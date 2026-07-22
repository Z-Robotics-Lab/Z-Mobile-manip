#!/usr/bin/env python3
"""Loopback-only, GPU-resident YOLOE service for fast 2-D/mask seeding.

The process reads pixels and text only.  It has no ROS, CAN, robot, planning, or
actuator imports. Model weights are baked into the service image; network access
is disabled by the service unit.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import re
import threading
import time
from typing import Any, Mapping


MODEL_ID = "yoloe-11s-seg.pt"
REQUEST_SCHEMA = "z_manip.local_grounding_request.v1"
RESPONSE_SCHEMA = "z_manip.local_grounding_response.v1"
MAX_REQUEST_BYTES = 2 * 1024 * 1024

_ZH_NOUNS: tuple[tuple[str, str], ...] = (
    ("电源适配器", "power adapter"),
    ("充电适配器", "charger"),
    ("充电器", "charger"),
    ("适配器", "adapter"),
    ("插头", "electrical plug"),
    ("遥控器", "remote control"),
    ("鼠标", "computer mouse"),
    ("手机", "mobile phone"),
    ("方块", "block"),
    ("积木", "block"),
    ("盒子", "box"),
    ("瓶子", "bottle"),
    ("杯子", "cup"),
    ("罐子", "can"),
    ("碗", "bowl"),
    ("球", "ball"),
)
_ZH_COLORS: tuple[tuple[str, str], ...] = (
    ("白色", "white"),
    ("白", "white"),
    ("黑色", "black"),
    ("黑", "black"),
    ("红色", "red"),
    ("红", "red"),
    ("绿色", "green"),
    ("绿", "green"),
    ("蓝色", "blue"),
    ("蓝", "blue"),
    ("黄色", "yellow"),
    ("黄", "yellow"),
    ("紫色", "purple"),
    ("紫", "purple"),
)


def grounding_prompt(instruction: str) -> str | None:
    """Return one concise English YOLOE class, or None for VLM fallback."""

    query = " ".join(str(instruction).strip().lower().split())
    if not query:
        return None
    noun = next((english for chinese, english in _ZH_NOUNS if chinese in query), None)
    color = next((english for chinese, english in _ZH_COLORS if chinese in query), None)
    if noun is not None:
        phrase = f"{color} {noun}" if color else noun
        return phrase
    if any("\u4e00" <= character <= "\u9fff" for character in query):
        return None
    cleaned = re.sub(r"[^a-z0-9\s_-]+", " ", query)
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(
        r"^(?:please\s+)?(?:pick(?:\s+up)?|grasp|grab|find|track|locate|approach)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned)
    if not cleaned:
        return None
    return cleaned


def select_detection(
    boxes_xyxy: object,
    scores: object,
    labels: object,
    *,
    width: int,
    height: int,
    minimum_confidence: float,
    maximum_area_ratio: float,
    minimum_border_margin_ratio: float = 0.002,
) -> dict[str, object] | None:
    """Select the strongest complete, finite, object-scale detection.

    A grasp seed must describe a fully observed object. Boxes clipped against
    an image edge are commonly the robot gripper, furniture, or another partial
    foreground object. Passing those boxes to the tracker creates a stable but
    geometrically meaningless mask, so reject them before confidence ranking.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum confidence must be within [0, 1]")
    if not 0.0 < maximum_area_ratio < 1.0:
        raise ValueError("maximum area ratio must be within (0, 1)")
    if not 0.0 <= minimum_border_margin_ratio < 0.5:
        raise ValueError("minimum border margin ratio must be within [0, 0.5)")
    candidates: list[tuple[float, float, dict[str, object]]] = []
    for index, raw_box in enumerate(boxes_xyxy):
        try:
            box = [float(value) for value in raw_box]
            confidence = float(scores[index])
        except (IndexError, TypeError, ValueError):
            continue
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            continue
        if not math.isfinite(confidence) or confidence < minimum_confidence:
            continue
        x1, y1, x2, y2 = box
        x1 = min(float(width), max(0.0, x1))
        y1 = min(float(height), max(0.0, y1))
        x2 = min(float(width), max(0.0, x2))
        y2 = min(float(height), max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        border_x = minimum_border_margin_ratio * width
        border_y = minimum_border_margin_ratio * height
        if (
            x1 <= border_x
            or y1 <= border_y
            or x2 >= width - border_x
            or y2 >= height - border_y
        ):
            continue
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
        if area_ratio < 0.0002 or area_ratio > maximum_area_ratio:
            continue
        label = "object"
        try:
            candidate_label = str(labels[index]).strip()
            if candidate_label:
                label = candidate_label
        except (IndexError, TypeError):
            pass
        result = {
            "label": label,
            "bbox_xyxy": [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ],
            "confidence": confidence,
            "area_ratio": area_ratio,
        }
        # Confidence is primary.  A tiny area tiebreak prevents a broad support
        # surface from winning when its score is effectively identical.
        candidates.append((confidence, -area_ratio, result))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


class GroundingRuntime:
    """One persistent CUDA YOLOE model guarded against concurrent forwards."""

    def __init__(
        self,
        *,
        model_id: str,
        minimum_confidence: float,
        maximum_area_ratio: float,
    ) -> None:
        self.model_id = model_id
        self.minimum_confidence = minimum_confidence
        self.maximum_area_ratio = maximum_area_ratio
        self._lock = threading.Lock()
        self._model: Any = None
        self._device = "unloaded"
        self._classes: tuple[str, ...] = ()

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("local grounding requires CUDA")
        torch.backends.cuda.matmul.allow_tf32 = True
        self._device = "cuda:0"
        self._model = YOLO(self.model_id, task="segment")
        self._model.to(self._device)

    def ground(self, image_bytes: bytes, instruction: str) -> dict[str, object]:
        prompt = grounding_prompt(instruction)
        if prompt is None:
            raise LookupError("instruction has no supported local noun phrase")
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        started = time.perf_counter()
        with self._lock:
            self.load()
            requested_classes = (prompt,)
            if self._classes != requested_classes:
                self._model.set_classes(list(requested_classes))
                self._classes = requested_classes
            result = self._model.predict(
                source=image,
                device=self._device,
                imgsz=640,
                conf=min(0.20, self.minimum_confidence),
                iou=0.55,
                # YOLOE builds new text embeddings whenever the prompt changes.
                # Ultralytics' half=True path permanently converts the detection
                # head to FP16, while a later MobileCLIP embedding is FP32.  The
                # next dynamic prompt then fails inside the head with a
                # Float/Half matmul mismatch.  Keep the persistent model in FP32;
                # TF32 is enabled in load(), so RTX inference remains fast and
                # arbitrary successive prompts remain type-stable.
                half=False,
                max_det=24,
                retina_masks=False,
                verbose=False,
            )[0]
        boxes = getattr(result, "boxes", None)
        xyxy = [] if boxes is None else boxes.xyxy.detach().cpu().tolist()
        scores = [] if boxes is None else boxes.conf.detach().cpu().tolist()
        class_ids = [] if boxes is None else boxes.cls.detach().cpu().tolist()
        names = getattr(result, "names", {0: prompt})
        labels = [str(names.get(int(class_id), prompt)) for class_id in class_ids]
        selected = select_detection(
            xyxy,
            scores,
            labels,
            width=width,
            height=height,
            minimum_confidence=self.minimum_confidence,
            maximum_area_ratio=self.maximum_area_ratio,
        )
        if selected is None:
            raise LookupError("local detector produced no qualified object box")
        return {
            "schema": RESPONSE_SCHEMA,
            "model": f"local/yoloe/{os.path.basename(self.model_id)}",
            "prompt": prompt,
            "target": selected,
            "latency_s": time.perf_counter() - started,
        }

    def warmup(self) -> None:
        """Run one synthetic forward so the first camera request stays fast."""

        from PIL import Image

        image = Image.new("RGB", (640, 480), color=(127, 127, 127))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        try:
            self.ground(encoded.getvalue(), "bottle")
        except LookupError:
            # A blank image should normally have no qualified detection; the
            # CUDA kernels and processor caches are warm regardless.
            pass

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device


class GroundingServer(ThreadingHTTPServer):
    runtime: GroundingRuntime


class RequestHandler(BaseHTTPRequestHandler):
    server: GroundingServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _json(self, status: HTTPStatus, document: Mapping[str, object]) -> None:
        payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._json(HTTPStatus.OK, {
            "schema": "z_manip.local_grounding_health.v1",
            "ready": self.server.runtime.loaded,
            "backend": "yoloe",
            "model": self.server.runtime.model_id,
            "device": self.server.runtime.device,
            "read_only": True,
            "motion_commands_published": 0,
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ground":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= content_length <= MAX_REQUEST_BYTES:
                raise ValueError("request size is invalid")
            document = json.loads(self.rfile.read(content_length))
            if not isinstance(document, Mapping) or document.get("schema") != REQUEST_SCHEMA:
                raise ValueError("request schema is invalid")
            instruction = str(document.get("instruction", "")).strip()
            encoded = document.get("image_base64")
            if not instruction or not isinstance(encoded, str):
                raise ValueError("instruction and image_base64 are required")
            image_bytes = base64.b64decode(encoded, validate=True)
            if not image_bytes:
                raise ValueError("decoded image is empty")
            response = self.server.runtime.ground(image_bytes, instruction)
        except LookupError as error:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {
                "schema": RESPONSE_SCHEMA,
                "error": str(error),
                "fallback_recommended": True,
            })
            return
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": f"{type(error).__name__}: {error}",
                "fallback_recommended": True,
            })
            return
        self._json(HTTPStatus.OK, response)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--minimum-confidence", type=float, default=0.35)
    parser.add_argument("--maximum-area-ratio", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.host not in {"127.0.0.1", "::1"}:
        raise ValueError("local grounding service may bind only to loopback")
    runtime = GroundingRuntime(
        model_id=args.model,
        minimum_confidence=args.minimum_confidence,
        maximum_area_ratio=args.maximum_area_ratio,
    )
    runtime.load()
    runtime.warmup()
    server = GroundingServer((args.host, args.port), RequestHandler)
    server.runtime = runtime
    print(
        f"local grounding ready on {args.host}:{args.port} model={args.model} "
        f"device={runtime.device}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
