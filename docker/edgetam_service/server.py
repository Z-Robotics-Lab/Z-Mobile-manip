#!/usr/bin/env python3
"""Standalone HTTP service for persistent EdgeTAM mask tracking.

Only RGB JPEG observations cross this boundary.  The service has no ROS or
simulator dependency and never receives object poses, depth, or ground truth.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import gc
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import re
import threading
import time
import traceback
from typing import Any, Mapping
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError


PROTOCOL_VERSION = "z-manip.edgetam/v1"
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class ServiceFault(RuntimeError):
    def __init__(self, status: HTTPStatus, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


class TrackingFailure(RuntimeError):
    """The model returned no trustworthy target mask."""


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8092
    model_id: str = "yonigozlan/EdgeTAM-hf"
    device: str = "cuda"
    session_timeout_s: float = 30.0
    max_sessions: int = 4
    # The streaming state is pruned every frame (``_prune_streaming_state``) so
    # GPU memory is bounded independently of frame count.  A low per-session
    # frame ceiling therefore only forced a periodic identity-destroying
    # re-acquisition (~2-3 min at field Hz) with no memory benefit.  Keep a
    # high default; operators can still lower it with the environment override.
    max_frames_per_session: int = 100_000
    # A frame whose model mask is empty/too-small/low-score does not end the
    # identity: EdgeTAM's streaming memory can bridge a brief occlusion and
    # relock the same track.  Coast (keep the session, advance the timeline,
    # return no mask) for up to this many *consecutive* such frames before
    # declaring the identity lost.  0 restores the legacy "kill on first
    # empty mask" behavior.  32 matches the default streaming-history horizon.
    max_coast_frames: int = 32
    max_request_bytes: int = 10_700_000
    max_jpeg_bytes: int = 8_000_000
    max_image_pixels: int = 4_194_304
    min_mask_pixels: int = 16
    min_score: float = 0.35
    vision_cache_frames: int = 8
    stream_history_frames: int = 32

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        config = cls(
            host=os.environ.get("EDGETAM_HOST", cls.host),
            port=int(os.environ.get("EDGETAM_PORT", cls.port)),
            model_id=os.environ.get("EDGETAM_MODEL_ID", cls.model_id),
            device=os.environ.get("EDGETAM_DEVICE", cls.device),
            session_timeout_s=float(
                os.environ.get("EDGETAM_SESSION_TIMEOUT_S", cls.session_timeout_s),
            ),
            max_sessions=int(os.environ.get("EDGETAM_MAX_SESSIONS", cls.max_sessions)),
            max_frames_per_session=int(
                os.environ.get(
                    "EDGETAM_MAX_FRAMES_PER_SESSION",
                    cls.max_frames_per_session,
                ),
            ),
            max_coast_frames=int(
                os.environ.get("EDGETAM_MAX_COAST_FRAMES", cls.max_coast_frames),
            ),
            max_request_bytes=int(
                os.environ.get("EDGETAM_MAX_REQUEST_BYTES", cls.max_request_bytes),
            ),
            max_jpeg_bytes=int(
                os.environ.get("EDGETAM_MAX_JPEG_BYTES", cls.max_jpeg_bytes),
            ),
            max_image_pixels=int(
                os.environ.get("EDGETAM_MAX_IMAGE_PIXELS", cls.max_image_pixels),
            ),
            min_mask_pixels=int(
                os.environ.get("EDGETAM_MIN_MASK_PIXELS", cls.min_mask_pixels),
            ),
            min_score=float(os.environ.get("EDGETAM_MIN_SCORE", cls.min_score)),
            vision_cache_frames=int(
                os.environ.get("EDGETAM_VISION_CACHE_FRAMES", cls.vision_cache_frames),
            ),
            stream_history_frames=int(
                os.environ.get(
                    "EDGETAM_STREAM_HISTORY_FRAMES",
                    cls.stream_history_frames,
                ),
            ),
        )
        if not (1 <= config.port <= 65535):
            raise ValueError("EDGETAM_PORT must be in [1, 65535]")
        if config.session_timeout_s <= 0.0:
            raise ValueError("EDGETAM_SESSION_TIMEOUT_S must be positive")
        if config.max_sessions < 1 or config.max_frames_per_session < 1:
            raise ValueError("session and frame limits must be positive")
        if config.max_coast_frames < 0:
            raise ValueError("EDGETAM_MAX_COAST_FRAMES cannot be negative")
        if config.max_request_bytes < 1024 or config.max_jpeg_bytes < 1024:
            raise ValueError("request and JPEG limits are too small")
        if config.max_image_pixels < 1 or config.min_mask_pixels < 1:
            raise ValueError("pixel limits must be positive")
        if not 0.0 <= config.min_score <= 1.0:
            raise ValueError("EDGETAM_MIN_SCORE must be in [0, 1]")
        if config.vision_cache_frames < 1 or config.stream_history_frames < 1:
            raise ValueError("EdgeTAM cache and streaming history limits must be positive")
        return config


@dataclass
class TrackingSession:
    session_id: str
    track_id: str
    inference_state: object
    image_size: tuple[int, int]
    last_frame_seq: int
    last_access_s: float
    lock: threading.RLock = field(default_factory=threading.RLock)
    closed: bool = False
    # Consecutive coasting frames (empty/low-score model mask) since the last
    # published mask.  Reset to zero on every tracking frame.
    coast_frames: int = 0


def _json_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_request",
            f"{name} must be an integer >= {minimum}",
        )
    return value


def _session_id(value: object) -> str:
    if not isinstance(value, str) or SESSION_RE.fullmatch(value) is None:
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_session_id",
            "session_id has an invalid format",
        )
    return value


def _protocol(document: Mapping[str, Any]) -> None:
    if document.get("protocol") != PROTOCOL_VERSION:
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "protocol_mismatch",
            f"protocol must be {PROTOCOL_VERSION}",
        )


def _bbox(value: object, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_bbox",
            "bbox_xyxy must contain four integer pixels",
        )
    x1, y1, x2, y2 = (
        _json_int(item, f"bbox_xyxy[{index}]")
        for index, item in enumerate(value)
    )
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_bbox",
            "bbox_xyxy is empty or outside the decoded image",
        )
    return x1, y1, x2, y2


def decode_jpeg(value: object, config: ServiceConfig) -> np.ndarray:
    if not isinstance(value, str):
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_jpeg",
            "image_jpeg_b64 must be a base64 string",
        )
    try:
        encoded = value.encode("ascii")
        raw = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_jpeg",
            "image_jpeg_b64 is not canonical base64",
        ) from exc
    if not 4 <= len(raw) <= config.max_jpeg_bytes:
        raise ServiceFault(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "invalid_jpeg",
            "decoded JPEG is empty or exceeds the configured limit",
        )
    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_jpeg",
            "payload is not a complete JPEG byte stream",
        )
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG":
                raise ServiceFault(
                    HTTPStatus.BAD_REQUEST,
                    "invalid_jpeg",
                    "decoded image is not JPEG",
                )
            width, height = image.size
            if width < 1 or height < 1 or width * height > config.max_image_pixels:
                raise ServiceFault(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "image_too_large",
                    "decoded image dimensions exceed the configured limit",
                )
            image.load()
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_jpeg",
            "JPEG could not be decoded safely",
        ) from exc
    if rgb.shape != (height, width, 3):
        raise ServiceFault(
            HTTPStatus.BAD_REQUEST,
            "invalid_jpeg",
            "JPEG decoder returned an unexpected image shape",
        )
    return rgb


def encode_coco_rle(mask: np.ndarray) -> dict[str, object]:
    array = np.asarray(mask, dtype=bool)
    flat = array.reshape(-1, order="F")
    # Vectorized run-length encoding.  This is byte-for-byte equivalent to the
    # previous per-pixel loop but avoids ~8 ms/frame of Python interpreter
    # overhead on a 640x480 mask, shortening the inference round trip.
    if flat.size == 0:
        counts: list[int] = [0]
    else:
        changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        boundaries = np.concatenate(
            (np.array([0], dtype=np.int64), changes, np.array([flat.size], dtype=np.int64)),
        )
        counts = np.diff(boundaries).tolist()
        # COCO RLE counts always begin with a background (False) run; prepend a
        # zero-length run when the first pixel is foreground.
        if bool(flat[0]):
            counts = [0] + counts
    return {
        "encoding": "coco_rle",
        "size": [int(array.shape[0]), int(array.shape[1])],
        "counts": [int(count) for count in counts],
    }


class EdgeTamBackend:
    """Lazily loaded official Transformers EdgeTAM streaming backend."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self._load_lock = threading.Lock()
        self._inference_lock = threading.RLock()
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None
        self._dtype = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            import torch
            from transformers import EdgeTamVideoModel, Sam2VideoProcessor

            device = torch.device(self.config.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "EDGETAM_DEVICE requests CUDA, but torch.cuda.is_available() is false",
                )
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
            processor = Sam2VideoProcessor.from_pretrained(self.config.model_id)
            model = EdgeTamVideoModel.from_pretrained(self.config.model_id)
            model = model.to(device=device, dtype=dtype)
            model.eval()
            self._torch = torch
            self._device = device
            self._dtype = dtype
            self._processor = processor
            self._model = model

    def initialize(
        self,
        image: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> tuple[object, np.ndarray, float]:
        self._load()
        with self._inference_lock, self._torch.inference_mode():
            state = self._processor.init_video_session(
                inference_device=self._device,
                dtype=self._dtype,
                max_vision_features_cache_size=self.config.vision_cache_frames,
            )
            try:
                frame, original_size = self._prepare_frame(image)
                self._processor.add_inputs_to_inference_session(
                    inference_session=state,
                    frame_idx=0,
                    obj_ids=1,
                    input_boxes=[[[float(item) for item in bbox_xyxy]]],
                    original_size=original_size,
                )
                output = self._model(
                    inference_session=state,
                    frame_idx=0,
                    frame=frame,
                )
                mask, score = self._result(output, original_size)
                return state, mask, score
            except Exception:
                reset = getattr(state, "reset_inference_session", None)
                if callable(reset):
                    reset()
                raise

    def update(
        self,
        state: object,
        image: np.ndarray,
        frame_seq: int,
    ) -> tuple[np.ndarray, float]:
        self._load()
        with self._inference_lock, self._torch.inference_mode():
            frame, original_size = self._prepare_frame(image)
            output = self._model(
                inference_session=state,
                frame_idx=frame_seq,
                frame=frame,
            )
            result = self._result(output, original_size)
            self._prune_streaming_state(state, frame_seq)
            return result

    def reset(self, state: object) -> None:
        with self._inference_lock:
            reset = getattr(state, "reset_inference_session", None)
            if callable(reset):
                reset()
            processed_frames = getattr(state, "processed_frames", None)
            if isinstance(processed_frames, dict):
                processed_frames.clear()
            # Transformers keeps the CUDA caching allocator populated even
            # after its session dictionaries are cleared.  Releasing those
            # blocks here lets an isolated grasp model share the GPU.
            gc.collect()
            if (
                self._torch is not None
                and self._device is not None
                and self._device.type == "cuda"
            ):
                self._torch.cuda.empty_cache()

    def _stream_history_limit(self) -> int:
        """Keep every model dependency while bounding streaming state."""
        model_config = getattr(self._model, "config", None)
        required = max(
            int(getattr(model_config, "num_maskmem", 1)),
            int(getattr(model_config, "max_object_pointers_in_encoder", 1)),
        )
        return max(self.config.stream_history_frames, required)

    def _prune_streaming_state(self, state: object, frame_seq: int) -> None:
        """Discard history that EdgeTAM can no longer consult in streaming mode.

        The upstream inference session retains all frames and all per-frame
        outputs by default.  EdgeTAM's streaming forward pass only consults a
        bounded recent window plus conditioning frames, so retaining older
        non-conditioning entries causes linear GPU growth without changing a
        future result.
        """
        history_limit = self._stream_history_limit()
        cutoff = frame_seq - history_limit + 1
        if cutoff <= 0:
            return

        def prune_indexed(mapping: object, *, before: int = cutoff) -> None:
            if not isinstance(mapping, dict):
                return
            for index in tuple(mapping):
                if isinstance(index, int) and index < before:
                    mapping.pop(index, None)

        # A processed frame is only needed while its vision features are being
        # computed.  Retaining the same bounded history is conservative and
        # also supports the model's small feature cache.
        prune_indexed(getattr(state, "processed_frames", None))
        for outputs in getattr(state, "output_dict_per_obj", {}).values():
            if isinstance(outputs, dict):
                prune_indexed(outputs.get("non_cond_frame_outputs"))
        for tracked in getattr(state, "frames_tracked_per_obj", {}).values():
            prune_indexed(tracked)

    def _prepare_frame(self, image: np.ndarray) -> tuple[object, tuple[int, int]]:
        inputs = self._processor(
            images=image,
            device=self._device,
            return_tensors="pt",
        )
        original = inputs.original_sizes[0]
        if hasattr(original, "detach"):
            original = original.detach().cpu().tolist()
        original_size = (int(original[0]), int(original[1]))
        return inputs.pixel_values[0], original_size

    def _result(self, output: object, original_size: tuple[int, int]) -> tuple[np.ndarray, float]:
        logits = getattr(output, "pred_masks", None)
        if logits is None:
            raise TrackingFailure("EdgeTAM returned no mask logits")
        restored = self._processor.post_process_masks(
            [logits],
            original_sizes=[original_size],
            binarize=False,
        )[0]
        if hasattr(restored, "detach"):
            restored = restored.detach().float().cpu().numpy()
        array = np.asarray(restored, dtype=np.float32)
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise TrackingFailure(f"unexpected EdgeTAM mask shape {array.shape}")
        mask = array > 0.0
        pixels = int(mask.sum())
        if pixels < self.config.min_mask_pixels:
            raise TrackingFailure("EdgeTAM returned an empty or too-small mask")
        # EdgeTAM Video exposes mask logits but no object-score field.  Mean
        # foreground probability is therefore the service confidence metric.
        foreground_logits = np.clip(array[mask], -30.0, 30.0)
        score = float(np.mean(1.0 / (1.0 + np.exp(-foreground_logits))))
        if not math.isfinite(score) or score < self.config.min_score:
            raise TrackingFailure("EdgeTAM confidence is below the configured threshold")
        return mask, score


class EdgeTamApplication:
    def __init__(self, config: ServiceConfig, backend: EdgeTamBackend | None = None):
        self.config = config
        self.backend = backend or EdgeTamBackend(config)
        self._sessions: dict[str, TrackingSession] = {}
        self._reserved: set[str] = set()
        self._sessions_lock = threading.RLock()

    def health(self) -> dict[str, object]:
        self._expire_sessions()
        with self._sessions_lock:
            active = len(self._sessions)
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "ok",
            "model_loaded": self.backend.loaded,
            "model_id": self.config.model_id,
            "device": self.config.device,
            "active_sessions": active,
            "max_sessions": self.config.max_sessions,
        }

    def init(self, document: Mapping[str, Any]) -> dict[str, object]:
        _protocol(document)
        session_id = _session_id(document.get("session_id"))
        frame_seq = _json_int(document.get("frame_seq"), "frame_seq")
        if frame_seq != 0:
            raise ServiceFault(
                HTTPStatus.CONFLICT,
                "out_of_order",
                "new sessions must start at frame_seq 0",
            )
        image = decode_jpeg(document.get("image_jpeg_b64"), self.config)
        height, width = image.shape[:2]
        bbox = _bbox(document.get("bbox_xyxy"), width, height)
        self._reserve(session_id)
        try:
            state, mask, score = self.backend.initialize(image, bbox)
            track = TrackingSession(
                session_id=session_id,
                track_id=uuid.uuid4().hex,
                inference_state=state,
                image_size=(width, height),
                last_frame_seq=0,
                last_access_s=time.monotonic(),
            )
            response = self._track_response(track, mask, score)
            with self._sessions_lock:
                self._reserved.discard(session_id)
                self._sessions[session_id] = track
            return response
        except TrackingFailure as exc:
            raise ServiceFault(
                HTTPStatus.GONE,
                "tracking_lost",
                str(exc),
            ) from exc
        finally:
            with self._sessions_lock:
                self._reserved.discard(session_id)

    def update(self, document: Mapping[str, Any]) -> dict[str, object]:
        _protocol(document)
        session_id = _session_id(document.get("session_id"))
        frame_seq = _json_int(document.get("frame_seq"), "frame_seq")
        image = decode_jpeg(document.get("image_jpeg_b64"), self.config)
        with self._locked_session(session_id) as session:
            expected_seq = session.last_frame_seq + 1
            if frame_seq != expected_seq:
                self._drop(session)
                raise ServiceFault(
                    HTTPStatus.CONFLICT,
                    "out_of_order",
                    f"expected frame_seq {expected_seq}, got {frame_seq}",
                )
            if frame_seq >= self.config.max_frames_per_session:
                self._drop(session)
                raise ServiceFault(
                    HTTPStatus.GONE,
                    "session_frame_limit",
                    "tracking session reached its configured frame limit",
                )
            height, width = image.shape[:2]
            if (width, height) != session.image_size:
                self._drop(session)
                raise ServiceFault(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "image_size_changed",
                    "image dimensions changed within a tracking session",
                )
            try:
                mask, score = self.backend.update(
                    session.inference_state,
                    image,
                    frame_seq,
                )
            except TrackingFailure as exc:
                # The model forward pass already advanced EdgeTAM's streaming
                # memory for this frame; only the *mask* was empty/low-score.
                # Coast (keep the identity, advance the timeline, publish no
                # mask) so a brief occlusion or motion-blur frame cannot destroy
                # a session that the streaming memory can still relock.
                if session.coast_frames < self.config.max_coast_frames:
                    session.coast_frames += 1
                    session.last_frame_seq = frame_seq
                    session.last_access_s = time.monotonic()
                    return self._coast_response(session)
                self._drop(session)
                raise ServiceFault(
                    HTTPStatus.GONE,
                    "tracking_lost",
                    str(exc),
                ) from exc
            except Exception:
                # A CUDA OOM or unexpected backend fault must not leave the
                # now-unreachable inference state resident in the session map.
                self._drop(session)
                raise
            session.coast_frames = 0
            session.last_frame_seq = frame_seq
            session.last_access_s = time.monotonic()
            return self._track_response(session, mask, score)

    def reset(self, document: Mapping[str, Any]) -> dict[str, object]:
        _protocol(document)
        session_id = _session_id(document.get("session_id"))
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)
            self._reserved.discard(session_id)
            if session is not None:
                session.closed = True
        if session is not None:
            with session.lock:
                self.backend.reset(session.inference_state)
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "reset",
            "session_id": session_id,
        }

    def close(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._reserved.clear()
            for session in sessions:
                session.closed = True
        for session in sessions:
            with session.lock:
                self.backend.reset(session.inference_state)

    def _reserve(self, session_id: str) -> None:
        self._expire_sessions()
        with self._sessions_lock:
            if session_id in self._sessions or session_id in self._reserved:
                raise ServiceFault(
                    HTTPStatus.CONFLICT,
                    "session_exists",
                    "reset the existing session before reusing session_id",
                )
            if len(self._sessions) + len(self._reserved) >= self.config.max_sessions:
                raise ServiceFault(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "session_capacity",
                    "EdgeTAM service has reached its configured session capacity",
                )
            self._reserved.add(session_id)

    @contextmanager
    def _locked_session(self, session_id: str):
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ServiceFault(
                    HTTPStatus.NOT_FOUND,
                    "unknown_session",
                    "tracking session does not exist",
                )
        with session.lock:
            now = time.monotonic()
            with self._sessions_lock:
                current = self._sessions.get(session_id)
                if current is not session or session.closed:
                    raise ServiceFault(
                        HTTPStatus.GONE,
                        "session_closed",
                        "tracking session is no longer active",
                    )
                if now - session.last_access_s > self.config.session_timeout_s:
                    self._sessions.pop(session_id, None)
                    session.closed = True
            if session.closed:
                self.backend.reset(session.inference_state)
                raise ServiceFault(
                    HTTPStatus.GONE,
                    "session_expired",
                    "tracking session exceeded its idle timeout",
                )
            yield session

    def _drop(self, session: TrackingSession) -> None:
        with self._sessions_lock:
            if self._sessions.get(session.session_id) is session:
                self._sessions.pop(session.session_id, None)
            session.closed = True
        self.backend.reset(session.inference_state)

    def _expire_sessions(self) -> None:
        with self._sessions_lock:
            candidates = list(self._sessions.values())
        for session in candidates:
            # An in-flight model call owns this lock and must not age out simply
            # because its GPU inference latency exceeds the idle threshold.
            if not session.lock.acquire(blocking=False):
                continue
            try:
                now = time.monotonic()
                with self._sessions_lock:
                    current = self._sessions.get(session.session_id)
                    if (
                        current is not session
                        or session.closed
                        or now - session.last_access_s <= self.config.session_timeout_s
                    ):
                        continue
                    self._sessions.pop(session.session_id, None)
                    session.closed = True
                self.backend.reset(session.inference_state)
            finally:
                session.lock.release()

    def _coast_response(self, session: TrackingSession) -> dict[str, object]:
        """Report a session-preserving coast with no mask observation.

        Identity fields (session_id, track_id, frame_seq, image_size) stay in
        strict lockstep with the tracking contract; only the mask/bbox/score
        are withheld because the model produced no trustworthy target this
        frame.  The consumer must treat this as keep-alive, never as an
        observation to publish or verify.
        """
        width, height = session.image_size
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "coasting",
            "session_id": session.session_id,
            "track_id": session.track_id,
            "frame_seq": session.last_frame_seq,
            "image_size": [width, height],
        }

    def _track_response(
        self,
        session: TrackingSession,
        mask: np.ndarray,
        score: float,
    ) -> dict[str, object]:
        mask = np.asarray(mask, dtype=bool)
        width, height = session.image_size
        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
            self._drop(session)
            raise ServiceFault(
                HTTPStatus.GONE,
                "tracking_lost",
                "model returned an invalid confidence score",
            )
        if mask.shape != (height, width):
            self._drop(session)
            raise ServiceFault(
                HTTPStatus.GONE,
                "tracking_lost",
                "model mask dimensions do not match the session image",
            )
        ys, xs = np.nonzero(mask)
        if len(xs) < self.config.min_mask_pixels:
            self._drop(session)
            raise ServiceFault(
                HTTPStatus.GONE,
                "tracking_lost",
                "model returned an empty or too-small target mask",
            )
        bbox = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "tracking",
            "session_id": session.session_id,
            "track_id": session.track_id,
            "frame_seq": session.last_frame_seq,
            "image_size": [width, height],
            "bbox_xyxy": bbox,
            "score": float(score),
            "mask_rle": encode_coco_rle(mask),
        }


class EdgeTamRequestHandler(BaseHTTPRequestHandler):
    server_version = "z-manip-edgetam/1"
    application: EdgeTamApplication

    def do_GET(self) -> None:
        if self.path != "/health":
            self._error(ServiceFault(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint"))
            return
        self._json(HTTPStatus.OK, self.application.health())

    def do_POST(self) -> None:
        try:
            document = self._read_document()
            routes = {
                "/v1/sessions/init": self.application.init,
                "/v1/sessions/update": self.application.update,
                "/v1/sessions/reset": self.application.reset,
            }
            action = routes.get(self.path)
            if action is None:
                raise ServiceFault(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
            self._json(HTTPStatus.OK, action(document))
        except ServiceFault as exc:
            # Request bodies can contain camera images and are deliberately
            # never logged.  The structured route/code/detail is enough to
            # distinguish confidence loss, session expiry and bad input in
            # field logs.
            print(
                f'edgetam_service fault route={self.path} '
                f'status={int(exc.status)} code={exc.code} detail={exc}',
                flush=True,
            )
            self._error(exc)
        except Exception:
            # Keep diagnostics server-side without returning request/image data.
            traceback.print_exc()
            self._error(
                ServiceFault(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "EdgeTAM service failed without a tracking result",
                ),
            )

    def _read_document(self) -> Mapping[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise ServiceFault(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "content_type",
                "Content-Type must be application/json",
            )
        length_raw = self.headers.get("Content-Length")
        try:
            length = int(length_raw or "")
        except ValueError as exc:
            raise ServiceFault(
                HTTPStatus.LENGTH_REQUIRED,
                "content_length",
                "a valid Content-Length is required",
            ) from exc
        if not 1 <= length <= self.application.config.max_request_bytes:
            raise ServiceFault(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request_too_large",
                "request body is empty or exceeds the configured limit",
            )
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ServiceFault(
                HTTPStatus.BAD_REQUEST,
                "truncated_request",
                "request body ended before Content-Length",
            )
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServiceFault(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "request body must be UTF-8 JSON",
            ) from exc
        if not isinstance(document, Mapping):
            raise ServiceFault(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "request JSON root must be an object",
            )
        return document

    def _error(self, fault: ServiceFault) -> None:
        self._json(
            fault.status,
            {
                "protocol": PROTOCOL_VERSION,
                "status": "error",
                "error": {"code": fault.code, "message": str(fault)},
            },
        )

    def _json(self, status: HTTPStatus, document: Mapping[str, Any]) -> None:
        payload = json.dumps(
            document,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        # Never log request bodies; BaseHTTPRequestHandler only supplies route/status.
        print(f"edgetam_service {self.address_string()} {format % args}", flush=True)


class EdgeTamHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    config = ServiceConfig.from_env()
    application = EdgeTamApplication(config)
    EdgeTamRequestHandler.application = application
    server = EdgeTamHttpServer((config.host, config.port), EdgeTamRequestHandler)
    print(
        f"EdgeTAM service listening on {config.host}:{config.port}; "
        f"model loads lazily from {config.model_id}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        application.close()


if __name__ == "__main__":
    main()
