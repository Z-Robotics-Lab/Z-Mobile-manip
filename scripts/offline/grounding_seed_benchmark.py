#!/usr/bin/env python3
"""Extract and replay exact grounding seeds from a closed rosbag.

The two subcommands are deliberately separated so ROS bag extraction can run
in the ROS runtime image while YOLOE replay runs in the CUDA detector image.
Neither subcommand initializes ROS, publishes a message, or opens a transport.
Deployment invocations must additionally use Docker ``--network none``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable


REQUEST_TOPIC = "/z_manip/grounding/request"
IMAGE_TOPIC = "/track_3d/exact_seed_image"
OFFER_TOPIC = "/track_3d/seed_offer_manifest"
BBOX_TOPIC = "/track_3d/init_bbox"
FRAME_TOPIC = "/track_3d/frame_manifest"
READ_TOPICS = {REQUEST_TOPIC, IMAGE_TOPIC, OFFER_TOPIC, BBOX_TOPIC, FRAME_TOPIC}

EXTRACT_SCHEMA = "z_mobile_manip.offline_grounding_seed_extract.v1"
REPLAY_SCHEMA = "z_mobile_manip.offline_grounding_seed_replay.v1"


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(samples),
        "min": min(samples, default=None),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "max": max(samples, default=None),
    }


def _json_document(message: object) -> dict[str, Any] | None:
    try:
        value = json.loads(message.data)
    except (AttributeError, TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _stamp_ns(message: object) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def _load_ros_modules() -> dict[str, Any]:
    try:
        from rclpy.serialization import deserialize_message
        import rosbag2_py
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:  # pragma: no cover - runtime-image only
        raise RuntimeError("run extraction in the ROS runtime image") from error
    return {
        "deserialize_message": deserialize_message,
        "rosbag2_py": rosbag2_py,
        "get_message": get_message,
    }


def correlate_seed_records(
    *,
    requests: dict[str, dict[str, Any]],
    offers: list[dict[str, Any]],
    image_record_ns: dict[int, int],
    bbox_record_ns: dict[int, int],
    first_frame_record_ns: dict[str, int],
) -> list[dict[str, Any]]:
    """Join immutable seed evidence without guessing by label or proximity."""

    records: list[dict[str, Any]] = []
    for offer in sorted(offers, key=lambda item: int(item["record_ns"])):
        stamp_ns = int(offer["stamp_ns"])
        if stamp_ns not in image_record_ns:
            continue
        request = requests.get(str(offer.get("request_id", "")))
        request_record_ns = None if request is None else int(request["record_ns"])
        offer_record_ns = int(offer["record_ns"])
        bbox_ns = bbox_record_ns.get(stamp_ns)
        first_frame_ns = first_frame_record_ns.get(str(offer.get("offer_token", "")))
        records.append(
            {
                "request_id": offer.get("request_id"),
                "instruction": None if request is None else request.get("instruction"),
                "stamp_ns": stamp_ns,
                "offer_token": offer.get("offer_token"),
                "recorded_init_bbox": bbox_ns is not None,
                "latency_s": {
                    "request_to_offer": None
                    if request_record_ns is None
                    else (offer_record_ns - request_record_ns) * 1e-9,
                    "offer_to_init_bbox": None
                    if bbox_ns is None
                    else (bbox_ns - offer_record_ns) * 1e-9,
                    "request_to_init_bbox": None
                    if request_record_ns is None or bbox_ns is None
                    else (bbox_ns - request_record_ns) * 1e-9,
                    "offer_to_first_tracker_frame": None
                    if first_frame_ns is None
                    else (first_frame_ns - offer_record_ns) * 1e-9,
                    "init_bbox_to_first_tracker_frame": None
                    if bbox_ns is None or first_frame_ns is None
                    else (first_frame_ns - bbox_ns) * 1e-9,
                },
            }
        )
    return records


def extract_bag(bag: Path, output: Path) -> dict[str, Any]:
    modules = _load_ros_modules()
    rosbag2_py = modules["rosbag2_py"]
    deserialize_message = modules["deserialize_message"]
    get_message = modules["get_message"]

    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    requests: dict[str, dict[str, Any]] = {}
    offers: list[dict[str, Any]] = []
    image_record_ns: dict[int, int] = {}
    image_bytes: dict[int, bytes] = {}
    bbox_record_ns: dict[int, int] = {}
    first_frame_record_ns: dict[str, int] = {}

    while reader.has_next():
        topic, raw, record_ns = reader.read_next()
        if topic not in READ_TOPICS:
            continue
        message = deserialize_message(raw, get_message(topic_types[topic]))
        if topic == REQUEST_TOPIC:
            document = _json_document(message)
            if document and document.get("schema") == "z_manip.grounding_request.v2":
                request_id = str(document.get("request_id", ""))
                if request_id:
                    requests[request_id] = {
                        "instruction": str(document.get("instruction", "")),
                        "record_ns": int(record_ns),
                    }
        elif topic == OFFER_TOPIC:
            document = _json_document(message)
            if document and document.get("schema") == "z_manip.seed_offer.v1":
                document = dict(document)
                document["record_ns"] = int(record_ns)
                offers.append(document)
        elif topic == IMAGE_TOPIC:
            stamp_ns = _stamp_ns(message)
            image_record_ns[stamp_ns] = int(record_ns)
            image_bytes[stamp_ns] = bytes(message.data)
        elif topic == BBOX_TOPIC:
            bbox_record_ns.setdefault(_stamp_ns(message), int(record_ns))
        elif topic == FRAME_TOPIC:
            document = _json_document(message)
            if document and document.get("schema") == "z_manip.tracker_frame.v1":
                seed_id = str(document.get("seed_id", ""))
                if seed_id:
                    first_frame_record_ns.setdefault(seed_id, int(record_ns))

    records = correlate_seed_records(
        requests=requests,
        offers=offers,
        image_record_ns=image_record_ns,
        bbox_record_ns=bbox_record_ns,
        first_frame_record_ns=first_frame_record_ns,
    )
    for index, record in enumerate(records):
        filename = f"{index:03d}-{int(record['stamp_ns'])}.jpg"
        (image_dir / filename).write_bytes(image_bytes[int(record["stamp_ns"])])
        record["image"] = str(Path("images") / filename)

    latency_keys = (
        "request_to_offer",
        "offer_to_init_bbox",
        "request_to_init_bbox",
        "offer_to_first_tracker_frame",
        "init_bbox_to_first_tracker_frame",
    )
    first_offer_latency: list[float] = []
    seen_request_ids: set[str] = set()
    for record in records:
        request_id = str(record.get("request_id") or "")
        latency = record["latency_s"]["request_to_offer"]
        if not request_id or request_id in seen_request_ids or latency is None:
            continue
        seen_request_ids.add(request_id)
        first_offer_latency.append(float(latency))
    report = {
        "schema": EXTRACT_SCHEMA,
        "bag": str(bag),
        "sample_count": len(records),
        "recorded_init_bbox_count": sum(
            bool(record["recorded_init_bbox"]) for record in records
        ),
        "stage_latency_s": {
            key: summarize(
                record["latency_s"][key]
                for record in records
                if record["latency_s"][key] is not None
            )
            for key in latency_keys
        },
        # A request can emit retry offers. This metric isolates the first offer
        # per immutable request identity instead of inflating latency with
        # intentionally delayed retries.
        "first_offer_per_request_latency_s": summarize(first_offer_latency),
        "samples": records,
    }
    (output / "seed_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _load_grounding_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("local_grounding_service", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load grounding service: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_yoloe(
    manifest_path: Path,
    *,
    grounding_service: Path,
    model: str,
    output: Path,
) -> dict[str, Any]:
    from PIL import Image

    module = _load_grounding_module(grounding_service)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = module.GroundingRuntime(
        model_id=model,
        minimum_confidence=0.20,
        maximum_area_ratio=0.45,
    )
    runtime.load()
    samples: list[dict[str, Any]] = []
    manifest_root = manifest_path.parent

    for source in manifest.get("samples", []):
        instruction = str(source.get("instruction") or "").strip()
        if not instruction:
            continue
        prompts = module.grounding_prompts(instruction)
        if not prompts:
            continue
        image_path = manifest_root / str(source["image"])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        def run_classes(classes: tuple[str, ...]) -> tuple[Any, dict[str, float], bool]:
            started = time.perf_counter()
            with runtime._lock:
                embedding_started = time.perf_counter()
                cache_hit = runtime._classes == classes
                if runtime._classes != classes:
                    embedding, cache_hit = runtime._embedding_for(classes)
                    runtime._model.set_classes(list(classes), embeddings=embedding)
                    runtime._classes = classes
                embedding_finished = time.perf_counter()
                result = runtime._model.predict(
                    source=image,
                    device=runtime._device,
                    imgsz=640,
                    conf=0.01,
                    iou=0.55,
                    half=False,
                    max_det=24,
                    retina_masks=False,
                    verbose=False,
                )[0]
            inference_finished = time.perf_counter()
            boxes = getattr(result, "boxes", None)
            xyxy = [] if boxes is None else boxes.xyxy.detach().cpu().tolist()
            scores = [] if boxes is None else boxes.conf.detach().cpu().tolist()
            class_ids = [] if boxes is None else boxes.cls.detach().cpu().tolist()
            names = getattr(result, "names", {0: classes[0]})
            labels = [
                str(names.get(int(class_id), classes[0])) for class_id in class_ids
            ]
            area_limits = {
                candidate: 0.12
                for candidate in classes
                if candidate.startswith("small ")
            }
            selected = module.select_detection(
                xyxy,
                scores,
                labels,
                width=width,
                height=height,
                minimum_confidence=0.0,
                maximum_area_ratio=0.45,
                maximum_area_ratio_by_label=area_limits,
            )
            postprocess_finished = time.perf_counter()
            return (
                selected,
                {
                    "embedding": embedding_finished - embedding_started,
                    "inference": inference_finished - embedding_finished,
                    "postprocess": postprocess_finished - inference_finished,
                    "total": postprocess_finished - started,
                },
                cache_hit,
            )

        selected, production_latency, cache_hit = run_classes(prompts)
        confidence = None if selected is None else float(selected["confidence"])
        single_prompt_scores: dict[str, float | None] = {}
        for prompt in prompts:
            single_selected, _, _ = run_classes((prompt,))
            single_prompt_scores[prompt] = (
                None
                if single_selected is None
                else float(single_selected["confidence"])
            )
        finite_single_scores = {
            prompt: score
            for prompt, score in single_prompt_scores.items()
            if score is not None
        }
        best_single_prompt = (
            max(finite_single_scores, key=finite_single_scores.get)
            if finite_single_scores
            else None
        )
        samples.append(
            {
                "image": source["image"],
                "instruction": instruction,
                "recorded_init_bbox": bool(source.get("recorded_init_bbox")),
                "prompt_count": len(prompts),
                "embedding_cache_hit": bool(cache_hit),
                "qualified_confidence": confidence,
                "single_prompt_scores": single_prompt_scores,
                "best_single_prompt": best_single_prompt,
                "best_single_confidence": None
                if best_single_prompt is None
                else finite_single_scores[best_single_prompt],
                "passes_detector_0_20": confidence is not None and confidence >= 0.20,
                "passes_wrist_0_55": confidence is not None and confidence >= 0.55,
                "latency_s": production_latency,
            }
        )

    confidence_groups: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        confidence = sample["qualified_confidence"]
        if confidence is None:
            continue
        key = "recorded_init_bbox" if sample["recorded_init_bbox"] else "no_init_bbox"
        confidence_groups[key].append(float(confidence))
    report = {
        "schema": REPLAY_SCHEMA,
        "source_manifest": str(manifest_path),
        "model": model,
        "raw_predict_confidence": 0.01,
        "production_detector_threshold": 0.20,
        "wrist_confirmation_threshold": 0.55,
        "sample_count": len(samples),
        "threshold_counts": {
            "passes_detector_0_20": sum(x["passes_detector_0_20"] for x in samples),
            "passes_wrist_0_55": sum(x["passes_wrist_0_55"] for x in samples),
        },
        "confidence_by_recorded_outcome": {
            key: {
                **summarize(values),
                "mean": statistics.fmean(values) if values else None,
                "values": values,
            }
            for key, values in confidence_groups.items()
        },
        "latency_s": {
            key: summarize(sample["latency_s"][key] for sample in samples)
            for key in ("embedding", "inference", "postprocess", "total")
        },
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--bag", required=True, type=Path)
    extract_parser.add_argument("--output", required=True, type=Path)
    replay_parser = subparsers.add_parser("replay-yoloe")
    replay_parser.add_argument("--manifest", required=True, type=Path)
    replay_parser.add_argument("--grounding-service", required=True, type=Path)
    replay_parser.add_argument("--model", required=True)
    replay_parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    if arguments.command == "extract":
        report = extract_bag(arguments.bag, arguments.output)
    else:
        report = replay_yoloe(
            arguments.manifest,
            grounding_service=arguments.grounding_service,
            model=arguments.model,
            output=arguments.output,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
