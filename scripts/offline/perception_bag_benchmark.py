#!/usr/bin/env python3
"""Benchmark perception evidence directly from a rosbag2 MCAP.

This tool is intentionally incapable of publishing ROS messages.  It uses
``rosbag2_py.SequentialReader`` plus ``rclpy.serialization`` only; it never
initializes rclpy, creates a node, imports robot drivers, or opens a network,
CAN, or WebRTC transport.  Run it in the runtime image with ``--network none``
to reproduce the deployment Python/ROS environment without joining a DDS
domain.

Two different measurements are reported and must not be conflated:

* ``recorded_fresh`` is the wall-clock time from a recorded grounding request
  to the first exact six-artifact perception bundle for that request.
* ``tracked_counterfactual`` asks whether the exact active tracker identity at
  a repeated same-instruction request already had a complete cached bundle,
  matching the production fast path.  It applies the production
  ``TrackingReuseContract``; it does not assume that a label match is
  sufficient.

The optional CPU replay decodes selected recorded bundles and reruns the same
point-cloud filtering, target exclusion, and antipodal grasp proposal used by
the read-only perception wrapper.  It does not rerun YOLOE/EdgeTAM because the
bag already contains their immutable output evidence.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable


REPORT_SCHEMA = "z_mobile_manip.offline_perception_bag_benchmark.v1"

REQUEST_TOPIC = "/z_manip/grounding/request"
STATUS_TOPIC = "/z_manip/perception/status"
VALID_TOPIC = "/z_manip/perception/valid"
MANIFEST_TOPIC = "/track_3d/frame_manifest"
OVERLAY_TOPIC = "/z_manip/perception/overlay"
MASK_TOPIC = "/z_manip/perception/target_mask"
TARGET_CLOUD_TOPIC = "/z_manip/perception/target_pointcloud"
SCENE_CLOUD_TOPIC = "/z_manip/perception/scene_pointcloud"
INFO_TOPIC = "/camera/color/camera_info"
RAW_COLOR_TOPIC = "/nuc/camera/color/image_raw/compressed"
RAW_DEPTH_TOPIC = "/nuc/camera/aligned_depth_to_color/image_raw/compressedDepth"
RAW_INFO_TOPIC = "/nuc/camera/color/camera_info"

BUNDLE_TOPICS = (
    OVERLAY_TOPIC,
    MASK_TOPIC,
    TARGET_CLOUD_TOPIC,
    SCENE_CLOUD_TOPIC,
    MANIFEST_TOPIC,
    INFO_TOPIC,
)
READ_TOPICS = set(BUNDLE_TOPICS) | {
    REQUEST_TOPIC,
    STATUS_TOPIC,
    VALID_TOPIC,
    RAW_COLOR_TOPIC,
    RAW_DEPTH_TOPIC,
    RAW_INFO_TOPIC,
}


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


def summarize_seconds(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(samples),
        "min_s": min(samples, default=None),
        "p50_s": percentile(samples, 50),
        "p95_s": percentile(samples, 95),
        "max_s": max(samples, default=None),
    }


def nearest_stamp_deltas_s(
    reference_stamps_ns: Iterable[int],
    candidate_stamps_ns: Iterable[int],
) -> list[float]:
    """Return absolute nearest-neighbour timestamp deltas in seconds."""

    candidates = sorted(int(value) for value in candidate_stamps_ns)
    if not candidates:
        return []
    result: list[float] = []
    for raw in reference_stamps_ns:
        stamp = int(raw)
        index = bisect_left(candidates, stamp)
        neighbours = candidates[max(0, index - 1) : min(len(candidates), index + 1)]
        result.append(min(abs(stamp - other) for other in neighbours) * 1e-9)
    return result


def _stamp_ns(message: object) -> int:
    header = getattr(message, "header")
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _status_values(message: object) -> list[dict[str, str]]:
    result = []
    for status in getattr(message, "status", []):
        values = {str(item.key): str(item.value) for item in status.values}
        if values.get("schema") == "z_manip.perception_status.v1":
            result.append(values)
    return result


def _bounded_bundle_insert(
    bundles: OrderedDict[int, dict[str, Any]],
    stamp_ns: int,
    topic: str,
    message: object,
    record_timestamp_ns: int,
    *,
    maximum: int = 360,
) -> dict[str, Any]:
    slot = bundles.setdefault(stamp_ns, {"messages": {}, "record_times_ns": {}})
    slot["messages"][topic] = message
    slot["record_times_ns"][topic] = int(record_timestamp_ns)
    bundles.move_to_end(stamp_ns)
    while len(bundles) > maximum:
        bundles.popitem(last=False)
    return slot


def _bundle_complete(slot: dict[str, Any]) -> bool:
    return all(topic in slot["messages"] for topic in BUNDLE_TOPICS)


def _load_ros_modules() -> dict[str, Any]:
    """Lazy imports keep unit tests and report parsing ROS-independent."""

    try:
        from rclpy.serialization import deserialize_message
        import rosbag2_py
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "ROS bag reader modules are unavailable; run in z-manip-runtime:jazzy"
        ) from error
    return {
        "deserialize_message": deserialize_message,
        "rosbag2_py": rosbag2_py,
        "get_message": get_message,
    }


def _request_document(message: object) -> dict[str, Any] | None:
    try:
        value = json.loads(message.data)
    except (AttributeError, TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema") != "z_manip.grounding_request.v2"
        or not str(value.get("request_id", "")).strip()
        or not str(value.get("instruction", "")).strip()
    ):
        return None
    return value


def _manifest_document(message: object) -> dict[str, Any] | None:
    try:
        value = json.loads(message.data)
        stamp_ns = int(value["result_stamp_ns"])
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if value.get("schema") != "z_manip.tracker_frame.v1" or stamp_ns <= 0:
        return None
    return value


def _new_contract(values: dict[str, str]) -> Any | None:
    from z_manip.perception.tracked_reuse import parse_tracking_reuse_contract

    instruction_sha256 = values.get("instruction_sha256", "")
    if not instruction_sha256:
        return None
    return parse_tracking_reuse_contract(
        values,
        expected_instruction_sha256=instruction_sha256,
    )


def _benchmark_bundle(slot: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Replay the CPU-heavy production post-processing for one exact bundle."""

    import cv2
    from cv_bridge import CvBridge
    import numpy as np
    from sensor_msgs_py import point_cloud2

    from z_manip.models.antipodal_grasp import AntipodalGraspSource
    from z_manip.models.grasp_source import GraspContext, GraspGenerationError
    from z_manip.perception.rgbd import filter_object_cloud, target_exclusion_mask

    messages = slot["messages"]
    bridge = CvBridge()
    measurements: list[dict[str, Any]] = []
    for _ in range(repeats):
        started = time.perf_counter()
        overlay = bridge.imgmsg_to_cv2(messages[OVERLAY_TOPIC], desired_encoding="bgr8")
        mask = bridge.imgmsg_to_cv2(messages[MASK_TOPIC], desired_encoding="passthrough")
        target = np.asarray(
            point_cloud2.read_points_numpy(
                messages[TARGET_CLOUD_TOPIC],
                field_names=("x", "y", "z"),
                skip_nans=True,
            ),
            dtype=np.float32,
        ).reshape(-1, 3)
        scene = np.asarray(
            point_cloud2.read_points_numpy(
                messages[SCENE_CLOUD_TOPIC],
                field_names=("x", "y", "z"),
                skip_nans=True,
            ),
            dtype=np.float32,
        ).reshape(-1, 3)
        decoded_at = time.perf_counter()

        filtered = filter_object_cloud(target, viewpoint=(0.0, 0.0, 0.0))
        excluded = target_exclusion_mask(scene, filtered, radius_m=0.012)
        collision_scene = np.ascontiguousarray(scene[~excluded], dtype=np.float32)
        filtered_at = time.perf_counter()

        context = GraspContext(
            object_points=filtered,
            bbox=None,
            source_frame=messages[TARGET_CLOUD_TOPIC].header.frame_id,
            t_target_src=np.eye(4),
            scene_points=collision_scene,
            progress_cb=lambda _phase, _progress: None,
        )
        candidates = None
        grasp_error = None
        try:
            candidates = AntipodalGraspSource(
                min_aperture_m=0.012,
                max_aperture_m=0.068,
                max_candidates=64,
                approach_samples=8,
                contact_angle_deg=55.0,
            ).generate(context)
        except GraspGenerationError as error:
            grasp_error = str(error)
        grasp_at = time.perf_counter()

        # Include representative immutable artifact serialization.  The files
        # live only in a temporary directory and are never fed to a runtime.
        with tempfile.TemporaryDirectory(prefix="z-mobile-perception-bench-") as temp:
            root = Path(temp)
            mask_u8 = np.asarray(mask, dtype=np.uint8)
            if mask_u8.max(initial=0) <= 1:
                mask_u8 *= 255
            cv2.imwrite(str(root / "overlay.png"), overlay)
            cv2.imwrite(str(root / "mask.png"), mask_u8)
            np.save(root / "target_points.npy", filtered)
            np.save(root / "scene_collision_points.npy", collision_scene)
            if candidates is not None:
                widths = getattr(candidates, "widths", None)
                if widths is None:
                    widths = np.zeros(len(candidates.grasps), dtype=np.float32)
                np.savez_compressed(
                    root / "grasp_candidates.npz",
                    grasps=np.asarray(candidates.grasps, dtype=np.float64),
                    scores=np.asarray(candidates.scores, dtype=np.float32),
                    widths=np.asarray(widths, dtype=np.float32),
                )
        finished = time.perf_counter()
        measurements.append({
            "decode_s": decoded_at - started,
            "filter_s": filtered_at - decoded_at,
            "grasp_generation_s": grasp_at - filtered_at,
            "artifact_write_s": finished - grasp_at,
            "total_s": finished - started,
            "target_points": int(len(target)),
            "filtered_target_points": int(len(filtered)),
            "scene_points": int(len(scene)),
            "collision_scene_points": int(len(collision_scene)),
            "grasp_candidates": 0 if candidates is None else int(len(candidates.grasps)),
            "grasp_error": grasp_error,
        })

    return {
        "repeats": repeats,
        "decode": summarize_seconds(item["decode_s"] for item in measurements),
        "filter": summarize_seconds(item["filter_s"] for item in measurements),
        "grasp_generation": summarize_seconds(
            item["grasp_generation_s"] for item in measurements
        ),
        "artifact_write": summarize_seconds(
            item["artifact_write_s"] for item in measurements
        ),
        "total": summarize_seconds(item["total_s"] for item in measurements),
        "sample": measurements[-1],
    }


def benchmark_bag(
    bag_path: Path,
    *,
    maximum_cpu_bundles: int,
    cpu_repeats: int,
) -> dict[str, Any]:
    modules = _load_ros_modules()
    rosbag2_py = modules["rosbag2_py"]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    message_types = {
        item.name: modules["get_message"](item.type)
        for item in reader.get_all_topics_and_types()
        if item.name in READ_TOPICS
    }
    missing = sorted(READ_TOPICS - set(message_types))
    if missing:
        raise RuntimeError(f"bag lacks required perception topics: {missing}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(READ_TOPICS)))

    requests: OrderedDict[str, dict[str, Any]] = OrderedDict()
    active_contract = None
    reuse_eligible_requests = 0
    reuse_unresolved = 0
    fresh_rows: dict[str, dict[str, Any]] = {}
    reuse_rows: dict[str, dict[str, Any]] = {}
    status_by_observation: dict[int, tuple[Any, int]] = {}
    bundles: OrderedDict[int, dict[str, Any]] = OrderedDict()
    cpu_slots: list[tuple[int, dict[str, Any]]] = []
    cpu_slot_stamps: set[int] = set()
    raw_stamps: dict[str, list[int]] = {
        RAW_COLOR_TOPIC: [],
        RAW_DEPTH_TOPIC: [],
        RAW_INFO_TOPIC: [],
    }
    topic_counts = {topic: 0 for topic in READ_TOPICS}
    started = time.perf_counter()

    def resolve(stamp_ns: int, slot: dict[str, Any]) -> None:
        if not _bundle_complete(slot):
            return
        manifest = slot["messages"][MANIFEST_TOPIC]
        frame_id = str(slot["messages"][TARGET_CLOUD_TOPIC].header.frame_id)
        available_ns = max(slot["record_times_ns"].values())
        status_entry = status_by_observation.get(stamp_ns)
        if status_entry is not None:
            contract, status_record_ns = status_entry
            request = requests.get(contract.request_id)
            if request is not None and contract.accepts_bundle(
                manifest,
                stamp_ns=stamp_ns,
                frame_id=frame_id,
            ):
                available_ns = max(available_ns, status_record_ns)
                if contract.request_id not in fresh_rows:
                    fresh_rows[contract.request_id] = {
                        "request_id": contract.request_id,
                        "instruction": request["instruction"],
                        "request_record_unix_ns": request["record_timestamp_ns"],
                        "bundle_stamp_ns": stamp_ns,
                        "bundle_available_unix_ns": available_ns,
                        "latency_s": (
                            available_ns - request["record_timestamp_ns"]
                        ) * 1e-9,
                    }
                    if (
                        maximum_cpu_bundles > 0
                        and len(cpu_slots) < maximum_cpu_bundles
                        and stamp_ns not in cpu_slot_stamps
                    ):
                        cpu_slots.append((stamp_ns, slot))
                        cpu_slot_stamps.add(stamp_ns)

    while reader.has_next():
        topic, serialized, record_timestamp_ns = reader.read_next()
        topic_counts[topic] += 1
        message = modules["deserialize_message"](serialized, message_types[topic])
        if topic in raw_stamps:
            raw_stamps[topic].append(_stamp_ns(message))
            continue
        if topic == REQUEST_TOPIC:
            request = _request_document(message)
            if request is None:
                continue
            request_id = str(request["request_id"])
            instruction = str(request["instruction"])
            instruction_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            requests[request_id] = {
                **request,
                "record_timestamp_ns": int(record_timestamp_ns),
                "instruction_sha256": instruction_sha256,
            }
            if (
                active_contract is not None
                and active_contract.instruction_sha256 == instruction_sha256
            ):
                reuse_eligible_requests += 1
                accepted = None
                for stamp_ns, slot in reversed(bundles.items()):
                    if not _bundle_complete(slot):
                        continue
                    manifest = slot["messages"][MANIFEST_TOPIC]
                    frame_id = str(
                        slot["messages"][TARGET_CLOUD_TOPIC].header.frame_id
                    )
                    if active_contract.accepts_bundle(
                        manifest,
                        stamp_ns=stamp_ns,
                        frame_id=frame_id,
                    ):
                        accepted = (stamp_ns, slot)
                        break
                if accepted is None:
                    reuse_unresolved += 1
                else:
                    stamp_ns, slot = accepted
                    available_ns = max(slot["record_times_ns"].values())
                    reuse_rows[request_id] = {
                        "request_id": request_id,
                        "instruction": instruction,
                        "source_request_id": active_contract.request_id,
                        "bundle_stamp_ns": stamp_ns,
                        "bundle_available_unix_ns": available_ns,
                        "request_record_unix_ns": int(record_timestamp_ns),
                        "cached_bundle_age_s": max(
                            0.0,
                            (int(record_timestamp_ns) - available_ns) * 1e-9,
                        ),
                        "recorded_ready_at_request": True,
                    }
            continue
        if topic == STATUS_TOPIC:
            for values in _status_values(message):
                contract = _new_contract(values)
                if contract is None:
                    continue
                active_contract = contract
                status_by_observation[contract.observation_stamp_ns] = (
                    contract,
                    int(record_timestamp_ns),
                )
                slot = bundles.get(contract.observation_stamp_ns)
                if slot is not None:
                    resolve(contract.observation_stamp_ns, slot)
            continue
        if topic == VALID_TOPIC:
            continue

        if topic == MANIFEST_TOPIC:
            manifest = _manifest_document(message)
            if manifest is None:
                continue
            stamp_ns = int(manifest["result_stamp_ns"])
            stored_message: object = manifest
        else:
            stamp_ns = _stamp_ns(message)
            stored_message = message
        slot = _bounded_bundle_insert(
            bundles,
            stamp_ns,
            topic,
            stored_message,
            int(record_timestamp_ns),
        )
        resolve(stamp_ns, slot)

    scan_elapsed_s = time.perf_counter() - started
    fresh_latencies = [row["latency_s"] for row in fresh_rows.values()]
    reuse_bundle_ages = [row["cached_bundle_age_s"] for row in reuse_rows.values()]
    reuse_within_half_second = sum(age <= 0.5 for age in reuse_bundle_ages)
    color_set = set(raw_stamps[RAW_COLOR_TOPIC])
    depth_set = set(raw_stamps[RAW_DEPTH_TOPIC])
    info_set = set(raw_stamps[RAW_INFO_TOPIC])

    cpu_results = []
    for stamp_ns, slot in cpu_slots:
        cpu_results.append({
            "stamp_ns": stamp_ns,
            **_benchmark_bundle(slot, cpu_repeats),
        })
    cpu_totals = [
        item["total"]["p50_s"]
        for item in cpu_results
        if item["total"]["p50_s"] is not None
    ]
    return {
        "schema": REPORT_SCHEMA,
        "read_only": True,
        "offline": True,
        "ros_publishers_created": 0,
        "rclpy_initialized": False,
        "robot_drivers_imported": False,
        "bag_path": str(bag_path.resolve()),
        "scan_elapsed_s": round(scan_elapsed_s, 6),
        "topic_counts": topic_counts,
        "raw_rgbd_sync": {
            "color_frames": len(color_set),
            "depth_frames": len(depth_set),
            "camera_info_frames": len(info_set),
            "exact_color_depth": len(color_set & depth_set),
            "exact_color_depth_info": len(color_set & depth_set & info_set),
            "color_to_depth_nearest": summarize_seconds(
                nearest_stamp_deltas_s(color_set, depth_set)
            ),
            "color_to_info_nearest": summarize_seconds(
                nearest_stamp_deltas_s(color_set, info_set)
            ),
        },
        "recorded_fresh": {
            "requests": len(requests),
            "exact_request_bundles": len(fresh_rows),
            "latency": summarize_seconds(fresh_latencies),
            "samples": list(fresh_rows.values()),
        },
        "tracked_counterfactual": {
            "eligible_same_instruction_requests": reuse_eligible_requests,
            "exact_cached_identity_bundles": len(reuse_rows),
            "cached_bundle_age_at_most_0_5_s": reuse_within_half_second,
            "unresolved": reuse_unresolved,
            "cached_bundle_age": summarize_seconds(reuse_bundle_ages),
            "samples": list(reuse_rows.values()),
        },
        "cpu_postprocess_replay": {
            "bundles": len(cpu_results),
            "repeats_per_bundle": cpu_repeats,
            "bundle_p50_total": summarize_seconds(cpu_totals),
            "samples": cpu_results,
        },
        "evidence_scope": {
            "fresh": (
                "recorded request-to-first exact six-artifact bundle; includes "
                "recorded grounding, tracking, and transport delay"
            ),
            "tracked": (
                "counterfactual immediate reuse of a complete bundle cached "
                "before the request; the production exact identity contract "
                "is applied and cache age is reported, not invented latency"
            ),
            "cpu": (
                "local replay of decode, filtering, target exclusion, antipodal "
                "proposal, and representative artifact writes; excludes API, "
                "container launch, YOLOE, and EdgeTAM inference"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cpu-bundles", type=int, default=3)
    parser.add_argument("--cpu-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.maximum_cpu_bundles < 0 or args.cpu_repeats < 1:
        parser.error("maximum-cpu-bundles must be >= 0 and cpu-repeats must be >= 1")
    report = benchmark_bag(
        args.bag,
        maximum_cpu_bundles=args.maximum_cpu_bundles,
        cpu_repeats=args.cpu_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
