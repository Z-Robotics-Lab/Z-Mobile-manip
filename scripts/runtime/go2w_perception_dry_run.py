#!/usr/bin/env python3
"""Ground, track, and generate grasp candidates without actuator publishers."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import cv2
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, String

from z_manip.models.antipodal_grasp import AntipodalGraspSource
from z_manip.models.grasp_source import (
    GEOMETRY_DENSE,
    GraspContext,
    GraspGenerationError,
    select_grasp_source,
)
from z_manip.perception.rgbd import (
    CameraIntrinsics,
    filter_object_cloud,
    target_exclusion_mask,
)
from z_manip.verification.passive_capture import validate_passive_capture


def _stamp_ns(message: object) -> int:
    header = getattr(message, "header")
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def _bounded_insert(cache: OrderedDict[int, object], key: int, value: object) -> None:
    cache[key] = value
    while len(cache) > 240:
        cache.popitem(last=False)


def _freshness_summary(samples_s: list[float]) -> dict[str, float | int | None]:
    """Summarize observed tracker-result lag against the latest camera stamp."""
    if not samples_s:
        return {
            "sample_count": 0,
            "max_lag_s": None,
            "p50_lag_s": None,
            "p95_lag_s": None,
            "p99_lag_s": None,
        }
    values = np.asarray(samples_s, dtype=np.float64)
    return {
        "sample_count": int(values.size),
        "max_lag_s": round(float(values.max()), 6),
        "p50_lag_s": round(float(np.percentile(values, 50)), 6),
        "p95_lag_s": round(float(np.percentile(values, 95)), 6),
        "p99_lag_s": round(float(np.percentile(values, 99)), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--passive-window",
        type=Path,
        help="live zero-TX passive joint report used to time-select the RGB-D bundle",
    )
    parser.add_argument(
        "--selected-passive-window",
        type=Path,
        help="immutable copy of the passive report matched to the selected bundle",
    )
    parser.add_argument("--timeout", type=float, default=105.0)
    parser.add_argument(
        "--min-bundle-target-points",
        type=int,
        default=40,
        help=(
            "wait for this many depth-supported target points before freezing "
            "the first tracker bundle"
        ),
    )
    parser.add_argument("--request-id", default="")
    parser.add_argument(
        "--reuse-valid-tracking",
        action="store_true",
        help=(
            "reuse a fresh persistent EdgeTAM track only when the bridge's "
            "instruction hash exactly matches this request"
        ),
    )
    parser.add_argument("--learned-endpoint", default="")
    parser.add_argument("--soak-duration", type=float, default=0.0)
    parser.add_argument("--max-recoveries", type=int, default=0)
    parser.add_argument(
        "--max-observed-result-lag",
        type=float,
        default=1.5,
        help=(
            "fail a soak if an observed valid result trails the latest camera "
            "stamp by more than this many seconds"
        ),
    )
    parser.add_argument(
        "--recovery-timeout",
        type=float,
        default=25.0,
        help="maximum wait for each recovery attempt before issuing the next one",
    )
    parser.add_argument(
        "--fallback-contact-angle-deg",
        type=float,
        default=55.0,
        help="proposal-only CPU fallback normal tolerance; candidates are never executed",
    )
    parser.add_argument(
        "--target-exclusion-radius-m",
        type=float,
        default=0.012,
        help=(
            "3-D target/scene decontamination radius applied after pixel-mask "
            "dilation; prevents intended finger contact from reappearing as "
            "an environment collision"
        ),
    )
    parser.add_argument(
        "--allow-no-grasp-candidates",
        action="store_true",
        help=(
            "keep a perception soak successful when the current frame has no "
            "geometric grasp proposal; the report still records the failure"
        ),
    )
    args = parser.parse_args()
    instruction = args.instruction.strip()
    if (
        not instruction
        or args.timeout <= 5.0
        or args.min_bundle_target_points < 40
        or args.max_recoveries < 0
        or args.recovery_timeout <= 5.0
        or not math.isfinite(args.max_observed_result_lag)
        or args.max_observed_result_lag <= 0.0
        or not 1.0 <= args.fallback_contact_angle_deg <= 89.0
        or not math.isfinite(args.target_exclusion_radius_m)
        or args.target_exclusion_radius_m <= 0.0
    ):
        parser.error(
            "instruction must be non-empty; timeouts must exceed 5 s; "
            "minimum bundle target points must be at least 40; "
            "max recoveries must be non-negative; observed result lag must be "
            "positive; fallback contact angle must be in [1, 89] degrees; "
            "and target exclusion radius must be positive"
        )
    if (args.passive_window is None) != (args.selected_passive_window is None):
        parser.error("passive-window and selected-passive-window must be provided together")
    request_id = args.request_id.strip() or f"dry-run-{uuid.uuid4().hex[:16]}"
    args.output.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Node("go2w_perception_grasp_read_only_dry_run")
    bridge = CvBridge()
    overlays: OrderedDict[int, Image] = OrderedDict()
    masks: OrderedDict[int, Image] = OrderedDict()
    clouds: OrderedDict[int, PointCloud2] = OrderedDict()
    scene_clouds: OrderedDict[int, PointCloud2] = OrderedDict()
    manifests: OrderedDict[int, dict[str, object]] = OrderedDict()
    infos: OrderedDict[int, CameraInfo] = OrderedDict()
    valid = False
    valid_transitions: list[tuple[float, bool]] = []
    message_counts = {
        "overlay": 0,
        "mask": 0,
        "cloud": 0,
        "scene_cloud": 0,
        "manifest": 0,
        "info": 0,
        "status": 0,
    }
    perception_failure = ""
    largest_bundle_target_points = 0
    instruction_sha256 = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    matching_tracking_request_id = ""
    matching_tracking_valid = False
    accepted_source_request_id = ""

    def valid_callback(message: Bool) -> None:
        nonlocal valid
        value = bool(message.data)
        if value != valid:
            valid = value
            valid_transitions.append((time.monotonic(), value))

    def cache_callback(name: str, cache: OrderedDict[int, object]):
        def callback(message: object) -> None:
            message_counts[name] += 1
            _bounded_insert(cache, _stamp_ns(message), message)

        return callback

    def manifest_callback(message: String) -> None:
        message_counts["manifest"] += 1
        try:
            value = json.loads(message.data)
            stamp = int(value["result_stamp_ns"])
            depth_filter = value["depth_filter"]
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if (
            not isinstance(value, dict)
            or value.get("schema") != "z_manip.tracker_frame.v1"
            or not isinstance(depth_filter, dict)
            or depth_filter.get("method") != "motion_adaptive_temporal_median"
            or depth_filter.get("applied_to")
            != ["target_pointcloud", "scene_pointcloud"]
        ):
            return
        _bounded_insert(manifests, stamp, value)

    def status_callback(message: DiagnosticArray) -> None:
        nonlocal perception_failure
        nonlocal matching_tracking_request_id, matching_tracking_valid
        message_counts["status"] += 1
        for status in message.status:
            values = {item.key: item.value for item in status.values}
            if values.get("schema") != "z_manip.perception_status.v1":
                continue
            status_request_id = values.get("request_id", "")
            instruction_matches = (
                values.get("instruction_sha256") == instruction_sha256
            )
            matching_tracking_valid = bool(
                instruction_matches and values.get("valid") == "true"
            )
            matching_tracking_request_id = (
                status_request_id if matching_tracking_valid else ""
            )
            failure = values.get("failure", "").strip()
            failure_matches = status_request_id == request_id or (
                accepted_source_request_id
                and status_request_id == accepted_source_request_id
            )
            if failure_matches and status.level == DiagnosticStatus.ERROR and failure:
                detail = values.get("failure_detail", "").strip()
                perception_failure = failure + (f": {detail}" if detail else "")

    latched_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    node.create_subscription(
        Bool,
        "/z_manip/perception/valid",
        valid_callback,
        latched_qos,
    )
    node.create_subscription(
        Image,
        "/z_manip/perception/overlay",
        cache_callback("overlay", overlays),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        Image,
        "/z_manip/perception/target_mask",
        cache_callback("mask", masks),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        PointCloud2,
        "/z_manip/perception/target_pointcloud",
        cache_callback("cloud", clouds),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        PointCloud2,
        "/z_manip/perception/scene_pointcloud",
        cache_callback("scene_cloud", scene_clouds),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        String,
        "/track_3d/frame_manifest",
        manifest_callback,
        qos_profile_sensor_data,
    )
    node.create_subscription(
        CameraInfo,
        "/camera/color/camera_info",
        cache_callback("info", infos),
        qos_profile_sensor_data,
    )
    node.create_subscription(
        DiagnosticArray,
        "/z_manip/perception/status",
        status_callback,
        latched_qos,
    )
    request_qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    publisher = node.create_publisher(
        String,
        "/z_manip/grounding/request",
        request_qos,
    )
    stage_timings: dict[str, float] = {}
    stage_started = time.monotonic()
    connection_deadline = time.monotonic() + 5.0
    while publisher.get_subscription_count() == 0 and time.monotonic() < connection_deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if publisher.get_subscription_count() == 0:
        raise RuntimeError("grounding request has no subscriber")
    stage_timings["subscriber_discovery_s"] = round(
        time.monotonic() - stage_started,
        6,
    )
    def publish_grounding(active_request_id: str) -> None:
        request = {
            "schema": "z_manip.grounding_request.v2",
            "request_id": active_request_id,
            "instruction": instruction,
            "scope": "grasp_only",
        }
        publisher.publish(String(data=json.dumps(request, separators=(",", ":"))))

    started = time.monotonic()
    deadline = started + args.timeout
    grounding_reused = False
    if args.reuse_valid_tracking:
        # Status is transient-local, so a late-joining warm runner receives the
        # current instruction hash immediately.  A single short discovery spin
        # is enough; mismatches publish fresh grounding without a fixed 200 ms
        # delay on every first request.
        reuse_probe_started = time.monotonic()
        reuse_deadline = min(deadline, started + 0.05)
        while time.monotonic() < reuse_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if matching_tracking_valid and matching_tracking_request_id:
                grounding_reused = True
                accepted_source_request_id = matching_tracking_request_id
                break
        stage_timings["tracking_reuse_probe_s"] = round(
            time.monotonic() - reuse_probe_started,
            6,
        )
    if not grounding_reused:
        overlays.clear()
        masks.clear()
        clouds.clear()
        scene_clouds.clear()
        manifests.clear()
        valid = False
        accepted_source_request_id = request_id
        publish_grounding(request_id)

    bundle_wait_started = time.monotonic()
    selected_stamp: int | None = None
    selected_passive_report: dict[str, object] | None = None
    passive_window_error = "waiting for first passive capture window"
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if perception_failure:
            break
        if valid:
            common = (
                overlays.keys()
                & masks.keys()
                & clouds.keys()
                & scene_clouds.keys()
                & manifests.keys()
                & infos.keys()
            )
            if common:
                supported = []
                for stamp in common:
                    point_count = int(
                        getattr(clouds[stamp], "width", 0)
                        * getattr(clouds[stamp], "height", 1)
                    )
                    largest_bundle_target_points = max(
                        largest_bundle_target_points,
                        point_count,
                    )
                    if point_count >= args.min_bundle_target_points:
                        supported.append(stamp)
                if not supported:
                    passive_window_error = (
                        "waiting for depth-supported target bundle "
                        f"({largest_bundle_target_points}/"
                        f"{args.min_bundle_target_points} points)"
                    )
                    continue
                if args.passive_window is None:
                    selected_stamp = max(supported)
                    break
                try:
                    candidate_report = json.loads(
                        args.passive_window.read_text(encoding="utf-8"),
                    )
                    capture = validate_passive_capture(candidate_report)
                    eligible = [
                        stamp
                        for stamp in supported
                        if capture.start_unix_ns - 250_000_000
                        <= stamp
                        <= capture.end_unix_ns + 250_000_000
                    ]
                    if eligible:
                        selected_stamp = min(
                            eligible,
                            key=lambda stamp: abs(stamp - capture.midpoint_unix_ns),
                        )
                        selected_passive_report = candidate_report
                        break
                    passive_window_error = (
                        "no exact perception bundle overlaps the latest passive window"
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as error:
                    passive_window_error = f"{type(error).__name__}: {error}"
    if selected_stamp is None:
        stage_timings["bundle_wait_s"] = round(
            time.monotonic() - bundle_wait_started,
            6,
        )
        report = {
            "read_only": True,
            "request_id": request_id,
            "source_grounding_request_id": accepted_source_request_id,
            "grounding_reused": grounding_reused,
            "instruction": instruction,
            "elapsed_s": round(time.monotonic() - started, 3),
            "stage": "perception_bundle",
            "perception_bundle_valid": False,
            "error": (
                "perception did not produce an exact valid "
                "overlay/mask/filtered target/filtered scene/manifest/K bundle"
            ),
            "message_counts": message_counts,
            "valid_transitions": [
                [round(transition_at - started, 3), value]
                for transition_at, value in valid_transitions
            ],
            "passive_window_required": args.passive_window is not None,
            "passive_window_error": passive_window_error,
            "perception_failure": perception_failure or None,
            "minimum_bundle_target_points": args.min_bundle_target_points,
            "largest_bundle_target_points": largest_bundle_target_points,
            "timings": stage_timings,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        node.destroy_node()
        rclpy.shutdown()
        return 5 if perception_failure else 2
    passive_capture_summary: dict[str, object] | None = None
    if selected_passive_report is not None:
        assert args.selected_passive_window is not None
        encoded_report = (
            json.dumps(selected_passive_report, indent=2, sort_keys=True) + "\n"
        )
        args.selected_passive_window.write_text(encoded_report, encoding="utf-8")
        capture = validate_passive_capture(selected_passive_report)
        passive_capture_summary = {
            "synchronized": True,
            "observation_start_unix_ns": capture.start_unix_ns,
            "observation_end_unix_ns": capture.end_unix_ns,
            "selected_stamp_ns": selected_stamp,
            "selected_offset_from_midpoint_s": round(
                abs(selected_stamp - capture.midpoint_unix_ns) * 1e-9,
                6,
            ),
            "report_sha256": hashlib.sha256(encoded_report.encode()).hexdigest(),
        }
    selected_at = time.monotonic()
    stage_timings["bundle_wait_s"] = round(
        selected_at - bundle_wait_started,
        6,
    )
    counts_at_selection = dict(message_counts)
    freshness_samples_s = [
        max(0, max(infos) - selected_stamp) * 1e-9,
    ]
    latest_freshness_stamp = selected_stamp

    postprocess_started = time.monotonic()
    overlay_message = overlays[selected_stamp]
    mask_message = masks[selected_stamp]
    cloud_message = clouds[selected_stamp]
    scene_cloud_message = scene_clouds[selected_stamp]
    frame_manifest = manifests[selected_stamp]
    info_message = infos[selected_stamp]
    overlay = bridge.imgmsg_to_cv2(overlay_message, desired_encoding="bgr8")
    mask = bridge.imgmsg_to_cv2(mask_message, desired_encoding="passthrough")
    temporal_depth = frame_manifest["depth_filter"]
    points = np.asarray(
        point_cloud2.read_points_numpy(
            cloud_message,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    camera = CameraIntrinsics(
        fx=float(info_message.k[0]),
        fy=float(info_message.k[4]),
        cx=float(info_message.k[2]),
        cy=float(info_message.k[5]),
        width=int(info_message.width),
        height=int(info_message.height),
    )
    if scene_cloud_message.header.frame_id != cloud_message.header.frame_id:
        raise RuntimeError("filtered target and scene clouds use different frames")
    scene_points = np.asarray(
        point_cloud2.read_points_numpy(
            scene_cloud_message,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    stage_timings["bundle_decode_s"] = round(
        time.monotonic() - postprocess_started,
        6,
    )
    filter_started = time.monotonic()
    filtered_points = filter_object_cloud(
        points,
        viewpoint=(0.0, 0.0, 0.0),
    )
    pixel_excluded_scene_points = np.ascontiguousarray(scene_points, dtype=np.float32)
    geometric_target_labels = target_exclusion_mask(
        pixel_excluded_scene_points,
        filtered_points,
        radius_m=args.target_exclusion_radius_m,
    )
    collision_scene_points = np.ascontiguousarray(
        pixel_excluded_scene_points[~geometric_target_labels],
        dtype=np.float32,
    )
    stage_timings["pointcloud_filter_s"] = round(
        time.monotonic() - filter_started,
        6,
    )
    context = GraspContext(
        object_points=filtered_points,
        bbox=None,
        source_frame=cloud_message.header.frame_id,
        t_target_src=np.eye(4),
        scene_points=collision_scene_points,
        progress_cb=lambda _phase, _progress: None,
    )
    backend = "antipodal"
    learned_error = ""
    grasp_error = ""
    candidates = None
    grasp_started = time.monotonic()
    if args.learned_endpoint.strip():
        import os

        os.environ["Z_MANIP_DENSE_PROVIDER"] = "anygrasp"
        os.environ["Z_MANIP_DENSE_ENDPOINT"] = args.learned_endpoint.strip()
        os.environ["Z_MANIP_DENSE_TIMEOUT_S"] = "10.0"
        os.environ["Z_MANIP_DENSE_MAX_GRASPS"] = "32"
        try:
            candidates = select_grasp_source(GEOMETRY_DENSE).generate(context)
            backend = "anygrasp"
        except GraspGenerationError as error:
            learned_error = str(error)
    if candidates is None:
        try:
            candidates = AntipodalGraspSource(
                min_aperture_m=0.012,
                max_aperture_m=0.068,
                max_candidates=64,
                approach_samples=8,
                contact_angle_deg=args.fallback_contact_angle_deg,
            ).generate(context)
        except GraspGenerationError as error:
            grasp_error = str(error)
    stage_timings["grasp_generation_s"] = round(
        time.monotonic() - grasp_started,
        6,
    )

    visualization_started = time.monotonic()
    visual = overlay.copy()
    intrinsic = np.asarray(info_message.k, dtype=float).reshape(3, 3)
    colors = ((0, 255, 255), (0, 200, 0), (255, 120, 0), (255, 0, 255), (0, 120, 255))

    def project(point: np.ndarray) -> tuple[int, int] | None:
        if point[2] <= 1e-6:
            return None
        pixel = intrinsic @ point
        return tuple(np.round(pixel[:2] / pixel[2]).astype(int))

    shown: list[dict[str, object]] = []
    if candidates is not None:
        order = np.argsort(-np.asarray(candidates.scores))
        for rank, index in enumerate(order[:3]):
            pose = np.asarray(candidates.grasps[index], dtype=float)
            origin = pose[:3, 3]
            closing = pose[:3, 0]
            approach = pose[:3, 2]
            width = float(candidates.widths[index])
            pixels = (
                project(origin),
                project(origin - 0.07 * approach),
                project(origin - 0.5 * width * closing),
                project(origin + 0.5 * width * closing),
            )
            if any(pixel is None for pixel in pixels):
                continue
            center, pregrasp, left, right = pixels
            color = colors[rank]
            cv2.arrowedLine(visual, pregrasp, center, color, 2, tipLength=0.25)
            cv2.line(visual, left, right, color, 2)
            cv2.circle(visual, center, 4, color, -1)
            cv2.putText(
                visual,
                f"#{rank + 1}",
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            shown.append(
                {
                    "rank": rank + 1,
                    "score": float(candidates.scores[index]),
                    "width_m": width,
                    "pose": pose.tolist(),
                },
            )

    banner = visual.copy()
    cv2.rectangle(banner, (0, visual.shape[0] - 54), (visual.shape[1], visual.shape[0]), (0, 0, 0), -1)
    visual = cv2.addWeighted(banner, 0.68, visual, 0.32, 0.0)
    cv2.putText(
        visual,
        (
            f"READ-ONLY {backend} candidates in camera frame; no IK/collision/execution"
            if candidates is not None
            else "READ-ONLY perception OK; no grasp proposal in this frame"
        ),
        (10, visual.shape[0] - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    legend = (
        "  ".join(
            f"#{item['rank']} score={item['score']:.3f} width={item['width_m'] * 1000:.1f}mm"
            for item in shown
        )
        if candidates is not None
        else grasp_error[:90]
    )
    cv2.putText(
        visual,
        legend,
        (10, visual.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 255, 210),
        1,
        cv2.LINE_AA,
    )

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.max(initial=0) <= 1:
        mask_u8 *= 255
    stage_timings["visualization_s"] = round(
        time.monotonic() - visualization_started,
        6,
    )
    artifact_write_started = time.monotonic()
    cv2.imwrite(str(args.output / "edgetam_overlay.png"), overlay)
    cv2.imwrite(str(args.output / "edgetam_mask.png"), mask_u8)
    cv2.imwrite(str(args.output / "grasp_candidates_overlay.png"), visual)
    np.save(args.output / "target_points.npy", filtered_points)
    np.save(args.output / "scene_collision_points.npy", collision_scene_points)
    if candidates is not None:
        np.savez_compressed(
            args.output / "grasp_candidates.npz",
            grasps=np.asarray(candidates.grasps, dtype=np.float64),
            scores=np.asarray(candidates.scores, dtype=np.float32),
            widths=(
                np.empty((0,), dtype=np.float32)
                if candidates.widths is None
                else np.asarray(candidates.widths, dtype=np.float32)
            ),
            centroid=np.asarray(candidates.centroid, dtype=np.float64),
            frame=np.asarray(candidates.frame),
            num_raw=np.asarray(candidates.num_raw, dtype=np.int64),
            stamp_ns=np.asarray(selected_stamp, dtype=np.int64),
        )
    stage_timings["artifact_write_s"] = round(
        time.monotonic() - artifact_write_started,
        6,
    )
    report = {
        "read_only": True,
        "request_id": request_id,
        "source_grounding_request_id": accepted_source_request_id,
        "grounding_reused": grounding_reused,
        "instruction": instruction,
        "elapsed_s": round(time.monotonic() - started, 3),
        "frame": candidates.frame if candidates is not None else cloud_message.header.frame_id,
        "stamp_ns": selected_stamp,
        "input_points": len(points),
        "minimum_bundle_target_points": args.min_bundle_target_points,
        "largest_bundle_target_points": largest_bundle_target_points,
        "filtered_target_points": len(filtered_points),
        "scene_points_total": len(scene_points),
        "scene_target_excluded_points": None,
        "scene_source": "edgetam_motion_adaptive_filtered_scene",
        "scene_target_geometric_excluded_points": int(
            np.count_nonzero(geometric_target_labels)
        ),
        "scene_target_exclusion_radius_m": args.target_exclusion_radius_m,
        "scene_collision_points": len(collision_scene_points),
        "temporal_depth_filter": temporal_depth,
        "grasp_backend": backend,
        "fallback_contact_angle_deg": args.fallback_contact_angle_deg,
        "learned_backend_error": learned_error,
        "grasp_generation_valid": candidates is not None,
        "grasp_generation_error": grasp_error,
        "raw_grasp_hypotheses": candidates.num_raw if candidates is not None else 0,
        "grasp_candidates": len(candidates.grasps) if candidates is not None else 0,
        "shown": shown,
        "result_freshness": _freshness_summary(freshness_samples_s),
        "max_observed_result_lag_s": args.max_observed_result_lag,
        "passive_capture": passive_capture_summary,
        "timings": stage_timings,
    }
    if args.soak_duration > 0.0:
        soak_deadline = time.monotonic() + args.soak_duration
        recovery_events: list[dict[str, object]] = []
        active_recovery: dict[str, object] | None = None
        recovery_exhausted = False
        latest_valid_stamp = selected_stamp
        while time.monotonic() < soak_deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if valid:
                common = (
                    overlays.keys()
                    & masks.keys()
                    & clouds.keys()
                    & scene_clouds.keys()
                    & manifests.keys()
                    & infos.keys()
                )
                if common:
                    newest_bundle_stamp = max(common)
                    if newest_bundle_stamp > latest_freshness_stamp:
                        freshness_samples_s.append(
                            max(0, max(infos) - newest_bundle_stamp) * 1e-9,
                        )
                        latest_freshness_stamp = newest_bundle_stamp
            if valid and active_recovery is not None:
                common = (
                    overlays.keys()
                    & masks.keys()
                    & clouds.keys()
                    & scene_clouds.keys()
                    & manifests.keys()
                    & infos.keys()
                )
                newer = [stamp for stamp in common if stamp > latest_valid_stamp]
                if newer:
                    latest_valid_stamp = max(newer)
                    active_recovery["recovered_at_s"] = round(now - started, 3)
                    active_recovery["recovery_latency_s"] = round(
                        now - float(active_recovery["requested_at_monotonic"]),
                        3,
                    )
                    active_recovery["recovered_stamp_ns"] = latest_valid_stamp
                    active_recovery.pop("requested_at_monotonic", None)
                    active_recovery = None
            elif not valid and active_recovery is None and not recovery_exhausted:
                if len(recovery_events) >= args.max_recoveries:
                    recovery_exhausted = True
                else:
                    recovery_number = len(recovery_events) + 1
                    recovery_request_id = (
                        f"{request_id}-recovery-{recovery_number}-{uuid.uuid4().hex[:8]}"
                    )
                    event: dict[str, object] = {
                        "number": recovery_number,
                        "request_id": recovery_request_id,
                        "requested_at_s": round(now - started, 3),
                        "requested_at_monotonic": now,
                        "after_stamp_ns": latest_valid_stamp,
                    }
                    recovery_events.append(event)
                    active_recovery = event
                    publish_grounding(recovery_request_id)
            if (
                active_recovery is not None
                and now - float(active_recovery["requested_at_monotonic"])
                > args.recovery_timeout
            ):
                active_recovery["error"] = "recovery timeout"
                active_recovery.pop("requested_at_monotonic", None)
                active_recovery = None
                if len(recovery_events) >= args.max_recoveries:
                    recovery_exhausted = True
        lost_after_selection = any(
            not value and transition_at >= selected_at
            for transition_at, value in valid_transitions
        )
        loss_active = False
        recovered_losses = 0
        for transition_at, value in valid_transitions:
            if transition_at < selected_at:
                continue
            if not value:
                loss_active = True
            elif loss_active:
                recovered_losses += 1
                loss_active = False
        all_losses_recovered = not loss_active
        for event in recovery_events:
            event.pop("requested_at_monotonic", None)
        freshness = _freshness_summary(freshness_samples_s)
        max_lag_s = freshness["max_lag_s"]
        freshness_within_limit = bool(
            isinstance(max_lag_s, float)
            and max_lag_s <= args.max_observed_result_lag
        )
        report["result_freshness"] = freshness
        report["soak"] = {
            "duration_s": args.soak_duration,
            "stable": bool(
                valid and all_losses_recovered and freshness_within_limit
            ),
            "uninterrupted": not lost_after_selection,
            "final_valid": valid,
            "freshness_within_limit": freshness_within_limit,
            "max_observed_result_lag_s": args.max_observed_result_lag,
            "max_recoveries": args.max_recoveries,
            "recovered_losses": recovered_losses,
            "successful_recovery_attempts": sum(
                "recovered_at_s" in event for event in recovery_events
            ),
            "recovery_exhausted": recovery_exhausted,
            "recovery_events": recovery_events,
            "message_counts": {
                name: message_counts[name] - counts_at_selection[name]
                for name in message_counts
            },
            "transitions": [
                [round(transition_at - started, 3), value]
                for transition_at, value in valid_transitions
            ],
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    if args.soak_duration > 0.0 and not report["soak"]["stable"]:
        return 3
    if candidates is None and not args.allow_no_grasp_candidates:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
