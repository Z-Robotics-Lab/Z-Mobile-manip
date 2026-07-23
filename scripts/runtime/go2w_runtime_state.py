#!/usr/bin/env python3
"""Bounded runtime-state validation for the Go2W manipulation workbench.

Extracted from ``go2w_planning_control.py`` (Stage 1 of the code-modularity
plan) with no behavioral change.  Contains the schema/limit constants and the
strict validator for one producer-written runtime snapshot.  The origin module
re-imports every public name here so ``go2w_planning_control.<name>`` continues
to resolve unchanged (existing tests pin those attributes).

Public surface (re-exported by ``go2w_planning_control``):
``RuntimeStateError``, ``validate_runtime_state``, and the content-limit
constants ``RUNTIME_SCHEMA``, ``MAX_RUNTIME_POINT_CLOUDS``,
``MAX_RUNTIME_POINTS``, ``MAX_RUNTIME_CANDIDATES``, ``MAX_RUNTIME_LINKS``,
``MAX_RUNTIME_PATH_POINTS``.

NOTE: ``MAX_RUNTIME_STATE_BYTES`` deliberately stays in the origin module.  A
test monkeypatches it on the origin module and ``RuntimeStateReader`` (which
also stays) reads it as a module global there; moving it would break that patch.
"""

from __future__ import annotations

import math
from typing import Any


RUNTIME_SCHEMA = "z_manip.runtime_state.v1"
MAX_RUNTIME_POINT_CLOUDS = 8
MAX_RUNTIME_POINTS = 50_000
MAX_RUNTIME_CANDIDATES = 512
MAX_RUNTIME_LINKS = 128
MAX_RUNTIME_PATH_POINTS = 2_000


class RuntimeStateError(ValueError):
    """A runtime snapshot is unsafe or incompatible with the fixed API."""


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeStateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeStateError(f"{label} must be a finite number")
    return result


def _finite_rows(
    value: object,
    *,
    columns: int,
    maximum: int,
    label: str,
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) > maximum:
        raise RuntimeStateError(f"{label} must contain at most {maximum} rows")
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise RuntimeStateError(f"{label} rows must contain {columns} numbers")
        result.append([
            _finite_number(item, f"{label} value")
            for item in row
        ])
    return result


def _transform(value: object, label: str) -> list[list[float]]:
    rows = _finite_rows(value, columns=4, maximum=4, label=label)
    if len(rows) != 4 or any(
        abs(rows[3][index] - expected) > 1e-8
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        raise RuntimeStateError(f"{label} must be a homogeneous 4x4 transform")
    return rows


def _strict_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise RuntimeStateError(f"{label} contains unsupported fields: {sorted(unknown)}")


def validate_runtime_state(document: object) -> dict[str, Any]:
    """Validate and normalize one bounded producer-written runtime snapshot."""

    if not isinstance(document, dict):
        raise RuntimeStateError("runtime state must contain a JSON object")
    allowed = {
        "schema",
        "sequence",
        "source_timestamp_ns",
        "joint_state_available",
        "joint_positions_rad",
        "robot_links",
        "point_clouds",
        "candidates",
        "plan_overlay",
        "telemetry",
        "kinematic_transforms",
    }
    _strict_keys(document, allowed, "runtime state")
    if document.get("schema") != RUNTIME_SCHEMA:
        raise RuntimeStateError(f"unsupported runtime schema: {document.get('schema')!r}")
    sequence = document.get("sequence")
    source_timestamp = document.get("source_timestamp_ns")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise RuntimeStateError("runtime sequence must be a non-negative integer")
    if (
        isinstance(source_timestamp, bool)
        or not isinstance(source_timestamp, int)
        or source_timestamp <= 0
    ):
        raise RuntimeStateError("source_timestamp_ns must be a positive integer")
    joint_state_available = document.get("joint_state_available", True)
    if not isinstance(joint_state_available, bool):
        raise RuntimeStateError("joint_state_available must be a boolean")
    joints = document.get("joint_positions_rad")
    expected_joint_count = 6 if joint_state_available else 0
    if not isinstance(joints, list) or len(joints) != expected_joint_count:
        raise RuntimeStateError(
            "joint_positions_rad must contain exactly six joints when feedback "
            "is available and be empty otherwise"
        )
    normalized: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "sequence": sequence,
        "source_timestamp_ns": source_timestamp,
        "joint_state_available": joint_state_available,
        "joint_positions_rad": [
            _finite_number(value, "joint position")
            for value in joints
        ],
    }

    kinematic_transforms = document.get("kinematic_transforms")
    if kinematic_transforms is not None:
        if not isinstance(kinematic_transforms, dict):
            raise RuntimeStateError("kinematic_transforms must be an object")
        _strict_keys(
            kinematic_transforms,
            {
                "schema",
                "verified",
                "source",
                "source_timestamp_ns",
                "joint_source_timestamp_ns",
                "camera_frame",
                "arm_base_frame",
                "platform_base_frame",
                "arm_base_from_camera",
                "platform_base_from_camera",
                "calibration_id",
                "calibration_synthetic",
            },
            "kinematic_transforms",
        )
        if kinematic_transforms.get("schema") != "z_manip.kinematic_transforms.v1":
            raise RuntimeStateError("unsupported kinematic transform schema")
        if kinematic_transforms.get("verified") is not True:
            raise RuntimeStateError("kinematic transforms must be verified")
        if kinematic_transforms.get("calibration_synthetic") is not False:
            raise RuntimeStateError("kinematic transforms require measured calibration")
        normalized_transforms: dict[str, object] = {
            "schema": "z_manip.kinematic_transforms.v1",
            "verified": True,
            "calibration_synthetic": False,
        }
        for key in (
            "source",
            "camera_frame",
            "arm_base_frame",
            "platform_base_frame",
            "calibration_id",
        ):
            value = kinematic_transforms.get(key)
            if not isinstance(value, str) or not value or len(value) > 256:
                raise RuntimeStateError(f"kinematic_transforms {key} must be a non-empty string")
            normalized_transforms[key] = value
        for key in ("source_timestamp_ns", "joint_source_timestamp_ns"):
            value = kinematic_transforms.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeStateError(f"kinematic_transforms {key} must be a positive integer")
            normalized_transforms[key] = value
        for key in ("arm_base_from_camera", "platform_base_from_camera"):
            normalized_transforms[key] = _transform(
                kinematic_transforms.get(key),
                f"kinematic_transforms {key}",
            )
        normalized["kinematic_transforms"] = normalized_transforms

    telemetry = document.get("telemetry")
    if telemetry is not None:
        if not isinstance(telemetry, dict):
            raise RuntimeStateError("telemetry must be an object")
        allowed_telemetry = {
            "ros_domain_id",
            "read_only",
            "motion_commands_published",
            "can_opened",
            "camera_rgbd_fresh",
            "point_cloud_fresh",
            "joint_state_available",
            "joint_topic",
            "joint_publisher_count",
            "fresh_topic_count",
            "configured_topic_count",
            "depth_filter",
            "tracker",
        }
        _strict_keys(telemetry, allowed_telemetry, "telemetry")
        boolean_keys = {
            "read_only",
            "can_opened",
            "camera_rgbd_fresh",
            "point_cloud_fresh",
            "joint_state_available",
        }
        integer_keys = {
            "ros_domain_id",
            "motion_commands_published",
            "joint_publisher_count",
            "fresh_topic_count",
            "configured_topic_count",
        }
        normalized_telemetry: dict[str, object] = {}
        for key in boolean_keys:
            if key in telemetry:
                if not isinstance(telemetry[key], bool):
                    raise RuntimeStateError(f"telemetry {key} must be a boolean")
                normalized_telemetry[key] = telemetry[key]
        for key in integer_keys:
            if key in telemetry:
                value = telemetry[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RuntimeStateError(
                        f"telemetry {key} must be a non-negative integer"
                    )
                normalized_telemetry[key] = value
        if "joint_topic" in telemetry:
            topic = telemetry["joint_topic"]
            if not isinstance(topic, str) or not topic.startswith("/") or len(topic) > 128:
                raise RuntimeStateError("telemetry joint_topic must be an absolute ROS topic")
            normalized_telemetry["joint_topic"] = topic
        depth_filter = telemetry.get("depth_filter")
        if depth_filter is not None:
            if not isinstance(depth_filter, dict):
                raise RuntimeStateError("telemetry depth_filter must be an object")
            _strict_keys(
                depth_filter,
                {"available", "fresh", "report"},
                "telemetry depth_filter",
            )
            normalized_filter: dict[str, object] = {}
            for key in ("available", "fresh"):
                if not isinstance(depth_filter.get(key), bool):
                    raise RuntimeStateError(
                        f"telemetry depth_filter {key} must be a boolean",
                    )
                normalized_filter[key] = depth_filter[key]
            report = depth_filter.get("report")
            if report is None:
                if depth_filter["available"] is True:
                    raise RuntimeStateError(
                        "available depth-filter telemetry requires a report",
                    )
                normalized_filter["report"] = None
            else:
                if not isinstance(report, dict) or depth_filter["available"] is not True:
                    raise RuntimeStateError("depth-filter report availability is inconsistent")
                _strict_keys(
                    report,
                    {
                        "method",
                        "frame_count",
                        "window_size",
                        "minimum_observations",
                        "mode",
                        "reset_reason",
                        "motion_threshold_mm",
                        "global_changed_fraction",
                        "dynamic_pixels",
                        "stable_pixels",
                        "rejected_low_support_pixels",
                        "rejected_unstable_pixels",
                        "mad_p95_mm",
                        "applied_to",
                    },
                    "telemetry depth_filter report",
                )
                mode = report.get("mode")
                applied_to = report.get("applied_to")
                if (
                    report.get("method") != "motion_adaptive_temporal_median"
                    or mode not in {
                        "warmup",
                        "static_temporal",
                        "local_motion",
                        "camera_motion_reset",
                    }
                    or applied_to != ["target_pointcloud", "scene_pointcloud"]
                ):
                    raise RuntimeStateError("unsupported depth-filter report contract")
                normalized_report = dict(report)
                integer_fields = (
                    "frame_count",
                    "window_size",
                    "minimum_observations",
                    "dynamic_pixels",
                    "stable_pixels",
                    "rejected_low_support_pixels",
                    "rejected_unstable_pixels",
                )
                for key in integer_fields:
                    value = report.get(key)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise RuntimeStateError(f"depth-filter {key} is invalid")
                if (
                    not 3 <= report["window_size"] <= 64
                    or not 1 <= report["frame_count"] <= report["window_size"]
                    or not 1
                    <= report["minimum_observations"]
                    <= report["frame_count"]
                    or any(report[key] > 10_000_000 for key in integer_fields[3:])
                ):
                    raise RuntimeStateError("depth-filter count bounds are invalid")
                for key in (
                    "motion_threshold_mm",
                    "global_changed_fraction",
                    "mad_p95_mm",
                ):
                    if _finite_number(report.get(key), f"depth-filter {key}") < 0.0:
                        raise RuntimeStateError(f"depth-filter {key} cannot be negative")
                if (
                    float(report["motion_threshold_mm"]) <= 0.0
                    or float(report["global_changed_fraction"]) > 1.0
                ):
                    raise RuntimeStateError("depth-filter thresholds violate bounds")
                reset_reason = report.get("reset_reason")
                if reset_reason not in {
                    None,
                    "shape_changed",
                    "stamp_not_increasing",
                    "input_gap",
                }:
                    raise RuntimeStateError("depth-filter reset_reason is unsupported")
                normalized_filter["report"] = normalized_report
            normalized_telemetry["depth_filter"] = normalized_filter
        tracker = telemetry.get("tracker")
        if tracker is not None:
            if not isinstance(tracker, dict):
                raise RuntimeStateError("telemetry tracker must be an object")
            _strict_keys(
                tracker,
                {
                    "phase",
                    "tracking",
                    "target_fresh",
                    "target_source_stamp_ns",
                    "failure",
                },
                "telemetry tracker",
            )
            phase = tracker.get("phase")
            if phase not in {
                "tracking",
                "target_stale",
                "idle_or_lost",
                "unobserved",
                "failed",
            }:
                raise RuntimeStateError("telemetry tracker phase is unsupported")
            tracking = tracker.get("tracking")
            if tracking is not None and not isinstance(tracking, bool):
                raise RuntimeStateError("telemetry tracker tracking must be boolean or null")
            target_fresh = tracker.get("target_fresh")
            if not isinstance(target_fresh, bool):
                raise RuntimeStateError("telemetry tracker target_fresh must be a boolean")
            target_stamp = tracker.get("target_source_stamp_ns")
            if (
                target_stamp is not None
                and (
                    isinstance(target_stamp, bool)
                    or not isinstance(target_stamp, int)
                    or target_stamp <= 0
                )
            ):
                raise RuntimeStateError(
                    "telemetry tracker target_source_stamp_ns must be positive or null",
                )
            failure = tracker.get("failure")
            normalized_failure: dict[str, object] | None = None
            if failure is not None:
                if not isinstance(failure, dict):
                    raise RuntimeStateError("telemetry tracker failure must be an object or null")
                _strict_keys(
                    failure,
                    {"seed_id", "seed_stamp_ns", "reason_code", "reason"},
                    "telemetry tracker failure",
                )
                normalized_failure = {}
                for key in ("seed_id", "seed_stamp_ns"):
                    value = failure.get(key)
                    if (
                        value is not None
                        and (
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 0
                        )
                    ):
                        raise RuntimeStateError(
                            f"telemetry tracker failure {key} must be non-negative or null",
                        )
                    normalized_failure[key] = value
                for key in ("reason_code", "reason"):
                    value = failure.get(key)
                    if value is not None and (
                        not isinstance(value, str) or len(value) > 512
                    ):
                        raise RuntimeStateError(
                            f"telemetry tracker failure {key} must be bounded text or null",
                        )
                    normalized_failure[key] = value
            normalized_telemetry["tracker"] = {
                "phase": phase,
                "tracking": tracking,
                "target_fresh": target_fresh,
                "target_source_stamp_ns": target_stamp,
                "failure": normalized_failure,
            }
        normalized["telemetry"] = normalized_telemetry

    links = document.get("robot_links")
    if links is not None:
        if not isinstance(links, list) or len(links) > MAX_RUNTIME_LINKS:
            raise RuntimeStateError(
                f"robot_links must contain at most {MAX_RUNTIME_LINKS} entries",
            )
        normalized_links = []
        names: set[str] = set()
        for index, link in enumerate(links):
            if not isinstance(link, dict):
                raise RuntimeStateError("robot link entries must be objects")
            _strict_keys(link, {"name", "transform"}, f"robot_links[{index}]")
            name = link.get("name")
            if not isinstance(name, str) or not name or len(name) > 128 or name in names:
                raise RuntimeStateError("robot link names must be unique non-empty strings")
            names.add(name)
            normalized_links.append({
                "name": name,
                "transform": _transform(link.get("transform"), f"robot link {name}"),
            })
        normalized["robot_links"] = normalized_links

    point_clouds = document.get("point_clouds")
    if point_clouds is not None:
        if not isinstance(point_clouds, dict) or len(point_clouds) > MAX_RUNTIME_POINT_CLOUDS:
            raise RuntimeStateError(
                f"point_clouds must contain at most {MAX_RUNTIME_POINT_CLOUDS} clouds",
            )
        normalized_clouds: dict[str, object] = {}
        total_points = 0
        for name, cloud in point_clouds.items():
            if not isinstance(name, str) or not name or len(name) > 64:
                raise RuntimeStateError("point-cloud names must be short non-empty strings")
            if not isinstance(cloud, dict):
                raise RuntimeStateError(f"point cloud {name!r} must be an object")
            _strict_keys(
                cloud,
                {"frame", "points_xyz_m", "colors_rgb"},
                f"point cloud {name!r}",
            )
            frame = cloud.get("frame")
            if not isinstance(frame, str) or not frame or len(frame) > 128:
                raise RuntimeStateError(f"point cloud {name!r} has an invalid frame")
            points = _finite_rows(
                cloud.get("points_xyz_m"),
                columns=3,
                maximum=MAX_RUNTIME_POINTS,
                label=f"point cloud {name!r}",
            )
            total_points += len(points)
            if total_points > MAX_RUNTIME_POINTS:
                raise RuntimeStateError(
                    f"runtime state exceeds the {MAX_RUNTIME_POINTS}-point total limit",
                )
            normalized_cloud: dict[str, object] = {
                "frame": frame,
                "points_xyz_m": points,
            }
            colors = cloud.get("colors_rgb")
            if colors is not None:
                if not isinstance(colors, list) or len(colors) != len(points):
                    raise RuntimeStateError("point-cloud colors must align with points")
                normalized_colors = []
                for color in colors:
                    if (
                        not isinstance(color, list)
                        or len(color) != 3
                        or any(
                            isinstance(channel, bool)
                            or not isinstance(channel, int)
                            or not 0 <= channel <= 255
                            for channel in color
                        )
                    ):
                        raise RuntimeStateError("point-cloud RGB values must be bytes")
                    normalized_colors.append(list(color))
                normalized_cloud["colors_rgb"] = normalized_colors
            normalized_clouds[name] = normalized_cloud
        normalized["point_clouds"] = normalized_clouds

    candidates = document.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list) or len(candidates) > MAX_RUNTIME_CANDIDATES:
            raise RuntimeStateError(
                f"candidates must contain at most {MAX_RUNTIME_CANDIDATES} entries",
            )
        normalized_candidates = []
        candidate_ids: set[int] = set()
        allowed_status = {"unevaluated", "active", "feasible", "rejected", "selected"}
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise RuntimeStateError("candidate entries must be objects")
            _strict_keys(
                candidate,
                {"candidate_id", "frame", "pose", "score", "width_m", "status"},
                f"candidates[{index}]",
            )
            candidate_id = candidate.get("candidate_id")
            frame = candidate.get("frame")
            status = candidate.get("status", "unevaluated")
            if (
                isinstance(candidate_id, bool)
                or not isinstance(candidate_id, int)
                or candidate_id < 0
                or candidate_id in candidate_ids
            ):
                raise RuntimeStateError("candidate IDs must be unique non-negative integers")
            if not isinstance(frame, str) or not frame or len(frame) > 128:
                raise RuntimeStateError("candidate frame must be a non-empty string")
            if status not in allowed_status:
                raise RuntimeStateError(f"unsupported candidate status: {status!r}")
            candidate_ids.add(candidate_id)
            normalized_candidate: dict[str, object] = {
                "candidate_id": candidate_id,
                "frame": frame,
                "pose": _transform(candidate.get("pose"), f"candidate {candidate_id} pose"),
                "status": status,
            }
            for key in ("score", "width_m"):
                if candidate.get(key) is not None:
                    normalized_candidate[key] = _finite_number(
                        candidate[key],
                        f"candidate {candidate_id} {key}",
                    )
            normalized_candidates.append(normalized_candidate)
        normalized["candidates"] = normalized_candidates

    plan = document.get("plan_overlay")
    if plan is not None:
        if not isinstance(plan, dict):
            raise RuntimeStateError("plan_overlay must be an object")
        _strict_keys(
            plan,
            {"frame", "joint_names", "segments", "tcp_path_xyz_m", "selected_candidate_id"},
            "plan_overlay",
        )
        frame = plan.get("frame")
        joint_names = plan.get("joint_names")
        if not isinstance(frame, str) or not frame or len(frame) > 128:
            raise RuntimeStateError("plan_overlay frame must be a non-empty string")
        if (
            not isinstance(joint_names, list)
            or len(joint_names) != 6
            or any(not isinstance(name, str) or not name for name in joint_names)
        ):
            raise RuntimeStateError("plan_overlay joint_names must contain six names")
        segments = plan.get("segments", {})
        if not isinstance(segments, dict) or set(segments) - {"transit", "approach", "lift"}:
            raise RuntimeStateError("plan_overlay has unsupported trajectory segments")
        normalized_segments: dict[str, object] = {}
        for name, segment in segments.items():
            if not isinstance(segment, dict):
                raise RuntimeStateError(f"plan segment {name!r} must be an object")
            _strict_keys(segment, {"positions_rad", "times_s"}, f"plan segment {name!r}")
            positions = _finite_rows(
                segment.get("positions_rad"),
                columns=6,
                maximum=MAX_RUNTIME_PATH_POINTS,
                label=f"plan segment {name!r}",
            )
            normalized_segment: dict[str, object] = {"positions_rad": positions}
            times = segment.get("times_s")
            if times is not None:
                if not isinstance(times, list) or len(times) != len(positions):
                    raise RuntimeStateError("plan segment times must align with positions")
                normalized_times = [
                    _finite_number(value, f"plan segment {name!r} time")
                    for value in times
                ]
                if any(second < first for first, second in zip(normalized_times, normalized_times[1:])):
                    raise RuntimeStateError("plan segment times must be monotonic")
                normalized_segment["times_s"] = normalized_times
            normalized_segments[name] = normalized_segment
        normalized_plan: dict[str, object] = {
            "frame": frame,
            "joint_names": list(joint_names),
            "segments": normalized_segments,
        }
        tcp_path = plan.get("tcp_path_xyz_m")
        if tcp_path is not None:
            normalized_plan["tcp_path_xyz_m"] = _finite_rows(
                tcp_path,
                columns=3,
                maximum=MAX_RUNTIME_PATH_POINTS,
                label="plan TCP path",
            )
        selected = plan.get("selected_candidate_id")
        if selected is not None:
            if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
                raise RuntimeStateError("selected_candidate_id must be non-negative")
            normalized_plan["selected_candidate_id"] = selected
        normalized["plan_overlay"] = normalized_plan
    return normalized


__all__ = [
    "RuntimeStateError",
    "validate_runtime_state",
    "RUNTIME_SCHEMA",
    "MAX_RUNTIME_POINT_CLOUDS",
    "MAX_RUNTIME_POINTS",
    "MAX_RUNTIME_CANDIDATES",
    "MAX_RUNTIME_LINKS",
    "MAX_RUNTIME_PATH_POINTS",
]
