#!/usr/bin/env python3
"""Loopback-only control plane for the Go2W manipulation workbench.

The HTTP layer accepts no command, path, environment, or robot-motion argument.
It exposes separate fixed perception and offline-planning actions, plus the
legacy combined planning-only action, one run at a time.  The only actuator
route is a fixed, server-owned, low-speed PiPER Home action; it accepts no
joint target, path, environment, or command from the browser.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat as stat_module
import subprocess
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import go2w_debug_ui
import go2w_interactive_sessions
import go2w_wrist_search
from go2w_live_perception import LivePerceptionRenderer

from z_manip.read_only_sessions import (
    ATTEMPT_SCHEMA,
    MANIFEST_SCHEMA,
    ReadOnlySessionService,
    SessionContractError,
    validate_session_id,
    validate_target_description,
)


MAX_LOG_TAIL_BYTES = 12_000
RUNTIME_SCHEMA = "z_manip.runtime_state.v1"
MAX_RUNTIME_STATE_BYTES = 8 * 1024 * 1024
MAX_RUNTIME_POINT_CLOUDS = 8
MAX_RUNTIME_POINTS = 50_000
MAX_RUNTIME_CANDIDATES = 512
MAX_RUNTIME_LINKS = 128
MAX_RUNTIME_PATH_POINTS = 2_000
RUNTIME_STALE_AFTER_S = 1.0
MAX_RUNTIME_FUTURE_S = 0.25
MAX_CAMERA_JPEG_BYTES = 512 * 1024
CAMERA_STALE_AFTER_S = 2.0
HOME_FAST_VERIFY_TOLERANCE_RAD = math.radians(1.0)
MAX_CAMERA_FUTURE_S = 0.25
MAX_INTERACTIVE_REQUEST_BYTES = 512
MAX_INTERACTIVE_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_INTERACTIVE_IMAGE_BYTES = 16 * 1024 * 1024
MAX_INTERACTIVE_BUNDLE_BYTES = 64 * 1024 * 1024
INTERACTIVE_ACTION_HEADER = "X-Z-Manip-Action"
INTERACTIVE_PERCEPTION_ACTION = "perception"
INTERACTIVE_PLANNING_ACTION = "planning"
HOME_ACTION = "home"
HOME_ROUTE = "/api/home"
HOME_STATUS_ROUTE = "/api/home/status"
GRASP_ACTION = "grasp"
GRASP_ROUTE = "/api/grasp"
DIRECT_GRASP_ACTION = "grasp-selected"
DIRECT_GRASP_ROUTE = "/api/grasp/selected"
GRASP_STATUS_ROUTE = "/api/grasp/status"
APPROACH_START_ACTION = "approach-start"
APPROACH_STOP_ACTION = "approach-stop"
APPROACH_START_ROUTE = "/api/approach/start"
APPROACH_STOP_ROUTE = "/api/approach/stop"
APPROACH_STATUS_ROUTE = "/api/approach/status"
PICK_HOLD_ACTION = "grasp-pick-hold"
PICK_HOLD_ROUTE = "/api/grasp/pick-hold"
RETURN_HOME_HOLDING_ACTION = "grasp-return-home-holding"
RETURN_HOME_HOLDING_ROUTE = "/api/grasp/return-home-holding"
PLACE_BACK_ACTION = "grasp-place-back"
PLACE_BACK_ROUTE = "/api/grasp/place-back"
SESSION_CLEAR_ACTION = "clear-demo"
SESSION_CLEAR_ROUTE = "/api/sessions/clear"
SERVICE_RESTART_ACTION = "restart-workbench"
SERVICE_RESTART_ROUTE = "/api/service/restart"
SERVICE_UNIT = "z-manip-planning-workbench.service"
COMPONENT_STATUS_ROUTE = "/api/components/status"
COMPONENT_RESTART_ROUTE = "/api/components/restart"
COMPONENT_BRINGUP_ROUTE = "/api/components/bringup"
COMPONENT_LOG_ROUTE_PREFIX = "/api/components/logs/"
COMPONENT_RESTART_ACTION = "restart-component"
COMPONENT_BRINGUP_ACTION = "bringup-components"
VISUAL_COMPONENTS = frozenset({
    "ui",
    "nuc-camera",
    "passive-feedback",
    "observer",
    "rgbd",
    "edgetam",
    "perception",
    "perception-all",
})
LOG_COMPONENTS = VISUAL_COMPONENTS | {"manager"}
INTERACTIVE_STATUS_ROUTE = "/api/sessions/status"
INTERACTIVE_PERCEPTION_ROUTE = "/api/sessions/perception"
INTERACTIVE_PLANNING_ROUTE = "/api/sessions/planning"
INTERACTIVE_PERCEPTION_ARTIFACTS = {
    "/api/sessions/perception/artifacts/mask.png": "edgetam_mask.png",
    "/api/sessions/perception/artifacts/overlay.png": "edgetam_overlay.png",
    "/api/sessions/perception/artifacts/candidates.png": "grasp_candidates_overlay.png",
}
INTERACTIVE_PLANNING_BUNDLE_ROUTE = "/api/sessions/planning/bundle"
LIVE_PERCEPTION_ROUTES = {
    "/api/perception/live/mask.png": "mask",
    "/api/perception/live/overlay.jpg": "overlay",
    "/api/perception/live/candidates.jpg": "candidates",
}


def _empty_display_bundle() -> dict[str, Any]:
    """Return a valid bundle representing no active demo task.

    Historical bundles stay on disk for diagnosis, but `/api/bundle` must not
    resurrect one after Home, Clear demo, a component restart, or a UI restart.
    """

    return {
        "schema": go2w_debug_ui.SCHEMA,
        "run_id": "no-active-task",
        "request_id": "display-cleared",
        "status": {"state": "cleared", "ok": False},
        "mode": {"read_only": True, "planning_only": True},
        "safety": {
            "motion_commands_published": 0,
            "transport_opened": False,
            "ros_imported": False,
            "can_opened": False,
        },
        "frames": {"perception": None, "planning": None},
        "inputs": {},
        "stages": [],
        "artifacts": {},
        "candidates": [],
        "planning": {"available": False, "plan_valid": False, "rejections": []},
        "visualization": {"frame": None, "images": {}},
        "display": {"cleared": True, "history_retained": True},
    }


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


class RuntimeStateReader:
    """Read one fixed, bounded runtime-state path with stat-based caching."""

    def __init__(
        self,
        path: Path | None,
        *,
        stale_after_s: float = RUNTIME_STALE_AFTER_S,
        clock_ns: Any = time.time_ns,
    ) -> None:
        if not math.isfinite(stale_after_s) or stale_after_s <= 0.0:
            raise ValueError("runtime stale threshold must be positive and finite")
        self.path = None if path is None else path.expanduser().resolve()
        self.stale_after_ns = round(stale_after_s * 1_000_000_000)
        self._clock_ns = clock_ns
        self._lock = threading.RLock()
        self._cache_key: object = object()
        self._cached: dict[str, Any] | None = None
        self._last_sequence: int | None = None
        self._last_digest: str | None = None

    def _offline(self, code: str, message: str, received_ns: int) -> dict[str, Any]:
        return {
            "schema": RUNTIME_SCHEMA,
            "status": "offline",
            "sequence": self._last_sequence,
            "source_timestamp_ns": None,
            "received_timestamp_ns": received_ns,
            "joint_positions_rad": None,
            "robot_links": [],
            "point_clouds": {},
            "candidates": [],
            "plan_overlay": None,
            "error": {"code": code, "message": message},
        }

    def _refresh(self, now_ns: int) -> None:
        if self.path is None:
            key = ("unconfigured",)
            if key != self._cache_key:
                self._cached = self._offline(
                    "RUNTIME_STATE_NOT_CONFIGURED",
                    "the server was started without --runtime-state",
                    now_ns,
                )
                self._cache_key = key
            return
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            key = ("missing", str(self.path))
            if key != self._cache_key:
                self._cached = self._offline(
                    "RUNTIME_STATE_MISSING",
                    "the fixed runtime-state file does not exist",
                    now_ns,
                )
                self._cache_key = key
            return
        except OSError as error:
            key = ("stat-error", str(error))
            if key != self._cache_key:
                self._cached = self._offline("RUNTIME_STATE_UNREADABLE", str(error), now_ns)
                self._cache_key = key
            return
        fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if fingerprint == self._cache_key:
            return
        self._cache_key = fingerprint
        if stat.st_size > MAX_RUNTIME_STATE_BYTES:
            self._cached = self._offline(
                "RUNTIME_STATE_TOO_LARGE",
                "runtime state exceeds the 8 MiB limit",
                now_ns,
            )
            return
        try:
            payload = self.path.read_bytes()
            if len(payload) > MAX_RUNTIME_STATE_BYTES:
                raise RuntimeStateError("runtime state exceeds the 8 MiB limit")
            document = json.loads(payload.decode("utf-8"))
            normalized = validate_runtime_state(document)
        except (OSError, UnicodeError, json.JSONDecodeError, RuntimeStateError) as error:
            self._cached = self._offline("RUNTIME_STATE_INVALID", str(error), now_ns)
            return
        digest = hashlib.sha256(payload).hexdigest()
        sequence = int(normalized["sequence"])
        if int(normalized["source_timestamp_ns"]) > now_ns + round(
            MAX_RUNTIME_FUTURE_S * 1_000_000_000,
        ):
            self._cached = self._offline(
                "RUNTIME_TIMESTAMP_IN_FUTURE",
                "runtime source timestamp is more than 250 ms in the future",
                now_ns,
            )
            return
        if self._last_sequence is not None and sequence < self._last_sequence:
            self._cached = self._offline(
                "RUNTIME_SEQUENCE_REGRESSION",
                "runtime sequence moved backwards",
                now_ns,
            )
            return
        if (
            self._last_sequence is not None
            and sequence == self._last_sequence
            and self._last_digest is not None
            and digest != self._last_digest
        ):
            self._cached = self._offline(
                "RUNTIME_SEQUENCE_NOT_ADVANCED",
                "runtime content changed without incrementing sequence",
                now_ns,
            )
            return
        self._last_sequence = sequence
        self._last_digest = digest
        normalized["received_timestamp_ns"] = now_ns
        normalized["error"] = None
        self._cached = normalized

    def snapshot(self) -> tuple[dict[str, Any], str]:
        with self._lock:
            now_ns = int(self._clock_ns())
            self._refresh(now_ns)
            assert self._cached is not None
            result = dict(self._cached)
            if result.get("error") is None:
                age_ns = max(0, now_ns - int(result["source_timestamp_ns"]))
                result["status"] = "live" if age_ns <= self.stale_after_ns else "stale"
                result["age_s"] = age_ns / 1_000_000_000.0
            etag_source = (
                result.get("status"),
                result.get("sequence"),
                result.get("source_timestamp_ns"),
                result.get("received_timestamp_ns"),
                None if result.get("error") is None else result["error"].get("code"),
            )
            etag = '"runtime-' + hashlib.sha256(repr(etag_source).encode()).hexdigest()[:20] + '"'
            return result, etag


class MeasuredHomeVerifier:
    """Verify Home from fresh read-only feedback without opening CAN.

    This is only a latency fast path.  Missing/stale/unsafe telemetry or any
    joint outside the tight tolerance falls back to the normal Home action.
    The full executor still performs its own fresh feedback, arm-status,
    enable, control-mode, and path-start checks before sending a path command.
    """

    def __init__(
        self,
        runtime_state: Path | None,
        home_config: Path,
        *,
        tolerance_rad: float = HOME_FAST_VERIFY_TOLERANCE_RAD,
    ) -> None:
        if not math.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
            raise ValueError("Home verification tolerance must be positive and finite")
        self.reader = RuntimeStateReader(runtime_state)
        self.tolerance_rad = float(tolerance_rad)
        document = json.loads(home_config.expanduser().resolve().read_text(encoding="utf-8"))
        joints = document.get("joint_radians") if isinstance(document, dict) else None
        if (
            not isinstance(document, dict)
            or document.get("schema") != "z_manip.piper_software_home.v1"
            or not isinstance(joints, list)
            or len(joints) != 6
        ):
            raise ValueError("PiPER Home config is invalid")
        self.home_joints = tuple(
            _finite_number(value, "Home joint")
            for value in joints
        )

    def verify(self) -> tuple[bool, str]:
        snapshot, _etag = self.reader.snapshot()
        if snapshot.get("status") != "live":
            return False, "runtime joint feedback is not live"
        telemetry = snapshot.get("telemetry")
        if (
            not isinstance(telemetry, dict)
            or telemetry.get("read_only") is not True
            or telemetry.get("motion_commands_published") != 0
            or snapshot.get("joint_state_available") is not True
        ):
            return False, "runtime joint feedback lacks read-only evidence"
        joints = snapshot.get("joint_positions_rad")
        if not isinstance(joints, list) or len(joints) != 6:
            return False, "runtime joint vector is unavailable"
        maximum_error = max(
            abs(_finite_number(actual, "runtime joint") - expected)
            for actual, expected in zip(joints, self.home_joints, strict=True)
        )
        if maximum_error > self.tolerance_rad:
            return False, f"measured Home error is {math.degrees(maximum_error):.3f}deg"
        return True, f"fresh read-only joints verify Home within {math.degrees(maximum_error):.3f}deg"


class CameraSnapshotReader:
    """Read one fixed, bounded, recent JPEG written by the ROS observer."""

    def __init__(
        self,
        path: Path | None,
        *,
        max_bytes: int = MAX_CAMERA_JPEG_BYTES,
        stale_after_s: float = CAMERA_STALE_AFTER_S,
        clock_ns: Any = time.time_ns,
    ) -> None:
        if max_bytes <= 0 or max_bytes > MAX_CAMERA_JPEG_BYTES:
            raise ValueError("camera byte limit must be within the fixed 512 KiB cap")
        if not math.isfinite(stale_after_s) or not 0.0 < stale_after_s <= 10.0:
            raise ValueError("camera stale threshold must be within 10 seconds")
        self.path = None if path is None else path.expanduser().resolve()
        self.max_bytes = int(max_bytes)
        self.stale_after_ns = round(stale_after_s * 1_000_000_000)
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._cache_key: object = object()
        self._payload: bytes | None = None
        self._etag: str | None = None

    def snapshot(self) -> tuple[str, bytes | None, str | None, float | None, str]:
        with self._lock:
            now_ns = int(self._clock_ns())
            if self.path is None:
                return "offline", None, None, None, "camera image is not configured"
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                return "offline", None, None, None, "camera image is unavailable"
            except OSError as error:
                return "offline", None, None, None, f"camera image is unreadable: {error}"
            if not stat_module.S_ISREG(stat.st_mode):
                return "invalid", None, None, None, "camera image is not a regular file"
            age_ns = now_ns - stat.st_mtime_ns
            age_s = max(0, age_ns) / 1_000_000_000.0
            if stat.st_mtime_ns > now_ns + round(MAX_CAMERA_FUTURE_S * 1_000_000_000):
                return "invalid", None, None, age_s, "camera image timestamp is in the future"
            if stat.st_size <= 4 or stat.st_size > self.max_bytes:
                return "invalid", None, None, age_s, "camera image violates the 512 KiB size limit"
            if age_ns > self.stale_after_ns:
                return "stale", None, self._etag, age_s, "camera image is stale"
            fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if fingerprint != self._cache_key:
                try:
                    payload = self.path.read_bytes()
                except OSError as error:
                    return "offline", None, None, age_s, f"camera image is unreadable: {error}"
                if (
                    len(payload) > self.max_bytes
                    or not payload.startswith(b"\xff\xd8")
                    or not payload.endswith(b"\xff\xd9")
                ):
                    return "invalid", None, None, age_s, "camera image is not a bounded JPEG"
                self._payload = payload
                self._etag = (
                    '"camera-'
                    + hashlib.sha256(payload).hexdigest()[:24]
                    + '"'
                )
                self._cache_key = fingerprint
            return "live", self._payload, self._etag, age_s, ""


class InteractiveArtifactError(ValueError):
    """A selected immutable interactive-session artifact is unavailable."""


class InteractiveArtifactReader:
    """Resolve only server-selected, manifest-bound perception/plan artifacts."""

    def __init__(self, run_root: Path, service: ReadOnlySessionService) -> None:
        self.run_root = run_root.expanduser().resolve()
        self.service = service

    @staticmethod
    def _load_bounded_json(path: Path, *, maximum: int, label: str) -> dict[str, Any]:
        try:
            stat = path.stat()
            if (
                path.is_symlink()
                or not stat_module.S_ISREG(stat.st_mode)
                or not 1 <= stat.st_size <= maximum
            ):
                raise InteractiveArtifactError(f"{label} is not a bounded regular file")
            document = json.loads(path.read_text(encoding="utf-8"))
        except InteractiveArtifactError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InteractiveArtifactError(f"cannot read {label}: {error}") from error
        if not isinstance(document, dict):
            raise InteractiveArtifactError(f"{label} must contain a JSON object")
        return document

    def _session(
        self,
        *,
        action: str,
        session_id: object,
        expected_status: str,
    ) -> Path:
        safe_id = validate_session_id(session_id)
        configured_action_root = self.run_root / action
        if configured_action_root.is_symlink():
            raise InteractiveArtifactError("interactive action root cannot be a symbolic link")
        try:
            action_root = configured_action_root.resolve(strict=True)
        except OSError as error:
            raise InteractiveArtifactError("interactive action root is unavailable") from error
        if action_root.parent != self.run_root:
            raise InteractiveArtifactError("interactive action root escaped its run root")
        candidate = action_root / safe_id
        if candidate.is_symlink():
            raise InteractiveArtifactError("interactive session cannot be a symbolic link")
        try:
            session = candidate.resolve(strict=True)
        except OSError as error:
            raise InteractiveArtifactError("interactive session is unavailable") from error
        if session.parent != action_root:
            raise InteractiveArtifactError("interactive session escaped its fixed run root")
        attempt = self._load_bounded_json(
            session / "attempt.json",
            maximum=MAX_INTERACTIVE_MANIFEST_BYTES,
            label=f"{action} attempt",
        )
        if (
            attempt.get("schema") != ATTEMPT_SCHEMA
            or attempt.get("action") != action
            or attempt.get("session_id") != safe_id
            or attempt.get("status") != expected_status
        ):
            raise InteractiveArtifactError(
                f"{action} session is not a verified {expected_status} attempt",
            )
        return session

    @staticmethod
    def _manifest_entry(
        manifest: dict[str, Any],
        *,
        relative_name: str,
    ) -> dict[str, Any]:
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise InteractiveArtifactError("artifact manifest schema is invalid")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise InteractiveArtifactError("artifact manifest file list is invalid")
        matches = [
            item
            for item in files
            if isinstance(item, dict) and item.get("name") == relative_name
        ]
        if len(matches) != 1:
            raise InteractiveArtifactError("artifact is not uniquely manifest-bound")
        entry = matches[0]
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise InteractiveArtifactError("artifact manifest entry is invalid")
        return entry

    def _verified_payload(
        self,
        *,
        session: Path,
        manifest_name: str,
        artifact_root: Path,
        relative_name: str,
        maximum: int,
    ) -> bytes:
        manifest = self._load_bounded_json(
            session / manifest_name,
            maximum=MAX_INTERACTIVE_MANIFEST_BYTES,
            label=manifest_name,
        )
        entry = self._manifest_entry(manifest, relative_name=relative_name)
        candidate = artifact_root / relative_name
        if artifact_root.is_symlink() or candidate.is_symlink():
            raise InteractiveArtifactError("interactive artifact cannot be a symbolic link")
        try:
            resolved_root = artifact_root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
            stat = resolved.stat()
        except OSError as error:
            raise InteractiveArtifactError("interactive artifact is unavailable") from error
        if resolved_root.parent != session:
            raise InteractiveArtifactError("interactive artifact root escaped its session")
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise InteractiveArtifactError("interactive artifact escaped its session") from error
        expected_size = entry["bytes"]
        if (
            not stat_module.S_ISREG(stat.st_mode)
            or not 1 <= stat.st_size <= maximum
            or stat.st_size != expected_size
        ):
            raise InteractiveArtifactError("interactive artifact size is invalid")
        try:
            payload = resolved.read_bytes()
        except OSError as error:
            raise InteractiveArtifactError("interactive artifact is unreadable") from error
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise InteractiveArtifactError("interactive artifact changed after selection")
        return payload

    def perception_png(self, relative_name: str) -> bytes:
        state = self.service.status()
        selected = state.get("selected_perception_session_id")
        if selected is None:
            actions = state.get("actions")
            perception = actions.get("perception") if isinstance(actions, dict) else None
            last_good = perception.get("last_good") if isinstance(perception, dict) else None
            selected = last_good.get("session_id") if isinstance(last_good, dict) else None
        if selected is None:
            raise InteractiveArtifactError("no successful perception session is selected")
        session = self._session(
            action="perception",
            session_id=selected,
            expected_status="succeeded",
        )
        payload = self._verified_payload(
            session=session,
            manifest_name="perception_manifest.json",
            artifact_root=session / "perception",
            relative_name=relative_name,
            maximum=MAX_INTERACTIVE_IMAGE_BYTES,
        )
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise InteractiveArtifactError("interactive image is not a PNG")
        return payload

    def perception_snapshot(self) -> tuple[str, dict[str, bytes], int]:
        """Return one manifest-verified perception anchor for live rendering."""

        state = self.service.status()
        selected = state.get("selected_perception_session_id")
        actions = state.get("actions")
        perception = actions.get("perception") if isinstance(actions, dict) else None
        latest = perception.get("latest_attempt") if isinstance(perception, dict) else None
        if selected is None:
            last_good = perception.get("last_good") if isinstance(perception, dict) else None
            selected = last_good.get("session_id") if isinstance(last_good, dict) else None
        if selected is None:
            raise InteractiveArtifactError("no successful perception session is selected")
        safe_id = validate_session_id(selected)
        if (
            isinstance(latest, dict)
            and latest.get("session_id") != safe_id
            and latest.get("status") == "failed"
        ):
            raise InteractiveArtifactError(
                "the newest perception attempt failed; the selected anchor is stale",
            )
        session = self._session(
            action="perception",
            session_id=safe_id,
            expected_status="succeeded",
        )
        artifact_root = session / "perception"
        names = {
            "mask": "edgetam_mask.png",
            "overlay": "edgetam_overlay.png",
            "candidates": "grasp_candidates_overlay.png",
        }
        payloads = {
            key: self._verified_payload(
                session=session,
                manifest_name="perception_manifest.json",
                artifact_root=artifact_root,
                relative_name=name,
                maximum=MAX_INTERACTIVE_IMAGE_BYTES,
            )
            for key, name in names.items()
        }
        if any(not payload.startswith(b"\x89PNG\r\n\x1a\n") for payload in payloads.values()):
            raise InteractiveArtifactError("interactive live-render anchor is not PNG")
        anchor_mtime_ns = min((artifact_root / name).stat().st_mtime_ns for name in names.values())
        return safe_id, payloads, anchor_mtime_ns


    def _planning_bundle_for_attempt(
        self,
        attempt: object,
        *,
        allow_missing_bundle: bool,
    ) -> bytes | None:
        if not isinstance(attempt, dict):
            return None
        status = attempt.get("status")
        if status not in {"succeeded", "blocked"}:
            return None
        session_id = attempt.get("session_id")
        if session_id is None:
            return None
        session = self._session(
            action="planning",
            session_id=session_id,
            expected_status=status,
        )
        manifest = self._load_bounded_json(
            session / "planning_manifest.json",
            maximum=MAX_INTERACTIVE_MANIFEST_BYTES,
            label="planning_manifest.json",
        )
        files = manifest.get("files") if manifest.get("schema") == MANIFEST_SCHEMA else None
        names = {
            item.get("name")
            for item in files
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        } if isinstance(files, list) else set()
        available = [
            name
            for name in ("debug_bundle.json", "planning/debug_bundle.json")
            if name in names
        ]
        if not available and allow_missing_bundle:
            return None
        if len(available) != 1:
            raise InteractiveArtifactError(
                "interactive planning attempt has no unique manifest-bound debug bundle",
            )
        payload = self._verified_payload(
            session=session,
            manifest_name="planning_manifest.json",
            artifact_root=session / "artifacts",
            relative_name=available[0],
            maximum=MAX_INTERACTIVE_BUNDLE_BYTES,
        )
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise InteractiveArtifactError("interactive debug bundle is invalid JSON") from error
        if (
            not isinstance(document, dict)
            or document.get("schema") != go2w_debug_ui.SCHEMA
            or not all(
                key in document
                for key in ("mode", "safety", "stages", "artifacts", "visualization")
            )
        ):
            raise InteractiveArtifactError("interactive debug bundle schema is invalid")
        return payload

    def planning_bundle(self) -> bytes:
        state = self.service.status()
        actions = state.get("actions")
        planning = actions.get("planning") if isinstance(actions, dict) else None
        latest_attempt = (
            planning.get("latest_attempt") if isinstance(planning, dict) else None
        )
        latest_payload = self._planning_bundle_for_attempt(
            latest_attempt,
            allow_missing_bundle=True,
        )
        if latest_payload is not None:
            return latest_payload

        last_good = planning.get("last_good") if isinstance(planning, dict) else None
        last_good_payload = self._planning_bundle_for_attempt(
            last_good,
            allow_missing_bundle=False,
        )
        if last_good_payload is None:
            raise InteractiveArtifactError("no interactive planning bundle is available")
        return last_good_payload


class PlanningOnlyRunner:
    """Serialize runs of one preconfigured planning-only script."""

    def __init__(self, session_script: Path, run_root: Path) -> None:
        self.session_script = session_script.expanduser().resolve()
        self.run_root = run_root.expanduser().resolve()
        if not self.session_script.is_file():
            raise FileNotFoundError(f"planning session script does not exist: {self.session_script}")
        if not self.session_script.stat().st_mode & 0o111:
            raise PermissionError(f"planning session script is not executable: {self.session_script}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_root / "web-planning-control.log"
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "available": True,
            "mode": "planning_only",
            "motion_commands_permitted": False,
            "running": False,
            "state": "idle",
            "outcome": None,
            "revision": 0,
            "started_unix_ns": None,
            "finished_unix_ns": None,
            "exit_code": None,
            "message": "Ready to start the fixed planning-only pipeline.",
        }

    def _log_tail(self) -> str:
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - MAX_LOG_TAIL_BYTES))
                return stream.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def status(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._status)
        result["log_tail"] = self._log_tail()
        latest = self.run_root / "latest" / "debug_bundle.json"
        result["latest_bundle_available"] = latest.is_file()
        if latest.is_file():
            result["latest_bundle"] = str(latest.resolve())
        latest_attempt = self.run_root / "latest_attempt" / "debug_bundle.json"
        result["latest_attempt_bundle_available"] = latest_attempt.is_file()
        if latest_attempt.is_file():
            result["latest_attempt_bundle"] = str(latest_attempt.resolve())
        return result

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._status["running"]:
                return {
                    "started": False,
                    "error": "a planning-only run is already in progress",
                    "control": dict(self._status),
                }
            self._status.update(
                {
                    "running": True,
                    "state": "running",
                    "outcome": None,
                    "revision": int(self._status["revision"]) + 1,
                    "started_unix_ns": time.time_ns(),
                    "finished_unix_ns": None,
                    "exit_code": None,
                    "message": "Running camera perception, passive CAN capture, and offline planning.",
                }
            )
        worker = threading.Thread(target=self._run, name="go2w-planning-only", daemon=True)
        worker.start()
        return {"started": True, "control": self.status()}

    def _run(self) -> None:
        return_code: int | None = None
        controller_error: str | None = None
        environment = os.environ.copy()
        environment["Z_MANIP_PLANNING_RUN_ROOT"] = str(self.run_root)
        environment["Z_MANIP_OPEN_BROWSER"] = "0"
        try:
            with self.log_path.open("w", encoding="utf-8") as log:
                log.write("Starting fixed Go2W planning-only pipeline; actuator commands are disabled.\n")
                log.flush()
                completed = subprocess.run(
                    [str(self.session_script)],
                    cwd=self.session_script.parents[2],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    shell=False,
                )
                return_code = completed.returncode
        except Exception as error:  # pragma: no cover - operating-system boundary
            controller_error = str(error)

        latest_available = (self.run_root / "latest" / "debug_bundle.json").is_file()
        latest_attempt_available = (
            self.run_root / "latest_attempt" / "debug_bundle.json"
        ).is_file()
        if controller_error is not None:
            outcome = "controller_error"
            message = f"Planning-only controller failed: {controller_error}"
        elif return_code == 0 and latest_available:
            outcome = "passed"
            message = "Planning-only pipeline passed; the dashboard loaded the new bundle."
        elif return_code != 0 and (latest_attempt_available or latest_available):
            outcome = "blocked"
            message = (
                "Planning-only pipeline was blocked safely; the dashboard retained "
                "the last successful bundle. Inspect latest-attempt diagnostics."
            )
        else:
            outcome = "controller_error"
            message = "Planning-only process ended without producing a debug bundle."
        with self._lock:
            self._status.update(
                {
                    "running": False,
                    "state": "finished",
                    "outcome": outcome,
                    "revision": int(self._status["revision"]) + 1,
                    "finished_unix_ns": time.time_ns(),
                    "exit_code": return_code,
                    "message": message,
                }
            )


class DepthServoRunner:
    """Own depth approach, bounded reacquisition, and optional grasp handoff.

    The browser is deliberately not the workflow controller.  Once accepted,
    an automatic run survives page refreshes and owns detection, visual
    approach, bounded tracker reacquisition, zero-speed handoff, and grasp
    startup on this loopback server.
    """

    def __init__(
        self,
        script: Path,
        status_path: Path,
        log_path: Path,
        *,
        session_service: Any | None = None,
        grasp_runner: Any | None = None,
        wrist_search: Any | None = None,
        max_reacquisitions: int = 3,
    ) -> None:
        self.script = script.expanduser().resolve()
        if not self.script.is_file():
            raise FileNotFoundError(f"depth-servo script does not exist: {self.script}")
        if not self.script.stat().st_mode & 0o111:
            raise PermissionError(f"depth-servo script is not executable: {self.script}")
        self.status_path = status_path.expanduser().resolve()
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path.expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # start()/stop() return status snapshots while holding the owner lock.
        # Use a re-entrant lock so an already-stopped Full Stop cannot deadlock
        # the threaded HTTP handler while building its response.
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._mode: str | None = None
        self._revision = 0
        self._message = "Run perception, then start Shadow or Live approach."
        self._session_service = session_service
        self._grasp_runner = grasp_runner
        self._wrist_search = wrist_search
        self._max_reacquisitions = max(0, int(max_reacquisitions))
        self._cancel = threading.Event()
        self._workflow: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "target": None,
            "auto_handoff": False,
            "operator_present": False,
            "speed_percent": 5,
            "reacquisition_attempts": 0,
            "last_reacquisition": None,
            "failure": None,
        }

    def _process_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _runtime_status(self) -> dict[str, Any]:
        runtime: dict[str, Any] = {}
        try:
            if self.status_path.is_file() and self.status_path.stat().st_size <= 64 * 1024:
                candidate = json.loads(self.status_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict) and candidate.get("schema") == "z_manip.depth_servo_status.v1":
                    runtime = candidate
        except (OSError, UnicodeError, json.JSONDecodeError):
            runtime = {}
        return runtime

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._process_running_locked()
            mode = self._mode
            revision = self._revision
            message = self._message
            exit_code = None if self._process is None else self._process.poll()
            workflow = dict(self._workflow)
            workflow_active = bool(workflow.get("active"))
        runtime = self._runtime_status()
        wrist_search = (
            None if self._wrist_search is None else self._wrist_search.status()
        )
        return {
            "schema": "z_manip.depth_servo_action.v1",
            "available": True,
            "running": running or workflow_active,
            "mode": mode,
            "phase": workflow.get("phase") if workflow_active else runtime.get(
                "phase", "starting" if running else "idle",
            ),
            "revision": revision,
            "message": message,
            "exit_code": exit_code,
            "runtime": runtime,
            "workflow": workflow,
            "wrist_search": wrist_search,
        }

    def _spawn_process_locked(self, mode: str) -> tuple[subprocess.Popen[bytes], Any]:
        self.status_path.unlink(missing_ok=True)
        log = self.log_path.open("ab")
        try:
            process = subprocess.Popen(
                [str(self.script), mode, str(self.status_path)],
                cwd=self.script.parents[2],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        except Exception:
            log.close()
            raise
        self._process = process
        self._mode = mode
        threading.Thread(
            target=self._watch,
            args=(process, log),
            name="z-manip-depth-servo-watch",
            daemon=True,
        ).start()
        return process, log

    def start(
        self,
        mode: str,
        *,
        target: str | None = None,
        acquire_target: bool = False,
        auto_handoff: bool = False,
        operator_present: bool = False,
        speed_percent: int = 5,
    ) -> dict[str, Any]:
        if mode not in {"shadow", "live"}:
            return {
                "started": False,
                "error": {"code": "INVALID_MODE", "message": "mode must be shadow or live"},
                "approach": self.status(),
            }
        if isinstance(speed_percent, bool) or not isinstance(speed_percent, int) or not 1 <= speed_percent <= 50:
            return {
                "started": False,
                "error": {"code": "INVALID_SPEED", "message": "speed_percent must be 1..50"},
                "approach": self.status(),
            }
        if target is not None:
            try:
                target = validate_target_description(target)
            except SessionContractError as error:
                return {
                    "started": False,
                    "error": {"code": error.code, "message": str(error)},
                    "approach": self.status(),
                }
        if (acquire_target or auto_handoff) and target is None:
            return {
                "started": False,
                "error": {"code": "TARGET_REQUIRED", "message": "automatic approach requires a target"},
                "approach": self.status(),
            }
        if auto_handoff and (mode != "live" or self._grasp_runner is None):
            return {
                "started": False,
                "error": {"code": "HANDOFF_UNAVAILABLE", "message": "automatic handoff requires live mode and a grasp runner"},
                "approach": self.status(),
            }
        with self._lock:
            if self._process_running_locked() or self._workflow.get("active") is True:
                return {"started": False, "approach": self.status()}
            self._mode = mode
            self._cancel = threading.Event()
            self._workflow = {
                "active": bool(acquire_target or target is not None),
                "phase": "detecting" if acquire_target else "starting",
                "target": target,
                "auto_handoff": bool(auto_handoff),
                "operator_present": bool(operator_present),
                "speed_percent": speed_percent,
                "reacquisition_attempts": 0,
                "last_reacquisition": None,
                "failure": None,
            }
            self._revision += 1
            self._message = (
                f"Detecting {target!r} before visual approach."
                if acquire_target
                else (
                    "Shadow mode is computing commands without publishing motion."
                    if mode == "shadow"
                    else "Live visual approach is publishing bounded Go2W velocity commands."
                )
            )
            if not acquire_target:
                try:
                    self._spawn_process_locked(mode)
                except Exception as error:
                    self._workflow.update(active=False, phase="blocked", failure=str(error))
                    self._message = f"Could not start depth servo: {error}"
                    return {
                        "started": False,
                        "error": {"code": "APPROACH_START_FAILED", "message": self._message},
                        "approach": self.status(),
                    }
        if acquire_target or target is not None:
            threading.Thread(
                target=self._supervise,
                args=(mode, bool(acquire_target)),
                name="z-manip-depth-servo-supervisor",
                daemon=True,
            ).start()
        return {"started": True, "approach": self.status()}

    def _set_workflow(self, **values: Any) -> None:
        with self._lock:
            if all(self._workflow.get(key) == value for key, value in values.items()):
                return
            self._workflow.update(values)
            self._revision += 1

    def _run_perception(self, target: str, *, reacquisition: bool) -> bool:
        if self._session_service is None:
            self._set_workflow(
                phase="blocked",
                failure="perception service is unavailable",
                active=False,
            )
            return False
        if reacquisition:
            with self._lock:
                attempts = int(self._workflow["reacquisition_attempts"]) + 1
                self._workflow.update(
                    phase="reacquiring",
                    reacquisition_attempts=attempts,
                    last_reacquisition=time.time_ns(),
                )
                self._message = f"Target lost; reacquiring {target!r} ({attempts}/{self._max_reacquisitions})."
        try:
            attempt = self._session_service.start_perception(target)
        except Exception as error:
            self._set_workflow(failure=f"perception reacquisition failed: {error}")
            return False
        return attempt.get("status") == "succeeded"

    def _handoff_after_base_stop(
        self,
        process: subprocess.Popen[bytes],
        *,
        target: str,
        cancel: threading.Event,
        terminal_phase: str,
    ) -> None:
        """Latch zero base motion before starting the fresh grasp transaction."""

        self._terminate_process(process, keep_status=True)
        with self._lock:
            auto_handoff = self._workflow.get("auto_handoff") is True
            speed = int(self._workflow.get("speed_percent", 5))
        if auto_handoff and not cancel.is_set():
            self._set_workflow(phase="handoff_to_grasp")
            # GraspRunner.start() owns a fresh close-range perception, IK,
            # planning, and execution transaction; no approach artifact is
            # reused after the base has moved.
            result = self._grasp_runner.start(target, speed)
            if result.get("started"):
                self._set_workflow(active=False, phase="grasp_started", failure=None)
                with self._lock:
                    self._message = (
                        "Base is stopped; fresh close-range perception, IK, "
                        "planning, and grasp started."
                    )
            else:
                self._set_workflow(
                    active=False,
                    phase="blocked",
                    failure="grasp handoff was rejected",
                )
        else:
            self._set_workflow(active=False, phase=terminal_phase, failure=None)
            with self._lock:
                self._message = "Visual approach stopped at the manipulation handoff."

    def _recover_view_with_stationary_base(
        self,
        process: subprocess.Popen[bytes],
        *,
        mode: str,
        target: str,
        cancel: threading.Event,
    ) -> bool:
        """Stop the base, run one bounded wrist search, then restart tracking."""

        # Wrist motion and base velocity are mutually exclusive. Terminating
        # the servo process first also invokes its zero-command cleanup.
        self._terminate_process(process, keep_status=True)
        if self._wrist_search is None:
            self._set_workflow(
                active=False,
                phase="blocked",
                failure="view recovery requires bounded wrist search",
            )
            with self._lock:
                self._message = "Target left the camera view; base is stopped."
            return False

        self._set_workflow(phase="wrist_search", failure=None)
        with self._lock:
            self._message = "Base stopped; running bounded wrist search to recover the target."
            speed = int(self._workflow.get("speed_percent", 5))
            operator_present = self._workflow.get("operator_present") is True
        try:
            found = self._wrist_search.run(
                target,
                mode=mode,
                speed_percent=speed,
                cancel=cancel,
                operator_present=operator_present,
            )
        except Exception as error:
            found = False
            self._set_workflow(failure=f"wrist search failed: {error}")
        if not found or cancel.is_set():
            search_status = self._wrist_search.status()
            failure = str(
                search_status.get("failure")
                or self._workflow.get("failure")
                or "target not found by bounded wrist search"
            )
            self._set_workflow(active=False, phase="blocked", failure=failure)
            with self._lock:
                self._message = "View recovery failed; arm and base are stopped."
            return False

        self._set_workflow(phase="seeding_tracker", failure=None)
        if not self._run_perception(target, reacquisition=True) or cancel.is_set():
            self._set_workflow(
                active=False,
                phase="blocked",
                failure="wrist search found the target but tracker reseeding failed",
            )
            with self._lock:
                self._message = "Target was found, but stable 3-D tracking did not restart."
            return False
        with self._lock:
            try:
                self._spawn_process_locked(mode)
            except Exception as error:
                self._workflow.update(active=False, phase="blocked", failure=str(error))
                self._message = f"Could not restart depth servo: {error}"
                return False
            self._workflow["phase"] = "waiting_for_track"
            self._message = "Target recovered with the base stationary; visual approach restarted."
        return True

    def _supervise(self, mode: str, acquire_target: bool) -> None:
        with self._lock:
            target = self._workflow.get("target")
            cancel = self._cancel
        assert isinstance(target, str)
        if acquire_target:
            if not self._run_perception(target, reacquisition=False):
                if self._wrist_search is None:
                    self._set_workflow(active=False, phase="blocked", failure="initial perception failed")
                    with self._lock:
                        self._message = "Initial target detection failed; base was never started."
                    return
                self._set_workflow(phase="wrist_search", failure=None)
                with self._lock:
                    self._message = "Target is outside the current D435 view; starting bounded wrist search."
                try:
                    found = self._wrist_search.run(
                        target,
                        mode=mode,
                        speed_percent=int(self._workflow.get("speed_percent", 5)),
                        cancel=cancel,
                        operator_present=self._workflow.get("operator_present") is True,
                    )
                except Exception as error:
                    found = False
                    self._set_workflow(failure=f"wrist search failed: {error}")
                if not found or cancel.is_set():
                    search_status = self._wrist_search.status()
                    failure = str(
                        search_status.get("failure")
                        or self._workflow.get("failure")
                        or "target not found by bounded wrist search"
                    )
                    self._set_workflow(active=False, phase="blocked", failure=failure)
                    with self._lock:
                        self._message = "Target was not confirmed; arm and base are stopped."
                    return
                self._set_workflow(phase="seeding_tracker", failure=None)
                if not self._run_perception(target, reacquisition=False):
                    self._set_workflow(
                        active=False,
                        phase="blocked",
                        failure="target was found but EdgeTAM seed acquisition failed",
                    )
                    with self._lock:
                        self._message = "Detector found the target, but stable 3-D tracking did not initialize."
                    return
            if cancel.is_set():
                return
            with self._lock:
                try:
                    self._spawn_process_locked(mode)
                except Exception as error:
                    self._workflow.update(active=False, phase="blocked", failure=str(error))
                    self._message = f"Could not start depth servo: {error}"
                    return
                self._workflow["phase"] = "approaching"
                self._message = "Target acquired; visual approach is running."

        last_reacquisition_s = 0.0
        while not cancel.wait(0.20):
            with self._lock:
                process = self._process
                active = self._workflow.get("active") is True
            if not active:
                return
            if process is None or process.poll() is not None:
                self._set_workflow(active=False, phase="blocked", failure="depth servo exited")
                return
            runtime = self._runtime_status()
            phase = runtime.get("phase")
            if phase in {"reached", "handoff_probe", "handoff_ready"}:
                self._handoff_after_base_stop(
                    process,
                    target=target,
                    cancel=cancel,
                    terminal_phase=str(phase),
                )
                return
            if phase in {"view_recovery", "search_required"}:
                now = time.monotonic()
                with self._lock:
                    attempts = int(self._workflow["reacquisition_attempts"])
                if attempts >= self._max_reacquisitions:
                    self._terminate_process(process, keep_status=True)
                    self._set_workflow(
                        active=False,
                        phase="blocked",
                        failure="view recovery budget exhausted",
                    )
                    with self._lock:
                        self._message = "Target remained outside the view; base is stopped."
                    return
                if now - last_reacquisition_s < 1.0:
                    continue
                last_reacquisition_s = now
                if not self._recover_view_with_stationary_base(
                    process,
                    mode=mode,
                    target=target,
                    cancel=cancel,
                ):
                    return
                continue
            if phase != "tracking_lost":
                if phase:
                    self._set_workflow(phase=str(phase))
                continue
            now = time.monotonic()
            with self._lock:
                attempts = int(self._workflow["reacquisition_attempts"])
            if attempts >= self._max_reacquisitions:
                self._terminate_process(process, keep_status=True)
                self._set_workflow(active=False, phase="blocked", failure="tracker reacquisition budget exhausted")
                with self._lock:
                    self._message = "Target remained lost after bounded reacquisition; base is stopped."
                return
            if now - last_reacquisition_s < 1.0:
                continue
            last_reacquisition_s = now
            if self._run_perception(target, reacquisition=True):
                self._set_workflow(phase="waiting_for_track")

    def _terminate_process(self, process: subprocess.Popen[bytes], *, keep_status: bool) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if not keep_status:
            self.status_path.unlink(missing_ok=True)

    def _watch(self, process: subprocess.Popen[bytes], log: Any) -> None:
        return_code = process.wait()
        log.close()
        with self._lock:
            if self._process is process:
                self._revision += 1
                self._message = (
                    "Depth servo stopped."
                    if return_code in (0, -signal.SIGTERM)
                    else f"Depth servo exited with code {return_code}; inspect its log."
                )

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._cancel.set()
            if self._wrist_search is not None:
                self._wrist_search.stop()
            self._workflow.update(active=False, phase="stopped", failure=None)
            process = self._process
            if process is None or process.poll() is not None:
                self.status_path.unlink(missing_ok=True)
                self._message = "Depth servo is already stopped."
                return {"stopped": True, "approach": self.status()}
            self._message = "Stopping visual approach and commanding zero velocity."
            self._revision += 1
        self._terminate_process(process, keep_status=False)
        return {"stopped": True, "approach": self.status()}


class PiperHomeRunner:
    """Serialize one fixed, server-owned low-speed PiPER Home action."""

    def __init__(
        self,
        script: Path,
        log_path: Path,
        *,
        on_home_reached: Callable[[], object] | None = None,
    ) -> None:
        self.script = script.expanduser().resolve()
        if not self.script.is_file():
            raise FileNotFoundError(f"PiPER Home script does not exist: {self.script}")
        if not self.script.stat().st_mode & 0o111:
            raise PermissionError(f"PiPER Home script is not executable: {self.script}")
        self.log_path = log_path.expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_home_reached = on_home_reached
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "schema": "z_manip.piper_home_action.v1",
            "available": True,
            "running": False,
            "state": "idle",
            "outcome": None,
            "at_home": None,
            "speed_percent": 2,
            "revision": 0,
            "started_unix_ns": None,
            "finished_unix_ns": None,
            "exit_code": None,
            "message": "Ready for fixed low-speed Home recovery.",
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(self, speed_percent: int = 2) -> dict[str, Any]:
        if isinstance(speed_percent, bool) or not isinstance(speed_percent, int) or not 1 <= speed_percent <= 50:
            return {
                "started": False,
                "error": {"code": "INVALID_SPEED", "message": "Home speed must be an integer from 1 to 50 percent"},
                "home": self.status(),
            }
        with self._lock:
            if self._status["running"]:
                return {"started": False, "home": dict(self._status)}
            self._status.update({
                "running": True,
                "state": "running",
                "outcome": None,
                "at_home": False,
                "speed_percent": speed_percent,
                "revision": int(self._status["revision"]) + 1,
                "started_unix_ns": time.time_ns(),
                "finished_unix_ns": None,
                "exit_code": None,
                "message": f"Resetting if required, holding current pose, then returning Home at {speed_percent}%.",
            })
        worker = threading.Thread(target=self._run, args=(speed_percent,), name="z-manip-piper-home", daemon=True)
        worker.start()
        return {"started": True, "home": self.status()}

    def _run(self, speed_percent: int) -> None:
        return_code: int | None = None
        controller_error: str | None = None
        completion: dict[str, Any] | None = None
        failure_detail: str | None = None
        try:
            with self.log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [str(self.script), str(speed_percent)],
                    cwd=self.script.parents[2],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    shell=False,
                    timeout=60.0,
                )
                return_code = completed.returncode
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if candidate.get("schema") == "z_manip.piper_home_recovery.v1":
                    completion = candidate
                    break
            for line in reversed(lines):
                if "SafetyError:" in line:
                    failure_detail = line.split("SafetyError:", 1)[1].strip()
                    break
        except Exception as error:  # pragma: no cover - operating-system boundary
            controller_error = str(error)

        passed = bool(
            controller_error is None
            and return_code == 0
            and completion is not None
            and completion.get("phase") == "complete"
        )
        with self._lock:
            self._status.update({
                "running": False,
                "state": "finished",
                "outcome": "passed" if passed else "blocked",
                "at_home": passed,
                "revision": int(self._status["revision"]) + 1,
                "finished_unix_ns": time.time_ns(),
                "exit_code": return_code,
                "message": (
                    "PiPER reached the measured Home pose."
                    if passed
                    else f"Home action stopped safely: {controller_error or failure_detail or 'inspect Home log'}"
                ),
                "result": completion,
            })
        if passed and self.on_home_reached is not None:
            try:
                self.on_home_reached()
            except Exception as error:  # pragma: no cover - cleanup must not falsify physical Home
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(f"Home task-context cleanup failed: {error}\n")


class PiperGraspRunner:
    """Run one fixed Home-planned pregrasp/close/lift sequence."""

    def __init__(
        self,
        script: Path,
        log_path: Path,
        receipt_root: Path,
        session_service: ReadOnlySessionService,
        session_run_root: Path,
        home_runner: PiperHomeRunner,
        home_verifier: MeasuredHomeVerifier | None = None,
    ) -> None:
        self.script = script.expanduser().resolve()
        if not self.script.is_file():
            raise FileNotFoundError(f"PiPER grasp stage script does not exist: {self.script}")
        if not self.script.stat().st_mode & 0o111:
            raise PermissionError(f"PiPER grasp stage script is not executable: {self.script}")
        self.log_path = log_path.expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_root = receipt_root.expanduser().resolve()
        self.receipt_root.mkdir(parents=True, exist_ok=True)
        self.session_service = session_service
        self.session_run_root = session_run_root.expanduser().resolve()
        self.home_runner = home_runner
        self.home_verifier = home_verifier
        self._lock = threading.Lock()
        self._workflow_path = self.receipt_root / "workflow.json"
        self._workflow: dict[str, Any] = {
            "phase": "ready_at_home",
            "artifact_id": None,
            "planning_session_id": None,
            "holding_object": False,
            "at_home": True,
            "receipt_dir": None,
            "planning_report": None,
            "planned_grasp": None,
        }
        self._restore_workflow()
        self._status: dict[str, Any] = {
            "schema": "z_manip.grasp_action.v1",
            "available": True,
            "running": False,
            "state": "idle",
            "phase": "idle",
            "outcome": None,
            "speed_percent": 5,
            "revision": 0,
            "started_unix_ns": None,
            "finished_unix_ns": None,
            "message": "Ready for one Home-planned pregrasp, close, and lift.",
        }

    def _restore_workflow(self) -> None:
        try:
            document = json.loads(self._workflow_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(document, dict) or document.get("schema") != "z_manip.grasp_workflow.v1":
            return
        phase = document.get("phase")
        if phase not in {"ready_at_home", "holding_at_lift", "holding_at_home", "placed_back_at_home"}:
            return
        self._workflow.update({key: document.get(key) for key in self._workflow})

    def _set_workflow(self, **values: Any) -> None:
        self._workflow.update(values)
        document = {"schema": "z_manip.grasp_workflow.v1", **self._workflow}
        temporary = self._workflow_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self._workflow_path)

    def reset_workflow(self) -> None:
        """Invalidate every staged receipt after an ordinary Home recovery."""
        with self._lock:
            self._set_workflow(
                phase="ready_at_home", artifact_id=None, planning_session_id=None,
                holding_object=False, at_home=True, receipt_dir=None,
                planning_report=None, planned_grasp=None,
            )

    def _reset_home_state(self, *, message: str) -> None:
        with self._lock:
            self._set_workflow(
                phase="ready_at_home", artifact_id=None, planning_session_id=None,
                holding_object=False, at_home=True, receipt_dir=None,
                planning_report=None, planned_grasp=None,
            )
            self._status.update({
                "running": False,
                "state": "idle",
                "phase": "idle",
                "outcome": None,
                "revision": int(self._status["revision"]) + 1,
                "started_unix_ns": None,
                "finished_unix_ns": time.time_ns(),
                "message": message,
            })

    def reset_for_home(self) -> None:
        """Release stale workflow locks as soon as Home recovery is accepted."""
        self._reset_home_state(
            message="Home recovery accepted; stale grasp workflow and action locks were cleared.",
        )

    def reset_after_home(self) -> None:
        """Clear every stale grasp lock after measured Home completes."""
        self._reset_home_state(
            message="Home verified; stale grasp workflow and action locks were cleared.",
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            workflow = getattr(self, "_workflow", {
                "phase": "ready_at_home",
                "artifact_id": None,
                "planning_session_id": None,
                "holding_object": False,
                "at_home": True,
            })
            return {**self._status, "workflow": dict(workflow)}

    def _update(self, **values: Any) -> None:
        with self._lock:
            values.setdefault("revision", int(self._status["revision"]) + 1)
            self._status.update(values)

    def _selected_planning(self) -> dict[str, Any]:
        state = self.session_service.status()
        selected = state.get("selected_perception_session_id")
        planning = state.get("actions", {}).get("planning", {})
        last_good = planning.get("last_good")
        if (
            not isinstance(selected, str)
            or not isinstance(last_good, dict)
            or last_good.get("status") != "succeeded"
            or last_good.get("selected_perception_session_id") != selected
        ):
            raise SessionContractError(
                "NO_SELECTED_PLANNING",
                "run planning for the currently selected perception before Direct Perform",
            )
        return dict(last_good)

    @staticmethod
    def _invalid_speed(speed_percent: int) -> dict[str, Any] | None:
        if isinstance(speed_percent, bool) or not isinstance(speed_percent, int) or not 1 <= speed_percent <= 50:
            return {
                "started": False,
                "error": {"code": "INVALID_SPEED", "message": "grasp speed must be an integer from 1 to 50 percent"},
            }
        return None

    def start(self, target: str, speed_percent: int = 5) -> dict[str, Any]:
        try:
            validated_target = validate_target_description(target)
        except SessionContractError as error:
            return {
                "started": False,
                "error": {"code": error.code, "message": str(error)},
                "grasp": self.status(),
            }
        invalid = self._invalid_speed(speed_percent)
        if invalid is not None:
            invalid["grasp"] = self.status()
            return invalid
        with self._lock:
            if self._status["running"]:
                return {"started": False, "grasp": dict(self._status)}
            if self._workflow.get("holding_object") is True:
                return {
                    "started": False,
                    "error": {"code": "OBJECT_ALREADY_HELD", "message": "finish Return Home Holding / Place Back before starting another grasp"},
                    "grasp": {**self._status, "workflow": dict(self._workflow)},
                }
            self._status.update({
                "running": True,
                "state": "running",
                "phase": "home",
                "outcome": None,
                "target": validated_target,
                "speed_percent": speed_percent,
                "revision": int(self._status["revision"]) + 1,
                "started_unix_ns": time.time_ns(),
                "finished_unix_ns": None,
                "message": f"Starting one complete perception-to-grasp run for {validated_target!r}; verifying measured Home first.",
            })
        worker = threading.Thread(
            target=self._run,
            args=(validated_target, speed_percent),
            name="z-manip-piper-grasp",
            daemon=True,
        )
        worker.start()
        return {"started": True, "grasp": self.status()}

    def start_selected(self, speed_percent: int = 5) -> dict[str, Any]:
        """Execute the current successful plan without Home/perception/replanning."""
        invalid = self._invalid_speed(speed_percent)
        if invalid is not None:
            invalid["grasp"] = self.status()
            return invalid
        with self._lock:
            if self._status["running"]:
                return {"started": False, "grasp": dict(self._status)}
            if self._workflow.get("holding_object") is True:
                return {
                    "started": False,
                    "error": {"code": "OBJECT_ALREADY_HELD", "message": "finish Return Home Holding / Place Back before Direct Perform"},
                    "grasp": {**self._status, "workflow": dict(self._workflow)},
                }
        try:
            planning = self._selected_planning()
        except SessionContractError as error:
            return {
                "started": False,
                "error": {"code": error.code, "message": str(error)},
                "grasp": self.status(),
            }
        with self._lock:
            if self._status["running"]:
                return {"started": False, "grasp": dict(self._status)}
            self._status.update({
                "running": True,
                "state": "running",
                "phase": "execute_selected",
                "outcome": None,
                "speed_percent": speed_percent,
                "planning_session_id": planning.get("session_id"),
                "revision": int(self._status["revision"]) + 1,
                "started_unix_ns": time.time_ns(),
                "finished_unix_ns": None,
                "message": f"Executing the selected plan directly at {speed_percent}% (no Home/perception/replanning).",
            })
        worker = threading.Thread(
            target=self._run_selected,
            args=(planning, speed_percent),
            name="z-manip-piper-selected-grasp",
            daemon=True,
        )
        worker.start()
        return {"started": True, "grasp": self.status()}

    def _start_workflow_worker(
        self,
        *,
        expected_phase: str | tuple[str, ...],
        running_phase: str,
        worker: Callable[[], None],
        speed_percent: int,
    ) -> dict[str, Any]:
        invalid = self._invalid_speed(speed_percent)
        if invalid is not None:
            invalid["grasp"] = self.status()
            return invalid
        with self._lock:
            if self._status["running"]:
                return {"started": False, "grasp": {**self._status, "workflow": dict(self._workflow)}}
            expected_phases = (expected_phase,) if isinstance(expected_phase, str) else expected_phase
            if self._workflow["phase"] not in expected_phases:
                expected_label = " or ".join(expected_phases)
                return {
                    "started": False,
                    "error": {
                        "code": "WORKFLOW_PHASE_MISMATCH",
                        "message": f"{running_phase} requires {expected_label}; current phase is {self._workflow['phase']}",
                    },
                    "grasp": {**self._status, "workflow": dict(self._workflow)},
                }
            self._status.update({
                "running": True,
                "state": "running",
                "phase": running_phase,
                "outcome": None,
                "speed_percent": speed_percent,
                "revision": int(self._status["revision"]) + 1,
                "started_unix_ns": time.time_ns(),
                "finished_unix_ns": None,
                "message": f"Starting {running_phase.replace('_', ' ')} at {speed_percent}%.",
            })
        thread = threading.Thread(target=worker, name=f"z-manip-{running_phase}", daemon=True)
        thread.start()
        return {"started": True, "grasp": self.status()}

    def start_pick_hold(self, target: str, speed_percent: int = 5) -> dict[str, Any]:
        try:
            validated = validate_target_description(target)
        except SessionContractError as error:
            return {"started": False, "error": {"code": error.code, "message": str(error)}, "grasp": self.status()}
        return self._start_workflow_worker(
            expected_phase="ready_at_home",
            running_phase="pick_hold",
            worker=lambda: self._run_pick_hold(validated, speed_percent),
            speed_percent=speed_percent,
        )

    def start_return_home_holding(self, speed_percent: int = 5) -> dict[str, Any]:
        return self._start_workflow_worker(
            expected_phase="holding_at_lift",
            running_phase="return_home_holding",
            worker=lambda: self._run_workflow_continuation("return-home-holding", "holding_at_home", speed_percent),
            speed_percent=speed_percent,
        )

    def start_place_back(self, speed_percent: int = 5) -> dict[str, Any]:
        return self._start_workflow_worker(
            expected_phase=("holding_at_lift", "holding_at_home"),
            running_phase="place_back",
            worker=lambda: self._run_workflow_continuation("place-back", "placed_back_at_home", speed_percent),
            speed_percent=speed_percent,
        )

    @staticmethod
    def _artifact_id(report: Path, archive: Path) -> str:
        return hashlib.sha256(report.read_bytes() + b"\0" + archive.read_bytes()).hexdigest()

    def _run_workflow_phase_remote(
        self,
        *,
        workflow_phase: str,
        report: Path,
        archive: Path,
        planning_session_id: str,
        receipt_dir: Path,
        speed_percent: int,
        prior_receipt_dir: Path | None = None,
    ) -> None:
        arguments = [
            str(self.script),
            "--planning-report", str(report),
            "--planned-grasp", str(archive),
            "--receipt-dir", str(receipt_dir),
            "--speed-percent", str(speed_percent),
            "--workflow-phase", workflow_phase,
            "--planning-session-id", planning_session_id,
        ]
        if prior_receipt_dir is not None:
            arguments.extend(("--prior-receipt-dir", str(prior_receipt_dir)))
        with self.log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                arguments,
                cwd=self.script.parents[2],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
                timeout=430.0,
            )
        if completed.returncode != 0 or not (receipt_dir / "workflow-state.json").is_file():
            raise RuntimeError(f"{workflow_phase} stopped safely; inspect {self.log_path}")

    def _run_pick_hold(self, target: str, speed_percent: int) -> None:
        action_dir = self.receipt_root / f"pick-hold-{time.time_ns()}"
        home_mode: str | None = None
        try:
            self.log_path.write_text("Starting staged pick-and-hold from measured Home.\n", encoding="utf-8")
            home_mode = self._wait_home(speed_percent)
            self._update(phase="pick_hold_perception", home_mode=home_mode)
            perception = self.session_service.start_perception(target)
            if perception.get("status") != "succeeded":
                raise RuntimeError("fresh perception failed")
            self._update(phase="pick_hold_planning", home_mode=home_mode)
            planning = self.session_service.start_planning()
            report, archive = self._planning_artifacts(planning)
            planning_session_id = validate_session_id(planning.get("session_id"))
            artifact_id = self._artifact_id(report, archive)
            self._update(phase="pick_hold_execute", planning_session_id=planning_session_id)
            self._run_workflow_phase_remote(
                workflow_phase="pick-hold", report=report, archive=archive,
                planning_session_id=planning_session_id, receipt_dir=action_dir,
                speed_percent=speed_percent,
            )
            with self._lock:
                self._set_workflow(
                    phase="holding_at_lift", artifact_id=artifact_id,
                    planning_session_id=planning_session_id, holding_object=True,
                    at_home=False, receipt_dir=str(action_dir),
                    planning_report=str(report), planned_grasp=str(archive),
                )
            self._update(
                running=False, state="finished", phase="holding_at_lift",
                outcome="passed", finished_unix_ns=time.time_ns(),
                message="Object grasped and lifted; grip remains closed.",
            )
        except Exception as error:  # pragma: no cover - hardware boundary
            self._update(
                running=False, state="finished", outcome="blocked",
                finished_unix_ns=time.time_ns(),
                message=f"Pick and Hold stopped during {self.status().get('phase')}: {error}",
            )

    def _run_workflow_continuation(
        self,
        workflow_phase: str,
        completed_phase: str,
        speed_percent: int,
    ) -> None:
        with self._lock:
            workflow = dict(self._workflow)
        action_dir = self.receipt_root / f"{workflow_phase}-{time.time_ns()}"
        try:
            report = Path(str(workflow["planning_report"])).resolve()
            archive = Path(str(workflow["planned_grasp"])).resolve()
            prior = Path(str(workflow["receipt_dir"])).resolve()
            planning_session_id = validate_session_id(workflow["planning_session_id"])
            if self._artifact_id(report, archive) != workflow["artifact_id"]:
                raise RuntimeError("stored workflow artifact changed after Pick and Hold")
            self._run_workflow_phase_remote(
                workflow_phase=workflow_phase, report=report, archive=archive,
                planning_session_id=planning_session_id, receipt_dir=action_dir,
                prior_receipt_dir=prior, speed_percent=speed_percent,
            )
            placed = completed_phase == "placed_back_at_home"
            with self._lock:
                self._set_workflow(
                    phase=completed_phase, holding_object=not placed, at_home=True,
                    receipt_dir=str(action_dir),
                )
            if placed:
                self.session_service.clear_current_context()
            self._update(
                running=False, state="finished", phase=completed_phase,
                outcome="passed", finished_unix_ns=time.time_ns(),
                message=(
                    "Object returned to its original grasp location; gripper opened and arm returned Home."
                    if placed else "Arm returned Home on the checked reverse path while preserving grip."
                ),
            )
        except Exception as error:  # pragma: no cover - hardware boundary
            self._update(
                running=False, state="finished", outcome="blocked",
                finished_unix_ns=time.time_ns(),
                message=f"{workflow_phase} stopped during {self.status().get('phase')}: {error}",
            )

    def _wait_home(self, speed_percent: int) -> str:
        if self.home_verifier is not None:
            verified, detail = self.home_verifier.verify()
            if verified:
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(f"Home fast verification: {detail}.\n")
                return "fresh_read_only_joint_feedback"
        result = self.home_runner.start(speed_percent)
        if not result.get("started"):
            raise RuntimeError("Home action could not start")
        deadline = time.monotonic() + 70.0
        while time.monotonic() < deadline:
            status = self.home_runner.status()
            if status.get("running") is not True:
                if status.get("outcome") != "passed":
                    raise RuntimeError(status.get("message") or "Home verification failed")
                return "full_home_action"
            time.sleep(0.1)
        raise RuntimeError("Home verification timed out")

    def _planning_artifacts(self, attempt: dict[str, Any]) -> tuple[Path, Path]:
        if attempt.get("status") != "succeeded":
            error = attempt.get("error")
            detail = error.get("message") if isinstance(error, dict) else None
            raise RuntimeError(detail or "fresh planning was blocked")
        session_id = validate_session_id(attempt.get("session_id"))
        planning_root = self.session_run_root / "planning" / session_id / "artifacts" / "planning"
        report = planning_root / "planning_report.json"
        archive = planning_root / "planned_grasp.npz"
        expected_parent = planning_root.resolve()
        for path in (report, archive):
            if path.is_symlink() or not path.is_file() or path.resolve().parent != expected_parent:
                raise RuntimeError("fresh planning omitted its immutable execution artifact")
        return report, archive

    def _run_full(
        self,
        *,
        report: Path,
        archive: Path,
        receipt_dir: Path,
        speed_percent: int,
    ) -> None:
        arguments = [
            str(self.script),
            "--planning-report", str(report),
            "--planned-grasp", str(archive),
            "--receipt-dir", str(receipt_dir),
            "--speed-percent", str(speed_percent),
        ]
        with self.log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                arguments,
                cwd=self.script.parents[2],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
                timeout=430.0,
            )
        if completed.returncode != 0 or not (receipt_dir / "lift-receipt.json").is_file():
            raise RuntimeError(f"full grasp stopped safely; inspect {self.log_path}")

    def _run(self, target: str, speed_percent: int) -> None:
        action_dir = self.receipt_root / f"grasp-{time.time_ns()}"
        action_started = time.monotonic()
        timings: dict[str, float] = {}
        home_mode: str | None = None
        try:
            self.log_path.write_text(
                "Starting fixed full grasp planned once from measured Home.\n",
                encoding="utf-8",
            )
            phase_started = time.monotonic()
            home_mode = self._wait_home(speed_percent)
            timings["home_verification"] = round(
                time.monotonic() - phase_started,
                6,
            )
            self._update(
                phase="perception",
                home_mode=home_mode,
                timings_s=dict(timings),
                message="Home verified. Capturing one fresh D435 RGB-D frame.",
            )
            phase_started = time.monotonic()
            perception = self.session_service.start_perception(target)
            timings["perception"] = round(time.monotonic() - phase_started, 6)
            if perception.get("status") != "succeeded":
                error = perception.get("error")
                detail = error.get("message") if isinstance(error, dict) else None
                raise RuntimeError(detail or "fresh perception failed")
            self._update(
                phase="planning",
                home_mode=home_mode,
                timings_s=dict(timings),
                message="Planning the complete transit, approach, and lift from Home.",
            )
            phase_started = time.monotonic()
            planning = self.session_service.start_planning()
            timings["planning"] = round(time.monotonic() - phase_started, 6)
            timings["pre_execution_total"] = round(
                time.monotonic() - action_started,
                6,
            )
            report, archive = self._planning_artifacts(planning)

            self._update(
                phase="execute_full",
                planning_session_id=planning.get("session_id"),
                home_mode=home_mode,
                timings_s=dict(timings),
                message=f"Executing one continuous checked pick, place-back, and Home return at {speed_percent}%.",
            )
            self._run_full(
                report=report,
                archive=archive,
                receipt_dir=action_dir,
                speed_percent=speed_percent,
            )
            self.session_service.clear_current_context()
            self._update(
                running=False,
                state="finished",
                phase="returned_home",
                outcome="passed",
                finished_unix_ns=time.time_ns(),
                receipt_dir=str(action_dir),
                home_mode=home_mode,
                timings_s=dict(timings),
                message="Grasp/lift verified, object placed back, and arm returned Home on the checked reverse path.",
            )
        except Exception as error:  # pragma: no cover - hardware boundary
            self._update(
                running=False,
                state="finished",
                outcome="blocked",
                finished_unix_ns=time.time_ns(),
                home_mode=home_mode,
                timings_s={
                    **timings,
                    "pre_execution_total": round(
                        time.monotonic() - action_started,
                        6,
                    ),
                },
                message=f"Grasp stopped safely during {self.status().get('phase')}: {error}",
            )

    def _run_selected(self, planning: dict[str, Any], speed_percent: int) -> None:
        action_dir = self.receipt_root / f"grasp-{time.time_ns()}"
        try:
            self.log_path.write_text(
                "Starting direct execution of the currently selected planning artifact.\n",
                encoding="utf-8",
            )
            report, archive = self._planning_artifacts(planning)
            self._run_full(
                report=report,
                archive=archive,
                receipt_dir=action_dir,
                speed_percent=speed_percent,
            )
            self.session_service.clear_current_context()
            self._update(
                running=False,
                state="finished",
                phase="returned_home",
                outcome="passed",
                finished_unix_ns=time.time_ns(),
                receipt_dir=str(action_dir),
                message="Selected plan executed directly; object placed back and arm returned Home.",
            )
        except Exception as error:  # pragma: no cover - hardware boundary
            self._update(
                running=False,
                state="finished",
                outcome="blocked",
                finished_unix_ns=time.time_ns(),
                message=f"Direct Perform stopped during {self.status().get('phase')}: {error}",
            )


class VisualComponentManager:
    """Bounded asynchronous owner for fixed UI and perception components."""

    def __init__(self, script: Path) -> None:
        resolved = script.expanduser().resolve()
        if resolved.is_symlink() or not resolved.is_file():
            raise ValueError(f"component manager is not a regular file: {resolved}")
        self.script = resolved
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active_component: str | None = None
        self._last_result: dict[str, Any] | None = None

    def _shell(self, *arguments: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.script), *arguments],
            cwd=self.script.parents[2],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
            timeout=timeout,
        )

    @staticmethod
    def _parse_status(output: str) -> dict[str, dict[str, str]]:
        components: dict[str, dict[str, str]] = {}
        for line in output.splitlines():
            fields = line.split("\t", 2)
            if len(fields) != 3 or fields[0] not in VISUAL_COMPONENTS:
                continue
            name, state, summary = fields
            if state not in {"healthy", "degraded", "offline"}:
                state = "degraded"
            components[name] = {
                "name": name,
                "state": state,
                "summary": summary[:300],
            }
        return components

    def status(self) -> dict[str, Any]:
        error: str | None = None
        components: dict[str, dict[str, str]] = {}
        try:
            completed = self._shell("status", "all", timeout=6.0)
            components = self._parse_status(completed.stdout)
            if completed.returncode != 0:
                error = f"component status exited {completed.returncode}"
        except (OSError, subprocess.SubprocessError) as exc:
            error = f"component status unavailable: {exc}"
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            active = self._active_component if running else None
            last_result = None if self._last_result is None else dict(self._last_result)
        for name in VISUAL_COMPONENTS:
            components.setdefault(
                name,
                {"name": name, "state": "offline", "summary": "status unavailable"},
            )
        return {
            "schema": "z_manip.visual_components.v1",
            "available": error is None,
            "busy": running,
            "active_component": active,
            "components": components,
            "last_result": last_result,
            "error": error,
        }

    def logs(self, component: str) -> dict[str, Any]:
        if component not in LOG_COMPONENTS:
            raise ValueError("unsupported visual component")
        completed = self._shell("logs", component, "100", timeout=8.0)
        return {
            "schema": "z_manip.visual_component_log.v1",
            "component": component,
            "ok": completed.returncode == 0,
            "text": completed.stdout[-24_000:],
        }

    def _run_restart(self, component: str) -> None:
        started_ns = time.time_ns()
        try:
            completed = self._shell("restart", component, timeout=95.0)
            result = {
                "component": component,
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "started_unix_ns": started_ns,
                "finished_unix_ns": time.time_ns(),
                "message": completed.stdout[-4000:],
            }
        except (OSError, subprocess.SubprocessError) as exc:
            result = {
                "component": component,
                "ok": False,
                "started_unix_ns": started_ns,
                "finished_unix_ns": time.time_ns(),
                "message": str(exc),
            }
        with self._lock:
            self._last_result = result
            self._active_component = None

    def _run_bringup(self) -> None:
        started_ns = time.time_ns()
        try:
            completed = self._shell("bringup", timeout=180.0)
            result = {
                "component": "bringup",
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "started_unix_ns": started_ns,
                "finished_unix_ns": time.time_ns(),
                "message": completed.stdout[-4000:],
            }
        except (OSError, subprocess.SubprocessError) as exc:
            result = {
                "component": "bringup",
                "ok": False,
                "started_unix_ns": started_ns,
                "finished_unix_ns": time.time_ns(),
                "message": str(exc),
            }
        with self._lock:
            self._last_result = result
            self._active_component = None

    def restart(self, component: str) -> dict[str, Any]:
        if component not in VISUAL_COMPONENTS or component == "ui":
            raise ValueError("unsupported asynchronous visual component")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {
                    "started": False,
                    "component": self._active_component,
                    "message": "another visual component restart is already running",
                }
            self._active_component = component
            self._thread = threading.Thread(
                target=self._run_restart,
                args=(component,),
                name=f"z-manip-restart-{component}",
                daemon=True,
            )
            self._thread.start()
        return {
            "started": True,
            "component": component,
            "message": f"restart started for {component}",
        }

    def bringup(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {
                    "started": False,
                    "component": self._active_component,
                    "message": "another visual component action is already running",
                }
            self._active_component = "bringup"
            self._thread = threading.Thread(
                target=self._run_bringup,
                name="z-manip-visual-bringup",
                daemon=True,
            )
            self._thread.start()
        return {
            "started": True,
            "component": "bringup",
            "message": "cold visual bringup started",
        }


def _runtime_handler(
    base_handler: type,
    reader: RuntimeStateReader,
    camera_reader: CameraSnapshotReader,
    live_perception: LivePerceptionRenderer,
    session_service: ReadOnlySessionService | None,
    session_artifacts: InteractiveArtifactReader | None,
    home_runner: PiperHomeRunner | None,
    grasp_runner: PiperGraspRunner | None,
    approach_runner: DepthServoRunner | None,
    component_manager: VisualComponentManager | None,
) -> type:
    """Add fixed-path runtime, camera, and read-only session endpoints."""

    class RuntimeDashboardHandler(base_handler):
        def _interactive_error(
            self,
            code: str,
            message: str,
            *,
            status: HTTPStatus,
            include_body: bool = True,
        ) -> None:
            busy = False
            if session_service is not None:
                try:
                    busy = session_service.status().get("busy") is True
                except Exception:
                    busy = False
            self._json(
                {
                    "schema": "z_manip.interactive_session_error.v1",
                    "ok": False,
                    "busy": busy,
                    "error": {"code": code, "message": message},
                },
                status=status,
                include_body=include_body,
            )

        def _interactive_host_valid(self, *, include_body: bool) -> bool:
            expected_host = f"{go2w_debug_ui.LOOPBACK}:{self.server.server_port}"
            hosts = self.headers.get_all("Host", failobj=[])
            if hosts != [expected_host]:
                self._interactive_error(
                    "INVALID_HOST",
                    "interactive session API requires its exact loopback Host",
                    status=HTTPStatus.FORBIDDEN,
                    include_body=include_body,
                )
                return False
            return True

        @staticmethod
        def _decode_strict_json(payload: bytes) -> object:
            def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError(f"duplicate JSON field: {key}")
                    result[key] = value
                return result

            return json.loads(
                payload.decode("utf-8", errors="strict"),
                object_pairs_hook=unique_object,
            )

        def _interactive_json_body(
            self,
            *,
            action: str,
        ) -> dict[str, object] | None:
            if self.headers.get_all("Transfer-Encoding", failobj=[]):
                self._interactive_error(
                    "INVALID_BODY_FRAMING",
                    "chunked or encoded interactive action bodies are forbidden",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return None
            if self.headers.get_all("Content-Encoding", failobj=[]):
                self._interactive_error(
                    "INVALID_BODY_ENCODING",
                    "interactive action bodies cannot be content-encoded",
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return None
            content_types = self.headers.get_all("Content-Type", failobj=[])
            normalized_type = [part.strip().lower() for part in content_types[0].split(";")] if len(content_types) == 1 else []
            if normalized_type not in (["application/json"], ["application/json", "charset=utf-8"]):
                self._interactive_error(
                    "INVALID_CONTENT_TYPE",
                    "interactive actions require application/json with optional UTF-8 charset",
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
                return None
            lengths = self.headers.get_all("Content-Length", failobj=[])
            if len(lengths) != 1 or not lengths[0].isdecimal():
                self._interactive_error(
                    "INVALID_CONTENT_LENGTH",
                    "interactive actions require one decimal Content-Length",
                    status=HTTPStatus.LENGTH_REQUIRED,
                )
                return None
            content_length = int(lengths[0])
            if not 1 <= content_length <= MAX_INTERACTIVE_REQUEST_BYTES:
                self._interactive_error(
                    "INVALID_BODY_SIZE",
                    f"interactive action JSON must be 1..{MAX_INTERACTIVE_REQUEST_BYTES} bytes",
                    status=(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                        if content_length > MAX_INTERACTIVE_REQUEST_BYTES
                        else HTTPStatus.BAD_REQUEST
                    ),
                )
                return None
            try:
                document = self._decode_strict_json(self.rfile.read(content_length))
            except (UnicodeError, json.JSONDecodeError, ValueError) as error:
                self._interactive_error(
                    "INVALID_JSON",
                    f"interactive action body is invalid JSON: {error}",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return None
            if not isinstance(document, dict):
                self._interactive_error(
                    "INVALID_JSON_OBJECT",
                    "interactive action body must contain one JSON object",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return None
            exact = (
                {"target"}
                if action == INTERACTIVE_PERCEPTION_ACTION
                else (
                    {"component"}
                    if action == COMPONENT_RESTART_ACTION
                    else (set() if action == APPROACH_START_ACTION else set())
                )
            )
            optional = {"speed_percent"} if action in (
                HOME_ACTION,
                DIRECT_GRASP_ACTION,
                RETURN_HOME_HOLDING_ACTION,
                PLACE_BACK_ACTION,
            ) else set()
            if action == APPROACH_START_ACTION:
                valid_fields = "mode" in document and set(document).issubset({
                    "mode", "target", "acquire_target", "auto_handoff", "operator_present", "speed_percent",
                })
            elif action in (GRASP_ACTION, PICK_HOLD_ACTION):
                valid_fields = "target" in document and set(document).issubset(
                    {"target", "speed_percent"}
                )
            else:
                valid_fields = (
                    set(document) == exact
                    if not optional
                    else set(document).issubset(optional)
                )
            if not valid_fields:
                if action == INTERACTIVE_PERCEPTION_ACTION:
                    field_message = "perception accepts exactly the target field"
                elif action == COMPONENT_RESTART_ACTION:
                    field_message = "component restart accepts exactly the component field"
                elif action in (GRASP_ACTION, PICK_HOLD_ACTION):
                    field_message = "grasp accepts a required target and an optional speed_percent"
                elif action == APPROACH_START_ACTION:
                    field_message = "approach requires mode and accepts target, acquire_target, auto_handoff, operator_present, and speed_percent"
                elif optional:
                    field_message = f"{action} accepts only an optional speed_percent"
                else:
                    field_message = f"{action} accepts an empty JSON object and no parameters"
                self._interactive_error(
                    "INVALID_ACTION_FIELDS",
                    field_message,
                    status=HTTPStatus.BAD_REQUEST,
                )
                return None
            if "speed_percent" in document:
                speed = document["speed_percent"]
                if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= 50:
                    self._interactive_error(
                        "INVALID_SPEED",
                        "speed_percent must be an integer from 1 to 50",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return None
            if action == APPROACH_START_ACTION and document.get("mode") not in {"shadow", "live"}:
                self._interactive_error(
                    "INVALID_MODE",
                    "visual approach mode must be shadow or live",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return None
            if action == APPROACH_START_ACTION:
                for field in ("acquire_target", "auto_handoff", "operator_present"):
                    if field in document and not isinstance(document[field], bool):
                        self._interactive_error(
                            "INVALID_APPROACH_OPTION",
                            f"{field} must be boolean",
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return None
                if "target" in document:
                    try:
                        validate_target_description(document["target"])
                    except SessionContractError as error:
                        self._interactive_error(
                            error.code,
                            str(error),
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return None
                if (document.get("acquire_target") or document.get("auto_handoff")) and "target" not in document:
                    self._interactive_error(
                        "TARGET_REQUIRED",
                        "automatic approach requires target",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return None
            if action in (GRASP_ACTION, PICK_HOLD_ACTION):
                try:
                    validate_target_description(document.get("target"))
                except SessionContractError as error:
                    self._interactive_error(
                        error.code,
                        str(error),
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return None
            if action == COMPONENT_RESTART_ACTION and document.get("component") not in VISUAL_COMPONENTS:
                self._interactive_error(
                    "INVALID_COMPONENT",
                    "component must be one of ui, nuc-camera, passive-feedback, observer, rgbd, edgetam, perception, perception-all",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return None
            return document

        def _component_get_route(self, *, include_body: bool) -> bool:
            if component_manager is None:
                return False
            route = urlsplit(self.path)
            is_status = route.path == COMPONENT_STATUS_ROUTE
            is_log = route.path.startswith(COMPONENT_LOG_ROUTE_PREFIX)
            if not is_status and not is_log:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "component endpoints accept no query string",
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return True
            if not self._interactive_host_valid(include_body=include_body):
                return True
            if is_status:
                self._json(component_manager.status(), include_body=include_body)
                return True
            component = route.path.removeprefix(COMPONENT_LOG_ROUTE_PREFIX)
            if component not in LOG_COMPONENTS or "/" in component:
                self._interactive_error(
                    "INVALID_COMPONENT",
                    "unsupported component log",
                    status=HTTPStatus.NOT_FOUND,
                    include_body=include_body,
                )
                return True
            try:
                self._json(component_manager.logs(component), include_body=include_body)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self._interactive_error(
                    "COMPONENT_LOG_UNAVAILABLE",
                    str(error),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    include_body=include_body,
                )
            return True

        def _interactive_get_route(self, *, include_body: bool) -> bool:
            if session_service is None or session_artifacts is None:
                return False
            route = urlsplit(self.path)
            known = (
                route.path == INTERACTIVE_STATUS_ROUTE
                or route.path in INTERACTIVE_PERCEPTION_ARTIFACTS
                or route.path == INTERACTIVE_PLANNING_BUNDLE_ROUTE
            )
            if not known:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "interactive session endpoints accept no query string",
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return True
            if not self._interactive_host_valid(include_body=include_body):
                return True
            try:
                if route.path == INTERACTIVE_STATUS_ROUTE:
                    self._json(session_service.status(), include_body=include_body)
                    return True
                if route.path in INTERACTIVE_PERCEPTION_ARTIFACTS:
                    payload = session_artifacts.perception_png(
                        INTERACTIVE_PERCEPTION_ARTIFACTS[route.path],
                    )
                    self._bytes(payload, "image/png", include_body=include_body)
                    return True
                payload = session_artifacts.planning_bundle()
                self._bytes(
                    payload,
                    "application/json; charset=utf-8",
                    include_body=include_body,
                )
                return True
            except (SessionContractError, InteractiveArtifactError) as error:
                code = getattr(error, "code", "INTERACTIVE_ARTIFACT_UNAVAILABLE")
                self._interactive_error(
                    code,
                    str(error),
                    status=HTTPStatus.CONFLICT,
                    include_body=include_body,
                )
                return True
            except Exception:
                self._interactive_error(
                    "INTERACTIVE_STATUS_UNAVAILABLE",
                    "interactive session state is unavailable",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    include_body=include_body,
                )
                return True

        def _display_bundle_route(self, *, include_body: bool) -> bool:
            """Serve only the bundle bound to the active interactive task."""

            if session_service is None or session_artifacts is None:
                return False
            route = urlsplit(self.path)
            if route.path != "/api/bundle":
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "display bundle accepts no query string",
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return True
            if not self._interactive_host_valid(include_body=include_body):
                return True
            try:
                state = session_service.status()
                if state.get("selected_perception_session_id") is None:
                    self._json(_empty_display_bundle(), include_body=include_body)
                    return True
                try:
                    payload = session_artifacts.planning_bundle()
                except InteractiveArtifactError:
                    self._json(_empty_display_bundle(), include_body=include_body)
                    return True
                self._bytes(
                    payload,
                    "application/json; charset=utf-8",
                    include_body=include_body,
                )
                return True
            except (SessionContractError, OSError, ValueError):
                self._interactive_error(
                    "DISPLAY_BUNDLE_UNAVAILABLE",
                    "current display task state is unavailable",
                    status=HTTPStatus.CONFLICT,
                    include_body=include_body,
                )
                return True

        def _home_get_route(self, *, include_body: bool) -> bool:
            if home_runner is None:
                return False
            route = urlsplit(self.path)
            if route.path != HOME_STATUS_ROUTE:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "Home status accepts no query string",
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return True
            if not self._interactive_host_valid(include_body=include_body):
                return True
            self._json(home_runner.status(), include_body=include_body)
            return True

        def _approach_get_route(self, *, include_body: bool) -> bool:
            if approach_runner is None:
                return False
            route = urlsplit(self.path)
            if route.path != APPROACH_STATUS_ROUTE:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "visual approach status accepts no query string",
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return True
            if not self._interactive_host_valid(include_body=include_body):
                return True
            self._json(approach_runner.status(), include_body=include_body)
            return True

        def _approach_post_route(self) -> bool:
            if approach_runner is None:
                return False
            route = urlsplit(self.path)
            actions = {
                APPROACH_START_ROUTE: APPROACH_START_ACTION,
                APPROACH_STOP_ROUTE: APPROACH_STOP_ACTION,
            }
            action = actions.get(route.path)
            if action is None:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "visual approach actions accept no query string",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            if not self._interactive_host_valid(include_body=True):
                return True
            expected_origin = f"http://{go2w_debug_ui.LOOPBACK}:{self.server.server_port}"
            if self.headers.get_all("Origin", failobj=[]) != [expected_origin]:
                self._interactive_error(
                    "CROSS_ORIGIN_FORBIDDEN",
                    "visual approach requires the exact loopback same-origin page",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            if self.headers.get_all(INTERACTIVE_ACTION_HEADER, failobj=[]) != [action]:
                self._interactive_error(
                    "ACTION_HEADER_REQUIRED",
                    f"{INTERACTIVE_ACTION_HEADER}: {action} is required",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            document = self._interactive_json_body(action=action)
            if document is None:
                return True
            if action == APPROACH_STOP_ACTION:
                result = approach_runner.stop()
                # Full Stop is the mobile-approach recovery boundary. Clear
                # selected task context asynchronously so an abandoned auto
                # handoff cannot restart from stale perception or planning.
                if session_service is not None:
                    threading.Thread(
                        target=session_service.clear_current_context,
                        name="z-manip-full-stop-context-reset",
                        daemon=True,
                    ).start()
                    result["task_context_clear_requested"] = True
                self._json(result, status=HTTPStatus.OK, include_body=True)
                return True
            active = bool(
                (session_service is not None and session_service.status().get("busy") is True)
                or (home_runner is not None and home_runner.status().get("running") is True)
                or (grasp_runner is not None and grasp_runner.status().get("running") is True)
            )
            if active:
                self._interactive_error(
                    "ACTION_BUSY",
                    "wait for the current Home, perception, planning, or grasp action to finish",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            result = approach_runner.start(
                str(document["mode"]),
                target=document.get("target"),
                acquire_target=document.get("acquire_target") is True,
                auto_handoff=document.get("auto_handoff") is True,
                operator_present=document.get("operator_present") is True,
                speed_percent=int(document.get("speed_percent", 5)),
            )
            self._json(
                result,
                status=HTTPStatus.ACCEPTED if result.get("started") else HTTPStatus.CONFLICT,
                include_body=True,
            )
            return True

        def _grasp_get_route(self, *, include_body: bool) -> bool:
            if grasp_runner is None:
                return False
            route = urlsplit(self.path)
            if route.path != GRASP_STATUS_ROUTE:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "grasp status accepts no query string",
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return True
            if not self._interactive_host_valid(include_body=include_body):
                return True
            self._json(grasp_runner.status(), include_body=include_body)
            return True

        def _home_post_route(self) -> bool:
            if home_runner is None:
                return False
            route = urlsplit(self.path)
            if route.path != HOME_ROUTE:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "Home accepts no query string",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            if not self._interactive_host_valid(include_body=True):
                return True
            expected_origin = f"http://{go2w_debug_ui.LOOPBACK}:{self.server.server_port}"
            if self.headers.get_all("Origin", failobj=[]) != [expected_origin]:
                self._interactive_error(
                    "CROSS_ORIGIN_FORBIDDEN",
                    "Home requires the exact loopback same-origin page",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            if self.headers.get_all(INTERACTIVE_ACTION_HEADER, failobj=[]) != [HOME_ACTION]:
                self._interactive_error(
                    "ACTION_HEADER_REQUIRED",
                    f"{INTERACTIVE_ACTION_HEADER}: {HOME_ACTION} is required",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            document = self._interactive_json_body(action=HOME_ACTION)
            if document is None:
                return True
            if grasp_runner is not None and grasp_runner.status().get("running") is True:
                self._interactive_error(
                    "ACTION_BUSY",
                    "a grasp action is still running",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            if approach_runner is not None and approach_runner.status().get("running") is True:
                self._interactive_error(
                    "ACTION_BUSY",
                    "stop the Go2W visual approach before moving the arm Home",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            # Home is also the software recovery button.  Clear stale grasp
            # state before launching the fixed physical recovery so a failed
            # prior workflow cannot keep the UI locked.  Read-only session
            # cleanup runs separately and is repeated after measured Home.
            if grasp_runner is not None:
                grasp_runner.reset_for_home()
            if session_service is not None:
                threading.Thread(
                    target=session_service.clear_current_context,
                    name="z-manip-home-context-reset",
                    daemon=True,
                ).start()
            speed = document.get("speed_percent")
            result = home_runner.start() if speed is None else home_runner.start(speed)
            self._json(
                result,
                status=HTTPStatus.ACCEPTED if result.get("started") else HTTPStatus.CONFLICT,
                include_body=True,
            )
            return True

        def _grasp_post_route(self) -> bool:
            if grasp_runner is None:
                return False
            route = urlsplit(self.path)
            actions = {
                GRASP_ROUTE: GRASP_ACTION,
                DIRECT_GRASP_ROUTE: DIRECT_GRASP_ACTION,
                PICK_HOLD_ROUTE: PICK_HOLD_ACTION,
                RETURN_HOME_HOLDING_ROUTE: RETURN_HOME_HOLDING_ACTION,
                PLACE_BACK_ROUTE: PLACE_BACK_ACTION,
            }
            action = actions.get(route.path)
            if action is None:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "grasp accepts no query string",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            if not self._interactive_host_valid(include_body=True):
                return True
            expected_origin = f"http://{go2w_debug_ui.LOOPBACK}:{self.server.server_port}"
            if self.headers.get_all("Origin", failobj=[]) != [expected_origin]:
                self._interactive_error(
                    "CROSS_ORIGIN_FORBIDDEN",
                    "grasp requires the exact loopback same-origin page",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            if self.headers.get_all(INTERACTIVE_ACTION_HEADER, failobj=[]) != [action]:
                self._interactive_error(
                    "ACTION_HEADER_REQUIRED",
                    f"{INTERACTIVE_ACTION_HEADER}: {action} is required",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            document = self._interactive_json_body(action=action)
            if document is None:
                return True
            if session_service is not None and session_service.status().get("busy") is True:
                self._interactive_error(
                    "ACTION_BUSY",
                    "perception or planning is still running",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            if home_runner is not None and home_runner.status().get("running") is True:
                self._interactive_error(
                    "ACTION_BUSY",
                    "Home is still running",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            if approach_runner is not None and approach_runner.status().get("running") is True:
                self._interactive_error(
                    "ACTION_BUSY",
                    "stop the Go2W visual approach before executing a grasp",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            speed = document.get("speed_percent")
            if action == DIRECT_GRASP_ACTION:
                result = (
                    grasp_runner.start_selected()
                    if speed is None
                    else grasp_runner.start_selected(speed)
                )
            elif action == PICK_HOLD_ACTION:
                target = validate_target_description(document.get("target"))
                result = (
                    grasp_runner.start_pick_hold(target)
                    if speed is None
                    else grasp_runner.start_pick_hold(target, speed)
                )
            elif action == RETURN_HOME_HOLDING_ACTION:
                result = (
                    grasp_runner.start_return_home_holding()
                    if speed is None
                    else grasp_runner.start_return_home_holding(speed)
                )
            elif action == PLACE_BACK_ACTION:
                result = (
                    grasp_runner.start_place_back()
                    if speed is None
                    else grasp_runner.start_place_back(speed)
                )
            else:
                target = validate_target_description(document.get("target"))
                result = (
                    grasp_runner.start(target)
                    if speed is None
                    else grasp_runner.start(target, speed)
                )
            self._json(
                result,
                status=HTTPStatus.ACCEPTED if result.get("started") else HTTPStatus.CONFLICT,
                include_body=True,
            )
            return True

        def _maintenance_post_route(self) -> bool:
            route = urlsplit(self.path)
            actions = {
                SESSION_CLEAR_ROUTE: SESSION_CLEAR_ACTION,
                SERVICE_RESTART_ROUTE: SERVICE_RESTART_ACTION,
            }
            action = actions.get(route.path)
            if action is None:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN", "maintenance actions accept no query string",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            if not self._interactive_host_valid(include_body=True):
                return True
            expected_origin = f"http://{go2w_debug_ui.LOOPBACK}:{self.server.server_port}"
            if self.headers.get_all("Origin", failobj=[]) != [expected_origin]:
                self._interactive_error(
                    "CROSS_ORIGIN_FORBIDDEN",
                    "maintenance actions require the exact loopback same-origin page",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            if self.headers.get_all(INTERACTIVE_ACTION_HEADER, failobj=[]) != [action]:
                self._interactive_error(
                    "ACTION_HEADER_REQUIRED",
                    f"{INTERACTIVE_ACTION_HEADER}: {action} is required",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            if self._interactive_json_body(action=action) is None:
                return True
            active = bool(
                (session_service is not None and session_service.status().get("busy") is True)
                or (home_runner is not None and home_runner.status().get("running") is True)
                or (grasp_runner is not None and grasp_runner.status().get("running") is True)
                or (approach_runner is not None and approach_runner.status().get("running") is True)
            )
            if active:
                self._interactive_error(
                    "ACTION_BUSY",
                    "wait for the current Home, perception, planning, or grasp action to finish",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            if action == SESSION_CLEAR_ACTION:
                if session_service is None:
                    self._interactive_error(
                        "SESSION_API_UNAVAILABLE", "session service is unavailable",
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return True
                result = session_service.clear_current_context()
                self._json(
                    {"ok": True, "result": result, "session": session_service.status()},
                    include_body=True,
                )
                return True
            completed = subprocess.run(
                [
                    "/usr/bin/systemd-run", "--user", "--collect",
                    "--unit=z-manip-workbench-restart", "--on-active=1s",
                    "/usr/bin/systemctl", "--user", "restart", SERVICE_UNIT,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=3.0,
            )
            if completed.returncode != 0:
                self._interactive_error(
                    "RESTART_SCHEDULE_FAILED",
                    "could not schedule the fixed workbench service restart",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return True
            self._json(
                {"ok": True, "scheduled": True, "service": SERVICE_UNIT},
                status=HTTPStatus.ACCEPTED,
                include_body=True,
            )
            return True

        def _component_post_route(self) -> bool:
            if component_manager is None:
                return False
            route = urlsplit(self.path)
            actions = {
                COMPONENT_RESTART_ROUTE: COMPONENT_RESTART_ACTION,
                COMPONENT_BRINGUP_ROUTE: COMPONENT_BRINGUP_ACTION,
            }
            action = actions.get(route.path)
            if action is None:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "component actions accept no query string",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            if not self._interactive_host_valid(include_body=True):
                return True
            expected_origin = f"http://{go2w_debug_ui.LOOPBACK}:{self.server.server_port}"
            if self.headers.get_all("Origin", failobj=[]) != [expected_origin]:
                self._interactive_error(
                    "CROSS_ORIGIN_FORBIDDEN",
                    "component actions require the exact loopback same-origin page",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            if self.headers.get_all(INTERACTIVE_ACTION_HEADER, failobj=[]) != [action]:
                self._interactive_error(
                    "ACTION_HEADER_REQUIRED",
                    f"{INTERACTIVE_ACTION_HEADER}: {action} is required",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            document = self._interactive_json_body(action=action)
            if document is None:
                return True
            active = bool(
                (session_service is not None and session_service.status().get("busy") is True)
                or (home_runner is not None and home_runner.status().get("running") is True)
                or (grasp_runner is not None and grasp_runner.status().get("running") is True)
            )
            if active:
                self._interactive_error(
                    "ACTION_BUSY",
                    "wait for the current Home, perception, planning, or grasp action to finish",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            component = str(document["component"]) if action == COMPONENT_RESTART_ACTION else "bringup"
            if component != "ui" and session_service is not None:
                session_service.clear_current_context()
            if action == COMPONENT_BRINGUP_ACTION:
                result = component_manager.bringup()
            elif component == "ui":
                completed = subprocess.run(
                    [
                        "/usr/bin/systemd-run", "--user", "--collect",
                        "--unit=z-manip-component-ui-restart", "--on-active=1s",
                        str(component_manager.script), "restart", "ui",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=3.0,
                )
                if completed.returncode != 0:
                    self._interactive_error(
                        "RESTART_SCHEDULE_FAILED",
                        "could not schedule the fixed UI restart",
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return True
                result = {"started": True, "component": "ui", "scheduled": True}
            else:
                result = component_manager.restart(component)
            self._json(
                {"ok": result.get("started") is True, "restart": result},
                status=(
                    HTTPStatus.ACCEPTED
                    if result.get("started") is True
                    else HTTPStatus.CONFLICT
                ),
                include_body=True,
            )
            return True

        def _interactive_post_route(self) -> bool:
            if session_service is None:
                return False
            route = urlsplit(self.path)
            actions = {
                INTERACTIVE_PERCEPTION_ROUTE: INTERACTIVE_PERCEPTION_ACTION,
                INTERACTIVE_PLANNING_ROUTE: INTERACTIVE_PLANNING_ACTION,
            }
            action = actions.get(route.path)
            if action is None:
                return False
            if route.query or route.fragment:
                self._interactive_error(
                    "QUERY_FORBIDDEN",
                    "interactive session actions accept no query string",
                    status=HTTPStatus.BAD_REQUEST,
                )
                return True
            if not self._interactive_host_valid(include_body=True):
                return True
            expected_origin = f"http://{go2w_debug_ui.LOOPBACK}:{self.server.server_port}"
            origins = self.headers.get_all("Origin", failobj=[])
            fetch_sites = self.headers.get_all("Sec-Fetch-Site", failobj=[])
            if origins != [expected_origin] or (
                fetch_sites and fetch_sites != ["same-origin"]
            ):
                self._interactive_error(
                    "CROSS_ORIGIN_FORBIDDEN",
                    "interactive actions require the exact loopback same-origin page",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            action_headers = self.headers.get_all(INTERACTIVE_ACTION_HEADER, failobj=[])
            if action_headers != [action]:
                self._interactive_error(
                    "ACTION_HEADER_REQUIRED",
                    f"{INTERACTIVE_ACTION_HEADER}: {action} is required",
                    status=HTTPStatus.FORBIDDEN,
                )
                return True
            document = self._interactive_json_body(action=action)
            if document is None:
                return True
            if grasp_runner is not None and grasp_runner.status().get("running") is True:
                self._interactive_error(
                    "ACTION_BUSY",
                    "a grasp action is still running",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            if approach_runner is not None and approach_runner.status().get("running") is True:
                self._interactive_error(
                    "ACTION_BUSY",
                    "stop the Go2W visual approach before starting perception or planning",
                    status=HTTPStatus.CONFLICT,
                )
                return True
            try:
                if action == INTERACTIVE_PERCEPTION_ACTION:
                    target = validate_target_description(document["target"])
                    attempt = session_service.start_perception(target)
                else:
                    attempt = session_service.start_planning()
                state = session_service.status()
            except SessionContractError as error:
                self._interactive_error(
                    error.code,
                    str(error),
                    status=(
                        HTTPStatus.CONFLICT
                        if error.code == "ACTION_BUSY"
                        else HTTPStatus.BAD_REQUEST
                    ),
                )
                return True
            except Exception:
                self._interactive_error(
                    "INTERACTIVE_ACTION_FAILED",
                    "interactive read-only action failed safely",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return True
            succeeded = attempt.get("status") == "succeeded"
            response = {
                "schema": "z_manip.interactive_action_response.v1",
                "ok": succeeded,
                "action": action,
                "status": attempt.get("status"),
                "busy": state.get("busy") is True,
                "attempt": attempt,
                "session": state,
            }
            self._json(
                response,
                status=HTTPStatus.OK if succeeded else HTTPStatus.CONFLICT,
                include_body=True,
            )
            return True

        def _camera_headers(
            self,
            status: HTTPStatus,
            *,
            content_type: str,
            length: int,
            camera_state: str,
            age_s: float | None,
            etag: str | None = None,
            perception_state: str | None = None,
            reference_age_s: float | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            if etag is not None:
                self.send_header("ETag", etag)
            self.send_header("X-Z-Manip-Camera-State", camera_state)
            if perception_state is not None:
                self.send_header("X-Z-Manip-Perception-State", perception_state)
            if reference_age_s is not None:
                self.send_header(
                    "X-Z-Manip-Reference-Age-Ms",
                    str(max(0, round(reference_age_s * 1000))),
                )
            if age_s is not None:
                self.send_header("X-Z-Manip-Camera-Age-Ms", str(max(0, round(age_s * 1000))))
            self.send_header("X-Z-Manip-Poll-Interval-Ms", "200")
            self.send_header("Content-Security-Policy", go2w_debug_ui.SECURITY_POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()

        def _camera_route(self, *, include_body: bool) -> bool:
            route = urlsplit(self.path)
            if route.path != "/api/camera/latest.jpg":
                return False
            if route.query or route.fragment:
                payload = b'{"error":"camera endpoint accepts no query string"}\n'
                self._camera_headers(
                    HTTPStatus.BAD_REQUEST,
                    content_type="application/json; charset=utf-8",
                    length=len(payload),
                    camera_state="invalid",
                    age_s=None,
                )
                if include_body:
                    self.wfile.write(payload)
                return True
            status, payload, etag, age_s, message = camera_reader.snapshot()
            if status == "live" and payload is not None and etag is not None:
                if self.headers.get("If-None-Match") == etag:
                    self._camera_headers(
                        HTTPStatus.NOT_MODIFIED,
                        content_type="image/jpeg",
                        length=0,
                        camera_state="live",
                        age_s=age_s,
                        etag=etag,
                    )
                    return True
                self._camera_headers(
                    HTTPStatus.OK,
                    content_type="image/jpeg",
                    length=len(payload),
                    camera_state="live",
                    age_s=age_s,
                    etag=etag,
                )
                if include_body:
                    self.wfile.write(payload)
                return True
            error_payload = (json.dumps({"error": message}, separators=(",", ":")) + "\n").encode("utf-8")
            http_status = {
                "stale": HTTPStatus.SERVICE_UNAVAILABLE,
                "invalid": HTTPStatus.CONFLICT,
            }.get(status, HTTPStatus.NOT_FOUND)
            self._camera_headers(
                http_status,
                content_type="application/json; charset=utf-8",
                length=len(error_payload),
                camera_state=status,
                age_s=age_s,
            )
            if include_body:
                self.wfile.write(error_payload)
            return True

        def _live_perception_route(self, *, include_body: bool) -> bool:
            route = urlsplit(self.path)
            kind = LIVE_PERCEPTION_ROUTES.get(route.path)
            if kind is None:
                return False
            if route.query or route.fragment:
                payload = b'{"error":"live perception endpoint accepts no query string"}\n'
                self._camera_headers(
                    HTTPStatus.BAD_REQUEST,
                    content_type="application/json; charset=utf-8",
                    length=len(payload),
                    camera_state="invalid",
                    age_s=None,
                )
                if include_body:
                    self.wfile.write(payload)
                return True
            state_name, payload, etag, camera_age_s, reference_age_s, detail = live_perception.snapshot(kind)
            content_type = "image/png" if kind == "mask" else "image/jpeg"
            if payload is not None and etag is not None and state_name in {"fresh", "tracked"}:
                if self.headers.get("If-None-Match") == etag:
                    self._camera_headers(
                        HTTPStatus.NOT_MODIFIED,
                        content_type=content_type,
                        length=0,
                        camera_state=state_name,
                        age_s=camera_age_s,
                        etag=etag,
                        perception_state=state_name,
                        reference_age_s=reference_age_s,
                    )
                else:
                    self._camera_headers(
                        HTTPStatus.OK,
                        content_type=content_type,
                        length=len(payload),
                        camera_state=state_name,
                        age_s=camera_age_s,
                        etag=etag,
                        perception_state=state_name,
                        reference_age_s=reference_age_s,
                    )
                    if include_body:
                        self.wfile.write(payload)
                return True
            error_payload = (json.dumps({
                "error": detail,
                "state": state_name,
                "reference_age_ms": None if reference_age_s is None else max(0, round(reference_age_s * 1000)),
            }, separators=(",", ":")) + "\n").encode("utf-8")
            self._camera_headers(
                HTTPStatus.CONFLICT if state_name in {"stale", "invalid"} else HTTPStatus.NOT_FOUND,
                content_type="application/json; charset=utf-8",
                length=len(error_payload),
                camera_state=state_name,
                age_s=camera_age_s,
                perception_state=state_name,
                reference_age_s=reference_age_s,
            )
            if include_body:
                self.wfile.write(error_payload)
            return True

        def _runtime_route(self, *, include_body: bool) -> bool:
            route = urlsplit(self.path)
            if route.path != "/api/runtime":
                return False
            if route.query or route.fragment:
                self._json(
                    {"error": "runtime endpoint accepts no query string or path argument"},
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return True
            document, etag = reader.snapshot()
            if self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                self.send_header("ETag", etag)
                self.send_header("X-Z-Manip-Poll-Interval-Ms", "200")
                self.send_header("Content-Security-Policy", go2w_debug_ui.SECURITY_POLICY)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.end_headers()
                return True
            payload = (json.dumps(document, separators=(",", ":")) + "\n").encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("ETag", etag)
            self.send_header("X-Z-Manip-Poll-Interval-Ms", "200")
            self.send_header("Content-Security-Policy", go2w_debug_ui.SECURITY_POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            if include_body:
                self.wfile.write(payload)
            return True

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            if (
                not self._component_get_route(include_body=True)
                and not self._approach_get_route(include_body=True)
                and not self._grasp_get_route(include_body=True)
                and not self._home_get_route(include_body=True)
                and not self._display_bundle_route(include_body=True)
                and not self._interactive_get_route(include_body=True)
                and not self._live_perception_route(include_body=True)
                and not self._camera_route(include_body=True)
                and not self._runtime_route(include_body=True)
            ):
                super().do_GET()

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            if (
                not self._component_get_route(include_body=False)
                and not self._approach_get_route(include_body=False)
                and not self._grasp_get_route(include_body=False)
                and not self._home_get_route(include_body=False)
                and not self._display_bundle_route(include_body=False)
                and not self._interactive_get_route(include_body=False)
                and not self._live_perception_route(include_body=False)
                and not self._camera_route(include_body=False)
                and not self._runtime_route(include_body=False)
            ):
                super().do_HEAD()

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            if (
                not self._component_post_route()
                and not self._approach_post_route()
                and not self._maintenance_post_route()
                and not self._grasp_post_route()
                and not self._home_post_route()
                and not self._interactive_post_route()
            ):
                super().do_POST()

    return RuntimeDashboardHandler


def create_server(
    bundle_path: Path,
    *,
    port: int,
    index_path: Path,
    control_backend: PlanningOnlyRunner,
    runtime_state: Path | None,
    camera_image: Path | None = None,
    interactive_service: ReadOnlySessionService | None = None,
    interactive_run_root: Path | None = None,
    home_runner: PiperHomeRunner | None = None,
    grasp_runner: PiperGraspRunner | None = None,
    approach_runner: DepthServoRunner | None = None,
    component_manager: VisualComponentManager | None = None,
) -> go2w_debug_ui.LoopbackHTTPServer:
    """Create the loopback dashboard plus a fixed-path runtime snapshot API."""

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    base_handler = go2w_debug_ui.make_handler(
        bundle_path,
        index_path,
        control_backend=control_backend,
        follow_bundle_symlink=True,
    )
    camera_reader = CameraSnapshotReader(camera_image)
    session_artifacts = (
        None
        if interactive_service is None
        else InteractiveArtifactReader(
            interactive_run_root or go2w_interactive_sessions.RUN_ROOT,
            interactive_service,
        )
    )
    handler = _runtime_handler(
        base_handler,
        RuntimeStateReader(runtime_state),
        camera_reader,
        LivePerceptionRenderer(camera_reader, session_artifacts),
        interactive_service,
        session_artifacts,
        home_runner,
        grasp_runner,
        approach_runner,
        component_manager,
    )
    return go2w_debug_ui.LoopbackHTTPServer(
        (go2w_debug_ui.LOOPBACK, port),
        handler,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session-script", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--runtime-state",
        type=Path,
        help=(
            "fixed producer-written z_manip.runtime_state.v1 JSON; omitted or "
            "missing files are reported as offline"
        ),
    )
    parser.add_argument(
        "--camera-image",
        type=Path,
        help="fixed observer-written JPEG served only at /api/camera/latest.jpg",
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--home-script", type=Path)
    parser.add_argument("--grasp-script", type=Path)
    parser.add_argument("--approach-script", type=Path)
    parser.add_argument("--approach-status", type=Path)
    parser.add_argument("--wrist-search-script", type=Path)
    parser.add_argument("--component-manager", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    run_root = args.run_root.expanduser().resolve()
    bundle_link = run_root / "latest" / "debug_bundle.json"
    runner = PlanningOnlyRunner(args.session_script, run_root)
    interactive_service = ReadOnlySessionService(
        go2w_interactive_sessions.RUN_ROOT,
        go2w_interactive_sessions.FixedReadOnlyBackend(),
    )
    home_runner = None if args.home_script is None else PiperHomeRunner(
        args.home_script,
        run_root / "piper-home.log",
        on_home_reached=interactive_service.clear_current_context,
    )
    grasp_runner = None
    if args.grasp_script is not None:
        if home_runner is None:
            raise ValueError("--grasp-script requires --home-script")
        grasp_runner = PiperGraspRunner(
            args.grasp_script,
            run_root / "piper-grasp.log",
            run_root / "execution-receipts",
            interactive_service,
            go2w_interactive_sessions.RUN_ROOT,
            home_runner,
            MeasuredHomeVerifier(
                args.runtime_state,
                home_runner.script.parents[2] / "configs" / "piper_home.json",
            ),
        )
        def clear_home_context() -> None:
            interactive_service.clear_current_context()
            # Home is the operator's recovery boundary.  A measured Home
            # completion invalidates every staged/held workflow, including a
            # stale holding_at_lift state after the arm was recovered by hand.
            grasp_runner.reset_after_home()
        home_runner.on_home_reached = clear_home_context
    wrist_search = None
    if args.camera_image is not None:
        home_path = (
            args.home_script.parents[2] / "configs" / "piper_home.json"
            if args.home_script is not None
            else Path(__file__).parents[2] / "configs" / "piper_home.json"
        )
        try:
            home_document = json.loads(home_path.read_text(encoding="utf-8"))
            home_joints = home_document["joint_radians"]
            motion = None
            if args.wrist_search_script is not None:
                motion = go2w_wrist_search.FixedWristMotion(
                    args.wrist_search_script,
                    run_root / "piper-wrist-search.log",
                )
            wrist_search = go2w_wrist_search.WristSearchCoordinator(
                home_joints,
                go2w_wrist_search.DetectorProbe(args.camera_image),
                motion=motion,
            )
        except (OSError, KeyError, TypeError, ValueError):
            wrist_search = None
    approach_runner = None
    if args.approach_script is not None:
        if args.approach_status is None:
            raise ValueError("--approach-script requires --approach-status")
        approach_runner = DepthServoRunner(
            args.approach_script,
            args.approach_status,
            run_root / "go2w-depth-servo.log",
            session_service=interactive_service,
            grasp_runner=grasp_runner,
            wrist_search=wrist_search,
        )
    component_script = (
        args.component_manager
        if args.component_manager is not None
        else Path(__file__).with_name("go2w_component_manager.sh")
    )
    component_manager = VisualComponentManager(component_script)
    server = create_server(
        bundle_link,
        port=args.port,
        index_path=args.index,
        control_backend=runner,
        runtime_state=args.runtime_state,
        camera_image=args.camera_image,
        interactive_service=interactive_service,
        interactive_run_root=go2w_interactive_sessions.RUN_ROOT,
        home_runner=home_runner,
        grasp_runner=grasp_runner,
        approach_runner=approach_runner,
        component_manager=component_manager,
    )
    host, port = server.server_address[:2]
    print(f"Z-Manip planning-only control dashboard: http://{host}:{port}/", flush=True)
    print(f"dynamic bundle: {bundle_link}", flush=True)
    print(
        "runtime state: "
        + ("offline (not configured)" if args.runtime_state is None else str(args.runtime_state.expanduser().resolve())),
        flush=True,
    )
    print(
        "camera image: "
        + ("offline (not configured)" if args.camera_image is None else str(args.camera_image.expanduser().resolve())),
        flush=True,
    )
    print(
        f"interactive sessions: {go2w_interactive_sessions.RUN_ROOT} "
        "(read-only perception + offline planning; fixed staged grasp only when configured)",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("planning-only control dashboard stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
