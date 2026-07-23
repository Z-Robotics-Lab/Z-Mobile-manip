#!/usr/bin/env python3
"""Subscribe-only ROS 2 runtime observer for the Go2W manipulation lab.

The module is importable without ROS. ROS message packages are loaded only by
``run_ros_observer``. The observer creates subscriptions, never publishers,
service clients, action clients, actuator transports, or SocketCAN handles.
It atomically replaces one JSON snapshot so offline dashboards never read a
partially-written runtime state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat as stat_module
import struct
import time
from typing import Any, Callable


SCHEMA = "z_manip.runtime_state.v1"
MAX_CLOUD_SAMPLE_POINTS = 1500
MAX_EXISTING_STATE_BYTES = 8 * 1024 * 1024
MAX_CAMERA_WIDTH = 640
MAX_CAMERA_HEIGHT = 480
MAX_CAMERA_JPEG_BYTES = 512 * 1024
CAMERA_JPEG_QUALITY = 82
# Fixed colorization window for the live depth tile.  A fixed near/far range
# keeps the turbo view stable frame-to-frame (per-frame min/max would flicker)
# and covers the typical D435 indoor manipulation working distance.  Invalid
# (zero / non-finite) depth pixels are rendered black.
DEPTH_COLORMAP_NEAR_M = 0.20
DEPTH_COLORMAP_FAR_M = 4.00
# The depth tile targets ~10 Hz.  Colorization is heavier than the RGB path, so
# the writer coalesces bursts to this minimum interval.  Because the depth topic
# arrives at the camera rate (30 Hz), the achieved write rate is quantized to
# writing every Nth frame: this interval selects "every 3rd frame" ~= 10 Hz.
# It sits below 3x the nominal 33 ms frame spacing on purpose — the live D435
# stream jitters (measured std dev ~11 ms), and a tighter threshold keeps a
# late 3rd frame from slipping to every-4th (~8.6 Hz measured at 0.08).  The
# browser polls slightly faster than this so it catches every distinct frame.
# CPU stays well under one core alongside the 30 Hz RGB path.
DEPTH_WRITE_MIN_INTERVAL_S = 0.075
DEFAULT_TOPICS = {
    "joint_state": ("/piper/state", "sensor_msgs/msg/JointState", 0.50),
    "color": ("/camera/color/image_raw", "sensor_msgs/msg/Image", 1.00),
    "depth": (
        "/camera/aligned_depth_to_color/image_raw",
        "sensor_msgs/msg/Image",
        1.00,
    ),
    "camera_info": (
        "/camera/color/camera_info",
        "sensor_msgs/msg/CameraInfo",
        1.00,
    ),
    "scene_cloud": (
        "/z_manip/perception/scene_pointcloud",
        "sensor_msgs/msg/PointCloud2",
        1.50,
    ),
    "target_cloud": (
        "/z_manip/perception/target_pointcloud",
        "sensor_msgs/msg/PointCloud2",
        1.50,
    ),
    "depth_filter": (
        "/track_3d/frame_manifest",
        "std_msgs/msg/String",
        1.50,
    ),
    "tracker_target": (
        "/track_3d/selected_target_pointcloud",
        "sensor_msgs/msg/PointCloud2",
        0.75,
    ),
    "tracker_state": ("/track_3d/is_tracking", "std_msgs/msg/Bool", 0.75),
    "tracker_failure": ("/track_3d/failure", "std_msgs/msg/String", 5.00),
    "tf": ("/tf", "tf2_msgs/msg/TFMessage", 2.00),
    "tf_static": ("/tf_static", "tf2_msgs/msg/TFMessage", 3600.00),
}


def load_initial_sequence(path: Path) -> int:
    """Recover the last committed sequence without trusting an unbounded file."""

    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
        if not stat_module.S_ISREG(stat.st_mode) or stat.st_size > MAX_EXISTING_STATE_BYTES:
            return 0
        payload = resolved.read_bytes()
        if len(payload) > MAX_EXISTING_STATE_BYTES:
            return 0
        document = json.loads(payload.decode("utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return 0
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        return 0
    sequence = document.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        return 0
    return sequence


def _finite_list(values: object) -> list[float] | None:
    if isinstance(values, (str, bytes, bytearray, dict)):
        return None
    try:
        numeric = [float(value) for value in values]  # type: ignore[union-attr]
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if all(math.isfinite(value) for value in numeric) else None


def stamp_ns(message: object) -> int | None:
    """Return a ROS header stamp without importing a ROS message class."""

    try:
        stamp = message.header.stamp
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        return None
    return seconds * 1_000_000_000 + nanoseconds


def frame_id(message: object) -> str:
    try:
        return str(message.header.frame_id)
    except AttributeError:
        return ""


def summarize_joint_state(message: object) -> dict[str, object]:
    names = [str(value) for value in getattr(message, "name", ())]
    positions = _finite_list(getattr(message, "position", ()))
    velocities = _finite_list(getattr(message, "velocity", ()))
    efforts = _finite_list(getattr(message, "effort", ()))
    error = ""
    if not names:
        error = "joint state has no names"
    elif len(set(names)) != len(names) or any(not name for name in names):
        error = "joint names are empty or duplicated"
    elif positions is None or len(positions) != len(names):
        error = "joint positions are non-finite or do not match names"
    elif velocities is None or len(velocities) not in (0, len(names)):
        error = "joint velocities are malformed"
    elif efforts is None or len(efforts) not in (0, len(names)):
        error = "joint efforts are malformed"
    return {
        "valid": not error,
        "error": error or None,
        "joint_count": len(names),
        "names": names,
        "positions_rad": positions or [],
        "velocities_rad_s": velocities or [],
        "efforts": efforts or [],
        "frame_id": frame_id(message),
    }


def summarize_image(message: object) -> dict[str, object]:
    try:
        width = int(message.width)
        height = int(message.height)
        step = int(message.step)
        encoding = str(message.encoding)
        data_bytes = len(message.data)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return {"valid": False, "error": f"invalid image metadata: {error}"}
    valid = bool(
        width > 0
        and height > 0
        and step > 0
        and encoding
        and data_bytes >= height * step
    )
    return {
        "valid": valid,
        "error": None if valid else "image dimensions, step, encoding, or payload are invalid",
        "frame_id": frame_id(message),
        "width": width,
        "height": height,
        "encoding": encoding,
        "step_bytes": step,
        "data_bytes": data_bytes,
    }


def encode_color_image_jpeg(message: object) -> tuple[bytes, dict[str, object]]:
    """Convert one bounded raw ROS color image into a dashboard JPEG."""

    import cv2
    import numpy as np

    try:
        source_width = int(message.width)
        source_height = int(message.height)
        step = int(message.step)
        encoding = str(message.encoding).lower()
        data = memoryview(message.data)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid raw color image: {error}") from error
    channels_by_encoding = {
        "mono8": 1,
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported raw color encoding: {encoding!r}")
    if (
        source_width <= 0
        or source_height <= 0
        or source_width > 8192
        or source_height > 8192
        or step < source_width * channels
        or step > source_width * channels + 65536
        or len(data) < source_height * step
    ):
        raise ValueError("raw color dimensions, step, or payload are invalid")
    rows = np.frombuffer(data, dtype=np.uint8, count=source_height * step).reshape(
        source_height,
        step,
    )
    pixels = rows[:, : source_width * channels].reshape(
        source_height,
        source_width,
        channels,
    )
    if encoding == "rgb8":
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    elif encoding == "bgra8":
        bgr = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    elif encoding == "mono8":
        bgr = cv2.cvtColor(pixels[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        bgr = np.ascontiguousarray(pixels)
    scale = min(
        1.0,
        MAX_CAMERA_WIDTH / source_width,
        MAX_CAMERA_HEIGHT / source_height,
    )
    width = max(1, min(MAX_CAMERA_WIDTH, round(source_width * scale)))
    height = max(1, min(MAX_CAMERA_HEIGHT, round(source_height * scale)))
    if (width, height) != (source_width, source_height):
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    encoded: bytes | None = None
    for quality in (CAMERA_JPEG_QUALITY, 70, 55, 40):
        ok, buffer = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if ok and len(buffer) <= MAX_CAMERA_JPEG_BYTES:
            encoded = buffer.tobytes()
            break
    if encoded is None:
        raise ValueError("encoded camera JPEG exceeds the 512 KiB limit")
    metadata: dict[str, object] = {
        "schema": "z_manip.camera_frame.v1",
        "source_timestamp_ns": stamp_ns(message),
        "source_frame": frame_id(message),
        "received_unix_ns": time.time_ns(),
        "source_width": source_width,
        "source_height": source_height,
        "source_encoding": encoding,
        "width": width,
        "height": height,
        "jpeg_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return encoded, metadata


def encode_depth_image_colormap(
    message: object,
    *,
    near_m: float = DEPTH_COLORMAP_NEAR_M,
    far_m: float = DEPTH_COLORMAP_FAR_M,
) -> tuple[bytes, dict[str, object]]:
    """Colorize one bounded raw ROS depth image into a turbo JPEG for the tile.

    Depth is normalized over a fixed [near_m, far_m] window and mapped through
    ``cv2.COLORMAP_TURBO``.  Zero and non-finite pixels (no return) are rendered
    black so the operator can distinguish missing depth from near-range depth.
    """

    import cv2
    import numpy as np

    try:
        source_width = int(message.width)
        source_height = int(message.height)
        step = int(message.step)
        encoding = str(message.encoding).lower()
        data = memoryview(message.data)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid raw depth image: {error}") from error
    dtype_by_encoding = {
        "16uc1": (np.uint16, 2, "mm"),
        "mono16": (np.uint16, 2, "mm"),
        "32fc1": (np.float32, 4, "m"),
    }
    spec = dtype_by_encoding.get(encoding)
    if spec is None:
        raise ValueError(f"unsupported raw depth encoding: {encoding!r}")
    dtype, bytes_per_pixel, units = spec
    if (
        source_width <= 0
        or source_height <= 0
        or source_width > 8192
        or source_height > 8192
        or step < source_width * bytes_per_pixel
        or step > source_width * bytes_per_pixel + 65536
        or len(data) < source_height * step
    ):
        raise ValueError("raw depth dimensions, step, or payload are invalid")
    rows = np.frombuffer(
        data, dtype=np.uint8, count=source_height * step
    ).reshape(source_height, step)
    pixels = rows[:, : source_width * bytes_per_pixel].reshape(
        source_height, source_width, bytes_per_pixel
    )
    depth = (
        np.ascontiguousarray(pixels)
        .view(dtype)
        .reshape(source_height, source_width)
    )
    if units == "mm":
        metres = depth.astype(np.float32) / 1000.0
    else:
        metres = depth.astype(np.float32)
    valid = np.isfinite(metres) & (metres > 0.0)
    span = max(1e-6, float(far_m) - float(near_m))
    normalized = np.clip((metres - float(near_m)) / span, 0.0, 1.0)
    scaled = np.zeros((source_height, source_width), dtype=np.uint8)
    scaled[valid] = (normalized[valid] * 255.0).astype(np.uint8)
    colorized = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colorized[~valid] = (0, 0, 0)
    scale = min(
        1.0,
        MAX_CAMERA_WIDTH / source_width,
        MAX_CAMERA_HEIGHT / source_height,
    )
    width = max(1, min(MAX_CAMERA_WIDTH, round(source_width * scale)))
    height = max(1, min(MAX_CAMERA_HEIGHT, round(source_height * scale)))
    if (width, height) != (source_width, source_height):
        # Nearest keeps invalid (black) regions crisp instead of bleeding colour.
        colorized = cv2.resize(
            colorized, (width, height), interpolation=cv2.INTER_NEAREST
        )
    encoded: bytes | None = None
    for quality in (CAMERA_JPEG_QUALITY, 70, 55, 40):
        ok, buffer = cv2.imencode(
            ".jpg",
            colorized,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if ok and len(buffer) <= MAX_CAMERA_JPEG_BYTES:
            encoded = buffer.tobytes()
            break
    if encoded is None:
        raise ValueError("encoded depth JPEG exceeds the 512 KiB limit")
    metadata: dict[str, object] = {
        "schema": "z_manip.depth_frame.v1",
        "source_timestamp_ns": stamp_ns(message),
        "source_frame": frame_id(message),
        "received_unix_ns": time.time_ns(),
        "source_width": source_width,
        "source_height": source_height,
        "source_encoding": encoding,
        "width": width,
        "height": height,
        "colormap": "turbo",
        "near_m": float(near_m),
        "far_m": float(far_m),
        "valid_fraction": round(float(valid.mean()), 4),
        "jpeg_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return encoded, metadata


def summarize_camera_info(message: object) -> dict[str, object]:
    try:
        width = int(message.width)
        height = int(message.height)
        matrix = _finite_list(message.k)
        distortion = _finite_list(message.d)
        model = str(message.distortion_model)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return {"valid": False, "error": f"invalid camera metadata: {error}"}
    valid = bool(
        width > 0
        and height > 0
        and matrix is not None
        and len(matrix) == 9
        and distortion is not None
    )
    return {
        "valid": valid,
        "error": None if valid else "camera dimensions or calibration arrays are invalid",
        "frame_id": frame_id(message),
        "width": width,
        "height": height,
        "distortion_model": model,
        "k": matrix or [],
        "distortion_coefficient_count": 0 if distortion is None else len(distortion),
    }


def summarize_point_cloud(message: object) -> dict[str, object]:
    try:
        width = int(message.width)
        height = int(message.height)
        point_step = int(message.point_step)
        row_step = int(message.row_step)
        data_bytes = len(message.data)
        fields = [str(value.name) for value in message.fields]
        dense = bool(message.is_dense)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        return {"valid": False, "error": f"invalid point-cloud metadata: {error}"}
    valid = bool(
        width >= 0
        and height >= 0
        and point_step > 0
        and row_step >= width * point_step
        and data_bytes >= height * row_step
    )
    result: dict[str, object] = {
        "valid": valid,
        "error": None if valid else "point-cloud dimensions or payload are invalid",
        "frame_id": frame_id(message),
        "width": width,
        "height": height,
        "point_count": width * height,
        "point_step_bytes": point_step,
        "row_step_bytes": row_step,
        "data_bytes": data_bytes,
        "fields": fields,
        "is_dense": dense,
    }
    if valid:
        result["points_xyz_m"] = sample_point_cloud_xyz(
            message,
            maximum=MAX_CLOUD_SAMPLE_POINTS,
        )
    return result


def sample_point_cloud_xyz(message: object, *, maximum: int) -> list[list[float]]:
    """Read a bounded XYZ sample from PointCloud2 without sensor_msgs_py."""

    if maximum <= 0:
        return []
    try:
        width = int(message.width)
        height = int(message.height)
        point_step = int(message.point_step)
        row_step = int(message.row_step)
        data = memoryview(message.data)
        big_endian = bool(message.is_bigendian)
        fields = {
            str(field.name): (int(field.offset), int(field.datatype), int(field.count))
            for field in message.fields
        }
    except (AttributeError, TypeError, ValueError, OverflowError):
        return []
    if not {"x", "y", "z"} <= set(fields) or width <= 0 or height <= 0:
        return []
    formats = {7: "f", 8: "d"}  # PointField.FLOAT32 / FLOAT64
    endian = ">" if big_endian else "<"
    unpackers: list[tuple[int, struct.Struct]] = []
    for name in ("x", "y", "z"):
        offset, datatype, count = fields[name]
        code = formats.get(datatype)
        if code is None or count != 1 or offset < 0:
            return []
        unpackers.append((offset, struct.Struct(endian + code)))
    total = width * height
    stride = max(1, math.ceil(total / maximum))
    output: list[list[float]] = []
    for flat_index in range(0, total, stride):
        row, column = divmod(flat_index, width)
        base = row * row_step + column * point_step
        try:
            point = [float(unpacker.unpack_from(data, base + offset)[0]) for offset, unpacker in unpackers]
        except (struct.error, ValueError, TypeError):
            continue
        if all(math.isfinite(value) for value in point):
            output.append(point)
        if len(output) >= maximum:
            break
    return output


def summarize_tf(message: object) -> dict[str, object]:
    transforms = list(getattr(message, "transforms", ()))
    pairs: list[dict[str, str]] = []
    stamps: list[int] = []
    for transform in transforms[:64]:
        parent = frame_id(transform)
        child = str(getattr(transform, "child_frame_id", ""))
        pairs.append({"parent": parent, "child": child})
        value = stamp_ns(transform)
        if value is not None:
            stamps.append(value)
    valid = bool(transforms and all(item["parent"] and item["child"] for item in pairs))
    return {
        "valid": valid,
        "error": None if valid else "TF message has no valid frame pair",
        "transform_count": len(transforms),
        "included_frame_pair_count": len(pairs),
        "frame_pairs": pairs,
        "latest_transform_stamp_ns": max(stamps, default=None),
    }


def summarize_depth_filter(message: object) -> dict[str, object]:
    """Extract bounded upstream D435 filter telemetry from a frame manifest."""

    payload = getattr(message, "data", None)
    if not isinstance(payload, str) or len(payload) > 4096:
        return {"valid": False, "error": "frame manifest is not a bounded string"}
    try:
        manifest = json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as error:
        return {"valid": False, "error": f"invalid frame manifest: {error}"}
    report = manifest.get("depth_filter") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "z_manip.tracker_frame.v1"
        or not isinstance(report, dict)
        or report.get("method") != "motion_adaptive_temporal_median"
    ):
        return {"valid": False, "error": "frame manifest has no supported depth filter"}
    allowed = {
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
    }
    if set(report) - allowed:
        return {"valid": False, "error": "depth-filter telemetry has unknown fields"}
    integer_fields = (
        "frame_count",
        "window_size",
        "minimum_observations",
        "dynamic_pixels",
        "stable_pixels",
        "rejected_low_support_pixels",
        "rejected_unstable_pixels",
    )
    if any(
        isinstance(report.get(key), bool) or not isinstance(report.get(key), int)
        for key in integer_fields
    ):
        return {"valid": False, "error": "depth-filter counts must be integers"}
    try:
        normalized = {
            "method": str(report["method"]),
            "frame_count": int(report["frame_count"]),
            "window_size": int(report["window_size"]),
            "minimum_observations": int(report["minimum_observations"]),
            "mode": str(report["mode"]),
            "reset_reason": report.get("reset_reason"),
            "motion_threshold_mm": float(report["motion_threshold_mm"]),
            "global_changed_fraction": float(report["global_changed_fraction"]),
            "dynamic_pixels": int(report["dynamic_pixels"]),
            "stable_pixels": int(report["stable_pixels"]),
            "rejected_low_support_pixels": int(report["rejected_low_support_pixels"]),
            "rejected_unstable_pixels": int(report["rejected_unstable_pixels"]),
            "mad_p95_mm": float(report["mad_p95_mm"]),
            "applied_to": [str(value) for value in report["applied_to"]],
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {"valid": False, "error": f"invalid depth-filter telemetry: {error}"}
    numeric = (
        normalized["motion_threshold_mm"],
        normalized["global_changed_fraction"],
        normalized["mad_p95_mm"],
    )
    counts = (
        normalized["frame_count"],
        normalized["window_size"],
        normalized["minimum_observations"],
        normalized["dynamic_pixels"],
        normalized["stable_pixels"],
        normalized["rejected_low_support_pixels"],
        normalized["rejected_unstable_pixels"],
    )
    reset_reason = normalized["reset_reason"]
    frame_count = normalized["frame_count"]
    window_size = normalized["window_size"]
    minimum = normalized["minimum_observations"]
    if (
        not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in numeric)
        or any(int(value) < 0 for value in counts)
        or not 3 <= int(window_size) <= 64
        or not 1 <= int(frame_count) <= int(window_size)
        or not 1 <= int(minimum) <= int(frame_count)
        or float(normalized["global_changed_fraction"]) > 1.0
        or float(normalized["motion_threshold_mm"]) <= 0.0
        or any(int(value) > 10_000_000 for value in counts[3:])
        or normalized["mode"] not in {
            "warmup",
            "static_temporal",
            "local_motion",
            "camera_motion_reset",
        }
        or reset_reason not in {
            None,
            "shape_changed",
            "stamp_not_increasing",
            "input_gap",
        }
        or normalized["applied_to"] != ["target_pointcloud", "scene_pointcloud"]
    ):
        return {"valid": False, "error": "depth-filter telemetry violates bounds"}
    return {"valid": True, "error": None, "depth_filter": normalized}


def summarize_bool(message: object) -> dict[str, object]:
    """Summarize a headerless ROS Bool without inventing a source stamp."""

    value = getattr(message, "data", None)
    if not isinstance(value, bool):
        return {"valid": False, "error": "Bool payload is not boolean"}
    return {"valid": True, "error": None, "value": value}


def summarize_tracker_failure(message: object) -> dict[str, object]:
    """Expose bounded, seed-correlated EdgeTAM terminal-failure evidence."""

    payload = getattr(message, "data", None)
    if not isinstance(payload, str) or len(payload) > 4096:
        return {"valid": False, "error": "tracker failure is not a bounded string"}
    try:
        report = json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as error:
        return {"valid": False, "error": f"invalid tracker failure: {error}"}
    if (
        not isinstance(report, dict)
        or report.get("schema") != "z_manip.tracker_failure.v1"
    ):
        return {"valid": False, "error": "unsupported tracker failure schema"}
    seed_id = report.get("seed_id")
    seed_stamp_ns = report.get("seed_stamp_ns")
    reason_code = report.get("reason_code")
    reason = report.get("reason")
    counts = {
        key: report.get(key)
        for key in (
            "replay_candidates",
            "replay_selected",
            "replay_span_ns",
            "acquisition_live_updates",
        )
    }
    if (
        not isinstance(seed_id, str)
        or not seed_id
        or len(seed_id) > 256
        or isinstance(seed_stamp_ns, bool)
        or not isinstance(seed_stamp_ns, int)
        or not 0 <= seed_stamp_ns <= (1 << 63) - 1
        or not isinstance(reason_code, str)
        or not isinstance(reason, str)
        or len(reason_code) > 128
        or len(reason) > 256
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 86_400_000_000_000
            for value in counts.values()
        )
    ):
        return {"valid": False, "error": "tracker failure fields violate bounds"}
    return {
        "valid": True,
        "error": None,
        "seed_id": seed_id,
        "seed_stamp_ns": seed_stamp_ns,
        "reason_code": reason_code,
        "reason": reason,
        **counts,
    }


@dataclass
class TopicState:
    topic: str
    message_type: str
    freshness_limit_s: float
    publisher_count: int = 0
    message_count: int = 0
    source_stamp_ns: int | None = None
    received_unix_ns: int | None = None
    received_monotonic_ns: int | None = None
    payload: dict[str, object] = field(default_factory=dict)
    subscription_error: str | None = None

    def observe(
        self,
        payload: dict[str, object],
        source_stamp_ns: int | None,
        *,
        received_unix_ns: int,
        received_monotonic_ns: int,
    ) -> None:
        self.message_count += 1
        self.source_stamp_ns = source_stamp_ns
        self.received_unix_ns = received_unix_ns
        self.received_monotonic_ns = received_monotonic_ns
        self.payload = dict(payload)

    def snapshot(self, now_monotonic_ns: int) -> dict[str, object]:
        age_s = None
        if self.received_monotonic_ns is not None:
            age_s = max(0.0, (now_monotonic_ns - self.received_monotonic_ns) * 1e-9)
        fresh = bool(
            age_s is not None
            and age_s <= self.freshness_limit_s
            and self.payload.get("valid") is True
        )
        if self.subscription_error:
            availability = "subscription_error"
        elif self.publisher_count <= 0:
            availability = "no_publishers"
        elif not self.message_count:
            availability = "waiting_for_message"
        elif not fresh:
            availability = "stale_or_invalid"
        else:
            availability = "fresh"
        result: dict[str, object] = {
            "topic": self.topic,
            "message_type": self.message_type,
            "publisher_count": self.publisher_count,
            "message_count": self.message_count,
            "received": self.message_count > 0,
            "source_stamp_ns": self.source_stamp_ns,
            "received_unix_ns": self.received_unix_ns,
            "age_s": age_s,
            "freshness_limit_s": self.freshness_limit_s,
            "fresh": fresh,
            "availability": availability,
            "subscription_error": self.subscription_error,
        }
        result.update(self.payload)
        return result


class RuntimeObserverState:
    """Pure-Python state and schema builder used by ROS and unit tests."""

    def __init__(
        self,
        topics: dict[str, tuple[str, str, float]],
        *,
        ros_domain_id: int,
        started_unix_ns: int | None = None,
        initial_sequence: int = 0,
    ) -> None:
        if (
            isinstance(initial_sequence, bool)
            or not isinstance(initial_sequence, int)
            or initial_sequence < 0
        ):
            raise ValueError("initial sequence must be a non-negative integer")
        self.sequence = initial_sequence
        self.ros_domain_id = int(ros_domain_id)
        self.started_unix_ns = time.time_ns() if started_unix_ns is None else started_unix_ns
        self.topics = {
            key: TopicState(topic, message_type, freshness)
            for key, (topic, message_type, freshness) in topics.items()
        }

    def observe(
        self,
        key: str,
        payload: dict[str, object],
        source_stamp_ns: int | None,
        *,
        received_unix_ns: int | None = None,
        received_monotonic_ns: int | None = None,
    ) -> None:
        self.topics[key].observe(
            payload,
            source_stamp_ns,
            received_unix_ns=time.time_ns() if received_unix_ns is None else received_unix_ns,
            received_monotonic_ns=(
                time.monotonic_ns()
                if received_monotonic_ns is None
                else received_monotonic_ns
            ),
        )

    def snapshot(
        self,
        *,
        publisher_counts: dict[str, int] | None = None,
        generated_unix_ns: int | None = None,
        now_monotonic_ns: int | None = None,
    ) -> dict[str, object]:
        if publisher_counts is not None:
            for key, count in publisher_counts.items():
                if key in self.topics:
                    self.topics[key].publisher_count = max(0, int(count))
        self.sequence += 1
        now_mono = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        topic_snapshots = {
            key: value.snapshot(now_mono)
            for key, value in self.topics.items()
        }
        fresh = {key for key, value in topic_snapshots.items() if value["fresh"]}
        observed = {key for key, value in topic_snapshots.items() if value["received"]}
        joint = topic_snapshots["joint_state"]
        joint_available = bool(
            joint["fresh"]
            and joint["publisher_count"] > 0
            and joint.get("valid") is True
        )
        tracker_state = topic_snapshots.get("tracker_state", {})
        tracker_target = topic_snapshots.get("tracker_target", {})
        tracker_failure = topic_snapshots.get("tracker_failure", {})
        failure_is_latest = bool(
            tracker_failure.get("fresh") is True
            and int(tracker_failure.get("received_unix_ns") or 0)
            >= int(tracker_state.get("received_unix_ns") or 0)
        )
        if failure_is_latest:
            tracker_phase = "failed"
        elif tracker_state.get("fresh") is True and tracker_state.get("value") is True:
            tracker_phase = (
                "tracking" if tracker_target.get("fresh") is True else "target_stale"
            )
        elif tracker_state.get("received") is True:
            tracker_phase = "idle_or_lost"
        else:
            tracker_phase = "unobserved"
        return {
            "schema": "z_manip.runtime_observer.v1",
            "sequence": self.sequence,
            "generated_unix_ns": (
                time.time_ns() if generated_unix_ns is None else generated_unix_ns
            ),
            "observer": {
                "read_only": True,
                "subscribe_only": True,
                "planning_only": True,
                "motion_commands_published": 0,
                "publishers_created": 0,
                "service_clients_created": 0,
                "action_clients_created": 0,
                "can_opened": False,
                "ros_domain_id": self.ros_domain_id,
                "started_unix_ns": self.started_unix_ns,
            },
            "summary": {
                "joint_state_available": joint_available,
                "camera_rgbd_fresh": {"color", "depth", "camera_info"} <= fresh,
                "point_cloud_fresh": bool(
                    {"scene_cloud", "target_cloud"} & fresh
                ),
                "depth_filter_fresh": "depth_filter" in fresh,
                "tracker_phase": tracker_phase,
                "tracker_target_fresh": "tracker_target" in fresh,
                "tf_available": bool({"tf", "tf_static"} & fresh),
                "observed_topic_count": len(observed),
                "fresh_topic_count": len(fresh),
                "configured_topic_count": len(topic_snapshots),
            },
            "joint_state": {
                "available": joint_available,
                "topic": joint["topic"],
                "publisher_count": joint["publisher_count"],
                "fresh": joint["fresh"],
                "source_stamp_ns": joint["source_stamp_ns"],
                "received_unix_ns": joint["received_unix_ns"],
                "names": joint.get("names", []),
                "positions_rad": joint.get("positions_rad", []),
                "velocities_rad_s": joint.get("velocities_rad_s", []),
                "efforts": joint.get("efforts", []),
                "reason": None if joint_available else joint["availability"],
            },
            "tracker": {
                "phase": tracker_phase,
                "tracking": (
                    tracker_state.get("value")
                    if tracker_state.get("fresh") is True
                    else None
                ),
                "target_fresh": tracker_target.get("fresh") is True,
                "target_source_stamp_ns": tracker_target.get("source_stamp_ns"),
                "failure": (
                    {
                        "seed_id": tracker_failure.get("seed_id"),
                        "seed_stamp_ns": tracker_failure.get("seed_stamp_ns"),
                        "reason_code": tracker_failure.get("reason_code"),
                        "reason": tracker_failure.get("reason"),
                    }
                    if failure_is_latest
                    else None
                ),
            },
            "topics": topic_snapshots,
        }


def build_runtime_state(
    diagnostic: dict[str, object],
    *,
    chain: object | None = None,
    calibration: dict[str, object] | None = None,
    platform_from_arm_base: object | None = None,
    platform_frame: str = "base_link",
) -> dict[str, object]:
    """Convert diagnostics into the bounded dashboard runtime contract."""

    import numpy as np

    generated_ns = int(diagnostic["generated_unix_ns"])
    joint = diagnostic.get("joint_state")
    joint = joint if isinstance(joint, dict) else {}
    positions = joint.get("positions_rad")
    positions = positions if isinstance(positions, list) else []
    available = bool(joint.get("available") is True and len(positions) == 6)
    summary = diagnostic.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    observer = diagnostic.get("observer")
    observer = observer if isinstance(observer, dict) else {}
    topics = diagnostic.get("topics")
    topics = topics if isinstance(topics, dict) else {}
    filter_topic = topics.get("depth_filter")
    filter_topic = filter_topic if isinstance(filter_topic, dict) else {}
    filter_telemetry = (
        filter_topic.get("depth_filter")
        if filter_topic.get("fresh") is True
        and isinstance(filter_topic.get("depth_filter"), dict)
        else None
    )
    filter_status = {
        "available": isinstance(filter_telemetry, dict),
        "fresh": filter_topic.get("fresh") is True,
        "report": filter_telemetry,
    }
    document: dict[str, object] = {
        "schema": SCHEMA,
        "sequence": int(diagnostic["sequence"]),
        "source_timestamp_ns": generated_ns,
        "joint_state_available": available,
        "joint_positions_rad": [float(value) for value in positions] if available else [],
        "telemetry": {
            "ros_domain_id": int(observer.get("ros_domain_id", 20)),
            "read_only": observer.get("read_only") is True,
            "motion_commands_published": int(observer.get("motion_commands_published", 0)),
            "can_opened": observer.get("can_opened") is True,
            "camera_rgbd_fresh": summary.get("camera_rgbd_fresh") is True,
            "point_cloud_fresh": summary.get("point_cloud_fresh") is True,
            "joint_state_available": available,
            "joint_topic": str(joint.get("topic", "/piper/state")),
            "joint_publisher_count": int(joint.get("publisher_count", 0)),
            "fresh_topic_count": int(summary.get("fresh_topic_count", 0)),
            "configured_topic_count": int(summary.get("configured_topic_count", 0)),
            "depth_filter": filter_status,
            "tracker": diagnostic.get("tracker", {}),
        },
    }
    if not available or chain is None:
        return document

    values = np.asarray(positions, dtype=float)
    transforms = chain.link_transforms(values)
    document["robot_links"] = [
        {"name": name, "transform": np.asarray(matrix, dtype=float).tolist()}
        for name, matrix in transforms.items()
    ]
    if not isinstance(calibration, dict):
        return document
    if (
        calibration.get("calibrated") is not True
        or calibration.get("mount_type") != "eye_in_hand"
        or calibration.get("tip_link") != chain.tip_link
    ):
        return document
    tip_from_camera = np.asarray(calibration.get("tip_from_camera"), dtype=float)
    if tip_from_camera.shape != (4, 4) or not np.all(np.isfinite(tip_from_camera)):
        return document
    base_from_camera = np.asarray(chain.forward(values), dtype=float) @ tip_from_camera
    camera_frame = str(calibration.get("camera_frame", ""))
    platform_transform = None
    if platform_from_arm_base is not None:
        candidate = np.asarray(platform_from_arm_base, dtype=float)
        if candidate.shape == (4, 4) and np.all(np.isfinite(candidate)):
            platform_transform = candidate @ base_from_camera
    # This is the transform source consumed by the mobile depth servo when
    # the ROS graph does not publish the PiPER/Go2W model frames.  It is
    # reconstructed from fresh passive joints, the deployed URDF, the
    # measured eye-in-hand calibration, and the fixed arm mount.  Keeping the
    # provenance in the artifact prevents a camera-z compatibility fallback
    # from being mistaken for base-frame geometry.
    if platform_transform is not None and camera_frame:
        document["kinematic_transforms"] = {
            "schema": "z_manip.kinematic_transforms.v1",
            "verified": True,
            "source": "passive_joints+deployed_urdf+measured_hand_eye",
            "source_timestamp_ns": generated_ns,
            "joint_source_timestamp_ns": joint.get("source_stamp_ns"),
            "camera_frame": camera_frame,
            "arm_base_frame": str(chain.base_link),
            "platform_base_frame": str(platform_frame),
            "arm_base_from_camera": base_from_camera.tolist(),
            "platform_base_from_camera": platform_transform.tolist(),
            "calibration_id": calibration.get("calibration_id"),
            "calibration_synthetic": calibration.get("synthetic"),
        }
    clouds: dict[str, object] = {}
    for output_name, topic_name in (("scene", "scene_cloud"), ("target", "target_cloud")):
        topic = topics.get(topic_name)
        if not isinstance(topic, dict) or topic.get("fresh") is not True:
            continue
        if str(topic.get("frame_id", "")) != camera_frame:
            continue
        points = np.asarray(topic.get("points_xyz_m"), dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            continue
        homogeneous = np.column_stack((points, np.ones(len(points), dtype=float)))
        transformed = (base_from_camera @ homogeneous.T).T[:, :3]
        clouds[output_name] = {
            "frame": chain.base_link,
            "points_xyz_m": transformed.tolist(),
        }
    if clouds:
        document["point_clouds"] = clouds
    return document


def atomic_write_json(path: Path, document: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class CameraFrameWriter:
    """Commit the JPEG first and its matching metadata manifest second."""

    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path.expanduser().resolve()
        self.metadata_path = self.image_path.with_suffix(".json")

    def write(self, message: object) -> None:
        jpeg, metadata = encode_color_image_jpeg(message)
        atomic_write_bytes(self.image_path, jpeg)
        atomic_write_json(self.metadata_path, metadata)


class DepthFrameWriter:
    """Colorize and commit the live depth tile, coalescing bursts to ~10 Hz.

    The depth topic runs at the camera rate (30 Hz today) but colorization is
    heavier than the RGB path, so writes are rate-limited to
    ``min_interval_s``.  Like ``CameraFrameWriter`` the JPEG is committed before
    its metadata manifest.
    """

    def __init__(
        self,
        image_path: Path,
        *,
        min_interval_s: float = DEPTH_WRITE_MIN_INTERVAL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.image_path = image_path.expanduser().resolve()
        self.metadata_path = self.image_path.with_suffix(".json")
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._monotonic = monotonic
        self._last_write_monotonic: float | None = None

    def _due(self, now: float) -> bool:
        return (
            self._last_write_monotonic is None
            or now - self._last_write_monotonic >= self.min_interval_s
        )

    def write(self, message: object) -> bool:
        """Colorize and persist one frame, or skip if the throttle is not due.

        Returns ``True`` when a frame was written so callers can reflect it in
        telemetry.
        """

        now = self._monotonic()
        if not self._due(now):
            return False
        jpeg, metadata = encode_depth_image_colormap(message)
        atomic_write_bytes(self.image_path, jpeg)
        atomic_write_json(self.metadata_path, metadata)
        self._last_write_monotonic = now
        return True


def _callback(
    state: RuntimeObserverState,
    key: str,
    summarizer: Callable[[object], dict[str, object]],
) -> Callable[[object], None]:
    def observe(message: object) -> None:
        source = stamp_ns(message)
        try:
            payload = summarizer(message)
        except Exception as error:  # Keep malformed telemetry observable.
            payload = {
                "valid": False,
                "error": f"{type(error).__name__}: {error}",
            }
        if key in ("tf", "tf_static"):
            source = payload.get("latest_transform_stamp_ns")
            if not isinstance(source, int):
                source = None
        state.observe(key, payload, source)

    return observe


def _color_callback(
    state: RuntimeObserverState,
    writer: CameraFrameWriter | None,
) -> Callable[[object], None]:
    def observe(message: object) -> None:
        payload = summarize_image(message)
        if writer is not None and payload.get("valid") is True:
            try:
                writer.write(message)
                payload["camera_jpeg_available"] = True
                payload["camera_jpeg_error"] = None
            except Exception as error:  # Preserve the last known-good camera JPEG.
                payload["camera_jpeg_available"] = False
                payload["camera_jpeg_error"] = f"{type(error).__name__}: {error}"[:512]
        state.observe("color", payload, stamp_ns(message))

    return observe


def _depth_callback(
    state: RuntimeObserverState,
    writer: DepthFrameWriter | None,
) -> Callable[[object], None]:
    def observe(message: object) -> None:
        payload = summarize_image(message)
        if writer is not None and payload.get("valid") is True:
            try:
                wrote = writer.write(message)
                if wrote:
                    payload["depth_jpeg_available"] = True
                    payload["depth_jpeg_error"] = None
            except Exception as error:  # Preserve the last known-good depth JPEG.
                payload["depth_jpeg_available"] = False
                payload["depth_jpeg_error"] = f"{type(error).__name__}: {error}"[:512]
        state.observe("depth", payload, stamp_ns(message))

    return observe


def run_ros_observer(args: argparse.Namespace) -> int:
    """Join ROS Domain 20 with subscriptions only and write snapshots."""

    import rclpy
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )
    from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
    from std_msgs.msg import Bool, String
    from tf2_msgs.msg import TFMessage

    chain = None
    calibration = None
    platform_from_arm_base = None
    if args.urdf is not None:
        from z_manip.kinematics import KinematicChain, fixed_transform_from_urdf

        chain = KinematicChain.from_urdf(
            args.urdf.expanduser().resolve(),
            args.base_link,
            args.tip_link,
        )
        platform_from_arm_base = fixed_transform_from_urdf(
            args.urdf.expanduser().resolve(),
            args.platform_urdf_link,
            args.base_link,
        )
    if args.calibration is not None:
        calibration = json.loads(
            args.calibration.expanduser().resolve().read_text(encoding="utf-8")
        )
        if not isinstance(calibration, dict):
            raise ValueError("camera calibration must contain a JSON object")

    topics = {
        "joint_state": (args.joint_topic, "sensor_msgs/msg/JointState", args.joint_age_s),
        "color": (args.color_topic, "sensor_msgs/msg/Image", args.camera_age_s),
        "depth": (args.depth_topic, "sensor_msgs/msg/Image", args.camera_age_s),
        "camera_info": (
            args.camera_info_topic,
            "sensor_msgs/msg/CameraInfo",
            args.camera_age_s,
        ),
        "scene_cloud": (
            args.scene_cloud_topic,
            "sensor_msgs/msg/PointCloud2",
            args.cloud_age_s,
        ),
        "target_cloud": (
            args.target_cloud_topic,
            "sensor_msgs/msg/PointCloud2",
            args.cloud_age_s,
        ),
        "depth_filter": (
            args.depth_filter_manifest_topic,
            "std_msgs/msg/String",
            args.cloud_age_s,
        ),
        "tracker_target": (
            args.tracker_target_topic,
            "sensor_msgs/msg/PointCloud2",
            args.tracker_age_s,
        ),
        "tracker_state": (
            args.tracker_state_topic,
            "std_msgs/msg/Bool",
            args.tracker_age_s,
        ),
        "tracker_failure": (
            args.tracker_failure_topic,
            "std_msgs/msg/String",
            args.tracker_failure_age_s,
        ),
        "tf": (args.tf_topic, "tf2_msgs/msg/TFMessage", args.tf_age_s),
        "tf_static": (
            args.tf_static_topic,
            "tf2_msgs/msg/TFMessage",
            args.tf_static_age_s,
        ),
    }
    state = RuntimeObserverState(
        topics,
        ros_domain_id=args.ros_domain_id,
        initial_sequence=load_initial_sequence(args.output),
    )
    camera_writer = (
        None if args.camera_output is None else CameraFrameWriter(args.camera_output)
    )
    depth_writer = (
        None
        if args.depth_output is None
        else DepthFrameWriter(
            args.depth_output,
            min_interval_s=args.depth_write_min_interval_s,
        )
    )
    rclpy.init()
    node = rclpy.create_node(
        "z_manip_runtime_observer_read_only",
        enable_rosout=False,
        start_parameter_services=False,
    )
    tf_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    tf_static_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    subscriptions = [
        node.create_subscription(
            JointState,
            args.joint_topic,
            _callback(state, "joint_state", summarize_joint_state),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            Image,
            args.color_topic,
            _color_callback(state, camera_writer),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            Image,
            args.depth_topic,
            _depth_callback(state, depth_writer),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            _callback(state, "camera_info", summarize_camera_info),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            PointCloud2,
            args.scene_cloud_topic,
            _callback(state, "scene_cloud", summarize_point_cloud),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            PointCloud2,
            args.target_cloud_topic,
            _callback(state, "target_cloud", summarize_point_cloud),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            String,
            args.depth_filter_manifest_topic,
            _callback(state, "depth_filter", summarize_depth_filter),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            PointCloud2,
            args.tracker_target_topic,
            _callback(state, "tracker_target", summarize_point_cloud),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            Bool,
            args.tracker_state_topic,
            _callback(state, "tracker_state", summarize_bool),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            String,
            args.tracker_failure_topic,
            _callback(state, "tracker_failure", summarize_tracker_failure),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            TFMessage,
            args.tf_topic,
            _callback(state, "tf", summarize_tf),
            tf_qos,
        ),
        node.create_subscription(
            TFMessage,
            args.tf_static_topic,
            _callback(state, "tf_static", summarize_tf),
            tf_static_qos,
        ),
    ]
    del subscriptions  # rclpy's node retains subscription ownership.
    started = time.monotonic()
    next_write = started
    try:
        while rclpy.ok():
            now = time.monotonic()
            if args.duration_s > 0.0 and now - started >= args.duration_s:
                break
            rclpy.spin_once(node, timeout_sec=min(0.10, args.write_period_s))
            now = time.monotonic()
            if now < next_write:
                continue
            counts = {
                key: node.count_publishers(topic.topic)
                for key, topic in state.topics.items()
            }
            diagnostic = state.snapshot(publisher_counts=counts)
            atomic_write_json(
                args.output,
                build_runtime_state(
                    diagnostic,
                    chain=chain,
                    calibration=calibration,
                    platform_from_arm_base=platform_from_arm_base,
                    platform_frame=args.platform_frame,
                ),
            )
            next_write = now + args.write_period_s
    finally:
        counts = {
            key: node.count_publishers(topic.topic)
            for key, topic in state.topics.items()
        }
        diagnostic = state.snapshot(publisher_counts=counts)
        atomic_write_json(
            args.output,
            build_runtime_state(
                diagnostic,
                chain=chain,
                calibration=calibration,
                platform_from_arm_base=platform_from_arm_base,
                platform_frame=args.platform_frame,
            ),
        )
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _positive_finite(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"{name} must be positive and finite")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera-output",
        type=Path,
        help="fixed atomically replaced 640x480-or-smaller JPEG artifact path",
    )
    parser.add_argument(
        "--depth-output",
        type=Path,
        help=(
            "fixed atomically replaced colorized (turbo) depth JPEG artifact "
            "path, written at ~10 Hz next to --camera-output"
        ),
    )
    parser.add_argument(
        "--depth-write-min-interval-s",
        type=float,
        default=DEPTH_WRITE_MIN_INTERVAL_S,
        help="minimum seconds between colorized depth-tile writes (~10 Hz)",
    )
    parser.add_argument("--ros-domain-id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "20")))
    parser.add_argument("--joint-topic", default=DEFAULT_TOPICS["joint_state"][0])
    parser.add_argument("--color-topic", default=DEFAULT_TOPICS["color"][0])
    parser.add_argument("--depth-topic", default=DEFAULT_TOPICS["depth"][0])
    parser.add_argument("--camera-info-topic", default=DEFAULT_TOPICS["camera_info"][0])
    parser.add_argument("--scene-cloud-topic", default=DEFAULT_TOPICS["scene_cloud"][0])
    parser.add_argument("--target-cloud-topic", default=DEFAULT_TOPICS["target_cloud"][0])
    parser.add_argument(
        "--depth-filter-manifest-topic",
        default=DEFAULT_TOPICS["depth_filter"][0],
    )
    parser.add_argument(
        "--tracker-target-topic",
        default=DEFAULT_TOPICS["tracker_target"][0],
    )
    parser.add_argument(
        "--tracker-state-topic",
        default=DEFAULT_TOPICS["tracker_state"][0],
    )
    parser.add_argument(
        "--tracker-failure-topic",
        default=DEFAULT_TOPICS["tracker_failure"][0],
    )
    parser.add_argument("--tf-topic", default=DEFAULT_TOPICS["tf"][0])
    parser.add_argument("--tf-static-topic", default=DEFAULT_TOPICS["tf_static"][0])
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--base-link", default="piper_base_link")
    parser.add_argument("--tip-link", default="piper_gripper_base")
    parser.add_argument(
        "--platform-urdf-link",
        default="base",
        help="URDF body link that is the fixed parent of the arm mount",
    )
    parser.add_argument(
        "--platform-frame",
        default="base_link",
        help="runtime frame name represented by platform-urdf-link",
    )
    parser.add_argument("--write-period-s", type=float, default=0.10)
    parser.add_argument("--joint-age-s", type=float, default=0.50)
    parser.add_argument("--camera-age-s", type=float, default=1.00)
    parser.add_argument("--cloud-age-s", type=float, default=1.50)
    parser.add_argument("--tracker-age-s", type=float, default=0.75)
    parser.add_argument("--tracker-failure-age-s", type=float, default=5.00)
    parser.add_argument("--tf-age-s", type=float, default=2.00)
    parser.add_argument("--tf-static-age-s", type=float, default=3600.00)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="optional bounded validation duration; zero runs until stopped",
    )
    values = parser.parse_args()
    if not 0 <= values.ros_domain_id <= 232:
        parser.error("ROS domain ID must be between 0 and 232")
    for name in (
        "write_period_s",
        "joint_age_s",
        "camera_age_s",
        "cloud_age_s",
        "tracker_age_s",
        "tracker_failure_age_s",
        "tf_age_s",
        "tf_static_age_s",
    ):
        _positive_finite(parser, name, float(getattr(values, name)))
    if not math.isfinite(values.duration_s) or values.duration_s < 0.0:
        parser.error("duration_s must be finite and non-negative")
    if values.camera_output is not None:
        if values.camera_output.suffix.lower() not in {".jpg", ".jpeg"}:
            parser.error("camera_output must use a .jpg or .jpeg suffix")
        if values.camera_output.expanduser().resolve() == values.output.expanduser().resolve():
            parser.error("camera_output must be different from output")
    if not math.isfinite(values.depth_write_min_interval_s) or values.depth_write_min_interval_s < 0.0:
        parser.error("depth_write_min_interval_s must be finite and non-negative")
    if values.depth_output is not None:
        if values.depth_output.suffix.lower() not in {".jpg", ".jpeg"}:
            parser.error("depth_output must use a .jpg or .jpeg suffix")
        resolved_depth = values.depth_output.expanduser().resolve()
        if resolved_depth == values.output.expanduser().resolve():
            parser.error("depth_output must be different from output")
        if (
            values.camera_output is not None
            and resolved_depth == values.camera_output.expanduser().resolve()
        ):
            parser.error("depth_output must be different from camera_output")
    for name in (
        "joint_topic",
        "color_topic",
        "depth_topic",
        "camera_info_topic",
        "scene_cloud_topic",
        "target_cloud_topic",
        "depth_filter_manifest_topic",
        "tf_topic",
        "tf_static_topic",
    ):
        topic = str(getattr(values, name))
        if not topic.startswith("/") or any(character.isspace() for character in topic):
            parser.error(f"{name} must be an absolute ROS topic without whitespace")
    return values


def main() -> int:
    return run_ros_observer(_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
