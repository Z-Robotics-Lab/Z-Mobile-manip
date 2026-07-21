#!/usr/bin/env python3
"""Replay recorded target/joint samples through the real whole-body controller.

This program is deliberately offline.  It opens a rosbag2 reader and writes a
temporary measured-state JSON file, but creates no ROS node, publisher, SPORT
client, CAN socket, or PiPER SDK object.  It is suitable for running inside a
``--network none`` container while the robot is unattended.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from z_manip.control.whole_body_runtime import WholeBodyRuntimeController


REPORT_SCHEMA = "z_mobile_manip.whole_body_bag_replay.v1"
TOPICS = (
    "/track_3d/selected_target_3d",
    "/piper/state",
    "/go2w/posture_state",
)


def _message_types(reader: rosbag2_py.SequentialReader) -> dict[str, type]:
    return {
        item.name: get_message(item.type)
        for item in reader.get_all_topics_and_types()
        if item.name in TOPICS
    }


def _joint_vector(message: Any) -> tuple[float, ...] | None:
    names = list(message.name)
    positions = list(message.position)
    if len(positions) < 6:
        return None
    expected = [f"piper_joint{index}" for index in range(1, 7)]
    if names and all(name in names for name in expected):
        result = tuple(float(positions[names.index(name)]) for name in expected)
    else:
        result = tuple(float(value) for value in positions[:6])
    return result if all(math.isfinite(value) for value in result) else None


def _target_xyz(message: Any) -> tuple[float, float, float] | None:
    position = message.bbox.center.position
    result = (float(position.x), float(position.y), float(position.z))
    if not all(math.isfinite(value) for value in result) or result[2] <= 0.0:
        return None
    return result


def _normalized_posture(recorded: dict[str, Any] | None) -> dict[str, Any]:
    """Preserve measured attitude but encode the observed Euler capability."""
    recorded = dict(recorded or {})
    attitude = recorded.get("attitude")
    if not isinstance(attitude, dict):
        attitude = {}
    recorded.update({
        "schema": "z_manip.go2w_posture_status.v1",
        "phase": "unsupported",
        "detail": "offline replay: Euler(1007) returned RPC 3203",
        "stop_latched": False,
        "capabilities": {
            **(
                recorded.get("capabilities")
                if isinstance(recorded.get("capabilities"), dict)
                else {}
            ),
            "euler": False,
            "body_height": False,
        },
        "feedback": {"fresh": True, "source": "recorded/replay"},
        "attitude": {
            **attitude,
            "current_roll_rad": float(attitude.get("current_roll_rad", 0.0)),
            "current_pitch_rad": float(attitude.get("current_pitch_rad", 0.0)),
        },
    })
    return recorded


def _fresh_arm_status(sequence: int) -> dict[str, Any]:
    return {
        "schema": "z_manip.piper_reactive_view_status.v1",
        "owner": "piper_reactive_view_executor",
        "ready": True,
        "stop_latched": False,
        "fault": None,
        "accepted_seq": int(sequence),
        "updated_unix_ns": time.time_ns(),
        "offline_replay": True,
    }


def _write_runtime_state(path: Path, joints: tuple[float, ...]) -> None:
    document = {
        "schema": "z_manip.runtime_state.v1",
        "source_timestamp_ns": time.time_ns(),
        "joint_state_available": True,
        "joint_positions_rad": list(joints),
        "offline_replay": True,
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def replay(
    *,
    bag_path: Path,
    urdf_path: Path,
    calibration_path: Path,
    sample_stride: int,
    maximum_samples: int,
) -> dict[str, Any]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    message_types = _message_types(reader)
    missing = sorted(set(TOPICS) - set(message_types))
    if missing:
        raise RuntimeError(f"bag lacks replay topics: {missing}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPICS)))

    mobile = WholeBodyRuntimeController(
        urdf_path=urdf_path,
        calibration_path=calibration_path,
    )
    posture_only = WholeBodyRuntimeController(
        urdf_path=urdf_path,
        calibration_path=calibration_path,
    )
    latest_joints: tuple[float, ...] | None = None
    latest_posture: dict[str, Any] | None = None
    seen_targets = 0
    samples: list[dict[str, Any]] = []
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="z-mobile-offline-replay-") as temp:
        runtime_path = Path(temp) / "runtime_state.json"
        while reader.has_next() and len(samples) < maximum_samples:
            topic, serialized, timestamp_ns = reader.read_next()
            message = deserialize_message(serialized, message_types[topic])
            if topic == "/piper/state":
                latest_joints = _joint_vector(message)
                continue
            if topic == "/go2w/posture_state":
                try:
                    value = json.loads(message.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(value, dict):
                    latest_posture = value
                continue

            target = _target_xyz(message)
            if target is None or latest_joints is None:
                continue
            seen_targets += 1
            if seen_targets % sample_stride:
                continue

            _write_runtime_state(runtime_path, latest_joints)
            posture = _normalized_posture(latest_posture)
            arm_status = _fresh_arm_status(len(samples))
            try:
                moving = mobile.solve(
                    camera_target_xyz_m=target,
                    posture_status=posture,
                    arm_view_status=arm_status,
                    runtime_state_path=runtime_path,
                    mode="live",
                    freeze_base=False,
                )
                frozen = posture_only.solve(
                    camera_target_xyz_m=target,
                    posture_status=posture,
                    arm_view_status=arm_status,
                    runtime_state_path=runtime_path,
                    mode="live",
                    freeze_base=True,
                )
            except Exception as error:  # report every replayed geometry failure
                failures.append(f"{type(error).__name__}: {error}")
                continue

            samples.append({
                "bag_timestamp_ns": int(timestamp_ns),
                "target_camera_xyz_m": list(target),
                "mobile": {
                    "base_forward_mps": moving.base_forward_mps,
                    "base_yaw_rps": moving.base_yaw_rps,
                    "body_roll_target_rad": moving.body_roll_target_rad,
                    "body_pitch_target_rad": moving.body_pitch_target_rad,
                    "arm_joint_velocity_rps": list(moving.arm_joint_velocity_rps),
                    "executable": moving.executable,
                    "transport": moving.document.get("transport"),
                },
                "posture_only": {
                    "base_forward_mps": frozen.base_forward_mps,
                    "base_yaw_rps": frozen.base_yaw_rps,
                    "body_roll_target_rad": frozen.body_roll_target_rad,
                    "body_pitch_target_rad": frozen.body_pitch_target_rad,
                    "arm_joint_velocity_rps": list(frozen.arm_joint_velocity_rps),
                    "executable": frozen.executable,
                    "transport": frozen.document.get("transport"),
                },
            })

    body_disabled = [
        sample for sample in samples
        if sample["mobile"]["transport"].get("body_enabled") is False
        and sample["posture_only"]["transport"].get("body_enabled") is False
    ]
    posture_base_zero = [
        sample for sample in samples
        if abs(sample["posture_only"]["base_forward_mps"]) <= 1e-10
        and abs(sample["posture_only"]["base_yaw_rps"]) <= 1e-10
    ]
    arm_active = [
        sample for sample in samples
        if max(abs(value) for value in sample["posture_only"]["arm_joint_velocity_rps"]) > 1e-5
    ]
    finite = [
        sample for sample in samples
        if all(
            math.isfinite(value)
            for branch in (sample["mobile"], sample["posture_only"])
            for value in (
                branch["base_forward_mps"],
                branch["base_yaw_rps"],
                branch["body_roll_target_rad"],
                branch["body_pitch_target_rad"],
                *branch["arm_joint_velocity_rps"],
            )
        )
    ]
    invariants = {
        "samples_present": bool(samples),
        "all_outputs_finite": len(finite) == len(samples),
        "euler_transport_disabled_for_all_samples": len(body_disabled) == len(samples),
        "base_locked_during_posture_only_for_all_samples": len(posture_base_zero) == len(samples),
        "arm_reallocated_for_at_least_one_sample": bool(arm_active),
        "no_replay_exceptions": not failures,
    }
    return {
        "schema": REPORT_SCHEMA,
        "offline": True,
        "network_required": False,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "bag_path": str(bag_path.resolve()),
        "sample_stride": sample_stride,
        "targets_seen": seen_targets,
        "samples_evaluated": len(samples),
        "replay_exception_counts": dict(Counter(failures)),
        "invariants": invariants,
        "passed": all(invariants.values()),
        "summary": {
            "euler_disabled_samples": len(body_disabled),
            "posture_only_base_locked_samples": len(posture_base_zero),
            "posture_only_arm_active_samples": len(arm_active),
            "max_abs_arm_rate_rps": max((
                max(abs(value) for value in sample["posture_only"]["arm_joint_velocity_rps"])
                for sample in samples
            ), default=0.0),
            "mobile_base_active_samples": sum(
                abs(sample["mobile"]["base_forward_mps"]) > 1e-5
                or abs(sample["mobile"]["base_yaw_rps"]) > 1e-5
                for sample in samples
            ),
        },
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=80)
    args = parser.parse_args()
    if args.sample_stride < 1 or args.max_samples < 1:
        parser.error("sample stride and maximum samples must be positive")
    report = replay(
        bag_path=args.bag.expanduser().resolve(),
        urdf_path=args.urdf.expanduser().resolve(),
        calibration_path=args.calibration.expanduser().resolve(),
        sample_stride=args.sample_stride,
        maximum_samples=args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "samples_evaluated": report["samples_evaluated"],
        "summary": report["summary"],
        "invariants": report["invariants"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
