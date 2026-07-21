#!/usr/bin/env python3
"""Append one safety-qualified, offline PiPER hand-eye pose sample.

The inputs are immutable reports from the read-only ChArUco subscriber and the
receive-only SocketCAN probe.  This program opens no ROS or CAN transport.  It
uses URDF forward kinematics only after proving that both observations overlap,
the arm stayed still, feedback was complete, and host CAN transmission was zero.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from z_manip.kinematics.chain import KinematicChain


def rigid_transform(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(f"{label} is not a right-handed rigid transform")
    return matrix


def planning_limit_violations(
    joints: np.ndarray,
    chain: KinematicChain,
) -> list[dict[str, object]]:
    """Describe real feedback outside the URDF motion-planning envelope."""

    violations: list[dict[str, object]] = []
    for index, (position, lower, upper) in enumerate(
        zip(joints, chain.lower_limits, chain.upper_limits),
        start=1,
    ):
        if lower <= position <= upper:
            continue
        boundary = lower if position < lower else upper
        direction = "below" if position < lower else "above"
        excess = abs(float(position - boundary))
        violations.append({
            "joint_index": index,
            "joint_name": chain.joint_names[index - 1],
            "position_rad": float(position),
            "position_deg": math.degrees(float(position)),
            "lower_limit_rad": float(lower),
            "lower_limit_deg": math.degrees(float(lower)),
            "upper_limit_rad": float(upper),
            "upper_limit_deg": math.degrees(float(upper)),
            "direction": direction,
            "excess_rad": excess,
            "excess_deg": math.degrees(excess),
        })
    return violations


def assemble_sample(
    camera_report: dict[str, object],
    joint_report: dict[str, object],
    chain: KinematicChain,
    *,
    max_joint_range_rad: float = 0.002,
    max_snapshot_span_s: float = 0.050,
    max_clock_skew_s: float = 0.250,
    max_abs_joint_feedback_rad: float = math.radians(190.0),
) -> dict[str, object]:
    if camera_report.get("schema") != "z_manip.charuco_camera_sample.v1":
        raise ValueError("unsupported camera-sample schema")
    if camera_report.get("read_only") is not True or camera_report.get("valid") is not True:
        raise ValueError("camera sample is not a valid read-only observation")
    if joint_report.get("schema") != "z_manip.piper_passive_joint_report.v1":
        raise ValueError("unsupported joint-report schema")
    if (
        joint_report.get("read_only") is not True
        or joint_report.get("complete_joint_feedback") is not True
        or joint_report.get("zero_transmit_verified") is not True
        or int(joint_report.get("interface_tx_packet_delta", -1)) != 0
    ):
        raise ValueError("joint report lacks complete, receive-only, zero-TX evidence")
    try:
        camera_stamp = int(camera_report["source_stamp_ns"])
        observation_start = int(joint_report["observation_start_unix_ns"])
        observation_end = int(joint_report["observation_end_unix_ns"])
        snapshot_span = float(joint_report["joint_snapshot_span_s"])
        reported_max_range = float(joint_report["max_joint_range_rad"])
        joints = np.asarray(joint_report["joint_positions_rad"], dtype=float)
        joint_ranges = np.asarray(joint_report["joint_ranges_rad"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("joint/camera timing or position evidence is invalid") from error
    if joints.shape != (chain.dof,) or joint_ranges.shape != (chain.dof,):
        raise ValueError(f"joint report must contain exactly {chain.dof} joints")
    numeric = np.concatenate((joints, joint_ranges, (snapshot_span, reported_max_range)))
    if not np.all(np.isfinite(numeric)) or np.any(joint_ranges < 0.0):
        raise ValueError("joint report contains invalid numeric evidence")
    if observation_end < observation_start:
        raise ValueError("joint observation time window is inverted")
    skew_ns = round(max_clock_skew_s * 1_000_000_000.0)
    if not observation_start - skew_ns <= camera_stamp <= observation_end + skew_ns:
        raise ValueError("camera stamp does not overlap the passive joint observation")
    if snapshot_span > max_snapshot_span_s:
        raise ValueError("joint feedback frames are too far apart for one FK snapshot")
    if reported_max_range != float(np.max(joint_ranges)):
        raise ValueError("reported maximum joint range is internally inconsistent")
    if reported_max_range > max_joint_range_rad:
        raise ValueError("arm moved during the camera/joint observation window")
    # The vendor URDF/SDK ranges are command/planning limits, not a validity
    # test for passive feedback collected in manual drag mode.  Hand-eye
    # calibration may retain real stationary feedback outside that envelope
    # because FK remains well-defined; automatic motion stays bound to URDF.
    if np.any(np.abs(joints) > max_abs_joint_feedback_rad):
        offenders = [
            f"J{index}={math.degrees(float(position)):+.3f}deg"
            for index, position in enumerate(joints, start=1)
            if abs(float(position)) > max_abs_joint_feedback_rad
        ]
        raise ValueError(
            "joint feedback exceeds the calibration plausibility envelope "
            f"(+/-{math.degrees(max_abs_joint_feedback_rad):.1f}deg): "
            + "; ".join(offenders)
        )
    limit_violations = planning_limit_violations(joints, chain)
    camera_from_target = rigid_transform(
        camera_report.get("camera_from_target"),
        "camera_from_target",
    )
    base_from_tip = chain.forward(joints)
    return {
        "source_stamp_ns": camera_stamp,
        "joint_observation_start_unix_ns": observation_start,
        "joint_observation_end_unix_ns": observation_end,
        "joint_names": list(chain.joint_names),
        "joint_positions_rad": joints.tolist(),
        "joint_ranges_rad": joint_ranges.tolist(),
        "joint_snapshot_span_s": snapshot_span,
        "base_from_tip": base_from_tip.tolist(),
        "camera_from_target": camera_from_target.tolist(),
        "safety_evidence": {
            "camera_read_only": True,
            "can_read_only": True,
            "can_tx_packet_delta": 0,
            "max_joint_range_rad": reported_max_range,
            "max_allowed_joint_range_rad": max_joint_range_rad,
            "max_allowed_snapshot_span_s": max_snapshot_span_s,
            "max_allowed_clock_skew_s": max_clock_skew_s,
            "max_abs_joint_feedback_rad": max_abs_joint_feedback_rad,
            "joint_limit_policy": "warning_only_for_passive_calibration",
            "planning_limits_enforced_for_automatic_motion": True,
            "planning_limit_violations": limit_violations,
        },
    }


def append_sample(
    dataset_path: Path,
    sample: dict[str, object],
    *,
    base_frame: str,
    tip_link: str,
    camera_frame: str,
    target_frame: str,
) -> dict[str, object]:
    destination = dataset_path.expanduser().resolve()
    if destination.exists():
        dataset = json.loads(destination.read_text(encoding="utf-8"))
        expected = {
            "schema": "z_manip.piper_hand_eye_samples.v1",
            "synthetic": False,
            "base_frame": base_frame,
            "tip_link": tip_link,
            "camera_frame": camera_frame,
            "target_frame": target_frame,
        }
        if not isinstance(dataset, dict) or any(dataset.get(key) != value for key, value in expected.items()):
            raise ValueError("existing hand-eye dataset metadata does not match this sample")
        samples = dataset.get("samples")
        if not isinstance(samples, list):
            raise ValueError("existing hand-eye dataset has no sample list")
    else:
        dataset = {
            "schema": "z_manip.piper_hand_eye_samples.v1",
            "synthetic": False,
            "base_frame": base_frame,
            "tip_link": tip_link,
            "camera_frame": camera_frame,
            "target_frame": target_frame,
            "samples": [],
        }
        samples = dataset["samples"]
    stamp = int(sample["source_stamp_ns"])
    if any(int(existing.get("source_stamp_ns", -1)) == stamp for existing in samples):
        raise ValueError("camera sample is already present in the hand-eye dataset")
    samples.append(sample)
    rendered = json.dumps(dataset, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(destination)
    return dataset


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-sample", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-link", default="piper_base_link")
    parser.add_argument("--tip-link", default="piper_gripper_base")
    parser.add_argument("--max-joint-range-rad", type=float, default=0.002)
    parser.add_argument("--max-snapshot-span-s", type=float, default=0.050)
    parser.add_argument("--max-clock-skew-s", type=float, default=0.250)
    parser.add_argument(
        "--max-abs-joint-feedback-rad",
        type=float,
        default=math.radians(190.0),
        help=(
            "generic corruption guard for passive joint feedback; independent "
            "of URDF planning limits (default: 190 degrees)"
        ),
    )
    values = parser.parse_args()
    limits = (
        values.max_joint_range_rad,
        values.max_snapshot_span_s,
        values.max_clock_skew_s,
        values.max_abs_joint_feedback_rad,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in limits):
        parser.error("all quality limits must be positive finite values")
    return values


def main() -> int:
    args = _arguments()
    camera_report = json.loads(args.camera_sample.read_text(encoding="utf-8"))
    joint_report = json.loads(args.joint_report.read_text(encoding="utf-8"))
    if not isinstance(camera_report, dict) or not isinstance(joint_report, dict):
        raise ValueError("camera and joint reports must be JSON objects")
    chain = KinematicChain.from_urdf(
        args.urdf.expanduser().resolve(),
        args.base_link,
        args.tip_link,
    )
    sample = assemble_sample(
        camera_report,
        joint_report,
        chain,
        max_joint_range_rad=args.max_joint_range_rad,
        max_snapshot_span_s=args.max_snapshot_span_s,
        max_clock_skew_s=args.max_clock_skew_s,
        max_abs_joint_feedback_rad=args.max_abs_joint_feedback_rad,
    )
    dataset = append_sample(
        args.dataset,
        sample,
        base_frame=args.base_link,
        tip_link=args.tip_link,
        camera_frame=str(camera_report.get("camera_frame", "")),
        target_frame=str(camera_report.get("target_frame", "")),
    )
    print(json.dumps({
        "dataset": str(args.dataset.expanduser().resolve()),
        "sample_count": len(dataset["samples"]),
        "source_stamp_ns": sample["source_stamp_ns"],
        "read_only_inputs": True,
        "planning_limit_violations": sample["safety_evidence"]["planning_limit_violations"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
