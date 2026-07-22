#!/usr/bin/env python3
"""Validate one synchronized real-perception planning session offline.

The gate reads immutable perception, passive-CAN, calibration, and URDF files.
It opens no ROS, network, CAN, or actuator transport.  A JSON report is always
written so the visualization layer can explain why planning was blocked.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from z_manip.kinematics.chain import KinematicChain


SCHEMA = "z_manip.piper_planning_session_gate.v1"
CAMERA_FRAME = "camera_color_optical_frame"
MAX_CLOCK_SKEW_S = 0.250
MAX_JOINT_RANGE_RAD = 0.002
MAX_SNAPSHOT_SPAN_S = 0.050
MAX_START_LIMIT_PROJECTION_RAD = 0.010
DIRECT_HANDOFF_RANGE_M = 0.60
MAX_HANDOFF_RANGE_M = 0.70


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _rigid(value: object, label: str) -> np.ndarray:
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


def _calibration_quality_passes(document: dict[str, Any]) -> bool:
    quality = document.get("quality")
    limits = document.get("quality_limits")
    if not isinstance(quality, dict) or not isinstance(limits, dict):
        return False
    try:
        return bool(
            int(document["sample_count"]) >= int(limits["min_samples"])
            and int(quality["rotation_axis_rank"])
            >= int(limits["min_rotation_axis_rank"])
            and float(quality["max_pair_rotation_rad"])
            >= float(limits["min_rotation_span_rad"])
            and 0.0
            <= float(quality["translation_rmse_m"])
            <= float(limits["max_translation_rmse_m"])
            and 0.0
            <= float(quality["rotation_rmse_rad"])
            <= float(limits["max_rotation_rmse_rad"])
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def classify_handoff_workspace(target_points_base: object) -> dict[str, object]:
    """Classify target range before the expensive handoff IK search.

    The median point is deliberately used instead of the cloud mean so a small
    number of depth outliers cannot turn a near target into a far target (or the
    reverse).  The 0.60--0.70 m gray zone remains available to the exact IK
    solver; only targets strictly beyond 0.70 m are routed back to base
    approach.
    """

    points = np.asarray(target_points_base, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 1
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("target point cloud in piper_base_link must have shape (N, 3)")

    robust_center = np.median(points, axis=0)
    target_range_m = float(np.linalg.norm(robust_center))
    if target_range_m > MAX_HANDOFF_RANGE_M:
        state = "NEED_BASE_APPROACH"
        planning_allowed = False
    elif target_range_m >= DIRECT_HANDOFF_RANGE_M:
        state = "PRECISION_IK"
        planning_allowed = True
    else:
        state = "NEAR_FIELD_IK"
        planning_allowed = True

    return {
        "state": state,
        "planning_allowed": planning_allowed,
        "frame": "piper_base_link",
        "target_range_m": target_range_m,
        "target_robust_center_base": robust_center.tolist(),
        "direct_handoff_range_m": DIRECT_HANDOFF_RANGE_M,
        "maximum_handoff_range_m": MAX_HANDOFF_RANGE_M,
    }


def evaluate_session(
    perception_dir: Path,
    joint_report_path: Path,
    calibration_path: Path,
    urdf_path: Path,
) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    def reject(code: str, message: str, **details: object) -> None:
        errors.append({"code": code, "message": message, "details": details})

    perception = _load_json(perception_dir / "report.json", "perception report")
    joints_report = _load_json(joint_report_path, "passive joint report")
    calibration = _load_json(calibration_path, "camera calibration")
    archive_path = perception_dir / "grasp_candidates.npz"
    try:
        archive = np.load(archive_path, allow_pickle=False)
        source_stamp_ns = int(archive["stamp_ns"].item())
        source_frame = str(archive["frame"].item())
        grasp_count = int(len(archive["grasps"]))
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid grasp candidate archive: {error}") from error

    if perception.get("read_only") is not True:
        reject("PERCEPTION_NOT_READ_ONLY", "perception report lacks read-only provenance")
    if not bool(perception.get("grasp_generation_valid")) or grasp_count < 1:
        reject(
            "NO_GRASP_CANDIDATES",
            "perception did not produce a valid grasp candidate",
            candidate_count=grasp_count,
            reported_error=perception.get("grasp_generation_error", ""),
        )
    if int(perception.get("stamp_ns", -1)) != source_stamp_ns:
        reject("PERCEPTION_STAMP_MISMATCH", "report and candidate archive stamps differ")
    if str(perception.get("frame", "")) != source_frame:
        reject("PERCEPTION_FRAME_MISMATCH", "report and candidate archive frames differ")

    joint_schema_ok = bool(
        joints_report.get("schema") == "z_manip.piper_passive_joint_report.v1"
        and joints_report.get("read_only") is True
        and joints_report.get("complete_joint_feedback") is True
        and joints_report.get("zero_transmit_verified") is True
        and int(joints_report.get("interface_tx_packet_delta", -1)) == 0
    )
    if not joint_schema_ok:
        reject("INVALID_PASSIVE_JOINT_REPORT", "joint report lacks complete zero-TX evidence")
    try:
        observation_start = int(joints_report["observation_start_unix_ns"])
        observation_end = int(joints_report["observation_end_unix_ns"])
        snapshot_span_s = float(joints_report["joint_snapshot_span_s"])
        max_joint_range_rad = float(joints_report["max_joint_range_rad"])
        current_joints = np.asarray(joints_report["joint_positions_rad"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("passive joint timing/position evidence is invalid") from error
    if current_joints.shape != (6,) or not np.all(np.isfinite(current_joints)):
        reject("INVALID_JOINT_VECTOR", "joint report must contain six finite radians")
    if max_joint_range_rad > MAX_JOINT_RANGE_RAD:
        reject(
            "ARM_MOVED_DURING_CAPTURE",
            "arm moved during the synchronized perception window",
            observed=max_joint_range_rad,
            allowed=MAX_JOINT_RANGE_RAD,
        )
    if snapshot_span_s > MAX_SNAPSHOT_SPAN_S:
        reject(
            "JOINT_SNAPSHOT_TOO_WIDE",
            "joint feedback frames are too far apart",
            observed=snapshot_span_s,
            allowed=MAX_SNAPSHOT_SPAN_S,
        )
    skew_ns = round(MAX_CLOCK_SKEW_S * 1_000_000_000.0)
    overlaps = observation_start - skew_ns <= source_stamp_ns <= observation_end + skew_ns
    if not overlaps:
        reject(
            "STALE_JOINT_REPORT",
            "perception stamp does not overlap passive joint observation",
            source_stamp_ns=source_stamp_ns,
            observation_start_unix_ns=observation_start,
            observation_end_unix_ns=observation_end,
        )

    calibration_ok = bool(
        calibration.get("schema") == "z_manip.piper_camera_calibration.v1"
        and calibration.get("calibrated") is True
        and calibration.get("synthetic") is False
        and str(calibration.get("calibration_id", "")).strip()
        and _calibration_quality_passes(calibration)
    )
    if not calibration_ok:
        reject("INVALID_CALIBRATION", "real camera calibration does not pass its quality gates")
    if source_frame != CAMERA_FRAME or calibration.get("camera_frame") != source_frame:
        reject(
            "CAMERA_FRAME_MISMATCH",
            "perception and calibration camera frames differ",
            perception_frame=source_frame,
            calibration_frame=calibration.get("camera_frame"),
        )

    chain = KinematicChain.from_urdf(
        urdf_path,
        "piper_base_link",
        "piper_gripper_base",
    )
    planning_start_joints = current_joints.copy()
    start_limit_projection = np.zeros_like(current_joints)
    if current_joints.shape == (chain.dof,):
        below = current_joints < chain.lower_limits
        above = current_joints > chain.upper_limits
        if np.any(below | above):
            planning_start_joints = np.clip(
                current_joints,
                chain.lower_limits,
                chain.upper_limits,
            )
            start_limit_projection = planning_start_joints - current_joints
            violations = []
            for index in np.flatnonzero(below | above):
                violations.append({
                    "joint_index": int(index + 1),
                    "position_deg": math.degrees(float(current_joints[index])),
                    "lower_deg": math.degrees(float(chain.lower_limits[index])),
                    "upper_deg": math.degrees(float(chain.upper_limits[index])),
                    "projection_deg": math.degrees(
                        float(start_limit_projection[index]),
                    ),
                })
            projection_max = float(np.max(np.abs(start_limit_projection)))
            if projection_max > MAX_START_LIMIT_PROJECTION_RAD:
                reject(
                    "PLANNING_START_OUTSIDE_URDF",
                    "measured planning start exceeds the allowed passive "
                    "feedback-to-command-limit reconciliation tolerance",
                    violations=violations,
                    projection_max_rad=projection_max,
                    allowed_projection_rad=MAX_START_LIMIT_PROJECTION_RAD,
                )
            else:
                warnings.append({
                    "code": "PLANNING_START_PROJECTED_TO_URDF_LIMIT",
                    "message": (
                        "planning-only start was projected by a small amount "
                        "onto the URDF command boundary"
                    ),
                    "details": {
                        "violations": violations,
                        "projection_max_rad": projection_max,
                        "allowed_projection_rad": (
                            MAX_START_LIMIT_PROJECTION_RAD
                        ),
                        "execution_requires_start_state_reconciliation": True,
                    },
                })

    base_from_camera: np.ndarray | None = None
    target_centroid_base: list[float] | None = None
    handoff_workspace: dict[str, object] | None = None
    planning_disposition = "INVALID_SESSION"
    try:
        tip_from_camera = _rigid(calibration.get("tip_from_camera"), "tip_from_camera")
        base_from_camera = chain.forward(current_joints) @ tip_from_camera
        target = np.load(perception_dir / "target_points.npy", allow_pickle=False)
        target = np.asarray(target, dtype=float)
        if target.ndim != 2 or target.shape[1] != 3 or not np.all(np.isfinite(target)):
            raise ValueError("target point cloud must have shape (N, 3)")
        target_base = target @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]
        target_centroid_base = np.mean(target_base, axis=0).tolist()
        handoff_workspace = classify_handoff_workspace(target_base)
        planning_disposition = str(handoff_workspace["state"])
        if not bool(handoff_workspace["planning_allowed"]):
            reject(
                "NEED_BASE_APPROACH",
                "target is outside the near-field handoff workspace; continue "
                "base approach before running IK",
                target_range_m=handoff_workspace["target_range_m"],
                maximum_handoff_range_m=MAX_HANDOFF_RANGE_M,
                frame="piper_base_link",
            )
    except (OSError, ValueError) as error:
        reject("INVALID_FRAME_TRANSFORM_INPUT", str(error))

    return {
        "schema": SCHEMA,
        "planning_ready": not errors,
        "read_only": True,
        "planning_only": True,
        "motion_commands_published": 0,
        "transport_opened": False,
        "source_stamp_ns": source_stamp_ns,
        "source_frame": source_frame,
        "candidate_count": grasp_count,
        "camera_calibration_id": calibration.get("calibration_id"),
        "joint_observation_overlaps": overlaps,
        "current_joints_rad": current_joints.tolist(),
        "measured_joints_rad": current_joints.tolist(),
        "planning_start_joints_rad": planning_start_joints.tolist(),
        "start_limit_projection_rad": start_limit_projection.tolist(),
        "start_limit_projection_max_rad": float(
            np.max(np.abs(start_limit_projection)),
        ),
        "execution_start_requires_limit_reconciliation": bool(
            np.any(start_limit_projection != 0.0),
        ),
        "joints_csv": ",".join(f"{value:.12g}" for value in current_joints),
        "planning_joints_csv": ",".join(
            f"{value:.12g}" for value in planning_start_joints
        ),
        "base_from_camera": None if base_from_camera is None else base_from_camera.tolist(),
        "target_centroid_base": target_centroid_base,
        "handoff_workspace": handoff_workspace,
        "planning_disposition": planning_disposition,
        "errors": errors,
        "warnings": warnings,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        report = evaluate_session(
            args.perception_dir.expanduser().resolve(),
            args.joint_report.expanduser().resolve(),
            args.calibration.expanduser().resolve(),
            args.urdf.expanduser().resolve(),
        )
    except ValueError as error:
        report = {
            "schema": SCHEMA,
            "planning_ready": False,
            "read_only": True,
            "planning_only": True,
            "motion_commands_published": 0,
            "transport_opened": False,
            "errors": [{"code": "INVALID_SESSION_INPUT", "message": str(error)}],
        }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["planning_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
