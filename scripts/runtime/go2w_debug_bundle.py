#!/usr/bin/env python3
"""Build one offline visualization/debug bundle from recorded artifacts.

The program only reads files and writes a JSON document.  It does not import
ROS, open SocketCAN, or contain an actuator transport.  Missing joint,
calibration, or planning evidence is represented as a blocked stage instead of
being silently treated as a successful end-to-end pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from z_manip.kinematics import KinematicChain


SCHEMA = "z_manip.debug_bundle.v1"
MAX_CLOUD_POINTS = 1_500
MAX_TRAJECTORY_POINTS = 200


def _load_json(path: Path | None, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, f"{label} was not supplied"
    if not path.is_file():
        return None, f"{label} does not exist: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"cannot read {label}: {error}"
    if not isinstance(value, dict):
        return None, f"{label} must contain a JSON object"
    return value, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ref(path: Path, output_parent: Path, **metadata: object) -> dict[str, object]:
    resolved = path.resolve()
    result: dict[str, object] = {
        "path": os.path.relpath(resolved, output_parent),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }
    result.update(metadata)
    return result


def _sample_rows(values: object, maximum: int) -> tuple[np.ndarray, list[int]]:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("sampled visualization array must be two-dimensional")
    if len(array) <= maximum:
        indices = np.arange(len(array), dtype=int)
    else:
        indices = np.unique(np.linspace(0, len(array) - 1, maximum, dtype=int))
    return array[indices], indices.tolist()


def _error(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, "details": details}


def _stage(
    name: str,
    status: str,
    *,
    metrics: dict[str, object] | None = None,
    error: dict[str, object] | None = None,
    input_refs: list[str] | None = None,
    output_refs: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "duration_ms": None,
        "input_refs": input_refs or [],
        "output_refs": output_refs or [],
        "metrics": metrics or {},
        "error": error,
    }


def _joint_gate(
    document: dict[str, Any] | None,
    load_error: str | None,
    source_stamp_ns: int,
    *,
    max_joint_range_rad: float = 0.002,
    max_snapshot_span_s: float = 0.050,
    max_clock_skew_s: float = 0.250,
) -> tuple[str, dict[str, object], dict[str, object] | None]:
    if document is None:
        return "blocked", {}, _error("MISSING_JOINT_REPORT", load_error or "joint report missing")
    required = (
        document.get("schema") == "z_manip.piper_passive_joint_report.v1"
        and document.get("read_only") is True
        and document.get("complete_joint_feedback") is True
        and document.get("zero_transmit_verified") is True
        and document.get("interface_tx_packet_delta") == 0
    )
    joints = np.asarray(document.get("joint_positions_rad", []), dtype=float)
    ranges = np.asarray(document.get("joint_ranges_rad", []), dtype=float)
    try:
        observation_start = int(document["observation_start_unix_ns"])
        observation_end = int(document["observation_end_unix_ns"])
        snapshot_span = float(document.get("joint_snapshot_span_s", 0.0))
        reported_max_range = float(document.get(
            "max_joint_range_rad",
            np.max(ranges) if ranges.shape == (6,) else float("nan"),
        ))
    except (KeyError, TypeError, ValueError, OverflowError):
        observation_start = 0
        observation_end = -1
        snapshot_span = float("nan")
        reported_max_range = float("nan")
    numeric_valid = (
        joints.shape == (6,)
        and ranges.shape == (6,)
        and np.all(np.isfinite(joints))
        and np.all(np.isfinite(ranges))
        and np.all(ranges >= 0.0)
        and np.isfinite(snapshot_span)
        and np.isfinite(reported_max_range)
    )
    skew_ns = round(max_clock_skew_s * 1_000_000_000.0)
    timing_valid = bool(
        source_stamp_ns > 0
        and observation_end >= observation_start
        and observation_start - skew_ns <= source_stamp_ns <= observation_end + skew_ns
    )
    metrics: dict[str, object] = {
        "complete_joint_feedback": document.get("complete_joint_feedback"),
        "zero_transmit_verified": document.get("zero_transmit_verified"),
        "interface_tx_packet_delta": document.get("interface_tx_packet_delta"),
        "joint_positions_rad": joints.tolist() if joints.shape == (6,) else [],
        "joint_ranges_rad": ranges.tolist() if ranges.shape == (6,) else [],
        "observation_start_unix_ns": observation_start,
        "observation_end_unix_ns": observation_end,
        "source_stamp_ns": source_stamp_ns,
        "source_stamp_overlaps_observation": timing_valid,
        "joint_snapshot_span_s": snapshot_span,
        "max_joint_range_rad": reported_max_range,
        "max_allowed_joint_range_rad": max_joint_range_rad,
        "max_allowed_snapshot_span_s": max_snapshot_span_s,
        "max_allowed_clock_skew_s": max_clock_skew_s,
    }
    if not required or not numeric_valid:
        return "failed", metrics, _error(
            "UNSAFE_OR_INVALID_JOINT_REPORT",
            "joint report lacks complete read-only zero-TX evidence or six finite joints",
        )
    if reported_max_range > max_joint_range_rad:
        return "failed", metrics, _error(
            "ARM_MOVED_DURING_OBSERVATION",
            "joint motion exceeded the read-only snapshot limit",
        )
    if snapshot_span > max_snapshot_span_s:
        return "failed", metrics, _error(
            "JOINT_SNAPSHOT_NOT_COHERENT",
            "the six-axis feedback frames are too far apart",
        )
    if not timing_valid:
        return "failed", metrics, _error(
            "STALE_JOINT_REPORT",
            "the camera artifact stamp does not overlap the passive joint observation",
        )
    if required and numeric_valid:
        return "ok", metrics, None
    raise AssertionError("unreachable joint gate state")


def _calibration_gate(document: dict[str, Any] | None, load_error: str | None) -> tuple[str, dict[str, object], dict[str, object] | None]:
    if document is None:
        return "blocked", {}, _error("MISSING_CALIBRATION", load_error or "calibration missing")
    metrics: dict[str, object] = {
        "schema": document.get("schema"),
        "calibrated": document.get("calibrated"),
        "synthetic": document.get("synthetic"),
        "calibration_id": document.get("calibration_id"),
        "mount_type": document.get("mount_type"),
        "camera_frame": document.get("camera_frame"),
        "sample_count": document.get("sample_count"),
        "quality": document.get("quality"),
        "quality_limits": document.get("quality_limits"),
    }
    if (
        document.get("schema") != "z_manip.piper_camera_calibration.v1"
        or document.get("calibrated") is not True
        or document.get("synthetic") is not False
        or not str(document.get("calibration_id", "")).strip()
    ):
        return "failed", metrics, _error(
            "UNCALIBRATED_OR_SYNTHETIC_CAMERA",
            "real planning requires a measured, calibrated, non-synthetic camera transform",
        )
    quality = document.get("quality")
    limits = document.get("quality_limits")
    try:
        passed = bool(
            isinstance(quality, dict)
            and isinstance(limits, dict)
            and int(document["sample_count"]) >= int(limits["min_samples"])
            and int(quality["rotation_axis_rank"]) >= int(limits["min_rotation_axis_rank"])
            and float(quality["max_pair_rotation_rad"]) >= float(limits["min_rotation_span_rad"])
            and float(quality["translation_rmse_m"]) <= float(limits["max_translation_rmse_m"])
            and float(quality["rotation_rmse_rad"]) <= float(limits["max_rotation_rmse_rad"])
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        passed = False
    if not passed:
        return "failed", metrics, _error(
            "CALIBRATION_QUALITY_GATE_FAILED",
            "camera calibration is missing quality evidence or exceeds its limits",
        )
    return "ok", metrics, None


def _load_cloud(path: Path, frame: str) -> tuple[dict[str, object] | None, str | None]:
    if not path.is_file():
        return None, f"missing point cloud: {path.name}"
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        return None, f"cannot load {path.name}: {error}"
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.all(np.isfinite(values)):
        return None, f"{path.name} is not a finite (N, 3) array"
    sampled, indices = _sample_rows(values, MAX_CLOUD_POINTS)
    return {
        "frame": frame,
        "source_count": len(values),
        "sample_count": len(sampled),
        "sample_indices": indices,
        "points_xyz_m": sampled.astype(float).tolist(),
        "source_dtype": str(values.dtype),
        "source_shape": list(values.shape),
    }, None


def _rigid_transform(value: object) -> np.ndarray | None:
    """Return a verified homogeneous rigid transform or ``None``."""

    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        transform.shape != (4, 4)
        or not np.all(np.isfinite(transform))
        or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
    ):
        return None
    return transform


def _transform_cloud(
    cloud: dict[str, object] | None,
    transform: np.ndarray,
    target_frame: str,
) -> dict[str, object] | None:
    if cloud is None:
        return None
    points = np.asarray(cloud["points_xyz_m"], dtype=float)
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    result = dict(cloud)
    result["source_frame"] = cloud.get("frame")
    result["frame"] = target_frame
    result["points_xyz_m"] = transformed.tolist()
    return result


def _load_candidates(path: Path) -> tuple[list[dict[str, object]], dict[str, object] | None, str | None]:
    if not path.is_file():
        return [], None, f"missing grasp archive: {path.name}"
    try:
        with np.load(path, allow_pickle=False) as archive:
            grasps = np.asarray(archive["grasps"], dtype=float)
            scores = np.asarray(archive["scores"], dtype=float)
            widths = np.asarray(archive["widths"], dtype=float)
            frame = str(archive["frame"].item())
            stamp_ns = int(archive["stamp_ns"].item())
            num_raw = int(archive["num_raw"].item())
    except (OSError, ValueError, KeyError) as error:
        return [], None, f"cannot load grasp archive: {error}"
    if (
        grasps.ndim != 3
        or grasps.shape[1:] != (4, 4)
        or scores.shape != (len(grasps),)
        or widths.shape not in ((0,), (len(grasps),))
        or not np.all(np.isfinite(grasps))
        or not np.all(np.isfinite(scores))
        or not np.all(np.isfinite(widths))
    ):
        return [], None, "grasp archive arrays are malformed or non-finite"
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    candidates = []
    for index, pose in enumerate(grasps):
        candidates.append({
            "candidate_id": index,
            "rank": int(ranks[index]),
            "source_score": float(scores[index]),
            "required_width_m": None if widths.size == 0 else float(widths[index]),
            "source_frame": frame,
            "pose_source": pose.tolist(),
            "pose_base": None,
            "status": "unevaluated",
            "rejections": [],
        })
    return candidates, {
        "frame": frame,
        "stamp_ns": stamp_ns,
        "raw_hypotheses": num_raw,
    }, None


def _trajectory_segment(
    values: object,
    times_s: object | None = None,
) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (6,) or not np.all(np.isfinite(array)):
        raise ValueError("trajectory segment must be a finite (N, 6) array")
    sampled, indices = _sample_rows(array, MAX_TRAJECTORY_POINTS)
    sampled_times: list[float] | None = None
    duration_s: float | None = None
    if times_s is not None:
        times = np.asarray(times_s, dtype=float)
        if (
            times.shape != (len(array),)
            or not np.all(np.isfinite(times))
            or not np.isclose(times[0], 0.0, atol=1e-9)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("trajectory times must be finite, increasing, and start at zero")
        sampled_times = times[indices].tolist()
        duration_s = float(times[-1])
    return {
        "source_waypoint_count": len(array),
        "sample_indices": indices,
        "positions_rad": sampled.tolist(),
        "times_s": sampled_times,
        "duration_s": duration_s,
    }


def _kinematic_visualization(
    urdf: Path,
    current_joints: object,
    segments: dict[str, object],
) -> dict[str, object]:
    chain = KinematicChain.from_urdf(
        urdf.expanduser().resolve(),
        "piper_base_link",
        "piper_gripper_base",
    )
    current = np.asarray(current_joints, dtype=float)
    frames = chain.link_transforms(current)
    names = list(frames)
    links = [
        [frames[first][:3, 3].tolist(), frames[second][:3, 3].tolist()]
        for first, second in zip(names, names[1:])
    ]
    trajectory: dict[str, list[list[float]]] = {}
    for name in ("transit", "approach", "lift"):
        segment = segments.get(name)
        if not isinstance(segment, dict):
            continue
        positions = np.asarray(segment.get("positions_rad"), dtype=float)
        if positions.ndim != 2 or positions.shape[1:] != (chain.dof,):
            raise ValueError(f"{name} visualization positions are invalid")
        trajectory[name] = [
            chain.forward(joints)[:3, 3].tolist()
            for joints in positions
        ]
    return {
        "robot_overlay": {
            "frame": chain.base_link,
            "link_names": names,
            "links_xyz_m": links,
        },
        "trajectory_xyz_m": trajectory,
        "kinematic_model": {
            "base_link": chain.base_link,
            "tip_link": chain.tip_link,
            "dof": chain.dof,
            "urdf_sha256": _sha256(urdf.expanduser().resolve()),
        },
    }


def build_bundle(
    perception_dir: Path,
    output: Path,
    *,
    planning_dir: Path | None = None,
    session_gate: Path | None = None,
    joint_report: Path | None = None,
    calibration: Path | None = None,
    urdf: Path | None = None,
) -> dict[str, object]:
    perception_dir = perception_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    output_parent = output.parent
    report_path = perception_dir / "report.json"
    perception, perception_error = _load_json(report_path, "perception report")
    perception = perception or {}
    request_id = str(perception.get("request_id", "unknown"))
    stamp_ns = int(perception.get("stamp_ns", 0) or 0)
    run_digest = hashlib.sha256(f"{request_id}:{stamp_ns}:{perception_dir}".encode()).hexdigest()[:16]
    frame = str(perception.get("frame", "unknown"))

    artifacts: dict[str, object] = {}
    for key, filename in (
        ("perception_report", "report.json"),
        ("segmentation_mask", "edgetam_mask.png"),
        ("segmentation_overlay", "edgetam_overlay.png"),
        ("candidate_overlay", "grasp_candidates_overlay.png"),
        ("candidate_archive", "grasp_candidates.npz"),
        ("target_cloud", "target_points.npy"),
        ("scene_cloud", "scene_collision_points.npy"),
    ):
        path = perception_dir / filename
        if path.is_file():
            artifacts[key] = _file_ref(path, output_parent)

    candidates, candidate_metadata, candidate_error = _load_candidates(
        perception_dir / "grasp_candidates.npz",
    )
    if candidate_metadata is not None:
        frame = str(candidate_metadata["frame"])
        stamp_ns = int(candidate_metadata["stamp_ns"])
    target_cloud, target_error = _load_cloud(perception_dir / "target_points.npy", frame)
    scene_cloud, scene_error = _load_cloud(
        perception_dir / "scene_collision_points.npy",
        frame,
    )

    joint, joint_error = _load_json(joint_report, "joint report")
    joint_status, joint_metrics, joint_stage_error = _joint_gate(
        joint,
        joint_error,
        stamp_ns,
    )
    if joint_report is not None and joint_report.is_file():
        artifacts["joint_report"] = _file_ref(joint_report, output_parent)

    calibration_document, calibration_error = _load_json(calibration, "camera calibration")
    calibration_status, calibration_metrics, calibration_stage_error = _calibration_gate(
        calibration_document,
        calibration_error,
    )
    if calibration is not None and calibration.is_file():
        artifacts["camera_calibration"] = _file_ref(calibration, output_parent)

    planning_report_path = None if planning_dir is None else planning_dir.expanduser().resolve() / "planning_report.json"
    planning, planning_error = _load_json(planning_report_path, "planning report")
    if planning_report_path is not None and planning_report_path.is_file():
        artifacts["planning_report"] = _file_ref(planning_report_path, output_parent)
    gate, gate_error = _load_json(session_gate, "planning session gate")
    if session_gate is not None and session_gate.is_file():
        artifacts["planning_session_gate"] = _file_ref(session_gate, output_parent)

    rejections: list[dict[str, object]] = []
    selected_plan: dict[str, object] | None = None
    planning_valid = bool(planning is not None and planning.get("plan_valid") is True)
    base_from_camera = None
    if planning is not None:
        base_from_camera = _rigid_transform(planning.get("base_from_camera"))
    if base_from_camera is None and gate is not None and gate.get("planning_ready") is True:
        base_from_camera = _rigid_transform(gate.get("base_from_camera"))
    if planning is not None:
        value = planning.get("rejections", [])
        if isinstance(value, list):
            rejections = [dict(item) for item in value if isinstance(item, dict)]
        selected_index = planning.get("candidate_index") if planning_valid else None
        for rejection in rejections:
            index = rejection.get("candidate_index")
            if isinstance(index, int) and 0 <= index < len(candidates):
                candidates[index]["rejections"].append(rejection)
                if candidates[index]["status"] == "unevaluated":
                    candidates[index]["status"] = "rejected"
        if isinstance(selected_index, int) and 0 <= selected_index < len(candidates):
            candidates[selected_index]["status"] = "selected"
    if base_from_camera is not None:
        for candidate in candidates:
            source_pose = np.asarray(candidate["pose_source"], dtype=float)
            candidate["pose_base"] = (base_from_camera @ source_pose).tolist()

    planned_archive_path = None if planning_dir is None else planning_dir.expanduser().resolve() / "planned_grasp.npz"
    trajectory_error = None
    if planning_valid:
        selected_plan = {
            "candidate_id": planning.get("candidate_index"),
            "symmetry_index": planning.get("symmetry_index"),
            "selected_global_rank": planning.get("selected_global_rank"),
            "higher_rank_rejection_count": planning.get(
                "higher_rank_rejection_count"
            ),
            "score": planning.get("score"),
            "required_width_m": planning.get("required_width_m"),
            "grasp_pose_base": planning.get("grasp_pose"),
            "pregrasp_pose_base": planning.get("pregrasp_pose"),
            "joint_names": [f"joint{i}" for i in range(1, 7)],
            "segments": {},
        }
        if planned_archive_path is None or not planned_archive_path.is_file():
            trajectory_error = "valid planning report has no planned_grasp.npz"
        else:
            artifacts["planned_grasp"] = _file_ref(planned_archive_path, output_parent)
            try:
                with np.load(planned_archive_path, allow_pickle=False) as archive:
                    selected_plan["segments"] = {
                        name: _trajectory_segment(
                            archive[name],
                            archive[f"{name}_times_s"]
                            if f"{name}_times_s" in archive.files
                            else None,
                        )
                        for name in ("transit", "approach", "lift")
                    }
                    selected_plan["current_joints_rad"] = np.asarray(
                        archive["current_joints"], dtype=float,
                    ).tolist()
            except (OSError, ValueError, KeyError) as error:
                trajectory_error = f"cannot load planned trajectory: {error}"

    perception_failed = perception_error is not None or perception.get("perception_bundle_valid") is False
    segmentation_ok = (perception_dir / "edgetam_mask.png").is_file() and (perception_dir / "edgetam_overlay.png").is_file()
    cloud_ok = target_cloud is not None and scene_cloud is not None
    grasp_ok = bool(candidates) and perception.get("grasp_generation_valid", True) is True
    stages = [
        _stage(
            "perception_bundle",
            "failed" if perception_failed else "ok",
            metrics={"stamp_ns": stamp_ns, "frame": frame, "request_id": request_id},
            error=None if not perception_failed else _error("PERCEPTION_BUNDLE_INVALID", perception_error or str(perception.get("error", "invalid perception bundle"))),
            output_refs=["artifacts.perception_report"],
        ),
        _stage(
            "segmentation",
            "ok" if segmentation_ok else "failed",
            metrics={"backend": "EdgeTAM"},
            error=None if segmentation_ok else _error("MISSING_SEGMENTATION_ARTIFACT", "mask or overlay image is missing"),
            output_refs=["artifacts.segmentation_mask", "artifacts.segmentation_overlay"],
        ),
        _stage(
            "point_cloud",
            "ok" if cloud_ok else "failed",
            metrics={
                "target_points": None if target_cloud is None else target_cloud["source_count"],
                "scene_points": None if scene_cloud is None else scene_cloud["source_count"],
            },
            error=None if cloud_ok else _error("INVALID_POINT_CLOUD", "; ".join(value for value in (target_error, scene_error) if value)),
            output_refs=["artifacts.target_cloud", "artifacts.scene_cloud"],
        ),
        _stage(
            "grasp_generation",
            "ok" if grasp_ok else "failed",
            metrics={
                "backend": perception.get("grasp_backend"),
                "candidate_count": len(candidates),
                "raw_hypotheses": None if candidate_metadata is None else candidate_metadata["raw_hypotheses"],
                "learned_backend_error": perception.get("learned_backend_error"),
            },
            error=None if grasp_ok else _error("GRASP_GENERATION_FAILED", candidate_error or str(perception.get("grasp_generation_error", "no candidates"))),
            output_refs=["artifacts.candidate_archive", "artifacts.candidate_overlay"],
        ),
        _stage("joint_state_gate", joint_status, metrics=joint_metrics, error=joint_stage_error, input_refs=["artifacts.joint_report"]),
        _stage("calibration_gate", calibration_status, metrics=calibration_metrics, error=calibration_stage_error, input_refs=["artifacts.camera_calibration"]),
    ]
    planning_frame = None if planning is None else planning.get("planning_frame")
    if (
        not isinstance(planning_frame, str)
        or not planning_frame.strip()
    ) and gate is not None and gate.get("planning_ready") is True:
        # This workbench's verified gate is defined for the fixed PiPER chain
        # rooted at piper_base_link.  Older gate artifacts predate an explicit
        # target_frame field, so keep them diagnosable instead of falsely
        # reporting the camera calibration as invalid.
        planning_frame = gate.get("target_frame", "piper_base_link")
    transform_status = "ok" if (
        calibration_status == "ok"
        and base_from_camera is not None
        and isinstance(planning_frame, str)
        and bool(planning_frame.strip())
    ) else "blocked"
    stages.append(_stage(
        "frame_transform",
        transform_status,
        metrics={
            "source_frame": frame,
            "planning_frame": planning_frame,
            "session_gate_error": gate_error,
        },
        error=None if transform_status == "ok" else _error(
            "MISSING_VERIFIED_FRAME_TRANSFORM",
            "verified rigid base-from-camera transform is unavailable",
        ),
    ))
    planning_prerequisites_ok = (
        joint_status == "ok"
        and calibration_status == "ok"
        and transform_status == "ok"
    )
    if planning is None:
        plan_error = _error("MISSING_PLANNING_REPORT", planning_error or "planning report missing")
        stages.extend([
            _stage("ik", "blocked", error=plan_error),
            _stage("collision_check", "blocked", error=plan_error),
            _stage("motion_plan", "blocked", error=plan_error),
        ])
    elif not planning_prerequisites_ok:
        prerequisite_error = _error(
            "UPSTREAM_GATE_NOT_VERIFIED",
            "recorded planning output is not trusted until joint, calibration, and frame-transform gates pass",
        )
        stages.extend([
            _stage("ik", "blocked", error=prerequisite_error),
            _stage("collision_check", "blocked", error=prerequisite_error),
            _stage("motion_plan", "blocked", error=prerequisite_error),
        ])
    elif planning_valid and trajectory_error is None:
        stages.extend([
            _stage("ik", "ok", metrics={"selected_candidate_id": planning.get("candidate_index")}),
            _stage("collision_check", "ok", metrics={"rejection_count": len(rejections)}),
            _stage("motion_plan", "ok", metrics={
                "transit_waypoints": planning.get("transit_waypoints"),
                "approach_waypoints": planning.get("approach_waypoints"),
                "lift_waypoints": planning.get("lift_waypoints"),
            }),
        ])
    else:
        message = trajectory_error or str(planning.get("error", "no valid plan"))
        rejection_stage_counts: dict[str, int] = {}
        for rejection in rejections:
            rejection_stage = str(rejection.get("stage", "unknown"))
            rejection_stage_counts[rejection_stage] = (
                rejection_stage_counts.get(rejection_stage, 0) + 1
            )

        ik_rejected = rejection_stage_counts.get("ik", 0)
        collision_rejected = sum(
            rejection_stage_counts.get(stage_name, 0)
            for stage_name in ("approach_collision", "lift_collision")
        )
        planning_rejected = rejection_stage_counts.get("planning", 0)

        # Each rejection is terminal for one candidate/symmetry hypothesis.  A
        # hypothesis rejected by collision necessarily passed IK; one rejected
        # by planning necessarily passed both IK and collision.  Do not mark an
        # entire upstream stage failed merely because some parallel hypotheses
        # were rejected there.
        ik_passed = collision_rejected + planning_rejected
        collision_passed = planning_rejected
        ik_metrics = {
            "passed_hypotheses": ik_passed,
            "rejected_hypotheses": ik_rejected,
        }
        collision_metrics = {
            "passed_hypotheses": collision_passed,
            "rejected_hypotheses": collision_rejected,
        }
        planning_metrics = {
            "attempted_hypotheses": planning_rejected,
            "failed_hypotheses": planning_rejected,
        }

        if ik_passed:
            ik_stage = _stage("ik", "ok", metrics=ik_metrics)
        else:
            ik_stage = _stage(
                "ik",
                "failed",
                metrics=ik_metrics,
                error=_error("IK_NOT_VERIFIED", message),
            )

        if collision_passed:
            collision_stage = _stage(
                "collision_check", "ok", metrics=collision_metrics
            )
        elif ik_passed:
            collision_stage = _stage(
                "collision_check",
                "failed",
                metrics=collision_metrics,
                error=_error("COLLISION_NOT_VERIFIED", message),
            )
        else:
            collision_stage = _stage(
                "collision_check",
                "blocked",
                metrics=collision_metrics,
                error=_error("UPSTREAM_IK_NOT_VERIFIED", message),
            )

        if planning_rejected:
            motion_plan_stage = _stage(
                "motion_plan",
                "failed",
                metrics=planning_metrics,
                error=_error("PLANNING_FAILED", message),
            )
        else:
            motion_plan_stage = _stage(
                "motion_plan",
                "blocked",
                metrics=planning_metrics,
                error=_error("UPSTREAM_COLLISION_NOT_VERIFIED", message),
            )
        stages.extend([
            ik_stage,
            collision_stage,
            motion_plan_stage,
        ])

    reported_motion = None if planning is None else planning.get("motion_commands_published")
    safety_error = None
    if reported_motion not in (None, 0):
        safety_error = _error("UPSTREAM_MOTION_REPORTED", "planning report says motion commands were published", reported=reported_motion)
    stages.append(_stage(
        "safety_gate",
        "ok" if safety_error is None else "failed",
        metrics={"bundle_generator_motion_commands_published": 0, "upstream_reported_motion_commands_published": reported_motion},
        error=safety_error,
    ))

    first_problem = next((stage for stage in stages if stage["status"] != "ok"), None)
    all_ok = first_problem is None
    source_rejection_count = 0 if planning is None else int(planning.get("rejection_count", len(rejections)) or 0)
    kinematic_visualization: dict[str, object] = {}
    kinematic_visualization_error: str | None = None
    overlay_allowed = calibration_status == "ok" and transform_status == "ok"
    current_visual_joints: object | None = None
    if planning is not None:
        current_visual_joints = planning.get("measured_joints_rad")
        if current_visual_joints is None:
            current_visual_joints = planning.get("current_joints_rad")
    if current_visual_joints is None and joint_status == "ok":
        current_visual_joints = joint_metrics.get("joint_positions_rad")
    if overlay_allowed and urdf is not None and current_visual_joints is not None:
        try:
            kinematic_visualization = _kinematic_visualization(
                urdf,
                current_visual_joints,
                {} if selected_plan is None else selected_plan.get("segments", {}),
            )
            kinematic_visualization["robot_overlay"]["pose_source"] = (
                "measured_passive_feedback"
            )
        except (OSError, ValueError) as error:
            kinematic_visualization_error = f"{type(error).__name__}: {error}"

    display_target_cloud = target_cloud
    display_scene_cloud = scene_cloud
    display_frame = frame
    reference_axes: list[dict[str, object]] = []
    if overlay_allowed and base_from_camera is not None and isinstance(planning_frame, str):
        display_target_cloud = _transform_cloud(target_cloud, base_from_camera, planning_frame)
        display_scene_cloud = _transform_cloud(scene_cloud, base_from_camera, planning_frame)
        display_frame = planning_frame
        reference_axes = [
            {"name": "base", "frame": planning_frame, "pose": np.eye(4).tolist()},
            {
                "name": "camera",
                "frame": planning_frame,
                "pose": base_from_camera.tolist(),
            },
        ]
    bundle: dict[str, object] = {
        "schema": SCHEMA,
        "run_id": f"debug-{run_digest}",
        "request_id": request_id,
        "created_unix_ns": time.time_ns(),
        "mode": {"read_only": True, "planning_only": True, "offline_artifact_reader": True},
        "status": {
            "ok": all_ok,
            "state": "ok" if all_ok else str(first_problem["status"]),
            "first_failed_stage": None if all_ok else first_problem["name"],
            "error_code": None if all_ok or first_problem["error"] is None else first_problem["error"]["code"],
            "message": None if all_ok or first_problem["error"] is None else first_problem["error"]["message"],
        },
        "safety": {
            "motion_commands_published": 0,
            "transport_opened": False,
            "ros_imported": False,
            "can_opened": False,
            "can_tx_packet_delta": None if joint is None else joint.get("interface_tx_packet_delta"),
            "zero_transmit_verified": None if joint is None else joint.get("zero_transmit_verified"),
            "upstream_reported_motion_commands_published": reported_motion,
        },
        "units": {"length": "m", "angle": "rad", "time": "s", "transform": "row-major target_from_source"},
        "frames": {"perception": frame, "planning": planning_frame},
        "inputs": {
            "instruction": perception.get("instruction"),
            "source_stamp_ns": stamp_ns,
            "joint_report_supplied": joint_report is not None,
            "calibration_supplied": calibration is not None,
            "planning_dir_supplied": planning_dir is not None,
            "urdf_supplied": urdf is not None,
            "kinematic_visualization_error": kinematic_visualization_error,
        },
        "stages": stages,
        "candidates": candidates,
        "planning": {
            "available": planning is not None,
            "plan_valid": planning_valid,
            "selection_status": (
                "selected"
                if planning_valid
                else "no_feasible_candidate"
                if planning is not None
                else "unavailable"
            ),
            "selected_global_rank": (
                None if planning is None else planning.get("selected_global_rank")
            ),
            "higher_rank_rejection_count": (
                None
                if planning is None
                else planning.get("higher_rank_rejection_count")
            ),
            "source_rejection_count": source_rejection_count,
            "included_rejection_count": len(rejections),
            "source_rejections_may_be_truncated": source_rejection_count > len(rejections),
            "rejections": rejections,
        },
        "selected_plan": selected_plan,
        "artifacts": artifacts,
        "visualization": {
            "frame": display_frame,
            "robot_overlay_allowed": overlay_allowed,
            "images": {
                name: artifacts[name]["path"]
                for name in ("segmentation_mask", "segmentation_overlay", "candidate_overlay")
                if name in artifacts
            },
            "target_cloud": display_target_cloud,
            "scene_cloud": display_scene_cloud,
            "reference_axes": reference_axes,
            "candidate_axes": [
                {
                    "candidate_id": item["candidate_id"],
                    "rank": item["rank"],
                    "status": item["status"],
                    "pose": item["pose_base"] or item["pose_source"],
                    "frame": (None if item["pose_base"] is None else (None if planning is None else planning.get("planning_frame"))) or frame,
                }
                for item in candidates
            ],
            "joint_trajectory": None if selected_plan is None else selected_plan.get("segments"),
            **kinematic_visualization,
        },
    }
    return bundle


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--planning-dir", type=Path)
    parser.add_argument("--session-gate", type=Path)
    parser.add_argument("--joint-report", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    output = args.output.expanduser().resolve()
    bundle = build_bundle(
        args.perception_dir,
        output,
        planning_dir=args.planning_dir,
        session_gate=args.session_gate,
        joint_report=args.joint_report,
        calibration=args.calibration,
        urdf=args.urdf,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "status": bundle["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
