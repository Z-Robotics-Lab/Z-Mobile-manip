#!/usr/bin/env python3
"""Filter recorded grasp candidates through PiPER IK and collision planning.

This program is deliberately ROS-free and has no actuator transport.  It only
loads immutable perception artifacts, an explicit measured joint vector, and
an explicit calibrated camera transform. For the wrist-mounted D435, the fixed
tip-to-camera calibration is composed with FK at the measured joint vector.
Missing calibration or joint feedback is a hard error rather than an
invitation to use a nominal transform on real hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Sequence

import numpy as np


COLLISION_WITNESS_REPLAY_LIMIT = 4


# A resident, network-disabled planning worker may call ``main`` repeatedly.
# Keep only the expensive immutable robot models warm.  Per-request scene,
# target, collision margins, controls, trajectories, and artifacts are always
# rebuilt below.  The key includes file identity and every option that changes
# StackConfig, so a deployed config/URDF update cannot silently reuse an old
# model.
_PLANNER_CACHE: dict[tuple[object, ...], tuple[object, object, object]] = {}


def grasp_centrality_scores(
    grasps: object,
    centroid: object,
    target_points: object,
) -> np.ndarray:
    """Score grasp origins in the robust object frame, from edge to centre."""

    poses = np.asarray(grasps, dtype=float)
    center = np.asarray(centroid, dtype=float)
    points = np.asarray(target_points, dtype=float)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not np.all(np.isfinite(poses))
    ):
        raise ValueError("grasp-centrality poses must be finite Nx4x4 poses")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("grasp-centrality centroid must be a finite 3-vector")
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 8
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("grasp-centrality target cloud must contain finite Nx3 points")

    centered_points = points - center
    _, _, object_axes = np.linalg.svd(centered_points, full_matrices=False)
    projected_points = centered_points @ object_axes.T
    robust_lower = np.quantile(projected_points, 0.10, axis=0)
    robust_upper = np.quantile(projected_points, 0.90, axis=0)
    # The 6 mm floor avoids making a thin/noisy depth axis dominate the score.
    robust_half_extent = np.maximum(0.5 * (robust_upper - robust_lower), 0.006)
    origin_offsets = (poses[:, :3, 3] - center) @ object_axes.T
    normalized_offsets = origin_offsets / robust_half_extent
    squared_distances = np.einsum(
        "ij,ij->i",
        normalized_offsets,
        normalized_offsets,
    )
    return np.exp(-0.5 * squared_distances)


def support_approach_scores(
    grasps: object,
    scores: object,
    lift_direction: object,
    *,
    weight: float,
    centralities: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefer central, supported-object approaches opposite the lift direction."""

    poses = np.asarray(grasps, dtype=float)
    values = np.asarray(scores, dtype=float)
    lift = np.asarray(lift_direction, dtype=float)
    prior_weight = float(weight)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not np.all(np.isfinite(poses))
    ):
        raise ValueError("support-approach grasps must be finite Nx4x4 poses")
    if values.shape != (len(poses),) or not np.all(np.isfinite(values)):
        raise ValueError("support-approach scores must align with grasps")
    if lift.shape != (3,) or not np.all(np.isfinite(lift)):
        raise ValueError("support-approach lift direction must be a finite 3-vector")
    lift_norm = float(np.linalg.norm(lift))
    if lift_norm <= 1e-9:
        raise ValueError("support-approach lift direction must be nonzero")
    if not math.isfinite(prior_weight) or not 0.0 <= prior_weight <= 0.5:
        raise ValueError("support-approach prior weight must be within [0, 0.5]")
    approaches = poses[:, :3, 2]
    approach_norms = np.linalg.norm(approaches, axis=1)
    if np.any(approach_norms <= 1e-9):
        raise ValueError("support-approach axes must be nonzero")
    approaches = approaches / approach_norms[:, None]
    lift = lift / lift_norm
    if centralities is None:
        center_scores = np.ones(len(poses), dtype=float)
    else:
        center_scores = np.asarray(centralities, dtype=float)
        if (
            center_scores.shape != (len(poses),)
            or not np.all(np.isfinite(center_scores))
            or np.any(center_scores < 0.0)
            or np.any(center_scores > 1.0)
        ):
            raise ValueError(
                "support-approach centralities must align with grasps and be within [0, 1]"
            )
    bonuses = (
        prior_weight
        * np.maximum(0.0, -(approaches @ lift))
        * center_scores
    )
    return values + bonuses, bonuses


def calibrated_scene_uncertainty(
    perception_report: object,
    calibration: object,
    target_points_camera: object,
    *,
    minimum_clearance_m: float = 0.006,
    maximum_clearance_m: float = 0.010,
) -> dict[str, float | int | str]:
    """Derive a bounded scene margin from measured depth and hand-eye residuals."""

    report = perception_report if isinstance(perception_report, dict) else {}
    metadata = calibration if isinstance(calibration, dict) else {}
    temporal = report.get("temporal_depth_filter")
    if not isinstance(temporal, dict):
        raise ValueError("temporal depth uncertainty evidence is missing")
    cloud = np.asarray(target_points_camera, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,) or not len(cloud):
        raise ValueError("target camera cloud is unavailable for uncertainty projection")
    finite_depth = cloud[:, 2][np.isfinite(cloud[:, 2]) & (cloud[:, 2] > 0.0)]
    if not finite_depth.size:
        raise ValueError("target camera cloud has no positive depth")
    try:
        sensor_mad_m = float(temporal["mad_p95_mm"]) * 1e-3
        translation_rmse_m = float(metadata["translation_rmse_m"])
        rotation_rmse_rad = float(metadata["rotation_rmse_rad"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("scene uncertainty evidence is malformed") from error
    values = (sensor_mad_m, translation_rmse_m, rotation_rmse_rad)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("scene uncertainty evidence must be finite and non-negative")
    target_depth_m = float(np.median(finite_depth))
    sensor_three_sigma_m = max(0.002, 3.0 * sensor_mad_m)
    rotational_position_rmse_m = rotation_rmse_rad * target_depth_m
    calibration_position_rmse_m = math.hypot(
        translation_rmse_m,
        rotational_position_rmse_m,
    )
    raw_clearance_m = math.hypot(
        calibration_position_rmse_m,
        sensor_three_sigma_m,
    )
    clearance_m = min(
        maximum_clearance_m,
        max(minimum_clearance_m, raw_clearance_m),
    )
    return {
        "method": "hand_eye_rmse_plus_temporal_depth_3sigma",
        "temporal_frame_count": int(temporal.get("frame_count", 0)),
        "sensor_mad_p95_m": sensor_mad_m,
        "sensor_three_sigma_m": sensor_three_sigma_m,
        "target_depth_m": target_depth_m,
        "translation_rmse_m": translation_rmse_m,
        "rotation_rmse_rad": rotation_rmse_rad,
        "rotational_position_rmse_m": rotational_position_rmse_m,
        "calibration_position_rmse_m": calibration_position_rmse_m,
        "raw_clearance_m": raw_clearance_m,
        "applied_clearance_m": clearance_m,
        "minimum_clearance_m": minimum_clearance_m,
        "maximum_clearance_m": maximum_clearance_m,
    }


def _closest_segment_witness(
    points: object,
    start: object,
    end: object,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the cloud point and capsule point for the closest pair."""

    cloud = np.asarray(points, dtype=float)
    first = np.asarray(start, dtype=float)
    second = np.asarray(end, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,) or not len(cloud):
        raise ValueError("collision witness cloud must be a non-empty (N, 3) array")
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("collision witness capsule endpoints must be three-vectors")
    delta = second - first
    denominator = float(delta @ delta)
    if denominator <= 1e-20:
        nearest_on_segment = np.repeat(first[None, :], len(cloud), axis=0)
    else:
        alpha = np.clip(((cloud - first) @ delta) / denominator, 0.0, 1.0)
        nearest_on_segment = first + alpha[:, None] * delta
    distances = np.linalg.norm(cloud - nearest_on_segment, axis=1)
    index = int(np.argmin(distances))
    return cloud[index], nearest_on_segment[index], float(distances[index])


def _replay_collision_witnesses(
    *,
    planner: object,
    candidates: object,
    failures: tuple[object, ...],
    scene_points: np.ndarray,
    target_points: np.ndarray,
    current_joints: np.ndarray,
    planning_control_type: object,
    maximum: int = COLLISION_WITNESS_REPLAY_LIMIT,
) -> tuple[dict[tuple[int, int], dict[str, object]], list[str]]:
    """Re-run only known collision hypotheses to produce bounded 3-D evidence."""

    from z_manip.collision import (
        PointCloudCollisionChecker,
        PointCloudCollisionConfig,
        SegmentCollisionResult,
        check_target_contact_approach,
    )
    from z_manip.ik.symmetry import expand_symmetry
    from z_manip.planning.grasp_pipeline import grasp_pregrasp_pose, tool_tip_pose

    results: dict[tuple[int, int], dict[str, object]] = {}
    errors: list[str] = []
    relevant = [
        failure
        for failure in failures
        if getattr(failure, "stage", None)
        in {"approach_collision", "lift_collision"}
        and isinstance(getattr(failure, "symmetry_index", None), int)
    ][:maximum]

    def checker(
        allowed: tuple[str, ...],
        *,
        attachment_joints: np.ndarray | None = None,
        collision_model: object | None = None,
    ) -> object:
        model = planner.collision_model if collision_model is None else collision_model
        value = PointCloudCollisionChecker(
            chain=planner.chain,
            model=model,
            frame_provider=planner.chain.link_transforms,
            config=PointCloudCollisionConfig(
                clearance=model.scene_clearance_m,
                point_radius=model.point_radius_m,
                scene_noise_tolerance=model.scene_noise_tolerance_m,
                scene_noise_min_support_points=(
                    model.scene_noise_min_support_points
                ),
                segment_joint_step=planner.config.rrt.collision_resolution,
            ),
            now_fn=lambda: 0.0,
            self_collision_checker=(
                None
                if planner.mesh_self_collision is None
                else planner.mesh_self_collision.check_state
            ),
        )
        value.update_scene(scene_points, stamp_s=0.0)
        if attachment_joints is None:
            value.update_target(target_points, allowed_contact_capsules=allowed)
        else:
            value.update_attached_target(
                target_points,
                attachment_joints=attachment_joints,
                allowed_contact_capsules=allowed,
                allow_initial_scene_contact=True,
                departure_direction_base=(
                    planner.config.grasp_plan.lift_direction_base
                ),
            )
        return value

    for failure in relevant:
        candidate_index = int(failure.candidate_index)
        symmetry_index = int(failure.symmetry_index)
        failure_stage = str(failure.stage)
        key = (candidate_index, symmetry_index)
        try:
            family = expand_symmetry(
                np.asarray(candidates.grasps[candidate_index], dtype=float),
                n_about_axis=planner.config.grasp_plan.symmetry_samples,
            )
            grasp = np.asarray(family[symmetry_index], dtype=float)
            pregrasp = grasp_pregrasp_pose(
                grasp,
                planner.config.grasp_plan.pregrasp_distance_m,
            )
            control = planning_control_type().limited_to(
                planner.config.grasp_plan.hypothesis_timeout_s,
                "collision witness replay",
            )
            pre_solution = planner.ik.solve(
                tool_tip_pose(pregrasp, planner.config.grasp_plan.tool_from_tip),
                current=current_joints,
                control=control,
            )
            poses = []
            for alpha in np.linspace(0.0, 1.0, planner.config.grasp_plan.approach_steps):
                pose = pregrasp.copy()
                pose[:3, 3] = (
                    pregrasp[:3, 3]
                    + alpha * (grasp[:3, 3] - pregrasp[:3, 3])
                )
                poses.append(pose)
            path = [np.asarray(pre_solution.joints, dtype=float)]
            seed = path[0]
            for pose in poses[1:]:
                solution = planner.ik.solve_continuation(
                    tool_tip_pose(pose, planner.config.grasp_plan.tool_from_tip),
                    current=seed,
                    max_joint_step_rad=(
                        planner.config.grasp_plan.max_cartesian_joint_step_rad
                    ),
                    control=control,
                )
                seed = np.asarray(solution.joints, dtype=float)
                path.append(seed)
            path_array = np.asarray(path, dtype=float)
            widths = getattr(candidates, "widths", None)
            required_width_m = (
                None
                if widths is None
                else float(np.asarray(widths, dtype=float)[candidate_index])
            )
            collision_aperture_m = planner.grasp_collision_aperture(
                required_width_m,
            )
            grasp_collision_model = planner._collision_model_for_grasp_width(
                required_width_m,
            )
            allowed = tuple(planner.collision_model.target_contact_capsules)
            replay_reason: str
            if failure_stage == "approach_collision":
                no_contact = checker(())
                open_contact = checker(allowed)
                closed_contact = checker(
                    allowed,
                    collision_model=grasp_collision_model,
                )
                replay = check_target_contact_approach(
                    path_array,
                    no_contact=no_contact,
                    finger_contact=open_contact,
                    allowed_contact_capsules=allowed,
                    control=control,
                )
                if replay.valid:
                    state = closed_contact.check_state(path_array[-1])
                    if state.valid:
                        raise ValueError(
                            "collision replay did not reproduce a structured collision",
                        )
                    segment_index = len(path_array) - 2
                    collision = SegmentCollisionResult(
                        False,
                        state.reason,
                        alpha=1.0,
                        sample_index=len(path_array) - 1,
                        state_result=state,
                    )
                    replay_reason = (
                        "closed-gripper final contact blocked: " + state.reason
                    )
                    geometry_checker = closed_contact
                else:
                    collision = replay.collision
                    state = None if collision is None else collision.state_result
                    if collision is None or state is None:
                        raise ValueError(
                            "collision replay omitted structured approach evidence",
                        )
                    match = re.search(r"approach segment (\d+)", replay.reason)
                    if match is None:
                        raise ValueError(
                            "collision replay omitted the blocked approach segment",
                        )
                    segment_index = int(match.group(1))
                    replay_reason = replay.reason
                    geometry_checker = (
                        open_contact
                        if replay.contact_entry_segment is not None
                        else no_contact
                    )
            else:
                lift = grasp.copy()
                lift[:3, 3] += (
                    planner.config.grasp_plan.lift_distance_m
                    * np.asarray(
                        planner.config.grasp_plan.lift_direction_base,
                        dtype=float,
                    )
                )
                lift_poses = []
                for alpha in np.linspace(
                    0.0,
                    1.0,
                    planner.config.grasp_plan.lift_steps,
                ):
                    pose = grasp.copy()
                    pose[:3, 3] = (
                        grasp[:3, 3]
                        + alpha * (lift[:3, 3] - grasp[:3, 3])
                    )
                    lift_poses.append(pose)
                lift_path = [path_array[-1]]
                seed = lift_path[0]
                for pose in lift_poses[1:]:
                    solution = planner.ik.solve_continuation(
                        tool_tip_pose(
                            pose,
                            planner.config.grasp_plan.tool_from_tip,
                        ),
                        current=seed,
                        max_joint_step_rad=(
                            planner.config.grasp_plan.max_cartesian_joint_step_rad
                        ),
                        control=control,
                    )
                    seed = np.asarray(solution.joints, dtype=float)
                    lift_path.append(seed)
                path_array = np.asarray(lift_path, dtype=float)
                attached = checker(
                    allowed,
                    attachment_joints=path_array[0],
                    collision_model=grasp_collision_model,
                )
                collision = None
                segment_index = -1
                for lift_index, (first, second) in enumerate(
                    zip(path_array, path_array[1:]),
                ):
                    checked = attached.check_segment(
                        first,
                        second,
                        control=control,
                    )
                    if not checked.valid:
                        collision = checked
                        segment_index = lift_index
                        break
                if collision is None or collision.state_result is None:
                    raise ValueError(
                        "lift replay did not reproduce a structured collision",
                    )
                state = collision.state_result
                replay_reason = (
                    f"lift segment {segment_index} blocked: {collision.reason}"
                )
                geometry_checker = attached

            alpha = float(0.0 if collision.alpha is None else collision.alpha)
            sample_joints = (
                path_array[segment_index]
                + alpha * (path_array[segment_index + 1] - path_array[segment_index])
            )
            record: dict[str, object] = {
                "schema": "z_manip.collision_witness.v1",
                "frame": planner.config.robot.base_link,
                "stage": failure_stage,
                "kind": state.kind,
                "capsules": list(state.capsules),
                "path_segment_index": segment_index,
                "sample_index": collision.sample_index,
                "alpha": alpha,
                "distance_m": None if state.distance is None else float(state.distance),
                "threshold_m": (
                    None if state.threshold is None else float(state.threshold)
                ),
                "clearance_margin_m": (
                    None
                    if state.threshold is None or state.distance is None
                    else float(state.distance) - float(state.threshold)
                ),
                "sample_joints_rad": sample_joints.tolist(),
                "grasp_pose_base": grasp.tolist(),
                "pregrasp_pose_base": pregrasp.tolist(),
                "replay_reason": replay_reason,
                "collision_gripper_aperture_m": collision_aperture_m,
            }
            if state.kind in {"scene", "target"} and state.capsules:
                capsules, frame_problem = geometry_checker._world_capsules(
                    sample_joints,
                )
                if frame_problem is not None or capsules is None:
                    raise ValueError(
                        "collision replay could not reconstruct capsule geometry",
                    )
                capsule = next(
                    item for item in capsules if item.spec.name == state.capsules[0]
                )
                witness_cloud = (
                    scene_points if state.kind == "scene" else target_points
                )
                witness_point, witness_capsule, measured_distance = (
                    _closest_segment_witness(
                        witness_cloud,
                        capsule.start,
                        capsule.end,
                    )
                )
                record.update({
                    "distance_m": (
                        measured_distance
                        if state.distance is None
                        else float(state.distance)
                    ),
                    "clearance_margin_m": (
                        None
                        if state.threshold is None
                        else measured_distance - float(state.threshold)
                    ),
                    "capsule_start_base": capsule.start.tolist(),
                    "capsule_end_base": capsule.end.tolist(),
                    "capsule_radius_m": float(capsule.spec.radius),
                    "witness_scene_point_base": witness_point.tolist(),
                    "witness_capsule_point_base": witness_capsule.tolist(),
                    "witness_segment_length_m": measured_distance,
                })
            if state.kind == "attached_target":
                base_t_tip = planner.chain.forward(sample_joints)
                attached_points = (
                    geometry_checker._attached_target_points_tip
                    @ base_t_tip[:3, :3].T
                    + base_t_tip[:3, 3]
                )

                def closest_to_tree(tree: object) -> tuple[float | None, list[float] | None, list[float] | None]:
                    if tree is None:
                        return None, None, None
                    distances, indices = tree.query(attached_points, k=1)
                    attached_index = int(np.argmin(distances))
                    scene_index = int(np.asarray(indices)[attached_index])
                    return (
                        float(np.asarray(distances)[attached_index]),
                        attached_points[attached_index].tolist(),
                        np.asarray(tree.data[scene_index], dtype=float).tolist(),
                    )

                nonsupport = closest_to_tree(
                    geometry_checker._attached_departure_scene_tree,
                )
                support = closest_to_tree(
                    geometry_checker._attached_departure_support_tree,
                )
                axial, lateral = geometry_checker._departure_progress(attached_points)
                record["attached_target_evidence"] = {
                    "payload_points": len(attached_points),
                    "nonsupport_scene_points": (
                        0
                        if geometry_checker._attached_departure_scene_tree is None
                        else len(geometry_checker._attached_departure_scene_tree.data)
                    ),
                    "support_scene_points": (
                        0
                        if geometry_checker._attached_departure_support_tree is None
                        else len(geometry_checker._attached_departure_support_tree.data)
                    ),
                    "nearest_nonsupport_distance_m": nonsupport[0],
                    "nearest_nonsupport_payload_point_base": nonsupport[1],
                    "nearest_nonsupport_scene_point_base": nonsupport[2],
                    "nearest_support_distance_m": support[0],
                    "nearest_support_payload_point_base": support[1],
                    "nearest_support_scene_point_base": support[2],
                    "departure_axial_progress_m": axial,
                    "departure_lateral_progress_m": lateral,
                    "support_contact_exempt": (
                        geometry_checker._departure_support_contact_is_exempt(
                            attached_points,
                            float(state.threshold),
                        )
                        if state.threshold is not None
                        else False
                    ),
                }
            results[key] = record
        except Exception as error:  # best-effort debug evidence cannot affect planning
            errors.append(
                f"#{candidate_index}/{symmetry_index}: "
                f"{type(error).__name__}: {error}"
            )
    return results, errors


def _failure_records(
    failures: tuple[object, ...],
    witnesses: dict[tuple[int, int], dict[str, object]],
) -> list[dict[str, object]]:
    records = []
    for failure in failures:
        record: dict[str, object] = {
            "candidate_index": failure.candidate_index,
            "symmetry_index": failure.symmetry_index,
            "stage": failure.stage,
            "reason": failure.reason,
        }
        key = (failure.candidate_index, failure.symmetry_index)
        if key in witnesses:
            record["collision_witness"] = witnesses[key]
        records.append(record)
    return records


def load_transform(
    path: str | Path,
    *,
    require_calibrated: bool = False,
) -> np.ndarray:
    """Load and validate one homogeneous transform from JSON or NPY.

    Real-camera planning must use a JSON document that explicitly declares
    ``calibrated: true``.  A bare matrix is intentionally insufficient proof:
    it could just as easily be a nominal CAD mount copied from the URDF.
    """

    source = Path(path).expanduser().resolve()
    if source.suffix == ".npy":
        if require_calibrated:
            raise ValueError(
                "real-camera planning requires JSON calibration metadata "
                "with calibrated=true",
            )
        matrix = np.load(source, allow_pickle=False)
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            if require_calibrated and value.get("calibrated") is not True:
                raise ValueError(
                    "real-camera planning requires calibrated=true; nominal "
                    "or unverified camera transforms are rejected",
                )
            value = value.get("base_from_camera")
        elif require_calibrated:
            raise ValueError(
                "real-camera planning requires JSON calibration metadata "
                "with calibrated=true",
            )
        matrix = np.asarray(value, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("base-from-camera transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("base-from-camera transform has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("base-from-camera rotation is not right-handed orthonormal")
    return matrix


def resolve_base_from_camera(
    path: str | Path,
    *,
    real_camera: bool,
    chain: object,
    joints: object,
    source_frame: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resolve a stamped camera pose without accepting nominal eye-in-hand CAD.

    Synthetic artifacts may continue to provide a direct base-from-camera
    transform. Real artifacts must provide explicit calibration metadata. For
    this robot's eye-in-hand D435, FK is evaluated at the exact joint vector
    supplied to the planner and composed with the fixed tip-from-camera result.
    """

    source = Path(path).expanduser().resolve()
    if not real_camera:
        return load_transform(source), {
            "calibrated": False,
            "mount_type": "synthetic",
        }
    if source.suffix == ".npy":
        raise ValueError(
            "real-camera planning requires a JSON camera calibration document",
        )
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("calibrated") is not True:
        raise ValueError(
            "real-camera planning requires calibrated=true; nominal or "
            "unverified camera transforms are rejected",
        )
    if document.get("schema") != "z_manip.piper_camera_calibration.v1":
        raise ValueError("real-camera calibration has an unsupported schema")
    if document.get("synthetic") is not False:
        raise ValueError("synthetic or unspecified calibration provenance is rejected")
    quality = document.get("quality")
    limits = document.get("quality_limits")
    if not isinstance(quality, dict) or not isinstance(limits, dict):
        raise ValueError("real-camera calibration quality evidence is missing")
    try:
        sample_count = int(document["sample_count"])
        min_samples = int(limits["min_samples"])
        axis_rank = int(quality["rotation_axis_rank"])
        min_axis_rank = int(limits["min_rotation_axis_rank"])
        rotation_span = float(quality["max_pair_rotation_rad"])
        min_rotation_span = float(limits["min_rotation_span_rad"])
        translation_rmse = float(quality["translation_rmse_m"])
        max_translation_rmse = float(limits["max_translation_rmse_m"])
        rotation_rmse = float(quality["rotation_rmse_rad"])
        max_rotation_rmse = float(limits["max_rotation_rmse_rad"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("real-camera calibration quality evidence is invalid") from error
    metrics = (
        rotation_span,
        min_rotation_span,
        translation_rmse,
        max_translation_rmse,
        rotation_rmse,
        max_rotation_rmse,
    )
    if (
        not all(math.isfinite(value) for value in metrics)
        or sample_count < min_samples
        or axis_rank < min_axis_rank
        or rotation_span < min_rotation_span
        or translation_rmse > max_translation_rmse
        or rotation_rmse > max_rotation_rmse
    ):
        raise ValueError("real-camera calibration does not satisfy its quality gates")
    calibration_id = str(document.get("calibration_id", "")).strip()
    if not calibration_id:
        raise ValueError("real-camera calibration requires a non-empty calibration_id")
    camera_frame = str(document.get("camera_frame", "")).strip()
    if camera_frame != source_frame:
        raise ValueError(
            f"camera calibration frame {camera_frame!r} does not match "
            f"artifact frame {source_frame!r}",
        )
    mount_type = str(document.get("mount_type", "")).strip()
    if mount_type == "eye_in_hand":
        tip_link = str(document.get("tip_link", "")).strip()
        chain_tip = str(getattr(chain, "tip_link", ""))
        if tip_link != chain_tip:
            raise ValueError(
                f"camera calibration tip {tip_link!r} does not match "
                f"kinematic-chain tip {chain_tip!r}",
            )
        tip_from_camera = np.asarray(document.get("tip_from_camera"), dtype=float)
        if tip_from_camera.shape != (4, 4):
            raise ValueError("tip_from_camera must be a 4x4 transform")
        matrix = tip_from_camera
        if not np.all(np.isfinite(matrix)):
            raise ValueError("tip_from_camera contains a non-finite value")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError("tip_from_camera has an invalid homogeneous row")
        rotation = matrix[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ):
            raise ValueError("tip_from_camera rotation is not right-handed orthonormal")
        base_from_tip = np.asarray(chain.forward(np.asarray(joints, dtype=float)))
        base_from_camera = base_from_tip @ matrix
    elif mount_type == "fixed_to_base":
        base_from_camera = np.asarray(document.get("base_from_camera"), dtype=float)
        if base_from_camera.shape != (4, 4):
            raise ValueError("base_from_camera must be a 4x4 transform")
        rotation = base_from_camera[:3, :3]
        if (
            not np.all(np.isfinite(base_from_camera))
            or not np.allclose(base_from_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        ):
            raise ValueError("base_from_camera is not a valid rigid transform")
    else:
        raise ValueError(
            "camera calibration mount_type must be eye_in_hand or fixed_to_base",
        )
    metadata = {
        "calibrated": True,
        "calibration_id": calibration_id,
        "mount_type": mount_type,
        "camera_frame": camera_frame,
        "sample_count": sample_count,
        "translation_rmse_m": translation_rmse,
        "rotation_rmse_rad": rotation_rmse,
    }
    return np.asarray(base_from_camera, dtype=float), metadata


def transform_points(points: object, target_from_source: object) -> np.ndarray:
    cloud = np.asarray(points, dtype=float)
    transform = np.asarray(target_from_source, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,) or not np.all(np.isfinite(cloud)):
        raise ValueError("point cloud must be a finite (N, 3) array")
    if transform.shape != (4, 4):
        raise ValueError("point transform must be 4x4")
    return np.ascontiguousarray(
        cloud @ transform[:3, :3].T + transform[:3, 3],
        dtype=np.float64,
    )


def transform_poses(poses: object, target_from_source: object) -> np.ndarray:
    values = np.asarray(poses, dtype=float)
    transform = np.asarray(target_from_source, dtype=float)
    if (
        values.ndim != 3
        or values.shape[1:] != (4, 4)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("grasp poses must be a finite (N, 4, 4) array")
    if transform.shape != (4, 4):
        raise ValueError("pose transform must be 4x4")
    return np.matmul(transform[None, :, :], values)


def rigid_pose_error(actual: object, target: object) -> tuple[float, float]:
    """Return translation and geodesic rotation error for two rigid poses."""

    measured = np.asarray(actual, dtype=float)
    desired = np.asarray(target, dtype=float)
    if measured.shape != (4, 4) or desired.shape != (4, 4):
        raise ValueError("pose-error inputs must be 4x4 transforms")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(desired)):
        raise ValueError("pose-error inputs must be finite")
    position_m = float(np.linalg.norm(measured[:3, 3] - desired[:3, 3]))
    relative = desired[:3, :3] @ measured[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return position_m, float(np.arccos(cosine))


def _joint_vector(value: str) -> np.ndarray:
    try:
        joints = np.fromstring(value, sep=",", dtype=float)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if joints.shape != (6,) or not np.all(np.isfinite(joints)):
        raise argparse.ArgumentTypeError(
            "--joints must contain exactly six finite comma-separated radians",
        )
    return joints


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _support_approach_prior_weight(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not math.isfinite(result) or not 0.0 <= result <= 0.5:
        raise argparse.ArgumentTypeError(
            "support-approach prior weight must be within [0, 0.5]",
        )
    return result


def _supervised_scene_clearance(value: str) -> float:
    result = _positive_float(value)
    if not 0.001 <= result <= 0.010:
        raise argparse.ArgumentTypeError(
            "supervised scene clearance must be within [0.001, 0.010] m",
        )
    return result


def _supervised_scene_point_radius(value: str) -> float:
    result = _positive_float(value)
    if not 0.001 <= result <= 0.005:
        raise argparse.ArgumentTypeError(
            "supervised scene point radius must be within [0.001, 0.005] m",
        )
    return result


def _gripper_scene_radius_scale(value: str) -> float:
    result = _positive_float(value)
    if not 0.5 <= result <= 1.0:
        raise argparse.ArgumentTypeError(
            "gripper scene radius scale must be within [0.5, 1.0]",
        )
    return result


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--joints", type=_joint_vector, required=True)
    parser.add_argument(
        "--planning-joints",
        type=_joint_vector,
        help=(
            "optional in-limit planning start; --joints remains the exact "
            "measured state used for eye-in-hand camera geometry"
        ),
    )
    parser.add_argument(
        "--camera-calibration",
        "--base-from-camera",
        dest="camera_calibration",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--search-timeout-s",
        type=_positive_float,
        help=(
            "planning-only grasp-search budget override; does not modify the "
            "runtime execution configuration"
        ),
    )
    parser.add_argument(
        "--symmetry-samples",
        type=_positive_int,
        help="planning-only approach-axis symmetry count",
    )
    parser.add_argument(
        "--max-hypotheses",
        type=_positive_int,
        help="planning-only grasp-hypothesis count",
    )
    parser.add_argument(
        "--max-feasible-plans",
        type=_positive_int,
        help="planning-only number of complete plans retained before returning",
    )
    parser.add_argument(
        "--support-approach-prior-weight",
        type=_support_approach_prior_weight,
        default=0.0,
        help=(
            "soft planning-only score bonus for approaches opposite the lift "
            "direction; zero disables the supported-object prior"
        ),
    )
    parser.add_argument(
        "--scene-clearance-m",
        type=_supervised_scene_clearance,
        help=(
            "fixed supervised planning-only scene margin; retains capsule and "
            "point radii and never changes the runtime execution configuration"
        ),
    )
    parser.add_argument(
        "--scene-point-radius-m",
        type=_supervised_scene_point_radius,
        help="fixed supervised planning-only radius assigned to each scene point",
    )
    parser.add_argument(
        "--scene-noise-tolerance-m",
        type=_supervised_scene_point_radius,
        default=0.003,
        help=(
            "planning-only D435 boundary-noise tolerance; a shallower scene "
            "penetration needs multiple supporting depth samples"
        ),
    )
    parser.add_argument(
        "--scene-noise-min-support-points",
        type=_positive_int,
        default=2,
        help="planning-only scene samples required inside the boundary-noise band",
    )
    parser.add_argument(
        "--gripper-scene-radius-scale",
        type=_gripper_scene_radius_scale,
        default=1.0,
        help=(
            "planning-only scale for palm/finger capsules against noisy scene points; "
            "URDF mesh self-collision remains unchanged"
        ),
    )
    return parser.parse_args(argv)


def _file_identity(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


def _load_planner(args: argparse.Namespace) -> tuple[object, object, bool]:
    """Return a process-local immutable planner template and cache evidence."""

    from z_manip.configuration import load_stack_config
    from z_manip_task.planning import OnlinePlanner

    environment = dict(os.environ)
    environment["Z_MANIP_ROBOT_URDF"] = str(args.urdf.expanduser().resolve())
    key = (
        _file_identity(args.config),
        _file_identity(args.urdf),
        os.environ.get("Z_MANIP_IK_BACKEND", "pinocchio"),
        args.search_timeout_s,
        args.symmetry_samples,
        args.max_hypotheses,
        args.max_feasible_plans,
    )
    cached = _PLANNER_CACHE.get(key)
    if cached is None:
        config = load_stack_config(args.config, environ=environment)
        if any(
            value is not None
            for value in (
                args.search_timeout_s,
                args.symmetry_samples,
                args.max_hypotheses,
                args.max_feasible_plans,
            )
        ):
            config = replace(
                config,
                grasp_plan=replace(
                    config.grasp_plan,
                    search_timeout_s=(
                        config.grasp_plan.search_timeout_s
                        if args.search_timeout_s is None
                        else args.search_timeout_s
                    ),
                    symmetry_samples=(
                        config.grasp_plan.symmetry_samples
                        if args.symmetry_samples is None
                        else args.symmetry_samples
                    ),
                    max_hypotheses=(
                        config.grasp_plan.max_hypotheses
                        if args.max_hypotheses is None
                        else args.max_hypotheses
                    ),
                    max_feasible_plans=(
                        config.grasp_plan.max_feasible_plans
                        if args.max_feasible_plans is None
                        else args.max_feasible_plans
                    ),
                ),
            )
        planner = OnlinePlanner(config)
        base_collision_model = planner.collision_model
        _PLANNER_CACHE.clear()
        _PLANNER_CACHE[key] = (config, planner, base_collision_model)
        return config, planner, False

    config, planner, base_collision_model = cached
    # The dry-run applies calibrated scene/gripper margins to this dataclass.
    # Restore the exact immutable template before every request.
    planner.collision_model = base_collision_model
    return config, planner, True


def main(argv: Sequence[str] | None = None) -> int:
    total_started = time.perf_counter()
    args = _arguments(argv)
    artifacts = args.artifacts.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = np.load(artifacts / "grasp_candidates.npz", allow_pickle=False)
    source_frame = str(archive["frame"].item())
    calibration_required = not source_frame.lower().startswith("synthetic")
    measured_joints = np.asarray(args.joints, dtype=float)
    planning_joints = (
        measured_joints.copy()
        if args.planning_joints is None
        else np.asarray(args.planning_joints, dtype=float)
    )

    from z_manip.models.grasp_source import GraspCandidates
    from z_manip.models.planner import PlanningError
    from z_manip.planning_control import PlanningControl
    config, planner, planner_cache_hit = _load_planner(args)
    if args.gripper_scene_radius_scale < 1.0:
        planner.collision_model = replace(
            planner.collision_model,
            capsules=tuple(
                replace(capsule, radius=capsule.radius * args.gripper_scene_radius_scale)
                if capsule.name == "palm" or capsule.name.startswith("finger_")
                else capsule
                for capsule in planner.collision_model.capsules
            ),
        )
    base_from_camera, calibration_metadata = resolve_base_from_camera(
        args.camera_calibration,
        real_camera=calibration_required,
        chain=planner.chain,
        joints=measured_joints,
        source_frame=source_frame,
    )
    target_points_camera = np.load(
        artifacts / "target_points.npy",
        allow_pickle=False,
    )
    if calibration_required:
        perception_report = json.loads(
            (artifacts / "report.json").read_text(encoding="utf-8"),
        )
        uncertainty = calibrated_scene_uncertainty(
            perception_report,
            calibration_metadata,
            target_points_camera,
            minimum_clearance_m=(
                0.006
                if args.scene_clearance_m is None
                else args.scene_clearance_m
            ),
            maximum_clearance_m=(
                0.010
                if args.scene_clearance_m is None
                else args.scene_clearance_m
            ),
        )
        uncertainty["planning_profile"] = (
            "calibrated_default"
            if args.scene_clearance_m is None
            else "supervised_fixed_clearance"
        )
        planner.collision_model = replace(
            planner.collision_model,
            scene_clearance_m=float(uncertainty["applied_clearance_m"]),
            point_radius_m=(
                planner.collision_model.point_radius_m
                if args.scene_point_radius_m is None
                else args.scene_point_radius_m
            ),
            scene_noise_tolerance_m=args.scene_noise_tolerance_m,
            scene_noise_min_support_points=args.scene_noise_min_support_points,
        )
    else:
        uncertainty = {
            "method": "synthetic_configured_margin",
            "applied_clearance_m": planner.collision_model.scene_clearance_m,
        }

    grasps_base = transform_poses(archive["grasps"], base_from_camera)
    centroid_base = transform_points(
        np.asarray(archive["centroid"], dtype=float).reshape(1, 3),
        base_from_camera,
    )[0]
    widths = np.asarray(archive["widths"], dtype=float)
    if widths.size == 0:
        widths = None
    target_base = transform_points(
        target_points_camera,
        base_from_camera,
    )
    grasp_centralities = grasp_centrality_scores(
        grasps_base,
        centroid_base,
        target_base,
    )
    raw_scores = np.asarray(archive["scores"], dtype=float)
    ranked_scores, support_approach_bonuses = support_approach_scores(
        grasps_base,
        raw_scores,
        config.grasp_plan.lift_direction_base,
        weight=args.support_approach_prior_weight,
        centralities=grasp_centralities,
    )
    candidates = GraspCandidates(
        grasps=grasps_base,
        scores=ranked_scores,
        centroid=centroid_base,
        frame=config.robot.base_link,
        num_raw=int(archive["num_raw"].item()),
        widths=widths,
    )
    scene_base = transform_points(
        np.load(artifacts / "scene_collision_points.npy", allow_pickle=False),
        base_from_camera,
    )
    stamp_s = 0.0
    timings_s: dict[str, float] = {
        "setup": time.perf_counter() - total_started,
    }
    report: dict[str, object] = {
        "read_only": True,
        "planning_only": True,
        "motion_commands_published": 0,
        "source_frame": source_frame,
        "planning_frame": config.robot.base_link,
        "calibration_metadata_required": calibration_required,
        "camera_calibration": calibration_metadata,
        "scene_uncertainty": uncertainty,
        "scene_clearance_m": planner.collision_model.scene_clearance_m,
        "scene_point_radius_m": planner.collision_model.point_radius_m,
        "scene_noise_tolerance_m": (
            planner.collision_model.scene_noise_tolerance_m
        ),
        "scene_noise_min_support_points": (
            planner.collision_model.scene_noise_min_support_points
        ),
        "gripper_scene_radius_scale": args.gripper_scene_radius_scale,
        "source_stamp_ns": int(archive["stamp_ns"].item()),
        "current_joints_rad": measured_joints.tolist(),
        "measured_joints_rad": measured_joints.tolist(),
        "planning_start_joints_rad": planning_joints.tolist(),
        "start_limit_projection_rad": (
            planning_joints - measured_joints
        ).tolist(),
        "execution_start_requires_limit_reconciliation": bool(
            np.any(planning_joints != measured_joints)
        ),
        "candidate_count": len(grasps_base),
        "ik_backend": getattr(planner, "ik_backend", "unknown"),
        "planner_model_cache_hit": planner_cache_hit,
        "scene_points": len(scene_base),
        "target_points": len(target_base),
        "base_from_camera": base_from_camera.tolist(),
        "plan_valid": False,
        "planning_search_timeout_s": config.grasp_plan.search_timeout_s,
        "planning_symmetry_samples": config.grasp_plan.symmetry_samples,
        "planning_max_hypotheses": config.grasp_plan.max_hypotheses,
        "planning_max_feasible_plans": config.grasp_plan.max_feasible_plans,
        "support_approach_prior_weight": args.support_approach_prior_weight,
        "lateral_approach_prior_weight": (
            config.grasp_plan.lateral_approach_prior_weight
        ),
        "overhead_approach_penalty_weight": (
            config.grasp_plan.overhead_approach_penalty_weight
        ),
        "grasp_centrality_method": "robust_object_frame_gaussian",
        "planning_pregrasp_distance_m": config.grasp_plan.pregrasp_distance_m,
        "planning_approach_steps": config.grasp_plan.approach_steps,
        "collision_witness_replay_limit": COLLISION_WITNESS_REPLAY_LIMIT,
        "timings_s": timings_s,
    }
    search_started = time.perf_counter()
    try:
        request_control = PlanningControl().limited_to(
            config.grasp_plan.search_timeout_s
            + config.grasp_plan.hypothesis_timeout_s
            + 1.0,
            "planning-only request",
        )
        planned = planner._plan(
            candidates,
            scene_points=scene_base,
            target_points=target_base,
            current_joints=planning_joints,
            stamp_s=stamp_s,
            control=request_control,
        )
        timings_s["search"] = time.perf_counter() - search_started
        retime_started = time.perf_counter()
        raw_transit = np.asarray(planned.transit.waypoints, dtype=float)
        raw_approach = np.vstack((raw_transit[-1], planned.approach_joints))
        raw_lift = np.vstack((raw_approach[-1], planned.lift_joints))
        from z_manip.planning.grasp_pipeline import tool_tip_pose

        contact_target = tool_tip_pose(
            planned.grasp_pose,
            config.grasp_plan.tool_from_tip,
        )
        contact_position_error_m, contact_orientation_error_rad = rigid_pose_error(
            planner.chain.forward(raw_approach[-1]),
            contact_target,
        )
        if planned.lift_pose is None:
            lift_target = np.asarray(planned.grasp_pose, dtype=float).copy()
            lift_target[:3, 3] += (
                config.grasp_plan.lift_distance_m
                * np.asarray(config.grasp_plan.lift_direction_base, dtype=float)
            )
        else:
            lift_target = np.asarray(planned.lift_pose, dtype=float).copy()
        lift_position_error_m, lift_orientation_error_rad = rigid_pose_error(
            planner.chain.forward(raw_lift[-1]),
            tool_tip_pose(lift_target, config.grasp_plan.tool_from_tip),
        )
        transit = planner._retime_joint_path(
            raw_transit,
            allow_stationary_hold=True,
        )
        approach = planner._retime_joint_path(raw_approach)
        lift = planner._retime_joint_path(raw_lift)
        timings_s["retime"] = time.perf_counter() - retime_started
    except (PlanningError, ValueError) as error:
        timings_s.setdefault("search", time.perf_counter() - search_started)
        report["error"] = f"{type(error).__name__}: {error}"
        failures = tuple(getattr(error, "failures", ()))
        replay_started = time.perf_counter()
        witnesses, witness_errors = _replay_collision_witnesses(
            planner=planner,
            candidates=candidates,
            failures=failures,
            scene_points=scene_base,
            target_points=target_base,
            current_joints=planning_joints,
            planning_control_type=PlanningControl,
        )
        timings_s["diagnostic_replay"] = time.perf_counter() - replay_started
        report["collision_witness_count"] = len(witnesses)
        report["collision_witness_replay_errors"] = witness_errors
        report["rejection_count"] = len(failures)
        report["rejections_truncated"] = False
        report["rejections"] = _failure_records(failures, witnesses)
        timings_s["total"] = time.perf_counter() - total_started
        (output / "planning_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))
        return 1

    artifact_started = time.perf_counter()
    planned_grasp_path = output / "planned_grasp.npz"
    np.savez_compressed(
        planned_grasp_path,
        grasp_pose=np.asarray(planned.grasp_pose, dtype=float),
        pregrasp_pose=np.asarray(planned.pregrasp_pose, dtype=float),
        transit_raw=raw_transit,
        approach_raw=raw_approach,
        lift_raw=raw_lift,
        transit=np.asarray(transit.positions, dtype=float),
        transit_times_s=np.asarray(transit.times_s, dtype=float),
        approach=np.asarray(approach.positions, dtype=float),
        approach_times_s=np.asarray(approach.times_s, dtype=float),
        lift=np.asarray(lift.positions, dtype=float),
        lift_times_s=np.asarray(lift.times_s, dtype=float),
        current_joints=planning_joints,
        measured_joints=measured_joints,
    )
    planned_grasp_sha256 = hashlib.sha256(
        planned_grasp_path.read_bytes(),
    ).hexdigest()
    timings_s["artifact_write"] = time.perf_counter() - artifact_started
    report.update({
        "plan_valid": True,
        "raw_paths_collision_validated": True,
        "planned_grasp_sha256": planned_grasp_sha256,
        "candidate_index": int(planned.candidate_index),
        "symmetry_index": int(planned.symmetry_index),
        "selected_global_rank": int(planned.selected_global_rank),
        "higher_rank_rejection_count": int(
            planned.higher_rank_rejection_count
        ),
        "score": float(planned.score),
        "raw_candidate_score": float(raw_scores[planned.candidate_index]),
        "support_approach_bonus": float(
            support_approach_bonuses[planned.candidate_index]
        ),
        "lateral_approach_bonus": float(planned.lateral_approach_bonus),
        "overhead_approach_penalty": float(
            planned.overhead_approach_penalty
        ),
        "approach_preference": "either_body_side_before_overhead",
        "grasp_centrality": float(grasp_centralities[planned.candidate_index]),
        "final_contact_fk_error": {
            "position_m": contact_position_error_m,
            "orientation_rad": contact_orientation_error_rad,
            "position_limit_m": config.ik.position_tolerance_m,
            "orientation_limit_rad": config.ik.orientation_tolerance_rad,
        },
        "final_lift_fk_error": {
            "position_m": lift_position_error_m,
            "orientation_rad": lift_orientation_error_rad,
            "position_limit_m": config.ik.position_tolerance_m,
            "orientation_limit_rad": config.ik.orientation_tolerance_rad,
        },
        "required_width_m": (
            None
            if planned.required_width_m is None
            else float(planned.required_width_m)
        ),
        "collision_gripper_aperture_m": planner.grasp_collision_aperture(
            planned.required_width_m,
        ),
        "transit_raw_waypoints": len(raw_transit),
        "approach_raw_waypoints": len(raw_approach),
        "lift_raw_waypoints": len(raw_lift),
        "transit_waypoints": len(transit.positions),
        "approach_waypoints": len(approach.positions),
        "lift_waypoints": len(lift.positions),
        "transit_duration_s": float(transit.times_s[-1]),
        "approach_duration_s": float(approach.times_s[-1]),
        "lift_duration_s": float(lift.times_s[-1]),
        "grasp_pose": np.asarray(planned.grasp_pose, dtype=float).tolist(),
        "pregrasp_pose": np.asarray(planned.pregrasp_pose, dtype=float).tolist(),
        "lift_pose": np.asarray(lift_target, dtype=float).tolist(),
        "trajectory_refinement": (
            None
            if planned.trajectory_refinement is None
            else planned.trajectory_refinement.document()
        ),
        "rejection_count": len(planned.failures),
        "rejections_truncated": False,
        "rejections": _failure_records(planned.failures, {}),
    })
    timings_s["total"] = time.perf_counter() - total_started
    (output / "planning_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
