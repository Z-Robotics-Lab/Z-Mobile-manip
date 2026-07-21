import math

import numpy as np
import pytest

from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.planner import PlanningError
from z_manip.planning.work_pose import (
    BoundedSE2WorkPoseOptimizer,
    WorkPoseConfig,
    WorkPoseFailureCode,
    WorkPoseObservation,
    WorkPoseOptimizationError,
)


def _observation(*, mount=None):
    target = np.eye(4)
    target[:3, 3] = (1.50, -0.68, 0.05)
    side_grasp = np.eye(4)
    side_grasp[:3, 3] = (1.48, -0.69, 0.08)
    angled_grasp = np.eye(4)
    angled_grasp[:3, :3] = np.array((
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ))
    angled_grasp[:3, 3] = (1.52, -0.67, 0.04)
    candidates = GraspCandidates(
        grasps=np.stack((side_grasp, angled_grasp)),
        scores=np.array((0.8, 0.7)),
        centroid=target[:3, 3].copy(),
        frame="piper_base_link",
        num_raw=7,
        widths=np.array((0.03, 0.04)),
    )
    return WorkPoseObservation(
        target_pose=target,
        candidates=candidates,
        scene_points=np.array((
            (1.30, -0.80, -0.15, 0.1),
            (1.60, -0.50, 0.20, 0.8),
        )),
        current_joints=np.linspace(-0.2, 0.3, 6),
        T_platform_piper=np.eye(4) if mount is None else mount,
    )


def test_off_axis_target_requires_lateral_motion_and_yaw_to_enter_arm_corridor():
    observation = _observation()
    choice = BoundedSE2WorkPoseOptimizer().select(observation)

    assert abs(choice.relative_base_pose[1]) > 0.20
    assert abs(choice.relative_base_pose[2]) > math.radians(5.0)
    predicted_target = choice.predicted_target_pose[:3, 3]
    assert 0.30 <= predicted_target[0] <= 0.75
    assert -0.32 <= predicted_target[1] <= 0.32
    assert abs(predicted_target[1]) < 1e-7

    expected_grasps = np.einsum(
        "ij,njk->nik",
        choice.T_new_piper_current_piper,
        observation.candidates.grasps,
    )
    assert choice.candidates.grasps == pytest.approx(expected_grasps)
    expected_scene_xyz = (
        observation.scene_points[:, :3]
        @ choice.T_new_piper_current_piper[:3, :3].T
        + choice.T_new_piper_current_piper[:3, 3]
    )
    assert choice.scene_points[:, :3] == pytest.approx(expected_scene_xyz)
    assert choice.scene_points[:, 3] == pytest.approx(observation.scene_points[:, 3])


def test_mount_is_conjugated_for_complete_target_grasp_and_scene_transforms():
    mount = np.eye(4)
    mount_yaw = math.radians(18.0)
    mount[:3, :3] = np.array((
        (math.cos(mount_yaw), -math.sin(mount_yaw), 0.0),
        (math.sin(mount_yaw), math.cos(mount_yaw), 0.0),
        (0.0, 0.0, 1.0),
    ))
    mount[:3, 3] = (0.12, -0.09, 0.41)
    observation = _observation(mount=mount)

    choice = BoundedSE2WorkPoseOptimizer().select(observation)

    base_pose = choice.relative_base_pose
    cosine = math.cos(float(base_pose[2]))
    sine = math.sin(float(base_pose[2]))
    T_current_platform_new_platform = np.array((
        (cosine, -sine, 0.0, base_pose[0]),
        (sine, cosine, 0.0, base_pose[1]),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    expected = np.linalg.inv(T_current_platform_new_platform @ mount) @ mount
    assert choice.T_new_piper_current_piper == pytest.approx(expected)
    assert choice.predicted_target_pose == pytest.approx(expected @ observation.target_pose)


def test_exact_evaluator_can_reject_cheap_pose_and_select_reachable_side_pose():
    evaluated = []

    def evaluate(candidate, *, control=None):
        del control
        pose = candidate.relative_base_pose.copy()
        evaluated.append(pose)
        if pose[1] > -0.70:
            raise PlanningError("IK requires the farther lateral work pose")
        return {"score": 2.0 - abs(float(pose[2]))}

    choice = BoundedSE2WorkPoseOptimizer().select(
        _observation(),
        evaluate=evaluate,
    )

    assert len(evaluated) > 1
    assert choice.relative_base_pose[1] <= -0.70
    assert choice.exact_evaluation is not None
    assert choice.diagnostics.rejection_count(WorkPoseFailureCode.EXACT_REJECTED) > 0
    assert choice.diagnostics.exact_evaluations <= WorkPoseConfig().max_exact_evaluations


def test_history_rejects_repeated_pose_and_selects_a_distinct_alternative():
    optimizer = BoundedSE2WorkPoseOptimizer()
    first = optimizer.select(_observation())

    second = optimizer.select(
        _observation(),
        history_relative_base_poses=(first.relative_base_pose,),
    )

    assert not np.allclose(second.relative_base_pose, first.relative_base_pose)
    assert second.diagnostics.rejection_count(WorkPoseFailureCode.HISTORY_DUPLICATE) == 1


def test_history_only_sample_fails_with_typed_diagnostic():
    config = WorkPoseConfig(
        radial_distances_m=(0.52,),
        target_lateral_offsets_m=(0.0,),
        yaw_offsets_rad=(0.0,),
        max_sampled_hypotheses=1,
        max_ranked_candidates=1,
        max_exact_evaluations=1,
        max_feasible_choices=1,
    )
    optimizer = BoundedSE2WorkPoseOptimizer(config)
    first = optimizer.select(_observation())

    with pytest.raises(WorkPoseOptimizationError) as raised:
        optimizer.select(
            _observation(),
            history_relative_base_poses=(first.relative_base_pose,),
        )

    diagnostics = raised.value.diagnostics
    assert diagnostics.rejection_count(WorkPoseFailureCode.HISTORY_DUPLICATE) == 1
    assert diagnostics.geometric_candidates == 0


def test_pure_forward_pose_that_retains_large_lateral_error_is_rejected():
    target_bearing = math.atan2(-0.68, 1.50)
    config = WorkPoseConfig(
        radial_distances_m=(0.52,),
        target_lateral_offsets_m=(-0.68,),
        yaw_offsets_rad=(-target_bearing,),
        max_sampled_hypotheses=1,
        max_ranked_candidates=1,
        max_exact_evaluations=1,
        max_feasible_choices=1,
    )

    with pytest.raises(WorkPoseOptimizationError) as raised:
        BoundedSE2WorkPoseOptimizer(config).select(_observation())

    diagnostic = raised.value.diagnostics
    assert diagnostic.rejection_count(WorkPoseFailureCode.OUTSIDE_MANIP_CORRIDOR) == 1
    failure_pose = diagnostic.failures[0].relative_base_pose
    assert failure_pose is not None
    assert failure_pose[0] > 0.9
    assert abs(failure_pose[1]) < 1e-8
    assert abs(failure_pose[2]) < 1e-8


def test_invalid_exact_result_is_a_typed_candidate_failure():
    config = WorkPoseConfig(
        radial_distances_m=(0.52,),
        target_lateral_offsets_m=(0.0,),
        yaw_offsets_rad=(0.0,),
        max_sampled_hypotheses=1,
        max_ranked_candidates=1,
        max_exact_evaluations=1,
        max_feasible_choices=1,
    )

    with pytest.raises(WorkPoseOptimizationError) as raised:
        BoundedSE2WorkPoseOptimizer(config).select(
            _observation(),
            evaluate=lambda _candidate: {"score": "not-a-number"},
        )

    assert raised.value.diagnostics.rejection_count(
        WorkPoseFailureCode.INVALID_EXACT_RESULT,
    ) == 1


def test_candidate_and_exact_budgets_are_hard_bounded():
    evaluated = []
    config = WorkPoseConfig(
        max_sampled_hypotheses=6,
        max_ranked_candidates=4,
        max_exact_evaluations=3,
        max_feasible_choices=1,
    )

    def reject(candidate):
        evaluated.append(candidate.relative_base_pose.copy())
        raise PlanningError("unreachable")

    with pytest.raises(WorkPoseOptimizationError) as raised:
        BoundedSE2WorkPoseOptimizer(config).select(
            _observation(),
            evaluate=reject,
        )

    diagnostics = raised.value.diagnostics
    assert diagnostics.sampled_hypotheses == 6
    assert diagnostics.sample_budget_exhausted
    assert diagnostics.exact_evaluations == 3
    assert diagnostics.exact_budget_exhausted
    assert len(evaluated) == 3
