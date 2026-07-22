import numpy as np

from z_manip.kinematics.robust_ik import IKSolution
from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.planner import JointTrajectory
from z_manip.planning.staged_trajectory import (
    GraspStage,
    GraspTrajectoryTarget,
    SidePreference,
    StagedGraspRequest,
    StagedGraspTrajectoryBuilder,
)
from z_manip.planning.time_parameterization import TimeParameterizationConfig


class _PoseEncodingIK:
    def solve(self, target, current=None):
        xyz = np.asarray(target)[:3, 3]
        joints = np.array((xyz[0], xyz[1], xyz[2], xyz.sum(), xyz[2] - xyz[0], 0.1))
        return IKSolution(joints, 0.0, 0.0, 1.0, 1, 0)


class _MidpointPlanner:
    def plan_joint(self, start_joints, goal_joints, *, timeout_s=5.0):
        start = np.asarray(start_joints)
        goal = np.asarray(goal_joints)
        return JointTrajectory(
            joint_names=tuple(f"j{index}" for index in range(6)),
            waypoints=np.vstack((start, 0.5 * (start + goal), goal)),
        )


def _target():
    pose = np.eye(4)
    pose[:3, 3] = (0.52, 0.02, 0.14)
    return GraspTrajectoryTarget(pose, candidate_index=7, score=0.91, required_width_m=0.04)


def _builder(planner=None):
    return StagedGraspTrajectoryBuilder(
        _PoseEncodingIK(),
        velocity_limits=np.full(6, 0.8),
        acceleration_limits=np.full(6, 1.4),
        joint_planner=planner,
        time_config=TimeParameterizationConfig(
            sample_period_s=0.01,
            min_segment_time_s=0.05,
            velocity_scale=0.7,
            acceleration_scale=0.6,
        ),
    )


def test_builds_continuous_speed_limited_four_stage_contract():
    builder = _builder(_MidpointPlanner())
    plan = builder.build(
        StagedGraspRequest(
            current_joints=np.zeros(6),
            target=_target(),
            pregrasp_offset_m=0.08,
            approach_clearance_m=0.05,
            lift_distance_m=0.09,
        ),
        plan_id="initial",
    )

    assert tuple(segment.stage for segment in plan.segments) == tuple(GraspStage)
    assert plan.schema == "z_manip.staged_grasp_trajectory.v1"
    for previous, following in zip(plan.segments, plan.segments[1:]):
        assert np.allclose(previous.goal_joints, following.start_joints)
    flattened = plan.flattened()
    assert np.all(np.diff(flattened.times_s) > 0.0)
    measured_velocity = np.diff(flattened.positions, axis=0) / np.diff(flattened.times_s)[:, None]
    assert np.max(np.abs(measured_velocity)) <= 0.8 * 0.7 * 1.01
    legacy = plan.as_joint_trajectory(tuple(f"piper_joint{i}" for i in range(1, 7)))
    assert np.allclose(legacy.waypoints, flattened.positions)
    assert np.allclose(legacy.times, flattened.times_s)


def test_side_entry_bias_tapers_to_exact_grasp_pose():
    plan = _builder().build(
        StagedGraspRequest(
            current_joints=np.zeros(6),
            target=_target(),
            side_preference=SidePreference.LEFT,
            side_entry_offset_m=0.04,
        ),
    )

    grasp_y = _target().grasp_pose[1, 3]
    assert np.isclose(plan.segment(GraspStage.APPROACH).target_pose[1, 3], grasp_y + 0.04)
    assert np.isclose(plan.segment(GraspStage.PREGRASP).target_pose[1, 3], grasp_y + 0.02)
    assert np.isclose(plan.segment(GraspStage.GRASP).target_pose[1, 3], grasp_y)


def test_rolling_replan_starts_at_fresh_measurement_and_replaces_only_suffix():
    builder = _builder()
    initial = builder.build(
        StagedGraspRequest(
            np.zeros(6),
            _target(),
            pregrasp_offset_m=0.075,
            approach_clearance_m=0.045,
        ),
        plan_id="first-plan",
    )
    measured = initial.segment(GraspStage.PREGRASP).goal_joints + 0.015
    updated_pose = np.array(_target().grasp_pose, copy=True)
    updated_pose[0, 3] += 0.012

    replanned = builder.replan_remaining(
        initial,
        measured,
        from_stage=GraspStage.GRASP,
        updated_grasp_pose=updated_pose,
    )

    assert replanned.parent_plan_id == "first-plan"
    assert replanned.revision == 1
    assert replanned.start_stage is GraspStage.GRASP
    assert replanned.pregrasp_offset_m == 0.075
    assert replanned.approach_clearance_m == 0.045
    assert tuple(segment.stage for segment in replanned.segments) == (
        GraspStage.GRASP,
        GraspStage.LIFT,
    )
    assert np.allclose(replanned.segments[0].start_joints, measured)
    assert np.isclose(replanned.target.grasp_pose[0, 3], updated_pose[0, 3])


def test_existing_grasp_candidates_adapt_without_schema_translation():
    pose = np.eye(4)
    pose[0, 3] = 0.4
    candidates = GraspCandidates(
        grasps=np.stack((np.eye(4), pose)),
        scores=np.array((0.2, 0.9)),
        centroid=np.array((0.4, 0.0, 0.1)),
        frame="piper_base_link",
        num_raw=2,
        widths=np.array((0.02, 0.05)),
    )

    target = GraspTrajectoryTarget.from_candidates(candidates, 1)

    assert target.candidate_index == 1
    assert target.score == 0.9
    assert target.required_width_m == 0.05
    assert np.allclose(target.grasp_pose, pose)


def test_builder_uses_pinocchio_reduced_model_velocity_limit_shape():
    class _ReducedModel:
        arm_velocity_limits = np.full(6, 0.6)

    builder = StagedGraspTrajectoryBuilder.from_kinematic_model(
        _PoseEncodingIK(),
        _ReducedModel(),
        acceleration_limits=np.full(6, 1.2),
    )

    assert np.allclose(builder.velocity_limits, 0.6)
