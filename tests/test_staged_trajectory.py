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


def _segment_final_speed(segment) -> float:
    positions = segment.trajectory.positions
    times = segment.trajectory.times_s
    step = float(times[-1] - times[-2])
    return float(np.max(np.abs(positions[-1] - positions[-2])) / step)


def _segment_speeds(segment) -> np.ndarray:
    positions = segment.trajectory.positions
    times = segment.trajectory.times_s
    return np.max(np.abs(np.diff(positions, axis=0)), axis=1) / np.diff(times)


def _direct_request(**overrides) -> StagedGraspRequest:
    base = dict(
        current_joints=np.zeros(6),
        target=_target(),
        pregrasp_offset_m=0.08,
        approach_clearance_m=0.05,
        lift_distance_m=0.09,
    )
    base.update(overrides)
    return StagedGraspRequest(**base)


def test_direct_approach_is_the_default_and_is_recorded_on_the_plan():
    plan = _builder(_MidpointPlanner()).build(_direct_request())

    assert plan.direct_approach is True
    assert plan.contact_speed_scale == 0.5
    # The four-stage structure and suffix contract are preserved.
    assert tuple(segment.stage for segment in plan.segments) == tuple(GraspStage)
    for previous, following in zip(plan.segments, plan.segments[1:]):
        assert np.allclose(previous.goal_joints, following.start_joints)


def test_direct_approach_passes_the_standoff_via_without_halting():
    builder = _builder(_MidpointPlanner())
    direct = builder.build(_direct_request(direct_approach=True))
    staged = builder.build(_direct_request(direct_approach=False))

    # The pregrasp segment ends AT the standoff.  Direct mode keeps a real
    # velocity through that via; the staged fallback stops dead there.
    direct_via_speed = _segment_final_speed(direct.segment(GraspStage.PREGRASP))
    staged_via_speed = _segment_final_speed(staged.segment(GraspStage.PREGRASP))
    assert direct_via_speed > 0.05
    assert staged_via_speed < 1e-3
    assert direct_via_speed > 20.0 * staged_via_speed


def test_direct_approach_speed_profile_slows_into_contact():
    plan = _builder(_MidpointPlanner()).build(_direct_request(contact_speed_scale=0.4))

    grasp = plan.segment(GraspStage.GRASP)
    speeds = _segment_speeds(grasp)
    quarter = max(1, len(speeds) // 4)
    # The blended descent decelerates as it reaches contact: the final quarter
    # of the grasp segment is slower than its opening quarter, and it ends at
    # (near) rest exactly at the grasp pose.
    assert np.max(speeds[-quarter:]) < np.max(speeds[:quarter])
    assert _segment_final_speed(grasp) < 0.05
    # Peak descent speed honours the contact speed cap (0.8 vel * 0.7 scale * 0.4).
    assert np.max(speeds) <= 0.8 * 0.7 * 0.4 * 1.01


def test_direct_approach_flattened_profile_is_velocity_continuous_and_bounded():
    plan = _builder(_MidpointPlanner()).build(_direct_request(contact_speed_scale=0.6))
    flattened = plan.flattened()

    assert np.all(np.diff(flattened.times_s) > 0.0)
    velocity = np.diff(flattened.positions, axis=0) / np.diff(flattened.times_s)[:, None]
    # No stage anywhere exceeds the base velocity envelope.
    assert np.max(np.abs(velocity)) <= 0.8 * 0.7 * 1.01


def test_direct_approach_preserves_the_reverse_replay_corridor():
    builder = _builder(_MidpointPlanner())
    direct = builder.build(_direct_request(direct_approach=True))
    staged = builder.build(_direct_request(direct_approach=False))

    # Same IK / joint paths: the geometry (standoff corners) is untouched, only
    # the timing changed.  A reverse joint replay therefore visits the exact
    # same standoffs whether the plan was blended or staged.
    for stage in GraspStage:
        assert np.allclose(
            direct.segment(stage).goal_joints,
            staged.segment(stage).goal_joints,
        )
        assert np.allclose(
            direct.segment(stage).start_joints,
            staged.segment(stage).start_joints,
        )

    standoffs = [direct.segment(stage).goal_joints for stage in GraspStage]
    forward = direct.flattened().positions
    reverse = forward[::-1]
    for corner in standoffs:
        assert np.min(np.max(np.abs(forward - corner), axis=1)) < 1e-6
        assert np.min(np.max(np.abs(reverse - corner), axis=1)) < 1e-6


def test_direct_approach_segments_start_and_end_on_exact_checked_joints():
    plan = _builder(_MidpointPlanner()).build(_direct_request())

    pregrasp = plan.segment(GraspStage.PREGRASP)
    grasp = plan.segment(GraspStage.GRASP)
    # The shared standoff via is an exact endpoint of both blended slices.
    assert np.allclose(pregrasp.trajectory.positions[-1], pregrasp.goal_joints)
    assert np.allclose(grasp.trajectory.positions[0], grasp.start_joints)
    assert np.allclose(pregrasp.goal_joints, grasp.start_joints)
    # The grasp slice terminates exactly at the contact pose joints.
    assert np.allclose(grasp.trajectory.positions[-1], grasp.goal_joints)


def test_direct_approach_replan_from_grasp_falls_back_to_single_descent():
    builder = _builder(_MidpointPlanner())
    initial = builder.build(_direct_request(), plan_id="p0")

    measured = initial.segment(GraspStage.PREGRASP).goal_joints + 0.01
    replanned = builder.replan_remaining(
        initial,
        measured,
        from_stage=GraspStage.GRASP,
    )

    # Direct mode is propagated across the rolling replan.
    assert replanned.direct_approach is True
    assert replanned.contact_speed_scale == initial.contact_speed_scale
    assert tuple(s.stage for s in replanned.segments) == (GraspStage.GRASP, GraspStage.LIFT)
    # A lone grasp descent still decelerates to rest at contact.
    assert _segment_final_speed(replanned.segment(GraspStage.GRASP)) < 0.05


def test_direct_approach_can_be_disabled_for_the_staged_fallback():
    plan = _builder(_MidpointPlanner()).build(_direct_request(direct_approach=False))

    assert plan.direct_approach is False
    # Every stage stops at rest at its own boundary (classic staged behaviour).
    for stage in (GraspStage.APPROACH, GraspStage.PREGRASP, GraspStage.GRASP):
        assert _segment_final_speed(plan.segment(stage)) < 1e-3


def test_contact_speed_scale_is_validated():
    import pytest

    with pytest.raises(ValueError, match="contact speed scale"):
        _direct_request(contact_speed_scale=0.0)
    with pytest.raises(ValueError, match="contact speed scale"):
        _direct_request(contact_speed_scale=1.5)


def test_direct_chain_is_continuous_through_the_outer_standoff():
    builder = _builder(_MidpointPlanner())
    direct = builder.build(_direct_request(direct_approach=True))
    staged = builder.build(_direct_request(direct_approach=False))

    # The transit (approach segment) no longer halts at the outer standoff:
    # its final speed is a real slow-down via, and the first pregrasp interval
    # continues at (essentially) the same speed -- one continuous profile.
    direct_out = _segment_final_speed(direct.segment(GraspStage.APPROACH))
    staged_out = _segment_final_speed(staged.segment(GraspStage.APPROACH))
    pregrasp_start = float(_segment_speeds(direct.segment(GraspStage.PREGRASP))[0])
    assert direct_out > 0.005
    assert staged_out < 1e-3
    assert abs(direct_out - pregrasp_start) <= 0.5 * max(direct_out, pregrasp_start)


def test_direct_chain_does_not_crawl_relative_to_staged():
    builder = _builder(_MidpointPlanner())
    direct = builder.build(_direct_request(direct_approach=True))
    staged = builder.build(_direct_request(direct_approach=False))

    # Removing the stops must not be paid for with a globally slower profile:
    # the continuous chain finishes within a modest factor of the stop-and-go
    # plan (it is typically faster since it never re-accelerates from rest).
    def pre_contact(plan):
        return sum(
            plan.segment(stage).duration_s
            for stage in (GraspStage.APPROACH, GraspStage.PREGRASP, GraspStage.GRASP)
        )

    assert pre_contact(direct) <= 1.3 * pre_contact(staged)


def test_direct_chain_descent_respects_contact_cap_at_the_junction():
    plan = _builder(_MidpointPlanner()).build(_direct_request(contact_speed_scale=0.4))

    # Everything past the outer standoff (the descent) obeys the contact cap.
    contact_cap = 0.8 * 0.7 * 0.4
    for stage in (GraspStage.PREGRASP, GraspStage.GRASP):
        assert np.max(_segment_speeds(plan.segment(stage))) <= contact_cap * 1.02
    # And the profile is time-parameterized monotonically across the chain.
    flattened = plan.flattened()
    assert np.all(np.diff(flattened.times_s) > 0.0)
