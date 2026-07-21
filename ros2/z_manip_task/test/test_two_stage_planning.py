"""Unit contracts for post-pregrasp fresh-scene replanning."""

from dataclasses import fields, FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning.grasp_pipeline import CandidateFailure, PlannedGrasp
from z_manip.planning.time_parameterization import TimeParameterizationConfig
from z_manip_task.planning import (
    GraspCompletionProgram,
    OnlinePlanner,
    PerceptionObservation,
    PregraspTransitProgram,
)


def _pose(x: float) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _observation(serial: int, *, scene_offset: float = 0.0) -> PerceptionObservation:
    target = np.array([[0.4, 0.0, 0.2], [0.41, 0.01, 0.2]])
    return PerceptionObservation(
        serial=serial,
        stamp_s=10.0 + serial,
        target_points=target.copy(),
        target_collision_points=target.copy(),
        scene_points=np.array([[1.0 + scene_offset, 0.0, 0.0]]),
        target_position_camera=np.array([0.0, 0.0, 0.4]),
        camera_origin_piper=np.zeros(3),
        camera_rotation_piper=np.eye(3),
        affordance=None,
    )


def _planned(
    *,
    candidate_index: int,
    transit: object,
    approach: object,
    lift: object,
    required_width_m: float,
) -> PlannedGrasp:
    return PlannedGrasp(
        candidate_index=candidate_index,
        symmetry_index=candidate_index + 1,
        grasp_pose=_pose(0.5 + candidate_index),
        pregrasp_pose=_pose(0.4 + candidate_index),
        transit=JointTrajectory(
            joint_names=('joint_a', 'joint_b'),
            waypoints=np.asarray(transit, dtype=float),
        ),
        approach_joints=np.asarray(approach, dtype=float),
        lift_joints=np.asarray(lift, dtype=float),
        required_width_m=required_width_m,
        score=0.8 + candidate_index,
        failures=(CandidateFailure(7, None, 'ik', 'earlier candidate rejected'),),
    )


def _planner(plans: list[PlannedGrasp]):
    planner = OnlinePlanner.__new__(OnlinePlanner)
    planner.chain = SimpleNamespace(
        dof=2,
        velocity_limits=np.array([2.0, 2.0]),
    )
    planner.config = SimpleNamespace(
        robot=SimpleNamespace(acceleration_limits=(4.0, 4.0)),
        time_parameterization=TimeParameterizationConfig(
            sample_period_s=0.05,
            min_segment_time_s=0.05,
            velocity_scale=1.0,
            acceleration_scale=1.0,
        ),
    )
    calls: list[dict[str, object]] = []

    def candidates(observation, control=None):
        return f'candidates-{observation.serial}'

    def plan(candidates_value, **kwargs):
        calls.append({'candidates': candidates_value, **kwargs})
        return plans[len(calls) - 1]

    planner.candidates = candidates
    planner._plan = plan
    return planner, calls


def _stage_one_plan() -> PlannedGrasp:
    return _planned(
        candidate_index=0,
        transit=((0.0, 0.0), (0.2, 0.1)),
        approach=((0.2, 0.1), (9.0, 9.0)),
        lift=((9.0, 9.0), (10.0, 10.0)),
        required_width_m=0.01,
    )


def _fresh_plan() -> PlannedGrasp:
    return _planned(
        candidate_index=3,
        transit=((0.21, 0.09), (0.3, 0.15)),
        approach=((0.3, 0.15), (0.4, 0.2), (0.5, 0.25)),
        lift=((0.5, 0.25), (0.5, 0.4)),
        required_width_m=0.055,
    )


def test_stage_one_exposes_only_retimed_pregrasp_transit():
    planner, calls = _planner([_stage_one_plan()])
    observation = _observation(11)

    program = planner.pregrasp_program(observation, np.zeros(2))

    assert isinstance(program, PregraspTransitProgram)
    assert {field.name for field in fields(program)} == {
        'observation_serial',
        'candidate_index',
        'symmetry_index',
        'score',
        'failures',
        'transit',
    }
    assert not hasattr(program, 'planned')
    assert not hasattr(program, 'approach')
    assert not hasattr(program, 'lift')
    assert not hasattr(program, 'required_width_m')
    assert np.allclose(program.transit.positions[0], (0.0, 0.0))
    assert np.allclose(program.transit.positions[-1], (0.2, 0.1))
    assert calls[0]['candidates'] == 'candidates-11'
    assert calls[0]['scene_points'] is observation.scene_points
    assert calls[0]['target_points'] is observation.target_collision_points

    with pytest.raises(FrozenInstanceError):
        program.score = 0.0
    with pytest.raises(ValueError, match='read-only'):
        program.transit.positions[0, 0] = 1.0


def test_stage_one_at_pregrasp_emits_bounded_stationary_handoff():
    already_there = _planned(
        candidate_index=0,
        transit=((0.2, 0.1), (0.2, 0.1)),
        approach=((0.2, 0.1), (0.3, 0.2)),
        lift=((0.3, 0.2), (0.3, 0.3)),
        required_width_m=0.03,
    )
    planner, _calls = _planner([already_there])

    program = planner.pregrasp_program(
        _observation(11),
        np.array((0.2, 0.1)),
    )

    np.testing.assert_allclose(
        program.transit.positions,
        ((0.2, 0.1), (0.2, 0.1)),
    )
    np.testing.assert_allclose(program.transit.times_s, (0.0, 0.05))


def test_completion_rejects_stale_observation_before_replanning():
    planner, calls = _planner([_stage_one_plan()])
    observation = _observation(11)
    pregrasp = planner.pregrasp_program(observation, np.zeros(2))

    with pytest.raises(PlanningError, match='newer than pregrasp'):
        planner.grasp_completion_program(pregrasp, observation, (0.2, 0.1))

    assert len(calls) == 1


def test_completion_replans_from_fresh_scene_and_measured_joints():
    planner, calls = _planner([_stage_one_plan(), _fresh_plan()])
    initial = _observation(11)
    fresh = _observation(12, scene_offset=0.7)
    measured = np.array([0.21, 0.09])
    pregrasp = planner.pregrasp_program(initial, np.zeros(2))

    completion = planner.grasp_completion_program(pregrasp, fresh, measured)

    assert isinstance(completion, GraspCompletionProgram)
    assert len(calls) == 2
    assert calls[1]['candidates'] == 'candidates-12'
    assert calls[1]['scene_points'] is fresh.scene_points
    assert calls[1]['target_points'] is fresh.target_collision_points
    assert np.array_equal(calls[1]['current_joints'], measured)
    assert calls[1]['stamp_s'] == fresh.stamp_s
    assert completion.observation_serial == fresh.serial
    assert completion.candidate_index == 3
    assert completion.required_width_m == pytest.approx(0.055)
    assert np.allclose(completion.approach.positions[0], measured)
    assert np.allclose(completion.approach.positions[-1], (0.5, 0.25))
    assert any(
        np.allclose(position, (0.3, 0.15))
        for position in completion.approach.positions
    )
    assert np.allclose(completion.lift.positions[0], (0.5, 0.25))
    assert np.allclose(completion.lift.positions[-1], (0.5, 0.4))
    assert not np.any(np.all(np.isclose(completion.approach.positions, 9.0), axis=1))
    assert not completion.grasp_pose.flags.writeable
    assert not completion.pregrasp_pose.flags.writeable


def test_completion_rejects_discontinuous_replanned_approach():
    discontinuous = _planned(
        candidate_index=2,
        transit=((0.21, 0.09), (0.3, 0.15)),
        approach=((0.35, 0.15), (0.5, 0.25)),
        lift=((0.5, 0.25), (0.5, 0.4)),
        required_width_m=0.04,
    )
    planner, _calls = _planner([_stage_one_plan(), discontinuous])
    pregrasp = planner.pregrasp_program(_observation(11), np.zeros(2))

    with pytest.raises(PlanningError, match='discontinuous phase boundary'):
        planner.grasp_completion_program(
            pregrasp,
            _observation(12),
            np.array([0.21, 0.09]),
        )
