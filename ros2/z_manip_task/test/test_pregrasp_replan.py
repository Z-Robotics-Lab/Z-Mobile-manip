"""Execution-order contracts for fresh planning after pregrasp arrival."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip('rclpy')
from std_msgs.msg import String  # noqa: E402

from z_manip.orchestration.mobile_manipulation import (  # noqa: E402
    MobileManipulationStateMachine,
    Stage,
)
from z_manip.planning.time_parameterization import (  # noqa: E402
    TimedJointTrajectory,
)
from z_manip_task.core import (  # noqa: E402
    RuntimePhase,
    RuntimeSafetyCore,
    TaskGenerationGuard,
)
from z_manip_task.node import (  # noqa: E402
    _ApproachExecutionJointFence,
    _ApproachPlanningAnchor,
    _fresh_approach_joint_positions,
    _JointFeedback,
    _PlanningObservationChanged,
    _PlanningObservationIdentity,
    _PlanningObservationPending,
    _PregraspDispatchFence,
    _PregraspHandoff,
    _target_geometry_signature,
    _validate_target_geometry_change,
    MobileManipulationRuntime,
)
from z_manip_task.planning import (  # noqa: E402
    GraspCompletionProgram,
    PregraspTransitProgram,
)


def _identity(*, stamp_ns: int = 10_000_000_000):
    return _PlanningObservationIdentity(
        request_id='request-a',
        producer_epoch='tracker-a',
        generation=3,
        stamp_ns=stamp_ns,
        frame_id='wrist_depth_optical_frame',
        target_position_camera=(0.0, 0.0, 0.5),
    )


def _pregrasp_program() -> PregraspTransitProgram:
    return PregraspTransitProgram(
        observation_serial=1,
        candidate_index=2,
        symmetry_index=1,
        score=0.8,
        failures=(),
        transit=TimedJointTrajectory(
            positions=np.array((np.zeros(6), (0.0, 1.0, -0.7, 0.0, 0.0, 0.0))),
            times_s=np.array((0.0, 1.0)),
        ),
    )


def _completion_program() -> GraspCompletionProgram:
    pose = np.eye(4)
    return GraspCompletionProgram(
        observation_serial=2,
        candidate_index=3,
        symmetry_index=1,
        grasp_pose=pose.copy(),
        pregrasp_pose=pose.copy(),
        required_width_m=0.04,
        score=0.9,
        failures=(),
        approach=TimedJointTrajectory(
            positions=np.array((
                (0.0, 1.0, -0.7, 0.0, 0.0, 0.0),
                (0.0, 1.1, -0.8, 0.0, 0.0, 0.0),
            )),
            times_s=np.array((0.0, 0.5)),
        ),
        lift=TimedJointTrajectory(
            positions=np.array((
                (0.0, 1.1, -0.8, 0.0, 0.0, 0.0),
                (0.0, 1.0, -0.6, 0.0, 0.0, 0.0),
            )),
            times_s=np.array((0.0, 0.5)),
        ),
    )


def test_pregrasp_validation_is_async_and_freezes_feedback() -> None:
    """Freeze dispatch watermarks only after asynchronous validation."""
    class DonePlan:
        @staticmethod
        def done() -> bool:
            return True

        @staticmethod
        def result() -> PregraspTransitProgram:
            return _pregrasp_program()

    class ValidationFuture:
        def __init__(self) -> None:
            self.complete = False

        def done(self) -> bool:
            return self.complete

        @staticmethod
        def result() -> bool:
            return True

        @staticmethod
        def cancel() -> bool:
            return False

    class Worker:
        def __init__(self) -> None:
            self.calls = []
            self.future = ValidationFuture()

        def submit(self, function, *args, **kwargs):
            self.calls.append((function, args, kwargs))
            return self.future

    identity = _identity()
    generation = TaskGenerationGuard()
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLANNING
    core.planned_serial = 1
    task = MobileManipulationStateMachine()
    task.stage = Stage.OBSERVE_GRASP
    worker = Worker()
    clock = [10.0]
    planner = SimpleNamespace(
        chain=SimpleNamespace(dof=6),
        validate_path=lambda *_args, **_kwargs: pytest.fail(
            'path validation must not run in the executor callback',
        ),
    )
    harness = SimpleNamespace(
        _future=DonePlan(),
        _future_kind='pregrasp',
        _future_serial=1,
        _future_generation=generation.current,
        _future_cancel_event=None,
        _future_base_anchor=None,
        _future_observation_identity=identity,
        _future_observation_wait=None,
        _task_generation=generation,
        _approach_planning_anchor=None,
        _pregrasp_program=None,
        _pregrasp_planning_identity=None,
        _pregrasp_dispatch_fence=None,
        _program=None,
        _core=core,
        _task=task,
        _joint_state=np.zeros(6),
        _joint_sequence=5,
        _joint_stamp_ns=10_000_000_000,
        _odom_sequence=8,
        _odom_stamp_ns=10_000_000_000,
        _scene_cloud=np.zeros((4, 3)),
        _target_cloud=np.zeros((4, 3)),
        _serial_gate=SimpleNamespace(
            snapshot=lambda _now: SimpleNamespace(stamp_s=10.0),
        ),
        _grounding_observation_authorized=lambda _sync: True,
        _validate_grasp_planning_observation=lambda _identity, **_kwargs: None,
        _planner=planner,
        _worker=worker,
        _now_s=lambda: clock[0],
        _recover_precontact=lambda *_args: False,
        _apply_safety=lambda _action: pytest.fail(
            'valid staged plan must not fail',
        ),
        get_parameter=lambda name: SimpleNamespace(value={
            'max_trajectory_start_error_rad': 0.04,
            'grasp_planning_budget_s': 15.0,
            'perception_loss_timeout_s': 0.6,
            'pregrasp_dispatch_feedback_wait_timeout_s': 1.0,
        }[name]),
    )
    harness._planning_control = lambda kind: (
        MobileManipulationRuntime._planning_control(harness, kind)
    )
    harness._start_pregrasp_transit_validation = lambda owner, serial: (
        MobileManipulationRuntime._start_pregrasp_transit_validation(
            harness,
            owner,
            serial,
        )
    )
    harness._freeze_pregrasp_dispatch_fence = lambda: (
        MobileManipulationRuntime._freeze_pregrasp_dispatch_fence(harness)
    )

    MobileManipulationRuntime._poll_planning(harness)

    assert core.phase is RuntimePhase.PLANNING
    assert task.stage is Stage.OBSERVE_GRASP
    assert harness._future_kind == 'pregrasp_validation'
    assert harness._pregrasp_dispatch_fence is None
    assert len(worker.calls) == 1
    assert worker.calls[0][0] is planner.validate_path

    harness._joint_sequence = 7
    harness._joint_stamp_ns = 10_400_000_000
    harness._odom_sequence = 11
    harness._odom_stamp_ns = 10_400_000_000
    clock[0] = 10.4
    worker.future.complete = True
    MobileManipulationRuntime._poll_planning(harness)

    fence = harness._pregrasp_dispatch_fence
    assert isinstance(fence, _PregraspDispatchFence)
    assert fence.minimum_joint_sequence == 7
    assert fence.minimum_joint_stamp_ns == 10_400_000_000
    assert fence.minimum_odom_sequence == 11
    assert fence.minimum_odom_stamp_ns == 10_400_000_000
    assert fence.deadline_s == pytest.approx(11.4)
    assert harness._future is None
    assert core.phase is RuntimePhase.PLANNING


@pytest.mark.parametrize('planning_stage', [
    Stage.OBSERVE_GRASP,
    Stage.PLAN_GRASP,
])
def test_pregrasp_dispatch_waits_for_new_joint_and_odom_feedback(
    planning_stage: Stage,
) -> None:
    """Require new joint and odometry samples before transit publication."""
    identity = _identity()
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLANNING
    core.planned_serial = 1
    task = MobileManipulationStateMachine()
    task.stage = planning_stage
    program = _pregrasp_program()
    published = []
    zero_commands = []
    gripper_commands = []
    harness = SimpleNamespace(
        _core=core,
        _task=task,
        _pregrasp_program=program,
        _pregrasp_planning_identity=identity,
        _pregrasp_dispatch_fence=_PregraspDispatchFence(
            deadline_s=11.0,
            minimum_joint_sequence=5,
            minimum_joint_stamp_ns=10_000_000_000,
            minimum_odom_sequence=8,
            minimum_odom_stamp_ns=10_000_000_000,
        ),
        _joint_history=[_JointFeedback(
            received_at_s=10.0,
            source_stamp_ns=10_000_000_000,
            sequence=5,
            positions=np.zeros(6),
        )],
        _odom_seen_at=10.0,
        _odom_stamp_ns=10_000_000_000,
        _odom_sequence=8,
        _planner=SimpleNamespace(chain=SimpleNamespace(dof=6)),
        _publish_zero=lambda: zero_commands.append(True),
        _arm_is_still=lambda _now: True,
        _validate_grasp_planning_observation=lambda _identity: None,
        _guard_active_posture=lambda _now: True,
        _recover_precontact=lambda *_args: False,
        _apply_safety=lambda _action: pytest.fail(
            'fresh dispatch must not fail',
        ),
        _gripper_pub=SimpleNamespace(publish=gripper_commands.append),
        _publish_program_segment=lambda name, **kwargs: published.append(
            (name, kwargs),
        ),
        get_parameter=lambda name: SimpleNamespace(value={
            'pregrasp_joint_state_max_age_s': 0.25,
            'posture_state_max_age_s': 0.5,
            'max_trajectory_start_error_rad': 0.04,
            'open_aperture_m': 0.07,
        }[name]),
    )

    MobileManipulationRuntime._pregrasp_result_execution_tick(harness, 10.05)
    assert published == []
    assert core.phase is RuntimePhase.PLANNING

    harness._joint_history.append(_JointFeedback(
        received_at_s=10.10,
        source_stamp_ns=10_100_000_000,
        sequence=6,
        positions=np.zeros(6),
    ))
    MobileManipulationRuntime._pregrasp_result_execution_tick(harness, 10.11)
    assert published == []
    assert core.phase is RuntimePhase.PLANNING

    harness._odom_seen_at = 10.12
    harness._odom_stamp_ns = 10_120_000_000
    harness._odom_sequence = 9
    MobileManipulationRuntime._pregrasp_result_execution_tick(harness, 10.13)

    assert len(zero_commands) == 3
    assert len(gripper_commands) == 1
    assert published == [('transit', {'path_prevalidated': True})]
    assert harness._pregrasp_dispatch_fence is None
    assert core.phase is RuntimePhase.TRANSIT
    assert task.stage is not Stage.OBSERVE_GRASP


def test_pregrasp_dispatch_feedback_timeout_never_publishes() -> None:
    """Fail closed when post-validation feedback never arrives."""
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLANNING
    task = MobileManipulationStateMachine()
    task.stage = Stage.OBSERVE_GRASP
    recoveries = []
    published = []
    harness = SimpleNamespace(
        _core=core,
        _task=task,
        _pregrasp_program=_pregrasp_program(),
        _pregrasp_planning_identity=_identity(),
        _pregrasp_dispatch_fence=_PregraspDispatchFence(
            deadline_s=10.5,
            minimum_joint_sequence=5,
            minimum_joint_stamp_ns=10_000_000_000,
            minimum_odom_sequence=8,
            minimum_odom_stamp_ns=10_000_000_000,
        ),
        _publish_zero=lambda: None,
        _recover_precontact=lambda kind, detail: (
            recoveries.append((kind, detail)) or False
        ),
        _apply_safety=lambda action: published.append(action),
    )

    MobileManipulationRuntime._pregrasp_result_execution_tick(harness, 10.51)

    assert len(recoveries) == 1
    assert 'fresh post-validation' in recoveries[0][1]
    assert core.phase is RuntimePhase.FAILED
    assert len(published) == 1


def test_arm_stillness_uses_source_time_coverage_without_receipt_phase_bias() -> None:
    positions = np.array([0.0, 1.0, -0.71, 0.0, 0.0, 0.0])
    history = [
        _JointFeedback(
            received_at_s=10.04 + 0.05 * index,
            source_stamp_ns=10_000_000_000 + 50_000_000 * index,
            sequence=index + 1,
            positions=positions.copy(),
        )
        for index in range(5)
    ]
    harness = SimpleNamespace(
        _joint_history=history,
        get_parameter=lambda name: SimpleNamespace(value={
            'arm_still_window_s': 0.20,
            'arm_still_tolerance_rad': 0.01,
        }[name]),
    )

    assert MobileManipulationRuntime._arm_is_still(harness, 10.25)
    assert not MobileManipulationRuntime._arm_is_still(harness, 10.45)


def test_arm_stillness_rejects_source_time_excursion() -> None:
    positions = np.zeros(6)
    history = [
        _JointFeedback(
            received_at_s=20.0 + 0.05 * index,
            source_stamp_ns=20_000_000_000 + 50_000_000 * index,
            sequence=index + 1,
            positions=positions.copy(),
        )
        for index in range(5)
    ]
    history[2].positions[1] = 0.02
    harness = SimpleNamespace(
        _joint_history=history,
        get_parameter=lambda name: SimpleNamespace(value={
            'arm_still_window_s': 0.20,
            'arm_still_tolerance_rad': 0.01,
        }[name]),
    )

    assert not MobileManipulationRuntime._arm_is_still(harness, 20.21)


def test_arm_stillness_rejects_freshly_received_stale_source_window() -> None:
    positions = np.zeros(6)
    history = [
        _JointFeedback(
            received_at_s=30.0 + 0.05 * index,
            source_stamp_ns=20_000_000_000 + 50_000_000 * index,
            sequence=index + 1,
            positions=positions.copy(),
        )
        for index in range(5)
    ]
    harness = SimpleNamespace(
        _joint_history=history,
        get_parameter=lambda name: SimpleNamespace(value={
            'arm_still_window_s': 0.20,
            'arm_still_tolerance_rad': 0.01,
        }[name]),
    )

    assert not MobileManipulationRuntime._arm_is_still(harness, 30.21)


def _target_points() -> np.ndarray:
    return np.array([
        (x, y, z)
        for x in np.linspace(-0.08, 0.08, 9)
        for y in np.linspace(-0.03, 0.03, 5)
        for z in np.linspace(-0.02, 0.02, 3)
    ])


def _geometry(points: np.ndarray | None = None):
    return _target_geometry_signature(
        _target_points() if points is None else points,
        min_points=40,
        trim_mad_scale=4.5,
        extent_percentile=2.0,
    )


def test_transit_completion_enters_reobserve_without_publishing_approach():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.TRANSIT
    core.planned_serial = 1
    core.trajectory_sent(
        'transit',
        executor_epoch='executor-a',
        published_at_s=10.0,
        trajectory_token='trajectory-transit',
    )
    effects = []
    harness = SimpleNamespace(
        _lock=threading.RLock(),
        _core=core,
        _pregrasp_program=_pregrasp_program(),
        _pregrasp_planning_identity=_identity(),
        _pregrasp_handoff=None,
        _pregrasp_stable_joint_sequence=None,
        _pregrasp_stable_joint_stamp_ns=None,
        _pregrasp_joint_error_rad=None,
        _approach_planning_anchor=None,
        _planner=SimpleNamespace(chain=SimpleNamespace(dof=6)),
        _joint_sequence=7,
        _joint_stamp_ns=10_000_000_000,
        _execution_status=None,
        _execution_status_seen_s=None,
        _latest_gripper_command_id=0,
        _gripper_feedback=[],
        _trajectory_deadline_s=20.0,
        _now_s=lambda: 10.5,
        _guard_active_posture=lambda _now: True,
        get_parameter=lambda name: SimpleNamespace(value={
            'pregrasp_reobserve_timeout_s': 8.0,
        }[name]),
        _publish_zero=lambda: effects.append('zero'),
        _publish_program_segment=lambda name: effects.append(f'publish:{name}'),
        _apply_safety=lambda action: (
            effects.append(f'safety:{action.reason}')
            if (
                action.stop_base
                or action.cancel_navigation
                or action.cancel_arm
                or action.reason
            )
            else None
        ),
    )
    harness._begin_pregrasp_reobserve = lambda status, now: (
        MobileManipulationRuntime._begin_pregrasp_reobserve(
            harness,
            status,
            now,
        )
    )

    MobileManipulationRuntime._execution_cb(
        harness,
        String(data=(
            'active;owner=trajectory;segment=transit;command_id=9;'
            'executor_epoch=executor-a;trajectory_token=trajectory-transit;'
            'trajectory_received_at=10.100000'
        )),
    )
    MobileManipulationRuntime._execution_cb(
        harness,
        String(data=(
            'succeeded;owner=trajectory;segment=transit;command_id=9;'
            'executor_epoch=executor-a;trajectory_token=trajectory-transit;'
            'trajectory_received_at=10.100000'
        )),
    )

    assert core.phase is RuntimePhase.PREGRASP_REOBSERVE
    assert harness._pregrasp_handoff is not None
    assert harness._pregrasp_handoff.minimum_joint_sequence == 7
    assert effects == ['zero']


def test_reobserve_requires_new_joint_and_new_exact_observation_before_replan():
    endpoint = np.array((0.0, 1.0, -0.7, 0.0, 0.0, 0.0))
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PREGRASP_REOBSERVE
    core.planned_serial = 1
    initial_identity = _identity()
    handoff = _PregraspHandoff(
        observation_serial=1,
        observation_identity=initial_identity,
        endpoint_joints=endpoint.copy(),
        executor_epoch='executor-a',
        command_id=9,
        trajectory_received_at=10.1,
        completed_at_s=10.1,
        completion_source_stamp_ns=10_100_000_000,
        deadline_s=18.1,
        minimum_joint_sequence=3,
        minimum_joint_stamp_ns=10_000_000_000,
    )
    started = []
    recoveries = []
    fresh_identity = _identity(stamp_ns=10_200_000_000)
    harness = SimpleNamespace(
        _core=core,
        _pregrasp_handoff=handoff,
        _pregrasp_program=_pregrasp_program(),
        _pregrasp_stable_joint_sequence=None,
        _pregrasp_stable_joint_stamp_ns=None,
        _pregrasp_joint_error_rad=None,
        _approach_planning_anchor=None,
        _joint_history=[_JointFeedback(
            received_at_s=10.15,
            source_stamp_ns=10_110_000_000,
            sequence=4,
            positions=endpoint.copy(),
        )],
        _bound_perception_request_id=initial_identity.request_id,
        _bound_perception_producer_epoch=initial_identity.producer_epoch,
        _bound_perception_generation=initial_identity.generation,
        _valid_observation_frame_id=initial_identity.frame_id,
        _valid_observation_stamp_ns=10_200_000_000,
        _publish_zero=lambda: None,
        _arm_is_still=lambda _now: True,
        _grounding_observation_authorized=lambda _sync: True,
        _semantic_observation=lambda serial, stamp: SimpleNamespace(
            serial=serial,
            stamp_s=stamp,
            target_collision_points=_target_points(),
        ),
        _capture_planning_observation_identity=lambda _observation: fresh_identity,
        _start_planning=lambda kind, observation, **kwargs: started.append(
            (kind, observation, kwargs['planning_joints'].copy()),
        ),
        _recover_precontact=lambda kind, detail: (
            recoveries.append((kind, detail)) or False
        ),
        _apply_safety=lambda _action: pytest.fail('valid handoff must not fail'),
        get_parameter=lambda name: SimpleNamespace(value={
            'pregrasp_joint_state_max_age_s': 0.25,
            'pregrasp_joint_tolerance_rad': 0.05,
            'pregrasp_max_observation_joint_skew_s': 0.12,
            'semantic_min_points': 40,
            'approach_planning_geometry_trim_mad_scale': 4.5,
            'approach_planning_geometry_extent_percentile': 2.0,
        }[name]),
    )

    old = SimpleNamespace(serial=1, stamp_s=10.2)
    MobileManipulationRuntime._pregrasp_reobserve_tick(harness, 10.2, old)
    assert harness._pregrasp_stable_joint_sequence == 4
    assert started == []

    MobileManipulationRuntime._pregrasp_reobserve_tick(harness, 10.21, old)
    assert started == []
    assert core.phase is RuntimePhase.PREGRASP_REOBSERVE

    fresh = SimpleNamespace(serial=2, stamp_s=10.2)
    MobileManipulationRuntime._pregrasp_reobserve_tick(harness, 10.22, fresh)

    assert core.phase is RuntimePhase.APPROACH_PLANNING
    assert len(started) == 1
    assert started[0][0] == 'approach'
    assert started[0][1].serial == 2
    assert started[0][1].stamp_s == 10.2
    np.testing.assert_allclose(started[0][2], endpoint)
    assert harness._approach_planning_anchor.observation_serial == 2
    assert recoveries == []


def test_completed_second_plan_waits_for_new_joint_before_approach_execution():
    class DoneFuture:
        @staticmethod
        def done() -> bool:
            return True

        @staticmethod
        def result() -> GraspCompletionProgram:
            return _completion_program()

    identity = _identity(stamp_ns=10_200_000_000)
    joints = np.array((0.0, 1.0, -0.7, 0.0, 0.0, 0.0))
    anchor = _ApproachPlanningAnchor(
        observation_identity=identity,
        observation_serial=2,
        joint_sequence=5,
        joint_stamp_ns=10_190_000_000,
        joint_positions=joints.copy(),
        target_geometry=_geometry(),
    )
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.APPROACH_PLANNING
    core.planned_serial = 2
    task = MobileManipulationStateMachine()
    task.stage = Stage.PLAN_GRASP
    generation = TaskGenerationGuard()
    published = []
    harness = SimpleNamespace(
        _future=DoneFuture(),
        _future_kind='approach',
        _future_serial=2,
        _future_generation=generation.current,
        _future_cancel_event=None,
        _future_base_anchor=None,
        _future_observation_identity=identity,
        _future_observation_wait=None,
        _task_generation=generation,
        _approach_planning_anchor=anchor,
        _pregrasp_program=_pregrasp_program(),
        _program=None,
        _approach_execution_joint_fence=None,
        _core=core,
        _task=task,
        _validate_grasp_planning_observation=lambda _identity, **_kwargs: None,
        _joint_history=[_JointFeedback(
            received_at_s=10.2,
            source_stamp_ns=10_200_000_000,
            sequence=anchor.joint_sequence + 1,
            positions=joints.copy(),
        )],
        _planner=SimpleNamespace(chain=SimpleNamespace(dof=6)),
        _now_s=lambda: 10.2,
        get_parameter=lambda name: SimpleNamespace(value={
            'approach_execution_joint_wait_timeout_s': 1.0,
            'approach_execution_joint_state_max_age_s': 0.25,
        }[name]),
        _publish_zero=lambda: published.append('zero'),
        _publish_program_segment=lambda name: published.append(name),
        _publish_debug_plan=lambda: published.append('debug'),
        _recover_precontact=lambda *_args: False,
        _apply_safety=lambda _action: pytest.fail('valid approach plan must not fail'),
    )

    MobileManipulationRuntime._poll_planning(harness)

    assert isinstance(harness._program, GraspCompletionProgram)
    assert core.phase is RuntimePhase.APPROACH_PLANNING
    assert task.stage is Stage.PLAN_GRASP
    assert isinstance(
        harness._approach_execution_joint_fence,
        _ApproachExecutionJointFence,
    )
    assert (
        harness._approach_execution_joint_fence.deadline_s
        == pytest.approx(11.2)
    )
    assert (
        harness._approach_execution_joint_fence.minimum_joint_sequence
        == anchor.joint_sequence + 1
    )
    assert published == []

    MobileManipulationRuntime._approach_result_execution_tick(harness, 10.21)
    assert core.phase is RuntimePhase.APPROACH_PLANNING
    assert task.stage is Stage.PLAN_GRASP
    assert published == ['zero']

    harness._joint_history.append(_JointFeedback(
        received_at_s=10.22,
        source_stamp_ns=10_220_000_000,
        sequence=anchor.joint_sequence + 2,
        positions=joints.copy(),
    ))
    MobileManipulationRuntime._approach_result_execution_tick(harness, 10.22)

    assert core.phase is RuntimePhase.APPROACH
    assert task.stage is Stage.EXECUTE_GRASP
    assert published == ['zero', 'zero', 'approach', 'debug']


def test_approach_execution_requires_fresh_joint_feedback_after_anchor():
    identity = _identity(stamp_ns=10_200_000_000)
    joints = np.array((0.0, 1.0, -0.7, 0.0, 0.0, 0.0))
    anchor = _ApproachPlanningAnchor(
        observation_identity=identity,
        observation_serial=2,
        joint_sequence=5,
        joint_stamp_ns=10_190_000_000,
        joint_positions=joints.copy(),
        target_geometry=_geometry(),
    )
    fresh = _JointFeedback(
        received_at_s=10.21,
        source_stamp_ns=10_210_000_000,
        sequence=6,
        positions=joints.copy(),
    )

    measured = _fresh_approach_joint_positions(
        fresh,
        anchor,
        now_s=10.22,
        maximum_age_s=0.25,
        dof=6,
    )
    np.testing.assert_allclose(measured, joints)

    with pytest.raises(ValueError, match='did not advance'):
        _fresh_approach_joint_positions(
            fresh,
            anchor,
            now_s=10.22,
            maximum_age_s=0.25,
            dof=6,
            minimum_sequence=fresh.sequence,
            minimum_source_stamp_ns=fresh.source_stamp_ns,
        )

    with pytest.raises(ValueError, match='did not advance'):
        _fresh_approach_joint_positions(
            _JointFeedback(
                received_at_s=10.21,
                source_stamp_ns=anchor.joint_stamp_ns,
                sequence=anchor.joint_sequence,
                positions=joints.copy(),
            ),
            anchor,
            now_s=10.22,
            maximum_age_s=0.25,
            dof=6,
        )

    with pytest.raises(ValueError, match='stale'):
        _fresh_approach_joint_positions(
            fresh,
            anchor,
            now_s=10.60,
            maximum_age_s=0.25,
            dof=6,
        )


def test_target_geometry_gate_rejects_translation_scale_and_observable_rotation():
    reference_points = _target_points()
    reference = _geometry(reference_points)
    limits = {
        'max_center_drift_m': 0.025,
        'max_extent_change_m': 0.008,
        'max_extent_ratio': 1.25,
        'axis_separation_ratio': 1.25,
        'max_orientation_change_rad': np.deg2rad(20.0),
    }

    noisy = reference_points.copy()
    noisy[::11] += np.array((0.0005, -0.0004, 0.0003))
    _validate_target_geometry_change(reference, _geometry(noisy), **limits)

    with pytest.raises(ValueError, match='center drifted'):
        _validate_target_geometry_change(
            reference,
            _geometry(reference_points + np.array((0.03, 0.0, 0.0))),
            **limits,
        )

    scaled = reference_points.copy()
    scaled[:, 0] *= 1.35
    with pytest.raises(ValueError, match='extent changed'):
        _validate_target_geometry_change(reference, _geometry(scaled), **limits)

    angle = np.deg2rad(25.0)
    rotation = np.array((
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    ))
    with pytest.raises(ValueError, match='principal axis'):
        _validate_target_geometry_change(
            reference,
            _geometry(reference_points @ rotation.T),
            **limits,
        )

    absolute_only = reference_points.copy()
    absolute_only[:, 0] *= 1.06
    with pytest.raises(ValueError, match='extent changed'):
        _validate_target_geometry_change(
            reference,
            _geometry(absolute_only),
            **limits,
        )

    thin_reference_points = reference_points.copy()
    thin_reference_points[:, 2] *= 0.25
    thin_current_points = thin_reference_points.copy()
    thin_current_points[:, 2] *= 1.30
    with pytest.raises(ValueError, match='extent changed'):
        _validate_target_geometry_change(
            _geometry(thin_reference_points),
            _geometry(thin_current_points),
            **limits,
        )


def test_approach_result_waits_for_new_exact_bundle_and_rechecks_geometry():
    identity = _identity(stamp_ns=10_200_000_000)
    reference_points = _target_points()
    parameters = {
        'approach_planning_target_drift_tolerance_m': 0.025,
        'semantic_min_points': 40,
        'approach_planning_geometry_trim_mad_scale': 4.5,
        'approach_planning_geometry_extent_percentile': 2.0,
        'approach_planning_geometry_max_extent_change_m': 0.008,
        'approach_planning_geometry_max_extent_ratio': 1.25,
        'approach_planning_geometry_axis_separation_ratio': 1.25,
        'approach_planning_geometry_max_orientation_change_rad': np.deg2rad(20.0),
    }
    harness = SimpleNamespace(
        _bound_perception_request_id=identity.request_id,
        _bound_perception_producer_epoch=identity.producer_epoch,
        _bound_perception_generation=identity.generation,
        _required_perception_request_id=identity.request_id,
        _affordance_request_id=identity.request_id,
        _affordance_producer_epoch=identity.producer_epoch,
        _affordance_generation=identity.generation,
        _valid_observation_stamp_ns=identity.stamp_ns,
        _valid_observation_frame_id=identity.frame_id,
        _target_camera=np.asarray(identity.target_position_camera),
        _target_cloud=reference_points.copy(),
        _serial_gate=SimpleNamespace(snapshot=lambda _now: object()),
        _now_s=lambda: 10.21,
        _grounding_observation_authorized=lambda _sync: True,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    with pytest.raises(_PlanningObservationPending, match='strictly newer'):
        MobileManipulationRuntime._validate_grasp_planning_observation(
            harness,
            identity,
            target_geometry=_geometry(reference_points),
        )

    harness._valid_observation_stamp_ns += 10_000_000
    MobileManipulationRuntime._validate_grasp_planning_observation(
        harness,
        identity,
        target_geometry=_geometry(reference_points),
    )

    angle = np.deg2rad(25.0)
    rotation = np.array((
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    ))
    harness._target_cloud = reference_points @ rotation.T
    with pytest.raises(_PlanningObservationChanged, match='geometry changed'):
        MobileManipulationRuntime._validate_grasp_planning_observation(
            harness,
            identity,
            target_geometry=_geometry(reference_points),
        )
