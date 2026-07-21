"""Tests for explicit, terminal task cancellation."""

import threading

import pytest

from z_manip.orchestration.mobile_manipulation import (
    FailureKind,
    MobileManipulationStateMachine,
    RetryBudget,
)
from z_manip_task.core import (
    parse_execution_status,
    RuntimePhase,
    RuntimeSafetyCore,
    TaskGenerationGuard,
    terminal_result,
)


def test_active_cancel_is_terminal_and_requests_every_stop() -> None:
    core = RuntimeSafetyCore()
    core.begin('pick the observed bottle')
    core.phase = RuntimePhase.TRANSIT
    core.trajectory_sent(
        'transit', executor_epoch='executor-a', published_at_s=0.0,
        trajectory_token='trajectory-transit',
    )

    action = core.cancel()

    assert core.phase is RuntimePhase.CANCELED
    assert core.instruction == 'pick the observed bottle'
    assert core.failure_reason == ''
    assert core.execution_segment == ''
    assert core.expected_command_id is None
    assert action.stop_base and action.cancel_navigation and action.cancel_arm
    assert terminal_result(core.phase) == 'canceled'


def test_idle_and_repeated_cancel_are_safe_and_new_task_can_start() -> None:
    core = RuntimeSafetyCore()

    first = core.cancel()
    second = core.cancel()

    assert core.phase is RuntimePhase.CANCELED
    assert first == second
    assert terminal_result(core.phase) == 'canceled'

    core.begin('start a later task')
    assert core.phase is RuntimePhase.POSE_SETTLE
    assert core.instruction == 'start a later task'
    assert terminal_result(core.phase) == ''


def test_stale_execution_status_cannot_revive_a_canceled_task() -> None:
    core = RuntimeSafetyCore()
    core.begin('pick')
    core.phase = RuntimePhase.TRANSIT
    core.trajectory_sent(
        'transit', executor_epoch='executor-a', published_at_s=0.0,
        trajectory_token='trajectory-transit',
    )
    core.execution_update(parse_execution_status(
        'active;owner=trajectory;segment=transit;command_id=7;'
        'executor_epoch=executor-a;trajectory_token=trajectory-transit;'
        'trajectory_received_at=7.0',
    ))
    core.cancel()

    action = core.execution_update(parse_execution_status(
        'succeeded;owner=trajectory;segment=transit;command_id=7;'
        'executor_epoch=executor-a;trajectory_token=trajectory-transit;'
        'trajectory_received_at=7.0',
    ))

    assert core.phase is RuntimePhase.CANCELED
    assert not action.stop_base and not action.cancel_navigation and not action.cancel_arm

    core.begin('new task')
    core.phase = RuntimePhase.TRANSIT
    core.trajectory_sent(
        'transit', executor_epoch='executor-a', published_at_s=8.0,
        trajectory_token='trajectory-new',
    )
    assert core.minimum_command_id == 8
    core.execution_update(parse_execution_status(
        'active;owner=trajectory;segment=transit;command_id=7;'
        'executor_epoch=executor-a;trajectory_token=trajectory-transit;'
        'trajectory_received_at=7.0',
    ))
    assert core.phase is RuntimePhase.TRANSIT
    assert core.expected_command_id is None


def test_task_generation_rejects_stale_future_results() -> None:
    generation = TaskGenerationGuard()
    first = generation.current
    assert generation.accepts(first)

    second = generation.advance()

    assert not generation.accepts(first)
    assert generation.accepts(second)
    assert not generation.accepts(None)
    assert not generation.accepts(True)


def test_canceled_node_drops_stale_execution_before_segment_publish() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick')
            self._core.phase = RuntimePhase.TRANSIT
            self._core.trajectory_sent(
                'transit', executor_epoch='executor-a', published_at_s=0.0,
                trajectory_token='trajectory-transit',
            )
            self._core.cancel()
            self._execution_status = None
            self._execution_status_seen_s = None
            self._latest_gripper_command_id = 0
            self._gripper_feedback = []
            self.effects: list[str] = []

        def _now_s(self) -> float:
            return 12.0

        def _apply_safety(self, action) -> None:
            if action.stop_base or action.cancel_navigation or action.cancel_arm:
                self.effects.append('safety')

        def _publish_program_segment(self, segment: str) -> None:
            self.effects.append(f'publish:{segment}')

    harness = Harness()
    MobileManipulationRuntime._execution_cb(
        harness,
        String(data='active;owner=trajectory;segment=transit;command_id=9'),
    )
    MobileManipulationRuntime._execution_cb(
        harness,
        String(data='succeeded;owner=trajectory;segment=transit;command_id=9'),
    )

    assert harness._core.phase is RuntimePhase.CANCELED
    assert harness.effects == []


def test_canceled_node_drops_stale_future_before_reading_result() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class StaleFuture:
        def done(self) -> bool:
            return True

        def result(self):
            raise AssertionError('stale planning result was read')

    class Harness:
        def __init__(self) -> None:
            self._future = StaleFuture()
            self._future_kind = 'final'
            self._future_serial = 8
            self._future_cancel_event = threading.Event()
            self._task_generation = TaskGenerationGuard()
            self._future_generation = self._task_generation.current
            self._task_generation.advance()

    harness = Harness()
    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is None
    assert harness._future_kind == ''
    assert harness._future_serial == 0
    assert harness._future_generation is None
    assert harness._future_cancel_event is None


def test_invalidating_async_work_signals_running_planner_before_cancel() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    cancel_event = threading.Event()

    class Future:
        def cancel(self) -> bool:
            assert cancel_event.is_set()
            return False

    class Harness:
        def __init__(self) -> None:
            self._future = Future()
            self._future_kind = 'standoff'
            self._future_serial = 7
            self._future_generation = 0
            self._future_cancel_event = cancel_event
            self._task_generation = TaskGenerationGuard()
            self._pregrasp_dispatch_fence = object()

    harness = Harness()
    MobileManipulationRuntime._invalidate_async_work(harness)

    assert cancel_event.is_set()
    assert harness._future is None
    assert harness._future_cancel_event is None
    assert harness._pregrasp_dispatch_fence is None


def test_task_perception_invalidation_publishes_reset_and_clears_validity() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import Empty
    from z_manip_task.node import MobileManipulationRuntime

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    class Harness:
        def __init__(self) -> None:
            self._perception_valid = True
            self._valid_seen_at = 12.0
            self._valid_perception_request_id = 'request-1'
            self._valid_perception_producer_epoch = 'producer-1'
            self._valid_perception_generation = 3
            self._valid_observation_stamp_ns = 123
            self._valid_observation_frame_id = 'camera'
            self._bound_perception_request_id = 'request-1'
            self._bound_perception_producer_epoch = 'producer-1'
            self._bound_perception_generation = 3
            self._required_perception_request_id = 'request-1'
            self._required_perception_generation = 3
            self._required_affordance_generation = 3
            self._grounding_reset_pub = Publisher()

    harness = Harness()

    MobileManipulationRuntime._invalidate_perception_session(harness)

    assert not harness._perception_valid
    assert harness._valid_seen_at is None
    assert harness._required_perception_request_id is None
    assert harness._bound_perception_producer_epoch is None
    assert len(harness._grounding_reset_pub.messages) == 1
    assert isinstance(harness._grounding_reset_pub.messages[0], Empty)


def test_task_cancel_invalidates_perception_before_terminal_status() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import Bool
    from z_manip_task.node import MobileManipulationRuntime

    class Resettable:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            self.calls += 1

    class Harness:
        _release_terminal_ownership = (
            MobileManipulationRuntime._release_terminal_ownership
        )

        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('cancel this task')
            self._approach = Resettable()
            self._visual_search = Resettable()
            self._verifier = Resettable()
            self._program = object()
            self._carry_program = object()
            self._place_programs = [object()]
            self._place_trajectory = object()
            self._place_contract = object()
            self._place_goal_id = 'old-goal'
            self._place_planning_started_at = 1.0
            self._release_started_at = 1.0
            self._desired_depth = 1.0
            self._approximate_displacement = 1.0
            self._closing_started_at = 1.0
            self._verification_started_at = 1.0
            self._commanded_close_aperture = 1.0
            self._expected_gripper_command_id = 1
            self._gripper_command_sent_s = 1.0
            self._gripper_feedback = [object()]
            self._trajectory_deadline_s = 1.0
            self._coarse_nav_ready = True
            self._pose_settle_until = 1.0
            self._pose_settle_started_at = 0.5
            self._pose_settle_last_tick_at = 0.75
            self._visual_search_settle_reference = object()
            self._lookout_pending = True
            self._visual_search_pending = True
            self._visual_search_edge_direction = 1
            self._visual_search_error_rad = 1.0
            self._visual_search_reason = 'old'
            self._required_affordance_generation = 1
            self._required_perception_generation = 1
            self._reground_started_at = 1.0
            self.events = []

        def _invalidate_async_work(self) -> None:
            self.events.append('async_invalidated')

        def _invalidate_perception_session(self) -> None:
            self.events.append('perception_invalidated')

        def _reset_execution_occlusion(self) -> None:
            return

        def _apply_safety(self, _action) -> None:
            self.events.append('safety_applied')

        def _publish_status(self, *, force: bool) -> None:
            assert force
            self.events.append('status_published')

    harness = Harness()

    MobileManipulationRuntime._task_cancel_cb(harness, Bool(data=True))

    assert harness._core.phase is RuntimePhase.CANCELED
    assert harness.events == [
        'async_invalidated',
        'perception_invalidated',
        'safety_applied',
        'status_published',
    ]
    assert harness._approach.calls == 1
    assert harness._visual_search.calls == 1
    assert harness._verifier.calls == 1
    assert harness._visual_search_settle_reference is None
    assert harness._pose_settle_until is None
    assert harness._pose_settle_started_at is None
    assert harness._pose_settle_last_tick_at is None


def test_precontact_recovery_cooperatively_cancels_running_planner() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    cancel_event = threading.Event()

    class Future:
        def cancel(self) -> bool:
            assert cancel_event.is_set()
            return False

    class Resettable:
        def __init__(self) -> None:
            self.reset_called = False

        def reset(self) -> None:
            self.reset_called = True

    class Parameter:
        def __init__(self, value) -> None:
            self.value = value

    class Harness:
        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the observed bottle')
            self._core.phase = RuntimePhase.STANDOFF
            self._task = MobileManipulationStateMachine()
            self._task_generation = TaskGenerationGuard()
            self._future = Future()
            self._future_kind = 'standoff'
            self._future_serial = 4
            self._future_generation = self._task_generation.current
            self._future_cancel_event = cancel_event
            self._program = object()
            self._carry_program = object()
            self._approach = Resettable()
            self._coarse_nav_ready = True
            self._required_affordance_generation = 1
            self._required_perception_generation = 1
            self._reground_started_at = 1.0
            self._visual_search_pending = False
            self._visual_search_reason = ''
            self._visual_search_edge_direction = 0
            self._serial_gate = object()
            self._perception_valid = True
            self._valid_seen_at = 1.0
            self._target_camera = object()
            self._target_piper = object()
            self._target_cloud = object()
            self._target_uv = object()
            self._scene_cloud = object()
            self._camera_origin_piper = object()
            self._camera_rotation_piper = object()
            self._affordance = object()
            self._pose_settle_until = 1.0
            self._visual_search_settle_reference = object()
            self._lookout_pending = False
            self.perception_auth_cleared = False
            self.safety_actions = []

        def _apply_safety(self, action) -> None:
            self.safety_actions.append(action)

        def get_parameter(self, name: str) -> Parameter:
            return Parameter({
                'sync_slop_s': 1e-6,
                'max_perception_age_s': 0.35,
            }[name])

        def _clear_perception_authorization(self) -> None:
            self.perception_auth_cleared = True

        def _reset_execution_occlusion(self) -> None:
            return

        _invalidate_async_work = MobileManipulationRuntime._invalidate_async_work

    harness = Harness()
    generation = harness._task_generation.current

    recovered = MobileManipulationRuntime._recover_precontact(
        harness,
        FailureKind.TARGET_LOST,
        'tracker dropped while standoff planning was running',
    )

    assert recovered
    assert cancel_event.is_set()
    assert harness._future is None
    assert harness._future_cancel_event is None
    assert harness._task_generation.current == generation + 1
    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search_pending
    assert harness._visual_search_settle_reference is None
    assert harness._approach.reset_called
    assert harness.perception_auth_cleared


def test_terminal_precontact_recovery_cancels_running_planner() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    cancel_event = threading.Event()

    class Future:
        def cancel(self) -> bool:
            assert cancel_event.is_set()
            return False

    class Harness:
        _release_terminal_ownership = (
            MobileManipulationRuntime._release_terminal_ownership
        )

        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the observed bottle')
            self._core.phase = RuntimePhase.STANDOFF
            self._task = MobileManipulationStateMachine(RetryBudget(
                tracker_reacquisitions=0,
            ))
            self._task_generation = TaskGenerationGuard()
            self._future = Future()
            self._future_kind = 'standoff'
            self._future_serial = 4
            self._future_generation = self._task_generation.current
            self._future_cancel_event = cancel_event
            self._visual_search_settle_reference = object()
            self._terminal_ownership_released = False
            self.perception_invalidations = 0
            self.safety_actions = []

        def _apply_safety(self, action) -> None:
            self.safety_actions.append(action)

        def _invalidate_perception_session(self) -> None:
            self.perception_invalidations += 1

        def _reset_execution_occlusion(self) -> None:
            return

        _invalidate_async_work = MobileManipulationRuntime._invalidate_async_work

    harness = Harness()
    generation = harness._task_generation.current

    recovered = MobileManipulationRuntime._recover_precontact(
        harness,
        FailureKind.TARGET_LOST,
        'tracker retry budget exhausted while planning was running',
    )

    assert recovered
    assert cancel_event.is_set()
    assert harness._future is None
    assert harness._future_cancel_event is None
    assert harness._task_generation.current == generation + 1
    assert harness._task.terminal
    assert harness._core.phase is RuntimePhase.FAILED
    assert harness._visual_search_settle_reference is None
    assert harness.perception_invalidations == 1
