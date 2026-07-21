"""Focused ownership, timing, and odometry lifecycle guards."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip_task.core import (
    ContinuousMotionQuietWindow,
    parse_execution_status,
    RuntimePhase,
    RuntimeSafetyCore,
    SafetyAction,
)


PLACE_GOAL = 'place-7-1000000000'
EXECUTOR_EPOCH = 'executor-epoch-a'
TRAJECTORY_TOKEN = 'trajectory-current'


class _CapturedPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _FixedClock:
    class _Now:
        @staticmethod
        def to_msg():
            return SimpleNamespace(sec=12, nanosec=34)

    @staticmethod
    def now():
        return _FixedClock._Now()


def _place_executor_snapshot():
    return {
        'goal_id': PLACE_GOAL,
        'trajectory_contract_id': PLACE_GOAL,
        'executor_epoch': EXECUTOR_EPOCH,
        'trajectory_command_highwater': 7,
        'trajectory_source_highwater_ns': 7_000_000_000,
        'gripper_command_highwater': 3,
        'gripper_source_highwater_ns': 6_500_000_000,
    }


def test_debug_plan_replaces_transient_history_and_terminal_clears_it() -> None:
    pytest.importorskip('rclpy')
    from visualization_msgs.msg import Marker
    from z_manip_task.node import MobileManipulationRuntime

    identity = np.eye(4, dtype=float)

    class Chain:
        @staticmethod
        def forward(joints):
            pose = identity.copy()
            pose[0, 3] = float(np.sum(joints))
            return pose

    class Harness:
        _publish_debug_plan = MobileManipulationRuntime._publish_debug_plan
        _clear_debug_plan = MobileManipulationRuntime._clear_debug_plan

        def __init__(self) -> None:
            grasp = identity.copy()
            grasp[2, 3] = 0.2
            self._pregrasp_program = SimpleNamespace(
                transit=SimpleNamespace(
                    positions=np.asarray(((0.0,), (0.1,))),
                ),
            )
            self._program = SimpleNamespace(
                pregrasp_pose=identity.copy(),
                grasp_pose=grasp,
                approach=SimpleNamespace(
                    positions=np.asarray(((0.1,), (0.2,))),
                ),
                lift=SimpleNamespace(positions=np.asarray(((0.2,), (0.3,)))),
            )
            self._piper_t_platform = identity.copy()
            self._planner = SimpleNamespace(chain=Chain())
            self._config = SimpleNamespace(
                robot=SimpleNamespace(platform_base_frame='base_link'),
            )
            self._markers_pub = _CapturedPublisher()
            self._path_pub = _CapturedPublisher()

        @staticmethod
        def get_clock():
            return _FixedClock()

    harness = Harness()
    harness._publish_debug_plan()

    published = harness._markers_pub.messages[-1].markers
    assert published[0].action == Marker.DELETEALL
    assert {(marker.ns, marker.id, marker.action) for marker in published[1:]} == {
        ('z_manip_grasp', 0, Marker.ADD),
        ('z_manip_grasp', 1, Marker.ADD),
    }
    assert harness._path_pub.messages[-1].poses

    harness._clear_debug_plan()

    cleared = harness._markers_pub.messages[-1].markers
    assert len(cleared) == 1
    assert cleared[0].action == Marker.DELETEALL
    assert harness._path_pub.messages[-1].poses == []


def _execution(
    trajectory: str,
    segment: str,
    command_id: int,
    received_at: float,
    *,
    contract_id: str = 'none',
    executor_epoch: str = EXECUTOR_EPOCH,
    trajectory_token: str = TRAJECTORY_TOKEN,
    gripper_command_id: int = 3,
    gripper_received_at: float = 6.5,
) -> str:
    return (
        f'{trajectory};owner=trajectory;segment={segment};'
        f'command_id={command_id};trajectory_contract_id={contract_id};'
        f'executor_epoch={executor_epoch};'
        f'trajectory_token={trajectory_token};'
        f'trajectory_received_at={received_at:.6f};'
        f'gripper_command_id={gripper_command_id};'
        f'gripper_received_at={gripper_received_at:.6f}'
    )


def _core_waiting_for_place_approach():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    carry = parse_execution_status(_execution('succeeded', 'carry', 7, 7.0))
    core.place_request_sent(
        place_contract_id=PLACE_GOAL,
        executor_state=carry,
    )
    core.place_plan_ready(_place_executor_snapshot())
    core.trajectory_sent(
        'place_transit',
        place_contract_id=PLACE_GOAL,
        executor_state=carry,
        trajectory_token=TRAJECTORY_TOKEN,
    )
    core.execution_update(parse_execution_status(
        _execution('active', 'place_transit', 8, 8.0),
    ))
    transit_succeeded = parse_execution_status(
        _execution('succeeded', 'place_transit', 8, 8.0),
    )
    core.execution_update(transit_succeeded)
    core.trajectory_sent(
        'place_approach',
        place_contract_id=PLACE_GOAL,
        executor_state=transit_succeeded,
        trajectory_token=TRAJECTORY_TOKEN,
    )
    return core, transit_succeeded


def _place_approach_callback_harness():
    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core, self._execution_status = (
                _core_waiting_for_place_approach()
            )
            self._execution_status_seen_s = 9.0
            self._latest_gripper_command_id = 0
            self._gripper_feedback = []
            self._trajectory_deadline_s = 20.0
            self.release_calls = []
            self.safety_reasons = []

        def _now_s(self) -> float:
            return 10.0

        def _guard_active_posture(self, _now: float) -> bool:
            return True

        def _apply_safety(self, action) -> None:
            if action.reason:
                self.safety_reasons.append(action.reason)

        def _start_release(self, now: float) -> None:
            self.release_calls.append(now)

    return Harness()


@pytest.mark.parametrize(
    'phase',
    (
        RuntimePhase.PICK_COMPLETE,
        RuntimePhase.COMPLETE,
        RuntimePhase.CANCELED,
        RuntimePhase.FAILED,
    ),
)
def test_terminal_safety_releases_async_and_perception_ownership_once(
    phase: RuntimePhase,
) -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        _release_terminal_ownership = (
            MobileManipulationRuntime._release_terminal_ownership
        )

        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.phase = phase
            self._terminal_ownership_released = False
            self._future = object()
            self._future_cancel_event = threading.Event()
            self._future_kind = 'final'
            self._future_generation = 3
            self.events = []

        def _invalidate_async_work(self) -> None:
            self.events.append('async')
            self._future = None
            self._future_cancel_event = None
            self._future_kind = ''
            self._future_generation = None

        def _invalidate_perception_session(self) -> None:
            self.events.append('perception')

        def _reset_execution_occlusion(self) -> None:
            return

    harness = Harness()

    MobileManipulationRuntime._apply_safety(harness, SafetyAction())
    MobileManipulationRuntime._apply_safety(harness, SafetyAction())

    assert harness.events == ['async', 'perception']
    assert harness._terminal_ownership_released


def test_posture_failure_releases_ownership_before_safety_and_status() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Resettable:
        def reset(self) -> None:
            return

    class Harness:
        _release_terminal_ownership = (
            MobileManipulationRuntime._release_terminal_ownership
        )

        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._approach = Resettable()
            self._visual_search = Resettable()
            self._verifier = Resettable()
            self._program = object()
            self._carry_program = object()
            self._place_programs = {'place_transit': object()}
            self._place_trajectory = object()
            self._place_contract = object()
            self._place_planning_started_at = 1.0
            self._release_started_at = 1.0
            self._closing_started_at = 1.0
            self._verification_started_at = 1.0
            self._trajectory_deadline_s = 1.0
            self._expected_gripper_command_id = 1
            self._gripper_command_sent_s = 1.0
            self._lookout_pending = True
            self._pose_settle_until = 2.0
            self._pose_settle_started_at = 1.0
            self._pose_settle_last_tick_at = 1.0
            self._visual_search_settle_reference = object()
            self._visual_search_pending = True
            self._terminal_ownership_released = False
            self.events = []

        def _invalidate_async_work(self) -> None:
            self.events.append('async')

        def _invalidate_perception_session(self) -> None:
            self.events.append('perception')

        def _reset_execution_occlusion(self) -> None:
            return

        def _apply_safety(self, _action) -> None:
            self.events.append('safety')

        def _publish_status(self, *, force: bool) -> None:
            assert force
            self.events.append('status')

    harness = Harness()

    MobileManipulationRuntime._fail_posture(harness, 'roll exceeded limit')

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.events == ['async', 'perception', 'safety', 'status']
    assert harness._visual_search_settle_reference is None


def test_observed_verification_after_retreat_releases_ownership_once() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        _apply_safety = MobileManipulationRuntime._apply_safety
        _release_terminal_ownership = (
            MobileManipulationRuntime._release_terminal_ownership
        )
        _begin_post_release_verification = (
            MobileManipulationRuntime._begin_post_release_verification
        )
        _complete_post_release_verification = (
            MobileManipulationRuntime._complete_post_release_verification
        )

        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core, _transit_succeeded = (
                _core_waiting_for_place_approach()
            )
            self._core.execution_update(parse_execution_status(
                _execution(
                    'active',
                    'place_approach',
                    9,
                    9.0,
                    contract_id=PLACE_GOAL,
                ),
            ))
            approach_succeeded = parse_execution_status(_execution(
                'succeeded',
                'place_approach',
                9,
                9.0,
                contract_id=PLACE_GOAL,
            ))
            self._core.execution_update(approach_succeeded)
            release_status = parse_execution_status(_execution(
                'succeeded',
                'place_approach',
                9,
                9.0,
                contract_id=PLACE_GOAL,
                gripper_command_id=4,
                gripper_received_at=9.2,
            ))
            self._core.execution_update(release_status)
            self._core.release_complete()
            self._core.trajectory_sent(
                'place_retreat',
                place_contract_id=PLACE_GOAL,
                executor_state=release_status,
                trajectory_token=TRAJECTORY_TOKEN,
            )
            self._execution_status = release_status
            self._execution_status_seen_s = None
            self._latest_gripper_command_id = 0
            self._gripper_feedback = []
            self._trajectory_deadline_s = 12.0
            self._task = SimpleNamespace(
                stage=SimpleNamespace(value='already_complete'),
            )
            self._place_observation_identity = object()
            self._post_release_release_command_id = 1
            self._post_release_pending_evidence = None
            self._post_release_verification_started_at_s = None
            self._post_release_verification_started_wall_s = None
            self._post_release_verification_last_tick_s = None
            self._post_release_verified_evidence = None
            self._terminal_ownership_released = False
            self.events = []

        def _now_s(self) -> float:
            return 10.0

        def _guard_active_posture(self, _now: float) -> bool:
            return True

        def _invalidate_async_work(self) -> None:
            self.events.append('async')

        def _invalidate_perception_session(self) -> None:
            self.events.append('perception')

        def _reset_execution_occlusion(self) -> None:
            return

    harness = Harness()
    active = String(
        data=_execution(
            'active',
            'place_retreat',
            10,
            10.0,
            contract_id=PLACE_GOAL,
            gripper_command_id=4,
            gripper_received_at=9.2,
        ),
    )
    succeeded = String(
        data=_execution(
            'succeeded',
            'place_retreat',
            10,
            10.0,
            contract_id=PLACE_GOAL,
            gripper_command_id=4,
            gripper_received_at=9.2,
        ),
    )

    MobileManipulationRuntime._execution_cb(harness, active)
    MobileManipulationRuntime._execution_cb(harness, succeeded)

    assert harness._core.phase is RuntimePhase.POST_RELEASE_VERIFICATION
    assert harness.events == []
    MobileManipulationRuntime._complete_post_release_verification(
        harness,
        SimpleNamespace(payload={}),
    )

    assert harness._core.phase is RuntimePhase.COMPLETE
    assert harness.events == ['async', 'perception']
    assert harness._trajectory_deadline_s is None


@pytest.mark.parametrize(
    'succeeded',
    (
        _execution(
            'succeeded',
            'place_approach',
            9,
            9.0,
            contract_id='place-other-goal',
        ),
        _execution(
            'succeeded',
            'place_approach',
            9,
            9.0,
            contract_id=PLACE_GOAL,
            executor_epoch='executor-epoch-b',
        ),
        _execution(
            'succeeded',
            'place_approach',
            9,
            9.5,
            contract_id=PLACE_GOAL,
        ),
    ),
)
def test_place_approach_callback_rejects_changed_transaction_before_release(
    succeeded: str,
) -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    harness = _place_approach_callback_harness()
    active = _execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    )

    MobileManipulationRuntime._execution_cb(harness, String(data=active))
    MobileManipulationRuntime._execution_cb(harness, String(data=succeeded))

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.release_calls == []
    assert len(harness.safety_reasons) == 1


def test_place_approach_callback_rejects_duplicate_identity_before_release() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    harness = _place_approach_callback_harness()
    duplicated = (
        _execution(
            'active',
            'place_approach',
            9,
            9.0,
            contract_id=PLACE_GOAL,
        )
        + ';executor_epoch=executor-epoch-b'
    )

    MobileManipulationRuntime._execution_cb(
        harness,
        String(data=duplicated),
    )

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.release_calls == []
    assert harness.safety_reasons == [
        'execution status invalid: execution status repeats field '
        "'executor_epoch'",
    ]


def test_place_approach_callback_releases_once_for_exact_active_success() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    harness = _place_approach_callback_harness()
    active = String(data=_execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))
    succeeded = String(data=_execution(
        'succeeded',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))

    MobileManipulationRuntime._execution_cb(harness, active)
    MobileManipulationRuntime._execution_cb(harness, succeeded)
    MobileManipulationRuntime._execution_cb(harness, succeeded)

    assert harness._core.phase is RuntimePhase.RELEASING
    assert harness.release_calls == [10.0]
    assert harness.safety_reasons == []


def test_place_approach_callback_rejects_highwater_replay_after_active() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    harness = _place_approach_callback_harness()
    active = String(data=_execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))
    frozen_highwater = String(data=_execution(
        'succeeded',
        'place_transit',
        8,
        8.0,
    ))
    succeeded = String(data=_execution(
        'succeeded',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))

    MobileManipulationRuntime._execution_cb(harness, active)
    MobileManipulationRuntime._execution_cb(harness, frozen_highwater)
    MobileManipulationRuntime._execution_cb(harness, succeeded)

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.release_calls == []
    assert harness.safety_reasons == [
        'place execution replayed the frozen high-water after active',
    ]


def test_place_planning_callback_fails_immediately_on_executor_change() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.phase = RuntimePhase.PLACE_GROUNDING
            carry = parse_execution_status(
                _execution('succeeded', 'carry', 7, 7.0),
            )
            self._core.place_request_sent(
                place_contract_id=PLACE_GOAL,
                executor_state=carry,
            )
            self._execution_status = carry
            self._execution_status_seen_s = 9.0
            self._latest_gripper_command_id = 0
            self._gripper_feedback = []
            self._trajectory_deadline_s = None
            self.safety_reasons = []

        def _now_s(self) -> float:
            return 10.0

        def _guard_active_posture(self, _now: float) -> bool:
            return True

        def _apply_safety(self, action) -> None:
            if action.reason:
                self.safety_reasons.append(action.reason)

    harness = Harness()

    MobileManipulationRuntime._execution_cb(
        harness,
        String(data=_execution('succeeded', 'carry', 8, 8.0)),
    )

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.safety_reasons == [
        'place planning executor snapshot changed',
    ]


def test_malformed_execution_status_releases_before_actuator_stops() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.node import MobileManipulationRuntime

    class Publisher:
        def __init__(self, name: str, events: list[str]) -> None:
            self._name = name
            self._events = events

        def publish(self, _message) -> None:
            self._events.append(self._name)

    class Logger:
        def __init__(self, events: list[str]) -> None:
            self._events = events

        def error(self, _message: str) -> None:
            self._events.append('log')

    class Harness:
        _apply_safety = MobileManipulationRuntime._apply_safety
        _release_terminal_ownership = (
            MobileManipulationRuntime._release_terminal_ownership
        )

        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._terminal_ownership_released = False
            self.events = []
            self._visual_search_active_pub = Publisher('search_off', self.events)
            self._cancel_nav_pub = Publisher('cancel_nav', self.events)
            self._arm_cancel_pub = Publisher('cancel_arm', self.events)
            self._logger = Logger(self.events)

        def _now_s(self) -> float:
            return 10.0

        def _guard_active_posture(self, _now: float) -> bool:
            return True

        def _invalidate_async_work(self) -> None:
            self.events.append('async')

        def _invalidate_perception_session(self) -> None:
            self.events.append('perception')

        def _reset_execution_occlusion(self) -> None:
            return

        def _publish_zero(self) -> None:
            self.events.append('zero')

        def get_logger(self) -> Logger:
            return self._logger

    harness = Harness()

    MobileManipulationRuntime._execution_cb(harness, String(data=''))

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.events == [
        'async',
        'perception',
        'search_off',
        'zero',
        'cancel_nav',
        'cancel_arm',
        'log',
    ]


def test_destroy_releases_ownership_before_worker_shutdown(monkeypatch) -> None:
    pytest.importorskip('rclpy')
    from rclpy.node import Node
    from z_manip_task.node import MobileManipulationRuntime

    class Publisher:
        def __init__(self, events: list[str]) -> None:
            self._events = events

        def publish(self, _message) -> None:
            self._events.append('cancel_arm')

    class Worker:
        def __init__(self, events: list[str]) -> None:
            self._events = events

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert not wait
            assert cancel_futures
            self._events.append('worker')

    class Harness(MobileManipulationRuntime):
        def __init__(self) -> None:
            self.events = []
            self._terminal_ownership_released = False
            self._arm_cancel_pub = Publisher(self.events)
            self._worker = Worker(self.events)

        def _invalidate_async_work(self) -> None:
            self.events.append('async')

        def _invalidate_perception_session(self) -> None:
            self.events.append('perception')

        def _reset_execution_occlusion(self) -> None:
            return

        def _publish_zero(self) -> None:
            self.events.append('zero')

    monkeypatch.setattr(
        Node,
        'destroy_node',
        lambda self: self.events.append('super') or True,
    )
    harness = Harness()

    result = harness.destroy_node()

    assert result
    assert harness.events == [
        'async', 'perception', 'zero', 'cancel_arm', 'worker', 'super',
    ]


@pytest.mark.parametrize('authorized', (False, True))
def test_wait_fresh_observation_requires_exact_authorization(
    authorized: bool,
) -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    synchronized = SimpleNamespace(serial=4, stamp_s=12.5)

    class Harness:
        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._core.phase = RuntimePhase.WAIT_FRESH_OBSERVATION
            self._core.required_replan_serial = 4
            self._future = None
            self.authorizations = []
            self.started = []

        def _grounding_observation_authorized(self, candidate) -> bool:
            self.authorizations.append(candidate)
            return authorized

        def _semantic_observation(self, serial: int, stamp_s: float):
            return (serial, stamp_s)

        def _start_planning(self, kind: str, observation) -> None:
            self.started.append((kind, observation))

        def _recover_precontact(self, _kind, _detail: str) -> bool:
            raise AssertionError('valid observation should not enter recovery')

        def _apply_safety(self, _action) -> None:
            raise AssertionError('valid observation should not fail')

    harness = Harness()

    MobileManipulationRuntime._wait_fresh_observation_tick(
        harness,
        synchronized,
    )

    assert harness.authorizations == [synchronized]
    if authorized:
        assert harness._core.phase is RuntimePhase.PLANNING
        assert harness.started == [('pregrasp', (4, 12.5))]
    else:
        assert harness._core.phase is RuntimePhase.WAIT_FRESH_OBSERVATION
        assert harness.started == []


def test_reground_timer_rejects_ros_time_rollback() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._core.phase = RuntimePhase.NEAR_GROUNDING
            self._reground_started_at = 10.0
            self._reground_last_tick_at = 10.8
            self.actions = []

        def get_parameter(self, name: str):
            assert name == 'semantic_reground_timeout_s'
            return SimpleNamespace(value=2.0)

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

    harness = Harness()

    MobileManipulationRuntime._reground_tick(harness, 10.5, None)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'clock moved backwards' in harness._core.failure_reason
    assert len(harness.actions) == 1


def test_pose_settle_timer_rejects_ros_time_rollback() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._lookout_pending = False
            self._pose_settle_started_at = 10.0
            self._pose_settle_last_tick_at = 10.8
            self._pose_settle_until = 11.0
            self._visual_search_pending = False
            self.actions = []
            self.zero_commands = 0
            self.status_publishes = 0

        def _now_s(self) -> float:
            return 10.5

        def _guard_active_posture(self, _now: float) -> bool:
            return True

        def _poll_planning(self) -> None:
            return

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

        def _publish_zero(self) -> None:
            self.zero_commands += 1

        def _publish_status(self) -> None:
            self.status_publishes += 1

    harness = Harness()

    MobileManipulationRuntime._tick(harness)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'pose settle clock moved backwards' in harness._core.failure_reason
    assert len(harness.actions) == 1
    assert harness.zero_commands == 1
    assert harness.status_publishes == 1


def test_stationary_wait_keeps_zero_ownership_and_rejects_time_rollback() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._core.begin_visual_search()
            self._core.mark_visual_search_complete()
            self._lookout_pending = False
            self._visual_search_pending = False
            self._pose_settle_started_at = 9.0
            self._pose_settle_last_tick_at = 9.0
            self._pose_settle_until = 10.0
            self._visual_search_settle_reference = SimpleNamespace(
                position_anchor_xy=(1.0, 2.0),
                target_yaw_rad=0.3,
                started_at_s=9.0,
                stop_started_at_s=9.0,
                minimum_odom_sequence=1,
                minimum_odom_stamp_ns=8_000_000_000,
                correction_deadline_s=12.0,
                absolute_deadline_s=13.0,
                stationary_deadline_s=13.0,
                reacquire_count=0,
            )
            self._position_xy = (1.02, 2.0)
            self._yaw = 0.3
            self._odom_seen_at = 9.4
            self._odom_stamp_ns = 9_400_000_000
            self._odom_sequence = 2
            self._base_linear_speed_mps = 0.0744
            self._base_angular_speed_rps = 0.0929
            self._visual_search = SimpleNamespace(config=SimpleNamespace(
                max_planar_drift_m=0.15,
                position_completion_tolerance_m=0.05,
                moving_rebound_reacquire_m=0.10,
                settle_yaw_tolerance_rad=0.0349065850,
                settle_max_linear_speed_mps=0.035,
                settle_max_angular_speed_rps=0.05,
                stationary_quiet_window_s=0.35,
            ))
            self._visual_search_stationarity = ContinuousMotionQuietWindow(
                quiet_window_s=0.35,
                max_odom_gap_s=0.15,
                max_linear_speed_mps=0.035,
                max_angular_speed_rps=0.05,
            )
            self._visual_search_stationarity.reset(
                stop_received_at_s=9.0,
                minimum_odom_sequence=1,
                minimum_odom_stamp_ns=8_000_000_000,
            )
            self.now = 9.5
            self._visual_search_active_pub = SimpleNamespace(messages=[])
            self._visual_search_active_pub.publish = (
                self._visual_search_active_pub.messages.append
            )
            self.actions = []
            self.zero_commands = 0
            self.grounding_requests = 0
            self.status_publishes = 0

        def _now_s(self) -> float:
            return self.now

        def _guard_active_posture(self, _now: float) -> bool:
            return True

        def _poll_planning(self) -> None:
            return

        def _publish_zero(self) -> None:
            self.zero_commands += 1

        def _finish_pose_settle(self, now: float) -> None:
            MobileManipulationRuntime._finish_pose_settle(self, now)

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

        def _publish_grounding_request(self) -> None:
            self.grounding_requests += 1

        def _publish_status(self) -> None:
            self.status_publishes += 1

        def get_parameter(self, name: str):
            assert name == 'visual_search_odom_timeout_s'
            return SimpleNamespace(value=0.5)

    harness = Harness()

    MobileManipulationRuntime._tick(harness)
    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness.zero_commands == 1
    assert harness._visual_search_active_pub.messages[-1].data is True

    harness.now = 10.0
    harness._odom_seen_at = 9.9
    harness._odom_stamp_ns = 9_900_000_000
    harness._odom_sequence = 3
    MobileManipulationRuntime._tick(harness)
    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search_settle_reference is not None
    assert harness.zero_commands == 2
    assert all(
        message.data is True
        for message in harness._visual_search_active_pub.messages
    )
    assert harness.actions == []

    harness.now = 10.5
    harness._odom_seen_at = 10.4
    harness._odom_stamp_ns = 10_400_000_000
    harness._odom_sequence = 4
    MobileManipulationRuntime._tick(harness)
    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness.zero_commands == 3

    harness.now = 10.4
    MobileManipulationRuntime._tick(harness)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'pose settle clock moved backwards' in harness._core.failure_reason
    assert harness.grounding_requests == 0
    assert harness.zero_commands == 4
    assert len(harness.actions) == 1


def test_planner_recovery_dispatches_lookout_before_pose_settle_validation() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message.data)

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._core.phase = RuntimePhase.STANDOFF
            self._lookout_pending = False
            self._pose_settle_started_at = None
            self._pose_settle_last_tick_at = None
            self._pose_settle_until = None
            self._named_pose_pub = Publisher()
            self.status_publishes = 0

        def _now_s(self) -> float:
            return 10.0

        def _guard_active_posture(self, _now: float) -> bool:
            return True

        def _poll_planning(self) -> None:
            self._core.restart_grounding()
            self._lookout_pending = True

        def _dispatch_pending_lookout(self, now: float) -> None:
            MobileManipulationRuntime._dispatch_pending_lookout(self, now)

        def get_parameter(self, name: str):
            assert name == 'lookout_settle_s'
            return SimpleNamespace(value=3.0)

        def _topic_value(self, name: str) -> str:
            assert name == 'lookout_pose'
            return 'LOOKOUT'

        def _publish_status(self) -> None:
            self.status_publishes += 1

    harness = Harness()

    MobileManipulationRuntime._tick(harness)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert not harness._lookout_pending
    assert harness._named_pose_pub.messages == ['LOOKOUT']
    assert harness._pose_settle_started_at == pytest.approx(10.0)
    assert harness._pose_settle_last_tick_at == pytest.approx(10.0)
    assert harness._pose_settle_until == pytest.approx(13.0)
    assert harness.status_publishes == 1


def test_pending_lookout_cannot_dispatch_after_terminal_transition() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    harness = SimpleNamespace(
        _core=RuntimeSafetyCore(),
        _lookout_pending=True,
        _pose_settle_started_at=1.0,
        _pose_settle_last_tick_at=1.0,
        _pose_settle_until=2.0,
        _named_pose_pub=Publisher(),
    )
    harness._core.begin('pick the bottle')
    harness._core.fail('terminal test')

    MobileManipulationRuntime._dispatch_pending_lookout(harness, 1.5)

    assert not harness._lookout_pending
    assert harness._pose_settle_started_at is None
    assert harness._pose_settle_last_tick_at is None
    assert harness._pose_settle_until is None
    assert harness._named_pose_pub.messages == []


def test_odometry_source_sequence_ignores_duplicates_and_rejects_rollback() -> None:
    pytest.importorskip('rclpy')
    from nav_msgs.msg import Odometry
    from z_manip_task.node import MobileManipulationRuntime

    class PostureGuard:
        def update(self, *_args, **_kwargs) -> None:
            return

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._roll = 0.0
            self._pitch = 0.0
            self._yaw = None
            self._position_xy = None
            self._odom_seen_at = None
            self._odom_stamp_ns = None
            self._odom_payload = None
            self._odom_sequence = 0
            self._base_linear_speed_mps = None
            self._base_angular_speed_rps = None
            self._nav_speed = float('inf')
            self._posture_guard = PostureGuard()
            self.failures = []
            self.now = 10.5

        def _now_s(self) -> float:
            return self.now

        def get_parameter(self, name: str):
            return SimpleNamespace(value={
                'platform_odometry_parent_frame': 'map',
                'platform_odometry_child_frame': 'base_link',
            }[name])

        def _guard_active_posture(self, _now: float) -> bool:
            return False

        def _fail_posture(self, reason: str) -> None:
            self.failures.append(reason)

    def message(stamp_ns: int, *, x: float = 0.0) -> Odometry:
        result = Odometry()
        result.header.frame_id = 'map'
        result.child_frame_id = 'base_link'
        result.header.stamp.sec = stamp_ns // 1_000_000_000
        result.header.stamp.nanosec = stamp_ns % 1_000_000_000
        result.pose.pose.orientation.w = 1.0
        result.pose.pose.position.x = x
        return result

    harness = Harness()
    first = message(10_000_000_000)
    second = message(10_100_000_000)

    MobileManipulationRuntime._odom_cb(harness, first)
    first_seen_at = harness._odom_seen_at
    harness.now = 10.9
    MobileManipulationRuntime._odom_cb(harness, first)
    assert harness._odom_sequence == 1
    assert harness._odom_seen_at == first_seen_at

    MobileManipulationRuntime._odom_cb(
        harness,
        message(10_000_000_000, x=0.1),
    )
    assert harness._odom_sequence == 1
    assert harness.failures == [
        'platform odometry payload changed at the same source stamp',
    ]

    MobileManipulationRuntime._odom_cb(harness, second)
    assert harness._odom_sequence == 2
    assert harness._odom_stamp_ns == 10_100_000_000

    MobileManipulationRuntime._odom_cb(harness, first)
    assert harness._odom_sequence == 2
    assert harness.failures == [
        'platform odometry payload changed at the same source stamp',
        'platform odometry source time moved backwards',
    ]


def test_joint_source_duplicate_conflict_and_rollback_are_fail_closed() -> None:
    pytest.importorskip('rclpy')
    import numpy as np
    from sensor_msgs.msg import JointState
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._planner = SimpleNamespace(
                chain=SimpleNamespace(joint_names=('j1', 'j2')),
            )
            self._joint_stamp_ns = None
            self._joint_sequence = 0
            self._joint_state = None
            self._joint_history = []
            self.now = 10.1
            self.actions = []

        def _now_s(self) -> float:
            return self.now

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

    def message(stamp_ns: int, positions=(0.1, 0.2)) -> JointState:
        result = JointState()
        result.header.stamp.sec = stamp_ns // 1_000_000_000
        result.header.stamp.nanosec = stamp_ns % 1_000_000_000
        result.name = ['j1', 'j2']
        result.position = list(positions)
        return result

    duplicate = Harness()
    first = message(10_000_000_000)
    MobileManipulationRuntime._joint_cb(duplicate, first)
    received_at = duplicate._joint_history[-1].received_at_s
    duplicate.now = 10.2
    MobileManipulationRuntime._joint_cb(duplicate, first)
    assert duplicate._joint_sequence == 1
    assert len(duplicate._joint_history) == 1
    assert duplicate._joint_history[-1].received_at_s == received_at

    MobileManipulationRuntime._joint_cb(
        duplicate,
        message(10_000_000_000, (0.1, 0.25)),
    )
    assert duplicate._core.phase is RuntimePhase.FAILED
    assert 'same source stamp' in duplicate._core.failure_reason
    assert np.array_equal(duplicate._joint_state, (0.1, 0.2))

    rollback = Harness()
    MobileManipulationRuntime._joint_cb(
        rollback,
        message(10_100_000_000),
    )
    MobileManipulationRuntime._joint_cb(
        rollback,
        message(10_200_000_000, (0.2, 0.3)),
    )
    MobileManipulationRuntime._joint_cb(
        rollback,
        message(10_100_000_000),
    )
    assert rollback._core.phase is RuntimePhase.FAILED
    assert 'source time moved backwards' in rollback._core.failure_reason
    assert rollback._joint_stamp_ns == 10_200_000_000
    assert rollback._joint_sequence == 2
    assert len(rollback._joint_history) == 2
