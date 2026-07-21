"""Task-side placement abort and terminal-failure protocol tests."""

import json
import threading
from types import SimpleNamespace

import pytest

from z_manip_task.place_transaction import (
    parse_terminal_place_status,
    PlaceTransactionProtocolError,
    transaction_abort_json,
)


def _failure(**updates):
    value = {
        'schema': 'z_manip.place_status.v2',
        'state': 'failed',
        'terminal': True,
        'goal_id': 'place-7',
        'executor_epoch': 'executor-a',
        'reason': 'armed transaction timed out',
    }
    value.update(updates)
    return value


def test_abort_payload_is_exact_and_rejects_empty_identity():
    assert json.loads(transaction_abort_json(
        goal_id='place-7',
        executor_epoch='executor-a',
    )) == {
        'schema': 'z_manip.place_transaction_control.v1',
        'action': 'abort',
        'goal_id': 'place-7',
        'executor_epoch': 'executor-a',
    }
    with pytest.raises(PlaceTransactionProtocolError, match='goal_id'):
        transaction_abort_json(goal_id='', executor_epoch='executor-a')


def test_terminal_status_parser_accepts_only_exact_v2_failure():
    status = parse_terminal_place_status(json.dumps(_failure()))
    assert status is not None
    assert status.goal_id == 'place-7'
    assert status.executor_epoch == 'executor-a'
    assert status.reason == 'armed transaction timed out'

    assert parse_terminal_place_status(json.dumps({
        'state': 'planned',
        'detail': 'legacy informational status',
    })) is None


def test_terminal_status_rejects_duplicates_extensions_and_nonterminal_v2():
    with pytest.raises(PlaceTransactionProtocolError, match='duplicate JSON key'):
        parse_terminal_place_status(
            '{"schema":"z_manip.place_status.v2","state":"failed",'
            '"terminal":true,"goal_id":"place-7","goal_id":"place-8",'
            '"executor_epoch":"executor-a","reason":"failed"}',
        )
    with pytest.raises(PlaceTransactionProtocolError, match='keys mismatch'):
        parse_terminal_place_status(json.dumps(_failure(extension='forbidden')))
    with pytest.raises(PlaceTransactionProtocolError, match='terminal failure'):
        parse_terminal_place_status(json.dumps(_failure(state='planned')))


def test_task_publishes_transaction_abort_exactly_once():
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    harness = SimpleNamespace(
        _place_transaction_requested=True,
        _place_transaction_abort_sent=False,
        _place_goal_id='place-7',
        _core=SimpleNamespace(place_executor_epoch='executor-a'),
        _place_transaction_control_pub=Publisher(),
    )

    MobileManipulationRuntime._publish_place_abort_once(harness)
    MobileManipulationRuntime._publish_place_abort_once(harness)

    assert len(harness._place_transaction_control_pub.messages) == 1
    assert json.loads(
        harness._place_transaction_control_pub.messages[0].data,
    ) == {
        'schema': 'z_manip.place_transaction_control.v1',
        'action': 'abort',
        'goal_id': 'place-7',
        'executor_epoch': 'executor-a',
    }


@pytest.mark.parametrize('phase_name', [
    'PLACE_PLANNING',
    'PLACE_TRANSIT',
    'PLACE_APPROACH',
    'RELEASING',
    'PLACE_RETREAT',
    'POST_RELEASE_VERIFICATION',
])
def test_exact_terminal_failure_stops_every_place_phase(phase_name):
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.core import RuntimePhase, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self):
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('place the observed object')
            self._core.phase = getattr(RuntimePhase, phase_name)
            self._core.place_executor_epoch = 'executor-a'
            self._place_goal_id = 'place-7'
            self.actions = []
            self.statuses = 0

        def _apply_safety(self, action):
            self.actions.append(action)

        def _publish_status(self, *, force=False):
            self.statuses += int(force)

    harness = Harness()
    MobileManipulationRuntime._place_status_cb(
        harness,
        String(data=json.dumps(_failure(reason='late executor failure'))),
    )

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness._core.failure_reason.endswith('late executor failure')
    assert len(harness.actions) == 1
    assert harness.actions[0].cancel_arm
    assert harness.statuses == 1


def test_foreign_terminal_failure_is_ignored_after_planning():
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.core import RuntimePhase, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self):
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('place')
            self._core.phase = RuntimePhase.PLACE_APPROACH
            self._core.place_executor_epoch = 'executor-a'
            self._place_goal_id = 'place-7'

        def _apply_safety(self, _action):
            raise AssertionError('foreign status must not stop this task')

    harness = Harness()
    MobileManipulationRuntime._place_status_cb(
        harness,
        String(data=json.dumps(_failure(goal_id='place-other'))),
    )
    MobileManipulationRuntime._place_status_cb(
        harness,
        String(data=json.dumps(_failure(executor_epoch='executor-other'))),
    )

    assert harness._core.phase is RuntimePhase.PLACE_APPROACH


def test_place_planning_wall_timeout_fires_while_ros_time_is_paused(monkeypatch):
    pytest.importorskip('rclpy')
    from z_manip_task.core import RuntimePhase, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self):
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('place')
            self._core.phase = RuntimePhase.PLACE_PLANNING
            self._place_planning_started_wall_s = 10.0
            self.actions = []
            self.statuses = 0

        @staticmethod
        def get_parameter(_name):
            return SimpleNamespace(value=5.0)

        def _apply_safety(self, action):
            self.actions.append(action)

        def _publish_status(self, *, force=False):
            self.statuses += int(force)

    monkeypatch.setattr('z_manip_task.node.time.monotonic', lambda: 15.0)
    harness = Harness()
    MobileManipulationRuntime._place_planning_wall_timeout_tick(harness)

    assert harness._core.phase is RuntimePhase.FAILED
    assert harness._core.failure_reason == 'placement planning wall timeout'
    assert len(harness.actions) == 1
    assert harness.statuses == 1
