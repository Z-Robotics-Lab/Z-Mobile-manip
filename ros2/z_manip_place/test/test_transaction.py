"""Placement transaction protocol and dual-clock lifecycle tests."""

import json

import pytest

from z_manip_place.transaction import (
    parse_region_transaction_identity,
    parse_transaction_control,
    PlacementTransactionLifecycle,
    PlaceTerminalFailure,
    PlaceTransactionControl,
    PlaceTransactionError,
    PlaceTransactionIdentity,
)


def _identity(goal: str = 'place-7', epoch: str = 'executor-a'):
    return PlaceTransactionIdentity(goal, epoch)


def _control(goal: str = 'place-7', epoch: str = 'executor-a'):
    return PlaceTransactionControl('abort', _identity(goal, epoch))


def _armed(*, ros_timeout_s: float = 3.0, wall_timeout_s: float = 5.0):
    lifecycle = PlacementTransactionLifecycle(
        ros_timeout_s=ros_timeout_s,
        wall_timeout_s=wall_timeout_s,
    )
    token = lifecycle.begin(
        _identity(),
        now_ros_ns=10_000_000_000,
        now_wall_s=20.0,
    )
    lifecycle.start_planning(token)
    lifecycle.arm(token, now_ros_ns=10_000_000_000, now_wall_s=20.0)
    return lifecycle


def test_control_parser_is_strict_and_round_trips_exact_identity():
    control = parse_transaction_control(_control().to_json())

    assert control == _control()
    with pytest.raises(PlaceTransactionError, match='duplicate JSON key'):
        parse_transaction_control(
            '{"schema":"z_manip.place_transaction_control.v1",'
            '"action":"abort","goal_id":"a","goal_id":"b",'
            '"executor_epoch":"e"}',
        )
    payload = _control().to_payload()
    payload['extension'] = True
    with pytest.raises(PlaceTransactionError, match='keys mismatch'):
        parse_transaction_control(json.dumps(payload))
    payload.pop('extension')
    payload['action'] = 'reset'
    with pytest.raises(PlaceTransactionError, match='must be abort'):
        parse_transaction_control(json.dumps(payload))


def test_invalid_region_failure_can_recover_only_unambiguous_v2_identity():
    payload = json.dumps({
        'schema_version': 2,
        'goal_id': 'place-7',
        'executor_epoch': 'executor-a',
        'malformed_body': True,
    })
    assert parse_region_transaction_identity(payload) == _identity()

    with pytest.raises(PlaceTransactionError, match='duplicate JSON key'):
        parse_region_transaction_identity(
            '{"schema_version":2,"goal_id":"place-7",'
            '"goal_id":"foreign","executor_epoch":"executor-a"}',
        )
    with pytest.raises(PlaceTransactionError, match='schema'):
        parse_region_transaction_identity(json.dumps({
            'schema_version': 1,
            'goal_id': 'place-7',
            'executor_epoch': 'executor-a',
        }))


def test_foreign_abort_is_ignored_but_exact_abort_unblocks_retry():
    lifecycle = _armed()

    assert not lifecycle.abort(_control(epoch='executor-b'))
    assert lifecycle.armed
    assert lifecycle.abort(_control())
    assert not lifecycle.active

    retry = lifecycle.begin(
        _identity('place-8'),
        now_ros_ns=12_000_000_000,
        now_wall_s=22.0,
    )
    lifecycle.start_planning(retry)
    assert lifecycle.matches(retry)


def test_planning_abort_invalidates_worker_token_before_arm_or_publish():
    lifecycle = PlacementTransactionLifecycle(
        ros_timeout_s=3.0,
        wall_timeout_s=5.0,
    )
    token = lifecycle.begin(_identity(), now_ros_ns=10, now_wall_s=20.0)
    lifecycle.start_planning(token)

    assert lifecycle.abort(_control())
    assert not lifecycle.matches(token)
    retry = lifecycle.begin(
        _identity('place-8'),
        now_ros_ns=11,
        now_wall_s=20.1,
    )
    lifecycle.start_planning(retry)
    with pytest.raises(PlaceTransactionError, match='cannot arm'):
        lifecycle.arm(token, now_ros_ns=10, now_wall_s=20.0)
    assert lifecycle.matches(retry)


def test_ros_timeout_wall_timeout_and_clock_rollback_fail_closed():
    ros = _armed(ros_timeout_s=1.0, wall_timeout_s=20.0)
    assert ros.watchdog_reason(
        now_ros_ns=11_000_000_001,
        now_wall_s=20.1,
    ) == 'placement transaction armed ROS-time deadline exceeded'
    assert not ros.active

    wall = _armed(ros_timeout_s=20.0, wall_timeout_s=1.0)
    assert wall.watchdog_reason(
        now_ros_ns=10_100_000_000,
        now_wall_s=21.01,
    ) == 'placement transaction armed wall-time deadline exceeded'
    assert not wall.active

    rollback = _armed()
    assert rollback.watchdog_reason(
        now_ros_ns=10_100_000_000,
        now_wall_s=20.1,
    ) is None
    assert rollback.watchdog_reason(
        now_ros_ns=10_099_999_999,
        now_wall_s=20.2,
    ) == 'placement transaction ROS clock moved backwards'
    assert not rollback.active

    wall_rollback = _armed()
    assert wall_rollback.watchdog_reason(
        now_ros_ns=10_100_000_000,
        now_wall_s=19.9,
    ) == 'placement transaction wall clock moved backwards'
    assert not wall_rollback.active


def test_pending_planning_and_armed_share_the_begin_deadline():
    pending = PlacementTransactionLifecycle(
        ros_timeout_s=1.0,
        wall_timeout_s=2.0,
    )
    pending.begin(_identity(), now_ros_ns=100, now_wall_s=10.0)
    assert pending.watchdog_reason(
        now_ros_ns=100,
        now_wall_s=12.0,
    ) == 'placement transaction pending wall-time deadline exceeded'

    planning = PlacementTransactionLifecycle(
        ros_timeout_s=1.0,
        wall_timeout_s=2.0,
    )
    token = planning.begin(_identity(), now_ros_ns=100, now_wall_s=10.0)
    planning.start_planning(token)
    assert planning.watchdog_reason(
        now_ros_ns=1_000_000_100,
        now_wall_s=10.1,
    ) == 'placement transaction planning ROS-time deadline exceeded'

    armed = PlacementTransactionLifecycle(
        ros_timeout_s=10.0,
        wall_timeout_s=1.0,
    )
    token = armed.begin(_identity(), now_ros_ns=100, now_wall_s=10.0)
    armed.start_planning(token)
    armed.arm(token, now_ros_ns=200, now_wall_s=10.9)
    assert armed.watchdog_reason(
        now_ros_ns=300,
        now_wall_s=11.0,
    ) == 'placement transaction armed wall-time deadline exceeded'


def test_worker_cannot_arm_or_publish_after_begin_deadline():
    lifecycle = PlacementTransactionLifecycle(
        ros_timeout_s=1.0,
        wall_timeout_s=2.0,
    )
    token = lifecycle.begin(_identity(), now_ros_ns=100, now_wall_s=10.0)
    lifecycle.start_planning(token)

    with pytest.raises(PlaceTransactionError, match='deadline exceeded before arm'):
        lifecycle.arm(
            token,
            now_ros_ns=1_000_000_100,
            now_wall_s=10.1,
        )
    assert lifecycle.state == 'planning'


def test_terminal_failure_payload_is_minimal_correlated_v2():
    failure = PlaceTerminalFailure(_identity(), 'retreat acknowledgement missing')
    assert json.loads(failure.to_json()) == {
        'schema': 'z_manip.place_status.v2',
        'state': 'failed',
        'terminal': True,
        'goal_id': 'place-7',
        'executor_epoch': 'executor-a',
        'reason': 'retreat acknowledgement missing',
    }


@pytest.mark.parametrize('name,value', [
    ('ros_timeout_s', 0.0),
    ('ros_timeout_s', float('nan')),
    ('wall_timeout_s', -1.0),
    ('wall_timeout_s', float('inf')),
])
def test_deadlines_must_be_finite_and_positive(name, value):
    values = {'ros_timeout_s': 1.0, 'wall_timeout_s': 1.0}
    values[name] = value
    with pytest.raises(PlaceTransactionError, match=name):
        PlacementTransactionLifecycle(**values)
