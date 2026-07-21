"""Tests for runtime safety invariants."""

import json
import math
from types import SimpleNamespace

import pytest

from z_manip.trajectory_digest import canonical_joint_trajectory_sha256

from z_manip_task.core import (
    base_twist_speed_magnitudes,
    BoundedYawSearch,
    ContinuousMotionQuietWindow,
    grasp_close_aperture,
    horizontal_edge_direction,
    ObservationSerialGate,
    parse_execution_status,
    parse_place_contract,
    RuntimePhase,
    RuntimeSafetyCore,
    split_placement_trajectory,
    trajectory_segment_frame_id,
    validate_grasp_aperture_contract,
    validate_place_trajectory_content,
    validate_platform_odometry_frames,
    validate_position_hold_frame_contract,
    vertical_edge_direction,
    VisualSearchConfig,
)


PLACE_GOAL = 'place-7-1000000000'
EXECUTOR_EPOCH = 'executor-epoch-a'
TRAJECTORY_TOKEN = 'trajectory-current'


def _place_executor_snapshot(**updates):
    snapshot = {
        'goal_id': PLACE_GOAL,
        'trajectory_contract_id': PLACE_GOAL,
        'executor_epoch': EXECUTOR_EPOCH,
        'trajectory_command_highwater': 7,
        'trajectory_source_highwater_ns': 7_000_000_000,
        'gripper_command_highwater': 3,
        'gripper_source_highwater_ns': 6_500_000_000,
    }
    snapshot.update(updates)
    return snapshot


def _place_contract_payload(**updates):
    contract = {
        'schema': 'z_manip.place_contract.v2',
        'schema_version': 2,
        'goal_id': PLACE_GOAL,
        'frame_id': 'base_link',
        'joint_names': ['joint1', 'joint2'],
        'phase_start_indices': {'transit': 0, 'approach': 2, 'retreat': 4},
        'point_count': 6,
        'trajectory_topic': '/z_manip/place/trajectory',
        'trajectory_digest_sha256': '0' * 64,
        'request_id': 'request-a',
        'producer_epoch': 'perception-a',
        'generation': 4,
        'observation_stamp_ns': 5_000_000_000,
        'observation_frame_id': 'wrist_camera_optical_frame',
        **_place_executor_snapshot(),
    }
    contract.update(updates)
    return contract


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
    event_token: str | None = None,
    event_received_at: float | None = None,
    suffix: str = '',
):
    event_fields = ''
    if event_token is not None or event_received_at is not None:
        assert event_token is not None and event_received_at is not None
        event_fields = (
            f';trajectory_event_token={event_token};'
            f'trajectory_event_received_at={event_received_at:.6f}'
        )
    return parse_execution_status(
        f'{trajectory};owner=trajectory;segment={segment};'
        f'command_id={command_id};trajectory_contract_id={contract_id};'
        f'executor_epoch={executor_epoch};'
        f'trajectory_token={trajectory_token};'
        f'trajectory_received_at={received_at:.6f};'
        f'gripper_command_id={gripper_command_id};'
        f'gripper_received_at={gripper_received_at:.6f}'
        f'{event_fields}{suffix}',
    )


def _send_non_place(
    core: RuntimeSafetyCore,
    segment: str,
    *,
    executor_epoch: str = EXECUTOR_EPOCH,
    published_at_s: float = 0.0,
    trajectory_token: str = TRAJECTORY_TOKEN,
) -> None:
    core.trajectory_sent(
        segment,
        executor_epoch=executor_epoch,
        published_at_s=published_at_s,
        trajectory_token=trajectory_token,
    )


def _seed_executor_highwater(
    core: RuntimeSafetyCore,
    command_id: int,
    received_at: float,
    *,
    executor_epoch: str = EXECUTOR_EPOCH,
) -> None:
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(
        core,
        'transit',
        executor_epoch=executor_epoch,
        published_at_s=max(0.0, received_at - 0.1),
    )
    core.execution_update(_execution(
        'active',
        'transit',
        command_id,
        received_at,
        executor_epoch=executor_epoch,
    ))
    core.execution_update(_execution(
        'succeeded',
        'transit',
        command_id,
        received_at,
        executor_epoch=executor_epoch,
    ))


def _assert_execution_binding_cleared(core):
    assert core.execution_segment == ''
    assert not core.execution_seen_active
    assert core.expected_command_id is None
    assert core.expected_executor_epoch == ''
    assert core.execution_publish_executor_epoch == ''
    assert core.execution_publish_token == ''
    assert core.expected_trajectory_received_at is None
    assert core.minimum_trajectory_received_at is None
    assert core.execution_command_highwater_snapshot == {}


def test_place_contract_v2_parser_accepts_exact_executor_snapshot():
    parsed = parse_place_contract(json.dumps(_place_contract_payload()))

    assert parsed['schema'] == 'z_manip.place_contract.v2'
    assert parsed['executor_epoch'] == EXECUTOR_EPOCH
    assert parsed['trajectory_contract_id'] == PLACE_GOAL
    assert parsed['trajectory_command_highwater'] == 7
    assert parsed['gripper_source_highwater_ns'] == 6_500_000_000
    assert parsed['trajectory_digest_sha256'] == '0' * 64


def _place_trajectory():
    points = []
    for index in range(6):
        points.append(SimpleNamespace(
            positions=[0.1 * index, -0.2 * index],
            velocities=[],
            accelerations=[],
            effort=[],
            time_from_start=SimpleNamespace(sec=index, nanosec=index * 1000),
        ))
    return SimpleNamespace(
        header=SimpleNamespace(
            frame_id='base_link',
            stamp=SimpleNamespace(sec=12, nanosec=345),
        ),
        joint_names=['joint1', 'joint2'],
        points=points,
    )


def _bound_place_contract(trajectory=None, **updates):
    message = _place_trajectory() if trajectory is None else trajectory
    digest = canonical_joint_trajectory_sha256(
        frame_id=message.header.frame_id,
        header_stamp=message.header.stamp,
        joint_names=message.joint_names,
        points=message.points,
    )
    return _place_contract_payload(
        trajectory_digest_sha256=digest,
        **updates,
    )


def test_place_trajectory_content_accepts_only_the_exact_separate_message():
    trajectory = _place_trajectory()
    contract = _bound_place_contract(trajectory)

    validate_place_trajectory_content(
        contract,
        trajectory,
        expected_topic='/z_manip/place/trajectory',
    )


@pytest.mark.parametrize(
    'mutation',
    (
        'joint_name',
        'position',
        'velocity',
        'acceleration',
        'effort',
        'time',
        'header_stamp',
    ),
)
def test_place_trajectory_content_rejects_stale_or_mutated_payload(mutation):
    expected = _place_trajectory()
    contract = _bound_place_contract(expected)
    received = _place_trajectory()
    if mutation == 'joint_name':
        received.joint_names[1] = 'foreign_joint'
    elif mutation == 'position':
        received.points[2].positions[0] += 0.01
    elif mutation == 'velocity':
        received.points[2].velocities = [0.0, 0.0]
    elif mutation == 'acceleration':
        received.points[2].accelerations = [0.0, 0.0]
    elif mutation == 'effort':
        received.points[2].effort = [0.0, 0.0]
    elif mutation == 'time':
        received.points[2].time_from_start.nanosec += 1
    else:
        received.header.stamp.nanosec += 1

    with pytest.raises(ValueError, match='digest'):
        validate_place_trajectory_content(
            contract,
            received,
            expected_topic='/z_manip/place/trajectory',
        )


def test_place_trajectory_content_rejects_frame_topic_and_contract_digest():
    trajectory = _place_trajectory()
    contract = _bound_place_contract(trajectory)

    foreign_frame = _place_trajectory()
    foreign_frame.header.frame_id = 'foreign_base'
    with pytest.raises(ValueError, match='frame'):
        validate_place_trajectory_content(
            contract,
            foreign_frame,
            expected_topic='/z_manip/place/trajectory',
        )

    with pytest.raises(ValueError, match='topic'):
        validate_place_trajectory_content(
            contract,
            trajectory,
            expected_topic='/foreign/place/trajectory',
        )

    tampered = dict(contract, trajectory_digest_sha256='f' * 64)
    with pytest.raises(ValueError, match='digest'):
        validate_place_trajectory_content(
            tampered,
            trajectory,
            expected_topic='/z_manip/place/trajectory',
        )


@pytest.mark.parametrize(
    'digest',
    ('0' * 63, 'G' * 64, 'A' * 64),
)
def test_place_contract_v2_rejects_noncanonical_digest(digest):
    with pytest.raises(ValueError, match='lowercase SHA-256'):
        parse_place_contract(json.dumps(_place_contract_payload(
            trajectory_digest_sha256=digest,
        )))


def test_place_contract_v2_rejects_missing_and_duplicate_digest_field():
    missing = _place_contract_payload()
    del missing['trajectory_digest_sha256']
    with pytest.raises(ValueError, match='fields are not exact'):
        parse_place_contract(json.dumps(missing))

    raw = json.dumps(_place_contract_payload())
    duplicate = raw[:-1] + ',"trajectory_digest_sha256":"' + 'f' * 64 + '"}'
    with pytest.raises(ValueError, match='repeats field'):
        parse_place_contract(duplicate)


@pytest.mark.parametrize('field', tuple(_place_executor_snapshot()))
def test_place_contract_v2_rejects_each_missing_executor_snapshot_field(field):
    contract = _place_contract_payload()
    del contract[field]

    with pytest.raises(ValueError, match='fields are not exact'):
        parse_place_contract(json.dumps(contract))


@pytest.mark.parametrize('version', (1, 2.0, True))
def test_place_contract_v2_rejects_invalid_unknown_duplicate_and_nonfinite_fields(
    version,
):
    old = _place_contract_payload(schema_version=version)
    with pytest.raises(ValueError, match='unsupported'):
        parse_place_contract(json.dumps(old))

    unknown = _place_contract_payload(legacy_executor_snapshot='unsafe')
    with pytest.raises(ValueError, match='unknown'):
        parse_place_contract(json.dumps(unknown))

    raw = json.dumps(_place_contract_payload())
    duplicate = raw[:-1] + ',"executor_epoch":"other"}'
    with pytest.raises(ValueError, match='repeats field'):
        parse_place_contract(duplicate)

    nonfinite = raw.replace('6500000000', 'NaN')
    with pytest.raises(ValueError, match='non-finite'):
        parse_place_contract(nonfinite)


@pytest.mark.parametrize(
    ('field', 'mutated'),
    (
        ('goal_id', 'old-place-goal'),
        ('trajectory_contract_id', 'other-place-goal'),
        ('executor_epoch', 'restarted-executor'),
        ('trajectory_command_highwater', 6),
        ('trajectory_source_highwater_ns', 6_999_999_999),
        ('gripper_command_highwater', 2),
        ('gripper_source_highwater_ns', 6_499_999_999),
    ),
)
def test_place_plan_rejects_each_mutated_executor_snapshot_field(field, mutated):
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    core.place_request_sent(
        place_contract_id=PLACE_GOAL,
        executor_state=_execution('succeeded', 'carry', 7, 7.0),
    )
    contract = _place_executor_snapshot(**{field: mutated})

    with pytest.raises(ValueError, match=field):
        core.place_plan_ready(contract)

    assert core.phase is RuntimePhase.PLACE_PLANNING


def _core_waiting_for_place_approach() -> tuple[
    RuntimeSafetyCore,
    object,
]:
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    carry = _execution('succeeded', 'carry', 7, 7.0)
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
    core.execution_update(_execution('active', 'place_transit', 8, 8.0))
    transit_succeeded = _execution(
        'succeeded',
        'place_transit',
        8,
        8.0,
    )
    core.execution_update(transit_succeeded)
    core.trajectory_sent(
        'place_approach',
        place_contract_id=PLACE_GOAL,
        executor_state=transit_succeeded,
        trajectory_token=TRAJECTORY_TOKEN,
    )
    return core, transit_succeeded


def test_place_execution_metadata_carries_bounded_contract_identity():
    assert trajectory_segment_frame_id(
        'place_approach',
        'place-7-1000000000',
    ) == 'place_approach|contract=place-7-1000000000'
    assert trajectory_segment_frame_id(
        'place_retreat',
        'place-7-1000000000',
    ) == 'place_retreat|contract=place-7-1000000000'
    assert trajectory_segment_frame_id('carry', None) == 'carry'
    assert trajectory_segment_frame_id(
        'approach',
        None,
        execution_token='trajectory-abc',
    ) == 'approach|token=trajectory-abc'
    assert trajectory_segment_frame_id(
        'place_approach',
        'place-7-1000000000',
        execution_token='trajectory-abc',
    ) == (
        'place_approach|contract=place-7-1000000000|token=trajectory-abc'
    )


@pytest.mark.parametrize(
    'contract_id',
    (
        '', 'contains whitespace', 'contains;delimiter', 'contains|delimiter',
        'contains=delimiter', 'x' * 129,
    ),
)
def test_place_execution_metadata_rejects_ambiguous_contract_id(contract_id):
    with pytest.raises(ValueError, match='contract ID'):
        trajectory_segment_frame_id('place_retreat', contract_id)


def test_observation_gate_requires_sync_freshness_and_new_versions():
    gate = ObservationSerialGate(sync_slop_s=0.1, max_age_s=0.3)
    gate.update('target', 1.00)
    gate.update('target_cloud', 1.02)
    gate.update('scene_cloud', 1.04)
    first = gate.snapshot(1.05)
    assert first is not None and first.serial == 1
    assert gate.snapshot(1.06).serial == 1
    gate.update('target', 1.10)
    gate.update('target_cloud', 1.11)
    gate.update('scene_cloud', 1.12)
    assert gate.snapshot(1.13).serial == 2
    assert gate.snapshot(1.50) is None


def test_exact_pipeline_rejects_adjacent_frame_splicing():
    gate = ObservationSerialGate(sync_slop_s=1e-6, max_age_s=0.3)
    gate.update('target', 1.0)
    gate.update('target_cloud', 1.0)
    gate.update('scene_cloud', 1.1)
    assert gate.snapshot(1.1) is None


def test_duplicate_callback_at_same_frame_stamp_is_idempotent():
    gate = ObservationSerialGate(sync_slop_s=1e-6, max_age_s=0.3)
    for stream in gate.streams:
        gate.update(stream, 1.0)
    assert gate.snapshot(1.0).serial == 1
    gate.update('target', 1.0)
    assert gate.snapshot(1.0).serial == 1


def test_post_servo_plan_uses_first_exact_bundle_from_new_grounding_session():
    core = RuntimeSafetyCore()
    core.begin('pick the requested item')
    assert core.phase is RuntimePhase.POSE_SETTLE
    core.mark_pose_settled()
    core.mark_standoff(7)
    assert core.phase is RuntimePhase.COARSE_NAV
    core.mark_coarse_ready()
    assert core.phase is RuntimePhase.NEAR_GROUNDING
    core.mark_near_grounded(9)
    core.mark_servo_complete_for_reground()
    assert core.phase is RuntimePhase.FINAL_GROUNDING
    assert core.required_replan_serial == 1
    with pytest.raises(RuntimeError, match='newly synchronized'):
        core.begin_replan(0)
    core.begin_replan(1)
    core.plan_ready()
    assert core.phase is RuntimePhase.TRANSIT


def test_executor_must_be_active_before_succeeded_advances():
    core = RuntimeSafetyCore()
    core.begin('pick')
    core.mark_pose_settled()
    core.mark_standoff(1)
    core.mark_coarse_ready()
    core.mark_near_grounded(1)
    core.mark_servo_complete_for_reground()
    core.begin_replan(1)
    core.plan_ready()
    _send_non_place(core, 'transit')
    core.execution_update(_execution('succeeded', 'transit', 1, 1.0))
    assert core.phase is RuntimePhase.TRANSIT
    assert core.expected_command_id is None
    core.execution_update(_execution('active', 'transit', 2, 2.0))
    assert core.expected_command_id == 2
    assert core.expected_executor_epoch == EXECUTOR_EPOCH
    assert core.expected_trajectory_received_at == pytest.approx(2.0)
    core.execution_update(_execution('succeeded', 'transit', 2, 2.0))
    assert core.phase is RuntimePhase.PREGRASP_REOBSERVE
    _assert_execution_binding_cleared(core)
    with pytest.raises(RuntimeError, match='newer post-pregrasp'):
        core.begin_approach_replan(1)
    core.begin_approach_replan(2)
    assert core.phase is RuntimePhase.APPROACH_PLANNING
    core.approach_plan_ready()
    assert core.phase is RuntimePhase.APPROACH


@pytest.mark.parametrize(
    ('command_id', 'executor_epoch', 'received_at'),
    (
        (4, EXECUTOR_EPOCH, 5.0),
        (6, EXECUTOR_EPOCH, 5.0),
        (5, 'restarted-executor', 5.0),
        (5, EXECUTOR_EPOCH, 5.1),
    ),
)
def test_non_place_execution_rejects_identity_change_after_active(
    command_id,
    executor_epoch,
    received_at,
):
    core = RuntimeSafetyCore()
    core.execution_update(_execution('succeeded', 'old', 4, 4.0))
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit')
    core.execution_update(_execution('active', 'transit', 5, 5.0))

    action = core.execution_update(_execution(
        'succeeded',
        'transit',
        command_id,
        received_at,
        executor_epoch=executor_epoch,
    ))

    assert core.phase is RuntimePhase.FAILED
    assert action.stop_base and action.cancel_navigation and action.cancel_arm
    _assert_execution_binding_cleared(core)


def test_non_place_execution_requires_complete_source_identity_after_active():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit')
    core.execution_update(_execution('active', 'transit', 1, 1.0))

    action = core.execution_update(parse_execution_status(
        'succeeded;owner=trajectory;segment=transit;command_id=1',
    ))

    assert core.phase is RuntimePhase.FAILED
    assert action.cancel_arm
    _assert_execution_binding_cleared(core)


@pytest.mark.parametrize('terminal', ('cancel', 'fail'))
def test_terminal_transition_clears_non_place_execution_binding(terminal):
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit')
    core.execution_update(_execution('active', 'transit', 1, 1.0))

    if terminal == 'cancel':
        core.cancel()
        assert core.phase is RuntimePhase.CANCELED
    else:
        core.fail('test failure')
        assert core.phase is RuntimePhase.FAILED

    _assert_execution_binding_cleared(core)


@pytest.mark.parametrize('phase', (
    RuntimePhase.PREGRASP_REOBSERVE,
    RuntimePhase.APPROACH_PLANNING,
))
def test_reobservation_clears_every_execution_binding_field(phase):
    core = RuntimeSafetyCore()
    core.phase = phase
    core.planned_serial = 8
    core.execution_segment = 'approach'
    core.execution_seen_active = True
    core.expected_command_id = 12
    core.expected_executor_epoch = EXECUTOR_EPOCH
    core.execution_publish_executor_epoch = EXECUTOR_EPOCH
    core.expected_trajectory_received_at = 12.5

    core.request_reobservation(8)

    assert core.phase is RuntimePhase.WAIT_FRESH_OBSERVATION
    _assert_execution_binding_cleared(core)


def test_new_trajectory_clears_prior_binding_without_losing_segment():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.TRANSIT
    core.execution_segment = 'old-segment'
    core.execution_seen_active = True
    core.expected_command_id = 12
    core.expected_executor_epoch = EXECUTOR_EPOCH
    core.expected_trajectory_received_at = 12.5

    _send_non_place(core, 'transit')

    assert core.execution_segment == 'transit'
    assert not core.execution_seen_active
    assert core.expected_command_id is None
    assert core.expected_executor_epoch == ''
    assert core.expected_trajectory_received_at is None


def test_blocked_second_stage_can_request_a_new_pregrasp_observation():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.APPROACH_PLANNING
    core.planned_serial = 8

    core.request_reobservation(8)

    assert core.phase is RuntimePhase.WAIT_FRESH_OBSERVATION
    assert core.required_replan_serial == 9


def test_rejection_requests_all_fail_closed_outputs():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit')
    action = core.execution_update(_execution(
        'rejected:velocity',
        'transit',
        1,
        0.0,
        event_token=TRAJECTORY_TOKEN,
        event_received_at=0.1,
    ))
    assert core.phase is RuntimePhase.FAILED
    assert action.stop_base and action.cancel_navigation and action.cancel_arm


def test_historical_canceled_status_is_ignored_before_trajectory_ownership():
    core = RuntimeSafetyCore()
    core.begin('pick the bottle')
    action = core.execution_update(parse_execution_status(
        'canceled;owner=named_pose;aperture=0.07',
    ))
    assert core.phase is RuntimePhase.POSE_SETTLE
    assert not action.cancel_arm


def test_pick_complete_is_not_mobile_manipulation_complete():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.VERIFY
    core.verification_complete(carry_only=True)
    assert core.phase is RuntimePhase.CARRY
    _send_non_place(core, 'carry')
    core.execution_update(_execution('succeeded', 'carry', 1, 1.0))
    assert core.phase is RuntimePhase.CARRY
    core.execution_update(_execution('active', 'carry', 1, 1.0))
    core.execution_update(_execution('succeeded', 'carry', 1, 1.0))
    assert core.phase is RuntimePhase.PICK_COMPLETE


def test_non_carry_task_waits_for_observed_place_after_planned_carry():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.VERIFY
    core.verification_complete(carry_only=False)
    _send_non_place(core, 'carry')
    core.execution_update(_execution('active', 'carry', 1, 1.0))
    core.execution_update(_execution('succeeded', 'carry', 1, 1.0))
    assert core.phase is RuntimePhase.PLACE_GROUNDING


def test_observed_place_requires_plan_release_and_retreat_execution():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    carry = _execution('succeeded', 'carry', 7, 7.0)
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
    core.execution_update(_execution('active', 'place_transit', 8, 8.0))
    transit_succeeded = _execution(
        'succeeded',
        'place_transit',
        8,
        8.0,
    )
    core.execution_update(transit_succeeded)
    assert core.phase is RuntimePhase.PLACE_APPROACH
    core.trajectory_sent(
        'place_approach',
        place_contract_id=PLACE_GOAL,
        executor_state=transit_succeeded,
        trajectory_token=TRAJECTORY_TOKEN,
    )
    core.execution_update(_execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))
    approach_succeeded = _execution(
        'succeeded',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    )
    core.execution_update(approach_succeeded)
    assert core.phase is RuntimePhase.RELEASING
    assert core.place_approach_command_id == 9
    assert core.place_approach_received_at == pytest.approx(9.0)
    release_status = _execution(
        'succeeded',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
        gripper_command_id=4,
        gripper_received_at=9.2,
    )
    core.execution_update(release_status)
    assert core.place_release_gripper_command_id == 4
    assert core.place_release_gripper_received_at == pytest.approx(9.2)
    core.release_complete()
    core.trajectory_sent(
        'place_retreat',
        place_contract_id=PLACE_GOAL,
        executor_state=release_status,
        trajectory_token=TRAJECTORY_TOKEN,
    )
    core.execution_update(_execution(
        'active',
        'place_retreat',
        10,
        10.0,
        contract_id=PLACE_GOAL,
        gripper_command_id=4,
        gripper_received_at=9.2,
    ))
    core.execution_update(_execution(
        'succeeded',
        'place_retreat',
        10,
        10.0,
        contract_id=PLACE_GOAL,
        gripper_command_id=4,
        gripper_received_at=9.2,
    ))
    assert core.phase is RuntimePhase.POST_RELEASE_VERIFICATION
    core.post_release_verification_complete()
    assert core.phase is RuntimePhase.COMPLETE


def test_post_release_verification_cannot_bypass_measured_retreat():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_RETREAT

    with pytest.raises(RuntimeError, match='cannot finish'):
        core.post_release_verification_complete()


def test_joined_placement_trajectory_splits_with_boundary_points():
    segments = split_placement_trajectory(
        ((0.0,), (0.1,), (0.2,), (0.3,), (0.4,), (0.5,)),
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        {'transit': 0, 'approach': 2, 'retreat': 4},
    )
    assert tuple(segment.name for segment in segments) == (
        'place_transit', 'place_approach', 'place_retreat',
    )
    assert segments[1].positions == ((0.1,), (0.2,), (0.3,))
    assert segments[1].times_s == pytest.approx((0.0, 0.2, 0.4))


def test_execution_status_validates_aperture():
    state = parse_execution_status(
        'active;owner=trajectory;gripper=accepted:0.02;aperture=0.019',
    )
    assert state.trajectory == 'active'
    assert state.aperture_m == pytest.approx(0.019)
    assert state.accepted_gripper_aperture_m == pytest.approx(0.02)
    with pytest.raises(ValueError):
        parse_execution_status('active;aperture=nan')


def test_execution_status_retains_strict_executor_transaction_identity():
    state = _execution(
        'active',
        'place_approach',
        12,
        81.25,
        contract_id=PLACE_GOAL,
    )

    assert state.executor_epoch == EXECUTOR_EPOCH
    assert state.trajectory_contract_id == PLACE_GOAL
    assert state.trajectory_received_at == pytest.approx(81.25)
    assert state.gripper_received_at == pytest.approx(6.5)


@pytest.mark.parametrize(
    'raw',
    (
        'active;gripper_command_id=4',
        'active;gripper_received_at=1.0',
        'active;gripper_command_id=0;gripper_received_at=1.0',
        'active;gripper_command_id=4;gripper_received_at=nan',
        'active;gripper_command_id=4;gripper_received_at=-1.0',
    ),
)
def test_gripper_command_requires_exact_finite_source_identity(raw):
    with pytest.raises(ValueError):
        parse_execution_status(raw)


@pytest.mark.parametrize(
    'field',
    ('executor_epoch', 'trajectory_contract_id', 'trajectory_received_at'),
)
def test_place_execution_status_rejects_missing_identity_field(field):
    fields = {
        'owner': 'trajectory',
        'segment': 'place_approach',
        'command_id': '12',
        'executor_epoch': EXECUTOR_EPOCH,
        'trajectory_contract_id': PLACE_GOAL,
        'trajectory_received_at': '81.25',
    }
    fields.pop(field)
    raw = 'active;' + ';'.join(
        f'{key}={value}' for key, value in fields.items()
    )

    with pytest.raises(ValueError, match='identity fields'):
        parse_execution_status(raw)


@pytest.mark.parametrize('value', ('nan', 'inf', '-0.001', 'none'))
def test_place_execution_status_rejects_invalid_source_time(value):
    raw = (
        'active;owner=trajectory;segment=place_approach;command_id=12;'
        f'executor_epoch={EXECUTOR_EPOCH};'
        f'trajectory_contract_id={PLACE_GOAL};'
        f'trajectory_received_at={value}'
    )

    with pytest.raises(ValueError, match='trajectory_received_at'):
        parse_execution_status(raw)


def test_execution_status_rejects_duplicate_or_malformed_fields():
    with pytest.raises(ValueError, match='repeats field'):
        parse_execution_status(
            'active;owner=trajectory;owner=named_pose',
        )
    with pytest.raises(ValueError, match='malformed'):
        parse_execution_status('active;owner=trajectory;broken')


@pytest.mark.parametrize(
    'fields',
    (
        'trajectory_event_token=trajectory-a',
        'trajectory_event_received_at=1.0',
        'trajectory_event_token=none;trajectory_event_received_at=1.0',
    ),
)
def test_execution_status_rejects_incomplete_attempt_event_identity(fields):
    with pytest.raises(ValueError, match='event token and source identity'):
        parse_execution_status(f'idle;{fields}')


def test_legacy_non_place_execution_status_remains_compatible():
    state = parse_execution_status(
        'active;owner=trajectory;segment=transit;command_id=3',
    )

    assert state.segment == 'transit'
    assert state.executor_epoch == ''
    assert state.trajectory_contract_id == ''
    assert state.trajectory_received_at is None


def test_place_handoff_tolerates_only_the_exact_frozen_highwater():
    core, transit_succeeded = _core_waiting_for_place_approach()

    action = core.execution_update(transit_succeeded)

    assert not action.cancel_arm
    assert core.phase is RuntimePhase.PLACE_APPROACH
    assert core.expected_command_id is None

    changed_identity = _execution(
        'succeeded',
        'place_approach',
        8,
        8.0,
        contract_id=PLACE_GOAL,
    )
    action = core.execution_update(changed_identity)

    assert action.cancel_arm
    assert core.phase is RuntimePhase.FAILED


def test_place_execution_rejects_frozen_highwater_replay_after_active():
    core, transit_succeeded = _core_waiting_for_place_approach()
    core.execution_update(_execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))

    action = core.execution_update(transit_succeeded)

    assert action.stop_base and action.cancel_navigation and action.cancel_arm
    assert core.phase is RuntimePhase.FAILED


@pytest.mark.parametrize(
    ('command_id', 'received_at', 'contract_id', 'executor_epoch'),
    (
        (9, 9.0, 'foreign-goal', EXECUTOR_EPOCH),
        (9, 9.0, PLACE_GOAL, 'foreign-epoch'),
        (8, 9.0, PLACE_GOAL, EXECUTOR_EPOCH),
        (9, 8.0, PLACE_GOAL, EXECUTOR_EPOCH),
    ),
)
def test_place_approach_rejects_foreign_or_nonadvancing_active_identity(
    command_id,
    received_at,
    contract_id,
    executor_epoch,
):
    core, _ = _core_waiting_for_place_approach()

    action = core.execution_update(_execution(
        'active',
        'place_approach',
        command_id,
        received_at,
        contract_id=contract_id,
        executor_epoch=executor_epoch,
    ))

    assert action.stop_base and action.cancel_navigation and action.cancel_arm
    assert core.phase is RuntimePhase.FAILED


def test_place_approach_requires_active_then_same_source_succeeded():
    core, _ = _core_waiting_for_place_approach()
    succeeded = _execution(
        'succeeded',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    )

    action = core.execution_update(succeeded)

    assert action.cancel_arm
    assert core.phase is RuntimePhase.FAILED

    core, _ = _core_waiting_for_place_approach()
    core.execution_update(_execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))
    action = core.execution_update(_execution(
        'succeeded',
        'place_approach',
        9,
        9.1,
        contract_id=PLACE_GOAL,
    ))

    assert action.cancel_arm
    assert core.phase is RuntimePhase.FAILED


def test_place_request_epoch_cannot_change_before_first_segment():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    carry = _execution('succeeded', 'carry', 7, 7.0)
    core.place_request_sent(
        place_contract_id=PLACE_GOAL,
        executor_state=carry,
    )
    core.place_plan_ready(_place_executor_snapshot())
    restarted = _execution(
        'succeeded',
        'carry',
        1,
        8.0,
        executor_epoch='restarted-executor',
    )

    with pytest.raises(ValueError, match='epoch changed'):
        core.trajectory_sent(
            'place_transit',
            place_contract_id=PLACE_GOAL,
            executor_state=restarted,
            trajectory_token=TRAJECTORY_TOKEN,
        )


def test_place_planning_retains_only_the_exact_request_executor_snapshot():
    carry = _execution('succeeded', 'carry', 7, 7.0)
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    core.place_request_sent(
        place_contract_id=PLACE_GOAL,
        executor_state=carry,
    )

    action = core.execution_update(carry)

    assert not action.cancel_arm
    assert core.phase is RuntimePhase.PLACE_PLANNING


@pytest.mark.parametrize(
    ('command_id', 'received_at', 'executor_epoch'),
    (
        (8, 8.0, EXECUTOR_EPOCH),
        (7, 7.1, EXECUTOR_EPOCH),
        (7, 7.0, 'restarted-executor'),
    ),
)
def test_place_planning_fails_immediately_when_executor_snapshot_changes(
    command_id,
    received_at,
    executor_epoch,
):
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    core.place_request_sent(
        place_contract_id=PLACE_GOAL,
        executor_state=_execution('succeeded', 'carry', 7, 7.0),
    )

    action = core.execution_update(_execution(
        'succeeded',
        'carry',
        command_id,
        received_at,
        executor_epoch=executor_epoch,
    ))

    assert action.stop_base and action.cancel_navigation and action.cancel_arm
    assert core.phase is RuntimePhase.FAILED


def test_place_release_feedback_remains_bound_to_approach_epoch_and_contract():
    core, _ = _core_waiting_for_place_approach()
    core.execution_update(_execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))
    core.execution_update(_execution(
        'succeeded',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
    ))
    assert core.phase is RuntimePhase.RELEASING

    action = core.execution_update(_execution(
        'succeeded',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
        executor_epoch='restarted-executor',
    ))

    assert action.cancel_arm
    assert core.phase is RuntimePhase.FAILED


@pytest.mark.parametrize('replayed_segment', ('old', 'transit'))
def test_same_epoch_old_active_command_is_ignored_before_current_binding(
    replayed_segment,
):
    core = RuntimeSafetyCore()
    _seed_executor_highwater(core, 7, 7.0)
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit', published_at_s=8.0)

    assert core.execution_command_highwater_snapshot == {
        EXECUTOR_EPOCH: 7,
    }
    action = core.execution_update(_execution(
        'active',
        replayed_segment,
        7,
        7.0,
    ))

    assert not action.cancel_arm
    assert core.phase is RuntimePhase.TRANSIT
    assert core.expected_command_id is None


def test_new_executor_epoch_can_restart_command_ids_from_one():
    core = RuntimeSafetyCore()
    _seed_executor_highwater(
        core, 70, 70.0, executor_epoch='old-executor',
    )
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(
        core, 'transit', executor_epoch='new-executor', published_at_s=0.5,
    )

    action = core.execution_update(_execution(
        'active',
        'transit',
        1,
        1.0,
        executor_epoch='new-executor',
    ))

    assert not action.cancel_arm
    assert core.phase is RuntimePhase.TRANSIT
    assert core.expected_command_id == 1
    assert core.expected_executor_epoch == 'new-executor'
    core.execution_update(_execution(
        'succeeded',
        'transit',
        1,
        1.0,
        executor_epoch='new-executor',
    ))
    assert core.phase is RuntimePhase.PREGRASP_REOBSERVE


def test_delayed_unseen_active_before_publish_fence_cannot_bind_new_motion():
    core = RuntimeSafetyCore()
    _seed_executor_highwater(core, 7, 70.0)
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit', published_at_s=100.0)

    stale_active = core.execution_update(_execution(
        'active', 'transit', 8, 80.0, trajectory_token='trajectory-old',
    ))
    stale_success = core.execution_update(_execution(
        'succeeded', 'transit', 8, 80.0, trajectory_token='trajectory-old',
    ))

    assert not stale_active.cancel_arm and not stale_success.cancel_arm
    assert core.phase is RuntimePhase.TRANSIT
    assert core.expected_command_id is None

    core.execution_update(_execution('active', 'transit', 9, 100.1))
    core.execution_update(_execution('succeeded', 'transit', 9, 100.1))
    assert core.phase is RuntimePhase.PREGRASP_REOBSERVE


def test_delayed_unseen_active_with_newer_receive_time_fails_and_cancels():
    core = RuntimeSafetyCore()
    _seed_executor_highwater(core, 7, 70.0)
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(
        core,
        'transit',
        published_at_s=100.0,
        trajectory_token='trajectory-new',
    )

    action = core.execution_update(_execution(
        'active',
        'transit',
        8,
        100.5,
        trajectory_token='trajectory-old',
    ))

    assert action.cancel_arm
    assert core.phase is RuntimePhase.FAILED
    assert core.expected_command_id is None


def test_matching_rejected_attempt_event_fails_without_mutating_accepted_id():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.APPROACH
    _send_non_place(
        core,
        'approach',
        published_at_s=100.0,
        trajectory_token='trajectory-new',
    )

    action = core.execution_update(_execution(
        'rejected:start_state',
        'transit',
        7,
        90.0,
        trajectory_token='trajectory-old',
        event_token='trajectory-new',
        event_received_at=100.1,
    ))

    assert action.cancel_arm
    assert core.phase is RuntimePhase.FAILED
    assert core.expected_command_id is None


def test_place_execution_cancels_newly_accepted_old_token():
    core, _ = _core_waiting_for_place_approach()

    action = core.execution_update(_execution(
        'active',
        'place_approach',
        9,
        9.0,
        contract_id=PLACE_GOAL,
        trajectory_token='trajectory-old',
    ))

    assert action.cancel_arm
    assert core.phase is RuntimePhase.FAILED
    assert core.expected_command_id is None


def test_unowned_execution_status_cannot_poison_executor_highwater():
    core = RuntimeSafetyCore()

    core.execution_update(_execution('succeeded', 'transit', 999, 999.0))

    assert core.executor_command_highwater == {}
    assert core.highest_command_id == 0


def test_trajectory_uses_frozen_epoch_highwater_snapshot():
    core = RuntimeSafetyCore()
    _seed_executor_highwater(core, 7, 7.0)
    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit', published_at_s=7.5)
    core.execution_update(_execution('succeeded', 'transit', 8, 8.0))

    assert core.executor_command_highwater[EXECUTOR_EPOCH] == 7
    assert core.execution_command_highwater_snapshot[EXECUTOR_EPOCH] == 7
    action = core.execution_update(_execution('active', 'transit', 8, 8.0))

    assert not action.cancel_arm
    assert core.expected_command_id == 8
    core.execution_update(_execution('succeeded', 'transit', 8, 8.0))
    assert core.phase is RuntimePhase.PREGRASP_REOBSERVE


def test_begin_and_restart_preserve_per_epoch_command_highwater():
    core = RuntimeSafetyCore()
    _seed_executor_highwater(core, 7, 7.0)

    core.begin('pick the mug')
    assert core.executor_command_highwater == {EXECUTOR_EPOCH: 7}
    core.restart_grounding()
    assert core.executor_command_highwater == {EXECUTOR_EPOCH: 7}

    core.phase = RuntimePhase.TRANSIT
    _send_non_place(core, 'transit', published_at_s=8.0)
    action = core.execution_update(_execution('active', 'transit', 7, 7.0))

    assert not action.cancel_arm
    assert core.phase is RuntimePhase.TRANSIT
    assert core.expected_command_id is None


def test_place_request_ignores_highwater_from_a_different_executor_epoch():
    core = RuntimeSafetyCore()
    _seed_executor_highwater(
        core, 70, 70.0, executor_epoch='old-executor',
    )
    core.phase = RuntimePhase.PLACE_GROUNDING
    carry = _execution(
        'succeeded',
        'carry',
        1,
        1.0,
        executor_epoch='new-executor',
    )

    core.place_request_sent(
        place_contract_id=PLACE_GOAL,
        executor_state=carry,
    )

    assert core.phase is RuntimePhase.PLACE_PLANNING
    assert core.place_executor_epoch == 'new-executor'
    assert core.executor_command_highwater == {
        'old-executor': 70,
        'new-executor': 1,
    }


def test_precontact_recovery_preserves_instruction_and_requires_new_frame():
    core = RuntimeSafetyCore()
    core.begin('pick and place the mug')
    core.mark_pose_settled()
    core.restart_grounding()
    assert core.instruction == 'pick and place the mug'
    assert core.phase is RuntimePhase.POSE_SETTLE
    core.phase = RuntimePhase.PLANNING
    core.request_reobservation(8)
    assert core.phase is RuntimePhase.WAIT_FRESH_OBSERVATION
    assert core.required_replan_serial == 9


def test_visual_search_requires_turn_then_stationary_regrounding():
    core = RuntimeSafetyCore()
    core.begin('pick the requested object')
    core.begin_visual_search()
    assert core.phase is RuntimePhase.VISUAL_SEARCH
    core.mark_visual_search_complete()
    assert core.phase is RuntimePhase.POSE_SETTLE
    core.mark_pose_settled()
    assert core.phase is RuntimePhase.GROUNDING


def test_bounded_yaw_search_alternates_measured_viewpoints_across_wrap():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.4,
        max_yaw_offset_rad=0.8,
        yaw_tolerance_rad=0.02,
        yaw_gain=2.0,
        max_yaw_rate_rps=0.3,
        turn_timeout_s=2.0,
    ))
    assert search.start(3.0, now_s=1.0, current_position_xy=(0.0, 0.0))
    assert search.target_offset_rad == pytest.approx(0.4)
    assert search.update(
        3.0, now_s=1.1, current_position_xy=(0.0, 0.0),
    ).angular_z == pytest.approx(0.3)
    target = search.target_yaw_rad
    assert target is not None
    assert search.update(
        target, now_s=1.2, current_position_xy=(0.0, 0.0),
    ).complete

    assert search.start(
        target, now_s=1.3, current_position_xy=(0.0, 0.0),
    )
    assert search.target_offset_rad == pytest.approx(-0.4)


def test_bounded_yaw_search_reacquires_same_target_without_consuming_view() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.4,
        max_yaw_offset_rad=0.8,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        yaw_gain=1.0,
        min_yaw_rate_rps=0.06,
        max_yaw_rate_rps=0.3,
    ))
    assert search.start(
        0.0,
        now_s=1.0,
        current_position_xy=(2.0, -1.0),
    )
    target = search.target_yaw_rad
    target_offset = search.target_offset_rad
    assert target is not None
    assert search.update(
        target,
        now_s=1.2,
        current_position_xy=(2.0, -1.0),
    ).complete
    attempt = search.attempt
    scan_index = search._scan_index

    search.reacquire_current_target(
        target + 0.06,
        now_s=1.3,
        current_position_xy=(2.01, -1.0),
        deadline_s=2.0,
    )

    assert search.active
    assert search.attempt == attempt
    assert search._scan_index == scan_index
    assert search.target_yaw_rad == target
    assert search.target_offset_rad == target_offset
    assert search.position_anchor_xy == (2.0, -1.0)
    assert search.allocated_timeout_s == pytest.approx(0.7)
    update = search.update(
        target + 0.06,
        now_s=1.4,
        current_position_xy=(2.01, -1.0),
    )
    assert update.angular_z == pytest.approx(-0.06)
    assert not update.complete


def test_bounded_yaw_search_reacquisition_requires_remaining_deadline() -> None:
    search = BoundedYawSearch()
    assert search.start(
        0.0,
        now_s=1.0,
        current_position_xy=(0.0, 0.0),
    )
    target = search.target_yaw_rad
    assert target is not None
    assert search.update(
        target,
        now_s=1.1,
        current_position_xy=(0.0, 0.0),
    ).complete

    with pytest.raises(ValueError, match='finite and bounded'):
        search.reacquire_current_target(
            target + 0.05,
            now_s=2.0,
            current_position_xy=(0.0, 0.0),
            deadline_s=2.0,
        )
    assert not search.active
    assert search.attempt == 1


def test_directional_recenter_cannot_exceed_configured_yaw_coverage():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.25,
        max_yaw_offset_rad=0.5,
        yaw_tolerance_rad=0.01,
        turn_timeout_s=1.0,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(0.0, 0.0),
        image_edge_direction=1,
    )
    assert search.update(
        0.25, now_s=0.1, current_position_xy=(0.0, 0.0),
    ).complete
    assert search.start(
        0.25, now_s=0.2, current_position_xy=(0.0, 0.0),
        image_edge_direction=1,
    )
    assert search.update(
        0.5, now_s=0.3, current_position_xy=(0.0, 0.0),
    ).complete
    assert not search.start(
        0.5, now_s=0.4, current_position_xy=(0.0, 0.0),
        image_edge_direction=1,
    )


def test_visual_search_timeout_commands_zero_and_fails_closed():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        turn_timeout_s=0.5,
    ))
    assert search.start(
        0.0, now_s=2.0, current_position_xy=(0.0, 0.0),
    )
    update = search.update(
        0.0, now_s=2.6, current_position_xy=(0.0, 0.0),
    )
    assert update.timed_out
    assert update.angular_z == 0.0
    assert not search.active


@pytest.mark.parametrize('direction', (-1, 1))
def test_visual_search_commands_configured_minimum_outside_yaw_gate(direction):
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        yaw_gain=1.0,
        min_yaw_rate_rps=0.06,
        max_yaw_rate_rps=0.30,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(0.0, 0.0),
        image_edge_direction=direction,
    )
    target = search.target_yaw_rad
    assert target is not None

    update = search.update(
        target - direction * 0.02,
        now_s=0.1,
        current_position_xy=(0.0, 0.0),
    )

    assert update.angular_z == pytest.approx(direction * 0.06)
    assert not update.complete


def test_visual_search_zero_minimum_preserves_proportional_yaw_command():
    assert VisualSearchConfig().min_yaw_rate_rps == 0.0
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        yaw_gain=1.0,
        min_yaw_rate_rps=0.0,
        max_yaw_rate_rps=0.30,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(0.0, 0.0),
    )
    target = search.target_yaw_rad
    assert target is not None

    update = search.update(
        target - 0.02,
        now_s=0.1,
        current_position_xy=(0.0, 0.0),
    )

    assert update.angular_z == pytest.approx(0.02)
    assert not update.complete


@pytest.mark.parametrize('direction', (-1, 1))
def test_visual_search_accepts_target_crossing_within_settle_gate(direction):
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        yaw_gain=1.0,
        min_yaw_rate_rps=0.06,
        max_yaw_rate_rps=0.30,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(0.0, 0.0),
        image_edge_direction=direction,
    )
    target = search.target_yaw_rad
    assert target is not None
    before_crossing = search.update(
        target - direction * 0.02,
        now_s=0.1,
        current_position_xy=(0.0, 0.0),
    )
    assert not before_crossing.complete
    assert before_crossing.angular_z == pytest.approx(direction * 0.06)

    crossed = search.update(
        target + direction * 0.025,
        now_s=0.2,
        current_position_xy=(0.0, 0.0),
    )

    assert crossed.complete
    assert crossed.angular_z == 0.0
    assert not search.active


@pytest.mark.parametrize('direction', (-1, 1))
def test_visual_search_accepts_bounded_crossing_across_angle_wrap(direction):
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        yaw_gain=1.0,
        min_yaw_rate_rps=0.06,
        max_yaw_rate_rps=0.30,
    ))
    origin = direction * (math.pi - 0.305)
    assert search.start(
        origin,
        now_s=0.0,
        current_position_xy=(0.0, 0.0),
        image_edge_direction=direction,
    )
    target = search.target_yaw_rad
    assert target is not None
    assert abs(target) == pytest.approx(math.pi - 0.005)
    before = target - direction * 0.02
    after = math.atan2(
        math.sin(target + direction * 0.025),
        math.cos(target + direction * 0.025),
    )
    assert math.copysign(1.0, before) != math.copysign(1.0, after)

    assert not search.update(
        before,
        now_s=0.1,
        current_position_xy=(0.0, 0.0),
    ).complete
    crossed = search.update(
        after,
        now_s=0.2,
        current_position_xy=(0.0, 0.0),
    )

    assert crossed.complete
    assert crossed.angular_z == 0.0


@pytest.mark.parametrize('direction', (-1, 1))
def test_visual_search_reverses_after_target_crossing_outside_settle_gate(direction):
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        yaw_gain=1.0,
        min_yaw_rate_rps=0.06,
        max_yaw_rate_rps=0.30,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(0.0, 0.0),
        image_edge_direction=direction,
    )
    target = search.target_yaw_rad
    assert target is not None
    search.update(
        target - direction * 0.02,
        now_s=0.1,
        current_position_xy=(0.0, 0.0),
    )

    crossed = search.update(
        target + direction * 0.04,
        now_s=0.2,
        current_position_xy=(0.0, 0.0),
    )

    assert not crossed.complete
    assert crossed.angular_z == pytest.approx(-direction * 0.06)
    assert search.active


def test_visual_search_start_and_reset_reinitialize_crossing_history():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        yaw_gain=1.0,
        min_yaw_rate_rps=0.06,
        max_yaw_rate_rps=0.30,
    ))
    assert search.previous_error_rad is None
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(0.0, 0.0),
    )
    first_target = search.target_yaw_rad
    assert first_target is not None
    assert search.previous_error_rad == pytest.approx(0.3)
    assert search.update(
        first_target - 0.005,
        now_s=0.1,
        current_position_xy=(0.0, 0.0),
    ).complete

    assert search.start(
        first_target,
        now_s=0.2,
        current_position_xy=(0.0, 0.0),
    )
    second_target = search.target_yaw_rad
    assert second_target is not None
    assert search.previous_error_rad < 0.0
    second_update = search.update(
        second_target + 0.025,
        now_s=0.3,
        current_position_xy=(0.0, 0.0),
    )
    assert not second_update.complete
    assert second_update.angular_z == pytest.approx(-0.06)

    search.reset()
    assert search.previous_error_rad is None


@pytest.mark.parametrize('minimum', (-0.01, float('nan'), 0.31))
def test_visual_search_minimum_yaw_rate_must_be_bounded(minimum):
    with pytest.raises(ValueError, match='minimum yaw rate'):
        VisualSearchConfig(
            min_yaw_rate_rps=minimum,
            max_yaw_rate_rps=0.30,
        )


def test_visual_search_first_complete_sample_wins_inside_yaw_deadline_grace():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.02,
        turn_timeout_s=0.5,
        deadline_grace_s=0.05,
    ))
    assert search.start(
        0.0, now_s=2.0, current_position_xy=(0.0, 0.0),
    )
    target = search.target_yaw_rad
    assert target is not None

    update = search.update(
        target,
        now_s=2.51,
        current_position_xy=(0.0, 0.0),
    )
    assert update.complete
    assert not update.timed_out
    assert update.angular_z == 0.0
    assert not search.active


def test_visual_search_rejects_complete_sample_after_yaw_deadline_grace():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.02,
        turn_timeout_s=0.5,
        deadline_grace_s=0.05,
    ))
    assert search.start(0.0, now_s=2.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    update = search.update(
        target,
        now_s=2.56,
        current_position_xy=(0.0, 0.0),
    )

    assert update.timed_out
    assert not update.complete
    assert update.timeout_phase == 'yaw_turn'
    assert not search.active


def test_visual_search_timeout_scales_with_measured_angular_travel():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.4,
        max_yaw_offset_rad=0.8,
        yaw_tolerance_rad=0.02,
        turn_timeout_s=2.0,
        max_turn_timeout_s=6.0,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(1.0, -2.0),
    )
    first_target = search.target_yaw_rad
    assert first_target is not None
    assert search.allocated_timeout_s == pytest.approx(2.0)
    assert search.update(
        first_target, now_s=1.0, current_position_xy=(1.0, -2.0),
    ).complete

    assert search.start(
        first_target, now_s=2.0, current_position_xy=(1.0, -2.0),
    )
    assert search.target_offset_rad == pytest.approx(-0.4)
    assert search.allocated_timeout_s == pytest.approx(4.0)
    still_turning = search.update(
        0.0, now_s=4.1, current_position_xy=(1.0, -2.0),
    )
    assert not still_turning.timed_out
    timed_out = search.update(
        0.0, now_s=6.1, current_position_xy=(1.0, -2.0),
    )
    assert timed_out.timed_out
    assert timed_out.angular_z == 0.0


def test_visual_search_scaled_timeout_never_exceeds_hard_cap():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.2,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        turn_timeout_s=1.0,
        max_turn_timeout_s=1.5,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(0.0, 0.0),
    )
    first_target = search.target_yaw_rad
    assert first_target is not None
    assert search.update(
        first_target, now_s=0.5, current_position_xy=(0.0, 0.0),
    ).complete
    assert search.start(
        first_target, now_s=1.0, current_position_xy=(0.0, 0.0),
    )
    assert search.allocated_timeout_s == pytest.approx(1.5)

    with pytest.raises(ValueError, match='maximum timeout'):
        VisualSearchConfig(turn_timeout_s=2.0, max_turn_timeout_s=1.0)


def test_visual_search_fails_closed_on_planar_drift_before_timeout():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        turn_timeout_s=1.0,
        max_turn_timeout_s=2.0,
        max_planar_drift_m=0.10,
        moving_rebound_reacquire_m=0.075,
    ))
    assert search.start(
        0.0, now_s=0.0, current_position_xy=(2.0, 3.0),
    )

    update = search.update(
        0.1, now_s=0.2, current_position_xy=(2.08, 3.08),
    )

    assert update.drift_exceeded
    assert update.planar_drift_m == pytest.approx(math.hypot(0.08, 0.08))
    assert update.angular_z == 0.0
    assert update.linear_x == 0.0
    assert update.linear_y == 0.0
    assert not update.timed_out
    assert not search.active


def test_visual_search_rotates_map_anchor_error_into_current_base_frame():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        position_hold_deadband_m=0.01,
        position_hold_gain_s_inv=2.0,
        max_position_hold_speed_mps=0.2,
    ))
    assert search.start(
        math.pi / 2.0,
        now_s=0.0,
        current_position_xy=(2.0, 3.0),
    )

    update = search.update(
        math.pi / 2.0,
        now_s=0.1,
        current_position_xy=(1.95, 3.0),
    )

    # The anchor is +map-X from the robot. At +90 degree yaw that is -base-Y.
    assert update.linear_x == pytest.approx(0.0, abs=1e-12)
    assert update.linear_y == pytest.approx(-0.10)
    assert search.position_error_base_xy == pytest.approx((0.0, -0.05))
    assert update.angular_z == pytest.approx(0.3)


def test_visual_search_position_hold_applies_deadband_and_vector_speed_cap():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        max_planar_drift_m=0.6,
        position_hold_deadband_m=0.02,
        position_hold_gain_s_inv=10.0,
        max_position_hold_speed_mps=0.07,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))

    inside = search.update(
        0.0,
        now_s=0.1,
        current_position_xy=(-0.006, -0.008),
    )
    assert inside.linear_x == 0.0
    assert inside.linear_y == 0.0

    capped = search.update(
        0.0,
        now_s=0.2,
        current_position_xy=(-0.3, -0.4),
    )
    assert math.hypot(capped.linear_x, capped.linear_y) == pytest.approx(0.07)
    assert capped.linear_x == pytest.approx(0.042)
    assert capped.linear_y == pytest.approx(0.056)


@pytest.mark.parametrize(
    ('distance_m', 'expected_speed_mps'),
    (
        (0.10, 0.05000),
        (0.09, 0.04250),
        (0.08, 0.03500),
        (0.07, 0.02750),
        (0.061, 0.02075),
    ),
)
def test_visual_search_position_hold_slows_continuously_before_completion(
    distance_m: float,
    expected_speed_mps: float,
) -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        max_planar_drift_m=0.15,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.06,
        moving_rebound_reacquire_m=0.08,
        position_hold_gain_s_inv=1.0,
        max_position_hold_speed_mps=0.05,
        position_hold_slowdown_radius_m=0.10,
        min_position_hold_speed_mps=0.02,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    update = search.update(
        target,
        now_s=0.1,
        current_position_xy=(-distance_m, 0.0),
        measured_angular_speed_rps=0.0,
    )

    speed = math.hypot(update.linear_x, update.linear_y)
    assert speed == pytest.approx(expected_speed_mps)
    assert update.linear_x > 0.0
    assert update.linear_y < 0.0


@pytest.mark.parametrize(
    ('slowdown_radius_m', 'minimum_speed_mps'),
    ((0.10, 0.0), (0.0, 0.02), (0.05, 0.02), (0.16, 0.02), (0.10, 0.06)),
)
def test_visual_search_position_slowdown_contract_is_bounded(
    slowdown_radius_m: float,
    minimum_speed_mps: float,
) -> None:
    with pytest.raises(ValueError, match='slowdown|minimum position'):
        VisualSearchConfig(
            max_planar_drift_m=0.15,
            position_completion_tolerance_m=0.06,
            moving_rebound_reacquire_m=0.08,
            max_position_hold_speed_mps=0.05,
            position_hold_slowdown_radius_m=slowdown_radius_m,
            min_position_hold_speed_mps=minimum_speed_mps,
        )


def test_visual_search_waits_for_position_hold_after_yaw_and_reset_reanchors():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.01,
        position_hold_gain_s_inv=1.0,
        max_position_hold_speed_mps=0.1,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(1.0, 2.0))
    target = search.target_yaw_rad
    assert target is not None

    correcting = search.update(
        target,
        now_s=0.5,
        current_position_xy=(1.05, 2.0),
    )
    assert correcting.angular_z == pytest.approx(0.0)
    assert correcting.linear_x < 0.0
    assert not correcting.complete
    assert search.active

    complete = search.update(
        target,
        now_s=0.6,
        current_position_xy=(1.005, 2.0),
    )
    assert complete.complete
    assert complete.linear_x == 0.0
    assert complete.linear_y == 0.0
    assert not search.active

    # A bounded multi-view search keeps the first XY anchor while its yaw
    # origin and coverage index are still active.
    assert search.start(target, now_s=0.7, current_position_xy=(1.005, 2.0))
    assert search.position_anchor_xy == (1.0, 2.0)

    search.reset()
    assert search.position_anchor_xy is None
    assert search.position_error_base_xy == (0.0, 0.0)
    assert search.linear_command_base_xy == (0.0, 0.0)
    assert search.start(0.0, now_s=1.0, current_position_xy=(4.0, -3.0))
    assert search.position_anchor_xy == (4.0, -3.0)


def test_visual_search_decouples_position_hold_after_yaw_reaches_tolerance():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.04,
        settle_yaw_tolerance_rad=0.04,
        yaw_gain=1.5,
        turn_timeout_s=2.0,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.015,
        position_hold_gain_s_inv=1.0,
        max_position_hold_speed_mps=0.1,
        position_hold_timeout_s=3.0,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(1.0, 2.0))
    target = search.target_yaw_rad
    assert target is not None

    # This reproduces the live failure: yaw is inside tolerance while measured
    # translation remains outside its deadband. Residual yaw must be zero so the
    # whole-body policy can close XY without fighting a simultaneous turn.
    update = search.update(
        target - 0.03,
        now_s=1.0,
        current_position_xy=(1.04, 2.0),
    )
    assert not update.complete
    assert not update.timed_out
    assert update.angular_z == 0.0
    assert update.linear_x < 0.0
    assert search.position_hold_started_at_s == pytest.approx(1.0)

    # The position phase owns an independent budget after the yaw phase.
    after_turn_budget = search.update(
        target - 0.03,
        now_s=2.5,
        current_position_xy=(1.02, 2.0),
    )
    assert not after_turn_budget.timed_out
    assert after_turn_budget.angular_z == 0.0
    assert after_turn_budget.linear_x < 0.0


def test_visual_search_releases_xy_hold_inside_final_parking_envelope():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.05,
        position_hold_gain_s_inv=1.0,
        max_position_hold_speed_mps=0.1,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    outside_parking_envelope = search.update(
        target - 0.04,
        now_s=0.1,
        current_position_xy=(0.04, 0.0),
    )
    assert outside_parking_envelope.angular_z > 0.0
    assert outside_parking_envelope.linear_x < 0.0

    inside_parking_envelope = search.update(
        target - 0.02,
        now_s=0.2,
        current_position_xy=(0.04, 0.0),
    )
    assert not inside_parking_envelope.complete
    assert inside_parking_envelope.angular_z > 0.0
    assert inside_parking_envelope.linear_x == 0.0
    assert inside_parking_envelope.linear_y == 0.0
    assert search.linear_command_base_xy == (0.0, 0.0)


def test_visual_search_position_hold_timeout_fails_closed():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.04,
        settle_yaw_tolerance_rad=0.04,
        turn_timeout_s=2.0,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.02,
        position_hold_timeout_s=0.5,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None
    assert not search.update(
        target,
        now_s=0.5,
        current_position_xy=(0.04, 0.0),
    ).timed_out

    timed_out = search.update(
        target,
        now_s=1.1,
        current_position_xy=(0.04, 0.0),
    )
    assert timed_out.timed_out
    assert timed_out.timeout_phase == 'position_hold'
    assert timed_out.angular_z == 0.0
    assert timed_out.linear_x == 0.0
    assert timed_out.linear_y == 0.0
    assert not search.active
    assert search.linear_command_base_xy == (0.0, 0.0)


def test_visual_search_first_complete_sample_wins_inside_position_deadline_grace():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.04,
        settle_yaw_tolerance_rad=0.04,
        turn_timeout_s=2.0,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.02,
        position_hold_timeout_s=0.5,
        deadline_grace_s=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None
    holding = search.update(
        target,
        now_s=0.5,
        current_position_xy=(0.04, 0.0),
    )
    assert not holding.complete
    assert search.position_hold_started_at_s == pytest.approx(0.5)

    complete = search.update(
        target,
        now_s=1.01,
        current_position_xy=(0.019, 0.0),
    )
    assert complete.complete
    assert not complete.timed_out
    assert complete.angular_z == 0.0
    assert complete.linear_x == 0.0
    assert complete.linear_y == 0.0
    assert not search.active


def test_visual_search_rejects_complete_sample_after_position_deadline_grace():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.04,
        settle_yaw_tolerance_rad=0.04,
        turn_timeout_s=2.0,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.02,
        position_hold_timeout_s=0.5,
        deadline_grace_s=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None
    assert not search.update(
        target,
        now_s=0.5,
        current_position_xy=(0.04, 0.0),
    ).complete

    update = search.update(
        target,
        now_s=1.06,
        current_position_xy=(0.019, 0.0),
    )

    assert update.timed_out
    assert not update.complete
    assert update.timeout_phase == 'position_hold'
    assert not search.active


def test_visual_search_reacquires_disturbed_yaw_without_coupled_translation():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.02,
        yaw_gain=1.5,
        turn_timeout_s=2.0,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.02,
        position_hold_timeout_s=2.0,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None
    search.update(
        target,
        now_s=0.5,
        current_position_xy=(0.04, 0.0),
    )

    reacquire = search.update(
        target - 0.071,
        now_s=0.6,
        current_position_xy=(0.04, 0.0),
    )
    assert reacquire.angular_z > 0.0
    assert reacquire.linear_x == 0.0
    assert reacquire.linear_y == 0.0
    assert search.linear_command_base_xy == (0.0, 0.0)


def test_visual_search_rejects_time_rollback_after_position_hold_starts():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.02,
        turn_timeout_s=20.0,
        max_turn_timeout_s=20.0,
        position_completion_tolerance_m=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None
    holding = search.update(
        target,
        now_s=10.0,
        current_position_xy=(0.10, 0.0),
    )
    assert not holding.complete
    assert search.position_hold_started_at_s == pytest.approx(10.0)
    assert search.last_update_at_s == pytest.approx(10.0)

    with pytest.raises(ValueError, match='finite and monotonic'):
        search.update(
            target,
            now_s=5.0,
            current_position_xy=(0.09, 0.0),
        )
    assert search.active
    assert search.last_update_at_s == pytest.approx(10.0)

    search.reset()
    assert search.last_update_at_s is None
    assert search.start(0.0, now_s=1.0, current_position_xy=(0.0, 0.0))
    assert search.last_update_at_s == pytest.approx(1.0)


def test_visual_search_position_hold_timeout_must_be_positive():
    with pytest.raises(ValueError, match='limits.*positive'):
        VisualSearchConfig(position_hold_timeout_s=0.0)


def test_visual_search_settle_yaw_tolerance_bounds_control_hysteresis():
    with pytest.raises(ValueError, match='settle yaw tolerance.*at least'):
        VisualSearchConfig(
            yaw_tolerance_rad=0.03,
            settle_yaw_tolerance_rad=0.02,
        )
    with pytest.raises(ValueError, match='settle yaw tolerance.*smaller'):
        VisualSearchConfig(
            yaw_step_rad=0.3,
            settle_yaw_tolerance_rad=0.3,
        )
    with pytest.raises(ValueError, match='heading reacquire tolerance'):
        VisualSearchConfig(
            settle_yaw_tolerance_rad=0.04,
            position_heading_reacquire_tolerance_rad=0.04,
        )
    with pytest.raises(ValueError, match='heading reacquire tolerance'):
        VisualSearchConfig(
            yaw_step_rad=0.2,
            position_heading_reacquire_tolerance_rad=0.2,
        )


def test_visual_search_control_converges_inside_settle_hysteresis() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3490658504,
        max_yaw_offset_rad=1.0471975512,
        yaw_tolerance_rad=0.0174532925,
        settle_yaw_tolerance_rad=0.0349065850,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    # The old 2 degree completion boundary parked here and then rebounded out
    # of tolerance. Keep commanding until the measured turn reaches 1 degree.
    outside_inner = search.update(
        target - 0.0347158,
        now_s=1.0,
        current_position_xy=(0.0, 0.0),
    )
    assert not outside_inner.complete
    assert outside_inner.angular_z > 0.0

    inside_inner = search.update(
        target - 0.0174,
        now_s=1.1,
        current_position_xy=(0.0, 0.0),
    )
    assert inside_inner.complete
    assert inside_inner.angular_z == 0.0


def test_visual_search_enters_xy_hold_at_measured_go2w_yaw_residual() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3490658504,
        max_yaw_offset_rad=1.0471975512,
        yaw_tolerance_rad=0.0087266463,
        settle_yaw_tolerance_rad=0.0349065850,
        yaw_gain=1.5,
        max_yaw_rate_rps=0.30,
        min_yaw_rate_rps=0.06,
        turn_timeout_s=10.0,
        max_turn_timeout_s=30.0,
        max_planar_drift_m=0.15,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.05,
        position_hold_gain_s_inv=1.0,
        max_position_hold_speed_mps=0.10,
        position_hold_timeout_s=4.0,
        deadline_grace_s=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    # Reproduce the measured Office recovery endpoint: the old 0.2 degree
    # gate timed out here instead of handing the 5.7 cm drift to XY recovery.
    holding = search.update(
        target - 0.00514656,
        now_s=10.04,
        current_position_xy=(0.057413, 0.0),
    )
    assert not holding.complete
    assert not holding.timed_out
    assert not holding.drift_exceeded
    assert holding.angular_z == 0.0
    assert math.hypot(holding.linear_x, holding.linear_y) == pytest.approx(0.057413)
    assert search.position_hold_started_at_s == pytest.approx(10.04)
    assert search.active

    parked = search.update(
        target - 0.00514656,
        now_s=10.2,
        current_position_xy=(0.049, 0.0),
    )
    assert parked.complete
    assert parked.angular_z == 0.0
    assert parked.linear_x == 0.0
    assert parked.linear_y == 0.0
    assert not search.active


def test_visual_search_hands_slow_outer_gate_to_sticky_position_hold() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3490658504,
        max_yaw_offset_rad=1.0471975512,
        yaw_tolerance_rad=0.0087266463,
        settle_yaw_tolerance_rad=0.0349065850,
        min_yaw_rate_rps=0.06,
        turn_timeout_s=10.0,
        max_turn_timeout_s=30.0,
        max_planar_drift_m=0.15,
        position_completion_tolerance_m=0.05,
        position_hold_timeout_s=4.0,
        settle_max_angular_speed_rps=0.05,
        deadline_grace_s=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    holding = search.update(
        target - 0.0308726854,
        now_s=10.04,
        current_position_xy=(0.0537638226, 0.0),
        measured_angular_speed_rps=0.0140741908,
    )
    assert not holding.complete
    assert not holding.timed_out
    assert holding.angular_z == 0.0
    assert math.hypot(holding.linear_x, holding.linear_y) == pytest.approx(
        0.0537638226,
    )
    assert search.position_hold_started_at_s == pytest.approx(10.04)

    # The outer-gate handoff is sticky. Do not resume the tighter yaw chase
    # while XY closes and heading remains inside the final settle envelope.
    parked = search.update(
        target - 0.0308,
        now_s=10.2,
        current_position_xy=(0.049, 0.0),
        measured_angular_speed_rps=0.02,
    )
    assert parked.complete
    assert parked.angular_z == 0.0
    assert not search.active


def test_visual_search_outer_gate_requires_low_measured_angular_speed() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3490658504,
        max_yaw_offset_rad=1.0471975512,
        yaw_tolerance_rad=0.0087266463,
        settle_yaw_tolerance_rad=0.0349065850,
        min_yaw_rate_rps=0.06,
        turn_timeout_s=10.0,
        max_turn_timeout_s=30.0,
        position_completion_tolerance_m=0.05,
        settle_max_angular_speed_rps=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    turning = search.update(
        target - 0.03,
        now_s=9.0,
        current_position_xy=(0.053, 0.0),
        measured_angular_speed_rps=0.051,
    )
    assert turning.angular_z > 0.0
    assert search.position_hold_started_at_s is None


def test_visual_search_position_hold_reacquires_yaw_outside_outer_gate() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        position_heading_reacquire_tolerance_rad=0.06,
        position_completion_tolerance_m=0.02,
        settle_max_angular_speed_rps=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None
    holding = search.update(
        target - 0.02,
        now_s=1.0,
        current_position_xy=(0.04, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert holding.angular_z == 0.0
    started = search.position_hold_started_at_s

    reacquire = search.update(
        target - 0.061,
        now_s=1.1,
        current_position_xy=(0.04, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert reacquire.angular_z > 0.0
    assert reacquire.linear_x == 0.0
    assert reacquire.linear_y == 0.0
    assert search.position_hold_started_at_s == started


def test_visual_search_position_hold_retains_xy_ownership_through_yaw_rebound() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3490658504,
        max_yaw_offset_rad=1.0471975512,
        yaw_tolerance_rad=0.0087266463,
        settle_yaw_tolerance_rad=0.0349065850,
        position_heading_reacquire_tolerance_rad=0.0698131701,
        min_yaw_rate_rps=0.06,
        max_planar_drift_m=0.15,
        position_completion_tolerance_m=0.06,
        moving_rebound_reacquire_m=0.08,
        max_position_hold_speed_mps=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    handoff = search.update(
        target - 0.03477,
        now_s=1.0,
        current_position_xy=(0.08065, 0.0),
        measured_angular_speed_rps=0.0129,
    )
    assert handoff.angular_z == 0.0
    assert math.hypot(handoff.linear_x, handoff.linear_y) > 0.0

    rebound = search.update(
        target - 0.03559,
        now_s=1.1,
        current_position_xy=(0.0811, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert rebound.angular_z == 0.0
    assert math.hypot(rebound.linear_x, rebound.linear_y) > 0.0

    reacquire = search.update(
        target - 0.0700,
        now_s=1.2,
        current_position_xy=(0.0811, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert reacquire.angular_z > 0.0
    assert reacquire.linear_x == 0.0
    assert reacquire.linear_y == 0.0

    yaw_only = search.update(
        target - 0.04,
        now_s=1.3,
        current_position_xy=(0.05, 0.0),
        measured_angular_speed_rps=0.06,
    )
    assert not yaw_only.complete
    assert yaw_only.angular_z > 0.0
    assert yaw_only.linear_x == 0.0
    assert yaw_only.linear_y == 0.0

    complete = search.update(
        target - 0.03,
        now_s=1.4,
        current_position_xy=(0.05, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert complete.complete
    assert complete.angular_z == 0.0


def test_visual_search_fast_outer_reentry_does_not_resume_position_hold() -> None:
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3,
        max_yaw_offset_rad=0.6,
        yaw_tolerance_rad=0.01,
        settle_yaw_tolerance_rad=0.03,
        position_heading_reacquire_tolerance_rad=0.06,
        position_completion_tolerance_m=0.02,
        settle_max_angular_speed_rps=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))
    target = search.target_yaw_rad
    assert target is not None

    holding = search.update(
        target - 0.02,
        now_s=1.0,
        current_position_xy=(0.04, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert holding.angular_z == 0.0
    assert holding.linear_x != 0.0
    started = search.position_hold_started_at_s

    outside = search.update(
        target - 0.061,
        now_s=1.1,
        current_position_xy=(0.04, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert outside.angular_z > 0.0
    assert outside.linear_x == 0.0

    fast_reentry = search.update(
        target - 0.025,
        now_s=1.2,
        current_position_xy=(0.04, 0.0),
        measured_angular_speed_rps=0.051,
    )
    assert fast_reentry.angular_z > 0.0
    assert fast_reentry.linear_x == 0.0
    assert fast_reentry.linear_y == 0.0
    assert search.position_hold_started_at_s == started

    slow_reentry = search.update(
        target - 0.02,
        now_s=1.3,
        current_position_xy=(0.04, 0.0),
        measured_angular_speed_rps=0.01,
    )
    assert slow_reentry.angular_z == 0.0
    assert slow_reentry.linear_x != 0.0
    assert search.position_hold_started_at_s == started


@pytest.mark.parametrize('speed', (-0.01, float('nan'), float('inf')))
def test_visual_search_rejects_invalid_measured_angular_speed(speed) -> None:
    search = BoundedYawSearch()
    assert search.start(0.0, now_s=0.0, current_position_xy=(0.0, 0.0))

    with pytest.raises(ValueError, match='measured angular speed'):
        search.update(
            0.0,
            now_s=0.1,
            current_position_xy=(0.0, 0.0),
            measured_angular_speed_rps=speed,
        )


def test_visual_search_accepts_measured_viewpoint_within_completion_tolerance():
    search = BoundedYawSearch(VisualSearchConfig(
        yaw_step_rad=0.3490658504,
        max_yaw_offset_rad=1.0471975512,
        yaw_tolerance_rad=0.0349065850,
        position_hold_deadband_m=0.01,
        position_completion_tolerance_m=0.05,
    ))
    assert search.start(0.0, now_s=0.0, current_position_xy=(1.0, 2.0))
    target = search.target_yaw_rad
    assert target is not None

    # Reproduce the terminal live sample: yaw and 3.8 cm XY drift are valid for
    # a measured search viewpoint even though XY is outside the command deadband.
    complete = search.update(
        target - 0.028946,
        now_s=1.0,
        current_position_xy=(1.038, 2.0),
    )
    assert complete.complete
    assert complete.angular_z == 0.0
    assert complete.linear_x == 0.0
    assert complete.linear_y == 0.0
    assert search.position_hold_started_at_s is None
    assert search.linear_command_base_xy == (0.0, 0.0)


def test_visual_search_position_completion_tolerance_is_bounded():
    with pytest.raises(ValueError, match='completion tolerance.*at least.*deadband'):
        VisualSearchConfig(
            position_hold_deadband_m=0.02,
            position_completion_tolerance_m=0.01,
        )
    with pytest.raises(ValueError, match='completion tolerance.*below.*drift'):
        VisualSearchConfig(
            position_completion_tolerance_m=0.15,
            max_planar_drift_m=0.15,
        )
    with pytest.raises(ValueError, match='moving rebound trigger'):
        VisualSearchConfig(
            position_completion_tolerance_m=0.05,
            moving_rebound_reacquire_m=0.05,
        )
    with pytest.raises(ValueError, match='moving rebound trigger'):
        VisualSearchConfig(
            moving_rebound_reacquire_m=0.15,
            max_planar_drift_m=0.15,
        )


def test_visual_search_position_deadband_must_remain_inside_hard_drift_gate():
    with pytest.raises(ValueError, match='deadband.*below.*drift'):
        VisualSearchConfig(
            position_hold_deadband_m=0.15,
            max_planar_drift_m=0.15,
        )


def test_platform_odometry_requires_configured_base_frame_contract():
    validate_platform_odometry_frames(
        'map', 'base_link',
        expected_parent_frame='map',
        expected_child_frame='base_link',
    )

    with pytest.raises(ValueError, match='map -> base_link.*map -> sensor'):
        validate_platform_odometry_frames(
            'map', 'sensor',
            expected_parent_frame='map',
            expected_child_frame='base_link',
        )
    with pytest.raises(ValueError, match='expected frames must be non-empty'):
        validate_platform_odometry_frames(
            'map', 'base_link',
            expected_parent_frame='',
            expected_child_frame='base_link',
        )


def test_position_hold_feedback_and_command_share_the_same_base_origin():
    validate_position_hold_frame_contract('base_link', 'base_link')

    with pytest.raises(ValueError, match='position-hold frame mismatch'):
        validate_position_hold_frame_contract('sensor', 'base_link')


def test_base_twist_speed_magnitudes_keep_si_domains_separate():
    linear_mps, angular_rps = base_twist_speed_magnitudes(
        (3.0, 4.0, 0.0),
        (0.0, 0.0, 0.2),
    )
    assert linear_mps == pytest.approx(5.0)
    assert angular_rps == pytest.approx(0.2)

    with pytest.raises(ValueError, match='finite linear and angular'):
        base_twist_speed_magnitudes(
            (float('nan'), 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    'override',
    (
        {'settle_max_linear_speed_mps': 0.0},
        {'settle_max_angular_speed_rps': float('nan')},
    ),
)
def test_visual_search_settle_speed_limits_must_be_finite_and_positive(override):
    with pytest.raises(ValueError, match='limits.*finite and positive'):
        VisualSearchConfig(**override)


@pytest.mark.parametrize('timeout_s', (-0.01, float('nan'), float('inf')))
def test_visual_search_stationary_wait_timeout_is_finite_and_non_negative(
    timeout_s,
):
    with pytest.raises(ValueError, match='stationary wait timeout'):
        VisualSearchConfig(stationary_wait_timeout_s=timeout_s)


def test_visual_search_stationary_wait_is_disabled_by_default():
    assert VisualSearchConfig().stationary_wait_timeout_s == 0.0


@pytest.mark.parametrize(
    'override',
    (
        {'stationary_quiet_window_s': 0.0},
        {'stationary_quiet_window_s': float('nan')},
        {'stationary_max_odom_gap_s': 0.0},
        {
            'stationary_quiet_window_s': 0.35,
            'stationary_max_odom_gap_s': 0.35,
        },
        {'settle_reacquire_budget_s': -0.01},
        {'settle_reacquire_budget_s': float('inf')},
    ),
)
def test_visual_search_stationarity_budgets_are_bounded(override):
    with pytest.raises(ValueError, match='stationary|reacquire'):
        VisualSearchConfig(**override)


def _quiet_window() -> ContinuousMotionQuietWindow:
    return ContinuousMotionQuietWindow(
        quiet_window_s=0.35,
        max_odom_gap_s=0.15,
        max_linear_speed_mps=0.035,
        max_angular_speed_rps=0.05,
    )


def test_motion_quiet_window_requires_both_receipt_and_source_duration():
    window = _quiet_window()
    window.reset(
        stop_received_at_s=10.0,
        minimum_odom_sequence=4,
        minimum_odom_stamp_ns=10_000_000_000,
    )
    window.observe(
        received_at_s=10.01,
        odom_sequence=5,
        odom_stamp_ns=10_010_000_000,
        linear_speed_mps=0.02,
        angular_speed_rps=0.04,
    )
    window.observe(
        received_at_s=10.37,
        odom_sequence=6,
        odom_stamp_ns=10_090_000_000,
        linear_speed_mps=0.02,
        angular_speed_rps=0.04,
    )
    assert not window.ready(
        odom_sequence=6,
        odom_stamp_ns=10_090_000_000,
        odom_seen_at_s=10.37,
    )

    window.observe(
        received_at_s=10.41,
        odom_sequence=7,
        odom_stamp_ns=10_410_000_000,
        linear_speed_mps=0.02,
        angular_speed_rps=0.04,
    )
    assert not window.ready(
        odom_sequence=7,
        odom_stamp_ns=10_410_000_000,
        odom_seen_at_s=10.41,
    )
    assert window.last_reset_reason == 'odom_discontinuity'


def test_motion_quiet_window_rejects_buffered_burst_across_receipt_gap():
    window = _quiet_window()
    window.reset(
        stop_received_at_s=10.0,
        minimum_odom_sequence=4,
        minimum_odom_stamp_ns=10_000_000_000,
    )
    samples = (
        (10.01, 5, 10.01),
        (10.40, 6, 10.11),
        (10.401, 7, 10.21),
        (10.402, 8, 10.31),
        (10.41, 9, 10.41),
    )
    for received_at_s, sequence, stamp_s in samples:
        window.observe(
            received_at_s=received_at_s,
            odom_sequence=sequence,
            odom_stamp_ns=int(round(stamp_s * 1e9)),
            linear_speed_mps=0.02,
            angular_speed_rps=0.04,
        )

    assert not window.ready(
        odom_sequence=9,
        odom_stamp_ns=10_410_000_000,
        odom_seen_at_s=10.41,
    )
    assert window.last_reset_reason == 'odom_discontinuity'
    assert window.stable_duration_s == pytest.approx(0.01)


def test_motion_quiet_window_cannot_age_one_cached_sample_into_success():
    window = _quiet_window()
    window.reset(
        stop_received_at_s=10.0,
        minimum_odom_sequence=4,
        minimum_odom_stamp_ns=10_000_000_000,
    )
    window.observe(
        received_at_s=10.04,
        odom_sequence=5,
        odom_stamp_ns=10_040_000_000,
        linear_speed_mps=0.0,
        angular_speed_rps=0.0,
    )
    assert not window.ready(
        odom_sequence=5,
        odom_stamp_ns=10_040_000_000,
        odom_seen_at_s=10.04,
    )
    assert window.stable_duration_s == 0.0


def test_motion_quiet_window_replays_go2w_ringdown_without_false_acceptance():
    window = _quiet_window()
    window.reset(
        stop_received_at_s=219.16,
        minimum_odom_sequence=1108,
        minimum_odom_stamp_ns=219_130_000_000,
    )
    sequence = 1108

    def observe(stamp_s: float, linear: float, angular: float) -> bool:
        nonlocal sequence
        sequence += 1
        stamp_ns = int(round(stamp_s * 1e9))
        window.observe(
            received_at_s=stamp_s,
            odom_sequence=sequence,
            odom_stamp_ns=stamp_ns,
            linear_speed_mps=linear,
            angular_speed_rps=angular,
        )
        return window.ready(
            odom_sequence=sequence,
            odom_stamp_ns=stamp_ns,
            odom_seen_at_s=stamp_s,
        )

    # Every short low-speed island in the recorded policy ring-down is unsafe.
    assert not observe(219.21, 0.0242, 0.0067)
    assert not observe(219.25, 0.0251, 0.0097)
    assert not observe(219.29, 0.0365, 0.0184)
    for stamp in (220.37, 220.41, 220.45, 220.49, 220.53):
        assert not observe(stamp, 0.032, 0.025)
    assert not observe(220.57, 0.042, 0.030)
    for stamp in (220.65, 220.69, 220.73, 220.77, 220.81, 220.85, 220.89):
        assert not observe(stamp, 0.030, 0.022)
    assert not observe(220.93, 0.036, 0.010)

    ready = False
    for index in range(10):
        stamp = 221.41 + 0.04 * index
        ready = observe(stamp, 0.032, 0.025)
        assert ready is (index == 9)
    assert ready
    assert window.stable_duration_s == pytest.approx(0.36)


@pytest.mark.parametrize('grace_s', (-0.01, float('nan'), float('inf')))
def test_visual_search_deadline_grace_is_finite_and_non_negative(grace_s):
    with pytest.raises(ValueError, match='deadline grace'):
        VisualSearchConfig(deadline_grace_s=grace_s)


def test_visual_search_deadline_grace_is_disabled_by_default():
    assert VisualSearchConfig().deadline_grace_s == 0.0


def test_horizontal_edge_gate_drives_recenter_and_rejects_flooded_box():
    assert horizontal_edge_direction(
        (0.02, 0.2, 0.06, 0.8), margin_ratio=0.08,
    ) == 1
    assert horizontal_edge_direction(
        (0.90, 0.2, 0.98, 0.8), margin_ratio=0.08,
    ) == -1
    assert horizontal_edge_direction(
        (0.20, 0.2, 0.40, 0.8), margin_ratio=0.08,
    ) == 0
    with pytest.raises(ValueError, match='both horizontal'):
        horizontal_edge_direction((0.01, 0.2, 0.99, 0.3), margin_ratio=0.08)


def test_vertical_edge_gate_requests_view_change_and_rejects_flooded_box():
    assert vertical_edge_direction(
        (0.2, 0.01, 0.4, 0.08), margin_ratio=0.08,
    ) == 1
    assert vertical_edge_direction(
        (0.2, 0.78, 0.4, 1.0), margin_ratio=0.08,
    ) == -1
    assert vertical_edge_direction(
        (0.2, 0.2, 0.4, 0.8), margin_ratio=0.08,
    ) == 0
    with pytest.raises(ValueError, match='both vertical'):
        vertical_edge_direction((0.2, 0.01, 0.3, 0.99), margin_ratio=0.08)


def test_image_edge_gate_rejects_zero_height_box():
    with pytest.raises(ValueError, match='no image area'):
        horizontal_edge_direction((0.2, 0.3, 0.4, 0.3), margin_ratio=0.08)


def test_carry_planning_ignores_status_before_trajectory_is_sent():
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.VERIFY
    core.verification_complete(carry_only=True)
    core.execution_update(parse_execution_status(
        'active;owner=trajectory;segment=lift;command_id=4',
    ))
    core.execution_update(parse_execution_status(
        'succeeded;owner=trajectory;segment=lift;command_id=4',
    ))
    assert core.phase is RuntimePhase.CARRY


def test_close_aperture_scales_with_candidate_width_and_is_bounded():
    values = {'fallback_m': 0.014, 'squeeze_m': 0.006, 'minimum_m': 0.008,
              'maximum_m': 0.075}
    assert grasp_close_aperture(0.050, **values) == pytest.approx(0.044)
    assert grasp_close_aperture(0.010, **values) == pytest.approx(0.008)
    assert grasp_close_aperture(None, **values) == pytest.approx(0.014)


def test_aperture_contract_rejects_unverifiable_planner_widths():
    common = {
        'candidate_min_m': 0.012,
        'open_aperture_m': 0.070,
        'squeeze_m': 0.006,
        'command_min_m': 0.0,
        'command_max_m': 0.075,
        'contact_margin_m': 0.0015,
    }
    validate_grasp_aperture_contract(candidate_max_m=0.068, **common)
    with pytest.raises(ValueError, match='actuator contract'):
        validate_grasp_aperture_contract(candidate_max_m=0.075, **common)
