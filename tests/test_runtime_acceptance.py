"""Contract tests for the autonomous mobile-manipulation acceptance runner."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = (
    ROOT / 'scripts' / 'runtime' / 'mobile_manipulation_acceptance.py'
)
SPEC = importlib.util.spec_from_file_location(
    'mobile_manip_acceptance',
    ACCEPTANCE,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _task_status(phase: str, **updates: object) -> str:
    value = {
        'schema': MODULE.TASK_STATUS_SCHEMA,
        'phase': phase,
        'instruction': 'place the bottle',
        'result': '',
        'place_goal_id': None,
        'place_plan_available': False,
        'failure': '',
    }
    value.update(updates)
    return json.dumps(value)


def _execution(
    trajectory: str,
    *,
    command_id: int = 8,
    segment: str = 'place_retreat',
    gripper_id: int = 4,
    accepted: float = 0.070,
    measured: float = 0.0698,
    contract_id: str = 'place-42',
    executor_epoch: str = 'executor-epoch-a',
    trajectory_received_at: float = 7.5,
    gripper_received_at: float = 8.5,
) -> str:
    return (
        f'{trajectory};owner=trajectory;command_id={command_id};'
        f'segment={segment};gripper=accepted:{accepted:.4f};'
        f'gripper_command_id={gripper_id};aperture={measured:.4f};'
        f'trajectory_contract_id={contract_id};'
        f'executor_epoch={executor_epoch};'
        f'trajectory_received_at={trajectory_received_at:.6f};'
        f'gripper_received_at={gripper_received_at:.6f}'
    )


def _post_release_verification(**updates: object) -> str:
    value = {
        'schema': MODULE.POST_RELEASE_VERIFICATION_SCHEMA,
        'state': 'verified',
        'result': MODULE.POST_RELEASE_VERIFICATION_RESULT,
        'failure': '',
        'observation_source': MODULE.POST_RELEASE_OBSERVATION_SOURCE,
        'goal_id': 'place-42',
        'place_goal_id': 'place-42',
        'release_gripper_command_id': 4,
        'request_id': 'request-7',
        'producer_epoch': 'verifier-epoch-2',
        'generation': 3,
        'frame_id': 'camera_color_optical_frame',
        'geometry_frame_id': 'piper_link',
        'planning_observation_stamp_ns': 8_000_000_000,
        'release_ack_stamp_ns': 9_000_000_000,
        'observation_start_stamp_ns': 9_900_000_000,
        'first_observation_stamp_ns': 10_000_000_000,
        'last_observation_stamp_ns': 10_500_000_000,
        'first_status_stamp_ns': 10_000_000_000,
        'last_status_stamp_ns': 10_500_000_000,
        'first_rgb_stamp_ns': 10_000_000_000,
        'first_depth_stamp_ns': 10_000_000_000,
        'first_target_stamp_ns': 10_000_000_000,
        'last_rgb_stamp_ns': 10_500_000_000,
        'last_depth_stamp_ns': 10_500_000_000,
        'last_target_stamp_ns': 10_500_000_000,
        'last_joint_stamp_ns': 10_490_000_000,
        'last_execution_status_received_ns': 10_400_000_000,
        'sample_count': 4,
        'target_point_count': 120,
        'stable_duration_s': 0.50,
        'max_target_motion_m': 0.012,
        'region_support_fraction': 0.92,
        'target_gripper_clearance_m': 0.08,
        'target_depth_correspondence_max_error_m': 0.008,
        'object_position_error_m': 0.02,
        'object_orientation_error_rad': 0.12,
        'object_upright_error_rad': 0.10,
        'object_registration_inlier_fraction': 0.82,
        'object_registration_rms_m': 0.011,
        'object_orientation_mode': 'axial',
        'planned_object_pose': [
            [1.0, 0.0, 0.0, 0.50],
            [0.0, 1.0, 0.0, 0.00],
            [0.0, 0.0, 1.0, 0.72],
            [0.0, 0.0, 0.0, 1.00],
        ],
        'observed_object_center_m': [0.51, 0.0, 0.72],
        'rejected_sample_count': 0,
        'rejected_sample_reasons': [],
    }
    value.update(updates)
    return json.dumps(value)


def _terminal_verification(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        'expected_goal_id': 'place-42',
        'expected_release_gripper_command_id': 4,
        'expected_request_id': 'request-7',
        'expected_producer_epoch': 'verifier-epoch-2',
        'expected_generation': 3,
        'expected_frame_id': 'camera_color_optical_frame',
        'expected_planning_observation_stamp_ns': 8_000_000_000,
        'verified': True,
        'sample_count': 4,
        'stable_duration_s': 0.50,
    }
    value.update(updates)
    return value


def _complete_task_status(**updates: object) -> str:
    value: dict[str, object] = {
        'result': 'mobile_manip_complete',
        'place_goal_id': 'place-42',
        'place_plan_available': True,
        'post_release_verification': _terminal_verification(),
    }
    value.update(updates)
    return _task_status('complete', **value)


def _observe_approach(
    evidence: object,
    *,
    command_id: int = 7,
    contract_id: str = 'place-42',
    executor_epoch: str = 'executor-epoch-a',
    source_s: float = 7.0,
) -> None:
    common = {
        'command_id': command_id,
        'segment': 'place_approach',
        'gripper_id': 3,
        'accepted': 0.030,
        'measured': 0.031,
        'contract_id': contract_id,
        'executor_epoch': executor_epoch,
        'trajectory_received_at': source_s,
        'gripper_received_at': source_s - 0.5,
    }
    evidence.observe_execution_status(_execution('active', **common))
    evidence.observe_execution_status(_execution('succeeded', **common))


def _observe_release(
    evidence: object,
    *,
    command_id: int = 7,
    contract_id: str = 'place-42',
    executor_epoch: str = 'executor-epoch-a',
    approach_source_s: float = 7.0,
    gripper_source_s: float = 8.5,
    gripper_id: int = 4,
    measurements: tuple[float, ...] = (0.0691, 0.0694, 0.0695),
) -> None:
    for measured in measurements:
        evidence.observe_execution_status(_execution(
            'succeeded',
            command_id=command_id,
            segment='place_approach',
            gripper_id=gripper_id,
            accepted=0.070,
            measured=measured,
            contract_id=contract_id,
            executor_epoch=executor_epoch,
            trajectory_received_at=approach_source_s,
            gripper_received_at=gripper_source_s,
        ))


def _observe_retreat(
    evidence: object,
    *,
    command_id: int = 8,
    contract_id: str = 'place-42',
    executor_epoch: str = 'executor-epoch-a',
    source_s: float = 9.5,
    gripper_source_s: float = 8.5,
) -> None:
    common = {
        'command_id': command_id,
        'segment': 'place_retreat',
        'gripper_id': 4,
        'accepted': 0.070,
        'measured': 0.0695,
        'contract_id': contract_id,
        'executor_epoch': executor_epoch,
        'trajectory_received_at': source_s,
        'gripper_received_at': gripper_source_s,
    }
    evidence.observe_execution_status(_execution('active', **common))
    evidence.observe_execution_status(_execution('succeeded', **common))


def _observe_execution_chain(evidence: object) -> None:
    _observe_approach(evidence)
    _observe_release(evidence)
    _observe_retreat(evidence)


def _observe_required_task_phases(evidence: object) -> None:
    for phase in (*MODULE.PICK_SUCCESS_PHASES, *MODULE.SUCCESS_PHASES[:-1]):
        evidence.observe_task_status(_task_status(phase))


def _prepared_evidence():
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_place_status(json.dumps({
        'state': 'planned',
        'detail': json.dumps({'goal_id': 'place-42'}),
    }))
    _observe_required_task_phases(evidence)
    return evidence


def test_success_requires_correlated_place_release_and_retreat_evidence():
    """A pass combines terminal, planner, release, retreat, and bag proof."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_place_status(json.dumps({
        'state': 'planned',
        'detail': json.dumps({'goal_id': 'place-42'}),
    }))
    _observe_required_task_phases(evidence)

    _observe_execution_chain(evidence)
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=True)
    assert checks
    assert all(checks.values())
    assert checks['pick_two_stage_phase_order_observed']
    summary = evidence.summary()
    assert summary['pick_two_stage_phase_order_observed']
    assert summary['phase_history'] == [
        *MODULE.PICK_SUCCESS_PHASES,
        *MODULE.SUCCESS_PHASES,
    ]


def test_pick_two_stage_phase_order_is_required_for_acceptance() -> None:
    """All pick phases in the wrong order cannot satisfy the full verdict."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_place_status(json.dumps({
        'state': 'planned',
        'detail': json.dumps({'goal_id': 'place-42'}),
    }))
    reordered = list(MODULE.PICK_SUCCESS_PHASES)
    reobserve = reordered.index('pregrasp_reobserve')
    approach_planning = reordered.index('approach_planning')
    reordered[reobserve], reordered[approach_planning] = (
        reordered[approach_planning],
        reordered[reobserve],
    )
    for phase in (*reordered, *MODULE.SUCCESS_PHASES[:-1]):
        evidence.observe_task_status(_task_status(phase))
    _observe_execution_chain(evidence)
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=True)
    assert not checks['pick_two_stage_phase_order_observed']
    assert not evidence.summary()['pick_two_stage_phase_order_observed']
    assert not all(checks.values())


def test_pick_phase_evidence_cannot_follow_place_transit() -> None:
    """Independent subsequences cannot reverse the pick/place chronology."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    observed = (*MODULE.SUCCESS_PHASES[:-1], *MODULE.PICK_SUCCESS_PHASES)
    for phase in observed:
        evidence.observe_task_status(_task_status(phase))
    evidence.observe_task_status(_complete_task_status())

    assert MODULE._ordered_subsequence(MODULE.PICK_SUCCESS_PHASES, observed)
    assert MODULE._ordered_subsequence(MODULE.SUCCESS_PHASES, (
        *observed,
        'complete',
    ))
    assert not evidence.checks(
        bag_closed_cleanly=True,
    )['pick_two_stage_phase_order_observed']


def test_pick_phase_evidence_cannot_span_two_planning_attempts() -> None:
    """A new planning phase discards every partial prior pick attempt."""
    split = 3
    observed = (
        *MODULE.PICK_SUCCESS_PHASES[:split],
        'planning',
        *MODULE.PICK_SUCCESS_PHASES[split:],
        *MODULE.SUCCESS_PHASES[:-1],
    )
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    for phase in observed:
        evidence.observe_task_status(_task_status(phase))
    evidence.observe_task_status(_complete_task_status())

    assert MODULE._ordered_subsequence(MODULE.PICK_SUCCESS_PHASES, observed)
    assert not evidence.checks(
        bag_closed_cleanly=True,
    )['pick_two_stage_phase_order_observed']


@pytest.mark.parametrize(
    'missing',
    (
        'place_plan',
        'release',
        'retreat_active',
        'retreat_succeeded',
        'post_release',
        'bag',
    ),
)
def test_terminal_complete_alone_cannot_pass_observed_place(missing: str):
    """Each independent observed-place predicate is mandatory."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    if missing != 'place_plan':
        evidence.observe_place_status(json.dumps({
            'state': 'planned',
            'detail': json.dumps({'goal_id': 'place-42'}),
        }))
    _observe_required_task_phases(evidence)
    _observe_approach(evidence)
    if missing != 'release':
        _observe_release(evidence)
    if missing != 'retreat_active':
        common = {
            'command_id': 8,
            'segment': 'place_retreat',
            'trajectory_received_at': 9.5,
            'gripper_received_at': 8.5,
        }
        evidence.observe_execution_status(_execution('active', **common))
    if missing != 'retreat_succeeded':
        common = {
            'command_id': 8,
            'segment': 'place_retreat',
            'trajectory_received_at': 9.5,
            'gripper_received_at': 8.5,
        }
        evidence.observe_execution_status(_execution('succeeded', **common))
    if missing != 'post_release':
        evidence.observe_post_release_verification(
            _post_release_verification(),
        )
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=missing != 'bag')
    assert not all(checks.values())
    if missing != 'bag':
        assert checks['observed_place_success'] is False


def test_motion_success_without_post_release_geometry_fails_closed():
    """Current runtime cannot pass by opening and retreating alone."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_place_status(json.dumps({
        'state': 'planned',
        'detail': json.dumps({'goal_id': 'place-42'}),
    }))
    _observe_required_task_phases(evidence)
    _observe_execution_chain(evidence)
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=True)
    assert checks['place_execution_evidence']
    assert not checks['post_release_target_stable_in_region_observed']
    assert not checks['observed_place_success']
    assert not all(checks.values())


def test_retreat_with_wrong_or_missing_executor_contract_fails_closed(
) -> None:
    """A retreat from another place transaction cannot satisfy completion."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_place_status(json.dumps({
        'state': 'planned',
        'detail': json.dumps({'goal_id': 'place-42'}),
    }))
    _observe_required_task_phases(evidence)
    _observe_approach(evidence)
    _observe_release(evidence)
    _observe_retreat(evidence, contract_id='other-place-goal')
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    assert not evidence.checks(
        bag_closed_cleanly=True,
    )['observed_place_success']


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('observation_source', 'simulator_ground_truth'),
        ('sample_count', 1),
        ('target_point_count', 4),
        ('stable_duration_s', 0.1),
        ('max_target_motion_m', 0.20),
        ('region_support_fraction', 0.20),
        ('target_gripper_clearance_m', 0.005),
        ('target_depth_correspondence_max_error_m', 0.013),
        ('object_position_error_m', 0.041),
        ('object_orientation_error_rad', 0.351),
        ('object_upright_error_rad', 0.261),
        ('object_registration_inlier_fraction', 0.549),
        ('object_registration_rms_m', 0.026),
        ('planned_object_pose', [[1.0, 0.0], [0.0, 1.0]]),
        ('observed_object_center_m', [0.0, float('nan'), 0.0]),
        ('last_observation_stamp_ns', 10_100_000_000),
    ),
)
def test_post_release_geometry_contract_rejects_weak_or_truth_evidence(
    field: str,
    value: object,
):
    """One flag cannot replace bounded multi-frame RGB-D geometry."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_post_release_verification(
        _post_release_verification(**{field: value}),
    )
    assert not evidence.post_release_verifications


def test_post_release_old_schema_or_missing_v2_field_is_rejected() -> None:
    """Only a complete v2 independent geometry payload is retained."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_post_release_verification(_post_release_verification(
        schema='z_manip.post_release_verification.v1',
    ))
    assert not evidence.post_release_verifications

    value = json.loads(_post_release_verification())
    value.pop('object_registration_rms_m')
    evidence.observe_post_release_verification(json.dumps(value))
    assert not evidence.post_release_verifications


def test_stale_status_is_ignored_and_second_task_publication_is_rejected():
    """A latched old task cannot satisfy the new one or trigger a retry."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    assert evidence.observe_task_status(_task_status('failed')) is None
    evidence.mark_task_published()
    stale = json.loads(_task_status('complete'))
    stale['instruction'] = 'an old task'
    assert evidence.observe_task_status(json.dumps(stale)) is None
    assert evidence.task_status_count == 0
    with pytest.raises(MODULE.AcceptanceError, match='more than once'):
        evidence.mark_task_published()


def test_terminal_drain_starts_at_first_terminal_receipt():
    """Repeated transient-local terminal samples cannot postpone shutdown."""
    received_at = MODULE._latch_terminal_receipt(None, 'failed', 10.0)
    assert received_at == 10.0
    assert MODULE._latch_terminal_receipt(
        received_at,
        'failed',
        30.0,
    ) == 10.0
    assert MODULE._latch_terminal_receipt(None, 'coarse_nav', 10.0) is None


def test_release_feedback_must_be_stable_and_retreat_identity_must_match():
    """Noisy opening or split active/success identities fail closed."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_task_status(_task_status('place_approach'))
    _observe_approach(evidence)
    _observe_release(
        evidence,
        measurements=(0.0651, 0.0680, 0.0652),
    )
    evidence.observe_execution_status(_execution(
        'active',
        command_id=10,
        segment='place_retreat',
        trajectory_received_at=9.5,
    ))
    evidence.observe_execution_status(_execution(
        'succeeded',
        command_id=11,
        segment='place_retreat',
        trajectory_received_at=9.6,
    ))

    checks = evidence.checks(bag_closed_cleanly=True)
    assert not checks['stable_measured_gripper_release_observed']
    assert not checks['place_retreat_active_then_succeeded_observed']


def test_retreat_before_release_cannot_be_reordered_into_success() -> None:
    """Receipt order prevents set aggregation from inventing a valid chain."""
    evidence = _prepared_evidence()
    _observe_approach(evidence)
    _observe_retreat(evidence)
    _observe_release(evidence)
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=True)
    assert not checks['physical_place_event_order_observed']
    assert not checks['observed_place_success']


def test_release_without_observed_approach_active_fails_closed() -> None:
    """Release feedback alone cannot stand in for approach completion."""
    evidence = _prepared_evidence()
    _observe_release(evidence)
    _observe_retreat(evidence)
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=True)
    assert not checks['place_approach_active_then_succeeded_observed']
    assert not checks['stable_measured_gripper_release_observed']
    assert not checks['observed_place_success']


@pytest.mark.parametrize(
    'fault',
    ('same_command', 'foreign_epoch', 'success_first'),
)
def test_retreat_identity_and_transition_order_are_immutable(
    fault: str,
) -> None:
    """Retreat must be a later transition in the frozen executor epoch."""
    evidence = _prepared_evidence()
    _observe_approach(evidence)
    _observe_release(evidence)
    if fault == 'same_command':
        _observe_retreat(evidence, command_id=7)
    elif fault == 'foreign_epoch':
        _observe_retreat(evidence, executor_epoch='executor-epoch-b')
    else:
        common = {
            'command_id': 8,
            'segment': 'place_retreat',
            'trajectory_received_at': 9.5,
            'gripper_received_at': 8.5,
        }
        evidence.observe_execution_status(_execution('succeeded', **common))
        evidence.observe_execution_status(_execution('active', **common))
        evidence.observe_execution_status(_execution('succeeded', **common))
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    assert not evidence.checks(
        bag_closed_cleanly=True,
    )['physical_place_event_order_observed']


def test_same_gripper_command_cannot_change_source_identity() -> None:
    """Stable samples cannot mix two source times under one command ID."""
    evidence = _prepared_evidence()
    _observe_approach(evidence)
    _observe_release(evidence, measurements=(0.0691, 0.0692))
    _observe_release(
        evidence,
        gripper_source_s=8.6,
        measurements=(0.0693,),
    )
    _observe_release(evidence, measurements=(0.0694,))
    _observe_retreat(evidence)
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=True)
    assert not checks['stable_measured_gripper_release_observed']
    assert not checks['observed_place_success']


@pytest.mark.parametrize(
    ('field', 'foreign'),
    (
        ('expected_goal_id', 'place-foreign'),
        ('expected_release_gripper_command_id', 99),
        ('expected_request_id', 'request-foreign'),
        ('expected_producer_epoch', 'verifier-epoch-foreign'),
        ('expected_generation', 99),
        ('expected_frame_id', 'foreign_frame'),
        ('expected_planning_observation_stamp_ns', 8_000_000_001),
        ('verified', False),
        ('sample_count', 5),
        ('stable_duration_s', 0.6),
    ),
)
def test_terminal_must_bind_exact_post_release_identity(
    field: str,
    foreign: object,
) -> None:
    """Every task expectation must equal the independent v2 evidence."""
    evidence = _prepared_evidence()
    _observe_execution_chain(evidence)
    evidence.observe_post_release_verification(_post_release_verification())
    expected = _terminal_verification(**{field: foreign})
    evidence.observe_task_status(_complete_task_status(
        post_release_verification=expected,
    ))

    checks = evidence.checks(bag_closed_cleanly=True)
    assert not checks['terminal_post_release_identity_correlated']
    assert not checks['observed_place_success']


def test_terminal_missing_expected_field_or_foreign_v2_replay_fails() -> None:
    """Incomplete terminal identity and a valid foreign replay both fail."""
    evidence = _prepared_evidence()
    _observe_execution_chain(evidence)
    evidence.observe_post_release_verification(_post_release_verification(
        request_id='request-foreign',
    ))
    expected = _terminal_verification()
    expected.pop('expected_frame_id')
    evidence.observe_task_status(_complete_task_status(
        post_release_verification=expected,
    ))

    assert not evidence.checks(
        bag_closed_cleanly=True,
    )['terminal_post_release_identity_correlated']


@pytest.mark.parametrize('fault', ('release_ack', 'observation_start'))
def test_executor_source_time_must_precede_visual_verification(
    fault: str,
) -> None:
    """Executor ROS source stamps must precede downstream verification."""
    evidence = _prepared_evidence()
    _observe_approach(evidence)
    if fault == 'release_ack':
        _observe_release(evidence, gripper_source_s=9.1)
        _observe_retreat(evidence, source_s=9.5, gripper_source_s=9.1)
    else:
        _observe_release(evidence)
        _observe_retreat(evidence, source_s=9.95)
    evidence.observe_post_release_verification(_post_release_verification())
    evidence.observe_task_status(_complete_task_status())

    checks = evidence.checks(bag_closed_cleanly=True)
    assert checks['physical_place_event_order_observed']
    assert not checks['terminal_post_release_identity_correlated']


def test_duplicate_json_and_execution_fields_are_rejected() -> None:
    """Ambiguous serialized evidence is rejected instead of overwritten."""
    assert MODULE._json_object('{"a":1,"a":2}') is None
    assert MODULE._json_object('{"a":NaN}') is None
    duplicated = _execution('active') + ';executor_epoch=duplicate'
    assert MODULE._execution_fields(duplicated) == {}


def test_first_terminal_task_status_is_immutable() -> None:
    """A repeated terminal sample cannot replace the first outcome."""
    evidence = MODULE.AcceptanceEvidence('place the bottle')
    evidence.mark_task_published()
    evidence.observe_task_status(_task_status('failed', failure='first'))
    evidence.observe_task_status(_complete_task_status())
    assert evidence.terminal_status is not None
    assert evidence.terminal_status['phase'] == 'failed'
    assert evidence.terminal_status['failure'] == 'first'


def test_default_topics_exclude_truth_and_include_local_velocity():
    """Evidence observes runtime contracts, not simulator object odometry."""
    topics = set(MODULE.DEFAULT_BAG_TOPICS)
    assert '/local_movement_cmd_vel' in topics
    assert '/z_manip/task/status' in topics
    assert '/z_manip/place/status' in topics
    assert MODULE.POST_RELEASE_VERIFICATION_TOPIC in topics
    assert '/piper/execution_status' in topics
    assert '/track_3d/seed_request' in topics
    assert '/track_3d/seed_offer_manifest' in topics
    assert '/track_3d/seed_status' in topics
    assert '/track_3d/exact_seed_image' in topics
    assert not any(topic.startswith('/objects/') for topic in topics)


def test_acceptance_and_supervisor_share_the_complete_critical_graph() -> None:
    """Acceptance cannot weaken the singleton supervisor readiness contract."""
    supervisor = runpy.run_path(str(
        ROOT / 'scripts' / 'runtime' / 'mobile_manipulation_supervisor.py'
    ))
    assert (
        MODULE.DEFAULT_CRITICAL_NODES == supervisor['DEFAULT_CRITICAL_NODES']
    )
    assert (
        MODULE.DEFAULT_CRITICAL_TOPICS == supervisor['DEFAULT_CRITICAL_TOPICS']
    )
    assert set(MODULE.DEFAULT_CRITICAL_NODES) == {
        'z_manip_urdf_root_alias',
        'vlm_edgetam_bridge',
        'z_manip_edgetam',
        'z_manip_complete_joint_state',
        'z_manip_robot_state_publisher',
        'z_manip_coarse_navigation',
        'z_manip_observed_placement',
        'z_manip_task_runtime',
    }


@pytest.mark.parametrize(
    ('owner_kind', 'owner'),
    [
        *[('node', name) for name in MODULE.DEFAULT_CRITICAL_NODES],
        *[('topic', name) for name in MODULE.DEFAULT_CRITICAL_TOPICS],
    ],
)
@pytest.mark.parametrize('fault_count', (0, 2))
def test_acceptance_rejects_each_missing_or_duplicate_critical_owner(
    owner_kind: str,
    owner: str,
    fault_count: int,
) -> None:
    """No individual subsystem owner may disappear or multiply at readiness."""
    nodes = {name: 1 for name in MODULE.DEFAULT_CRITICAL_NODES}
    topics = {name: 1 for name in MODULE.DEFAULT_CRITICAL_TOPICS}
    (nodes if owner_kind == 'node' else topics)[owner] = fault_count

    assert not MODULE._critical_graph_ready(nodes, topics)


def test_real_upstream_readiness_does_not_require_simulation_clock():
    """Real deployment retains sensors but does not wait for `/clock`."""
    simulation = MODULE._default_upstream_topics('true')
    real = MODULE._default_upstream_topics('false')
    assert '/clock' in simulation
    assert '/clock' not in real
    assert set(real) == set(simulation) - {'/clock'}


def test_task_platform_parameters_are_optional_validated_and_forwarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance selects no robot pose vocabulary without an explicit file."""
    monkeypatch.delenv('Z_MANIP_TASK_PLATFORM_PARAMETERS', raising=False)
    profile = tmp_path / 'go2w_sim.yaml'
    profile.write_text('z_manip_task_runtime:\n  ros__parameters: {}\n')

    assert MODULE._arguments([]).task_platform_parameters == ''
    assert MODULE._optional_path_argument('', 'task profile') is None
    assert MODULE._optional_path_argument(
        str(profile),
        'task profile',
    ) == str(profile)
    with pytest.raises(MODULE.AcceptanceError, match='not a file'):
        MODULE._optional_path_argument(
            str(tmp_path / 'missing.yaml'),
            'task profile',
        )

    arguments = MODULE._arguments([
        '--task-platform-parameters',
        str(profile),
    ])
    assert arguments.task_platform_parameters == str(profile)
    source = ACCEPTANCE.read_text()
    assert 'task_platform_file = _optional_path_argument(' in source
    assert "f'task_platform_parameters:={task_platform_file}'" in source


def test_runtime_image_installs_acceptance_entrypoint():
    """The production image exposes and smoke-checks the runner."""
    dockerfile = (ROOT / 'docker' / 'runtime' / 'Dockerfile').read_text()
    dockerignore = (
        ROOT / 'docker' / 'runtime' / 'Dockerfile.dockerignore'
    ).read_text()
    smoke = (ROOT / 'docker' / 'runtime' / 'smoke.sh').read_text()
    assert 'scripts/runtime/mobile_manipulation_acceptance.py' in dockerfile
    entrypoint = '/usr/local/bin/z-manip-mobile-manipulation-acceptance'
    assert entrypoint in dockerfile
    assert '!scripts/runtime/mobile_manipulation_acceptance.py' in dockerignore
    assert 'command -v z-manip-mobile-manipulation-acceptance' in smoke
    assert 'rosbag2_storage_mcap' in smoke
    assert ACCEPTANCE.stat().st_mode & 0o111


def test_compose_wires_complete_external_runtime_contract() -> None:
    """Supply DDS, model paths, assets, and a private environment file."""
    compose = (ROOT / 'docker' / 'runtime' / 'compose.yaml').read_text()
    assert 'FASTDDS_BUILTIN_TRANSPORTS: UDPv4' in compose
    assert 'Z_MANIP_ENV_FILE' in compose
    assert 'required: false' in compose
    assert 'OPENROUTER_API_KEY' not in compose
    assert (
        'Z_MANIP_STACK_CONFIG: /opt/z_manip/configs/go2w_piper.json'
        in compose
    )
    assert (
        'Z_MANIP_ROBOT_URDF: /robot/assets/urdf/go2w_sensored.urdf'
        in compose
    )
    assert (
        'Z_MANIP_ROBOT_DESCRIPTION_FILE: '
        '/robot/assets/urdf/go2w_sensored.urdf'
        in compose
    )
    assert (
        'Z_MANIP_COLLISION_MODEL_FILE: '
        '/opt/z_manip/configs/piper_collision_capsules.json'
        in compose
    )
    assert ':/robot/assets:ro' in compose


def test_runtime_readme_passes_complete_model_paths_to_supervisor() -> None:
    """Documented launch cannot omit meshes or place collision geometry."""
    readme = (ROOT / 'docker' / 'runtime' / 'README.md').read_text()
    assert 'source /opt/z_manip_ws/install/setup.bash' in readme
    assert (
        'robot_description_file:=/robot/assets/urdf/go2w_sensored.urdf'
        in readme
    )
    assert (
        'collision_model_file:='
        '/opt/z_manip/configs/piper_collision_capsules.json'
        in readme
    )
    assert (
        'task_platform_parameters:='
        '/opt/z_manip_ws/install/share/z_manip_task/config/go2w_sim.yaml'
        in readme
    )
    assert 'Z_MANIP_ROBOT_ASSETS' in readme
    assert 'Z_MANIP_ENV_FILE' in readme
