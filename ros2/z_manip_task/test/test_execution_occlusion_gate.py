"""Fail-closed lifecycle tests for bounded wrist-camera occlusion."""

from types import SimpleNamespace

import pytest

from z_manip_task.core import (
    ExecutionOcclusionConfig,
    ExecutionOcclusionGate,
    RuntimePhase,
)


ENDPOINT = (0.1, -0.2, 0.3)
OBSERVATION_STAMP_NS = 9_950_000_000
OBSERVATION_FRAME = 'wrist_depth_optical_frame'


def _source(now: float) -> tuple[int, int]:
    return int(round(now * 1000.0)), int(round((now - 0.01) * 1e9))


def _mark_loss(gate: ExecutionOcclusionGate, now: float) -> None:
    sequence, stamp_ns = _source(now)
    gate.mark_loss(
        now,
        joint_source_stamp_ns=stamp_ns,
        joint_sequence=sequence,
    )


def _evaluate(gate: ExecutionOcclusionGate, **kwargs):
    sequence, stamp_ns = _source(kwargs['now_s'])
    kwargs.setdefault('joint_source_stamp_ns', stamp_ns)
    kwargs.setdefault('joint_sequence', sequence)
    return gate.evaluate(**kwargs)


def _restore_tracking(gate: ExecutionOcclusionGate, now: float) -> bool:
    return gate.tracking_restored(
        now,
        request_id='task-request',
        producer_epoch='bridge-epoch',
        generation=4,
        observation_serial=int(round(now * 1000.0)),
        observation_stamp_ns=int(round((now - 0.01) * 1e9)),
        observation_frame_id=OBSERVATION_FRAME,
    )


def _arm(
    gate: ExecutionOcclusionGate,
    *,
    now: float = 10.0,
    measured=ENDPOINT,
    joint_seen: float = 9.9,
    exact: bool = True,
) -> None:
    gate.arm_near_contact(
        now_s=now,
        exact_authorized=exact,
        request_id='task-request',
        producer_epoch='bridge-epoch',
        generation=4,
        observation_serial=12,
        observation_stamp_ns=OBSERVATION_STAMP_NS,
        observation_frame_id=OBSERVATION_FRAME,
        measured_joints=measured,
        approach_endpoint_joints=ENDPOINT,
        joint_seen_at_s=joint_seen,
        joint_source_stamp_ns=_source(now)[1],
        joint_sequence=_source(now)[0],
    )


@pytest.mark.parametrize(
    ('exact', 'measured', 'joint_seen', 'reason'),
    (
        (False, ENDPOINT, 9.9, 'exactly authorized'),
        (True, (0.1, -0.2, 0.5), 9.9, 'approach endpoint'),
        (True, ENDPOINT, 9.0, 'joint feedback is stale'),
    ),
)
def test_near_contact_arming_requires_exact_geometry_and_measured_endpoint(
    exact,
    measured,
    joint_seen,
    reason,
) -> None:
    gate = ExecutionOcclusionGate()

    with pytest.raises(ValueError, match=reason):
        _arm(
            gate,
            exact=exact,
            measured=measured,
            joint_seen=joint_seen,
        )

    assert not gate.armed


def test_execution_occlusion_duration_has_a_non_configurable_hard_ceiling() -> None:
    assert ExecutionOcclusionConfig(max_duration_s=3.0).max_duration_s == 3.0
    with pytest.raises(ValueError, match='3.0 s hard limit'):
        ExecutionOcclusionConfig(max_duration_s=3.000_001)


def test_loss_floor_uses_latest_exact_bundle_and_rejects_replay_or_split() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    assert gate.retain_exact_observation(
        request_id='task-request',
        producer_epoch='bridge-epoch',
        generation=4,
        observation_serial=13,
        observation_stamp_ns=10_050_000_000,
        observation_frame_id=OBSERVATION_FRAME,
    )
    _mark_loss(gate, 10.1)

    assert not gate.tracking_restored(
        10.2,
        request_id='task-request',
        producer_epoch='bridge-epoch',
        generation=4,
        observation_serial=13,
        observation_stamp_ns=10_050_000_000,
        observation_frame_id=OBSERVATION_FRAME,
    )
    assert gate.loss_active
    assert not gate.tracking_restored(
        10.21,
        request_id='task-request',
        producer_epoch='bridge-epoch',
        generation=4,
        observation_serial=14,
        observation_stamp_ns=10_050_000_000,
        observation_frame_id=OBSERVATION_FRAME,
    )
    assert gate.loss_active
    with pytest.raises(ValueError, match='does not match'):
        gate.tracking_restored(
            10.22,
            request_id='task-request',
            producer_epoch='bridge-epoch',
            generation=4,
            observation_serial=14,
            observation_stamp_ns=10_210_000_000,
            observation_frame_id='different_camera_frame',
        )
    assert gate.loss_active
    assert gate.tracking_restored(
        10.23,
        request_id='task-request',
        producer_epoch='bridge-epoch',
        generation=4,
        observation_serial=14,
        observation_stamp_ns=10_210_000_000,
        observation_frame_id=OBSERVATION_FRAME,
    )


def test_prediction_requires_a_new_monotonic_joint_source_after_loss() -> None:
    gate = ExecutionOcclusionGate(ExecutionOcclusionConfig(
        joint_state_max_age_s=1.0,
        command_ack_timeout_s=0.1,
    ))
    _arm(gate)
    loss_sequence, loss_stamp_ns = _source(10.1)
    _mark_loss(gate, 10.1)

    replay = gate.evaluate(
        now_s=10.11,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.11,
        joint_source_stamp_ns=loss_stamp_ns,
        joint_sequence=loss_sequence,
        close_command_sent_at_s=10.1,
    )
    assert replay.allowed
    assert replay.mode == 'awaiting_joint_sample'

    expired_wait = gate.evaluate(
        now_s=10.21,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.21,
        joint_source_stamp_ns=loss_stamp_ns,
        joint_sequence=loss_sequence,
        close_command_sent_at_s=10.1,
    )
    assert not expired_wait.allowed
    assert 'acknowledgement timed out' in expired_wait.reason

    accepted = gate.evaluate(
        now_s=10.22,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.22,
        joint_source_stamp_ns=10_210_000_000,
        joint_sequence=loss_sequence + 1,
        close_command_sent_at_s=10.1,
        close_acknowledged=True,
        execution_status_seen_at_s=10.22,
    )
    assert accepted.allowed
    duplicate = gate.evaluate(
        now_s=10.23,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.22,
        joint_source_stamp_ns=10_210_000_000,
        joint_sequence=loss_sequence + 1,
        close_command_sent_at_s=10.1,
        close_acknowledged=True,
        execution_status_seen_at_s=10.23,
    )
    assert duplicate.allowed
    split = gate.evaluate(
        now_s=10.24,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.24,
        joint_source_stamp_ns=10_210_000_000,
        joint_sequence=loss_sequence + 2,
        close_command_sent_at_s=10.1,
        close_acknowledged=True,
        execution_status_seen_at_s=10.24,
    )
    assert not split.allowed
    assert 'backwards or split' in split.reason


def test_awaiting_joint_sample_still_enforces_closing_ack_deadline() -> None:
    gate = ExecutionOcclusionGate(ExecutionOcclusionConfig(
        joint_state_max_age_s=1.0,
        command_ack_timeout_s=0.4,
    ))
    _arm(gate)
    loss_sequence, loss_stamp_ns = _source(10.2)
    _mark_loss(gate, 10.2)

    decision = gate.evaluate(
        now_s=10.41,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.2,
        joint_source_stamp_ns=loss_stamp_ns,
        joint_sequence=loss_sequence,
        close_command_sent_at_s=10.0,
    )

    assert not decision.allowed
    assert 'acknowledgement timed out' in decision.reason


def test_identity_checked_lift_completion_is_an_acknowledged_lift_sample() -> None:
    path = (ENDPOINT, (0.14, -0.16, 0.34))

    def gate_at_loss() -> ExecutionOcclusionGate:
        gate = ExecutionOcclusionGate(ExecutionOcclusionConfig(
            joint_state_max_age_s=1.0,
        ))
        _arm(gate)
        gate.confirm_contact(10.01)
        gate.note_lift_sent(10.02)
        _mark_loss(gate, 10.2)
        return gate

    completed = gate_at_loss()
    accepted = completed.evaluate(
        now_s=10.6,
        phase=RuntimePhase.LIFT,
        measured_joints=path[-1],
        joint_seen_at_s=10.6,
        joint_source_stamp_ns=10_590_000_000,
        joint_sequence=10_600,
        execution_status_seen_at_s=10.6,
        lift_path=path,
        lift_execution_completed=True,
    )
    assert accepted.allowed
    assert accepted.mode == 'predicted_lift'

    unacknowledged = gate_at_loss().evaluate(
        now_s=10.6,
        phase=RuntimePhase.LIFT,
        measured_joints=path[-1],
        joint_seen_at_s=10.6,
        joint_source_stamp_ns=10_590_000_000,
        joint_sequence=10_600,
        execution_status_seen_at_s=10.6,
        lift_path=path,
    )
    assert not unacknowledged.allowed
    assert 'acknowledgement timed out' in unacknowledged.reason


def test_lift_path_cannot_advance_on_a_reused_joint_source_sample() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    _mark_loss(gate, 10.1)
    gate.confirm_contact(10.2)
    gate.note_lift_sent(10.21)
    path = (ENDPOINT, (0.12, -0.18, 0.32), (0.14, -0.16, 0.34))
    sequence, stamp_ns = _source(10.3)
    first = gate.evaluate(
        now_s=10.3,
        phase=RuntimePhase.LIFT,
        measured_joints=path[1],
        joint_seen_at_s=10.3,
        joint_source_stamp_ns=stamp_ns,
        joint_sequence=sequence,
        execution_status_seen_at_s=10.3,
        lift_path=path,
        lift_execution_active=True,
    )
    replayed_progress = gate.evaluate(
        now_s=10.31,
        phase=RuntimePhase.LIFT,
        measured_joints=path[2],
        joint_seen_at_s=10.31,
        joint_source_stamp_ns=stamp_ns,
        joint_sequence=sequence,
        execution_status_seen_at_s=10.31,
        lift_path=path,
        lift_execution_active=True,
    )
    assert first.allowed
    assert not replayed_progress.allowed
    assert 'without a new joint source sample' in replayed_progress.reason


def test_loss_before_near_contact_is_never_predicted() -> None:
    gate = ExecutionOcclusionGate()

    with pytest.raises(ValueError, match='not armed'):
        _mark_loss(gate, 10.0)


def test_prediction_is_forbidden_during_approach_even_if_gate_was_armed() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    _mark_loss(gate, 10.1)

    decision = _evaluate(
        gate,
        now_s=10.2,
        phase=RuntimePhase.APPROACH,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.2,
    )

    assert not decision.allowed
    assert 'forbidden during approach' in decision.reason


def test_closing_requires_ack_within_grace_and_fresh_matching_feedback() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    _mark_loss(gate, 10.1)

    grace = _evaluate(
        gate,
        now_s=10.2,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.2,
        close_command_sent_at_s=10.1,
    )
    assert grace.allowed
    assert grace.mode == 'predicted_closing'

    timed_out = _evaluate(
        gate,
        now_s=10.6,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.6,
        close_command_sent_at_s=10.1,
    )
    assert not timed_out.allowed
    assert 'acknowledgement timed out' in timed_out.reason

    acknowledged = ExecutionOcclusionGate()
    _arm(acknowledged)
    _mark_loss(acknowledged, 10.1)
    accepted = _evaluate(
        acknowledged,
        now_s=10.6,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.6,
        close_command_sent_at_s=10.1,
        close_acknowledged=True,
        execution_status_seen_at_s=10.55,
    )
    assert accepted.allowed

    stale = _evaluate(
        acknowledged,
        now_s=10.9,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.9,
        close_command_sent_at_s=10.1,
        close_acknowledged=True,
        execution_status_seen_at_s=10.55,
    )
    assert not stale.allowed
    assert 'feedback is stale' in stale.reason


def test_lift_requires_contact_executor_identity_and_validated_path_progress() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    _mark_loss(gate, 10.1)
    gate.confirm_contact(10.2)
    gate.note_lift_sent(10.3)
    path = (
        ENDPOINT,
        (0.12, -0.18, 0.32),
        (0.14, -0.16, 0.34),
    )

    accepted = _evaluate(
        gate,
        now_s=10.4,
        phase=RuntimePhase.LIFT,
        measured_joints=path[1],
        joint_seen_at_s=10.4,
        execution_status_seen_at_s=10.39,
        lift_path=path,
        lift_execution_active=True,
    )
    assert accepted.allowed
    assert accepted.path_index == 1

    deviation = _evaluate(
        gate,
        now_s=10.5,
        phase=RuntimePhase.LIFT,
        measured_joints=(0.5, 0.5, 0.5),
        joint_seen_at_s=10.5,
        execution_status_seen_at_s=10.49,
        lift_path=path,
        lift_execution_active=True,
    )
    assert not deviation.allowed
    assert 'left the validated lift path' in deviation.reason


def test_lift_progress_cannot_regress_beyond_the_sample_bound() -> None:
    gate = ExecutionOcclusionGate(ExecutionOcclusionConfig(
        max_path_regression_samples=1,
    ))
    _arm(gate)
    _mark_loss(gate, 10.1)
    gate.confirm_contact(10.2)
    gate.note_lift_sent(10.3)
    path = tuple(
        (ENDPOINT[0] + index * 0.02, ENDPOINT[1], ENDPOINT[2])
        for index in range(6)
    )

    forward = _evaluate(
        gate,
        now_s=10.4,
        phase=RuntimePhase.LIFT,
        measured_joints=path[4],
        joint_seen_at_s=10.4,
        execution_status_seen_at_s=10.4,
        lift_path=path,
        lift_execution_active=True,
    )
    backward = _evaluate(
        gate,
        now_s=10.5,
        phase=RuntimePhase.LIFT,
        measured_joints=path[2],
        joint_seen_at_s=10.5,
        execution_status_seen_at_s=10.5,
        lift_path=path,
        lift_execution_active=True,
    )

    assert forward.allowed
    assert not backward.allowed
    assert 'moved backwards' in backward.reason


def test_live_tracking_flicker_cannot_erase_lift_progress() -> None:
    gate = ExecutionOcclusionGate(ExecutionOcclusionConfig(
        max_path_regression_samples=1,
    ))
    _arm(gate)
    _mark_loss(gate, 10.1)
    gate.confirm_contact(10.2)
    gate.note_lift_sent(10.3)
    path = tuple(
        (ENDPOINT[0] + index * 0.02, ENDPOINT[1], ENDPOINT[2])
        for index in range(6)
    )
    forward = _evaluate(
        gate,
        now_s=10.4,
        phase=RuntimePhase.LIFT,
        measured_joints=path[4],
        joint_seen_at_s=10.4,
        execution_status_seen_at_s=10.4,
        lift_path=path,
        lift_execution_active=True,
    )
    assert _restore_tracking(gate, 10.45)
    _mark_loss(gate, 10.46)
    backward = _evaluate(
        gate,
        now_s=10.5,
        phase=RuntimePhase.LIFT,
        measured_joints=path[2],
        joint_seen_at_s=10.5,
        execution_status_seen_at_s=10.5,
        lift_path=path,
        lift_execution_active=True,
    )

    assert forward.allowed
    assert gate.maximum_lift_path_index == 4
    assert not backward.allowed
    assert 'moved backwards' in backward.reason


def test_verification_requires_completed_lift_and_total_window_remains_bounded() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    _mark_loss(gate, 10.1)
    gate.confirm_contact(10.2)
    gate.note_lift_sent(10.3)
    path = (ENDPOINT, (0.15, -0.15, 0.35))

    incomplete = _evaluate(
        gate,
        now_s=10.4,
        phase=RuntimePhase.VERIFY,
        measured_joints=path[-1],
        joint_seen_at_s=10.4,
        lift_path=path,
    )
    assert not incomplete.allowed
    assert 'completed lift evidence' in incomplete.reason

    gate.note_lift_completed(10.5)
    accepted = _evaluate(
        gate,
        now_s=10.6,
        phase=RuntimePhase.VERIFY,
        measured_joints=path[-1],
        joint_seen_at_s=10.6,
        lift_path=path,
    )
    assert accepted.allowed
    assert accepted.mode == 'predicted_verify'

    expired = _evaluate(
        gate,
        now_s=13.01,
        phase=RuntimePhase.VERIFY,
        measured_joints=path[-1],
        joint_seen_at_s=13.01,
        lift_path=path,
    )
    assert not expired.allowed
    assert 'window expired' in expired.reason


def test_tracking_restoration_cannot_hide_an_overlong_blackout() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    _mark_loss(gate, 10.1)

    with pytest.raises(ValueError, match='returned after'):
        _restore_tracking(gate, 13.01)


def test_tracking_restoration_never_extends_the_aggregate_execution_window() -> None:
    gate = ExecutionOcclusionGate()
    _arm(gate)
    path = (ENDPOINT, (0.15, -0.15, 0.35))

    _mark_loss(gate, 10.1)
    assert _evaluate(
        gate,
        now_s=10.15,
        phase=RuntimePhase.CLOSING,
        measured_joints=ENDPOINT,
        joint_seen_at_s=10.15,
        close_command_sent_at_s=10.1,
    ).allowed
    assert _restore_tracking(gate, 10.2)
    gate.confirm_contact(10.3)
    gate.note_lift_sent(10.4)
    _mark_loss(gate, 11.0)
    assert _evaluate(
        gate,
        now_s=11.1,
        phase=RuntimePhase.LIFT,
        measured_joints=path[-1],
        joint_seen_at_s=11.1,
        execution_status_seen_at_s=11.1,
        lift_path=path,
        lift_execution_active=True,
    ).allowed
    assert _restore_tracking(gate, 11.2)
    gate.note_lift_completed(12.0)
    _mark_loss(gate, 12.5)
    assert _evaluate(
        gate,
        now_s=12.6,
        phase=RuntimePhase.VERIFY,
        measured_joints=path[-1],
        joint_seen_at_s=12.6,
        lift_path=path,
    ).allowed
    assert _restore_tracking(gate, 12.7)

    with pytest.raises(ValueError, match='already expired'):
        _mark_loss(gate, 13.01)


def test_near_contact_runtime_arming_copies_the_exact_snapshot() -> None:
    pytest.importorskip('rclpy')
    import numpy as np

    from z_manip_task.node import MobileManipulationRuntime

    target = np.asarray((0.4, 0.1, 0.6))
    target_cloud = np.asarray(((0.4, 0.1, 0.6), (0.41, 0.1, 0.6)))
    scene_cloud = np.asarray(((1.0, 1.0, 1.0),))
    harness = SimpleNamespace(
        _serial_gate=SimpleNamespace(
            snapshot=lambda _now: SimpleNamespace(serial=8, stamp_s=9.95),
        ),
        _grounding_observation_authorized=lambda _snapshot: True,
        _program=SimpleNamespace(
            approach=SimpleNamespace(positions=(ENDPOINT, ENDPOINT)),
        ),
        _joint_history=[SimpleNamespace(
            received_at_s=9.9,
            source_stamp_ns=9_990_000_000,
            sequence=10_000,
            positions=ENDPOINT,
        )],
        _bound_perception_request_id='task-request',
        _bound_perception_producer_epoch='bridge-epoch',
        _bound_perception_generation=4,
        _valid_observation_stamp_ns=OBSERVATION_STAMP_NS,
        _valid_observation_frame_id=OBSERVATION_FRAME,
        _execution_occlusion=ExecutionOcclusionGate(),
        _target_piper=target,
        _target_cloud=target_cloud,
        _scene_cloud=scene_cloud,
        _execution_occlusion_target_piper=None,
        _execution_occlusion_target_cloud=None,
        _execution_occlusion_scene_cloud=None,
    )
    harness._cache_execution_occlusion_geometry = (
        lambda: MobileManipulationRuntime._cache_execution_occlusion_geometry(
            harness,
        )
    )

    MobileManipulationRuntime._arm_execution_occlusion(harness, 10.0)

    assert harness._execution_occlusion.armed
    assert harness._execution_occlusion.observation_serial == 8
    assert harness._execution_occlusion_target_piper is not target
    assert harness._execution_occlusion_target_cloud is not target_cloud
    assert harness._execution_occlusion_scene_cloud is not scene_cloud
    target[:] = 99.0
    target_cloud[:] = 99.0
    scene_cloud[:] = 99.0
    assert tuple(harness._execution_occlusion_target_piper) == (0.4, 0.1, 0.6)
    assert harness._execution_occlusion_target_cloud[0, 0] == pytest.approx(0.4)
    assert harness._execution_occlusion_scene_cloud[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    'phase',
    (RuntimePhase.CLOSING, RuntimePhase.LIFT, RuntimePhase.VERIFY),
)
def test_execution_tick_never_uses_generic_validity_as_exact_authorization(
    phase: RuntimePhase,
) -> None:
    pytest.importorskip('rclpy')
    import threading

    from z_manip_task.core import RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick')
            self._core.phase = phase
            self._lookout_pending = False
            self._trajectory_deadline_s = None
            self._serial_gate = SimpleNamespace(
                snapshot=lambda _now: SimpleNamespace(serial=12),
            )
            self._execution_occlusion = SimpleNamespace(loss_active=False)
            self._execution_occlusion_last_decision = SimpleNamespace(
                reason='exact observation watermark mismatch',
            )
            self._perception_valid = True
            self._valid_seen_at = 10.0
            self.loss_checks = []
            self.actions = []
            self.status_publishes = 0

        @staticmethod
        def _now_s() -> float:
            return 10.1

        @staticmethod
        def _guard_active_posture(_now: float) -> bool:
            return True

        @staticmethod
        def _poll_planning() -> None:
            return

        @staticmethod
        def _grounding_observation_authorized(_snapshot) -> bool:
            return False

        def _execution_occlusion_allows_loss(
            self,
            now: float,
            detail: str,
        ) -> bool:
            self.loss_checks.append((now, detail))
            return False

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

        def _publish_status(self) -> None:
            self.status_publishes += 1

    harness = Harness()

    MobileManipulationRuntime._tick(harness)

    assert harness._core.phase is RuntimePhase.FAILED
    assert len(harness.loss_checks) == 1
    assert 'exact-authorized' in harness.loss_checks[0][1]
    assert 'watermark mismatch' in harness._core.failure_reason
    assert harness.status_publishes == 1


@pytest.mark.parametrize(
    ('status_text', 'seen_active', 'expected'),
    (
        (
            'succeeded;owner=trajectory;segment=lift;command_id=7',
            True,
            True,
        ),
        (
            'succeeded;owner=other;segment=lift;command_id=7',
            True,
            False,
        ),
        (
            'succeeded;owner=trajectory;segment=carry;command_id=7',
            True,
            False,
        ),
        (
            'succeeded;owner=trajectory;segment=lift;command_id=8',
            True,
            False,
        ),
        (
            'succeeded;owner=trajectory;segment=lift;command_id=7',
            False,
            False,
        ),
    ),
)
def test_lift_completion_identity_is_captured_before_core_transition(
    status_text: str,
    seen_active: bool,
    expected: bool,
) -> None:
    pytest.importorskip('rclpy')

    from z_manip_task.core import parse_execution_status, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    core = RuntimeSafetyCore()
    core.begin('pick')
    core.phase = RuntimePhase.LIFT
    core.execution_segment = 'lift'
    core.expected_command_id = 7
    core.execution_seen_active = seen_active
    harness = SimpleNamespace(_core=core)

    matches = MobileManipulationRuntime._lift_completion_identity_matches(
        harness,
        parse_execution_status(status_text),
    )

    assert matches is expected


def test_loss_active_lift_accepts_matching_active_to_succeeded_transition() -> None:
    pytest.importorskip('rclpy')
    import threading

    from std_msgs.msg import String

    from z_manip_task.core import (
        ExecutionOcclusionDecision,
        parse_execution_status,
        RuntimeSafetyCore,
    )
    from z_manip_task.node import MobileManipulationRuntime

    class Task:
        stage = SimpleNamespace(value='execute_grasp')

        def __init__(self) -> None:
            self.results = []

        def apply(self, result) -> None:
            self.results.append(result)

    class Harness:
        _execution_occlusion_decision = (
            MobileManipulationRuntime._execution_occlusion_decision
        )
        _execution_perception_admitted = (
            MobileManipulationRuntime._execution_perception_admitted
        )
        _lift_completion_identity_matches = (
            MobileManipulationRuntime._lift_completion_identity_matches
        )

        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick')
            self._core.phase = RuntimePhase.LIFT
            self._core.trajectory_sent(
                'lift', executor_epoch='executor-a', published_at_s=9.9,
                trajectory_token='trajectory-lift',
            )
            active = parse_execution_status(
                'active;owner=trajectory;segment=lift;command_id=7;'
                'executor_epoch=executor-a;'
                'trajectory_token=trajectory-lift;'
                'trajectory_received_at=10.0',
            )
            self._core.execution_update(active)
            self._execution_occlusion = ExecutionOcclusionGate()
            _arm(self._execution_occlusion)
            self._execution_occlusion.confirm_contact(10.01)
            self._execution_occlusion.note_lift_sent(10.02)
            _mark_loss(self._execution_occlusion, 10.2)
            self._joint_history = [SimpleNamespace(
                received_at_s=10.6,
                source_stamp_ns=10_590_000_000,
                sequence=10_600,
                positions=ENDPOINT,
            )]
            self._program = SimpleNamespace(
                lift=SimpleNamespace(positions=(ENDPOINT, ENDPOINT)),
            )
            self._execution_status = active
            self._execution_status_seen_s = 10.2
            self._expected_gripper_command_id = None
            self._commanded_close_aperture = None
            self._gripper_command_sent_s = None
            self._serial_gate = SimpleNamespace(snapshot=lambda _now: None)
            self._execution_occlusion_last_decision = (
                ExecutionOcclusionDecision(False, 'not evaluated')
            )
            self._execution_occlusion_loss_detail = 'tracker lost'
            self._latest_gripper_command_id = 0
            self._gripper_feedback = []
            self._trajectory_deadline_s = 20.0
            self._verification_started_at = None
            self._task = Task()
            self.safety_reasons = []

        @staticmethod
        def _now_s() -> float:
            return 10.6

        @staticmethod
        def _guard_active_posture(_now: float) -> bool:
            return True

        @staticmethod
        def _grounding_observation_authorized(_snapshot) -> bool:
            return False

        def _apply_safety(self, action) -> None:
            if action.reason:
                self.safety_reasons.append(action.reason)

    harness = Harness()
    succeeded = String(
        data=(
            'succeeded;owner=trajectory;segment=lift;command_id=7;'
            'executor_epoch=executor-a;trajectory_token=trajectory-lift;'
            'trajectory_received_at=10.0'
        ),
    )

    MobileManipulationRuntime._execution_cb(harness, succeeded)

    assert harness._core.phase is RuntimePhase.VERIFY
    assert harness._execution_occlusion.lift_completed_at_s == pytest.approx(10.6)
    assert harness._execution_occlusion_last_decision.allowed
    assert harness.safety_reasons == []
    assert len(harness._task.results) == 1


def test_perception_failure_is_predicted_only_for_armed_closing() -> None:
    pytest.importorskip('rclpy')
    import threading

    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    from z_manip_task.core import parse_execution_status, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        _revoke_perception_success = (
            MobileManipulationRuntime._revoke_perception_success
        )
        _begin_execution_occlusion_loss = (
            MobileManipulationRuntime._begin_execution_occlusion_loss
        )
        _execution_occlusion_decision = (
            MobileManipulationRuntime._execution_occlusion_decision
        )

        def __init__(self, phase: RuntimePhase, *, armed: bool) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.begin('pick')
            self._core.phase = phase
            self._required_perception_request_id = 'task-request'
            self._required_grounding_scope = 'grasp_only'
            self._bound_perception_request_id = 'task-request'
            self._bound_perception_producer_epoch = 'bridge-epoch'
            self._bound_perception_generation = 4
            self._perception_generation = 4
            self._perception_valid = True
            self._valid_seen_at = 10.0
            self._valid_perception_request_id = 'task-request'
            self._valid_perception_producer_epoch = 'bridge-epoch'
            self._valid_perception_generation = 4
            self._valid_observation_stamp_ns = 9_950_000_000
            self._valid_observation_frame_id = 'wrist_depth_optical_frame'
            self._handled_perception_failure = None
            self._execution_occlusion = ExecutionOcclusionGate()
            if armed:
                _arm(self._execution_occlusion)
            self._joint_history = [SimpleNamespace(
                received_at_s=10.19,
                source_stamp_ns=10_190_000_000,
                sequence=10_200,
                positions=ENDPOINT,
            )]
            self._serial_gate = SimpleNamespace(snapshot=lambda _now: None)
            self._execution_status = parse_execution_status(
                'idle;gripper=accepted:0.020;gripper_command_id=5;'
                'gripper_received_at=8.000000;aperture=0.03',
            )
            self._expected_gripper_command_id = 5
            self._commanded_close_aperture = 0.02
            self._gripper_command_sent_s = 10.0
            self._execution_status_seen_s = 10.19
            self._program = SimpleNamespace(
                lift=SimpleNamespace(positions=(ENDPOINT, ENDPOINT)),
            )
            self._execution_occlusion_last_decision = None
            self._execution_occlusion_loss_detail = ''
            self.actions = []
            self.recoveries = []

        @staticmethod
        def _now_s() -> float:
            return 10.2

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

        def _recover_precontact(self, kind, detail: str) -> bool:
            self.recoveries.append((kind, detail))
            return False

        @staticmethod
        def _grounding_observation_authorized(_snapshot) -> bool:
            return False

    failure = DiagnosticArray(status=[DiagnosticStatus(
        level=DiagnosticStatus.ERROR,
        message='failed',
        values=[
            KeyValue(key='schema', value='z_manip.perception_status.v1'),
            KeyValue(key='request_id', value='task-request'),
            KeyValue(key='grounding_scope', value='grasp_only'),
            KeyValue(key='producer_epoch', value='bridge-epoch'),
            KeyValue(key='generation', value='4'),
            KeyValue(key='valid', value='false'),
            KeyValue(key='failure', value='tracker_lost'),
        ],
    )])
    closing = Harness(RuntimePhase.CLOSING, armed=True)
    transit = Harness(RuntimePhase.TRANSIT, armed=False)

    MobileManipulationRuntime._perception_status_cb(closing, failure)
    MobileManipulationRuntime._perception_status_cb(transit, failure)

    assert closing._core.phase is RuntimePhase.CLOSING
    assert closing._execution_occlusion.loss_active
    assert closing._execution_occlusion_last_decision.allowed
    assert closing.actions == []
    awaiting = closing._execution_occlusion_decision(10.21)
    assert awaiting.allowed
    assert awaiting.mode == 'awaiting_joint_sample'
    closing._joint_history.append(SimpleNamespace(
        received_at_s=10.33,
        source_stamp_ns=10_330_000_000,
        sequence=10_201,
        positions=ENDPOINT,
    ))
    next_joint_tick = closing._execution_occlusion_decision(10.34)
    assert next_joint_tick.allowed
    assert next_joint_tick.mode == 'predicted_closing'
    assert transit._core.phase is RuntimePhase.FAILED
    assert not transit._execution_occlusion.loss_active
    assert len(transit.recoveries) == 1


def test_cached_lift_scene_invalidation_prevents_publication() -> None:
    pytest.importorskip('rclpy')
    import numpy as np

    from z_manip_task.core import parse_execution_status, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Planner:
        chain = SimpleNamespace(joint_names=('j1', 'j2', 'j3'))

        def __init__(self) -> None:
            self.calls = []

        def validate_path(self, path, **kwargs) -> bool:
            self.calls.append((path, kwargs))
            return False

    class Harness:
        _execution_occlusion_decision = (
            MobileManipulationRuntime._execution_occlusion_decision
        )

        @staticmethod
        def _execution_perception_admitted(_now, _phase) -> bool:
            return True

        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.begin('pick')
            self._core.phase = RuntimePhase.LIFT
            self._execution_occlusion = ExecutionOcclusionGate()
            _arm(self._execution_occlusion)
            _mark_loss(self._execution_occlusion, 10.1)
            self._execution_occlusion.confirm_contact(10.2)
            path = np.asarray((ENDPOINT, (0.15, -0.15, 0.35)))
            self._program = SimpleNamespace(
                lift=SimpleNamespace(positions=path, times_s=(0.0, 1.0)),
            )
            self._carry_program = None
            self._place_programs = {}
            self._joint_state = np.asarray(ENDPOINT)
            self._joint_history = [SimpleNamespace(
                received_at_s=10.2,
                source_stamp_ns=10_240_000_000,
                sequence=10_250,
                positions=self._joint_state.copy(),
            )]
            self._execution_status = parse_execution_status(
                'idle;gripper=accepted:0.020;gripper_command_id=5;'
                'gripper_received_at=8.000000;aperture=0.03',
            )
            self._expected_gripper_command_id = 5
            self._commanded_close_aperture = 0.02
            self._gripper_command_sent_s = 10.0
            self._execution_status_seen_s = 10.2
            self._execution_occlusion_scene_cloud = np.asarray(((1.0, 2.0, 3.0),))
            self._execution_occlusion_target_cloud = np.asarray(((0.1, 0.2, 0.3),))
            self._planner = Planner()
            self._serial_gate = SimpleNamespace(snapshot=lambda _now: None)
            self._trajectory_pub = SimpleNamespace(publish=lambda _msg: pytest.fail(
                'invalidated lift was published',
            ))
            self.actions = []

        def _now_s(self) -> float:
            return 10.25

        @staticmethod
        def _guard_active_posture(_now: float) -> bool:
            return True

        def get_parameter(self, name: str):
            assert name == 'max_trajectory_start_error_rad'
            return SimpleNamespace(value=0.1)

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

    harness = Harness()

    MobileManipulationRuntime._publish_program_segment(harness, 'lift')

    assert harness._core.phase is RuntimePhase.FAILED
    assert len(harness._planner.calls) == 1
    _, kwargs = harness._planner.calls[0]
    assert kwargs['scene_points'] is harness._execution_occlusion_scene_cloud
    assert kwargs['target_points'] is harness._execution_occlusion_target_cloud
    assert 'invalidates the path' in harness._core.failure_reason
