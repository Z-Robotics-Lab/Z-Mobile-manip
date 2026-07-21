"""Focused tests for continuous task posture safety."""

from types import SimpleNamespace

import pytest

from z_manip_task.core import (
    PostureSafetyGate,
    PostureState,
    RuntimePhase,
    RuntimeSafetyCore,
)


def _gate() -> PostureSafetyGate:
    return PostureSafetyGate(
        max_roll_rad=0.17,
        max_pitch_rad=0.21,
        max_age_s=0.50,
        acquisition_timeout_s=1.0,
    )


def test_first_posture_sample_waits_without_authorizing_motion() -> None:
    gate = _gate()
    gate.begin(10.0)

    waiting = gate.assess(10.8)

    assert waiting.state is PostureState.WAITING
    assert not waiting.safe
    assert 'first state-estimation' in waiting.reason

    unavailable = gate.assess(11.01)
    assert unavailable.state is PostureState.UNSAFE
    assert 'acquisition timeout' in unavailable.reason


def test_preexisting_fresh_posture_sample_authorizes_a_new_task() -> None:
    gate = _gate()
    gate.update(0.04, -0.05, seen_at_s=4.9)
    gate.begin(5.0)

    assert gate.assess(5.05).safe


def test_new_task_discards_pre_clock_posture_and_waits_for_a_fresh_sample() -> None:
    gate = _gate()
    gate.update(0.01, -0.02, seen_at_s=0.0)
    gate.begin(54.54)

    waiting = gate.assess(54.60)
    assert waiting.state is PostureState.WAITING

    gate.update(0.02, -0.03, seen_at_s=54.61)
    assert gate.assess(54.62).safe


def test_sim_clock_activation_restarts_posture_acquisition_from_zero() -> None:
    gate = _gate()
    gate.update(0.01, -0.02, seen_at_s=0.0)
    gate.begin(0.0)

    waiting = gate.assess(61.09)
    assert waiting.state is PostureState.WAITING

    gate.update(0.02, -0.03, seen_at_s=61.10)
    assert gate.assess(61.11).safe


def test_fresh_post_clock_sample_survives_first_nonzero_assessment() -> None:
    gate = _gate()
    gate.begin(0.0)
    gate.update(0.02, -0.03, seen_at_s=61.10)

    assert gate.assess(61.11).safe


@pytest.mark.parametrize(
    ('roll', 'pitch', 'seen_at', 'now', 'reason'),
    (
        (0.18, 0.0, 2.0, 2.0, 'base posture limit exceeded'),
        (0.0, -0.22, 2.0, 2.0, 'base posture limit exceeded'),
        (0.0, 0.0, 2.0, 2.51, 'is stale'),
        (0.0, 0.0, 2.1, 2.0, 'in the future'),
        (float('nan'), 0.0, 2.0, 2.0, 'non-finite attitude'),
    ),
)
def test_invalid_or_stale_posture_is_unsafe(
    roll: float,
    pitch: float,
    seen_at: float,
    now: float,
    reason: str,
) -> None:
    gate = _gate()
    gate.begin(2.0)
    gate.update(roll, pitch, seen_at_s=seen_at)

    assessment = gate.assess(now)

    assert assessment.state is PostureState.UNSAFE
    assert reason in assessment.reason


@pytest.mark.parametrize(
    'phase',
    tuple(sorted(RuntimeSafetyCore._ACTIVE_PHASES, key=lambda phase: phase.value)),
)
def test_posture_loss_cancels_every_motion_owner_in_every_active_phase(
    phase: RuntimePhase,
) -> None:
    core = RuntimeSafetyCore()
    core.phase = phase

    action = core.posture_invalid('measured platform attitude exceeded limits')

    assert core.phase is RuntimePhase.FAILED
    assert core.failure_reason == 'measured platform attitude exceeded limits'
    assert action.stop_base and action.cancel_navigation and action.cancel_arm
    assert action.reason == core.failure_reason


@pytest.mark.parametrize(
    'phase',
    (
        RuntimePhase.IDLE,
        RuntimePhase.PICK_COMPLETE,
        RuntimePhase.COMPLETE,
        RuntimePhase.CANCELED,
        RuntimePhase.FAILED,
    ),
)
def test_posture_updates_do_not_change_idle_or_terminal_tasks(
    phase: RuntimePhase,
) -> None:
    core = RuntimeSafetyCore()
    core.phase = phase

    action = core.posture_invalid('irrelevant while task does not own motion')

    assert core.phase is phase
    assert not action.stop_base
    assert not action.cancel_navigation
    assert not action.cancel_arm


def test_node_gate_holds_missing_state_then_fails_after_timeout() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.begin('pick the bottle')
            self._posture_guard = _gate()
            self._posture_guard.begin(3.0)
            self.actions = []

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

        def _fail_posture(self, reason: str) -> None:
            self.actions.append(self._core.posture_invalid(reason))

    harness = Harness()

    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.5)
    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness.actions[-1].stop_base
    assert not harness.actions[-1].cancel_arm

    assert not MobileManipulationRuntime._guard_active_posture(harness, 4.01)
    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.actions[-1].stop_base
    assert harness.actions[-1].cancel_navigation
    assert harness.actions[-1].cancel_arm


class _NodePostureHarness:
    def __init__(
        self,
        *,
        phase: RuntimePhase,
        dwell_s: float = 0.15,
    ) -> None:
        self._core = RuntimeSafetyCore()
        self._core.phase = phase
        self._posture_guard = _gate()
        self._posture_guard.begin(3.0)
        self._coarse_nav_posture_violation_started_at_s = None
        self._dwell_s = dwell_s
        self.actions = []
        self.fail_reasons = []

    def get_parameter(self, name: str):
        assert name == 'coarse_nav_posture_violation_dwell_s'
        return SimpleNamespace(value=self._dwell_s)

    def _apply_safety(self, action) -> None:
        self.actions.append(action)

    def _fail_posture(self, reason: str) -> None:
        self._coarse_nav_posture_violation_started_at_s = None
        self.fail_reasons.append(reason)
        self.actions.append(self._core.posture_invalid(reason))


def test_coarse_navigation_requires_a_continuous_posture_violation() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _NodePostureHarness(phase=RuntimePhase.COARSE_NAV)
    harness._posture_guard.update(0.19, 0.0, seen_at_s=3.0)

    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.0)
    assert harness._core.phase is RuntimePhase.COARSE_NAV
    assert harness.actions == []

    harness._posture_guard.update(0.19, 0.0, seen_at_s=3.10)
    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.10)
    assert harness._core.phase is RuntimePhase.COARSE_NAV
    assert harness.actions == []

    harness._posture_guard.update(0.01, 0.0, seen_at_s=3.11)
    assert MobileManipulationRuntime._guard_active_posture(harness, 3.11)
    assert harness._coarse_nav_posture_violation_started_at_s is None

    harness._posture_guard.update(0.19, 0.0, seen_at_s=3.12)
    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.12)
    harness._posture_guard.update(0.19, 0.0, seen_at_s=3.271)
    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.271)
    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.actions[-1].cancel_navigation
    assert harness.actions[-1].cancel_arm


def test_posture_limit_outside_coarse_navigation_fails_immediately() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _NodePostureHarness(phase=RuntimePhase.VISUAL_SERVO)
    harness._posture_guard.update(0.19, 0.0, seen_at_s=3.0)

    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.0)
    assert harness._core.phase is RuntimePhase.FAILED
    assert harness.actions[-1].cancel_navigation
    assert harness.actions[-1].cancel_arm


def test_stale_posture_during_coarse_navigation_fails_immediately() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _NodePostureHarness(phase=RuntimePhase.COARSE_NAV)
    harness._posture_guard.update(0.0, 0.0, seen_at_s=3.0)

    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.51)
    assert harness._core.phase is RuntimePhase.FAILED
    assert 'stale' in harness.fail_reasons[-1]


@pytest.mark.parametrize('dwell_s', (0.0, -0.1, float('nan')))
def test_invalid_coarse_navigation_posture_dwell_fails_closed(
    dwell_s: float,
) -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _NodePostureHarness(
        phase=RuntimePhase.COARSE_NAV,
        dwell_s=dwell_s,
    )
    harness._posture_guard.update(0.19, 0.0, seen_at_s=3.0)

    assert not MobileManipulationRuntime._guard_active_posture(harness, 3.0)
    assert harness._core.phase is RuntimePhase.FAILED
    assert 'finite and positive' in harness.fail_reasons[-1]


def test_state_estimation_quaternion_must_be_finite_and_non_degenerate() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import _quaternion_roll_pitch

    normalized = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=2.0)
    assert _quaternion_roll_pitch(normalized) == pytest.approx((0.0, 0.0))

    zero = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)
    with pytest.raises(ValueError, match='degenerate'):
        _quaternion_roll_pitch(zero)


def test_posture_boundary_clears_visual_search_settle_reference_when_idle() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._visual_search_settle_reference = object()

    harness = Harness()

    MobileManipulationRuntime._fail_posture(harness, 'late posture callback')

    assert harness._visual_search_settle_reference is None
