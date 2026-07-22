"""Bounded tracker-loss recovery tests with no ROS graph or model calls."""

from types import SimpleNamespace

from z_manip_ros.contract import ContractPhase, FailureCode, TrackingContract
from z_manip_ros.vlm_edgetam_bridge import VlmEdgeTamBridge


class _ParameterSource:
    values = {
        'tracker_auto_reacquire_enabled': True,
        'tracker_auto_reacquire_max_attempts': 2,
        'tracker_auto_reacquire_backoff_s': 0.25,
        'tracker_auto_reacquire_window_s': 8.0,
    }

    def get_parameter(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(value=self.values[name])


class _Harness(_ParameterSource):
    def __init__(self) -> None:
        self.now = 10.0
        self._contract = TrackingContract(
            data_timeout_s=0.5,
            min_cloud_points=2,
        )
        generation = self._contract.request(
            'pick the bottle',
            now_s=1.0,
            request_id='task-a',
        )
        self._contract.grounding_started(generation, now_s=1.1)
        self._contract.grounding_succeeded(
            generation,
            target_label='bottle',
            confidence=0.9,
            now_s=1.2,
        )
        self._contract.tracker_failed(now_s=1.3)
        self._tracker_failure_detail = 'mask_continuity'
        self._tracker_reacquire_attempts = 0
        self._tracker_reacquire_due_monotonic_s = None
        self._tracker_reacquire_deadline_monotonic_s = None
        self._tracker_reacquire_instruction = ''
        self._tracker_reacquire_request_id = ''
        self._tracker_reacquire_state = 'idle'
        self._expected_edge_seed_id = 'old-seed'
        self._expected_edge_seed_stamp_ns = 123
        self.events: list[str] = []
        self._coarse_nav_authorization = SimpleNamespace(
            reset=lambda: self.events.append('auth_reset'),
        )

    def _monotonic_s(self) -> float:
        return self.now

    def _clear_tracker_messages(self) -> None:
        self.events.append('clear')

    def _cancel_pending_grounding(self) -> None:
        self.events.append('cancel_grounding')

    def _publish_seed_command(self, action: str) -> None:
        self.events.append(f'seed:{action}')

    def _publish_zero_velocity(self) -> None:
        self.events.append('zero')

    def _publish_contract(self) -> None:
        self.events.append('status')

    def get_logger(self) -> SimpleNamespace:
        return SimpleNamespace(
            warn=lambda detail: self.events.append(f'warn:{detail}'),
        )


def test_tracker_loss_revokes_old_geometry_then_starts_fresh_generation(
) -> None:
    harness = _Harness()
    snapshot = harness._contract.snapshot

    assert VlmEdgeTamBridge._schedule_tracker_reacquire(harness, snapshot)
    assert harness._tracker_reacquire_state == 'scheduled'
    assert not VlmEdgeTamBridge._maybe_start_tracker_reacquire(harness, 2.0)
    assert harness._contract.phase is ContractPhase.FAILED

    harness.now = 10.25
    assert VlmEdgeTamBridge._maybe_start_tracker_reacquire(harness, 2.0)
    assert harness._contract.phase is ContractPhase.WAITING_FRAME
    assert harness._contract.snapshot.request_id == 'task-a'
    assert harness._contract.snapshot.instruction == 'pick the bottle'
    assert harness._tracker_reacquire_attempts == 1
    assert harness._expected_edge_seed_id == ''
    assert harness._expected_edge_seed_stamp_ns is None
    assert harness.events == [
        'auth_reset',
        'clear',
        'cancel_grounding',
        'seed:arm',
        'zero',
        (
            'warn:tracker lost; started bounded fresh-frame '
            'perception reacquire 1/2'
        ),
        'status',
    ]


def test_reacquire_is_bounded_by_attempt_count_and_window() -> None:
    harness = _Harness()
    snapshot = harness._contract.snapshot
    harness._tracker_reacquire_attempts = 2

    assert not VlmEdgeTamBridge._schedule_tracker_reacquire(harness, snapshot)
    assert harness._tracker_reacquire_state == 'exhausted'
    assert harness._tracker_reacquire_due_monotonic_s is None

    harness._tracker_reacquire_attempts = 0
    assert VlmEdgeTamBridge._schedule_tracker_reacquire(harness, snapshot)
    harness.now = 18.01
    assert not VlmEdgeTamBridge._maybe_start_tracker_reacquire(harness, 2.0)
    assert harness._tracker_reacquire_state == 'expired'
    assert harness._contract.phase is ContractPhase.FAILED
    assert harness.events == []


def test_non_tracker_failure_never_retries() -> None:
    harness = _Harness()
    snapshot = SimpleNamespace(
        failure=FailureCode.GROUNDING_FAILED,
        instruction='pick the bottle',
        request_id='task-a',
    )

    assert not VlmEdgeTamBridge._schedule_tracker_reacquire(harness, snapshot)
    assert harness._tracker_reacquire_state == 'not_tracker_loss'
    assert harness._tracker_reacquire_due_monotonic_s is None
