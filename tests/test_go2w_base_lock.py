"""Unit tests for the transport-free Go2W base body-posture lock.

Covers the owner state machine (:class:`BaseLockController`) and the NUC latch
(:class:`BaseLockLatch`): idempotency, unlock-on-abort, the never-lock-during-
servo guard, watchdog expiry, acknowledgement transitions, and fail-open
transport behavior.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_base_lock.py"
SPEC = importlib.util.spec_from_file_location("go2w_base_lock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LOCK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LOCK
SPEC.loader.exec_module(LOCK)


class _RecordingEmit:
    """Injectable transport that records commands and returns a scripted ack."""

    def __init__(self, *, lock_state: str | None = "locked",
                 unlock_state: str | None = "unlocked", delivered: bool = True,
                 raises: bool = False) -> None:
        self.commands: list[dict] = []
        self._lock_state = lock_state
        self._unlock_state = unlock_state
        self._delivered = delivered
        self._raises = raises

    def __call__(self, command: dict) -> "LOCK.BaseLockAck":
        self.commands.append(dict(command))
        if self._raises:
            raise RuntimeError("simulated transport failure")
        nuc_state = self._lock_state if command["lock"] else self._unlock_state
        return LOCK.BaseLockAck(delivered=self._delivered, nuc_state=nuc_state)

    @property
    def lock_commands(self) -> list[dict]:
        return [c for c in self.commands if c["lock"]]

    @property
    def unlock_commands(self) -> list[dict]:
        return [c for c in self.commands if not c["lock"]]


def _controller(emit, **kwargs):
    kwargs.setdefault("heartbeat_period_s", None)  # no background thread in tests
    return LOCK.BaseLockController(emit=emit, **kwargs)


# --------------------------------------------------------------------------
# Owner state machine
# --------------------------------------------------------------------------

def test_lock_then_unlock_full_cycle_with_acknowledgement():
    emit = _RecordingEmit()
    controller = _controller(emit)

    assert controller.state == LOCK.BaseLockState.UNLOCKED
    assert controller.request_lock("grasp", base_stopped=True) == LOCK.BaseLockState.LOCKED
    assert controller.request_unlock("grasp") == LOCK.BaseLockState.UNLOCKED
    assert controller.state == LOCK.BaseLockState.UNLOCKED
    assert len(emit.lock_commands) == 1
    assert len(emit.unlock_commands) == 1
    # Sequence numbers strictly increase across the lifecycle.
    assert emit.lock_commands[0]["seq"] == 1
    assert emit.unlock_commands[0]["seq"] == 2


def test_lock_request_is_idempotent():
    emit = _RecordingEmit()
    controller = _controller(emit)

    controller.request_lock("grasp", base_stopped=True)
    controller.request_lock("grasp", base_stopped=True)
    controller.request_lock("someone-else", base_stopped=True)

    # Only one lock command was emitted; repeats are no-ops that keep the
    # original owner.
    assert len(emit.lock_commands) == 1
    assert controller.status_field()["source"] == "grasp"


def test_unlock_request_is_idempotent():
    emit = _RecordingEmit()
    controller = _controller(emit)

    controller.request_lock("grasp", base_stopped=True)
    controller.request_unlock("grasp")
    controller.request_unlock("grasp")
    controller.request_unlock("grasp")

    assert len(emit.unlock_commands) == 1


def test_unlock_on_abort_from_locked_always_emits_release():
    # An operator abort / exception mid-grasp lands here: unlock must fire even
    # though the lock is fully engaged.
    emit = _RecordingEmit()
    controller = _controller(emit)
    controller.request_lock("grasp", base_stopped=True)
    assert controller.state == LOCK.BaseLockState.LOCKED

    state = controller.request_unlock("abort")
    assert state == LOCK.BaseLockState.UNLOCKED
    assert emit.unlock_commands[-1]["lock"] is False


def test_unlock_before_any_lock_is_a_noop():
    emit = _RecordingEmit()
    controller = _controller(emit)
    assert controller.request_unlock("grasp") == LOCK.BaseLockState.UNLOCKED
    assert emit.commands == []


def test_never_lock_while_base_is_moving_during_servo():
    emit = _RecordingEmit()
    controller = _controller(emit)

    state = controller.request_lock("servo", base_stopped=False)

    assert state == LOCK.BaseLockState.UNLOCKED
    assert emit.commands == []  # nothing was transmitted to the NUC
    assert "base is not stopped" in controller.status_field()["last_refusal"]

    # And once the base has stopped, the same request now locks.
    assert controller.request_lock("grasp", base_stopped=True) == LOCK.BaseLockState.LOCKED
    assert controller.status_field()["last_refusal"] is None


def test_lock_requested_stays_pending_without_acknowledgement():
    # Transport delivered the command but could not read the NUC state back:
    # the owner stays LOCK_REQUESTED (not falsely LOCKED) until a heartbeat or
    # observed status confirms it.
    emit = _RecordingEmit(lock_state=None)
    controller = _controller(emit)
    assert controller.request_lock("grasp", base_stopped=True) == LOCK.BaseLockState.LOCK_REQUESTED
    assert controller.is_engaged() is True


def test_heartbeat_renews_and_confirms_pending_lock():
    emit = _RecordingEmit(lock_state=None)
    controller = _controller(emit)
    controller.request_lock("grasp", base_stopped=True)
    assert controller.state == LOCK.BaseLockState.LOCK_REQUESTED

    # A later heartbeat whose transport now reads the NUC back confirms LOCKED
    # and renews the lease (re-emits lock=True) without changing seq.
    emit._lock_state = "locked"
    controller.heartbeat()
    assert controller.state == LOCK.BaseLockState.LOCKED
    assert len(emit.lock_commands) == 2
    assert emit.lock_commands[0]["seq"] == emit.lock_commands[1]["seq"]


def test_observe_status_advances_pending_states():
    emit = _RecordingEmit(lock_state=None, unlock_state=None)
    controller = _controller(emit)
    controller.request_lock("grasp", base_stopped=True)
    assert controller.state == LOCK.BaseLockState.LOCK_REQUESTED
    controller.observe_status({"base_lock": {"state": "locked"}})
    assert controller.state == LOCK.BaseLockState.LOCKED
    controller.request_unlock("grasp")
    assert controller.state == LOCK.BaseLockState.UNLOCK_REQUESTED
    controller.observe_status({"base_lock": {"state": "unlocked"}})
    assert controller.state == LOCK.BaseLockState.UNLOCKED


def test_transport_failure_is_fail_open_and_never_raises():
    emit = _RecordingEmit(raises=True)
    logged: list[str] = []
    controller = _controller(emit, logger=logged.append)

    # A raising transport must not propagate: the grasp proceeds unlocked.
    state = controller.request_lock("grasp", base_stopped=True)
    assert state == LOCK.BaseLockState.LOCK_REQUESTED  # emitted, undelivered
    assert controller.status_field()["delivered"] is False
    assert any("transport failed" in message for message in logged)
    # Unlock in the finally path likewise never raises.
    assert controller.request_unlock("grasp") == LOCK.BaseLockState.UNLOCK_REQUESTED


def test_status_field_reports_since_and_source():
    clock = {"t": 100.0}
    emit = _RecordingEmit()
    controller = _controller(emit, clock=lambda: clock["t"])
    controller.request_lock("mobile_handoff_grasp", base_stopped=True)
    clock["t"] = 103.5
    field = controller.status_field()
    assert field["state"] == "locked"
    assert field["source"] == "mobile_handoff_grasp"
    assert abs(field["since_s"] - 3.5) < 1e-9


# --------------------------------------------------------------------------
# NUC latch + watchdog
# --------------------------------------------------------------------------

def _lock_command(*, seq: int = 1, lease_s: float = 120.0, source: str = "grasp") -> dict:
    return LOCK.build_command(lock=True, source=source, seq=seq, lease_s=lease_s)


def _unlock_command(*, seq: int = 2, source: str = "grasp") -> dict:
    return LOCK.build_command(lock=False, source=source, seq=seq, lease_s=120.0)


def test_latch_locks_and_unlocks_on_commands():
    clock = {"t": 0.0}
    latch = LOCK.BaseLockLatch(clock=lambda: clock["t"])
    assert latch.locked() is False
    assert latch.apply(_lock_command()) == "locked"
    assert latch.locked() is True
    assert latch.apply(_unlock_command()) == "unlocked"
    assert latch.locked() is False


def test_latch_lock_is_idempotent_and_renews_lease():
    clock = {"t": 0.0}
    latch = LOCK.BaseLockLatch(lease_s=120.0, clock=lambda: clock["t"])
    assert latch.apply(_lock_command(seq=1)) == "locked"
    clock["t"] = 30.0
    # A repeat lock renews the lease (does not re-lock) and pushes the deadline.
    assert latch.apply(_lock_command(seq=2)) == "renewed"
    field = latch.status_field()
    assert field["state"] == "locked"
    # Lease remaining reflects the renewed deadline (30 + 120 - 30 = 120).
    assert abs(field["lease_remaining_s"] - 120.0) < 1e-9
    # since_s tracks the original lock, not the renewal.
    assert abs(field["since_s"] - 30.0) < 1e-9


def test_latch_watchdog_expires_after_lease():
    clock = {"t": 0.0}
    latch = LOCK.BaseLockLatch(lease_s=120.0, clock=lambda: clock["t"])
    latch.apply(_lock_command(lease_s=120.0))

    clock["t"] = 119.0
    assert latch.poll() is None
    assert latch.locked() is True

    clock["t"] = 120.5
    assert latch.poll() == "watchdog_expired"
    assert latch.locked() is False
    assert latch.status_field()["watchdog_expiries"] == 1


def test_latch_renewal_prevents_watchdog_expiry():
    clock = {"t": 0.0}
    latch = LOCK.BaseLockLatch(lease_s=120.0, clock=lambda: clock["t"])
    latch.apply(_lock_command(seq=1, lease_s=120.0))
    # Heartbeat renewals arrive every 30 s; the lease never lapses.
    for renew_at in (30.0, 60.0, 90.0, 121.0, 150.0):
        clock["t"] = renew_at
        assert latch.poll() is None
        latch.apply(_lock_command(seq=1, lease_s=120.0))
    assert latch.locked() is True


def test_latch_ignores_malformed_and_stale_commands():
    latch = LOCK.BaseLockLatch()
    assert latch.apply("not-a-dict") == "ignored"
    assert latch.apply({"schema": "wrong", "lock": True}) == "ignored"
    assert latch.apply({"schema": LOCK.BASE_LOCK_COMMAND_SCHEMA, "lock": "yes"}) == "ignored"
    # Unlock while already unlocked is a no-op.
    assert latch.apply(_unlock_command()) == "ignored"


def test_latch_records_stance_capability_and_residual():
    latch = LOCK.BaseLockLatch()
    latch.apply(_lock_command())
    latch.note_stance(supported=False, residual="StandUp returned 3203; StopMove-hold only")
    field = latch.status_field()
    assert field["stance_lock_supported"] is False
    assert "3203" in field["residual"]
    assert field["state"] == "locked"


def test_build_command_shape():
    command = LOCK.build_command(lock=True, source="grasp", seq=7, lease_s=90.0)
    assert command["schema"] == LOCK.BASE_LOCK_COMMAND_SCHEMA
    assert command["lock"] is True
    assert command["source"] == "grasp"
    assert command["seq"] == 7
    assert command["lease_s"] == 90.0
    assert isinstance(command["requested_unix_ns"], int)
