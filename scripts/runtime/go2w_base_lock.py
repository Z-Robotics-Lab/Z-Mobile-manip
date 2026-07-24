#!/usr/bin/env python3
"""Transport-free Go2W base body-posture lock: owner state machine + NUC latch.

The operator observed that during a grasp the Go2W keeps running its
stepping/balance stand.  Those balance micro-motions shake the PiPER wrist
while the arm is closing on the target and corrupt the measured close-range
joints.  This module is the *pure* core of a mechanism that latches the base
into its stillest available stance for exactly the window between the
servo->grasp handoff (base already stopped) and the arm returning Home.

Design goals baked into the state machine (see the task brief "不要串行乱锁"):

* one owner -- the PC orchestrator drives :class:`BaseLockController`; the NUC
  live service only *reacts* to the commands it emits;
* idempotent -- repeating a lock or unlock request is a no-op, never a second
  transport command;
* never lock while the base is still moving -- ``request_lock`` refuses unless
  the caller proves the base has stopped (the servo phase drives base *and*
  arm and must never be locked);
* fail-open -- a failed transport is a logged warning, not a raised exception;
  the lock is an accuracy improvement, not a safety gate, so the grasp still
  runs;
* self-healing -- the NUC :class:`BaseLockLatch` auto-unlocks after a lease if
  the owner stops heartbeating (process death / lost session), so a dropped
  unlock can never strand the base in a locked stand.

Nothing here imports ROS, SSH, or Unitree symbols, so both the orchestrator
(which imports it directly) and the NUC WebRTC process (which imports the
latch inside its live loop) can use it, and every branch is unit testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any, Callable


# The command travels on a new subject inside the *existing* PC<->NUC control
# channel (the ``/go2w/*`` Domain-20 topics the live service already owns), and
# the acknowledgement rides back on the live service's ``/go2w/posture_state``
# status document under a ``base_lock`` block.
BASE_LOCK_COMMAND_TOPIC = "/go2w/base_lock"
BASE_LOCK_COMMAND_SCHEMA = "z_manip.go2w_base_lock_command.v1"
BASE_LOCK_STATUS_SCHEMA = "z_manip.go2w_base_lock_status.v1"

# Watchdog lease.  Chosen as 120 s because it must comfortably outlast any
# single grasp *stage* (the staged executor stages run at a 480 s hard ceiling,
# but each individual approach/close/lift/return move is tens of seconds) while
# still freeing the base for the next operator within ~2 min if the PC dies.
# The owner renews every 30 s, so four consecutive lost heartbeats -- a
# genuinely dead owner, not a transient Wi-Fi/SSH hiccup -- are required before
# the base is released.
DEFAULT_LOCK_LEASE_S = 120.0
DEFAULT_HEARTBEAT_PERIOD_S = 30.0


class BaseLockState(str, Enum):
    """Owner-side lifecycle.  ``*_REQUESTED`` means "emitted, awaiting ack"."""

    UNLOCKED = "unlocked"
    LOCK_REQUESTED = "lock_requested"
    LOCKED = "locked"
    UNLOCK_REQUESTED = "unlock_requested"


@dataclass(frozen=True)
class BaseLockAck:
    """Result of one emit attempt returned by the injected transport.

    ``delivered`` is whether the command reached the NUC at all; ``nuc_state``
    is the base-lock state the NUC reported back (``"locked"``/``"unlocked"``)
    or ``None`` when no acknowledgement could be read.  Either field being
    unavailable never raises -- it only keeps the owner in a ``*_REQUESTED``
    state that the next heartbeat reconciles.
    """

    delivered: bool = False
    nuc_state: str | None = None


def build_command(*, lock: bool, source: str, seq: int, lease_s: float,
                  now_unix_ns: int | None = None) -> dict[str, Any]:
    """Build one JSON-safe lock/unlock command document for the wire."""

    return {
        "schema": BASE_LOCK_COMMAND_SCHEMA,
        "lock": bool(lock),
        "source": str(source),
        "seq": int(seq),
        "lease_s": float(lease_s),
        "requested_unix_ns": int(time.time_ns() if now_unix_ns is None else now_unix_ns),
    }


class BaseLockController:
    """Single-owner base-lock state machine driven by the PC orchestrator.

    The controller is transport agnostic: it is constructed with an ``emit``
    callable that ships one command document to the NUC and returns a
    :class:`BaseLockAck`.  ``emit`` must never raise for a delivery failure --
    but if it does, the controller treats it as an undelivered attempt and
    keeps going, because the grasp must not be bricked by a lock failure.
    """

    def __init__(
        self,
        *,
        emit: Callable[[dict[str, Any]], BaseLockAck],
        clock: Callable[[], float] = time.monotonic,
        lease_s: float = DEFAULT_LOCK_LEASE_S,
        heartbeat_period_s: float | None = DEFAULT_HEARTBEAT_PERIOD_S,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self._emit = emit
        self._clock = clock
        self._lease_s = float(lease_s)
        self._heartbeat_period_s = heartbeat_period_s
        self._log = logger or (lambda _message: None)
        self._lock = threading.RLock()
        self._state = BaseLockState.UNLOCKED
        self._source: str | None = None
        self._since: float | None = None
        self._seq = 0
        self._last_refusal: str | None = None
        self._delivered = False
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

    # -- introspection -----------------------------------------------------
    @property
    def state(self) -> BaseLockState:
        with self._lock:
            return self._state

    def is_engaged(self) -> bool:
        """True once a lock has been requested and not yet released."""
        with self._lock:
            return self._state in (BaseLockState.LOCK_REQUESTED, BaseLockState.LOCKED)

    def status_field(self, now: float | None = None) -> dict[str, Any]:
        """Compact dict for the dashboard status stream: state/since_s/source."""
        with self._lock:
            moment = self._clock() if now is None else now
            since_s = None if self._since is None else max(0.0, moment - self._since)
            return {
                "state": self._state.value,
                "since_s": since_s,
                "source": self._source,
                "seq": self._seq,
                "delivered": self._delivered,
                "last_refusal": self._last_refusal,
            }

    # -- owner transitions -------------------------------------------------
    def request_lock(
        self,
        source: str,
        *,
        base_stopped: bool,
        now: float | None = None,
    ) -> BaseLockState:
        """Request the base lock.  No-op if already (being) locked.

        ``base_stopped`` is the never-lock-during-servo guard: the caller must
        prove the depth servo has fully stopped the base before locking.  A
        request with ``base_stopped=False`` is refused and recorded, and no
        command is emitted -- the servo phase moves base *and* arm together and
        must stay unlocked.
        """
        with self._lock:
            moment = self._clock() if now is None else now
            if not base_stopped:
                self._last_refusal = "refused: base is not stopped (servo phase)"
                self._log(
                    "base-lock: refused lock request from %r; base is not stopped"
                    % source
                )
                return self._state
            self._last_refusal = None
            if self._state in (BaseLockState.LOCK_REQUESTED, BaseLockState.LOCKED):
                # Idempotent: an already-engaged lock stays owned by its first
                # source; renewal is the heartbeat's job, not a re-request.
                return self._state
            self._seq += 1
            self._state = BaseLockState.LOCK_REQUESTED
            self._source = source
            self._since = moment
            command = build_command(
                lock=True, source=source, seq=self._seq, lease_s=self._lease_s,
            )
            ack = self._safe_emit(command)
            self._delivered = ack.delivered
            if ack.delivered and ack.nuc_state == "locked":
                self._state = BaseLockState.LOCKED
            self._start_heartbeat_locked()
            return self._state

    def request_unlock(self, source: str, *, now: float | None = None) -> BaseLockState:
        """Release the base lock.  No-op if already unlocked/unlocking.

        Safe to call from a ``finally`` on every grasp exit path -- success,
        failure, exception, or operator abort -- because an already-unlocked
        controller simply returns.
        """
        with self._lock:
            if self._state in (BaseLockState.UNLOCKED, BaseLockState.UNLOCK_REQUESTED):
                return self._state
            self._seq += 1
            self._state = BaseLockState.UNLOCK_REQUESTED
            command = build_command(
                lock=False, source=source, seq=self._seq, lease_s=self._lease_s,
            )
            ack = self._safe_emit(command)
            self._delivered = ack.delivered
            if ack.delivered and ack.nuc_state == "unlocked":
                self._finish_unlock_locked()
            return self._state

    def observe_status(self, nuc_status: Any, now: float | None = None) -> BaseLockState:
        """Reconcile against a NUC status document read out of band (optional).

        Advances ``LOCK_REQUESTED``->``LOCKED`` and finishes a pending unlock
        when the NUC's own ``base_lock`` block confirms the transition.  Used
        when the transport can echo the live-service status back to the PC.
        """
        with self._lock:
            nuc_state = _extract_nuc_lock_state(nuc_status)
            if nuc_state is None:
                return self._state
            if nuc_state == "locked" and self._state == BaseLockState.LOCK_REQUESTED:
                self._state = BaseLockState.LOCKED
            elif nuc_state == "unlocked" and self._state == BaseLockState.UNLOCK_REQUESTED:
                self._finish_unlock_locked()
            return self._state

    def heartbeat(self, now: float | None = None) -> BaseLockState:
        """Renew the NUC lease (and finish a pending unlock whose ack was lost).

        Called periodically by the internal heartbeat thread while engaged.
        Renewing keeps the NUC watchdog from expiring during a long grasp; the
        renewal also re-confirms ``LOCKED`` and can complete an
        ``UNLOCK_REQUESTED`` whose original acknowledgement was dropped.
        """
        with self._lock:
            if self._state in (BaseLockState.LOCK_REQUESTED, BaseLockState.LOCKED):
                command = build_command(
                    lock=True,
                    source=self._source or "heartbeat",
                    seq=self._seq,
                    lease_s=self._lease_s,
                )
                ack = self._safe_emit(command)
                self._delivered = ack.delivered
                if ack.delivered and ack.nuc_state == "locked":
                    self._state = BaseLockState.LOCKED
            elif self._state == BaseLockState.UNLOCK_REQUESTED:
                command = build_command(
                    lock=False,
                    source=self._source or "heartbeat",
                    seq=self._seq,
                    lease_s=self._lease_s,
                )
                ack = self._safe_emit(command)
                if ack.delivered and ack.nuc_state == "unlocked":
                    self._finish_unlock_locked()
            return self._state

    def close(self) -> None:
        """Stop the heartbeat thread; used on orchestrator shutdown."""
        self._stop_heartbeat()

    # -- internals ---------------------------------------------------------
    def _safe_emit(self, command: dict[str, Any]) -> BaseLockAck:
        try:
            ack = self._emit(command)
        except Exception as error:  # noqa: BLE001 - transport must never brick grasp
            self._log(
                "base-lock: transport failed for %s (seq=%s): %s"
                % (command.get("lock"), command.get("seq"), error)
            )
            return BaseLockAck(delivered=False, nuc_state=None)
        if not isinstance(ack, BaseLockAck):
            return BaseLockAck(delivered=False, nuc_state=None)
        return ack

    def _finish_unlock_locked(self) -> None:
        self._state = BaseLockState.UNLOCKED
        self._source = None
        self._since = None
        self._stop_heartbeat_locked()

    def _start_heartbeat_locked(self) -> None:
        if self._heartbeat_period_s is None or self._heartbeat_period_s <= 0.0:
            return
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()
        thread = threading.Thread(
            target=self._heartbeat_loop,
            name="z-manip-base-lock-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat_locked(self) -> None:
        self._heartbeat_stop.set()

    def _stop_heartbeat(self) -> None:
        with self._lock:
            self._stop_heartbeat_locked()
            thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _heartbeat_loop(self) -> None:
        period = self._heartbeat_period_s or DEFAULT_HEARTBEAT_PERIOD_S
        while not self._heartbeat_stop.wait(period):
            with self._lock:
                engaged = self._state in (
                    BaseLockState.LOCK_REQUESTED,
                    BaseLockState.LOCKED,
                    BaseLockState.UNLOCK_REQUESTED,
                )
            if not engaged:
                return
            self.heartbeat()


def _extract_nuc_lock_state(nuc_status: Any) -> str | None:
    """Pull ``locked``/``unlocked`` out of a NUC status doc, else ``None``."""
    if not isinstance(nuc_status, dict):
        return None
    block = nuc_status.get("base_lock")
    if not isinstance(block, dict):
        return None
    state = block.get("state")
    if state in ("locked", "unlocked"):
        return state
    return None


class BaseLockLatch:
    """NUC-side latch with a self-expiring lease (the watchdog).

    Pure logic only: :meth:`apply` folds one command in, :meth:`poll` expires
    the lease, :meth:`locked` gates ``cmd_vel``, and :meth:`status_field`
    renders the acknowledgement block the owner reads back.  The live WebRTC
    node wraps this with the actual SPORT stance calls; unit tests exercise the
    latch directly.
    """

    def __init__(
        self,
        *,
        lease_s: float = DEFAULT_LOCK_LEASE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._default_lease_s = float(lease_s)
        self._clock = clock
        self._locked = False
        self._source: str | None = None
        self._since: float | None = None
        self._deadline: float | None = None
        self._seq: int | None = None
        self._lease_s = self._default_lease_s
        self._stance_supported: bool | None = None
        self._residual: str | None = None
        self._expiries = 0

    def apply(self, command: Any, now: float | None = None) -> str:
        """Fold one command document in; return the event that resulted.

        Events: ``"locked"`` (fresh lock), ``"renewed"`` (lease extended on an
        already-locked latch), ``"unlocked"`` (released), ``"ignored"``
        (malformed/duplicate that changed nothing).
        """
        moment = self._clock() if now is None else now
        if not isinstance(command, dict):
            return "ignored"
        if command.get("schema") != BASE_LOCK_COMMAND_SCHEMA:
            return "ignored"
        lock = command.get("lock")
        if not isinstance(lock, bool):
            return "ignored"
        seq = command.get("seq")
        source = command.get("source")
        lease_s = command.get("lease_s")
        lease_value = (
            float(lease_s)
            if isinstance(lease_s, (int, float)) and not isinstance(lease_s, bool)
            and float(lease_s) > 0.0
            else self._default_lease_s
        )
        if lock:
            already = self._locked
            self._lease_s = lease_value
            self._deadline = moment + lease_value
            self._seq = seq if isinstance(seq, int) and not isinstance(seq, bool) else self._seq
            if not already:
                self._locked = True
                self._source = str(source) if source is not None else None
                self._since = moment
                return "locked"
            return "renewed"
        # unlock
        if not self._locked:
            return "ignored"
        self._release()
        return "unlocked"

    def poll(self, now: float | None = None) -> str | None:
        """Expire the lease.  Returns ``"watchdog_expired"`` on auto-unlock."""
        moment = self._clock() if now is None else now
        if self._locked and self._deadline is not None and moment > self._deadline:
            self._expiries += 1
            self._release()
            return "watchdog_expired"
        return None

    def locked(self) -> bool:
        return self._locked

    def note_stance(self, *, supported: bool, residual: str | None = None) -> None:
        """Record the SPORT stance-lock capability observed by the live node."""
        self._stance_supported = supported
        self._residual = residual

    def status_field(self, now: float | None = None) -> dict[str, Any]:
        moment = self._clock() if now is None else now
        since_s = None if self._since is None else max(0.0, moment - self._since)
        lease_remaining_s = (
            None if (self._deadline is None or not self._locked)
            else max(0.0, self._deadline - moment)
        )
        return {
            "schema": BASE_LOCK_STATUS_SCHEMA,
            "state": "locked" if self._locked else "unlocked",
            "source": self._source,
            "since_s": since_s,
            "lease_s": self._lease_s if self._locked else None,
            "lease_remaining_s": lease_remaining_s,
            "seq": self._seq,
            "stance_lock_supported": self._stance_supported,
            "residual": self._residual,
            "watchdog_expiries": self._expiries,
        }

    def _release(self) -> None:
        self._locked = False
        self._source = None
        self._since = None
        self._deadline = None
        self._lease_s = self._default_lease_s
