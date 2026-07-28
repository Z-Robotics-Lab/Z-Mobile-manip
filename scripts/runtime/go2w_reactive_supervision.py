#!/usr/bin/env python3
"""Pure watchdog and JSONL replay helpers for the reactive mobile workflow.

This module has no ROS, WebRTC, CAN, or robot SDK dependency.  It exists so
historical depth-servo traces can exercise the same bounded-wait policy used by
the loopback workbench supervisor without constructing any actuator transport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

# systemd runs the workbench as ``python3 <abs path>/go2w_planning_control.py``
# with no PYTHONPATH, so ``sys.path[0]`` is ``scripts/runtime`` and the repo
# root is NOT importable by default.  ``go2w_interactive_sessions`` happens to
# insert it, and happens to be imported one line earlier -- do not depend on
# that: this module is also imported directly by ``go2w_reactive_replay.py``.
# Mirrors the pattern in go2w_interactive_sessions.py.
STACK_ROOT = Path(__file__).resolve().parent.parent.parent
if str(STACK_ROOT) not in sys.path:
    sys.path.insert(0, str(STACK_ROOT))

from z_manip.control.servo_phase import (  # noqa: E402
    MIN_EFFECTIVE_BASE_COMMAND,
    NO_PROGRESS_DEADLINE_S,
    BaseOwner,
    ExpiryAction,
    ServoPhase,
    is_known_phase,
    phase_policy,
)
from z_manip.control.servo_phase import (  # noqa: E402
    HANDOFF_PHASES,
    HEARTBEAT_EXEMPT_PHASES,
    POSTURE_WAIT_PHASES,
    TERMINAL_PHASES,
)


# NOTE: every phase set above is now DERIVED from the single
# ``z_manip.control.servo_phase.PHASE_POLICY`` table and re-exported here only
# so existing importers keep working.  Do not hand-write a phase set in this
# module again.
#
# The previous design assigned ``deadline_s`` ONLY inside a three-member
# ``POSTURE_WAIT_PHASES`` set whose strings ("posture_adjust",
# "posture_shadow_verified", "posture_blocked") have ZERO occurrences in any
# recorded corpus -- the deployed servo never emits them.  Every other phase,
# including the ``whole_body_shadow`` phase that held the base at (0, 0) for a
# recorded 23.1 s with tracking green, carried ``deadline_s = None`` and could
# not time out.  ``HEARTBEAT_EXEMPT_PHASES`` did not help: the heartbeat only
# checks that ``updated_unix_ns`` advances, which a servo publishing zeros at
# 20 Hz satisfies forever.
#
# The table inverts that: EVERY phase carries a deadline and the exemptions
# are the enumerated terminal states.
#
# A per-phase deadline is still not enough on its own, because it is measured
# from the phase transition and so is rearmed by ANY phase change -- including
# a change between two phases that both publish (0, 0).  The recorded corpus
# flaps ``whole_body_approach``/``whole_body_shadow`` at 0.05-0.75 s intervals
# (they are the two arms of one ``command.executable`` ternary) and splits long
# approach runs with single ``transform_unavailable`` ticks.  So the watchdog
# ALSO carries a cross-phase accumulator, ``no_progress_s``, which is cleared
# only by a real base command and bounded by ``NO_PROGRESS_DEADLINE_S``.
__all__ = [
    "HANDOFF_PHASES",
    "HEARTBEAT_EXEMPT_PHASES",
    "POSTURE_WAIT_PHASES",
    "ReactivePhaseWatchdog",
    "ReactiveWatchdogConfig",
    "ReactiveWatchdogDecision",
    "TERMINAL_PHASES",
    "load_jsonl",
    "ownership_snapshot",
    "posture_feedback_snapshot",
    "replay_trace",
]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def posture_feedback_snapshot(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the depth-servo wrapper and the direct posture document."""

    wrapper = _mapping(runtime.get("posture_status"))
    document = _mapping(wrapper.get("document")) or wrapper
    feedback = _mapping(document.get("feedback"))
    wrapper_age = _finite(wrapper.get("age_s"))
    body = _mapping(document.get("body_height"))
    feedback_age = _finite(body.get("feedback_age_s"))
    age_s = wrapper_age if wrapper_age is not None else feedback_age
    return {
        "available": document.get("schema") == "z_manip.go2w_posture_status.v1",
        "fresh": feedback.get("fresh") is True,
        "age_s": age_s,
        "phase": str(document.get("phase", "unavailable")),
        "owner": str(document.get("command_owner", "unavailable")),
        "detail": str(document.get("detail", "")),
        "current_height_m": _finite(body.get("current_m")),
        "target_height_m": _finite(body.get("target_m")),
        "current_pitch_rad": _finite(
            _mapping(document.get("attitude")).get("current_pitch_rad")
        ),
        "target_pitch_rad": _finite(
            _mapping(document.get("attitude")).get("target_pitch_rad")
        ),
    }


def base_command_magnitude(runtime: Mapping[str, Any]) -> float:
    """Return the largest base command component the servo actually issued.

    ``live`` mode is judged on the PUBLISHED command; every other mode is
    judged on the PROPOSED command, because a shadow servo publishes zero by
    construction (``published = value if live else 0.0``).  Judging shadow
    mode on published values would report every healthy shadow run as a
    stalled base.
    """

    output = _mapping(runtime.get("output"))
    live = str(runtime.get("mode", "")) == "live"
    linear_key = "published_linear_x" if live else "proposed_linear_x"
    yaw_key = "published_angular_z" if live else "proposed_angular_z"
    linear = _finite(output.get(linear_key)) or 0.0
    yaw = _finite(output.get(yaw_key)) or 0.0
    return max(abs(linear), abs(yaw))


def base_is_commanding(runtime: Mapping[str, Any]) -> bool:
    """True only when the base command is large enough to move the chassis.

    The threshold is ``MIN_EFFECTIVE_BASE_COMMAND``, not zero: a servo
    publishing 0.001 m/s forever is not driving anything (the shipped
    ``min_forward_mps`` is 0.10 precisely because of the Go2W dead zone), and
    treating it as progress would rearm every stall timer indefinitely.
    """

    return base_command_magnitude(runtime) >= MIN_EFFECTIVE_BASE_COMMAND


def ownership_snapshot(runtime: Mapping[str, Any]) -> dict[str, str]:
    """Return one inspectable owner per actuator group.

    Missing feedback is reported as ``unavailable``; an arm-view intent alone
    is deliberately labelled ``intent_only`` rather than pretending an arm
    executor accepted it.

    ``base`` is derived from the MEASURED command only.  It used to
    short-circuit on the phase term::

        base_owner = "visual_servo" if (
            phase in {"approach", "base_approach", "whole_body_approach"}
            or abs(published_linear) > 1e-9
            or abs(published_yaw) > 1e-9
        ) else "zero_hold"

    so any row whose phase was one of those three read ``visual_servo`` even
    while publishing exactly (0, 0).  That is reachable in live mode:
    ``_whole_body_output`` emits ``whole_body_approach`` whenever the QP
    command is executable, and the forward clip can return 0.0 while the
    yaw deadband zeroes the turn on the same tick -- a parked robot reported
    as an actively servoing one.  The phase's CLAIM now lives in
    ``PhasePolicy.expected_base_owner``; comparing it against this measured
    owner is what detects the stall, instead of hiding it.
    """

    posture = posture_feedback_snapshot(runtime)
    reactive = _mapping(runtime.get("reactive"))
    arm_view = _mapping(reactive.get("arm_view"))
    arm_feedback = _mapping(runtime.get("arm_view_status"))
    base_owner = (
        BaseOwner.VISUAL_SERVO.value
        if base_is_commanding(runtime)
        else BaseOwner.ZERO_HOLD.value
    )
    arm_owner = str(arm_feedback.get("owner", ""))
    if not arm_owner:
        arm_owner = (
            "intent_only"
            if str(arm_view.get("mode", "hold")) not in {"", "hold"}
            else "none"
        )
    whole_body = _mapping(runtime.get("whole_body"))
    optimizer = _mapping(whole_body.get("command")) or _mapping(runtime.get("optimizer"))
    optimizer_owner = str(
        optimizer.get("backend")
        or optimizer.get("owner")
        or ("initializing" if whole_body.get("enabled") is True else "unavailable")
    )
    return {
        "base": base_owner,
        "body": str(posture["owner"]),
        "arm_view": arm_owner,
        "optimizer": optimizer_owner,
    }


@dataclass(frozen=True)
class ReactiveWatchdogConfig:
    posture_wait_timeout_s: float = 12.0
    feedback_freshness_s: float = 0.75
    state_heartbeat_timeout_s: float = 1.5
    max_state_future_s: float = 0.25

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.posture_wait_timeout_s)
            or self.posture_wait_timeout_s <= 0
        ):
            raise ValueError("posture wait timeout must be finite and positive")
        if (
            not math.isfinite(self.feedback_freshness_s)
            or self.feedback_freshness_s <= 0
        ):
            raise ValueError("feedback freshness must be finite and positive")
        if (
            not math.isfinite(self.state_heartbeat_timeout_s)
            or self.state_heartbeat_timeout_s <= 0
        ):
            raise ValueError("state heartbeat timeout must be finite and positive")
        if not math.isfinite(self.max_state_future_s) or self.max_state_future_s < 0:
            raise ValueError("maximum state future allowance must be finite and non-negative")


@dataclass(frozen=True)
class ReactiveWatchdogDecision:
    phase: str
    phase_elapsed_s: float
    deadline_s: float | None
    timed_out: bool
    code: str | None
    message: str
    feedback: dict[str, Any]
    owners: dict[str, str]
    heartbeat_required: bool
    heartbeat_valid: bool
    heartbeat_updated_unix_ns: int | None
    heartbeat_age_s: float | None
    heartbeat_elapsed_s: float
    heartbeat_deadline_s: float | None
    # --- phase-table fields -------------------------------------------
    #: What the supervisor must DO when this decision reports ``timed_out``.
    #: ``"none"`` only ever appears on a decision that has not timed out.
    on_expiry: str = ExpiryAction.NONE.value
    #: Who the phase CLAIMS is driving the base.
    expected_base_owner: str = BaseOwner.ZERO_HOLD.value
    #: True when the phase claims ``visual_servo`` but the measured command
    #: is exactly zero.  This is the stall signature, and it is the reason
    #: ``phase_elapsed_s`` is accumulating for a driving phase.
    base_owner_mismatch: bool = False
    is_terminal: bool = False
    #: False when the phase string is not in ``ServoPhase``; such a phase is
    #: NOT exempt, it inherits the fail-closed unknown-phase policy.
    phase_known: bool = True
    #: Seconds the base has gone without an effective command, ACROSS phase
    #: changes.  ``phase_elapsed_s`` is rearmed by any phase transition; this
    #: is not.  It is the value that actually bounds a flapping stall.
    no_progress_s: float = 0.0
    #: The bound on ``no_progress_s``; None while the current phase does not
    #: accumulate it (terminal phases and ``starting``).
    no_progress_deadline_s: float | None = None

    def document(self) -> dict[str, Any]:
        return asdict(self)


class ReactivePhaseWatchdog:
    """Bound one blocking reactive phase and expose the exact missing owner."""

    def __init__(self, config: ReactiveWatchdogConfig | None = None) -> None:
        self.config = config or ReactiveWatchdogConfig()
        self.reset()

    def deadline_for(self, phase: str) -> float | None:
        """Return the configured deadline for ``phase``.

        The posture bounded wait keeps honouring the operator-configured
        ``posture_wait_timeout_s`` so the shipped CLI flag still works; every
        other phase takes its deadline from the table.
        """

        policy = phase_policy(phase)
        if policy.is_posture_wait:
            return self.config.posture_wait_timeout_s
        return policy.deadline_s

    def note_supervisor_progress(self) -> None:
        """Clear the cross-phase no-progress budget after a bounded recovery.

        A stationary wrist search or a tracker reseed moves no wheels, so the
        base-command accumulator would otherwise still be full when the servo
        restarts and would fire on the first observation of the recovered run
        -- burning the whole reacquisition budget in a few seconds instead of
        giving each attempt its bound.  The supervisor's recovery attempts are
        themselves counted against ``max_reacquisitions``, so this cannot
        become an unbounded rearm loop.

        It deliberately does NOT touch the heartbeat state or the per-phase
        timer: only the cross-phase budget is a supervisor-level quantity.
        """

        self._no_progress_s = 0.0

    def reset(self) -> None:
        self._phase: str | None = None
        self._phase_started_s: float | None = None
        #: Cross-phase accumulator.  Cleared only by an effective base command
        #: (or an explicit supervisor recovery); PAUSED, never cleared, by a
        #: non-counting phase.  Deliberately not keyed on the phase string:
        #: keying it on the phase is the flapping hole this closes.
        self._no_progress_s: float = 0.0
        self._last_observe_s: float | None = None
        self._heartbeat_stamp_ns: int | None = None
        self._heartbeat_progress_s: float | None = None
        self._last = ReactiveWatchdogDecision(
            phase="idle",
            phase_elapsed_s=0.0,
            deadline_s=None,
            timed_out=False,
            code=None,
            message="idle",
            feedback={},
            owners={},
            heartbeat_required=False,
            heartbeat_valid=False,
            heartbeat_updated_unix_ns=None,
            heartbeat_age_s=None,
            heartbeat_elapsed_s=0.0,
            heartbeat_deadline_s=None,
        )

    @property
    def last(self) -> ReactiveWatchdogDecision:
        return self._last

    def observe(
        self,
        runtime: Mapping[str, Any],
        *,
        now_s: float,
        now_unix_ns: int | None = None,
    ) -> ReactiveWatchdogDecision:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("watchdog timestamp must be finite")
        phase = str(
            runtime.get("phase")
            or _mapping(runtime.get("reactive")).get("phase")
            or "idle"
        )
        policy = phase_policy(phase)
        phase_group = "posture_wait" if policy.is_posture_wait else phase
        if phase_group != self._phase:
            self._phase = phase_group
            self._phase_started_s = now
        feedback = posture_feedback_snapshot(runtime)
        owners = ownership_snapshot(runtime)

        # A phase that CLAIMS to be driving the base measures its deadline
        # against a continuous zero-command span, not against time in phase:
        # a legitimately long approach issues commands and rearms the timer,
        # while a phase that stops commanding without leaving the phase --
        # the recorded 23.1 s ``whole_body_shadow`` hold with tracking green
        # -- accumulates until it expires.  A parked phase measures plain
        # time in phase, because for it a zero command is correct.
        commanding = base_is_commanding(runtime)
        expects_drive = policy.expected_base_owner is BaseOwner.VISUAL_SERVO
        base_owner_mismatch = expects_drive and not commanding
        if expects_drive and commanding:
            self._phase_started_s = now
        assert self._phase_started_s is not None
        elapsed = max(0.0, now - self._phase_started_s)

        # The cross-phase accumulator.  ``elapsed`` above is rearmed by any
        # phase transition, which is why it alone cannot bound the recorded
        # flapping stall: ``whole_body_approach``/``whole_body_shadow`` are
        # two arms of one ternary and flip at 0.05-0.75 s in the live trace,
        # and a single ``transform_unavailable`` tick splits an approach run
        # six times in the same trace.  This accumulator is cleared ONLY by an
        # effective base command; a non-counting phase (terminal, or
        # ``starting``) pauses it rather than clearing it, because clearing on
        # a phase change is the hole.
        delta_s = (
            0.0
            if self._last_observe_s is None
            else max(0.0, now - self._last_observe_s)
        )
        self._last_observe_s = now
        if commanding:
            self._no_progress_s = 0.0
        elif policy.counts_no_progress:
            self._no_progress_s += delta_s
        no_progress_s = self._no_progress_s
        no_progress_deadline_s = (
            NO_PROGRESS_DEADLINE_S if policy.counts_no_progress else None
        )
        no_progress_timed_out = (
            no_progress_deadline_s is not None
            and no_progress_s >= no_progress_deadline_s
        )

        deadline_s = self.deadline_for(phase)
        phase_timed_out = deadline_s is not None and elapsed >= deadline_s
        posture_timed_out = policy.is_posture_wait and phase_timed_out

        wall_ns = time.time_ns() if now_unix_ns is None else now_unix_ns
        if isinstance(wall_ns, bool) or not isinstance(wall_ns, int) or wall_ns <= 0:
            raise ValueError("watchdog wall timestamp must be a positive integer")
        raw_stamp = runtime.get("updated_unix_ns")
        stamp_ns = (
            raw_stamp
            if isinstance(raw_stamp, int) and not isinstance(raw_stamp, bool) and raw_stamp > 0
            else None
        )
        heartbeat_required = policy.heartbeat_required
        heartbeat_valid = False
        heartbeat_age_s: float | None = None
        heartbeat_invalid_reason: str | None = None
        if stamp_ns is not None:
            heartbeat_age_s = (wall_ns - stamp_ns) / 1e9
            if heartbeat_age_s < -self.config.max_state_future_s:
                heartbeat_invalid_reason = "state heartbeat timestamp is in the future"
            elif (
                self._heartbeat_stamp_ns is not None
                and stamp_ns < self._heartbeat_stamp_ns
            ):
                heartbeat_invalid_reason = "state heartbeat timestamp moved backwards"
            else:
                heartbeat_valid = True
                if self._heartbeat_stamp_ns != stamp_ns:
                    self._heartbeat_stamp_ns = stamp_ns
                    self._heartbeat_progress_s = now
        else:
            heartbeat_invalid_reason = "state document has no updated_unix_ns heartbeat"
        if self._heartbeat_progress_s is None:
            self._heartbeat_progress_s = now
        heartbeat_elapsed_s = max(0.0, now - self._heartbeat_progress_s)
        heartbeat_timed_out = False
        if heartbeat_required:
            if heartbeat_invalid_reason is not None:
                # A handoff may start an arm/gripper transaction, so it is
                # never accepted from an unverified status document. Other
                # active phases receive one bounded startup grace period.
                heartbeat_timed_out = (
                    policy.is_handoff
                    or heartbeat_elapsed_s >= self.config.state_heartbeat_timeout_s
                )
            elif (
                heartbeat_age_s is not None
                and heartbeat_age_s > self.config.state_heartbeat_timeout_s
            ):
                heartbeat_timed_out = True
                heartbeat_invalid_reason = "state heartbeat is older than its deadline"
            elif heartbeat_elapsed_s >= self.config.state_heartbeat_timeout_s:
                heartbeat_timed_out = True
                heartbeat_invalid_reason = "state heartbeat stopped advancing"

        timed_out = phase_timed_out or no_progress_timed_out or heartbeat_timed_out
        code: str | None = None
        message = "phase is progressing within its bounded wait"
        # Keep the more specific posture diagnosis when both deadlines expire
        # on the same observation; the state heartbeat deadline still causes
        # the same stationary terminal transition.
        if posture_timed_out:
            age_s = feedback.get("age_s")
            feedback_stale = (
                not feedback.get("available")
                or not feedback.get("fresh")
                or not isinstance(age_s, (int, float))
                or float(age_s) > self.config.feedback_freshness_s
            )
            if feedback_stale:
                code = "POSTURE_FEEDBACK_TIMEOUT"
                message = (
                    "posture executor feedback is unavailable or stale; "
                    "reactive control degraded to a stationary terminal state"
                )
            elif owners.get("arm_view") == "intent_only":
                code = "ARM_VIEW_FEEDBACK_TIMEOUT"
                message = (
                    "arm-view intent has no measured executor owner; reactive "
                    "control degraded to a stationary terminal state"
                )
            else:
                code = "POSTURE_SETTLE_TIMEOUT"
                message = (
                    "measured posture did not settle before the bounded deadline; "
                    "reactive control degraded to a stationary terminal state"
                )
        elif heartbeat_timed_out:
            code = "REACTIVE_STATE_HEARTBEAT_TIMEOUT"
            message = (
                f"{heartbeat_invalid_reason or 'state heartbeat deadline expired'}; "
                "reactive control degraded to a stationary terminal state"
            )
        elif phase_timed_out:
            if base_owner_mismatch:
                code = "BASE_COMMAND_STALL_TIMEOUT"
                message = (
                    f"phase {phase!r} claims the base "
                    f"({policy.expected_base_owner.value}) but has commanded "
                    f"exactly zero for {elapsed:.2f}s (deadline {deadline_s:.2f}s); "
                    "the base is stopped with no owner driving it"
                )
            elif not is_known_phase(phase):
                code = "UNKNOWN_PHASE_TIMEOUT"
                message = (
                    f"phase {phase!r} is not in the servo phase table; it was "
                    f"bounded by the fail-closed unknown-phase deadline "
                    f"({deadline_s:.2f}s) after {elapsed:.2f}s"
                )
            else:
                code = "PHASE_DEADLINE_TIMEOUT"
                message = (
                    f"phase {phase!r} exceeded its {deadline_s:.2f}s bounded "
                    f"wait ({elapsed:.2f}s elapsed) without reaching a "
                    "terminal state"
                )
        elif no_progress_timed_out:
            # The flapping case: no single phase held long enough to expire,
            # but the base has not moved across all of them.
            code = "BASE_PROGRESS_STALL_TIMEOUT"
            message = (
                f"the base has issued no effective command for "
                f"{no_progress_s:.2f}s across phase changes (bound "
                f"{no_progress_deadline_s:.2f}s); the current phase "
                f"{phase!r} has only been held {elapsed:.2f}s, so no "
                "per-phase deadline expired -- the run is not advancing"
            )
        self._last = ReactiveWatchdogDecision(
            phase=phase,
            phase_elapsed_s=elapsed,
            deadline_s=deadline_s,
            timed_out=timed_out,
            code=code,
            message=message,
            feedback=feedback,
            owners=owners,
            heartbeat_required=heartbeat_required,
            heartbeat_valid=heartbeat_valid,
            heartbeat_updated_unix_ns=stamp_ns,
            heartbeat_age_s=heartbeat_age_s,
            heartbeat_elapsed_s=heartbeat_elapsed_s,
            heartbeat_deadline_s=(
                self.config.state_heartbeat_timeout_s
                if heartbeat_required else None
            ),
            # A heartbeat failure always stops: the status document itself is
            # untrustworthy, so no phase-specific escalation is justified.
            on_expiry=(
                ExpiryAction.STOP_AND_DEGRADE.value
                if heartbeat_timed_out
                else policy.on_expiry.value
                if (phase_timed_out or no_progress_timed_out)
                else ExpiryAction.NONE.value
            ),
            expected_base_owner=policy.expected_base_owner.value,
            base_owner_mismatch=base_owner_mismatch,
            is_terminal=policy.is_terminal,
            phase_known=is_known_phase(phase),
            no_progress_s=no_progress_s,
            no_progress_deadline_s=no_progress_deadline_s,
        )
        return self._last


def _no_progress_runs(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find zero-command runs that SURVIVE phase changes.

    This is the offline twin of the deployed watchdog's cross-phase
    accumulator, and uses the same rule so an offline verdict transfers: a run
    is broken only by an effective base command, and a non-counting phase
    (terminal, or ``starting``) pauses it rather than breaking it.
    """

    runs: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for record in rows:
        stamp_ns = record.get("updated_unix_ns")
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            continue
        phase = str(record.get("phase", "unknown"))
        policy = phase_policy(phase)
        if base_is_commanding(record):
            active = None
            continue
        if not policy.counts_no_progress:
            # Paused, not broken: an interleaved terminal/starting row must
            # not be able to launder a stall into two short runs.
            continue
        if active is None:
            active = {
                "start_unix_ns": stamp_ns,
                "end_unix_ns": stamp_ns,
                "samples": 1,
                "phases": [phase],
            }
            runs.append(active)
        else:
            active["end_unix_ns"] = stamp_ns
            active["samples"] += 1
            if phase not in active["phases"]:
                active["phases"].append(phase)
    stalls: list[dict[str, Any]] = []
    for run in runs:
        duration_s = max(0.0, (run["end_unix_ns"] - run["start_unix_ns"]) / 1e9)
        if duration_s < NO_PROGRESS_DEADLINE_S:
            continue
        stalls.append({
            **run,
            "duration_s": duration_s,
            "deadline_s": NO_PROGRESS_DEADLINE_S,
            "code": "BASE_PROGRESS_STALL",
            "recommended_terminal_phase": ServoPhase.DEGRADED.value,
        })
    return stalls


def replay_trace(
    records: Iterable[Mapping[str, Any]],
    *,
    stall_threshold_s: float = 5.0,
) -> dict[str, Any]:
    """Summarize contiguous phase spans and identify zero-command stalls.

    Two detectors run, and both must be clean for ``passed``:

    ``stalls``
        The per-span heuristic: any NON-TERMINAL phase that held the base at
        exactly zero for longer than the smaller of ``stall_threshold_s`` and
        the phase's own deadline.  It used to be restricted to
        ``POSTURE_WAIT_PHASES``, whose three strings the deployed servo never
        emits -- which is why replaying the live trace that contains a 23.1 s
        zero-command ``whole_body_shadow`` span reported ``passed: True``.
        This detector is per-span, so a trace that FLAPS between two parked
        phases splits into short spans and hides from it.

    ``no_progress_stalls``
        The cross-phase detector, which mirrors the deployed watchdog's
        ``NO_PROGRESS_DEADLINE_S`` bound exactly: contiguous rows in counting
        phases with no effective base command, regardless of how the phase
        string changes underneath.  Without it a green replay did not
        transfer to the live guard in the fail-open direction, and a red one
        could name a stall the live guard would never stop.
    """

    rows = [record for record in records if isinstance(record, Mapping)]
    no_progress_stalls = _no_progress_runs(rows)
    spans: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for record in rows:
        stamp_ns = record.get("updated_unix_ns")
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            continue
        phase = str(record.get("phase", "unknown"))
        output = _mapping(record.get("output"))
        linear = _finite(output.get("published_linear_x")) or 0.0
        yaw = _finite(output.get("published_angular_z")) or 0.0
        reason = str(output.get("reason", ""))
        if active is None or active["phase"] != phase:
            if active is not None:
                spans.append(active)
            active = {
                "phase": phase,
                "start_unix_ns": stamp_ns,
                "end_unix_ns": stamp_ns,
                "samples": 1,
                "all_zero_command": abs(linear) <= 1e-9 and abs(yaw) <= 1e-9,
                "last_reason": reason,
            }
        else:
            active["end_unix_ns"] = stamp_ns
            active["samples"] += 1
            active["all_zero_command"] = bool(active["all_zero_command"]) and (
                abs(linear) <= 1e-9 and abs(yaw) <= 1e-9
            )
            if reason:
                active["last_reason"] = reason
    if active is not None:
        spans.append(active)
    stalls: list[dict[str, Any]] = []
    for span in spans:
        duration_s = max(0.0, (span["end_unix_ns"] - span["start_unix_ns"]) / 1e9)
        span["duration_s"] = duration_s
        policy = phase_policy(span["phase"])
        span["deadline_s"] = policy.deadline_s
        span["expected_base_owner"] = policy.expected_base_owner.value
        span["is_terminal"] = policy.is_terminal
        if policy.is_terminal or policy.deadline_s is None:
            continue
        budget_s = min(float(stall_threshold_s), float(policy.deadline_s))
        if span["all_zero_command"] and duration_s >= budget_s:
            stalls.append({
                **span,
                "code": (
                    "POSTURE_WAIT_STALL"
                    if policy.is_posture_wait
                    else "BASE_COMMAND_STALL"
                    if policy.expected_base_owner is BaseOwner.VISUAL_SERVO
                    else "PHASE_DEADLINE_STALL"
                ),
                "on_expiry": policy.on_expiry.value,
                "recommended_terminal_phase": ServoPhase.DEGRADED.value,
            })
    return {
        "schema": "z_manip.reactive_trace_replay.v1",
        "record_count": len(rows),
        "span_count": len(spans),
        "spans": spans,
        "stalls": stalls,
        "no_progress_stalls": no_progress_stalls,
        "no_progress_deadline_s": NO_PROGRESS_DEADLINE_S,
        "passed": not stalls and not no_progress_stalls,
    }


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"trace line {line_number} is not an object")
            records.append(value)
    return records
