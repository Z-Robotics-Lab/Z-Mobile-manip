#!/usr/bin/env python3
"""The single source of truth for the depth-servo phase vocabulary.

Why this module exists
----------------------
Before it, "the current phase" was written six independent times with
divergent memberships:

* ``LOSS_STAIR_PHASES`` (go2w_depth_servo.py)
* an inline literal set in the whole-body branch of the same file
* ``POSTURE_WAIT_PHASES`` (go2w_reactive_supervision.py)
* ``HEARTBEAT_EXEMPT_PHASES`` (same file)
* ``HANDOFF_PHASES`` / ``HANDOFF_TERMINAL_PHASES`` (both files)
* the supervisor's inline ``{view_recovery, search_required}`` set

Nothing forced those six to agree, and they did not.  ``ReactivePhase``
emitted ``"reacquire"`` while every consumer spelled ``"reacquiring"``, so
the emitted string matched no consumer at all.

Two invariants replace that:

1. :class:`ServoPhase` enumerates every value that can appear in the
   ``phase`` field of ``z_manip.depth_servo_status.v1``.  Every
   :class:`~z_manip.control.reactive_servo.ReactivePhase` value is a member
   (checked at import).
2. :data:`PHASE_POLICY` carries one :class:`PhasePolicy` row for *every*
   member (checked at import).  Deadlines are the default and exemptions are
   the listed exception -- the inversion of the previous design, where a
   deadline was assigned only inside a three-member set whose strings the
   deployed servo never emits.

Deadline semantics
------------------
``deadline_s`` bounds *time without progress*, and ``expected_base_owner``
defines what progress means for that phase:

``BaseOwner.VISUAL_SERVO``
    The phase exists in order to drive the base.  Progress is a non-zero
    base command, so the timer resets on every commanded tick and only
    accumulates across a continuous zero-command span.  This is the recorded
    stall signature: 23.1 s of ``whole_body_shadow`` in ``mode: live`` with
    ``published_linear_x == published_angular_z == 0.0`` and ``tracking:
    true`` (depth-servo.trace.jsonl, 101 consecutive samples).

``BaseOwner.ZERO_HOLD``
    The phase is a parked wait.  A zero command is correct, so progress is
    only leaving the phase, and the timer runs on time-in-phase.

The measured command is the *published* one in ``live`` mode and the
*proposed* one otherwise: a shadow-mode servo publishes zero by construction,
so judging it on published values would bound every healthy shadow run.

Why a per-phase deadline is not sufficient on its own
-----------------------------------------------------
A per-phase deadline is measured from the moment the phase is entered, so it
is reset by any phase change -- INCLUDING a change from one zero-command
phase to another zero-command phase.  Phase flapping is not hypothetical: it
is the dominant pattern in the recorded corpus.  ``whole_body_approach`` and
``whole_body_shadow`` are the two arms of ONE ternary on
``command.executable`` (go2w_depth_servo.py), and the recorded trace contains
15 flips between them with inter-flip intervals of 0.05-0.75 s; the same trace
shows single ``transform_unavailable`` ticks splitting long
``whole_body_approach`` runs six times.  All of ``whole_body_shadow``,
``whole_body_approach``, ``whole_body_blocked``, ``whole_body_posture``,
``transform_unavailable`` and ``tracking_hold`` can publish exactly (0, 0), so
a base parked with tracking green can hop between them forever and reset every
per-phase timer on the way.

:data:`NO_PROGRESS_DEADLINE_S` closes that.  It bounds a base-command stall
that SURVIVES phase changes: one accumulator, cleared only by a real base
command, paused (never cleared) while the servo sits in a phase for which a
zero command is not evidence of a stall (:attr:`PhasePolicy.counts_no_progress`
is False -- the terminal phases and ``starting``).  It is pinned at import
time to be >= every counting phase's own deadline, so it can never fire
earlier than a contiguous span of that phase would have: it only removes the
flapping escape hatch, it does not tighten any single-phase bound.

This module is pure: stdlib only, no ROS/CAN/WebRTC/pinocchio/casadi, so the
supervisor's decisions stay testable on a host that cannot build the robot
stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping

from .reactive_servo import ReactivePhase


class ExpiryAction(str, Enum):
    """What the supervisor must DO when a phase deadline expires.

    Every member is an action that stops or escalates.  There is deliberately
    no "log and continue" member: a deadline whose expiry does nothing is the
    bug this table exists to remove.
    """

    #: Terminate the servo process, latch the workflow to ``degraded`` and
    #: leave the base stopped.  Used where continuing cannot help.
    STOP_AND_DEGRADE = "stop_and_degrade"
    #: Spend one bounded reacquisition attempt (perception reseed / wrist
    #: search) with the base stationary.  The reacquisition budget is itself
    #: bounded and terminates in ``blocked`` when exhausted, so this is an
    #: escalation with its own terminal, not an open loop.
    ESCALATE_VIEW_RECOVERY = "escalate_view_recovery"
    #: Reserved for terminal phases, which carry no deadline.
    NONE = "none"


class BaseOwner(str, Enum):
    """Who the phase claims is driving the mobile base."""

    VISUAL_SERVO = "visual_servo"
    ZERO_HOLD = "zero_hold"


class ServoPhase(str, Enum):
    """Every value that can appear in ``depth_servo_status.v1["phase"]``.

    Grouped by writer.  The ``ReactivePhase`` block MUST keep string parity
    with :class:`~z_manip.control.reactive_servo.ReactivePhase`; the import
    check at the bottom of this module enforces it.
    """

    # --- process lifecycle (DepthServoNode) -------------------------------
    IDLE = "idle"
    STARTING = "starting"
    STOPPED = "stopped"
    EXITED = "exited"

    # --- ReactivePhase parity block --------------------------------------
    WAITING_TARGET = "waiting_target"
    TRANSFORM_UNAVAILABLE = "transform_unavailable"
    BASE_APPROACH = "base_approach"
    POSTURE_ADJUST = "posture_adjust"
    REACQUIRE = "reacquiring"
    TRACKING_HOLD = "tracking_hold"
    VIEW_RECOVERY = "view_recovery"
    SEARCH_REQUIRED = "search_required"
    HANDOFF_PROBE = "handoff_probe"
    HANDOFF_SETTLE = "handoff_settle"
    HANDOFF_READY = "handoff_ready"

    # --- DepthServoCore presentation remaps ------------------------------
    APPROACH = "approach"
    REACHED = "reached"
    POSTURE_SHADOW_VERIFIED = "posture_shadow_verified"
    POSTURE_BLOCKED = "posture_blocked"
    SETTLING = "settling"
    TRACKING_LOST = "tracking_lost"

    # --- whole-body branch (DepthServoNode._whole_body_output) ------------
    WHOLE_BODY_APPROACH = "whole_body_approach"
    WHOLE_BODY_POSTURE = "whole_body_posture"
    WHOLE_BODY_SHADOW = "whole_body_shadow"
    WHOLE_BODY_BLOCKED = "whole_body_blocked"

    # --- bounded exit for the one-way handoff latch -----------------------
    HANDOFF_ABANDONED = "handoff_abandoned"

    # --- workflow terminals mirrored into the same field ------------------
    BLOCKED = "blocked"
    DEGRADED = "degraded"
    GRASP_STARTED = "grasp_started"


@dataclass(frozen=True)
class PhasePolicy:
    """The bounded-wait contract for exactly one phase."""

    deadline_s: float | None
    on_expiry: ExpiryAction
    heartbeat_required: bool
    expected_base_owner: BaseOwner
    is_terminal: bool
    #: The servo has lost or frozen its live 3-D track.  Consumers must not
    #: solve a whole-body step on the retained target, and a bundle arriving
    #: out of one of these spans the loss dwell (view-damping EMA skip).
    is_loss_stair: bool = False
    #: The servo is asking the supervisor to stop it and open the close-range
    #: grasp transaction.  Membership here starts an arm/gripper transaction,
    #: so it is deliberately narrow: ``handoff_settle`` (still settling) and
    #: ``handoff_abandoned`` (the latch gave up) are NOT members.
    is_handoff: bool = False
    #: Members of the posture bounded wait; they share one budget so a
    #: sub-phase flip cannot rearm the timer.
    is_posture_wait: bool = False
    #: Whether time spent in this phase with the base at zero counts toward
    #: the cross-phase :data:`NO_PROGRESS_DEADLINE_S` accumulator.
    #:
    #: True for every phase where a parked base means the run is not
    #: advancing -- which is every non-terminal phase except ``starting``,
    #: where a zero command is the CORRECT output while the container, rclpy
    #: and the first TF read come up.  A False row PAUSES the accumulator; it
    #: never clears it, because clearing on a phase change is exactly the
    #: flapping escape hatch this field exists to close.
    counts_no_progress: bool = True

    def __post_init__(self) -> None:
        if self.deadline_s is not None:
            if not math.isfinite(self.deadline_s) or self.deadline_s < 0.0:
                raise ValueError("phase deadline must be finite and non-negative")
            if self.on_expiry is ExpiryAction.NONE:
                raise ValueError(
                    "a phase with a deadline must name a real expiry action"
                )
        elif self.on_expiry is not ExpiryAction.NONE:
            raise ValueError("a phase without a deadline cannot name an expiry action")
        if self.deadline_s is None and not self.is_terminal:
            raise ValueError(
                "only a terminal phase may be exempt from a deadline; "
                "every non-terminal phase must carry one"
            )
        if self.is_terminal and self.counts_no_progress:
            raise ValueError(
                "a terminal phase must not accumulate the cross-phase "
                "no-progress budget; the supervisor acts on it immediately"
            )


_STOP = ExpiryAction.STOP_AND_DEGRADE
_RECOVER = ExpiryAction.ESCALATE_VIEW_RECOVERY
_NONE = ExpiryAction.NONE
_DRIVE = BaseOwner.VISUAL_SERVO
_HOLD = BaseOwner.ZERO_HOLD


def _terminal(*, heartbeat_required: bool, is_handoff: bool = False) -> PhasePolicy:
    return PhasePolicy(
        deadline_s=None,
        on_expiry=_NONE,
        heartbeat_required=heartbeat_required,
        expected_base_owner=_HOLD,
        is_terminal=True,
        is_handoff=is_handoff,
        counts_no_progress=False,
    )


#: Deadline every non-terminal phase inherits unless it names its own.  12.0 s
#: matches the shipped ``posture_wait_timeout_s`` and sits well below the
#: recorded 23.1 s zero-command stall.
DEFAULT_DEADLINE_S = 12.0

#: The cross-phase bound on a base-command stall.  Unlike ``deadline_s`` this
#: budget is NOT reset by a phase change; only a real base command clears it.
#: It exists because ``deadline_s`` alone is defeated by phase flapping, which
#: the recorded corpus shows happening at 0.05-0.75 s intervals between two
#: phases that both publish (0, 0).
#:
#: The value is pinned by ``_check_no_progress_bound`` below to be >= every
#: counting phase's own ``deadline_s``, so this bound can only fire in a
#: situation no single contiguous phase would have tolerated either.  Raising
#: any per-phase deadline above it is an import-time error, not a silent
#: inversion.
NO_PROGRESS_DEADLINE_S = 20.0

#: A published/proposed base command below this magnitude is not motion.
#:
#: The shipped Go2W settings snap any non-trivial forward command up to
#: ``min_forward_mps = 0.10`` precisely because of the chassis dead zone, and
#: the yaw slew limiter steps by ``max_yaw_step_rps = 0.015``, so the smallest
#: command the servo can legitimately hold is two orders of magnitude above
#: this floor.  Without a floor the stall timer is rearmed forever by a servo
#: publishing 0.001 m/s -- a number that moves nothing.
MIN_EFFECTIVE_BASE_COMMAND = 5e-3

#: A phase string this table does not know is a fail-closed condition, not a
#: free pass: it gets a short bounded life and a hard stop.  The previous
#: design gave any unlisted phase ``deadline_s = None`` forever.
UNKNOWN_PHASE_POLICY = PhasePolicy(
    deadline_s=5.0,
    on_expiry=_STOP,
    heartbeat_required=True,
    expected_base_owner=_HOLD,
    is_terminal=False,
)


PHASE_POLICY: Mapping[ServoPhase, PhasePolicy] = {
    # ---- lifecycle -------------------------------------------------------
    ServoPhase.IDLE: _terminal(heartbeat_required=False),
    ServoPhase.STOPPED: _terminal(heartbeat_required=False),
    # The heartbeat exemption list is deliberately IDENTICAL to the one this
    # table replaced ({idle, stopped, blocked, degraded, grasp_started}).
    # ``exited`` and ``starting`` both rewrite the status document, so they
    # keep the heartbeat contract; exempting them would have widened a
    # 1.5 s bound to 30 s.
    ServoPhase.EXITED: _terminal(heartbeat_required=True),
    # Container spawn + rclpy init + the first TF/runtime-state read is
    # measured in seconds, not tens of seconds; 30 s is a spawn backstop on
    # top of (not instead of) the state heartbeat.
    # ``counts_no_progress=False``: a zero base command during spawn is the
    # correct output, not a stall, and 30 s of it must not consume the 20 s
    # cross-phase budget before the approach has issued its first command.
    ServoPhase.STARTING: PhasePolicy(
        deadline_s=30.0,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        counts_no_progress=False,
    ),

    # ---- acquisition / loss stair ---------------------------------------
    ServoPhase.WAITING_TARGET: PhasePolicy(
        deadline_s=20.0,
        on_expiry=_RECOVER,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_loss_stair=True,
    ),
    # A persistent TF / runtime-observer outage is a real fault: no amount of
    # wrist searching produces a transform, so escalating to recovery would
    # spend the reacquisition budget on nothing.
    ServoPhase.TRANSFORM_UNAVAILABLE: PhasePolicy(
        deadline_s=8.0,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_loss_stair=True,
    ),
    ServoPhase.TRACKING_LOST: PhasePolicy(
        deadline_s=20.0,
        on_expiry=_RECOVER,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_loss_stair=True,
    ),
    # Frozen on stale data.  The supervisor had NO handler for this phase at
    # all: it fell into the "just mirror the phase" branch and hung.
    #
    # STOP, not RECOVER.  ``ReactiveServoConfig`` bounds this phase itself:
    # ``tracking_hold_s`` defaults to 0.0 and is validated ``<
    # tracking_loss_grace_s`` (0.75 s), after which the controller steps to
    # VIEW_RECOVERY on its own.  So an 8 s ``tracking_hold`` means the
    # reactive FSM has stopped stepping -- and a wrist search cannot fix a
    # frozen FSM, it just sweeps the camera off a target that is still in
    # view.  Same reasoning as ``transform_unavailable`` below.
    ServoPhase.TRACKING_HOLD: PhasePolicy(
        deadline_s=8.0,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_loss_stair=True,
    ),
    # STOP for the same reason: REACQUIRE means "rebuilding a stable 3-D track
    # after posture motion" (reactive_servo.py), i.e. the target IS in view
    # and the tracker is re-converging on it.  Pointing the wrist somewhere
    # else is the one action guaranteed not to help.
    ServoPhase.REACQUIRE: PhasePolicy(
        deadline_s=12.0,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_loss_stair=True,
    ),
    ServoPhase.VIEW_RECOVERY: PhasePolicy(
        deadline_s=20.0,
        on_expiry=_RECOVER,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_loss_stair=True,
    ),
    ServoPhase.SEARCH_REQUIRED: PhasePolicy(
        deadline_s=20.0,
        on_expiry=_RECOVER,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_loss_stair=True,
    ),

    # ---- posture bounded wait (one shared budget) ------------------------
    ServoPhase.POSTURE_ADJUST: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_posture_wait=True,
    ),
    ServoPhase.POSTURE_SHADOW_VERIFIED: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_posture_wait=True,
    ),
    ServoPhase.POSTURE_BLOCKED: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_posture_wait=True,
        is_loss_stair=True,
    ),
    ServoPhase.WHOLE_BODY_POSTURE: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
        is_posture_wait=True,
    ),
    ServoPhase.SETTLING: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
    ),

    # ---- driving phases: the deadline bounds a ZERO-COMMAND span ----------
    ServoPhase.APPROACH: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_DRIVE,
        is_terminal=False,
    ),
    ServoPhase.BASE_APPROACH: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_DRIVE,
        is_terminal=False,
    ),
    ServoPhase.WHOLE_BODY_APPROACH: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_DRIVE,
        is_terminal=False,
    ),
    # ``whole_body_shadow``/``whole_body_blocked`` are the NON-EXECUTABLE
    # variants of the approach: the QP refused, so the servo publishes zero
    # while still claiming to be approaching.  This is the recorded 23.1 s
    # live stall (backend "fixed-fixture-collision-gate", success false,
    # tracking true).  They expect to drive, so the deadline runs.
    ServoPhase.WHOLE_BODY_SHADOW: PhasePolicy(
        deadline_s=DEFAULT_DEADLINE_S,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_DRIVE,
        is_terminal=False,
    ),
    ServoPhase.WHOLE_BODY_BLOCKED: PhasePolicy(
        deadline_s=8.0,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_DRIVE,
        is_terminal=False,
    ),

    # ---- handoff transaction boundary ------------------------------------
    # The supervisor acts on these on the tick it observes them, so they are
    # terminal for the servo.  The heartbeat stays REQUIRED: a handoff opens
    # an arm/gripper transaction and is never accepted on an unverified doc.
    ServoPhase.HANDOFF_PROBE: _terminal(heartbeat_required=True, is_handoff=True),
    ServoPhase.HANDOFF_READY: _terminal(heartbeat_required=True, is_handoff=True),
    ServoPhase.REACHED: _terminal(heartbeat_required=True, is_handoff=True),
    ServoPhase.HANDOFF_SETTLE: PhasePolicy(
        deadline_s=5.0,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=False,
    ),
    # The one-way handoff latch gave up waiting for the supervisor to stop
    # this process.  Escalate on the first observation; the base is already
    # held at zero and must never resume on its own.  Deliberately NOT
    # ``is_handoff``: an abandoned latch must never be read as a request to
    # open a grasp transaction.
    ServoPhase.HANDOFF_ABANDONED: PhasePolicy(
        deadline_s=0.0,
        on_expiry=_STOP,
        heartbeat_required=True,
        expected_base_owner=_HOLD,
        is_terminal=True,
        counts_no_progress=False,
    ),

    # ---- workflow terminals mirrored into the same field -----------------
    ServoPhase.BLOCKED: _terminal(heartbeat_required=False),
    ServoPhase.DEGRADED: _terminal(heartbeat_required=False),
    ServoPhase.GRASP_STARTED: _terminal(heartbeat_required=False),
}


def _missing_rows() -> tuple[str, ...]:
    return tuple(sorted(p.value for p in ServoPhase if p not in PHASE_POLICY))


_MISSING = _missing_rows()
if _MISSING:
    raise RuntimeError(
        "every ServoPhase member needs a PHASE_POLICY row; missing: "
        + ", ".join(_MISSING)
    )

_UNKNOWN_REACTIVE = tuple(sorted(
    phase.value
    for phase in ReactivePhase
    if phase.value not in {member.value for member in ServoPhase}
))
if _UNKNOWN_REACTIVE:
    raise RuntimeError(
        "ReactivePhase emits strings ServoPhase does not know: "
        + ", ".join(_UNKNOWN_REACTIVE)
    )


def _check_no_progress_bound() -> None:
    """The cross-phase budget must never be tighter than a phase's own bound.

    If some phase legitimately needs ``deadline_s = 25.0`` then 25 s of a
    parked base is, by that row's own claim, tolerable -- and a 20 s
    cross-phase budget would kill it mid-phase.  Raising a per-phase deadline
    above this bound must therefore be an import-time error that forces the
    author to raise the cross-phase bound with it, not a silent inversion.
    """

    offenders = sorted(
        f"{phase.value}={policy.deadline_s}"
        for phase, policy in PHASE_POLICY.items()
        if policy.counts_no_progress
        and policy.deadline_s is not None
        and policy.deadline_s > NO_PROGRESS_DEADLINE_S
    )
    if offenders:
        raise RuntimeError(
            "NO_PROGRESS_DEADLINE_S="
            f"{NO_PROGRESS_DEADLINE_S} is tighter than the per-phase deadline "
            "of a counting phase, so the cross-phase budget would cut a phase "
            "short: " + ", ".join(offenders)
        )
    # ``UNKNOWN_PHASE_POLICY`` counts too: an unlisted phase must not be able
    # to out-live the cross-phase budget either.
    assert UNKNOWN_PHASE_POLICY.counts_no_progress
    assert UNKNOWN_PHASE_POLICY.deadline_s is not None
    if UNKNOWN_PHASE_POLICY.deadline_s > NO_PROGRESS_DEADLINE_S:
        raise RuntimeError("the unknown-phase deadline exceeds the no-progress bound")


_check_no_progress_bound()


def phase_policy(phase: object) -> PhasePolicy:
    """Return the bounded-wait contract for ``phase``, fail-closed.

    An unrecognized string is NOT exempt; it receives
    :data:`UNKNOWN_PHASE_POLICY`.
    """

    if isinstance(phase, ServoPhase):
        return PHASE_POLICY[phase]
    try:
        return PHASE_POLICY[ServoPhase(str(phase))]
    except (ValueError, KeyError):
        return UNKNOWN_PHASE_POLICY


def is_known_phase(phase: object) -> bool:
    try:
        ServoPhase(str(phase))
    except ValueError:
        return False
    return True


def _select(predicate) -> frozenset[str]:
    return frozenset(
        phase.value for phase, policy in PHASE_POLICY.items() if predicate(policy)
    )


# Every membership below is DERIVED from the one table.  Nothing may
# hand-write a phase set again.
LOSS_STAIR_PHASES: frozenset[str] = _select(lambda p: p.is_loss_stair)
HANDOFF_PHASES: frozenset[str] = _select(lambda p: p.is_handoff)
#: Historical alias.  ``HANDOFF_PHASES`` (supervisor) and
#: ``HANDOFF_TERMINAL_PHASES`` (servo) were two hand-written copies of the
#: same three strings; they are now one derived set.
HANDOFF_TERMINAL_PHASES: frozenset[str] = HANDOFF_PHASES
POSTURE_WAIT_PHASES: frozenset[str] = _select(lambda p: p.is_posture_wait)
HEARTBEAT_EXEMPT_PHASES: frozenset[str] = _select(lambda p: not p.heartbeat_required)
TERMINAL_PHASES: frozenset[str] = _select(lambda p: p.is_terminal)
DRIVING_PHASES: frozenset[str] = _select(
    lambda p: p.expected_base_owner is BaseOwner.VISUAL_SERVO
)
RECOVERY_ESCALATION_PHASES: frozenset[str] = _select(
    lambda p: p.on_expiry is ExpiryAction.ESCALATE_VIEW_RECOVERY
)
#: Phases whose zero-command time accumulates the cross-phase stall budget.
NO_PROGRESS_PHASES: frozenset[str] = _select(lambda p: p.counts_no_progress)

#: Phase strings that runtime scripts must reference through :class:`ServoPhase`
#: rather than quoting.  The grep guard in ``tests/test_servo_phase_table.py``
#: enforces it.
GUARDED_PHASE_LITERALS: frozenset[str] = frozenset(
    phase.value
    for phase in ServoPhase
    # ``blocked``/``degraded``/``idle``/``stopped`` are also generic English
    # words used for perception-attempt outcomes and workflow bookkeeping in
    # the supervisor; guarding them would produce false positives that train
    # engineers to disable the guard.  They are not servo-emitted phases.
    if phase not in {
        ServoPhase.BLOCKED,
        ServoPhase.DEGRADED,
        ServoPhase.GRASP_STARTED,
        ServoPhase.IDLE,
    }
)


__all__ = [
    "BaseOwner",
    "DEFAULT_DEADLINE_S",
    "DRIVING_PHASES",
    "ExpiryAction",
    "GUARDED_PHASE_LITERALS",
    "HANDOFF_PHASES",
    "HANDOFF_TERMINAL_PHASES",
    "HEARTBEAT_EXEMPT_PHASES",
    "LOSS_STAIR_PHASES",
    "MIN_EFFECTIVE_BASE_COMMAND",
    "NO_PROGRESS_DEADLINE_S",
    "NO_PROGRESS_PHASES",
    "PHASE_POLICY",
    "POSTURE_WAIT_PHASES",
    "PhasePolicy",
    "RECOVERY_ESCALATION_PHASES",
    "ServoPhase",
    "TERMINAL_PHASES",
    "UNKNOWN_PHASE_POLICY",
    "is_known_phase",
    "phase_policy",
]
