"""State-graph reachability: no phase may be absorbing.

The 4-area audit's structural finding (c) was that latches have no reset edges
and that a phase can be held forever.  Stages 1-4 gave every phase a deadline
and added a cross-phase no-progress budget, but *that the graph as a whole has
no absorbing non-terminal state* was never asserted anywhere.  This module
asserts it three ways:

1.  STATICALLY, over the table: every non-terminal row carries a finite
    deadline AND an expiry action that actually ends or escalates the run.
2.  OVER THE EMITTED VOCABULARY: every phase string ``go2w_depth_servo.py`` can
    put in the status document is a ``ServoPhase`` with a policy row.
3.  DYNAMICALLY, by driving the REAL ``ReactivePhaseWatchdog`` with the
    deployed input set -- including the degenerate documents a foreign or
    partial writer can produce -- and requiring every one of them to stop.

(3) is the one that matters, because it is the only check that exercises the
value that actually decides the transition rather than the table that is
supposed to describe it.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "scripts" / "runtime"
for _p in (str(REPO_ROOT), str(RUNTIME_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from z_manip.control.servo_phase import (  # noqa: E402
    NO_PROGRESS_DEADLINE_S,
    PHASE_POLICY,
    UNKNOWN_PHASE_POLICY,
    ExpiryAction,
    ServoPhase,
    phase_policy,
)

SERVO_SOURCE = RUNTIME_DIR / "go2w_depth_servo.py"


def _supervision_module():
    spec = importlib.util.spec_from_file_location(
        "supervision_under_reachability_test",
        RUNTIME_DIR / "go2w_reactive_supervision.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1. Statically, over the table.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phase", sorted(ServoPhase, key=lambda p: p.value))
def test_no_non_terminal_phase_is_absorbing(phase: ServoPhase) -> None:
    """A non-terminal phase must carry a way out of itself.

    "Terminal" means the run is over -- ``reached``, ``stopped``, ``blocked``,
    ``degraded`` and the handoff hand-offs are absorbing *on purpose*, because
    the supervisor stops supervising once it sees them.  Every OTHER phase is a
    state the robot sits in while a task is live, so it needs both a finite
    deadline and an expiry that does something.
    """

    policy = PHASE_POLICY[phase]
    if policy.is_terminal:
        pytest.skip(f"{phase.value} is terminal by design")

    assert policy.deadline_s is not None, (
        f"{phase.value} is non-terminal and carries no deadline, so a servo "
        f"that holds it holds it forever"
    )
    assert policy.deadline_s > 0.0, f"{phase.value} has a non-positive deadline"
    assert policy.on_expiry is not ExpiryAction.NONE, (
        f"{phase.value} expires into ExpiryAction.NONE -- the deadline fires "
        f"and nothing happens, which is an absorbing state wearing a timer"
    )


def test_the_unknown_phase_fallback_is_itself_not_absorbing() -> None:
    """A phase string nobody in this repo has ever heard of must still stop."""

    assert UNKNOWN_PHASE_POLICY.is_terminal is False
    assert UNKNOWN_PHASE_POLICY.deadline_s is not None
    assert UNKNOWN_PHASE_POLICY.on_expiry is not ExpiryAction.NONE


# --------------------------------------------------------------------------
# 2. Over the vocabulary the servo can actually emit.
# --------------------------------------------------------------------------


def test_every_phase_the_servo_can_emit_has_a_policy_row() -> None:
    """Derive the emitted vocabulary from the servo's own source.

    ``ServoPhase.<MEMBER>.value`` is the only spelling the servo is permitted
    to use (pinned by the AST guard in ``test_servo_phase_table.py``), so
    walking the AST for that attribute chain enumerates every phase the process
    can publish.
    """

    tree = ast.parse(SERVO_SOURCE.read_text())
    emitted: set[str] = set()
    for node in ast.walk(tree):
        # Match `ServoPhase.<NAME>.value`
        if not isinstance(node, ast.Attribute) or node.attr != "value":
            continue
        inner = node.value
        if not isinstance(inner, ast.Attribute):
            continue
        if not isinstance(inner.value, ast.Name) or inner.value.id != "ServoPhase":
            continue
        emitted.add(inner.attr)

    assert emitted, "found no ServoPhase references in the servo; the scan is broken"

    missing = sorted(name for name in emitted if not hasattr(ServoPhase, name))
    assert not missing, f"servo emits ServoPhase members that do not exist: {missing}"

    for name in sorted(emitted):
        member = getattr(ServoPhase, name)
        assert member in PHASE_POLICY, (
            f"servo can emit {member.value!r} and the policy table has no row "
            f"for it, so the supervisor would never bound it"
        )


# --------------------------------------------------------------------------
# 3. Dynamically, against the real watchdog and the deployed input set.
# --------------------------------------------------------------------------


def _parked_document(phase: object, *, stamp_ns: int, omit_phase: bool = False):
    """A status document from a servo that is ALIVE and doing nothing.

    Base published at exactly (0, 0), tracking green, heartbeat fresh -- the
    recorded shape of SYMPTOM A.  This is the input under which an absorbing
    phase parks the robot indefinitely with every health signal green.
    """

    document = {
        "schema": "z_manip.depth_servo_status.v1",
        "mode": "live",
        "tracking": True,
        "updated_unix_ns": stamp_ns,
        "output": {
            "published_linear_x": 0.0,
            "published_angular_z": 0.0,
            "proposed_linear_x": 0.0,
            "proposed_angular_z": 0.0,
        },
    }
    if not omit_phase:
        document["phase"] = phase
    return document


def _drive_until_stop(make_document, *, horizon_s: float = 300.0):
    """Run the real watchdog; return (stopped, elapsed_s, decision)."""

    supervision = _supervision_module()
    watchdog = supervision.ReactivePhaseWatchdog()
    watchdog.reset()

    base_unix_ns = 1_785_216_710_000_000_000
    elapsed = 0.0
    decision = None
    while elapsed <= horizon_s:
        stamp_ns = base_unix_ns + int(elapsed * 1e9)
        decision = watchdog.observe(
            make_document(stamp_ns), now_s=elapsed, now_unix_ns=stamp_ns
        )
        if decision.timed_out:
            return True, elapsed, decision
        elapsed += 0.05
    return False, elapsed, decision


NON_TERMINAL = [p for p in ServoPhase if not PHASE_POLICY[p].is_terminal]


@pytest.mark.parametrize("phase", sorted(NON_TERMINAL, key=lambda p: p.value))
def test_a_parked_servo_is_stopped_in_every_non_terminal_phase(
    phase: ServoPhase,
) -> None:
    """Hold one non-terminal phase forever with the base at (0, 0)."""

    stopped, elapsed, decision = _drive_until_stop(
        lambda ns: _parked_document(phase.value, stamp_ns=ns)
    )
    assert stopped, (
        f"the base published (0, 0) with tracking green in {phase.value!r} for "
        f"{elapsed:.0f}s and the supervisor never stopped it"
    )
    # Whichever bound governs THIS phase: the cross-phase no-progress budget,
    # or the phase's own deadline when it is exempt from that budget
    # (``starting`` is, deliberately -- a zero command while the process spawns
    # is the correct output, and its own 30 s row is the backstop).
    policy = PHASE_POLICY[phase]
    governing_s = policy.deadline_s or 0.0
    if policy.counts_no_progress:
        governing_s = min(governing_s, NO_PROGRESS_DEADLINE_S) or governing_s
    assert elapsed <= governing_s + 1.0, (
        f"{phase.value} was only stopped after {elapsed:.2f}s, past its own "
        f"governing bound of {governing_s}s"
    )
    assert decision.code, f"{phase.value} timed out with no diagnosis code"


def test_a_phase_cycle_over_the_whole_vocabulary_is_still_bounded() -> None:
    """The flap defence, generalised.

    A per-phase deadline is defeated by rotating phases.  Rotate through EVERY
    non-terminal phase, one per tick, with the base parked -- the cross-phase
    no-progress budget is the only thing that can stop this, which is exactly
    why it exists.
    """

    order = sorted(NON_TERMINAL, key=lambda p: p.value)
    counter = {"i": 0}

    def make(stamp_ns: int):
        phase = order[counter["i"] % len(order)]
        counter["i"] += 1
        return _parked_document(phase.value, stamp_ns=stamp_ns)

    stopped, elapsed, decision = _drive_until_stop(make)
    assert stopped, (
        f"rotating through all {len(order)} non-terminal phases parked the "
        f"robot for {elapsed:.0f}s without a single deadline firing"
    )
    assert decision.code == "BASE_PROGRESS_STALL_TIMEOUT", (
        "the rotation was stopped by a per-phase deadline; this test is "
        "supposed to exercise the CROSS-phase budget"
    )

    # Exact expected wall time, not a loose ceiling. Phases with
    # ``counts_no_progress=False`` PAUSE the accumulator rather than clearing
    # it (clearing on a phase change is precisely the hole stage 1 repaired),
    # so a rotation containing them stretches the 20 s budget by the ratio of
    # rotated phases to counting ones -- and no further.
    counting = [p for p in order if PHASE_POLICY[p].counts_no_progress]
    expected_s = NO_PROGRESS_DEADLINE_S * len(order) / len(counting)
    assert elapsed <= expected_s + 0.1, (
        f"stopped at {elapsed:.2f}s, later than the {expected_s:.2f}s a "
        f"paused-not-cleared accumulator can account for -- a phase change is "
        f"buying time it should not"
    )


def test_an_unknown_phase_string_is_bounded_not_exempt() -> None:
    stopped, elapsed, decision = _drive_until_stop(
        lambda ns: _parked_document("a_phase_from_a_future_release", stamp_ns=ns)
    )
    assert stopped, "an unrecognized phase string ran forever"
    assert decision.code == "UNKNOWN_PHASE_TIMEOUT"
    assert elapsed <= (UNKNOWN_PHASE_POLICY.deadline_s or 0.0) + 1.0


@pytest.mark.parametrize(
    "label,make_document",
    [
        ("empty string", lambda ns: _parked_document("", stamp_ns=ns)),
        ("null", lambda ns: _parked_document(None, stamp_ns=ns)),
        ("key absent", lambda ns: _parked_document(None, stamp_ns=ns, omit_phase=True)),
    ],
)
def test_a_document_without_a_phase_is_not_laundered_into_a_terminal(
    label: str, make_document
) -> None:
    """The last absorbing state, and the audit's failure shape exactly.

    ``observe`` used to resolve a missing/blank phase to the literal
    ``"idle"``.  ``idle`` is TERMINAL, heartbeat-exempt and carries no
    deadline, so a readable, schema-valid, heartbeat-fresh status document that
    simply omits ``phase`` parked the robot forever with ``timed_out`` False
    and ``phase_known`` True -- it read as *checked and fine*.

    The shipped servo always writes a real phase, so this is not reachable from
    ``go2w_depth_servo.py`` today.  It is reachable from any other writer of
    that path: an older container image, a partially-written document from a
    non-atomic writer, or a future producer.  "No phase information" is not
    evidence that there is nothing to wait for.
    """

    stopped, elapsed, decision = _drive_until_stop(make_document)
    assert stopped, (
        f"a status document whose phase is {label} parked the robot for "
        f"{elapsed:.0f}s: phase={decision.phase!r} deadline={decision.deadline_s} "
        f"terminal={decision.is_terminal} -- absorbing"
    )
    assert decision.is_terminal is False, (
        f"a {label} phase was reported terminal; the run is not over, the "
        f"servo simply said nothing"
    )
    assert elapsed <= (UNKNOWN_PHASE_POLICY.deadline_s or 0.0) + 1.0


def test_an_explicit_idle_report_is_still_treated_as_terminal() -> None:
    """The narrowing above must not change what an explicit ``idle`` means.

    Fake servos in ``test_go2w_planning_control.py`` and
    ``test_staged_pick_hold_contract.py`` report ``"phase": "idle"`` to mean
    "no approach is running", and the supervise loop exits that case on its own
    ``active``/``poll()`` checks rather than on a watchdog deadline.  Only the
    ABSENCE of a phase changed meaning.
    """

    assert phase_policy("idle").is_terminal is True
    assert phase_policy("idle").deadline_s is None
