"""Stage 1 of the state-machine hardening: the phase table is the foundation.

Every test here is regression coverage for a CONFIRMED defect found by
replaying the recorded live corpus, not a shape assertion:

R7  Only three phase strings ever carried a deadline, and the deployed servo
    emits none of them.  The recorded 23.1 s zero-command stall could not be
    timed out, and ``ownership_snapshot`` reported ``base: visual_servo`` on
    rows publishing exactly (0, 0).
R11 ``ReactivePhase.REACQUIRE`` emitted "reacquire" while every consumer
    spelled "reacquiring"; the vocabulary was hand-written six times.
R10 The handoff latch was set by one tick and had no reset and no expiry.
R7b The per-phase deadline added by R7 was itself defeated by phase
    FLAPPING: ``_phase_started_s`` was rearmed by any phase-group change, and
    the recorded corpus flaps between two phases that both publish (0, 0) at
    0.05-0.75 s intervals.  A single injected ``transform_unavailable`` tick
    made the recorded 23.1 s stall stop timing out again.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from z_manip.control.reactive_servo import ReactivePhase
from z_manip.control.servo_phase import (
    EAGER_RECOVERY_PHASES,
    LOSS_STAIR_PHASES,
    MIN_EFFECTIVE_BASE_COMMAND,
    NO_PROGRESS_DEADLINE_S,
    PHASE_POLICY,
    SHIPPED_TRACKING_LOSS_GRACE_S,
    BaseOwner,
    ExpiryAction,
    GUARDED_PHASE_LITERALS,
    PhasePolicy,
    ServoPhase,
    phase_policy,
    view_recovery_deadline_s,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "runtime"))
SUPERVISION_PATH = ROOT / "scripts" / "runtime" / "go2w_reactive_supervision.py"
SERVO_PATH = ROOT / "scripts" / "runtime" / "go2w_depth_servo.py"
FIXTURE = ROOT / "tests" / "fixtures" / "go2w_whole_body_zero_stall.jsonl"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPERVISION = _load("go2w_reactive_supervision", SUPERVISION_PATH)


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# R7 -- every phase carries a deadline, and the table is complete.
# ---------------------------------------------------------------------------


def test_every_phase_enum_member_has_a_policy_row():
    """FAILS the moment ServoPhase gains a member the table lacks.

    ``z_manip.control.servo_phase`` also raises at import time on a missing
    row, so this cannot be bypassed by not importing the table.
    """

    missing = sorted(p.value for p in ServoPhase if p not in PHASE_POLICY)
    assert missing == [], f"phases with no bounded-wait contract: {missing}"

    extra = sorted(
        str(p) for p in PHASE_POLICY if not isinstance(p, ServoPhase)
    )
    assert extra == [], f"policy rows for non-members: {extra}"


def test_only_terminal_phases_may_be_exempt_from_a_deadline():
    """The inversion: deadlines are the default, exemptions are enumerated."""

    for phase, policy in PHASE_POLICY.items():
        if policy.deadline_s is None:
            assert policy.is_terminal, (
                f"{phase.value} has no deadline but is not terminal; "
                "an unbounded non-terminal phase is exactly the defect "
                "that let a 23.1 s zero-command stall go untimed"
            )
            assert policy.on_expiry is ExpiryAction.NONE
        else:
            assert policy.deadline_s >= 0.0
            # An expiry action must STOP or ESCALATE.  There is no
            # log-and-continue member, by construction of the enum.
            assert policy.on_expiry in {
                ExpiryAction.STOP_AND_DEGRADE,
                ExpiryAction.ESCALATE_VIEW_RECOVERY,
            }, f"{phase.value} expiry is not a real action"


def test_every_phase_row_carries_the_full_contract():
    for phase, policy in PHASE_POLICY.items():
        assert isinstance(policy.heartbeat_required, bool)
        assert isinstance(policy.expected_base_owner, BaseOwner)
        assert isinstance(policy.is_terminal, bool)


def test_unknown_phase_is_bounded_and_fails_closed():
    """An unlisted phase string used to be exempt forever."""

    policy = phase_policy("some_phase_nobody_declared")
    assert policy.deadline_s is not None and policy.deadline_s > 0.0
    assert policy.on_expiry is ExpiryAction.STOP_AND_DEGRADE
    assert policy.heartbeat_required is True
    assert policy.is_terminal is False


def test_every_phase_in_the_recorded_corpus_carries_a_deadline():
    """Replay the live fixture: no observed phase may be unbounded.

    Pre-change, ``whole_body_shadow``/``whole_body_approach``/
    ``tracking_hold``/``transform_unavailable`` all resolved to
    ``deadline_s = None``.
    """

    observed = {str(row.get("phase")) for row in _fixture_rows()}
    assert observed, "fixture is empty"
    unbounded = sorted(
        phase
        for phase in observed
        if phase_policy(phase).deadline_s is None
        and not phase_policy(phase).is_terminal
    )
    assert unbounded == [], f"recorded phases with no deadline: {unbounded}"


# ---------------------------------------------------------------------------
# R7 -- the recorded 23.1 s zero-command stall must now time out.
# ---------------------------------------------------------------------------


def test_watchdog_bounds_the_recorded_23s_zero_command_stall():
    """The live SYMPTOM A trace: tracking green, base parked, no timeout.

    Fixture provenance: ``artifacts/go2w_real/latest/depth-servo.trace.jsonl``
    (+ .1), the longest contiguous ``whole_body_shadow`` span with
    ``published_linear_x == published_angular_z == 0.0``.  101 samples,
    23.101 s, ``mode: live``, ``tracking: true``, whole-body backend
    ``fixed-fixture-collision-gate`` reporting ``success: false``.  Three
    preceding rows are kept so the timer is exercised across a genuinely
    commanding ``whole_body_approach`` tick first.

    Pre-change the watchdog reported ``deadline_s = None`` for every one of
    these rows and never timed out.
    """

    rows = _fixture_rows()
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(
            # The recorded trace samples at ~4-5 Hz, far slower than the
            # 20 Hz status document, so the state-heartbeat deadline is not
            # the mechanism under test here.
            state_heartbeat_timeout_s=60.0,
        )
    )
    base_ns = rows[0]["updated_unix_ns"]
    fired = None
    for row in rows:
        now_s = (row["updated_unix_ns"] - base_ns) / 1e9
        decision = watchdog.observe(
            row,
            now_s=now_s,
            now_unix_ns=row["updated_unix_ns"] + 20_000_000,
        )
        if decision.timed_out:
            fired = (now_s, decision)
            break

    assert fired is not None, "the recorded zero-command stall never timed out"
    elapsed_s, decision = fired
    assert decision.code == "BASE_COMMAND_STALL_TIMEOUT"
    assert decision.phase == ServoPhase.WHOLE_BODY_SHADOW.value
    assert decision.on_expiry == ExpiryAction.STOP_AND_DEGRADE.value
    assert decision.base_owner_mismatch is True
    assert decision.owners["base"] == BaseOwner.ZERO_HOLD.value
    assert decision.expected_base_owner == BaseOwner.VISUAL_SERVO.value
    # It must fire well inside the recorded 23.1 s hold, not at the very end.
    assert 0.0 < elapsed_s <= 16.0
    assert decision.deadline_s is not None


def test_replay_trace_flags_the_recorded_zero_command_stall():
    """``replay_trace`` used to only look inside POSTURE_WAIT_PHASES."""

    report = SUPERVISION.replay_trace(_fixture_rows(), stall_threshold_s=5.0)

    assert report["passed"] is False, "replay declared the live stall healthy"
    codes = {stall["code"] for stall in report["stalls"]}
    assert "BASE_COMMAND_STALL" in codes
    stall = next(s for s in report["stalls"] if s["code"] == "BASE_COMMAND_STALL")
    assert stall["phase"] == ServoPhase.WHOLE_BODY_SHADOW.value
    assert stall["duration_s"] == pytest.approx(23.101, abs=0.01)
    assert stall["on_expiry"] == ExpiryAction.STOP_AND_DEGRADE.value


def test_a_commanding_driving_phase_is_not_timed_out():
    """The deadline must not kill a healthy, actually-moving approach.

    A driving phase measures a zero-command span, not time in phase, so a
    servo that keeps issuing commands survives arbitrarily long.
    """

    watchdog = SUPERVISION.ReactivePhaseWatchdog()
    for index in range(2000):  # 100 s at 20 Hz
        decision = watchdog.observe(
            {
                "phase": ServoPhase.WHOLE_BODY_APPROACH.value,
                "mode": "live",
                "updated_unix_ns": 10_000_000_000 + index * 50_000_000,
                "output": {
                    "published_linear_x": 0.12,
                    "published_angular_z": 0.0,
                },
            },
            now_s=index * 0.05,
            now_unix_ns=10_010_000_000 + index * 50_000_000,
        )
        assert decision.timed_out is False, f"healthy approach killed at {index}"
    assert decision.base_owner_mismatch is False


def test_shadow_mode_is_judged_on_the_proposed_command():
    """A shadow servo publishes zero by construction.

    ``published = value if live else 0.0``, so judging shadow mode on the
    published command would report every healthy shadow run as a stalled
    base and degrade the operator's diagnostic mode.
    """

    watchdog = SUPERVISION.ReactivePhaseWatchdog()
    for index in range(600):  # 30 s at 20 Hz, past the 12 s deadline
        decision = watchdog.observe(
            {
                "phase": ServoPhase.APPROACH.value,
                "mode": "shadow",
                "updated_unix_ns": 10_000_000_000 + index * 50_000_000,
                "output": {
                    "proposed_linear_x": 0.15,
                    "proposed_angular_z": 0.0,
                    "published_linear_x": 0.0,
                    "published_angular_z": 0.0,
                },
            },
            now_s=index * 0.05,
            now_unix_ns=10_010_000_000 + index * 50_000_000,
        )
        assert decision.timed_out is False
    # ...but a shadow servo that stops PROPOSING anything is still stalled.
    stalled = SUPERVISION.ReactivePhaseWatchdog()
    for index in range(600):
        decision = stalled.observe(
            {
                "phase": ServoPhase.APPROACH.value,
                "mode": "shadow",
                "updated_unix_ns": 10_000_000_000 + index * 50_000_000,
                "output": {
                    "proposed_linear_x": 0.0,
                    "proposed_angular_z": 0.0,
                    "published_linear_x": 0.0,
                    "published_angular_z": 0.0,
                },
            },
            now_s=index * 0.05,
            now_unix_ns=10_010_000_000 + index * 50_000_000,
        )
        if decision.timed_out:
            break
    assert decision.timed_out is True
    assert decision.code == "BASE_COMMAND_STALL_TIMEOUT"


# ---------------------------------------------------------------------------
# R7 -- base ownership must be measured, never inferred from the phase name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase",
    (
        ServoPhase.APPROACH.value,
        ServoPhase.BASE_APPROACH.value,
        ServoPhase.WHOLE_BODY_APPROACH.value,
    ),
)
def test_base_owner_is_zero_hold_when_the_published_command_is_zero(phase):
    """The phase term used to short-circuit the ownership snapshot.

    ``whole_body_approach`` is emitted whenever the QP command is executable,
    and on that same tick the forward clip can return 0.0 while the yaw
    deadband zeroes the turn -- a parked robot reported as ``visual_servo``.
    """

    owners = SUPERVISION.ownership_snapshot({
        "phase": phase,
        "mode": "live",
        "output": {"published_linear_x": 0.0, "published_angular_z": 0.0},
    })

    assert owners["base"] == BaseOwner.ZERO_HOLD.value, (
        f"{phase} claimed the base while publishing exactly (0, 0)"
    )


def test_base_owner_is_visual_servo_only_on_a_real_command():
    owners = SUPERVISION.ownership_snapshot({
        "phase": ServoPhase.WHOLE_BODY_SHADOW.value,
        "mode": "live",
        "output": {"published_linear_x": 0.0, "published_angular_z": -0.2},
    })

    assert owners["base"] == BaseOwner.VISUAL_SERVO.value


# ---------------------------------------------------------------------------
# R11 -- one vocabulary.
# ---------------------------------------------------------------------------


def test_every_reactive_phase_value_is_a_servo_phase_value():
    """``ReactivePhase.REACQUIRE`` emitted a string no consumer spelled."""

    servo_values = {phase.value for phase in ServoPhase}
    orphans = sorted(
        phase.value for phase in ReactivePhase if phase.value not in servo_values
    )
    assert orphans == [], (
        f"ReactivePhase emits strings the phase table does not know: {orphans}"
    )


def test_the_emitted_reacquire_string_is_the_one_consumers_match():
    """The concrete R11 defect, pinned.

    The servo's loss stair, the whole-body branch's bail-out and the legacy
    branch all spelled "reacquiring"; the controller emitted "reacquire".
    """

    from z_manip.control.servo_phase import LOSS_STAIR_PHASES

    # The defect, stated as the property that matters: the string the
    # controller EMITS must be one a consumer MATCHES.
    assert ReactivePhase.REACQUIRE.value in LOSS_STAIR_PHASES, (
        f"{ReactivePhase.REACQUIRE.value!r} is emitted but no loss-stair "
        f"consumer matches it; consumers spell {sorted(LOSS_STAIR_PHASES)}"
    )
    assert ReactivePhase.REACQUIRE.value == "reacquiring"


def test_the_six_phase_sets_are_derived_from_one_table():
    """No consumer may hand-write a phase membership any more."""

    from z_manip.control import servo_phase as table

    assert table.HANDOFF_PHASES == table.HANDOFF_TERMINAL_PHASES
    assert table.HANDOFF_PHASES == frozenset({
        ServoPhase.REACHED.value,
        ServoPhase.HANDOFF_PROBE.value,
        ServoPhase.HANDOFF_READY.value,
    })
    # The supervisor re-exports the derived sets rather than redefining them.
    assert SUPERVISION.HANDOFF_PHASES is table.HANDOFF_PHASES
    assert SUPERVISION.POSTURE_WAIT_PHASES is table.POSTURE_WAIT_PHASES
    assert SUPERVISION.HEARTBEAT_EXEMPT_PHASES is table.HEARTBEAT_EXEMPT_PHASES
    # The heartbeat exemption list is unchanged from the one it replaced.
    assert table.HEARTBEAT_EXEMPT_PHASES == frozenset({
        ServoPhase.IDLE.value,
        ServoPhase.STOPPED.value,
        ServoPhase.BLOCKED.value,
        ServoPhase.DEGRADED.value,
        ServoPhase.GRASP_STARTED.value,
    })


#: Files that must reference phases through ``ServoPhase``, never by quoting.
_GUARDED_RUNTIME_FILES = (
    SERVO_PATH,
    SUPERVISION_PATH,
)


def _quoted_phase_literals(path: Path) -> list[str]:
    """Return every executable string literal that is a phase name.

    Docstrings and bare string expression statements (the prose that has to
    NAME these strings to explain them) are excluded by walking the AST
    rather than the text, so a comment can quote "reacquire" while code
    cannot.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    exempt: set[int] = set()
    for node in ast.walk(tree):
        # A phase name used as a DICT KEY is a status-document field name,
        # not a phase value (e.g. ``{"handoff_ready": decision.handoff_ready}``).
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    exempt.add(id(key))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for statement in body:
            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant
            ) and isinstance(statement.value.value, str):
                exempt.add(id(statement.value))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in GUARDED_PHASE_LITERALS
            and id(node) not in exempt
        ):
            offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")
    return offenders


def test_runtime_scripts_quote_no_phase_names():
    """Grep guard: a quoted phase name in a runtime script is a new fork.

    Every one of the six divergent phase memberships started as a quoted
    string in exactly these two files.  A quoted phase name is how
    ``"reacquire"``/``"reacquiring"`` drifted apart with the suite green.
    """

    offenders: list[str] = []
    for path in _GUARDED_RUNTIME_FILES:
        offenders.extend(_quoted_phase_literals(path))
    assert offenders == [], (
        "runtime scripts must use ServoPhase.<MEMBER>.value, not a quoted "
        "phase name:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# R10 -- the one-way handoff latch has a bounded, stop-only exit.
# ---------------------------------------------------------------------------


def test_handoff_latch_is_held_until_its_bound_then_abandoned():
    """The latch had no reset edge and no expiry at all.

    Set by a single tick, it made ``_tick`` return early forever while the
    status document kept advertising ``needs_ik_probe`` -- a permanent,
    unbounded request for a transaction nobody was going to open.
    """

    servo = _load("go2w_depth_servo_latch_probe", SERVO_PATH)

    latched = servo.DepthServoOutput(
        phase=ServoPhase.HANDOFF_PROBE.value,
        proposed_linear_x=0.0,
        proposed_angular_z=0.0,
        published_linear_x=0.0,
        published_angular_z=0.0,
        depth_error_m=0.1,
        yaw_error_rad=0.0,
        target_age_s=0.1,
        done=False,
        reason="corridor reached",
        reactive_phase=ServoPhase.HANDOFF_PROBE.value,
        needs_ik_probe=True,
    )

    # Inside the bound the latch is preserved byte for byte: the supervisor
    # must still see the handoff request.
    held = servo._handoff_latch_output(
        latched, latched_since_s=100.0, now_s=100.0 + 19.0, timeout_s=20.0
    )
    assert held is latched

    expired = servo._handoff_latch_output(
        latched, latched_since_s=100.0, now_s=100.0 + 20.5, timeout_s=20.0
    )
    assert expired is not latched
    assert expired.phase == ServoPhase.HANDOFF_ABANDONED.value
    # The base never resumes: the exit is always a stop.
    assert expired.published_linear_x == 0.0
    assert expired.published_angular_z == 0.0
    assert expired.proposed_linear_x == 0.0
    assert expired.proposed_angular_z == 0.0
    # ...and it must stop reading as a handoff REQUEST, or the supervisor
    # would open a grasp transaction instead of degrading.
    assert expired.needs_ik_probe is False
    assert expired.reactive_phase is None
    # Idempotent: an already-abandoned latch is not re-abandoned.
    assert servo._handoff_latch_output(
        expired, latched_since_s=100.0, now_s=1_000.0, timeout_s=20.0
    ) is expired


def test_abandoned_handoff_is_not_a_handoff_request_and_is_bounded():
    """The abandoned latch must escalate, not park."""

    planning = _load(
        "go2w_planning_control_probe",
        ROOT / "scripts" / "runtime" / "go2w_planning_control.py",
    )
    runtime = {
        "phase": ServoPhase.HANDOFF_ABANDONED.value,
        "mode": "live",
        "output": {
            "phase": ServoPhase.HANDOFF_ABANDONED.value,
            "needs_ik_probe": False,
            "reactive_phase": None,
            "published_linear_x": 0.0,
            "published_angular_z": 0.0,
        },
    }

    assert planning.DepthServoRunner._runtime_requests_handoff(runtime) is False

    policy = phase_policy(ServoPhase.HANDOFF_ABANDONED)
    assert policy.deadline_s == 0.0
    assert policy.on_expiry is ExpiryAction.STOP_AND_DEGRADE

    watchdog = SUPERVISION.ReactivePhaseWatchdog()
    runtime["updated_unix_ns"] = 10_000_000_000
    decision = watchdog.observe(runtime, now_s=1.0, now_unix_ns=10_050_000_000)
    assert decision.timed_out is True
    assert decision.on_expiry == ExpiryAction.STOP_AND_DEGRADE.value


def test_ik_probe_still_has_no_producer_in_this_repository():
    """R10's decision, pinned so it cannot silently change.

    HANDOFF_READY is kept (it is the fail-closed upgrade path) and the
    deployed terminal is HANDOFF_PROBE.  If a producer is ever added, this
    test fails and whoever added it must re-read the contract recorded at
    ``IK_PROBE_SCHEMA`` in go2w_depth_servo.py.
    """

    # Only TRACKED files. An rglob walk descends into .claude/worktrees/, where
    # this repo keeps ~20 git worktrees, and reports every one of their copies of
    # go2w_depth_servo.py -- a failure that says nothing about a producer
    # existing. It is also invisible from inside a worktree, where ROOT has no
    # nested worktrees, so the check passes there and fails only after a merge.
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", "replace").split("\0")

    hits: list[str] = []
    for name in tracked:
        if not name:
            continue
        path = ROOT / name
        if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".json", ".html"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if "reactive/ik_probe" in text:
            hits.append(name)

    assert sorted(hits) == [
        "scripts/runtime/go2w_depth_servo.py",
        "tests/test_servo_phase_table.py",
    ], f"unexpected ik_probe references: {sorted(hits)}"


# ---------------------------------------------------------------------------
# R7b -- the deadline must survive phase FLAPPING.
#
# The per-phase deadline is measured from the phase transition, so it is
# rearmed by ANY phase change -- including a change between two phases that
# both publish (0, 0).  Every test below drives a base that never moves and
# asserts the watchdog still stops it.  Each one passes with a single
# contiguous phase and FAILS the moment the phase string alternates, which is
# the defect: the value that decided the transition (phase-string identity)
# was not the value the guard claimed to measure (a zero-command span).
# ---------------------------------------------------------------------------


def _parked_row(phase: str, index: int, *, linear: float = 0.0) -> dict:
    """One status document: mode live, tracking green, base at ``linear``."""

    return {
        "phase": phase,
        "mode": "live",
        "tracking": True,
        "updated_unix_ns": 10_000_000_000 + index * 50_000_000,
        "output": {
            "published_linear_x": linear,
            "published_angular_z": 0.0,
            "proposed_linear_x": linear,
            "proposed_angular_z": 0.0,
        },
    }


def _first_timeout(phase_at, *, seconds: float = 90.0, hz: float = 20.0):
    """Drive the watchdog with a parked base and return the first timeout."""

    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        # The heartbeat advances every tick in these sequences; it is not the
        # mechanism under test, and leaving it at 1.5 s would let a heartbeat
        # timeout mask whether the PHASE bound works.
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=600.0)
    )
    last = None
    for index in range(int(seconds * hz)):
        now_s = index / hz
        row = _parked_row(phase_at(index), index)
        last = watchdog.observe(
            row, now_s=now_s, now_unix_ns=row["updated_unix_ns"] + 10_000_000
        )
        if last.timed_out:
            return now_s, last
    return None, last


#: Every pair below is two phases the deployed servo can emit back to back
#: while publishing exactly (0, 0).
_FLAP_SEQUENCES = {
    # ``whole_body_approach`` and ``whole_body_shadow`` are the two arms of
    # ONE ternary on ``command.executable`` (go2w_depth_servo.py
    # ``_whole_body_output``), and the recorded live trace flips between them
    # 15 times with inter-flip intervals of 0.05-0.75 s.  The approach arm can
    # publish (0, 0) too: the forward clip returns 0.0 while the yaw deadband
    # zeroes the turn -- the same reachability the ownership fix rests on.
    "whole_body_executable_gate": (
        lambda i: ServoPhase.WHOLE_BODY_SHADOW.value
        if i % 2
        else ServoPhase.WHOLE_BODY_APPROACH.value
    ),
    # The handoff corridor: ``_whole_body_output`` emits ``whole_body_posture``
    # if executable else ``whole_body_shadow``, both parked BY DESIGN.  They
    # sit in different groups (posture wait vs driving), so the flip rearmed
    # the watchdog AND reset ``whole_body_handoff_settle_cycles`` -- an
    # unbounded close-range park with tracking green.
    "handoff_corridor_gate": (
        lambda i: ServoPhase.WHOLE_BODY_POSTURE.value
        if (i // 20) % 2
        else ServoPhase.WHOLE_BODY_SHADOW.value
    ),
    # A single ``transform_unavailable`` tick splitting a long run: this
    # happens six times in the recorded corpus.
    "single_tick_transform_hiccup": (
        lambda i: ServoPhase.TRANSFORM_UNAVAILABLE.value
        if i % 180 == 0
        else ServoPhase.WHOLE_BODY_SHADOW.value
    ),
    # The loss stair rotating slowly.  None of these three had an eager
    # handler either, so this was an unbounded park with no diagnosis at all.
    "loss_stair_rotation": (
        lambda i: (
            ServoPhase.TRACKING_HOLD.value,
            ServoPhase.REACQUIRE.value,
            ServoPhase.TRANSFORM_UNAVAILABLE.value,
        )[(i // 100) % 3]
    ),
}


@pytest.mark.parametrize("name", sorted(_FLAP_SEQUENCES))
def test_phase_flapping_cannot_rearm_the_stall_timer(name):
    """A parked base must be bounded however the phase string moves."""

    fired_at, decision = _first_timeout(_FLAP_SEQUENCES[name])

    assert fired_at is not None, (
        f"{name}: the base published (0, 0) for 90s with tracking green and "
        "the watchdog never timed out; phase flapping rearmed every deadline"
    )
    # It must be bounded by the CROSS-PHASE budget, and the diagnosis must say
    # so rather than blaming whichever phase happened to be current.
    assert decision.code == "BASE_PROGRESS_STALL_TIMEOUT"
    assert decision.no_progress_s >= NO_PROGRESS_DEADLINE_S
    assert decision.no_progress_deadline_s == NO_PROGRESS_DEADLINE_S
    assert fired_at <= NO_PROGRESS_DEADLINE_S + 1.0
    # An expiry that does nothing is the defect the table exists to remove.
    assert decision.on_expiry in {
        ExpiryAction.STOP_AND_DEGRADE.value,
        ExpiryAction.ESCALATE_VIEW_RECOVERY.value,
    }


def test_one_injected_hiccup_tick_cannot_hide_the_recorded_23s_stall():
    """The regression, driven from the real fixture.

    ``test_watchdog_bounds_the_recorded_23s_zero_command_stall`` passes on the
    contiguous recording.  Change ONE row's phase in the middle of that stall
    -- exactly what a single ``transform_unavailable`` tick does, six times in
    the recorded corpus -- and the per-phase deadline is rearmed and never
    expires again.  The recording is 23.1 s of a live, parked, tracking-green
    base, so it must still be caught.
    """

    rows = _fixture_rows()
    stall_rows = [
        index
        for index, row in enumerate(rows)
        if row.get("phase") == ServoPhase.WHOLE_BODY_SHADOW.value
    ]
    assert len(stall_rows) > 50, "fixture no longer contains the recorded stall"
    injected = rows[stall_rows[len(stall_rows) // 2]]
    injected["phase"] = ServoPhase.TRANSFORM_UNAVAILABLE.value

    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=60.0)
    )
    base_ns = rows[0]["updated_unix_ns"]
    fired = None
    for row in rows:
        now_s = (row["updated_unix_ns"] - base_ns) / 1e9
        decision = watchdog.observe(
            row, now_s=now_s, now_unix_ns=row["updated_unix_ns"] + 20_000_000
        )
        if decision.timed_out:
            fired = (now_s, decision)
            break

    assert fired is not None, (
        "one injected phase tick made the recorded 23.1s zero-command stall "
        "stop timing out"
    )
    elapsed_s, decision = fired
    assert elapsed_s <= 23.0, "the stall must be caught inside the recording"
    assert decision.no_progress_s >= NO_PROGRESS_DEADLINE_S


def test_replay_trace_flags_a_flapping_zero_command_run():
    """A green replay must transfer to the deployed guard.

    ``replay_trace``'s per-span detector groups by phase STRING, so a flapping
    trace splits into short spans and reads healthy -- while the same input
    parks the robot.  The cross-phase detector mirrors the live bound exactly,
    so a replay verdict now means what it says.
    """

    rows = [
        _parked_row(
            ServoPhase.WHOLE_BODY_SHADOW.value
            if index % 2
            else ServoPhase.WHOLE_BODY_APPROACH.value,
            index,
        )
        for index in range(int(40 * 20))  # 40 s at 20 Hz
    ]

    report = SUPERVISION.replay_trace(rows, stall_threshold_s=5.0)

    assert report["passed"] is False, (
        "replay declared a 40s parked, flapping base healthy"
    )
    assert report["no_progress_stalls"], "no cross-phase stall was reported"
    stall = report["no_progress_stalls"][0]
    assert stall["code"] == "BASE_PROGRESS_STALL"
    assert stall["duration_s"] >= NO_PROGRESS_DEADLINE_S
    assert set(stall["phases"]) == {
        ServoPhase.WHOLE_BODY_APPROACH.value,
        ServoPhase.WHOLE_BODY_SHADOW.value,
    }
    # The per-span detector alone is blind to this input: every span is one
    # 0.05 s sample.  That is the whole point of the second detector.
    assert report["stalls"] == []


def test_a_terminal_row_does_not_launder_a_zero_command_run():
    """A non-counting phase PAUSES the budget; it must not clear it.

    Clearing on a phase change is the defect.  If an interleaved terminal (or
    ``starting``) row reset the accumulator, the same flap that this fix
    closes would reopen through the exempt rows instead.
    """

    def phase_at(index: int) -> str:
        if index % 200 == 199:
            return ServoPhase.STARTING.value
        return ServoPhase.WHOLE_BODY_SHADOW.value

    fired_at, decision = _first_timeout(phase_at)

    assert fired_at is not None, (
        "an interleaved non-counting phase laundered a continuous "
        "zero-command run into short ones"
    )
    assert decision.no_progress_s >= NO_PROGRESS_DEADLINE_S


def test_starting_does_not_spend_the_cross_phase_budget():
    """...but the pause must be real, or a slow spawn is killed.

    ``starting`` carries its own 30 s spawn backstop and publishes zero
    because that is the CORRECT output while the container, rclpy and the
    first TF read come up.  If it accumulated the 20 s cross-phase budget it
    would be stopped at 20 s by a bound meant for a stalled approach.
    """

    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=600.0)
    )
    decision = None
    for index in range(int(25 * 20)):  # 25 s, inside the 30 s spawn backstop
        row = _parked_row(ServoPhase.STARTING.value, index)
        decision = watchdog.observe(
            row, now_s=index / 20.0, now_unix_ns=row["updated_unix_ns"] + 10_000_000
        )
        assert decision.timed_out is False, f"spawn killed at {index / 20.0:.1f}s"
    assert decision.no_progress_s == 0.0
    assert decision.no_progress_deadline_s is None


def test_the_cross_phase_bound_is_never_tighter_than_a_phase_deadline():
    """The invariant that makes the new bound safe to add.

    If some row legitimately needs a deadline longer than the cross-phase
    budget then that row's own claim is that a parked base is tolerable for
    that long, and the cross-phase bound would cut it short mid-phase.  The
    module raises at import time rather than letting that invert silently.
    """

    for phase, policy in PHASE_POLICY.items():
        if not policy.counts_no_progress or policy.deadline_s is None:
            continue
        assert policy.deadline_s <= NO_PROGRESS_DEADLINE_S, (
            f"{phase.value} tolerates {policy.deadline_s}s of a parked base "
            f"but the cross-phase bound is {NO_PROGRESS_DEADLINE_S}s, so it "
            "would be stopped before its own deadline"
        )
    # A terminal phase is acted on immediately; it must never accumulate.
    for phase, policy in PHASE_POLICY.items():
        if policy.is_terminal:
            assert policy.counts_no_progress is False, phase.value


def test_a_commanding_approach_survives_the_cross_phase_bound():
    """The new bound must not kill a healthy, actually-moving approach.

    A real command clears the accumulator, so an approach that keeps driving
    runs arbitrarily long even while its phase flaps between the two arms of
    the executable gate.
    """

    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=600.0)
    )
    for index in range(int(120 * 20)):  # 120 s, 6x the cross-phase bound
        row = _parked_row(
            ServoPhase.WHOLE_BODY_SHADOW.value
            if index % 2
            else ServoPhase.WHOLE_BODY_APPROACH.value,
            index,
            linear=0.12,
        )
        decision = watchdog.observe(
            row, now_s=index / 20.0, now_unix_ns=row["updated_unix_ns"] + 10_000_000
        )
        assert decision.timed_out is False, f"healthy approach killed at {index}"
    assert decision.no_progress_s == 0.0


def test_a_sub_deadzone_command_is_not_progress():
    """0.001 m/s forever is not driving; it must not rearm the timer.

    The shipped settings snap any non-trivial forward command up to
    ``min_forward_mps = 0.10`` precisely because of the Go2W dead zone, and
    the yaw slew step is 0.015 rad/s, so nothing the servo legitimately holds
    is anywhere near this floor.
    """

    assert MIN_EFFECTIVE_BASE_COMMAND > 0.001

    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=600.0)
    )
    fired = False
    for index in range(int(60 * 20)):
        row = _parked_row(ServoPhase.APPROACH.value, index, linear=0.001)
        decision = watchdog.observe(
            row, now_s=index / 20.0, now_unix_ns=row["updated_unix_ns"] + 10_000_000
        )
        if decision.timed_out:
            fired = True
            break
    assert fired, "a 0.001 m/s command rearmed the stall timer forever"
    # ...and the reported owner must agree with the timer, or the operator
    # reads ``visual_servo`` next to a stall diagnosis.
    assert decision.owners["base"] == BaseOwner.ZERO_HOLD.value

    # A command at the floor IS progress, and is reported as such.
    owners = SUPERVISION.ownership_snapshot(
        _parked_row(
            ServoPhase.APPROACH.value, 0, linear=MIN_EFFECTIVE_BASE_COMMAND
        )
    )
    assert owners["base"] == BaseOwner.VISUAL_SERVO.value


def test_a_counted_recovery_clears_only_the_cross_phase_budget():
    """A stationary recovery is supervisory progress.

    A wrist search moves no wheels, so without this the accumulator is full
    when the servo restarts and expires on the first observation of the
    recovered run -- collapsing the whole (bounded) reacquisition budget into
    a few seconds instead of giving each attempt its own bound.
    """

    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=600.0)
    )
    for index in range(int(15 * 20)):
        row = _parked_row(ServoPhase.TRACKING_LOST.value, index)
        watchdog.observe(
            row, now_s=index / 20.0, now_unix_ns=row["updated_unix_ns"] + 10_000_000
        )
    assert watchdog.last.no_progress_s > 10.0

    watchdog.note_supervisor_progress()
    row = _parked_row(ServoPhase.WAITING_TARGET.value, 300)
    decision = watchdog.observe(
        row, now_s=15.0, now_unix_ns=row["updated_unix_ns"] + 10_000_000
    )
    # One tick's worth of fresh accumulation, not the 15 s that preceded it.
    assert decision.no_progress_s < 0.2
    assert decision.timed_out is False
    # It must NOT be a general amnesty: the heartbeat state is untouched, so a
    # dead servo is still caught.
    assert decision.heartbeat_required is True


# ---------------------------------------------------------------------------
# R10 follow-up -- an abandoned handoff must not still read as a request.
# ---------------------------------------------------------------------------


def test_abandoned_handoff_scrubs_the_frozen_reactive_request():
    """The latch cleared the OUTPUT but not the status document's block.

    During the latch ``_tick`` returns before ``core.tick`` runs, so
    ``reactive_status`` stays frozen at the handoff decision with
    ``needs_ik_probe: True``.  ``_runtime_requests_handoff`` reads that block
    as a fallback, so an abandoned latch still answered "open a close-range
    grasp transaction".  It was safe only because ``_supervise`` evaluates the
    timeout first -- a live-motion guarantee resting on statement order in a
    different file.
    """

    servo = _load("go2w_depth_servo_reactive_scrub", SERVO_PATH)
    planning = _load(
        "go2w_planning_control_scrub",
        ROOT / "scripts" / "runtime" / "go2w_planning_control.py",
    )
    frozen = {
        "phase": ServoPhase.HANDOFF_PROBE.value,
        "needs_ik_probe": True,
        "handoff_ready": False,
        "reason": "close-range corridor reached",
    }

    # Before the latch expires the request must survive untouched.
    held = servo._abandoned_reactive_status(
        dict(frozen), phase=ServoPhase.HANDOFF_PROBE.value
    )
    assert held == frozen

    scrubbed = servo._abandoned_reactive_status(
        dict(frozen), phase=ServoPhase.HANDOFF_ABANDONED.value
    )
    assert scrubbed["needs_ik_probe"] is False
    assert scrubbed["handoff_ready"] is False
    assert scrubbed["phase"] == ServoPhase.HANDOFF_ABANDONED.value

    runtime = {
        "phase": ServoPhase.HANDOFF_ABANDONED.value,
        "mode": "live",
        "output": {
            "phase": ServoPhase.HANDOFF_ABANDONED.value,
            "needs_ik_probe": False,
            "reactive_phase": None,
            "published_linear_x": 0.0,
            "published_angular_z": 0.0,
        },
        "reactive": scrubbed,
    }
    assert planning.DepthServoRunner._runtime_requests_handoff(runtime) is False

    # Guard the claim: with the UNSCRUBBED block the answer is yes, which is
    # what made the safety depend on statement order.
    runtime["reactive"] = frozen
    assert planning.DepthServoRunner._runtime_requests_handoff(runtime) is True


def test_only_no_target_phases_escalate_into_the_wrist_search():
    """Bound the NEW arm motion this stage makes reachable.

    ``ExpiryAction.ESCALATE_VIEW_RECOVERY`` routes into
    ``_recover_view_with_stationary_base``, which COMMANDS WRIST MOTION.
    Before this stage only ``view_recovery``/``search_required`` could reach
    it.  Adding deadlines -- and especially adding a cross-phase bound that
    actually fires -- makes every escalating phase a new trigger for an arm
    sweep that has not been exercised on hardware from that phase.

    So the escalating set is restricted to phases that mean "there is no
    usable target", which is what the bounded wrist search was written for.
    ``tracking_hold`` and ``reacquiring`` are excluded on purpose: both mean
    the target IS in view and the tracker is the thing that is not producing
    (``tracking_hold_s`` defaults to 0.0 and is validated below the 0.75 s
    loss grace, and REACQUIRE is "rebuilding a stable 3-D track after posture
    motion"), so sweeping the camera away from it cannot help.

    NARROWED SINCE, twice, and both directions are the safe one:

    * ``view_recovery`` LEFT the set.  Its deadline now means "the servo did
      not step out of ``view_recovery`` within its own full configured
      ``tracking_loss_grace_s``", i.e. the reactive FSM is not stepping -- and
      a wrist search cannot fix a frozen FSM.  Exactly the argument already
      made for ``tracking_hold``/``reacquiring``.  The happy path is
      unaffected: a healthy servo steps to ``search_required`` by itself at
      the grace, and THAT phase is escalated on sight.
    * ``acquiring`` was never added.  It means the controller has never held a
      3-D target, so there is no last-known viewing ray for a sweep to return
      to; it stops and degrades instead.
    """

    escalating = {
        phase.value
        for phase, policy in PHASE_POLICY.items()
        if policy.on_expiry is ExpiryAction.ESCALATE_VIEW_RECOVERY
    }

    assert escalating == {
        ServoPhase.WAITING_TARGET.value,
        ServoPhase.TRACKING_LOST.value,
        ServoPhase.SEARCH_REQUIRED.value,
    }, (
        "a phase gained the power to start an unexercised wrist sweep; "
        "justify it against the recorded corpus before widening this set"
    )
    for phase in (
        ServoPhase.TRACKING_HOLD,
        ServoPhase.REACQUIRE,
        ServoPhase.VIEW_RECOVERY,
        ServoPhase.ACQUIRING,
    ):
        assert PHASE_POLICY[phase].on_expiry is ExpiryAction.STOP_AND_DEGRADE
    # Every escapee must still have a real action; none may silently park.
    for policy in PHASE_POLICY.values():
        if policy.deadline_s is not None:
            assert policy.on_expiry is not ExpiryAction.NONE


# ---------------------------------------------------------------------------
# ACQUIRING -- "not yet" is not "lost", and it must still END.
# ---------------------------------------------------------------------------


def test_acquiring_is_bounded_terminal_and_stationary():
    """Landing ACQUIRING without a bounded, terminal exit is a hard fail-open.

    ``_lost()`` answered its ``_last_geometry is None`` branch with
    SEARCH_REQUIRED -- the stair's most severe verdict -- for a controller
    that had simply never received a bundle.  That is the first tick of every
    session (12 of 12 rows at ``bundle_count == 1`` in the recorded corpus),
    and the supervisor spent a reacquisition attempt sweeping the wrist away
    from a freshly seeded target (symptom B).

    But splitting the phase out is only safe if the new phase ends by itself:
    a row with ``deadline_s=None`` would stall forever, and the offline replay
    harness used to flag zero-command stalls only inside POSTURE_WAIT_PHASES,
    so it would have stayed green on that infinite stall.  This test pins
    every property that makes the split safe.
    """

    policy = PHASE_POLICY[ServoPhase.ACQUIRING]

    # A finite deadline, and an expiry that is terminal AND stationary.
    assert policy.deadline_s is not None
    assert policy.deadline_s > 0.0 and policy.deadline_s < float("inf")
    assert policy.on_expiry is ExpiryAction.STOP_AND_DEGRADE
    assert policy.expected_base_owner is BaseOwner.ZERO_HOLD
    assert policy.heartbeat_required is True
    assert policy.is_terminal is False
    # It must not be able to start an unexercised wrist sweep.
    assert policy.escalate_on_observation is False
    assert ServoPhase.ACQUIRING.value not in EAGER_RECOVERY_PHASES
    # It is a loss-stair member so the whole-body branch never solves on it.
    assert ServoPhase.ACQUIRING.value in LOSS_STAIR_PHASES
    # Its zero-command time must accumulate the cross-phase stall budget.
    assert policy.counts_no_progress is True

    # It must OUTLAST the supervisor's deliberate acquisition hold, or the
    # supervisor's hold would be cut short by this very deadline.
    planning_path = ROOT / "scripts" / "runtime" / "go2w_planning_control.py"
    grace_s = None
    for node in ast.walk(ast.parse(planning_path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SERVO_ACQUISITION_GRACE_S"
            and isinstance(node.value, ast.Constant)
        ):
            grace_s = float(node.value.value)
    assert grace_s is not None
    assert policy.deadline_s > grace_s, (
        f"acquiring expires at {policy.deadline_s}s, inside the supervisor's "
        f"{grace_s}s acquisition hold; the hold could never complete"
    )


def test_the_offline_harness_flags_a_stalled_acquiring_run():
    """The replay harness must not stay green on a parked ``acquiring``.

    This is the specific fail-open the phase split could have created:
    ``replay_trace`` once flagged zero-command stalls only for
    POSTURE_WAIT_PHASES, so a servo parked in a brand-new phase forever would
    have replayed as ``passed: True``.
    """

    rows = [
        _parked_row(ServoPhase.ACQUIRING.value, index)
        for index in range(int(30 * 20))  # 30 s at 20 Hz
    ]

    report = SUPERVISION.replay_trace(rows, stall_threshold_s=5.0)

    assert report["passed"] is False
    assert report["no_progress_stalls"], "the cross-phase detector missed it"
    codes = {stall["code"] for stall in report["stalls"]}
    assert "PHASE_DEADLINE_STALL" in codes
    stall = next(s for s in report["stalls"] if s["code"] == "PHASE_DEADLINE_STALL")
    assert stall["phase"] == ServoPhase.ACQUIRING.value
    assert stall["on_expiry"] == ExpiryAction.STOP_AND_DEGRADE.value


def test_a_parked_acquiring_servo_times_out_in_the_live_watchdog():
    """The same bound, in the deployed watchdog rather than the replay."""

    fired_s, decision = _first_timeout(lambda _index: ServoPhase.ACQUIRING.value)

    assert fired_s is not None, "an acquiring servo never timed out"
    assert decision.timed_out is True
    assert decision.on_expiry == ExpiryAction.STOP_AND_DEGRADE.value
    assert fired_s <= PHASE_POLICY[ServoPhase.ACQUIRING].deadline_s + 0.10


def test_every_eager_phase_agrees_with_its_own_table_row():
    """The supervisor may not overrule the table about the same phase.

    ``_EAGER_VIEW_RECOVERY_PHASES`` was hand-written next to a table that said
    something different, which is how ``view_recovery`` ended up being acted on
    at 0.80 s while its own row called for a 20 s wait.
    """

    assert EAGER_RECOVERY_PHASES
    for phase in EAGER_RECOVERY_PHASES:
        policy = PHASE_POLICY[ServoPhase(phase)]
        assert policy.escalate_on_observation is True
        assert policy.on_expiry is ExpiryAction.ESCALATE_VIEW_RECOVERY
        assert policy.is_terminal is False

    # And the invariant is enforced at construction, not just observed here.
    with pytest.raises(ValueError):
        PhasePolicy(
            deadline_s=5.0,
            on_expiry=ExpiryAction.STOP_AND_DEGRADE,
            heartbeat_required=True,
            expected_base_owner=BaseOwner.ZERO_HOLD,
            is_terminal=False,
            escalate_on_observation=True,
        )


def test_view_recovery_deadline_is_the_servos_number_not_a_second_opinion():
    """R5: one bound for one event, sourced from the servo's own document."""

    assert PHASE_POLICY[ServoPhase.VIEW_RECOVERY].on_expiry is (
        ExpiryAction.STOP_AND_DEGRADE
    )
    assert ServoPhase.VIEW_RECOVERY.value not in EAGER_RECOVERY_PHASES

    watchdog = SUPERVISION.ReactivePhaseWatchdog()
    phase = ServoPhase.VIEW_RECOVERY.value
    assert watchdog.deadline_for(phase, {"limits": {"tracking_loss_grace_s": 5.5}}) == 5.5
    # Absent limits fall back to the SHIPPED launcher value, not to a looser
    # table default.
    assert watchdog.deadline_for(phase) == SHIPPED_TRACKING_LOSS_GRACE_S
    assert SHIPPED_TRACKING_LOSS_GRACE_S == _launcher_flag_value(
        "--tracking-loss-grace-s"
    ), (
        "the launcher's --tracking-loss-grace-s and the supervisor's fallback "
        "drifted apart; that divergence IS the defect"
    )
    # Never looser than the cross-phase stall budget.
    assert view_recovery_deadline_s(1e6) <= NO_PROGRESS_DEADLINE_S


def _launcher_flag_value(flag: str) -> float:
    launcher = ROOT / "scripts" / "runtime" / "go2w_depth_servo.sh"
    tokens = launcher.read_text(encoding="utf-8").split()
    return float(tokens[tokens.index(flag) + 1])
