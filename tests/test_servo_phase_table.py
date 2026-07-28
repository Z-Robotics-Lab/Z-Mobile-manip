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
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from z_manip.control.reactive_servo import ReactivePhase
from z_manip.control.servo_phase import (
    PHASE_POLICY,
    BaseOwner,
    ExpiryAction,
    GUARDED_PHASE_LITERALS,
    ServoPhase,
    phase_policy,
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

    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git/" in str(path):
            continue
        if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".json", ".html"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if "reactive/ik_probe" in text:
            hits.append(str(path.relative_to(ROOT)))

    assert sorted(hits) == [
        "scripts/runtime/go2w_depth_servo.py",
        "tests/test_servo_phase_table.py",
    ], f"unexpected ik_probe references: {sorted(hits)}"
