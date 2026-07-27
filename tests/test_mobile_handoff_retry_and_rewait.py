"""WS-B items #5 and #14: bounded mobile-handoff recapture and joint re-wait.

Item #5 -- one shared recapture budget for the fresh close-range capture,
consumed by EITHER a transient perception-process/camera fault OR a
passive-window straddle, strictly capped at 1x, with the base stopped and
locked across the whole block.  Geometry/grounding failures fail closed.

Item #14 -- the passive-joint readiness wait is 3.5 s with exactly one bounded
re-wait, and the not_before_unix_ns ordering gate is passed unchanged to every
poll in every window.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_planning_control.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("go2w_planning_control", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTROL = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("go2w_planning_control", CONTROL)
SPEC.loader.exec_module(CONTROL)


BOUNDARY = 1_800_000_000_000_000_000


def _executor_start_evidence() -> dict[str, object]:
    return {
        "executor_started_unix_ns": 1_800_000_001_000_000_000,
        "executor_started_monotonic_ns": 123_456_789,
        "monotonic_clock_domain": "nuc_piper_executor_process",
    }


class _FakeBaseLock:
    """Records lock/unlock calls in order."""

    def __init__(self):
        self.calls = []
        self.locked = False

    def request_lock(self, source, *, base_stopped, now=None):
        self.calls.append(("lock", source, base_stopped))
        self.locked = True
        return "locked"

    def request_unlock(self, source, *, now=None):
        self.calls.append(("unlock", source))
        self.locked = False
        return "unlocked"

    def status_field(self, now=None):
        return {"state": "locked" if self.locked else "unlocked"}


def _retry_runner(tmp_path, *, perception_results, validate, base_lock=None):
    """Build a PiperGraspRunner whose fresh capture returns a scripted sequence.

    ``perception_results`` is consumed one entry per start_perception call; the
    last entry repeats.  Each call records whether the base was locked at the
    moment perception ran, so "base stopped+locked across every retry" is
    provable.
    """

    events: list[object] = []

    class FakeSessions:
        def __init__(self):
            self.calls = 0

        def clear_current_context(self):
            events.append("clear")

        def start_perception(self, target):
            index = min(self.calls, len(perception_results) - 1)
            self.calls += 1
            events.append((
                "perception",
                self.calls,
                "locked" if (base_lock is not None and base_lock.locked) else "unlocked",
            ))
            result = dict(perception_results[index])
            result.setdefault("session_id", f"20260723-12000{self.calls}")
            return result

        def start_planning(self):
            events.append("planning")
            return {"status": "succeeded", "session_id": "20260723-120009"}

    class FreshJoints:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            return True, "fresh", {
                "sequence": 42,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            }

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.receipt_root = tmp_path / "receipts"
    runner.receipt_root.mkdir()
    runner.session_service = FakeSessions()
    if base_lock is not None:
        runner.base_lock = base_lock
    runner._lock = threading.Lock()
    runner._status = {
        "revision": 0,
        "running": True,
        "phase": "handoff_settle",
        "outcome": None,
    }
    runner._planning_artifacts = lambda attempt: (
        tmp_path / "planning_report.json",
        tmp_path / "planned_grasp.npz",
    )
    runner.home_verifier = FreshJoints()
    runner._validate_mobile_handoff_capture_evidence = validate

    def execute(**kwargs):
        events.append("execute")
        return _executor_start_evidence()

    runner._run_full = execute
    runner._events = events
    return runner


def _perception_calls(runner):
    return [event for event in runner._events if event[0] == "perception"]


# ---------------------------------------------------------------------------
# Item #5: transient perception-process / camera faults recapture once.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "code",
    ["PERCEPTION_PROCESS_FAILED", "PERCEPTION_CAMERA_FRAME_TIMEOUT"],
)
def test_transient_perception_fault_recaptures_once(tmp_path, code):
    runner = _retry_runner(
        tmp_path,
        perception_results=[
            {"status": "failed", "error": {"code": code, "message": "transient"}},
            {"status": "succeeded"},
        ],
        validate=lambda **_kwargs: {"validated": True},
    )
    runner._run_mobile_handoff(
        "white charger", 9, "20260723-115900", BOUNDARY, place_back=False,
    )

    # Exactly one recapture: the fresh capture ran twice, then succeeded.
    assert len(_perception_calls(runner)) == 2
    assert runner.status()["outcome"] == "passed"


def test_transient_recapture_is_capped_at_one(tmp_path):
    # A persistent transient fault must not loop: one recapture, then closed.
    runner = _retry_runner(
        tmp_path,
        perception_results=[
            {"status": "failed", "error": {"code": "PERCEPTION_PROCESS_FAILED"}},
        ],
        validate=lambda **_kwargs: pytest.fail("validation must never run"),
    )
    runner._run_mobile_handoff(
        "white charger", 9, "20260723-115900", BOUNDARY, place_back=False,
    )

    assert len(_perception_calls(runner)) == 2  # 1 initial + 1 capped retry
    assert runner.status()["outcome"] == "blocked"


@pytest.mark.parametrize(
    "code",
    ["GRASP_GEOMETRY_FAILED", "PERCEPTION_TARGET_NOT_FOUND", "PERCEPTION_TRACKER_LOST"],
)
def test_geometry_and_grounding_faults_fail_closed_without_retry(tmp_path, code):
    runner = _retry_runner(
        tmp_path,
        perception_results=[
            {"status": "failed", "error": {"code": code, "message": "deterministic"}},
        ],
        validate=lambda **_kwargs: pytest.fail("validation must never run"),
    )
    runner._run_mobile_handoff(
        "white charger", 9, "20260723-115900", BOUNDARY, place_back=False,
    )

    # Deterministic for the frozen scene: no recapture, fail closed.
    assert len(_perception_calls(runner)) == 1
    assert runner.status()["outcome"] == "blocked"


def test_straddle_recapture_still_works(tmp_path):
    attempts = {"n": 0}

    def flaky_validate(**_kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError(
                "passive handoff feedback is not strictly post-stop and ordered",
            )
        return {"validated": True}

    runner = _retry_runner(
        tmp_path,
        perception_results=[{"status": "succeeded"}],
        validate=flaky_validate,
    )
    runner._run_mobile_handoff(
        "white charger", 9, "20260723-115900", BOUNDARY, place_back=False,
    )

    assert attempts["n"] == 2
    assert len(_perception_calls(runner)) == 2
    assert runner.status()["outcome"] == "passed"


def test_shared_budget_transient_then_straddle_fails_closed(tmp_path):
    # A transient fault spends the single shared recapture budget; a subsequent
    # straddle on the retried frame is NOT granted a second recapture.
    attempts = {"n": 0}

    def straddle_validate(**_kwargs):
        attempts["n"] += 1
        raise RuntimeError(
            "passive handoff feedback is not strictly post-stop and ordered",
        )

    runner = _retry_runner(
        tmp_path,
        perception_results=[
            {"status": "failed", "error": {"code": "PERCEPTION_PROCESS_FAILED"}},
            {"status": "succeeded"},
        ],
        validate=straddle_validate,
    )
    runner._run_mobile_handoff(
        "white charger", 9, "20260723-115900", BOUNDARY, place_back=False,
    )

    # 2 perception calls (transient recapture), validation attempted once on the
    # retried frame, then closed -- never a third capture.
    assert len(_perception_calls(runner)) == 2
    assert attempts["n"] == 1
    assert runner.status()["outcome"] == "blocked"


def test_base_stays_stopped_and_locked_across_the_transient_retry(tmp_path):
    base_lock = _FakeBaseLock()
    runner = _retry_runner(
        tmp_path,
        perception_results=[
            {"status": "failed", "error": {"code": "PERCEPTION_CAMERA_FRAME_TIMEOUT"}},
            {"status": "succeeded"},
        ],
        validate=lambda **_kwargs: {"validated": True},
        base_lock=base_lock,
    )
    runner._run_mobile_handoff(
        "white charger", 9, "20260723-115900", BOUNDARY, place_back=False,
    )

    # Locked exactly once (base_stopped=True), unlocked exactly once at the end.
    assert base_lock.calls == [
        ("lock", "mobile_handoff_grasp", True),
        ("unlock", "mobile_handoff_grasp"),
    ]
    # Both the initial capture and the recapture ran while the base was locked.
    assert [event[2] for event in _perception_calls(runner)] == ["locked", "locked"]
    assert runner.status()["outcome"] == "passed"


def test_retry_codes_are_exactly_the_transient_perception_faults():
    assert CONTROL.MOBILE_HANDOFF_PERCEPTION_RETRY_CODES == frozenset({
        "PERCEPTION_PROCESS_FAILED",
        "PERCEPTION_CAMERA_FRAME_TIMEOUT",
    })


# ---------------------------------------------------------------------------
# Item #14: 3.5 s joint window with one bounded re-wait; ordering gate intact.
# ---------------------------------------------------------------------------

def test_joint_ready_timeout_widened_with_one_rewait():
    assert CONTROL.MOBILE_HANDOFF_JOINT_READY_TIMEOUT_S == 3.5
    assert CONTROL.MOBILE_HANDOFF_JOINT_READY_REWAITS == 1


def _clock(monkeypatch):
    state = {"t": 0.0}
    monkeypatch.setattr(CONTROL.time, "monotonic", lambda: state["t"])
    monkeypatch.setattr(
        CONTROL.time, "sleep", lambda dt: state.__setitem__("t", state["t"] + max(dt, 0.0)),
    )
    return state


def test_joint_wait_rewaits_exactly_once_then_fails_closed(tmp_path, monkeypatch):
    _clock(monkeypatch)
    seen = []

    class NeverReady:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            seen.append(not_before_unix_ns)
            return False, "predates stop", {"source_timestamp_ns": not_before_unix_ns - 1}

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.home_verifier = NeverReady()

    with pytest.raises(RuntimeError) as excinfo:
        runner._wait_mobile_handoff_joints(
            not_before_unix_ns=BOUNDARY, timeout_s=0.1,
        )

    message = str(excinfo.value)
    assert "not observed after base stop" in message
    assert "x 2 windows" in message  # 1 initial + 1 bounded re-wait
    # The ordering gate was never relaxed: every poll demanded the same bound.
    assert seen and all(bound == BOUNDARY for bound in seen)
    # Exactly one re-wait was logged (not a loop).
    rewaits = [
        line
        for line in runner.log_path.read_text(encoding="utf-8").splitlines()
        if "re-waiting one bounded window" in line
    ]
    assert len(rewaits) == 1


def test_joint_wait_recovers_in_the_rewait_without_relaxing_the_gate(tmp_path, monkeypatch):
    clock = _clock(monkeypatch)
    seen = []

    class ReadyInSecondWindow:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            seen.append(not_before_unix_ns)
            # Not ready during the first 0.1 s window; a strictly post-stop
            # sample only arrives after the re-wait begins.
            if clock["t"] < 0.15:
                return False, "predates stop", {
                    "source_timestamp_ns": not_before_unix_ns - 1,
                }
            return True, "fresh", {
                "sequence": 7,
                "source_timestamp_ns": not_before_unix_ns + 5,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            }

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.home_verifier = ReadyInSecondWindow()

    evidence = runner._wait_mobile_handoff_joints(
        not_before_unix_ns=BOUNDARY, timeout_s=0.1,
    )

    # Recovered in the re-wait with a strictly post-stop sample.
    assert evidence["source_timestamp_ns"] == BOUNDARY + 5
    assert all(bound == BOUNDARY for bound in seen)
    assert "re-waiting one bounded window" in runner.log_path.read_text(encoding="utf-8")


def test_joint_wait_returns_in_first_window_without_rewait(tmp_path, monkeypatch):
    _clock(monkeypatch)

    class ImmediatelyReady:
        def current_joint_snapshot(self, *, not_before_unix_ns):
            return True, "fresh", {
                "sequence": 1,
                "source_timestamp_ns": not_before_unix_ns + 1,
                "joint_positions_rad": [0.0] * 6,
                "read_only": True,
            }

    runner = object.__new__(CONTROL.PiperGraspRunner)
    runner.log_path = tmp_path / "grasp.log"
    runner.home_verifier = ImmediatelyReady()

    evidence = runner._wait_mobile_handoff_joints(
        not_before_unix_ns=BOUNDARY, timeout_s=0.1,
    )

    assert evidence["source_timestamp_ns"] == BOUNDARY + 1
    assert "re-waiting one bounded window" not in runner.log_path.read_text(
        encoding="utf-8",
    )
