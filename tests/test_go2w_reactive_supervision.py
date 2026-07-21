from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_reactive_supervision.py"
SPEC = importlib.util.spec_from_file_location("go2w_reactive_supervision", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUPERVISION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUPERVISION
SPEC.loader.exec_module(SUPERVISION)


def _posture_runtime(*, arm_mode: str = "hold", arm_owner: str | None = None):
    runtime = {
        "phase": "posture_adjust",
        "posture_status": {
            "age_s": 0.1,
            "document": {
                "schema": "z_manip.go2w_posture_status.v1",
                "phase": "settling",
                "command_owner": "posture_adapter",
                "feedback": {"fresh": True},
                "body_height": {"current_m": 0.24, "target_m": 0.20},
                "attitude": {"current_pitch_rad": 0.1, "target_pitch_rad": 0.2},
            },
        },
        "reactive": {"arm_view": {"mode": arm_mode}},
        "output": {"published_linear_x": 0.0, "published_angular_z": 0.0},
    }
    if arm_owner is not None:
        runtime["arm_view_status"] = {"owner": arm_owner}
    return runtime


def test_historical_fixture_reproduces_posture_wait_stall():
    records = SUPERVISION.load_jsonl(
        ROOT / "tests" / "fixtures" / "go2w_posture_stall.jsonl"
    )

    report = SUPERVISION.replay_trace(records, stall_threshold_s=5.0)

    assert report["passed"] is False
    assert len(report["stalls"]) == 1
    assert report["stalls"][0]["duration_s"] == 118.6
    assert report["stalls"][0]["code"] == "POSTURE_WAIT_STALL"
    assert report["stalls"][0]["recommended_terminal_phase"] == "degraded"


def test_watchdog_bounds_missing_posture_feedback():
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(posture_wait_timeout_s=2.0)
    )
    runtime = {"phase": "posture_adjust", "reactive": {"arm_view": {"mode": "track"}}}

    assert watchdog.observe(runtime, now_s=10.0).timed_out is False
    decision = watchdog.observe(runtime, now_s=12.0)

    assert decision.timed_out is True
    assert decision.code == "POSTURE_FEEDBACK_TIMEOUT"
    assert decision.owners["arm_view"] == "intent_only"


def test_watchdog_distinguishes_arm_intent_without_executor_feedback():
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(posture_wait_timeout_s=1.0)
    )
    runtime = _posture_runtime(arm_mode="track")

    watchdog.observe(runtime, now_s=2.0)
    decision = watchdog.observe(runtime, now_s=3.0)

    assert decision.code == "ARM_VIEW_FEEDBACK_TIMEOUT"
    assert decision.feedback["current_height_m"] == 0.24


def test_watchdog_reports_measured_posture_settle_timeout():
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(posture_wait_timeout_s=1.0)
    )
    runtime = _posture_runtime(arm_mode="track", arm_owner="arm_view_executor")

    watchdog.observe(runtime, now_s=4.0)
    decision = watchdog.observe(runtime, now_s=5.0)

    assert decision.code == "POSTURE_SETTLE_TIMEOUT"
    assert decision.owners == {
        "base": "zero_hold",
        "body": "posture_adapter",
        "arm_view": "arm_view_executor",
        "optimizer": "unavailable",
    }


def test_posture_subphase_changes_do_not_reset_the_wait_budget():
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(posture_wait_timeout_s=2.0)
    )
    runtime = _posture_runtime(arm_mode="track", arm_owner="arm_view_executor")

    watchdog.observe(runtime, now_s=20.0)
    runtime["phase"] = "posture_shadow_verified"
    decision = watchdog.observe(runtime, now_s=22.0)

    assert decision.timed_out is True
    assert decision.phase == "posture_shadow_verified"
