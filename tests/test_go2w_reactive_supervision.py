from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import pytest


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


@pytest.mark.parametrize(
    "phase",
    (
        "waiting_target",
        "approach",
        "base_approach",
        "tracking_lost",
        "reacquiring",
        "view_recovery",
        "search_required",
    ),
)
def test_all_active_servo_phases_require_an_advancing_state_heartbeat(phase):
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=0.5)
    )
    runtime = {"phase": phase, "updated_unix_ns": 10_000_000_000}

    first = watchdog.observe(
        runtime,
        now_s=1.0,
        now_unix_ns=10_100_000_000,
    )
    frozen = watchdog.observe(
        runtime,
        now_s=1.5,
        now_unix_ns=10_600_000_000,
    )

    assert first.timed_out is False
    assert frozen.timed_out is True
    assert frozen.code == "REACTIVE_STATE_HEARTBEAT_TIMEOUT"
    assert frozen.heartbeat_updated_unix_ns == 10_000_000_000
    assert frozen.heartbeat_elapsed_s == pytest.approx(0.5)


def test_phase_changes_do_not_hide_a_frozen_state_heartbeat():
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=0.6)
    )
    runtime = {"phase": "approach", "updated_unix_ns": 20_000_000_000}
    watchdog.observe(runtime, now_s=4.0, now_unix_ns=20_050_000_000)
    runtime["phase"] = "tracking_lost"
    watchdog.observe(runtime, now_s=4.3, now_unix_ns=20_350_000_000)
    runtime["phase"] = "reacquiring"

    decision = watchdog.observe(
        runtime,
        now_s=4.6,
        now_unix_ns=20_650_000_000,
    )

    assert decision.timed_out is True
    assert decision.code == "REACTIVE_STATE_HEARTBEAT_TIMEOUT"


def test_advancing_state_heartbeat_refreshes_deadline():
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=0.5)
    )
    for index in range(6):
        decision = watchdog.observe(
            {
                "phase": "base_approach",
                "updated_unix_ns": 30_000_000_000 + index * 100_000_000,
            },
            now_s=7.0 + index * 0.1,
            now_unix_ns=30_050_000_000 + index * 100_000_000,
        )
        assert decision.timed_out is False
        assert decision.heartbeat_valid is True


@pytest.mark.parametrize("phase", ("reached", "handoff_probe", "handoff_ready"))
def test_grasp_handoff_rejects_missing_or_stale_state_heartbeat(phase):
    watchdog = SUPERVISION.ReactivePhaseWatchdog(
        SUPERVISION.ReactiveWatchdogConfig(state_heartbeat_timeout_s=0.5)
    )

    missing = watchdog.observe(
        {"phase": phase},
        now_s=1.0,
        now_unix_ns=40_000_000_000,
    )

    assert missing.timed_out is True
    assert missing.code == "REACTIVE_STATE_HEARTBEAT_TIMEOUT"

    watchdog.reset()
    stale = watchdog.observe(
        {"phase": phase, "updated_unix_ns": 40_000_000_000},
        now_s=2.0,
        now_unix_ns=41_000_000_000,
    )
    assert stale.timed_out is True
    assert stale.code == "REACTIVE_STATE_HEARTBEAT_TIMEOUT"
