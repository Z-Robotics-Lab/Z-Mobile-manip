from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "handoff_stage_budget.py"
SPEC = importlib.util.spec_from_file_location("handoff_stage_budget", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parallel_critical_path_waits_for_slower_read_only_input() -> None:
    result = MODULE.simulate({
        "fresh_perception_s": 1.2,
        "passive_ready_s": 1.5,
        "planning_dispatch_s": 0.1,
        "planning_s": 1.0,
        "grasp_start_dispatch_s": 0.1,
    })

    assert result["critical_prerequisite"] == "passive_feedback"
    assert result["parallel_saved_vs_serial_s"] == 1.2
    assert result["events_s_after_stop"]["plan_finished"] == 2.6
    assert result["events_s_after_stop"]["grasp_started"] == 2.7
    assert result["grasp_start_under_goal"] is True


def test_unknown_executor_latency_never_claims_grasp_budget() -> None:
    result = MODULE.simulate({
        "fresh_perception_s": 1.68,
        "passive_ready_s": 0.05,
        "planning_dispatch_s": 0.010228,
        "planning_s": 1.266,
        "grasp_start_dispatch_s": None,
    })

    assert result["events_s_after_stop"]["plan_finished"] == 2.956228
    assert result["remaining_grasp_start_budget_s"] == 0.043772
    assert result["grasp_start_under_goal"] is None
    assert result["grasp_start_timing_known"] is False


def test_recorded_passive_sample_exposes_budget_shortfall() -> None:
    result = MODULE.simulate({
        "fresh_perception_s": 1.68,
        "passive_ready_s": 1.840508,
        "planning_dispatch_s": 0.010228,
        "planning_s": 1.266,
    })

    assert result["critical_prerequisite"] == "passive_feedback"
    assert result["events_s_after_stop"]["plan_finished"] == 3.116736
    assert result["plan_finish_shortfall_s"] == 0.116736
    assert result["remaining_grasp_start_budget_s"] == -0.116736


def test_evidence_gate_requires_explicit_post_stop_epochs_and_grasp_start() -> None:
    result = MODULE.audit_evidence({
        "base_stop_unix_ns": 1_000,
        "fresh_data_source_unix_ns": 1_100,
        "passive_source_unix_ns": 1_050,
        "plan_finished_unix_ns": 1_200,
        "plan_succeeded": True,
    }, goal_s=3.0)

    assert result["complete"] is False
    assert result["observed_under_goal"] is None
    assert result["missing_or_invalid"] == ["grasp_started_after_plan"]


def test_evidence_gate_rejects_stale_source_and_failed_plan() -> None:
    result = MODULE.audit_evidence({
        "base_stop_unix_ns": 2_000_000_000,
        "fresh_data_source_unix_ns": 1_900_000_000,
        "passive_source_unix_ns": 2_100_000_000,
        "plan_finished_unix_ns": 2_500_000_000,
        "grasp_started_unix_ns": 2_600_000_000,
        "plan_succeeded": False,
    }, goal_s=3.0)

    assert result["complete"] is False
    assert "fresh_data_is_post_stop" in result["missing_or_invalid"]
    assert "plan_succeeded" in result["missing_or_invalid"]


def test_evidence_gate_accepts_complete_chronological_trace() -> None:
    result = MODULE.audit_evidence({
        "base_stop_unix_ns": 1_000_000_000,
        "fresh_data_source_unix_ns": 1_500_000_000,
        "passive_source_unix_ns": 1_200_000_000,
        "plan_finished_unix_ns": 2_800_000_000,
        "grasp_started_unix_ns": 2_900_000_000,
        "plan_succeeded": True,
    }, goal_s=3.0)

    assert result["complete"] is True
    assert result["observed_stop_to_grasp_start_s"] == 1.9
    assert result["observed_under_goal"] is True


def test_negative_or_nonfinite_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="passive_ready_s"):
        MODULE.simulate({
            "fresh_perception_s": 1.0,
            "passive_ready_s": -1.0,
            "planning_s": 1.0,
        })
