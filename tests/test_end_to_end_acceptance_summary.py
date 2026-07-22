from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "end_to_end_acceptance_summary.py"
SPEC = importlib.util.spec_from_file_location("end_to_end_acceptance_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


START_NS = 1_767_225_600_000_000_000
END_NS = START_NS + 20_000_000_000
SOURCE_NS = START_NS + 2_000_000_000
EXECUTOR_NS = START_NS + 7_000_000_000


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs() -> tuple[dict, dict, dict]:
    bag = {
        "offline": True,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "complete": True,
        "integrity": {
            "rosbag": {
                "path": "/stopped-bag",
                "framing_valid": True,
                "required_topics_missing": [],
                "required_topics_empty": [],
                "declared_message_count": 100,
                "topic_count": 12,
                "starting_time_unix_ns": START_NS,
                "ending_time_unix_ns_exclusive": END_NS,
            }
        },
    }
    perception = {
        "recorded_fresh": {
            "requests": 2,
            "exact_request_bundles": 2,
            "latency": {"count": 2, "p50_s": 1.0, "p95_s": 1.5},
        },
        "tracked_counterfactual": {
            "eligible_same_instruction_requests": 2,
            "exact_cached_identity_bundles": 2,
            "cached_bundle_age_at_most_0_5_s": 2,
            "unresolved": 0,
            "cached_bundle_age": {"count": 2, "p95_s": 0.1},
        },
    }
    planning = {
        "offline": True,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "summary": {
            "handoff": {
                "eligible_trials": 2,
                "succeeded": 2,
                "planner_wall_s": {"p50": 1.0, "p95": 1.5},
                "planner_search_s": {"p50": 0.8, "p95": 1.2},
            }
        },
    }
    return bag, perception, planning


def _make_interactive_chain(tmp_path: Path) -> tuple[Path, Path]:
    interactive = tmp_path / "interactive"
    perception_report = _write(
        interactive / "perception" / "p1" / "perception" / "report.json",
        {"stamp_ns": SOURCE_NS},
    )
    assert perception_report.exists()

    valid_report = _write(
        interactive / "planning" / "plan-ok" / "artifacts" / "planning"
        / "planning_report.json",
        {
            "plan_valid": True,
            "planned_grasp_sha256": "grasp-hash",
            "source_stamp_ns": SOURCE_NS,
            "rejections": [],
            "rejection_count": 0,
            "rejections_truncated": False,
        },
    )
    _write(
        interactive / "planning" / "plan-ok" / "attempt.json",
        {
            "status": "succeeded",
            "selected_perception_session_id": "p1",
            "finished_at": "2026-01-01T00:00:05Z",
        },
    )
    _write(
        interactive / "planning" / "plan-ik" / "artifacts" / "planning"
        / "planning_report.json",
        {
            "plan_valid": False,
            "source_stamp_ns": SOURCE_NS + 1,
            "rejections": [
                {"stage": "ik", "reason": "no solution"},
                {"stage": "ik", "reason": "no solution"},
            ],
            "rejection_count": 2,
            "rejections_truncated": False,
        },
    )

    receipts = tmp_path / "receipts"
    report_hash = hashlib.sha256(valid_report.read_bytes()).hexdigest()
    _write(
        receipts / "run-1" / "pregrasp-receipt.json",
        {
            "schema": "z_manip.piper_stage_receipt.v1",
            "stage": "pregrasp",
            "success": True,
            "motion_feedback_verified": True,
            "started_unix_ns": EXECUTOR_NS,
            "planning_report_sha256": report_hash,
            "planned_grasp_sha256": "grasp-hash",
        },
    )
    return interactive, receipts


def test_complete_hash_linked_chain_is_accepted(tmp_path: Path) -> None:
    bag, perception, planning = _inputs()
    interactive, receipts = _make_interactive_chain(tmp_path)

    report = MODULE.build_report(
        bag_report=bag,
        perception_report=perception,
        planning_replay_report=planning,
        interactive_root=interactive,
        receipts_root=receipts,
    )

    assert report["verdict"] == "accepted"
    assert report["stages"]["executor_start"]["measurement_status"] == "measured"
    assert report["stages"]["executor_start"]["metrics"]["valid_linked_starts"] == 1
    assert report["stages"]["all_ik_disposition"]["metrics"][
        "disposition_counts"
    ]["exhaustive_all_ik"] == 1
    assert report["motion_commands_sent"] == 0


def test_missing_receipt_remains_unmeasured(tmp_path: Path) -> None:
    bag, perception, planning = _inputs()
    interactive, _ = _make_interactive_chain(tmp_path)
    empty_receipts = tmp_path / "empty-receipts"
    empty_receipts.mkdir()

    report = MODULE.build_report(
        bag_report=bag,
        perception_report=perception,
        planning_replay_report=planning,
        interactive_root=interactive,
        receipts_root=empty_receipts,
    )

    executor = report["stages"]["executor_start"]
    assert report["verdict"] == "incomplete_evidence"
    assert executor["measurement_status"] == "unmeasured"
    assert executor["passed"] is None
    assert "executor_start" in report["incomplete_evidence_stages"]
    assert "UNMEASURED" in MODULE.render_markdown(report)


def test_unlinked_process_like_receipt_is_not_execution_evidence(tmp_path: Path) -> None:
    bag, perception, planning = _inputs()
    interactive, receipts = _make_interactive_chain(tmp_path)
    receipt_path = receipts / "run-1" / "pregrasp-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["planning_report_sha256"] = "not-an-artifact-hash"
    _write(receipt_path, receipt)

    report = MODULE.build_report(
        bag_report=bag,
        perception_report=perception,
        planning_replay_report=planning,
        interactive_root=interactive,
        receipts_root=receipts,
    )

    executor = report["stages"]["executor_start"]
    assert executor["measurement_status"] == "unmeasured"
    assert executor["metrics"]["valid_linked_starts"] == 0
    assert "planning_report_sha256" in executor["metrics"]["invalid_candidates"][0][
        "errors"
    ][0]


def test_all_ik_classifier_fails_closed_on_truncation_and_mixed_stages() -> None:
    exhaustive = {
        "plan_valid": False,
        "rejections": [{"stage": "ik"}, {"stage": "ik"}],
        "rejection_count": 2,
        "rejections_truncated": False,
    }
    assert MODULE.classify_planning_report(exhaustive) == "exhaustive_all_ik"
    assert MODULE.classify_planning_report({
        **exhaustive,
        "rejections_truncated": True,
    }) == "truncated_failure"
    assert MODULE.classify_planning_report({
        **exhaustive,
        "rejections": [{"stage": "ik"}, {"stage": "approach_collision"}],
    }) == "mixed_failure"


def test_counterfactual_tracking_never_satisfies_runtime_measurement() -> None:
    _, perception, _ = _inputs()
    _, tracked = MODULE.summarize_perception(perception, fresh_p95_goal_s=2.0)

    assert tracked["measurement_status"] == "counterfactual"
    assert tracked["passed"] is None

