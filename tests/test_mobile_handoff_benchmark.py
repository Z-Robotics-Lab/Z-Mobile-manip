from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "mobile_handoff_benchmark.py"
SPEC = importlib.util.spec_from_file_location("mobile_handoff_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def _write_attempt(root, action, session, started, finished, status, **extra):
    path = root / action / session
    path.mkdir(parents=True)
    (path / "attempt.json").write_text(json.dumps({
        "action": action, "session_id": session, "started_at": started,
        "finished_at": finished, "status": status, **extra,
    }), encoding="utf-8")
    return path


def _bag(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  duration:\n    nanoseconds: 20000000000\n"
        "  starting_time:\n    nanoseconds_since_epoch: 100000000000\n"
        "  message_count: 42\n", encoding="utf-8",
    )
    return bag


def test_report_is_windowed_and_links_perception_to_planning(tmp_path):
    bag = _bag(tmp_path)
    sessions = tmp_path / "sessions"
    perception = _write_attempt(sessions, "perception", "p1", "1970-01-01T00:01:41Z", "1970-01-01T00:01:42.5Z", "succeeded")
    report_dir = perception / "perception"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(json.dumps({"elapsed_s": 1.0, "grasp_candidates": 32}), encoding="utf-8")
    planning = _write_attempt(sessions, "planning", "q1", "1970-01-01T00:01:42.6Z", "1970-01-01T00:01:44Z", "succeeded", selected_perception_session_id="p1")
    planning_dir = planning / "artifacts" / "planning"
    planning_dir.mkdir(parents=True)
    (planning_dir / "planning_report.json").write_text(json.dumps({"timings_s": {"setup": 0.2, "search": 0.7, "total": 0.9}, "rejections": [], "rejection_count": 0}), encoding="utf-8")
    (planning / "planning.log").write_text("\n".join((
        "planner human output",
        json.dumps({
            "schema": "z_manip.interactive_timing.v1",
            "stage": "planning_session_gate",
            "elapsed_s": 0.12,
        }),
        json.dumps({
            "schema": "z_manip.interactive_timing.v1",
            "stage": "planning_ready_pre_visualization",
            "elapsed_s": 1.01,
        }),
        json.dumps({
            "schema": "z_manip.interactive_timing.v1",
            "stage": "planning_visualization_and_audit",
            "elapsed_s": 0.2,
        }),
        json.dumps({
            "schema": "z_manip.interactive_timing.v1",
            "stage": "planning_total",
            "elapsed_s": 1.21,
        }),
    )) + "\n", encoding="utf-8")
    _write_attempt(sessions, "perception", "stale", "1970-01-01T00:00:01Z", "1970-01-01T00:00:02Z", "succeeded")

    report = BENCH.build_report(bag=bag, sessions_root=sessions)

    assert report["offline"] is True
    assert report["transport_opened"] is False
    assert report["motion_commands_sent"] == 0
    assert report["perception"]["attempts"] == 1
    assert report["planning"]["attempts"] == 1
    assert report["transactions"]["count"] == 1
    assert report["transactions"]["duration_s"]["p50"] == 3.0
    assert report["transactions"]["orchestration_gap_s"]["p50"] == 0.1
    assert report["perception"]["wrapper_overhead_s"]["p50"] == 0.5
    assert report["planning"]["wrapper_stages_s"][
        "planning_ready_pre_visualization"
    ] == {"samples": 1, "p50": 1.01, "p95": 1.01}
    assert report["critical_path_audit"]["plan_ready_marker_status"] == "observed"
    assert report["critical_path_audit"]["visualization_deferral_applied"] is False


def test_timing_parser_does_not_infer_missing_or_invalid_markers(tmp_path):
    log = tmp_path / "planning.log"
    log.write_text("\n".join((
        "not json",
        json.dumps({
            "schema": "other.schema",
            "stage": "planning_ready_pre_visualization",
            "elapsed_s": 0.3,
        }),
        json.dumps({
            "schema": "z_manip.interactive_timing.v1",
            "stage": "planning_ready_pre_visualization",
            "elapsed_s": -1.0,
        }),
    )), encoding="utf-8")

    assert BENCH._timing_stages(log) == {}


def test_report_classifies_ik_dominated_planning_failure(tmp_path):
    bag = _bag(tmp_path)
    sessions = tmp_path / "sessions"
    planning = _write_attempt(sessions, "planning", "q1", "1970-01-01T00:01:41Z", "1970-01-01T00:01:47Z", "blocked", error={"code": "OFFLINE_PLANNER_BLOCKED", "message": "rejections={ik:64}"})
    planning_dir = planning / "artifacts" / "planning"
    planning_dir.mkdir(parents=True)
    (planning_dir / "planning_report.json").write_text(json.dumps({"timings_s": {"setup": 0.3, "search": 5.5, "total": 5.8}, "rejections": [{"stage": "ik"} for _ in range(64)], "rejection_count": 64}), encoding="utf-8")

    report = BENCH.build_report(bag=bag, sessions_root=sessions)

    assert report["planning"]["failure_counts"] == {"OFFLINE_PLANNER_BLOCKED": 1}
    assert report["planning"]["rejection_stages"] == {"ik": 64}
    assert "planning search is dominated by IK rejection" in report["bottlenecks"]


def test_servo_timing_quantifies_handoff_transitions():
    report = BENCH.servo_timing([
        {"phase": "handoff_settle", "updated_unix_ns": 1_000_000_000},
        {"phase": "handoff_settle", "updated_unix_ns": 1_100_000_000},
        {"phase": "handoff_probe", "updated_unix_ns": 1_300_000_000},
        {"phase": "stopped", "updated_unix_ns": 1_500_000_000},
    ])

    assert report["phase_counts"] == {
        "handoff_probe": 1, "handoff_settle": 2, "stopped": 1,
    }
    assert report["handoff_settle_to_probe_s"]["p50"] == 0.3
    assert report["handoff_probe_to_stop_s"]["p50"] == 0.2


def test_window_is_strict_and_lifecycle_does_not_invent_grasp_start(tmp_path):
    bag = _bag(tmp_path)
    sessions = tmp_path / "sessions"
    perception = _write_attempt(
        sessions, "perception", "p1",
        "1970-01-01T00:01:41Z", "1970-01-01T00:01:42Z", "succeeded",
    )
    (perception / "perception").mkdir()
    (perception / "perception" / "report.json").write_text(
        json.dumps({"elapsed_s": 0.8}), encoding="utf-8",
    )
    planning = _write_attempt(
        sessions, "planning", "q1",
        "1970-01-01T00:01:42.1Z", "1970-01-01T00:01:43.5Z", "blocked",
        selected_perception_session_id="p1",
        error={"code": "OFFLINE_PLANNER_BLOCKED"},
    )
    planning_dir = planning / "artifacts" / "planning"
    planning_dir.mkdir(parents=True)
    (planning_dir / "planning_report.json").write_text(
        json.dumps({"timings_s": {"total": 1.2}, "rejections": []}),
        encoding="utf-8",
    )
    # Exact metadata end is outside the half-open bag window.
    _write_attempt(
        sessions, "perception", "at-end",
        "1970-01-01T00:02:00Z", "1970-01-01T00:02:01Z", "succeeded",
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(item) for item in [
        {"phase": "stopped", "updated_unix_ns": 100_500_000_000},
        {"phase": "stopped", "updated_unix_ns": 120_000_000_000},
    ]), encoding="utf-8")
    grasp_log = tmp_path / "piper-grasp.log"
    grasp_log.write_text(
        "Mobile handoff joint readiness passed: sequence=7 "
        "source_timestamp_ns=100900000000\n",
        encoding="utf-8",
    )

    report = BENCH.build_report(
        bag=bag, sessions_root=sessions,
        trace_jsonl=trace, grasp_log=grasp_log,
    )

    assert report["perception"]["attempts"] == 1
    assert report["servo"]["records"] == 1
    lifecycle = report["handoff_lifecycle"]
    assert lifecycle["paired_base_stops"] == 1
    assert lifecycle["paired_joint_sources"] == 1
    assert lifecycle["grasp_start_events"] == 0
    assert lifecycle["grasp_start_status"] == "not_observed_in_artifacts"
    assert lifecycle["items"][0]["base_stop_to_fresh_perception_start_s"] == 0.5
    assert lifecycle["items"][0]["base_stop_to_joint_source_s"] == 0.4
    assert lifecycle["items"][0]["joint_source_to_fresh_perception_start_s"] == 0.1
