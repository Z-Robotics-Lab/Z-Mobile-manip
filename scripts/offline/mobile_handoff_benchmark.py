#!/usr/bin/env python3
"""Benchmark mobile-handoff perception and planning from immutable artifacts.

This tool is deliberately filesystem-only. It does not import ROS, open a
network transport, or load any robot driver. Session attempts are restricted
to the rosbag time window so stale experiments do not contaminate a run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "z_mobile_manip.mobile_handoff_benchmark.v1"
TIMING_SCHEMA = "z_manip.interactive_timing.v1"
MAX_TIMING_LOG_BYTES = 4 * 1024 * 1024
PERCEPTION_GOAL_S = 2.0
HANDOFF_GOAL_S = 3.0


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_json_stream(path: Path) -> list[dict[str, Any]]:
    """Load line-delimited or concatenated trace records."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        value, index = decoder.raw_decode(text, index)
        if isinstance(value, dict):
            records.append(value)
    return records


def _stamp(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1e9)


def _duration_s(attempt: dict[str, Any]) -> float | None:
    started = _stamp(attempt.get("started_at"))
    finished = _stamp(attempt.get("finished_at"))
    if started is None or finished is None or finished < started:
        return None
    return (finished - started) / 1e9


def bag_window(bag: Path) -> dict[str, Any]:
    metadata = bag / "metadata.yaml" if bag.is_dir() else bag.parent / "metadata.yaml"
    text = metadata.read_text(encoding="utf-8")
    start_match = re.search(r"(?m)^\s*nanoseconds_since_epoch:\s*(\d+)\s*$", text)
    duration_match = re.search(
        r"(?ms)^\s*duration:\s*\n\s*nanoseconds:\s*(\d+)\s*$",
        text,
    )
    count_match = re.search(r"(?m)^\s*message_count:\s*(\d+)\s*$", text)
    if start_match is None or duration_match is None:
        raise ValueError(f"rosbag metadata has no bounded time window: {metadata}")
    start_ns = int(start_match.group(1))
    duration_ns = int(duration_match.group(1))
    return {
        "path": str(bag.resolve()),
        "start_unix_ns": start_ns,
        "end_unix_ns": start_ns + duration_ns,
        "duration_s": duration_ns / 1e9,
        "message_count": int(count_match.group(1)) if count_match else None,
    }


def _failure_code(attempt: dict[str, Any]) -> str | None:
    error = attempt.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str) and code:
            return code
        error = error.get("message")
    if not isinstance(error, str) or not error:
        return None
    lower = error.lower()
    if "grounding" in lower:
        return "GROUNDING_FAILED"
    if "obb dimension" in lower:
        return "GRASP_GEOMETRY_FAILED"
    if "ik=" in lower or "ik failed" in lower:
        return "IK_FAILED"
    if "collision" in lower:
        return "COLLISION_FAILED"
    return "UNKNOWN_FAILED"


def _attempts(
    sessions_root: Path,
    action: str,
    *,
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted((sessions_root / action).glob("*/attempt.json")):
        attempt = _load_json(path)
        if attempt is None:
            continue
        started_ns = _stamp(attempt.get("started_at"))
        # Rosbag windows are half-open.  An attempt beginning exactly at the
        # metadata end stamp belongs to the next recording, not this one.
        if started_ns is None or not start_ns <= started_ns < end_ns:
            continue
        result.append({
            **attempt,
            "attempt_path": str(path.resolve()),
            "started_unix_ns": started_ns,
            "finished_unix_ns": _stamp(attempt.get("finished_at")),
            "duration_s": _duration_s(attempt),
            "failure_code": _failure_code(attempt),
        })
    return result


def _nested_number(document: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timing_stages(path: Path) -> dict[str, float]:
    """Read bounded, machine-readable timing markers from one action log.

    Human log text is deliberately ignored.  A repeated stage keeps the last
    valid marker because it represents the completed retry/attempt that the
    backend returned.  This is measurement only: missing markers are never
    inferred from attempt wall time.
    """

    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or not 1 <= metadata.st_size <= MAX_TIMING_LOG_BYTES
        ):
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}
    stages: dict[str, float] = {}
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("schema") != TIMING_SCHEMA:
            continue
        stage = record.get("stage")
        elapsed = record.get("elapsed_s")
        if (
            not isinstance(stage, str)
            or not stage
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
        ):
            continue
        number = float(elapsed)
        if math.isfinite(number) and number >= 0.0:
            stages[stage] = number
    return stages


def enrich_attempts(attempts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for attempt in attempts:
        root = Path(attempt["attempt_path"]).parent
        action = str(attempt.get("action", ""))
        item = dict(attempt)
        timing_stages = _timing_stages(root / f"{action}.log")
        item["timing_stages_s"] = timing_stages
        if action == "perception":
            report = _load_json(root / "perception" / "report.json")
            item["kernel_elapsed_s"] = _nested_number(report, "elapsed_s")
            item["grasp_candidates"] = (
                report.get("grasp_candidates") if isinstance(report, dict) else None
            )
        elif action == "planning":
            report = _load_json(root / "artifacts" / "planning" / "planning_report.json")
            item["planner_setup_s"] = _nested_number(report, "timings_s", "setup")
            item["planner_search_s"] = _nested_number(report, "timings_s", "search")
            item["planner_total_s"] = _nested_number(report, "timings_s", "total")
            item["rejection_count"] = (
                report.get("rejection_count") if isinstance(report, dict) else None
            )
            rejections = report.get("rejections") if isinstance(report, dict) else None
            item["rejection_stages"] = dict(sorted(Counter(
                str(entry.get("stage", "unknown"))
                for entry in rejections or []
                if isinstance(entry, dict)
            ).items()))
            item["plan_ready_pre_visualization_s"] = timing_stages.get(
                "planning_ready_pre_visualization",
            )
        internal = item.get("kernel_elapsed_s", item.get("planner_total_s"))
        duration = item.get("duration_s")
        item["wrapper_overhead_s"] = (
            max(0.0, float(duration) - float(internal))
            if isinstance(duration, (int, float)) and isinstance(internal, (int, float))
            else None
        )
        enriched.append(item)
    return enriched


def _percentile(values: Iterable[float | None], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 6)


def summarize(attempts: list[dict[str, Any]], action: str) -> dict[str, Any]:
    durations = [item.get("duration_s") for item in attempts]
    succeeded = [item for item in attempts if item.get("status") == "succeeded"]
    failures = Counter(
        str(item.get("failure_code") or "UNCLASSIFIED")
        for item in attempts
        if item.get("status") != "succeeded"
    )
    result = {
        "attempts": len(attempts),
        "succeeded": len(succeeded),
        "success_rate": round(len(succeeded) / len(attempts), 6) if attempts else None,
        "duration_s": {
            "min": _percentile(durations, 0.0),
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": _percentile(durations, 1.0),
        },
        "wrapper_overhead_s": {
            "p50": _percentile((item.get("wrapper_overhead_s") for item in attempts), 0.50),
            "p95": _percentile((item.get("wrapper_overhead_s") for item in attempts), 0.95),
        },
        "failure_counts": dict(sorted(failures.items())),
    }
    if action == "perception":
        result["kernel_elapsed_s"] = {
            "min": _percentile((item.get("kernel_elapsed_s") for item in attempts), 0.0),
            "p50": _percentile((item.get("kernel_elapsed_s") for item in attempts), 0.50),
            "p95": _percentile((item.get("kernel_elapsed_s") for item in attempts), 0.95),
        }
    else:
        result["planner_search_s"] = {
            "min": _percentile((item.get("planner_search_s") for item in attempts), 0.0),
            "p50": _percentile((item.get("planner_search_s") for item in attempts), 0.50),
            "p95": _percentile((item.get("planner_search_s") for item in attempts), 0.95),
        }
        rejection_stages: Counter[str] = Counter()
        for item in attempts:
            rejection_stages.update(item.get("rejection_stages") or {})
        result["rejection_stages"] = dict(sorted(rejection_stages.items()))
        stage_names = (
            "planning_session_gate",
            "planning_search",
            "planning_ready_pre_visualization",
            "planning_visualization_and_audit",
            "planning_total",
        )
        result["wrapper_stages_s"] = {
            stage: {
                "samples": sum(
                    stage in (item.get("timing_stages_s") or {})
                    for item in attempts
                ),
                "p50": _percentile(
                    (
                        (item.get("timing_stages_s") or {}).get(stage)
                        for item in attempts
                    ),
                    0.50,
                ),
                "p95": _percentile(
                    (
                        (item.get("timing_stages_s") or {}).get(stage)
                        for item in attempts
                    ),
                    0.95,
                ),
            }
            for stage in stage_names
        }
    return result


def pair_transactions(
    perception: list[dict[str, Any]],
    planning: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_perception: dict[str, list[dict[str, Any]]] = {}
    for item in planning:
        selected = item.get("selected_perception_session_id")
        if isinstance(selected, str):
            by_perception.setdefault(selected, []).append(item)
    result = []
    for capture in perception:
        session_id = capture.get("session_id")
        linked = sorted(
            by_perception.get(str(session_id), []),
            key=lambda item: int(item.get("started_unix_ns") or 0),
        )
        for plan in linked:
            start_ns = capture.get("started_unix_ns")
            capture_end_ns = capture.get("finished_unix_ns")
            plan_start_ns = plan.get("started_unix_ns")
            end_ns = plan.get("finished_unix_ns")
            total = (
                (int(end_ns) - int(start_ns)) / 1e9
                if isinstance(start_ns, int) and isinstance(end_ns, int) and end_ns >= start_ns
                else None
            )
            orchestration_gap = (
                (int(plan_start_ns) - int(capture_end_ns)) / 1e9
                if isinstance(capture_end_ns, int)
                and isinstance(plan_start_ns, int)
                and plan_start_ns >= capture_end_ns
                else None
            )
            result.append({
                "perception_session_id": session_id,
                "planning_session_id": plan.get("session_id"),
                "perception_status": capture.get("status"),
                "planning_status": plan.get("status"),
                "perception_s": capture.get("duration_s"),
                "planning_s": plan.get("duration_s"),
                "plan_ready_pre_visualization_s": plan.get(
                    "plan_ready_pre_visualization_s",
                ),
                "perception_to_planning_gap_s": orchestration_gap,
                "perception_to_plan_finish_s": total,
                "failure_code": plan.get("failure_code"),
                "rejection_stages": plan.get("rejection_stages"),
                "perception_started_unix_ns": start_ns,
                "perception_finished_unix_ns": capture_end_ns,
                "planning_started_unix_ns": plan_start_ns,
                "planning_finished_unix_ns": end_ns,
            })
    return result


def _record_stamp(record: dict[str, Any]) -> int | None:
    try:
        return int(record.get("updated_unix_ns", record.get("sample_unix_ns")))
    except (TypeError, ValueError):
        return None


def window_trace_records(
    records: Iterable[dict[str, Any]],
    *,
    start_ns: int,
    end_ns: int,
) -> list[dict[str, Any]]:
    """Return trace records in the same strict ``[start, end)`` bag window."""
    return [
        record for record in records
        if (stamp := _record_stamp(record)) is not None
        and start_ns <= stamp < end_ns
    ]


def servo_timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    phases = Counter(str(item.get("phase", "unknown")) for item in records)
    transitions: Counter[str] = Counter()
    previous_phase: str | None = None
    last_settle_ns: int | None = None
    last_probe_ns: int | None = None
    settle_to_probe: list[float] = []
    probe_to_stop: list[float] = []
    for item in records:
        phase = str(item.get("phase", "unknown"))
        stamp_ns = _record_stamp(item)
        if phase != previous_phase:
            if previous_phase is not None:
                transitions[f"{previous_phase}->{phase}"] += 1
            if phase == "handoff_settle":
                last_settle_ns = stamp_ns
            elif phase == "handoff_probe":
                if stamp_ns is not None and last_settle_ns is not None and stamp_ns >= last_settle_ns:
                    settle_to_probe.append((stamp_ns - last_settle_ns) / 1e9)
                last_probe_ns = stamp_ns
                last_settle_ns = None
            elif phase == "stopped":
                if stamp_ns is not None and last_probe_ns is not None and stamp_ns >= last_probe_ns:
                    probe_to_stop.append((stamp_ns - last_probe_ns) / 1e9)
                last_probe_ns = None
            previous_phase = phase
    return {
        "records": len(records),
        "phase_counts": dict(sorted(phases.items())),
        "transition_counts": dict(sorted(transitions.items())),
        "handoff_settle_to_probe_s": {
            "samples": len(settle_to_probe),
            "p50": _percentile(settle_to_probe, 0.50),
            "p95": _percentile(settle_to_probe, 0.95),
        },
        "handoff_probe_to_stop_s": {
            "samples": len(probe_to_stop),
            "p50": _percentile(probe_to_stop, 0.50),
            "p95": _percentile(probe_to_stop, 0.95),
        },
    }


def parse_joint_source_evidence(path: Path | None) -> list[int]:
    """Read passive-joint source stamps recorded by the handoff readiness gate.

    The current log is intentionally conservative evidence: it records the
    source timestamp of the accepted passive sample, but not a separately
    timestamped gate-completion or actuator-start event.
    """
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    return [
        int(match.group(1))
        for match in re.finditer(r"source_timestamp_ns=(\d+)", text)
    ]


def handoff_lifecycle(
    transactions: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
    *,
    joint_source_stamps: Iterable[int] = (),
    stop_lookback_s: float = 15.0,
) -> dict[str, Any]:
    """Pair stop, fresh capture and planning without inventing executor events."""
    stops = sorted(
        stamp for record in trace_records
        if record.get("phase") == "stopped"
        and (stamp := _record_stamp(record)) is not None
    )
    joint_sources = sorted(int(stamp) for stamp in joint_source_stamps)
    used_stops: set[int] = set()
    used_joint_sources: set[int] = set()
    items: list[dict[str, Any]] = []
    max_lookback_ns = int(stop_lookback_s * 1e9)

    for transaction in sorted(
        transactions,
        key=lambda item: int(item.get("perception_started_unix_ns") or 0),
    ):
        perception_start = transaction.get("perception_started_unix_ns")
        perception_finish = transaction.get("perception_finished_unix_ns")
        planning_start = transaction.get("planning_started_unix_ns")
        planning_finish = transaction.get("planning_finished_unix_ns")
        if not isinstance(perception_start, int):
            continue
        eligible_stops = [
            stamp for stamp in stops
            if stamp not in used_stops
            and stamp <= perception_start
            and perception_start - stamp <= max_lookback_ns
        ]
        stop = eligible_stops[-1] if eligible_stops else None
        if stop is not None:
            used_stops.add(stop)

        eligible_joints = [
            stamp for stamp in joint_sources
            if stamp not in used_joint_sources
            and (stop is None or stamp >= stop)
            and stamp <= perception_start
        ]
        joint_source = eligible_joints[-1] if eligible_joints else None
        if joint_source is not None:
            used_joint_sources.add(joint_source)

        def delta(later: object, earlier: object) -> float | None:
            if isinstance(later, int) and isinstance(earlier, int) and later >= earlier:
                return round((later - earlier) / 1e9, 9)
            return None

        items.append({
            "perception_session_id": transaction.get("perception_session_id"),
            "planning_session_id": transaction.get("planning_session_id"),
            "base_stop_unix_ns": stop,
            "joint_source_unix_ns": joint_source,
            "perception_started_unix_ns": perception_start,
            "planning_finished_unix_ns": planning_finish,
            "base_stop_to_fresh_perception_start_s": delta(perception_start, stop),
            "base_stop_to_joint_source_s": delta(joint_source, stop),
            "joint_source_to_fresh_perception_start_s": delta(
                perception_start, joint_source,
            ),
            "fresh_perception_s": delta(perception_finish, perception_start),
            "perception_to_planning_gap_s": delta(planning_start, perception_finish),
            "planning_s": delta(planning_finish, planning_start),
            "base_stop_to_plan_finish_s": delta(planning_finish, stop),
            # Neither attempt.json nor the depth-servo trace records an
            # executor-start stamp.  A blocked plan must never be represented
            # as a grasp start merely because the handoff worker was launched.
            "grasp_start_observed": False,
        })

    stage_names = (
        "base_stop_to_fresh_perception_start_s",
        "base_stop_to_joint_source_s",
        "joint_source_to_fresh_perception_start_s",
        "fresh_perception_s",
        "perception_to_planning_gap_s",
        "planning_s",
        "base_stop_to_plan_finish_s",
    )
    stages = {}
    for stage in stage_names:
        values = [item.get(stage) for item in items]
        observed = [value for value in values if isinstance(value, (int, float))]
        stages[stage] = {
            "samples": len(observed),
            "min": _percentile(observed, 0.0),
            "p50": _percentile(observed, 0.50),
            "p95": _percentile(observed, 0.95),
            "max": _percentile(observed, 1.0),
        }
    return {
        "transactions": len(items),
        "paired_base_stops": sum(item["base_stop_unix_ns"] is not None for item in items),
        "paired_joint_sources": sum(
            item["joint_source_unix_ns"] is not None for item in items
        ),
        "grasp_start_events": 0,
        "grasp_start_status": "not_observed_in_artifacts",
        "stages": stages,
        "items": items,
    }


def build_report(
    *,
    bag: Path,
    sessions_root: Path,
    trace_jsonl: Path | None = None,
    grasp_log: Path | None = None,
) -> dict[str, Any]:
    window = bag_window(bag)
    perception = enrich_attempts(_attempts(
        sessions_root, "perception",
        start_ns=window["start_unix_ns"], end_ns=window["end_unix_ns"],
    ))
    planning = enrich_attempts(_attempts(
        sessions_root, "planning",
        start_ns=window["start_unix_ns"], end_ns=window["end_unix_ns"],
    ))
    transactions = pair_transactions(perception, planning)
    perception_summary = summarize(perception, "perception")
    planning_summary = summarize(planning, "planning")
    transaction_times = [item.get("perception_to_plan_finish_s") for item in transactions]
    orchestration_gaps = [
        item.get("perception_to_planning_gap_s") for item in transactions
    ]
    goals = {
        "perception_under_2s": bool(
            perception_summary["duration_s"]["p95"] is not None
            and perception_summary["duration_s"]["p95"] < PERCEPTION_GOAL_S
        ),
        "fresh_perception_plan_under_3s": bool(
            transaction_times and _percentile(transaction_times, 0.95) < HANDOFF_GOAL_S
        ),
    }
    bottlenecks = []
    if not goals["perception_under_2s"]:
        bottlenecks.append("perception wrapper p95 exceeds 2.0 s")
    if planning_summary.get("rejection_stages", {}).get("ik", 0):
        bottlenecks.append("planning search is dominated by IK rejection")
    if not goals["fresh_perception_plan_under_3s"]:
        bottlenecks.append("fresh perception plus planning p95 exceeds 3.0 s")
    report = {
        "schema": SCHEMA,
        "offline": True,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "bag": window,
        "perception": perception_summary,
        "planning": planning_summary,
        "transactions": {
            "count": len(transactions),
            "orchestration_gap_s": {
                "min": _percentile(orchestration_gaps, 0.0),
                "p50": _percentile(orchestration_gaps, 0.50),
                "p95": _percentile(orchestration_gaps, 0.95),
                "max": _percentile(orchestration_gaps, 1.0),
            },
            "duration_s": {
                "min": _percentile(transaction_times, 0.0),
                "p50": _percentile(transaction_times, 0.50),
                "p95": _percentile(transaction_times, 0.95),
                "max": _percentile(transaction_times, 1.0),
            },
            "items": transactions,
        },
        "goals": goals,
        "bottlenecks": bottlenecks,
        "attempts": {"perception": perception, "planning": planning},
    }
    wrapper_stages = planning_summary.get("wrapper_stages_s", {})
    plan_ready_stage = wrapper_stages.get(
        "planning_ready_pre_visualization",
        {},
    )
    visualization_stage = wrapper_stages.get(
        "planning_visualization_and_audit",
        {},
    )
    report["critical_path_audit"] = {
        "plan_ready_marker_status": (
            "observed"
            if plan_ready_stage.get("samples", 0) > 0
            else "not_observed"
        ),
        "plan_ready_pre_visualization_s": plan_ready_stage,
        "visualization_and_audit_s": visualization_stage,
        "visualization_deferral_applied": False,
        "reason": (
            "the current immutable planning contract builds and audits the "
            "debug bundle before the backend returns, then manifests and "
            "freezes the complete action tree; no executor receipt is "
            "observed in this bag, so subtracting UI work would be an "
            "unverified counterfactual"
        ),
    }
    if trace_jsonl is not None:
        trace_records = window_trace_records(
            _load_json_stream(trace_jsonl),
            start_ns=window["start_unix_ns"],
            end_ns=window["end_unix_ns"],
        )
        report["servo"] = servo_timing(trace_records)
        report["handoff_lifecycle"] = handoff_lifecycle(
            transactions,
            trace_records,
            joint_source_stamps=parse_joint_source_evidence(grasp_log),
        )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    bag = report["bag"]
    perception = report["perception"]
    planning = report["planning"]
    transactions = report["transactions"]
    lines = [
        "# Mobile handoff benchmark", "",
        f"- Bag: `{bag['path']}`",
        f"- Window: {bag['duration_s']:.3f} s, {bag['message_count']} messages",
        f"- Perception: {perception['succeeded']}/{perception['attempts']} succeeded; p50 {perception['duration_s']['p50']} s, p95 {perception['duration_s']['p95']} s",
        f"- Planning: {planning['succeeded']}/{planning['attempts']} succeeded; p50 {planning['duration_s']['p50']} s, p95 {planning['duration_s']['p95']} s",
        f"- Fresh perception -> plan finish: {transactions['count']} linked; p50 {transactions['duration_s']['p50']} s, p95 {transactions['duration_s']['p95']} s",
        f"- Perception -> planner orchestration gap: p50 {transactions['orchestration_gap_s']['p50']} s, p95 {transactions['orchestration_gap_s']['p95']} s",
        "", "## Bottlenecks", "",
    ]
    lines.extend(f"- {item}" for item in report["bottlenecks"])
    critical = report.get("critical_path_audit")
    if isinstance(critical, dict):
        visualization = critical.get("visualization_and_audit_s", {})
        lines.extend((
            "", "## Critical-path audit", "",
            f"- Plan-ready marker: {critical['plan_ready_marker_status']}",
            f"- Visualization + safety audit: p50 {visualization.get('p50')} s, p95 {visualization.get('p95')} s",
            "- Visualization deferral: not applied; immutable manifest/freeze and executor receipt evidence remain fail-closed",
        ))
    lifecycle = report.get("handoff_lifecycle")
    if isinstance(lifecycle, dict):
        stages = lifecycle["stages"]
        lines.extend((
            "", "## Recorded handoff lifecycle", "",
            f"- Base stop -> fresh perception start: p50 {stages['base_stop_to_fresh_perception_start_s']['p50']} s, p95 {stages['base_stop_to_fresh_perception_start_s']['p95']} s",
            f"- Fresh perception: p50 {stages['fresh_perception_s']['p50']} s, p95 {stages['fresh_perception_s']['p95']} s",
            f"- Perception -> planning gap: p50 {stages['perception_to_planning_gap_s']['p50']} s, p95 {stages['perception_to_planning_gap_s']['p95']} s",
            f"- Planning: p50 {stages['planning_s']['p50']} s, p95 {stages['planning_s']['p95']} s",
            f"- Base stop -> plan finish: p50 {stages['base_stop_to_plan_finish_s']['p50']} s, p95 {stages['base_stop_to_plan_finish_s']['p95']} s",
            f"- Grasp start: {lifecycle['grasp_start_status']}",
        ))
    lines.extend(("", "## Safety evidence", "", "- Offline filesystem analysis only", "- Network/ROS/CAN/WebRTC transports opened: no", "- Motion commands sent: 0", ""))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--sessions-root", required=True, type=Path)
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--grasp-log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--strict-goals", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        bag=args.bag,
        sessions_root=args.sessions_root,
        trace_jsonl=args.trace_jsonl,
        grasp_log=args.grasp_log,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 1 if args.strict_goals and not all(report["goals"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
