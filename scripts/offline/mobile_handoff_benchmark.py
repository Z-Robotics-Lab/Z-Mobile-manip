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
        if started_ns is None or not start_ns <= started_ns <= end_ns:
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


def enrich_attempts(attempts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for attempt in attempts:
        root = Path(attempt["attempt_path"]).parent
        action = str(attempt.get("action", ""))
        item = dict(attempt)
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
                "perception_to_planning_gap_s": orchestration_gap,
                "perception_to_plan_finish_s": total,
                "failure_code": plan.get("failure_code"),
                "rejection_stages": plan.get("rejection_stages"),
            })
    return result


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
        try:
            stamp_ns = int(item.get("updated_unix_ns", item.get("sample_unix_ns")))
        except (TypeError, ValueError):
            stamp_ns = None
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


def build_report(
    *,
    bag: Path,
    sessions_root: Path,
    trace_jsonl: Path | None = None,
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
    if trace_jsonl is not None:
        report["servo"] = servo_timing(_load_json_stream(trace_jsonl))
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
    lines.extend(("", "## Safety evidence", "", "- Offline filesystem analysis only", "- Network/ROS/CAN/WebRTC transports opened: no", "- Motion commands sent: 0", ""))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--sessions-root", required=True, type=Path)
    parser.add_argument("--trace-jsonl", type=Path)
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
