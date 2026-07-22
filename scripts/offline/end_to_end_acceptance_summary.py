#!/usr/bin/env python3
"""Build a fail-closed, artifact-only mobile-manipulation acceptance summary.

The evaluator consumes reports produced by existing offline replay tools plus
immutable interactive-session artifacts.  It does not import ROS, open a
transport, or invoke a robot SDK.  A process log is never treated as evidence
that execution started: only a receipt whose planning-report and grasp hashes
match an on-disk planning artifact is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "z_mobile_manip.offline_end_to_end_acceptance.v1"
RECEIPT_SCHEMA = "z_manip.piper_stage_receipt.v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso_to_unix_ns(value: object) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp() * 1e9)


def _stage(
    measurement_status: str,
    passed: bool | None,
    basis: str,
    **metrics: Any,
) -> dict[str, Any]:
    return {
        "measurement_status": measurement_status,
        "passed": passed,
        "basis": basis,
        "metrics": metrics,
    }


def _bag_window(bag_report: dict[str, Any]) -> tuple[int | None, int | None]:
    rosbag = bag_report.get("integrity", {}).get("rosbag", {})
    return (
        _int(rosbag.get("starting_time_unix_ns")),
        _int(rosbag.get("ending_time_unix_ns_exclusive")),
    )


def summarize_bag_integrity(report: dict[str, Any]) -> dict[str, Any]:
    rosbag = report.get("integrity", {}).get("rosbag", {})
    missing = rosbag.get("required_topics_missing", [])
    empty = rosbag.get("required_topics_empty", [])
    checks = {
        "offline": report.get("offline") is True,
        "transport_closed": report.get("transport_opened") is False,
        "zero_motion_commands": report.get("motion_commands_sent") == 0,
        "mcap_framing_valid": rosbag.get("framing_valid") is True,
        "required_topics_present": isinstance(missing, list) and not missing,
        "required_topics_nonempty": isinstance(empty, list) and not empty,
        "replay_complete": report.get("complete") is True,
    }
    return _stage(
        "measured",
        all(checks.values()),
        "stopped rosbag metadata, MCAP framing, and offline replay report",
        checks=checks,
        bag_path=rosbag.get("path"),
        message_count=rosbag.get("declared_message_count"),
        topic_count=rosbag.get("topic_count"),
        start_unix_ns=rosbag.get("starting_time_unix_ns"),
        end_unix_ns_exclusive=rosbag.get("ending_time_unix_ns_exclusive"),
    )


def summarize_perception(
    report: dict[str, Any], *, fresh_p95_goal_s: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    fresh = report.get("recorded_fresh", {})
    latency = fresh.get("latency", {})
    requests = _int(fresh.get("requests")) or 0
    exact = _int(fresh.get("exact_request_bundles")) or 0
    p95 = _finite(latency.get("p95_s"))
    exact_complete = requests > 0 and exact == requests
    fresh_passed = exact_complete and p95 is not None and p95 <= fresh_p95_goal_s
    fresh_stage = _stage(
        "measured",
        fresh_passed,
        "recorded request-to-first exact six-artifact bundle",
        requests=requests,
        exact_request_bundles=exact,
        exact_bundle_coverage=(exact / requests if requests else None),
        latency_s=latency,
        p95_goal_s=fresh_p95_goal_s,
        exact_identity_complete=exact_complete,
    )

    tracked = report.get("tracked_counterfactual", {})
    eligible = _int(tracked.get("eligible_same_instruction_requests")) or 0
    exact_cached = _int(tracked.get("exact_cached_identity_bundles")) or 0
    fresh_cached = _int(tracked.get("cached_bundle_age_at_most_0_5_s")) or 0
    tracked_stage = _stage(
        "counterfactual",
        None,
        "exact recorded cache identity evaluated after the fact; no runtime reuse is claimed",
        eligible_same_instruction_requests=eligible,
        exact_cached_identity_bundles=exact_cached,
        cached_bundle_age_at_most_0_5_s=fresh_cached,
        unresolved=tracked.get("unresolved"),
        cached_bundle_age_s=tracked.get("cached_bundle_age"),
        production_fresh_cache_coverage=(fresh_cached / eligible if eligible else None),
    )
    return fresh_stage, tracked_stage


def summarize_close_planning(
    report: dict[str, Any], *, planning_p95_goal_s: float
) -> dict[str, Any]:
    handoff = report.get("summary", {}).get("handoff", {})
    eligible = _int(handoff.get("eligible_trials")) or 0
    succeeded = _int(handoff.get("succeeded")) or 0
    wall = handoff.get("planner_wall_s", {})
    p95 = _finite(wall.get("p95"))
    passed = (
        report.get("offline") is True
        and report.get("motion_commands_sent") == 0
        and report.get("transport_opened") is False
        and eligible > 0
        and succeeded == eligible
        and p95 is not None
        and p95 <= planning_p95_goal_s
    )
    return _stage(
        "measured",
        passed,
        "network-disabled replay trials inside the near-field handoff gate",
        eligible_trials=eligible,
        succeeded=succeeded,
        success_rate=(succeeded / eligible if eligible else None),
        planner_wall_s=wall,
        planner_search_s=handoff.get("planner_search_s"),
        p95_goal_s=planning_p95_goal_s,
    )


def classify_planning_report(report: dict[str, Any]) -> str:
    if report.get("plan_valid") is True:
        return "valid_plan"
    rejections = report.get("rejections")
    count = _int(report.get("rejection_count"))
    if report.get("rejections_truncated") is True:
        return "truncated_failure"
    if not isinstance(rejections, list) or not rejections:
        return "empty_failure"
    if count is None or count != len(rejections):
        return "incomplete_failure"
    stages = {
        item.get("stage")
        for item in rejections
        if isinstance(item, dict)
    }
    if stages == {"ik"}:
        return "exhaustive_all_ik"
    return "mixed_failure"


def _planning_reports(root: Path) -> Iterable[Path]:
    return root.glob("planning/*/artifacts/planning/planning_report.json")


def summarize_all_ik_disposition(
    interactive_root: Path,
    *,
    bag_start_ns: int | None,
    bag_end_ns: int | None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    counts = {
        "valid_plan": 0,
        "exhaustive_all_ik": 0,
        "mixed_failure": 0,
        "truncated_failure": 0,
        "empty_failure": 0,
        "incomplete_failure": 0,
    }
    items: list[dict[str, Any]] = []
    report_by_sha: dict[str, Path] = {}
    for path in sorted(_planning_reports(interactive_root)):
        try:
            report = _load_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        digest = _sha256(path)
        report_by_sha[digest] = path
        source_stamp_ns = _int(report.get("source_stamp_ns"))
        if bag_start_ns is not None and (
            source_stamp_ns is None or source_stamp_ns < bag_start_ns
        ):
            continue
        if bag_end_ns is not None and (
            source_stamp_ns is None or source_stamp_ns >= bag_end_ns
        ):
            continue
        disposition = classify_planning_report(report)
        counts[disposition] += 1
        items.append({
            "planning_report": str(path),
            "planning_report_sha256": digest,
            "source_stamp_ns": source_stamp_ns,
            "disposition": disposition,
            "runtime_action": (
                "NEED_BASE_APPROACH"
                if disposition == "exhaustive_all_ik" else None
            ),
        })
    total = sum(counts.values())
    ambiguous = (
        counts["truncated_failure"]
        + counts["empty_failure"]
        + counts["incomplete_failure"]
    )
    passed = total > 0 and counts["exhaustive_all_ik"] > 0 and ambiguous == 0
    return (
        _stage(
            "measured",
            passed,
            "complete, non-truncated planning reports whose every rejection is IK",
            reports_in_bag_window=total,
            disposition_counts=counts,
            ambiguous_failures=ambiguous,
            items=items,
        ),
        report_by_sha,
    )


def _attempt_for_planning_report(path: Path) -> Path:
    # .../<session>/artifacts/planning/planning_report.json
    return path.parents[2] / "attempt.json"


def _validate_executor_start(
    receipt_path: Path,
    receipt: dict[str, Any],
    *,
    report_by_sha: dict[str, Path],
    interactive_root: Path,
    bag_start_ns: int | None,
    bag_end_ns: int | None,
) -> dict[str, Any]:
    errors: list[str] = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        errors.append("unsupported receipt schema")
    if receipt.get("stage") != "pregrasp":
        errors.append("not a pregrasp executor-start receipt")
    if receipt.get("success") is not True:
        errors.append("receipt did not report success")
    if receipt.get("motion_feedback_verified") is not True:
        errors.append("motion feedback was not verified")
    started_ns = _int(receipt.get("started_unix_ns"))
    if started_ns is None:
        errors.append("missing executor start timestamp")
    if bag_start_ns is not None and started_ns is not None and started_ns < bag_start_ns:
        errors.append("executor start predates bag window")
    if bag_end_ns is not None and started_ns is not None and started_ns >= bag_end_ns:
        errors.append("executor start is outside bag window")

    report_sha = receipt.get("planning_report_sha256")
    report_path = report_by_sha.get(report_sha) if isinstance(report_sha, str) else None
    report: dict[str, Any] = {}
    if report_path is None:
        errors.append("planning_report_sha256 does not match an interactive artifact")
    else:
        report = _load_object(report_path)
        if report.get("plan_valid") is not True:
            errors.append("linked planning report is not valid")
        if receipt.get("planned_grasp_sha256") != report.get("planned_grasp_sha256"):
            errors.append("planned grasp hash mismatch")

    attempt_path = _attempt_for_planning_report(report_path) if report_path else None
    attempt: dict[str, Any] = {}
    perception_path: Path | None = None
    perception: dict[str, Any] = {}
    if attempt_path is None or not attempt_path.exists():
        errors.append("linked planning attempt is missing")
    else:
        attempt = _load_object(attempt_path)
        if attempt.get("status") != "succeeded":
            errors.append("linked planning attempt did not succeed")
        finished_ns = _iso_to_unix_ns(attempt.get("finished_at"))
        if finished_ns is None:
            errors.append("linked planning attempt lacks a finish timestamp")
        if started_ns is not None and finished_ns is not None and started_ns < finished_ns:
            errors.append("executor started before planning finished")
        perception_id = attempt.get("selected_perception_session_id")
        if isinstance(perception_id, str) and perception_id:
            perception_path = (
                interactive_root / "perception" / perception_id
                / "perception" / "report.json"
            )
        if perception_path is None or not perception_path.exists():
            errors.append("linked perception report is missing")
        else:
            perception = _load_object(perception_path)
            if _int(perception.get("stamp_ns")) != _int(report.get("source_stamp_ns")):
                errors.append("perception/planning source stamp mismatch")

    return {
        "measurement_status": "measured" if not errors else "invalid",
        "valid": not errors,
        "errors": errors,
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "executor_started_unix_ns": started_ns,
        "planning_report": str(report_path) if report_path else None,
        "planning_report_sha256": report_sha,
        "planned_grasp_sha256": receipt.get("planned_grasp_sha256"),
        "planning_session_id": (
            report_path.parents[2].name if report_path is not None else None
        ),
        "perception_report": str(perception_path) if perception_path else None,
        "perception_stamp_ns": _int(perception.get("stamp_ns")),
        "source_to_executor_start_s": (
            round((started_ns - int(perception["stamp_ns"])) / 1e9, 6)
            if started_ns is not None and _int(perception.get("stamp_ns")) is not None
            else None
        ),
        "planning_finish_to_executor_start_s": (
            round((started_ns - int(_iso_to_unix_ns(attempt.get("finished_at")))) / 1e9, 6)
            if started_ns is not None and _iso_to_unix_ns(attempt.get("finished_at")) is not None
            else None
        ),
    }


def summarize_executor_start(
    receipts_root: Path,
    *,
    report_by_sha: dict[str, Path],
    interactive_root: Path,
    bag_start_ns: int | None,
    bag_end_ns: int | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in sorted(receipts_root.glob("*/pregrasp-receipt.json")):
        try:
            receipt = _load_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        started_ns = _int(receipt.get("started_unix_ns"))
        if bag_start_ns is not None and (started_ns is None or started_ns < bag_start_ns):
            continue
        if bag_end_ns is not None and (started_ns is None or started_ns >= bag_end_ns):
            continue
        candidates.append(_validate_executor_start(
            path,
            receipt,
            report_by_sha=report_by_sha,
            interactive_root=interactive_root,
            bag_start_ns=bag_start_ns,
            bag_end_ns=bag_end_ns,
        ))
    valid = [item for item in candidates if item["valid"]]
    if valid:
        status = "measured"
        passed: bool | None = True
        basis = "hash-linked pregrasp receipt with verified motion feedback"
    else:
        status = "unmeasured"
        passed = None
        basis = "no valid hash-linked pregrasp executor-start receipt was recorded"
    return _stage(
        status,
        passed,
        basis,
        candidate_receipts=len(candidates),
        valid_linked_starts=len(valid),
        valid_starts=valid,
        invalid_candidates=[item for item in candidates if not item["valid"]],
    )


def build_report(
    *,
    bag_report: dict[str, Any],
    perception_report: dict[str, Any],
    planning_replay_report: dict[str, Any],
    interactive_root: Path,
    receipts_root: Path,
    fresh_p95_goal_s: float = 2.0,
    planning_p95_goal_s: float = 3.0,
    source_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    bag_start_ns, bag_end_ns = _bag_window(bag_report)
    fresh, tracked = summarize_perception(
        perception_report, fresh_p95_goal_s=fresh_p95_goal_s,
    )
    all_ik, report_by_sha = summarize_all_ik_disposition(
        interactive_root,
        bag_start_ns=bag_start_ns,
        bag_end_ns=bag_end_ns,
    )
    stages = {
        "artifact_integrity": summarize_bag_integrity(bag_report),
        "fresh_perception": fresh,
        "tracked_perception": tracked,
        "close_planning": summarize_close_planning(
            planning_replay_report,
            planning_p95_goal_s=planning_p95_goal_s,
        ),
        "all_ik_disposition": all_ik,
        "executor_start": summarize_executor_start(
            receipts_root,
            report_by_sha=report_by_sha,
            interactive_root=interactive_root,
            bag_start_ns=bag_start_ns,
            bag_end_ns=bag_end_ns,
        ),
    }
    mandatory = (
        "artifact_integrity",
        "fresh_perception",
        "close_planning",
        "all_ik_disposition",
        "executor_start",
    )
    incomplete = [
        name for name in mandatory
        if stages[name]["measurement_status"] != "measured"
        or stages[name]["passed"] is None
    ]
    failed = [name for name in mandatory if stages[name]["passed"] is False]
    if incomplete:
        verdict = "incomplete_evidence"
    elif failed:
        verdict = "rejected"
    else:
        verdict = "accepted"
    return {
        "schema": SCHEMA,
        "offline": True,
        "read_only": True,
        "robot_drivers_imported": False,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "verdict": verdict,
        "accepted": verdict == "accepted",
        "mandatory_stages": list(mandatory),
        "failed_stages": failed,
        "incomplete_evidence_stages": incomplete,
        "stages": stages,
        "sources": {
            **(source_paths or {}),
            "interactive_root": str(interactive_root),
            "receipts_root": str(receipts_root),
        },
        "evidence_policy": {
            "tracked_reuse": "counterfactual only; never upgraded to measured runtime latency",
            "executor_start": "requires a pregrasp receipt linked by planning-report and grasp hashes",
            "missing_receipt": "unmeasured, never inferred from process or log presence",
            "all_ik": "requires complete non-truncated rejections and every rejection stage equal to ik",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Offline end-to-end acceptance summary",
        "",
        f"**Verdict: {report['verdict'].upper()}**",
        "",
        "This report is artifact-only and read-only. It did not import robot drivers, open a transport, or send a motion command.",
        "",
        "| Stage | Evidence | Result | Basis |",
        "|---|---|---:|---|",
    ]
    for name, stage in report["stages"].items():
        passed = stage["passed"]
        result = "PASS" if passed is True else "FAIL" if passed is False else "UNMEASURED"
        lines.append(
            f"| `{name}` | {stage['measurement_status'].upper()} | {result} | {stage['basis']} |"
        )
    fresh = report["stages"]["fresh_perception"]["metrics"]
    tracked = report["stages"]["tracked_perception"]["metrics"]
    close = report["stages"]["close_planning"]["metrics"]
    ik = report["stages"]["all_ik_disposition"]["metrics"]
    executor = report["stages"]["executor_start"]["metrics"]
    lines.extend([
        "",
        "## Machine-checkable findings",
        "",
        f"- Fresh exact bundles: `{fresh['exact_request_bundles']}/{fresh['requests']}`; recorded p95 `{fresh['latency_s'].get('p95_s')}` s against `{fresh['p95_goal_s']}` s.",
        f"- Tracked reuse: `{tracked['cached_bundle_age_at_most_0_5_s']}/{tracked['eligible_same_instruction_requests']}` eligible requests had an exact cached identity at most 0.5 s old. This is **COUNTERFACTUAL**, not measured runtime reuse.",
        f"- Close planning: `{close['succeeded']}/{close['eligible_trials']}` succeeded; p95 `{close['planner_wall_s'].get('p95')}` s against `{close['p95_goal_s']}` s.",
        f"- All-IK disposition counts: `{json.dumps(ik['disposition_counts'], sort_keys=True)}`.",
        f"- Strict executor starts: `{executor['valid_linked_starts']}` hash-linked receipt(s).",
    ])
    starts = executor.get("valid_starts", [])
    if starts:
        lines.extend(["", "## Verified executor-start chains", ""])
        for item in starts:
            lines.append(
                "- "
                f"planning `{item['planning_session_id']}` -> pregrasp receipt; "
                f"source-to-start `{item['source_to_executor_start_s']}` s, "
                f"plan-finish-to-start `{item['planning_finish_to_executor_start_s']}` s."
            )
    else:
        lines.extend([
            "",
            "## Executor-start evidence",
            "",
            "**UNMEASURED.** No valid, hash-linked pregrasp receipt was recorded. Process or log presence is not execution evidence.",
        ])
    lines.extend([
        "",
        "## Evidence semantics",
        "",
        "- `MEASURED` means the value exists in a stopped bag, replay report, immutable planning artifact, or cryptographically linked executor receipt.",
        "- `COUNTERFACTUAL` describes an offline reuse opportunity and cannot satisfy a mandatory runtime-evidence gate.",
        "- `UNMEASURED` remains unknown. The evaluator never fabricates a receipt or infers execution from a launched process.",
        "",
    ])
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-replay-report", type=Path, required=True)
    parser.add_argument("--perception-report", type=Path, required=True)
    parser.add_argument("--planning-replay-report", type=Path, required=True)
    parser.add_argument("--interactive-root", type=Path, required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--fresh-p95-goal-s", type=float, default=2.0)
    parser.add_argument("--planning-p95-goal-s", type=float, default=3.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--require-acceptance", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(
        bag_report=_load_object(args.bag_replay_report),
        perception_report=_load_object(args.perception_report),
        planning_replay_report=_load_object(args.planning_replay_report),
        interactive_root=args.interactive_root,
        receipts_root=args.receipts_root,
        fresh_p95_goal_s=args.fresh_p95_goal_s,
        planning_p95_goal_s=args.planning_p95_goal_s,
        source_paths={
            "bag_replay_report": str(args.bag_replay_report),
            "perception_report": str(args.perception_report),
            "planning_replay_report": str(args.planning_replay_report),
        },
    )
    rendered_json = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    rendered_markdown = render_markdown(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json, encoding="utf-8")
    else:
        print(rendered_json, end="")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(rendered_markdown, encoding="utf-8")
    if args.require_acceptance and not report["accepted"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
