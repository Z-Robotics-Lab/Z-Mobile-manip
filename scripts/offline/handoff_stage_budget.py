#!/usr/bin/env python3
"""Simulate and audit the mobile-handoff latency critical path.

This program is intentionally transport-free.  It consumes JSON benchmark
artifacts, performs arithmetic, and emits JSON.  It never imports ROS or a
robot SDK and cannot issue a motion command.

The simulator keeps two questions separate:

* How fast would the required stages be when fresh perception and passive
  feedback acquisition run concurrently?
* Do recorded timestamps prove that a real grasp executor actually started?

A projection can meet a latency budget without being accepted as live proof.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "z_mobile_manip.handoff_stage_budget.v1"
DEFAULT_GOAL_S = 3.0


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _seconds(value: object, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def simulate(timing: dict[str, Any], *, goal_s: float = DEFAULT_GOAL_S) -> dict[str, Any]:
    """Return the parallel critical path for one timing profile."""
    goal = _seconds(goal_s, "goal_s")
    assert goal is not None
    perception_start = _seconds(
        timing.get("perception_start_after_stop_s", 0.0),
        "perception_start_after_stop_s",
    )
    passive_start = _seconds(
        timing.get("passive_start_after_stop_s", 0.0),
        "passive_start_after_stop_s",
    )
    perception = _seconds(timing.get("fresh_perception_s"), "fresh_perception_s")
    passive = _seconds(timing.get("passive_ready_s"), "passive_ready_s")
    planning_dispatch = _seconds(
        timing.get("planning_dispatch_s", 0.0), "planning_dispatch_s",
    )
    planning = _seconds(timing.get("planning_s"), "planning_s")
    grasp_dispatch = _seconds(
        timing.get("grasp_start_dispatch_s"),
        "grasp_start_dispatch_s",
        optional=True,
    )
    assert all(value is not None for value in (
        perception_start, passive_start, perception, passive,
        planning_dispatch, planning,
    ))

    fresh_ready = perception_start + perception
    passive_ready = passive_start + passive
    prerequisites_ready = max(fresh_ready, passive_ready)
    if abs(fresh_ready - passive_ready) <= 1e-9:
        controlling = "tie"
    elif fresh_ready > passive_ready:
        controlling = "fresh_perception"
    else:
        controlling = "passive_feedback"
    planning_started = prerequisites_ready + planning_dispatch
    plan_finished = planning_started + planning
    grasp_started = (
        plan_finished + grasp_dispatch if grasp_dispatch is not None else None
    )

    # This serial reference represents the same two required read-only jobs
    # launched one after the other at the earliest start offset.  It is only a
    # comparison baseline; it is never used as evidence of execution.
    earliest_start = min(perception_start, passive_start)
    serial_prerequisites = earliest_start + perception + passive
    parallel_saved = max(0.0, serial_prerequisites - prerequisites_ready)
    remaining = goal - plan_finished
    return {
        "goal_s": round(goal, 6),
        "events_s_after_stop": {
            "fresh_data_ready": round(fresh_ready, 6),
            "passive_feedback_ready": round(passive_ready, 6),
            "planning_started": round(planning_started, 6),
            "plan_finished": round(plan_finished, 6),
            "grasp_started": round(grasp_started, 6) if grasp_started is not None else None,
        },
        "critical_prerequisite": controlling,
        "parallel_saved_vs_serial_s": round(parallel_saved, 6),
        "plan_finish_under_goal": plan_finished < goal,
        "plan_finish_shortfall_s": round(max(0.0, plan_finished - goal), 6),
        "remaining_grasp_start_budget_s": round(remaining, 6),
        "grasp_start_timing_known": grasp_started is not None,
        "grasp_start_under_goal": (
            grasp_started < goal if grasp_started is not None else None
        ),
        "grasp_start_shortfall_s": (
            round(max(0.0, grasp_started - goal), 6)
            if grasp_started is not None else None
        ),
    }


def audit_evidence(evidence: dict[str, Any], *, goal_s: float) -> dict[str, Any]:
    """Fail closed unless stop, epochs, successful plan, and grasp start exist."""
    fields = (
        "base_stop_unix_ns",
        "fresh_data_source_unix_ns",
        "passive_source_unix_ns",
        "plan_finished_unix_ns",
        "grasp_started_unix_ns",
    )
    stamps: dict[str, int | None] = {}
    for field in fields:
        try:
            stamps[field] = int(evidence[field])
        except (KeyError, TypeError, ValueError):
            stamps[field] = None

    stop = stamps["base_stop_unix_ns"]
    fresh = stamps["fresh_data_source_unix_ns"]
    passive = stamps["passive_source_unix_ns"]
    plan = stamps["plan_finished_unix_ns"]
    grasp = stamps["grasp_started_unix_ns"]
    checks = {
        "base_stop_observed": stop is not None,
        "fresh_data_is_post_stop": (
            stop is not None and fresh is not None and fresh >= stop
        ),
        "passive_feedback_is_post_stop": (
            stop is not None and passive is not None and passive >= stop
        ),
        "plan_succeeded": evidence.get("plan_succeeded") is True,
        "plan_finished_after_inputs": (
            plan is not None and fresh is not None and passive is not None
            and plan >= max(fresh, passive)
        ),
        "grasp_started_after_plan": (
            grasp is not None and plan is not None and grasp >= plan
        ),
    }
    complete = all(checks.values())
    elapsed = (
        (grasp - stop) / 1e9
        if complete and stop is not None and grasp is not None else None
    )
    return {
        "checks": checks,
        "complete": complete,
        "observed_stop_to_grasp_start_s": round(elapsed, 6) if elapsed is not None else None,
        "observed_under_goal": elapsed < goal_s if elapsed is not None else None,
        "missing_or_invalid": [name for name, passed in checks.items() if not passed],
    }


def recorded_evidence(report: dict[str, Any], *, goal_s: float) -> dict[str, Any]:
    """Audit lifecycle items without upgrading inferred events to evidence."""
    lifecycle = report.get("handoff_lifecycle")
    transactions = report.get("transactions")
    lifecycle_items = lifecycle.get("items", []) if isinstance(lifecycle, dict) else []
    transaction_items = transactions.get("items", []) if isinstance(transactions, dict) else []
    transaction_by_plan = {
        item.get("planning_session_id"): item
        for item in transaction_items if isinstance(item, dict)
    }
    audits = []
    for item in lifecycle_items:
        if not isinstance(item, dict):
            continue
        transaction = transaction_by_plan.get(item.get("planning_session_id"), {})
        # attempt completion proves an artifact was written, not that its RGB-D
        # source epoch was post-stop.  The strict field therefore stays absent
        # unless a source stamp was explicitly recorded.
        evidence = {
            "base_stop_unix_ns": item.get("base_stop_unix_ns"),
            "fresh_data_source_unix_ns": item.get("fresh_data_source_unix_ns"),
            "passive_source_unix_ns": item.get("joint_source_unix_ns"),
            "plan_finished_unix_ns": item.get("planning_finished_unix_ns"),
            "grasp_started_unix_ns": item.get("grasp_started_unix_ns"),
            "plan_succeeded": transaction.get("planning_status") == "succeeded",
        }
        audits.append({
            "perception_session_id": item.get("perception_session_id"),
            "planning_session_id": item.get("planning_session_id"),
            **audit_evidence(evidence, goal_s=goal_s),
        })
    return {
        "transactions": len(audits),
        "complete_evidence": sum(item["complete"] for item in audits),
        "under_goal_with_complete_evidence": sum(
            item["observed_under_goal"] is True for item in audits
        ),
        "items": audits,
    }


def build_report(
    profile: dict[str, Any],
    *,
    handoff_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    goal_s = _seconds(profile.get("goal_s", DEFAULT_GOAL_S), "goal_s")
    assert goal_s is not None
    scenarios = profile.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("profile.scenarios must be a non-empty list")
    projected = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict) or not isinstance(scenario.get("timing_s"), dict):
            raise ValueError(f"scenario {index} must contain timing_s")
        projected.append({
            "name": str(scenario.get("name", f"scenario-{index + 1}")),
            "evidence_class": str(scenario.get("evidence_class", "projection")),
            "source": scenario.get("source"),
            "simulation": simulate(scenario["timing_s"], goal_s=goal_s),
            "evidence": audit_evidence(
                scenario.get("evidence", {})
                if isinstance(scenario.get("evidence"), dict) else {},
                goal_s=goal_s,
            ),
        })
    report = {
        "schema": SCHEMA,
        "offline": True,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "goal_s": goal_s,
        "scenarios": projected,
    }
    if handoff_report is not None:
        report["recorded_evidence"] = recorded_evidence(
            handoff_report, goal_s=goal_s,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--handoff-report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        _load_object(args.profile),
        handoff_report=(
            _load_object(args.handoff_report) if args.handoff_report else None
        ),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
