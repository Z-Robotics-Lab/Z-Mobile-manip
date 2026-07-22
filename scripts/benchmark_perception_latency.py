#!/usr/bin/env python3
"""Summarize immutable interactive perception latency artifacts.

The benchmark is intentionally offline: it reads report/log files only and
never imports ROS or opens a transport.  It accepts either the perception
session root or a single captured session directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


DEFAULT_BUDGETS = {
    "passive_capture_window_p95_s": 0.30,
    "fresh_internal_p50_s": 1.50,
    "fresh_internal_p95_s": 1.70,
    "fresh_wrapper_overhead_p50_s": 0.20,
    "fresh_wrapper_overhead_p95_s": 0.30,
    "fresh_wrapper_total_p50_s": 1.80,
    "fresh_wrapper_total_p95_s": 2.00,
    "successful_candidate_count_p50_min": 32.0,
}


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "samples": len(samples),
        "min_s": min(samples) if samples else None,
        "p50_s": _percentile(samples, 0.50),
        "p90_s": _percentile(samples, 0.90),
        "p95_s": _percentile(samples, 0.95),
        "max_s": max(samples) if samples else None,
    }


def _value_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "samples": len(samples),
        "min": min(samples) if samples else None,
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "p95": _percentile(samples, 0.95),
        "max": max(samples) if samples else None,
    }


def _json_lines(path: Path) -> Iterable[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _session_dir(path: Path) -> Path:
    parent = path.parent
    if parent.name == "perception":
        parent = parent.parent
    return parent


def _session_selected(path: Path, not_before_session: str | None) -> bool:
    return (
        not_before_session is None
        or _session_dir(path).name >= not_before_session
    )


def collect(
    root: Path,
    *,
    not_before_session: str | None = None,
) -> dict[str, object]:
    reports = sorted(root.rglob("report.json"))
    internal: list[float] = []
    successful_internal: list[float] = []
    reused: list[float] = []
    fresh: list[float] = []
    unclassified: list[float] = []
    candidate_counts: list[float] = []
    passive_capture_windows: list[float] = []
    internal_by_session: dict[Path, float] = {}
    mode_by_session: dict[Path, str] = {}
    failures = 0
    instrumented_reports = 0
    legacy_reports = 0
    stage_values: dict[str, list[float]] = {}
    for path in reports:
        if not _session_selected(path, not_before_session):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict) or report.get("read_only") is not True:
            continue
        elapsed = report.get("elapsed_s")
        failed = bool(report.get("perception_failure"))
        if failed:
            failures += 1
        if isinstance(elapsed, (int, float)):
            elapsed_s = float(elapsed)
            internal.append(elapsed_s)
            if not failed:
                successful_internal.append(elapsed_s)
                if report.get("grounding_reused") is True:
                    reused.append(elapsed_s)
                    grounding_mode = "reused_tracking"
                elif report.get("grounding_reused") is False:
                    fresh.append(elapsed_s)
                    grounding_mode = "fresh_grounding"
                else:
                    unclassified.append(elapsed_s)
                    grounding_mode = "unclassified"
                session_dir = _session_dir(path)
                resolved_session = session_dir.resolve()
                internal_by_session[resolved_session] = elapsed_s
                mode_by_session[resolved_session] = grounding_mode
                candidate_count = report.get("grasp_candidates")
                if (
                    report.get("grasp_generation_valid") is True
                    and isinstance(candidate_count, (int, float))
                ):
                    candidate_counts.append(float(candidate_count))
        passive_capture = report.get("passive_capture")
        if isinstance(passive_capture, dict):
            start_ns = passive_capture.get("observation_start_unix_ns")
            end_ns = passive_capture.get("observation_end_unix_ns")
            if (
                isinstance(start_ns, int)
                and isinstance(end_ns, int)
                and end_ns >= start_ns
            ):
                passive_capture_windows.append((end_ns - start_ns) * 1e-9)
        timings = report.get("timings")
        if isinstance(timings, dict):
            instrumented_reports += 1
            for name, value in timings.items():
                if isinstance(name, str) and isinstance(value, (int, float)):
                    stage_values.setdefault(name, []).append(float(value))
        else:
            legacy_reports += 1

    attempts: list[float] = []
    totals: list[float] = []
    reused_totals: list[float] = []
    fresh_totals: list[float] = []
    wrapper_overhead: list[float] = []
    reused_wrapper_overhead: list[float] = []
    fresh_wrapper_overhead: list[float] = []
    wrapper_stage_values: dict[str, list[float]] = {}
    for log_path in root.rglob("perception.log"):
        if not _session_selected(log_path, not_before_session):
            continue
        session_total: float | None = None
        for event in _json_lines(log_path):
            event_stage = event.get("stage")
            if isinstance(event_stage, str):
                for name, value in event.items():
                    if (
                        name != "elapsed_s"
                        and name.endswith("_s")
                        and isinstance(value, (int, float))
                    ):
                        wrapper_stage_values.setdefault(
                            f"{event_stage}.{name}",
                            [],
                        ).append(float(value))
            elapsed = event.get("elapsed_s")
            if not isinstance(elapsed, (int, float)):
                continue
            if event.get("stage") == "perception_attempt":
                attempts.append(float(elapsed))
            elif event.get("stage") == "perception_total":
                session_total = float(elapsed)
                totals.append(session_total)
        resolved_session = log_path.parent.resolve()
        internal_elapsed = internal_by_session.get(resolved_session)
        if session_total is not None and internal_elapsed is not None:
            overhead = max(0.0, session_total - internal_elapsed)
            wrapper_overhead.append(overhead)
            mode = mode_by_session.get(resolved_session)
            if mode == "reused_tracking":
                reused_totals.append(session_total)
                reused_wrapper_overhead.append(overhead)
            elif mode == "fresh_grounding":
                fresh_totals.append(session_total)
                fresh_wrapper_overhead.append(overhead)

    return {
        "schema": "z_manip.perception_latency_benchmark.v1",
        "offline": True,
        "root": str(root.resolve()),
        "not_before_session": not_before_session,
        "reports": len(internal),
        "failures": failures,
        "instrumentation": {
            "instrumented_reports": instrumented_reports,
            "legacy_reports": legacy_reports,
        },
        "internal": _summary(internal),
        "successful_internal": _summary(successful_internal),
        "fresh_grounding": _summary(fresh),
        "reused_tracking": _summary(reused),
        "unclassified_grounding": _summary(unclassified),
        "successful_candidate_count": _value_summary(candidate_counts),
        "passive_capture_window": _summary(passive_capture_windows),
        "wrapper_attempt": _summary(attempts),
        "wrapper_total": _summary(totals),
        "fresh_grounding_wrapper_total": _summary(fresh_totals),
        "reused_tracking_wrapper_total": _summary(reused_totals),
        "wrapper_overhead": _summary(wrapper_overhead),
        "fresh_grounding_wrapper_overhead": _summary(fresh_wrapper_overhead),
        "reused_tracking_wrapper_overhead": _summary(reused_wrapper_overhead),
        "stages": {
            name: _summary(values)
            for name, values in sorted(stage_values.items())
        },
        "wrapper_stages": {
            name: _summary(values)
            for name, values in sorted(wrapper_stage_values.items())
        },
        "targets": {
            "perception_ui_total_s": 2.0,
            "internal_under_target": sum(
                value <= 2.0 for value in successful_internal
            ),
            "wrapper_under_target": sum(value <= 2.0 for value in totals),
        },
    }


def evaluate_budget(
    result: dict[str, object],
    *,
    budgets: dict[str, float] | None = None,
    minimum_samples: int = 5,
) -> dict[str, object]:
    """Evaluate the two-second target without inventing missing evidence."""

    limits = dict(DEFAULT_BUDGETS if budgets is None else budgets)
    metric_paths = {
        "passive_capture_window_p95_s": ("passive_capture_window", "p95_s", "max"),
        "fresh_internal_p50_s": ("fresh_grounding", "p50_s", "max"),
        "fresh_internal_p95_s": ("fresh_grounding", "p95_s", "max"),
        "fresh_wrapper_overhead_p50_s": (
            "fresh_grounding_wrapper_overhead", "p50_s", "max"
        ),
        "fresh_wrapper_overhead_p95_s": (
            "fresh_grounding_wrapper_overhead", "p95_s", "max"
        ),
        "fresh_wrapper_total_p50_s": (
            "fresh_grounding_wrapper_total", "p50_s", "max"
        ),
        "fresh_wrapper_total_p95_s": (
            "fresh_grounding_wrapper_total", "p95_s", "max"
        ),
        "successful_candidate_count_p50_min": (
            "successful_candidate_count", "p50", "min"
        ),
    }
    checks: dict[str, object] = {}
    passed = True
    for name, limit in limits.items():
        section_name, value_name, direction = metric_paths[name]
        section = result.get(section_name)
        samples = section.get("samples", 0) if isinstance(section, dict) else 0
        measured = section.get(value_name) if isinstance(section, dict) else None
        enough = isinstance(samples, int) and samples >= minimum_samples
        if not enough or not isinstance(measured, (int, float)):
            ok = False
            reason = "insufficient_samples"
        elif direction == "max":
            ok = float(measured) <= limit
            reason = "within_budget" if ok else "over_budget"
        else:
            ok = float(measured) >= limit
            reason = "within_budget" if ok else "below_quality_floor"
        passed = passed and ok
        checks[name] = {
            "passed": ok,
            "reason": reason,
            "samples": samples,
            "measured": measured,
            "limit": limit,
            "direction": direction,
        }
    return {
        "schema": "z_manip.perception_stage_budget.v1",
        "passed": passed,
        "minimum_samples": minimum_samples,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--not-before-session")
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 2 unless every latency and candidate-quality budget passes",
    )
    args = parser.parse_args()
    result = collect(args.root, not_before_session=args.not_before_session)
    result["budget"] = evaluate_budget(
        result,
        minimum_samples=max(1, args.minimum_samples),
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if not args.check or result["budget"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
