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


def collect(root: Path) -> dict[str, object]:
    reports = sorted(root.rglob("report.json"))
    internal: list[float] = []
    successful_internal: list[float] = []
    reused: list[float] = []
    fresh: list[float] = []
    internal_by_session: dict[Path, float] = {}
    failures = 0
    stage_values: dict[str, list[float]] = {}
    for path in reports:
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
                (reused if report.get("grounding_reused") is True else fresh).append(
                    elapsed_s,
                )
                session_dir = path.parent
                if session_dir.name == "perception":
                    session_dir = session_dir.parent
                internal_by_session[session_dir.resolve()] = elapsed_s
        timings = report.get("timings")
        if isinstance(timings, dict):
            for name, value in timings.items():
                if isinstance(name, str) and isinstance(value, (int, float)):
                    stage_values.setdefault(name, []).append(float(value))

    attempts: list[float] = []
    totals: list[float] = []
    wrapper_overhead: list[float] = []
    for log_path in root.rglob("perception.log"):
        session_total: float | None = None
        for event in _json_lines(log_path):
            elapsed = event.get("elapsed_s")
            if not isinstance(elapsed, (int, float)):
                continue
            if event.get("stage") == "perception_attempt":
                attempts.append(float(elapsed))
            elif event.get("stage") == "perception_total":
                session_total = float(elapsed)
                totals.append(session_total)
        internal_elapsed = internal_by_session.get(log_path.parent.resolve())
        if session_total is not None and internal_elapsed is not None:
            wrapper_overhead.append(max(0.0, session_total - internal_elapsed))

    return {
        "schema": "z_manip.perception_latency_benchmark.v1",
        "offline": True,
        "root": str(root.resolve()),
        "reports": len(reports),
        "failures": failures,
        "internal": _summary(internal),
        "successful_internal": _summary(successful_internal),
        "fresh_grounding": _summary(fresh),
        "reused_tracking": _summary(reused),
        "wrapper_attempt": _summary(attempts),
        "wrapper_total": _summary(totals),
        "wrapper_overhead": _summary(wrapper_overhead),
        "stages": {
            name: _summary(values)
            for name, values in sorted(stage_values.items())
        },
        "targets": {
            "perception_ui_total_s": 2.0,
            "internal_under_target": sum(
                value <= 2.0 for value in successful_internal
            ),
            "wrapper_under_target": sum(value <= 2.0 for value in totals),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = collect(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
