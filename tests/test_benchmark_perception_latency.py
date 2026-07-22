from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_perception_latency.py"
SPEC = importlib.util.spec_from_file_location("benchmark_perception_latency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_collect_separates_reused_tracking_and_wrapper_latency(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "report.json").write_text(json.dumps({
        "read_only": True,
        "elapsed_s": 1.25,
        "grounding_reused": True,
        "timings": {"bundle_wait_s": 0.25},
    }))
    (second / "report.json").write_text(json.dumps({
        "read_only": True,
        "elapsed_s": 3.0,
        "grounding_reused": False,
        "timings": {"bundle_wait_s": 2.0},
    }))
    (first / "perception.log").write_text(
        json.dumps({
            "schema": "z_manip.interactive_timing.v1",
            "stage": "perception_total",
            "elapsed_s": 1.75,
            "runner_probe_s": 0.05,
            "passive_capture_s": 0.25,
        }) + "\n",
    )
    (second / "perception.log").write_text(
        json.dumps({
            "schema": "z_manip.interactive_timing.v1",
            "stage": "perception_total",
            "elapsed_s": 4.0,
            "runner_probe_s": 0.15,
            "passive_capture_s": 0.5,
        }) + "\n",
    )

    result = BENCHMARK.collect(tmp_path)

    assert result["internal"]["samples"] == 2
    assert result["reused_tracking"]["p50_s"] == 1.25
    assert result["fresh_grounding"]["p50_s"] == 3.0
    assert result["wrapper_total"]["p50_s"] == 2.875
    assert result["reused_tracking_wrapper_total"]["p50_s"] == 1.75
    assert result["fresh_grounding_wrapper_total"]["p50_s"] == 4.0
    assert result["reused_tracking_wrapper_overhead"]["p50_s"] == 0.5
    assert result["fresh_grounding_wrapper_overhead"]["p50_s"] == 1.0
    assert result["stages"]["bundle_wait_s"]["p50_s"] == 1.125
    assert result["wrapper_stages"][
        "perception_total.runner_probe_s"
    ]["p50_s"] == 0.1
    assert result["wrapper_stages"][
        "perception_total.passive_capture_s"
    ]["p50_s"] == 0.375
    assert result["instrumentation"] == {
        "instrumented_reports": 2,
        "legacy_reports": 0,
    }
    assert result["targets"]["internal_under_target"] == 1


def test_collect_marks_legacy_reports_without_stage_timings(tmp_path):
    session = tmp_path / "legacy"
    session.mkdir()
    (session / "report.json").write_text(json.dumps({
        "read_only": True,
        "elapsed_s": 0.9,
        "grounding_reused": True,
    }))

    result = BENCHMARK.collect(tmp_path)

    assert result["instrumentation"] == {
        "instrumented_reports": 0,
        "legacy_reports": 1,
    }
    assert result["reused_tracking"]["p50_s"] == 0.9
    assert result["reused_tracking_wrapper_total"]["samples"] == 0
