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
        }) + "\n",
    )

    result = BENCHMARK.collect(tmp_path)

    assert result["internal"]["samples"] == 2
    assert result["reused_tracking"]["p50_s"] == 1.25
    assert result["fresh_grounding"]["p50_s"] == 3.0
    assert result["wrapper_total"]["p50_s"] == 1.75
    assert result["wrapper_overhead"]["p50_s"] == 0.5
    assert result["stages"]["bundle_wait_s"]["p50_s"] == 1.125
    assert result["targets"]["internal_under_target"] == 1
