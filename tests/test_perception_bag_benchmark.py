from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "perception_bag_benchmark.py"
SPEC = importlib.util.spec_from_file_location("perception_bag_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates_and_empty_is_none() -> None:
    assert MODULE.percentile([], 50) is None
    assert MODULE.percentile([1.0], 95) == 1.0
    assert MODULE.percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert MODULE.percentile([1.0, 2.0, 3.0], 95) == 2.9


def test_nearest_stamp_deltas_are_absolute_seconds() -> None:
    assert MODULE.nearest_stamp_deltas_s([], [1]) == []
    assert MODULE.nearest_stamp_deltas_s([100, 220, 400], []) == []
    assert MODULE.nearest_stamp_deltas_s(
        [1_000_000_000, 2_020_000_000],
        [1_010_000_000, 2_000_000_000],
    ) == pytest.approx([0.01, 0.02])


def test_request_and_manifest_parsers_fail_closed() -> None:
    class Message:
        def __init__(self, data: str) -> None:
            self.data = data

    assert MODULE._request_document(Message("not json")) is None
    assert MODULE._request_document(Message('{"schema":"wrong"}')) is None
    request = MODULE._request_document(
        Message(
            '{"schema":"z_manip.grounding_request.v2",'
            '"request_id":"r1","instruction":"charger"}'
        )
    )
    assert request is not None and request["request_id"] == "r1"

    assert MODULE._manifest_document(Message("{}")) is None
    manifest = MODULE._manifest_document(
        Message(
            '{"schema":"z_manip.tracker_frame.v1",'
            '"result_stamp_ns":123,"track_id":"t1"}'
        )
    )
    assert manifest is not None and manifest["result_stamp_ns"] == 123


def test_bundle_completion_requires_all_six_artifacts() -> None:
    slot = {"messages": {topic: object() for topic in MODULE.BUNDLE_TOPICS}}
    assert MODULE._bundle_complete(slot)
    slot["messages"].pop(MODULE.SCENE_CLOUD_TOPIC)
    assert not MODULE._bundle_complete(slot)


def test_summary_rejects_no_samples_without_inventing_latency() -> None:
    summary = MODULE.summarize_seconds([])
    assert summary == {
        "count": 0,
        "min_s": None,
        "p50_s": None,
        "p95_s": None,
        "max_s": None,
    }
