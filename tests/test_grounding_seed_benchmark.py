from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "offline" / "grounding_seed_benchmark.py"
SPEC = importlib.util.spec_from_file_location("grounding_seed_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_correlate_seed_records_uses_exact_stamp_and_token() -> None:
    records = MODULE.correlate_seed_records(
        requests={"r1": {"instruction": "charger", "record_ns": 100}},
        offers=[
            {
                "request_id": "r1",
                "stamp_ns": 123,
                "offer_token": "seed-1",
                "record_ns": 200,
            },
            {
                "request_id": "r1",
                "stamp_ns": 999,
                "offer_token": "seed-missing",
                "record_ns": 300,
            },
        ],
        image_record_ns={123: 190},
        bbox_record_ns={123: 500},
        first_frame_record_ns={"seed-1": 700},
    )
    assert len(records) == 1
    assert records[0]["recorded_init_bbox"] is True
    assert records[0]["instruction"] == "charger"
    assert records[0]["latency_s"] == pytest.approx(
        {
            "request_to_offer": 1e-7,
            "offer_to_init_bbox": 3e-7,
            "request_to_init_bbox": 4e-7,
            "offer_to_first_tracker_frame": 5e-7,
            "init_bbox_to_first_tracker_frame": 2e-7,
        }
    )


def test_correlate_preserves_unmatched_request_as_unknown_negative() -> None:
    records = MODULE.correlate_seed_records(
        requests={},
        offers=[
            {
                "request_id": "earlier-request",
                "stamp_ns": 123,
                "offer_token": "seed-1",
                "record_ns": 200,
            }
        ],
        image_record_ns={123: 190},
        bbox_record_ns={},
        first_frame_record_ns={},
    )
    assert records[0]["instruction"] is None
    assert records[0]["recorded_init_bbox"] is False
    assert all(value is None for value in records[0]["latency_s"].values())


def test_summary_does_not_invent_empty_measurements() -> None:
    assert MODULE.summarize([]) == {
        "count": 0,
        "min": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
