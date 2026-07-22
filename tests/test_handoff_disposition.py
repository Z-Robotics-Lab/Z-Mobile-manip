from __future__ import annotations

from z_manip.planning.handoff_disposition import (
    NEED_BASE_APPROACH,
    cached_failure_report,
    classify_planning_report,
)


def _report(stages: list[str]) -> dict[str, object]:
    return {
        "plan_valid": False,
        "source_stamp_ns": 123,
        "candidate_count": len(stages),
        "rejection_count": len(stages),
        "rejections_truncated": False,
        "rejections": [{"stage": stage} for stage in stages],
        "timings_s": {"search": 5.6, "total": 6.2},
    }


def test_complete_all_ik_failure_requests_a_changed_base_pose():
    disposition = classify_planning_report(_report(["ik"] * 64))

    assert disposition is not None
    assert disposition.state == NEED_BASE_APPROACH
    assert disposition.retryable is True
    assert disposition.rejection_count == 64


def test_collision_or_truncated_evidence_remains_fail_closed():
    mixed = _report(["ik", "approach_collision"])
    truncated = _report(["ik"] * 4)
    truncated["rejections_truncated"] = True

    assert classify_planning_report(mixed) is None
    assert classify_planning_report(truncated) is None


def test_cached_report_zeroes_search_without_changing_hard_evidence():
    original = _report(["ik"] * 64)
    cached = cached_failure_report(original, elapsed_s=0.002)

    assert cached["planning_disposition"] == NEED_BASE_APPROACH
    assert cached["cached_all_ik_failure"] is True
    assert cached["rejections"] == original["rejections"]
    assert cached["timings_s"]["search"] == 0.0
    assert cached["timings_s"]["total"] == 0.002
