"""Typed, evidence-backed dispositions for close-range grasp planning.

The mobile controller must distinguish an exhaustive inverse-kinematics miss
from a malformed session or a collision failure.  The former means that the
base should acquire a meaningfully different manipulation pose before asking
the arm planner to solve the same problem again.  It is not permission to
relax URDF limits, IK tolerances, or collision checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


NEED_BASE_APPROACH = "NEED_BASE_APPROACH"


@dataclass(frozen=True)
class PlanningDisposition:
    """Recoverable planning outcome proven by a complete rejection report."""

    state: str
    reason: str
    retryable: bool
    source_stamp_ns: int
    rejection_count: int

    def document(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "retryable": self.retryable,
            "source_stamp_ns": self.source_stamp_ns,
            "rejection_count": self.rejection_count,
        }


def classify_planning_report(
    report: Mapping[str, Any] | object,
) -> PlanningDisposition | None:
    """Return ``NEED_BASE_APPROACH`` only for a complete all-IK failure.

    A single IK miss, truncated diagnostics, aperture failure, or collision
    failure remains an ordinary fail-closed planner error.  Requiring one
    rejection for every evaluated hypothesis prevents a partial/timeout
    report from being misclassified as recoverable.
    """

    if not isinstance(report, Mapping) or report.get("plan_valid") is not False:
        return None
    if report.get("rejections_truncated") is not False:
        return None
    candidate_count = report.get("candidate_count")
    rejection_count = report.get("rejection_count")
    source_stamp_ns = report.get("source_stamp_ns")
    rejections = report.get("rejections")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or candidate_count < 1
        or not isinstance(rejection_count, int)
        or isinstance(rejection_count, bool)
        or rejection_count < candidate_count
        or not isinstance(source_stamp_ns, int)
        or isinstance(source_stamp_ns, bool)
        or source_stamp_ns <= 0
        or not isinstance(rejections, list)
        or len(rejections) != rejection_count
    ):
        return None
    if not all(
        isinstance(rejection, Mapping) and rejection.get("stage") == "ik"
        for rejection in rejections
    ):
        return None
    return PlanningDisposition(
        state=NEED_BASE_APPROACH,
        reason="exhaustive_all_ik_failed",
        retryable=True,
        source_stamp_ns=source_stamp_ns,
        rejection_count=rejection_count,
    )


def cached_failure_report(
    report: Mapping[str, Any],
    *,
    elapsed_s: float,
) -> dict[str, Any]:
    """Return a bounded retry report without repeating the exhaustive solve."""

    disposition = classify_planning_report(report)
    if disposition is None:
        raise ValueError("report is not a complete all-IK failure")
    result = dict(report)
    result.update({
        "planning_disposition": disposition.state,
        "planning_retryable": disposition.retryable,
        "planning_disposition_evidence": disposition.document(),
        "cached_all_ik_failure": True,
        "error": (
            "NEED_BASE_APPROACH: this immutable source and measured joint "
            "state already exhausted every IK hypothesis; acquire a changed "
            "base pose before retrying"
        ),
    })
    timings = dict(result.get("timings_s") or {})
    timings.update({
        "search": 0.0,
        "diagnostic_replay": 0.0,
        "total": max(0.0, float(elapsed_s)),
        "cached_failure_lookup": max(0.0, float(elapsed_s)),
    })
    result["timings_s"] = timings
    return result


__all__ = [
    "NEED_BASE_APPROACH",
    "PlanningDisposition",
    "cached_failure_report",
    "classify_planning_report",
]
