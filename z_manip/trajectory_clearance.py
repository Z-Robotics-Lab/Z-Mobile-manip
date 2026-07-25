"""Deprecated import path — moved to :mod:`z_manip.collision.trajectory_clearance`.

Kept as a thin re-export shim so existing ``from z_manip.trajectory_clearance
import ...`` call sites keep working. New code should import from
:mod:`z_manip.collision` (or :mod:`z_manip.collision.trajectory_clearance`)
directly.
"""

from __future__ import annotations

from z_manip.collision.trajectory_clearance import (
    FixedFixtureStateGuard,
    FixedFixtureTrajectoryEvidence,
    SmoothClearancePenalty,
    TrajectoryCollisionWitness,
    TrajectorySegmentClearance,
    evaluate_fixed_fixture_trajectory,
    evaluate_smooth_clearance_penalty,
)

__all__ = [
    "FixedFixtureStateGuard",
    "FixedFixtureTrajectoryEvidence",
    "SmoothClearancePenalty",
    "TrajectoryCollisionWitness",
    "TrajectorySegmentClearance",
    "evaluate_fixed_fixture_trajectory",
    "evaluate_smooth_clearance_penalty",
]
