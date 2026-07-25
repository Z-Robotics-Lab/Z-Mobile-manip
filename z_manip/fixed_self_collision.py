"""Deprecated import path — moved to :mod:`z_manip.collision.fixed_fixture`.

Kept as a thin re-export shim so existing ``from z_manip.fixed_self_collision
import ...`` call sites keep working. New code should import from
:mod:`z_manip.collision` (or :mod:`z_manip.collision.fixed_fixture`) directly.
"""

from __future__ import annotations

from z_manip.collision.fixed_fixture import (
    CollisionState,
    CollisionWitness,
    FixedCapsule,
    FixedSelfCollisionGuard,
    StepDecision,
)

__all__ = [
    "CollisionState",
    "CollisionWitness",
    "FixedCapsule",
    "FixedSelfCollisionGuard",
    "StepDecision",
]
