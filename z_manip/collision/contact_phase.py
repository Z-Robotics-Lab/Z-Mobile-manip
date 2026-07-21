"""Geometry-triggered target-contact phases for grasp approaches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from z_manip.planning_control import PlanningControl, checkpoint

from .pointcloud import SegmentCollisionResult


class SegmentCollisionChecker(Protocol):
    """Minimal collision-checker contract needed by an approach audit."""

    def check_segment(
        self,
        start_joints: object,
        end_joints: object,
        *,
        max_joint_step: float | None = None,
        control: PlanningControl | None = None,
    ) -> SegmentCollisionResult:
        ...


@dataclass(frozen=True)
class TargetContactApproachResult:
    """Result of a path audit with a geometry-triggered contact suffix."""

    valid: bool
    reason: str
    contact_entry_segment: int | None = None
    collision: SegmentCollisionResult | None = None


def check_target_contact_approach(
    path: object,
    *,
    no_contact: SegmentCollisionChecker,
    finger_contact: SegmentCollisionChecker,
    allowed_contact_capsules: Sequence[str],
    control: PlanningControl | None = None,
) -> TargetContactApproachResult:
    """Audit an approach while allowing only geometry-observed finger contact.

    ``no_contact`` retains the target as an obstacle. Its first target collision
    is the contact-entry signal, and it may start the contact suffix only when
    every reported capsule belongs to the explicit finger allowlist. The whole
    entry segment and every later segment are then rechecked by
    ``finger_contact``. That checker may ignore those finger/target contacts,
    but must still reject target contact by the palm, wrist, or arm as well as
    every scene and self collision.

    The transition is derived from the target point cloud and robot capsules;
    it is not tied to a waypoint index, approach direction, or object class.
    """

    checkpoint(control, "target-contact approach audit")
    positions = np.asarray(path, dtype=float)
    if (
        positions.ndim != 2
        or len(positions) < 2
        or positions.shape[1] < 1
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("approach path must be a finite (N, dof) array with N >= 2")

    allowed = frozenset(allowed_contact_capsules)
    if (
        any(not isinstance(name, str) or not name for name in allowed)
        or len(allowed) != len(tuple(allowed_contact_capsules))
    ):
        raise ValueError("target contact capsules must be unique non-empty names")

    contact_entry: int | None = None
    for index, (first, second) in enumerate(zip(positions, positions[1:])):
        checkpoint(control, f"target-contact approach segment {index}")
        if contact_entry is None:
            strict = no_contact.check_segment(first, second, control=control)
            if strict.valid:
                continue
            state = strict.state_result
            observed = frozenset(() if state is None else state.capsules)
            if (
                state is None
                or state.kind != "target"
                or not observed
                or not observed.issubset(allowed)
            ):
                return TargetContactApproachResult(
                    False,
                    f"approach segment {index} is blocked before finger contact: "
                    f"{strict.reason}",
                    collision=strict,
                )
            contact_entry = index

        permissive = finger_contact.check_segment(first, second, control=control)
        if not permissive.valid:
            return TargetContactApproachResult(
                False,
                f"approach segment {index} is blocked after finger contact entry: "
                f"{permissive.reason}",
                contact_entry_segment=contact_entry,
                collision=permissive,
            )

    checkpoint(control, "target-contact approach audit")
    return TargetContactApproachResult(
        True,
        "collision-free approach"
        if contact_entry is None
        else f"finger-only target contact from segment {contact_entry}",
        contact_entry_segment=contact_entry,
    )
