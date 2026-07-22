"""Continuous fixed-fixture trajectory evidence and smooth clearance cost.

The module is deliberately transport-free.  It samples every joint-space edge
at a bounded resolution through ``FixedSelfCollisionGuard.check_state`` and
returns immutable evidence suitable for planning logs.  A softplus clearance
cost and stable finite-difference gradient provide an optimizer-facing seam
without coupling the collision implementation to CasADi.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np


ARM_DOF = 6


class _WitnessLike(Protocol):
    pair: tuple[str, str]
    distance_m: float
    threshold_m: float
    margin_m: float


class _CollisionStateLike(Protocol):
    valid: bool
    minimum_margin_m: float
    witness: _WitnessLike


class FixedFixtureStateGuard(Protocol):
    def check_state(self, joints: object) -> _CollisionStateLike: ...


@dataclass(frozen=True)
class TrajectoryCollisionWitness:
    pair: tuple[str, str]
    distance_m: float
    threshold_m: float
    margin_m: float

    @classmethod
    def from_state(cls, state: _CollisionStateLike) -> "TrajectoryCollisionWitness":
        witness = state.witness
        return cls(
            pair=tuple(str(name) for name in witness.pair),
            distance_m=float(witness.distance_m),
            threshold_m=float(witness.threshold_m),
            margin_m=float(witness.margin_m),
        )

    def document(self) -> dict[str, object]:
        return {
            "pair": list(self.pair),
            "distance_m": self.distance_m,
            "threshold_m": self.threshold_m,
            "margin_m": self.margin_m,
        }


@dataclass(frozen=True)
class TrajectorySegmentClearance:
    segment_index: int
    interval_count: int
    sample_count: int
    valid: bool
    minimum_margin_m: float
    minimum_alpha: float
    witness: TrajectoryCollisionWitness

    def document(self) -> dict[str, object]:
        return {
            "segment_index": self.segment_index,
            "waypoint_indices": [self.segment_index, self.segment_index + 1],
            "interval_count": self.interval_count,
            "sample_count": self.sample_count,
            "valid": self.valid,
            "minimum_margin_m": self.minimum_margin_m,
            "minimum_alpha": self.minimum_alpha,
            "witness": self.witness.document(),
        }


@dataclass(frozen=True)
class FixedFixtureTrajectoryEvidence:
    valid: bool
    minimum_margin_m: float
    minimum_segment_index: int
    witness: TrajectoryCollisionWitness
    segments: tuple[TrajectorySegmentClearance, ...]
    maximum_joint_step_rad: float
    state_checks: int

    def document(self) -> dict[str, object]:
        return {
            "schema": "z_manip.fixed_fixture_trajectory_clearance.v1",
            "valid": self.valid,
            "minimum_margin_m": self.minimum_margin_m,
            "minimum_segment_index": self.minimum_segment_index,
            "witness": self.witness.document(),
            "continuous_sampling": {
                "maximum_joint_step_rad": self.maximum_joint_step_rad,
                "state_checks": self.state_checks,
            },
            "segments": [segment.document() for segment in self.segments],
        }


@dataclass(frozen=True)
class SmoothClearancePenalty:
    value: float
    gradient: np.ndarray | None
    minimum_margin_m: float
    witness: TrajectoryCollisionWitness
    required_margin_m: float
    softness_m: float
    sample_count: int

    def document(self) -> dict[str, object]:
        return {
            "schema": "z_manip.smooth_clearance_penalty.v1",
            "value": self.value,
            "gradient_available": self.gradient is not None,
            "gradient_shape": (
                None if self.gradient is None else list(self.gradient.shape)
            ),
            "minimum_margin_m": self.minimum_margin_m,
            "witness": self.witness.document(),
            "required_margin_m": self.required_margin_m,
            "softness_m": self.softness_m,
            "sample_count": self.sample_count,
        }


def _trajectory(value: object) -> np.ndarray:
    trajectory = np.asarray(value, dtype=float)
    if (
        trajectory.ndim != 2
        or trajectory.shape[0] < 2
        or trajectory.shape[1] != ARM_DOF
        or not np.isfinite(trajectory).all()
    ):
        raise ValueError("joint trajectory must be a finite Nx6 array with N >= 2")
    return trajectory


def _positive(value: float, *, label: str, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return result


def _interval_counts(trajectory: np.ndarray, maximum_step: float) -> tuple[int, ...]:
    return tuple(
        max(1, int(math.ceil(float(np.linalg.norm(end - start)) / maximum_step)))
        for start, end in zip(trajectory[:-1], trajectory[1:])
    )


def _state_margin(state: _CollisionStateLike) -> float:
    margin = float(state.minimum_margin_m)
    if not math.isfinite(margin):
        raise ValueError("fixed-fixture guard returned a non-finite margin")
    return margin


def evaluate_fixed_fixture_trajectory(
    joint_trajectory: object,
    *,
    guard: FixedFixtureStateGuard,
    max_joint_step_rad: float = 0.01,
) -> FixedFixtureTrajectoryEvidence:
    """Return resolution-bounded collision evidence for every trajectory edge."""

    trajectory = _trajectory(joint_trajectory)
    maximum_step = _positive(max_joint_step_rad, label="maximum joint step")
    intervals = _interval_counts(trajectory, maximum_step)
    segments: list[TrajectorySegmentClearance] = []
    global_margin = math.inf
    global_witness: TrajectoryCollisionWitness | None = None
    global_segment = 0
    state_checks = 0
    for segment_index, (start, end, interval_count) in enumerate(
        zip(trajectory[:-1], trajectory[1:], intervals),
    ):
        minimum_margin = math.inf
        minimum_alpha = 0.0
        minimum_witness: TrajectoryCollisionWitness | None = None
        valid = True
        for alpha in np.linspace(0.0, 1.0, interval_count + 1):
            state = guard.check_state(start + float(alpha) * (end - start))
            state_checks += 1
            margin = _state_margin(state)
            valid = valid and bool(state.valid) and margin > 0.0
            if margin < minimum_margin:
                minimum_margin = margin
                minimum_alpha = float(alpha)
                minimum_witness = TrajectoryCollisionWitness.from_state(state)
        assert minimum_witness is not None
        segment = TrajectorySegmentClearance(
            segment_index=segment_index,
            interval_count=interval_count,
            sample_count=interval_count + 1,
            valid=valid,
            minimum_margin_m=minimum_margin,
            minimum_alpha=minimum_alpha,
            witness=minimum_witness,
        )
        segments.append(segment)
        if minimum_margin < global_margin:
            global_margin = minimum_margin
            global_witness = minimum_witness
            global_segment = segment_index
    assert global_witness is not None
    return FixedFixtureTrajectoryEvidence(
        valid=all(segment.valid for segment in segments),
        minimum_margin_m=global_margin,
        minimum_segment_index=global_segment,
        witness=global_witness,
        segments=tuple(segments),
        maximum_joint_step_rad=maximum_step,
        state_checks=state_checks,
    )


def _sample_margins(
    trajectory: np.ndarray,
    intervals: Sequence[int],
    guard: FixedFixtureStateGuard,
) -> np.ndarray:
    margins: list[float] = []
    first = guard.check_state(trajectory[0])
    margins.append(_state_margin(first))
    for start, end, interval_count in zip(
        trajectory[:-1],
        trajectory[1:],
        intervals,
    ):
        for step in range(1, interval_count + 1):
            alpha = step / interval_count
            margins.append(_state_margin(guard.check_state(start + alpha * (end - start))))
    return np.asarray(margins, dtype=float)


def _penalty_from_margins(
    margins: np.ndarray,
    required_margin_m: float,
    softness_m: float,
) -> float:
    scaled = (required_margin_m - margins) / softness_m
    soft_violation = softness_m * np.logaddexp(0.0, scaled)
    return float(0.5 * np.mean(soft_violation * soft_violation))


def evaluate_smooth_clearance_penalty(
    joint_trajectory: object,
    *,
    guard: FixedFixtureStateGuard,
    required_margin_m: float = 0.02,
    softness_m: float = 0.01,
    max_joint_step_rad: float = 0.01,
    finite_difference_step_rad: float = 1e-5,
    compute_gradient: bool = True,
) -> SmoothClearancePenalty:
    """Evaluate a softplus clearance cost and optional waypoint gradient.

    The interpolation counts are frozen from the nominal trajectory while the
    central finite differences are evaluated.  This prevents a one-sample
    change at a subdivision threshold from polluting the planner gradient.
    """

    trajectory = _trajectory(joint_trajectory)
    required_margin = _positive(
        required_margin_m,
        label="required clearance margin",
        allow_zero=True,
    )
    softness = _positive(softness_m, label="clearance softness")
    maximum_step = _positive(max_joint_step_rad, label="maximum joint step")
    difference_step = _positive(
        finite_difference_step_rad,
        label="finite-difference step",
    )
    intervals = _interval_counts(trajectory, maximum_step)
    margins = _sample_margins(trajectory, intervals, guard)
    value = _penalty_from_margins(margins, required_margin, softness)
    evidence = evaluate_fixed_fixture_trajectory(
        trajectory,
        guard=guard,
        max_joint_step_rad=maximum_step,
    )
    gradient: np.ndarray | None = None
    if compute_gradient:
        gradient = np.zeros_like(trajectory)
        for waypoint_index in range(trajectory.shape[0]):
            for joint_index in range(ARM_DOF):
                plus = trajectory.copy()
                minus = trajectory.copy()
                plus[waypoint_index, joint_index] += difference_step
                minus[waypoint_index, joint_index] -= difference_step
                plus_value = _penalty_from_margins(
                    _sample_margins(plus, intervals, guard),
                    required_margin,
                    softness,
                )
                minus_value = _penalty_from_margins(
                    _sample_margins(minus, intervals, guard),
                    required_margin,
                    softness,
                )
                gradient[waypoint_index, joint_index] = (
                    plus_value - minus_value
                ) / (2.0 * difference_step)
    return SmoothClearancePenalty(
        value=value,
        gradient=gradient,
        minimum_margin_m=evidence.minimum_margin_m,
        witness=evidence.witness,
        required_margin_m=required_margin,
        softness_m=softness,
        sample_count=len(margins),
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
