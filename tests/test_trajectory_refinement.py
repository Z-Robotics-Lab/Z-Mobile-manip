from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from z_manip.planning.trajectory_refinement import (
    TrajectoryRefinementConfig,
    TrajectoryRefinementUnavailable,
    evaluate_generic_trajectory,
    refine_joint_trajectory,
)


@dataclass(frozen=True)
class _Witness:
    pair: tuple[str, str] = ("arm", "fixture")
    distance_m: float = 1.0
    threshold_m: float = 0.1
    margin_m: float = 0.9


@dataclass(frozen=True)
class _State:
    valid: bool
    minimum_margin_m: float
    witness: _Witness


class _ConstantGuard:
    def __init__(self, margin: float = 0.25) -> None:
        self.margin = float(margin)

    def check_state(self, joints: object) -> _State:
        del joints
        return _State(
            valid=self.margin > 0.0,
            minimum_margin_m=self.margin,
            witness=_Witness(
                distance_m=0.1 + self.margin,
                margin_m=self.margin,
            ),
        )


class _CircularFixtureGuard:
    """A circular forbidden region in the first two joint coordinates."""

    def __init__(self, radius: float = 0.42) -> None:
        self.radius = float(radius)

    def check_state(self, joints: object) -> _State:
        values = np.asarray(joints, dtype=float)
        margin = float(np.linalg.norm(values[:2]) - self.radius)
        return _State(
            valid=margin > 0.0,
            minimum_margin_m=margin,
            witness=_Witness(
                distance_m=self.radius + margin,
                threshold_m=self.radius,
                margin_m=margin,
            ),
        )


LOWER = np.full(6, -2.0)
UPPER = np.full(6, 2.0)


def _zigzag_seed() -> np.ndarray:
    path = np.zeros((6, 6))
    path[:, 0] = (0.0, 0.40, -0.25, 0.50, 0.15, 0.80)
    path[:, 1] = (0.0, -0.25, 0.45, -0.15, 0.35, 0.20)
    return path


def test_refinement_smooths_seed_and_preserves_endpoints_and_limits() -> None:
    seed = _zigzag_seed()
    result = refine_joint_trajectory(
        seed,
        lower_limits=LOWER,
        upper_limits=UPPER,
        state_valid=lambda joints: True,
        fixed_guard=_ConstantGuard(),
        backend="scipy",
    )

    assert result.accepted
    assert result.backend == "scipy-lbfgsb"
    assert result.objective_after < result.objective_before
    assert np.array_equal(result.trajectory[0], seed[0])
    assert np.array_equal(result.trajectory[-1], seed[-1])
    assert np.all(result.trajectory >= LOWER)
    assert np.all(result.trajectory <= UPPER)
    assert result.generic_after.valid
    assert result.fixed_after is not None and result.fixed_after.valid
    assert (
        result.fixed_after.minimum_margin_m
        >= result.fixed_before.minimum_margin_m - 1e-9  # type: ignore[union-attr]
    )
    assert not result.trajectory.flags.writeable


def test_refinement_never_cuts_through_generic_configuration_obstacle() -> None:
    seed = np.zeros((5, 6))
    seed[:, :2] = (
        (-0.80, 0.0),
        (-0.55, 0.65),
        (0.0, 0.82),
        (0.55, 0.65),
        (0.80, 0.0),
    )

    def state_valid(joints: np.ndarray) -> bool:
        return bool(np.linalg.norm(joints[:2]) > 0.42)

    result = refine_joint_trajectory(
        seed,
        lower_limits=LOWER,
        upper_limits=UPPER,
        state_valid=state_valid,
        fixed_guard=_ConstantGuard(),
        backend="scipy",
        config=TrajectoryRefinementConfig(max_joint_step_rad=0.015),
    )

    evidence = evaluate_generic_trajectory(
        result.trajectory,
        state_valid=state_valid,
        max_joint_step_rad=0.015,
    )
    assert evidence.valid
    assert np.array_equal(result.trajectory[0], seed[0])
    assert np.array_equal(result.trajectory[-1], seed[-1])


def test_refinement_never_reduces_fixed_fixture_minimum_margin() -> None:
    seed = np.zeros((5, 6))
    seed[:, :2] = (
        (-0.80, 0.0),
        (-0.55, 0.65),
        (0.0, 0.82),
        (0.55, 0.65),
        (0.80, 0.0),
    )
    guard = _CircularFixtureGuard()
    result = refine_joint_trajectory(
        seed,
        lower_limits=LOWER,
        upper_limits=UPPER,
        fixed_guard=guard,
        backend="scipy",
        config=TrajectoryRefinementConfig(max_joint_step_rad=0.015),
    )

    assert result.fixed_before is not None
    assert result.fixed_after is not None
    assert result.fixed_after.valid
    assert (
        result.fixed_after.minimum_margin_m
        >= result.fixed_before.minimum_margin_m - 1e-9
    )


def test_rejected_candidate_returns_exact_seed_with_auditable_evidence() -> None:
    seed = _zigzag_seed()
    result = refine_joint_trajectory(
        seed,
        lower_limits=LOWER,
        upper_limits=UPPER,
        state_valid=lambda joints: False,
        fixed_guard=_ConstantGuard(),
        backend="scipy",
    )

    assert not result.accepted
    assert np.array_equal(result.trajectory, seed)
    assert result.objective_after == result.objective_before
    assert not result.generic_after.valid
    assert result.document()["schema"] == "z_manip.trajectory_refinement.v1"


def test_two_point_seed_is_a_safe_noop() -> None:
    seed = np.zeros((2, 6))
    seed[-1, 0] = 0.4
    result = refine_joint_trajectory(
        seed,
        lower_limits=LOWER,
        upper_limits=UPPER,
        fixed_guard=_ConstantGuard(),
        backend="scipy",
    )
    assert not result.accepted
    assert np.array_equal(result.trajectory, seed)
    assert result.reason == "smoothing objective did not improve"


def test_invalid_seed_limits_and_backend_are_rejected() -> None:
    seed = _zigzag_seed()
    outside = seed.copy()
    outside[2, 0] = 3.0
    with pytest.raises(ValueError, match="violates joint limits"):
        refine_joint_trajectory(
            outside,
            lower_limits=LOWER,
            upper_limits=UPPER,
        )
    with pytest.raises(ValueError, match="backend"):
        refine_joint_trajectory(
            seed,
            lower_limits=LOWER,
            upper_limits=UPPER,
            backend="unknown",
        )


def test_explicit_casadi_backend_is_used_or_fails_loudly() -> None:
    seed = _zigzag_seed()
    try:
        result = refine_joint_trajectory(
            seed,
            lower_limits=LOWER,
            upper_limits=UPPER,
            fixed_guard=_ConstantGuard(),
            backend="casadi",
        )
    except TrajectoryRefinementUnavailable as error:
        assert "CasADi" in str(error)
    else:
        assert result.backend.startswith("casadi-")

