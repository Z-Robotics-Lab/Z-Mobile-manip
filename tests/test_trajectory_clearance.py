"""Offline tests for continuous fixed-fixture trajectory clearance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from z_manip.fixed_self_collision import FixedSelfCollisionGuard
from z_manip.trajectory_clearance import (
    evaluate_fixed_fixture_trajectory,
    evaluate_smooth_clearance_penalty,
)


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
COLLISION_MODEL = ROOT / "configs/piper_collision_capsules.json"


def _guard() -> FixedSelfCollisionGuard:
    return FixedSelfCollisionGuard(urdf_path=URDF, model_path=COLLISION_MODEL)


@dataclass(frozen=True)
class _Witness:
    pair: tuple[str, str]
    distance_m: float
    threshold_m: float
    margin_m: float


@dataclass(frozen=True)
class _State:
    valid: bool
    minimum_margin_m: float
    witness: _Witness


class _MidSegmentGuard:
    """Endpoints are clear while the middle of the edge is in collision."""

    def check_state(self, joints):
        q0 = float(np.asarray(joints)[0])
        margin = abs(q0) - 0.20
        witness = _Witness(
            pair=("platform_head", "wrist"),
            distance_m=margin + 0.30,
            threshold_m=0.30,
            margin_m=margin,
        )
        return _State(margin > 0.0, margin, witness)


class _LinearMarginGuard:
    def check_state(self, joints):
        margin = float(np.asarray(joints)[0])
        witness = _Witness(
            pair=("mid360", "forearm"),
            distance_m=margin + 0.20,
            threshold_m=0.20,
            margin_m=margin,
        )
        return _State(margin > 0.0, margin, witness)


def test_continuous_sampling_finds_collision_between_safe_waypoints():
    trajectory = np.zeros((2, 6))
    trajectory[0, 0] = -1.0
    trajectory[1, 0] = 1.0

    evidence = evaluate_fixed_fixture_trajectory(
        trajectory,
        guard=_MidSegmentGuard(),
        max_joint_step_rad=0.05,
    )

    assert not evidence.valid
    assert evidence.minimum_margin_m == pytest.approx(-0.20)
    assert evidence.witness.pair == ("platform_head", "wrist")
    assert evidence.segments[0].minimum_alpha == pytest.approx(0.5)
    assert evidence.segments[0].interval_count == 40
    assert evidence.segments[0].sample_count == 41
    assert evidence.segments[0].document()["valid"] is False


def test_recorded_mid360_path_reports_global_and_per_segment_witnesses():
    clear = np.asarray([-0.049, 0.188, -0.009, -0.045, 0.331, 0.0])
    near = np.asarray([-0.139, 0.313, -0.009, -0.121, 0.358, 0.0])
    collision = np.asarray([-0.176, 0.775, 0.003, -0.196, 0.367, -0.096])

    evidence = evaluate_fixed_fixture_trajectory(
        np.vstack((clear, near, collision)),
        guard=_guard(),
        max_joint_step_rad=0.01,
    )
    document = evidence.document()

    assert not evidence.valid
    assert len(evidence.segments) == 2
    assert evidence.minimum_margin_m < -0.05
    assert "mid360" in evidence.witness.pair
    assert evidence.minimum_segment_index == 1
    assert document["continuous_sampling"]["maximum_joint_step_rad"] == 0.01
    assert document["segments"][1]["minimum_margin_m"] < 0.0


def test_smooth_penalty_returns_finite_difference_gradient_for_planner():
    trajectory = np.zeros((2, 6))
    trajectory[:, 0] = (0.12, 0.08)

    evaluation = evaluate_smooth_clearance_penalty(
        trajectory,
        guard=_LinearMarginGuard(),
        required_margin_m=0.15,
        softness_m=0.02,
        max_joint_step_rad=0.02,
        finite_difference_step_rad=1e-5,
    )
    descended = trajectory - 0.2 * evaluation.gradient
    descended_value = evaluate_smooth_clearance_penalty(
        descended,
        guard=_LinearMarginGuard(),
        required_margin_m=0.15,
        softness_m=0.02,
        max_joint_step_rad=0.02,
        compute_gradient=False,
    ).value

    assert evaluation.value > 0.0
    assert evaluation.gradient.shape == (2, 6)
    assert np.isfinite(evaluation.gradient).all()
    assert np.all(evaluation.gradient[:, 0] < 0.0)
    assert np.count_nonzero(evaluation.gradient[:, 1:]) == 0
    assert descended_value < evaluation.value
    assert evaluation.minimum_margin_m == pytest.approx(0.08)
    assert evaluation.document()["gradient_available"] is True


@pytest.mark.parametrize(
    "trajectory",
    (
        np.zeros((1, 6)),
        np.zeros((2, 5)),
        np.full((2, 6), np.nan),
    ),
)
def test_invalid_trajectory_is_rejected(trajectory):
    with pytest.raises(ValueError, match="trajectory"):
        evaluate_fixed_fixture_trajectory(
            trajectory,
            guard=_LinearMarginGuard(),
        )
