"""Pure offline tests for proactive whole-body fixed-fixture gating."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from z_manip.control.whole_body_collision import select_collision_safe_arm_step
from z_manip.fixed_self_collision import FixedSelfCollisionGuard


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
COLLISION_MODEL = ROOT / "configs/piper_collision_capsules.json"


def _guard() -> FixedSelfCollisionGuard:
    return FixedSelfCollisionGuard(urdf_path=URDF, model_path=COLLISION_MODEL)


def test_recorded_mid360_entry_is_reduced_to_largest_safe_current_side_step():
    guard = _guard()
    current = np.asarray([-0.049, 0.188, -0.009, -0.045, 0.331, 0.0])
    collision = np.asarray([-0.176, 0.775, 0.003, -0.196, 0.367, -0.096])

    selected = select_collision_safe_arm_step(
        current_joints=current,
        primary_arm_velocity=collision - current,
        horizon_dt_s=1.0,
        tool_lateral_jacobian=np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        guard=guard,
        candidate_improves_task=lambda _velocity: True,
    )

    assert selected.allowed
    assert selected.strategy == "current_side_scale_0.125"
    np.testing.assert_allclose(
        selected.arm_velocity_rps,
        0.125 * (collision - current),
    )
    assert selected.selected_attempt is not None
    assert "mid360" in selected.selected_attempt.pair
    assert selected.selected_attempt.target_margin_m > 0.0


@dataclass(frozen=True)
class _Witness:
    pair: tuple[str, str] = ("mid360", "wrist")
    margin_m: float = -0.01


@dataclass(frozen=True)
class _Decision:
    allowed: bool
    escaping: bool
    reason: str
    witness: _Witness
    current_margin_m: float
    target_margin_m: float


class _SignGuard:
    def check_step(self, current, target):
        allowed = float(np.asarray(target)[0]) < float(np.asarray(current)[0])
        return _Decision(
            allowed=allowed,
            escaping=False,
            reason="opposite side is clear" if allowed else "current side hits mid360",
            witness=_Witness(margin_m=0.02 if allowed else -0.01),
            current_margin_m=0.01,
            target_margin_m=0.02 if allowed else -0.01,
        )


def test_opposite_lateral_candidate_is_tried_after_current_side_scales_fail():
    selected = select_collision_safe_arm_step(
        current_joints=np.zeros(6),
        primary_arm_velocity=np.asarray((0.4, 0.1, 0.0, 0.0, 0.0, 0.0)),
        horizon_dt_s=0.2,
        tool_lateral_jacobian=np.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        guard=_SignGuard(),
        candidate_improves_task=lambda _velocity: True,
    )

    assert selected.allowed
    assert selected.strategy == "opposite_side"
    assert selected.current_side == "left"
    assert selected.selected_side == "right"
    assert selected.arm_velocity_rps[0] < 0.0
    assert [attempt.strategy for attempt in selected.attempts[:5]] == [
        "current_side",
        "current_side_scale_0.750",
        "current_side_scale_0.500",
        "current_side_scale_0.250",
        "current_side_scale_0.125",
    ]


def test_existing_intrusion_only_admits_a_step_that_increases_clearance():
    guard = _guard()
    current = np.asarray([-0.139, 0.313, -0.009, -0.121, 0.358, 0.0])
    clear = np.asarray([-0.049, 0.188, -0.009, -0.045, 0.331, 0.0])

    escape = select_collision_safe_arm_step(
        current_joints=current,
        primary_arm_velocity=clear - current,
        horizon_dt_s=1.0,
        tool_lateral_jacobian=np.zeros(6),
        guard=guard,
        candidate_improves_task=lambda _velocity: True,
    )
    deeper = select_collision_safe_arm_step(
        current_joints=current,
        primary_arm_velocity=current - clear,
        horizon_dt_s=1.0,
        tool_lateral_jacobian=np.zeros(6),
        guard=guard,
        candidate_improves_task=lambda _velocity: True,
    )

    assert escape.allowed
    assert escape.selected_attempt is not None
    assert escape.selected_attempt.escaping
    assert escape.selected_attempt.target_margin_m > escape.selected_attempt.current_margin_m
    # The task direction is unsafe, but the selector must not trap an arm that
    # already starts inside a conservative envelope.  It synthesizes a short
    # clearance-first recovery instead.
    assert deeper.allowed
    assert deeper.strategy.startswith("collision_escape_")
    assert deeper.selected_attempt is not None
    assert deeper.selected_attempt.escaping
    assert deeper.selected_attempt.target_margin_m > deeper.selected_attempt.current_margin_m


def test_recorded_submillimetre_mid360_boundary_synthesizes_escape():
    guard = _guard()
    current = np.asarray([
        -0.08126252997285598,
        0.36140532821046584,
        -0.01005309649148734,
        -0.04991641660703782,
        0.3155206221755349,
        0.0,
    ])

    selected = select_collision_safe_arm_step(
        current_joints=current,
        primary_arm_velocity=np.zeros(6),
        horizon_dt_s=0.2,
        tool_lateral_jacobian=np.zeros(6),
        guard=guard,
        candidate_improves_task=lambda _velocity: False,
    )

    assert selected.allowed
    assert selected.strategy.startswith("collision_escape_")
    assert selected.selected_attempt is not None
    assert selected.selected_attempt.current_margin_m < 0.0
    assert selected.selected_attempt.target_margin_m > 0.0


def test_geometry_safe_candidate_is_rejected_when_task_replay_does_not_improve():
    selected = select_collision_safe_arm_step(
        current_joints=np.zeros(6),
        primary_arm_velocity=np.ones(6) * 0.1,
        horizon_dt_s=0.2,
        tool_lateral_jacobian=np.zeros(6),
        guard=_SignGuard(),
        candidate_improves_task=lambda _velocity: False,
    )

    assert not selected.allowed
    assert selected.strategy == "fail_closed"
    assert all(not attempt.task_improved for attempt in selected.attempts)
    assert np.count_nonzero(selected.arm_velocity_rps) == 0
