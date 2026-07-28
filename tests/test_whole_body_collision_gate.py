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
    # 2026-07-28: `current` re-recorded against the corrected Mid-360 geometry.
    # The pose used here before is itself inside the sensor body, which would
    # send the selector down the escape branch instead of the scale branch this
    # test is about.  See tests/test_lidar_keepout.py.
    current = np.asarray([-0.01715, 0.0658, -0.00315, -0.01575, 0.11585, 0.0])
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
    # 2026-07-28: re-recorded on the corrected Mid-360 envelope.  This is the
    # pose on the recorded strike approach that sits 0.17 mm inside the new
    # boundary; the previous fixture was the 0.1 mm boundary of the superseded
    # capsule and is now 37 mm deep, which is no longer a boundary case.  Do not
    # weaken the assertions below to accommodate a stale pose -- move the pose.
    current = np.asarray([
        -0.032413,
        0.124362,
        -0.005954,
        -0.029768,
        0.218956,
        0.0,
    ])
    assert -0.001 < guard.check_state(current).minimum_margin_m < 0.0

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


# ---------------------------------------------------------------------------
# What the runtime does when the gate authorizes nothing.
#
# The gate judges the ARM: select_collision_safe_arm_step is handed the six arm
# joints and the arm block of the primary velocity, and every capsule it
# measures is anchored to piper_base_link. Zeroing the chassis DOFs too stopped
# the whole robot on evidence about the arm -- recorded 2026-07-28 as 147 of 251
# trace rows with tracking TRUE and base velocity exactly 0.0, one stall of
# 23.3 s with the target a metre away.
#
# These run without casadi or pinocchio, which the whole-body runtime needs and
# which are not importable on this host -- that is the point of testing the
# decision as a pure function rather than through the controller.

from z_manip.control.whole_body_collision import (  # noqa: E402
    CHASSIS_CONTROL_DOF,
    hold_arm_release_chassis,
    hold_whole_body,
)
from z_manip.control.whole_body_model import CONTROL_DOF, CONTROL_NAMES  # noqa: E402


def test_a_blocked_gate_holds_the_arm_and_keeps_the_chassis_intent():
    velocity = np.asarray([0.18, -0.05, 0.01, -0.02, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
    held = hold_arm_release_chassis(velocity)

    np.testing.assert_array_equal(held[:CHASSIS_CONTROL_DOF], velocity[:CHASSIS_CONTROL_DOF])
    np.testing.assert_array_equal(held[CHASSIS_CONTROL_DOF:], np.zeros(CONTROL_DOF - CHASSIS_CONTROL_DOF))
    # The input must not be mutated: the caller still reports the primary intent.
    assert velocity[4] == 0.3


def test_the_split_matches_the_control_vector_layout():
    """The load-bearing assumption, pinned against the model.

    If the control vector is ever reordered, a fixed index-4 split would zero
    the wrong half -- silently commanding the arm while freezing the base, which
    is the exact inverse of this gate's intent and would drive the arm into the
    fixture the gate exists to protect.
    """

    assert len(CONTROL_NAMES) == CONTROL_DOF
    chassis = CONTROL_NAMES[:CHASSIS_CONTROL_DOF]
    arm = CONTROL_NAMES[CHASSIS_CONTROL_DOF:]
    assert chassis == (
        "base_forward_mps",
        "base_yaw_rps",
        "body_roll_rps",
        "body_pitch_rps",
    )
    assert all(name.startswith("piper_joint") for name in arm), arm
    assert len(arm) == 6


def test_a_blocked_gate_never_leaves_arm_authority():
    """Whatever the optimizer wanted, the arm goes to zero.

    When this path runs the measured margin is at or below zero, so no arm step
    is authorized at all -- including the one the QP preferred.
    """

    for magnitude in (0.0, 0.01, 1.0, 1e3):
        velocity = np.full(CONTROL_DOF, magnitude)
        held = hold_arm_release_chassis(velocity)
        assert not np.any(held[CHASSIS_CONTROL_DOF:]), magnitude


def test_a_malformed_control_vector_is_rejected():
    for bad in (np.zeros((2, 10)), np.zeros(4), np.full(10, np.nan)):
        try:
            hold_arm_release_chassis(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted a malformed control vector: {bad!r}")


def test_a_blocked_fixed_fixture_gate_holds_the_whole_body():
    """The one confirmed regression of 2026-07-28, pinned.

    Releasing the chassis while the arm is frozen was geometrically defensible
    -- the gate is handed only the arm joints -- but it broke the visual-servo
    loop: measured on 433 rows of that day's traces, the gate blocked the arm on
    13 and on at least two of them the chassis was still driving at 0.18 m/s, so
    the pose the grasp is computed from is not the pose the servo was
    converging to. On a plug-sized target that is the reported shift.
    """

    velocity = np.asarray([0.18, -0.05, 0.01, -0.02, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
    held = hold_whole_body(velocity)

    np.testing.assert_array_equal(held, np.zeros(CONTROL_DOF))
    # The caller still reports the primary intent, so the input is untouched.
    assert velocity[0] == 0.18


def test_the_runtime_uses_the_whole_body_hold_not_the_arm_only_hold():
    """Bind the runtime to the right primitive.

    ``hold_arm_release_chassis`` is retained as the correct primitive for a gate
    that genuinely only constrains the arm. Wiring it back into the
    fixed-fixture path would silently reinstate the regression, and no
    behavioural test can catch it here because WholeBodyRuntimeController needs
    casadi, which is not importable on this host.
    """

    source = (ROOT / "z_manip" / "control" / "whole_body_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "hold_whole_body(" in source
    assert "hold_arm_release_chassis" not in source
