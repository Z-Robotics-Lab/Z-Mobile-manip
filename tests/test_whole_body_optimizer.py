import math
from pathlib import Path

import numpy as np
import pytest

from z_manip.control.whole_body_model import (
    ARM_DOF,
    CONTROL_DOF,
    PinocchioReducedWholeBodyModel,
    ReducedWholeBodyState,
    ReducedWholeBodyVelocity,
)
from z_manip.control.whole_body_optimizer import (
    CAMERA_DEPTH_BELOW_MINIMUM,
    CasadiBoxQP,
    ScipyReferenceBoxQP,
    WholeBodyOptimizerConfig,
    WholeBodyShadowOptimizer,
    WholeBodyTask,
    WholeBodyVisibilityError,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _ry(pitch):
    c, s = math.cos(pitch), math.sin(pitch)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


class ReplayKinematics:
    """Deterministic reduced geometry; no actuator or middleware imports."""

    camera_frame = "camera_optical"
    tool_frame = "tool"
    arm_lower_limits = np.full(ARM_DOF, -math.pi)
    arm_upper_limits = np.full(ARM_DOF, math.pi)
    arm_velocity_limits = np.full(ARM_DOF, 1.0)

    def frame_pose(self, state, frame):
        q = np.asarray(state.arm_joints_rad)
        transform = np.eye(4)
        yaw = state.base_yaw_rad + 0.35 * q[0]
        pitch = state.body_pitch_rad + 0.25 * q[1]
        if frame == self.camera_frame:
            # Optical z forward, x right, y down at zero posture.
            base_from_optical = np.asarray(
                ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
            )
            transform[:3, :3] = _rz(yaw) @ _ry(pitch) @ base_from_optical
            transform[:3, 3] = (
                state.base_x_m + 0.12 * math.cos(state.base_yaw_rad),
                state.base_y_m + 0.12 * math.sin(state.base_yaw_rad),
                state.body_height_m + 0.32 + 0.04 * q[2],
            )
        elif frame == self.tool_frame:
            transform[:3, :3] = _rz(yaw) @ _ry(pitch)
            local = np.asarray((0.38 + 0.08 * q[2], 0.06 * q[3], 0.18 + 0.06 * q[4]))
            transform[:3, 3] = (
                np.asarray((state.base_x_m, state.base_y_m, state.body_height_m))
                + _rz(state.base_yaw_rad) @ local
            )
        else:
            raise ValueError(frame)
        return transform

    def integrate(self, state, velocity, dt_s):
        value = state.as_vector()
        control = velocity.as_vector()
        value[0] += math.cos(state.base_yaw_rad) * control[0] * dt_s
        value[1] += math.sin(state.base_yaw_rad) * control[0] * dt_s
        value[2] += control[1] * dt_s
        value[3:6] += control[2:5] * dt_s
        value[6:] = np.clip(
            value[6:] + control[5:] * dt_s,
            self.arm_lower_limits,
            self.arm_upper_limits,
        )
        return ReducedWholeBodyState.from_vector(value)

    def frame_jacobian(self, state, frame):
        pose = self.frame_pose(state, frame)
        result = np.zeros((6, CONTROL_DOF))
        epsilon = 1e-6
        for index in range(CONTROL_DOF):
            command = np.zeros(CONTROL_DOF)
            command[index] = 1.0
            moved = self.frame_pose(
                self.integrate(
                    state,
                    ReducedWholeBodyVelocity.from_vector(command),
                    epsilon,
                ),
                frame,
            )
            result[:3, index] = (moved[:3, 3] - pose[:3, 3]) / epsilon
        return result

    def arm_manipulability(self, state):
        q = np.asarray(state.arm_joints_rad)
        return float(0.20 + 0.05 * math.cos(q[1]) * math.cos(q[2]))


def _state(**changes):
    values = dict(
        base_x_m=0.0,
        base_y_m=0.0,
        base_yaw_rad=0.0,
        body_height_m=0.0,
        body_roll_rad=0.0,
        body_pitch_rad=0.0,
        arm_joints_rad=(0.0,) * ARM_DOF,
    )
    values.update(changes)
    return ReducedWholeBodyState(**values)


def test_reduced_state_has_nonholonomic_integration_and_six_arm_joints():
    model = ReplayKinematics()
    state = _state(base_yaw_rad=math.pi / 2)
    velocity = ReducedWholeBodyVelocity.from_vector(
        (0.2, 0.1, -0.02, 0.0, 0.03, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    moved = model.integrate(state, velocity, 0.5)

    assert moved.base_x_m == pytest.approx(0.0, abs=1e-12)
    assert moved.base_y_m == pytest.approx(0.1)
    assert moved.base_yaw_rad == pytest.approx(math.pi / 2 + 0.05)
    assert moved.body_height_m == pytest.approx(-0.01)
    assert len(moved.arm_joints_rad) == 6


def test_far_target_uses_ground_plane_distance_and_commands_forward():
    optimizer = WholeBodyShadowOptimizer(ReplayKinematics())
    state = _state()
    task = WholeBodyTask(target_world_xyz_m=(1.30, 0.0, 0.05))

    result = optimizer.solve(state, task)

    assert result.near_weight == pytest.approx(0.0)
    assert result.planar_distance_m == pytest.approx(1.30)
    assert result.velocity.base_forward_mps > 0.10
    assert result.objective_after < result.objective_before
    assert result.status_document()["motion_commands_sent"] == 0
    assert result.status_document()["transport_opened"] is False


def test_near_low_target_couples_body_and_arm_without_blocking_phase():
    optimizer = WholeBodyShadowOptimizer(ReplayKinematics())
    state = _state()
    task = WholeBodyTask(
        target_world_xyz_m=(0.56, 0.0, -0.12),
        desired_target_height_in_body_m=-0.10,
    )

    result = optimizer.solve(state, task)

    assert result.near_weight > 0.9
    assert result.velocity.body_height_mps < 0.0
    assert np.linalg.norm(result.velocity.arm_joint_velocity_rps) > 0.01
    assert result.objective_after < result.objective_before


def test_locked_untransported_controls_remain_exactly_stationary():
    optimizer = WholeBodyShadowOptimizer(ReplayKinematics())
    result = optimizer.solve(
        _state(),
        WholeBodyTask(target_world_xyz_m=(0.56, 0.0, -0.12)),
        locked_control_indices=(0, 1, 5, 6, 7, 8, 9, 10),
    )

    velocity = result.velocity.as_vector()
    assert velocity[(0, 1, 5, 6, 7, 8, 9, 10),] == pytest.approx(0.0)
    assert np.linalg.norm(velocity[2:5]) > 0.0


def test_invalid_locked_control_index_is_rejected():
    optimizer = WholeBodyShadowOptimizer(ReplayKinematics())
    with pytest.raises(ValueError, match="out of range"):
        optimizer.solve(
            _state(),
            WholeBodyTask(target_world_xyz_m=(0.80, 0.0, 0.0)),
            locked_control_indices=(CONTROL_DOF,),
        )


@pytest.mark.parametrize(
    "target_x_m",
    (
        0.05,  # Behind the camera origin (camera is at base x + 0.12 m).
        0.17,  # In front, but only 0.05 m deep and below the 0.10 m gate.
    ),
)
def test_target_behind_or_too_close_fails_closed_with_stationary_intent(
    target_x_m,
):
    optimizer = WholeBodyShadowOptimizer(ReplayKinematics())
    state = _state()
    task = WholeBodyTask(target_world_xyz_m=(target_x_m, 0.0, 0.0))

    result = optimizer.solve(state, task)
    status = result.status_document()

    assert result.success is False
    assert result.failure_code == CAMERA_DEPTH_BELOW_MINIMUM
    assert result.camera_depth_m <= optimizer.config.camera_min_depth_m
    assert result.velocity.as_vector() == pytest.approx(np.zeros(CONTROL_DOF))
    assert result.predicted_state == state
    assert status["executable_intent"] is False
    assert status["failure_code"] == CAMERA_DEPTH_BELOW_MINIMUM
    assert status["schedule"]["camera_depth_valid"] is False
    assert set(status["intent"].values()) == {0.0}


@pytest.mark.parametrize("target_x_m", (0.05, 0.17))
def test_invalid_camera_depth_cannot_be_linearized(target_x_m):
    optimizer = WholeBodyShadowOptimizer(ReplayKinematics())

    with pytest.raises(WholeBodyVisibilityError) as caught:
        optimizer.linearize(
            _state(),
            WholeBodyTask(target_world_xyz_m=(target_x_m, 0.0, 0.0)),
        )

    assert caught.value.camera_depth_m <= optimizer.config.camera_min_depth_m


def test_box_qp_respects_velocity_and_one_step_joint_position_bounds():
    model = ReplayKinematics()
    state = _state(arm_joints_rad=(math.pi - 0.001, 0.0, 0.0, 0.0, 0.0, 0.0))
    optimizer = WholeBodyShadowOptimizer(
        model,
        WholeBodyOptimizerConfig(horizon_dt_s=0.2),
    )
    problem = optimizer.linearize(
        state,
        WholeBodyTask(target_world_xyz_m=(0.60, 0.20, 0.0)),
    )

    assert problem.upper[5] <= 0.005 + 1e-12
    value = ScipyReferenceBoxQP().solve(problem)
    assert np.all(value >= problem.lower - 1e-9)
    assert np.all(value <= problem.upper + 1e-9)


def test_forty_tick_shadow_replay_converges_without_a_posture_blocking_phase():
    model = ReplayKinematics()
    optimizer = WholeBodyShadowOptimizer(model)
    task = WholeBodyTask(target_world_xyz_m=(1.25, 0.25, -0.08))
    state = _state()
    previous = None
    initial_residual = None
    final_residual = None
    initial_standoff_error = None
    final_standoff_error = None

    for _tick in range(40):
        result = optimizer.solve(state, task, previous_velocity=previous)
        velocity = result.velocity.as_vector()
        problem = optimizer.linearize(state, task, previous_velocity=previous)
        assert np.all(velocity >= problem.lower - 1e-9)
        assert np.all(velocity <= problem.upper + 1e-9)
        assert result.status_document()["mode"] == "shadow"
        assert "phase" not in result.status_document()
        assert optimizer.config.body_height_min_m - 1e-9 <= result.predicted_state.body_height_m
        assert result.predicted_state.body_height_m <= optimizer.config.body_height_max_m + 1e-9
        assert abs(result.predicted_state.body_roll_rad) <= optimizer.config.body_roll_abs_max_rad + 1e-9
        assert abs(result.predicted_state.body_pitch_rad) <= optimizer.config.body_pitch_abs_max_rad + 1e-9
        image_error = float(np.linalg.norm(result.residual_before[:2]))
        standoff_error = abs(
            result.planar_distance_m - task.desired_planar_standoff_m,
        )
        if initial_residual is None:
            initial_residual = image_error
            initial_standoff_error = standoff_error
        final_residual = image_error
        final_standoff_error = standoff_error
        state = result.predicted_state
        previous = result.velocity

    assert final_residual < 0.20 * initial_residual
    assert final_standoff_error < 0.15 * initial_standoff_error


def test_previous_velocity_penalty_reduces_command_discontinuity():
    model = ReplayKinematics()
    optimizer = WholeBodyShadowOptimizer(model)
    state = _state()
    previous = optimizer.solve(
        state,
        WholeBodyTask(target_world_xyz_m=(0.85, 0.30, -0.05)),
    ).velocity
    changed = WholeBodyTask(target_world_xyz_m=(0.85, -0.30, -0.05))

    unsmoothed = optimizer.solve(state, changed).velocity.as_vector()
    smoothed = optimizer.solve(
        state,
        changed,
        previous_velocity=previous,
    ).velocity.as_vector()

    assert np.linalg.norm(smoothed - previous.as_vector()) < np.linalg.norm(
        unsmoothed - previous.as_vector(),
    )


def test_casadi_and_reference_qp_agree_when_optional_runtime_is_present():
    pytest.importorskip("casadi")
    optimizer = WholeBodyShadowOptimizer(ReplayKinematics())
    problem = optimizer.linearize(
        _state(),
        WholeBodyTask(target_world_xyz_m=(0.72, 0.12, -0.05)),
    )

    reference = ScipyReferenceBoxQP().solve(problem)
    casadi = CasadiBoxQP().solve(problem)

    assert casadi == pytest.approx(reference, abs=3e-5)


def test_real_urdf_pinocchio_reduces_to_virtual_body_plus_piper_six():
    pytest.importorskip("pinocchio")
    if not REAL_URDF.is_file():
        pytest.skip("deployed Go2W + PiPER URDF is unavailable")
    model = PinocchioReducedWholeBodyModel(REAL_URDF)
    state = _state()

    camera = model.frame_pose(state, model.camera_frame)
    tool = model.frame_pose(state, model.tool_frame)
    camera_jacobian = model.frame_jacobian(state, model.camera_frame)

    assert model.model.nq == 6
    assert tuple(model.model.names[1:]) == tuple(f"piper_joint{i}" for i in range(1, 7))
    assert camera.shape == tool.shape == (4, 4)
    assert camera_jacobian.shape == (6, 11)
    assert np.isfinite(camera_jacobian).all()
    assert np.count_nonzero(np.linalg.norm(camera_jacobian[:, 5:], axis=0) > 1e-8) >= 4
