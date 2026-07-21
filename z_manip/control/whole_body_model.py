"""Reduced whole-body kinematics for Go2W + PiPER shadow control.

The Unitree controller already owns leg-level whole-body control.  The
manipulation stack therefore optimizes only the commands it can actually send:

``[base forward, base yaw, body roll, body pitch, arm qdot(6)]``.

The corresponding configuration still contains planar ``x/y/yaw``, body
height/attitude, and all six PiPER joints.  This makes the non-holonomic base
constraint structural: there is no lateral velocity decision variable.

Pinocchio is imported lazily.  Importing this module on the lightweight UI or
test host cannot pull in Pinocchio, CasADi, ROS, WebRTC, or an actuator SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from z_manip.kinematics.chain import rotation_log


ARM_DOF = 6
STATE_DOF = 6 + ARM_DOF
CONTROL_DOF = 4 + ARM_DOF
STATE_NAMES = (
    "base_x_m",
    "base_y_m",
    "base_yaw_rad",
    "body_height_m",
    "body_roll_rad",
    "body_pitch_rad",
    *(f"piper_joint{index}_rad" for index in range(1, ARM_DOF + 1)),
)
CONTROL_NAMES = (
    "base_forward_mps",
    "base_yaw_rps",
    "body_roll_rps",
    "body_pitch_rps",
    *(f"piper_joint{index}_rps" for index in range(1, ARM_DOF + 1)),
)


class PinocchioWholeBodyUnavailable(RuntimeError):
    """The optional Pinocchio runtime is unavailable or incompatible."""


def _finite_vector(value: Sequence[float], size: int, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must have shape ({size},) and be finite")
    return vector


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


@dataclass(frozen=True)
class ReducedWholeBodyState:
    """Measured reduced state; no desired or synthetic values are allowed."""

    base_x_m: float
    base_y_m: float
    base_yaw_rad: float
    body_height_m: float
    body_roll_rad: float
    body_pitch_rad: float
    arm_joints_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        _finite_vector(self.as_vector(), STATE_DOF, label="whole-body state")

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            (
                self.base_x_m,
                self.base_y_m,
                self.base_yaw_rad,
                self.body_height_m,
                self.body_roll_rad,
                self.body_pitch_rad,
                *self.arm_joints_rad,
            ),
            dtype=float,
        )

    @classmethod
    def from_vector(cls, value: Sequence[float]) -> "ReducedWholeBodyState":
        vector = _finite_vector(value, STATE_DOF, label="whole-body state")
        return cls(
            base_x_m=float(vector[0]),
            base_y_m=float(vector[1]),
            base_yaw_rad=float(vector[2]),
            body_height_m=float(vector[3]),
            body_roll_rad=float(vector[4]),
            body_pitch_rad=float(vector[5]),
            arm_joints_rad=tuple(float(item) for item in vector[6:]),
        )


@dataclass(frozen=True)
class ReducedWholeBodyVelocity:
    """Commands exposed by the high-level optimizer, never actuator messages."""

    base_forward_mps: float
    base_yaw_rps: float
    body_roll_rps: float
    body_pitch_rps: float
    arm_joint_velocity_rps: tuple[float, ...]

    def __post_init__(self) -> None:
        _finite_vector(self.as_vector(), CONTROL_DOF, label="whole-body velocity")

    def as_vector(self) -> np.ndarray:
        return np.asarray(
            (
                self.base_forward_mps,
                self.base_yaw_rps,
                self.body_roll_rps,
                self.body_pitch_rps,
                *self.arm_joint_velocity_rps,
            ),
            dtype=float,
        )

    @classmethod
    def from_vector(cls, value: Sequence[float]) -> "ReducedWholeBodyVelocity":
        vector = _finite_vector(value, CONTROL_DOF, label="whole-body velocity")
        return cls(
            base_forward_mps=float(vector[0]),
            base_yaw_rps=float(vector[1]),
            body_roll_rps=float(vector[2]),
            body_pitch_rps=float(vector[3]),
            arm_joint_velocity_rps=tuple(float(item) for item in vector[4:]),
        )


class WholeBodyKinematics(Protocol):
    """Small seam used by the optimizer and deterministic replay tests."""

    arm_lower_limits: np.ndarray
    arm_upper_limits: np.ndarray
    arm_velocity_limits: np.ndarray
    camera_frame: str
    tool_frame: str

    def frame_pose(self, state: ReducedWholeBodyState, frame: str) -> np.ndarray: ...

    def frame_jacobian(self, state: ReducedWholeBodyState, frame: str) -> np.ndarray: ...

    def arm_manipulability(self, state: ReducedWholeBodyState) -> float: ...

    def target_in_body(
        self,
        state: ReducedWholeBodyState,
        target_world_xyz_m: Sequence[float],
    ) -> np.ndarray: ...

    def integrate(
        self,
        state: ReducedWholeBodyState,
        velocity: ReducedWholeBodyVelocity,
        dt_s: float,
    ) -> ReducedWholeBodyState: ...


@dataclass(frozen=True)
class PinocchioWholeBodyFrames:
    root_frame: str = "base"
    camera_frame: str = "d435_link"
    tool_frame: str = "piper_gripper_base"
    arm_joint_names: tuple[str, ...] = tuple(
        f"piper_joint{index}" for index in range(1, ARM_DOF + 1)
    )


class PinocchioReducedWholeBodyModel:
    """Pinocchio FK/Jacobians wrapped by a virtual Go2W body pose.

    The deployed URDF is reduced by locking the leg, wheel and gripper joints.
    A virtual planar/body transform is applied outside Pinocchio because Go2W's
    SPORT API exposes body pose commands rather than leg joint targets.
    """

    _EPS = 1e-6

    def __init__(
        self,
        urdf_path: str | Path,
        frames: PinocchioWholeBodyFrames | None = None,
        *,
        camera_frame_from_optical: Sequence[Sequence[float]] | None = None,
        tool_from_camera_optical: Sequence[Sequence[float]] | None = None,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as error:  # pragma: no cover - optional runtime
            raise PinocchioWholeBodyUnavailable(
                "Pinocchio whole-body model requires the optional pinocchio package",
            ) from error

        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise ValueError(f"robot URDF does not exist: {self.urdf_path}")
        selected = frames or PinocchioWholeBodyFrames()
        self.root_frame = selected.root_frame
        self.camera_frame = selected.camera_frame
        self.tool_frame = selected.tool_frame
        self.arm_joint_names = selected.arm_joint_names
        if len(self.arm_joint_names) != ARM_DOF:
            raise ValueError("the reduced whole-body model requires six arm joints")

        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        missing = set(self.arm_joint_names) - set(full_model.names)
        if missing:
            raise ValueError(f"URDF is missing arm joints: {sorted(missing)}")
        locked = [
            joint_id
            for joint_id, name in enumerate(full_model.names)
            if joint_id and name not in set(self.arm_joint_names)
        ]
        self.model = pin.buildReducedModel(full_model, locked, pin.neutral(full_model))
        if tuple(self.model.names[1:]) != self.arm_joint_names or self.model.nq != ARM_DOF:
            raise ValueError(
                "reduced Pinocchio joint order does not match requested PiPER joints",
            )
        self.data = self.model.createData()
        self._frame_ids: dict[str, int] = {}
        if camera_frame_from_optical is not None and tool_from_camera_optical is not None:
            raise ValueError("select either a URDF-camera or measured tool-camera extrinsic")
        required_frames = [self.root_frame, self.tool_frame]
        if tool_from_camera_optical is None:
            required_frames.append(self.camera_frame)
        for frame in required_frames:
            frame_id = self.model.getFrameId(frame)
            if frame_id >= self.model.nframes:
                raise ValueError(f"Pinocchio model has no frame {frame!r}")
            self._frame_ids[frame] = frame_id

        self.arm_lower_limits = np.asarray(self.model.lowerPositionLimit, dtype=float)
        self.arm_upper_limits = np.asarray(self.model.upperPositionLimit, dtype=float)
        self.arm_velocity_limits = np.asarray(self.model.velocityLimit, dtype=float)
        transform = np.eye(4)
        if camera_frame_from_optical is not None:
            transform = np.asarray(camera_frame_from_optical, dtype=float)
        elif tool_from_camera_optical is not None:
            transform = np.asarray(tool_from_camera_optical, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera optical extrinsic must be a finite 4x4 matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError("camera optical extrinsic must be homogeneous")
        self.camera_frame_from_optical = transform.copy()
        self._measured_tool_camera = tool_from_camera_optical is not None

    @staticmethod
    def _world_from_root(state: ReducedWholeBodyState) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = _rotation_rpy(
            state.body_roll_rad,
            state.body_pitch_rad,
            state.base_yaw_rad,
        )
        transform[:3, 3] = (
            state.base_x_m,
            state.base_y_m,
            state.body_height_m,
        )
        return transform

    def _pin_pose(self, state: ReducedWholeBodyState, frame: str) -> np.ndarray:
        try:
            frame_id = self._frame_ids[frame]
        except KeyError as error:
            raise ValueError(f"unsupported whole-body frame {frame!r}") from error
        q = _finite_vector(state.arm_joints_rad, ARM_DOF, label="arm joints")
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[frame_id]
        result = np.eye(4)
        result[:3, :3] = np.asarray(placement.rotation, dtype=float)
        result[:3, 3] = np.asarray(placement.translation, dtype=float).reshape(3)
        return result

    def frame_pose(self, state: ReducedWholeBodyState, frame: str) -> np.ndarray:
        pin_frame = self.tool_frame if (
            frame == self.camera_frame and self._measured_tool_camera
        ) else frame
        result = self._world_from_root(state) @ self._pin_pose(state, pin_frame)
        if frame == self.camera_frame:
            result = result @ self.camera_frame_from_optical
        return result

    def frame_jacobian(self, state: ReducedWholeBodyState, frame: str) -> np.ndarray:
        """Return a world-aligned 6x10 geometric velocity Jacobian."""

        pose = self.frame_pose(state, frame)
        jacobian = np.zeros((6, CONTROL_DOF), dtype=float)
        zero = ReducedWholeBodyVelocity.from_vector(np.zeros(CONTROL_DOF))
        # Virtual body columns use the exact state integrator.  This avoids an
        # Euler-rate convention mismatch with SPORT API 1007.
        for index in range(4):
            tangent = zero.as_vector()
            tangent[index] = 1.0
            stepped = self.integrate(
                state,
                ReducedWholeBodyVelocity.from_vector(tangent),
                self._EPS,
            )
            perturbed = self.frame_pose(stepped, frame)
            jacobian[:3, index] = (perturbed[:3, 3] - pose[:3, 3]) / self._EPS
            jacobian[3:, index] = rotation_log(
                perturbed[:3, :3] @ pose[:3, :3].T,
            ) / self._EPS

        q = _finite_vector(state.arm_joints_rad, ARM_DOF, label="arm joints")
        pin_frame = self.tool_frame if (
            frame == self.camera_frame and self._measured_tool_camera
        ) else frame
        frame_id = self._frame_ids[pin_frame]
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        arm = np.asarray(
            self.pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                frame_id,
                self.pin.LOCAL_WORLD_ALIGNED,
            ),
            dtype=float,
        )
        world_rotation = self._world_from_root(state)[:3, :3]
        jacobian[:3, 4:] = world_rotation @ arm[:3]
        jacobian[3:, 4:] = world_rotation @ arm[3:]
        if frame == self.camera_frame:
            # Shift the Pinocchio Jacobian from the URDF/hand-eye parent frame
            # to the optical origin.  The calibrated transform is deliberately
            # external to the URDF and must not be silently replaced by CAD.
            parent_pose = self._world_from_root(state) @ self._pin_pose(state, pin_frame)
            offset_world = (
                parent_pose[:3, :3] @ self.camera_frame_from_optical[:3, 3]
            )
            for column in range(4, CONTROL_DOF):
                jacobian[:3, column] += np.cross(
                    jacobian[3:, column],
                    offset_world,
                )
        return jacobian

    def arm_manipulability(self, state: ReducedWholeBodyState) -> float:
        linear = self.frame_jacobian(state, self.tool_frame)[:3, 4:]
        singular_values = np.linalg.svd(linear, compute_uv=False)
        return float(np.prod(np.maximum(singular_values, 0.0)))

    def target_in_body(
        self,
        state: ReducedWholeBodyState,
        target_world_xyz_m: Sequence[float],
    ) -> np.ndarray:
        """Express a world target in the local Go2W/PiPER root frame."""

        target = _finite_vector(target_world_xyz_m, 3, label="world target")
        root_from_world = np.linalg.inv(self._world_from_root(state))
        return (root_from_world @ np.append(target, 1.0))[:3]

    def integrate(
        self,
        state: ReducedWholeBodyState,
        velocity: ReducedWholeBodyVelocity,
        dt_s: float,
    ) -> ReducedWholeBodyState:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("integration interval must be finite and positive")
        control = velocity.as_vector()
        vector = state.as_vector()
        # Non-holonomic ground motion: only forward speed and yaw rate exist.
        vector[0] += math.cos(state.base_yaw_rad) * control[0] * dt
        vector[1] += math.sin(state.base_yaw_rad) * control[0] * dt
        vector[2] += control[1] * dt
        # Body height is a fixed reference/optional observation, not a SPORT
        # commandable degree of freedom on current Go2W firmware.
        vector[4] += control[2] * dt
        vector[5] += control[3] * dt
        vector[6:] += control[4:] * dt
        vector[6:] = np.clip(
            vector[6:],
            self.arm_lower_limits,
            self.arm_upper_limits,
        )
        return ReducedWholeBodyState.from_vector(vector)


__all__ = [
    "ARM_DOF",
    "CONTROL_DOF",
    "CONTROL_NAMES",
    "STATE_DOF",
    "STATE_NAMES",
    "PinocchioReducedWholeBodyModel",
    "PinocchioWholeBodyFrames",
    "PinocchioWholeBodyUnavailable",
    "ReducedWholeBodyState",
    "ReducedWholeBodyVelocity",
    "WholeBodyKinematics",
]
