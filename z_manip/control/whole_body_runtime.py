"""Measured-state adapter for the Pinocchio/CasADi whole-body controller.

This module deliberately stops at command intents.  The ROS node owns topic
publication and the NUC owns SPORT transport, so importing or testing this
adapter cannot move either robot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .whole_body_model import (
    PinocchioReducedWholeBodyModel,
    PinocchioWholeBodyFrames,
    ReducedWholeBodyState,
    ReducedWholeBodyVelocity,
)
from .whole_body_optimizer import (
    CasadiBoxQP,
    WholeBodyOptimizerConfig,
    WholeBodyShadowOptimizer,
    WholeBodyTask,
)


RUNTIME_SCHEMA = "z_manip.whole_body_runtime.v1"


@dataclass(frozen=True)
class WholeBodyRuntimeCommand:
    base_forward_mps: float
    base_yaw_rps: float
    body_height_target_m: float
    body_roll_target_rad: float
    body_pitch_target_rad: float
    arm_joint_velocity_rps: tuple[float, ...]
    executable: bool
    document: dict[str, Any]


def _json_document(path: Path, *, maximum_bytes: int = 2_000_000) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"required JSON artifact is missing: {resolved}")
    if resolved.stat().st_size > maximum_bytes:
        raise ValueError(f"JSON artifact exceeds bounded size: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {resolved}")
    return value


def _finite_tuple(value: Sequence[float], size: int, *, label: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain {size} finite values")
    return result


class WholeBodyRuntimeController:
    """Turn synchronized perception and measured robot state into one QP step."""

    def __init__(
        self,
        *,
        urdf_path: Path,
        calibration_path: Path,
        desired_standoff_m: float = 0.52,
    ) -> None:
        calibration = _json_document(calibration_path)
        if calibration.get("calibrated") is not True or calibration.get("synthetic") is not False:
            raise ValueError("whole-body runtime requires measured accepted hand-eye calibration")
        tip_from_camera = np.asarray(calibration.get("tip_from_camera"), dtype=float)
        if tip_from_camera.shape != (4, 4) or not np.isfinite(tip_from_camera).all():
            raise ValueError("calibration lacks a finite tip_from_camera transform")
        frames = PinocchioWholeBodyFrames(
            camera_frame=str(calibration.get("camera_frame", "camera_color_optical_frame")),
            tool_frame=str(calibration.get("tip_link", "piper_gripper_base")),
        )
        self.model = PinocchioReducedWholeBodyModel(
            urdf_path,
            frames,
            tool_from_camera_optical=tip_from_camera,
        )
        config = WholeBodyOptimizerConfig(
            horizon_dt_s=0.20,
            handoff_planar_m=float(desired_standoff_m),
            base_forward_min_mps=-0.02,
            base_forward_max_mps=0.18,
            base_yaw_max_rps=0.12,
            # Keep the first live integration deliberately smooth.  These are
            # rates, while the NUC still enforces the absolute SPORT envelope.
            body_height_rate_max_mps=0.035,
            body_roll_rate_max_rps=math.radians(3.0),
            body_pitch_rate_max_rps=math.radians(5.0),
            arm_velocity_scale=0.18,
        )
        self.optimizer = WholeBodyShadowOptimizer(
            self.model,
            config,
            solver=CasadiBoxQP(),
        )
        self.desired_standoff_m = float(desired_standoff_m)
        self.previous_velocity: ReducedWholeBodyVelocity | None = None

    @staticmethod
    def _measured_posture(document: dict[str, Any] | None) -> tuple[float, float, float, bool, str]:
        if not isinstance(document, dict):
            return 0.0, 0.0, 0.0, False, "NUC posture state unavailable"
        feedback = document.get("feedback")
        height = document.get("body_height")
        attitude = document.get("attitude")
        try:
            values = (
                float(height["current_m"]),
                float(attitude["current_roll_rad"]),
                float(attitude["current_pitch_rad"]),
            )
        except (KeyError, TypeError, ValueError):
            return 0.0, 0.0, 0.0, False, "NUC measured posture fields unavailable"
        fresh = (
            document.get("schema") == "z_manip.go2w_posture_status.v1"
            and isinstance(feedback, dict)
            and feedback.get("fresh") is True
            and document.get("stop_latched") is False
            and all(math.isfinite(item) for item in values)
        )
        height_source = str(height.get("source", "unknown")) if isinstance(height, dict) else "unknown"
        detail = (
            f"measured SPORT attitude + height source {height_source}"
            if fresh
            else "NUC posture feedback stale or latched"
        )
        return (*values, fresh, detail)

    @staticmethod
    def _measured_joints(runtime_state: dict[str, Any]) -> tuple[tuple[float, ...], bool, str]:
        try:
            joints = _finite_tuple(runtime_state["joint_positions_rad"], 6, label="PiPER joints")
        except (KeyError, TypeError, ValueError) as error:
            return (0.0,) * 6, False, str(error)
        timestamp_ns = int(runtime_state.get("source_timestamp_ns", 0))
        age_s = (time.time_ns() - timestamp_ns) / 1e9
        fresh = (
            runtime_state.get("schema") == "z_manip.runtime_state.v1"
            and runtime_state.get("joint_state_available") is True
            and -0.5 <= age_s <= 0.75
        )
        return joints, fresh, f"passive joint age {age_s:.3f}s"

    def solve(
        self,
        *,
        camera_target_xyz_m: Sequence[float],
        posture_status: dict[str, Any] | None,
        runtime_state_path: Path,
        mode: str,
    ) -> WholeBodyRuntimeCommand:
        camera_target = np.asarray(camera_target_xyz_m, dtype=float)
        if camera_target.shape != (3,) or not np.isfinite(camera_target).all() or camera_target[2] <= 0.0:
            raise ValueError("camera target must be a visible finite optical-frame point")
        runtime_state = _json_document(runtime_state_path)
        joints, joints_fresh, joint_detail = self._measured_joints(runtime_state)
        height, roll, pitch, posture_fresh, posture_detail = self._measured_posture(posture_status)
        state = ReducedWholeBodyState(
            base_x_m=0.0,
            base_y_m=0.0,
            base_yaw_rad=0.0,
            body_height_m=height,
            body_roll_rad=roll,
            body_pitch_rad=pitch,
            arm_joints_rad=joints,
        )
        target_world = self.model.frame_pose(state, self.model.camera_frame) @ np.append(camera_target, 1.0)
        task = WholeBodyTask(
            target_world_xyz_m=tuple(float(item) for item in target_world[:3]),
            desired_planar_standoff_m=self.desired_standoff_m,
        )
        result = self.optimizer.solve(
            state,
            task,
            previous_velocity=self.previous_velocity,
        )
        self.previous_velocity = result.velocity if result.success else None
        velocity = result.velocity
        dt = self.optimizer.config.horizon_dt_s
        body_height = float(np.clip(
            height + velocity.body_height_mps * dt,
            self.optimizer.config.body_height_min_m,
            self.optimizer.config.body_height_max_m,
        ))
        body_roll = float(np.clip(
            roll + velocity.body_roll_rps * dt,
            -self.optimizer.config.body_roll_abs_max_rad,
            self.optimizer.config.body_roll_abs_max_rad,
        ))
        body_pitch = float(np.clip(
            pitch + velocity.body_pitch_rps * dt,
            -self.optimizer.config.body_pitch_abs_max_rad,
            self.optimizer.config.body_pitch_abs_max_rad,
        ))
        executable = bool(
            mode == "live" and result.success and joints_fresh and posture_fresh
        )
        document = result.status_document()
        document.update({
            "schema": RUNTIME_SCHEMA,
            "mode": mode,
            "executable": executable,
            "measured_state": {
                "posture_fresh": posture_fresh,
                "posture_detail": posture_detail,
                "joints_fresh": joints_fresh,
                "joint_detail": joint_detail,
                "state": asdict(state),
            },
            "transport": {
                "base_and_body_enabled": executable,
                "arm_enabled": False,
                "arm_reason": "first live gate: PiPER qdot remains diagnostic until base/body test passes",
            },
            "posture_target": {
                "body_height_delta_m": body_height,
                "roll_delta_rad": body_roll,
                "pitch_delta_rad": body_pitch,
                "yaw_delta_rad": 0.0,
            },
        })
        return WholeBodyRuntimeCommand(
            base_forward_mps=velocity.base_forward_mps,
            base_yaw_rps=velocity.base_yaw_rps,
            body_height_target_m=body_height,
            body_roll_target_rad=body_roll,
            body_pitch_target_rad=body_pitch,
            arm_joint_velocity_rps=velocity.arm_joint_velocity_rps,
            executable=executable,
            document=document,
        )

    def reset(self) -> None:
        self.previous_velocity = None


__all__ = ["RUNTIME_SCHEMA", "WholeBodyRuntimeCommand", "WholeBodyRuntimeController"]
