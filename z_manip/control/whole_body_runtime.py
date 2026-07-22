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


def _euler_actuation_available(document: dict[str, Any] | None) -> bool:
    """Return false only after the robot explicitly rejects Euler support."""
    if not isinstance(document, dict):
        return True
    capabilities = document.get("capabilities")
    return not (
        isinstance(capabilities, dict)
        and capabilities.get("euler") is False
    )


def _locked_control_indices(
    *,
    freeze_base: bool,
    euler_available: bool,
    mode: str,
    arm_ready: bool,
) -> tuple[int, ...]:
    """Remove control DOFs that have no confirmed live actuator owner.

    Locking the rejected body attitude DOFs is important: otherwise the QP
    can improve its mathematical objective with roll/pitch commands that the
    Go2W will never execute, starving the PiPER joints of the same view task.
    """
    locked: list[int] = []
    if freeze_base:
        locked.extend((0, 1))
    if not euler_available:
        locked.extend((2, 3))
    if mode == "live" and not arm_ready:
        locked.extend(range(4, 10))
    return tuple(locked)


@dataclass(frozen=True)
class WholeBodyRuntimeCommand:
    base_forward_mps: float
    base_yaw_rps: float
    body_height_target_m: float | None
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
            # The wrist-mounted D435 must retain a usable depth image for the
            # manipulation handoff.  This is a deployed sensor limit, not a
            # generic optimizer tuning knob.
            camera_min_depth_m=0.38,
            base_forward_min_mps=-0.02,
            base_forward_max_mps=0.18,
            base_yaw_max_rps=0.12,
            # Keep the first live integration deliberately smooth.  These are
            # rates, while the NUC still enforces the absolute SPORT envelope.
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
    def _measured_posture(document: dict[str, Any] | None) -> tuple[float, float, bool, str]:
        if not isinstance(document, dict):
            return 0.0, 0.0, False, "NUC posture state unavailable"
        feedback = document.get("feedback")
        attitude = document.get("attitude")
        try:
            values = (
                float(attitude["current_roll_rad"]),
                float(attitude["current_pitch_rad"]),
            )
        except (KeyError, TypeError, ValueError):
            return 0.0, 0.0, False, "NUC measured Euler fields unavailable"
        fresh = (
            document.get("schema") == "z_manip.go2w_posture_status.v1"
            and isinstance(feedback, dict)
            and feedback.get("fresh") is True
            and document.get("stop_latched") is False
            and all(math.isfinite(item) for item in values)
        )
        detail = (
            "measured SPORT roll/pitch; BodyHeight is not a control DOF"
            if fresh
            else "NUC Euler feedback stale or latched"
        )
        return (*values, fresh, detail)

    @staticmethod
    def _arm_executor_ready(document: dict[str, Any] | None) -> tuple[bool, str]:
        if not isinstance(document, dict):
            return False, "PiPER reactive executor status unavailable"
        try:
            updated_ns = int(document["updated_unix_ns"])
        except (KeyError, TypeError, ValueError):
            return False, "PiPER reactive executor timestamp unavailable"
        age_s = (time.time_ns() - updated_ns) / 1e9
        ready = (
            document.get("schema") == "z_manip.piper_reactive_view_status.v1"
            and document.get("owner") == "piper_reactive_view_executor"
            and document.get("ready") is True
            and document.get("stop_latched") is False
            and -0.5 <= age_s <= 0.5
        )
        return ready, (
            f"PiPER reactive executor age {age_s:.3f}s"
            if ready else f"PiPER reactive executor unavailable/stale ({age_s:.3f}s)"
        )

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
        arm_view_status: dict[str, Any] | None = None,
        runtime_state_path: Path,
        mode: str,
        freeze_base: bool = False,
    ) -> WholeBodyRuntimeCommand:
        camera_target = np.asarray(camera_target_xyz_m, dtype=float)
        if camera_target.shape != (3,) or not np.isfinite(camera_target).all() or camera_target[2] <= 0.0:
            raise ValueError("camera target must be a visible finite optical-frame point")
        runtime_state = _json_document(runtime_state_path)
        joints, joints_fresh, joint_detail = self._measured_joints(runtime_state)
        roll, pitch, posture_fresh, posture_detail = self._measured_posture(posture_status)
        euler_available = _euler_actuation_available(posture_status)
        arm_ready, arm_detail = self._arm_executor_ready(arm_view_status)
        state = ReducedWholeBodyState(
            base_x_m=0.0,
            base_y_m=0.0,
            base_yaw_rad=0.0,
            # Current Go2W firmware exposes no commandable BodyHeight.  Keep a
            # fixed virtual root height; all target geometry is body-relative.
            body_height_m=0.0,
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
            locked_control_indices=_locked_control_indices(
                freeze_base=freeze_base,
                euler_available=euler_available,
                mode=mode,
                arm_ready=arm_ready,
            ),
        )
        self.previous_velocity = result.velocity if result.success else None
        velocity = result.velocity
        dt = self.optimizer.config.horizon_dt_s
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
        base_enabled = bool(executable and not freeze_base)
        body_enabled = bool(executable and euler_available)
        arm_enabled = bool(executable and arm_ready)
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
                "arm_executor_fresh": arm_ready,
                "arm_executor_detail": arm_detail,
                "state": asdict(state),
            },
            "transport": {
                # Kept for older dashboard readers; true only when both
                # transports are actually available in this solve.
                "base_and_body_enabled": bool(base_enabled and body_enabled),
                "base_enabled": base_enabled,
                "body_enabled": body_enabled,
                "arm_enabled": arm_enabled,
                "body_reason": (
                    "Euler actuator accepted by the active Go2W service"
                    if body_enabled
                    else (
                        "Euler unavailable; view task reallocated to PiPER"
                        if not euler_available
                        else result.reason
                    )
                ),
                "arm_reason": (
                    "fresh measured PiPER reactive executor owns CAN"
                    if arm_enabled else (
                        result.reason if not result.success else arm_detail
                    )
                ),
                "enabled_dofs": [
                    *(["base_forward", "base_yaw"] if base_enabled else []),
                    *(["body_roll", "body_pitch"] if body_enabled else []),
                    *(list(self.model.arm_joint_names) if arm_enabled else []),
                ],
                "disabled_dofs": {
                    "body_height": "unsupported by current Go2W SPORT firmware",
                    **({
                        "base": (
                            "locked in the close-range handoff zone"
                            if freeze_base else result.reason
                        ),
                    } if not base_enabled else {}),
                    **({
                        "body_roll_pitch": (
                            "Euler(1007) rejected with RPC_ERR_SERVER_API_NOT_IMPL; "
                            "optimizer DOFs locked"
                        ),
                    } if not euler_available else {}),
                    **({
                        "piper_arm": (
                            result.reason if not result.success else arm_detail
                        ),
                    } if not arm_enabled else {}),
                },
            },
            "posture_target": {
                "body_height_delta_m": None,
                "roll_delta_rad": body_roll,
                "pitch_delta_rad": body_pitch,
                "yaw_delta_rad": 0.0,
            },
        })
        return WholeBodyRuntimeCommand(
            base_forward_mps=velocity.base_forward_mps,
            base_yaw_rps=velocity.base_yaw_rps,
            body_height_target_m=None,
            body_roll_target_rad=body_roll,
            body_pitch_target_rad=body_pitch,
            arm_joint_velocity_rps=velocity.arm_joint_velocity_rps,
            executable=executable,
            document=document,
        )

    def reset(self) -> None:
        self.previous_velocity = None


__all__ = ["RUNTIME_SCHEMA", "WholeBodyRuntimeCommand", "WholeBodyRuntimeController"]
