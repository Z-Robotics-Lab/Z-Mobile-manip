#!/usr/bin/env python3
"""Minimal EdgeTAM-to-Go2W depth visual-servo runtime.

The node subscribes to EdgeTAM's selected 3-D target and publishes bounded
body velocity commands.  ``shadow`` mode computes and reports the command but
never publishes it.  ``live`` mode publishes on the existing guarded command
path; this module contains no Unitree/WebRTC transport.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import statistics
import threading
import time
from typing import Any

import numpy as np

from z_manip.control.reactive_servo import (
    ArmViewIntent,
    BaseMotionIntent,
    PostureIntent,
    ReactivePhase,
    ReactiveServoConfig,
    ReactiveServoDecision,
    ReactiveTargetController,
    TargetGeometry,
)
from z_manip.control.visual_servo import VisualServoConfig, VisualServoController


STATUS_SCHEMA = "z_manip.depth_servo_status.v1"


@dataclass(frozen=True)
class DepthServoSettings:
    mode: str = "shadow"
    desired_depth_m: float = 0.50
    # PiPER can solve the final centimetres.  The legged base only needs to
    # enter a coarse near-field corridor; demanding camera-perfect alignment
    # makes body sway repeatedly reset the handoff window.
    depth_tolerance_m: float = 0.01
    lateral_tolerance_m: float = 0.12
    settle_time_s: float = 0.10
    handoff_depth_m: float = 0.52
    handoff_bearing_rad: float = math.radians(20.0)
    linear_gain: float = 0.65
    yaw_gain: float = 0.70
    # Go2W's low-speed gait is inconsistent around 0.05--0.10 m/s: the API
    # accepts the command while the body can stop making forward progress.
    # Cruise briskly in the far field and keep a gait-maintaining floor until
    # the coarse handoff cone is reached.
    min_forward_mps: float = 0.10
    max_forward_mps: float = 0.18
    max_reverse_mps: float = 0.05
    max_yaw_rps: float = 0.12
    rotate_only_bearing_rad: float = math.radians(25.0)
    yaw_deadband_rad: float = math.radians(6.0)
    target_timeout_s: float = 0.25
    tracking_loss_grace_s: float = 0.75
    target_filter_window: int = 5
    target_filter_alpha: float = 0.55
    max_target_jump_m: float = 0.20
    outlier_rebase_samples: int = 3
    outlier_rebase_spread_m: float = 0.05
    base_frame: str = "base_link"
    arm_base_frame: str = "piper_base_link"
    transform_timeout_s: float = 0.25
    # Explicit test-only compatibility seam. Deployed ROS construction always
    # leaves this false: missing transforms must never fall back to optical z.
    allow_legacy_optical_depth_for_tests: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "live"}:
            raise ValueError("mode must be shadow or live")
        if not math.isfinite(self.target_timeout_s) or self.target_timeout_s <= 0.0:
            raise ValueError("target timeout must be finite and positive")
        if not math.isfinite(self.tracking_loss_grace_s) or self.tracking_loss_grace_s < self.target_timeout_s:
            raise ValueError("tracking-loss grace must be at least the target timeout")
        if self.target_filter_window < 1:
            raise ValueError("target filter window must be positive")
        if not 0.0 < self.target_filter_alpha <= 1.0:
            raise ValueError("target filter alpha must be in (0, 1]")
        if not math.isfinite(self.max_target_jump_m) or self.max_target_jump_m <= 0.0:
            raise ValueError("maximum target jump must be finite and positive")
        if self.outlier_rebase_samples < 2:
            raise ValueError("outlier rebase requires at least two samples")
        if (
            not math.isfinite(self.outlier_rebase_spread_m)
            or self.outlier_rebase_spread_m <= 0.0
        ):
            raise ValueError("outlier rebase spread must be finite and positive")
        if not self.base_frame.strip() or not self.arm_base_frame.strip():
            raise ValueError("base and arm-base frames must be non-empty")
        if not math.isfinite(self.transform_timeout_s) or self.transform_timeout_s <= 0.0:
            raise ValueError("transform timeout must be finite and positive")
        if not math.isfinite(self.handoff_depth_m) or self.handoff_depth_m <= 0.0:
            raise ValueError("handoff depth must be finite and positive")
        if (
            not math.isfinite(self.min_forward_mps)
            or not 0.0 < self.min_forward_mps <= self.max_forward_mps
        ):
            raise ValueError("minimum forward speed must be in (0, max_forward_mps]")
        if (
            not math.isfinite(self.handoff_bearing_rad)
            or not 0.0 < self.handoff_bearing_rad < math.pi / 2.0
        ):
            raise ValueError("handoff bearing must be in (0, pi/2)")


@dataclass(frozen=True)
class DepthServoOutput:
    phase: str
    proposed_linear_x: float
    proposed_angular_z: float
    published_linear_x: float
    published_angular_z: float
    depth_error_m: float | None
    yaw_error_rad: float | None
    target_age_s: float | None
    done: bool = False
    reason: str = ""
    reactive_phase: str | None = None
    needs_ik_probe: bool = False


def _rigid_transform_matrix(
    translation_xyz: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """Build a target-from-source transform from a ROS-style transform."""

    translation = np.asarray(translation_xyz, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if (
        translation.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(translation).all()
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError("transform components must be finite xyz and xyzw values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("transform quaternion must have non-zero norm")
    x, y, z, w = quaternion / norm
    rotation = np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _validated_matrix(value: object, *, name: str) -> np.ndarray:
    """Accept only a finite right-handed rigid 4x4 transform."""

    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-4):
        raise ValueError(f"{name} rotation is not right-handed")
    return matrix


def _runtime_state_transforms(
    path: Path,
    *,
    source_frame: str,
    base_frame: str,
    arm_base_frame: str,
    now_unix_ns: int,
    max_age_s: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read fresh verified model transforms from the subscribe-only observer.

    This is the deployed fallback for ROS graphs that publish RealSense TF but
    not the combined Go2W/PiPER model frames.  It is deliberately stricter
    than ordinary JSON loading: an old artifact, synthetic calibration, frame
    mismatch, or malformed rigid transform stops the base.
    """

    artifact = path.expanduser().resolve()
    if not artifact.is_file():
        raise ValueError(f"runtime observer state is missing: {artifact}")
    if artifact.stat().st_size > 2_000_000:
        raise ValueError("runtime observer state exceeds the bounded size")
    document = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "z_manip.runtime_state.v1":
        raise ValueError("runtime observer state schema is invalid")
    transforms = document.get("kinematic_transforms")
    if not isinstance(transforms, dict):
        raise ValueError("runtime observer has no kinematic transforms")
    if (
        transforms.get("schema") != "z_manip.kinematic_transforms.v1"
        or transforms.get("verified") is not True
        or transforms.get("calibration_synthetic") is not False
    ):
        raise ValueError("runtime kinematic transform evidence is not verified")
    if str(transforms.get("camera_frame", "")) != source_frame:
        raise ValueError("runtime camera frame does not match the target cloud")
    if str(transforms.get("platform_base_frame", "")) != base_frame:
        raise ValueError("runtime platform frame does not match the servo base frame")
    if str(transforms.get("arm_base_frame", "")) != arm_base_frame:
        raise ValueError("runtime arm frame does not match the servo arm frame")
    timestamp_ns = int(transforms.get("source_timestamp_ns", 0))
    age_s = (int(now_unix_ns) - timestamp_ns) / 1e9
    if age_s < -0.50 or age_s > max_age_s:
        raise ValueError(f"runtime kinematic transforms are stale ({age_s:.3f}s)")
    base = _validated_matrix(
        transforms.get("platform_base_from_camera"),
        name="platform_base_from_camera",
    )
    arm = _validated_matrix(
        transforms.get("arm_base_from_camera"),
        name="arm_base_from_camera",
    )
    return base, arm, timestamp_ns


def _posture_feedback_state(
    document: dict[str, Any] | None,
    *,
    age_s: float,
    timeout_s: float = 0.75,
) -> tuple[bool, bool, bool, str]:
    """Reduce posture-adapter telemetry to the reactive-controller contract.

    Shadow verification is deliberately distinct from measured settling: it
    is useful diagnostic evidence, but can never unlock manipulation handoff.
    """

    if document is None or not math.isfinite(age_s) or age_s > timeout_s:
        return False, False, False, "posture status unavailable or stale"
    if document.get("schema") != "z_manip.go2w_posture_status.v1":
        return False, False, False, "posture status schema is invalid"
    phase = str(document.get("phase", ""))
    mode = str(document.get("mode", ""))
    detail = str(document.get("detail", ""))
    stop_latched = document.get("stop_latched") is True
    blocked = stop_latched or phase in {
        "blocked",
        "fault",
        "stopped",
        "stopping",
    }
    feedback = document.get("feedback")
    feedback_fresh = (
        isinstance(feedback, dict) and feedback.get("fresh") is True
    )
    settled = (
        mode == "live"
        and phase == "reached"
        and feedback_fresh
        and not stop_latched
    )
    shadow_verified = mode == "shadow" and phase == "shadow" and not blocked
    return settled, blocked, shadow_verified, detail


class DepthServoCore:
    """ROS-free state for deterministic testing and a thin ROS adapter."""

    def __init__(self, settings: DepthServoSettings) -> None:
        self.settings = settings
        self.controller = VisualServoController(VisualServoConfig(
            desired_depth_m=settings.desired_depth_m,
            depth_tolerance_m=settings.depth_tolerance_m,
            lateral_tolerance_m=settings.lateral_tolerance_m,
            settle_time_s=settings.settle_time_s,
            linear_gain=settings.linear_gain,
            yaw_gain=settings.yaw_gain,
            max_forward_mps=settings.max_forward_mps,
            max_reverse_mps=settings.max_reverse_mps,
            max_yaw_rps=settings.max_yaw_rps,
            rotate_only_bearing_rad=settings.rotate_only_bearing_rad,
            yaw_deadband_rad=settings.yaw_deadband_rad,
        ))
        self.reactive = ReactiveTargetController(ReactiveServoConfig(
            desired_planar_standoff_m=settings.desired_depth_m,
            posture_entry_planar_m=max(settings.handoff_depth_m + 0.20, 0.80),
            handoff_planar_max_m=settings.handoff_depth_m,
            linear_gain=settings.linear_gain,
            yaw_gain=settings.yaw_gain,
            max_forward_mps=settings.max_forward_mps,
            max_yaw_rps=settings.max_yaw_rps,
            yaw_deadband_rad=settings.yaw_deadband_rad,
            tracking_loss_grace_s=settings.tracking_loss_grace_s,
        ))
        self._target: tuple[float, float, float] | None = None
        self._raw_target: tuple[float, float, float] | None = None
        self._target_received_s: float | None = None
        self._samples: deque[tuple[float, float, float]] = deque(
            maxlen=settings.target_filter_window,
        )
        self._outlier_samples: deque[tuple[float, float, float]] = deque(
            maxlen=settings.outlier_rebase_samples,
        )
        self._accepted_observations = 0
        self._rejected_observations = 0
        self._rebases = 0
        self._geometry: TargetGeometry | None = None
        self._transforms_received_s: float | None = None
        self._transform_error: str | None = "no synchronized target transforms"
        self._ik_feasible: bool | None = None
        self._last_decision: ReactiveServoDecision | None = None
        self._done = False

    @property
    def target(self) -> tuple[float, float, float] | None:
        return self._target

    @property
    def camera_geometry(self) -> dict[str, float] | None:
        """Return camera-frame 3-D metrics without inventing base-frame data."""

        if self._target is None:
            return None
        x_m, y_m, z_m = self._target
        return {
            "camera_range_m": math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m),
            "camera_elevation_rad": math.atan2(-y_m, z_m),
        }

    @property
    def geometry(self) -> TargetGeometry | None:
        return self._geometry

    @property
    def reactive_status(self) -> dict[str, Any] | None:
        decision = self._last_decision
        if decision is None:
            return None
        return {
            "phase": decision.phase.value,
            "reason": decision.reason,
            "handoff_ready": decision.handoff_ready,
            "needs_ik_probe": decision.needs_ik_probe,
            "posture": asdict(decision.posture),
            "arm_view": {
                **asdict(decision.arm_view),
                "mode": decision.arm_view.mode.value,
            },
        }

    @property
    def transform_status(self) -> dict[str, Any]:
        return {
            "valid": self._geometry is not None,
            "error": self._transform_error,
            "received_monotonic_s": self._transforms_received_s,
        }

    def set_ik_probe_result(self, feasible: bool | None) -> None:
        """Record a downstream read-only IK probe result for handoff gating."""

        self._ik_feasible = None if feasible is None else bool(feasible)

    @property
    def filter_stats(self) -> dict[str, int | float | None]:
        return {
            "window_samples": len(self._samples),
            "accepted": self._accepted_observations,
            "rejected_outliers": self._rejected_observations,
            "outlier_cluster_samples": len(self._outlier_samples),
            "rebases": self._rebases,
            "raw_x_m": None if self._raw_target is None else self._raw_target[0],
            "raw_y_m": None if self._raw_target is None else self._raw_target[1],
            "raw_z_m": None if self._raw_target is None else self._raw_target[2],
        }

    def observe_target(
        self,
        *,
        x_m: float,
        z_m: float,
        stamp_s: float,
        y_m: float = 0.0,
        T_base_camera: np.ndarray | None = None,
        T_arm_camera: np.ndarray | None = None,
        transform_error: str | None = None,
    ) -> bool:
        """Observe a complete optical-frame target centroid.

        ``y_m`` defaults to zero only for backward-compatible callers.  The
        ROS adapter always supplies the measured optical y coordinate.
        """

        values = (float(x_m), float(y_m), float(z_m), float(stamp_s))
        if not all(math.isfinite(value) for value in values) or z_m <= 0.0:
            return False
        transforms_available = T_base_camera is not None and T_arm_camera is not None
        # A fresh camera observation accompanied by failed TF must stop motion
        # instead of silently retaining older valid target geometry.
        if not transforms_available:
            self._geometry = None
            self._transforms_received_s = None
            self._transform_error = (
                transform_error or "synchronized transforms unavailable"
            )
        raw = (float(x_m), float(y_m), float(z_m))
        self._raw_target = raw
        if self._target is not None:
            jump_m = math.sqrt(sum(
                (raw[index] - self._target[index]) ** 2 for index in range(3)
            ))
            if jump_m > self.settings.max_target_jump_m:
                self._outlier_samples.append(raw)
                self._rejected_observations += 1
                if len(self._outlier_samples) < self.settings.outlier_rebase_samples:
                    return False
                cluster_median = tuple(
                    statistics.median(sample[index] for sample in self._outlier_samples)
                    for index in range(3)
                )
                cluster_spread = max(
                    math.sqrt(sum(
                        (sample[index] - cluster_median[index]) ** 2
                        for index in range(3)
                    ))
                    for sample in self._outlier_samples
                )
                if cluster_spread > self.settings.outlier_rebase_spread_m:
                    return False
                # A coherent replacement cluster is a real target relocation,
                # not isolated depth noise. Rebase the filter so the old EMA
                # cannot reject the new stable track forever.
                self._samples.clear()
                self._samples.extend(self._outlier_samples)
                self._outlier_samples.clear()
                self._target = None
                self._rebases += 1
            else:
                self._outlier_samples.clear()
        if self._target is not None or not self._samples:
            self._samples.append(raw)
        median = (
            statistics.median(sample[0] for sample in self._samples),
            statistics.median(sample[1] for sample in self._samples),
            statistics.median(sample[2] for sample in self._samples),
        )
        if self._target is None:
            self._target = median
        else:
            alpha = self.settings.target_filter_alpha
            self._target = (
                alpha * median[0] + (1.0 - alpha) * self._target[0],
                alpha * median[1] + (1.0 - alpha) * self._target[1],
                alpha * median[2] + (1.0 - alpha) * self._target[2],
            )
        self._target_received_s = float(stamp_s)
        self._accepted_observations += 1
        if transforms_available:
            self._geometry = TargetGeometry.from_camera(
                self._target,
                T_base_camera=T_base_camera,
                T_arm_camera=T_arm_camera,
            )
            self._transforms_received_s = float(stamp_s)
            self._transform_error = None
        return True

    def reset(self) -> None:
        self._target = None
        self._raw_target = None
        self._target_received_s = None
        self._samples.clear()
        self._outlier_samples.clear()
        self._accepted_observations = 0
        self._rejected_observations = 0
        self._rebases = 0
        self._geometry = None
        self._transforms_received_s = None
        self._transform_error = "no synchronized target transforms"
        self._ik_feasible = None
        self._last_decision = None
        self._done = False
        self.controller.reset()
        self.reactive.reset()

    def _zero(self, phase: str, age_s: float | None) -> DepthServoOutput:
        self.controller.reset()
        return DepthServoOutput(
            phase=phase,
            proposed_linear_x=0.0,
            proposed_angular_z=0.0,
            published_linear_x=0.0,
            published_angular_z=0.0,
            depth_error_m=None,
            yaw_error_rad=None,
            target_age_s=age_s,
            done=self._done,
            reason=self._transform_error or "",
        )

    def _reactive_tick(
        self,
        *,
        now_s: float,
        age_s: float,
        tracking: bool | None,
        body_settled: bool,
        posture_blocked: bool,
        posture_shadow_verified: bool,
        posture_detail: str,
    ) -> DepthServoOutput:
        fresh_tracking = (
            tracking is True and age_s <= self.settings.target_timeout_s
        )
        transform_age_s = (
            None
            if self._transforms_received_s is None
            else max(0.0, now_s - self._transforms_received_s)
        )
        transform_fresh = (
            self._geometry is not None
            and transform_age_s is not None
            and transform_age_s <= self.settings.transform_timeout_s
        )
        if not transform_fresh:
            reason = self._transform_error or (
                f"synchronized transforms are stale ({transform_age_s:.3f}s)"
                if transform_age_s is not None
                else "synchronized transforms unavailable"
            )
            self._last_decision = ReactiveServoDecision(
                phase=ReactivePhase.TRANSFORM_UNAVAILABLE,
                base=BaseMotionIntent(),
                posture=PostureIntent(),
                arm_view=ArmViewIntent(),
                geometry=None,
                reason=reason,
            )
            output = self._zero("transform_unavailable", age_s)
            return DepthServoOutput(
                **{
                    **asdict(output),
                    "reason": reason,
                    "reactive_phase": ReactivePhase.TRANSFORM_UNAVAILABLE.value,
                },
            )
        if posture_blocked and self._last_decision is not None and (
            self._last_decision.phase is ReactivePhase.POSTURE_ADJUST
        ):
            output = self._zero("posture_blocked", age_s)
            return DepthServoOutput(
                **{
                    **asdict(output),
                    "reason": posture_detail or "posture adapter blocked the intent",
                    "reactive_phase": ReactivePhase.POSTURE_ADJUST.value,
                },
            )
        decision = self.reactive.update(
            self._geometry if fresh_tracking and transform_fresh else None,
            now_s=now_s,
            tracking=fresh_tracking,
            # The depth-servo runtime does not own posture hardware. A posture
            # adapter may later feed measured settling; exposing intents here
            # must never manufacture an active body command.
            body_settled=body_settled,
            ik_feasible=self._ik_feasible,
        )
        self._last_decision = decision
        phase = decision.phase.value
        if decision.phase is ReactivePhase.HANDOFF_READY:
            self._done = True
            phase = "reached"
        elif decision.phase is ReactivePhase.BASE_APPROACH:
            phase = "approach"
        elif (
            posture_shadow_verified
            and decision.phase is ReactivePhase.POSTURE_ADJUST
        ):
            phase = "posture_shadow_verified"
        linear_x = decision.base.linear_x_mps
        angular_z = decision.base.angular_z_rps
        live = self.settings.mode == "live"
        geometry = decision.geometry
        depth_error = None
        yaw_error = None
        if geometry is not None:
            depth_error = (
                geometry.base_planar_distance_m
                - self.settings.desired_depth_m
            )
            yaw_error = geometry.base_bearing_rad
        return DepthServoOutput(
            phase=phase,
            proposed_linear_x=linear_x,
            proposed_angular_z=angular_z,
            published_linear_x=linear_x if live else 0.0,
            published_angular_z=angular_z if live else 0.0,
            depth_error_m=depth_error,
            yaw_error_rad=yaw_error,
            target_age_s=age_s,
            done=self._done,
            reason=decision.reason,
            reactive_phase=decision.phase.value,
            needs_ik_probe=decision.needs_ik_probe,
        )

    def tick(
        self,
        *,
        now_s: float,
        tracking: bool | None,
        body_settled: bool = False,
        posture_blocked: bool = False,
        posture_shadow_verified: bool = False,
        posture_detail: str = "",
    ) -> DepthServoOutput:
        now = float(now_s)
        if self._done:
            return self._zero("reached", 0.0)
        if self._target is None or self._target_received_s is None:
            return self._zero("waiting_target", None)
        age_s = max(0.0, now - self._target_received_s)
        if not self.settings.allow_legacy_optical_depth_for_tests:
            return self._reactive_tick(
                now_s=now,
                age_s=age_s,
                tracking=tracking,
                body_settled=body_settled,
                posture_blocked=posture_blocked,
                posture_shadow_verified=posture_shadow_verified,
                posture_detail=posture_detail,
            )
        if tracking is not True or age_s > self.settings.target_timeout_s:
            phase = (
                "reacquiring"
                if age_s <= self.settings.tracking_loss_grace_s
                else "tracking_lost"
            )
            return self._zero(phase, age_s)
        x_m, y_m, z_m = self._target
        yaw_error = math.atan2(x_m, z_m)
        # A Go2W body pose is not a precision fixture: one footstep can move
        # the camera by several centimetres and degrees.  Stop the base as
        # soon as the object enters the arm's coarse near-field cone, latch
        # that decision, and let fresh perception + IK solve the final pose.
        # This is intentionally one-sided in depth; we never ask the base to
        # back away after it has entered the manipulation workspace.
        if (
            z_m <= self.settings.handoff_depth_m
            and abs(yaw_error) <= self.settings.handoff_bearing_rad
        ):
            self._done = True
            return DepthServoOutput(
                phase="reached",
                proposed_linear_x=0.0,
                proposed_angular_z=0.0,
                published_linear_x=0.0,
                published_angular_z=0.0,
                depth_error_m=z_m - self.settings.desired_depth_m,
                yaw_error_rad=yaw_error,
                target_age_s=age_s,
                done=True,
            )
        # This first mobile-manipulation flow is approach-only: once the target
        # is at or inside the requested standoff band, never reverse away from
        # it.  Continue yaw centering, settle, then hand off to manipulation.
        control_z_m = max(z_m, self.settings.desired_depth_m)
        command = self.controller.update((x_m, y_m, control_z_m), stamp_s=now)
        linear_x = command.linear_x
        # Keep Go2W above its observed low-speed dead zone while it is still
        # outside the manipulation handoff. If it is already near but not
        # roughly aligned, rotate without advancing past the target.
        if linear_x > 0.0 and z_m > self.settings.handoff_depth_m:
            linear_x = max(linear_x, self.settings.min_forward_mps)
        elif z_m <= self.settings.handoff_depth_m:
            linear_x = 0.0
        phase = "approach"
        if command.converged:
            self._done = True
            phase = "reached"
        elif linear_x == 0.0 and command.angular_z == 0.0:
            phase = "settling"
        live = self.settings.mode == "live"
        return DepthServoOutput(
            phase=phase,
            proposed_linear_x=linear_x,
            proposed_angular_z=command.angular_z,
            published_linear_x=linear_x if live else 0.0,
            published_angular_z=command.angular_z if live else 0.0,
            depth_error_m=z_m - self.settings.desired_depth_m,
            yaw_error_rad=command.yaw_error_rad,
            target_age_s=age_s,
            done=self._done,
        )


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, document: dict[str, Any]) -> None:
    """Append compact bounded diagnostics without ever storing camera data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 2_000_000:
        rotated = path.with_suffix(path.suffix + ".1")
        rotated.unlink(missing_ok=True)
        os.replace(path, rotated)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--target-topic", default="/track_3d/selected_target_pointcloud")
    parser.add_argument("--tracking-topic", default="/track_3d/is_tracking")
    parser.add_argument("--velocity-topic", default="/cmd_vel")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--arm-base-frame", default="piper_base_link")
    parser.add_argument(
        "--runtime-state",
        type=Path,
        help="fresh subscribe-only runtime-observer.json transform source",
    )
    parser.add_argument("--runtime-transform-timeout-s", type=float, default=0.50)
    parser.add_argument("--desired-depth-m", type=float, default=0.50)
    parser.add_argument("--handoff-depth-m", type=float, default=0.52)
    parser.add_argument("--handoff-bearing-deg", type=float, default=20.0)
    parser.add_argument("--min-forward-mps", type=float, default=0.10)
    parser.add_argument("--max-forward-mps", type=float, default=0.18)
    parser.add_argument("--max-yaw-rps", type=float, default=0.12)
    parser.add_argument("--target-timeout-s", type=float, default=0.25)
    parser.add_argument("--tracking-loss-grace-s", type=float, default=0.75)
    parser.add_argument("--transform-timeout-s", type=float, default=0.25)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    return parser.parse_args()


def _run_ros(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Bool, String
    from tf2_ros import Buffer, TransformException, TransformListener

    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0.0:
        raise ValueError("rate must be finite and positive")
    if (
        not math.isfinite(args.runtime_transform_timeout_s)
        or args.runtime_transform_timeout_s <= 0.0
    ):
        raise ValueError("runtime transform timeout must be finite and positive")
    settings = DepthServoSettings(
        mode=args.mode,
        desired_depth_m=args.desired_depth_m,
        handoff_depth_m=args.handoff_depth_m,
        handoff_bearing_rad=math.radians(args.handoff_bearing_deg),
        min_forward_mps=args.min_forward_mps,
        max_forward_mps=args.max_forward_mps,
        max_yaw_rps=args.max_yaw_rps,
        target_timeout_s=args.target_timeout_s,
        tracking_loss_grace_s=args.tracking_loss_grace_s,
        base_frame=args.base_frame,
        arm_base_frame=args.arm_base_frame,
        transform_timeout_s=args.transform_timeout_s,
    )

    class DepthServoNode(Node):
        def __init__(self) -> None:
            super().__init__("z_manip_depth_servo")
            self.core = DepthServoCore(settings)
            self.tracking: bool | None = None
            self.last_source_stamp_ns: int | None = None
            self.last_source_frame: str | None = None
            self.last_transform_error: str | None = "no target transforms received"
            self.last_transform_success_s: float | None = None
            self.last_transform_source: str | None = None
            self.last_transform_stamps_ns: dict[str, int | None] = {
                settings.base_frame: None,
                settings.arm_base_frame: None,
            }
            self.posture_status: dict[str, Any] | None = None
            self.posture_status_received_s: float | None = None
            self.last_posture_intent: tuple[float, float] | None = None
            self.last_posture_intent_s = 0.0
            self.last_output = self.core.tick(now_s=time.monotonic(), tracking=False)
            self.last_trace_phase: str | None = None
            self.last_trace_s = 0.0
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.publisher = self.create_publisher(TwistStamped, args.velocity_topic, 1)
            self.posture_intent_publisher = self.create_publisher(
                String,
                "/z_manip/reactive/posture_intent",
                qos,
            )
            self.create_subscription(PointCloud2, args.target_topic, self._target, qos)
            self.create_subscription(Bool, args.tracking_topic, self._tracking, qos)
            self.create_subscription(
                String,
                "/go2w/posture_state",
                self._posture_state,
                qos,
            )
            self.create_timer(1.0 / args.rate_hz, self._tick)
            self._write_status("starting")

        @staticmethod
        def _matrix(transform_stamped: Any) -> np.ndarray:
            transform = transform_stamped.transform
            return _rigid_transform_matrix(
                (
                    float(transform.translation.x),
                    float(transform.translation.y),
                    float(transform.translation.z),
                ),
                (
                    float(transform.rotation.x),
                    float(transform.rotation.y),
                    float(transform.rotation.z),
                    float(transform.rotation.w),
                ),
            )

        @staticmethod
        def _stamp_ns(transform_stamped: Any) -> int:
            stamp = transform_stamped.header.stamp
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        def _target_transforms(
            self,
            *,
            source_frame: str,
            source_stamp: Any,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not source_frame:
                raise ValueError("target point cloud has an empty frame_id")
            query_time = Time.from_msg(source_stamp)
            timeout = Duration(seconds=settings.transform_timeout_s)
            tf_error: Exception | None = None
            try:
                base = self.tf_buffer.lookup_transform(
                    settings.base_frame,
                    source_frame,
                    query_time,
                    timeout=timeout,
                )
                arm = self.tf_buffer.lookup_transform(
                    settings.arm_base_frame,
                    source_frame,
                    query_time,
                    timeout=timeout,
                )
                self.last_transform_stamps_ns = {
                    settings.base_frame: self._stamp_ns(base),
                    settings.arm_base_frame: self._stamp_ns(arm),
                }
                self.last_transform_success_s = time.monotonic()
                self.last_transform_source = "tf2"
                self.last_transform_error = None
                return self._matrix(base), self._matrix(arm)
            except TransformException as error:
                tf_error = error
            if args.runtime_state is None:
                raise ValueError(f"TF lookup failed and no runtime state is configured: {tf_error}")
            try:
                base_matrix, arm_matrix, stamp_ns = _runtime_state_transforms(
                    args.runtime_state,
                    source_frame=source_frame,
                    base_frame=settings.base_frame,
                    arm_base_frame=settings.arm_base_frame,
                    now_unix_ns=time.time_ns(),
                    max_age_s=args.runtime_transform_timeout_s,
                )
            except (OSError, ValueError, json.JSONDecodeError) as runtime_error:
                raise ValueError(
                    f"TF lookup failed ({tf_error}); runtime model failed ({runtime_error})",
                ) from runtime_error
            self.last_transform_stamps_ns = {
                settings.base_frame: stamp_ns,
                settings.arm_base_frame: stamp_ns,
            }
            self.last_transform_success_s = time.monotonic()
            self.last_transform_source = "runtime_observer_kinematics"
            self.last_transform_error = None
            return base_matrix, arm_matrix

        def _target(self, message: PointCloud2) -> None:
            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            for point in point_cloud2.read_points(
                message,
                field_names=("x", "y", "z"),
                skip_nans=True,
            ):
                x_m, y_m, z_m = float(point[0]), float(point[1]), float(point[2])
                if all(math.isfinite(value) for value in (x_m, y_m, z_m)) and z_m > 0.0:
                    xs.append(x_m)
                    ys.append(y_m)
                    zs.append(z_m)
                if len(xs) >= 5000:
                    break
            if not xs:
                return
            source_frame = str(message.header.frame_id or "")
            transform_error: str | None = None
            T_base_camera: np.ndarray | None = None
            T_arm_camera: np.ndarray | None = None
            try:
                T_base_camera, T_arm_camera = self._target_transforms(
                    source_frame=source_frame,
                    source_stamp=message.header.stamp,
                )
            except (TransformException, ValueError) as error:
                transform_error = (
                    f"TF {source_frame or '<empty>'} -> "
                    f"({settings.base_frame}, {settings.arm_base_frame}) unavailable: {error}"
                )
                self.last_transform_error = transform_error
            accepted = self.core.observe_target(
                x_m=statistics.median(xs),
                y_m=statistics.median(ys),
                z_m=statistics.median(zs),
                stamp_s=time.monotonic(),
                T_base_camera=T_base_camera,
                T_arm_camera=T_arm_camera,
                transform_error=transform_error,
            )
            if accepted:
                self.last_source_frame = source_frame or None
                self.last_source_stamp_ns = (
                    int(message.header.stamp.sec) * 1_000_000_000
                    + int(message.header.stamp.nanosec)
                )

        def _tracking(self, message: Bool) -> None:
            self.tracking = bool(message.data)

        def _posture_state(self, message: String) -> None:
            try:
                document = json.loads(message.data)
            except json.JSONDecodeError:
                return
            if (
                isinstance(document, dict)
                and document.get("schema") == "z_manip.go2w_posture_status.v1"
            ):
                self.posture_status = document
                self.posture_status_received_s = time.monotonic()

        def _posture_feedback(self) -> tuple[bool, bool, bool, str]:
            age_s = (
                math.inf
                if self.posture_status_received_s is None
                else time.monotonic() - self.posture_status_received_s
            )
            return _posture_feedback_state(
                self.posture_status,
                age_s=age_s,
            )

        def _publish_posture_intent(self, *, blocked: bool = False) -> None:
            if blocked:
                return
            status = self.core.reactive_status
            if status is None or status.get("phase") not in {
                ReactivePhase.POSTURE_ADJUST.value,
                ReactivePhase.VIEW_RECOVERY.value,
            }:
                return
            posture = status.get("posture")
            if not isinstance(posture, dict):
                return
            target = (
                float(posture.get("body_height_delta_m", 0.0)),
                float(posture.get("pitch_delta_rad", 0.0)),
            )
            now_s = time.monotonic()
            if (
                self.last_posture_intent == target
                and now_s - self.last_posture_intent_s < 0.25
            ):
                return
            message = String()
            message.data = json.dumps(
                {
                    "schema": "z_manip.go2w_posture_intent.v1",
                    "body_height_delta_m": target[0],
                    "pitch_delta_rad": target[1],
                },
                separators=(",", ":"),
                allow_nan=False,
            )
            self.posture_intent_publisher.publish(message)
            self.last_posture_intent = target
            self.last_posture_intent_s = now_s

        def _publish(self, linear_x: float, angular_z: float) -> None:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "base_link"
            message.twist.linear.x = float(linear_x)
            message.twist.angular.z = float(angular_z)
            self.publisher.publish(message)

        def _write_status(self, state: str | None = None, *, running: bool = True) -> None:
            target = self.core.target
            geometry = self.core.geometry
            transform_received_s = self.core.transform_status[
                "received_monotonic_s"
            ]
            geometry_age_s = (
                None
                if transform_received_s is None
                else max(0.0, time.monotonic() - transform_received_s)
            )
            lookup_age_s = (
                None
                if self.last_transform_success_s is None
                else max(0.0, time.monotonic() - self.last_transform_success_s)
            )
            transform_fresh = (
                geometry is not None
                and geometry_age_s is not None
                and geometry_age_s <= settings.transform_timeout_s
            )
            posture_age_s = (
                None
                if self.posture_status_received_s is None
                else max(0.0, time.monotonic() - self.posture_status_received_s)
            )
            document = {
                "schema": STATUS_SCHEMA,
                "running": running,
                "mode": settings.mode,
                "phase": state or self.last_output.phase,
                "tracking": self.tracking,
                "target": None if target is None else {
                    "x_m": target[0],
                    "y_m": target[1],
                    "z_m": target[2],
                    "frame_id": self.last_source_frame,
                },
                "geometry": (
                    asdict(geometry)
                    if geometry is not None
                    else self.core.camera_geometry
                ),
                "reactive": self.core.reactive_status,
                "transforms": {
                    "valid": transform_fresh,
                    "error": (
                        self.last_transform_error
                        or self.core.transform_status["error"]
                        or (
                            "synchronized transforms are stale"
                            if not transform_fresh else None
                        )
                    ),
                    "geometry_age_s": geometry_age_s,
                    "lookup_age_s": lookup_age_s,
                    "source_frame": self.last_source_frame,
                    "base_frame": settings.base_frame,
                    "arm_base_frame": settings.arm_base_frame,
                    "stamps_ns": self.last_transform_stamps_ns,
                    "source": self.last_transform_source,
                },
                "posture_status": {
                    "age_s": posture_age_s,
                    "document": self.posture_status,
                    "last_intent": None if self.last_posture_intent is None else {
                        "body_height_delta_m": self.last_posture_intent[0],
                        "pitch_delta_rad": self.last_posture_intent[1],
                    },
                },
                "source_stamp_ns": self.last_source_stamp_ns,
                "output": asdict(self.last_output),
                "filter": self.core.filter_stats,
                "trace_file": None if args.trace_file is None else str(args.trace_file),
                "pid": os.getpid(),
                "updated_unix_ns": time.time_ns(),
            }
            _atomic_json(args.status_file, document)
            now_s = time.monotonic()
            if args.trace_file is not None and (
                self.last_output.phase != self.last_trace_phase
                or now_s - self.last_trace_s >= 1.0
            ):
                _append_jsonl(args.trace_file, {
                    "schema": "z_manip.depth_servo_trace.v1",
                    "updated_unix_ns": document["updated_unix_ns"],
                    "mode": settings.mode,
                    "phase": document["phase"],
                    "tracking": self.tracking,
                    "target": document["target"],
                    "source_stamp_ns": self.last_source_stamp_ns,
                    "output": document["output"],
                    "filter": document["filter"],
                })
                self.last_trace_phase = self.last_output.phase
                self.last_trace_s = now_s

        def _tick(self) -> None:
            settled, blocked, shadow_verified, detail = self._posture_feedback()
            self.last_output = self.core.tick(
                now_s=time.monotonic(),
                tracking=self.tracking,
                body_settled=settled,
                posture_blocked=blocked,
                posture_shadow_verified=shadow_verified,
                posture_detail=detail,
            )
            self._publish_posture_intent(blocked=blocked)
            if settings.mode == "live":
                self._publish(
                    self.last_output.published_linear_x,
                    self.last_output.published_angular_z,
                )
            self._write_status()

        def stop(self, phase: str = "stopped") -> None:
            if settings.mode == "live":
                for _ in range(3):
                    self._publish(0.0, 0.0)
            self.last_output = DepthServoOutput(
                phase=phase,
                proposed_linear_x=0.0,
                proposed_angular_z=0.0,
                published_linear_x=0.0,
                published_angular_z=0.0,
                depth_error_m=None,
                yaw_error_rad=None,
                target_age_s=None,
                done=False,
            )
            self._write_status(phase, running=False)

    rclpy.init()
    node = DepthServoNode()
    stopped = threading.Event()
    stop_published = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()
        # Publish the final zero while the ROS context is still valid.  Calling
        # shutdown first made the finally block raise and could leave the
        # transport relying only on its watchdog stop.
        if not stop_published.is_set():
            node.stop("stopped")
            stop_published.set()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        rclpy.spin(node)
    finally:
        if not stop_published.is_set():
            node.stop("stopped" if stopped.is_set() else "exited")
            stop_published.set()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    return _run_ros(_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
