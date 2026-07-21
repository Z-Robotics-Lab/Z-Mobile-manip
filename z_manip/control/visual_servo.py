"""Fail-closed position-based visual servo for the Go2W base."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class VisualServoConfig:
    desired_depth_m: float = 0.52
    depth_tolerance_m: float = 0.05
    lateral_tolerance_m: float = 0.035
    depth_exit_hysteresis_m: float = 0.01
    lateral_exit_hysteresis_m: float = 0.005
    settle_time_s: float = 2.0
    linear_gain: float = 0.65
    yaw_gain: float = 1.4
    max_forward_mps: float = 0.18
    max_reverse_mps: float = 0.06
    max_yaw_rps: float = 0.35
    rotate_only_bearing_rad: float = math.radians(18.0)
    yaw_deadband_rad: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.desired_depth_m,
            self.depth_tolerance_m,
            self.lateral_tolerance_m,
            self.settle_time_s,
            self.linear_gain,
            self.yaw_gain,
            self.max_forward_mps,
            self.max_reverse_mps,
            self.max_yaw_rps,
            self.rotate_only_bearing_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("visual-servo settings must be finite and positive")
        hysteresis = (
            self.depth_exit_hysteresis_m,
            self.lateral_exit_hysteresis_m,
            self.yaw_deadband_rad,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in hysteresis):
            raise ValueError("visual-servo hysteresis must be finite and non-negative")
        if self.depth_exit_hysteresis_m > self.depth_tolerance_m:
            raise ValueError("depth exit hysteresis cannot exceed depth tolerance")
        if self.lateral_exit_hysteresis_m > self.lateral_tolerance_m:
            raise ValueError("lateral exit hysteresis cannot exceed lateral tolerance")
        if self.rotate_only_bearing_rad >= math.pi / 2.0:
            raise ValueError("rotate-only bearing must be below pi/2")


@dataclass(frozen=True)
class ServoCommand:
    linear_x: float
    angular_z: float
    converged: bool
    depth_error_m: float = 0.0
    yaw_error_rad: float = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class VisualServoController:
    """Turn target optical XYZ into bounded base velocity commands.

    Optical ``x`` points right and ``z`` points forward. ROS base yaw is
    positive left, hence the negative sign on the image bearing. Missing or
    non-finite observations produce an immediate zero command and reset the
    continuous convergence window.
    """

    def __init__(self, config: VisualServoConfig) -> None:
        self.config = config
        self._settled_since: Optional[float] = None

    def reset(self) -> None:
        self._settled_since = None

    def update(
        self,
        position_camera: Optional[Sequence[float]],
        *,
        stamp_s: float,
        desired_depth_m: float | None = None,
    ) -> ServoCommand:
        if position_camera is None or len(position_camera) != 3:
            self.reset()
            return ServoCommand(0.0, 0.0, False)
        x, _y, z = (float(value) for value in position_camera)
        if not all(math.isfinite(value) for value in (x, z, stamp_s)) or z <= 0.0:
            self.reset()
            return ServoCommand(0.0, 0.0, False)

        desired_depth = (
            self.config.desired_depth_m
            if desired_depth_m is None
            else float(desired_depth_m)
        )
        if not math.isfinite(desired_depth) or desired_depth <= 0.0:
            self.reset()
            return ServoCommand(0.0, 0.0, False)
        depth_error = z - desired_depth
        yaw_error = math.atan2(x, z)
        if self._settled_since is not None and stamp_s < self._settled_since:
            self._settled_since = None
        settling = self._settled_since is not None
        depth_limit = self.config.depth_tolerance_m + (
            self.config.depth_exit_hysteresis_m if settling else 0.0
        )
        lateral_limit = self.config.lateral_tolerance_m + (
            self.config.lateral_exit_hysteresis_m if settling else 0.0
        )
        inside = (
            abs(depth_error) <= depth_limit
            and abs(x) <= lateral_limit
        )
        if inside:
            if self._settled_since is None or stamp_s < self._settled_since:
                self._settled_since = stamp_s
            converged = (
                stamp_s - self._settled_since + 1e-9
                >= self.config.settle_time_s
            )
            return ServoCommand(0.0, 0.0, converged, depth_error, yaw_error)

        self._settled_since = None
        linear = _clamp(
            self.config.linear_gain * depth_error,
            -self.config.max_reverse_mps,
            self.config.max_forward_mps,
        )
        angular = _clamp(
            -self.config.yaw_gain * yaw_error,
            -self.config.max_yaw_rps,
            self.config.max_yaw_rps,
        )
        # A legged base rocks laterally during each step.  Chasing a few
        # degrees of image-bearing noise makes it alternate left/right foot
        # turns instead of progressing toward the target.  Deployments can
        # therefore opt into a small zero-yaw region while wheeled/sim users
        # keep the legacy default of zero deadband.
        if abs(yaw_error) <= self.config.yaw_deadband_rad:
            angular = 0.0
        # Large bearing error: rotate first instead of driving a blind arc.
        if abs(yaw_error) > self.config.rotate_only_bearing_rad:
            linear = 0.0
        return ServoCommand(linear, angular, False, depth_error, yaw_error)
