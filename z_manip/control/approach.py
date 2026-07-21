"""Exclusive-owner handoff from approximate navigation to visual servo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from .visual_servo import ServoCommand, VisualServoConfig, VisualServoController


class ApproachPhase(str, Enum):
    FAR_NAV = "far_nav"
    HANDOFF = "handoff"
    VISUAL_SERVO = "visual_servo"
    COMPLETE = "complete"
    FAILED = "failed"


class VelocityOwner(str, Enum):
    NAVIGATION = "navigation"
    MANIP_SERVO = "manip_servo"
    NONE = "none"


@dataclass(frozen=True)
class TwoStageApproachConfig:
    near_stage_threshold_m: float = 1.4
    tracker_lock_time_s: float = 0.35
    handoff_quiet_time_s: float = 0.30
    navigation_quiet_speed_mps: float = 0.025
    navigation_quiet_yaw_rate_rps: float = 0.035
    track_loss_timeout_s: float = 0.6
    handoff_timeout_s: float = 60.0
    timeout_s: float = 60.0
    max_roll_rad: float = math.radians(10.0)
    max_pitch_rad: float = math.radians(12.0)
    visual_servo: VisualServoConfig = VisualServoConfig()

    def __post_init__(self) -> None:
        positive = (
            self.near_stage_threshold_m,
            self.tracker_lock_time_s,
            self.handoff_quiet_time_s,
            self.navigation_quiet_speed_mps,
            self.navigation_quiet_yaw_rate_rps,
            self.track_loss_timeout_s,
            self.handoff_timeout_s,
            self.timeout_s,
            self.max_roll_rad,
            self.max_pitch_rad,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("approach thresholds and timeouts must be positive")


@dataclass(frozen=True)
class ApproachInput:
    stamp_s: float
    approximate_range_m: float | None
    target_position_camera: Optional[Sequence[float]]
    tracker_locked: bool
    navigation_speed_mps: float
    navigation_yaw_rate_rps: float
    base_roll_rad: float
    base_pitch_rad: float
    desired_depth_m: float | None = None


@dataclass(frozen=True)
class ApproachOutput:
    phase: ApproachPhase
    owner: VelocityOwner
    cancel_navigation: bool
    servo: ServoCommand
    reason: str = ""


_STOP = ServoCommand(0.0, 0.0, False)


class TwoStageApproachController:
    """Coordinate far navigation and near visual control without command overlap.

    Approximate map/navigation range is sufficient for the far phase, but never
    declares completion. Only a persistent visual track can trigger handoff and
    only the visual servo convergence gate can complete the approach.
    """

    def __init__(self, config: TwoStageApproachConfig | None = None):
        self.config = config or TwoStageApproachConfig()
        self.visual_servo = VisualServoController(self.config.visual_servo)
        self.phase = ApproachPhase.FAR_NAV
        self._phase_started_at: float | None = None
        self._last_stamp: float | None = None
        self._track_since: float | None = None
        self._quiet_since: float | None = None
        self._lost_since: float | None = None

    def reset(self) -> None:
        self.phase = ApproachPhase.FAR_NAV
        self._phase_started_at = None
        self._last_stamp = None
        self._track_since = None
        self._quiet_since = None
        self._lost_since = None
        self.visual_servo.reset()

    def _fail(self, reason: str) -> ApproachOutput:
        self.phase = ApproachPhase.FAILED
        self.visual_servo.reset()
        return ApproachOutput(self.phase, VelocityOwner.NONE, True, _STOP, reason)

    @staticmethod
    def _has_target(value: ApproachInput) -> bool:
        if not value.tracker_locked or value.target_position_camera is None:
            return False
        if len(value.target_position_camera) != 3:
            return False
        x, y, z = (float(item) for item in value.target_position_camera)
        return all(math.isfinite(item) for item in (x, y, z)) and z > 0.0

    def update(self, value: ApproachInput) -> ApproachOutput:
        stamp = float(value.stamp_s)
        finite_scalars = (
            stamp,
            float(value.navigation_speed_mps),
            float(value.navigation_yaw_rate_rps),
            float(value.base_roll_rad),
            float(value.base_pitch_rad),
        )
        if not all(math.isfinite(item) for item in finite_scalars):
            return self._fail("approach input contains a non-finite value")
        if self._last_stamp is not None and stamp < self._last_stamp:
            return self._fail("simulation time moved backwards")
        self._last_stamp = stamp
        if self.phase in (ApproachPhase.COMPLETE, ApproachPhase.FAILED):
            return ApproachOutput(self.phase, VelocityOwner.NONE, False, _STOP)
        if self._phase_started_at is None:
            self._phase_started_at = stamp
        phase_timeout_s = (
            self.config.timeout_s
            if self.phase == ApproachPhase.VISUAL_SERVO
            else self.config.handoff_timeout_s
        )
        if stamp - self._phase_started_at > phase_timeout_s:
            reason = (
                "visual servo timeout"
                if self.phase == ApproachPhase.VISUAL_SERVO
                else "approach handoff timeout"
            )
            return self._fail(reason)
        if (
            abs(value.base_roll_rad) > self.config.max_roll_rad
            or abs(value.base_pitch_rad) > self.config.max_pitch_rad
        ):
            return self._fail("base attitude gate exceeded")
        has_target = self._has_target(value)
        if has_target:
            if self._track_since is None:
                self._track_since = stamp
        else:
            self._track_since = None
        track_stable = (
            self._track_since is not None
            and stamp - self._track_since >= self.config.tracker_lock_time_s
        )

        if self.phase == ApproachPhase.FAR_NAV:
            visual_range = (
                float(value.target_position_camera[2]) if has_target else float("inf")
            )
            approximate_range = (
                float(value.approximate_range_m)
                if value.approximate_range_m is not None
                and math.isfinite(float(value.approximate_range_m))
                else float("inf")
            )
            if (
                track_stable
                and min(visual_range, approximate_range)
                <= self.config.near_stage_threshold_m
            ):
                self.phase = ApproachPhase.HANDOFF
                self._phase_started_at = stamp
                self._quiet_since = None
                self.visual_servo.reset()
                return ApproachOutput(self.phase, VelocityOwner.NONE, True, _STOP)
            return ApproachOutput(
                self.phase,
                VelocityOwner.NAVIGATION,
                False,
                _STOP,
                "waiting for stable near-field visual track"
                if approximate_range <= self.config.near_stage_threshold_m else "",
            )

        if self.phase == ApproachPhase.HANDOFF:
            if not has_target:
                self._quiet_since = None
                return ApproachOutput(
                    self.phase,
                    VelocityOwner.NONE,
                    True,
                    _STOP,
                    "waiting for target reacquisition before visual control",
                )
            if (
                abs(value.navigation_speed_mps)
                <= self.config.navigation_quiet_speed_mps
                and abs(value.navigation_yaw_rate_rps)
                <= self.config.navigation_quiet_yaw_rate_rps
            ):
                if self._quiet_since is None:
                    self._quiet_since = stamp
            else:
                self._quiet_since = None
            if (
                track_stable
                and self._quiet_since is not None
                and stamp - self._quiet_since + 1e-9
                >= self.config.handoff_quiet_time_s
            ):
                self.phase = ApproachPhase.VISUAL_SERVO
                self._phase_started_at = stamp
                self._lost_since = None
                command = self.visual_servo.update(
                    value.target_position_camera,
                    stamp_s=stamp,
                    desired_depth_m=value.desired_depth_m,
                )
                return ApproachOutput(
                    self.phase, VelocityOwner.MANIP_SERVO, True, command,
                )
            return ApproachOutput(self.phase, VelocityOwner.NONE, True, _STOP)

        assert self.phase == ApproachPhase.VISUAL_SERVO
        if not has_target:
            self.visual_servo.reset()
            if self._lost_since is None:
                self._lost_since = stamp
            if stamp - self._lost_since > self.config.track_loss_timeout_s:
                return self._fail("persistent target tracking lost during visual servo")
            return ApproachOutput(
                self.phase,
                VelocityOwner.MANIP_SERVO,
                True,
                _STOP,
                "holding zero while tracker reacquires target",
            )
        self._lost_since = None
        command = self.visual_servo.update(
            value.target_position_camera,
            stamp_s=stamp,
            desired_depth_m=value.desired_depth_m,
        )
        if command.converged:
            self.phase = ApproachPhase.COMPLETE
            return ApproachOutput(self.phase, VelocityOwner.NONE, True, _STOP)
        return ApproachOutput(
            self.phase, VelocityOwner.MANIP_SERVO, True, command,
        )
