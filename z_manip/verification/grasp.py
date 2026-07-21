"""Verify a grasp from proprioception and persistent visual tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class VerificationState(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class GraspVerificationConfig:
    min_held_aperture_m: float = 0.008
    max_held_aperture_m: float = 0.068
    min_lift_m: float = 0.06
    max_relative_drift_m: float = 0.045
    min_target_lift_ratio: float = 0.65
    hold_time_s: float = 0.25
    track_loss_timeout_s: float = 0.45
    lift_direction_base: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.min_held_aperture_m < self.max_held_aperture_m:
            raise ValueError("invalid held-aperture interval")
        if self.min_lift_m <= 0.0 or self.max_relative_drift_m <= 0.0:
            raise ValueError("verification motion thresholds must be positive")
        if not 0.0 < self.min_target_lift_ratio <= 1.0:
            raise ValueError("target lift ratio must be in (0, 1]")
        if self.hold_time_s <= 0.0 or self.track_loss_timeout_s <= 0.0:
            raise ValueError("verification timers must be positive")
        direction = np.asarray(self.lift_direction_base, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("lift direction must be a finite three-vector")
        if np.linalg.norm(direction) < 1e-9:
            raise ValueError("lift direction must be nonzero")


@dataclass(frozen=True)
class VerificationSample:
    stamp_s: float
    gripper_aperture_m: float
    ee_position_base: Sequence[float]
    target_centroid_base: Optional[Sequence[float]]
    tracker_locked: bool


@dataclass(frozen=True)
class VerificationResult:
    state: VerificationState
    reason: str = ""
    lift_m: float = 0.0
    target_lift_m: float = 0.0
    relative_target_drift_m: float = 0.0


class GraspVerifier:
    """Fuse gripper aperture, FK motion and observed target co-motion."""

    def __init__(self, config: GraspVerificationConfig | None = None):
        self.config = config or GraspVerificationConfig()
        self._direction = np.asarray(self.config.lift_direction_base, dtype=float)
        self._direction /= np.linalg.norm(self._direction)
        self.reset()

    def reset(self) -> None:
        self._start_stamp: float | None = None
        self._last_stamp: float | None = None
        self._start_ee: np.ndarray | None = None
        self._start_target: np.ndarray | None = None
        self._start_relative: np.ndarray | None = None
        self._lost_since: float | None = None
        self._hold_since: float | None = None
        self._terminal: VerificationResult | None = None

    @staticmethod
    def _vector(value: Sequence[float], label: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must be a finite three-vector")
        return vector

    def _fail(self, reason: str) -> VerificationResult:
        self._terminal = VerificationResult(VerificationState.FAILED, reason)
        return self._terminal

    def update(self, sample: VerificationSample) -> VerificationResult:
        if self._terminal is not None:
            return self._terminal
        try:
            stamp = float(sample.stamp_s)
            aperture = float(sample.gripper_aperture_m)
            ee = self._vector(sample.ee_position_base, "end-effector position")
        except (TypeError, ValueError) as error:
            return self._fail(f"invalid verification sample: {error}")
        if not math.isfinite(stamp) or not math.isfinite(aperture):
            return self._fail("verification sample contains a non-finite scalar")
        if self._last_stamp is not None and stamp < self._last_stamp:
            return self._fail("verification time moved backwards")
        self._last_stamp = stamp
        if aperture <= self.config.min_held_aperture_m:
            return self._fail("gripper closed to empty-grasp aperture")
        if aperture >= self.config.max_held_aperture_m:
            return self._fail("gripper remains effectively open")

        target = None
        if sample.tracker_locked and sample.target_centroid_base is not None:
            try:
                target = self._vector(sample.target_centroid_base, "target centroid")
            except ValueError as error:
                return self._fail(str(error))
            self._lost_since = None
        else:
            if self._lost_since is None:
                self._lost_since = stamp
            if stamp - self._lost_since > self.config.track_loss_timeout_s:
                return self._fail("persistent target tracking unavailable during lift")

        if self._start_stamp is None:
            if target is None:
                return VerificationResult(VerificationState.PENDING, "waiting for target lock")
            self._start_stamp = stamp
            self._start_ee = ee
            self._start_target = target
            self._start_relative = target - ee
            return VerificationResult(VerificationState.PENDING)

        assert self._start_ee is not None
        assert self._start_target is not None
        assert self._start_relative is not None
        lift = float(np.dot(ee - self._start_ee, self._direction))
        target_lift = 0.0
        drift = 0.0
        if target is not None:
            target_lift = float(np.dot(target - self._start_target, self._direction))
            drift = float(np.linalg.norm((target - ee) - self._start_relative))
            if drift > self.config.max_relative_drift_m:
                return self._fail(
                    f"tracked target relative drift {drift:.3f} m exceeds limit",
                )
        conditions = (
            target is not None
            and lift >= self.config.min_lift_m
            and target_lift >= self.config.min_target_lift_ratio * self.config.min_lift_m
            and drift <= self.config.max_relative_drift_m
        )
        if not conditions:
            self._hold_since = None
            return VerificationResult(
                VerificationState.PENDING,
                lift_m=lift,
                target_lift_m=target_lift,
                relative_target_drift_m=drift,
            )
        if self._hold_since is None:
            self._hold_since = stamp
        if stamp - self._hold_since < self.config.hold_time_s:
            return VerificationResult(
                VerificationState.PENDING,
                lift_m=lift,
                target_lift_m=target_lift,
                relative_target_drift_m=drift,
            )
        self._terminal = VerificationResult(
            VerificationState.SUCCESS,
            lift_m=lift,
            target_lift_m=target_lift,
            relative_target_drift_m=drift,
        )
        return self._terminal
