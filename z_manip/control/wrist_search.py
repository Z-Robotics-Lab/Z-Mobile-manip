"""Pure, transport-free policy for bounded wrist-camera target search.

This module deliberately cannot open ROS, CAN, SSH, or a robot SDK.  It only
turns measured joint feedback and detector observations into a finite sequence
of fixed wrist viewpoints.  A runtime adapter must separately authorize and
execute each returned target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np


class WristSearchPhase(str, Enum):
    IDLE = "idle"
    MOVE = "move"
    SETTLE = "settle"
    OBSERVE = "observe"
    FOUND = "found"
    EXHAUSTED = "exhausted"
    STOPPED = "stopped"


@dataclass(frozen=True)
class WristSearchConfig:
    """Finite search envelope around a measured, known-clear anchor pose."""

    yaw_joint_index: int = 3  # PiPER J4: wrist azimuth around the forearm.
    pitch_joint_index: int = 4  # PiPER J5: wrist elevation.
    yaw_step_rad: float = math.radians(18.0)
    pitch_step_rad: float = math.radians(14.0)
    max_yaw_offset_rad: float = math.radians(36.0)
    max_pitch_offset_rad: float = math.radians(28.0)
    settle_s: float = 0.35
    detector_hz: float = 5.0
    observations_per_view: int = 3
    # The resident grounding service already rejects candidates below 0.20
    # and applies finite/border/area gates. Search adds temporal confirmation,
    # not a contradictory second confidence gate.
    confidence_threshold: float = 0.20
    confirmations_required: int = 2
    joint_tolerance_rad: float = math.radians(1.0)
    # At 5% each smooth 18 degree wrist edge takes about 1.9 seconds.  The
    # deadline therefore covers the finite grid once without permitting an
    # unbounded scan.
    max_search_s: float = 75.0

    def __post_init__(self) -> None:
        if self.yaw_joint_index == self.pitch_joint_index:
            raise ValueError("wrist search joints must be distinct")
        if min(self.yaw_joint_index, self.pitch_joint_index) < 0:
            raise ValueError("wrist search joint indices must be non-negative")
        positive = (
            self.yaw_step_rad,
            self.pitch_step_rad,
            self.max_yaw_offset_rad,
            self.max_pitch_offset_rad,
            self.settle_s,
            self.detector_hz,
            self.joint_tolerance_rad,
            self.max_search_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("wrist search limits must be finite and positive")
        if self.yaw_step_rad > self.max_yaw_offset_rad:
            raise ValueError("yaw step exceeds the search envelope")
        if self.pitch_step_rad > self.max_pitch_offset_rad:
            raise ValueError("pitch step exceeds the search envelope")
        if self.observations_per_view < 1 or self.confirmations_required < 1:
            raise ValueError("wrist search observation counts must be positive")
        if self.confirmations_required > self.observations_per_view:
            raise ValueError("confirmations cannot exceed observations per view")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in (0, 1]")

    @property
    def observation_period_s(self) -> float:
        return 1.0 / self.detector_hz


@dataclass(frozen=True)
class WristView:
    index: int
    yaw_offset_rad: float
    pitch_offset_rad: float


@dataclass(frozen=True)
class WristSearchDecision:
    phase: WristSearchPhase
    view: WristView | None
    target_joints_rad: tuple[float, ...] | None
    confidence: float | None = None
    confirmations: int = 0
    message: str = ""


class BoundedWristSearch:
    """Deterministic raster search with multi-frame detector confirmation."""

    def __init__(self, config: WristSearchConfig | None = None) -> None:
        self.config = config or WristSearchConfig()
        self._views = self._make_views()
        self.reset()

    def _make_views(self) -> tuple[WristView, ...]:
        # Near-center views are intentionally visited first.  A deterministic
        # nearest-neighbour walk reduces large cross-sweeps; the executor still
        # interpolates every selected edge into small measured increments.
        yaw_levels = [0.0]
        radius = self.config.yaw_step_rad
        while radius <= self.config.max_yaw_offset_rad + 1e-9:
            yaw_levels.extend((radius, -radius))
            radius += self.config.yaw_step_rad
        pitch_levels = [0.0]
        radius = self.config.pitch_step_rad
        while radius <= self.config.max_pitch_offset_rad + 1e-9:
            pitch_levels.extend((radius, -radius))
            radius += self.config.pitch_step_rad
        remaining = {
            (yaw, pitch)
            for pitch in pitch_levels
            for yaw in yaw_levels
            if not (yaw == 0.0 and pitch == 0.0)
        }
        offsets: list[tuple[float, float]] = [(0.0, 0.0)]
        while remaining:
            current_yaw, current_pitch = offsets[-1]
            next_offset = min(
                remaining,
                key=lambda value: (
                    math.hypot(value[0] - current_yaw, value[1] - current_pitch),
                    math.hypot(value[0], value[1]),
                    abs(value[1]),
                    abs(value[0]),
                    value[1],
                    value[0],
                ),
            )
            remaining.remove(next_offset)
            offsets.append(next_offset)
        return tuple(
            WristView(index=index, yaw_offset_rad=yaw, pitch_offset_rad=pitch)
            for index, (yaw, pitch) in enumerate(offsets)
        )

    @property
    def views(self) -> tuple[WristView, ...]:
        return self._views

    def reset(self) -> None:
        self.phase = WristSearchPhase.IDLE
        self.anchor_joints_rad: np.ndarray | None = None
        self.started_at_s: float | None = None
        self.view_started_at_s: float | None = None
        self.last_observation_at_s: float | None = None
        self.view_index = 0
        self.observation_count = 0
        self.confirmation_count = 0
        self.last_confidence: float | None = None

    def start(self, joints_rad: Sequence[float], *, now_s: float) -> WristSearchDecision:
        joints = np.asarray(joints_rad, dtype=float)
        required = max(self.config.yaw_joint_index, self.config.pitch_joint_index) + 1
        if joints.ndim != 1 or len(joints) < required or not np.isfinite(joints).all():
            raise ValueError("wrist search anchor must contain finite measured joints")
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("wrist search timestamp must be finite")
        self.reset()
        self.anchor_joints_rad = joints.copy()
        self.started_at_s = now
        self.view_started_at_s = now
        self.phase = WristSearchPhase.MOVE
        return self._decision(message="move to the first bounded search view")

    def stop(self) -> WristSearchDecision:
        self.phase = WristSearchPhase.STOPPED
        return self._decision(message="wrist search stopped")

    def _target(self) -> np.ndarray:
        if self.anchor_joints_rad is None:
            raise RuntimeError("wrist search has no measured anchor")
        view = self._views[self.view_index]
        target = self.anchor_joints_rad.copy()
        target[self.config.yaw_joint_index] += view.yaw_offset_rad
        target[self.config.pitch_joint_index] += view.pitch_offset_rad
        return target

    def _decision(
        self,
        *,
        confidence: float | None = None,
        message: str = "",
    ) -> WristSearchDecision:
        view = None
        target = None
        if self.anchor_joints_rad is not None and self.view_index < len(self._views):
            view = self._views[self.view_index]
            target = tuple(float(value) for value in self._target())
        return WristSearchDecision(
            phase=self.phase,
            view=view,
            target_joints_rad=target,
            confidence=confidence,
            confirmations=self.confirmation_count,
            message=message,
        )

    def update_motion(self, joints_rad: Sequence[float], *, now_s: float) -> WristSearchDecision:
        if self.phase not in {WristSearchPhase.MOVE, WristSearchPhase.SETTLE}:
            raise RuntimeError("wrist search is not waiting for motion")
        now = self._validate_time(now_s)
        joints = np.asarray(joints_rad, dtype=float)
        target = self._target()
        if joints.shape != target.shape or not np.isfinite(joints).all():
            raise ValueError("wrist search feedback does not match the anchor")
        if float(np.max(np.abs(joints - target))) > self.config.joint_tolerance_rad:
            self.phase = WristSearchPhase.MOVE
            return self._decision(message="waiting for measured wrist viewpoint")
        if self.phase is WristSearchPhase.MOVE:
            self.phase = WristSearchPhase.SETTLE
            self.view_started_at_s = now
        if now - float(self.view_started_at_s) < self.config.settle_s:
            return self._decision(message="letting the wrist camera settle")
        self.phase = WristSearchPhase.OBSERVE
        self.observation_count = 0
        self.confirmation_count = 0
        self.last_observation_at_s = None
        return self._decision(message="sampling detector confidence")

    def observe(
        self,
        *,
        visible: bool,
        confidence: float | None,
        now_s: float,
    ) -> WristSearchDecision:
        if self.phase is not WristSearchPhase.OBSERVE:
            raise RuntimeError("wrist search is not accepting observations")
        now = self._validate_time(now_s)
        if (
            self.last_observation_at_s is not None
            and now - self.last_observation_at_s + 1e-9
            < self.config.observation_period_s
        ):
            return self._decision(
                confidence=self.last_confidence,
                message="waiting for the next detector cadence",
            )
        value = None if confidence is None else float(confidence)
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError("detector confidence must be finite and within [0, 1]")
        self.last_observation_at_s = now
        self.last_confidence = value
        self.observation_count += 1
        qualified = visible and value is not None and value >= self.config.confidence_threshold
        self.confirmation_count = self.confirmation_count + 1 if qualified else 0
        if self.confirmation_count >= self.config.confirmations_required:
            self.phase = WristSearchPhase.FOUND
            return self._decision(confidence=value, message="target confirmed")
        if self.observation_count < self.config.observations_per_view:
            return self._decision(confidence=value, message="collecting detector confirmations")
        self.view_index += 1
        self.observation_count = 0
        self.confirmation_count = 0
        if self.view_index >= len(self._views):
            self.phase = WristSearchPhase.EXHAUSTED
            return WristSearchDecision(
                phase=self.phase,
                view=None,
                target_joints_rad=None,
                confidence=value,
                message="bounded wrist search exhausted",
            )
        self.phase = WristSearchPhase.MOVE
        self.view_started_at_s = now
        return self._decision(confidence=value, message="advance to the next search view")

    def _validate_time(self, now_s: float) -> float:
        now = float(now_s)
        if self.started_at_s is None or not math.isfinite(now) or now < self.started_at_s:
            raise ValueError("wrist search time must be finite and monotonic")
        if now - self.started_at_s > self.config.max_search_s:
            self.phase = WristSearchPhase.EXHAUSTED
            raise TimeoutError("bounded wrist search exceeded its hard deadline")
        return now
