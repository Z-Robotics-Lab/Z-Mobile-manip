"""Conservative jerk-continuous time parameterization for joint paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeParameterizationConfig:
    sample_period_s: float = 0.02
    min_segment_time_s: float = 0.10
    velocity_scale: float = 0.65
    acceleration_scale: float = 0.55

    def __post_init__(self) -> None:
        if self.sample_period_s <= 0.0 or self.min_segment_time_s <= 0.0:
            raise ValueError("trajectory sample period and segment time must be positive")
        if not 0.0 < self.velocity_scale <= 1.0:
            raise ValueError("velocity scale must be in (0, 1]")
        if not 0.0 < self.acceleration_scale <= 1.0:
            raise ValueError("acceleration scale must be in (0, 1]")


@dataclass(frozen=True, eq=False)
class TimedJointTrajectory:
    positions: np.ndarray
    times_s: np.ndarray


def _validate_limits(values: object, dof: int, label: str) -> np.ndarray:
    limits = np.asarray(values, dtype=float)
    if (
        limits.shape != (dof,)
        or not np.all(np.isfinite(limits))
        or np.any(limits <= 0.0)
    ):
        raise ValueError(
            f"{label} limits must align with the path and be finite positive",
        )
    return limits


def retime_path(
    waypoints: object,
    velocity_limits: object,
    acceleration_limits: object,
    config: TimeParameterizationConfig | None = None,
) -> TimedJointTrajectory:
    """Sample rest-to-rest quintic segments under per-joint dynamic limits.

    The standard quintic smoothstep has maximum normalized velocity 1.875 and
    maximum normalized acceleration about 5.774. Segment durations are chosen
    analytically from those bounds. Stopping at path vertices is conservative,
    predictable, and suitable for the first hardware-independent executor.
    """

    settings = config or TimeParameterizationConfig()
    path = np.asarray(waypoints, dtype=float)
    if path.ndim != 2 or len(path) < 2:
        raise ValueError("joint path needs at least two waypoints")
    if not np.all(np.isfinite(path)):
        raise ValueError("joint path contains a non-finite value")
    dof = path.shape[1]
    velocity = (
        _validate_limits(velocity_limits, dof, "velocity")
        * settings.velocity_scale
    )
    acceleration = (
        _validate_limits(acceleration_limits, dof, "acceleration")
        * settings.acceleration_scale
    )
    if np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) < 1e-12):
        raise ValueError("joint path contains no motion")

    positions = [path[0].copy()]
    times = [0.0]
    elapsed = 0.0
    for start, end in zip(path, path[1:]):
        delta = end - start
        if np.linalg.norm(delta) < 1e-12:
            continue
        duration_velocity = float(np.max(1.875 * np.abs(delta) / velocity))
        duration_acceleration = float(
            np.max(np.sqrt(5.774 * np.abs(delta) / acceleration)),
        )
        duration = max(
            settings.min_segment_time_s,
            duration_velocity,
            duration_acceleration,
        )
        samples = max(2, int(np.ceil(duration / settings.sample_period_s)) + 1)
        for local_time in np.linspace(0.0, duration, samples)[1:]:
            tau = local_time / duration
            blend = 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5
            positions.append(start + blend * delta)
            times.append(elapsed + local_time)
        elapsed += duration
    return TimedJointTrajectory(np.asarray(positions), np.asarray(times))
