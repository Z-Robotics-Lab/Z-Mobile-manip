"""Conservative jerk-continuous time parameterization for joint paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeParameterizationConfig:
    sample_period_s: float = 0.02
    min_segment_time_s: float = 0.10
    velocity_scale: float = 0.65
    acceleration_scale: float = 0.55
    # Share of the acceleration cap reserved for the direction change at a path
    # corner; the rest drives along-path ramping.  A polyline vertex reverses
    # commanded velocity direction inside ONE sample period, so this is what
    # decides how fast a corner may be flown.  0.0 restores the legacy
    # rest-to-rest-at-every-vertex behaviour exactly.
    #
    # ponytail: measured on the deployed side-grasp reach -- 0.6 gives peak
    # |a| 0.58x cap at 9.0 s; 0.9 gives 0.83x at 15.4 s.  Re-measure if the
    # arm's real tracking disagrees; this is the calibration knob.
    corner_acceleration_budget: float = 0.6

    def __post_init__(self) -> None:
        if self.sample_period_s <= 0.0 or self.min_segment_time_s <= 0.0:
            raise ValueError("trajectory sample period and segment time must be positive")
        if not 0.0 < self.velocity_scale <= 1.0:
            raise ValueError("velocity scale must be in (0, 1]")
        if not 0.0 < self.acceleration_scale <= 1.0:
            raise ValueError("acceleration scale must be in (0, 1]")
        if not 0.0 <= self.corner_acceleration_budget < 1.0:
            raise ValueError("corner acceleration budget must be in [0, 1)")


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
    """Sample ONE velocity-continuous profile along the path's own polyline.

    Interior vertices are *vias*, not stops.  The commanded samples ride the
    exact polyline edges, so collision coverage is identical to the geometry
    that was checked -- no corner is ever shortcut and no new edge is created.

    Speed is bounded three ways: by the per-joint velocity cap on each edge,
    by the per-joint acceleration cap while ramping along an edge, and -- the
    one a per-segment retimer never had to think about -- by the direction
    change at each corner, which a moving arm must absorb within one sample
    period.  A forward/backward pass then slows down only *near* the corners
    that need it and cruises between them.

    ``corner_acceleration_budget = 0.0`` reproduces the legacy behaviour of
    stopping dead at every vertex.
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

    budget = settings.corner_acceleration_budget
    if budget <= 0.0:
        return _retime_rest_to_rest(path, velocity, acceleration, settings)

    # Drop zero-length edges; they carry no direction and would divide by zero.
    keep = np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-12
    vertices = np.vstack((path[0], path[1:][keep]))
    edges = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    headings = edges / lengths[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    period = settings.sample_period_s
    count = len(lengths)

    # Speed ceilings. Along an edge the joint rates are speed * heading, so the
    # velocity cap gives one ceiling per edge.
    edge_ceiling = np.min(velocity / np.abs(headings).clip(1e-12), axis=1)
    node_speed = np.empty(count + 1)
    node_speed[0] = node_speed[count] = 0.0
    # A vertex always gets its own sample, and the sampler keeps neighbouring
    # samples at least half a period away, so half a period is the shortest
    # window a corner's direction change can be commanded across.  Budgeting
    # against a full period would under-price every corner by 2x.
    corner_window = 0.5 * period
    for index in range(1, count):
        # Crossing a corner swings the joint rates by speed * (u_out - u_in)
        # within one sample; that swing must fit the reserved share.
        swing = np.abs(headings[index] - headings[index - 1])
        worst = float(np.max(swing / (budget * acceleration * corner_window)))
        corner_ceiling = np.inf if worst < 1e-12 else 1.0 / worst
        node_speed[index] = min(
            corner_ceiling,
            edge_ceiling[index - 1],
            edge_ceiling[index],
        )
    # Whatever the corners did not claim is available for ramping along a path.
    ramp = (1.0 - budget) * float(np.min(acceleration))
    for index in range(1, count + 1):
        node_speed[index] = min(
            node_speed[index],
            math.sqrt(node_speed[index - 1] ** 2 + 2.0 * ramp * lengths[index - 1]),
        )
    for index in range(count - 1, -1, -1):
        node_speed[index] = min(
            node_speed[index],
            math.sqrt(node_speed[index + 1] ** 2 + 2.0 * ramp * lengths[index]),
        )

    # Accelerate / cruise / decelerate within each edge.  A plain trapezoid
    # between the two node speeds cannot express an edge whose BOTH ends are
    # slow -- a two-waypoint path is exactly that, and it would integrate to an
    # unbounded duration.
    profile = [
        _edge_profile(node_speed[k], node_speed[k + 1], lengths[k], ramp,
                      edge_ceiling[k])
        for k in range(count)
    ]
    node_time = np.concatenate(([0.0], np.cumsum([leg.duration for leg in profile])))
    span = float(node_time[count])
    total = max(span, settings.min_segment_time_s)
    # Stretching to the floor slows the profile down; it never speeds it up,
    # so the velocity and acceleration bounds above still hold.
    stretch = total / span if span > 1e-12 else 1.0

    samples = max(2, int(np.ceil(total / period)) + 1)
    grid = np.linspace(0.0, total, samples)
    # Land a sample exactly on every original vertex.  Callers slice staged
    # trajectories at a vertex (the pregrasp standoff), and a slice endpoint
    # that merely lands NEAR the checked joint would have to be nudged onto it
    # -- a step, at the one sample where the profile is moving slowest.
    # Drop grid points that would crowd a vertex time: two samples a
    # microsecond apart are a division by nearly zero for anything that
    # differentiates this trajectory, and buy no fidelity.
    vertex_times = node_time * stretch
    crowded = np.min(
        np.abs(grid[:, None] - vertex_times[None, :]),
        axis=1,
    ) < 0.5 * period
    times = np.unique(np.concatenate((grid[~crowded], vertex_times)))
    positions = np.empty((len(times), dof))
    for sample, moment in enumerate(times):
        clock = moment / stretch
        index = int(np.clip(np.searchsorted(node_time, clock) - 1, 0, count - 1))
        travelled = profile[index].distance_at(clock - node_time[index])
        positions[sample] = _on_polyline(
            vertices,
            cumulative,
            cumulative[index] + travelled,
        )
    # Pin the endpoints so the caller's checked start/goal survive float drift.
    positions[0] = vertices[0]
    positions[-1] = vertices[-1]
    return TimedJointTrajectory(positions, times)


@dataclass(frozen=True)
class _EdgeProfile:
    """One edge's accelerate/cruise/decelerate speed profile."""

    entry_speed: float
    peak_speed: float
    ramp: float
    accel_time: float
    cruise_time: float
    decel_time: float
    accel_distance: float
    cruise_distance: float

    @property
    def duration(self) -> float:
        return self.accel_time + self.cruise_time + self.decel_time

    def distance_at(self, elapsed: float) -> float:
        moment = min(max(elapsed, 0.0), self.duration)
        if moment <= self.accel_time:
            return self.entry_speed * moment + 0.5 * self.ramp * moment ** 2
        moment -= self.accel_time
        if moment <= self.cruise_time:
            return self.accel_distance + self.peak_speed * moment
        moment -= self.cruise_time
        return (
            self.accel_distance
            + self.cruise_distance
            + self.peak_speed * moment
            - 0.5 * self.ramp * moment ** 2
        )


def _edge_profile(
    entry_speed: float,
    exit_speed: float,
    length: float,
    ramp: float,
    ceiling: float,
) -> _EdgeProfile:
    """Fastest accel/cruise/decel profile across one edge under ``ramp``."""

    # Highest speed reachable that still leaves room to decelerate to the exit.
    peak = min(
        ceiling,
        math.sqrt(max(0.5 * (entry_speed ** 2 + exit_speed ** 2) + ramp * length, 0.0)),
    )
    peak = max(peak, entry_speed, exit_speed, 1e-9)
    accel_distance = max((peak ** 2 - entry_speed ** 2) / (2.0 * ramp), 0.0)
    decel_distance = max((peak ** 2 - exit_speed ** 2) / (2.0 * ramp), 0.0)
    cruise_distance = max(length - accel_distance - decel_distance, 0.0)
    return _EdgeProfile(
        entry_speed=entry_speed,
        peak_speed=peak,
        ramp=ramp,
        accel_time=max(peak - entry_speed, 0.0) / ramp,
        cruise_time=cruise_distance / peak,
        decel_time=max(peak - exit_speed, 0.0) / ramp,
        accel_distance=accel_distance,
        cruise_distance=cruise_distance,
    )


def _on_polyline(
    vertices: np.ndarray,
    cumulative: np.ndarray,
    arclength: float,
) -> np.ndarray:
    """Point at ``arclength`` along the polyline, riding its exact edges."""

    index = int(np.clip(
        np.searchsorted(cumulative, arclength) - 1,
        0,
        len(vertices) - 2,
    ))
    span = cumulative[index + 1] - cumulative[index]
    fraction = 0.0 if span <= 1e-12 else (arclength - cumulative[index]) / span
    fraction = min(max(fraction, 0.0), 1.0)
    return vertices[index] + fraction * (vertices[index + 1] - vertices[index])


def _retime_rest_to_rest(
    path: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    settings: TimeParameterizationConfig,
) -> TimedJointTrajectory:
    """Legacy shape: a rest-to-rest quintic per edge, stopping at every vertex."""

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
