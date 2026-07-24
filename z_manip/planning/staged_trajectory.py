"""Pure-data contract for rolling, staged grasp trajectories.

This module deliberately owns neither ROS nor hardware.  It composes an IK
backend and an optional joint-space planner into four explicit phases:

``approach -> pregrasp -> grasp -> lift``.

In direct-approach mode (the default) every pre-contact phase is retimed as
ONE velocity-continuous chain -- the standoffs are slow-down vias, not stops --
while the aggregate contract still verifies that no hidden joint jump exists
at a phase boundary.  A measured joint state can therefore replace any phase
boundary and replan only the remaining suffix, and the staged fallback keeps
the classic independently retimed rest-to-rest phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import inspect
import math
from typing import Callable, Protocol, Sequence
from uuid import uuid4

import numpy as np

from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.planner import JointTrajectory
from z_manip.planning.grasp_pipeline import grasp_pregrasp_pose
from z_manip.planning.time_parameterization import (
    TimedJointTrajectory,
    TimeParameterizationConfig,
    retime_path,
)


class GraspStage(str, Enum):
    APPROACH = "approach"
    PREGRASP = "pregrasp"
    GRASP = "grasp"
    LIFT = "lift"


class SidePreference(str, Enum):
    AUTO = "auto"
    LEFT = "left"
    RIGHT = "right"


_STAGE_ORDER = tuple(GraspStage)


def _interpolate_arclength(path: np.ndarray, cumulative: np.ndarray, arclength: float) -> np.ndarray:
    """Sample a joint polyline at a cumulative chord length (rides the edges)."""

    total = float(cumulative[-1])
    clamped = min(max(arclength, 0.0), total)
    index = int(np.searchsorted(cumulative, clamped, side="right") - 1)
    index = min(max(index, 0), len(path) - 2)
    span = float(cumulative[index + 1] - cumulative[index])
    if span <= 1e-12:
        return np.array(path[index], dtype=float, copy=True)
    fraction = (clamped - float(cumulative[index])) / span
    return path[index] + fraction * (path[index + 1] - path[index])


def _readonly_vector(value: object, label: str, dof: int | None = None) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    expected = (dof,) if dof is not None else None
    if vector.ndim != 1 or (expected is not None and vector.shape != expected):
        suffix = "a vector" if expected is None else f"shape {expected}"
        raise ValueError(f"{label} must have {suffix}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be finite")
    vector.setflags(write=False)
    return vector


def _readonly_pose(value: object, label: str) -> np.ndarray:
    pose = np.array(value, dtype=float, copy=True)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{label} must be homogeneous")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError(f"{label} rotation must be orthonormal")
    pose.setflags(write=False)
    return pose


def _readonly_trajectory(value: TimedJointTrajectory) -> TimedJointTrajectory:
    positions = np.array(value.positions, dtype=float, copy=True)
    times_s = np.array(value.times_s, dtype=float, copy=True)
    positions.setflags(write=False)
    times_s.setflags(write=False)
    return TimedJointTrajectory(positions=positions, times_s=times_s)


@dataclass(frozen=True, eq=False)
class GraspTrajectoryTarget:
    """One selected grasp hypothesis in the arm planning frame."""

    grasp_pose: np.ndarray
    candidate_index: int | None = None
    score: float | None = None
    required_width_m: float | None = None
    source_frame: str = "piper_base_link"

    def __post_init__(self) -> None:
        object.__setattr__(self, "grasp_pose", _readonly_pose(self.grasp_pose, "grasp pose"))
        if self.candidate_index is not None and self.candidate_index < 0:
            raise ValueError("candidate index cannot be negative")
        for name in ("score", "required_width_m"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if self.required_width_m is not None and self.required_width_m <= 0.0:
            raise ValueError("required gripper width must be positive")
        if not self.source_frame:
            raise ValueError("source frame cannot be empty")

    @classmethod
    def from_candidates(
        cls,
        candidates: GraspCandidates,
        index: int,
    ) -> "GraspTrajectoryTarget":
        """Adapt the existing grasp-source batch without changing its schema."""

        poses = np.asarray(candidates.grasps, dtype=float)
        scores = np.asarray(candidates.scores, dtype=float)
        selected = int(index)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("candidate grasps must have shape (N, 4, 4)")
        if selected < 0 or selected >= len(poses) or scores.shape != (len(poses),):
            raise ValueError("candidate index or score array is invalid")
        width = None
        if candidates.widths is not None:
            widths = np.asarray(candidates.widths, dtype=float)
            if widths.shape != (len(poses),):
                raise ValueError("candidate widths must align with grasp poses")
            width = float(widths[selected])
        return cls(
            grasp_pose=poses[selected],
            candidate_index=selected,
            score=float(scores[selected]),
            required_width_m=width,
            source_frame=str(candidates.frame),
        )


@dataclass(frozen=True, eq=False)
class StagedGraspRequest:
    current_joints: np.ndarray
    target: GraspTrajectoryTarget
    pregrasp_offset_m: float = 0.10
    approach_clearance_m: float = 0.06
    lift_distance_m: float = 0.10
    lift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    side_preference: SidePreference = SidePreference.AUTO
    side_entry_offset_m: float = 0.0
    # Direct-approach is the accuracy/smoothness default: both standoffs become
    # velocity-continuous vias inside one blended transit-and-descent chain
    # into contact, instead of full stops between stages.
    # ``direct_approach=False`` keeps the classic per-stage rest-to-rest
    # fallback.
    direct_approach: bool = True
    # Speed-cap multiplier for the descent arcs (pregrasp->grasp) of the
    # blended chain so the tool creeps into contact for accuracy.  The profile
    # already decelerates to zero exactly at the grasp pose; this caps the
    # descent cruise.
    contact_speed_scale: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_joints",
            _readonly_vector(self.current_joints, "current joints"),
        )
        object.__setattr__(self, "side_preference", SidePreference(self.side_preference))
        object.__setattr__(self, "direct_approach", bool(self.direct_approach))
        positive = (
            self.pregrasp_offset_m,
            self.approach_clearance_m,
            self.lift_distance_m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("pregrasp, approach, and lift distances must be positive")
        if not np.isfinite(self.contact_speed_scale) or not 0.0 < self.contact_speed_scale <= 1.0:
            raise ValueError("contact speed scale must be within (0, 1]")
        if not np.isfinite(self.side_entry_offset_m) or self.side_entry_offset_m < 0.0:
            raise ValueError("side entry offset must be finite and non-negative")
        direction = np.asarray(self.lift_direction, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("lift direction must be a finite three-vector")
        if np.linalg.norm(direction) < 1e-9:
            raise ValueError("lift direction cannot be zero")


@dataclass(frozen=True, eq=False)
class GraspTrajectorySegment:
    stage: GraspStage
    target_pose: np.ndarray
    start_joints: np.ndarray
    goal_joints: np.ndarray
    trajectory: TimedJointTrajectory

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", GraspStage(self.stage))
        start = _readonly_vector(self.start_joints, "segment start joints")
        goal = _readonly_vector(self.goal_joints, "segment goal joints", len(start))
        pose = _readonly_pose(self.target_pose, "segment target pose")
        trajectory = _readonly_trajectory(self.trajectory)
        positions = trajectory.positions
        times = trajectory.times_s
        if positions.ndim != 2 or positions.shape[1:] != (len(start),) or len(positions) < 2:
            raise ValueError("segment trajectory must be a (T, dof) array with T >= 2")
        if times.shape != (len(positions),) or not np.all(np.isfinite(times)):
            raise ValueError("segment trajectory times must align with positions")
        if abs(float(times[0])) > 1e-9 or not np.all(np.diff(times) > 0.0):
            raise ValueError("segment times must start at zero and strictly increase")
        if not np.allclose(positions[0], start, atol=1e-8, rtol=0.0):
            raise ValueError("segment trajectory does not start at its declared state")
        if not np.allclose(positions[-1], goal, atol=1e-8, rtol=0.0):
            raise ValueError("segment trajectory does not end at its declared state")
        object.__setattr__(self, "target_pose", pose)
        object.__setattr__(self, "start_joints", start)
        object.__setattr__(self, "goal_joints", goal)
        object.__setattr__(self, "trajectory", trajectory)

    @property
    def duration_s(self) -> float:
        return float(self.trajectory.times_s[-1])


@dataclass(frozen=True, eq=False)
class StagedGraspTrajectory:
    """Immutable executable suffix, suitable for rolling replacement."""

    plan_id: str
    revision: int
    parent_plan_id: str | None
    target: GraspTrajectoryTarget
    side_preference: SidePreference
    segments: tuple[GraspTrajectorySegment, ...]
    pregrasp_offset_m: float = 0.10
    approach_clearance_m: float = 0.06
    lift_distance_m: float = 0.10
    lift_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    side_entry_offset_m: float = 0.0
    # Records which approach mode produced this plan so the executor receipt /
    # trace can attest it: ``True`` == one blended pregrasp->grasp descent,
    # ``False`` == the legacy discrete pregrasp stop.
    direct_approach: bool = True
    contact_speed_scale: float = 0.5
    schema: str = "z_manip.staged_grasp_trajectory.v1"

    def __post_init__(self) -> None:
        if not self.plan_id or self.revision < 0 or not self.segments:
            raise ValueError("a staged plan needs an id, revision, and at least one segment")
        object.__setattr__(self, "side_preference", SidePreference(self.side_preference))
        stages = tuple(segment.stage for segment in self.segments)
        indices = tuple(_STAGE_ORDER.index(stage) for stage in stages)
        if indices != tuple(range(indices[0], len(_STAGE_ORDER))):
            raise ValueError("segments must be a contiguous suffix of grasp stage order")
        for previous, following in zip(self.segments, self.segments[1:]):
            if not np.allclose(previous.goal_joints, following.start_joints, atol=1e-8):
                raise ValueError("staged trajectory contains a joint discontinuity")

    @property
    def start_stage(self) -> GraspStage:
        return self.segments[0].stage

    @property
    def duration_s(self) -> float:
        return float(sum(segment.duration_s for segment in self.segments))

    def segment(self, stage: GraspStage | str) -> GraspTrajectorySegment:
        selected = GraspStage(stage)
        for segment in self.segments:
            if segment.stage is selected:
                return segment
        raise KeyError(f"stage {selected.value!r} is not in this plan suffix")

    def flattened(self) -> TimedJointTrajectory:
        """Return the existing timed-trajectory schema with cumulative times."""

        positions: list[np.ndarray] = []
        times: list[np.ndarray] = []
        elapsed = 0.0
        for index, segment in enumerate(self.segments):
            stage_positions = segment.trajectory.positions
            stage_times = segment.trajectory.times_s + elapsed
            if index:
                stage_positions = stage_positions[1:]
                stage_times = stage_times[1:]
            positions.append(stage_positions)
            times.append(stage_times)
            elapsed += segment.duration_s
        return _readonly_trajectory(
            TimedJointTrajectory(
                positions=np.vstack(positions),
                times_s=np.concatenate(times),
            ),
        )

    def as_joint_trajectory(self, joint_names: Sequence[str]) -> JointTrajectory:
        """Adapt to the planner/executor's existing ``JointTrajectory`` type."""

        names = tuple(joint_names)
        dof = len(self.segments[0].start_joints)
        if len(names) != dof:
            raise ValueError("joint names must align with staged trajectory DOF")
        flattened = self.flattened()
        return JointTrajectory(
            joint_names=names,
            waypoints=flattened.positions,
            times=flattened.times_s,
        )


class _IKSolver(Protocol):
    def solve(self, target: np.ndarray, current: np.ndarray | None = None) -> object: ...


class _JointPlanner(Protocol):
    def plan_joint(self, start_joints: object, goal_joints: object, **kwargs: object) -> object: ...


def _accepts_timeout(callback: Callable[..., object]) -> bool:
    try:
        signature = inspect.signature(callback)
        signature.bind(np.zeros(1), np.zeros(1), timeout_s=1.0)
    except (TypeError, ValueError):
        return False
    return True


class StagedGraspTrajectoryBuilder:
    """Build and roll a four-stage trajectory without touching hardware."""

    def __init__(
        self,
        ik_solver: _IKSolver | Callable[[np.ndarray, np.ndarray], object],
        velocity_limits: object,
        acceleration_limits: object,
        *,
        joint_planner: _JointPlanner | Callable[[np.ndarray, np.ndarray], object] | None = None,
        time_config: TimeParameterizationConfig | None = None,
        planning_timeout_s: float = 4.0,
    ) -> None:
        self.ik_solver = ik_solver
        self.velocity_limits = _readonly_vector(velocity_limits, "velocity limits")
        self.acceleration_limits = _readonly_vector(
            acceleration_limits,
            "acceleration limits",
            len(self.velocity_limits),
        )
        if np.any(self.velocity_limits <= 0.0) or np.any(self.acceleration_limits <= 0.0):
            raise ValueError("joint dynamic limits must be positive")
        self.joint_planner = joint_planner
        self.time_config = time_config or TimeParameterizationConfig()
        self.planning_timeout_s = float(planning_timeout_s)
        if not np.isfinite(self.planning_timeout_s) or self.planning_timeout_s <= 0.0:
            raise ValueError("planning timeout must be finite and positive")

    @classmethod
    def from_kinematic_model(
        cls,
        ik_solver: _IKSolver | Callable[[np.ndarray, np.ndarray], object],
        model: object,
        acceleration_limits: object,
        **kwargs: object,
    ) -> "StagedGraspTrajectoryBuilder":
        """Use either ``KinematicChain`` or reduced Pinocchio velocity limits."""

        velocity = getattr(model, "arm_velocity_limits", None)
        if velocity is None:
            velocity = getattr(model, "velocity_limits", None)
        if velocity is None:
            raise ValueError("kinematic model does not expose arm velocity limits")
        return cls(ik_solver, velocity, acceleration_limits, **kwargs)

    def _solve(self, pose: np.ndarray, seed: np.ndarray) -> np.ndarray:
        callback = getattr(self.ik_solver, "solve", self.ik_solver)
        result = callback(pose, current=seed) if hasattr(self.ik_solver, "solve") else callback(pose, seed)
        joints = getattr(result, "joints", result)
        return _readonly_vector(joints, "IK solution", len(self.velocity_limits))

    def _path(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if self.joint_planner is None:
            return np.vstack((start, goal))
        callback = getattr(self.joint_planner, "plan_joint", self.joint_planner)
        if _accepts_timeout(callback):
            result = callback(start, goal, timeout_s=self.planning_timeout_s)
        else:
            result = callback(start, goal)
        path = getattr(result, "waypoints", result)
        values = np.asarray(path, dtype=float)
        if values.ndim != 2 or values.shape[1:] != (len(start),) or len(values) < 2:
            raise ValueError("joint planner must return a (T, dof) path")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint planner returned a non-finite path")
        if not np.allclose(values[0], start, atol=1e-7) or not np.allclose(values[-1], goal, atol=1e-7):
            raise ValueError("joint planner path endpoints do not match the requested segment")
        return values

    def _retime(self, path: np.ndarray) -> TimedJointTrajectory:
        if np.linalg.norm(path[-1] - path[0]) < 1e-12:
            duration = self.time_config.min_segment_time_s
            return TimedJointTrajectory(
                positions=np.vstack((path[0], path[-1])),
                times_s=np.asarray((0.0, duration)),
            )
        return retime_path(
            path,
            self.velocity_limits,
            self.acceleration_limits,
            self.time_config,
        )

    def _retime_direct_chain(
        self,
        stage_paths: Sequence[np.ndarray],
        speed_scales: Sequence[float],
    ) -> list[TimedJointTrajectory] | None:
        """Retime consecutive stage paths as ONE velocity-continuous chain.

        The collision-checked joint sub-paths are concatenated into a single
        polyline whose shared stage vertices become interior *vias* rather
        than stops.  A forward/backward path-velocity pass (TOPP-style)
        produces one time profile that is at rest only at the chain ends:

        * per-arc speed caps: transit at full scale, descent at the contact
          scale, so the tool still creeps into contact;
        * interior C0 corners get angle-tapered local speed caps -- a
          slow-down, never a dwell -- because a corner crossed at speed asks
          for a velocity-direction jump;
        * the backward pass starts braking as early as the acceleration
          limits require (e.g. a long transit brakes into a short
          contact-speed descent well before the standoff), which no single
          rest-to-rest quintic over the whole chain can express.

        The dense samples are then sliced back at the vias into per-stage
        trajectories, so the plan keeps its four-stage structure (and its
        reverse-replay corridor) while the flattened trajectory is one
        continuous velocity profile.  The geometry is untouched: samples ride
        the exact checked polyline edges, so collision coverage is identical
        to the staged fallback.  Returns ``None`` for degenerate chains; the
        caller then falls back to per-stage rest-to-rest retiming.
        """

        paths = [np.asarray(path, dtype=float) for path in stage_paths]
        scales = [float(scale) for scale in speed_scales]
        if len(paths) < 2 or len(paths) != len(scales):
            return None
        full = paths[0]
        via_indices: list[int] = []
        for path in paths[1:]:
            via_indices.append(len(full) - 1)
            full = np.vstack((full, path[1:]))
        edges = np.diff(full, axis=0)
        edge_lengths = np.linalg.norm(edges, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        total = float(cumulative[-1])
        via_arcs = [float(cumulative[index]) for index in via_indices]
        arc_bounds = np.asarray([0.0, *via_arcs, total])
        if total < 1e-9 or np.any(np.diff(arc_bounds) <= 1e-9):
            return None

        active = edge_lengths > 1e-12
        slope = np.zeros_like(edges)
        slope[active] = edges[active] / edge_lengths[active, None]
        velocity = self.velocity_limits * self.time_config.velocity_scale
        acceleration = self.acceleration_limits * self.time_config.acceleration_scale

        # Interior polyline corners.  The residual one-sample acceleration
        # spike exactly AT a corner is a C0-geometry artifact, not a command
        # the hardware must track instantaneously: targets are streamed at
        # the sample period and the controller's internal generator smooths
        # between them, exactly as the executor already streams C0 raw
        # polylines today.  Corner-adjacent samples are therefore excluded
        # from the discrete acceleration guard below; velocity is guarded
        # everywhere and the corner speed itself is capped by angle.
        corner_arcs: list[float] = []
        corner_windows: list[float] = []
        corner_fractions: list[float] = []
        for vertex in range(1, len(full) - 1):
            before = vertex - 1
            if not active[before] or not active[vertex]:
                continue
            cosine = float(np.clip(
                np.dot(slope[before], slope[vertex]), -1.0, 1.0,
            ))
            angle = math.acos(cosine)
            if angle < math.radians(0.5):
                continue
            corner_arcs.append(float(cumulative[vertex]))
            corner_windows.append(0.5 * float(min(
                edge_lengths[before], edge_lengths[vertex],
            )))
            corner_fractions.append(
                max(0.06, (1.0 - min(angle, math.pi) / math.pi) ** 3),
            )

        # Arc-length nodes: dense grid plus every polyline vertex (so edge
        # ownership never straddles a node) plus vias and corners.
        grid = np.unique(np.concatenate((
            np.linspace(0.0, total, 512),
            cumulative,
        )))
        node_edge = np.clip(
            np.searchsorted(cumulative, grid, side="right") - 1,
            0,
            len(edges) - 1,
        )
        interval_edge = node_edge[:-1]

        def _arc_scale(arc: float) -> float:
            index = int(np.clip(
                np.searchsorted(arc_bounds, arc, side="right") - 1,
                0,
                len(scales) - 1,
            ))
            return scales[index]

        interval_scale = np.asarray([
            _arc_scale(0.5 * float(grid[i] + grid[i + 1]))
            for i in range(len(grid) - 1)
        ])
        with np.errstate(divide="ignore", invalid="ignore"):
            per_edge_speed = np.where(
                np.abs(slope) > 1e-12,
                velocity / np.maximum(np.abs(slope), 1e-12),
                np.inf,
            ).min(axis=1)
            per_edge_accel = np.where(
                np.abs(slope) > 1e-12,
                acceleration / np.maximum(np.abs(slope), 1e-12),
                np.inf,
            ).min(axis=1)
        interval_speed_cap = interval_scale * per_edge_speed[interval_edge]
        interval_accel = per_edge_accel[interval_edge]
        if not np.all(np.isfinite(interval_speed_cap)):
            return None

        node_cap = np.empty_like(grid)
        node_cap[0] = 0.0
        node_cap[-1] = 0.0
        node_cap[1:-1] = np.minimum(
            interval_speed_cap[:-1],
            interval_speed_cap[1:],
        )
        for corner, fraction in zip(corner_arcs, corner_fractions):
            index = int(np.argmin(np.abs(grid - corner)))
            if 0 < index < len(grid) - 1:
                node_cap[index] = min(node_cap[index], node_cap[index] * fraction)

        # Forward/backward reachability under the acceleration limits.
        speeds = node_cap.copy()
        deltas = np.diff(grid)
        for i in range(len(grid) - 1):
            reachable = math.sqrt(
                speeds[i] ** 2 + 2.0 * float(interval_accel[i]) * float(deltas[i]),
            )
            speeds[i + 1] = min(speeds[i + 1], reachable)
        for i in range(len(grid) - 2, -1, -1):
            reachable = math.sqrt(
                speeds[i + 1] ** 2 + 2.0 * float(interval_accel[i]) * float(deltas[i]),
            )
            speeds[i] = min(speeds[i], reachable)

        # Integrate time along the arc (speed floored to keep the ends finite).
        mean_speed = np.maximum(0.5 * (speeds[:-1] + speeds[1:]), 1e-4)
        node_times = np.concatenate(([0.0], np.cumsum(deltas / mean_speed)))
        duration = float(node_times[-1])
        if not math.isfinite(duration) or duration <= 0.0:
            return None

        via_times = np.asarray([
            float(np.interp(value, grid, node_times)) for value in via_arcs
        ])
        samples = max(
            2,
            int(np.ceil(duration / self.time_config.sample_period_s)) + 1,
        )
        base = np.linspace(0.0, duration, samples)
        keep = np.min(
            np.abs(base[:, None] - via_times[None, :]),
            axis=1,
        ) > 1e-6
        times = np.sort(np.concatenate((base[keep], via_times)))
        arc_positions = np.interp(times, node_times, grid)
        positions = np.stack([
            _interpolate_arclength(full, cumulative, value)
            for value in arc_positions
        ])

        def _limit_ratio(
            local_times: np.ndarray,
            local_positions: np.ndarray,
            local_arcs: np.ndarray,
        ) -> float:
            """Worst velocity/acceleration utilization of the sampled profile.

            Velocity is guarded at every sample; the discrete acceleration is
            guarded away from C0 corner vertices (see corner note above).
            """
            sample_deltas = np.diff(local_times)
            velocities = np.diff(local_positions, axis=0) / sample_deltas[:, None]
            worst = float(np.max(np.max(np.abs(velocities) / velocity, axis=1)))
            if len(velocities) > 1:
                midpoints = 0.5 * (local_times[:-1] + local_times[1:])
                accelerations = (
                    np.diff(velocities, axis=0) / np.diff(midpoints)[:, None]
                )
                acceleration_ratios = np.max(
                    np.abs(accelerations) / acceleration, axis=1,
                )
                exclude = np.zeros(len(acceleration_ratios), dtype=bool)
                sample_arcs = local_arcs[1:-1]
                for corner, window in zip(corner_arcs, corner_windows):
                    exclude |= np.abs(sample_arcs - corner) <= max(window, 1e-6)
                for via in via_arcs:
                    exclude |= np.abs(sample_arcs - via) <= 1e-6
                acceleration_ratios = acceleration_ratios[~exclude]
                if len(acceleration_ratios):
                    worst = max(
                        worst,
                        math.sqrt(max(float(np.max(acceleration_ratios)), 0.0)),
                    )
            return worst

        # Guarantee pass: discretization error can push a sample slightly
        # over a limit; uniform time dilation by ``f`` scales velocity by 1/f
        # and acceleration by 1/f^2 with the positions (the checked geometry)
        # untouched.
        factor = _limit_ratio(times, positions, arc_positions)
        if factor > 1.001:
            times = times * (factor * 1.02)
            via_times = via_times * (factor * 1.02)

        result: list[TimedJointTrajectory] = []
        previous_index = 0
        for arc, path in enumerate(paths):
            if arc < len(via_times):
                end_index = int(np.flatnonzero(
                    np.isclose(times, via_times[arc], rtol=0.0, atol=1e-9),
                )[0])
            else:
                end_index = len(times) - 1
            segment_positions = positions[previous_index:end_index + 1].copy()
            segment_times = (
                times[previous_index:end_index + 1] - times[previous_index]
            )
            # Pin shared endpoints to the exact checked joints (no float drift).
            segment_positions[0] = path[0]
            segment_positions[-1] = path[-1]
            if len(segment_positions) < 2:
                return None
            result.append(TimedJointTrajectory(
                positions=segment_positions,
                times_s=segment_times,
            ))
            previous_index = end_index
        return result

    @staticmethod
    def _poses(request: StagedGraspRequest) -> dict[GraspStage, np.ndarray]:
        grasp = np.array(request.target.grasp_pose, copy=True)
        pregrasp = grasp_pregrasp_pose(grasp, request.pregrasp_offset_m)
        approach = grasp_pregrasp_pose(
            grasp,
            request.pregrasp_offset_m + request.approach_clearance_m,
        )
        if request.side_preference is not SidePreference.AUTO and request.side_entry_offset_m:
            sign = 1.0 if request.side_preference is SidePreference.LEFT else -1.0
            # Side bias tapers to zero at contact; the grasp pose itself remains
            # exactly the candidate pose produced by perception.
            approach[1, 3] += sign * request.side_entry_offset_m
            pregrasp[1, 3] += sign * request.side_entry_offset_m * 0.5
        lift = grasp.copy()
        direction = np.asarray(request.lift_direction, dtype=float)
        lift[:3, 3] += request.lift_distance_m * direction / np.linalg.norm(direction)
        return {
            GraspStage.APPROACH: approach,
            GraspStage.PREGRASP: pregrasp,
            GraspStage.GRASP: grasp,
            GraspStage.LIFT: lift,
        }

    def build(
        self,
        request: StagedGraspRequest,
        *,
        start_stage: GraspStage | str = GraspStage.APPROACH,
        revision: int = 0,
        parent_plan_id: str | None = None,
        plan_id: str | None = None,
    ) -> StagedGraspTrajectory:
        current = _readonly_vector(
            request.current_joints,
            "current joints",
            len(self.velocity_limits),
        )
        selected = GraspStage(start_stage)
        start_index = _STAGE_ORDER.index(selected)
        poses = self._poses(request)
        # First resolve every stage's collision-checked joint path, then retime.
        # Splitting the two phases lets the direct-approach mode fuse the
        # pregrasp and grasp descents into one blended, velocity-continuous
        # profile without duplicating IK or path planning.
        stage_paths: dict[GraspStage, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for stage in _STAGE_ORDER[start_index:]:
            goal = self._solve(poses[stage], current)
            path = self._path(current, goal)
            stage_paths[stage] = (current, goal, path)
            current = goal

        # The blended chain covers every pre-contact stage present: transit
        # (APPROACH) at full speed and the descent (PREGRASP + GRASP) at the
        # contact scale, with the speed ramped down inside the transit tail.
        # Both standoffs therefore become velocity-continuous vias; motion
        # comes to rest only at the grasp contact (for the close) and after
        # the lift.  LIFT stays a separate rest-to-rest segment because the
        # gripper must close while the arm is stationary at the grasp pose.
        chain_stages = [
            stage
            for stage in (GraspStage.APPROACH, GraspStage.PREGRASP, GraspStage.GRASP)
            if stage in stage_paths
        ]
        chain_scales = {
            GraspStage.APPROACH: 1.0,
            GraspStage.PREGRASP: request.contact_speed_scale,
            GraspStage.GRASP: request.contact_speed_scale,
        }
        trajectories: dict[GraspStage, TimedJointTrajectory] = {}
        if request.direct_approach and len(chain_stages) >= 2:
            chained = self._retime_direct_chain(
                [stage_paths[stage][2] for stage in chain_stages],
                [chain_scales[stage] for stage in chain_stages],
            )
            if chained is not None:
                for stage, trajectory in zip(chain_stages, chained):
                    trajectories[stage] = trajectory
        for stage, (_, _, path) in stage_paths.items():
            if stage not in trajectories:
                trajectories[stage] = self._retime(path)

        segments: list[GraspTrajectorySegment] = [
            GraspTrajectorySegment(
                stage=stage,
                target_pose=poses[stage],
                start_joints=start_joints,
                goal_joints=goal_joints,
                trajectory=trajectories[stage],
            )
            for stage, (start_joints, goal_joints, _) in stage_paths.items()
        ]
        return StagedGraspTrajectory(
            plan_id=plan_id or uuid4().hex,
            revision=int(revision),
            parent_plan_id=parent_plan_id,
            target=request.target,
            side_preference=request.side_preference,
            segments=tuple(segments),
            pregrasp_offset_m=request.pregrasp_offset_m,
            approach_clearance_m=request.approach_clearance_m,
            lift_distance_m=request.lift_distance_m,
            lift_direction=request.lift_direction,
            side_entry_offset_m=request.side_entry_offset_m,
            direct_approach=request.direct_approach,
            contact_speed_scale=request.contact_speed_scale,
        )

    def replan_remaining(
        self,
        previous: StagedGraspTrajectory,
        measured_joints: object,
        *,
        from_stage: GraspStage | str,
        updated_grasp_pose: object | None = None,
        pregrasp_offset_m: float | None = None,
        approach_clearance_m: float | None = None,
        lift_distance_m: float | None = None,
        lift_direction: tuple[float, float, float] | None = None,
        side_entry_offset_m: float | None = None,
        direct_approach: bool | None = None,
        contact_speed_scale: float | None = None,
    ) -> StagedGraspTrajectory:
        """Replace a remaining suffix from a fresh measured joint boundary."""

        target = previous.target
        if updated_grasp_pose is not None:
            target = replace(target, grasp_pose=_readonly_pose(updated_grasp_pose, "grasp pose"))
        request = StagedGraspRequest(
            current_joints=np.asarray(measured_joints, dtype=float),
            target=target,
            pregrasp_offset_m=(
                previous.pregrasp_offset_m
                if pregrasp_offset_m is None
                else pregrasp_offset_m
            ),
            approach_clearance_m=(
                previous.approach_clearance_m
                if approach_clearance_m is None
                else approach_clearance_m
            ),
            lift_distance_m=(
                previous.lift_distance_m
                if lift_distance_m is None
                else lift_distance_m
            ),
            lift_direction=(
                previous.lift_direction if lift_direction is None else lift_direction
            ),
            side_preference=previous.side_preference,
            side_entry_offset_m=(
                previous.side_entry_offset_m
                if side_entry_offset_m is None
                else side_entry_offset_m
            ),
            direct_approach=(
                previous.direct_approach
                if direct_approach is None
                else direct_approach
            ),
            contact_speed_scale=(
                previous.contact_speed_scale
                if contact_speed_scale is None
                else contact_speed_scale
            ),
        )
        return self.build(
            request,
            start_stage=from_stage,
            revision=previous.revision + 1,
            parent_plan_id=previous.plan_id,
        )


__all__ = [
    "GraspStage",
    "GraspTrajectorySegment",
    "GraspTrajectoryTarget",
    "SidePreference",
    "StagedGraspRequest",
    "StagedGraspTrajectory",
    "StagedGraspTrajectoryBuilder",
]
