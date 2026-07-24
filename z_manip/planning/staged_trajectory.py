"""Pure-data contract for rolling, staged grasp trajectories.

This module deliberately owns neither ROS nor hardware.  It composes an IK
backend and an optional joint-space planner into four explicit phases:

``approach -> pregrasp -> grasp -> lift``.

Every phase is independently retimed, while the aggregate contract verifies
that no hidden joint jump exists at a phase boundary.  A measured joint state
can therefore replace any phase boundary and replan only the remaining suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import inspect
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


def _quintic_blend(u: float) -> float:
    """Normalized rest-to-rest 5th-order smoothstep ``10u^3-15u^4+6u^5``.

    ``s(0)=0``, ``s(1)=1`` with zero first and second derivative at both ends;
    peak normalized velocity ``1.875`` and acceleration ``5.774`` at the middle.
    """

    return (u * u * u) * (10.0 + u * (-15.0 + 6.0 * u))


def _inverse_quintic_blend(y: float) -> float:
    """Return ``u in [0, 1]`` with ``_quintic_blend(u) == y`` (monotone)."""

    if y <= 0.0:
        return 0.0
    if y >= 1.0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(64):
        mid = 0.5 * (low + high)
        if _quintic_blend(mid) < y:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


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
    # Direct-approach is the accuracy default: the pregrasp standoff becomes a
    # velocity-continuous via inside one blended descent into contact, instead
    # of a full stop before the final approach.  ``direct_approach=False`` keeps
    # the classic per-stage rest-to-rest fallback.
    direct_approach: bool = True
    # Peak descent speed multiplier applied to the blended pregrasp->grasp
    # segment so the tool creeps into contact for accuracy.  The quintic already
    # decelerates to zero exactly at the grasp pose; this caps the peak.
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

    def _retime_direct_pair(
        self,
        pregrasp_path: np.ndarray,
        grasp_path: np.ndarray,
        *,
        contact_speed_scale: float,
    ) -> tuple[TimedJointTrajectory, TimedJointTrajectory]:
        """Retime the pregrasp->grasp descent as ONE blended, sliced profile.

        The two collision-checked joint sub-paths are concatenated into a single
        polyline whose shared standoff vertex becomes an interior *via* rather
        than a stop.  A single rest-to-rest quintic in cumulative chord length
        keeps a non-zero velocity through that via and decelerates smoothly to
        zero exactly at the grasp contact; ``contact_speed_scale`` caps the peak
        so the tool creeps into contact.  The dense samples are then sliced back
        at the via into the pregrasp and grasp segment trajectories so the plan
        keeps its four-stage structure (and its reverse-replay corridor) while
        the flattened trajectory is one continuous velocity profile.

        The geometry is untouched: the samples ride the exact checked polyline
        edges, so collision coverage is identical to the staged fallback.
        """

        pregrasp = np.asarray(pregrasp_path, dtype=float)
        grasp = np.asarray(grasp_path, dtype=float)
        # Continuous polyline through the shared standoff via (dropped duplicate).
        full = np.vstack((pregrasp, grasp[1:]))
        via_index = len(pregrasp) - 1
        edges = np.diff(full, axis=0)
        edge_lengths = np.linalg.norm(edges, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        total = float(cumulative[-1])
        via_arclength = float(cumulative[via_index])
        if (
            total < 1e-9
            or via_arclength <= 1e-9
            or via_arclength >= total - 1e-9
        ):
            # Degenerate corridor (no real standoff separation): fall back to the
            # conservative per-segment rest-to-rest retiming.
            return self._retime(pregrasp), self._retime(grasp)

        active = edge_lengths > 1e-12
        slope = np.zeros_like(edges)
        slope[active] = edges[active] / edge_lengths[active, None]
        max_slope = np.max(np.abs(slope), axis=0)
        # Peak speed of the single quintic over the whole descent is capped so no
        # joint exceeds its (contact-scaled) velocity or its acceleration limit.
        velocity = (
            self.velocity_limits
            * self.time_config.velocity_scale
            * float(contact_speed_scale)
        )
        acceleration = self.acceleration_limits * self.time_config.acceleration_scale
        with np.errstate(divide="ignore", invalid="ignore"):
            duration_velocity = float(np.max(1.875 * total * max_slope / velocity))
            duration_acceleration = float(
                np.max(np.sqrt(5.774 * total * max_slope / acceleration)),
            )
        duration = max(
            self.time_config.min_segment_time_s,
            duration_velocity,
            duration_acceleration,
        )

        via_time = _inverse_quintic_blend(via_arclength / total) * duration
        samples = max(2, int(np.ceil(duration / self.time_config.sample_period_s)) + 1)
        grid = np.linspace(0.0, duration, samples)
        # Guarantee the via is an exact sample so the slice endpoints land on the
        # true standoff joints; drop any grid point that would collide with it.
        keep = np.abs(grid - via_time) > 1e-6
        times = np.sort(np.concatenate((grid[keep], (via_time,))))
        positions = np.stack([
            _interpolate_arclength(
                full,
                cumulative,
                _quintic_blend(sample_time / duration) * total,
            )
            for sample_time in times
        ])

        boundary = int(np.searchsorted(times, via_time))
        pregrasp_positions = positions[: boundary + 1].copy()
        grasp_positions = positions[boundary:].copy()
        # Pin the shared endpoints to the exact checked joints (kill float drift).
        pregrasp_positions[0] = full[0]
        pregrasp_positions[-1] = full[via_index]
        grasp_positions[0] = full[via_index]
        grasp_positions[-1] = full[-1]
        return (
            TimedJointTrajectory(
                positions=pregrasp_positions,
                times_s=times[: boundary + 1].copy(),
            ),
            TimedJointTrajectory(
                positions=grasp_positions,
                times_s=times[boundary:] - via_time,
            ),
        )

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

        use_direct = (
            request.direct_approach
            and GraspStage.PREGRASP in stage_paths
            and GraspStage.GRASP in stage_paths
        )
        trajectories: dict[GraspStage, TimedJointTrajectory] = {}
        if use_direct:
            _, _, pregrasp_path = stage_paths[GraspStage.PREGRASP]
            _, _, grasp_path = stage_paths[GraspStage.GRASP]
            pregrasp_traj, grasp_traj = self._retime_direct_pair(
                pregrasp_path,
                grasp_path,
                contact_speed_scale=request.contact_speed_scale,
            )
            trajectories[GraspStage.PREGRASP] = pregrasp_traj
            trajectories[GraspStage.GRASP] = grasp_traj
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
