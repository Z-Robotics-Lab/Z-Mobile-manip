"""Fail-closed collision checks against a filtered perception point cloud.

The checker deliberately owns no robot-specific geometry.  Capsule endpoints
refer to URDF link frames and a caller-supplied ``frame_provider`` evaluates
those frames for a joint state.  This keeps the same collision model usable
with an in-process URDF chain, ROS TF, or a hardware kinematics service.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from z_manip.kinematics import KinematicChain
from z_manip.planning_control import PlanningControl, checkpoint


_Pair = tuple[str, str]
_Offset = tuple[float, float, float]
FrameProvider = Callable[[np.ndarray], Mapping[str, np.ndarray]]
SelfCollisionChecker = Callable[[np.ndarray], "CollisionResult"]


def _validate_offset(value: object, label: str) -> None:
    offset = np.asarray(value, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError(f"{label} must be a finite three-vector")


def _normalize_pair(pair: Sequence[str]) -> _Pair:
    if len(pair) != 2 or not all(isinstance(name, str) and name for name in pair):
        raise ValueError("a self-collision pair must contain two non-empty names")
    if pair[0] == pair[1]:
        raise ValueError("a capsule cannot be paired with itself")
    return tuple(sorted((pair[0], pair[1])))


@dataclass(frozen=True)
class CapsuleSpec:
    """One capsule with endpoints expressed in named URDF link frames."""

    name: str
    start_frame: str
    end_frame: str
    radius: float
    start_offset: _Offset = (0.0, 0.0, 0.0)
    end_offset: _Offset = (0.0, 0.0, 0.0)
    check_scene: bool = True
    check_target: bool = True
    # Keep this capsule in the lightweight self-collision pass even when a
    # mesh backend owns normal arm self collision.  This is intended for
    # platform fixtures that are not part of the active arm chain.
    supplemental_self_collision: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.start_frame or not self.end_frame:
            raise ValueError("capsule and frame names must be non-empty")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("capsule radius must be finite and positive")
        _validate_offset(self.start_offset, "capsule start_offset")
        _validate_offset(self.end_offset, "capsule end_offset")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CapsuleSpec":
        """Build a capsule from a JSON/YAML-compatible mapping."""

        return cls(
            name=str(data["name"]),
            start_frame=str(data["start_frame"]),
            end_frame=str(data["end_frame"]),
            radius=float(data["radius"]),
            start_offset=tuple(data.get("start_offset", (0.0, 0.0, 0.0))),
            end_offset=tuple(data.get("end_offset", (0.0, 0.0, 0.0))),
            check_scene=bool(data.get("check_scene", True)),
            check_target=bool(data.get("check_target", True)),
            supplemental_self_collision=bool(
                data.get("supplemental_self_collision", False)
            ),
        )


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Capsule pairs checked for self collision.

    ``pairs=()`` disables self checks.  ``pairs=None`` checks every capsule
    pair, after removing ``ignore_pairs``.  An explicit list is preferable for
    articulated robots because adjacent link capsules normally overlap.
    """

    pairs: tuple[_Pair, ...] | None = ()
    ignore_pairs: tuple[_Pair, ...] = ()

    def __post_init__(self) -> None:
        if self.pairs is not None:
            for pair in self.pairs:
                _normalize_pair(pair)
        for pair in self.ignore_pairs:
            _normalize_pair(pair)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SelfCollisionConfig":
        if data is None:
            return cls()
        pairs = data.get("pairs", ())
        parsed_pairs = None if pairs is None else tuple(tuple(pair) for pair in pairs)
        return cls(
            pairs=parsed_pairs,
            ignore_pairs=tuple(tuple(pair) for pair in data.get("ignore_pairs", ())),
        )


@dataclass(frozen=True)
class RobotCollisionModel:
    """Robot geometry loaded independently of scene or planner settings."""

    capsules: tuple[CapsuleSpec, ...]
    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    target_contact_capsules: tuple[str, ...] = ()
    scene_clearance_m: float = 0.02
    point_radius_m: float = 0.005
    scene_noise_tolerance_m: float = 0.0
    scene_noise_min_support_points: int = 1
    # Optional, default-off relaxation for the finger-vs-scene approach check.
    # When enabled the checker fits the horizontal support surface beneath the
    # grasp target and drops scene samples inside a tight in-plane band from the
    # cloud that finger capsules are tested against.  Every other capsule (palm,
    # wrist, arm) and every other collision family keep the full cloud, so only
    # table/floor grazing under a top-down grasp is exempted, not real obstacles.
    finger_support_plane_exclusion: bool = False
    finger_support_plane_band_m: float = 0.006
    finger_support_plane_radius_m: float = 0.18
    finger_support_plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not self.capsules:
            raise ValueError("a collision model needs at least one capsule")
        if (
            not np.isfinite(self.scene_clearance_m)
            or self.scene_clearance_m < 0.0
            or not np.isfinite(self.point_radius_m)
            or self.point_radius_m < 0.0
            or not np.isfinite(self.scene_noise_tolerance_m)
            or self.scene_noise_tolerance_m < 0.0
        ):
            raise ValueError("scene collision margins must be finite and non-negative")
        if self.scene_noise_min_support_points < 1:
            raise ValueError("scene noise support count must be positive")
        if (
            not np.isfinite(self.finger_support_plane_band_m)
            or self.finger_support_plane_band_m < 0.0
            or not np.isfinite(self.finger_support_plane_radius_m)
            or self.finger_support_plane_radius_m <= 0.0
        ):
            raise ValueError(
                "finger support-plane band/radius must be finite, band>=0, radius>0",
            )
        normal = np.asarray(self.finger_support_plane_normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)) or float(
            np.linalg.norm(normal)
        ) <= 1e-9:
            raise ValueError("finger support-plane normal must be a finite nonzero 3-vector")
        names = [capsule.name for capsule in self.capsules]
        if len(names) != len(set(names)):
            raise ValueError("capsule names must be unique")
        known = set(names)
        unknown_contact = set(self.target_contact_capsules) - known
        if unknown_contact:
            raise ValueError(
                "target_contact_capsules references unknown capsules: "
                f"{sorted(unknown_contact)}",
            )
        configured = () if self.self_collision.pairs is None else self.self_collision.pairs
        for pair in (*configured, *self.self_collision.ignore_pairs):
            unknown = set(pair) - known
            if unknown:
                raise ValueError(f"self-collision pair references unknown capsules: {sorted(unknown)}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RobotCollisionModel":
        return cls(
            capsules=tuple(CapsuleSpec.from_mapping(item) for item in data["capsules"]),
            self_collision=SelfCollisionConfig.from_mapping(data.get("self_collision")),
            target_contact_capsules=tuple(data.get("target_contact_capsules", ())),
            scene_clearance_m=float(data.get("scene_clearance_m", 0.02)),
            point_radius_m=float(data.get("point_radius_m", 0.005)),
            scene_noise_tolerance_m=float(data.get("scene_noise_tolerance_m", 0.0)),
            scene_noise_min_support_points=int(
                data.get("scene_noise_min_support_points", 1)
            ),
            finger_support_plane_exclusion=bool(
                data.get("finger_support_plane_exclusion", False)
            ),
            finger_support_plane_band_m=float(
                data.get("finger_support_plane_band_m", 0.006)
            ),
            finger_support_plane_radius_m=float(
                data.get("finger_support_plane_radius_m", 0.18)
            ),
            finger_support_plane_normal=tuple(
                data.get("finger_support_plane_normal", (0.0, 0.0, 1.0))
            ),
        )


@dataclass(frozen=True)
class PointCloudCollisionConfig:
    """Perception uncertainty and continuous joint-segment resolution."""

    clearance: float = 0.02
    point_radius: float = 0.005
    scene_noise_tolerance: float = 0.0
    scene_noise_min_support_points: int = 1
    max_scene_age_s: float = 0.35
    future_tolerance_s: float = 0.05
    min_scene_points: int = 32
    segment_joint_step: float = 0.025
    max_attached_points: int = 512
    support_axial_tolerance_scale: float = 0.10
    support_normal_min_alignment: float = 0.75
    support_normal_max_surface_variation: float = 0.30
    support_normal_neighbor_count: int = 24
    support_region_plane_tolerance_scale: float = 0.30
    support_region_max_tolerance_scale: float = 0.35
    support_region_mad_scale: float = 3.0
    support_region_radius_scale: float = 1.5
    departure_lateral_tolerance_scale: float = 0.25

    def __post_init__(self) -> None:
        finite_nonnegative = (
            self.clearance,
            self.point_radius,
            self.scene_noise_tolerance,
            self.future_tolerance_s,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in finite_nonnegative):
            raise ValueError("collision clearances and time tolerance must be non-negative")
        if not np.isfinite(self.max_scene_age_s) or self.max_scene_age_s <= 0.0:
            raise ValueError("max_scene_age_s must be finite and positive")
        if self.min_scene_points < 1:
            raise ValueError("min_scene_points must be positive")
        if self.scene_noise_min_support_points < 1:
            raise ValueError("scene_noise_min_support_points must be positive")
        if self.max_attached_points < 1:
            raise ValueError("max_attached_points must be positive")
        if not np.isfinite(self.segment_joint_step) or self.segment_joint_step <= 0.0:
            raise ValueError("segment_joint_step must be finite and positive")
        scales = (
            self.support_axial_tolerance_scale,
            self.support_region_plane_tolerance_scale,
            self.support_region_max_tolerance_scale,
            self.support_region_mad_scale,
            self.support_region_radius_scale,
            self.departure_lateral_tolerance_scale,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in scales):
            raise ValueError("support and departure tolerance scales must be non-negative")
        bounded = (
            self.support_normal_min_alignment,
            self.support_normal_max_surface_variation,
        )
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("support normal thresholds must be within [0, 1]")
        if self.support_normal_neighbor_count < 3:
            raise ValueError("support_normal_neighbor_count must be at least three")
        if (
            self.support_region_max_tolerance_scale
            < self.support_region_plane_tolerance_scale
        ):
            raise ValueError("support region maximum tolerance must cover its base tolerance")


@dataclass(frozen=True)
class CollisionResult:
    valid: bool
    reason: str
    kind: str | None = None
    capsules: tuple[str, ...] = ()
    distance: float | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class SegmentCollisionResult:
    valid: bool
    reason: str
    alpha: float | None = None
    sample_index: int | None = None
    state_result: CollisionResult | None = None


@dataclass(frozen=True)
class _WorldCapsule:
    spec: CapsuleSpec
    start: np.ndarray
    end: np.ndarray


def _point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = end - start
    length_squared = float(np.dot(delta, delta))
    if length_squared <= 1e-20:
        return np.linalg.norm(points - start, axis=1)
    projection = ((points - start) @ delta) / length_squared
    projection = np.clip(projection, 0.0, 1.0)
    closest = start + projection[:, None] * delta
    return np.linalg.norm(points - closest, axis=1)


def _segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Exact shortest distance between two finite 3-D line segments."""

    first = first_end - first_start
    second = second_end - second_start
    between = first_start - second_start
    aa = float(np.dot(first, first))
    bb = float(np.dot(first, second))
    cc = float(np.dot(second, second))
    dd = float(np.dot(first, between))
    ee = float(np.dot(second, between))
    epsilon = 1e-14

    if aa <= epsilon and cc <= epsilon:
        return float(np.linalg.norm(first_start - second_start))
    if aa <= epsilon:
        return float(_point_segment_distances(first_start[None, :], second_start, second_end)[0])
    if cc <= epsilon:
        return float(_point_segment_distances(second_start[None, :], first_start, first_end)[0])

    denominator = aa * cc - bb * bb
    s_numerator = 0.0 if denominator <= epsilon else bb * ee - cc * dd
    s_denominator = denominator
    t_numerator = aa * ee - bb * dd
    t_denominator = denominator

    if denominator <= epsilon:
        s_numerator, s_denominator = 0.0, 1.0
        t_numerator, t_denominator = ee, cc
    elif s_numerator < 0.0:
        s_numerator = 0.0
        t_numerator, t_denominator = ee, cc
    elif s_numerator > s_denominator:
        s_numerator = s_denominator
        t_numerator, t_denominator = ee + bb, cc

    if t_numerator < 0.0:
        t_numerator = 0.0
        if -dd < 0.0:
            s_numerator = 0.0
        elif -dd > aa:
            s_numerator, s_denominator = s_denominator, s_denominator
        else:
            s_numerator, s_denominator = -dd, aa
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if -dd + bb < 0.0:
            s_numerator = 0.0
        elif -dd + bb > aa:
            s_numerator, s_denominator = s_denominator, s_denominator
        else:
            s_numerator, s_denominator = -dd + bb, aa

    first_fraction = 0.0 if abs(s_numerator) <= epsilon else s_numerator / s_denominator
    second_fraction = 0.0 if abs(t_numerator) <= epsilon else t_numerator / t_denominator
    separation = between + first_fraction * first - second_fraction * second
    return float(np.linalg.norm(separation))


class PointCloudCollisionChecker:
    """Check joint states and interpolated segments against perceived points."""

    def __init__(
        self,
        *,
        chain: KinematicChain,
        model: RobotCollisionModel,
        frame_provider: FrameProvider,
        config: PointCloudCollisionConfig | None = None,
        now_fn: Callable[[], float] = time.time,
        self_collision_checker: SelfCollisionChecker | None = None,
    ) -> None:
        self.chain = chain
        self.model = model
        self.frame_provider = frame_provider
        self.config = config or PointCloudCollisionConfig()
        self._now_fn = now_fn
        self._self_collision_checker = self_collision_checker
        self._points: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._scene_stamp_s: float | None = None
        self._scene_problem = "no perception cloud has been received"
        # Lazily fitted finger-vs-scene support-plane exclusion (default off).
        self._finger_scene_tree: cKDTree | None = None
        self._finger_scene_points: np.ndarray | None = None
        self._finger_scene_excluded = 0
        self._finger_scene_ready = False
        self._target_points: np.ndarray | None = None
        self._target_tree: cKDTree | None = None
        self._target_allowed_capsules: frozenset[str] = frozenset()
        self._target_clearance = 0.0
        self._attached_target_points_tip: np.ndarray | None = None
        self._attached_target_tree_tip: cKDTree | None = None
        self._attached_target_allow_scene_contact = False
        self._attached_departure_scene_tree: cKDTree | None = None
        self._attached_departure_support_tree: cKDTree | None = None
        self._attached_departure_direction_base: np.ndarray | None = None
        self._attached_departure_origin_centroid_base: np.ndarray | None = None
        self._attached_departure_contact_mode = False

        chain_frames = {chain.base_link, chain.tip_link}
        for joint in chain.joints:
            chain_frames.update((joint.parent, joint.child))
        requested_frames = {
            frame
            for capsule in model.capsules
            for frame in (capsule.start_frame, capsule.end_frame)
        }
        unknown = requested_frames - chain_frames
        if unknown:
            raise ValueError(f"capsules reference frames outside the kinematic chain: {sorted(unknown)}")
        self._self_pairs = self._resolve_self_pairs()

    def _resolve_self_pairs(self) -> tuple[_Pair, ...]:
        names = tuple(capsule.name for capsule in self.model.capsules)
        configured = self.model.self_collision.pairs
        pairs = itertools.combinations(names, 2) if configured is None else configured
        ignored = {_normalize_pair(pair) for pair in self.model.self_collision.ignore_pairs}
        return tuple(
            pair
            for pair in dict.fromkeys(_normalize_pair(pair) for pair in pairs)
            if pair not in ignored
        )

    def _clear_finger_scene_exclusion(self) -> None:
        self._finger_scene_tree = None
        self._finger_scene_points = None
        self._finger_scene_excluded = 0
        self._finger_scene_ready = False

    def _clear_scene(self, reason: str) -> None:
        self._points = None
        self._tree = None
        self._clear_departure_contact()
        self._clear_finger_scene_exclusion()
        self._scene_stamp_s = None
        self._scene_problem = reason

    def _clear_departure_contact(self) -> None:
        self._attached_departure_scene_tree = None
        self._attached_departure_support_tree = None
        self._attached_departure_direction_base = None
        self._attached_departure_origin_centroid_base = None
        self._attached_departure_contact_mode = False

    def update_scene(
        self,
        points: object,
        *,
        stamp_s: float,
        target_mask: object | None = None,
    ) -> int:
        """Replace the base-frame scene cloud and omit target-mask points.

        ``target_mask`` must be aligned with the point array.  It removes the
        intended grasp object so contact with that object does not invalidate
        an otherwise safe approach.  Malformed updates invalidate the previous
        cloud before raising, preventing accidental reuse of stale geometry.
        """

        try:
            cloud = np.asarray(points, dtype=float)
            if cloud.ndim != 2 or cloud.shape[1] != 3:
                raise ValueError("scene points must have shape (N, 3)")
            if not np.isfinite(stamp_s):
                raise ValueError("scene timestamp must be finite")
            if target_mask is None:
                excluded = np.zeros(len(cloud), dtype=bool)
            else:
                excluded = np.asarray(target_mask)
                if excluded.shape != (len(cloud),) or excluded.dtype != np.bool_:
                    raise ValueError("target_mask must be a boolean vector aligned with scene points")
            usable = np.all(np.isfinite(cloud), axis=1) & ~excluded
            filtered = np.ascontiguousarray(cloud[usable], dtype=float)
        except (TypeError, ValueError) as error:
            self._clear_scene(f"invalid perception update: {error}")
            raise

        self._points = filtered
        self._tree = cKDTree(filtered) if len(filtered) else None
        self._clear_departure_contact()
        self._clear_finger_scene_exclusion()
        self._scene_stamp_s = float(stamp_s)
        if len(filtered) < self.config.min_scene_points:
            self._scene_problem = (
                f"perception cloud has {len(filtered)} usable points; "
                f"requires {self.config.min_scene_points}"
            )
        else:
            self._scene_problem = ""
        return len(filtered)

    def update_target(
        self,
        points: object,
        *,
        allowed_contact_capsules: Sequence[str] = (),
        clearance: float = 0.0,
    ) -> int:
        """Install target geometry with an explicit link-contact allowlist.

        The target stays separate from the environment cloud. This permits the
        fingers to contact it without granting the wrist or arm permission to
        pass through it during transit, approach, or lift.
        """
        cloud = np.asarray(points, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError("target points must have shape (N, 3)")
        filtered = np.ascontiguousarray(cloud[np.all(np.isfinite(cloud), axis=1)])
        if len(filtered) < 1:
            raise ValueError("target cloud contains no finite points")
        names = {capsule.name for capsule in self.model.capsules}
        allowed = frozenset(str(name) for name in allowed_contact_capsules)
        unknown = allowed - names
        if unknown:
            raise ValueError(f"target contact allowlist has unknown capsules: {sorted(unknown)}")
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("target clearance must be finite and non-negative")
        self._target_points = filtered
        self._target_tree = cKDTree(filtered)
        self._target_allowed_capsules = allowed
        self._target_clearance = float(clearance)
        self._attached_target_points_tip = None
        self._attached_target_tree_tip = None
        self._attached_target_allow_scene_contact = False
        self._clear_departure_contact()
        self._clear_finger_scene_exclusion()
        return len(filtered)

    def _support_contact_mask(
        self,
        target_points: np.ndarray,
        departure_direction: np.ndarray,
        contact_threshold: float,
    ) -> np.ndarray:
        """Classify scene samples on a locally planar support-facing manifold."""

        assert self._points is not None and self._tree is not None
        target_tree = cKDTree(target_points)
        distances, nearest_indices = target_tree.query(self._points, k=1)
        nearest_target = target_points[np.asarray(nearest_indices, dtype=int)]
        separation = self._points - nearest_target
        axial_separation = separation @ departure_direction
        axial_tolerance = max(
            1e-6,
            self.config.support_axial_tolerance_scale * contact_threshold,
        )
        candidate_mask = (
            (np.asarray(distances) <= contact_threshold)
            & (axial_separation <= axial_tolerance)
        )
        candidates = np.flatnonzero(candidate_mask)
        support = np.zeros(len(self._points), dtype=bool)
        normal_seeds = np.zeros(len(self._points), dtype=bool)
        neighbor_count = min(
            self.config.support_normal_neighbor_count,
            len(self._points),
        )
        if neighbor_count < 3:
            return support
        for index in candidates:
            _, neighbor_indices = self._tree.query(
                self._points[index],
                k=neighbor_count,
            )
            neighbors = self._points[np.asarray(neighbor_indices, dtype=int)]
            centered = neighbors - np.mean(neighbors, axis=0)
            covariance = centered.T @ centered / float(len(neighbors))
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            total_variation = float(np.sum(eigenvalues))
            if total_variation <= 1e-12 or float(eigenvalues[1]) <= 1e-12:
                continue
            surface_variation = float(eigenvalues[0]) / total_variation
            normal_alignment = abs(float(
                eigenvectors[:, 0] @ departure_direction,
            ))
            if (
                surface_variation
                <= self.config.support_normal_max_surface_variation
                and normal_alignment >= self.config.support_normal_min_alignment
            ):
                normal_seeds[index] = True
        seed_indices = np.flatnonzero(normal_seeds)
        if not len(seed_indices):
            return support
        # Each seed independently proves a locally support-facing surface.
        # The robust global plane below is only a guard for growing through
        # edge samples whose own normals are unreliable; it must not discard
        # valid seeds on a warped or quantized support observation.
        support[seed_indices] = True
        projections = self._points @ departure_direction
        seed_projections = projections[seed_indices]
        support_plane = float(np.median(seed_projections))
        median_deviation = float(np.median(np.abs(
            seed_projections - support_plane,
        )))
        base_tolerance = max(
            1e-6,
            self.config.support_region_plane_tolerance_scale
            * contact_threshold,
        )
        maximum_tolerance = max(
            base_tolerance,
            self.config.support_region_max_tolerance_scale
            * contact_threshold,
        )
        plane_tolerance = min(
            max(
                base_tolerance,
                self.config.support_region_mad_scale * median_deviation,
            ),
            maximum_tolerance,
        )
        eligible_indices = np.flatnonzero(
            candidate_mask
            & (np.abs(projections - support_plane) <= plane_tolerance)
        )
        if not len(eligible_indices):
            return support
        eligible_points = self._points[eligible_indices]
        eligible_tree = cKDTree(eligible_points)
        eligible_lookup = {
            int(global_index): local_index
            for local_index, global_index in enumerate(eligible_indices)
        }
        pending = [
            eligible_lookup[int(index)]
            for index in seed_indices
            if int(index) in eligible_lookup
        ]
        visited = np.zeros(len(eligible_indices), dtype=bool)
        region_radius = max(
            1e-6,
            self.config.support_region_radius_scale * contact_threshold,
        )
        while pending:
            local_index = pending.pop()
            if visited[local_index]:
                continue
            visited[local_index] = True
            pending.extend(
                neighbor
                for neighbor in eligible_tree.query_ball_point(
                    eligible_points[local_index],
                    region_radius,
                )
                if not visited[neighbor]
            )
        support[eligible_indices[visited]] = True
        return support

    def update_attached_target(
        self,
        points: object,
        *,
        attachment_joints: object,
        allowed_contact_capsules: Sequence[str] = (),
        clearance: float = 0.0,
        allow_scene_contact: bool = False,
        allow_initial_scene_contact: bool = False,
        departure_direction_base: object | None = None,
    ) -> int:
        """Attach observed target geometry rigidly to the kinematic tip.

        ``points`` are measured in the chain base frame at
        ``attachment_joints``. They are stored in the tip frame and transformed
        for every checked joint state, so lift and carry planning audit the
        moving payload against both the scene and non-contact robot capsules.

        ``allow_initial_scene_contact`` permits only initial contact samples on
        the support-facing target surface opposite ``departure_direction_base``.
        Side contacts remain obstacles, and segment checks require monotonic
        motion away from that support. ``allow_scene_contact`` remains reserved
        for final placement.
        """
        cloud = np.asarray(points, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError("attached target points must have shape (N, 3)")
        filtered = np.ascontiguousarray(cloud[np.all(np.isfinite(cloud), axis=1)])
        if len(filtered) < 1:
            raise ValueError("attached target cloud contains no finite points")
        joints, joint_problem = self._validate_joints(attachment_joints)
        if joint_problem is not None or joints is None:
            raise ValueError(joint_problem.reason if joint_problem else "invalid attachment joints")
        names = {capsule.name for capsule in self.model.capsules}
        allowed = frozenset(str(name) for name in allowed_contact_capsules)
        unknown = allowed - names
        if unknown:
            raise ValueError(
                f"target contact allowlist has unknown capsules: {sorted(unknown)}",
            )
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("target clearance must be finite and non-negative")
        if allow_scene_contact and allow_initial_scene_contact:
            raise ValueError(
                "full and initial attached-target scene contact modes are exclusive",
            )
        departure_direction = None
        if allow_initial_scene_contact:
            if self._points is None or self._tree is None:
                raise ValueError(
                    "initial attached-target contact requires a valid scene cloud",
                )
            try:
                departure_direction = np.asarray(
                    departure_direction_base,
                    dtype=float,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "departure_direction_base must be a finite nonzero three-vector",
                ) from error
            if (
                departure_direction.shape != (3,)
                or not np.all(np.isfinite(departure_direction))
            ):
                raise ValueError(
                    "departure_direction_base must be a finite nonzero three-vector",
                )
            direction_norm = float(np.linalg.norm(departure_direction))
            if direction_norm <= 1e-9:
                raise ValueError(
                    "departure_direction_base must be a finite nonzero three-vector",
                )
            departure_direction = departure_direction / direction_norm
        observed_target = filtered
        if len(filtered) > self.config.max_attached_points:
            indices = np.linspace(
                0,
                len(filtered) - 1,
                self.config.max_attached_points,
                dtype=np.int64,
            )
            filtered = filtered[indices]
        base_t_tip = self.chain.forward(joints)
        tip_t_base = np.linalg.inv(base_t_tip)
        points_tip = filtered @ tip_t_base[:3, :3].T + tip_t_base[:3, 3]
        self._target_points = None
        self._target_tree = None
        self._attached_target_points_tip = np.ascontiguousarray(points_tip)
        self._attached_target_tree_tip = cKDTree(self._attached_target_points_tip)
        self._target_allowed_capsules = allowed
        self._target_clearance = float(clearance)
        self._attached_target_allow_scene_contact = bool(allow_scene_contact)
        self._clear_departure_contact()
        self._attached_departure_contact_mode = bool(allow_initial_scene_contact)
        if allow_initial_scene_contact:
            assert departure_direction is not None
            assert self._points is not None
            contact_threshold = (
                self.config.clearance
                + self.config.point_radius
                + self._target_clearance
            )
            support_mask = self._support_contact_mask(
                observed_target,
                departure_direction,
                contact_threshold,
            )
            departure_scene = np.ascontiguousarray(
                self._points[~support_mask],
                dtype=float,
            )
            support_scene = np.ascontiguousarray(
                self._points[support_mask],
                dtype=float,
            )
            self._attached_departure_scene_tree = (
                cKDTree(departure_scene) if len(departure_scene) else None
            )
            self._attached_departure_support_tree = (
                cKDTree(support_scene) if len(support_scene) else None
            )
            self._attached_departure_direction_base = departure_direction.copy()
            self._attached_departure_origin_centroid_base = np.mean(
                filtered,
                axis=0,
            )
        return len(points_tip)

    def _scene_status(self) -> CollisionResult | None:
        if self._points is None or self._tree is None or self._scene_stamp_s is None:
            return CollisionResult(False, self._scene_problem, kind="perception")
        if len(self._points) < self.config.min_scene_points:
            return CollisionResult(False, self._scene_problem, kind="perception")
        try:
            now = float(self._now_fn())
        except (TypeError, ValueError, RuntimeError) as error:
            return CollisionResult(
                False,
                f"collision clock failed: {type(error).__name__}: {error}",
                kind="perception",
            )
        if not np.isfinite(now):
            return CollisionResult(False, "collision clock is non-finite", kind="perception")
        age = now - self._scene_stamp_s
        if age > self.config.max_scene_age_s:
            return CollisionResult(
                False,
                f"perception cloud is stale by {age:.3f} s",
                kind="perception",
            )
        if age < -self.config.future_tolerance_s:
            return CollisionResult(
                False,
                f"perception cloud timestamp is {-age:.3f} s in the future",
                kind="perception",
            )
        return None

    def _validate_joints(self, joints: object) -> tuple[np.ndarray | None, CollisionResult | None]:
        expected = (self.chain.dof,)
        try:
            values = np.asarray(joints, dtype=float)
        except (TypeError, ValueError):
            return None, CollisionResult(
                False,
                f"joint state must be a finite {expected} vector",
                kind="kinematics",
            )
        if values.shape != expected or not np.all(np.isfinite(values)):
            return None, CollisionResult(
                False,
                f"joint state must be a finite {expected} vector",
                kind="kinematics",
            )
        if np.any(values < self.chain.lower_limits) or np.any(values > self.chain.upper_limits):
            return None, CollisionResult(False, "joint state violates URDF limits", kind="kinematics")
        return values, None

    @staticmethod
    def _valid_transform(transform: object) -> bool:
        try:
            matrix = np.asarray(transform, dtype=float)
        except (TypeError, ValueError):
            return False
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return False
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
            return False
        rotation = matrix[:3, :3]
        return bool(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        )

    def _world_capsules(
        self,
        joints: np.ndarray,
    ) -> tuple[tuple[_WorldCapsule, ...] | None, CollisionResult | None]:
        try:
            frames = self.frame_provider(joints.copy())
        except Exception as error:  # A kinematics service failure must fail closed.
            return None, CollisionResult(
                False,
                f"frame provider failed: {type(error).__name__}: {error}",
                kind="kinematics",
            )
        if not isinstance(frames, Mapping):
            return None, CollisionResult(
                False,
                "frame provider did not return a frame mapping",
                kind="kinematics",
            )
        requested = {
            frame
            for capsule in self.model.capsules
            for frame in (capsule.start_frame, capsule.end_frame)
        }
        missing = requested - set(frames)
        if missing:
            return None, CollisionResult(
                False,
                f"frame provider omitted links: {sorted(missing)}",
                kind="kinematics",
            )
        for frame in requested:
            if not self._valid_transform(frames[frame]):
                return None, CollisionResult(
                    False,
                    f"frame provider returned an invalid transform for {frame!r}",
                    kind="kinematics",
                )

        world_capsules = []
        for capsule in self.model.capsules:
            start_transform = np.asarray(frames[capsule.start_frame], dtype=float)
            end_transform = np.asarray(frames[capsule.end_frame], dtype=float)
            start = start_transform[:3, :3] @ np.asarray(capsule.start_offset) + start_transform[:3, 3]
            end = end_transform[:3, :3] @ np.asarray(capsule.end_offset) + end_transform[:3, 3]
            world_capsules.append(_WorldCapsule(capsule, start, end))
        return tuple(world_capsules), None

    @staticmethod
    def _is_finger_capsule(spec: CapsuleSpec) -> bool:
        """Identify parallel-jaw finger capsules by their conventional prefix."""

        return spec.name.startswith("finger")

    def _ensure_finger_scene_exclusion(self) -> None:
        """Fit the support surface beneath the target and drop its in-plane band.

        The reduced cloud is consulted only for finger-capsule scene checks and
        only when the model explicitly enables it.  Fitting is deterministic:
        the support plane is horizontal (a configured base-frame normal) and its
        offset is anchored to the lowest target extent, i.e. the object's contact
        with the table or floor.  A tight symmetric band about that plane, capped
        to a lateral radius under the target, is removed.  Genuine obstacles above
        the band or outside the radius stay in the finger cloud.
        """

        if self._finger_scene_ready:
            return
        self._finger_scene_ready = True
        self._finger_scene_tree = self._tree
        self._finger_scene_points = self._points
        self._finger_scene_excluded = 0
        if (
            not self.model.finger_support_plane_exclusion
            or self._points is None
            or self._tree is None
            or self._target_points is None
            or len(self._points) == 0
        ):
            return
        normal = np.asarray(self.model.finger_support_plane_normal, dtype=float)
        normal = normal / float(np.linalg.norm(normal))
        target = self._target_points
        centroid = np.mean(target, axis=0)
        band = float(self.model.finger_support_plane_band_m)
        radius = float(self.model.finger_support_plane_radius_m)
        points = self._points
        offsets = points - centroid
        axial = offsets @ normal
        lateral = np.linalg.norm(offsets - axial[:, None] * normal, axis=1)
        near = lateral <= radius
        if not np.any(near):
            return
        # Anchor the plane to the object's support contact: the lowest target
        # extent along the normal is where the object meets the table or floor.
        support_height = float(np.min(target @ normal))
        heights = points @ normal
        inlier = near & (np.abs(heights - support_height) <= band)
        excluded = int(np.count_nonzero(inlier))
        if excluded == 0:
            return
        kept = np.ascontiguousarray(points[~inlier], dtype=float)
        self._finger_scene_points = kept
        self._finger_scene_tree = cKDTree(kept) if len(kept) else None
        self._finger_scene_excluded = excluded

    def _check_scene_capsule(
        self,
        capsule: _WorldCapsule,
        *,
        tree: cKDTree | None = None,
        points: np.ndarray | None = None,
    ) -> CollisionResult | None:
        if tree is None and points is None:
            tree, points = self._tree, self._points
        if tree is None or points is None or len(points) == 0:
            return None
        center = 0.5 * (capsule.start + capsule.end)
        half_length = 0.5 * float(np.linalg.norm(capsule.end - capsule.start))
        threshold = capsule.spec.radius + self.config.clearance + self.config.point_radius
        candidate_indices = tree.query_ball_point(center, half_length + threshold)
        if not candidate_indices:
            return None
        candidates = points[np.asarray(candidate_indices, dtype=int)]
        distances = _point_segment_distances(candidates, capsule.start, capsule.end)
        closest = float(np.min(distances))
        deep_threshold = max(0.0, threshold - self.config.scene_noise_tolerance)
        deep_support = int(np.count_nonzero(distances <= deep_threshold))
        # D435 samples inside the outer boundary-noise band are not hard
        # collisions.  Rejection requires locally supported penetration beyond
        # that band, so a threshold-grazing mini-cluster cannot veto a plan.
        if deep_support >= self.config.scene_noise_min_support_points:
            return CollisionResult(
                False,
                f"capsule {capsule.spec.name!r} intersects perceived scene",
                kind="scene",
                capsules=(capsule.spec.name,),
                distance=closest,
                threshold=threshold,
            )
        return None

    def _check_target_capsule(
        self,
        capsule: _WorldCapsule,
        *,
        base_t_tip: np.ndarray | None = None,
    ) -> CollisionResult | None:
        tree = self._target_tree
        start = capsule.start
        end = capsule.end
        if self._attached_target_tree_tip is not None:
            if base_t_tip is None:
                raise RuntimeError("attached target check requires the tip transform")
            tip_t_base = np.linalg.inv(base_t_tip)
            start = tip_t_base[:3, :3] @ start + tip_t_base[:3, 3]
            end = tip_t_base[:3, :3] @ end + tip_t_base[:3, 3]
            tree = self._attached_target_tree_tip
        if tree is None:
            return None
        if capsule.spec.name in self._target_allowed_capsules:
            return None
        center = 0.5 * (start + end)
        half_length = 0.5 * float(np.linalg.norm(end - start))
        threshold = (
            capsule.spec.radius
            + self.config.point_radius
            + self._target_clearance
        )
        indices = tree.query_ball_point(center, half_length + threshold)
        if not indices:
            return None
        points = (
            self._attached_target_points_tip
            if self._attached_target_tree_tip is not None
            else self._target_points
        )
        assert points is not None
        candidates = points[np.asarray(indices, dtype=int)]
        closest = float(np.min(_point_segment_distances(
            candidates,
            start,
            end,
        )))
        if closest <= threshold:
            return CollisionResult(
                False,
                f"capsule {capsule.spec.name!r} intersects grasp target",
                kind="target",
                capsules=(capsule.spec.name,),
                distance=closest,
                threshold=threshold,
            )
        return None

    def _attached_scene_distance(
        self,
        tree: cKDTree | None,
        points: np.ndarray,
    ) -> float | None:
        if tree is None:
            return None
        distances, _ = tree.query(points, k=1)
        return float(np.min(distances))

    def _departure_support_contact_is_exempt(
        self,
        points: np.ndarray,
        threshold: float,
    ) -> bool:
        if (
            not self._attached_departure_contact_mode
            or self._attached_departure_support_tree is None
            or self._attached_departure_direction_base is None
            or self._attached_departure_origin_centroid_base is None
        ):
            return False
        axial_progress, lateral_progress = self._departure_progress(points)
        direction_tolerance = 1e-6
        lateral_tolerance = max(
            direction_tolerance,
            self.config.departure_lateral_tolerance_scale * threshold,
        )
        if (
            axial_progress < -direction_tolerance
            or lateral_progress > axial_progress + lateral_tolerance
        ):
            return False
        closest_support = self._attached_scene_distance(
            self._attached_departure_support_tree,
            points,
        )
        assert closest_support is not None
        return closest_support <= threshold

    def _departure_progress(self, points: np.ndarray) -> tuple[float, float]:
        assert self._attached_departure_direction_base is not None
        assert self._attached_departure_origin_centroid_base is not None
        displacement = (
            np.mean(points, axis=0)
            - self._attached_departure_origin_centroid_base
        )
        axial = float(displacement @ self._attached_departure_direction_base)
        lateral = float(np.linalg.norm(
            displacement - axial * self._attached_departure_direction_base,
        ))
        return axial, lateral

    def _departure_progress_at_joints(self, joints: np.ndarray) -> float | None:
        if (
            not self._attached_departure_contact_mode
            or self._attached_target_points_tip is None
            or self._attached_departure_direction_base is None
            or self._attached_departure_origin_centroid_base is None
        ):
            return None
        base_t_tip = self.chain.forward(joints)
        points = (
            self._attached_target_points_tip @ base_t_tip[:3, :3].T
            + base_t_tip[:3, 3]
        )
        axial, _ = self._departure_progress(points)
        return axial

    def _check_attached_target_scene(self, base_t_tip: np.ndarray) -> CollisionResult | None:
        if (
            self._attached_target_points_tip is None
            or self._attached_target_allow_scene_contact
        ):
            return None
        points = (
            self._attached_target_points_tip @ base_t_tip[:3, :3].T
            + base_t_tip[:3, 3]
        )
        threshold = (
            self.config.clearance
            + self.config.point_radius
            + self._target_clearance
        )
        if self._attached_departure_contact_mode:
            closest = self._attached_scene_distance(
                self._attached_departure_scene_tree,
                points,
            )
            if (
                closest is None or closest > threshold
            ) and self._departure_support_contact_is_exempt(points, threshold):
                return None
            if closest is None or closest > threshold:
                closest = self._attached_scene_distance(
                    self._attached_departure_support_tree,
                    points,
                )
        else:
            closest = self._attached_scene_distance(self._tree, points)
        if closest is not None and closest <= threshold:
            return CollisionResult(
                False,
                "attached grasp target intersects perceived scene",
                kind="attached_target",
                distance=closest,
                threshold=threshold,
            )
        return None

    def _check_self_collision(
        self,
        capsules: tuple[_WorldCapsule, ...],
        *,
        supplemental_only: bool = False,
    ) -> CollisionResult | None:
        by_name = {capsule.spec.name: capsule for capsule in capsules}
        for first_name, second_name in self._self_pairs:
            first, second = by_name[first_name], by_name[second_name]
            if supplemental_only and not (
                first.spec.supplemental_self_collision
                or second.spec.supplemental_self_collision
            ):
                continue
            distance = _segment_distance(first.start, first.end, second.start, second.end)
            threshold = first.spec.radius + second.spec.radius + self.config.clearance
            if distance <= threshold:
                return CollisionResult(
                    False,
                    f"capsules {first_name!r} and {second_name!r} self-collide",
                    kind="self",
                    capsules=(first_name, second_name),
                    distance=distance,
                    threshold=threshold,
                )
        return None

    def check_state(self, joints: object) -> CollisionResult:
        """Return a diagnostic state result; missing perception is invalid."""

        scene_problem = self._scene_status()
        if scene_problem is not None:
            return scene_problem
        values, joint_problem = self._validate_joints(joints)
        if joint_problem is not None:
            return joint_problem
        assert values is not None
        capsules, frame_problem = self._world_capsules(values)
        if frame_problem is not None:
            return frame_problem
        assert capsules is not None
        base_t_tip = self.chain.forward(values)
        # With no mesh backend, the capsule model owns all configured pairs.
        # With a mesh backend, keep only explicitly supplemental platform
        # fixtures here; the mesh checker continues to own the existing arm
        # pairs.  This preserves the validated deployed arm model while adding
        # Go2W head/Mid-360 obstacles that are outside the active arm chain.
        self_problem = self._check_self_collision(
            capsules,
            supplemental_only=self._self_collision_checker is not None,
        )
        if self_problem is not None:
            return self_problem
        if self._self_collision_checker is not None:
            try:
                self_problem = self._self_collision_checker(values.copy())
            except Exception as error:
                self_problem = CollisionResult(
                    False,
                    "self-collision backend failed closed: "
                    f"{type(error).__name__}: {error}",
                    kind="kinematics",
                )
            if self_problem.valid:
                self_problem = None
            if self_problem is not None:
                return self_problem
        attached_scene_collision = self._check_attached_target_scene(base_t_tip)
        if attached_scene_collision is not None:
            return attached_scene_collision
        finger_exclusion = (
            self.model.finger_support_plane_exclusion
            and self._target_points is not None
        )
        if finger_exclusion:
            self._ensure_finger_scene_exclusion()
        for capsule in capsules:
            if capsule.spec.check_scene:
                if finger_exclusion and self._is_finger_capsule(capsule.spec):
                    scene_collision = self._check_scene_capsule(
                        capsule,
                        tree=self._finger_scene_tree,
                        points=self._finger_scene_points,
                    )
                else:
                    scene_collision = self._check_scene_capsule(capsule)
                if scene_collision is not None:
                    return scene_collision
            if capsule.spec.check_target:
                target_collision = self._check_target_capsule(
                    capsule,
                    base_t_tip=base_t_tip,
                )
                if target_collision is not None:
                    return target_collision
        return CollisionResult(True, "collision-free")

    def is_state_valid(self, joints: object) -> bool:
        """Planner-compatible state-validity callback."""

        return self.check_state(joints).valid

    def check_segment(
        self,
        start_joints: object,
        end_joints: object,
        *,
        max_joint_step: float | None = None,
        control: PlanningControl | None = None,
    ) -> SegmentCollisionResult:
        """Audit an entire linearly interpolated joint-space segment."""

        checkpoint(control, "point-cloud segment collision checking")
        start, start_problem = self._validate_joints(start_joints)
        if start_problem is not None:
            return SegmentCollisionResult(False, start_problem.reason, 0.0, 0, start_problem)
        end, end_problem = self._validate_joints(end_joints)
        if end_problem is not None:
            return SegmentCollisionResult(False, end_problem.reason, 1.0, None, end_problem)
        assert start is not None and end is not None
        try:
            step = self.config.segment_joint_step if max_joint_step is None else float(max_joint_step)
        except (TypeError, ValueError):
            step = float("nan")
        if not np.isfinite(step) or step <= 0.0:
            result = CollisionResult(False, "max_joint_step must be finite and positive", "kinematics")
            return SegmentCollisionResult(False, result.reason, state_result=result)
        sample_count = max(1, int(np.ceil(np.linalg.norm(end - start) / step)))
        previous_departure_progress = None
        for index, alpha in enumerate(np.linspace(0.0, 1.0, sample_count + 1)):
            checkpoint(control, "point-cloud collision interpolation")
            sample = start + alpha * (end - start)
            departure_progress = self._departure_progress_at_joints(sample)
            if departure_progress is not None and (
                departure_progress < -1e-6
                or (
                    previous_departure_progress is not None
                    and departure_progress < previous_departure_progress - 1e-6
                )
            ):
                result = CollisionResult(
                    False,
                    "attached target reverses the initial departure direction",
                    kind="attached_target",
                )
                return SegmentCollisionResult(
                    False,
                    result.reason,
                    float(alpha),
                    index,
                    result,
                )
            previous_departure_progress = departure_progress
            result = self.check_state(sample)
            if not result.valid:
                return SegmentCollisionResult(False, result.reason, float(alpha), index, result)
        checkpoint(control, "point-cloud segment collision checking")
        return SegmentCollisionResult(True, "collision-free segment")

    def is_segment_valid(
        self,
        start_joints: object,
        end_joints: object,
        *,
        max_joint_step: float | None = None,
        control: PlanningControl | None = None,
    ) -> bool:
        return self.check_segment(
            start_joints,
            end_joints,
            max_joint_step=max_joint_step,
            control=control,
        ).valid
