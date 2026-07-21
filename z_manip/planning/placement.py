"""Plan placement from synchronized RGB-D geometry, never scene ground truth.

The planner deliberately stops at the perception/motion boundary.  It fits and
audits support geometry here, then asks an injected evaluator to perform the
robot-specific IK, continuous collision checking, and motion planning.  Scene
descriptions, semantic object classes, and simulator poses are not inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from z_manip.models.planner import PlanningError
from z_manip.planning_control import (
    PlanningAborted,
    PlanningControl,
    checkpoint,
)


@dataclass(frozen=True)
class NormalizedPlacementRegion:
    """VLM-proposed image region and regions that must not support the object."""

    xyxy: tuple[float, float, float, float]
    avoid_xyxy: tuple[tuple[float, float, float, float], ...] = ()

    def __post_init__(self) -> None:
        _validate_normalized_box(self.xyxy, "placement region")
        for box in self.avoid_xyxy:
            _validate_normalized_box(box, "avoid region")


@dataclass(frozen=True)
class PlacementConstraints:
    """Typed geometric constraints produced from the VLM placement request."""

    min_clearance_m: float = 0.025
    max_surface_tilt_rad: float | None = None
    preferred_yaw_rad: float | None = None
    yaw_tolerance_rad: float = np.pi
    min_support_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_clearance_m) or self.min_clearance_m < 0.0:
            raise ValueError("placement clearance must be finite and non-negative")
        if self.max_surface_tilt_rad is not None and (
            not np.isfinite(self.max_surface_tilt_rad)
            or not 0.0 <= self.max_surface_tilt_rad < np.pi / 2.0
        ):
            raise ValueError("surface tilt constraint must be in [0, pi/2)")
        if self.preferred_yaw_rad is not None and not np.isfinite(self.preferred_yaw_rad):
            raise ValueError("preferred yaw must be finite")
        if not np.isfinite(self.yaw_tolerance_rad) or not 0.0 <= self.yaw_tolerance_rad <= np.pi:
            raise ValueError("yaw tolerance must be in [0, pi]")
        if not 0.0 < self.min_support_fraction <= 1.0:
            raise ValueError("support fraction must be in (0, 1]")


@dataclass(frozen=True, eq=False)
class ObservedPlacementInput:
    """Synchronized perception and carried-object geometry for placement.

    ``organized_points`` is an ``(H, W, 3)`` cloud aligned to the image used by
    the VLM. ``scene_points`` is the collision cloud in the same metric frame.
    The carried target must already be excluded from that scene cloud using its
    persistent observed mask. ``tool_from_object`` and ``object_extent_m`` are
    estimates retained from the observed grasp, not simulator state.
    """

    organized_points: object
    scene_points: object
    region: NormalizedPlacementRegion
    constraints: PlacementConstraints
    gravity: object
    object_extent_m: object
    tool_from_object: object
    organized_frame: str
    scene_frame: str
    organized_stamp_s: float
    scene_stamp_s: float


@dataclass(frozen=True)
class ObservedPlacementConfig:
    """Robot- and sensor-independent placement search parameters."""

    max_sync_skew_s: float = 0.04
    ransac_iterations: int = 320
    ransac_distance_m: float = 0.008
    max_plane_rms_m: float = 0.006
    min_plane_points: int = 80
    min_plane_inlier_ratio: float = 0.45
    max_surface_tilt_rad: float = np.deg2rad(35.0)
    max_ransac_points: int = 6000
    sample_spacing_m: float = 0.035
    support_neighbor_radius_m: float = 0.035
    boundary_margin_m: float = 0.008
    footprint_samples_per_axis: int = 5
    plane_exclusion_m: float = 0.012
    obstacle_height_margin_m: float = 0.025
    tool_clearance_radius_m: float = 0.045
    preplace_distance_m: float = 0.12
    retreat_distance_m: float = 0.14
    yaw_samples: int = 8
    max_geometric_candidates: int = 96
    seed: int = 7
    support_score_weight: float = 0.3
    clearance_score_weight: float = 0.15
    centrality_score_weight: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.max_sync_skew_s,
            self.ransac_distance_m,
            self.max_plane_rms_m,
            self.sample_spacing_m,
            self.support_neighbor_radius_m,
            self.plane_exclusion_m,
            self.tool_clearance_radius_m,
            self.preplace_distance_m,
            self.retreat_distance_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("placement metric thresholds must be finite and positive")
        if self.boundary_margin_m < 0.0 or self.obstacle_height_margin_m < 0.0:
            raise ValueError("placement margins cannot be negative")
        if self.ransac_iterations < 1 or self.min_plane_points < 3:
            raise ValueError("plane fitting needs iterations and at least three points")
        if self.max_ransac_points < self.min_plane_points:
            raise ValueError("max_ransac_points cannot be below min_plane_points")
        if not 0.0 < self.min_plane_inlier_ratio <= 1.0:
            raise ValueError("plane inlier ratio must be in (0, 1]")
        if not 0.0 <= self.max_surface_tilt_rad < np.pi / 2.0:
            raise ValueError("maximum surface tilt must be in [0, pi/2)")
        if self.footprint_samples_per_axis < 2 or self.yaw_samples < 1:
            raise ValueError("placement sampling counts are too small")
        if self.max_geometric_candidates < 1:
            raise ValueError("at least one geometric candidate must be evaluated")
        if min(
            self.support_score_weight,
            self.clearance_score_weight,
            self.centrality_score_weight,
        ) < 0.0:
            raise ValueError("placement score weights cannot be negative")


@dataclass(frozen=True, eq=False)
class SupportPlane:
    origin: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    inlier_count: int
    inlier_ratio: float
    rms_error_m: float


@dataclass(frozen=True, eq=False)
class PlacementCandidate:
    """One geometrically supported 6-DoF tool motion family."""

    support_position: np.ndarray
    surface_normal: np.ndarray
    yaw_rad: float
    object_pose: np.ndarray
    preplace_pose: np.ndarray
    place_pose: np.ndarray
    retreat_pose: np.ndarray
    support_fraction: float
    obstacle_clearance_m: float
    geometric_score: float


@dataclass(frozen=True, eq=False)
class PlacementMotionEvaluation:
    """Recommended return type for the injected robot-specific evaluator."""

    score: float
    motion: object = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.score):
            raise ValueError("placement motion score must be finite")


@dataclass(frozen=True, eq=False)
class PlannedPlacement:
    plane: SupportPlane
    candidate: PlacementCandidate
    evaluation: object
    score: float
    geometric_candidates: int
    rejected_by_motion: int
    failures: tuple[str, ...] = field(default_factory=tuple)


def _validate_normalized_box(box: object, label: str) -> None:
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must contain four finite normalized coordinates")
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{label} must satisfy 0 <= x1 < x2 <= 1 and y likewise")


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        raise PlanningError(f"{label} is not a usable direction")
    return vector / norm


def _evaluation_score(value: object) -> float:
    if isinstance(value, Mapping):
        score = value.get("score")
    else:
        score = getattr(value, "score", None)
    if score is None or not np.isfinite(float(score)):
        raise PlanningError("placement evaluator returned no finite score")
    return float(score)


class ObservedPlacementPlanner:
    """Fit support geometry and rank fully motion-checked placement poses."""

    def __init__(self, config: ObservedPlacementConfig | None = None) -> None:
        self.config = config or ObservedPlacementConfig()

    def _validate(self, observation: ObservedPlacementInput) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        organized = np.asarray(observation.organized_points, dtype=float)
        scene = np.asarray(observation.scene_points, dtype=float)
        extent = np.asarray(observation.object_extent_m, dtype=float)
        tool_from_object = np.asarray(observation.tool_from_object, dtype=float)
        gravity = np.asarray(observation.gravity, dtype=float)
        if organized.ndim != 3 or organized.shape[2] != 3:
            raise PlanningError("organized RGB-D points must have shape (H, W, 3)")
        if scene.ndim != 2 or scene.shape[1] != 3:
            raise PlanningError("scene cloud must have shape (N, 3)")
        scene = scene[np.all(np.isfinite(scene), axis=1)]
        if len(scene) < self.config.min_plane_points:
            raise PlanningError("scene cloud has too few finite points")
        if extent.shape != (3,) or not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
            raise PlanningError("observed object extent must be a positive finite 3-vector")
        if tool_from_object.shape != (4, 4) or not np.all(np.isfinite(tool_from_object)):
            raise PlanningError("tool_from_object must be a finite SE(3) matrix")
        if not np.allclose(tool_from_object[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
            raise PlanningError("tool_from_object has an invalid homogeneous row")
        rotation = tool_from_object[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
            raise PlanningError("tool_from_object rotation is not right-handed orthonormal")
        if not observation.organized_frame or observation.organized_frame != observation.scene_frame:
            raise PlanningError("organized and scene clouds must share one non-empty frame")
        stamps = np.asarray((observation.organized_stamp_s, observation.scene_stamp_s), dtype=float)
        if not np.all(np.isfinite(stamps)) or abs(stamps[0] - stamps[1]) > self.config.max_sync_skew_s:
            raise PlanningError("RGB-D and scene clouds are not synchronized")
        return organized, scene, extent, tool_from_object, _unit(gravity, "gravity")

    @staticmethod
    def _region_mask(shape: tuple[int, int], region: NormalizedPlacementRegion) -> np.ndarray:
        height, width = shape
        x = (np.arange(width, dtype=float) + 0.5) / width
        y = (np.arange(height, dtype=float) + 0.5) / height
        xx, yy = np.meshgrid(x, y)
        x1, y1, x2, y2 = region.xyxy
        mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
        for ax1, ay1, ax2, ay2 in region.avoid_xyxy:
            mask &= ~((xx >= ax1) & (xx <= ax2) & (yy >= ay1) & (yy <= ay2))
        return mask

    def _fit_plane(
        self,
        organized: np.ndarray,
        region: NormalizedPlacementRegion,
        gravity: np.ndarray,
        max_tilt_rad: float,
        control: PlanningControl | None = None,
    ) -> tuple[SupportPlane, np.ndarray]:
        mask = self._region_mask(organized.shape[:2], region)
        points = organized[mask]
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < self.config.min_plane_points:
            raise PlanningError("VLM placement region contains too few valid depth points")
        if len(points) > self.config.max_ransac_points:
            indices = np.linspace(0, len(points) - 1, self.config.max_ransac_points, dtype=int)
            fit_points = points[indices]
        else:
            fit_points = points
        up = -gravity
        min_up_alignment = float(np.cos(max_tilt_rad))
        rng = np.random.default_rng(self.config.seed)
        best_count = -1
        best_normal: np.ndarray | None = None
        best_origin: np.ndarray | None = None
        for _ in range(self.config.ransac_iterations):
            checkpoint(control, "placement support-plane RANSAC")
            sample = fit_points[rng.choice(len(fit_points), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal /= norm
            if float(np.dot(normal, up)) < 0.0:
                normal = -normal
            if float(np.dot(normal, up)) < min_up_alignment:
                continue
            distances = np.abs((fit_points - sample[0]) @ normal)
            count = int(np.count_nonzero(distances <= self.config.ransac_distance_m))
            if count > best_count:
                best_count = count
                best_normal = normal
                best_origin = sample[0]
        if best_normal is None or best_origin is None:
            raise PlanningError("no gravity-consistent support plane was observed")
        initial_distances = np.abs((points - best_origin) @ best_normal)
        inliers = points[initial_distances <= self.config.ransac_distance_m]
        ratio = len(inliers) / len(points)
        if len(inliers) < self.config.min_plane_points or ratio < self.config.min_plane_inlier_ratio:
            raise PlanningError("observed support plane has insufficient RANSAC consensus")
        origin = np.mean(inliers, axis=0)
        covariance = (inliers - origin).T @ (inliers - origin) / len(inliers)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        if float(np.dot(normal, up)) < 0.0:
            normal = -normal
        if float(np.dot(normal, up)) < min_up_alignment:
            raise PlanningError("refined support plane violates the gravity tilt constraint")
        distances = np.abs((points - origin) @ normal)
        inliers = points[distances <= self.config.ransac_distance_m]
        ratio = len(inliers) / len(points)
        rms = float(np.sqrt(np.mean(((inliers - origin) @ normal) ** 2)))
        if len(inliers) < self.config.min_plane_points or ratio < self.config.min_plane_inlier_ratio:
            raise PlanningError("refined support plane has insufficient support")
        if not np.isfinite(rms) or rms > self.config.max_plane_rms_m:
            raise PlanningError("observed support plane is too rough for placement")

        canonical = np.eye(3)[int(np.argmin(np.abs(normal)))]
        tangent_u = _unit(np.cross(canonical, normal), "support tangent")
        tangent_v = _unit(np.cross(normal, tangent_u), "support tangent")
        plane = SupportPlane(
            origin=origin,
            normal=normal,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            inlier_count=len(inliers),
            inlier_ratio=ratio,
            rms_error_m=rms,
        )
        return plane, inliers

    def _yaw_values(self, constraints: PlacementConstraints) -> np.ndarray:
        if constraints.preferred_yaw_rad is None:
            return np.linspace(0.0, 2.0 * np.pi, self.config.yaw_samples, endpoint=False)
        if constraints.yaw_tolerance_rad <= 1e-9 or self.config.yaw_samples == 1:
            return np.asarray([constraints.preferred_yaw_rad])
        return constraints.preferred_yaw_rad + np.linspace(
            -constraints.yaw_tolerance_rad,
            constraints.yaw_tolerance_rad,
            self.config.yaw_samples,
        )

    def _candidate_geometry(
        self,
        support_position: np.ndarray,
        yaw: float,
        plane: SupportPlane,
        extent: np.ndarray,
        tool_from_object: np.ndarray,
        support_tree: cKDTree,
        scene_tree: cKDTree,
        scene: np.ndarray,
        constraints: PlacementConstraints,
        region_radius: float,
    ) -> PlacementCandidate | None:
        cosine, sine = float(np.cos(yaw)), float(np.sin(yaw))
        axis_x = cosine * plane.tangent_u + sine * plane.tangent_v
        axis_y = -sine * plane.tangent_u + cosine * plane.tangent_v
        half_x = 0.5 * extent[0]
        half_y = 0.5 * extent[1]
        boundary_x = half_x + self.config.boundary_margin_m
        boundary_y = half_y + self.config.boundary_margin_m
        grid_x = np.linspace(-boundary_x, boundary_x, self.config.footprint_samples_per_axis)
        grid_y = np.linspace(-boundary_y, boundary_y, self.config.footprint_samples_per_axis)
        footprint = np.asarray([
            support_position + dx * axis_x + dy * axis_y
            for dx in grid_x for dy in grid_y
        ])
        support_uv = np.column_stack((
            (footprint - plane.origin) @ plane.tangent_u,
            (footprint - plane.origin) @ plane.tangent_v,
        ))
        support_distances, _ = support_tree.query(support_uv, k=1)
        support_fraction = float(np.mean(
            support_distances <= self.config.support_neighbor_radius_m,
        ))
        if support_fraction + 1e-12 < constraints.min_support_fraction:
            return None

        clearance = max(constraints.min_clearance_m, 0.0)
        object_height = extent[2]
        query_radius = float(np.hypot(
            np.hypot(half_x + clearance, half_y + clearance),
            object_height + self.config.obstacle_height_margin_m,
        ))
        nearby_indices = scene_tree.query_ball_point(support_position, query_radius)
        minimum_clearance = np.inf
        if nearby_indices:
            delta = scene[np.asarray(nearby_indices, dtype=int)] - support_position
            height = delta @ plane.normal
            above = (
                (height > self.config.plane_exclusion_m)
                & (height < object_height + self.config.obstacle_height_margin_m)
            )
            if np.any(above):
                delta = delta[above]
                local_x = np.abs(delta @ axis_x)
                local_y = np.abs(delta @ axis_y)
                dx = np.maximum(local_x - half_x, 0.0)
                dy = np.maximum(local_y - half_y, 0.0)
                gaps = np.hypot(dx, dy)
                minimum_clearance = float(np.min(gaps))
                if minimum_clearance + 1e-12 < clearance:
                    return None

        object_pose = np.eye(4)
        object_pose[:3, :3] = np.column_stack((axis_x, axis_y, plane.normal))
        object_pose[:3, 3] = support_position + 0.5 * extent[2] * plane.normal
        place_pose = object_pose @ np.linalg.inv(tool_from_object)

        tool_neighbors = scene_tree.query_ball_point(
            place_pose[:3, 3], self.config.tool_clearance_radius_m,
        )
        if tool_neighbors:
            tool_points = scene[np.asarray(tool_neighbors, dtype=int)]
            plane_height = np.abs((tool_points - plane.origin) @ plane.normal)
            if np.any(plane_height > self.config.plane_exclusion_m):
                return None

        preplace_pose = place_pose.copy()
        preplace_pose[:3, 3] += self.config.preplace_distance_m * plane.normal
        retreat_pose = place_pose.copy()
        retreat_pose[:3, 3] += self.config.retreat_distance_m * plane.normal
        center_distance = float(np.linalg.norm(support_position - plane.origin))
        normalized_centrality = max(0.0, 1.0 - center_distance / max(region_radius, 1e-6))
        scored_clearance = (
            clearance + self.config.sample_spacing_m
            if not np.isfinite(minimum_clearance)
            else min(minimum_clearance, clearance + self.config.sample_spacing_m)
        )
        geometric_score = (
            self.config.support_score_weight * support_fraction
            + self.config.clearance_score_weight
            * scored_clearance / max(clearance + self.config.sample_spacing_m, 1e-6)
            + self.config.centrality_score_weight * normalized_centrality
        )
        return PlacementCandidate(
            support_position=support_position.copy(),
            surface_normal=plane.normal.copy(),
            yaw_rad=float(yaw),
            object_pose=object_pose,
            preplace_pose=preplace_pose,
            place_pose=place_pose,
            retreat_pose=retreat_pose,
            support_fraction=support_fraction,
            obstacle_clearance_m=minimum_clearance,
            geometric_score=geometric_score,
        )

    def plan(
        self,
        observation: ObservedPlacementInput,
        *,
        current_joints: object,
        evaluate: Callable[[PlacementCandidate, np.ndarray], object],
        control: PlanningControl | None = None,
    ) -> PlannedPlacement:
        """Return the highest-ranked geometry and robot-motion feasible placement."""

        checkpoint(control, "observed placement setup")
        organized, scene, extent, tool_from_object, gravity = self._validate(observation)
        current = np.asarray(current_joints, dtype=float)
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise PlanningError("current joints must be a finite vector")
        configured_tilt = self.config.max_surface_tilt_rad
        requested_tilt = observation.constraints.max_surface_tilt_rad
        max_tilt = configured_tilt if requested_tilt is None else min(configured_tilt, requested_tilt)
        plane, support_points = self._fit_plane(
            organized, observation.region, gravity, max_tilt, control,
        )

        support_uv = np.column_stack((
            (support_points - plane.origin) @ plane.tangent_u,
            (support_points - plane.origin) @ plane.tangent_v,
        ))
        support_tree = cKDTree(support_uv)
        scene_tree = cKDTree(scene)
        lower = np.min(support_uv, axis=0)
        upper = np.max(support_uv, axis=0)
        u_values = np.arange(lower[0], upper[0] + 0.5 * self.config.sample_spacing_m,
                             self.config.sample_spacing_m)
        v_values = np.arange(lower[1], upper[1] + 0.5 * self.config.sample_spacing_m,
                             self.config.sample_spacing_m)
        samples = np.asarray([(u, v) for u in u_values for v in v_values], dtype=float)
        if len(samples) == 0:
            raise PlanningError("observed support plane has no sampleable area")
        samples = samples[np.argsort(np.linalg.norm(samples, axis=1), kind="stable")]
        region_radius = float(np.max(np.linalg.norm(support_uv, axis=1)))

        candidates: list[PlacementCandidate] = []
        for uv in samples:
            checkpoint(control, "placement geometric candidate generation")
            support_position = plane.origin + uv[0] * plane.tangent_u + uv[1] * plane.tangent_v
            for yaw in self._yaw_values(observation.constraints):
                candidate = self._candidate_geometry(
                    support_position,
                    float(yaw),
                    plane,
                    extent,
                    tool_from_object,
                    support_tree,
                    scene_tree,
                    scene,
                    observation.constraints,
                    region_radius,
                )
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            raise PlanningError(
                "no placement pose has full boundary support and observed obstacle clearance",
            )
        candidates.sort(key=lambda candidate: candidate.geometric_score, reverse=True)
        candidates = candidates[: self.config.max_geometric_candidates]

        feasible: list[tuple[float, PlacementCandidate, object]] = []
        failures: list[str] = []
        for index, candidate in enumerate(candidates):
            checkpoint(control, "placement motion candidate evaluation")
            try:
                evaluation = evaluate(candidate, current.copy())
                score = _evaluation_score(evaluation) + candidate.geometric_score
            except PlanningAborted:
                raise
            except (PlanningError, ValueError) as error:
                failures.append(f"candidate {index}: {type(error).__name__}: {error}")
                continue
            feasible.append((score, candidate, evaluation))
        checkpoint(control, "placement candidate selection")
        if not feasible:
            summary = "; ".join(failures[-8:])
            raise PlanningError(
                f"no observed placement candidate survived IK/collision/motion evaluation: {summary}",
            )
        score, candidate, evaluation = max(feasible, key=lambda item: item[0])
        return PlannedPlacement(
            plane=plane,
            candidate=candidate,
            evaluation=evaluation,
            score=score,
            geometric_candidates=len(candidates),
            rejected_by_motion=len(candidates) - len(feasible),
            failures=tuple(failures),
        )


__all__ = [
    "NormalizedPlacementRegion",
    "ObservedPlacementConfig",
    "ObservedPlacementInput",
    "ObservedPlacementPlanner",
    "PlacementCandidate",
    "PlacementConstraints",
    "PlacementMotionEvaluation",
    "PlannedPlacement",
    "SupportPlane",
]
