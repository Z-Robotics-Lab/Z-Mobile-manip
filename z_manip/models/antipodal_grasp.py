"""Object-agnostic, multi-direction 6-DoF antipodal grasp generation.

The generator estimates outward surface normals from the observed object cloud,
finds opposing contact pairs within the configured gripper aperture, and samples
approach directions around each contact-pair closing axis. It therefore produces
side, oblique, and vertical grasps instead of assuming a tabletop top grasp.

This CPU backend is the deterministic fallback and regression anchor. A learned
grasp model may replace candidate proposal, but all candidates still pass the
same width, collision, IK, and motion-planning filters downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .grasp_source import GraspCandidates, GraspContext, GraspGenerationError
from .grasp_ordering import directionally_diverse_indices


def _normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _estimate_outward_normals(points: np.ndarray, neighbours: int) -> np.ndarray:
    tree = cKDTree(points)
    k = min(max(6, neighbours), len(points))
    _, indices = tree.query(points, k=k)
    normals = np.empty_like(points)
    centroid = np.median(points, axis=0)
    for index, nearby in enumerate(indices):
        local = points[np.atleast_1d(nearby)]
        centred = local - local.mean(axis=0)
        covariance = centred.T @ centred / max(1, len(local) - 1)
        _, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        if np.dot(normal, points[index] - centroid) < 0.0:
            normal = -normal
        normals[index] = normal
    return normals


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, reference))) > 0.88:
        reference = np.array([0.0, 1.0, 0.0])
    first = _normalize(np.cross(axis, reference))
    if first is None:
        raise GraspGenerationError("could not construct approach basis")
    second = _normalize(np.cross(axis, first))
    if second is None:
        raise GraspGenerationError("could not construct approach basis")
    return first, second


def _preferred_approach(affordance: object) -> Optional[np.ndarray]:
    if affordance is None:
        return None
    value = None
    if isinstance(affordance, Mapping):
        value = affordance.get("preferred_approach")
    else:
        value = getattr(affordance, "preferred_approach", None)
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise GraspGenerationError("preferred_approach must be a 3-vector")
    return _normalize(vector)


@dataclass(frozen=True)
class _Candidate:
    pose: np.ndarray
    score: float
    width: float


class AntipodalGraspSource:
    """Generate diverse antipodal candidates from an object point cloud."""

    def __init__(
        self,
        *,
        min_aperture_m: float = 0.012,
        max_aperture_m: float = 0.085,
        contact_angle_deg: float = 42.0,
        normal_neighbours: int = 18,
        approach_samples: int = 8,
        max_surface_points: int = 420,
        max_contact_pairs: int = 96,
        max_candidates: int = 64,
    ) -> None:
        self.min_aperture_m = float(min_aperture_m)
        self.max_aperture_m = float(max_aperture_m)
        self.contact_cosine = math.cos(math.radians(contact_angle_deg))
        self.normal_neighbours = int(normal_neighbours)
        self.approach_samples = max(6, int(approach_samples))
        self.max_surface_points = max(32, int(max_surface_points))
        self.max_contact_pairs = max(1, int(max_contact_pairs))
        self.max_candidates = max(1, int(max_candidates))

    def generate(self, context: GraspContext) -> GraspCandidates:
        if context.object_points is None:
            raise GraspGenerationError("antipodal generation requires an object cloud")
        points = np.asarray(context.object_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise GraspGenerationError(f"object cloud must be Nx3, got {points.shape}")
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 24:
            raise GraspGenerationError(f"object cloud has only {len(points)} valid points")
        if len(points) > self.max_surface_points:
            take = np.linspace(0, len(points) - 1, self.max_surface_points).astype(int)
            points = points[take]

        centred = points - np.median(points, axis=0)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        projected = centred @ axes.T
        oriented_extent = np.ptp(projected, axis=0)
        if float(np.min(oriented_extent)) > self.max_aperture_m * 1.05:
            raise GraspGenerationError(
                "object has no OBB dimension within gripper aperture; "
                f"extent={oriented_extent.round(3).tolist()}"
            )

        context.progress_cb("estimate_normals", 0.15)
        normals = _estimate_outward_normals(points, self.normal_neighbours)
        preferred = _preferred_approach(context.affordance)
        robust_lower = np.quantile(projected, 0.10, axis=0)
        robust_upper = np.quantile(projected, 0.90, axis=0)
        robust_half_extent = np.maximum(
            0.5 * (robust_upper - robust_lower),
            0.5 * self.min_aperture_m,
        )
        pairs = self._contact_pairs(
            points,
            normals,
            object_center=np.median(points, axis=0),
            object_axes=axes,
            robust_half_extent=robust_half_extent,
        )
        if not pairs:
            # D435 surface normals can become locally one-sided on small,
            # textureless cuboids.  The object geometry is still usable: its
            # robust OBB gives a centred closing axis and an aperture-bounded
            # width.  Keep this strictly as a fallback so measured antipodal
            # contacts retain priority whenever they exist.
            candidates = self._obb_fallback_candidates(
                object_center=np.median(points, axis=0),
                object_axes=axes,
                robust_lower=robust_lower,
                robust_upper=robust_upper,
                preferred=preferred,
            )
            if not candidates:
                extent = np.ptp(points, axis=0)
                raise GraspGenerationError(
                    "no antipodal or aperture-bounded OBB grasp; "
                    f"observed extent={extent.round(3).tolist()}"
                )
            return self._as_grasp_candidates(
                candidates,
                frame=context.source_frame,
                num_raw=len(candidates),
                centroid=np.median(points, axis=0),
            )

        context.progress_cb("sample_approaches", 0.45)
        candidates: list[_Candidate] = []
        for pair_index, (i, j, pair_score, width) in enumerate(pairs):
            closing = _normalize(points[j] - points[i])
            if closing is None:
                continue
            first, second = _orthogonal_basis(closing)
            midpoint = 0.5 * (points[i] + points[j])
            for sample in range(self.approach_samples):
                angle = 2.0 * math.pi * sample / self.approach_samples
                approach = _normalize(math.cos(angle) * first + math.sin(angle) * second)
                if approach is None:
                    continue
                binormal = _normalize(np.cross(approach, closing))
                if binormal is None:
                    continue
                # Re-orthogonalize to limit accumulated floating-point error.
                approach = _normalize(np.cross(closing, binormal))
                rotation = np.column_stack((closing, binormal, approach))
                pose = np.eye(4)
                pose[:3, :3] = rotation
                pose[:3, 3] = midpoint
                semantic_score = 0.0
                if preferred is not None:
                    semantic_score = 0.35 * max(0.0, float(np.dot(approach, preferred)))
                diversity = 0.02 * math.cos(angle)
                candidates.append(_Candidate(
                    pose=pose,
                    score=pair_score + semantic_score + diversity,
                    width=width,
                ))
            if pair_index + 1 >= self.max_contact_pairs:
                break

        if not candidates:
            raise GraspGenerationError("antipodal contacts produced no valid 6-DoF poses")
        context.progress_cb("antipodal_candidates", 1.0)
        return self._as_grasp_candidates(
            candidates,
            frame=context.source_frame,
            num_raw=len(pairs) * self.approach_samples,
            centroid=np.median(points, axis=0),
        )

    def _as_grasp_candidates(
        self,
        candidates: Sequence[_Candidate],
        *,
        frame: str,
        num_raw: int,
        centroid: np.ndarray,
    ) -> GraspCandidates:
        poses = np.stack([candidate.pose for candidate in candidates])
        scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float32)
        order = directionally_diverse_indices(poses, scores, self.max_candidates)
        selected = [candidates[index] for index in order]
        return GraspCandidates(
            grasps=np.stack([candidate.pose for candidate in selected]),
            scores=np.asarray([candidate.score for candidate in selected], dtype=np.float32),
            centroid=np.asarray(centroid, dtype=float),
            frame=frame,
            num_raw=num_raw,
            widths=np.asarray([candidate.width for candidate in selected], dtype=np.float32),
        )

    def _obb_fallback_candidates(
        self,
        *,
        object_center: np.ndarray,
        object_axes: np.ndarray,
        robust_lower: np.ndarray,
        robust_upper: np.ndarray,
        preferred: Optional[np.ndarray],
    ) -> list[_Candidate]:
        robust_widths = robust_upper - robust_lower
        robust_midpoint = object_center + (0.5 * (robust_lower + robust_upper)) @ object_axes
        feasible_axes = [
            index
            for index, width in enumerate(robust_widths)
            if self.min_aperture_m <= float(width) <= self.max_aperture_m
        ]
        feasible_axes.sort(key=lambda index: float(robust_widths[index]))
        candidates: list[_Candidate] = []
        for axis_rank, axis_index in enumerate(feasible_axes):
            width = float(robust_widths[axis_index])
            closing = _normalize(np.asarray(object_axes[axis_index], dtype=float))
            if closing is None:
                continue
            first, second = _orthogonal_basis(closing)
            for sample in range(self.approach_samples):
                angle = 2.0 * math.pi * sample / self.approach_samples
                approach = _normalize(math.cos(angle) * first + math.sin(angle) * second)
                if approach is None:
                    continue
                binormal = _normalize(np.cross(approach, closing))
                if binormal is None:
                    continue
                approach = _normalize(np.cross(closing, binormal))
                pose = np.eye(4)
                pose[:3, :3] = np.column_stack((closing, binormal, approach))
                pose[:3, 3] = robust_midpoint
                semantic_score = (
                    0.18 * max(0.0, float(np.dot(approach, preferred)))
                    if preferred is not None
                    else 0.0
                )
                candidates.append(
                    _Candidate(
                        pose=pose,
                        # Stay below genuine contact-pair scores while keeping
                        # the thinnest feasible closing axis first.
                        score=0.42 - 0.03 * axis_rank + semantic_score + 0.01 * math.cos(angle),
                        width=width,
                    )
                )
        return candidates

    def _contact_pairs(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        *,
        object_center: np.ndarray,
        object_axes: np.ndarray,
        robust_half_extent: np.ndarray,
    ) -> list[tuple[int, int, float, float]]:
        pairs = []
        for i in range(len(points) - 1):
            delta = points[i + 1:] - points[i]
            widths = np.linalg.norm(delta, axis=1)
            possible = np.flatnonzero(
                (widths >= self.min_aperture_m) & (widths <= self.max_aperture_m)
            )
            for relative in possible:
                j = i + 1 + int(relative)
                width = float(widths[relative])
                closing = delta[relative] / width
                alignment_a = float(np.dot(normals[i], -closing))
                alignment_b = float(np.dot(normals[j], closing))
                if alignment_a < self.contact_cosine or alignment_b < self.contact_cosine:
                    continue
                opposing = max(0.0, float(-np.dot(normals[i], normals[j])))
                width_centre = 0.5 * (self.min_aperture_m + self.max_aperture_m)
                width_score = 1.0 - abs(width - width_centre) / self.max_aperture_m
                midpoint = 0.5 * (points[i] + points[j])
                midpoint_local = (midpoint - object_center) @ object_axes.T
                normalized_offset = midpoint_local / robust_half_extent
                centrality = math.exp(
                    -0.5 * float(normalized_offset @ normalized_offset),
                )
                score = (
                    0.45 * (alignment_a + alignment_b)
                    + 0.25 * opposing
                    + 0.1 * width_score
                    + 0.30 * centrality
                )
                pairs.append((i, j, score, width))
        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs[: self.max_contact_pairs]
