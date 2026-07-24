"""Object-agnostic geometric grasp generation for a TWO-FINGER parallel gripper.

The generator treats the observed object cloud as an oriented bounding box (OBB)
and reasons about which *face* a parallel jaw can actually straddle.  A face is
graspable only when the object extent along the closing axis fits the aperture
with clearance for the finger pads; wide faces are rejected outright.  It then
samples approach directions around each graspable closing axis, places the grasp
point at the OBB mid-plane (not the observed near surface), and vetoes approaches
whose finger-sweep corridor is occupied by the surrounding scene (e.g. an object
sitting against a wall).

This CPU backend is the deterministic fallback and regression anchor.  A learned
grasp model may replace candidate proposal, but all candidates still pass the
same width, collision, IK, and motion-planning filters downstream.

Convention: a grasp pose is source-gripper (Franka TCP).  Its rotation columns
are ``(closing, binormal, approach)`` so tool-X is the finger closing axis and
tool-Z is the approach direction the arm drives along into contact (matches
``grasp_pipeline.grasp_pregrasp_pose`` and ``grasp_ordering``).  The translation
is the TCP contact point; finger/palm offsets live in ``grasp_plan.tool_from_tip``
so the module only has to put the TCP at the object mid-plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .grasp_source import GraspCandidates, GraspContext, GraspGenerationError
from .grasp_ordering import directionally_diverse_indices

# Scoring weights for a graspable (closing-axis, approach) candidate.  Width
# margin dominates so the narrowest graspable face is preferred; the preferred
# (affordance) approach and corridor clearance shape the rest of the ordering.
_SCORE_BASE = 0.30
_W_WIDTH_MARGIN = 0.28
_W_CORRIDOR = 0.22
_W_PREFERRED = 0.35
_W_DIVERSITY = 0.02


def _normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _estimate_outward_normals(points: np.ndarray, neighbours: int) -> np.ndarray:
    tree = cKDTree(points)
    k = min(max(6, neighbours), len(points))
    _, indices = tree.query(points, k=k)
    local = points[np.asarray(indices)]
    centred = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred, optimize=True)
    covariance /= max(1, k - 1)
    # ``numpy.linalg.eigh`` accepts a stack of symmetric matrices.  This is
    # mathematically identical to the former Python loop but removes hundreds
    # of small LAPACK dispatches from every perception request.
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    centroid = np.median(points, axis=0)
    inward = np.einsum("ni,ni->n", normals, points - centroid) < 0.0
    normals[inward] *= -1.0
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
class _OBB:
    """Robust oriented bounding box fit to the observed object cloud."""

    center: np.ndarray        # (3,) robust mid-plane centre in the cloud frame
    median: np.ndarray        # (3,) cloud median (frame anchor for projections)
    axes: np.ndarray          # (3, 3) orthonormal principal axes as rows
    half_extent: np.ndarray   # (3,) robust half-extent along each principal axis
    full_extent: np.ndarray   # (3,) robust full extent along each principal axis


@dataclass(frozen=True)
class _ClosingAxis:
    """A candidate closing direction with its object extent along that axis."""

    axis: np.ndarray  # (3,) unit closing direction
    width: float      # object extent along ``axis`` (required jaw opening)


@dataclass(frozen=True)
class _Candidate:
    pose: np.ndarray
    score: float
    width: float
    approach: np.ndarray


class AntipodalGraspSource:
    """Generate face-aware parallel-jaw candidates from an object point cloud."""

    def __init__(
        self,
        *,
        min_aperture_m: float = 0.012,
        max_aperture_m: float = 0.085,
        aperture_clearance_m: float = 0.010,
        contact_angle_deg: float = 42.0,
        normal_neighbours: int = 18,
        approach_samples: int = 8,
        max_surface_points: int = 420,
        max_contact_pairs: int = 96,
        max_candidates: int = 64,
        max_extra_closing_axes: int = 4,
        extra_axis_min_angle_deg: float = 18.0,
        corridor_depth_m: float = 0.10,
        corridor_fingertip_m: float = 0.02,
        corridor_binormal_half_m: float = 0.045,
        corridor_closing_pad_m: float = 0.012,
        corridor_block_count: int = 4,
    ) -> None:
        self.min_aperture_m = float(min_aperture_m)
        self.max_aperture_m = float(max_aperture_m)
        # A parallel jaw needs a little slack on each finger to straddle a face
        # without its pads bumping the object, so the usable closing extent is
        # the aperture minus this clearance rather than the raw aperture.
        self.aperture_clearance_m = max(0.0, float(aperture_clearance_m))
        self.graspable_extent_m = max(
            self.min_aperture_m,
            self.max_aperture_m - self.aperture_clearance_m,
        )
        self.contact_cosine = math.cos(math.radians(contact_angle_deg))
        self.normal_neighbours = int(normal_neighbours)
        self.approach_samples = max(6, int(approach_samples))
        self.max_surface_points = max(32, int(max_surface_points))
        self.max_contact_pairs = max(1, int(max_contact_pairs))
        self.max_candidates = max(1, int(max_candidates))
        self.max_extra_closing_axes = max(0, int(max_extra_closing_axes))
        self.extra_axis_min_cos = math.cos(math.radians(extra_axis_min_angle_deg))
        self.corridor_depth_m = max(0.0, float(corridor_depth_m))
        self.corridor_fingertip_m = max(0.0, float(corridor_fingertip_m))
        self.corridor_binormal_half_m = max(1e-3, float(corridor_binormal_half_m))
        self.corridor_closing_pad_m = max(0.0, float(corridor_closing_pad_m))
        self.corridor_block_count = max(1, int(corridor_block_count))

    # -- public API -------------------------------------------------------

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

        obb = self._fit_obb(points)
        if float(np.min(obb.full_extent)) > self.max_aperture_m * 1.05:
            raise GraspGenerationError(
                "object has no OBB dimension within gripper aperture; "
                f"extent={obb.full_extent.round(3).tolist()}"
            )

        context.progress_cb("closing_axes", 0.15)
        closing_axes = self._candidate_closing_axes(points, obb)
        graspable = [
            candidate
            for candidate in closing_axes
            if self.min_aperture_m <= candidate.width <= self.graspable_extent_m
        ]
        if not graspable:
            raise GraspGenerationError(
                "no graspable face fits the gripper aperture with clearance; "
                f"closing widths={[round(c.width, 3) for c in closing_axes]}, "
                f"usable aperture={round(self.graspable_extent_m, 3)}"
            )

        context.progress_cb("approach_corridors", 0.55)
        environment = self._environment_points(context.scene_points, obb)
        preferred = _preferred_approach(context.affordance)

        candidates = self._build_candidates(
            graspable,
            obb=obb,
            environment=environment,
            preferred=preferred,
            enforce_corridor=True,
        )
        if not candidates:
            # Every corridor was occupied (a fully surrounded object).  Keep the
            # geometry usable by ranking on clearance instead of hard-vetoing;
            # downstream IK/collision remains authoritative.
            candidates = self._build_candidates(
                graspable,
                obb=obb,
                environment=environment,
                preferred=preferred,
                enforce_corridor=False,
            )
        if not candidates:
            raise GraspGenerationError(
                "graspable faces produced no valid 6-DoF poses"
            )

        context.progress_cb("grasp_candidates", 1.0)
        return self._as_grasp_candidates(
            candidates,
            frame=context.source_frame,
            num_raw=len(graspable) * self.approach_samples,
            centroid=obb.center,
        )

    # -- geometry ---------------------------------------------------------

    def _fit_obb(self, points: np.ndarray) -> _OBB:
        median = np.median(points, axis=0)
        centred = points - median
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        projected = centred @ axes.T
        # 1st/99th percentiles instead of raw min/max: FFS depth is clean but a
        # handful of stray points must not inflate a face width and reject an
        # otherwise-graspable axis.
        lower = np.quantile(projected, 0.01, axis=0)
        upper = np.quantile(projected, 0.99, axis=0)
        full_extent = upper - lower
        half_extent = 0.5 * np.maximum(full_extent, self.min_aperture_m)
        # The mid-plane centre, mapped back to the cloud frame.  A single-view
        # cloud biases the median toward the camera-facing surface; taking the
        # centre of the robust extent box recovers the object's mid-plane, which
        # is where the TCP must sit so the fingers wrap AROUND the object rather
        # than pinching its near edge.
        center = median + (0.5 * (lower + upper)) @ axes
        return _OBB(
            center=center,
            median=median,
            axes=axes,
            half_extent=half_extent,
            full_extent=full_extent,
        )

    def _robust_extent_along(self, points: np.ndarray, obb: _OBB, axis: np.ndarray) -> float:
        projection = (points - obb.median) @ axis
        lower = float(np.quantile(projection, 0.01))
        upper = float(np.quantile(projection, 0.99))
        return upper - lower

    def _candidate_closing_axes(
        self,
        points: np.ndarray,
        obb: _OBB,
    ) -> list[_ClosingAxis]:
        """Enumerate closing directions: the 3 OBB axes plus antipodal pairs.

        The OBB axes are the primary, robust closing directions for the box-like
        objects this stack manipulates.  Antipodal contact pairs contribute extra
        closing directions for rounded objects whose graspable axis is not a
        principal axis; each is de-duplicated against the OBB axes by angle.
        """

        axes: list[_ClosingAxis] = []
        for index in range(3):
            axis = _normalize(np.asarray(obb.axes[index], dtype=float))
            if axis is None:
                continue
            axes.append(_ClosingAxis(axis=axis, width=float(obb.full_extent[index])))

        if self.max_extra_closing_axes <= 0:
            return axes

        pair_directions = self._contact_pair_directions(points, obb)
        added = 0
        for direction in pair_directions:
            if added >= self.max_extra_closing_axes:
                break
            if any(
                abs(float(np.dot(direction, existing.axis))) > self.extra_axis_min_cos
                for existing in axes
            ):
                continue
            width = self._robust_extent_along(points, obb, direction)
            axes.append(_ClosingAxis(axis=direction, width=width))
            added += 1
        return axes

    def _contact_pair_directions(
        self,
        points: np.ndarray,
        obb: _OBB,
    ) -> list[np.ndarray]:
        """Sign-normalized closing directions from opposing antipodal contacts."""

        try:
            normals = _estimate_outward_normals(points, self.normal_neighbours)
        except Exception:
            return []
        directions: list[np.ndarray] = []
        scores: list[float] = []
        for i in range(len(points) - 1):
            delta = points[i + 1:] - points[i]
            widths = np.linalg.norm(delta, axis=1)
            possible = np.flatnonzero(
                (widths >= self.min_aperture_m) & (widths <= self.graspable_extent_m)
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
                direction = closing if closing[np.argmax(np.abs(closing))] >= 0 else -closing
                directions.append(direction)
                scores.append(alignment_a + alignment_b + opposing)
        if not directions:
            return []
        order = np.argsort(scores)[::-1][: self.max_contact_pairs]
        return [directions[index] for index in order]

    # -- candidate assembly ----------------------------------------------

    def _build_candidates(
        self,
        graspable: Sequence[_ClosingAxis],
        *,
        obb: _OBB,
        environment: Optional[np.ndarray],
        preferred: Optional[np.ndarray],
        enforce_corridor: bool,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        usable = max(self.graspable_extent_m - self.min_aperture_m, 1e-6)
        for closing_axis in graspable:
            closing = closing_axis.axis
            width = closing_axis.width
            width_margin = float(
                np.clip((self.graspable_extent_m - width) / usable, 0.0, 1.0)
            )
            first, second = _orthogonal_basis(closing)
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
                if approach is None:
                    continue

                clearance, blocked = self._corridor_clearance(
                    environment,
                    grasp_point=obb.center,
                    approach=approach,
                    closing=closing,
                    binormal=binormal,
                    width=width,
                )
                if enforce_corridor and blocked:
                    continue

                pose = np.eye(4)
                pose[:3, :3] = np.column_stack((closing, binormal, approach))
                # Mid-plane fix: the grasp point is the OBB centre, not the
                # observed near surface, so the jaw closes on the object centre.
                pose[:3, 3] = obb.center

                preferred_score = 0.0
                if preferred is not None:
                    preferred_score = _W_PREFERRED * max(
                        0.0, float(np.dot(approach, preferred))
                    )
                score = (
                    _SCORE_BASE
                    + _W_WIDTH_MARGIN * width_margin
                    + _W_CORRIDOR * clearance
                    + preferred_score
                    + _W_DIVERSITY * math.cos(angle)
                )
                candidates.append(
                    _Candidate(
                        pose=pose,
                        score=score,
                        width=width,
                        approach=approach,
                    )
                )
        return candidates

    def _environment_points(
        self,
        scene_points: Optional[object],
        obb: _OBB,
    ) -> Optional[np.ndarray]:
        """Scene points with the object itself removed, for corridor checks."""

        if scene_points is None:
            return None
        scene = np.asarray(scene_points, dtype=np.float64)
        if scene.ndim != 2 or scene.shape[1] != 3 or len(scene) == 0:
            return None
        scene = scene[np.isfinite(scene).all(axis=1)]
        if len(scene) == 0:
            return None
        local = np.abs((scene - obb.median) @ obb.axes.T)
        # Drop everything inside the (slightly grown) object box so the object
        # never vetoes its own approaches; what remains is the surrounding scene
        # (wall, support rail, neighbours).
        margin = obb.half_extent + 0.015
        inside = np.all(local <= margin, axis=1)
        environment = scene[~inside]
        return environment if len(environment) else None

    def _corridor_clearance(
        self,
        environment: Optional[np.ndarray],
        *,
        grasp_point: np.ndarray,
        approach: np.ndarray,
        closing: np.ndarray,
        binormal: np.ndarray,
        width: float,
    ) -> tuple[float, bool]:
        """Occupancy of the finger-sweep corridor behind the grasp point.

        The hand travels from the pregrasp (``grasp_point - depth * approach``)
        into contact, so the swept volume is a box extending along ``-approach``
        from the grasp point, as wide as the jaw plus a finger pad and as tall as
        the finger span.  A wall or support inside that box means the approach
        drives the fingers through the obstacle.  Returns ``(clearance, blocked)``
        with clearance in ``(0, 1]`` for soft ranking and a hard-veto flag.
        """

        if environment is None or len(environment) == 0:
            return 1.0, False
        delta = environment - grasp_point
        along = delta @ approach
        lateral = delta @ closing
        span = delta @ binormal
        in_corridor = (
            (along >= -self.corridor_depth_m)
            & (along <= self.corridor_fingertip_m)
            & (np.abs(lateral) <= 0.5 * width + self.corridor_closing_pad_m)
            & (np.abs(span) <= self.corridor_binormal_half_m)
        )
        occupied = int(np.count_nonzero(in_corridor))
        clearance = 1.0 / (1.0 + occupied)
        blocked = occupied >= self.corridor_block_count
        return clearance, blocked

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
