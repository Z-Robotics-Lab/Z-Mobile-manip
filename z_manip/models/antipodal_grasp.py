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
    vertical_index: Optional[int] = None  # row of ``axes`` snapped to gravity, or None


@dataclass(frozen=True)
class _RoundSection:
    """A recovered upright rotationally-symmetric cross-section (cylinder/bottle).

    A single wrist view sees only the front arc of a round object; its raw OBB
    axes therefore mislabel the front-to-back *radius* as a graspable width and
    put the observed centre ~r in front of the true axis.  Fitting a circle to
    the horizontal cross-section recovers the true axis centre and diameter and
    lets the closing axis be any horizontal direction (the gripper picks the one
    whose perpendicular approach is reachable).
    """

    center: np.ndarray        # (3,) true axis centre at the chosen grasp height
    diameter: float           # true object diameter (jaw opening), not the arc depth
    closing_axes: list["_ClosingAxis"]  # horizontal closing fan, all width == diameter


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
        up_axis: Sequence[float] = (0.0, 0.0, 1.0),
        gravity_snap_deg: float = 20.0,
        rotational_symmetry: bool = True,
        symmetry_closing_fan: int = 6,
        symmetry_span_deg: float = 165.0,
        symmetry_rms_ratio: float = 0.08,
        symmetry_min_slab_points: int = 40,
        height_search: bool = True,
        height_slabs: int = 7,
        height_offset_cap_m: float = 0.06,
        height_margin_weight: float = 0.5,
        height_min_extent_m: float = 0.08,
        corridor_backfill_min_candidates: int = 8,
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
        # Gravity/upright prior.  The cloud is expressed in the arm base frame
        # (``grasp_pipeline`` lifts every observation into it before calling a
        # backend, and its lift direction is base +Z), so the default up axis is
        # base +Z.  Tests and other frames may override it.
        up = np.asarray(up_axis, dtype=np.float64)
        up_unit = _normalize(up)
        self.up_axis = up_unit if up_unit is not None else np.array([0.0, 0.0, 1.0])
        self.gravity_snap_cos = math.cos(math.radians(max(0.0, float(gravity_snap_deg))))
        self.rotational_symmetry = bool(rotational_symmetry)
        self.symmetry_closing_fan = max(2, int(symmetry_closing_fan))
        self.symmetry_span_cos_bins = max(1, int(round(float(symmetry_span_deg) / 10.0)))
        self.symmetry_rms_ratio = max(1e-3, float(symmetry_rms_ratio))
        self.symmetry_min_slab_points = max(12, int(symmetry_min_slab_points))
        self.height_search = bool(height_search)
        self.height_slabs = max(1, int(height_slabs))
        self.height_offset_cap_m = max(0.0, float(height_offset_cap_m))
        self.height_margin_weight = max(0.0, float(height_margin_weight))
        # Short objects offer no meaningful cross-section variation; searching
        # a handful of noisy sparse slabs just makes the TCP jump run-to-run.
        self.height_min_extent_m = max(0.0, float(height_min_extent_m))
        # A small object on a support surface can have almost every oblique
        # corridor occupied by the support itself; hard-vetoing down to one or
        # two candidates starves downstream IK of alternatives.  Below this
        # floor the blocked poses are appended back with a strong score penalty
        # (downstream collision checking remains authoritative).  Zero disables.
        self.corridor_backfill_min_candidates = max(
            0, int(corridor_backfill_min_candidates)
        )

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
        # A single wrist view of a curved object exposes only its front arc, so
        # the raw OBB mislabels the front-to-back radius as a graspable width and
        # sits ~r in front of the true axis.  When the horizontal cross-section
        # fits a circle, recover the true axis/diameter and let the closing axis
        # be any horizontal direction; otherwise use the OBB faces (boxes).
        grasp_center = obb.center
        graspable: list[_ClosingAxis] = []
        section = self._round_cross_section(points, obb)
        if section is not None:
            graspable = [
                candidate
                for candidate in section.closing_axes
                if self.min_aperture_m <= candidate.width <= self.graspable_extent_m
            ]
            if graspable:
                grasp_center = section.center
        if not graspable:
            closing_axes = self._candidate_closing_axes(points, obb)
            graspable = [
                candidate
                for candidate in closing_axes
                if self.min_aperture_m <= candidate.width <= self.graspable_extent_m
            ]
            grasp_center = obb.center
            if not graspable:
                raise GraspGenerationError(
                    "no graspable face fits the gripper aperture with clearance; "
                    f"closing widths={[round(c.width, 3) for c in closing_axes]}, "
                    f"usable aperture={round(self.graspable_extent_m, 3)}"
                )

        context.progress_cb("approach_corridors", 0.55)
        environment = self._environment_points(context.scene_points, obb)
        preferred = _preferred_approach(context.affordance)

        blocked: list[_Candidate] = []
        candidates = self._build_candidates(
            graspable,
            environment=environment,
            preferred=preferred,
            grasp_center=grasp_center,
            enforce_corridor=True,
            blocked_out=blocked,
        )
        if not candidates:
            # Every corridor was occupied (a fully surrounded object).  Keep the
            # geometry usable by ranking on clearance instead of hard-vetoing;
            # downstream IK/collision remains authoritative.
            candidates = self._build_candidates(
                graspable,
                environment=environment,
                preferred=preferred,
                grasp_center=grasp_center,
                enforce_corridor=False,
            )
        elif len(candidates) < self.corridor_backfill_min_candidates and blocked:
            # Partially surrounded object (e.g. sitting on its support): a
            # couple of clean corridors is too thin a set for downstream IK.
            # Re-admit the vetoed poses at strongly penalized scores so clean
            # candidates always rank first; the planner's own swept-path
            # collision check is the authority on whether any of them is safe.
            blocked.sort(key=lambda candidate: -candidate.score)
            shortfall = self.corridor_backfill_min_candidates - len(candidates)
            candidates = list(candidates) + [
                _Candidate(
                    pose=candidate.pose,
                    score=0.25 * candidate.score,
                    width=candidate.width,
                    approach=candidate.approach,
                )
                for candidate in blocked[:shortfall]
            ]
        if not candidates:
            raise GraspGenerationError(
                "graspable faces produced no valid 6-DoF poses"
            )

        context.progress_cb("grasp_candidates", 1.0)
        return self._as_grasp_candidates(
            candidates,
            frame=context.source_frame,
            num_raw=len(graspable) * self.approach_samples,
            centroid=grasp_center,
        )

    # -- geometry ---------------------------------------------------------

    def _fit_obb(self, points: np.ndarray) -> _OBB:
        median = np.median(points, axis=0)
        centred = points - median
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        # Gravity/upright prior: PCA in-plane directions on a front-facing curved
        # patch are noisy, so a principal axis near vertical arrives tilted and
        # the gripper rotates off the graspable face.  Snap a near-vertical axis
        # to exact vertical and re-orthonormalize the other two into the exact
        # horizontal plane, so the two horizontal closing axes are level.
        axes, vertical_index = self._snap_axes_to_gravity(axes)
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
            vertical_index=vertical_index,
        )

    def _snap_axes_to_gravity(
        self,
        axes: np.ndarray,
    ) -> tuple[np.ndarray, Optional[int]]:
        """Snap a near-vertical principal axis to exact gravity, level the rest.

        Returns the (possibly rewritten) orthonormal axis rows and the index of
        the vertical axis, or ``None`` when no axis is within the snap cone (the
        object is not upright — leave PCA untouched).
        """

        up = self.up_axis
        axes = np.asarray(axes, dtype=np.float64)
        dots = axes @ up
        vertical_index = int(np.argmax(np.abs(dots)))
        if abs(float(dots[vertical_index])) < self.gravity_snap_cos:
            return axes, None
        others = [i for i in range(3) if i != vertical_index]
        # First horizontal axis: strip the vertical component from an existing
        # in-plane axis so the level gripper keeps the observed orientation as
        # much as possible; fall back to any horizontal direction if degenerate.
        horiz0 = axes[others[0]] - float(axes[others[0]] @ up) * up
        candidate = _normalize(horiz0)
        if candidate is None:
            candidate = _normalize(np.cross(up, axes[others[1]]))
        if candidate is None:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(up @ reference)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            candidate = _normalize(np.cross(up, reference))
        assert candidate is not None
        horiz1 = np.cross(up, candidate)
        snapped = axes.copy()
        snapped[vertical_index] = up * (1.0 if dots[vertical_index] >= 0.0 else -1.0)
        snapped[others[0]] = candidate
        snapped[others[1]] = horiz1
        return snapped, vertical_index

    # -- rotationally-symmetric (upright curved) completion ---------------

    @staticmethod
    def _fit_circle(uv: np.ndarray) -> tuple[float, float, float]:
        """Algebraic (Kåsa) circle fit; returns ``(u_center, v_center, radius)``."""

        design = np.column_stack((2.0 * uv[:, 0], 2.0 * uv[:, 1], np.ones(len(uv))))
        rhs = (uv ** 2).sum(axis=1)
        (uc, vc, c), *_ = np.linalg.lstsq(design, rhs, rcond=None)
        radius = math.sqrt(max(c + uc * uc + vc * vc, 1e-12))
        return float(uc), float(vc), float(radius)

    def _fit_round_slab(
        self,
        uv: np.ndarray,
        max_horizontal_extent: float,
        *,
        center_reference: tuple[float, float],
        diameter_band: tuple[float, float] = (0.6, 1.4),
    ) -> Optional[tuple[float, float, float]]:
        """Robustly fit a circle to one horizontal slab and gate its roundness.

        Iteratively rejects inliers beyond 3*MAD (labels, holes, stray depth),
        then accepts the fit only when the inlier RMS is a small fraction of the
        radius AND the inliers wrap a wide angular arc.  A flat face or a box
        corner fails one of those, so boxes fall back to the OBB face path.

        Sparse-cloud hard floors (small objects at range): a near-straight arc
        makes an algebraic circle fit ill-conditioned — it can hallucinate a
        huge radius whose centre sits far outside the object while still
        passing a *relative* RMS gate.  The fit is therefore rejected unless it
        keeps enough inliers, its diameter stays commensurate with the observed
        OBB horizontal extent (``diameter_band``), and its centre lands near the
        OBB centre (inside the observed footprint).  Any violation falls back
        to the OBB mid-plane path.
        Returns ``(u_center, v_center, radius)`` or ``None``.
        """

        if len(uv) < self.symmetry_min_slab_points:
            return None
        inliers = np.ones(len(uv), dtype=bool)
        uc = vc = radius = 0.0
        for _ in range(5):
            uc, vc, radius = self._fit_circle(uv[inliers])
            if radius <= 1e-6:
                return None
            r = np.hypot(uv[:, 0] - uc, uv[:, 1] - vc)
            mad = float(np.median(np.abs(r[inliers] - np.median(r[inliers])))) + 1e-9
            updated = np.abs(r - radius) < 3.0 * mad
            if int(updated.sum()) < 12:
                break
            inliers = updated
        if int(inliers.sum()) < self.symmetry_min_slab_points:
            return None
        r = np.hypot(uv[:, 0] - uc, uv[:, 1] - vc)
        rms = float(np.sqrt(np.mean((r[inliers] - radius) ** 2)))
        diameter = 2.0 * radius
        if diameter <= self.min_aperture_m:
            return None
        low_band, high_band = diameter_band
        if not (
            low_band * max_horizontal_extent
            <= diameter
            <= high_band * max_horizontal_extent
        ):
            return None
        centre_offset = math.hypot(uc - center_reference[0], vc - center_reference[1])
        if centre_offset > 0.6 * max_horizontal_extent + 0.005:
            return None
        if rms / radius > self.symmetry_rms_ratio:
            return None
        # Angular coverage of the on-circle inliers: a box's near-circle points
        # cluster at a few corners (narrow span); a real arc wraps continuously.
        on_circle = np.abs(r - radius) < 0.15 * radius
        angles = np.degrees(np.arctan2(uv[on_circle, 1] - vc, uv[on_circle, 0] - uc))
        occupied = np.zeros(36, dtype=bool)
        occupied[((angles % 360.0) // 10.0).astype(int)] = True
        if int(occupied.sum()) < self.symmetry_span_cos_bins:
            return None
        return uc, vc, radius

    def _round_cross_section(
        self,
        points: np.ndarray,
        obb: _OBB,
    ) -> Optional[_RoundSection]:
        """Recover a true upright round cross-section, or ``None`` for non-round.

        Fits circles to horizontal slabs along the vertical axis, confirms the
        object is round at mid-height, then (optionally) picks the grasp height
        whose cross-section leaves the most aperture margin while staying near
        the vertical mass centre.  The closing axis is left free: a fan of
        horizontal directions is returned, each with the true diameter, so the
        downstream reachable approach dictates which diameter the jaw closes on.
        """

        if not self.rotational_symmetry or obb.vertical_index is None:
            return None
        up = self.up_axis
        k = obb.vertical_index
        horizontal = [obb.axes[i] for i in range(3) if i != k]
        e1, e2 = horizontal[0], horizontal[1]
        max_horizontal_extent = float(max(obb.full_extent[i] for i in range(3) if i != k))
        heights = points @ up
        mass_center_h = float(np.median(heights))
        vertical_extent = float(obb.full_extent[k])
        # A thick slab gives a robust roundness verdict at mid-height; a thinner
        # slab resolves how the cross-section varies along the axis (neck vs
        # body) for the grasp-height search.
        detect_half = max(0.02, 0.25 * vertical_extent)
        eval_half = float(
            np.clip(0.5 * vertical_extent / max(1, self.height_slabs - 1), 0.015, 0.030)
        )

        obb_center_uv = (float(obb.center @ e1), float(obb.center @ e2))

        def fit_at(
            height: float,
            half: float,
            diameter_band: tuple[float, float],
        ) -> Optional[tuple[float, float, float]]:
            mask = np.abs(heights - height) <= half
            slab = points[mask]
            if len(slab) < self.symmetry_min_slab_points:
                return None
            uv = np.column_stack((slab @ e1, slab @ e2))
            return self._fit_round_slab(
                uv,
                max_horizontal_extent,
                center_reference=obb_center_uv,
                diameter_band=diameter_band,
            )

        reference = fit_at(mass_center_h, detect_half, (0.6, 1.4))
        if reference is None:
            return None

        best_height = mass_center_h
        best_fit = reference
        if (
            self.height_search
            and self.height_slabs > 1
            and vertical_extent >= self.height_min_extent_m
        ):
            # Prefer a narrower graspable cross-section (more jaw margin) but
            # penalise straying from the vertical mass centre so the grasp stays
            # stable; both terms are in metres and the offset is capped at the
            # smaller of the configured cap and half the object height.
            cap = min(self.height_offset_cap_m, 0.5 * vertical_extent)
            lo = float(np.quantile(heights, 0.10))
            hi = float(np.quantile(heights, 0.90))
            best_cost = 2.0 * reference[2]
            for height in np.linspace(lo, hi, self.height_slabs):
                offset = abs(float(height) - mass_center_h)
                if offset > cap:
                    continue
                # A validated round detection anchors the evaluation slabs, so
                # a genuinely narrower neck (well under the body diameter) may
                # pass; the centre-footprint and inlier floors still apply.
                fit = fit_at(float(height), eval_half, (0.3, 1.4))
                if fit is None:
                    continue
                cost = 2.0 * fit[2] + self.height_margin_weight * offset
                if cost < best_cost:
                    best_cost = cost
                    best_height = float(height)
                    best_fit = fit

        uc, vc, radius = best_fit
        center = uc * e1 + vc * e2 + best_height * up
        diameter = 2.0 * radius
        closing_axes: list[_ClosingAxis] = []
        for index in range(self.symmetry_closing_fan):
            angle = math.pi * index / self.symmetry_closing_fan
            axis = _normalize(math.cos(angle) * e1 + math.sin(angle) * e2)
            if axis is None:
                continue
            closing_axes.append(_ClosingAxis(axis=axis, width=diameter))
        if not closing_axes:
            return None
        return _RoundSection(center=center, diameter=diameter, closing_axes=closing_axes)

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
        environment: Optional[np.ndarray],
        preferred: Optional[np.ndarray],
        grasp_center: np.ndarray,
        enforce_corridor: bool,
        blocked_out: Optional[list["_Candidate"]] = None,
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

                clearance, corridor_blocked = self._corridor_clearance(
                    environment,
                    grasp_point=grasp_center,
                    approach=approach,
                    closing=closing,
                    binormal=binormal,
                    width=width,
                )
                if enforce_corridor and corridor_blocked and blocked_out is None:
                    continue

                pose = np.eye(4)
                pose[:3, :3] = np.column_stack((closing, binormal, approach))
                # The grasp point is the recovered object centre (round-section
                # axis for curved objects, OBB mid-plane otherwise), never the
                # observed near surface, so the jaw closes on the object centre.
                pose[:3, 3] = grasp_center

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
                candidate = _Candidate(
                    pose=pose,
                    score=score,
                    width=width,
                    approach=approach,
                )
                if enforce_corridor and corridor_blocked:
                    blocked_out.append(candidate)
                else:
                    candidates.append(candidate)
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
