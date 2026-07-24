"""Face selection and mid-plane geometry for the two-finger parallel gripper.

These fixtures exercise the geometric grasp redesign directly:

* a box with one un-graspable wide face must be grasped across a narrow face;
* a wall behind the object must veto the through-wall approach;
* the narrower graspable face must outrank the wider one;
* the grasp point must sit at the OBB mid-plane, not the observed near surface
  (the "slightly off / closed short" near-miss regression).
"""

import numpy as np
import pytest

from z_manip.models.antipodal_grasp import AntipodalGraspSource
from z_manip.models.grasp_source import GraspContext, GraspGenerationError


def _context(points, *, scene_points=None, affordance=None):
    return GraspContext(
        object_points=np.asarray(points, dtype=np.float32),
        bbox=None,
        source_frame="base_link",
        t_target_src=np.eye(4),
        scene_points=None if scene_points is None else np.asarray(scene_points, dtype=np.float32),
        progress_cb=lambda _phase, _progress: None,
        affordance=affordance,
    )


def _box_cloud(half, center=(0.5, 0.0, 0.25), per_edge=8, front_axis=None, front_extra=0):
    """Dense point cloud on all six faces of an axis-aligned box.

    ``front_axis`` (e.g. ``(0, -1, 0)``) plus ``front_extra`` add extra samples on
    one face to emulate a single-view cloud whose observed centroid is biased
    toward the camera; a correct implementation must still grasp the mid-plane.
    """

    hx, hy, hz = (float(value) for value in half)
    cx, cy, cz = (float(value) for value in center)
    us = np.linspace(-1.0, 1.0, per_edge)
    points = []
    for sign in (-1.0, 1.0):
        points += [(cx + sign * hx, cy + u * hy, cz + w * hz) for u in us for w in us]
        points += [(cx + u * hx, cy + sign * hy, cz + w * hz) for u in us for w in us]
        points += [(cx + u * hx, cy + w * hy, cz + sign * hz) for u in us for w in us]
    if front_axis is not None and front_extra > 0:
        axis = np.asarray(front_axis, dtype=float)
        vs = np.linspace(-1.0, 1.0, front_extra)
        offset = np.array([cx, cy, cz]) + axis * np.array([hx, hy, hz])
        # Tangent samples across the biased face.
        tangents = [t for t in (np.array([hx, 0, 0]), np.array([0, hy, 0]), np.array([0, 0, hz]))
                    if abs(float(np.dot(t / (np.linalg.norm(t) + 1e-9), axis))) < 0.5]
        for a in vs:
            for b in vs:
                points.append(tuple(offset + a * tangents[0] + b * tangents[1]))
    return np.asarray(points, dtype=np.float64)


def _wall_plane(y, center, half_span=0.15, n=24):
    us = np.linspace(-half_span, half_span, n)
    cx, _, cz = center
    return np.array([(cx + u, y, cz + w) for u in us for w in us], dtype=np.float64)


def test_wide_face_rejected_grasp_uses_narrow_axis():
    # 0.11 x 0.05 x 0.03 box: the 0.11 m face never fits a two-finger jaw.
    points = _box_cloud((0.055, 0.025, 0.015))
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))

    # No candidate spans anything close to the 0.11 m axis (all graspable
    # closing axes stay well under the usable aperture).
    assert np.max(candidates.widths) < 0.06
    # The narrowest axis is the 0.03 m (z) face; the best grasp closes across it,
    # never across the 0.11 m x-axis.
    best_closing = candidates.grasps[0, :3, 0]
    assert abs(float(best_closing @ np.array([0.0, 0.0, 1.0]))) > 0.9
    assert abs(float(best_closing @ np.array([1.0, 0.0, 0.0]))) < 0.1
    assert candidates.widths[0] < 0.04


def test_wide_face_rejected_raises_when_only_axis_is_too_wide():
    # A slab whose two small faces are still wider than the usable aperture.
    points = _box_cloud((0.06, 0.05, 0.045))
    with pytest.raises(GraspGenerationError, match="aperture|graspable"):
        AntipodalGraspSource(max_aperture_m=0.085).generate(_context(points))


def test_wall_behind_object_vetoes_through_wall_approach():
    center = (0.45, 0.0, 0.20)
    half = (0.02, 0.025, 0.03)
    obj = _box_cloud(half, center=center)
    wall = _wall_plane(center[1] + half[1] + 0.02, center)  # 2 cm behind back face
    scene = np.vstack((obj, wall))

    source = AntipodalGraspSource(max_candidates=48)

    walled = source.generate(_context(obj, scene_points=scene))
    approaches = walled.grasps[:, :3, 2]
    wall_outward = np.array([0.0, 1.0, 0.0])
    # The through-wall approach (pregrasp behind the wall) points opposite the
    # wall's outward normal; it must be gone.
    assert np.all(approaches @ wall_outward >= -0.5)
    assert len(walled.grasps) >= 4

    # Without the scene the same object DOES expose the through-wall approach,
    # proving the corridor check — not the geometry — removed it.
    open_scene = source.generate(_context(obj))
    open_approaches = open_scene.grasps[:, :3, 2]
    assert np.any(open_approaches @ wall_outward < -0.5)


def test_narrower_graspable_face_outranks_wider_face():
    # x = 0.03 (narrow, graspable), y = 0.06 (wide, graspable), z = 0.10 (rejected).
    points = _box_cloud((0.015, 0.03, 0.05))
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))

    # Best grasp closes across the narrow 0.03 m x-axis.
    best_closing = candidates.grasps[0, :3, 0]
    assert abs(float(best_closing @ np.array([1.0, 0.0, 0.0]))) > 0.9
    assert candidates.widths[0] < 0.04
    assert candidates.widths[0] <= float(np.min(candidates.widths)) + 1e-6


def test_grasp_point_is_obb_midplane_not_front_surface():
    center = np.array([0.5, 0.0, 0.25])
    half = np.array([0.02, 0.025, 0.03])
    # Front (camera-facing, -y) face over-sampled: a surface-weighted contact
    # midpoint would drift toward y = -0.025.  The mid-plane fix must not.
    points = _box_cloud(
        tuple(half),
        center=tuple(center),
        front_axis=(0.0, -1.0, 0.0),
        front_extra=14,
    )
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))

    front_surface_y = center[1] - half[1]
    for translation in candidates.grasps[:, :3, 3]:
        # Every grasp point is the object centre, within a few millimetres.
        assert np.linalg.norm(translation - center) < 0.004
        # And it is far from the near surface — fingers wrap the object, they do
        # not pinch its front edge.
        assert abs(translation[1] - center[1]) < abs(translation[1] - front_surface_y)
        assert abs(translation[1] - front_surface_y) > 0.015


def test_width_margin_scales_with_face_narrowness():
    narrow = _box_cloud((0.015, 0.03, 0.05))
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(narrow))
    widths = np.asarray(candidates.widths, dtype=float)
    scores = np.asarray(candidates.scores, dtype=float)
    # Narrow-face grasps carry a higher width-margin bonus, so the mean score of
    # the narrowest quartile exceeds that of the widest quartile.
    order = np.argsort(widths)
    narrowest = scores[order[: max(1, len(order) // 4)]]
    widest = scores[order[-max(1, len(order) // 4):]]
    assert float(np.mean(narrowest)) > float(np.mean(widest))
