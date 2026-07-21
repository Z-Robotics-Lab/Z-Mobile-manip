import numpy as np
import pytest

import z_manip.models.antipodal_grasp as antipodal_module
from z_manip.models.antipodal_grasp import AntipodalGraspSource
from z_manip.models.grasp_source import GraspContext, GraspGenerationError


def _context(points, affordance=None):
    return GraspContext(
        object_points=np.asarray(points, dtype=np.float32),
        bbox=None,
        source_frame="base_link",
        t_target_src=np.eye(4),
        scene_points=None,
        progress_cb=lambda _phase, _progress: None,
        affordance=affordance,
    )


def _cylinder_cloud(radius=0.032, height=0.14, n_angle=48, n_height=10):
    angles = np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False)
    zs = np.linspace(0.08, 0.08 + height, n_height)
    side = np.array([
        [0.48 + radius * np.cos(a), radius * np.sin(a), z]
        for a in angles for z in zs
    ])
    rings = np.linspace(0.0, radius, 7)
    caps = np.array([
        [0.48 + r * np.cos(a), r * np.sin(a), z]
        for z in (0.08, 0.08 + height) for r in rings for a in angles
    ])
    return np.vstack((side, caps))


def test_antipodal_source_generates_multi_direction_six_dof_grasps():
    candidates = AntipodalGraspSource(
        max_candidates=32,
        approach_samples=8,
    ).generate(_context(_cylinder_cloud()))

    assert candidates.frame == "base_link"
    assert candidates.grasps.shape[1:] == (4, 4)
    assert len(candidates.grasps) >= 8
    assert np.all((candidates.widths > 0.055) & (candidates.widths < 0.08))
    rotations = candidates.grasps[:, :3, :3]
    assert np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5)
    approach_axes = rotations[:, :, 2]
    assert np.any(np.abs(approach_axes[:, 2]) < 0.5), "side grasps must be present"
    assert np.any(np.abs(approach_axes[:, 2]) > 0.8), "vertical grasps may coexist"


def test_affordance_direction_changes_candidate_ranking():
    points = _cylinder_cloud()
    preferred = {"preferred_approach": (1.0, 0.0, 0.0)}
    candidates = AntipodalGraspSource(max_candidates=24).generate(
        _context(points, affordance=preferred)
    )
    best_approach = candidates.grasps[0, :3, 2]
    assert np.dot(best_approach, preferred["preferred_approach"]) > 0.65


def test_vertical_affordance_keeps_lateral_grasp_under_small_candidate_cap():
    candidates = AntipodalGraspSource(max_candidates=8).generate(
        _context(
            _cylinder_cloud(),
            affordance={"preferred_approach": (0.0, 0.0, 1.0)},
        ),
    )
    approach_z = candidates.grasps[:, 2, 2]

    assert np.max(approach_z) > 0.9
    assert np.any(np.abs(approach_z) < 0.35)


def test_best_antipodal_grasp_prefers_object_interior_over_edge_contacts():
    points = _cylinder_cloud(radius=0.032, height=0.14)
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    center = np.median(points, axis=0)
    extent = np.quantile(points, 0.90, axis=0) - np.quantile(points, 0.10, axis=0)
    normalized_offset = np.abs(candidates.grasps[0, :3, 3] - center) / np.maximum(
        extent,
        0.012,
    )

    assert np.linalg.norm(normalized_offset) < 0.35


def test_antipodal_source_rejects_object_wider_than_gripper():
    points = _cylinder_cloud(radius=0.07)
    with pytest.raises(GraspGenerationError, match="antipodal|aperture"):
        AntipodalGraspSource(max_aperture_m=0.085).generate(_context(points))


def test_default_generator_is_not_top_down_only():
    source = AntipodalGraspSource()
    assert source.approach_samples >= 6


def test_obb_fallback_keeps_small_cuboid_graspable_when_normals_are_one_sided(
    monkeypatch,
):
    xs = np.linspace(-0.020, 0.020, 8)
    ys = np.linspace(-0.032, 0.032, 10)
    zs = np.linspace(0.48, 0.508, 7)
    points = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    monkeypatch.setattr(
        antipodal_module,
        "_estimate_outward_normals",
        lambda sampled, _neighbours: np.tile((0.0, 0.0, 1.0), (len(sampled), 1)),
    )

    candidates = AntipodalGraspSource(
        max_aperture_m=0.068,
        max_candidates=32,
    ).generate(_context(points))

    assert len(candidates.grasps) >= 8
    assert np.all(candidates.widths >= 0.012)
    assert np.all(candidates.widths <= 0.068)
    assert np.min(candidates.widths) < 0.035
    assert np.allclose(candidates.centroid, np.median(points, axis=0), atol=0.005)
