import numpy as np
import pytest

import z_manip.models.antipodal_grasp as antipodal_module
from z_manip.models.antipodal_grasp import AntipodalGraspSource
from z_manip.models.grasp_source import GraspContext, GraspGenerationError
from z_manip.ik.symmetry import expand_symmetry

_UP = np.array([0.0, 0.0, 1.0])


def _angle_to_vertical_deg(vector):
    unit = np.asarray(vector, dtype=float)
    unit = unit / np.linalg.norm(unit)
    return float(np.degrees(np.arccos(np.clip(abs(unit @ _UP), 0.0, 1.0))))


def _upright_half_cylinder(cx=0.5, cy=0.0, radius=0.030, z0=0.10, height=0.18,
                           n_angle=60, n_height=20):
    """Front-only 180-degree arc of an upright cylinder (a single wrist view)."""

    angles = np.linspace(-np.pi, 0.0, n_angle)  # front half, y <= cy
    zs = np.linspace(z0, z0 + height, n_height)
    return np.array([
        [cx + radius * np.cos(a), cy + radius * np.sin(a), z]
        for a in angles for z in zs
    ])


def _context(points, affordance=None, scene_points=None):
    return GraspContext(
        object_points=np.asarray(points, dtype=np.float32),
        bbox=None,
        source_frame="base_link",
        t_target_src=np.eye(4),
        scene_points=(
            None if scene_points is None
            else np.asarray(scene_points, dtype=np.float32)
        ),
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


def _reference_normals(points, neighbours):
    tree = antipodal_module.cKDTree(points)
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


def test_vectorized_normals_match_reference_geometry():
    points = _cylinder_cloud(n_angle=20, n_height=6).astype(np.float64)
    actual = antipodal_module._estimate_outward_normals(points, 18)
    expected = _reference_normals(points, 18)

    # Eigenvector signs are fixed outward by both implementations. Compare
    # directions, allowing only numerical noise from batched LAPACK dispatch.
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)


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


# -- partial-view curved-object recovery (single wrist view) -----------------


def test_half_cylinder_front_view_grasps_across_true_diameter():
    # A single wrist view sees only the FRONT arc of a standing bottle.  The raw
    # OBB would call the front-to-back radius a graspable width and sit ~r in
    # front of the true axis; the circle completion must recover the true axis
    # and close horizontally across the real diameter.
    radius = 0.030
    points = _upright_half_cylinder(cx=0.5, cy=0.0, radius=radius)
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))

    best_closing = candidates.grasps[0, :3, 0]
    assert _angle_to_vertical_deg(best_closing) > 87.0  # horizontal within 3 deg
    tcp = candidates.grasps[0, :3, 3]
    # TCP lands on the true cylinder axis, not the observed front surface.
    assert np.hypot(tcp[0] - 0.5, tcp[1] - 0.0) < 0.005
    # Commanded width is the true diameter, within 10 percent.
    assert abs(float(candidates.widths[0]) - 2.0 * radius) <= 0.10 * (2.0 * radius)
    # Every closing axis is horizontal (all fan members are level).
    assert np.all(np.abs(candidates.grasps[:, 2, 0]) < 0.06)


def test_tilted_upright_object_snaps_closing_axis_to_gravity():
    # An upright thin box tilted 12 deg: PCA yields a tilted vertical axis, so
    # without the gravity prior the level jaw arrives rotated off the face.
    half = np.array([0.014, 0.028, 0.09])
    center = np.array([0.5, 0.0, 0.30])
    tilt = np.radians(12.0)
    # Tilt about Y so the narrow (x) closing axis genuinely leaves the horizontal
    # plane; the vertical (z) axis stays inside the snap cone.
    rot_y = np.array([
        [np.cos(tilt), 0.0, np.sin(tilt)],
        [0.0, 1.0, 0.0],
        [-np.sin(tilt), 0.0, np.cos(tilt)],
    ])
    us = np.linspace(-1.0, 1.0, 9)
    local = []
    for sign in (-1.0, 1.0):
        local += [(sign * half[0], u * half[1], w * half[2]) for u in us for w in us]
        local += [(u * half[0], sign * half[1], w * half[2]) for u in us for w in us]
        local += [(u * half[0], w * half[1], sign * half[2]) for u in us for w in us]
    points = (np.asarray(local) @ rot_y.T) + center

    snapped = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    assert _angle_to_vertical_deg(snapped.grasps[0, :3, 0]) > 87.0  # level jaw

    # With the prior disabled the same cloud keeps the ~12 deg PCA tilt, proving
    # the snap — not the geometry — leveled the closing axis.
    unsnapped = AntipodalGraspSource(
        max_candidates=32,
        gravity_snap_deg=0.0,
    ).generate(_context(points))
    assert _angle_to_vertical_deg(unsnapped.grasps[0, :3, 0]) < 84.0


def test_varying_cross_section_prefers_narrower_graspable_height():
    # A fat body (D=72 mm) with a narrower neck (D=40 mm) above it: the grasp
    # height should move up to the neck for aperture margin, staying within the
    # stability cap of the mass centre.
    cx, cy = 0.5, 0.0
    points = []
    for radius, z0, z1, n_height in ((0.036, 0.10, 0.20, 18), (0.020, 0.205, 0.265, 12)):
        for a in np.linspace(-np.pi, 0.0, 50):
            for z in np.linspace(z0, z1, n_height):
                points.append([cx + radius * np.cos(a), cy + radius * np.sin(a), z])
    points = np.asarray(points)
    mass_center_z = float(np.median(points[:, 2]))

    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    best = candidates.grasps[0]
    # Grasp height rose toward the neck rather than staying at the fat body.
    assert best[2, 3] > mass_center_z + 0.02
    assert best[2, 3] >= 0.205
    # And it closes across the narrow neck diameter, not the 72 mm body.
    assert float(candidates.widths[0]) < 0.050
    assert abs(float(candidates.widths[0]) - 0.040) <= 0.15 * 0.040
    assert _angle_to_vertical_deg(best[:3, 0]) > 87.0


def test_uniform_cylinder_grasp_height_stays_near_mass_center():
    # The height-margin term must not move a uniform object off its centre.
    points = _cylinder_cloud(radius=0.032, height=0.16)
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    mass_center_z = float(np.median(points[:, 2]))
    assert abs(float(candidates.grasps[0, 2, 3]) - mass_center_z) < 0.02


def test_closing_axis_convention_survives_symmetry_expansion():
    # End-to-end convention pin: column 0 is the physical jaw-opening axis
    # (tool-X per grasp_plan.tool_from_tip).  For an upright cylinder it must be
    # horizontal, and NO approach-axis symmetry member may rotate it toward the
    # object body (the observed "jaws parallel to the bottle" 90-degree failure).
    points = _cylinder_cloud(radius=0.032, height=0.16)
    candidates = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    grasp = candidates.grasps[0]

    # Column order is (closing, binormal, approach); tool-Z (approach) is col 2.
    assert _angle_to_vertical_deg(grasp[:3, 0]) > 87.0  # closing horizontal
    rotation = grasp[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)

    family = expand_symmetry(grasp, n_about_axis=4)
    for member in family:
        # Approach axis is preserved; closing never swings toward vertical.
        assert np.allclose(member[:3, 2], grasp[:3, 2], atol=1e-6)
        assert _angle_to_vertical_deg(member[:3, 0]) > 60.0


# -- small-object robustness (sparse single-view clouds) ---------------------


def _rng_thin(points, keep, seed=7):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=min(keep, len(points)), replace=False)
    return np.asarray(points)[np.sort(indices)]


def _small_box_cloud(half=(0.025, 0.018, 0.008), center=(0.47, -0.05, 0.0),
                     per_edge=7, noise=0.0015, seed=3):
    hx, hy, hz = half
    cx, cy, cz = center
    us = np.linspace(-1.0, 1.0, per_edge)
    pts = []
    for sign in (-1.0, 1.0):
        pts += [(cx + sign * hx, cy + u * hy, cz + w * hz) for u in us for w in us]
        pts += [(cx + u * hx, cy + sign * hy, cz + w * hz) for u in us for w in us]
        pts += [(cx + u * hx, cy + w * hy, cz + sign * hz) for u in us for w in us]
    rng = np.random.default_rng(seed)
    return np.asarray(pts) + rng.normal(0.0, noise, (len(pts), 3))


def _tcp_error_mm(candidates, true_center):
    best = candidates.grasps[0, :3, 3]
    return 1000.0 * float(np.linalg.norm(best - np.asarray(true_center)))


def test_small_sparse_box_localizes_no_worse_than_obb_path():
    # A charger-sized box (50 x 36 x 16 mm) at realistic sparse density with
    # depth noise: the circle-completion path must never hijack it and push the
    # TCP away — small boxes localize exactly as the plain OBB path does.
    center = (0.47, -0.05, 0.0)
    points = _rng_thin(_small_box_cloud(center=center), 260)
    new = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    obb_only = AntipodalGraspSource(
        max_candidates=32,
        rotational_symmetry=False,
    ).generate(_context(points))
    assert _tcp_error_mm(new, center) <= _tcp_error_mm(obb_only, center) + 1.0
    assert _tcp_error_mm(new, center) < 8.0


def test_small_sparse_half_cylinder_localizes_no_worse_than_obb_path():
    # A 32 mm-diameter bottle neck seen as a sparse front arc: the completion
    # should help (or at worst match) the raw OBB mid-plane.
    radius, cx, cy = 0.016, 0.47, -0.05
    dense = _upright_half_cylinder(cx=cx, cy=cy, radius=radius, z0=-0.03,
                                   height=0.06, n_angle=30, n_height=10)
    rng = np.random.default_rng(11)
    points = _rng_thin(dense + rng.normal(0.0, 0.001, dense.shape), 170)
    axis_xy = np.array([cx, cy])

    def horizontal_error_mm(candidates):
        tcp = candidates.grasps[0, :3, 3]
        return 1000.0 * float(np.hypot(tcp[0] - axis_xy[0], tcp[1] - axis_xy[1]))

    new = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    obb_only = AntipodalGraspSource(
        max_candidates=32,
        rotational_symmetry=False,
    ).generate(_context(points))
    assert horizontal_error_mm(new) <= horizontal_error_mm(obb_only) + 1.0
    assert horizontal_error_mm(new) < 8.0


def test_shallow_arc_does_not_hallucinate_far_circle_center():
    # A nearly flat curved patch: an unguarded algebraic circle fit recovers a
    # huge radius whose centre lies far outside the object.  The hard floors
    # (diameter band, centre-in-footprint) must reject the fit so the grasp
    # point stays at the OBB mid-plane.
    cx, cy, radius = 0.47, -0.05, 0.30  # 600 mm circle: locally almost flat
    arc = np.linspace(-0.06, 0.06, 26)  # ~34 mm-wide shallow patch
    zs = np.linspace(-0.05, 0.05, 12)
    points = np.array([
        [cx + radius * (np.cos(a) - 1.0), cy + radius * np.sin(a), z]
        for a in arc for z in zs
    ])
    rng = np.random.default_rng(5)
    points = points + rng.normal(0.0, 0.0008, points.shape)

    new = AntipodalGraspSource(max_candidates=32).generate(_context(points))
    obb_only = AntipodalGraspSource(
        max_candidates=32,
        rotational_symmetry=False,
    ).generate(_context(points))
    # Identical grasp point: the round path declined, OBB mid-plane won.
    assert np.allclose(new.grasps[0, :3, 3], obb_only.grasps[0, :3, 3], atol=1e-6)


def test_corridor_backfill_rescues_starved_candidate_set():
    # A charger lying on a support surface: the support occupies almost every
    # oblique finger corridor, which live starved the planner down to two
    # candidates.  The backfill floor re-admits vetoed poses at strongly
    # penalized scores; clean candidates always outrank them.
    center = (0.47, -0.05, 0.012)
    obj = _small_box_cloud(half=(0.025, 0.018, 0.008), center=center, noise=0.0005)
    xs = np.linspace(-0.12, 0.12, 22)
    support = np.array([
        [center[0] + u, center[1] + v, 0.0] for u in xs for v in xs
    ])
    scene = np.vstack((obj, support))
    source = AntipodalGraspSource(max_candidates=32)
    starved = AntipodalGraspSource(
        max_candidates=32,
        corridor_backfill_min_candidates=0,
    ).generate(_context(obj, scene_points=scene))
    rescued = source.generate(_context(obj, scene_points=scene))
    assert len(rescued.grasps) >= min(
        source.corridor_backfill_min_candidates,
        len(starved.grasps) + 1,
    )
    assert len(rescued.grasps) > len(starved.grasps)
    # Penalized backfill never outranks a corridor-clean candidate.
    assert np.max(rescued.scores) == pytest.approx(np.max(starved.scores), abs=1e-5)
