import math

import numpy as np
import pytest
from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# The upright prior must be evaluated against real gravity, not against the
# cloud frame's Z.
#
# Every recorded perception session writes frame = camera_color_optical_frame,
# where +Z is the camera boresight.  up_axis defaults to (0, 0, 1) and NO
# production caller ever passes it, so _snap_axes_to_gravity was snapping the
# boresight to "vertical".  A flat patch seen obliquely then reports its
# projected spread as a third object dimension -- and because the scoring
# explicitly prefers the NARROWEST graspable face, that phantom axis wins.


def _camera_from_base(pitch_rad: float) -> np.ndarray:
    """A wrist camera looking forward and down by ``pitch_rad``."""

    cos, sin = np.cos(pitch_rad), np.sin(pitch_rad)
    optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    transform = np.eye(4)
    transform[:3, :3] = optical @ pitch
    transform[:3, 3] = (0.05, 0.0, 0.30)
    return transform


def test_gravity_comes_from_the_cloud_frame_not_from_a_fixed_axis():
    source = AntipodalGraspSource()
    context = SimpleNamespace(t_target_src=_camera_from_base(0.61))

    up = source._gravity_in_cloud_frame(context)

    assert np.isclose(np.linalg.norm(up), 1.0)
    # Base +Z carried into the camera frame is nowhere near the boresight.
    angle_deg = np.degrees(np.arccos(abs(float(up[2]))))
    assert angle_deg > 45.0, (
        f"true up is only {angle_deg:.1f} deg from the boresight; the fixture no "
        "longer represents a wrist camera and cannot catch the defect"
    )


def test_an_identity_transform_keeps_the_configured_axis():
    """A cloud already in the base frame must be unaffected."""

    source = AntipodalGraspSource()
    up = source._gravity_in_cloud_frame(SimpleNamespace(t_target_src=np.eye(4)))
    np.testing.assert_allclose(up, source.up_axis)


@pytest.mark.parametrize(
    "transform",
    [None, "not a transform", np.eye(3), np.full((4, 4), np.nan)],
)
def test_an_unusable_transform_falls_back_rather_than_snapping_to_garbage(transform):
    source = AntipodalGraspSource()
    up = source._gravity_in_cloud_frame(SimpleNamespace(t_target_src=transform))
    np.testing.assert_allclose(up, source.up_axis)


def test_a_scaled_transform_is_refused():
    """A non-orthonormal rotation cannot be inverted into a meaningful axis."""

    scaled = np.eye(4)
    scaled[:3, :3] = np.diag((1.05, 1.05, 1.05))
    source = AntipodalGraspSource()
    up = source._gravity_in_cloud_frame(SimpleNamespace(t_target_src=scaled))
    np.testing.assert_allclose(up, source.up_axis)


def _bench(z=0.0, half=0.30, step=0.006):
    """A flat support patch, as the scene cloud delivers it."""

    grid = np.arange(-half, half + step, step)
    return np.array([[0.48 + u, v, z] for u in grid for v in grid])


def test_top_only_view_anchors_the_grasp_to_the_support_not_the_lid():
    # A steep wrist view of a 0.12 m can on a bench returns its top face and
    # almost nothing else, so the cloud's own mid-plane IS the lid.  The
    # support is still visible all around it and fixes the height.
    top_z, radius = 0.12, 0.033
    angles = np.linspace(0.0, 2.0 * np.pi, 60, endpoint=False)
    lid = np.array([
        [0.48 + r * np.cos(a), r * np.sin(a), top_z - 0.001 * (r / radius)]
        for r in np.linspace(0.0, radius, 12) for a in angles
    ])
    source = AntipodalGraspSource(max_candidates=8)

    blind = source.generate(_context(lid))
    assert float(blind.grasps[0, 2, 3]) > top_z - 0.005          # on the lid

    anchored = source.generate(_context(lid, scene_points=_bench()))
    assert abs(float(anchored.grasps[0, 2, 3]) - 0.5 * top_z) < 0.01


def test_a_round_object_wider_than_the_aperture_refuses_instead_of_guessing():
    # The OBB fallback would re-measure the visible arc's DEPTH as a closing
    # width and centre the grasp on the front surface, so a genuinely too-wide
    # round object must fail loudly.
    wide = _upright_half_cylinder(radius=0.055, z0=0.05, height=0.20)
    source = AntipodalGraspSource(max_candidates=8)
    with pytest.raises(GraspGenerationError, match="outside the usable aperture"):
        source.generate(_context(wide, scene_points=_bench()))


def _look_down_cylinder(radius=0.033, height=0.120, arc_deg=137.0,
                        wall_depth_m=0.010, lid_rings=14, n_angle=48, n_wall=3):
    """One steep wrist view of an upright can standing on the bench.

    The lid faces the camera and samples densely; the wall is foreshortened to
    a sliver near the rim, which is what the D435 returns once the base drives
    close and the arm looks down on the bench (the can's observed vertical
    extent collapses from 0.117 m at eye level to 0.0037 m at 70 degrees).
    ``arc_deg`` defaults to the visible side-wall arc measured on this can at
    low elevation, 137 degrees, which occupies 14 of the 36 ten-degree bins.
    """

    lid_angles = np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False)
    lid = np.array([
        [0.48 + r * np.cos(a), r * np.sin(a), height]
        for r in np.linspace(0.0, radius, lid_rings) for a in lid_angles
    ])
    half = np.radians(arc_deg) / 2.0
    wall_angles = np.linspace(-np.pi - half, -np.pi + half, n_angle)
    wall = np.array([
        [0.48 + radius * np.cos(a), radius * np.sin(a), z]
        for a in wall_angles
        for z in np.linspace(height - wall_depth_m, height - 0.001, n_wall)
    ])
    return np.vstack((lid, wall))


def _look_down_carton(half=(0.0225, 0.0325, 0.050), wall_depth_m=0.099, n=11, n_wall=8):
    """The same view of an upright juice carton: top face plus the two near walls."""

    hx, hy, hz = half
    us = np.linspace(-1.0, 1.0, n)
    zs = np.linspace(2.0 * hz - wall_depth_m, 2.0 * hz - 0.001, n_wall)
    return np.asarray(
        [(0.48 + u * hx, v * hy, 2.0 * hz) for u in us for v in us]
        + [(0.48 + u * hx, -hy, z) for u in us for z in zs]
        + [(0.48 - hx, v * hy, z) for v in us for z in zs],
        dtype=np.float64,
    )


def test_top_down_view_of_a_cylinder_is_grasped_on_its_side_not_its_lid():
    # The complaint this whole anchor exists for: after the servo the robot is
    # tall, the reconstructed surface is nearly all top face, and the grasp
    # lands on the lid -- useless on a cup, on the cap of a bottle, and it
    # drives the fingers down toward the bench.
    can = _look_down_cylinder()
    source = AntipodalGraspSource(max_candidates=8)

    blind = source.generate(_context(can))
    # 0.115 m: the mid-plane of the observed 0.110-0.120 m sliver, 55 mm above
    # the can's true 0.060 m mid-height and 5 mm under the rim.
    assert float(blind.grasps[0, 2, 3]) > 0.110

    anchored = source.generate(_context(can, scene_points=_bench()))
    pose = anchored.grasps[0]
    assert abs(float(pose[2, 3]) - 0.060) < 0.005
    # A side grasp is BOTH: the jaw closes across a horizontal diameter and the
    # tool drives in horizontally.  A vertical closing axis would be a rim
    # pinch and a vertical approach a lid press, at the same TCP height.
    assert _angle_to_vertical_deg(pose[:3, 0]) > 80.0
    assert _angle_to_vertical_deg(pose[:3, 2]) > 80.0
    assert 0.012 <= float(anchored.widths[0]) <= source.graspable_extent_m


def test_a_partial_arc_still_recovers_the_true_diameter():
    # A single camera sees at most 180 degrees of a cylinder, and grazing
    # dropout takes that to 137 degrees on this can -- 14 of the 36 bins.  The
    # old 165-degree span asked for 16, so the arc gate was the sole binding
    # rejection and every cylinder fell through to the OBB path, to be grasped
    # across the visible chord (0.062 m) instead of the diameter (0.066 m).
    # 110 degrees is 11 bins, 3 below the measured arc.  Boxes stay out on the
    # roundness residual rather than on this gate (see the carton test).
    can = _look_down_cylinder(wall_depth_m=0.118, n_wall=6, n_angle=56)
    source = AntipodalGraspSource(max_candidates=8)
    obb = source._fit_obb(can, source.up_axis)
    anchor = source._support_anchored_height(can, _bench(), source.up_axis)

    section = source._round_cross_section(can, obb, source.up_axis, anchor_h=anchor)
    assert section is not None
    assert section.diameter == pytest.approx(0.066, abs=0.002)

    strict = AntipodalGraspSource(max_candidates=8, symmetry_span_deg=165.0)
    assert strict._round_cross_section(can, obb, strict.up_axis, anchor_h=anchor) is None


@pytest.mark.parametrize("wall_depth_m", [0.099, 0.030, 0.008])
def test_a_carton_is_not_completed_into_a_cylinder(wall_depth_m):
    # Lowering the arc gate must not let a box become a cylinder -- but the arc
    # gate was never what kept boxes out.  On the full-height view (the only
    # one here whose mid-height slab has points at all) this carton's
    # near-circle inliers occupy exactly 11 bins, i.e. it MEETS the new
    # 11-bin threshold; what rejects it is the roundness residual, rms/radius
    # 0.138 against the 0.08 gate.  On the two steep views the mid-height slab
    # is empty and the round path never starts.
    carton = _look_down_carton(wall_depth_m=wall_depth_m)
    source = AntipodalGraspSource(max_candidates=8)
    obb = source._fit_obb(carton, source.up_axis)
    anchor = source._support_anchored_height(carton, _bench(), source.up_axis)

    assert source._round_cross_section(carton, obb, source.up_axis) is None
    assert source._round_cross_section(
        carton, obb, source.up_axis, anchor_h=anchor
    ) is None

    # With the arc gate removed entirely (one bin) the carton is STILL not
    # round, which is what makes 165 -> 110 safe: the box rejection never
    # rested on this threshold.
    open_gate = AntipodalGraspSource(max_candidates=8, symmetry_span_deg=10.0)
    assert open_gate.symmetry_span_cos_bins == 1
    assert open_gate._round_cross_section(
        carton, obb, open_gate.up_axis, anchor_h=anchor
    ) is None


def _rounded_carton(fillet_m, elevation_deg=30.0, half=(0.0225, 0.0325, 0.050),
                    noise_m=0.0008, seed=0):
    """ONE wrist view of an upright carton with rounded vertical edges.

    ``_look_down_carton`` is a perfectly sharp box, and the sharp case is the
    easy one -- its corners are what the roundness residual keys on.  Real juice
    boxes are filleted, and the misfit only appears in a genuine single view
    with real depth noise, so this reproduces both.
    """

    hx, hy, hz = half
    ax, ay = hx - fillet_m, hy - fillet_m
    profile = []
    for cx, cy, a0 in ((ax, ay, 0.0), (-ax, ay, 0.5 * np.pi),
                       (-ax, -ay, np.pi), (ax, -ay, 1.5 * np.pi)):
        for a in np.linspace(a0, a0 + 0.5 * np.pi, 50):
            profile.append((cx + fillet_m * np.cos(a), cy + fillet_m * np.sin(a)))
    for t in np.linspace(-ax, ax, 100):
        profile += [(t, hy), (t, -hy)]
    for t in np.linspace(-ay, ay, 100):
        profile += [(hx, t), (-hx, t)]

    rng = np.random.default_rng(seed)
    base = np.array([0.48, 0.0, 0.0])
    elevation = math.radians(elevation_deg)
    camera = base + np.array([0.0, 0.0, hz]) + 0.55 * np.array(
        [-math.cos(elevation), 0.0, math.sin(elevation)]
    )
    points, normals = [], []
    for z in np.linspace(0.002, 2.0 * hz, 70):                      # walls
        for (x, y) in profile:
            outward = np.array([x, y, 0.0])
            points.append(base + np.array([x, y, z]))
            normals.append(outward / max(float(np.linalg.norm(outward)), 1e-9))
    for t in np.linspace(0.0, 1.0, 26):                             # top face
        for (x, y) in profile:
            points.append(base + np.array([x * t, y * t, 2.0 * hz]))
            normals.append(np.array([0.0, 0.0, 1.0]))
    points, normals = np.asarray(points), np.asarray(normals)
    ray = camera - points
    ray /= np.linalg.norm(ray, axis=1)[:, None]
    keep = np.einsum("ij,ij->i", normals, ray) > 0.15
    return points[keep] + rng.normal(0.0, noise_m, (int(keep.sum()), 3))


def test_a_rounded_corner_carton_is_not_completed_into_a_cylinder():
    # test_a_carton_is_not_completed_into_a_cylinder proves it for a SHARP box
    # and generalises to "boxes stay out on the roundness residual".  That
    # generalisation is false: a filleted carton in one noisy view passes both
    # the residual AND the arc-bin gate -- its corners scatter inliers right
    # around the circle -- and gets completed into an ~87 mm cylinder.  Since
    # the caller now REFUSES a too-wide round object rather than falling
    # through to the OBB faces, that misfit is a total grasp failure on the
    # only objects on this bench the jaw can actually close around.
    #
    # What catches it is that the fitted circle overshoots the observed
    # footprint (1.31x measured) where a true cylinder does not (1.03-1.06x).
    # 8 mm of fillet on a 45x65 mm carton.  A 4 mm fillet is a milder misfit
    # that stays inside the band, so it is not claimed here.
    carton = _rounded_carton(0.008)
    source = AntipodalGraspSource(max_candidates=8)
    obb = source._fit_obb(carton, source.up_axis)
    anchor = source._support_anchored_height(carton, _bench(), source.up_axis)

    assert source._round_cross_section(
        carton, obb, source.up_axis, anchor_h=anchor
    ) is None

    # Guard the guard: with the inherited 1.4 band it IS mistaken for a
    # cylinder, so this test fails for the reason it claims to.
    loose = AntipodalGraspSource(max_candidates=8, round_diameter_band_max=1.4)
    section = loose._round_cross_section(
        carton, obb, loose.up_axis, anchor_h=anchor
    )
    assert section is not None
    assert section.diameter > 1.2 * max(
        obb.full_extent[i] for i in range(3) if i != obb.vertical_index
    )

    # ...and with the gate in place the carton still gets its real 45 mm face.
    grasps = source.generate(_context(carton))
    assert grasps.widths[int(np.argmax(grasps.scores))] == pytest.approx(0.045, abs=0.006)


def test_a_carton_seen_from_above_is_also_anchored_to_the_bench():
    # Boxes are dragged onto their top face by the same mechanism as cans, so
    # the anchor must fix them too -- and must not move the closing axis.
    carton = _look_down_carton(wall_depth_m=0.015, n_wall=3)
    source = AntipodalGraspSource(max_candidates=8)

    blind = source.generate(_context(carton))
    assert abs(float(blind.grasps[0, 2, 3]) - 0.050) > 0.030

    anchored = source.generate(_context(carton, scene_points=_bench()))
    assert abs(float(anchored.grasps[0, 2, 3]) - 0.050) < 0.005
    np.testing.assert_allclose(
        anchored.grasps[0, :3, 0], blind.grasps[0, :3, 0], atol=1e-9
    )


def test_a_floor_mode_under_a_bench_edge_is_refused_as_a_support():
    # The support is the densest horizontal layer near the object, and for an
    # object at the bench's front edge that layer can be the FLOOR: 0.489 m
    # below the bench top on this rig.  Anchoring to it puts the TCP at
    # -0.185 m, i.e. 185 mm inside the flight case -- worse than the lid grasp.
    can = _look_down_cylinder()
    source = AntipodalGraspSource(max_candidates=8)
    floor = _bench(z=-0.489353)

    assert source._support_anchored_height(can, floor, source.up_axis) is None
    # Only the band declines it; without one the anchor commands -0.185 m.
    unguarded = AntipodalGraspSource(
        max_candidates=8, support_max_object_height_m=10.0
    )
    assert unguarded._support_anchored_height(
        can, floor, unguarded.up_axis
    ) == pytest.approx(-0.1847, abs=0.001)
    # A real bench under the same object is still accepted.
    assert source._support_anchored_height(
        can, _bench(), source.up_axis
    ) == pytest.approx(0.060, abs=0.001)
