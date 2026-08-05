"""The observed point cloud, in both collision checkers at once.

The deployed planner does not plan against an empty world.  ``online_planner``
builds a :class:`PointCloudCollisionChecker` per request, feeds it
``scene_points`` -- the perceived cloud, already expressed in the PiPER base
frame -- and blocks a state when any point lies within
``capsule.radius + scene_clearance_m + point_radius_m`` of a capsule segment
(13 mm of margin on top of the geometry, from
``configs/piper_collision_capsules.json``).  roboplan's equivalent is
``Scene.addOcTreeGeometry``.  This file plans nothing; it puts the SAME cloud
into both and asks where the two disagree.  Four things came out of it, all
measured on the real model.

1.  **The cloud does not land where you put it.**  ``addOcTreeGeometry(name,
    parent_frame, octree, tform, color)`` reads ``tform`` relative to the parent
    frame's parent JOINT, not relative to the named link.  ``piper_base_link``
    is welded to the Go2-W base by a fixed joint that pinocchio absorbs, so
    passing the cloud with ``parent_frame="piper_base_link"`` and an identity
    ``tform`` -- the obvious call, and the frame the cloud is genuinely in --
    silently places every obstacle 152 mm away from where it was observed
    (measured: the mount offset is exactly (0.1285, 0, 0.0818) m).  Nothing
    raises; the scene simply becomes a plausible lie.  Every arm link is the
    child of a revolute joint whose frame IS the link frame, so the bug is
    invisible on any frame except a fixed one.  ``_base_placement`` is the fix.

2.  **The 13 mm perception margin is not reproduced, and how much of it
    survives is an accident of grid alignment.**  The octree enforces geometric
    intersection only.  Sampling ~450 collision-free configurations across the
    contact boundary of a table + object cloud:

        table at z = -0.0615 m (surface 1.5 mm below a voxel boundary):
            75 of 452 configurations blocked by the deployed checker are
            reported COLLISION-FREE by roboplan; their true clearances span
            2.2 mm .. 12.5 mm -- i.e. the whole margin is given away.
        table at z = -0.0800 m (surface exactly on a voxel boundary):
            0 of 436 -- the occupied voxel column happens to stand 20 mm proud
            of the real surface, so roboplan is *more* conservative there, and
            21 configurations are blocked that the deployed checker clears.

    A real table height is not a multiple of the leaf size, so the first row is
    the case to plan for.  Whoever wires this must add the margin back (inflate
    the capsule bodies in ``roboplan_scene``, or dilate the cloud); voxelisation
    is not a substitute for it.

3.  **What roboplan never does is miss a real penetration.**  An octomap leaf
    always CONTAINS the point it was built from, so the occupied set is a
    superset of the cloud: over ~1750 configurations at four table heights,
    there was not one case of a capsule overlapping a raw point that roboplan
    called free.  The disagreement is one-sided and bounded on both ends --
    permissive only inside the 13 mm margin, conservative only out to one voxel
    diagonal.  That is what the two bound assertions below pin.

4.  **``check_scene: false`` does not survive the trip.**  Six capsules
    (chassis, NUC, head, Mid-360, bracket) are deliberately never tested
    against the cloud -- they are the robot's own body and the cloud contains
    the surface it stands on.  roboplan pairs an added geometry with every
    body in the model, so those six become enforced against the perceived
    scene the moment the octree is added.  ``setCollisions(body, name, False)``
    turns them back off, is the only mitigation that works, and does NOT
    survive the geometry it was made against -- the six calls belong to every
    per-frame add, not to bringup.  A cloud that
    only holds a table in front of the robot never reaches them -- dropping
    the mitigation changes not one verdict in the comparison below -- so this
    is pinned by its own test: it bites on a cloud that extends back over the
    platform, which the multi-frame scene accumulator can produce and one
    frame of a forward-looking D435 cannot.

Cost, measured here and printed by the last test: octree construction is linear
at ~0.18 us per point (0.25 ms for 1k, 4.5 ms for the 26k cloud below, 26 ms
for a 57k one -- the size of the recorded lab cloud in
``tests/data/place_lab_crate_scene.npz``, whose points are 1.6 mm apart);
``addOcTreeGeometry`` 0.9-1.5 ms and near flat in the point count; a collision
check with the arm over the cloud 57 us against 15 us with no cloud present.
For scale, the same Scene with URDF mesh geometry instead of capsules cost
2.8 ms per check.

Needs roboplan, so it runs in the 4090 container only -- except the first test,
which pins the oracle every other assertion is written in terms of and runs
anywhere.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from z_manip.collision.pointcloud import (
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
)
from z_manip.kinematics.chain import KinematicChain

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
MODEL = ROOT / "configs/piper_collision_capsules.json"
HOME = ROOT / "configs/piper_home.json"
BASE_LINK = "piper_base_link"
TIP_LINK = "piper_gripper_base"
# configs/go2w_piper.json robot.acceleration_limits; scene_handle validates the
# count against the group, nothing here depends on the values.
ACCELERATION_LIMITS = (1.5, 1.5, 1.5, 2.0, 2.0, 2.0)

pytestmark = pytest.mark.skipif(not URDF.exists(), reason="robot URDF not checked out")

# Octree leaf size.  The scene accumulator already voxel-downsamples the scene
# cloud at 0.01 m (scene_accumulator.py:54), so anything finer buys resolution
# the cloud does not carry; 0.02 m keeps the leaf count (and the 4.5 ms build
# for a 26k cloud) inside one perception frame.
RESOLUTION_M = 0.02
# An octomap leaf contains the point it was built from but is not centred on it,
# so a capsule can be blocked by a voxel whose generating point is up to one
# full leaf diagonal away.  The 1 mm is the capsule -> rigid-body inflation
# roboplan_scene adds, which test_roboplan_scene pins below 1e-3 m.
VOXEL_BOUND_M = RESOLUTION_M * np.sqrt(3.0) + 1.0e-3
STAMP_S = 1000.0
# Where the synthetic table sits, in the PiPER base frame.  -0.0615 puts the
# surface 1.5 mm below a voxel boundary (adverse: voxelisation adds almost
# nothing); -0.08 puts it exactly on one (aligned: voxelisation adds a full
# 20 mm).  The recorded lab cloud in tests/data/place_lab_crate_scene.npz has
# its support plane at z = -0.09..-0.08, so both are realistic heights.
TABLE_HEIGHTS_M = (-0.0615, -0.0800)


@lru_cache(maxsize=1)
def _model() -> RobotCollisionModel:
    return RobotCollisionModel.from_mapping(json.loads(MODEL.read_text()))


@lru_cache(maxsize=1)
def _chain() -> KinematicChain:
    return KinematicChain.from_urdf(URDF, BASE_LINK, TIP_LINK)


@lru_cache(maxsize=1)
def _home() -> np.ndarray:
    return np.asarray(json.loads(HOME.read_text())["joint_radians"], dtype=float)


def _margin_m() -> float:
    model = _model()
    return model.scene_clearance_m + model.point_radius_m


def _cloud(table_z_m: float, spacing_m: float = 0.004) -> np.ndarray:
    """A table plane plus a box object on it, in the PiPER base frame.

    Geometry taken from the recorded lab scene: the support plane spans
    x in [0.30, 0.85] (work_pose.corridor is [0.48, 0.78]) and the object sits
    at the preferred target x = 0.56.  4 mm spacing is coarser than the real
    stride-2 D435 cloud (1.6 mm nearest-neighbour) and far finer than either
    checker's margin, so the lattice itself never decides a verdict.
    """

    xs = np.arange(0.30, 0.8501, spacing_m)
    ys = np.arange(-0.35, 0.3501, spacing_m)
    xx, yy = np.meshgrid(xs, ys)
    table = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, table_z_m)))
    bx = np.arange(0.50, 0.5601, spacing_m)
    by = np.arange(-0.03, 0.0301, spacing_m)
    bz = np.arange(table_z_m, 0.0401, spacing_m)
    gx, gy, gz = (axis.ravel() for axis in np.meshgrid(bx, by, bz))
    # Only the object's shell is observable, and a solid interior would hide
    # penetration behind the surface points that caused it.
    shell = (
        (np.abs(gx - 0.53) > 0.0299) | (np.abs(gy) > 0.0299) | (gz > 0.0399)
    )
    return np.vstack((table, np.column_stack((gx, gy, gz))[shell]))


def _checker(cloud: np.ndarray) -> PointCloudCollisionChecker:
    """The checker exactly as ``online_planner._plan`` builds it."""

    model = _model()
    checker = PointCloudCollisionChecker(
        chain=_chain(),
        model=model,
        frame_provider=_chain().link_transforms,
        config=PointCloudCollisionConfig(
            clearance=model.scene_clearance_m,
            point_radius=model.point_radius_m,
            scene_noise_tolerance=model.scene_noise_tolerance_m,
            scene_noise_min_support_points=model.scene_noise_min_support_points,
            segment_joint_step=0.035,
        ),
        now_fn=lambda: STAMP_S,
    )
    checker.update_scene(cloud, stamp_s=STAMP_S)
    return checker


def _segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    axis = end - start
    span = float(axis @ axis)
    if span <= 1e-20:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip((points - start) @ axis / span, 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * axis), axis=1)


def _gap(cloud: np.ndarray, tree: cKDTree, q_arm: np.ndarray) -> float:
    """Signed clearance between the cloud and the scene-checked capsules.

    Independent of both checkers: capsule endpoints from ``KinematicChain``,
    exact point-to-segment distance, minus the capsule radius.  Negative means
    a raw cloud point is inside the capsule.  ``+inf`` when no point is within
    80 mm of any capsule, which is past every threshold in this file.
    """

    transforms = _chain().link_transforms(q_arm)
    worst = np.inf
    for capsule in _model().capsules:
        if not capsule.check_scene:
            continue
        start_tf = transforms[capsule.start_frame]
        end_tf = transforms[capsule.end_frame]
        start = start_tf[:3, :3] @ np.asarray(capsule.start_offset, float) + start_tf[:3, 3]
        end = end_tf[:3, :3] @ np.asarray(capsule.end_offset, float) + end_tf[:3, 3]
        reach = 0.5 * float(np.linalg.norm(end - start)) + capsule.radius + 0.08
        near = tree.query_ball_point(0.5 * (start + end), reach)
        if near:
            distances = _segment_distances(cloud[np.asarray(near, dtype=int)], start, end)
            worst = min(worst, float(distances.min()) - capsule.radius)
    return worst


@lru_cache(maxsize=len(TABLE_HEIGHTS_M))
def _scene_sample(table_z_m: float) -> tuple[np.ndarray, cKDTree, tuple[np.ndarray, ...]]:
    """Cloud, its KD-tree, and configurations concentrated on the contact boundary.

    Uniform sampling puts ~1% of configurations anywhere near the table, which
    is precisely the band the two checkers can differ in, so each ray from home
    to a penetrating configuration is bisected onto the contact surface and
    sampled around it.  The uniform tail is kept so the comparison also covers
    configurations that are nowhere near the cloud.
    """

    chain, home = _chain(), _home()
    cloud = _cloud(table_z_m)
    tree = cKDTree(cloud)
    rng = np.random.default_rng(20260803)

    rays, attempts = [], 0
    while len(rays) < 40 and attempts < 6000:
        attempts += 1
        target = rng.uniform(chain.lower_limits, chain.upper_limits)
        tip = chain.forward(target)[:3, 3]
        if not (0.30 < tip[0] < 0.80 and abs(tip[1]) < 0.30 and -0.10 < tip[2] < 0.05):
            continue
        if _gap(cloud, tree, target) <= 0.0:
            rays.append(target)

    configs = [home]
    for target in rays:
        low, high = 0.0, 1.0
        for _ in range(14):
            middle = 0.5 * (low + high)
            if _gap(cloud, tree, home + middle * (target - home)) <= 0.0:
                high = middle
            else:
                low = middle
        contact = 0.5 * (low + high)
        for offset in (-0.06, -0.04, -0.025, -0.015, -0.008, -0.004, 0.0, 0.004, 0.008, 0.015, 0.03):
            step = contact + offset
            if 0.0 <= step <= 1.0:
                configs.append(home + step * (target - home))
    configs.extend(rng.uniform(chain.lower_limits, chain.upper_limits) for _ in range(200))
    return cloud, tree, tuple(configs)


def _handle():
    """The real-model Scene, shared with every other roboplan test in the run.

    ``scene_handle`` caches per config, so the ~0.65 s build is paid once.  That
    also means every ``addOcTreeGeometry`` here MUTATES a Scene other modules
    will query -- hence ``_octree`` removing in a ``finally``.
    """

    from z_manip.planning.roboplan_runtime import RoboplanConfig, scene_handle

    class _RobotConfig:
        urdf_path = URDF
        base_link = BASE_LINK
        tip_link = TIP_LINK
        acceleration_limits = ACCELERATION_LIMITS

    return scene_handle(RoboplanConfig(enabled=True), _RobotConfig())


def _base_placement(scene, handle) -> np.ndarray:
    """Where the arm base sits in the frame ``addOcTreeGeometry`` measures from.

    See finding 1 in the module docstring: this is NOT identity, and leaving it
    out is silent.
    """

    return np.asarray(scene.forwardKinematics(handle.seed, BASE_LINK), dtype=float)


def _boxes(points: np.ndarray, resolution: float = RESOLUTION_M) -> np.ndarray:
    """``roboplan.core.OcTree`` boxes: (x, y, z, size, occupancy, threshold).

    The ``size`` column is ignored -- measured, a leaf comes out at the
    constructor's ``resolution`` whether this says 1 mm or 200 mm -- but coal's
    ``OcTree::toBoxes`` writes it, so it is filled in honestly.  Built as one
    (N, 6) array rather than a list of 6-vectors: same result, half the time
    (26 ms vs 50 ms for 57k points).
    """

    points = np.atleast_2d(np.asarray(points, dtype=float))
    boxes = np.empty((len(points), 6), dtype=float)
    boxes[:, :3] = points
    boxes[:, 3] = resolution
    boxes[:, 4] = 1.0  # occupancy
    boxes[:, 5] = 0.5  # occupancy threshold
    return boxes


@contextmanager
def _octree(scene, tform, points, *, name="test_scene_cloud", resolution=RESOLUTION_M, disable=()):
    """Add the cloud as an octree for the duration of a block, then remove it."""

    import roboplan.core as rc

    scene.addOcTreeGeometry(
        name,
        BASE_LINK,
        rc.OcTree(_boxes(points, resolution), resolution),
        tform,
        np.array([0.5, 0.5, 0.5, 0.4]),
    )
    try:
        for body in disable:
            scene.setCollisions(body, name, False)
        yield name
    finally:
        scene.removeGeometry(name)


def _unchecked_bodies() -> tuple[str, ...]:
    """The capsule bodies ``check_scene: false`` excludes from cloud checks."""

    return tuple(f"cap_{c.name}" for c in _model().capsules if not c.check_scene)


def test_the_deployed_checker_blocks_exactly_at_the_capsule_margin():
    """Licence for the oracle every roboplan assertion below is written against.

    ``_gap`` is not the checker's own arithmetic; if the two ever part company
    the disagreement numbers in this file describe nothing.  Runs without
    roboplan, so a host suite still exercises it.
    """

    margin = _margin_m()
    cloud, tree, configs = _scene_sample(TABLE_HEIGHTS_M[0])
    checker = _checker(cloud)

    scene_verdicts = blocked = 0
    for q_arm in configs:
        result = checker.check_state(q_arm)
        # The checker tests self-collision first and applies ``clearance`` to
        # those pairs too, so a near-self-collision answers before the scene is
        # ever consulted.  Those configurations carry no scene verdict.
        if result.kind not in (None, "scene"):
            continue
        scene_verdicts += 1
        blocked += not result.valid
        gap = _gap(cloud, tree, q_arm)
        assert result.valid == (gap > margin), (
            f"deployed checker says valid={result.valid} at a clearance of "
            f"{gap * 1e3:.2f} mm, margin is {margin * 1e3:.1f} mm"
        )
    assert scene_verdicts > 200, scene_verdicts
    # Both verdicts, in quantity: an oracle agreed with on an all-clear sample
    # has been agreed with about nothing.
    assert 20 < blocked < scene_verdicts - 20, (blocked, scene_verdicts)


def test_the_cloud_lands_where_the_arm_thinks_it_is():
    """Finding 1: ``tform`` is joint-relative, and getting it wrong is silent.

    A single voxel is dropped on the palm capsule's own centre, with every
    other body's pair switched off so only the palm can answer.  Passing the
    point in the frame the cloud is genuinely in -- ``piper_base_link``, the
    frame named in the very same call -- misses the palm entirely and lands
    152 mm away, where at home it quietly hits the shoulder and mount capsules
    instead.  Nothing raises either way, so only a geometric assertion catches
    it.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")

    handle = _handle()
    home = _home()
    transforms = _chain().link_transforms(home)
    palm = next(c for c in _model().capsules if c.name == "palm")
    start_tf, end_tf = transforms[palm.start_frame], transforms[palm.end_frame]
    centre = 0.5 * (
        (start_tf[:3, :3] @ np.asarray(palm.start_offset, float) + start_tf[:3, 3])
        + (end_tf[:3, :3] @ np.asarray(palm.end_offset, float) + end_tf[:3, 3])
    )
    q_full = handle.expand(home)
    others = tuple(f"cap_{c.name}" for c in _model().capsules if c.name != "palm")

    with handle.exclusive() as scene:
        placement = _base_placement(scene, handle)
        assert not scene.hasCollisions(q_full)
        with _octree(scene, placement, [centre], disable=others):
            assert scene.hasCollisions(q_full), "the cloud never reached the palm"
        with _octree(scene, np.eye(4), [centre], disable=others):
            assert not scene.hasCollisions(q_full), (
                "an identity tform now places the cloud correctly -- roboplan "
                "changed its frame convention, re-read finding 1"
            )
        # The arm base is welded to the Go2-W base, so the offset is a constant
        # of the URDF; a floating base would make it configuration-dependent and
        # every octree placement in this file wrong.
        stretched = handle.expand(np.array([0.4, 0.6, -0.6, 0.3, 0.2, 0.1]))
        assert np.allclose(placement, _base_placement(scene, handle))
        assert np.allclose(
            placement,
            np.asarray(scene.forwardKinematics(stretched, BASE_LINK), dtype=float),
        )

    assert np.allclose(placement[:3, :3], np.eye(3))
    assert float(np.linalg.norm(placement[:3, 3])) == pytest.approx(0.1522, abs=5e-4)


@pytest.mark.parametrize(
    "table_z_m, permissive_expected",
    ((TABLE_HEIGHTS_M[0], True), (TABLE_HEIGHTS_M[1], False)),
    ids=("grid-adverse", "grid-aligned"),
)
def test_octree_and_cloud_checker_disagree_only_inside_the_measured_bounds(
    table_z_m, permissive_expected, capsys,
):
    """Findings 2 and 3, over the contact boundary of a table + object cloud.

    Two bounds hold whatever the grid alignment: roboplan never clears a
    configuration whose capsules overlap a raw cloud point (an octomap leaf
    contains its point), and never blocks one further than a leaf diagonal
    away.  Between them the verdicts differ, one-sided by construction, and the
    permissive side is exactly the 13 mm margin the deployed checker adds and
    the octree does not.

    ``permissive_expected`` pins that the margin really is lost at a realistic
    table height.  If it starts passing with no permissive disagreements,
    somebody added the margin back -- delete the parameter, do not relax it.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")

    margin = _margin_m()
    cloud, tree, configs = _scene_sample(table_z_m)
    checker = _checker(cloud)
    handle = _handle()

    rows = []
    with handle.exclusive() as scene:
        placement = _base_placement(scene, handle)
        # Self-collision would answer for both checkers and mask the scene
        # verdict, so the sample is cut down to configurations the model
        # already accepts with no cloud present.
        free = [q for q in configs if not scene.hasCollisions(handle.expand(q))]
        with _octree(scene, placement, cloud, disable=_unchecked_bodies()):
            for q_arm in free:
                result = checker.check_state(q_arm)
                if result.kind not in (None, "scene"):
                    continue
                rows.append((
                    _gap(cloud, tree, q_arm),
                    bool(scene.hasCollisions(handle.expand(q_arm))),
                    not result.valid,
                ))

    blocked_roboplan = sum(1 for _, rp, _ in rows if rp)
    blocked_cloud = sum(1 for _, _, pc in rows if pc)
    missed = sorted(gap for gap, rp, _ in rows if gap <= 0.0 and not rp)
    phantom = sorted(gap for gap, rp, _ in rows if rp and gap > VOXEL_BOUND_M)
    permissive = sorted(gap for gap, rp, pc in rows if pc and not rp)
    conservative = sorted(gap for gap, rp, pc in rows if rp and not pc)

    with capsys.disabled():
        print(
            f"\n  table z={table_z_m:+.4f} m, {len(rows)} scene verdicts,"
            f" {len(cloud)} points"
            f"\n    blocked: roboplan {blocked_roboplan}, cloud checker {blocked_cloud}"
            f"\n    roboplan clears what the cloud checker blocks: {len(permissive)}"
            + (f" (clearances {permissive[0] * 1e3:.1f}..{permissive[-1] * 1e3:.1f} mm)"
               if permissive else "")
            + f"\n    roboplan blocks what the cloud checker clears: {len(conservative)}"
            + (f" (clearances {conservative[0] * 1e3:.1f}..{conservative[-1] * 1e3:.1f} mm)"
               if conservative else "")
        )

    assert len(rows) > 200, f"only {len(rows)} usable configurations"
    assert 20 < blocked_cloud < len(rows) - 20, "the sample does not straddle the boundary"
    assert 20 < blocked_roboplan < len(rows) - 20, "the sample does not straddle the boundary"
    assert not missed, (
        "roboplan reported COLLISION-FREE at a capsule/point overlap of "
        f"{-missed[0] * 1e3:.2f} mm -- an octree leaf no longer contains the "
        f"point it was built from ({len(missed)} configurations)"
    )
    assert not phantom, (
        f"roboplan blocked {len(phantom)} configurations further than a leaf "
        f"diagonal from the cloud, worst {phantom[-1] * 1e3:.2f} mm > "
        f"{VOXEL_BOUND_M * 1e3:.2f} mm"
    )
    assert not permissive or (0.0 < permissive[0] and permissive[-1] <= margin), (
        f"a permissive disagreement outside the perception margin: "
        f"{permissive[0] * 1e3:.2f}..{permissive[-1] * 1e3:.2f} mm vs "
        f"{margin * 1e3:.1f} mm"
    )
    assert not conservative or conservative[0] > margin, (
        f"a conservative disagreement inside the perception margin: "
        f"{conservative[0] * 1e3:.2f} mm"
    )
    if permissive_expected:
        assert permissive, (
            "the octree now reproduces the deployed 13 mm perception margin at "
            "an unaligned table height; finding 2 no longer holds"
        )


def test_capsules_excluded_from_cloud_checks_are_enforced_until_disabled():
    """Finding 4: ``check_scene: false`` does not cross the seam by itself.

    The chassis capsule is never tested against perception by the deployed
    checker -- the cloud contains the surface the robot stands on, and its own
    body is not an obstacle.  roboplan pairs an added geometry with every body,
    so an unmitigated wiring turns the six platform capsules into permanent
    obstacles and every plan through them fails for no visible reason.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")

    handle = _handle()
    _, _, configs = _scene_sample(TABLE_HEIGHTS_M[0])

    def centre(name: str) -> np.ndarray:
        capsule = next(c for c in _model().capsules if c.name == name)
        # Both these capsules are welded to piper_base_link, so their position
        # does not depend on the arm configuration.
        assert capsule.start_frame == capsule.end_frame == BASE_LINK
        return 0.5 * (
            np.asarray(capsule.start_offset, float) + np.asarray(capsule.end_offset, float)
        )

    chassis = next(c for c in _model().capsules if c.name == "go2w_chassis")
    mount = next(c for c in _model().capsules if c.name == "mount")
    assert not chassis.check_scene and mount.check_scene
    # min_scene_points is 32: one point is not a cloud as far as the checker is
    # concerned, and a one-point cloud fails closed on perception grounds.
    ignored = _checker(np.repeat(centre("go2w_chassis")[None, :], 64, axis=0))
    control = _checker(np.repeat(centre("mount")[None, :], 64, axis=0))

    with handle.exclusive() as scene:
        # home is not usable here: the deployed checker adds its 10 mm
        # clearance to SELF pairs too, and home clears mount-vs-wrist by only
        # 3.8 mm, so it answers "self" before the scene is ever consulted.
        q_arm = next(
            q for q in configs
            if not scene.hasCollisions(handle.expand(q)) and ignored.check_state(q).valid
        )
        q_full = handle.expand(q_arm)
        # The control: a point buried in the mount, which is base-fixed in the
        # same way and differs only in check_scene.  The deployed checker does
        # block on that one, so the verdict above is the flag, not the family.
        assert control.check_state(q_arm).kind == "scene"

        with _octree(scene, _base_placement(scene, handle), [centre("go2w_chassis")]):
            assert scene.hasCollisions(q_full), (
                "roboplan no longer pairs added geometry with base-fixed bodies; "
                "the mitigation below may have become a no-op"
            )
        with _octree(
            scene,
            _base_placement(scene, handle),
            [centre("go2w_chassis")],
            disable=_unchecked_bodies(),
        ):
            assert not scene.hasCollisions(q_full)
        # The override dies with the geometry it was made against: the next
        # perception frame gets fresh pairs, enabled again.  Disabling belongs
        # to every add, not to bringup.
        with _octree(scene, _base_placement(scene, handle), [centre("go2w_chassis")]):
            assert scene.hasCollisions(q_full), (
                "setCollisions now survives removeGeometry -- a wiring that "
                "disables once at bringup would become correct"
            )


def test_a_removed_octree_leaves_no_stale_obstacle():
    """A cloud that outlives its ``removeGeometry`` is a plan through a wall.

    Every perception frame replaces the cloud, so add/remove runs at frame rate
    for the life of the process.  Compared over a batch rather than one pose:
    a leftover leaf only shows up where the arm actually goes.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    import roboplan.core as rc

    handle = _handle()
    cloud, _, configs = _scene_sample(TABLE_HEIGHTS_M[0])
    batch = [handle.expand(q) for q in configs[:120]]

    with handle.exclusive() as scene:
        placement = _base_placement(scene, handle)
        before = [scene.hasCollisions(q) for q in batch]

        with _octree(scene, placement, cloud) as name:
            during = [scene.hasCollisions(q) for q in batch]
            assert sum(during) > sum(before), "the cloud blocked nothing at all"
            # A second geometry under a live name would leave a copy behind on
            # removal; roboplan refuses instead.
            with pytest.raises(RuntimeError, match="already exists"):
                scene.addOcTreeGeometry(
                    name,
                    BASE_LINK,
                    rc.OcTree(_boxes(cloud[:1]), RESOLUTION_M),
                    placement,
                    np.array([0.5, 0.5, 0.5, 0.4]),
                )

            # Moving the cloud out of the workspace must clear every verdict the
            # cloud caused, and moving it back must restore them exactly.
            away = placement.copy()
            away[:3, 3] = placement[:3, 3] + np.array([12.0, 0.0, 0.0])
            scene.updateGeometryPlacement(name, BASE_LINK, away)
            assert [scene.hasCollisions(q) for q in batch] == before
            scene.updateGeometryPlacement(name, BASE_LINK, placement)
            assert [scene.hasCollisions(q) for q in batch] == during

        assert [scene.hasCollisions(q) for q in batch] == before, (
            "a removed octree still blocks configurations -- stale perception "
            "geometry is live in the planning scene"
        )
        with pytest.raises(RuntimeError, match="Could not find object"):
            scene.removeGeometry("test_scene_cloud")
        with pytest.raises(RuntimeError, match="Could not find object"):
            scene.updateGeometryPlacement("test_scene_cloud", BASE_LINK, placement)

        # Re-add: the second frame of perception must behave exactly like the
        # first, verdict for verdict.
        with _octree(scene, placement, cloud):
            assert [scene.hasCollisions(q) for q in batch] == during


def test_a_nonpositive_resolution_silently_discards_the_whole_cloud():
    """``OcTree(boxes, 0.0)`` builds an empty tree and reports collision-free.

    No exception, no empty-geometry complaint from ``addOcTreeGeometry``: a
    mis-typed resolution deletes the perceived world and the scene answers
    "clear" for everything.  Whatever wires this must validate the resolution
    itself, the way every other config value in this repo is validated.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    import roboplan.core as rc

    handle = _handle()
    home = _home()
    transforms = _chain().link_transforms(home)
    palm = next(c for c in _model().capsules if c.name == "palm")
    start_tf = transforms[palm.start_frame]
    centre = start_tf[:3, :3] @ np.asarray(palm.start_offset, float) + start_tf[:3, 3]
    q_full = handle.expand(home)

    with handle.exclusive() as scene:
        placement = _base_placement(scene, handle)
        with _octree(scene, placement, [centre]):
            assert scene.hasCollisions(q_full)
        for resolution in (0.0, -RESOLUTION_M):
            tree = rc.OcTree(_boxes([centre], RESOLUTION_M), resolution)
            scene.addOcTreeGeometry(
                "bad_resolution", BASE_LINK, tree, placement, np.array([0.5, 0.5, 0.5, 0.4]),
            )
            try:
                assert not scene.hasCollisions(q_full), (
                    f"resolution {resolution} now keeps the cloud -- the "
                    "fail-open documented here is fixed, drop this test"
                )
            finally:
                scene.removeGeometry("bad_resolution")


def test_insertion_cost_and_check_throughput(capsys):
    """What a perception frame costs, and what it costs to keep checking.

    The number that matters is the second one: a Scene built from the URDF
    meshes cost 2.8 ms per collision check, which is why roboplan_scene emits
    capsules instead.  An octree that put the check back into the millisecond
    range would take the capsule Scene's whole reason for existing with it.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    import roboplan.core as rc

    handle = _handle()
    cloud, tree, configs = _scene_sample(TABLE_HEIGHTS_M[0])
    rng = np.random.default_rng(5)

    sizes, build_ms, add_ms = [], [], []
    with handle.exclusive() as scene:
        placement = _base_placement(scene, handle)
        for count in (1000, 4000, 16000, len(cloud)):
            points = cloud[rng.choice(len(cloud), size=count, replace=False)]
            started = time.perf_counter()
            octree = rc.OcTree(_boxes(points), RESOLUTION_M)
            build = time.perf_counter() - started
            started = time.perf_counter()
            scene.addOcTreeGeometry(
                "cost", BASE_LINK, octree, placement, np.array([0.5, 0.5, 0.5, 0.4]),
            )
            add = time.perf_counter() - started
            scene.removeGeometry("cost")
            sizes.append(count)
            build_ms.append(build * 1e3)
            add_ms.append(add * 1e3)

        # Configurations that actually reach the cloud: away from it the
        # broadphase rejects every pair and the octree is free.
        near = [handle.expand(q) for q in configs if _gap(cloud, tree, q) < 0.05][:24]
        assert len(near) > 10, len(near)

        def sweep() -> float:
            started = time.perf_counter()
            for _ in range(40):
                for q_full in near:
                    scene.hasCollisions(q_full)
            return (time.perf_counter() - started) / (40 * len(near))

        bare = sweep()
        with _octree(scene, placement, cloud):
            loaded = sweep()

    with capsys.disabled():
        print(
            "\n  octree build / addOcTreeGeometry:\n"
            + "".join(
                f"    {n:6d} pts  {b:6.2f} ms  {a:5.2f} ms\n"
                for n, b, a in zip(sizes, build_ms, add_ms)
            )
            + f"    collision check near the cloud: {loaded * 1e6:.1f} us"
            f" ({bare * 1e6:.1f} us with no cloud, {loaded / bare:.1f}x)"
        )

    # Measured 26 ms for 57k points; 4x that still leaves the insert inside one
    # 10 Hz perception frame on a slower host.
    assert build_ms[-1] < 0.4e3 * len(cloud) / 57000.0, build_ms[-1]
    assert add_ms[-1] < 50.0, add_ms[-1]
    # Measured 48-68 us against 16 us bare.  The ceiling is the 2.8 ms mesh
    # figure this whole capsule detour exists to avoid.
    assert loaded < 0.5e-3, f"{loaded * 1e6:.1f} us per check with the cloud present"
