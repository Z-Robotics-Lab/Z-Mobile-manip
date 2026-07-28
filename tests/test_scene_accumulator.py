"""Offline tests for multi-view scene-cloud accumulation.

Every case here is a synthetic two-viewpoint scene built to isolate one property
of the merge. That is deliberate: the accumulator's whole job is to decide what
one viewpoint may say about geometry another viewpoint measured, and a recorded
bag cannot pin that down because the ground-truth surface is unknown. The
numbers that DO come from hardware -- the cross-view registration disagreement
this module is tuned against -- enter here as the noise magnitudes swept in
``test_thickening_stays_bounded_under_registration_noise`` and
``test_systematic_registration_bias_degrades_to_single_frame``.
"""

from __future__ import annotations

import numpy as np
import pytest

from z_manip.perception.scene_accumulator import (
    AccumulationReport,
    SceneAccumulatorConfig,
    SceneCloudAccumulator,
    voxel_downsample,
)


# Camera optical convention: +Z forward, +X right, +Y down. This is the rotation
# that maps that frame onto the arm base frame when the wrist yaw is zero.
_CAMERA_IN_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def _pose(yaw_rad: float, origin: np.ndarray) -> np.ndarray:
    """A base_from_camera transform for a camera yawed about the base Z axis."""

    cos, sin = np.cos(yaw_rad), np.sin(yaw_rad)
    yaw = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = yaw @ _CAMERA_IN_BASE
    transform[:3, 3] = origin
    return transform


def _visible(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Mask of points inside a D435-like 69 x 42 deg frustum from ``transform``."""

    rotation, origin = transform[:3, :3], transform[:3, 3]
    camera = (points - origin) @ rotation
    depth = camera[:, 2]
    ahead = depth > 1e-3
    tangent_x = np.zeros(len(points))
    tangent_y = np.zeros(len(points))
    tangent_x[ahead] = camera[ahead, 0] / depth[ahead]
    tangent_y[ahead] = camera[ahead, 1] / depth[ahead]
    return ahead & (np.abs(tangent_x) < 0.687) & (np.abs(tangent_y) < 0.384)


def _wall(rng: np.random.Generator, count: int = 2400) -> np.ndarray:
    """A flat slab at x = 0.55 m, fully inside both viewpoints used below."""

    return np.column_stack(
        [
            np.full(count, 0.55),
            rng.uniform(-0.12, 0.12, count),
            rng.uniform(0.05, 0.35, count),
        ]
    )


_EYE = np.array([0.05, 0.0, 0.25])


def _feed(
    accumulator: SceneCloudAccumulator,
    scene: np.ndarray,
    poses: list[np.ndarray],
    *,
    offsets: list[np.ndarray] | None = None,
) -> list[AccumulationReport]:
    reports = []
    for index, transform in enumerate(poses):
        frame = scene[_visible(scene, transform)].copy()
        if offsets is not None:
            frame = frame + offsets[index]
        reports.append(
            accumulator.update(
                frame, stamp_s=1.0 + 0.2 * index, base_from_camera=transform
            )
        )
    return reports


def test_update_accepts_a_first_frame():
    """Regression: the class shipped unable to accept ANY frame.

    ``_Frame`` gained ``origin``/``rotation`` fields that ``update`` never
    passed, so every call raised TypeError. Nothing imported the module and it
    had no tests, so the breakage was invisible.
    """

    rng = np.random.default_rng(0)
    accumulator = SceneCloudAccumulator()
    report = accumulator.update(
        _wall(rng), stamp_s=1.0, base_from_camera=_pose(0.0, _EYE)
    )

    assert report.accepted
    assert report.frames == 1
    assert report.total_points > 0
    assert len(accumulator.points) == report.total_points


def test_retains_geometry_the_newest_frame_can_no_longer_see():
    """The coverage bug this module exists to close.

    A wrist camera aimed at the grasp target cannot hold the cabinet side wall
    the elbow sweeps through. Planning on the newest frame alone plans straight
    through it.
    """

    rng = np.random.default_rng(1)
    target = np.column_stack(
        [
            np.full(400, 0.55),
            rng.uniform(-0.06, 0.06, 400),
            rng.uniform(0.10, 0.22, 400),
        ]
    )
    side_wall = np.column_stack(
        [
            rng.uniform(0.35, 0.65, 600),
            np.full(600, -0.25),
            rng.uniform(0.05, 0.45, 600),
        ]
    )
    scene = np.vstack([target, side_wall])
    poses = [_pose(-0.35, _EYE), _pose(0.0, _EYE)]

    accumulator = SceneCloudAccumulator()
    _feed(accumulator, scene, poses)

    def wall_coverage(cloud: np.ndarray) -> int:
        """Occupied 1 cm voxels on the side wall.

        Raw point counts are not comparable here: the merged cloud is voxel
        downsampled and a raw single frame is not, so counting points would
        credit the un-downsampled baseline for duplicates.
        """

        distance = np.linalg.norm(
            cloud[:, None, :] - side_wall[None, :, :], axis=2
        ).min(axis=1)
        on_wall = cloud[distance < 0.02]
        if not len(on_wall):
            return 0
        return len(voxel_downsample(on_wall, SceneAccumulatorConfig().voxel_size_m))

    single_frame = scene[_visible(scene, poses[-1])]
    assert wall_coverage(accumulator.points) > wall_coverage(single_frame)


def test_newest_frame_overrules_a_surface_it_can_still_see():
    """Where the newest frame has an opinion, the retained copy is discarded.

    Averaging or concatenating instead would smear the surface by most of the
    planner's clearance budget and turn reachable grasps into false collisions.
    """

    rng = np.random.default_rng(2)
    wall = _wall(rng)
    accumulator = SceneCloudAccumulator()
    reports = _feed(accumulator, wall, [_pose(-0.20, _EYE), _pose(0.0, _EYE)])

    assert reports[-1].carved_authoritative > 0
    # The slab is one voxel thick in the merged cloud, exactly as one view sees it.
    thickness = np.percentile(accumulator.points[:, 0], 99) - np.percentile(
        accumulator.points[:, 0], 1
    )
    assert thickness <= SceneAccumulatorConfig().voxel_size_m


@pytest.mark.parametrize("p90_mm", [5.0, 10.0, 18.0, 30.0])
def test_thickening_stays_bounded_under_registration_noise(p90_mm):
    """Random cross-view disagreement must not thicken a doubly-seen surface.

    The hardware figure this is tuned against is a p90 of 18 mm over 44
    same-scene pairs. 30 mm is included to show the behaviour past
    ``authoritative_radius_m`` (25 mm) rather than only inside it.
    """

    rng = np.random.default_rng(3)
    wall = _wall(rng)
    # |N(0, s)| in three dimensions has p90 near 2.5 s.
    sigma = (p90_mm / 1000.0) / 2.5
    poses = [_pose(-0.20, _EYE), _pose(0.0, _EYE)]
    accumulator = SceneCloudAccumulator()
    reports = _feed(
        accumulator,
        wall,
        poses,
        offsets=[rng.normal(0.0, sigma, (1, 3)) * 0.0 + rng.normal(0.0, sigma, 3), np.zeros(3)],
    )

    surviving = reports[-1].retained_points
    newest = reports[-1].frame_points
    # A retained copy of an already-visible surface is the failure mode. Some
    # leakage is inevitable once the disagreement approaches the authoritative
    # radius; it must stay a small minority rather than doubling the surface.
    assert surviving < 0.25 * newest


def test_systematic_registration_bias_degrades_to_single_frame():
    """Documents the real limit of the 25 mm authoritative radius.

    Registration error is systematic per viewpoint (hand-eye plus forward
    kinematics), not random per point. Once a whole view is displaced past the
    authoritative radius along the surface normal, the retained cloud and the
    newest frame disagree everywhere they overlap, and the consistency gate
    drops the buffer. That is FAIL-SAFE -- no phantom obstacles -- but it means
    the accumulator quietly stops accumulating and the planner is back to single
    -frame coverage. The reset reason is the only signal, so a caller that does
    not surface it will never know it lost the feature.
    """

    rng = np.random.default_rng(4)
    wall = _wall(rng)
    poses = [_pose(-0.20, _EYE), _pose(0.0, _EYE)]

    within = SceneCloudAccumulator()
    reports = _feed(
        within, wall, poses, offsets=[np.array([0.018, 0.0, 0.0]), np.zeros(3)]
    )
    assert not reports[-1].reset_reason

    beyond = SceneCloudAccumulator()
    reports = _feed(
        beyond, wall, poses, offsets=[np.array([0.030, 0.0, 0.0]), np.zeros(3)]
    )
    assert reports[-1].reset_reason
    assert "disagrees" in reports[-1].reset_reason
    assert beyond.frame_count == 1
    assert reports[-1].retained_points == 0


def test_a_stationary_viewpoint_supersedes_instead_of_filling_the_buffer():
    """Scene clouds arrive at 10-30 Hz; a ring buffer of one viewpoint is useless.

    Without keyframe admission, eight frames of a stationary wrist span under a
    second and carry exactly the coverage of a single frame.
    """

    rng = np.random.default_rng(5)
    side_wall = np.column_stack(
        [
            rng.uniform(0.35, 0.65, 600),
            np.full(600, -0.25),
            rng.uniform(0.05, 0.45, 600),
        ]
    )
    scene = np.vstack([_wall(rng, count=400), side_wall])
    transform = _pose(0.0, _EYE)
    accumulator = SceneCloudAccumulator()

    for index in range(6):
        report = accumulator.update(
            scene[_visible(scene, transform)],
            stamp_s=1.0 + 0.05 * index,
            base_from_camera=transform,
        )
        assert report.accepted
        assert report.superseded_viewpoint == (index > 0)

    assert accumulator.frame_count == 1

    moved = _pose(-0.35, _EYE)
    report = accumulator.update(
        scene[_visible(scene, moved)], stamp_s=1.4, base_from_camera=moved
    )
    assert not report.superseded_viewpoint
    assert accumulator.frame_count == 2


def test_freshness_tracks_the_newest_contribution():
    """Downstream staleness gates must not age out on old retained geometry.

    The scene needs geometry the second view cannot see, or the carve deletes
    the first frame outright and there is no retained contribution to age.
    """

    rng = np.random.default_rng(6)
    side_wall = np.column_stack(
        [
            rng.uniform(0.35, 0.65, 600),
            np.full(600, -0.25),
            rng.uniform(0.05, 0.45, 600),
        ]
    )
    scene = np.vstack([_wall(rng, count=400), side_wall])
    accumulator = SceneCloudAccumulator()
    _feed(accumulator, scene, [_pose(-0.35, _EYE), _pose(0.0, _EYE)])

    assert accumulator.newest_stamp_s == pytest.approx(1.2)
    assert accumulator.oldest_stamp_s == pytest.approx(1.0)


def test_reported_base_motion_and_backwards_stamps_drop_the_buffer():
    """No usable base odometry exists, so callers that DO know must be obeyed."""

    rng = np.random.default_rng(7)
    wall = _wall(rng)
    transform = _pose(0.0, _EYE)
    frame = wall[_visible(wall, transform)]

    accumulator = SceneCloudAccumulator()
    accumulator.update(frame, stamp_s=1.0, base_from_camera=_pose(-0.30, _EYE))
    report = accumulator.update(
        frame, stamp_s=1.2, base_from_camera=transform, base_moved=True
    )
    assert report.reset_reason == "caller reported base motion"
    assert accumulator.frame_count == 1

    accumulator = SceneCloudAccumulator()
    accumulator.update(frame, stamp_s=5.0, base_from_camera=_pose(-0.30, _EYE))
    report = accumulator.update(frame, stamp_s=1.0, base_from_camera=transform)
    assert report.reset_reason == "scene stamp went backwards"
    assert accumulator.frame_count == 1


def test_a_thin_frame_is_a_dropout_not_evidence_of_a_changed_world():
    rng = np.random.default_rng(8)
    wall = _wall(rng)
    transform = _pose(0.0, _EYE)
    accumulator = SceneCloudAccumulator()
    accumulator.update(
        wall[_visible(wall, transform)], stamp_s=1.0, base_from_camera=transform
    )

    report = accumulator.update(
        np.zeros((4, 3)), stamp_s=1.1, base_from_camera=transform
    )
    assert not report.accepted
    assert "usable points" in report.reason
    assert accumulator.frame_count == 1


def test_the_newest_frame_survives_the_point_budget():
    """Evicting it would leave the planner worse off than the single-frame path."""

    rng = np.random.default_rng(9)
    config = SceneAccumulatorConfig(max_points=50, max_frames=8)
    accumulator = SceneCloudAccumulator(config)
    wall = _wall(rng, count=600)

    for index, yaw in enumerate(np.linspace(-0.6, 0.6, 5)):
        transform = _pose(float(yaw), _EYE)
        accumulator.update(
            wall[_visible(wall, transform)],
            stamp_s=1.0 + 0.2 * index,
            base_from_camera=transform,
        )

    assert accumulator.frame_count >= 1
    assert accumulator.newest_stamp_s == pytest.approx(1.8)
    assert len(accumulator.points) > 0


def test_frames_expire_by_age():
    rng = np.random.default_rng(10)
    config = SceneAccumulatorConfig(max_frame_age_s=1.0)
    accumulator = SceneCloudAccumulator(config)
    wall = _wall(rng)

    accumulator.update(
        wall[_visible(wall, _pose(-0.30, _EYE))],
        stamp_s=1.0,
        base_from_camera=_pose(-0.30, _EYE),
    )
    accumulator.update(
        wall[_visible(wall, _pose(0.30, _EYE))],
        stamp_s=9.0,
        base_from_camera=_pose(0.30, _EYE),
    )
    assert accumulator.frame_count == 1


def test_voxel_downsample_prefers_the_higher_priority_point():
    points = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    kept = voxel_downsample(points, 0.01, priority=np.array([0.0, 1.0]))
    assert kept.shape == (1, 3)
    assert kept[0, 0] == pytest.approx(0.001)


def test_malformed_input_is_rejected_rather_than_silently_accepted():
    accumulator = SceneCloudAccumulator()
    wall = _wall(np.random.default_rng(11))

    with pytest.raises(ValueError):
        accumulator.update(wall, stamp_s=1.0, base_from_camera=np.eye(3))

    skewed = np.eye(4)
    skewed[:3, :3] = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        accumulator.update(wall, stamp_s=1.0, base_from_camera=skewed)

    with pytest.raises(ValueError):
        accumulator.update(
            wall, stamp_s=float("nan"), base_from_camera=_pose(0.0, _EYE)
        )
