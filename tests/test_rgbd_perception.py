import numpy as np
import pytest
from scipy.spatial import cKDTree

from z_manip.perception import rgbd as rgbd_module
from z_manip.perception.rgbd import (
    BoundingBox,
    CameraIntrinsics,
    ColorDepthTracker,
    KDTREE_MAX_WORKERS,
    KDTREE_PARALLEL_MIN_POINTS,
    depth_bbox_observation,
    depth_to_pointcloud,
    depth_to_scene_cloud,
    filter_object_cloud,
    target_exclusion_mask,
)


class _RecordingKDTree:
    """cKDTree stand-in that records the ``workers`` used by each query."""

    def __init__(self, data, calls):
        self._tree = cKDTree(data)
        self._calls = calls

    def query(self, points, **kwargs):
        self._calls.append(kwargs.get("workers", 1))
        return self._tree.query(points, **kwargs)


def _recorded_workers(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        rgbd_module,
        "cKDTree",
        lambda data: _RecordingKDTree(data, calls),
    )
    return calls


def test_depth_bbox_is_backprojected_without_ground_truth():
    depth = np.zeros((80, 100), dtype=np.uint16)
    depth[30:50, 55:75] = 1200
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0,
                                  width=100, height=80)

    observation = depth_bbox_observation(
        depth,
        BoundingBox(55, 30, 75, 50),
        intrinsics,
        label="red can",
        stamp_s=3.5,
    )

    assert observation.label == "red can"
    assert np.allclose(observation.position_camera, (0.18, -0.012, 1.2), atol=0.02)
    assert observation.valid_points == 400


def test_color_depth_tracker_follows_target_and_rejects_background():
    image0 = np.full((90, 120, 3), 35, dtype=np.uint8)
    depth0 = np.full((90, 120), 2600, dtype=np.uint16)
    image0[30:60, 45:65] = (210, 35, 25)
    depth0[30:60, 45:65] = 1300

    tracker = ColorDepthTracker(color_tolerance=45.0, depth_tolerance_mm=350)
    tracker.initialize(image0, depth0, BoundingBox(43, 28, 67, 62))

    image1 = np.full_like(image0, 35)
    depth1 = np.full_like(depth0, 2600)
    image1[32:65, 55:78] = (208, 38, 28)
    depth1[32:65, 55:78] = 1120

    tracked = tracker.update(image1, depth1)

    assert tracked is not None
    assert 53 <= tracked.x1 <= 57
    assert 76 <= tracked.x2 <= 80
    assert tracker.depth_mm == 1120


def test_color_depth_tracker_reports_lost_instead_of_reusing_stale_pose():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    depth = np.full((60, 80), 1000, dtype=np.uint16)
    image[20:40, 30:50] = (200, 20, 20)
    tracker = ColorDepthTracker(min_pixels=30)
    tracker.initialize(image, depth, BoundingBox(28, 18, 52, 42))

    assert tracker.update(np.zeros_like(image), np.zeros_like(depth)) is None


def test_aligned_depth_backprojects_masked_metric_pointcloud_and_transform():
    depth = np.zeros((4, 6), dtype=np.uint16)
    depth[1:3, 2:5] = 1000
    mask = np.zeros_like(depth, dtype=bool)
    mask[1:3, 3:5] = True
    intrinsics = CameraIntrinsics(100.0, 100.0, 2.5, 1.5, 6, 4)
    transform = np.eye(4)
    transform[:3, 3] = (0.5, -0.2, 0.1)

    points = depth_to_pointcloud(
        depth,
        intrinsics,
        mask=mask,
        transform=transform,
        min_depth_m=0.2,
        max_depth_m=2.0,
    )

    assert points.shape == (4, 3)
    assert np.allclose(points[:, 2], 1.1)
    assert np.mean(points[:, 0]) > 0.5
    np.testing.assert_allclose(np.mean(points[:, 1]), -0.2, atol=0.01)


def test_target_cloud_exclusion_marks_only_nearby_scene_points():
    scene = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.2, 0.0, 1.0]])
    target = np.array([[0.005, 0.0, 1.0], [0.006, 0.0, 1.0]])

    excluded = target_exclusion_mask(scene, target, radius_m=0.02, min_target_points=2)

    assert excluded.tolist() == [True, True, False]


def test_object_cloud_filter_removes_background_leakage_and_sparse_fliers():
    rng = np.random.default_rng(4)
    target = rng.normal((0.4, 0.0, 1.2), (0.015, 0.02, 0.012), size=(300, 3))
    background = rng.normal((0.4, 0.0, 1.8), 0.02, size=(30, 3))
    fliers = np.array([[2.0, 2.0, 0.5], [-1.0, 1.0, 0.8]])

    filtered = filter_object_cloud(
        np.vstack((target, background, fliers)),
        viewpoint=(0.0, 0.0, 0.0),
        min_points=40,
    )

    assert 260 <= len(filtered) <= 305
    assert np.max(filtered[:, 2]) < 1.4


def _dense_object_cloud(count):
    rng = np.random.default_rng(17)
    core = rng.normal((0.4, 0.0, 1.2), (0.015, 0.02, 0.012), size=(count, 3))
    background = rng.normal((0.4, 0.0, 1.8), 0.02, size=(count // 10, 3))
    return np.vstack((core, background))


def test_neighbour_query_runs_parallel_only_above_the_thread_spawn_threshold(monkeypatch):
    calls = _recorded_workers(monkeypatch)

    filter_object_cloud(_dense_object_cloud(KDTREE_PARALLEL_MIN_POINTS * 2))
    filter_object_cloud(_dense_object_cloud(60))

    assert calls == [KDTREE_MAX_WORKERS, 1]


def test_target_exclusion_query_runs_parallel_only_above_the_threshold(monkeypatch):
    rng = np.random.default_rng(23)
    target = rng.normal((0.4, 0.0, 1.2), 0.02, size=(400, 3))
    big_scene = rng.normal((0.4, 0.0, 1.2), 0.2, size=(KDTREE_PARALLEL_MIN_POINTS * 2, 3))
    small_scene = big_scene[:32]
    calls = _recorded_workers(monkeypatch)

    target_exclusion_mask(big_scene, target)
    target_exclusion_mask(small_scene, target)

    assert calls == [KDTREE_MAX_WORKERS, 1]


def test_parallel_kdtree_queries_return_the_serial_result_exactly():
    cloud = _dense_object_cloud(KDTREE_PARALLEL_MIN_POINTS * 4)
    rng = np.random.default_rng(29)
    scene = rng.normal((0.4, 0.0, 1.2), 0.2, size=(4000, 3))

    filtered = filter_object_cloud(cloud)
    excluded = target_exclusion_mask(scene, filtered)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(rgbd_module, "_kdtree_workers", lambda _count: 1)
        serial_filtered = filter_object_cloud(cloud)
        serial_excluded = target_exclusion_mask(scene, serial_filtered)

    assert np.array_equal(filtered, serial_filtered)
    assert np.array_equal(excluded, serial_excluded)


def test_scene_cloud_keeps_segmentation_labels_aligned_after_depth_filtering():
    depth = np.full((6, 8), 1200, dtype=np.uint16)
    depth[0, 0] = 0
    target = np.zeros_like(depth, dtype=bool)
    target[2:5, 3:6] = True
    intrinsics = CameraIntrinsics(100.0, 100.0, 4.0, 3.0, 8, 6)

    points, excluded = depth_to_scene_cloud(
        depth,
        intrinsics,
        target_mask=target,
        target_dilation_px=0,
        stride=1,
    )

    assert len(points) == 47
    assert excluded.dtype == np.bool_
    assert int(excluded.sum()) == 9
    assert np.all(points[excluded, 2] == pytest.approx(1.2))


def _reference_dilation(mask, radius):
    """Original full-square scipy dilation, kept here as the equivalence oracle."""
    from scipy.ndimage import binary_dilation

    size = 2 * radius + 1
    return binary_dilation(mask, structure=np.ones((size, size), dtype=bool))


def _dilation_equivalence_cases():
    rng = np.random.default_rng(20260725)
    cases = [
        np.ones((17, 23), dtype=bool),
        np.zeros((17, 23), dtype=bool),
        np.array([[True]]),
        np.array([[False]]),
        np.zeros((3, 40), dtype=bool),
    ]
    border = np.zeros((21, 19), dtype=bool)
    border[0, :] = True
    border[:, 0] = True
    border[-1, -1] = True
    cases.append(border)
    for _ in range(20):
        shape = (int(rng.integers(1, 30)), int(rng.integers(1, 30)))
        cases.append(rng.random(shape) < 0.15)
    return cases


def test_separable_dilation_matches_the_full_square_structuring_element():
    for mask in _dilation_equivalence_cases():
        for radius in range(1, 6):
            assert np.array_equal(
                rgbd_module.dilate_mask(mask, radius),
                _reference_dilation(mask, radius),
            ), f"radius {radius} diverged on shape {mask.shape}"


def test_scene_cloud_dilation_never_evaluates_a_full_square_element(monkeypatch):
    # scipy evaluates a (2r+1)x(2r+1) element in O(r^2) per pixel. Nothing on
    # this path may call it; the separable passes are the whole point.
    def refuse(*_args, **_kwargs):
        raise AssertionError("dilation must not evaluate a full square element")

    monkeypatch.setattr(rgbd_module, "binary_dilation", refuse, raising=False)
    depth = np.full((6, 8), 1200, dtype=np.uint16)
    target = np.zeros_like(depth, dtype=bool)
    target[2:5, 3:6] = True
    intrinsics = CameraIntrinsics(100.0, 100.0, 4.0, 3.0, 8, 6)

    _points, excluded = depth_to_scene_cloud(
        depth,
        intrinsics,
        target_mask=target,
        target_dilation_px=1,
        stride=1,
    )

    assert int(excluded.sum()) == int(_reference_dilation(target, 1).sum())


def test_zero_radius_dilation_leaves_the_mask_untouched():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:3, 2:5] = True

    assert np.array_equal(rgbd_module.dilate_mask(mask, 0), mask)
