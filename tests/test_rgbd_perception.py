import numpy as np
import pytest

from z_manip.perception.rgbd import (
    BoundingBox,
    CameraIntrinsics,
    ColorDepthTracker,
    depth_bbox_observation,
    depth_to_pointcloud,
    depth_to_scene_cloud,
    filter_object_cloud,
    target_exclusion_mask,
)


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
