"""The organized cloud placement needs must agree with the live target cloud.

``tests/data/place_target_backprojection.npz`` freezes one recorded perception
frame: the EdgeTAM mask, the metric depth of its pixels, the recovered colour
intrinsics, and the ``target_points.npy`` the live stack actually published for
that frame.  If ``backproject_depth`` ever disagrees with the back-projection
the hand-eye calibration was solved against, this fails.
"""

from pathlib import Path

import numpy as np
import pytest

from z_manip.perception.rgbd import (
    CameraIntrinsics,
    backproject_depth,
    depth_to_pointcloud,
)


FIXTURE = Path(__file__).resolve().parent / "data/place_target_backprojection.npz"


def _frame():
    return np.load(FIXTURE, allow_pickle=False)


def test_organized_backprojection_round_trips_the_recorded_target_cloud():
    frame = _frame()
    mask = frame["mask"]
    recorded = frame["target_points_camera"]

    organized = backproject_depth(
        frame["depth_m"],
        frame["camera_matrix"],
        np.eye(4),
        min_depth_m=float(frame["min_depth_m"]),
        max_depth_m=float(frame["max_depth_m"]),
    )

    assert organized.shape == (*mask.shape, 3)
    finite = np.all(np.isfinite(organized), axis=2)
    # Depth exists only where the tracker segmented the object in this fixture.
    np.testing.assert_array_equal(finite, mask)
    # The recorded cloud is the masked pixels in row-major order.
    rows, columns = np.nonzero(mask)
    error = np.linalg.norm(organized[rows, columns] - recorded, axis=1)
    assert float(error.max()) < 1e-3
    # 1 mm is the contract; half-pixel or transposed conventions would land
    # around 0.3 mm at this range and must not pass silently.
    assert float(error.max()) < 1e-5


def test_organized_backprojection_applies_the_measured_hand_eye_transform():
    frame = _frame()
    mask = frame["mask"]
    transform = frame["base_from_camera"]
    recorded = frame["target_points_camera"]

    organized = backproject_depth(
        frame["depth_m"],
        frame["camera_matrix"],
        transform,
        min_depth_m=float(frame["min_depth_m"]),
        max_depth_m=float(frame["max_depth_m"]),
    )

    rows, columns = np.nonzero(mask)
    expected = recorded @ transform[:3, :3].T + transform[:3, 3]
    assert float(np.abs(organized[rows, columns] - expected).max()) < 1e-8


def test_organized_and_flat_clouds_share_one_backprojection():
    frame = _frame()
    matrix = frame["camera_matrix"]
    depth_m = frame["depth_m"]
    height, width = depth_m.shape
    intrinsics = CameraIntrinsics(
        fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
        width=width, height=height,
    )
    depth_mm = np.rint(depth_m * 1000.0).astype(np.uint16)

    flat = depth_to_pointcloud(
        depth_mm, intrinsics, mask=frame["mask"], min_depth_m=0.28, max_depth_m=5.0,
    )
    organized = backproject_depth(
        depth_m, matrix, np.eye(4), min_depth_m=0.28, max_depth_m=5.0,
    )

    rows, columns = np.nonzero(frame["mask"])
    assert float(np.abs(organized[rows, columns] - flat).max()) < 1e-6


def test_organized_backprojection_holes_out_of_range_depth_and_fails_closed():
    frame = _frame()
    matrix = frame["camera_matrix"]
    depth_m = frame["depth_m"]
    observed = depth_m[frame["mask"]]

    organized = backproject_depth(
        depth_m, matrix, np.eye(4),
        min_depth_m=float(observed.max()) + 0.01, max_depth_m=5.0,
    )
    assert not np.any(np.isfinite(organized))

    with pytest.raises(ValueError, match="depth interval"):
        backproject_depth(depth_m, matrix, np.eye(4), min_depth_m=1.0, max_depth_m=0.5)
    with pytest.raises(ValueError, match="homogeneous matrix"):
        backproject_depth(
            depth_m, matrix, np.eye(3), min_depth_m=0.28, max_depth_m=5.0,
        )
    with pytest.raises(ValueError, match="focal lengths"):
        backproject_depth(
            depth_m, np.zeros((3, 3)), np.eye(4), min_depth_m=0.28, max_depth_m=5.0,
        )
    with pytest.raises(ValueError, match="HxW metric depth"):
        backproject_depth(
            depth_m[None], matrix, np.eye(4), min_depth_m=0.28, max_depth_m=5.0,
        )
