"""Equivalence and cost guards for the tracker's target-mask dilation.

``project_scene_depth`` grows the tracked-target mask before back-projection so
the contact corridor stays reserved. The dilation must keep the exact geometry
of a full ``(2r+1) x (2r+1)`` structuring element -- including at the image
border -- while costing far less than evaluating that element per pixel.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "z_manip_edgetam"))

from z_manip_edgetam import core as tracker_core  # noqa: E402


def _reference_dilation(mask, radius):
    """Full-square structuring element, kept here as the equivalence oracle."""
    size = 2 * radius + 1
    return binary_dilation(
        np.ascontiguousarray(mask, dtype=bool),
        structure=np.ones((size, size), dtype=bool),
    )


def _equivalence_cases():
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
    # A strided view, to prove the contiguity handling is not assumed away.
    strided = np.zeros((30, 60), dtype=bool)
    strided[5:25, 10:50] = True
    cases.append(strided[:, ::2])
    return cases


def test_tracker_dilation_matches_the_full_square_structuring_element():
    for mask in _equivalence_cases():
        for radius in range(1, 6):
            assert np.array_equal(
                tracker_core.dilate_mask(mask, radius),
                _reference_dilation(mask, radius),
            ), f"radius {radius} diverged on shape {mask.shape}"


def test_tracker_dilation_delegates_to_opencv_instead_of_shifting_in_python(monkeypatch):
    # The previous form ORed (2r+1)**2 shifted copies in a Python loop. OpenCV
    # evaluates the separable rectangle in C, so the call must actually happen.
    calls = []
    original_dilate = tracker_core.cv2.dilate

    def recording_dilate(*args, **kwargs):
        calls.append(kwargs.get("borderType"))
        return original_dilate(*args, **kwargs)

    monkeypatch.setattr(tracker_core.cv2, "dilate", recording_dilate)
    mask = np.zeros((24, 32), dtype=bool)
    mask[8:16, 10:22] = True

    dilated = tracker_core.dilate_mask(mask, 3)

    assert calls == [tracker_core.cv2.BORDER_CONSTANT]
    assert np.array_equal(dilated, _reference_dilation(mask, 3))


def test_zero_radius_dilation_leaves_the_mask_untouched():
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:3, 2:5] = True

    assert np.array_equal(tracker_core.dilate_mask(mask, 0), mask)


def test_scene_projection_still_excludes_the_dilated_target_corridor():
    intrinsics = tracker_core.CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=6.0)
    depth = np.full((12, 16), 1.2, dtype=np.float64)
    mask = np.zeros((12, 16), dtype=bool)
    mask[5:7, 7:9] = True

    points = tracker_core.project_scene_depth(
        mask,
        depth,
        intrinsics,
        target_dilation_px=2,
        stride=1,
        min_depth_m=0.3,
        max_depth_m=3.0,
        max_points=1024,
    )

    retained = depth.size - int(_reference_dilation(mask, 2).sum())
    assert len(points) == retained


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_tracker_and_library_dilation_agree(radius):
    # Two implementations exist on different dependency budgets; pin them to
    # one another so the copies cannot silently diverge.
    from z_manip.perception.rgbd import dilate_mask as library_dilate

    rng = np.random.default_rng(41)
    mask = rng.random((37, 53)) < 0.2

    assert np.array_equal(
        tracker_core.dilate_mask(mask, radius),
        library_dilate(mask, radius),
    )
