from __future__ import annotations

import numpy as np
import pytest

from z_manip_edgetam.core import MotionAdaptiveDepthFilter


def _filter() -> MotionAdaptiveDepthFilter:
    return MotionAdaptiveDepthFilter(
        window_size=5,
        min_valid_fraction=0.6,
        max_mad_m=0.006,
        motion_threshold_m=0.012,
        global_motion_fraction=0.15,
        min_motion_pixels=16,
    )


def test_static_d435_jitter_uses_temporal_median_and_rejects_isolated_spike():
    depth_filter = _filter()
    outputs = []
    reports = []
    for index, offset_mm in enumerate((0, 2, -2, 1, 3), start=1):
        depth = np.full((24, 24), 1.0 + offset_mm * 0.001, dtype=np.float32)
        if index == 5:
            depth[4, 4] = 1.08
        output, report = depth_filter.update(depth, stamp_ns=index * 100_000_000)
        outputs.append(output)
        reports.append(report)

    assert reports[-1]["mode"] == "static_temporal"
    assert reports[-1]["frame_count"] == 5
    assert outputs[-1][4, 4] == pytest.approx(1.0, abs=0.004)
    assert np.std(outputs[-1][10:14, 10:14]) < 1e-6
    assert reports[-1]["applied_to"] == [
        "target_pointcloud",
        "scene_pointcloud",
    ]


def test_local_object_motion_bypasses_median_without_global_reset():
    depth_filter = _filter()
    for index in range(1, 5):
        depth_filter.update(
            np.full((30, 30), 1.0, dtype=np.float32),
            stamp_ns=index * 100_000_000,
        )
    moved = np.full((30, 30), 1.0, dtype=np.float32)
    moved[10:16, 10:16] = 1.05

    output, report = depth_filter.update(moved, stamp_ns=500_000_000)

    assert report["mode"] == "local_motion"
    assert report["dynamic_pixels"] >= 16
    np.testing.assert_allclose(output[11:15, 11:15], 1.05)
    np.testing.assert_allclose(output[:8, :8], 1.0)


def test_eye_in_hand_camera_motion_resets_history_and_tracks_latest_frame():
    depth_filter = _filter()
    for index in range(1, 5):
        depth_filter.update(
            np.full((20, 20), 0.8, dtype=np.float32),
            stamp_ns=index * 100_000_000,
        )

    moved = np.full((20, 20), 0.92, dtype=np.float32)
    output, report = depth_filter.update(moved, stamp_ns=500_000_000)

    assert report["mode"] == "camera_motion_reset"
    assert report["global_changed_fraction"] == pytest.approx(1.0)
    assert report["frame_count"] == 1
    np.testing.assert_allclose(output, moved)


def test_gap_and_non_monotonic_stamp_reset_and_bad_config_rejected():
    depth_filter = _filter()
    depth = np.ones((8, 8), dtype=np.float32)
    depth_filter.update(depth, stamp_ns=100_000_000)
    _output, gap = depth_filter.update(depth, stamp_ns=800_000_000)
    assert gap["reset_reason"] == "input_gap"
    _output, order = depth_filter.update(depth, stamp_ns=700_000_000)
    assert order["reset_reason"] == "stamp_not_increasing"

    with pytest.raises(ValueError):
        MotionAdaptiveDepthFilter(window_size=2)
    with pytest.raises(ValueError):
        MotionAdaptiveDepthFilter(global_motion_fraction=1.1)


def test_permanently_invalid_pixels_are_quiet_and_remain_invalid(recwarn):
    depth_filter = _filter()
    output = None
    for index in range(1, 6):
        depth = np.ones((12, 12), dtype=np.float32)
        depth[2:5, 3:7] = 0.0
        output, _report = depth_filter.update(
            depth,
            stamp_ns=index * 100_000_000,
        )

    assert output is not None
    assert np.count_nonzero(output[2:5, 3:7]) == 0
    assert output.flags.writeable is False
    assert not [item for item in recwarn if issubclass(item.category, RuntimeWarning)]
