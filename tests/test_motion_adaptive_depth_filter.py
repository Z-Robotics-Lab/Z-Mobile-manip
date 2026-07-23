"""Regression tests for the motion-adaptive temporal depth filter.

The half-resolution median tiles each decimated value over its 2x2 block, so
the camera-motion detector must compare on the decimated grid.  Testing the
tiled output against full-resolution input reads the intra-block spatial
gradient (one pixel of surface slope) as scene motion on every frame, which
permanently resets the temporal window on any oblique surface.  Observed live
at 640x480/30fps: a fully static floor reported ~21% "changed" pixels each
frame, the mode never left camera_motion_reset, and mad_p95 stayed 0.0 because
the median never engaged.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2" / "z_manip_edgetam"))

from z_manip_edgetam.core import MotionAdaptiveDepthFilter  # noqa: E402

STEP_NS = 33_000_000


def _oblique_floor(
    height: int = 32,
    width: int = 32,
    row_step_m: float = 0.015,
) -> np.ndarray:
    # 15mm between neighbouring rows exceeds the 12mm motion threshold, like
    # carpet viewed at a shallow angle from ~2m.  A 2x2 block that borrows its
    # leader's value therefore differs from the raw pixel by more than the
    # threshold even when nothing moves.
    rows = 1.0 + np.arange(height, dtype=np.float32) * row_step_m
    return np.tile(rows[:, None], (1, width))


def test_static_oblique_surface_keeps_temporal_median_at_half_resolution():
    filt = MotionAdaptiveDepthFilter(half_resolution=True)
    frame = _oblique_floor()
    report: dict[str, object] = {}
    for index in range(8):
        _out, report = filt.update(
            frame.copy(),
            stamp_ns=(index + 1) * STEP_NS,
        )
        assert report["mode"] != "camera_motion_reset", (
            f"static scene misread as camera motion at frame {index + 1}: "
            f"changed_fraction={report['global_changed_fraction']}"
        )
    assert report["mode"] == "static_temporal"
    assert report["frame_count"] == 5
    assert report["global_changed_fraction"] == 0.0


def test_global_scene_shift_still_resets_window_at_half_resolution():
    filt = MotionAdaptiveDepthFilter(half_resolution=True)
    frame = _oblique_floor()
    for index in range(4):
        filt.update(frame.copy(), stamp_ns=(index + 1) * STEP_NS)
    _out, report = filt.update(frame + 0.2, stamp_ns=5 * STEP_NS)
    assert report["mode"] == "camera_motion_reset"
    assert report["frame_count"] == 1


def test_local_object_motion_passes_through_at_half_resolution():
    filt = MotionAdaptiveDepthFilter(half_resolution=True)
    frame = _oblique_floor()
    for index in range(5):
        filt.update(frame.copy(), stamp_ns=(index + 1) * STEP_NS)
    moved = frame.copy()
    moved[8:16, 8:16] -= 0.5
    out, report = filt.update(moved, stamp_ns=6 * STEP_NS)
    assert report["mode"] == "local_motion"
    assert out[10, 10] == np.float32(moved[10, 10])
