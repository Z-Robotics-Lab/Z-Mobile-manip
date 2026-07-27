"""Passive-window overlap observability in the read-only perception dry run.

The overlap gate decides whether a perception bundle may be matched to a
zero-TX passive window, and a rejection is what makes a request wait for
another fixed observation period.  Rejections leave no trace in the selected
report, so the recorded corpus can only ever show the offset of a bundle that
was *accepted* -- the measurement that would size the gate is censored exactly
where it matters.

These tests pin the reported margin so both edges are measurable without
changing which bundle is admitted, and pin it to the WORST miss.  The whole
point of the number is to let a later wave widen a fail-closed tolerance from
uncensored data; recording the closest near-miss instead would under-state the
tolerance needed, which is the wrong direction for a safety gate to be wrong
in.  They also pin one shape for the rejection counters across the success and
failure reports, so a single query can read them across a corpus.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
DRY_RUN = ROOT / "scripts" / "runtime" / "go2w_perception_dry_run.py"


def _dry_run_module():
    """Import the dry run with its ROS-only dependencies stubbed out."""

    stubs = {
        "cv2": {},
        "cv_bridge": {"CvBridge": object},
        "diagnostic_msgs": {},
        "diagnostic_msgs.msg": {
            "DiagnosticArray": object,
            "DiagnosticStatus": object,
        },
        "rclpy": {"ok": lambda: False},
        "rclpy.node": {"Node": object},
        "rclpy.qos": {
            "DurabilityPolicy": object,
            "QoSProfile": object,
            "ReliabilityPolicy": object,
            "qos_profile_sensor_data": object(),
        },
        "sensor_msgs": {},
        "sensor_msgs.msg": {
            "CameraInfo": object,
            "Image": object,
            "PointCloud2": object,
        },
        "sensor_msgs_py": {},
        "sensor_msgs_py.point_cloud2": {"read_points_numpy": lambda *a, **k: None},
        "std_msgs": {},
        "std_msgs.msg": {"Bool": object, "String": object},
    }
    installed = []
    for name, attributes in stubs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        sys.modules[name] = module
        installed.append(name)
    try:
        spec = importlib.util.spec_from_file_location(
            "go2w_perception_dry_run_overlap", DRY_RUN,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["go2w_perception_dry_run_overlap"] = module
        spec.loader.exec_module(module)
    finally:
        for name in installed:
            sys.modules.pop(name, None)
    return module


MODULE = _dry_run_module()

# One fixed observation period, expressed the way the remote probe reports it.
WINDOW_START_NS = 1_785_125_385_000_000_000
WINDOW_END_NS = WINDOW_START_NS + 250_000_000


def _window():
    from z_manip.verification.passive_capture import PassiveCaptureWindow

    import numpy as np

    return PassiveCaptureWindow(
        start_unix_ns=WINDOW_START_NS,
        end_unix_ns=WINDOW_END_NS,
        midpoint_unix_ns=(WINDOW_START_NS + WINDOW_END_NS) // 2,
        joint_positions_rad=np.zeros(6),
    )


def test_overlap_margin_is_none_without_a_supported_bundle():
    assert MODULE._widest_passive_overlap_margin_s((), _window()) is None


def test_overlap_margin_reports_the_worst_miss_not_the_closest():
    # The reported margin sizes a FAIL-CLOSED gate, so it must be the worst
    # miss the widened tolerance would have to cover.  Reporting the closest
    # near-miss instead systematically under-states it.
    window = _window()
    near = WINDOW_END_NS + 300_000_000   # misses the trailing edge by 50 ms
    far = WINDOW_END_NS + 900_000_000    # misses it by 650 ms
    assert MODULE._widest_passive_overlap_margin_s((near,), window) == -0.05
    assert MODULE._widest_passive_overlap_margin_s((far,), window) == -0.65
    assert MODULE._widest_passive_overlap_margin_s((near, far), window) == -0.65
    assert MODULE._widest_passive_overlap_margin_s((far, near), window) == -0.65


def test_overlap_margin_is_negative_for_every_rejected_bundle():
    window = _window()
    # 0.35 s of sensor-stamp transport lag: the measured 0.311-0.493 s range on
    # /camera/ffs_depth_aligned puts a fresh bundle before the window start.
    lagging = WINDOW_START_NS - 350_000_000
    assert MODULE._widest_passive_overlap_margin_s((lagging,), window) == -0.1


def test_overlap_margin_is_symmetric_about_the_two_edges():
    window = _window()
    leading = MODULE._widest_passive_overlap_margin_s(
        (WINDOW_START_NS - 450_000_000,),
        window,
    )
    trailing = MODULE._widest_passive_overlap_margin_s(
        (WINDOW_END_NS + 450_000_000,),
        window,
    )
    assert leading == trailing == -0.2


def test_reported_margin_names_its_own_arithmetic():
    # The field is ``widest_rejected_overlap_margin_s``; the helper, its
    # docstring and the accumulator must all mean the same thing by "widest".
    source = DRY_RUN.read_text(encoding="utf-8")
    assert "def _widest_passive_overlap_margin_s(" in source
    assert "return round(min(margins) * 1e-9, 6) if margins else None" in source
    # Accumulated across rejections by keeping the WORST, not the closest.
    assert "or margin_s < widest_rejected_margin_s" in source


def test_rejection_stats_use_one_shape_on_both_reports():
    source = DRY_RUN.read_text(encoding="utf-8")
    # Same key, same nesting, on the failure and the success report, so a single
    # query can read them across a corpus.
    assert source.count('"passive_window_rejections": passive_window_rejections') == 2
    assert source.count(
        '"widest_rejected_overlap_margin_s": widest_rejected_margin_s',
    ) == 2
    # And never additionally nested inside the passive_capture sub-dict.
    assert '"rejections": passive_window_rejections' not in source


def test_overlap_gate_bounds_are_unchanged_by_the_margin_report():
    source = DRY_RUN.read_text(encoding="utf-8")
    # The margin is observability only: the admitted set is still the exact
    # symmetric 250 ms overlap the zero-TX evidence contract was verified with.
    assert "capture.start_unix_ns - 250_000_000" in source
    assert "capture.end_unix_ns + 250_000_000" in source
