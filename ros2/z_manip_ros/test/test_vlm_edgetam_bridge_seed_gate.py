"""Seed depth gate + confidence hygiene glue in the perception bridge (P2-1/2)."""

from collections import OrderedDict
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.perception.seed_gate import SeedDepthGateConfig
from z_manip_ros.vlm_edgetam_bridge import VlmEdgeTamBridge


def _depth_image(depth_mm: np.ndarray) -> SimpleNamespace:
    stamp = SimpleNamespace(sec=100, nanosec=500_000_000)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp),
        height=int(depth_mm.shape[0]),
        width=int(depth_mm.shape[1]),
        encoding='16UC1',
        data=depth_mm.astype(np.uint16).tobytes(),
    )


def test_decode_16uc1_depth_frame_scales_to_metres():
    mm = np.full((8, 8), 1500, dtype=np.uint16)
    frame = VlmEdgeTamBridge._decode_depth_frame(_depth_image(mm), 0.001)
    assert frame.shape == (8, 8)
    assert frame[0, 0] == pytest.approx(1.5, abs=1e-3)


def test_decode_rejects_unknown_encoding():
    msg = SimpleNamespace(height=4, width=4, encoding='rgb8', data=b'\x00' * 48)
    with pytest.raises(ValueError):
        VlmEdgeTamBridge._decode_depth_frame(msg, 0.001)


def _harness(depth_m: np.ndarray, stamp_ns: int) -> SimpleNamespace:
    frames: OrderedDict[int, np.ndarray] = OrderedDict()
    frames[stamp_ns] = depth_m
    return SimpleNamespace(
        _lock=threading.RLock(),
        _seed_depth_frames=frames,
        _seed_depth_max_join_age_s=0.2,
        _seed_depth_gate_cfg=SeedDepthGateConfig(),
    )


def test_measure_seed_depth_joins_matching_stamp():
    depth = np.full((100, 100), 0.0, dtype=np.float32)
    depth[40:60, 40:60] = 0.964
    stamp_ns = 100 * 1_000_000_000 + 500_000_000
    harness = _harness(depth, stamp_ns)
    header = SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=500_000_000))
    m = VlmEdgeTamBridge._measure_seed_depth(harness, header, (0.4, 0.4, 0.6, 0.6))
    assert m.median_z_m == pytest.approx(0.964, abs=1e-3)


def test_measure_seed_depth_abstains_when_no_frame_in_join_window():
    depth = np.full((100, 100), 1.0, dtype=np.float32)
    stamp_ns = 100 * 1_000_000_000 + 500_000_000
    harness = _harness(depth, stamp_ns)
    # A header a full second away exceeds the 0.2 s join window -> abstain.
    header = SimpleNamespace(stamp=SimpleNamespace(sec=101, nanosec=500_000_000))
    m = VlmEdgeTamBridge._measure_seed_depth(harness, header, (0.4, 0.4, 0.6, 0.6))
    assert m.median_z_m is None
