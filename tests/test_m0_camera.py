"""M0 camera gates — G-a (dual-stream hz + encoding + intrinsics), G-e (near
clip), G-d (RTF). Live, attach-only: skip unless the chain is green; a probe
that can't reach the chain skips (never errors).

Gate values are ``tests.contract`` (go2w source of truth); no threshold here is
softened to make a live run pass. G-a's hz gate is the M0 spec floor (≥10 fps
wall); if the running chain is configured slower (e.g. GO2W_CAM_SLOW), this test
reports that real shortfall rather than lowering the bar.
"""

from __future__ import annotations

import pytest

from tests import contract as C
from tests import helpers as H

pytestmark = [pytest.mark.m0]


def _probe(fn, *a, **k):
    """Call a live probe; convert an attach-only ProbeSkip into pytest.skip."""
    try:
        return fn(*a, **k)
    except H.ProbeSkip as exc:
        pytest.skip(str(exc))


# ------------------------------------------------------------------------ G-a
@pytest.mark.slow
def test_ga_color_hz(chain):
    """G-a: color stream ≥ 10 fps in SIM time (header stamps; wall = sim × RTF)."""
    r = _probe(H.topic_hz_sim, C.TOPIC_COLOR)
    assert r["fps_sim"] >= C.GA_HZ_MIN - C.GA_HZ_TOL, (
        f"color {C.TOPIC_COLOR} at {r['fps_sim']:.2f} fps-sim < gate {C.GA_HZ_MIN} "
        f"(wall {r['fps_wall']:.2f} Hz, rtf~{r['rtf_implied']:.2f}; "
        f"GO2W_CAM_SLOW on? stride misconfig?)"
    )


@pytest.mark.slow
def test_ga_depth_hz(chain):
    """G-a: aligned-depth stream ≥ 10 fps in SIM time (header stamps)."""
    r = _probe(H.topic_hz_sim, C.TOPIC_DEPTH_ALIGNED)
    assert r["fps_sim"] >= C.GA_HZ_MIN - C.GA_HZ_TOL, (
        f"depth {C.TOPIC_DEPTH_ALIGNED} at {r['fps_sim']:.2f} fps-sim < gate "
        f"{C.GA_HZ_MIN} (wall {r['fps_wall']:.2f} Hz)"
    )


def test_ga_color_encoding(chain):
    """G-a: color is rgb8 at 848x480."""
    enc, w, h = _probe(H.image_encoding, C.TOPIC_COLOR)
    assert enc == C.ENC_COLOR, f"color encoding {enc!r} != {C.ENC_COLOR!r}"
    assert (w, h) == (C.CAM_WIDTH, C.CAM_HEIGHT), f"color size {w}x{h}"


def test_ga_depth_encoding(chain):
    """G-a: aligned depth is 16UC1 at 848x480."""
    enc, w, h = _probe(H.image_encoding, C.TOPIC_DEPTH_ALIGNED)
    assert enc == C.ENC_DEPTH, f"depth encoding {enc!r} != {C.ENC_DEPTH!r}"
    assert (w, h) == (C.CAM_WIDTH, C.CAM_HEIGHT), f"depth size {w}x{h}"


def test_ga_camera_info(chain):
    """G-a: CameraInfo is 848x480 with fx in the D435 color-family window."""
    info = _probe(H.camera_info, C.TOPIC_COLOR_INFO)
    assert (info.width, info.height) == (C.CAM_WIDTH, C.CAM_HEIGHT), (
        f"CameraInfo size {info.width}x{info.height}"
    )
    assert C.FX_MIN <= info.fx <= C.FX_MAX, (
        f"fx={info.fx:.2f} outside gate [{C.FX_MIN},{C.FX_MAX}] "
        f"(nominal {C.FX_NOMINAL})"
    )


# ------------------------------------------------------------------------ G-e
@pytest.mark.slow
def test_ge_depth_near_clip(chain):
    """G-e: nearest non-zero depth in a frame ≥ 0.28 m (near-clip honoured)."""
    stats = _probe(H.depth_frame_stats, C.TOPIC_DEPTH_ALIGNED)
    assert stats.nonzero_frac > 0.0, "depth frame is entirely zero (no scene?)"
    assert stats.min_nonzero_m >= C.GE_MIN_DEPTH_M, (
        f"min non-zero depth {stats.min_nonzero_m:.3f} m < near-clip "
        f"{C.GE_MIN_DEPTH_M} m"
    )


# ------------------------------------------------------------------------ G-d
@pytest.mark.slow
def test_gd_rtf(chain):
    """G-d: real-time factor ≥ 0.15 (two /clock samples vs wall)."""
    rtf = _probe(H.clock_rtf, window_s=5.0)
    assert rtf >= C.GD_RTF_MIN, f"RTF {rtf:.3f} < gate {C.GD_RTF_MIN}"
