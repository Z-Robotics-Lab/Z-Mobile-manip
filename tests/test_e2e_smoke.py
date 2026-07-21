"""E2E M0 smoke — the whole M0 face, live, in one flow:

    chain green → LOOKOUT → both streams simultaneously alive → the depth frame
    actually SEES the scene (≥5% of pixels in [0.3, 3.0] m) → RTF ≥ 0.15 holds.

This is the "eyes on the sim" acceptance shape for M0 reduced to machine
judgement: not a flag or a message count, but the real streams carrying real
geometry while sim time runs near enough to real time. Attach-only; skips if the
chain isn't green or a probe can't reach it. Commands only a bounded LOOKOUT
publish; starts/tears down nothing.
"""

from __future__ import annotations

import pytest

from tests import contract as C
from tests import helpers as H

pytestmark = [pytest.mark.e2e, pytest.mark.m0, pytest.mark.slow]


def _probe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except H.ProbeSkip as exc:
        pytest.skip(str(exc))


def test_e2e_m0_lookout_sees_scene(chain):
    """LOOKOUT, both streams live, depth sees the scene, RTF holds — one flow."""
    # 1) put the camera to LOOKOUT (level, forward) and let it settle in sim time
    _probe(H.set_named_pose, C.POSE_LOOKOUT)
    _probe(H.wait_sim_seconds, C.SETTLE_SIM_S)

    # 2) both streams are simultaneously present and correctly typed
    topics = _probe(H.list_topics)
    for t in (C.TOPIC_COLOR, C.TOPIC_DEPTH_ALIGNED, C.TOPIC_COLOR_INFO):
        assert t in topics, f"{t} absent — dual stream not up"
    enc_c, _, _ = _probe(H.image_encoding, C.TOPIC_COLOR)
    assert enc_c == C.ENC_COLOR, f"color encoding {enc_c!r}"

    # 3) the depth frame carries real geometry: ≥5% of pixels in [0.3, 3.0] m,
    #    and the near-clip holds (nearest valid depth ≥ 0.28 m).
    stats = _probe(H.depth_frame_stats, C.TOPIC_DEPTH_ALIGNED)
    assert stats.inband_frac > C.E2E_INBAND_FRAC_MIN, (
        f"only {stats.inband_frac * 100:.1f}% of depth in [0.3,3.0] m "
        f"(≤{C.E2E_INBAND_FRAC_MIN * 100:.0f}%): camera not seeing the scene"
    )
    assert stats.min_nonzero_m >= C.GE_MIN_DEPTH_M, (
        f"near-clip violated: min depth {stats.min_nonzero_m:.3f} m"
    )

    # 4) sim time runs near real time throughout (servo loops depend on this)
    rtf = _probe(H.clock_rtf, window_s=5.0)
    assert rtf >= C.GD_RTF_MIN, f"RTF {rtf:.3f} < {C.GD_RTF_MIN}"
