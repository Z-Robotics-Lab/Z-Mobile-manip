"""M0 ground-truth stream — the sim-only object odom the verify oracle reads
(``/objects/<name>/odom``). Exists and publishes at ~5 sim-Hz, which folds
through RTF to a ≥0.5 Hz wall floor (task spec).

Parametrized from the versioned Office scene JSON. These topics are an external
acceptance oracle only: the runtime stack is forbidden from consuming object GT.
Attach-only; skips if the chain is not green or a probe can't reach it.
"""

from __future__ import annotations

import pytest

from tests import contract as C
from tests import helpers as H

pytestmark = [pytest.mark.m0]


def _probe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except H.ProbeSkip as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("prop", C.PROPS)
def test_prop_odom_exists(chain, prop):
    """GT odom topic for the prop is present in the live graph."""
    topic = C.prop_odom_topic(prop)
    assert _probe(H.topic_exists, topic), f"{topic} missing from graph"


@pytest.mark.slow
@pytest.mark.parametrize("prop", C.PROPS)
def test_prop_odom_rate(chain, prop):
    """GT odom ≥ 4 Hz in SIM time (design: 5 Hz sim, warehouse_nav step%20)."""
    topic = C.prop_odom_topic(prop)
    r = _probe(H.topic_hz_sim, topic,
               msg_module="nav_msgs.msg", msg_class="Odometry", n_msgs=10)
    assert r["fps_sim"] >= C.GT_SIM_HZ_MIN, (
        f"{topic} at {r['fps_sim']:.2f} Hz-sim < gate {C.GT_SIM_HZ_MIN} "
        f"(design {C.GT_SIM_HZ} Hz sim; wall {r['fps_wall']:.2f} Hz)"
    )
