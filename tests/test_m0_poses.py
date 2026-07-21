"""M0 pose gates — G-b (LOOKOUT camera level, TF pitch ≤ 5°) and G-c (three-pose
settle joint error < 0.05 rad). Both are STRICT: the arm-stiffness fix landed
(go2w 096f966 K100/D5→K400/D15; re-verified strict-PASS in 0d87149, and this
suite XPASSed them independently on 2026-07-10) — the former xfail markers
(task_6dd89fc1) were removed at closeout.

Settling is measured in SIM seconds via /clock — never a wall sleep (RTF 0.2
turns a 3 s wall sleep into 0.6 sim-s; §3 / pitfall 41). The only thing these
tests command is a bounded, idempotent ``/piper/named_pose`` publish on an
already-running arm; nothing is started or torn down.
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


# ------------------------------------------------------------------------ G-b
@pytest.mark.slow
def test_gb_lookout_camera_level(chain):
    """G-b: after LOOKOUT settles, the optical axis is within 5° of horizontal.

    Command LOOKOUT, let 3 sim-s pass, then read TF parent→optical and take the
    +Z boresight's elevation vs the parent's horizontal plane.
    """
    _probe(H.set_named_pose, C.POSE_LOOKOUT)
    elev = _probe(
        H.optical_axis_pitch_deg,
        C.FRAME_CAM_PARENT, C.FRAME_OPTICAL,
        settle_sim_s=C.SETTLE_SIM_S,
    )
    assert abs(elev) <= C.GB_PITCH_MAX_DEG, (
        f"LOOKOUT optical axis elevation {elev:.2f}° exceeds "
        f"±{C.GB_PITCH_MAX_DEG}° (camera not level)"
    )


# ------------------------------------------------------------------------ G-c
@pytest.mark.slow
@pytest.mark.parametrize("pose", C.POSE_CYCLE)
def test_gc_pose_settles(chain, pose):
    """G-c: each of STOW/LOOKOUT/CARRY settles to joint error < 0.05 rad.

    Command the pose, wait 3 sim-s for the arm to reach it, then compare
    ``/piper/state`` vs ``/piper/cmd`` per joint-name; the max must be < ε.
    """
    _probe(H.set_named_pose, pose)
    _probe(H.wait_sim_seconds, C.SETTLE_SIM_S)
    err = _probe(H.joint_error, C.TOPIC_JOINT_STATE, C.TOPIC_JOINT_CMD)
    worst = max(err.per_joint, key=err.per_joint.get)
    assert err.max_err < C.GC_JOINT_ERR_MAX, (
        f"{pose}: joint '{worst}' error {err.max_err:.4f} rad "
        f">= gate {C.GC_JOINT_ERR_MAX} (arm too soft?)  per={err.per_joint}"
    )
