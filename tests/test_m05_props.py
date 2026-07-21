"""M0.5 Office shelf and diverse graspable-object acceptance checks.

The go2W scene loader reads ``configs/manip_office_scene.json``. This suite reads
that same file, so adding a new shape does not require copying names or positions
into Python. GT is used only after the fact for simulator acceptance; it is never
an input to perception, planning, servoing, grasping, or verification.

Contract names, poses, shape families and support bounds are derived in
:mod:`tests.contract` from the versioned JSON.

Attach-only (z-manip invariant): starts / restarts / tears down nothing. Every
probe that cannot reach the chain — chain not green, topic absent (e.g. the
warehouse scene has no M0.5 props), or a timeout — becomes ``pytest.skip``, never
a hard error. So this file is green-or-skip on any host; it FAILS only when a prop
is actually observed off its spot or through the floor.
"""

from __future__ import annotations

import pytest

from tests import contract as C
from tests import helpers as H

pytestmark = [pytest.mark.m05]

# The M0.5 physics props with a configured GT spot (pallet is physics=False ⇒ no
# odom ⇒ not here). Drive parametrization off the contract table so a new prop is
# covered by adding one row there + to PROPS.
_M05_PROPS = sorted(C.PROP_XYZ)


def _probe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except H.ProbeSkip as exc:
        pytest.skip(str(exc))


def _require_topic(topic: str) -> None:
    """Skip (attach-only) unless the prop's GT topic is in the live graph.

    Absent ⇒ this is not the office scene (or the prop isn't loaded); G-p2 does
    not apply. Distinct from an observed-wrong failure.
    """
    if not _probe(H.topic_exists, topic):
        pytest.skip(f"{topic} absent (not office scene / prop not loaded)")


@pytest.mark.parametrize("prop", _M05_PROPS)
def test_m05_prop_odom_exists(chain, prop):
    """Each M0.5 physics prop publishes its /objects/<name>/odom GT topic."""
    _require_topic(C.prop_odom_topic(prop))


@pytest.mark.slow
@pytest.mark.parametrize("prop", _M05_PROPS)
def test_m05_prop_on_pallet_after_settle(chain, prop):
    """G-p2: after settle, prop GT xy on its configured spot ±0.10 m and z>0.05 m.

    z>0.05 m proves it is resting on the pallet (collider present, not穿地/toppled
    off); xy proximity proves it spawned where the scene config placed it. Waits
    the settle window in SIM time (never wall — RTF≈0.2) before sampling.
    """
    topic = C.prop_odom_topic(prop)
    _require_topic(topic)

    # Let the object come to rest (SIM seconds; wait_sim_seconds guards a stalled
    # clock and skips rather than hangs).
    _probe(H.wait_sim_seconds, C.SETTLE_SIM_S)

    pose = _probe(H.prop_odom_pose, topic)
    exp_x, exp_y, _exp_z = C.PROP_XYZ[prop]

    # GT must be expressed in the world frame with the prop's own child frame.
    assert pose["frame"] == "world", (
        f"{topic} header.frame_id={pose['frame']!r}, expected 'world'"
    )
    assert pose["child"] == prop, (
        f"{topic} child_frame_id={pose['child']!r}, expected {prop!r}"
    )

    dx = pose["x"] - exp_x
    dy = pose["y"] - exp_y
    assert abs(dx) <= C.PROP_POS_TOL_M and abs(dy) <= C.PROP_POS_TOL_M, (
        f"{prop} GT xy=({pose['x']:.3f},{pose['y']:.3f}) off configured spot "
        f"({exp_x:.3f},{exp_y:.3f}) by ({dx:+.3f},{dy:+.3f}) m > {C.PROP_POS_TOL_M} m"
    )
    assert pose["z"] > C.Z_ON_SUPPORT_MIN, (
        f"{prop} GT z={pose['z']:.3f} m <= {C.Z_ON_SUPPORT_MIN} m - fell through / "
        f"toppled off the shelf (missing collider or unstable spawn)"
    )


def test_m05_scene_config_is_scalable_and_reachable():
    """Config gate: diverse objects stay inside the arm-facing shelf edge band.

    上限=PiPER 626mm 的台缘可达窄带（CEO 2026-07-10 reach 裁定）；下限=不悬空出台。
    纯查 contract 常量与摆位表的一致性——摆位以后改大改远会被这里拦住，不靠眼力。
    """
    assert C.SCENE_CONFIG_PATH.is_file()
    assert len(C.PROP_XYZ) >= 6
    assert len(set(C.PROP_SHAPE_FAMILY.values())) >= 6
    cx, cy = C.SUPPORT_SURFACE_CENTER_XY
    hx, hy = C.SUPPORT_SURFACE_HALF_XY
    for prop, (x, y, _z) in C.PROP_XYZ.items():
        edge = min(hx - abs(x - cx), hy - abs(y - cy))
        assert edge >= C.PROP_EDGE_DIST_MIN_M, (
            f"{prop} 摆出/贴台缘 ({x},{y})：最近边距 {edge:.3f} m < "
            f"{C.PROP_EDGE_DIST_MIN_M}（悬空风险）"
        )
        assert edge <= C.PROP_EDGE_DIST_MAX_M, (
            f"{prop} 离台缘 {edge:.3f} m > {C.PROP_EDGE_DIST_MAX_M} m —— "
            f"超出 PiPER 台缘可达窄带（626mm 臂展账，plan §4a）"
        )


@pytest.mark.slow
@pytest.mark.parametrize("prop", _M05_PROPS)
def test_m05_prop_settled_and_optional_upright(chain, prop):
    """G-p7: every shape settles; explicitly upright objects also keep their axis.

    立正生成（Rx90°）后 (R·ŷ)·ẑ = 2(qy·qz+qw·qx) 应≈1；tilt 超阈 = 生成姿态错 /
    沉降翻倒 / 被撞倒。G-p2 只查位置查不出"躺着"——这是它的补盲。
    """
    import math

    topic = C.prop_odom_topic(prop)
    _require_topic(topic)
    _probe(H.wait_sim_seconds, C.SETTLE_SIM_S)
    pose = _probe(H.prop_odom_pose, topic)
    speed = math.sqrt(pose["vx"] ** 2 + pose["vy"] ** 2 + pose["vz"] ** 2)
    assert speed <= C.PROP_SETTLED_SPEED_MAX_MPS, (
        f"{prop} speed {speed:.4f} m/s > {C.PROP_SETTLED_SPEED_MAX_MPS} m/s "
        "after the settle window"
    )
    if not C.PROP_REQUIRES_UPRIGHT[prop]:
        return
    qw, qx, qy, qz = pose["qw"], pose["qx"], pose["qy"], pose["qz"]
    world_z_row = (
        2.0 * (qx * qz - qw * qy),
        2.0 * (qy * qz + qw * qx),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    local_up = C.PROP_UP_AXIS_LOCAL[prop]
    up_z = sum(a * b for a, b in zip(world_z_row, local_up))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    assert tilt <= C.PROP_TILT_MAX_DEG, (
        f"{prop} 竖轴倾角 {tilt:.1f}° > {C.PROP_TILT_MAX_DEG}°（躺倒/翻倒；"
        f"quat w={pose['qw']:.3f} x={pose['qx']:.3f} y={pose['qy']:.3f} z={pose['qz']:.3f}）"
    )
