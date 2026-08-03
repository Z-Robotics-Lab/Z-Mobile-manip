"""The capsule -> roboplan primitive conversion may never under-cover.

``roboplan_scene`` replaces the enforced capsules from
``configs/piper_collision_capsules.json`` with rigid primitive bodies so
roboplan's ``Scene`` can use them.  A capsule spans two link frames and is not
in general rigid in either, so each body is pinned to one frame and its radius
inflated by the measured residual.  If that inflation is ever too small the
enforced envelope shrinks silently -- the exact failure mode
``tests/test_lidar_keepout.py`` was written for.

These tests need neither roboplan nor pinocchio: the conversion is pure numpy
against ``KinematicChain``.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from z_manip.collision.pointcloud import RobotCollisionModel
from z_manip.kinematics.chain import KinematicChain
from z_manip.planning.roboplan_scene import (
    capsule_bodies,
    write_planning_model,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
SRDF = ROOT / "ros2/z_manip_motion/config/piper.srdf"
MODEL = ROOT / "configs/piper_collision_capsules.json"

pytestmark = pytest.mark.skipif(not URDF.exists(), reason="robot URDF not checked out")


@pytest.fixture(scope="module")
def fixtures():
    model = RobotCollisionModel.from_mapping(json.loads(MODEL.read_text()))
    chain = KinematicChain.from_urdf(URDF, "piper_base_link", "piper_gripper_base")
    return model, chain, capsule_bodies(chain, model)


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    axis = end - start
    span = float(axis @ axis)
    t = 0.0 if span < 1e-18 else float(np.clip((point - start) @ axis / span, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * axis)))


def test_emitted_bodies_contain_the_enforced_capsules(fixtures):
    """Every point of the true capsule stays inside the emitted, inflated one.

    Sampled at the joint limits as well as at random, because the residual is
    largest at the corners of the joint box.
    """

    model, chain, bodies = fixtures
    rng = np.random.default_rng(7)
    configs = np.vstack(
        (
            rng.uniform(chain.lower_limits, chain.upper_limits, size=(256, chain.dof)),
            chain.lower_limits,
            chain.upper_limits,
            np.zeros(chain.dof),
        )
    )
    by_name = {body.name: body for body in bodies}

    worst = 0.0
    for q in configs:
        transforms = chain.link_transforms(q)
        for capsule in model.capsules:
            body = by_name[capsule.name]
            start_tf = transforms[capsule.start_frame]
            end_tf = transforms[capsule.end_frame]
            true_start = start_tf[:3, :3] @ np.asarray(capsule.start_offset, float) + start_tf[:3, 3]
            true_end = end_tf[:3, :3] @ np.asarray(capsule.end_offset, float) + end_tf[:3, 3]

            parent = transforms[body.parent_frame]
            half = 0.5 * body.length * body.axis
            emitted_start = parent[:3, :3] @ (body.origin - half) + parent[:3, 3]
            emitted_end = parent[:3, :3] @ (body.origin + half) + parent[:3, 3]

            for t in np.linspace(0.0, 1.0, 9):
                point = true_start + t * (true_end - true_start)
                gap = _segment_distance(point, emitted_start, emitted_end) - body.inflation
                worst = max(worst, gap)
                assert gap <= 1e-9, (
                    f"{capsule.name} under-covered by {gap * 1e3:.4f} mm at q={q}"
                )
    # Guard the guard: a conversion that inflated everything hugely would also
    # pass the containment check while making the robot unable to move.
    assert max(body.inflation for body in bodies) < 1e-3


def test_no_source_collision_geometry_survives(fixtures, tmp_path):
    """The URDF's own <collision> tags stay INERT -- see the module docstring."""

    model, _, bodies = fixtures
    out_urdf, _ = write_planning_model(URDF, SRDF, model, bodies, tmp_path)
    robot = ElementTree.parse(out_urdf).getroot()

    emitted = {body.link_name for body in bodies}
    for link in robot.findall("link"):
        if link.get("name") in emitted:
            continue
        assert link.find("collision") is None, f"{link.get('name')} kept URDF collision geometry"
    assert robot.find(".//collision/geometry/mesh") is None
    assert len(robot.findall(".//collision")) == sum(
        3 if body.length > 1e-9 else 2 for body in bodies
    )


def _capsule_endpoints(chain, capsule, transforms):
    start_tf = transforms[capsule.start_frame]
    end_tf = transforms[capsule.end_frame]
    return (
        start_tf[:3, :3] @ np.asarray(capsule.start_offset, float) + start_tf[:3, 3],
        end_tf[:3, :3] @ np.asarray(capsule.end_offset, float) + end_tf[:3, 3],
    )


def _segment_gap(p1, q1, p2, q2):
    """Closest distance between two segments."""

    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f, c, b = d1 @ d1, d2 @ d2, d2 @ r, d1 @ r, d1 @ d2
    den = a * e - b * b
    s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
    t = float(np.clip((b * s + f) / e, 0.0, 1.0)) if e > 1e-12 else 0.0
    s = float(np.clip((b * t - c) / a, 0.0, 1.0)) if a > 1e-12 else 0.0
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def test_scene_verdicts_match_the_capsule_model(fixtures, tmp_path):
    """roboplan's verdict must equal the enforced capsule model's, pair for pair.

    This is the seam: everything above tests the emitted geometry, this tests
    that roboplan actually enforces it.  Skips where roboplan is absent, so a
    green suite off the 4090 does not mean this ran -- same caveat as the
    pinocchio IK tests.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    from z_manip.planning.roboplan_scene import build_scene

    model, chain, _ = fixtures
    scene = build_scene(
        URDF, SRDF, MODEL,
        base_link="piper_base_link", tip_link="piper_gripper_base", out_dir=tmp_path,
    )
    group = scene.getJointGroupInfo("piper_arm")
    q_indices = np.asarray(group.q_indices)
    q_full = scene.clampToValidConfiguration(scene.getCurrentJointPositions())
    by_name = {capsule.name: capsule for capsule in model.capsules}
    pairs = [tuple(pair) for pair in model.self_collision.pairs]

    home = np.asarray(
        json.loads((ROOT / "configs/piper_home.json").read_text())["joint_radians"], float
    )
    rng = np.random.default_rng(11)
    configs = np.vstack(
        (home, rng.uniform(chain.lower_limits, chain.upper_limits, size=(200, chain.dof)))
    )

    for q_arm in configs:
        transforms = chain.link_transforms(q_arm)
        expected = any(
            _segment_gap(
                *_capsule_endpoints(chain, by_name[a], transforms),
                *_capsule_endpoints(chain, by_name[b], transforms),
            )
            < by_name[a].radius + by_name[b].radius
            for a, b in pairs
        )
        q = q_full.copy()
        q[q_indices] = q_arm
        assert scene.hasCollisions(scene.clampToValidConfiguration(q)) == expected, (
            f"roboplan and the capsule model disagree at q={q_arm}"
        )

    # The commanded home pose clears every enforced pair -- by 3.8 mm at the
    # tightest (mount vs wrist).  If this fails the robot cannot go home.
    q = q_full.copy()
    q[q_indices] = home
    assert not scene.hasCollisions(scene.clampToValidConfiguration(q))


def test_acm_leaves_exactly_the_configured_pairs_enabled(fixtures, tmp_path):
    model, _, bodies = fixtures
    _, out_srdf = write_planning_model(URDF, SRDF, model, bodies, tmp_path)
    srdf = ElementTree.parse(out_srdf).getroot()

    disabled = {
        tuple(sorted((entry.get("link1"), entry.get("link2"))))
        for entry in srdf.findall("disable_collisions")
        if entry.get("link1", "").startswith("cap_")
    }
    names = [body.name for body in bodies]
    all_pairs = {
        tuple(sorted((f"cap_{a}", f"cap_{b}")))
        for i, a in enumerate(names)
        for b in names[i + 1:]
    }
    enabled = {tuple(sorted(pair)) for pair in model.self_collision.pairs}
    assert disabled == all_pairs - {
        tuple(sorted((f"cap_{a}", f"cap_{b}"))) for a, b in enabled
    }
    # The source SRDF's arm group must survive so roboplan can plan for it.
    assert srdf.find(".//group[@name='piper_arm']") is not None


# ---------------------------------------------------------------------------
# The joint-limit YAML.  The URDF declares no acceleration limits, so without
# this the Scene answers DBL_MAX and roboplan.toppra -- which reads its limits
# FROM the Scene and takes only a scalar scale -- runs with no acceleration
# constraint at all.  Measured that way, 1 densified path in 8 came back 1.72x
# over the requested cap.
# ---------------------------------------------------------------------------

def test_joint_limits_yaml_matches_the_chain(fixtures, tmp_path):
    from z_manip.planning.roboplan_scene import write_joint_limits

    _, chain, _ = fixtures
    limits = [1.5, 1.5, 1.5, 2.0, 2.0, 2.0]
    text = write_joint_limits(chain.joint_names, limits, tmp_path).read_text()
    for name, value in zip(chain.joint_names, limits):
        assert f"  {name}:" in text
        assert f"max_acceleration: [{value:g}]" in text


@pytest.mark.parametrize(
    "limits, match",
    [
        ([1.5, 1.5, 1.5], "do not match"),
        ([1.5, 1.5, 1.5, 2.0, 2.0, 0.0], "finite and positive"),
        ([1.5, 1.5, 1.5, 2.0, 2.0, np.inf], "finite and positive"),
    ],
)
def test_bad_acceleration_limits_fail_closed(fixtures, tmp_path, limits, match):
    from z_manip.planning.roboplan_scene import write_joint_limits

    _, chain, _ = fixtures
    with pytest.raises(ValueError, match=match):
        write_joint_limits(chain.joint_names, limits, tmp_path)


def test_the_scene_reports_the_configured_acceleration_not_dbl_max(tmp_path):
    """The regression that motivated the YAML: an unbounded Scene.

    This is the assertion that would have caught it -- ``getAccelerationLimit
    Vectors`` answering 1.798e308 is what let TOPP-RA ignore the cap.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    from z_manip.planning.roboplan_scene import build_scene

    limits = [1.5, 1.5, 1.5, 2.0, 2.0, 2.0]
    scene = build_scene(
        URDF, SRDF, MODEL,
        base_link="piper_base_link", tip_link="piper_gripper_base",
        out_dir=tmp_path / "with", acceleration_limits=limits,
    )
    _, upper = scene.getAccelerationLimitVectors("piper_arm")
    assert np.all(np.isfinite(upper))
    np.testing.assert_allclose(upper, limits)

    # And without: still DBL_MAX, so the caller cannot mistake one for the other.
    bare = build_scene(
        URDF, SRDF, MODEL,
        base_link="piper_base_link", tip_link="piper_gripper_base",
        out_dir=tmp_path / "bare",
    )
    _, unbounded = bare.getAccelerationLimitVectors("piper_arm")
    assert np.all(unbounded > 1e300)
