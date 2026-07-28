"""Enforced keep-out around the Livox Mid-360 on the Go2-W head.

Why this file exists
====================
The PiPER gripper physically struck the Mid-360 during live operation, on the
return leg, while carrying an object.  Two independent defects made that
possible, and both are pinned here:

1.  The URDF collision geometry on the head is INERT.
    ``PinocchioSelfCollisionChecker`` (``z_manip/collision/pinocchio_self.py``)
    reduces the model to the six active arm joints and then keeps a geometry
    only if its frame is ``piper_base_link`` or its post-reduction supporting
    joint is one of ``piper_joint1..6``.  ``base``, ``Head_upper``,
    ``Head_lower``, ``mid360_link``, ``mid360_bracket_link``, ``mount_plate``,
    ``nuc_weight`` and ``regulator`` are all fixed to ``base``, so after
    ``buildReducedModel`` their supporting joint is ``universe`` and every one
    of them is dropped before a single collision pair is built.

    Consequence: commit **e4be019** ("enlarge envelope") edited head collision
    geometry in ``go2w_sensored.urdf`` and had ZERO runtime effect, and commit
    **a8ecd46** then shrank the *enforced* capsule in
    ``configs/piper_collision_capsules.json`` from r=0.042 to r=0.0325 -- a
    silent revert that nobody noticed for months.  The URDF and the capsule
    file drifted apart because nothing tied them together.
    ``test_capsules_track_the_urdf_head_geometry`` is that tie.

2.  The capsule under-covered the hardware, and the CARRIED OBJECT was never
    tested against it at all (``check_target`` was ``false``), so an object in
    the gripper could be driven straight through the sensor while the gate
    reported the state collision-free.

Do not shrink ``mid360``/``mid360_bracket`` or flip their ``check_target`` off.
If a future change needs more workspace, move the *arm*, not the sensor model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from z_manip.collision import (
    CollisionResult,
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
)
from z_manip.fixed_self_collision import FixedSelfCollisionGuard, _segment_distance
from z_manip.kinematics import KinematicChain


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
MODEL = ROOT / "configs/piper_collision_capsules.json"

# The two poses below are recorded joint states from the live incident run.
#
# STRIKE is the pose the OLD model (r=0.0325 at base x=0.270, z=0.100) reported
# as clear with +25.5 mm of Mid-360 margin.  Measured against the real sensor
# body it is 5.4 mm INSIDE the metal.  That is the fail-open the operator hit.
STRIKE = (-0.049, 0.188, -0.009, -0.045, 0.331, 0.0)
# CONTROL is the same approach retracted to 35 % of its joint-space excursion.
# It is genuinely outside the corrected envelope (+20.3 mm) and must stay valid,
# so a regression that simply rejects everything cannot pass this file.
CONTROL = (-0.01715, 0.0658, -0.00315, -0.01575, 0.11585, 0.0)

# Hardware truth, independent of any capsule: the Mid-360 puck is a cylinder of
# radius 0.042 m and length 0.075 m whose axis passes through the sensor origin.
SENSOR_RADIUS_M = 0.042
SENSOR_LENGTH_M = 0.075


def _raw() -> dict:
    return json.loads(MODEL.read_text(encoding="utf-8"))


def _capsule(name: str) -> dict:
    return next(item for item in _raw()["capsules"] if item["name"] == name)


def _guard() -> FixedSelfCollisionGuard:
    return FixedSelfCollisionGuard(urdf_path=URDF, model_path=MODEL)


def _pair_margin(guard: FixedSelfCollisionGuard, joints, name: str) -> float:
    """Smallest margin over every configured pair that involves ``name``.

    ``check_state`` only reports the global argmin, which can be a different
    fixture.  These tests need the Mid-360 specifically.
    """

    world = guard._world_capsules(joints)
    margins = []
    for first, second in guard.pairs:
        if name not in (first, second):
            continue
        start_a, end_a, radius_a = world[first]
        start_b, end_b, radius_b = world[second]
        distance = _segment_distance(start_a, end_a, start_b, end_b)
        margins.append(distance - (radius_a + radius_b + guard.clearance_m))
    assert margins, f"capsule {name!r} participates in no self-collision pair"
    return min(margins)


# --------------------------------------------------------------------------
# 1.  The capsule file and the URDF may never drift apart again.
# --------------------------------------------------------------------------


def _urdf_origins() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ElementTree.parse(URDF).getroot()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        if origin is None:
            continue
        out[joint.attrib["name"]] = (
            np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" "),
            np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" "),
        )
    return out


def _rotation_y(pitch: float) -> np.ndarray:
    cos, sin = np.cos(pitch), np.sin(pitch)
    return np.asarray([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def test_capsules_track_the_urdf_head_geometry():
    """Recompute both head capsules from the URDF and require an exact match.

    This is the anti-regression device for e4be019 / a8ecd46: moving the sensor
    in the URDF without moving the enforced capsule (or vice versa) now fails
    the suite instead of silently disarming the gate.
    """

    origins = _urdf_origins()
    mount_xyz, mount_rpy = origins["piper_mount_joint"]
    np.testing.assert_allclose(mount_rpy, np.zeros(3), atol=1e-12)

    sensor_xyz, sensor_rpy = origins["mid360_joint"]
    assert sensor_rpy[0] == 0.0 and sensor_rpy[2] == 0.0
    axis = _rotation_y(float(sensor_rpy[1])) @ np.asarray([0.0, 0.0, 1.0])

    root = ElementTree.parse(URDF).getroot()
    link = next(
        item for item in root.findall("link") if item.attrib["name"] == "mid360_link"
    )
    cylinder = link.find("collision/geometry/cylinder")
    assert cylinder is not None, "mid360_link lost its collision cylinder"
    radius = float(cylinder.attrib["radius"])
    length = float(cylinder.attrib["length"])
    assert (radius, length) == (SENSOR_RADIUS_M, SENSOR_LENGTH_M)
    collision_origin = link.find("collision/origin")
    np.testing.assert_allclose(
        np.fromstring(collision_origin.attrib["xyz"], sep=" "),
        np.zeros(3),
        atol=1e-12,
    )

    half = 0.5 * length
    expected_start = sensor_xyz - half * axis - mount_xyz
    expected_end = sensor_xyz + half * axis - mount_xyz

    puck = _capsule("mid360")
    np.testing.assert_allclose(puck["start_offset"], expected_start, atol=1e-6)
    np.testing.assert_allclose(puck["end_offset"], expected_end, atol=1e-6)
    # The capsule must enclose the puck, never merely touch its axis.
    assert puck["radius"] >= radius

    # The bracket capsule runs along the vertical centre line of the URDF box
    # and its radius must reach the box's vertical edges.
    bracket_xyz, bracket_rpy = origins["mid360_bracket_joint"]
    np.testing.assert_allclose(bracket_rpy, np.zeros(3), atol=1e-12)
    box = next(
        item
        for item in root.findall("link")
        if item.attrib["name"] == "mid360_bracket_link"
    ).find("collision/geometry/box")
    assert box is not None, "mid360_bracket_link lost its collision box"
    size = np.fromstring(box.attrib["size"], sep=" ")

    spec = _capsule("mid360_bracket")
    np.testing.assert_allclose(
        spec["start_offset"],
        bracket_xyz - np.asarray([0.0, 0.0, 0.5 * size[2]]) - mount_xyz,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        spec["end_offset"],
        bracket_xyz + np.asarray([0.0, 0.0, 0.5 * size[2]]) - mount_xyz,
        atol=1e-6,
    )
    assert spec["radius"] >= float(np.hypot(0.5 * size[0], 0.5 * size[1])) - 1e-9

    # The bracket must actually reach the deck the sensor is bolted to; if the
    # torso box is ever re-measured this catches a floating bracket.
    assert (bracket_xyz[2] + 0.5 * size[2]) == pytest.approx(float(sensor_xyz[2]))


# --------------------------------------------------------------------------
# 2.  Nobody may quietly shrink or disarm the keep-out again.
# --------------------------------------------------------------------------


def test_head_keepout_is_not_shrunk_or_disarmed():
    """Pin every field a8ecd46-style edit could weaken.

    a8ecd46 changed a single number (0.042 -> 0.0325) and no test noticed.
    """

    for name, minimum_radius in (("mid360", 0.050), ("mid360_bracket", 0.0495)):
        spec = _capsule(name)
        assert spec["radius"] >= minimum_radius, (
            f"{name} radius shrank below the approved {minimum_radius} m envelope"
        )
        # supplemental_self_collision is what keeps the capsule alive on the
        # real robot: with the pinocchio mesh backend loaded, check_state calls
        # _check_self_collision(supplemental_only=True), so a head capsule
        # without this flag is silently inert in production while still
        # passing unit tests that build the checker without a mesh backend.
        assert spec["supplemental_self_collision"] is True
        # check_target is what tests the CARRIED OBJECT against the sensor.
        assert spec["check_target"] is True
        # The head is robot, not world: it must never be tested against its own
        # perceived point cloud.
        assert spec["check_scene"] is False
        assert spec["start_frame"] == "piper_base_link"
        assert spec["end_frame"] == "piper_base_link"

    # Physical extent in the `base` frame, recomputed from the offsets so a
    # move-without-resize is caught too.
    mount = np.asarray([0.128526599, 0.0, 0.081794287])
    puck = _capsule("mid360")
    start = np.asarray(puck["start_offset"]) + mount
    end = np.asarray(puck["end_offset"]) + mount
    radius = float(puck["radius"])
    assert max(start[0], end[0]) + radius >= 0.392
    assert max(start[2], end[2]) + radius >= 0.225
    assert radius >= 0.050


def test_every_arm_capsule_paired_with_the_puck_is_paired_with_the_bracket():
    """A capsule with no pairs is inert.  The bracket must mirror the puck."""

    pairs = _raw()["self_collision"]["pairs"]
    puck = {
        other
        for pair in pairs
        for other in pair
        if "mid360" in pair and other != "mid360"
    }
    bracket = {
        other
        for pair in pairs
        for other in pair
        if "mid360_bracket" in pair and other != "mid360_bracket"
    }
    assert puck, "mid360 lost all of its self-collision pairs"
    assert puck == bracket

    guard = _guard()
    assert len([pair for pair in guard.pairs if "mid360" in pair]) == 14
    assert len([pair for pair in guard.pairs if "mid360_bracket" in pair]) == 14


# --------------------------------------------------------------------------
# 3.  The corrected geometry actually changes what the planner rejects.
# --------------------------------------------------------------------------


def _mesh_backend_clear(_joints) -> CollisionResult:
    """Stand in for the pinocchio mesh backend, reporting the arm clear.

    This is not a convenience: pinocchio is unimportable in CI, and passing a
    ``self_collision_checker`` is the ONLY way to reach the production branch at
    ``pointcloud.py`` where ``check_state`` calls
    ``_check_self_collision(supplemental_only=True)``.  In that branch a head
    capsule is checked only if it carries ``supplemental_self_collision``.  With
    no backend every configured pair is checked and a missing flag would go
    unnoticed -- the exact way this codebase keeps shipping fail-open gates.
    Reporting the arm clear also removes the arm-owned pairs (``mount``/
    ``wrist``) from the answer, so a rejection below can only be the head.
    """

    return CollisionResult(True, "mesh backend clear")


def _pointcloud_checker(model: RobotCollisionModel) -> PointCloudCollisionChecker:
    chain = KinematicChain.from_urdf(URDF, "piper_base_link", "piper_gripper_base")
    checker = PointCloudCollisionChecker(
        chain=chain,
        model=model,
        frame_provider=chain.link_transforms,
        config=PointCloudCollisionConfig(
            clearance=model.scene_clearance_m,
            point_radius=model.point_radius_m,
        ),
        now_fn=lambda: 10.0,
        self_collision_checker=_mesh_backend_clear,
    )
    # A distant floor patch: perception must be fresh and populated or
    # check_state refuses to answer, and nothing here may be rejected because
    # of the scene.
    floor = np.asarray([
        (2.0 + 0.01 * i, 2.0 + 0.01 * j, -2.0)
        for i in range(8)
        for j in range(8)
    ])
    checker.update_scene(floor, stamp_s=10.0)
    return checker


def _superseded_model() -> RobotCollisionModel:
    """The pre-fix capsule (a8ecd46's r=0.0325 at the old, wrong pose)."""

    raw = _raw()
    for item in raw["capsules"]:
        if item["name"] == "mid360":
            item["radius"] = 0.0325
            item["start_offset"] = [0.13036, 0.0, -0.01233]
            item["end_offset"] = [0.15259, 0.0, 0.04875]
        if item["name"] == "mid360_bracket":
            item["radius"] = 1e-6
    return RobotCollisionModel.from_mapping(raw)


def test_planner_gate_rejects_a_pose_inside_the_real_sensor_and_keeps_the_control():
    """The recorded strike pose: accepted before, rejected now; control passes."""

    superseded = _pointcloud_checker(_superseded_model())
    assert superseded.check_state(np.asarray(STRIKE)).valid, (
        "fixture drift: STRIKE must be a pose the superseded model accepted"
    )

    checker = _pointcloud_checker(RobotCollisionModel.from_mapping(_raw()))
    strike = checker.check_state(np.asarray(STRIKE))
    assert not strike.valid
    assert strike.kind == "self"
    assert any(name.startswith("mid360") for name in strike.capsules)

    control = checker.check_state(np.asarray(CONTROL))
    assert control.valid, control.reason


def test_strike_pose_is_physically_inside_the_sensor_body():
    """Independent of any capsule: STRIKE penetrates the bare 42 mm hardware.

    Recomputed against the raw cylinder, so it stays true even if every capsule
    radius is retuned.  This is the receipt that the rejection above is a true
    positive and not conservatism.
    """

    chain = KinematicChain.from_urdf(URDF, "piper_base_link", "piper_gripper_base")
    origins = _urdf_origins()
    mount_xyz, _ = origins["piper_mount_joint"]
    sensor_xyz, sensor_rpy = origins["mid360_joint"]
    axis = _rotation_y(float(sensor_rpy[1])) @ np.asarray([0.0, 0.0, 1.0])
    half = 0.5 * SENSOR_LENGTH_M
    start = sensor_xyz - half * axis - mount_xyz
    end = sensor_xyz + half * axis - mount_xyz

    raw = _raw()
    tips = [
        item
        for item in raw["capsules"]
        if item["name"].startswith("finger_") and item["name"].endswith("_tip")
    ]
    frames = chain.link_transforms(np.asarray(STRIKE))
    penetration = []
    for item in tips:
        transform = frames[item["start_frame"]]
        tip_start = transform[:3, :3] @ np.asarray(item["start_offset"]) + transform[:3, 3]
        tip_end = transform[:3, :3] @ np.asarray(item["end_offset"]) + transform[:3, 3]
        distance = _segment_distance(start, end, tip_start, tip_end)
        penetration.append(distance - (SENSOR_RADIUS_M + item["radius"]))

    assert min(penetration) < 0.0, (
        "STRIKE no longer touches the bare sensor cylinder; re-record it"
    )
    # Measured 5.4 mm of metal-on-metal interference at this recorded pose.
    assert min(penetration) < -0.005


def test_superseded_capsule_reported_clearance_where_the_hardware_was_hit():
    """The exact fail-open: +25 mm of reported margin, 5 mm inside the sensor."""

    guard = _guard()
    assert _pair_margin(guard, np.asarray(STRIKE), "mid360") < 0.0

    raw = _raw()
    for item in raw["capsules"]:
        if item["name"] == "mid360":
            item["radius"] = 0.0325
            item["start_offset"] = [0.13036, 0.0, -0.01233]
            item["end_offset"] = [0.15259, 0.0, 0.04875]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "superseded.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        superseded = FixedSelfCollisionGuard(urdf_path=URDF, model_path=path)
    assert _pair_margin(superseded, np.asarray(STRIKE), "mid360") > 0.02


# --------------------------------------------------------------------------
# 4.  The operator's actual invariant: the CARRIED OBJECT, not just the arm.
# --------------------------------------------------------------------------


def _attached_checker(offset_base: np.ndarray) -> PointCloudCollisionChecker:
    """Hold a small payload whose points sit at ``offset_base`` in the base frame."""

    raw = _raw()
    model = RobotCollisionModel.from_mapping(raw)
    checker = _pointcloud_checker(model)
    grid = np.asarray([
        (x, y, z)
        for x in (-0.01, 0.0, 0.01)
        for y in (-0.01, 0.0, 0.01)
        for z in (-0.01, 0.0, 0.01)
    ])
    checker.update_attached_target(
        grid + offset_base,
        attachment_joints=np.asarray(CONTROL),
        allowed_contact_capsules=tuple(model.target_contact_capsules),
    )
    return checker


def test_carried_object_is_rejected_inside_the_lidar_even_when_the_arm_is_clear():
    """The reported accident: object in hand clips the Mid-360 on the way home.

    Before this change every head capsule had ``check_target: false`` and
    ``_check_attached_target_scene`` only ever compares the payload against the
    perceived SCENE cloud -- never against robot geometry -- so this exact
    motion was structurally unpreventable.
    """

    guard = _guard()
    assert guard.check_state(np.asarray(CONTROL)).valid, "arm itself must be clear"

    mount = np.asarray([0.128526599, 0.0, 0.081794287])
    sensor_centre = np.asarray([0.330, 0.0, 0.140]) - mount

    carried = _attached_checker(sensor_centre)
    result = carried.check_state(np.asarray(CONTROL))
    assert not result.valid
    assert result.kind == "target"
    assert any(name.startswith("mid360") for name in result.capsules)

    # Control: the same payload held well clear of the head stays valid, so the
    # rejection above is the geometry and not a blanket veto.
    clear_of_head = sensor_centre + np.asarray([0.0, 0.45, 0.0])
    assert _attached_checker(clear_of_head).check_state(np.asarray(CONTROL)).valid


def test_attached_object_threshold_does_not_silently_include_scene_clearance():
    """``_check_target_capsule`` uses radius + point_radius + target_clearance.

    It deliberately does NOT add ``config.clearance``, and no production call
    site passes ``clearance=``, so the payload is held to 53 mm against the
    Mid-360 while the arm is held to 60 mm.  Pinned so the asymmetry is a
    decision rather than a surprise.
    """

    raw = _raw()
    model = RobotCollisionModel.from_mapping(raw)
    puck = next(item for item in model.capsules if item.name == "mid360")
    assert puck.radius + model.point_radius_m == pytest.approx(0.053)
    assert model.scene_clearance_m == pytest.approx(0.01)
