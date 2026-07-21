import json
import math
from pathlib import Path

import numpy as np
import pytest

from z_manip.collision import RobotCollisionModel
from z_manip.configuration import load_stack_config
from z_manip.planning.grasp_pipeline import tool_tip_pose


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
MESH_ROOT = URDF.parents[1] / "piper_ros/src/piper_description/meshes"


def _stack_config():
    return load_stack_config(
        ROOT / "configs/go2w_piper.json",
        environ={"Z_MANIP_ROBOT_URDF": str(URDF)},
    )


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
    )


def _capsule_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    axis = end - start
    alpha = np.clip(((points - start) @ axis) / float(axis @ axis), 0.0, 1.0)
    return np.linalg.norm(points - (start + alpha[:, None] * axis), axis=1)


def _capsule_bounds(start: object, end: object, radius: float) -> tuple[np.ndarray, np.ndarray]:
    endpoints = np.asarray((start, end), dtype=float)
    return endpoints.min(axis=0) - radius, endpoints.max(axis=0) + radius


def _stl_vertices(name: str) -> np.ndarray:
    payload = (MESH_ROOT / name).read_bytes()
    triangle_count = int.from_bytes(payload[80:84], "little")
    record = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    assert len(payload) == 84 + triangle_count * record.itemsize
    triangles = np.frombuffer(payload, dtype=record, count=triangle_count, offset=84)
    return np.asarray(triangles["vertices"], dtype=float).reshape((-1, 3))


def test_piper_tool_transform_maps_candidate_closing_axis_to_gripper_y():
    """The grasp convention is x=closing, y=binormal, z=approach.

    ``tool_tip_pose`` inverts the configured transform. Therefore Rz(+90 deg)
    in ``tool_from_tip`` yields Rz(-90 deg) for the IK tip: a finite +Y tip
    displacement becomes +X in the candidate frame, while +Z stays approach.
    This fixes the sign as well as the previously missing axis permutation.
    """

    config = _stack_config()
    tool_from_tip = np.asarray(config.grasp_plan.tool_from_tip)
    np.testing.assert_allclose(tool_from_tip[:3, :3], _rotation_z(math.pi / 2.0), atol=1e-12)
    np.testing.assert_allclose(tool_from_tip[:3, :3].T @ tool_from_tip[:3, :3], np.eye(3))
    assert np.linalg.det(tool_from_tip[:3, :3]) == pytest.approx(1.0)

    base_from_tip = tool_tip_pose(np.eye(4), tool_from_tip)
    displacement = base_from_tip[:3, :3] @ np.array((0.0, 1e-3, 0.0))
    np.testing.assert_allclose(displacement, (1e-3, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(base_from_tip[:3, 2], (0.0, 0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(config.tool_geometry.tip_closing_axis, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(config.tool_geometry.tip_approach_axis, (0.0, 0.0, 1.0))


def test_piper_contact_tcp_is_inside_the_measured_distal_pad_region():
    """The composed PiPER mesh spans gripper-base z=0.0593..0.1358 m.

    link7/link8 are rooted at z=0.1358 m and their local Y extent is
    -0.0765..0 m; Rx(+90 deg) maps that extent onto gripper-base Z. The TCP is
    the center of the distal half of this observed contact interval: it retains
    19.125 mm of pad before the mesh edge while biasing contact away from the
    palm. It is neither the shrinking fingertip boundary nor the unsupported
    old 0.17 m offset beyond the mesh.
    """

    config = _stack_config()
    tool_from_tip = np.asarray(config.grasp_plan.tool_from_tip)
    mesh_z_min = 0.1358 - 0.0765
    mesh_z_max = 0.1358
    pad_center = 0.5 * (mesh_z_min + mesh_z_max)
    distal_half_center = 0.5 * (pad_center + mesh_z_max)
    assert mesh_z_min == pytest.approx(0.0593)
    assert config.tool_geometry.finger_contact_z_interval_m == pytest.approx(
        (mesh_z_min, mesh_z_max),
    )
    assert config.tool_geometry.contact_tcp_z_m == pytest.approx(distal_half_center)
    assert tool_from_tip[2, 3] == pytest.approx(distal_half_center)
    assert mesh_z_max - distal_half_center == pytest.approx(0.019125)

    base_from_tip = tool_tip_pose(np.eye(4), tool_from_tip)
    np.testing.assert_allclose(base_from_tip[:3, 3], (0.0, 0.0, -distal_half_center))


def test_piper_open_finger_capsules_follow_y_and_cover_mesh_envelope():
    """Four tapered capsules per finger cover the real maximum-open STL.

    A single 31 mm capsule covered the mesh but exaggerated the distal pad and
    fingertip by roughly 16 mm. The four-piece proxy follows the measured taper
    while retaining a small positive enclosure margin at every STL vertex.
    """

    raw = json.loads((ROOT / "configs/piper_collision_capsules.json").read_text())
    model = RobotCollisionModel.from_mapping(raw)
    fingers = tuple(item for item in model.capsules if item.name.startswith("finger_"))
    assert len(fingers) == 8
    assert model.scene_clearance_m == pytest.approx(0.01)
    assert model.point_radius_m == pytest.approx(0.003)

    half_open = 0.5 * _stack_config().tool_geometry.collision_open_aperture_m
    joint7_rotation = _rotation_x(1.5708)
    open_slide_local = np.array((0.0, 0.0, half_open))
    open_slide = joint7_rotation @ open_slide_local
    np.testing.assert_allclose(open_slide, (0.0, -0.035, 0.0), atol=3e-7)

    mesh = _stl_vertices("link7.STL")
    joint8_rotation = _rotation_z(-3.1416) @ joint7_rotation
    joint_origin = np.array((0.0, 0.0, 0.1358))
    open_meshes = {
        "right": (mesh + open_slide_local) @ joint7_rotation.T + joint_origin,
        "left": (mesh + open_slide_local) @ joint8_rotation.T + joint_origin,
    }
    for side, open_mesh in open_meshes.items():
        side_capsules = tuple(item for item in fingers if item.name.startswith(f"finger_{side}_"))
        signed_distances = []
        for finger in side_capsules:
            signed_distances.append(
                _capsule_distance(
                    open_mesh,
                    np.asarray(finger.start_offset),
                    np.asarray(finger.end_offset),
                ) - finger.radius
            )
        union_distance = np.min(np.stack(signed_distances), axis=0)
        assert float(np.max(union_distance)) <= 0.0
        assert float(np.max(union_distance)) > -0.001
        tip = next(item for item in side_capsules if item.name.endswith("_tip"))
        assert tip.radius == pytest.approx(0.015)

    assert not any(name.endswith("_proximal") for name in model.target_contact_capsules)
    assert len(model.target_contact_capsules) == 6


def test_piper_palm_capsule_covers_mesh_without_invading_contact_tcp():
    """The gripper-base STL AABB is x=-.0400..0305, y=+/-0.0725,
    z=0..0.063 m. The full STL fits inside a 46 mm Y-axis capsule through the
    X/Z box center with centerline y=+/-0.042 m. Unlike the old z=0..0.10
    capsule, its axial envelope ends at z=0.0775 m and remains clear of the
    configured contact TCP.
    """

    raw = json.loads((ROOT / "configs/piper_collision_capsules.json").read_text())
    model = RobotCollisionModel.from_mapping(raw)
    palm = next(item for item in model.capsules if item.name == "palm")
    np.testing.assert_allclose(palm.start_offset, (-0.00475, -0.042, 0.0315))
    np.testing.assert_allclose(palm.end_offset, (-0.00475, 0.042, 0.0315))
    assert palm.radius == pytest.approx(0.046)

    palm_mesh = _stl_vertices("gripper_base.STL")
    mesh_distances = _capsule_distance(
        palm_mesh,
        np.asarray(palm.start_offset),
        np.asarray(palm.end_offset),
    )
    mesh_distance = float(np.max(mesh_distances))
    assert mesh_distance <= palm.radius
    assert palm.radius - mesh_distance > 1e-4

    proxy_min, proxy_max = _capsule_bounds(
        palm.start_offset,
        palm.end_offset,
        palm.radius,
    )
    assert np.all(proxy_min <= (-0.04, -0.0725, 0.0))
    assert np.all(proxy_max >= (0.0305, 0.0725, 0.063))
    assert proxy_min[1] == pytest.approx(-0.088)
    assert proxy_max[1] == pytest.approx(0.088)
    assert proxy_max[2] == pytest.approx(0.0775)

    contact_tcp = np.array(((0.0, 0.0, _stack_config().tool_geometry.contact_tcp_z_m),))
    tcp_distance = _capsule_distance(
        contact_tcp,
        np.asarray(palm.start_offset),
        np.asarray(palm.end_offset),
    )[0]
    point_radius_m = 0.005
    assert tcp_distance == pytest.approx(0.085307, abs=1e-6)
    assert tcp_distance > palm.radius + point_radius_m
