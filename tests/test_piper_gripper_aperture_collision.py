from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from z_manip.collision import RobotCollisionModel
from z_manip.collision.gripper_aperture import (
    collision_aperture_for_grasp,
    with_parallel_gripper_aperture,
)
from z_manip.configuration import load_stack_config
from z_manip.kinematics import KinematicChain


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
MESH_ROOT = URDF.parents[1] / "piper_ros/src/piper_description/meshes"


def _stl_triangles(name: str) -> np.ndarray:
    payload = (MESH_ROOT / name).read_bytes()
    triangle_count = int.from_bytes(payload[80:84], "little")
    record = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    assert len(payload) == 84 + triangle_count * record.itemsize
    triangles = np.frombuffer(
        payload,
        dtype=record,
        count=triangle_count,
        offset=84,
    )
    return np.asarray(triangles["vertices"], dtype=float)


def _triangle_surface_samples(triangles: np.ndarray, subdivisions: int = 12) -> np.ndarray:
    samples = []
    for first in range(subdivisions + 1):
        for second in range(subdivisions + 1 - first):
            alpha = first / subdivisions
            beta = second / subdivisions
            samples.append(
                alpha * triangles[:, 0]
                + beta * triangles[:, 1]
                + (1.0 - alpha - beta) * triangles[:, 2]
            )
    return np.concatenate(samples, axis=0)


def _finger_surface(side: str, aperture_m: float) -> np.ndarray:
    if side == "right":
        link = "piper_link7"
        mesh = "link7.STL"
        joints = np.asarray((0.5 * aperture_m,))
    elif side == "left":
        link = "piper_link8"
        mesh = "link8.STL"
        joints = np.asarray((-0.5 * aperture_m,))
    else:  # pragma: no cover - test helper misuse
        raise ValueError(side)
    chain = KinematicChain.from_urdf(URDF, "piper_gripper_base", link)
    transform = chain.forward(joints)
    triangles = _stl_triangles(mesh)
    world = triangles @ transform[:3, :3].T + transform[:3, 3]
    return _triangle_surface_samples(world)


def _segment_distance(points: np.ndarray, start: object, end: object) -> np.ndarray:
    first = np.asarray(start, dtype=float)
    second = np.asarray(end, dtype=float)
    axis = second - first
    denominator = float(axis @ axis)
    alpha = (
        np.zeros(len(points))
        if denominator <= 1e-20
        else np.clip(((points - first) @ axis) / denominator, 0.0, 1.0)
    )
    return np.linalg.norm(points - (first + alpha[:, None] * axis), axis=1)


def _finger_union_distance(
    points: np.ndarray,
    model: RobotCollisionModel,
    side: str,
) -> np.ndarray:
    capsules = tuple(
        capsule
        for capsule in model.capsules
        if capsule.name.startswith(f"finger_{side}_")
    )
    assert len(capsules) == 4
    return np.min(np.stack([
        _segment_distance(points, capsule.start_offset, capsule.end_offset)
        - capsule.radius
        for capsule in capsules
    ]), axis=0)


def _config_and_model():
    config = load_stack_config(
        ROOT / "configs/go2w_piper.json",
        environ={"Z_MANIP_ROBOT_URDF": str(URDF)},
    )
    raw = json.loads(config.collision_model_path.read_text())
    return config, RobotCollisionModel.from_mapping(raw)


@pytest.mark.parametrize("required_width_m", (0.03, 0.05, 0.068))
def test_plan_aperture_capsules_cover_real_stl_triangle_surfaces(required_width_m):
    """The width-aware proxy covers faces, not only the STL vertex set."""

    config, open_model = _config_and_model()
    tool = config.tool_geometry
    collision_aperture = collision_aperture_for_grasp(
        required_width_m,
        open_aperture_m=tool.collision_open_aperture_m,
        grasp_margin_m=tool.collision_grasp_margin_m,
    )
    model = with_parallel_gripper_aperture(
        open_model,
        open_aperture_m=tool.collision_open_aperture_m,
        aperture_m=collision_aperture,
        closing_axis=tool.tip_closing_axis,
    )

    for side in ("left", "right"):
        proxy_surface = _finger_surface(side, collision_aperture)
        proxy_gap = _finger_union_distance(proxy_surface, model, side)
        assert float(np.max(proxy_gap)) <= open_model.point_radius_m

        # The 4 mm total margin leaves each proxy finger 2 mm outside the
        # expected object-contact aperture. The 3 mm point support still covers
        # the true surface, including the measured ~0.9 mm mid/pad union seam.
        contact_surface = _finger_surface(side, required_width_m)
        contact_gap = _finger_union_distance(contact_surface, model, side)
        assert float(np.max(contact_gap)) <= open_model.point_radius_m


def test_plan_aperture_moves_both_sides_inward_and_preserves_contact_policy():
    config, open_model = _config_and_model()
    aperture = 0.034
    shifted = with_parallel_gripper_aperture(
        open_model,
        open_aperture_m=config.tool_geometry.collision_open_aperture_m,
        aperture_m=aperture,
        closing_axis=config.tool_geometry.tip_closing_axis,
    )
    inward = 0.5 * (
        config.tool_geometry.collision_open_aperture_m - aperture
    )
    original = {capsule.name: capsule for capsule in open_model.capsules}
    current = {capsule.name: capsule for capsule in shifted.capsules}
    for name, capsule in original.items():
        if name.startswith("finger_left_"):
            assert current[name].start_offset[1] == pytest.approx(
                capsule.start_offset[1] - inward,
            )
        elif name.startswith("finger_right_"):
            assert current[name].start_offset[1] == pytest.approx(
                capsule.start_offset[1] + inward,
            )
        else:
            assert current[name] == capsule
    assert shifted.target_contact_capsules == open_model.target_contact_capsules
    assert not any(
        name.endswith("_proximal")
        for name in shifted.target_contact_capsules
    )


def test_plan_aperture_requires_width_and_rejects_a_side_axis_mismatch():
    config, model = _config_and_model()
    tool = config.tool_geometry
    with pytest.raises(ValueError, match="requires required_width_m"):
        collision_aperture_for_grasp(
            None,
            open_aperture_m=tool.collision_open_aperture_m,
            grasp_margin_m=tool.collision_grasp_margin_m,
        )
    with pytest.raises(ValueError, match="disagrees with its side name"):
        with_parallel_gripper_aperture(
            model,
            open_aperture_m=tool.collision_open_aperture_m,
            aperture_m=0.04,
            closing_axis=(0.0, -1.0, 0.0),
        )
