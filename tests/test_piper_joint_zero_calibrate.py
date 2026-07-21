from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from z_manip.kinematics import KinematicChain


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_joint_zero_calibrate.py"
SPEC = importlib.util.spec_from_file_location("piper_joint_zero_calibrate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def chain(tmp_path: Path) -> KinematicChain:
    axes = ("0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0")
    origins = ("0 0 .1", ".18 0 0", ".16 0 0", ".08 0 .03", ".08 0 0", ".06 0 0")
    links = ['<link name="piper_base_link"/>']
    joints = []
    parent = "piper_base_link"
    for index, (axis, origin) in enumerate(zip(axes, origins), 1):
        child = f"link{index}"
        links.append(f'<link name="{child}"/>')
        joints.append(
            f'<joint name="piper_joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{origin}" rpy="0 0 0"/><axis xyz="{axis}"/>'
            '<limit lower="-3" upper="3" effort="1" velocity="2"/>'
            '</joint>'
        )
        parent = child
    links.append('<link name="piper_gripper_base"/>')
    joints.append(
        f'<joint name="tip" type="fixed"><parent link="{parent}"/>'
        '<child link="piper_gripper_base"/><origin xyz=".08 0 0" rpy="0 0 0"/></joint>'
    )
    path = tmp_path / "robot.urdf"
    path.write_text('<robot name="test">' + ''.join(links + joints) + '</robot>')
    return KinematicChain.from_urdf(path, "piper_base_link", "piper_gripper_base")


def inputs(tmp_path: Path):
    model = chain(tmp_path)
    platform_arm = np.eye(4)
    platform_arm[:3, 3] = (0.06, 0.0, 0.07)
    platform_target = np.eye(4)
    platform_target[:3, 3] = (0.35, -0.1, 0.28)
    tip_camera = np.eye(4)
    tip_camera[:3, 3] = (-0.08, 0.0, 0.03)
    true_offset = np.radians((0.35, -0.45, 0.25, 0.3, -0.2, 0.15))
    samples = []
    rng = np.random.default_rng(4)
    for _ in range(24):
        q = rng.uniform(-0.9, 0.9, 6)
        camera_target = (
            np.linalg.inv(tip_camera)
            @ np.linalg.inv(model.forward(q + true_offset))
            @ np.linalg.inv(platform_arm)
            @ platform_target
        )
        samples.append({
            "joint_positions_rad": q.tolist(),
            "joint_names": list(model.joint_names),
            "base_from_tip": model.forward(q).tolist(),
            "camera_from_target": camera_target.tolist(),
            "safety_evidence": {
                "camera_read_only": True,
                "can_read_only": True,
                "can_tx_packet_delta": 0,
            },
        })
    dataset = {
        "schema": "z_manip.piper_hand_eye_samples.v1",
        "synthetic": False,
        "base_frame": model.base_link,
        "tip_link": model.tip_link,
        "camera_frame": "camera_color_optical_frame",
        "target_frame": "board",
        "samples": samples,
    }
    hand_eye = {
        "schema": "z_manip.piper_camera_calibration.v1",
        "calibrated": True,
        "synthetic": False,
        "tip_link": model.tip_link,
        "camera_frame": "camera_color_optical_frame",
        "tip_from_camera": tip_camera.tolist(),
        "dataset_sha256": "hand-eye-independent-source",
    }
    mount = {
        "schema": "z_manip.piper_mount_calibration.v1",
        "calibrated": True,
        "platform_frame": "base",
        "arm_base_frame": model.base_link,
        "platform_from_arm_base": platform_arm.tolist(),
        "input_sha256": {"samples": "mount-independent-source"},
    }
    anchor = {
        "schema": "z_manip.platform_target_anchor.v1",
        "platform_frame": "base",
        "target_frame": "board",
        "example_only": False,
        "independently_measured": True,
        "rigidly_attached_to_platform": True,
        "platform_from_target": platform_target.tolist(),
    }
    return model, dataset, hand_eye, mount, anchor, true_offset


def test_recovers_joint_zeros_and_passes_held_out_validation(tmp_path):
    model, dataset, hand_eye, mount, anchor, expected = inputs(tmp_path)
    report = MODULE.solve_joint_zeros(dataset, hand_eye, mount, anchor, model)
    actual = np.asarray(report["joint_zero_offsets_rad"])
    assert report["ready_for_manual_review"] is True
    assert report["observability"]["rank"] == 6
    assert report["urdf_modified"] is False
    assert report["motion_commands_published"] == 0
    np.testing.assert_allclose(actual, expected, atol=math_radians(0.05))
    assert report["quality"]["calibrated_validation"]["translation_rmse_m"] < 0.001
    assert report["provenance"]["independent_dataset_verified"] is True


def math_radians(value: float) -> float:
    return value * np.pi / 180.0


def test_rejects_example_anchor_or_too_few_samples(tmp_path):
    model, dataset, hand_eye, mount, anchor, _ = inputs(tmp_path)
    anchor["example_only"] = True
    with pytest.raises(ValueError, match="independent real"):
        MODULE.solve_joint_zeros(dataset, hand_eye, mount, anchor, model)


def test_rejects_reused_mount_dataset(tmp_path):
    model, dataset, hand_eye, mount, anchor, _ = inputs(tmp_path)
    mount["input_sha256"]["samples"] = MODULE._hash(dataset)
    with pytest.raises(ValueError, match="new dataset independent"):
        MODULE.solve_joint_zeros(dataset, hand_eye, mount, anchor, model)


def test_rejects_unexcited_or_nonzero_tx_dataset(tmp_path):
    model, dataset, hand_eye, mount, anchor, _ = inputs(tmp_path)
    for sample in dataset["samples"]:
        sample["joint_positions_rad"][5] = 0.0
        sample["base_from_tip"] = model.forward(
            np.asarray(sample["joint_positions_rad"]),
        ).tolist()
    with pytest.raises(ValueError, match="independently excite"):
        MODULE.solve_joint_zeros(dataset, hand_eye, mount, anchor, model)

    model, dataset, hand_eye, mount, anchor, _ = inputs(tmp_path)
    dataset["samples"][0]["safety_evidence"]["can_tx_packet_delta"] = 1
    with pytest.raises(ValueError, match="zero-TX evidence"):
        MODULE.solve_joint_zeros(dataset, hand_eye, mount, anchor, model)
    anchor["example_only"] = False
    dataset["samples"] = dataset["samples"][:12]
    with pytest.raises(ValueError, match="at least 16"):
        MODULE.solve_joint_zeros(dataset, hand_eye, mount, anchor, model)


def test_solver_has_no_robot_transport():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "import rclpy",
        "subprocess",
        "socket.",
        "piper_sdk",
        "cansend",
        "/piper/cmd",
    ):
        assert forbidden not in source
