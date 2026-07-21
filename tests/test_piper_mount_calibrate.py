from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_mount_calibrate.py"
SPEC = importlib.util.spec_from_file_location("piper_mount_calibrate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pose(xyz, rotvec=(0.0, 0.0, 0.0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = xyz
    return value


def inputs():
    platform_from_arm = pose((0.061, -0.004, 0.071), (0.01, -0.02, 0.03))
    platform_from_target = pose((0.18, 0.0, 0.22), (0.0, 0.0, 1.1))
    tip_from_camera = pose((-0.082, 0.001, 0.033), (0.02, -0.01, 0.04))
    arm_from_target = np.linalg.inv(platform_from_arm) @ platform_from_target
    samples = []
    for index in range(12):
        arm_from_tip = pose(
            (0.18 + 0.01 * index, -0.08 + 0.012 * index, 0.25 + 0.005 * index),
            (0.03 * index, -0.02 * index, 0.015 * index),
        )
        camera_from_target = (
            np.linalg.inv(tip_from_camera)
            @ np.linalg.inv(arm_from_tip)
            @ arm_from_target
        )
        samples.append({
            "base_from_tip": arm_from_tip.tolist(),
            "camera_from_target": camera_from_target.tolist(),
        })
    dataset = {
        "schema": MODULE.SAMPLES_SCHEMA,
        "synthetic": False,
        "base_frame": "piper_base_link",
        "tip_link": "piper_gripper_base",
        "camera_frame": "camera_color_optical_frame",
        "target_frame": "piper_charuco_board",
        "samples": samples,
    }
    hand_eye = {
        "schema": MODULE.CALIBRATION_SCHEMA,
        "synthetic": False,
        "calibrated": True,
        "tip_link": "piper_gripper_base",
        "camera_frame": "camera_color_optical_frame",
        "tip_from_camera": tip_from_camera.tolist(),
    }
    anchor = {
        "schema": MODULE.ANCHOR_SCHEMA,
        "platform_frame": "base",
        "target_frame": "piper_charuco_board",
        "rigidly_attached_to_platform": True,
        "independently_measured": True,
        "measurement_method": "cad_bracket",
        "translation_uncertainty_m": 0.001,
        "rotation_uncertainty_rad": 0.005,
        "platform_from_target": platform_from_target.tolist(),
    }
    return dataset, hand_eye, anchor, platform_from_arm


def test_recovers_mount_without_editing_or_transport():
    dataset, hand_eye, anchor, expected = inputs()
    report = MODULE.solve_mount(
        dataset,
        hand_eye,
        anchor,
        nominal_platform_from_arm=expected,
    )
    actual = np.asarray(report["platform_from_arm_base"])
    assert report["calibrated"] is True
    assert report["urdf_modified"] is False
    assert report["motion_commands_published"] == 0
    assert np.max(np.abs(actual - expected)) < 1e-10
    assert report["quality"]["translation_rmse_m"] < 1e-10
    assert report["quality"]["rotation_rmse_rad"] < 1e-10


def test_rejects_unobservable_external_board():
    dataset, hand_eye, anchor, _ = inputs()
    anchor["independently_measured"] = False
    anchor["rigidly_attached_to_platform"] = False
    with pytest.raises(ValueError, match="rigidly attached"):
        MODULE.solve_mount(dataset, hand_eye, anchor)


def test_rejects_identity_example_template():
    dataset, hand_eye, anchor, _ = inputs()
    anchor["example_only"] = True
    with pytest.raises(ValueError, match="real measured anchor"):
        MODULE.solve_mount(dataset, hand_eye, anchor)


def test_rejects_imprecise_anchor_and_large_nominal_change():
    dataset, hand_eye, anchor, expected = inputs()
    anchor["translation_uncertainty_m"] = 0.02
    nominal = expected.copy()
    nominal[0, 3] += 0.2
    report = MODULE.solve_mount(
        dataset,
        hand_eye,
        anchor,
        nominal_platform_from_arm=nominal,
    )
    assert report["calibrated"] is False
    assert report["nominal_comparison"]["within_manual_review_envelope"] is False


def test_solver_source_has_no_robot_transport():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "import rclpy",
        "python-can",
        "socket.",
        "subprocess",
        "/piper/cmd",
        "/piper/joint_trajectory",
    ):
        assert forbidden not in source
