from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_hand_eye_calibrate.py"
SPEC = importlib.util.spec_from_file_location("piper_hand_eye_calibrate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CALIBRATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATE)


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


def test_synthetic_eye_in_hand_is_recovered():
    tip_from_camera = transform((0.03, -0.18, 0.07), (-0.045, 0.015, 0.020))
    base_from_target = transform((0.1, -0.2, 0.3), (0.55, -0.04, 0.18))
    samples = []
    axes = (
        (1.0, 0.2, 0.1),
        (0.1, 1.0, 0.3),
        (0.2, 0.1, 1.0),
        (1.0, -0.5, 0.4),
    )
    for index in range(16):
        axis = np.asarray(axes[index % len(axes)], dtype=float)
        axis /= np.linalg.norm(axis)
        angle = -0.55 + 1.1 * index / 15.0
        base_from_tip = transform(
            axis * angle,
            (
                0.15 + 0.04 * math.cos(index),
                0.03 * math.sin(index * 0.7),
                0.25 + 0.025 * math.cos(index * 0.4),
            ),
        )
        camera_from_target = (
            np.linalg.inv(tip_from_camera)
            @ np.linalg.inv(base_from_tip)
            @ base_from_target
        )
        samples.append({
            "base_from_tip": base_from_tip.tolist(),
            "camera_from_target": camera_from_target.tolist(),
        })
    report = CALIBRATE.solve_hand_eye({
        "synthetic": True,
        "tip_link": "piper_gripper_base",
        "camera_frame": "camera_color_optical_frame",
        "target_frame": "charuco_board",
        "samples": samples,
    })

    assert report["calibrated"] is True
    assert report["synthetic"] is True
    np.testing.assert_allclose(report["tip_from_camera"], tip_from_camera, atol=1e-6)
    assert report["quality"]["translation_rmse_m"] < 1e-8
    assert report["quality"]["rotation_rmse_rad"] < 1e-8


def test_degenerate_motion_fails_quality_gate():
    pose = np.eye(4).tolist()
    report = CALIBRATE.solve_hand_eye({
        "synthetic": True,
        "tip_link": "piper_gripper_base",
        "camera_frame": "camera_color_optical_frame",
        "target_frame": "charuco_board",
        "samples": [
            {"base_from_tip": pose, "camera_from_target": pose}
            for _ in range(8)
        ],
    })

    assert report["calibrated"] is False
    assert report["quality"]["rotation_axis_rank"] < 2


def test_calibration_source_has_no_robot_transport():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden_imports = {"rclpy", "can", "socket", "piper_sdk", "pyAgxArm"}
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert imports.isdisjoint(forbidden_imports)
