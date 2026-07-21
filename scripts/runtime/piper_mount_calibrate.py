#!/usr/bin/env python3
"""Estimate the Go2W-body-to-PiPER-base mounting transform offline.

This solver deliberately requires an independently measured, rigid body target.
A board that is merely fixed in the room does not make the body-to-arm transform
observable: body-to-board and body-to-arm would both be unknown.  The program
opens no ROS, network, CAN, or actuator transport and never edits the URDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from z_manip.kinematics import fixed_transform_from_urdf, rotation_log


SCHEMA = "z_manip.piper_mount_calibration.v1"
ANCHOR_SCHEMA = "z_manip.platform_target_anchor.v1"
SAMPLES_SCHEMA = "z_manip.piper_hand_eye_samples.v1"
CALIBRATION_SCHEMA = "z_manip.piper_camera_calibration.v1"


def rigid_transform(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(f"{label} is not a right-handed rigid transform")
    return matrix


def _hash(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(
        np.stack([pose[:3, :3] for pose in poses]),
    ).mean().as_matrix()
    result[:3, 3] = np.mean(
        np.stack([pose[:3, 3] for pose in poses]),
        axis=0,
    )
    return result


def _pose_residuals(
    poses: list[np.ndarray],
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray([
        np.linalg.norm(pose[:3, 3] - reference[:3, 3])
        for pose in poses
    ])
    rotation = np.asarray([
        np.linalg.norm(rotation_log(reference[:3, :3].T @ pose[:3, :3]))
        for pose in poses
    ])
    return translation, rotation


def solve_mount(
    dataset: dict[str, Any],
    hand_eye: dict[str, Any],
    anchor: dict[str, Any],
    *,
    nominal_platform_from_arm: np.ndarray | None = None,
    min_samples: int = 8,
    max_translation_rmse_m: float = 0.010,
    max_rotation_rmse_rad: float = math.radians(2.0),
    max_anchor_translation_uncertainty_m: float = 0.003,
    max_anchor_rotation_uncertainty_rad: float = math.radians(1.0),
    max_nominal_translation_delta_m: float = 0.100,
    max_nominal_rotation_delta_rad: float = math.radians(15.0),
) -> dict[str, Any]:
    if dataset.get("schema") != SAMPLES_SCHEMA:
        raise ValueError("unsupported hand-eye sample schema")
    if hand_eye.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported hand-eye calibration schema")
    if anchor.get("schema") != ANCHOR_SCHEMA:
        raise ValueError("unsupported platform target anchor schema")
    if anchor.get("example_only") is True:
        raise ValueError("replace the example anchor with a real measured anchor")
    if dataset.get("synthetic") is not False or hand_eye.get("synthetic") is not False:
        raise ValueError("real mount calibration requires explicitly non-synthetic inputs")
    if hand_eye.get("calibrated") is not True:
        raise ValueError("hand-eye calibration has not passed its quality gate")
    if anchor.get("rigidly_attached_to_platform") is not True:
        raise ValueError("target must be rigidly attached to the robot platform")
    if anchor.get("independently_measured") is not True:
        raise ValueError(
            "platform_from_target must come from CAD, a metrology jig, or an "
            "external tracker; a fixed external board alone is unobservable"
        )
    method = str(anchor.get("measurement_method", "")).strip()
    if method not in {"cad_bracket", "metrology_jig", "external_tracker"}:
        raise ValueError("anchor measurement_method lacks independent metric provenance")

    arm_frame = str(dataset.get("base_frame", "")).strip()
    tip_link = str(dataset.get("tip_link", "")).strip()
    camera_frame = str(dataset.get("camera_frame", "")).strip()
    target_frame = str(dataset.get("target_frame", "")).strip()
    platform_frame = str(anchor.get("platform_frame", "")).strip()
    if not all((arm_frame, tip_link, camera_frame, target_frame, platform_frame)):
        raise ValueError("all platform, arm, tip, camera, and target frames are required")
    if (
        hand_eye.get("tip_link") != tip_link
        or hand_eye.get("camera_frame") != camera_frame
        or anchor.get("target_frame") != target_frame
    ):
        raise ValueError("sample, hand-eye, and platform-anchor frames do not match")

    platform_from_target = rigid_transform(
        anchor.get("platform_from_target"),
        "platform_from_target",
    )
    tip_from_camera = rigid_transform(
        hand_eye.get("tip_from_camera"),
        "tip_from_camera",
    )
    try:
        anchor_translation_uncertainty_m = float(
            anchor["translation_uncertainty_m"],
        )
        anchor_rotation_uncertainty_rad = float(
            anchor["rotation_uncertainty_rad"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("anchor uncertainty fields are required") from error
    if (
        not math.isfinite(anchor_translation_uncertainty_m)
        or not math.isfinite(anchor_rotation_uncertainty_rad)
        or anchor_translation_uncertainty_m < 0.0
        or anchor_rotation_uncertainty_rad < 0.0
    ):
        raise ValueError("anchor uncertainties must be finite and non-negative")

    samples = dataset.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("mount calibration requires at least three pose samples")
    estimates: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{index}] must be an object")
        arm_from_tip = rigid_transform(
            sample.get("base_from_tip"),
            f"samples[{index}].base_from_tip",
        )
        camera_from_target = rigid_transform(
            sample.get("camera_from_target"),
            f"samples[{index}].camera_from_target",
        )
        arm_from_target = arm_from_tip @ tip_from_camera @ camera_from_target
        estimates.append(platform_from_target @ np.linalg.inv(arm_from_target))

    platform_from_arm = _mean_pose(estimates)
    translation_errors, rotation_errors = _pose_residuals(
        estimates,
        platform_from_arm,
    )
    quality = {
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation_errors ** 2))),
        "rotation_max_rad": float(np.max(rotation_errors)),
        "per_sample_translation_error_m": translation_errors.tolist(),
        "per_sample_rotation_error_rad": rotation_errors.tolist(),
        "anchor_translation_uncertainty_m": anchor_translation_uncertainty_m,
        "anchor_rotation_uncertainty_rad": anchor_rotation_uncertainty_rad,
    }

    nominal_delta: dict[str, Any] | None = None
    nominal_review_passes = nominal_platform_from_arm is None
    if nominal_platform_from_arm is not None:
        nominal = rigid_transform(
            nominal_platform_from_arm,
            "nominal_platform_from_arm",
        )
        delta = np.linalg.inv(nominal) @ platform_from_arm
        translation_delta = float(np.linalg.norm(delta[:3, 3]))
        rotation_delta = float(np.linalg.norm(rotation_log(delta[:3, :3])))
        nominal_review_passes = bool(
            translation_delta <= max_nominal_translation_delta_m
            and rotation_delta <= max_nominal_rotation_delta_rad
        )
        nominal_delta = {
            "nominal_platform_from_arm": nominal.tolist(),
            "nominal_from_calibrated_delta": delta.tolist(),
            "translation_delta_m": translation_delta,
            "rotation_delta_rad": rotation_delta,
            "within_manual_review_envelope": nominal_review_passes,
        }

    quality_passes = bool(
        len(samples) >= min_samples
        and quality["translation_rmse_m"] <= max_translation_rmse_m
        and quality["rotation_rmse_rad"] <= max_rotation_rmse_rad
        and anchor_translation_uncertainty_m
        <= max_anchor_translation_uncertainty_m
        and anchor_rotation_uncertainty_rad
        <= max_anchor_rotation_uncertainty_rad
    )
    calibrated = bool(quality_passes and nominal_review_passes)
    fingerprint = _hash({
        "samples_sha256": _hash(dataset),
        "hand_eye_sha256": _hash(hand_eye),
        "anchor_sha256": _hash(anchor),
    })
    return {
        "schema": SCHEMA,
        "calibrated": calibrated,
        "read_only": True,
        "motion_commands_published": 0,
        "urdf_modified": False,
        "urdf_update_requires_manual_approval": True,
        "calibration_id": f"piper-mount-{fingerprint[:16]}",
        "platform_frame": platform_frame,
        "arm_base_frame": arm_frame,
        "tip_link": tip_link,
        "camera_frame": camera_frame,
        "target_frame": target_frame,
        "sample_count": len(samples),
        "platform_from_arm_base": platform_from_arm.tolist(),
        "quality": quality,
        "quality_limits": {
            "min_samples": min_samples,
            "max_translation_rmse_m": max_translation_rmse_m,
            "max_rotation_rmse_rad": max_rotation_rmse_rad,
            "max_anchor_translation_uncertainty_m": (
                max_anchor_translation_uncertainty_m
            ),
            "max_anchor_rotation_uncertainty_rad": (
                max_anchor_rotation_uncertainty_rad
            ),
            "max_nominal_translation_delta_m": max_nominal_translation_delta_m,
            "max_nominal_rotation_delta_rad": max_nominal_rotation_delta_rad,
        },
        "anchor": {
            "measurement_method": method,
            "independently_measured": True,
            "rigidly_attached_to_platform": True,
        },
        "nominal_comparison": nominal_delta,
        "input_sha256": {
            "samples": _hash(dataset),
            "hand_eye": _hash(hand_eye),
            "anchor": _hash(anchor),
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--hand-eye", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--platform-link", default="base")
    parser.add_argument("--arm-base-link", default="piper_base_link")
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    args = _arguments()
    destination = args.output.expanduser().resolve()
    try:
        nominal = None
        if args.urdf is not None:
            nominal = fixed_transform_from_urdf(
                args.urdf.expanduser().resolve(),
                args.platform_link,
                args.arm_base_link,
            )
        report = solve_mount(
            _read(args.samples),
            _read(args.hand_eye),
            _read(args.anchor),
            nominal_platform_from_arm=nominal,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "calibrated": False,
            "read_only": True,
            "motion_commands_published": 0,
            "urdf_modified": False,
            "urdf_update_requires_manual_approval": True,
            "error": f"{type(error).__name__}: {error}",
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["calibrated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
