#!/usr/bin/env python3
"""Solve an offline PiPER eye-in-hand calibration with fail-closed quality gates.

The input contains synchronized, already-recorded robot FK and calibration-board
poses. This program has no ROS, CAN, or actuator transport. It writes
``calibrated: true`` only when pose count, rotational diversity, and fixed-board
consistency all satisfy the configured limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def rigid_transform(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(f"{label} rotation is not right-handed orthonormal")
    return matrix


def motion_diversity(base_from_tip: list[np.ndarray]) -> dict[str, object]:
    axes: list[np.ndarray] = []
    angles: list[float] = []
    for first_index, first in enumerate(base_from_tip[:-1]):
        for second in base_from_tip[first_index + 1:]:
            vector = Rotation.from_matrix(
                first[:3, :3].T @ second[:3, :3],
            ).as_rotvec()
            angle = float(np.linalg.norm(vector))
            angles.append(angle)
            if angle >= math.radians(5.0):
                axes.append(vector / angle)
    if axes:
        singular_values = np.linalg.svd(np.stack(axes), compute_uv=False)
        reference = max(float(singular_values[0]), 1e-12)
        axis_rank = int(np.count_nonzero(singular_values / reference >= 0.1))
    else:
        singular_values = np.zeros(3)
        axis_rank = 0
    return {
        "max_pair_rotation_rad": max(angles, default=0.0),
        "rotation_axis_rank": axis_rank,
        "rotation_axis_singular_values": singular_values.tolist(),
    }


def fixed_target_residuals(
    base_from_tip: list[np.ndarray],
    camera_from_target: list[np.ndarray],
    tip_from_camera: np.ndarray,
) -> dict[str, float]:
    base_from_targets = [
        base_tip @ tip_from_camera @ camera_target
        for base_tip, camera_target in zip(base_from_tip, camera_from_target)
    ]
    translations = np.stack([pose[:3, 3] for pose in base_from_targets])
    mean_translation = translations.mean(axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)
    rotations = Rotation.from_matrix(
        np.stack([pose[:3, :3] for pose in base_from_targets]),
    )
    mean_rotation = rotations.mean()
    rotation_errors = (mean_rotation.inv() * rotations).magnitude()
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation_errors ** 2))),
        "rotation_max_rad": float(np.max(rotation_errors)),
    }


def solve_hand_eye(
    dataset: dict[str, object],
    *,
    min_samples: int = 8,
    min_rotation_span_rad: float = math.radians(20.0),
    max_translation_rmse_m: float = 0.010,
    max_rotation_rmse_rad: float = math.radians(2.0),
) -> dict[str, object]:
    samples = dataset.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("hand-eye dataset requires at least three pose samples")
    tip_link = str(dataset.get("tip_link", "")).strip()
    camera_frame = str(dataset.get("camera_frame", "")).strip()
    target_frame = str(dataset.get("target_frame", "")).strip()
    synthetic = dataset.get("synthetic")
    if not tip_link or not camera_frame or not target_frame:
        raise ValueError("tip_link, camera_frame, and target_frame are required")
    if not isinstance(synthetic, bool):
        raise ValueError("dataset synthetic provenance must be explicitly true or false")

    base_from_tip: list[np.ndarray] = []
    camera_from_target: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        base_from_tip.append(rigid_transform(
            sample.get("base_from_tip"),
            f"samples[{index}].base_from_tip",
        ))
        camera_from_target.append(rigid_transform(
            sample.get("camera_from_target"),
            f"samples[{index}].camera_from_target",
        ))

    diversity = motion_diversity(base_from_tip)
    candidates: list[dict[str, object]] = []
    rotations_gripper = [pose[:3, :3] for pose in base_from_tip]
    translations_gripper = [pose[:3, 3].reshape(3, 1) for pose in base_from_tip]
    rotations_target = [pose[:3, :3] for pose in camera_from_target]
    translations_target = [pose[:3, 3].reshape(3, 1) for pose in camera_from_target]
    for name, method in METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                rotations_gripper,
                translations_gripper,
                rotations_target,
                translations_target,
                method=method,
            )
            tip_from_camera = np.eye(4)
            tip_from_camera[:3, :3] = np.asarray(rotation, dtype=float)
            tip_from_camera[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
            tip_from_camera = rigid_transform(
                tip_from_camera,
                f"{name} tip_from_camera",
            )
            residuals = fixed_target_residuals(
                base_from_tip,
                camera_from_target,
                tip_from_camera,
            )
            score = (
                residuals["translation_rmse_m"] / max_translation_rmse_m
                + residuals["rotation_rmse_rad"] / max_rotation_rmse_rad
            )
            candidates.append({
                "method": name,
                "tip_from_camera": tip_from_camera,
                "score": score,
                **residuals,
            })
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            candidates.append({
                "method": name,
                "error": f"{type(error).__name__}: {error}",
            })
    valid_candidates = [
        candidate
        for candidate in candidates
        if "tip_from_camera" in candidate
    ]
    if not valid_candidates:
        raise ValueError("no hand-eye method produced a valid rigid transform")
    best = min(valid_candidates, key=lambda candidate: float(candidate["score"]))
    calibrated = bool(
        len(samples) >= min_samples
        and float(diversity["max_pair_rotation_rad"]) >= min_rotation_span_rad
        and int(diversity["rotation_axis_rank"]) >= 2
        and float(best["translation_rmse_m"]) <= max_translation_rmse_m
        and float(best["rotation_rmse_rad"]) <= max_rotation_rmse_rad
    )
    canonical_dataset = json.dumps(
        dataset,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dataset_hash = hashlib.sha256(canonical_dataset).hexdigest()
    calibration_id = f"piper-hand-eye-{dataset_hash[:16]}"
    candidate_report = []
    for candidate in candidates:
        candidate_report.append({
            key: value
            for key, value in candidate.items()
            if key != "tip_from_camera"
        })
    return {
        "schema": "z_manip.piper_camera_calibration.v1",
        "calibrated": calibrated,
        "synthetic": synthetic,
        "calibration_id": calibration_id,
        "mount_type": "eye_in_hand",
        "tip_link": tip_link,
        "camera_frame": camera_frame,
        "target_frame": target_frame,
        "sample_count": len(samples),
        "method": best["method"],
        "tip_from_camera": np.asarray(best["tip_from_camera"]).tolist(),
        "quality": {
            "translation_rmse_m": best["translation_rmse_m"],
            "translation_max_m": best["translation_max_m"],
            "rotation_rmse_rad": best["rotation_rmse_rad"],
            "rotation_max_rad": best["rotation_max_rad"],
            **diversity,
        },
        "quality_limits": {
            "min_samples": min_samples,
            "min_rotation_span_rad": min_rotation_span_rad,
            "min_rotation_axis_rank": 2,
            "max_translation_rmse_m": max_translation_rmse_m,
            "max_rotation_rmse_rad": max_rotation_rmse_rad,
        },
        "dataset_sha256": dataset_hash,
        "candidate_methods": candidate_report,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--min-rotation-span-deg", type=float, default=20.0)
    parser.add_argument("--max-translation-rmse-m", type=float, default=0.010)
    parser.add_argument("--max-rotation-rmse-deg", type=float, default=2.0)
    values = parser.parse_args()
    if (
        values.min_samples < 3
        or not 0.0 < values.min_rotation_span_deg < 180.0
        or values.max_translation_rmse_m <= 0.0
        or not 0.0 < values.max_rotation_rmse_deg < 180.0
    ):
        parser.error("invalid hand-eye quality limits")
    return values


def main() -> int:
    args = _arguments()
    dataset = json.loads(args.samples.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError("hand-eye dataset root must be an object")
    report = solve_hand_eye(
        dataset,
        min_samples=args.min_samples,
        min_rotation_span_rad=math.radians(args.min_rotation_span_deg),
        max_translation_rmse_m=args.max_translation_rmse_m,
        max_rotation_rmse_rad=math.radians(args.max_rotation_rmse_deg),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["calibrated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
