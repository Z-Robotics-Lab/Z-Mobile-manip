#!/usr/bin/env python3
"""Estimate provisional PiPER joint zero offsets from held-out metric poses.

The body target, body-to-arm mount, and wrist hand-eye transform must already
be independently established.  The solver is offline, opens no robot transport,
and never changes the URDF.  Results remain provisional until physical low-speed
validation because hand-eye, mounting, compliance, and link errors can correlate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from z_manip.kinematics import KinematicChain, rotation_log


SCHEMA = "z_manip.piper_joint_zero_calibration.v1"


def rigid(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(f"{label} is not a rigid transform")
    return matrix


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _pose_error(predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    delta = np.linalg.inv(expected) @ predicted
    return np.concatenate((delta[:3, 3], rotation_log(delta[:3, :3])))


def _metrics(errors: np.ndarray) -> dict[str, float]:
    translation = np.linalg.norm(errors[:, :3], axis=1)
    rotation = np.linalg.norm(errors[:, 3:], axis=1)
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation ** 2))),
        "translation_max_m": float(np.max(translation)),
        "rotation_rmse_rad": float(np.sqrt(np.mean(rotation ** 2))),
        "rotation_max_rad": float(np.max(rotation)),
    }


def solve_joint_zeros(
    dataset: dict[str, Any],
    hand_eye: dict[str, Any],
    mount: dict[str, Any],
    anchor: dict[str, Any],
    chain: KinematicChain,
    *,
    min_samples: int = 16,
    max_abs_offset_rad: float = math.radians(3.0),
    prior_sigma_rad: float = math.radians(1.0),
    fit_translation_sigma_m: float = 0.002,
    fit_rotation_sigma_rad: float = math.radians(0.3),
    min_joint_span_rad: float = math.radians(10.0),
    max_validation_translation_rmse_m: float = 0.010,
    max_validation_rotation_rmse_rad: float = math.radians(2.0),
    max_observability_condition: float = 10_000.0,
) -> dict[str, Any]:
    positive_parameters = {
        "max_abs_offset_rad": max_abs_offset_rad,
        "prior_sigma_rad": prior_sigma_rad,
        "fit_translation_sigma_m": fit_translation_sigma_m,
        "fit_rotation_sigma_rad": fit_rotation_sigma_rad,
        "min_joint_span_rad": min_joint_span_rad,
        "max_validation_translation_rmse_m": (
            max_validation_translation_rmse_m
        ),
        "max_validation_rotation_rmse_rad": max_validation_rotation_rmse_rad,
        "max_observability_condition": max_observability_condition,
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in positive_parameters.values()
    ):
        raise ValueError("all fit, prior, validation, and observability limits must be positive")
    if dataset.get("schema") != "z_manip.piper_hand_eye_samples.v1":
        raise ValueError("unsupported kinematic sample schema")
    if dataset.get("synthetic") is not False:
        raise ValueError("kinematic calibration requires real samples")
    if (
        hand_eye.get("schema") != "z_manip.piper_camera_calibration.v1"
        or hand_eye.get("calibrated") is not True
        or hand_eye.get("synthetic") is not False
    ):
        raise ValueError("real quality-gated hand-eye calibration is required")
    if (
        mount.get("schema") != "z_manip.piper_mount_calibration.v1"
        or mount.get("calibrated") is not True
    ):
        raise ValueError("quality-gated body-to-arm mounting calibration is required")
    if (
        anchor.get("schema") != "z_manip.platform_target_anchor.v1"
        or anchor.get("example_only") is True
        or anchor.get("independently_measured") is not True
        or anchor.get("rigidly_attached_to_platform") is not True
    ):
        raise ValueError("an independent real body target anchor is required")

    samples = dataset.get("samples")
    if not isinstance(samples, list) or len(samples) < min_samples:
        raise ValueError(f"at least {min_samples} real poses are required")
    if (
        dataset.get("base_frame") != chain.base_link
        or dataset.get("tip_link") != chain.tip_link
        or hand_eye.get("tip_link") != chain.tip_link
        or dataset.get("camera_frame") != hand_eye.get("camera_frame")
        or dataset.get("target_frame") != anchor.get("target_frame")
        or mount.get("arm_base_frame") != chain.base_link
        or mount.get("platform_frame") != anchor.get("platform_frame")
    ):
        raise ValueError("kinematic dataset/calibration frame contract does not match")

    platform_from_arm = rigid(
        mount.get("platform_from_arm_base"),
        "platform_from_arm_base",
    )
    tip_from_camera = rigid(hand_eye.get("tip_from_camera"), "tip_from_camera")
    platform_from_target = rigid(
        anchor.get("platform_from_target"),
        "platform_from_target",
    )
    dataset_hash = _hash(dataset)
    hand_eye_source_hash = str(hand_eye.get("dataset_sha256", "")).strip()
    mount_input_hashes = mount.get("input_sha256")
    mount_source_hash = ""
    if isinstance(mount_input_hashes, dict):
        mount_source_hash = str(mount_input_hashes.get("samples", "")).strip()
    independent_dataset = bool(
        hand_eye_source_hash
        and mount_source_hash
        and dataset_hash not in {hand_eye_source_hash, mount_source_hash}
    )
    if not independent_dataset:
        raise ValueError(
            "joint-zero fitting requires a new dataset independent of both "
            "hand-eye and mount fitting inputs"
        )

    observations: list[tuple[np.ndarray, np.ndarray]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{index}] must be an object")
        joints = np.asarray(sample.get("joint_positions_rad"), dtype=float)
        if joints.shape != (chain.dof,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"samples[{index}] joint vector is invalid")
        if sample.get("joint_names") != list(chain.joint_names):
            raise ValueError(
                f"samples[{index}] joint names do not match the URDF chain"
            )
        safety = sample.get("safety_evidence")
        if (
            not isinstance(safety, dict)
            or safety.get("camera_read_only") is not True
            or safety.get("can_read_only") is not True
            or safety.get("can_tx_packet_delta") != 0
        ):
            raise ValueError(
                f"samples[{index}] lacks read-only camera/CAN and zero-TX evidence"
            )
        reported_base_from_tip = rigid(
            sample.get("base_from_tip"),
            f"samples[{index}].base_from_tip",
        )
        nominal_base_from_tip = chain.forward(joints)
        nominal_contract_error = _pose_error(
            nominal_base_from_tip,
            reported_base_from_tip,
        )
        if (
            np.linalg.norm(nominal_contract_error[:3]) > 1e-6
            or np.linalg.norm(nominal_contract_error[3:]) > 1e-6
        ):
            raise ValueError(
                f"samples[{index}] base_from_tip was not generated from the "
                "selected URDF and reported joints"
            )
        camera_from_target = rigid(
            sample.get("camera_from_target"),
            f"samples[{index}].camera_from_target",
        )
        observations.append((joints, camera_from_target))

    joint_matrix = np.stack([joints for joints, _ in observations])
    joint_spans = np.ptp(joint_matrix, axis=0)
    excitation_singular_values = np.linalg.svd(
        joint_matrix - np.mean(joint_matrix, axis=0),
        compute_uv=False,
    )
    excitation_reference = max(float(excitation_singular_values[0]), 1e-12)
    excitation_rank = int(np.count_nonzero(
        excitation_singular_values / excitation_reference >= 0.05,
    ))
    if np.any(joint_spans < min_joint_span_rad) or excitation_rank != chain.dof:
        raise ValueError(
            "joint poses do not independently excite all six axes over the "
            "required range"
        )

    validation_indices = np.arange(0, len(observations), 4, dtype=int)
    training_indices = np.asarray([
        index
        for index in range(len(observations))
        if index not in set(validation_indices.tolist())
    ], dtype=int)
    if len(validation_indices) < 4 or len(training_indices) < 12:
        raise ValueError("train/validation split requires at least 12/4 poses")

    def errors(offsets: np.ndarray, indices: np.ndarray) -> np.ndarray:
        values = []
        for index in indices:
            joints, camera_from_target = observations[int(index)]
            predicted = (
                platform_from_arm
                @ chain.forward(joints + offsets)
                @ tip_from_camera
                @ camera_from_target
            )
            values.append(_pose_error(predicted, platform_from_target))
        return np.asarray(values)

    # Fit scales represent one-observation noise, whereas validation limits are
    # end-to-end acceptance gates.  Reusing the looser validation limits here
    # would let the zero-centred prior dominate a physically informative fit.
    translation_scale = fit_translation_sigma_m
    rotation_scale = fit_rotation_sigma_rad

    def objective(offsets: np.ndarray) -> np.ndarray:
        residual = errors(offsets, training_indices)
        scaled = np.column_stack((
            residual[:, :3] / translation_scale,
            residual[:, 3:] / rotation_scale,
        )).reshape(-1)
        return np.concatenate((scaled, offsets / prior_sigma_rad))

    result = least_squares(
        objective,
        np.zeros(chain.dof),
        bounds=(-max_abs_offset_rad, max_abs_offset_rad),
        method="trf",
        max_nfev=500,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    offsets = np.asarray(result.x, dtype=float)

    def observation_vector(values: np.ndarray) -> np.ndarray:
        residual = errors(values, training_indices)
        return np.column_stack((
            residual[:, :3] / translation_scale,
            residual[:, 3:] / rotation_scale,
        )).reshape(-1)

    step = 1e-5
    jacobian = np.column_stack([
        (
            observation_vector(offsets + np.eye(chain.dof)[axis] * step)
            - observation_vector(offsets - np.eye(chain.dof)[axis] * step)
        ) / (2.0 * step)
        for axis in range(chain.dof)
    ])
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    reference = max(float(singular_values[0]), 1e-12)
    observability_rank = int(np.count_nonzero(singular_values / reference >= 1e-4))
    observability_condition = float(
        singular_values[0] / max(singular_values[-1], 1e-12),
    )
    information = jacobian.T @ jacobian
    covariance = np.linalg.pinv(information, rcond=1e-12)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    nominal_training = _metrics(errors(np.zeros(chain.dof), training_indices))
    calibrated_training = _metrics(errors(offsets, training_indices))
    nominal_validation = _metrics(errors(np.zeros(chain.dof), validation_indices))
    calibrated_validation = _metrics(errors(offsets, validation_indices))
    validation_passes = bool(
        calibrated_validation["translation_rmse_m"]
        <= max_validation_translation_rmse_m
        and calibrated_validation["rotation_rmse_rad"]
        <= max_validation_rotation_rmse_rad
        and calibrated_validation["translation_rmse_m"]
        <= nominal_validation["translation_rmse_m"] * 1.05 + 1e-6
        and calibrated_validation["rotation_rmse_rad"]
        <= nominal_validation["rotation_rmse_rad"] * 1.05 + 1e-6
    )
    observable = bool(
        observability_rank == chain.dof
        and observability_condition <= max_observability_condition
    )
    not_on_bound = bool(np.all(np.abs(offsets) < max_abs_offset_rad * 0.98))
    ready = bool(result.success and observable and validation_passes and not_on_bound)
    input_hash = _hash({
        "dataset": dataset_hash,
        "hand_eye": _hash(hand_eye),
        "mount": _hash(mount),
        "anchor": _hash(anchor),
    })
    return {
        "schema": SCHEMA,
        "ready_for_manual_review": ready,
        "read_only": True,
        "motion_commands_published": 0,
        "urdf_modified": False,
        "calibration_id": f"piper-joint-zero-{input_hash[:16]}",
        "joint_names": list(chain.joint_names),
        "joint_zero_offsets_rad": offsets.tolist(),
        "joint_zero_offsets_deg": np.degrees(offsets).tolist(),
        "sample_count": len(observations),
        "training_indices": training_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "optimizer": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "evaluations": int(result.nfev),
            "max_abs_offset_rad": max_abs_offset_rad,
            "prior_sigma_rad": prior_sigma_rad,
        },
        "observability": {
            "rank": observability_rank,
            "required_rank": chain.dof,
            "condition": observability_condition,
            "max_condition": max_observability_condition,
            "singular_values": singular_values.tolist(),
            "linearized_standard_deviation_rad": standard_deviation.tolist(),
            "linearized_standard_deviation_deg": np.degrees(
                standard_deviation,
            ).tolist(),
            "passes": observable,
            "joint_excitation_rank": excitation_rank,
            "joint_excitation_required_rank": chain.dof,
            "joint_excitation_singular_values": (
                excitation_singular_values.tolist()
            ),
            "joint_spans_rad": joint_spans.tolist(),
            "minimum_joint_span_rad": min_joint_span_rad,
        },
        "provenance": {
            "independent_dataset_verified": independent_dataset,
            "read_only_capture_evidence_verified": True,
            "dataset_sha256": dataset_hash,
            "hand_eye_dataset_sha256": hand_eye_source_hash,
            "mount_dataset_sha256": mount_source_hash,
        },
        "quality": {
            "nominal_training": nominal_training,
            "calibrated_training": calibrated_training,
            "nominal_validation": nominal_validation,
            "calibrated_validation": calibrated_validation,
            "validation_passes": validation_passes,
            "offsets_inside_review_bounds": not_on_bound,
        },
        "quality_limits": {
            "fit_translation_sigma_m": fit_translation_sigma_m,
            "fit_rotation_sigma_rad": fit_rotation_sigma_rad,
            "min_joint_span_rad": min_joint_span_rad,
            "max_validation_translation_rmse_m": max_validation_translation_rmse_m,
            "max_validation_rotation_rmse_rad": max_validation_rotation_rmse_rad,
        },
        "caveat": (
            "provisional joint-zero fit; hand-eye, mounting, compliance, backlash, "
            "and link-geometry errors can correlate"
        ),
        "input_sha256": input_hash,
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--hand-eye", type=Path, required=True)
    parser.add_argument("--mount", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    try:
        chain = KinematicChain.from_urdf(
            args.urdf.expanduser().resolve(),
            "piper_base_link",
            "piper_gripper_base",
        )
        report = solve_joint_zeros(
            _read(args.samples),
            _read(args.hand_eye),
            _read(args.mount),
            _read(args.anchor),
            chain,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "ready_for_manual_review": False,
            "read_only": True,
            "motion_commands_published": 0,
            "urdf_modified": False,
            "error": f"{type(error).__name__}: {error}",
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_manual_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
