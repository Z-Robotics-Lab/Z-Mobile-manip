"""Plan-aware collision geometry for a symmetric parallel gripper."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .pointcloud import CapsuleSpec, RobotCollisionModel


def collision_aperture_for_grasp(
    required_width_m: object,
    *,
    open_aperture_m: float,
    grasp_margin_m: float,
) -> float:
    """Return the bounded collision aperture for one planned grasp width.

    ``grasp_margin_m`` is a total jaw-opening margin, so half of it remains on
    each finger.  A missing width is rejected: open-finger geometry is not a
    safe substitute for the closed-finger geometry used during lift.
    """

    if required_width_m is None:
        raise ValueError("plan-aware gripper collision requires required_width_m")
    try:
        required = float(required_width_m)
        opened = float(open_aperture_m)
        margin = float(grasp_margin_m)
    except (TypeError, ValueError) as error:
        raise ValueError("gripper collision apertures must be numeric") from error
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError("required gripper width must be finite and positive")
    if not math.isfinite(opened) or opened <= 0.0:
        raise ValueError("open gripper aperture must be finite and positive")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("gripper collision margin must be finite and non-negative")
    if required > opened + 1e-9:
        raise ValueError("required gripper width exceeds the collision-open aperture")
    return min(opened, required + margin)


def _shift_capsule(capsule: CapsuleSpec, shift: np.ndarray) -> CapsuleSpec:
    start = np.asarray(capsule.start_offset, dtype=float) + shift
    end = np.asarray(capsule.end_offset, dtype=float) + shift
    return replace(
        capsule,
        start_offset=tuple(float(value) for value in start),
        end_offset=tuple(float(value) for value in end),
    )


def with_parallel_gripper_aperture(
    model: RobotCollisionModel,
    *,
    open_aperture_m: float,
    aperture_m: float,
    closing_axis: object,
    positive_finger_prefix: str = "finger_left",
    negative_finger_prefix: str = "finger_right",
) -> RobotCollisionModel:
    """Translate fixed open-finger capsules to a requested jaw aperture.

    The positive finger starts on the positive side of ``closing_axis`` and
    moves in the negative direction while closing; the negative finger mirrors
    that motion.  Offsets must share one frame because the configured axis is
    expressed in that frame.  Palm and arm capsules, allowlists, and self-pairs
    are preserved unchanged.
    """

    try:
        opened = float(open_aperture_m)
        aperture = float(aperture_m)
        axis = np.asarray(closing_axis, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("parallel-gripper aperture geometry is malformed") from error
    if (
        not math.isfinite(opened)
        or opened <= 0.0
        or not math.isfinite(aperture)
        or aperture <= 0.0
        or aperture > opened + 1e-9
    ):
        raise ValueError("gripper collision aperture must be within (0, open]")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("gripper closing axis must be a finite three-vector")
    norm = float(np.linalg.norm(axis))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("gripper closing axis must be a unit vector")
    axis = axis / norm
    if not positive_finger_prefix or not negative_finger_prefix:
        raise ValueError("finger collision prefixes must be non-empty")

    inward = 0.5 * max(0.0, opened - aperture)
    positive_count = 0
    negative_count = 0
    shifted: list[CapsuleSpec] = []
    for capsule in model.capsules:
        if capsule.name == positive_finger_prefix or capsule.name.startswith(
            positive_finger_prefix + "_"
        ):
            positive_count += 1
            direction = -axis
            expected_sign = 1.0
        elif capsule.name == negative_finger_prefix or capsule.name.startswith(
            negative_finger_prefix + "_"
        ):
            negative_count += 1
            direction = axis
            expected_sign = -1.0
        else:
            shifted.append(capsule)
            continue
        if capsule.start_frame != capsule.end_frame:
            raise ValueError(
                f"finger capsule {capsule.name!r} must use one offset frame",
            )
        center = 0.5 * (
            np.asarray(capsule.start_offset, dtype=float)
            + np.asarray(capsule.end_offset, dtype=float)
        )
        if expected_sign * float(center @ axis) <= 0.0:
            raise ValueError(
                f"finger capsule {capsule.name!r} disagrees with its side name",
            )
        shifted.append(_shift_capsule(capsule, inward * direction))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("collision model must contain both parallel-gripper fingers")
    if positive_count != negative_count:
        raise ValueError("parallel-gripper collision sides must have equal capsule counts")
    return replace(model, capsules=tuple(shifted))


__all__ = [
    "collision_aperture_for_grasp",
    "with_parallel_gripper_aperture",
]
