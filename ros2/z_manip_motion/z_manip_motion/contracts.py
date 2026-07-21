"""Pure validation helpers shared by the ROS bridge and offline tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math


PIPER_ARM_JOINTS = tuple(f"piper_joint{index}" for index in range(1, 7))
PIPER_GRIPPER_JOINTS = ("piper_joint7", "piper_joint8")


class ContractError(ValueError):
    """A state, goal, or planned trajectory is unsafe to forward."""


def _finite(values: Sequence[float], label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be numeric") from error
    if not all(math.isfinite(value) for value in result):
        raise ContractError(f"{label} contains a non-finite value")
    return result


def validate_arm_joint_names(names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(names)
    if len(result) != 6 or len(set(result)) != 6:
        raise ContractError("the arm planning group must contain six unique joints")
    forbidden = set(result).intersection(PIPER_GRIPPER_JOINTS)
    if forbidden:
        raise ContractError(f"gripper joints cannot enter the arm group: {sorted(forbidden)}")
    if not all(isinstance(name, str) and name for name in result):
        raise ContractError("arm joint names must be non-empty strings")
    return result


def ordered_joint_positions(
    names: Sequence[str],
    positions: Sequence[float],
    *,
    expected: Sequence[str] = PIPER_ARM_JOINTS,
    allow_extra: bool,
) -> tuple[float, ...]:
    """Extract a finite joint vector by name, never by incoming array order."""

    expected_names = validate_arm_joint_names(expected)
    incoming_names = tuple(names)
    incoming_positions = _finite(positions, "joint positions")
    if len(incoming_names) != len(incoming_positions):
        raise ContractError("joint names and positions must have equal length")
    if len(set(incoming_names)) != len(incoming_names):
        raise ContractError("joint state contains duplicate names")
    mapping = dict(zip(incoming_names, incoming_positions))
    missing = set(expected_names) - set(mapping)
    if missing:
        raise ContractError(f"joint state is missing arm joints: {sorted(missing)}")
    extra = set(mapping) - set(expected_names)
    if extra and not allow_extra:
        raise ContractError(f"arm goal contains non-group joints: {sorted(extra)}")
    return tuple(mapping[name] for name in expected_names)


@dataclass(frozen=True)
class TrajectoryPointData:
    positions: Sequence[float]
    velocities: Sequence[float] = ()
    accelerations: Sequence[float] = ()
    effort: Sequence[float] = ()
    time_from_start_s: float = 0.0


def validate_planned_trajectory(
    joint_names: Sequence[str],
    points: Sequence[TrajectoryPointData],
    *,
    expected: Sequence[str] = PIPER_ARM_JOINTS,
) -> tuple[int, ...]:
    """Validate a MoveIt trajectory and return canonical reorder indices."""

    expected_names = validate_arm_joint_names(expected)
    incoming_names = tuple(joint_names)
    if len(incoming_names) != len(set(incoming_names)):
        raise ContractError("planned trajectory has duplicate joint names")
    if set(incoming_names) != set(expected_names):
        missing = sorted(set(expected_names) - set(incoming_names))
        extra = sorted(set(incoming_names) - set(expected_names))
        raise ContractError(
            f"planned trajectory does not match arm group; missing={missing}, extra={extra}",
        )
    if not points:
        raise ContractError("MoveIt returned an empty trajectory")

    joint_count = len(incoming_names)
    previous_time = -1.0
    for index, point in enumerate(points):
        positions = _finite(point.positions, f"trajectory point {index} positions")
        if len(positions) != joint_count:
            raise ContractError(f"trajectory point {index} positions have wrong length")
        for field_name in ("velocities", "accelerations", "effort"):
            values = _finite(getattr(point, field_name), f"trajectory point {index} {field_name}")
            if values and len(values) != joint_count:
                raise ContractError(f"trajectory point {index} {field_name} have wrong length")
        time_value = float(point.time_from_start_s)
        if not math.isfinite(time_value) or time_value < 0.0:
            raise ContractError(f"trajectory point {index} has invalid time_from_start")
        if time_value <= previous_time:
            raise ContractError("trajectory time_from_start must increase strictly")
        previous_time = time_value
    return tuple(incoming_names.index(name) for name in expected_names)


def normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Normalize an xyzw quaternion or reject a degenerate pose goal."""

    quaternion = _finite(values, "pose quaternion")
    if len(quaternion) != 4:
        raise ContractError("pose quaternion must have four components")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-8:
        raise ContractError("pose quaternion has near-zero norm")
    return tuple(value / norm for value in quaternion)


def validate_position(values: Sequence[float]) -> tuple[float, float, float]:
    position = _finite(values, "pose position")
    if len(position) != 3:
        raise ContractError("pose position must have three components")
    return position


def validate_trajectory_start(
    first_positions: Sequence[float],
    measured_positions: Sequence[float],
    *,
    tolerance: float,
) -> None:
    first = _finite(first_positions, "trajectory start")
    measured = _finite(measured_positions, "measured arm state")
    if len(first) != len(measured):
        raise ContractError("trajectory start and measured arm state sizes differ")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ContractError("trajectory start tolerance must be finite and positive")
    error = max((abs(planned - actual) for planned, actual in zip(first, measured)), default=0.0)
    if error > tolerance:
        raise ContractError(
            f"trajectory start differs from measured state by {error:.4f} rad",
        )


def validate_workspace_bounds(
    minimum: Sequence[float],
    maximum: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lower = _finite(minimum, "workspace minimum")
    upper = _finite(maximum, "workspace maximum")
    if len(lower) != 3 or len(upper) != 3:
        raise ContractError("workspace bounds must be three-vectors")
    if any(low >= high for low, high in zip(lower, upper)):
        raise ContractError("workspace minimum must be below maximum on every axis")
    return lower, upper
