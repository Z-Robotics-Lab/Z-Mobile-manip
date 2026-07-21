"""Canonical content identity for ROS-style joint trajectories."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Iterable, Sequence


_SCHEMA_PREFIX = b'z_manip.joint_trajectory.sha256.v1\x00'
_POINT_FIELDS = ('positions', 'velocities', 'accelerations', 'effort')


def _unsigned(value: object, label: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f'{label} is outside its unsigned integer range')
    return value


def _signed(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -(2**63) <= value < 2**63
    ):
        raise ValueError(f'{label} is outside the signed 64-bit range')
    return value


def _text(value: object, label: str) -> bytes:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f'{label} must be a non-empty trimmed string')
    encoded = value.encode('utf-8')
    if len(encoded) >= 2**32:
        raise ValueError(f'{label} exceeds the canonical string limit')
    return struct.pack('>I', len(encoded)) + encoded


def _float64_sequence(
    values: object,
    label: str,
    *,
    joint_count: int,
    required: bool,
) -> bytes:
    try:
        sequence = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f'{label} must be a numeric sequence') from error
    allowed_lengths = {joint_count} if required else {0, joint_count}
    if len(sequence) not in allowed_lengths:
        raise ValueError(
            f'{label} length must be '
            f'{joint_count}' + ('' if required else f' or zero'),
        )
    result = bytearray(struct.pack('>I', len(sequence)))
    for index, value in enumerate(sequence):
        if isinstance(value, bool):
            raise ValueError(f'{label}[{index}] must be finite float64 data')
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'{label}[{index}] must be finite float64 data',
            ) from error
        if not math.isfinite(number):
            raise ValueError(f'{label}[{index}] must be finite float64 data')
        result.extend(struct.pack('>d', number))
    return bytes(result)


def _duration(value: object, label: str) -> bytes:
    try:
        seconds = _signed(getattr(value, 'sec'), f'{label}.sec')
        nanoseconds = _unsigned(
            getattr(value, 'nanosec'),
            f'{label}.nanosec',
            maximum=999_999_999,
        )
    except AttributeError as error:
        raise ValueError(f'{label} must expose sec and nanosec') from error
    if seconds < 0:
        raise ValueError(f'{label} cannot be negative')
    return struct.pack('>qI', seconds, nanoseconds)


def canonical_joint_trajectory_sha256(
    *,
    frame_id: object,
    header_stamp: object,
    joint_names: Sequence[object] | Iterable[object],
    points: Sequence[object] | Iterable[object],
) -> str:
    """Return a canonical digest over every JointTrajectory payload field."""
    names = tuple(joint_names)
    if not names or len(names) >= 2**32:
        raise ValueError('joint_names must be a bounded non-empty sequence')
    if len(set(names)) != len(names):
        raise ValueError('joint_names must be unique')
    trajectory_points = tuple(points)
    if not trajectory_points or len(trajectory_points) >= 2**32:
        raise ValueError('trajectory points must be a bounded non-empty sequence')

    digest = hashlib.sha256()
    digest.update(_SCHEMA_PREFIX)
    digest.update(_text(frame_id, 'frame_id'))
    digest.update(_duration(header_stamp, 'header_stamp'))
    digest.update(struct.pack('>I', len(names)))
    for index, name in enumerate(names):
        digest.update(_text(name, f'joint_names[{index}]'))
    digest.update(struct.pack('>I', len(trajectory_points)))
    for point_index, point in enumerate(trajectory_points):
        for field in _POINT_FIELDS:
            try:
                values = getattr(point, field)
            except AttributeError as error:
                raise ValueError(
                    f'trajectory point {point_index} lacks {field}',
                ) from error
            digest.update(_float64_sequence(
                values,
                f'points[{point_index}].{field}',
                joint_count=len(names),
                required=field == 'positions',
            ))
        try:
            duration = getattr(point, 'time_from_start')
        except AttributeError as error:
            raise ValueError(
                f'trajectory point {point_index} lacks time_from_start',
            ) from error
        digest.update(_duration(
            duration,
            f'points[{point_index}].time_from_start',
        ))
    return digest.hexdigest()


__all__ = ['canonical_joint_trajectory_sha256']
