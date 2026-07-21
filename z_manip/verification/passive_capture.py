"""Pure validation for zero-transmit PiPER passive capture windows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


SCHEMA = "z_manip.piper_passive_joint_report.v1"


@dataclass(frozen=True)
class PassiveCaptureWindow:
    start_unix_ns: int
    end_unix_ns: int
    midpoint_unix_ns: int
    joint_positions_rad: np.ndarray


def validate_passive_capture(
    document: dict[str, Any],
    *,
    max_joint_range_rad: float = 0.002,
    max_snapshot_span_s: float = 0.050,
) -> PassiveCaptureWindow:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("unsupported passive joint report schema")
    if (
        document.get("read_only") is not True
        or document.get("complete_joint_feedback") is not True
        or document.get("zero_transmit_verified") is not True
        or int(document.get("interface_tx_packet_delta", -1)) != 0
    ):
        raise ValueError("passive joint report lacks complete zero-TX provenance")
    try:
        start = int(document["observation_start_unix_ns"])
        end = int(document["observation_end_unix_ns"])
        positions = np.asarray(document["joint_positions_rad"], dtype=float)
        ranges = np.asarray(document["joint_ranges_rad"], dtype=float)
        reported_max = float(document["max_joint_range_rad"])
        snapshot_span = float(document["joint_snapshot_span_s"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("passive joint report timing/vector fields are invalid") from error
    if start <= 0 or end <= start:
        raise ValueError("passive joint observation interval is invalid")
    if positions.shape != (6,) or ranges.shape != (6,):
        raise ValueError("passive joint report must contain six positions and ranges")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(ranges)):
        raise ValueError("passive joint vectors must be finite")
    if (
        not math.isfinite(reported_max)
        or reported_max < 0.0
        or not np.isclose(reported_max, float(np.max(ranges)), atol=1e-9)
        or reported_max > max_joint_range_rad
    ):
        raise ValueError("arm moved during passive joint capture")
    if (
        not math.isfinite(snapshot_span)
        or snapshot_span < 0.0
        or snapshot_span > max_snapshot_span_s
    ):
        raise ValueError("passive joint snapshot span is too wide")
    immutable = positions.copy()
    immutable.setflags(write=False)
    return PassiveCaptureWindow(
        start_unix_ns=start,
        end_unix_ns=end,
        midpoint_unix_ns=(start + end) // 2,
        joint_positions_rad=immutable,
    )
