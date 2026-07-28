"""Pure validation for zero-transmit PiPER passive capture windows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


SCHEMA = "z_manip.piper_passive_joint_report.v1"

#: How far AFTER a passive window closes a perception bundle stamp may still
#: fall and be admitted by the stamp-overlap gate.  This is the same 250 ms the
#: gate in ``go2w_perception_dry_run.py`` applies on both edges; it lives here
#: so the staleness predicate below is sized from the gate it must agree with
#: rather than from a second, independently written number.
PASSIVE_WINDOW_OVERLAP_TOLERANCE_NS = 250_000_000

#: How far BEFORE a run's bundle wait began a bundle's sensor stamp may still
#: legitimately be.
#:
#: The perception dry run clears every bundle cache immediately before it takes
#: ``bundle_wait_started``, so every stamp it can ever select arrives during the
#: wait -- but a message DELIVERED during the wait may have been STAMPED before
#: it, because DDS buffers (depth 5 sensor-data / depth 10 reliable at the
#: measured 7.5-8.1 Hz EdgeTAM mask rate, i.e. ~0.6-1.3 s) hold frames captured
#: before the subscription drained.  2 s is ~1.6x the widest of those buffers.
#:
#: This allowance exists only to keep the staleness verdict below CONSERVATIVE.
#: The two recorded instances of the defect it detects missed by 17.8 s and
#: 18.4 s (``widest_rejected_overlap_margin_s`` in the 20260728-030239 and
#: 20260728-070233 reports), an order of magnitude outside it.
PASSIVE_WINDOW_BUFFERED_BUNDLE_ALLOWANCE_NS = 2_000_000_000


def stale_passive_window_reason(
    *,
    window_end_unix_ns: int,
    bundle_wait_started_unix_ns: int,
    oldest_bundle_stamp_ns: int | None,
    overlap_tolerance_ns: int = PASSIVE_WINDOW_OVERLAP_TOLERANCE_NS,
    buffered_bundle_allowance_ns: int = PASSIVE_WINDOW_BUFFERED_BUNDLE_ALLOWANCE_NS,
) -> str | None:
    """Did this passive window close before its own bundle wait could begin?

    THE DEFECT THIS NAMES.  A perception request that retries -- either the
    inner fresh-seed retry or the resident-worker fingerprint self-heal -- used
    to inherit the PREVIOUS sub-attempt's ``selected_passive_joint_report.json``
    and ``live_passive_joint_report.json``.  The wrapper's capture loop skips
    opening a window while a valid selected report exists, so the retry ran with
    ``passive_capture_count == 0`` against a window that had already closed, and
    the stamp-overlap gate then rejected every single fresh bundle.  Recorded 6
    times out of 6 self-heals: 911 and 1127 rejections, the full 15 s budget
    burnt, ``PERCEPTION_PROCESS_FAILED``, and a plainly detected object routed
    into a wrist search.

    This predicate turns that into a NAMED, immediate verdict.  It never widens
    the gate: a window it accepts is still subject to the unchanged exact
    stamp-overlap requirement, and the only remedy for a window it rejects is to
    CAPTURE A NEW ONE.

    Both conditions must hold, which makes the verdict strictly rarer than
    either alone:

    1. the window closed (plus the gate's own trailing tolerance) more than
       ``buffered_bundle_allowance_ns`` before the bundle wait began, and
    2. it closed (plus that tolerance) before the OLDEST bundle stamp this run
       is actually holding, so no bundle in hand overlaps it and -- stamps being
       monotonic -- none that arrives later can either.

    Returns ``None`` when the window may still produce an overlap, otherwise a
    bounded reason string suitable for ``perception_failure``.
    """

    if oldest_bundle_stamp_ns is None:
        # Nothing to compare against yet.  A run with no bundle in hand has not
        # proven anything about this window.
        return None
    latest_admissible_ns = window_end_unix_ns + overlap_tolerance_ns
    if latest_admissible_ns >= bundle_wait_started_unix_ns - buffered_bundle_allowance_ns:
        return None
    if latest_admissible_ns >= oldest_bundle_stamp_ns:
        return None
    closed_before_wait_s = (
        bundle_wait_started_unix_ns - window_end_unix_ns
    ) * 1e-9
    behind_oldest_bundle_s = (oldest_bundle_stamp_ns - window_end_unix_ns) * 1e-9
    return (
        "passive_window_stale: the passive window closed "
        f"{closed_before_wait_s:.3f} s before this bundle wait began and "
        f"{behind_oldest_bundle_s:.3f} s before the oldest bundle in hand; "
        "no fresh capture can overlap it, so it must be re-captured"
    )


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
