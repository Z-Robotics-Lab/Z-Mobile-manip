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
#: The perception dry run clears its bundle caches immediately before it takes
#: ``bundle_wait_started`` ON THE PATH THAT RE-GROUNDS.  CORRECTION TO AN
#: EARLIER VERSION OF THIS COMMENT, WHICH CLAIMED IT ALWAYS DOES: the clear at
#: go2w_perception_dry_run.py sits under ``if not grounding_reused:``, and
#: ``infos`` is never cleared at all, so on the ``--reuse-valid-tracking`` path
#: -- which is what attempt 0 and the healed retry both take -- ``min(common)``
#: can be an entry that predates the wait.  The consequence is a FALSE NEGATIVE
#: only (condition 2 returns and the verdict is suppressed), never a false
#: positive; but the invariant as written was not true and the sizing argument
#: below must not rest on it.
#:
#: A message DELIVERED during the wait may still have been STAMPED before it,
#: because DDS buffers (depth 5 sensor-data / depth 10 reliable at the measured
#: 7.5-8.1 Hz EdgeTAM mask rate, i.e. ~0.6-1.3 s) hold frames captured before
#: the subscription drained.  2 s is ~1.6x the widest of those buffers.
#:
#: THE MARGIN, RE-DERIVED FROM THE RIGHT QUANTITY.  An earlier version of this
#: comment justified 2 s with "the recorded instances missed by 17.8 s and
#: 18.4 s".  That cites ``widest_rejected_overlap_margin_s``, which is the worst
#: margin over the whole 15 s wait -- NOT the quantity condition 1 tests, which
#: is ``bundle_wait_started - window_end``.  Measured properly over all 749
#: recorded sessions that hold a selected passive report
#: (``report.mtime - elapsed_s`` against ``observation_end_unix_ns``):
#:
#:     min -27.716  p50 -0.971  p95 +0.044  p99 4.086  max 4.640 s
#:
#: Exactly 10 sessions exceed 1 s and all 10 are the pathological ones (4.014 -
#: 4.640 s); every one of them failed, five on the full 15 s wait.  So the
#: 2.25 s trip point (this allowance plus the overlap tolerance) sits in an
#: empty band between 0.044 s and 4.014 s: no recorded healthy approach comes
#: near it, and the real margin on the recorded defect is ~1.8-2.4 s -- about
#: 2x, not the order of magnitude previously claimed.  A self-heal ~2 s faster
#: than the recorded restart would simply not produce the named error.
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
    the stamp-overlap gate then rejected every single fresh bundle.  Recorded on
    12 of the 13 self-heals that reached a second sub-attempt (an earlier
    version of this docstring said "6 of 6"; see
    ``_clear_inherited_attempt_outputs`` for the full re-count): 911 and 1127
    rejections, the full 15 s budget burnt, ``PERCEPTION_PROCESS_FAILED``, and a
    plainly detected object routed into a wrist search.

    IT IS NOW A RARE PATH, AND SAYING SO IS PART OF THE RECORD.  The commit that
    added this predicate ALSO made both retry paths clear the inherited window,
    which removes the only producer of it that has ever been observed.  What is
    left is the unobserved one: a capture cycle that stalls long enough for the
    live report to age past the trip point while fresh bundles keep arriving.
    The recorded per-capture time is 0.36 s median / 0.60 s p90 but its tail
    reaches 6.8-19.0 s, so the path is not closed -- but this verdict has never
    been seen to fire, and no test here can make it fire on real hardware.

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
