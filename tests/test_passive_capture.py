from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from z_manip.verification.passive_capture import (
    PASSIVE_WINDOW_OVERLAP_TOLERANCE_NS,
    stale_passive_window_reason,
    validate_passive_capture,
)


def report():
    return {
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": True,
        "interface_tx_packet_delta": 0,
        "observation_start_unix_ns": 1_000_000_000,
        "observation_end_unix_ns": 9_000_000_000,
        "joint_positions_rad": [0.0, 0.2, -0.4, 0.1, 0.0, 0.0],
        "joint_ranges_rad": [0.0] * 6,
        "max_joint_range_rad": 0.0,
        "joint_snapshot_span_s": 0.005,
    }


def test_valid_zero_transmit_window_is_immutable():
    value = validate_passive_capture(report())
    assert value.midpoint_unix_ns == 5_000_000_000
    assert value.joint_positions_rad.shape == (6,)
    assert value.joint_positions_rad.flags.writeable is False


@pytest.mark.parametrize("field,value", [
    ("interface_tx_packet_delta", 1),
    ("zero_transmit_verified", False),
    ("complete_joint_feedback", False),
])
def test_rejects_missing_zero_transmit_provenance(field, value):
    document = report()
    document[field] = value
    with pytest.raises(ValueError, match="zero-TX"):
        validate_passive_capture(document)


def test_rejects_motion_or_wide_snapshot():
    moving = report()
    moving["joint_ranges_rad"][2] = 0.01
    moving["max_joint_range_rad"] = 0.01
    with pytest.raises(ValueError, match="moved"):
        validate_passive_capture(moving)
    wide = report()
    wide["joint_snapshot_span_s"] = 0.2
    with pytest.raises(ValueError, match="too wide"):
        validate_passive_capture(wide)


def test_positions_are_exactly_six_finite_values():
    document = report()
    document["joint_positions_rad"] = [0.0] * 5
    with pytest.raises(ValueError, match="six positions"):
        validate_passive_capture(document)
    document = report()
    document["joint_positions_rad"][1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_passive_capture(document)


# ---------------------------------------------------------------------------
# R9: a passive window that closed before its own bundle wait began.
#
# Sized against the RECORDED failures.  Sessions 20260728-030239 and
# 20260728-070233 are self-healed perception retries that inherited the previous
# sub-attempt's window: their reports carry passive_window_rejections of 911 and
# 1127 and widest_rejected_overlap_margin_s of -17.807681 and -18.408020, with
# seed_accepted true and 781 / 658 target points -- the object was plainly
# detected, and the run burnt its whole 15 s budget rejecting every bundle.
# ---------------------------------------------------------------------------

_SECOND = 1_000_000_000


def test_a_window_that_closed_long_before_the_wait_is_named_stale():
    wait_started = 1_000 * _SECOND
    reason = stale_passive_window_reason(
        # The 20260728-030239 geometry: window ~17.8 s behind the wait.
        window_end_unix_ns=wait_started - 18 * _SECOND,
        bundle_wait_started_unix_ns=wait_started,
        oldest_bundle_stamp_ns=wait_started + _SECOND // 10,
    )
    assert reason is not None
    assert reason.startswith("passive_window_stale")
    # The reason must carry BOTH measurements an operator needs to tell this
    # apart from a near miss, not just the verdict.
    assert "18.000 s before this bundle wait began" in reason
    assert "before the oldest bundle in hand" in reason


def test_a_window_inside_the_buffered_bundle_allowance_is_not_stale():
    wait_started = 1_000 * _SECOND
    # 1.5 s behind the wait: inside the 2 s DDS-buffer allowance, so a bundle
    # delivered during the wait but STAMPED before it can still overlap.
    assert stale_passive_window_reason(
        window_end_unix_ns=wait_started - 1_500_000_000,
        bundle_wait_started_unix_ns=wait_started,
        oldest_bundle_stamp_ns=wait_started + _SECOND,
    ) is None


def test_a_window_that_still_overlaps_a_bundle_in_hand_is_never_stale():
    wait_started = 1_000 * _SECOND
    # Far behind the wait, but the run is holding a bundle stamped inside the
    # window's own trailing tolerance.  Condition 2 refuses the verdict: this
    # window can still be selected, and calling it stale would throw away
    # evidence the unchanged overlap gate would have accepted.
    window_end = wait_started - 30 * _SECOND
    assert stale_passive_window_reason(
        window_end_unix_ns=window_end,
        bundle_wait_started_unix_ns=wait_started,
        oldest_bundle_stamp_ns=window_end + 100_000_000,
    ) is None


def test_no_bundle_in_hand_never_produces_a_stale_verdict():
    wait_started = 1_000 * _SECOND
    assert stale_passive_window_reason(
        window_end_unix_ns=wait_started - 60 * _SECOND,
        bundle_wait_started_unix_ns=wait_started,
        oldest_bundle_stamp_ns=None,
    ) is None


def test_the_stale_predicate_agrees_with_the_gate_it_must_not_contradict():
    """The tolerance is the OVERLAP GATE's own 250 ms, not a second number.

    go2w_perception_dry_run.py admits a stamp in
    ``[start - 250 ms, end + 250 ms]``.  A window whose latest admissible stamp
    is still ahead of the oldest bundle in hand must never be called stale, or
    this predicate would reject evidence that gate would have taken.
    """

    assert PASSIVE_WINDOW_OVERLAP_TOLERANCE_NS == 250_000_000
    gate_source = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "runtime" / "go2w_perception_dry_run.py"
    ).read_text(encoding="utf-8")
    assert "capture.start_unix_ns - 250_000_000" in gate_source
    assert "capture.end_unix_ns + 250_000_000" in gate_source
