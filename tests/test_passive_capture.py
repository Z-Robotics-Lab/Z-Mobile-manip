from __future__ import annotations

import numpy as np
import pytest

from z_manip.verification.passive_capture import validate_passive_capture


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
