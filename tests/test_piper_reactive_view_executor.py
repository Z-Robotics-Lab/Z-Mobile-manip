"""Offline contract tests for the PiPER reactive-view executor.

These tests intentionally exercise only parsing, bounded integration, and the
pre-transport command-line gate.  They never import ROS or open SocketCAN.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_reactive_view_executor.py"
SPEC = importlib.util.spec_from_file_location("piper_reactive_view_executor_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def intent(**updates: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": MODULE.INTENT_SCHEMA,
        "seq": 7,
        "source_timestamp_ns": 1_000_000_000,
        "deadline_unix_ns": 1_250_000_000,
        "joint_velocity_rps": [0.0] * 6,
    }
    document.update(updates)
    return document


def test_validated_intent_accepts_fresh_schema_and_clips_velocity() -> None:
    qdot = [0.0, MODULE.MAX_QDOT_RPS * 10.0, -MODULE.MAX_QDOT_RPS * 10.0, 0, 0, 0]
    seq, source_ns, bounded = MODULE.validated_intent(
        intent(joint_velocity_rps=qdot),
        now_ns=1_050_000_000,
    )
    assert seq == 7
    assert source_ns == 1_000_000_000
    np.testing.assert_allclose(
        bounded,
        [0.0, MODULE.MAX_QDOT_RPS, -MODULE.MAX_QDOT_RPS, 0.0, 0.0, 0.0],
    )


@pytest.mark.parametrize(
    ("document", "now_ns"),
    [
        (intent(schema="wrong"), 1_050_000_000),
        (intent(seq=True), 1_050_000_000),
        (intent(seq=1.5), 1_050_000_000),
        (intent(deadline_unix_ns=999_999_999), 1_050_000_000),
        (intent(deadline_unix_ns=1_010_000_000), 1_050_000_000),
        (intent(), 1_400_000_001),
        (
            intent(source_timestamp_ns=1_500_000_001, deadline_unix_ns=1_750_000_000),
            1_000_000_000,
        ),
        (intent(joint_velocity_rps=[0.0] * 5), 1_050_000_000),
        (intent(joint_velocity_rps=[0.0, 0.0, 0.0, 0.0, 0.0, float("nan")]), 1_050_000_000),
        (intent(joint_velocity_rps=None), 1_050_000_000),
    ],
)
def test_validated_intent_rejects_bad_schema_time_and_vector(
    document: dict[str, object],
    now_ns: int,
) -> None:
    with pytest.raises(ValueError):
        MODULE.validated_intent(document, now_ns=now_ns)


def test_validated_intent_accepts_measured_pc_to_nuc_clock_skew() -> None:
    """The NUC can be about 310 ms behind the PC without invalidating a lease."""
    seq, source_ns, qdot = MODULE.validated_intent(
        intent(
            source_timestamp_ns=1_310_000_000,
            deadline_unix_ns=1_560_000_000,
            joint_velocity_rps=[0.1] * 6,
        ),
        now_ns=1_000_000_000,
    )
    assert seq == 7
    assert source_ns == 1_310_000_000
    np.testing.assert_allclose(qdot, [0.1] * 6)


def test_bounded_target_limits_qdot_step_and_joint_envelope() -> None:
    low = MODULE.piper.JOINT_LIMITS_RAD[:, 0] + MODULE.JOINT_MARGIN_RAD
    high = MODULE.piper.JOINT_LIMITS_RAD[:, 1] - MODULE.JOINT_MARGIN_RAD
    measured = (low + high) / 2.0
    target = MODULE.bounded_target(measured, np.full(6, 1000.0), 10.0)
    np.testing.assert_allclose(target - measured, np.full(6, MODULE.MAX_STEP_RAD))

    near_limits = np.asarray([high[0], low[1], high[2], low[3], high[4], low[5]])
    outward = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    clipped = MODULE.bounded_target(near_limits, outward, 1.0)
    assert np.all(clipped >= low)
    assert np.all(clipped <= high)
    np.testing.assert_allclose(clipped, near_limits)

    # A stale-intent/full-stop zero command must hold the measured pose.  It
    # must not silently pull a joint off its URDF stop merely to create margin.
    hard_edge = np.asarray([
        MODULE.piper.JOINT_LIMITS_RAD[0, 1],
        MODULE.piper.JOINT_LIMITS_RAD[1, 0],
        MODULE.piper.JOINT_LIMITS_RAD[2, 1],
        MODULE.piper.JOINT_LIMITS_RAD[3, 0],
        MODULE.piper.JOINT_LIMITS_RAD[4, 1],
        MODULE.piper.JOINT_LIMITS_RAD[5, 0],
    ])
    np.testing.assert_allclose(MODULE.bounded_target(hard_edge, np.zeros(6), 0.05), hard_edge)
    np.testing.assert_allclose(MODULE.bounded_target(hard_edge, outward, 0.05), hard_edge)


@pytest.mark.parametrize(
    ("measured", "qdot", "dt_s"),
    [
        (np.zeros(5), np.zeros(6), 0.05),
        (np.full(6, np.nan), np.zeros(6), 0.05),
        (np.zeros(6), np.zeros(5), 0.05),
        (np.zeros(6), np.full(6, np.inf), 0.05),
        (np.zeros(6), np.zeros(6), 0.0),
        (np.zeros(6), np.zeros(6), float("nan")),
    ],
)
def test_bounded_target_rejects_invalid_inputs(
    measured: np.ndarray,
    qdot: np.ndarray,
    dt_s: float,
) -> None:
    with pytest.raises(ValueError):
        MODULE.bounded_target(measured, qdot, dt_s)


def test_cli_requires_explicit_live_ack_before_ros_or_can_import() -> None:
    environment = os.environ.copy()
    environment.pop("Z_MANIP_PIPER_REACTIVE_ACK", None)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--execute"],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert result.returncode != 0
    assert "exact acknowledgement" in result.stderr
    assert "rclpy" not in result.stderr
    assert "pyAgxArm" not in result.stderr


def test_import_does_not_import_ros_or_hardware_transport() -> None:
    code = (
        "import runpy,sys;"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='offline_import');"
        "print(int('rclpy' in sys.modules), int('pyAgxArm' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "0 0"


def test_reactive_owner_uses_sensor_data_qos_for_standard_joint_topic() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'JOINT_STATE_TOPIC = "/piper/state"' in source
    assert 'JointState, JOINT_STATE_TOPIC, qos_profile_sensor_data' in source


def test_reactive_owner_does_not_starve_intent_callback_behind_can_timer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "MultiThreadedExecutor(num_threads=2)" in source
    assert "callback_group=self.intent_group" in source
    assert "callback_group=self.hardware_group" in source
    assert "with self.intent_lock:" in source


def test_reactive_owner_checks_fixed_fixtures_at_final_command_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FixedSelfCollisionGuard" in source
    assert "fixed_collision_guard.check_step" in source
    unsafe_hold = source.index("self.target = self.actual.copy()")
    transport = source.index("robot.move_j", unsafe_hold)
    assert unsafe_hold < transport
    assert '"fixed_collision_guard": collision_evidence' in source
    assert '"unsafe_target_forwarded": False' in source
    assert '"measured_hold_sent": self.measured_hold_sent' in source
    assert '"last_blocked_seq": self.last_blocked_seq' in source


def test_joint_state_fields_match_passive_bridge_contract() -> None:
    measured = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    frame_id, names, positions = MODULE.joint_state_fields(measured)
    assert frame_id == "piper_base_link"
    assert names == [f"piper_joint{index}" for index in range(1, 7)]
    assert positions == pytest.approx(measured.tolist())


@pytest.mark.parametrize("measured", [np.zeros(5), np.full(6, np.nan)])
def test_joint_state_fields_reject_invalid_feedback(measured: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite six-vector"):
        MODULE.joint_state_fields(measured)


def test_validated_joint_feedback_refreshes_only_on_new_sdk_stamp() -> None:
    measured = np.arange(6, dtype=float) / 10.0
    actual, stamp, receipt_s, age_s = MODULE.validated_joint_feedback(
        measured,
        11.0,
        previous_stamp=10.0,
        previous_receipt_s=2.0,
        now_s=5.0,
    )
    np.testing.assert_allclose(actual, measured)
    assert stamp == 11.0
    assert receipt_s == 5.0
    assert age_s == 0.0

    _actual, _stamp, unchanged_receipt_s, unchanged_age_s = MODULE.validated_joint_feedback(
        measured,
        11.0,
        previous_stamp=11.0,
        previous_receipt_s=receipt_s,
        now_s=5.0 + MODULE.MAX_FEEDBACK_AGE_S,
    )
    assert unchanged_receipt_s == 5.0
    assert unchanged_age_s == pytest.approx(MODULE.MAX_FEEDBACK_AGE_S)


def test_validated_joint_feedback_rejects_stale_unchanged_sdk_sample() -> None:
    with pytest.raises(ValueError, match="stale"):
        MODULE.validated_joint_feedback(
            np.zeros(6),
            11.0,
            previous_stamp=11.0,
            previous_receipt_s=5.0,
            now_s=5.0 + MODULE.MAX_FEEDBACK_AGE_S + 1e-6,
        )
