#!/usr/bin/env python3
"""Persistent single-CAN-owner executor for whole-body PiPER view intents.

This process is started only for an explicitly authorized live mobile-servo
session.  It integrates short-lived velocity intents from fresh measured joint
feedback, sends a bounded absolute target, and publishes measured ownership
status.  Importing this module cannot connect to CAN or enable the arm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import piper_staged_grasp_executor as piper


INTENT_SCHEMA = "z_manip.piper_reactive_view_intent.v1"
STATUS_SCHEMA = "z_manip.piper_reactive_view_status.v1"
INTENT_TOPIC = "/z_manip/reactive/arm_view_intent"
STATUS_TOPIC = "/z_manip/reactive/arm_view_status"
JOINT_STATE_TOPIC = "/piper/state"
JOINT_NAMES = tuple(f"piper_joint{index}" for index in range(1, 7))
JOINT_STATE_FRAME = "piper_base_link"
FULL_STOP_TOPIC = "/go2w/full_stop"
LIVE_ACK = "I_UNDERSTAND_PIPER_REACTIVE_VIEW_WILL_MOVE"
MAX_INTENT_AGE_S = 0.30
# The intent publisher runs on the 4090 PC while this executor runs on the
# NUC.  Their wall clocks are NTP-synchronised but are not phase locked.  The
# measured PC-to-NUC midpoint offset is about 0.31 s, so a 0.10 s future check
# rejects every otherwise-fresh command.  The local monotonic lease below
# remains the actual motion timeout; this allowance is used only when accepting
# a newly received cross-host document.
MAX_FUTURE_SKEW_S = 0.50
MAX_FEEDBACK_AGE_S = 0.30
MAX_QDOT_RPS = math.radians(12.0)
MAX_STEP_RAD = math.radians(2.0)
JOINT_MARGIN_RAD = math.radians(0.5)


def validated_intent(document: object, *, now_ns: int) -> tuple[int, int, np.ndarray]:
    if not isinstance(document, dict) or document.get("schema") != INTENT_SCHEMA:
        raise ValueError("invalid PiPER reactive intent schema")
    if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns <= 0:
        raise ValueError("now_ns must be a positive integer nanosecond stamp")

    def exact_integer(field: str, default: int) -> int:
        value = document.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"intent {field} must be an integer")
        return value

    seq = exact_integer("seq", -1)
    source_ns = exact_integer("source_timestamp_ns", 0)
    deadline_ns = exact_integer("deadline_unix_ns", 0)
    if seq < 0 or source_ns <= 0 or deadline_ns <= 0:
        raise ValueError("intent seq/source/deadline must be positive")
    if deadline_ns < source_ns:
        raise ValueError("intent deadline precedes its source timestamp")
    age_s = (now_ns - source_ns) / 1e9
    if now_ns > deadline_ns or age_s > MAX_INTENT_AGE_S or age_s < -MAX_FUTURE_SKEW_S:
        raise ValueError("PiPER reactive intent is stale")
    try:
        qdot = np.asarray(document.get("joint_velocity_rps"), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("intent must contain six finite joint velocities") from error
    if qdot.shape != (6,) or not np.isfinite(qdot).all():
        raise ValueError("intent must contain six finite joint velocities")
    return seq, source_ns, np.clip(qdot, -MAX_QDOT_RPS, MAX_QDOT_RPS)


def bounded_target(measured: np.ndarray, qdot: np.ndarray, dt_s: float) -> np.ndarray:
    measured = np.asarray(measured, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    if measured.shape != (6,) or not np.isfinite(measured).all():
        raise ValueError("measured joints must be a finite six-vector")
    if qdot.shape != (6,) or not np.isfinite(qdot).all():
        raise ValueError("joint velocities must be a finite six-vector")
    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("integration dt must be finite and positive")
    bounded_qdot = np.clip(qdot, -MAX_QDOT_RPS, MAX_QDOT_RPS)
    step = np.clip(bounded_qdot * dt_s, -MAX_STEP_RAD, MAX_STEP_RAD)
    low = piper.JOINT_LIMITS_RAD[:, 0] + JOINT_MARGIN_RAD
    high = piper.JOINT_LIMITS_RAD[:, 1] - JOINT_MARGIN_RAD
    # Do not turn a zero-velocity hold into an unsolicited margin-recovery
    # motion when an encoder is already between the margin and its URDF stop
    # (PiPER J2/J3 commonly rest there).  Permit movement back toward the safe
    # interior, while clipping any command that would move farther outward.
    lower_stop = np.minimum(measured, low)
    upper_stop = np.maximum(measured, high)
    return np.clip(measured + step, lower_stop, upper_stop)


def validated_joint_feedback(
    measured: object,
    feedback_stamp: object,
    *,
    previous_stamp: float,
    previous_receipt_s: float,
    now_s: float,
) -> tuple[np.ndarray, float, float, float]:
    """Validate one SDK feedback sample and compute its local freshness.

    The SDK stamp is used only as a change token: firmware variants do not all
    expose the same clock domain.  Freshness is therefore measured from the
    local monotonic time at which a new stamp was first observed.
    """
    try:
        actual = np.asarray(measured, dtype=float)
        stamp = float(feedback_stamp)
        previous_stamp = float(previous_stamp)
        receipt_s = float(previous_receipt_s)
        now_s = float(now_s)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid PiPER joint feedback") from error
    if actual.shape != (6,) or not np.isfinite(actual).all():
        raise ValueError("PiPER joint feedback must be a finite six-vector")
    if not all(math.isfinite(value) for value in (stamp, previous_stamp, receipt_s, now_s)):
        raise ValueError("PiPER feedback timestamps must be finite")
    if receipt_s < 0.0 or now_s < receipt_s:
        raise ValueError("PiPER feedback receipt time is invalid")
    if stamp != previous_stamp:
        receipt_s = now_s
    feedback_age_s = max(0.0, now_s - receipt_s)
    if feedback_age_s > MAX_FEEDBACK_AGE_S:
        raise ValueError("PiPER joint feedback is stale")
    return actual, stamp, receipt_s, feedback_age_s


def joint_state_fields(measured: object) -> tuple[str, list[str], list[float]]:
    """Return the measured JointState contract shared with the passive bridge."""
    try:
        actual = np.asarray(measured, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid PiPER joint feedback") from error
    if actual.shape != (6,) or not np.isfinite(actual).all():
        raise ValueError("PiPER joint feedback must be a finite six-vector")
    return JOINT_STATE_FRAME, list(JOINT_NAMES), [float(value) for value in actual]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v188", choices=("default", "v183", "v188", "v189"))
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute or __import__("os").environ.get("Z_MANIP_PIPER_REACTIVE_ACK") != LIVE_ACK:
        raise SystemExit("live PiPER reactive executor requires --execute and exact acknowledgement")
    if not 5.0 <= args.rate_hz <= 30.0 or not 1 <= args.speed_percent <= 12:
        raise SystemExit("reactive rate/speed is outside the bounded envelope")

    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty, String

    robot: Any | None = None
    try:
        robot, _effector = piper.connect_real_arm(args.channel, args.firmware)
        measured, _initial_feedback_stamp = piper.wait_for_initial_arm_feedback(robot)
        guard = piper.CommandGuard()
        piper.enter_can_joint_control(robot, guard, timeout_s=5.0)
        robot.set_speed_percent(args.speed_percent)

        class ExecutorNode(Node):
            def __init__(self) -> None:
                super().__init__("z_manip_piper_reactive_view_executor")
                self.latest: tuple[int, int, np.ndarray] | None = None
                self.last_seq = -1
                self.last_intent_s: float | None = None
                self.actual = measured.copy()
                self.target = measured.copy()
                self.feedback_stamp = float(_initial_feedback_stamp)
                self.feedback_receipt_s = time.monotonic()
                self.commands_sent = 0
                self.stop_latched = False
                self.fault: str | None = None
                self.rejected_intents = 0
                self.last_intent_error: str | None = None
                # CAN feedback + move_j can consume nearly the entire 50 ms
                # timer period.  A SingleThreadedExecutor may consequently
                # keep selecting the overdue timer and never dispatch the
                # intent subscription.  Put the short subscription callbacks
                # in a separate callback group/thread; only the timer touches
                # the hardware SDK.
                self.intent_lock = threading.Lock()
                self.intent_group = MutuallyExclusiveCallbackGroup()
                self.hardware_group = MutuallyExclusiveCallbackGroup()
                qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
                self.publisher = self.create_publisher(String, STATUS_TOPIC, qos)
                # The passive bridge is stopped while this process owns CAN.
                # Preserve its exact measured JointState contract so TF, the
                # runtime observer, and whole-body controller retain feedback.
                self.joint_publisher = self.create_publisher(
                    JointState, JOINT_STATE_TOPIC, qos_profile_sensor_data
                )
                self.create_subscription(
                    String,
                    INTENT_TOPIC,
                    self._intent,
                    qos,
                    callback_group=self.intent_group,
                )
                self.create_subscription(
                    Empty,
                    FULL_STOP_TOPIC,
                    self._stop,
                    qos,
                    callback_group=self.intent_group,
                )
                self.create_timer(
                    1.0 / args.rate_hz,
                    self._tick,
                    callback_group=self.hardware_group,
                )

            def _intent(self, message: String) -> None:
                try:
                    value = validated_intent(json.loads(message.data), now_ns=time.time_ns())
                    with self.intent_lock:
                        if value[0] <= self.last_seq:
                            return
                        self.latest = value
                        self.last_seq = value[0]
                        self.last_intent_s = time.monotonic()
                        self.last_intent_error = None
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    # A malformed or stale message must not leave the previous
                    # velocity command live for the rest of its lease.
                    with self.intent_lock:
                        self.latest = None
                        self.rejected_intents += 1
                        self.last_intent_error = str(error)

            def _stop(self, _message: Empty) -> None:
                with self.intent_lock:
                    self.stop_latched = True
                    self.latest = None

            def _tick(self) -> None:
                now = time.monotonic()
                try:
                    actual, feedback_stamp = piper.read_joint_feedback(robot)
                    (
                        self.actual,
                        self.feedback_stamp,
                        self.feedback_receipt_s,
                        _feedback_age_s,
                    ) = validated_joint_feedback(
                        actual,
                        feedback_stamp,
                        previous_stamp=self.feedback_stamp,
                        previous_receipt_s=self.feedback_receipt_s,
                        now_s=now,
                    )
                    frame_id, names, positions = joint_state_fields(self.actual)
                    joint_state = JointState()
                    joint_state.header.stamp = self.get_clock().now().to_msg()
                    joint_state.header.frame_id = frame_id
                    joint_state.name = names
                    joint_state.position = positions
                    self.joint_publisher.publish(joint_state)
                    piper.check_arm_status(robot, require_idle=False)
                    with self.intent_lock:
                        fresh_intent = (
                            self.latest is not None
                            and self.last_intent_s is not None
                            and now - self.last_intent_s <= MAX_INTENT_AGE_S
                            and not self.stop_latched
                        )
                        qdot = (
                            self.latest[2].copy()
                            if fresh_intent and self.latest is not None
                            else np.zeros(6)
                        )
                    self.target = bounded_target(self.actual, qdot, 1.0 / args.rate_hz)
                    # Stale/Full Stop holds the measured pose rather than
                    # unloading torque or replaying an old target.
                    guard.mark_before_command()
                    robot.move_j([float(value) for value in self.target])
                    self.commands_sent += 1
                    self.fault = None
                except Exception as error:  # noqa: BLE001
                    self.fault = f"{type(error).__name__}: {error}"
                with self.intent_lock:
                    accepted_seq = self.last_seq
                    stop_latched = self.stop_latched
                    rejected_intents = self.rejected_intents
                    last_intent_error = self.last_intent_error
                status = String()
                status.data = json.dumps({
                    "schema": STATUS_SCHEMA,
                    "owner": "piper_reactive_view_executor",
                    "ready": self.fault is None,
                    "stop_latched": stop_latched,
                    "accepted_seq": accepted_seq,
                    "actual_joints_rad": self.actual.tolist(),
                    "target_joints_rad": self.target.tolist(),
                    "max_error_rad": float(np.max(np.abs(self.target - self.actual))),
                    "feedback_age_s": (
                        max(0.0, time.monotonic() - self.feedback_receipt_s)
                        if self.feedback_receipt_s > 0.0 else None
                    ),
                    "commands_sent": self.commands_sent,
                    "rejected_intents": rejected_intents,
                    "last_intent_error": last_intent_error,
                    "future_clock_skew_tolerance_s": MAX_FUTURE_SKEW_S,
                    "fault": self.fault,
                    "updated_unix_ns": time.time_ns(),
                }, separators=(",", ":"), allow_nan=False)
                self.publisher.publish(status)

        rclpy.init()
        node = ExecutorNode()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        return 0
    finally:
        piper.disconnect_quietly(robot)


if __name__ == "__main__":
    raise SystemExit(main())
