#!/usr/bin/env python3
"""ROS intent/status relay between reactive geometry and the NUC owner.

Input is JSON on ``/z_manip/reactive/posture_intent`` using schema
``z_manip.go2w_posture_intent.v1``.  The deltas are relative to the measured
neutral manipulation stance, not integrated on every controller tick.  Output
is the bounded NUC wire topic ``/go2w/posture_cmd``.

Shadow is the default and publishes only diagnostic status.  Live publishing
requires ``--mode live`` and an exact environment acknowledgement; the NUC
independently applies its own live gate and feedback checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any


INTENT_TOPIC = "/z_manip/reactive/posture_intent"
INTENT_STATUS_TOPIC = "/z_manip/reactive/posture_status"
NUC_COMMAND_TOPIC = "/go2w/posture_cmd"
NUC_STATUS_TOPIC = "/go2w/posture_state"
FULL_STOP_INPUT_TOPIC = "/z_manip/reactive/full_stop"
FULL_STOP_OUTPUT_TOPIC = "/go2w/full_stop"
CONTROL_RESET_INPUT_TOPIC = "/z_manip/reactive/control_reset"
CONTROL_RESET_OUTPUT_TOPIC = "/go2w/control_reset"
INTENT_SCHEMA = "z_manip.go2w_posture_intent.v1"
LIVE_ACK = "I_UNDERSTAND_POSTURE_INTENTS_REACH_NUC"


def bounded_wire_target(document: dict[str, Any]) -> tuple[float, float, float, float]:
    """Convert one neutral-relative Euler intent to the NUC wire contract."""
    if document.get("schema") != INTENT_SCHEMA:
        raise ValueError(f"intent schema must be {INTENT_SCHEMA}")
    legacy_height = float(document.get("body_height_delta_m", 0.0))
    values = (
        float(document.get("roll_delta_rad", 0.0)),
        float(document["pitch_delta_rad"]),
        float(document.get("yaw_delta_rad", 0.0)),
    )
    if not math.isfinite(legacy_height) or not all(math.isfinite(value) for value in values):
        raise ValueError("posture intent must be finite")
    if abs(legacy_height) > 1e-6:
        raise ValueError("BodyHeight is unsupported; body_height_delta_m must be zero")
    roll, pitch, yaw = values
    return (
        0.0,
        min(max(roll, -math.radians(8.0)), math.radians(8.0)),
        min(max(pitch, -math.radians(12.0)), math.radians(12.0)),
        min(max(yaw, -math.radians(8.0)), math.radians(8.0)),
    )


def feedback_is_fresh(document: dict[str, Any], *, maximum_age_s: float = 0.50) -> bool:
    try:
        age_s = float(document["feedback"]["sport_state_age_s"])
        return (
            document.get("schema") == "z_manip.go2w_posture_status.v1"
            and document.get("mode") == "live"
            and document.get("stop_latched") is False
            and bool(document["feedback"]["fresh"])
            and math.isfinite(age_s)
            and 0.0 <= age_s <= maximum_age_s
        )
    except (KeyError, TypeError, ValueError):
        return False


def euler_is_available(document: dict[str, Any]) -> bool:
    """Require explicit same-epoch robot evidence before publishing Euler."""
    capabilities = document.get("capabilities")
    return bool(
        isinstance(capabilities, dict)
        and capabilities.get("euler") is True
        and capabilities.get("euler_state") == "SUPPORTED_OBSERVED"
    )


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--feedback-timeout-s", type=float, default=0.50)
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = _parse_args(args)
    if parsed.mode == "live" and os.environ.get("Z_MANIP_POSTURE_INTENT_LIVE_ACK") != LIVE_ACK:
        raise SystemExit(
            "live blocked: set Z_MANIP_POSTURE_INTENT_LIVE_ACK=" + LIVE_ACK
        )

    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Empty, String

    class PostureIntentBridge(Node):
        def __init__(self) -> None:
            super().__init__("z_manip_go2w_posture_intent_bridge")
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.nuc_status: dict[str, Any] | None = None
            self.nuc_status_received_s: float | None = None
            self.command_pub = self.create_publisher(TwistStamped, NUC_COMMAND_TOPIC, qos)
            self.status_pub = self.create_publisher(String, INTENT_STATUS_TOPIC, qos)
            self.stop_pub = self.create_publisher(Empty, FULL_STOP_OUTPUT_TOPIC, qos)
            self.reset_pub = self.create_publisher(Empty, CONTROL_RESET_OUTPUT_TOPIC, qos)
            self.create_subscription(String, INTENT_TOPIC, self._intent, qos)
            self.create_subscription(String, NUC_STATUS_TOPIC, self._nuc_status, qos)
            self.create_subscription(Empty, FULL_STOP_INPUT_TOPIC, self._full_stop, qos)
            self.create_subscription(Empty, CONTROL_RESET_INPUT_TOPIC, self._reset, qos)
            self._publish("idle", "shadow: no commands published" if parsed.mode == "shadow" else "waiting for fresh NUC feedback")

        def _publish(self, phase: str, detail: str, *, target: tuple[float, ...] | None = None) -> None:
            message = String()
            message.data = json.dumps(
                {
                    "schema": "z_manip.go2w_posture_intent_status.v1",
                    "mode": parsed.mode,
                    "phase": phase,
                    "detail": detail,
                    "wire_target": target,
                    "nuc_status": self.nuc_status,
                    "updated_unix_ns": time.time_ns(),
                },
                separators=(",", ":"), allow_nan=False,
            )
            self.status_pub.publish(message)

        def _nuc_status(self, message: String) -> None:
            try:
                document = json.loads(message.data)
                if isinstance(document, dict):
                    self.nuc_status = document
                    self.nuc_status_received_s = time.monotonic()
            except json.JSONDecodeError:
                return

        def _intent(self, message: String) -> None:
            try:
                document = json.loads(message.data)
                target = bounded_wire_target(document)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                self._publish("blocked", f"invalid posture intent: {error}")
                return
            if parsed.mode == "shadow":
                self._publish("shadow", "would publish bounded posture target", target=target)
                return
            local_age = (
                math.inf if self.nuc_status_received_s is None
                else time.monotonic() - self.nuc_status_received_s
            )
            if (
                self.nuc_status is None
                or local_age > parsed.feedback_timeout_s
                or not feedback_is_fresh(self.nuc_status, maximum_age_s=parsed.feedback_timeout_s)
            ):
                self._publish("blocked", "NUC posture feedback is missing/stale/latched", target=target)
                return
            if not euler_is_available(self.nuc_status):
                self._publish(
                    "degraded",
                    "Euler is not implemented by the active Go2W service; base + arm fallback",
                    target=target,
                )
                return
            command = TwistStamped()
            command.header.stamp = self.get_clock().now().to_msg()
            command.header.frame_id = "base_link"
            command.twist.linear.z = target[0]
            command.twist.angular.x = target[1]
            command.twist.angular.y = target[2]
            command.twist.angular.z = target[3]
            self.command_pub.publish(command)
            self._publish("commanding", "bounded posture target published to NUC owner", target=target)

        def _full_stop(self, _message: Empty) -> None:
            # Full Stop is always forwarded in live mode and never waits for
            # feedback. Shadow mode remains physically incapable of sending.
            if parsed.mode == "live":
                self.stop_pub.publish(Empty())
                self._publish("stopping", "Full Stop forwarded to NUC owner")
            else:
                self._publish("shadow", "would forward Full Stop")

        def _reset(self, _message: Empty) -> None:
            if parsed.mode == "live":
                self.reset_pub.publish(Empty())
                self._publish("resetting", "control-reset request forwarded to NUC")
            else:
                self._publish("shadow", "would forward control reset")

    rclpy.init()
    node = PostureIntentBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
