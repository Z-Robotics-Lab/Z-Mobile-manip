#!/usr/bin/env python3
"""Single-owner Go2W Move/posture/Full-Stop WebRTC bridge.

The default ``shadow`` process never constructs ``UnitreeControlNode`` and
therefore cannot open WebRTC.  ``live`` requires both a command-line mode and
an exact environment acknowledgement.  In live mode the inherited Move pump,
BodyHeight/Euler and StopMove share one connection and one asyncio request
lock.  Full Stop latches immediately, drops pending work, and owns the next
SPORT request slot.

``/go2w/posture_cmd`` is deliberately a transport message, not a geometry
intent: ``linear.z`` is the Unitree BodyHeight *offset from nominal* and
``angular.x/y/z`` are absolute roll/pitch/yaw targets in ``base_link``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_ros.unitree_control import UnitreeControlNode


POSTURE_COMMAND_TOPIC = "/go2w/posture_cmd"
POSTURE_CANCEL_TOPIC = "/go2w/posture_cancel"
FULL_STOP_TOPIC = "/go2w/full_stop"
CONTROL_RESET_TOPIC = "/go2w/control_reset"
POSTURE_STATE_TOPIC = "/go2w/posture_state"
STATUS_SCHEMA = "z_manip.go2w_posture_status.v1"
LIVE_ACK = "I_UNDERSTAND_GO2W_WILL_MOVE"


def _status_code(response: Any) -> int | None:
    try:
        return int(response["data"]["header"]["status"]["code"])
    except (KeyError, TypeError, ValueError):
        return None


def _finite_env(name: str, *, required: bool = False, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        if required:
            raise RuntimeError(f"{name} is required in live mode")
        return default
    value = float(raw)
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite")
    return value


class _StatusNode(Node):
    """Common ROS status/input surface; it owns no transport."""

    _MIN_HEIGHT_OFFSET_M = -0.12
    _MAX_HEIGHT_OFFSET_M = 0.02
    _MAX_ROLL_RAD = math.radians(8.0)
    _MAX_PITCH_RAD = math.radians(12.0)
    _MAX_YAW_RAD = math.radians(8.0)
    _STATE_TIMEOUT_S = 0.50
    _QUIET_LINEAR_MPS = 0.035
    _QUIET_YAW_RPS = 0.05
    _HEIGHT_TOLERANCE_M = 0.015
    _ANGLE_TOLERANCE_RAD = math.radians(2.5)

    def _init_status_surface(self, *, mode: str) -> None:
        self._bridge_mode = mode
        self._nominal_body_height_m = _finite_env(
            "Z_MANIP_GO2W_NOMINAL_BODY_HEIGHT_M",
            required=mode == "live",
        )
        self._sport_state: dict[str, Any] | None = None
        self._sport_state_received_s: float | None = None
        self._target: tuple[float, float, float, float] | None = None
        self._phase = "idle" if mode == "live" else "shadow"
        self._detail = (
            "single WebRTC owner ready"
            if mode == "live"
            else "shadow: transport was not constructed"
        )
        self._last_code: int | None = None
        self._stop_latched = False
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._posture_pub = self.create_publisher(String, POSTURE_STATE_TOPIC, qos)
        self.create_subscription(TwistStamped, POSTURE_COMMAND_TOPIC, self._posture_command, qos)
        self.create_subscription(Empty, POSTURE_CANCEL_TOPIC, self._full_stop, qos)
        self.create_subscription(Empty, FULL_STOP_TOPIC, self._full_stop, qos)
        self.create_subscription(Empty, CONTROL_RESET_TOPIC, self._control_reset, qos)
        self.create_timer(0.10, self._publish_status)

    def _validate_posture(self, message: TwistStamped) -> tuple[float, float, float, float]:
        if message.header.frame_id not in ("", "base_link"):
            raise ValueError("posture command frame must be base_link")
        values = (
            float(message.twist.linear.z),
            float(message.twist.angular.x),
            float(message.twist.angular.y),
            float(message.twist.angular.z),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("posture command must be finite")
        height, roll, pitch, yaw = values
        if not self._MIN_HEIGHT_OFFSET_M <= height <= self._MAX_HEIGHT_OFFSET_M:
            raise ValueError("body-height offset is outside the calibrated envelope")
        if abs(roll) > self._MAX_ROLL_RAD:
            raise ValueError("roll is outside the calibrated envelope")
        if abs(pitch) > self._MAX_PITCH_RAD:
            raise ValueError("pitch is outside the calibrated envelope")
        if abs(yaw) > self._MAX_YAW_RAD:
            raise ValueError("yaw is outside the calibrated envelope")
        return values

    def _feedback_age_s(self) -> float | None:
        if self._sport_state_received_s is None:
            return None
        return max(0.0, time.monotonic() - self._sport_state_received_s)

    def _fresh_feedback(self) -> tuple[bool, str]:
        age_s = self._feedback_age_s()
        if self._sport_state is None or age_s is None:
            return False, "measured SPORT state is unavailable"
        if age_s > self._STATE_TIMEOUT_S:
            return False, f"measured SPORT state is stale ({age_s:.3f}s)"
        return True, "measured SPORT state is fresh"

    def _base_quiet(self) -> tuple[bool, str]:
        fresh, detail = self._fresh_feedback()
        if not fresh:
            return False, detail
        assert self._sport_state is not None
        velocity = self._sport_state["velocity"]
        planar = math.hypot(float(velocity[0]), float(velocity[1]))
        if planar > self._QUIET_LINEAR_MPS or abs(float(velocity[2])) > self._QUIET_YAW_RPS:
            return False, "base must settle before changing posture"
        return True, "base is quiet"

    def _on_sport_mode_state(self, message: Any) -> None:
        try:
            data = message.get("data", message)
            imu = data.get("imu_state", {})
            rpy = imu.get("rpy", data.get("rpy")) if isinstance(imu, dict) else data.get("rpy")
            velocity = data.get("velocity")
            values = (
                float(data["body_height"]),
                float(rpy[0]), float(rpy[1]), float(rpy[2]),
                float(velocity[0]), float(velocity[1]), float(velocity[2]),
            )
            if not all(math.isfinite(value) for value in values):
                return
            self._sport_state = {
                "body_height": values[0],
                "rpy": list(values[1:4]),
                "velocity": list(values[4:7]),
            }
            self._sport_state_received_s = time.monotonic()
            self._update_reached_phase()
        except (KeyError, IndexError, TypeError, ValueError):
            return

    def _update_reached_phase(self) -> None:
        if self._target is None or self._sport_state is None or self._stop_latched:
            return
        offset, roll, pitch, yaw = self._target
        expected_height = self._nominal_body_height_m + offset
        measured_rpy = self._sport_state["rpy"]
        reached = (
            abs(float(self._sport_state["body_height"]) - expected_height)
            <= self._HEIGHT_TOLERANCE_M
            and abs(float(measured_rpy[0]) - roll) <= self._ANGLE_TOLERANCE_RAD
            and abs(float(measured_rpy[1]) - pitch) <= self._ANGLE_TOLERANCE_RAD
            and abs(float(measured_rpy[2]) - yaw) <= self._ANGLE_TOLERANCE_RAD
        )
        if reached:
            self._phase = "reached"
            self._detail = "measured body posture reached the requested target"

    def _status_document(self) -> dict[str, Any]:
        age_s = self._feedback_age_s()
        fresh = age_s is not None and age_s <= self._STATE_TIMEOUT_S
        state = self._sport_state or {}
        target = self._target
        current_height = state.get("body_height")
        target_height = (
            None if target is None else self._nominal_body_height_m + target[0]
        )
        current_rpy = state.get("rpy") or (None, None, None)
        velocity = state.get("velocity") or (None, None, None)
        linear_speed = (
            None if velocity[0] is None else math.hypot(float(velocity[0]), float(velocity[1]))
        )
        return {
            "schema": STATUS_SCHEMA,
            "mode": self._bridge_mode,
            "phase": self._phase,
            "command_owner": "full_stop" if self._stop_latched else (
                "posture" if self._phase in {"commanding", "settling"} else "none"
            ),
            "stop_latched": self._stop_latched,
            "detail": self._detail,
            "body_height": {
                "current_m": current_height,
                "nominal_m": self._nominal_body_height_m,
                "target_offset_m": None if target is None else target[0],
                "target_m": target_height,
                "error_m": (
                    None if current_height is None or target_height is None
                    else target_height - float(current_height)
                ),
                "feedback_age_s": age_s,
            },
            "attitude": {
                "current_roll_rad": current_rpy[0],
                "current_pitch_rad": current_rpy[1],
                "current_yaw_rad": current_rpy[2],
                "target_roll_rad": None if target is None else target[1],
                "target_pitch_rad": None if target is None else target[2],
                "target_yaw_rad": None if target is None else target[3],
            },
            "base": {
                "linear_speed_mps": linear_speed,
                "yaw_rate_rps": velocity[2],
            },
            "feedback": {"fresh": fresh, "source": "sport_mode_state"},
            "command": {"last_robot_code": self._last_code},
            "updated_unix_ns": time.time_ns(),
        }

    def _publish_status(self) -> None:
        message = String()
        message.data = json.dumps(self._status_document(), separators=(",", ":"), allow_nan=False)
        self._posture_pub.publish(message)


class ShadowReactiveControlNode(_StatusNode):
    """Observable command surface that never constructs a WebRTC owner."""

    def __init__(self) -> None:
        Node.__init__(self, "z_manip_go2w_reactive_shadow")
        self._init_status_surface(mode="shadow")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(TwistStamped, "/cmd_vel_safe", self._shadow_move, qos)
        self.get_logger().warning("SHADOW ONLY: no Unitree WebRTC connection was opened")

    def _shadow_move(self, message: TwistStamped) -> None:
        if self._stop_latched:
            return
        self._phase = "shadow"
        self._detail = "shadow: Move observed but not transmitted"

    def _posture_command(self, message: TwistStamped) -> None:
        try:
            self._target = self._validate_posture(message)
            self._phase = "shadow"
            self._detail = "shadow: BodyHeight/Euler observed but not transmitted"
        except ValueError as error:
            self._phase = "blocked"
            self._detail = str(error)

    def _full_stop(self, _message: Empty) -> None:
        self._stop_latched = True
        self._target = None
        self._phase = "stopped"
        self._detail = "shadow: queues cleared; transport does not exist"

    def _control_reset(self, _message: Empty) -> None:
        self._stop_latched = False
        self._phase = "shadow"
        self._detail = "shadow reset; transport does not exist"


class ReactiveUnitreeControlNode(UnitreeControlNode, _StatusNode):
    """The only process allowed to own live Go2W SPORT requests."""

    _MIN_COMMAND_PERIOD_S = 0.20

    def __init__(self) -> None:
        UnitreeControlNode.__init__(self)
        self._sport_request_lock = asyncio.Lock()
        self._posture_lock = threading.Lock()
        self._posture_active = False
        self._pending_posture: tuple[float, float, float, float] | None = None
        self._posture_generation = 0
        self._last_posture_command_s = 0.0
        self._allow_moving_posture = os.environ.get(
            "Z_MANIP_GO2W_ALLOW_POSTURE_WHILE_MOVING", "0"
        ) == "1"
        self._init_status_surface(mode="live")
        self.conn.datachannel.pub_sub.subscribe(
            RTC_TOPIC["SPORT_MOD_STATE"], self._on_sport_mode_state,
        )
        self.get_logger().info(
            "LIVE single-owner bridge enabled: Move + BodyHeight + Euler + StopMove"
        )

    def cmd_vel_callback(self, message: TwistStamped) -> None:
        if self._stop_latched:
            return
        UnitreeControlNode.cmd_vel_callback(self, message)

    async def _request_sport(self, name: str, parameter: dict[str, float]) -> int | None:
        async with self._sport_request_lock:
            response = await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD[name], "parameter": parameter},
                ),
                timeout=1.0,
            )
        return _status_code(response)

    async def _drain_move_commands(self) -> None:
        """Override upstream pump so Move shares the SPORT request lock."""
        try:
            while not self._stop_latched:
                wait_s = self._move_min_period_s - (self.loop.time() - self._move_last_sent_s)
                if wait_s > 0.0:
                    await asyncio.sleep(wait_s)
                with self._move_lock:
                    command = self._pending_move
                    self._pending_move = None
                if command is None or self._stop_latched:
                    return
                x, y, yaw = command
                try:
                    self._last_code = await self._request_sport(
                        "Move", {"x": x, "y": y, "z": yaw},
                    )
                    self._move_timeout_count = 0
                    self._move_send_count += 1
                except asyncio.TimeoutError:
                    self._move_timeout_count += 1
                    self.get_logger().warning("Move ACK timed out; latest-value stream continues")
                except Exception as error:  # noqa: BLE001
                    self.get_logger().error(f"Move request failed: {error}")
                finally:
                    self._move_last_sent_s = self.loop.time()
        finally:
            with self._move_lock:
                self._move_pump_running = False
                restart = self._pending_move is not None and not self._stop_latched
                if restart:
                    self._move_pump_running = True
            if restart:
                asyncio.create_task(self._drain_move_commands())

    def _posture_command(self, message: TwistStamped) -> None:
        try:
            target = self._validate_posture(message)
        except ValueError as error:
            self._phase = "blocked"
            self._detail = str(error)
            return
        if self._stop_latched:
            self._phase = "stopped"
            self._detail = "Full Stop is latched; publish /go2w/control_reset first"
            return
        fresh, detail = self._fresh_feedback()
        if not fresh:
            self._phase = "blocked"
            self._detail = detail
            return
        if not self._allow_moving_posture:
            quiet, detail = self._base_quiet()
            if not quiet:
                self._phase = "waiting_base_quiet"
                self._detail = detail
                return
        if time.monotonic() - self._last_posture_command_s < self._MIN_COMMAND_PERIOD_S:
            return
        self._target = target
        with self._posture_lock:
            self._pending_posture = target
            if self._posture_active:
                return
            self._posture_active = True
            generation = self._posture_generation
        self._phase = "commanding"
        self._detail = "dispatching serialized BodyHeight + Euler"
        asyncio.run_coroutine_threadsafe(self._drain_posture(generation), self.loop)

    async def _drain_posture(self, generation: int) -> None:
        try:
            while generation == self._posture_generation and not self._stop_latched:
                with self._posture_lock:
                    target = self._pending_posture
                    self._pending_posture = None
                if target is None:
                    return
                height, roll, pitch, yaw = target
                for name, parameter in (
                    ("BodyHeight", {"data": height}),
                    ("Euler", {"x": roll, "y": pitch, "z": yaw}),
                ):
                    if generation != self._posture_generation or self._stop_latched:
                        return
                    self._last_code = await self._request_sport(name, parameter)
                    if self._last_code not in (0, None):
                        self._phase = "fault"
                        self._detail = f"{name} refused by robot (code={self._last_code})"
                        return
                self._last_posture_command_s = time.monotonic()
                self._phase = "settling"
                self._detail = "command accepted; waiting for measured posture"
        except Exception as error:  # noqa: BLE001
            self._phase = "fault"
            self._detail = f"posture transport failed: {type(error).__name__}: {error}"
        finally:
            with self._posture_lock:
                self._posture_active = False
                restart = (
                    self._pending_posture is not None
                    and not self._stop_latched
                    and generation == self._posture_generation
                )
                if restart:
                    self._posture_active = True
            if restart:
                asyncio.create_task(self._drain_posture(generation))

    def _full_stop(self, _message: Empty) -> None:
        self._stop_latched = True
        with self._move_lock:
            self._pending_move = None
        with self._posture_lock:
            self._posture_generation += 1
            self._pending_posture = None
        self._target = None
        self._phase = "stopping"
        self._detail = "Full Stop latched; queues flushed; StopMove owns next SPORT slot"

        async def stop() -> None:
            try:
                self._last_code = await self._request_sport("StopMove", {})
                self._phase = "stopped"
                self._detail = f"Full Stop latched; StopMove response code={self._last_code}"
            except Exception as error:  # noqa: BLE001
                self._phase = "fault"
                self._detail = f"StopMove failed while latch remains active: {error}"

        asyncio.run_coroutine_threadsafe(stop(), self.loop)

    def _control_reset(self, _message: Empty) -> None:
        fresh, detail = self._fresh_feedback()
        if not fresh:
            self._phase = "blocked"
            self._detail = f"cannot release Full Stop: {detail}"
            return
        quiet, detail = self._base_quiet()
        if not quiet:
            self._phase = "blocked"
            self._detail = f"cannot release Full Stop: {detail}"
            return
        self._stop_latched = False
        self._phase = "idle"
        self._detail = "Full Stop latch released against fresh quiet feedback"


def _parse_args(args: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    return parser.parse_known_args(args)


def main(args: list[str] | None = None) -> None:
    parsed, ros_args = _parse_args(args)
    if parsed.mode == "live" and os.environ.get("Z_MANIP_GO2W_LIVE_ACK") != LIVE_ACK:
        raise SystemExit(
            "live blocked: set Z_MANIP_GO2W_LIVE_ACK=" + LIVE_ACK + " on the NUC"
        )
    rclpy.init(args=ros_args)
    node: Node | None = None
    try:
        node = (
            ReactiveUnitreeControlNode()
            if parsed.mode == "live"
            else ShadowReactiveControlNode()
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
