"""ROS node that publishes only complete, fresh robot joint states."""

from __future__ import annotations

import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

from .contracts import ContractError
from .robot_state import (
    ClockHandoverGuard,
    CompleteJointStateAssembler,
    movable_joint_names_from_urdf,
)


_CLOCK_GRAPH_PROBE_PERIOD_S = 0.1


class JointStateAssembler(Node):
    """Merge arm and platform proprioception for a collision-correct MoveIt state."""

    def __init__(self) -> None:
        super().__init__("z_manip_complete_joint_state")
        self.declare_parameter("robot_description_file", "")
        self.declare_parameter("input_topics", ["/piper/state", "/go2w/joint_states"])
        self.declare_parameter("output_topic", "/z_manip/motion/complete_joint_states")
        self.declare_parameter("state_max_age_s", 0.25)
        self.declare_parameter("state_max_stamp_skew_s", 0.25)
        self.declare_parameter("clock_handover_quiet_s", 0.5)

        description_path = Path(
            str(self.get_parameter("robot_description_file").value),
        ).expanduser()
        if not description_path.is_file():
            raise ContractError(f"robot description file does not exist: {description_path}")
        required = movable_joint_names_from_urdf(description_path.read_text())
        topics = tuple(dict.fromkeys(
            str(topic).strip() for topic in self.get_parameter("input_topics").value
        ))
        if not topics or any(not topic for topic in topics):
            raise ContractError("input_topics must contain non-empty ROS topic names")
        self._topics = topics
        self._use_sim_time = bool(self.get_parameter("use_sim_time").value)
        self._clock_handover_quiet_s = float(
            self.get_parameter("clock_handover_quiet_s").value,
        )
        if (
            not math.isfinite(self._clock_handover_quiet_s)
            or self._clock_handover_quiet_s <= 0.0
        ):
            raise ContractError("clock_handover_quiet_s must be finite and positive")
        self._assembler = CompleteJointStateAssembler(
            required,
            max_age_s=float(self.get_parameter("state_max_age_s").value),
            max_stamp_skew_s=float(
                self.get_parameter("state_max_stamp_skew_s").value,
            ),
            expected_sources=topics,
            require_clock=self._use_sim_time,
        )

        output_topic = str(self.get_parameter("output_topic").value).strip()
        if not output_topic:
            raise ContractError("output_topic must be non-empty")
        self._publisher = self.create_publisher(
            JointState,
            output_topic,
            qos_profile_sensor_data,
        )
        # Do not use Node._subscriptions: it is rclpy's internal entity registry.
        self._state_subscriptions = self._create_state_subscriptions()
        self._clock_subscription = None
        self._clock_handover = ClockHandoverGuard(self._clock_handover_quiet_s)
        self._next_clock_graph_probe = 0.0
        self._clock_graph_error_log_not_before = 0.0
        if self._use_sim_time:
            self._clock_subscription = self._create_clock_subscription()
        self._last_readiness: tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ] | None = None
        self.get_logger().info(
            f"waiting for {len(required)} movable joints from {list(topics)}; "
            f"complete states publish to {output_topic}",
        )

    def _create_state_subscriptions(self):
        return [
            self.create_subscription(
                JointState,
                topic,
                lambda message, source=topic: self._on_state(source, message),
                qos_profile_sensor_data,
            )
            for topic in self._topics
        ]

    def _create_clock_subscription(self):
        return self.create_subscription(
            Clock,
            "/clock",
            self._on_clock,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

    def _on_clock(self, message: Clock) -> None:
        stamp_ns = (
            int(message.clock.sec) * 1_000_000_000
            + int(message.clock.nanosec)
        )
        now = time.monotonic()
        try:
            if self._clock_handover.pending:
                publisher_gids = None
                if now >= self._next_clock_graph_probe:
                    publisher_gids = self._clock_publisher_gids()
                    self._next_clock_graph_probe = (
                        now + _CLOCK_GRAPH_PROBE_PERIOD_S
                    )
                if not self._clock_handover.observe(
                    stamp_ns,
                    now=now,
                    publisher_gids=publisher_gids,
                ):
                    return
                self._assembler.resume_clock(stamp_ns)
                self._clock_handover.finish()
                self.get_logger().warning(
                    f"ROS clock handover complete at {stamp_ns} ns",
                )
                return
            reset = self._assembler.observe_clock(stamp_ns)
        except ContractError as error:
            self.get_logger().error(f"rejecting malformed ROS clock: {error}")
            return
        if not reset:
            return

        self._clock_handover.begin(stamp_ns, now=now)
        self._next_clock_graph_probe = now
        self._last_readiness = None
        self.get_logger().warning(
            f"ROS clock reset to {stamp_ns} ns; state acceptance quarantined",
        )

    def _clock_publisher_gids(self) -> set[tuple[int, ...]]:
        try:
            endpoints = self.get_publishers_info_by_topic("/clock")
        except RuntimeError as error:
            now = time.monotonic()
            if now >= self._clock_graph_error_log_not_before:
                self.get_logger().warning(f"clock graph probe failed: {error}")
                self._clock_graph_error_log_not_before = now + 1.0
            return set()
        self._clock_graph_error_log_not_before = 0.0
        return {tuple(endpoint.endpoint_gid) for endpoint in endpoints}

    def _on_state(self, source: str, message: JointState) -> None:
        now = time.monotonic()
        try:
            self._assembler.update(
                message.name,
                message.position,
                source=source,
                stamp_ns=(
                    int(message.header.stamp.sec) * 1_000_000_000
                    + int(message.header.stamp.nanosec)
                ),
                received_at=now,
                reference_stamp_ns=(
                    None
                    if self._use_sim_time
                    else self.get_clock().now().nanoseconds
                ),
            )
            complete = self._assembler.next_snapshot(now=now)
        except ContractError as error:
            self.get_logger().error(f"rejecting malformed state from {source}: {error}")
            return

        if complete is None:
            readiness = self._assembler.readiness(now=now)
            key = (
                readiness.missing,
                readiness.stale,
                readiness.unstamped,
                readiness.inconsistent,
            )
            if key != self._last_readiness:
                self.get_logger().warning(
                    "MoveIt state blocked; "
                    f"missing={list(readiness.missing)}, "
                    f"stale={list(readiness.stale)}, "
                    f"unstamped={list(readiness.unstamped)}, "
                    f"inconsistent={list(readiness.inconsistent)}",
                )
                self._last_readiness = key
            return

        self._last_readiness = ((), (), (), ())
        output = JointState()
        output.header.stamp.sec = complete.stamp_ns // 1_000_000_000
        output.header.stamp.nanosec = complete.stamp_ns % 1_000_000_000
        output.name = list(complete.names)
        output.position = list(complete.positions)
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStateAssembler()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
