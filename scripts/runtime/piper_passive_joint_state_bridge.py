#!/usr/bin/env python3
"""Publish PiPER joint telemetry from receive-only SocketCAN feedback.

The CAN socket is filtered to the three joint-feedback identifiers and the
only socket operation after ``bind`` is ``recv``.  The ROS surface contains
one telemetry publisher and deliberately creates no subscriptions, services,
actions, or actuator transports.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import socket
import struct
import time


JOINT_FEEDBACK_IDS = (0x2A5, 0x2A6, 0x2A7)
PAIR_BY_ID = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)}
JOINT_NAMES = tuple(f"piper_joint{index}" for index in range(1, 7))
CAN_SFF_MASK = 0x7FF
CAN_FRAME = struct.Struct("=IB3x8s")


def decode_joint_pair(can_id: int, payload: bytes) -> tuple[tuple[int, float], ...]:
    frame_id = int(can_id) & CAN_SFF_MASK
    if frame_id not in PAIR_BY_ID:
        raise ValueError(f"unsupported PiPER feedback CAN ID 0x{frame_id:03X}")
    if len(payload) != 8:
        raise ValueError("PiPER joint feedback must contain exactly eight bytes")
    raw_first, raw_second = struct.unpack(">ii", payload)
    scale = 1e-3 * math.pi / 180.0
    indices = PAIR_BY_ID[frame_id]
    return ((indices[0], raw_first * scale), (indices[1], raw_second * scale))


def counter(interface: str, name: str) -> int:
    source = Path("/sys/class/net") / interface / "statistics" / name
    return int(source.read_text(encoding="ascii").strip())


def open_receive_socket(interface: str) -> socket.socket:
    channel = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    filters = b"".join(
        struct.pack("=II", frame_id, CAN_SFF_MASK)
        for frame_id in JOINT_FEEDBACK_IDS
    )
    channel.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
    channel.settimeout(0.10)
    channel.bind((interface,))
    return channel


def run(interface: str, topic: str, publish_hz: float, snapshot_span_s: float) -> int:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    tx_at_start = counter(interface, "tx_packets")
    rclpy.init()
    node = rclpy.create_node(
        "piper_passive_joint_state_bridge",
        enable_rosout=False,
        start_parameter_services=False,
    )
    publisher = node.create_publisher(JointState, topic, qos_profile_sensor_data)
    positions: list[float | None] = [None] * 6
    received_by_id: dict[int, float] = {}
    publish_period = 1.0 / publish_hz
    next_publish = time.monotonic()
    last_tx_check = next_publish
    try:
        with open_receive_socket(interface) as channel:
            while rclpy.ok():
                try:
                    frame = channel.recv(CAN_FRAME.size)
                except TimeoutError:
                    continue
                if len(frame) != CAN_FRAME.size:
                    continue
                can_id, dlc, data = CAN_FRAME.unpack(frame)
                frame_id = can_id & CAN_SFF_MASK
                if frame_id not in PAIR_BY_ID or dlc != 8:
                    continue
                now = time.monotonic()
                received_by_id[frame_id] = now
                for index, value in decode_joint_pair(frame_id, data[:dlc]):
                    positions[index] = value
                if now - last_tx_check >= 1.0:
                    if counter(interface, "tx_packets") != tx_at_start:
                        raise RuntimeError(
                            "can0 TX counter changed while passive bridge was active; "
                            "stopping rather than coexisting with an unknown transmitter"
                        )
                    last_tx_check = now
                if now < next_publish or any(value is None for value in positions):
                    continue
                if (
                    len(received_by_id) != len(JOINT_FEEDBACK_IDS)
                    or max(received_by_id.values()) - min(received_by_id.values())
                    > snapshot_span_s
                ):
                    continue
                message = JointState()
                message.header.stamp = node.get_clock().now().to_msg()
                message.header.frame_id = "piper_base_link"
                message.name = list(JOINT_NAMES)
                message.position = [float(value) for value in positions]
                publisher.publish(message)
                next_publish = now + publish_period
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--topic", default="/piper/state")
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--snapshot-span-s", type=float, default=0.05)
    values = parser.parse_args()
    if not values.interface or "/" in values.interface:
        parser.error("interface must be a plain network-interface name")
    if values.topic != "/piper/state":
        parser.error("the passive bridge topic is fixed to /piper/state")
    if not math.isfinite(values.publish_hz) or not 1.0 <= values.publish_hz <= 50.0:
        parser.error("publish-hz must be between 1 and 50")
    if (
        not math.isfinite(values.snapshot_span_s)
        or not 0.005 <= values.snapshot_span_s <= 0.25
    ):
        parser.error("snapshot-span-s must be between 0.005 and 0.25")
    return values


if __name__ == "__main__":
    raise SystemExit(run(**vars(arguments())))
