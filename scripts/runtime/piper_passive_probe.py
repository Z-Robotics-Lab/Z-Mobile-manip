#!/usr/bin/env python3
"""Passively decode PiPER joint feedback without transmitting CAN frames.

The probe deliberately does not configure or bring up the SocketCAN interface.
It opens a receive-only code path, filters for the three PiPER joint-feedback
identifiers, and verifies that the kernel TX packet counter did not increase
during the observation.  It is intended as the first real-hardware integration
gate before any ROS arm driver or controller is allowed to run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import struct
import time


JOINT_FEEDBACK_IDS = (0x2A5, 0x2A6, 0x2A7)
CAN_SFF_MASK = 0x7FF
CAN_FRAME = struct.Struct("=IB3x8s")
PAIR_BY_ID = {
    0x2A5: (0, 1),
    0x2A6: (2, 3),
    0x2A7: (4, 5),
}


def decode_joint_pair(can_id: int, payload: bytes) -> tuple[tuple[int, float], ...]:
    """Decode one two-joint PiPER feedback payload into radians."""

    frame_id = int(can_id) & CAN_SFF_MASK
    try:
        indices = PAIR_BY_ID[frame_id]
    except KeyError as error:
        raise ValueError(f"unsupported PiPER feedback CAN ID 0x{frame_id:03X}") from error
    data = bytes(payload)
    if len(data) != 8:
        raise ValueError(f"PiPER joint feedback must contain 8 bytes, got {len(data)}")
    raw_first, raw_second = struct.unpack(">ii", data)
    scale = 1e-3 * math.pi / 180.0
    return (
        (indices[0], raw_first * scale),
        (indices[1], raw_second * scale),
    )


def _counter(interface: str, name: str) -> int:
    path = Path("/sys/class/net") / interface / "statistics" / name
    return int(path.read_text(encoding="ascii").strip())


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="can0")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    values = parser.parse_args()
    if not values.interface or "/" in values.interface:
        parser.error("interface must be a non-empty network-interface name")
    if not math.isfinite(values.duration) or not 0.25 <= values.duration <= 60.0:
        parser.error("duration must be between 0.25 and 60 seconds")
    return values


def main() -> int:
    args = _arguments()
    observation_start_unix_ns = time.time_ns()
    tx_before = _counter(args.interface, "tx_packets")
    rx_before = _counter(args.interface, "rx_packets")
    counts = {frame_id: 0 for frame_id in JOINT_FEEDBACK_IDS}
    joints: list[float | None] = [None] * 6
    joint_minima: list[float | None] = [None] * 6
    joint_maxima: list[float | None] = [None] * 6
    first_received = None
    last_received = None
    first_received_unix_ns = None
    last_received_unix_ns = None
    last_by_id_unix_ns: dict[int, int] = {}
    total_frames = 0
    bus_error: str | None = None
    deadline = time.monotonic() + args.duration

    # The only socket operation after bind is recv().  In particular this
    # program contains no transmit call and installs filters before binding.
    with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as channel:
        filters = b"".join(
            struct.pack("=II", frame_id, CAN_SFF_MASK)
            for frame_id in JOINT_FEEDBACK_IDS
        )
        channel.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
        channel.settimeout(min(0.25, args.duration))
        try:
            channel.bind((args.interface,))
            while time.monotonic() < deadline:
                try:
                    frame = channel.recv(CAN_FRAME.size)
                except TimeoutError:
                    continue
                except OSError as error:
                    # can0 going bus-down mid-window (ENETDOWN [Errno 100] or
                    # ECONNRESET [Errno 104] seen after aborted executor runs,
                    # 2026-07-23) yields no further feedback.  Stop and fail
                    # closed with a legible reason instead of terminating on an
                    # unhandled traceback (SystemExit).
                    bus_error = f"can0 down during passive window: {error}"
                    break
                if len(frame) != CAN_FRAME.size:
                    continue
                can_id, dlc, data = CAN_FRAME.unpack(frame)
                frame_id = can_id & CAN_SFF_MASK
                if frame_id not in counts or dlc != 8:
                    continue
                received = time.monotonic()
                received_unix_ns = time.time_ns()
                if first_received is None:
                    first_received = received
                    first_received_unix_ns = received_unix_ns
                last_received = received
                last_received_unix_ns = received_unix_ns
                last_by_id_unix_ns[frame_id] = received_unix_ns
                total_frames += 1
                counts[frame_id] += 1
                for index, value in decode_joint_pair(frame_id, data[:dlc]):
                    joints[index] = value
                    joint_minima[index] = (
                        value
                        if joint_minima[index] is None
                        else min(joint_minima[index], value)
                    )
                    joint_maxima[index] = (
                        value
                        if joint_maxima[index] is None
                        else max(joint_maxima[index], value)
                    )
        except OSError as error:
            # bind() raises ENETDOWN when the interface is already down at the
            # start of the window; treat it identically to a mid-window drop so
            # the gate returns a legible fail-closed report rather than crashing.
            bus_error = f"can0 down during passive window: {error}"

    tx_after = _counter(args.interface, "tx_packets")
    rx_after = _counter(args.interface, "rx_packets")
    observation_end_unix_ns = time.time_ns()
    complete = all(value is not None for value in joints)
    joint_ranges = [
        None if lower is None or upper is None else upper - lower
        for lower, upper in zip(joint_minima, joint_maxima)
    ]
    tx_delta = tx_after - tx_before
    report = {
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "interface": args.interface,
        "duration_s": args.duration,
        "observation_start_unix_ns": observation_start_unix_ns,
        "observation_end_unix_ns": observation_end_unix_ns,
        "first_feedback_unix_ns": first_received_unix_ns,
        "last_feedback_unix_ns": last_received_unix_ns,
        "complete_joint_feedback": complete,
        "passive_window_error": bus_error,
        "joint_positions_rad": joints,
        "joint_positions_deg": [
            None if value is None else value * 180.0 / math.pi
            for value in joints
        ],
        "joint_ranges_rad": joint_ranges,
        "max_joint_range_rad": (
            None
            if any(value is None for value in joint_ranges)
            else max(float(value) for value in joint_ranges)
        ),
        "filtered_frame_counts": {
            f"0x{frame_id:03X}": counts[frame_id]
            for frame_id in JOINT_FEEDBACK_IDS
        },
        "filtered_frames": total_frames,
        "feedback_span_s": (
            0.0
            if first_received is None or last_received is None
            else last_received - first_received
        ),
        "joint_snapshot_span_s": (
            None
            if len(last_by_id_unix_ns) != len(JOINT_FEEDBACK_IDS)
            else (max(last_by_id_unix_ns.values()) - min(last_by_id_unix_ns.values()))
            / 1_000_000_000.0
        ),
        "interface_rx_packet_delta": rx_after - rx_before,
        "interface_tx_packet_delta": tx_delta,
        "zero_transmit_verified": tx_delta == 0,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if tx_delta != 0:
        print(
            "ERROR: can0 TX counter increased; another process may be transmitting",
            flush=True,
        )
        return 2
    if bus_error is not None:
        print(f"ERROR: {bus_error}", flush=True)
        return 1
    if not complete:
        print(
            "ERROR: no complete passive PiPER joint-feedback set was observed",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
