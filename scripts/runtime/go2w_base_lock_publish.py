#!/usr/bin/env python3
"""One-shot NUC-local publisher for the Go2W base-lock command channel.

The PC orchestrator owns the base-lock state machine but the workstation host
runs FastDDS only, while the reactive live service subscribes on the NUC's
Domain-20 CycloneDDS graph.  Rather than fight cross-host discovery, the
orchestrator SSHes to the NUC and runs this helper: it publishes one
``/go2w/base_lock`` command *co-located* with the live-service subscriber (where
discovery is immediate and reliable), then reads the live service's own
``/go2w/posture_state`` back so the orchestrator gets a real acknowledgement of
the resulting lock state.

It prints a single JSON line ``{"delivered": bool, "nuc_state": ...}`` on
stdout and exits.  It never commands the robot -- it only publishes the intent
message the live service acts on -- and any failure is a non-zero exit that the
orchestrator treats as an undelivered, best-effort attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import go2w_base_lock

POSTURE_STATE_TOPIC = "/go2w/posture_state"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", choices=("0", "1"), required=True)
    parser.add_argument("--source", default="orchestrator")
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--lease-s", type=float, default=go2w_base_lock.DEFAULT_LOCK_LEASE_S)
    parser.add_argument("--publish-count", type=int, default=3)
    parser.add_argument("--ack-timeout-s", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    command = go2w_base_lock.build_command(
        lock=args.lock == "1",
        source=args.source,
        seq=args.seq,
        lease_s=args.lease_s,
    )

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    node = Node("z_manip_go2w_base_lock_publisher")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    publisher = node.create_publisher(String, go2w_base_lock.BASE_LOCK_COMMAND_TOPIC, qos)

    observed: dict[str, object] = {"nuc_state": None}

    def _on_status(message: String) -> None:
        try:
            document = json.loads(message.data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return
        block = document.get("base_lock")
        if isinstance(block, dict) and block.get("state") in ("locked", "unlocked"):
            observed["nuc_state"] = block.get("state")

    node.create_subscription(String, POSTURE_STATE_TOPIC, _on_status, qos)

    payload = String()
    payload.data = json.dumps(command, separators=(",", ":"), allow_nan=False)

    try:
        deadline = time.monotonic() + max(0.1, args.ack_timeout_s)
        published = 0
        # Publish a few times so a RELIABLE late-joining match still delivers,
        # while spinning to collect the resulting live-service status.
        while time.monotonic() < deadline:
            if published < max(1, args.publish_count):
                publisher.publish(payload)
                published += 1
            rclpy.spin_once(node, timeout_sec=0.1)
            if observed["nuc_state"] is not None and published >= max(1, args.publish_count):
                break
        delivered = published > 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.stdout.write(
        json.dumps({"delivered": delivered, "nuc_state": observed["nuc_state"]}) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
