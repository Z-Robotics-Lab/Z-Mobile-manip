#!/usr/bin/env python3
"""Container-local protocol health check."""

import os

import msgpack
import zmq


context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.LINGER, 0)
socket.connect(f"tcp://127.0.0.1:{int(os.environ.get('ANYGRASP_PORT', '5557'))}")
socket.send(
    msgpack.packb(
        {
            "protocol": "z-manip.grasp.v1",
            "operation": "health",
            "payload": {},
        },
        use_bin_type=True,
    ),
)
if not socket.poll(4000, zmq.POLLIN):
    raise SystemExit(1)
response = msgpack.unpackb(socket.recv(), raw=False)
socket.close(linger=0)
context.term()
if not (
    isinstance(response, dict)
    and response.get("protocol") == "z-manip.grasp.v1"
    and response.get("status") == "ok"
    and response.get("ready") is True
):
    raise SystemExit(1)
