#!/usr/bin/env python3
"""AnyGrasp adapter for the versioned Z-Manip msgpack-over-ZMQ contract."""

from __future__ import annotations

from argparse import Namespace
import os
import time
from typing import Any

import msgpack
import numpy as np
import zmq


PROTOCOL = "z-manip.grasp.v1"
PROVIDER = "anygrasp"
MODEL = "anygrasp-gsnet"
MODEL_VERSION = "sdk-cu128-py311"
CONVENTION = "x_closing_y_binormal_z_approach"
OPERATIONS = ("health", "metadata", "infer")
_P_GRASPNET_TO_TCP = np.array(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


def _base(**values: Any) -> dict[str, Any]:
    response = {
        "protocol": PROTOCOL,
        "status": "ok",
        "provider": PROVIDER,
        "model": MODEL,
        "model_version": MODEL_VERSION,
    }
    response.update(values)
    return response


def _array(value: object, label: str) -> np.ndarray:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data"}:
        raise ValueError(f"{label} is not a canonical array envelope")
    dtype = np.dtype(value["dtype"])
    if dtype.hasobject or not (
        np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
    ):
        raise ValueError(f"{label} has an unsupported dtype")
    shape = tuple(int(item) for item in value["shape"])
    if any(item < 0 for item in shape):
        raise ValueError(f"{label} has a negative dimension")
    data = value["data"]
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} data is not bytes")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(data) != expected:
        raise ValueError(f"{label} byte length differs from its shape")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _wire(value: object) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": array.tobytes(),
    }


def _poses(translations: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    translations = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    if len(translations) != len(rotations):
        raise ValueError("AnyGrasp translations and rotations differ in count")
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(rotations), axis=0)
    poses[:, :3, :3] = rotations @ _P_GRASPNET_TO_TCP
    depth_offset = float(os.environ.get("ANYGRASP_DEPTH_OFFSET_M", "0"))
    poses[:, :3, 3] = translations + depth_offset * rotations[:, :, 0]
    return poses


class Server:
    def __init__(self) -> None:
        from gsnet import create_detector

        checkpoint = os.environ.get(
            "ANYGRASP_CHECKPOINT",
            "/opt/anygrasp/grasp_detection/log/checkpoint_detection.tar",
        )
        config = Namespace(
            checkpoint_path=checkpoint,
            max_gripper_width=float(os.environ.get("ANYGRASP_MAX_WIDTH_M", "0.10")),
            gripper_height=float(os.environ.get("ANYGRASP_GRIPPER_HEIGHT_M", "0.03")),
        )
        started = time.perf_counter()
        self.detector = create_detector(config)
        if self.detector is None:
            raise RuntimeError("AnyGrasp detector initialization failed (check license binding)")
        print(f"AnyGrasp detector ready in {time.perf_counter() - started:.3f}s", flush=True)

    def handle(self, request: object) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("request must be a mapping")
        if set(request) != {"protocol", "operation", "payload"}:
            raise ValueError("request envelope has unexpected fields")
        if request.get("protocol") != PROTOCOL:
            raise ValueError("protocol mismatch")
        operation = request.get("operation")
        payload = request.get("payload")
        if operation not in OPERATIONS or not isinstance(payload, dict):
            raise ValueError("unsupported operation or payload")
        if operation == "health":
            return _base(ready=True)
        if operation == "metadata":
            return _base(convention=CONVENTION, operations=list(OPERATIONS))
        return self.infer(payload)

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        expected = {
            "object_points",
            "colors",
            "scene_bounds",
            "frame",
            "convention",
            "max_grasps",
        }
        if set(payload) != expected:
            raise ValueError("infer payload fields do not match the contract")
        frame = payload["frame"]
        if not isinstance(frame, str) or not frame.strip() or len(frame) > 256:
            raise ValueError("observation frame is invalid")
        if payload["convention"] != CONVENTION:
            raise ValueError("grasp convention mismatch")
        max_grasps = payload["max_grasps"]
        if isinstance(max_grasps, bool) or not isinstance(max_grasps, int) or not 1 <= max_grasps <= 512:
            raise ValueError("max_grasps is invalid")
        points = np.asarray(_array(payload["object_points"], "object_points"), dtype=np.float32)
        bounds = np.asarray(_array(payload["scene_bounds"], "scene_bounds"), dtype=np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 24:
            raise ValueError("object_points must be a non-empty Nx3 cloud")
        if bounds.shape != (2, 3) or np.any(bounds[0] >= bounds[1]):
            raise ValueError("scene_bounds must have shape (2,3) and increasing limits")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(bounds)):
            raise ValueError("inference geometry contains non-finite values")
        options = {
            "dense_grasp": os.environ.get("ANYGRASP_DENSE", "0") == "1",
            "collision_detection": True,
            "region_steering": None,
            "approach_steering": None,
            "approach_thresh": np.pi,
        }
        started = time.perf_counter()
        group = self.detector.get_grasp(points, options)
        infer_s = time.perf_counter() - started
        if group is None or len(group) == 0:
            raise RuntimeError("AnyGrasp returned no grasp candidates")
        group = group.nms().sort_by_score()
        group = group[:max_grasps]
        poses = _poses(group.translations, group.rotation_matrices)
        scores = np.asarray(group.scores, dtype=np.float32)
        widths = np.asarray(group.widths, dtype=np.float32)
        if not (
            len(poses) == len(scores) == len(widths)
            and len(poses) > 0
            and np.all(np.isfinite(poses))
            and np.all(np.isfinite(scores))
            and np.all(np.isfinite(widths))
            and np.all((0.0 <= scores) & (scores <= 1.0))
            and np.all(widths > 0.0)
        ):
            raise RuntimeError("AnyGrasp returned malformed candidates")
        print(
            f"infer points={len(points)} grasps={len(poses)} elapsed={infer_s:.3f}s",
            flush=True,
        )
        return _base(
            frame=frame,
            convention=CONVENTION,
            grasps=_wire(poses),
            scores=_wire(scores),
            widths=_wire(widths),
            diagnostics={"inference_s": infer_s, "input_points": len(points)},
        )


def main() -> None:
    server = Server()
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, 256 * 1024 * 1024)
    port = int(os.environ.get("ANYGRASP_PORT", "5557"))
    socket.bind(f"tcp://0.0.0.0:{port}")
    print(f"Z-Manip AnyGrasp server listening on tcp://0.0.0.0:{port}", flush=True)
    try:
        while True:
            try:
                request = msgpack.unpackb(socket.recv(), raw=False, strict_map_key=False)
                response = server.handle(request)
            except Exception as error:
                response = _base(status="error", error=f"{type(error).__name__}: {error}")
            socket.send(msgpack.packb(response, use_bin_type=True))
    finally:
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
