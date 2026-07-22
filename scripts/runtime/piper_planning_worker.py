#!/usr/bin/env python3
"""Resident, network-free client/server for offline PiPER planning.

The server runs inside the existing ``--network none`` planning container and
keeps only immutable Pinocchio/FCL robot models warm.  Requests remain fixed
CLI arguments, are path-confined to the runner's read-only artifact mount and
writable scratch mount, and execute the same fail-closed dry-run entrypoint.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from importlib.machinery import SourceFileLoader
import io
import json
import os
from pathlib import Path
import socket
import stat
import sys
import time
from typing import Sequence


MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_SOCKET = Path("/workspace-planning-output/.planner.sock")
DRY_RUN = Path("/usr/local/bin/z-manip-piper-planning-dry-run")
ARTIFACT_ROOT = Path("/workspace-artifacts")
OUTPUT_ROOT = Path("/workspace-planning-output")
CONFIG = Path("/opt/z_manip/configs/go2w_piper.json")
URDF_ROOT = Path("/robot_assets")


def _contained(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_dry_run():
    loader = SourceFileLoader("piper_planning_dry_run", str(DRY_RUN))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("planner dry-run module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(module: object, argv: Sequence[str]) -> None:
    args = module._arguments(argv)
    if not _contained(args.artifacts, ARTIFACT_ROOT):
        raise ValueError("planner artifacts escape the immutable artifact root")
    if not _contained(args.output, OUTPUT_ROOT):
        raise ValueError("planner output escapes the bounded scratch root")
    if args.config.expanduser().resolve() != CONFIG.resolve():
        raise ValueError("planner config is not the fixed deployed config")
    if not _contained(args.urdf, URDF_ROOT):
        raise ValueError("planner URDF escapes the immutable robot asset root")
    if not _contained(args.camera_calibration, ARTIFACT_ROOT):
        raise ValueError("planner calibration escapes the immutable artifact root")


def _serve(socket_path: Path) -> int:
    module = _load_dry_run()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, stat.S_IRUSR | stat.S_IWUSR)
        server.listen(4)
        while True:
            connection, _ = server.accept()
            with connection:
                started = time.perf_counter()
                try:
                    payload = bytearray()
                    while len(payload) <= MAX_REQUEST_BYTES:
                        block = connection.recv(16 * 1024)
                        if not block:
                            break
                        payload.extend(block)
                    if len(payload) > MAX_REQUEST_BYTES:
                        raise ValueError("planner request exceeds bounded size")
                    request = json.loads(bytes(payload))
                    argv = request.get("argv")
                    backend = request.get("ik_backend", "pinocchio")
                    if (
                        not isinstance(argv, list)
                        or not all(isinstance(value, str) and "\x00" not in value for value in argv)
                    ):
                        raise ValueError("planner argv must be a bounded string list")
                    if backend not in {"pinocchio", "robust"}:
                        raise ValueError("unsupported planner IK backend")
                    _validate(module, argv)
                    previous_backend = os.environ.get("Z_MANIP_IK_BACKEND")
                    os.environ["Z_MANIP_IK_BACKEND"] = backend
                    output = io.StringIO()
                    try:
                        with redirect_stdout(output), redirect_stderr(output):
                            return_code = int(module.main(argv))
                    finally:
                        if previous_backend is None:
                            os.environ.pop("Z_MANIP_IK_BACKEND", None)
                        else:
                            os.environ["Z_MANIP_IK_BACKEND"] = previous_backend
                    response = {
                        "return_code": return_code,
                        "elapsed_s": time.perf_counter() - started,
                        "output": output.getvalue(),
                    }
                except (Exception, SystemExit) as error:  # keep requests fail-closed
                    response = {
                        "return_code": 70,
                        "elapsed_s": time.perf_counter() - started,
                        "output": f"planner worker rejected request: {type(error).__name__}: {error}\n",
                    }
                encoded = json.dumps(response).encode("utf-8")
                if len(encoded) > MAX_RESPONSE_BYTES:
                    encoded = json.dumps({
                        "return_code": 70,
                        "elapsed_s": time.perf_counter() - started,
                        "output": "planner worker response exceeded bounded size\n",
                    }).encode("utf-8")
                connection.sendall(encoded)


def _client(socket_path: Path, argv: Sequence[str]) -> int:
    request = json.dumps({
        "argv": list(argv),
        "ik_backend": os.environ.get("Z_MANIP_IK_BACKEND", "pinocchio"),
    }).encode("utf-8")
    if len(request) > MAX_REQUEST_BYTES:
        raise ValueError("planner request exceeds bounded size")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        response = bytearray()
        while len(response) <= MAX_RESPONSE_BYTES:
            block = client.recv(64 * 1024)
            if not block:
                break
            response.extend(block)
    if len(response) > MAX_RESPONSE_BYTES:
        raise RuntimeError("planner worker response exceeds bounded size")
    document = json.loads(bytes(response))
    output = document.get("output", "")
    if isinstance(output, str):
        sys.stdout.write(output)
    return_code = document.get("return_code", 70)
    return int(return_code) if isinstance(return_code, int) else 70


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "client"))
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parsed, remainder = parser.parse_known_args(argv)
    if parsed.mode == "serve":
        if remainder:
            parser.error("serve mode accepts no planner arguments")
        return _serve(parsed.socket)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("client mode requires planner arguments after --")
    return _client(parsed.socket, remainder)


if __name__ == "__main__":
    raise SystemExit(main())
