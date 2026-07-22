#!/usr/bin/env python3
"""Resident request worker for the read-only perception dry-run.

The server imports OpenCV, ROS, and the grasp stack once, then executes the
unchanged dry-run entrypoint for each bounded request.  It exposes no actuator
API and accepts outputs only below the mounted artifact root.
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
DEFAULT_SOCKET = Path("/workspace-artifacts/go2w_real/.perception_runner.sock")
DRY_RUN = Path("/usr/local/bin/z-manip-go2w-perception-dry-run")
ARTIFACT_ROOT = Path("/workspace-artifacts")
WORKER_FINGERPRINT = os.environ.get("Z_MANIP_RUNTIME_FINGERPRINT", "unknown")


def _contained(path: Path, root: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_dry_run():
    loader = SourceFileLoader("go2w_perception_dry_run", str(DRY_RUN))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        raise RuntimeError("perception dry-run module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(module: object, argv: Sequence[str]) -> None:
    args = module._arguments(argv)
    output = args.output.expanduser().resolve()
    if not _contained(output, ARTIFACT_ROOT):
        raise ValueError("perception output escapes the artifact root")
    for path in (args.passive_window, args.selected_passive_window):
        if path is None or not _contained(path, output):
            raise ValueError("passive evidence must remain inside request output")
    # The interactive path has no need for a learned endpoint.  Keeping it
    # empty prevents a caller from turning this resident worker into an
    # arbitrary network client.
    if args.learned_endpoint:
        raise ValueError("resident perception does not accept learned endpoints")


def _run_validated_request(module: object, argv: Sequence[str]) -> tuple[int, str]:
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        return_code = int(module.main(argv, manage_rclpy_context=False))
    return return_code, output.getvalue()


def _serve(socket_path: Path, *, max_requests: int | None = None) -> int:
    # Heavy imports happen exactly once here, rather than once per UI click.
    module = _load_dry_run()
    resident_node = module.start_resident_context()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, stat.S_IRUSR | stat.S_IWUSR)
            server.listen(4)
            completed_requests = 0
            fatal_worker_error = False
            while max_requests is None or completed_requests < max_requests:
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
                            raise ValueError("perception request exceeds bounded size")
                        request = json.loads(bytes(payload))
                        argv = request.get("argv")
                        if (
                            not isinstance(argv, list)
                            or not all(
                                isinstance(value, str) and "\x00" not in value
                                for value in argv
                            )
                        ):
                            raise ValueError(
                                "perception argv must be a bounded string list",
                            )
                        _validate(module, argv)
                        return_code, worker_output = _run_validated_request(
                            module,
                            argv,
                        )
                        response = {
                            "return_code": return_code,
                            "elapsed_s": time.perf_counter() - started,
                            "output": worker_output,
                            "worker_fingerprint": WORKER_FINGERPRINT,
                        }
                    except (Exception, SystemExit) as error:
                        # A request exception may leave a request-scoped ROS
                        # node alive.  Reply once, then exit so Docker's
                        # restart policy rebuilds a clean resident context.
                        fatal_worker_error = True
                        response = {
                            "return_code": 70,
                            "elapsed_s": time.perf_counter() - started,
                            "output": (
                                "perception worker rejected request: "
                                f"{type(error).__name__}: {error}\n"
                            ),
                            "worker_fingerprint": WORKER_FINGERPRINT,
                        }
                    encoded = json.dumps(response).encode("utf-8")
                    if len(encoded) > MAX_RESPONSE_BYTES:
                        encoded = json.dumps({
                            "return_code": 70,
                            "elapsed_s": time.perf_counter() - started,
                            "output": "perception worker response exceeded bounded size\n",
                            "worker_fingerprint": WORKER_FINGERPRINT,
                        }).encode("utf-8")
                    connection.sendall(encoded)
                    completed_requests += 1
                    if fatal_worker_error:
                        break
        return 70 if fatal_worker_error else 0
    finally:
        module.stop_resident_context(resident_node)


def _client(socket_path: Path, argv: Sequence[str]) -> int:
    request = json.dumps({"argv": list(argv)}).encode("utf-8")
    if len(request) > MAX_REQUEST_BYTES:
        raise ValueError("perception request exceeds bounded size")
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
        raise RuntimeError("perception worker response exceeds bounded size")
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
            parser.error("serve mode accepts no perception arguments")
        return _serve(parsed.socket)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    if not remainder:
        parser.error("client mode requires perception arguments after --")
    return _client(parsed.socket, remainder)


if __name__ == "__main__":
    raise SystemExit(main())
