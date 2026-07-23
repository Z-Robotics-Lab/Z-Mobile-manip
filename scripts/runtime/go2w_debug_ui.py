#!/usr/bin/env python3
"""Serve one recorded Z-Manip debug bundle on a loopback-only dashboard.

This process is an offline artifact reader.  It deliberately has no ROS,
SocketCAN, robot SDK, actuator transport, or subprocess integration.  The only
network listener is an HTTP server fixed to the IPv4 loopback address.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


SCHEMA = "z_manip.debug_bundle.v1"
LOOPBACK = "127.0.0.1"
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; "
    "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'none'"
)
CONTROL_ACTION_HEADER = "X-Z-Manip-Action"
CONTROL_ACTION_VALUE = "planning-only"


class BundleError(ValueError):
    """A bundle cannot be safely interpreted by this dashboard."""


def load_bundle(path: Path) -> dict[str, Any]:
    """Load and minimally validate one bounded v1 debug bundle."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BundleError(f"debug bundle does not exist: {resolved}")
    if resolved.stat().st_size > MAX_BUNDLE_BYTES:
        raise BundleError("debug bundle exceeds the 64 MiB display limit")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read debug bundle: {error}") from error
    if not isinstance(document, dict):
        raise BundleError("debug bundle must contain a JSON object")
    if document.get("schema") != SCHEMA:
        raise BundleError(f"unsupported debug bundle schema: {document.get('schema')!r}")
    for key in ("mode", "safety", "stages", "artifacts", "visualization"):
        if key not in document:
            raise BundleError(f"debug bundle is missing required field: {key}")
    if not isinstance(document["stages"], list):
        raise BundleError("debug bundle stages must be a list")
    if not isinstance(document["artifacts"], dict):
        raise BundleError("debug bundle artifacts must be an object")
    return document


def _artifact_path(bundle_path: Path, document: dict[str, Any], key: str) -> Path:
    """Resolve only a manifest-declared image; never accept a filesystem path."""

    bundle_path = bundle_path.expanduser().resolve()
    artifacts = document.get("artifacts", {})
    reference = artifacts.get(key) if isinstance(artifacts, dict) else None
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise FileNotFoundError(f"artifact key is not declared: {key}")
    candidate = (bundle_path.parent / reference["path"]).resolve()
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(f"artifact is not a display image: {key}")
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact file is missing: {key}")
    return candidate


class LoopbackHTTPServer(ThreadingHTTPServer):
    """Threaded local server with deterministic teardown for tests and tools."""

    daemon_threads = True
    allow_reuse_address = True


def make_handler(
    bundle_path: Path,
    index_path: Path,
    *,
    control_backend: Any | None = None,
    follow_bundle_symlink: bool = False,
) -> type[BaseHTTPRequestHandler]:
    """Create a handler for a fixed bundle or an opt-in planning-only control UI."""

    configured_bundle_path = bundle_path.expanduser().absolute()
    fixed_bundle_path = configured_bundle_path.resolve()
    index_path = index_path.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"dashboard HTML does not exist: {index_path}")
    runtime_scene_path = index_path.parent / "runtime_scene.js"

    def current_bundle_path() -> Path:
        return configured_bundle_path.resolve() if follow_bundle_symlink else fixed_bundle_path

    load_bundle(current_bundle_path())

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "ZManipArtifactDashboard/1"

        def _headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", SECURITY_POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()

        def _bytes(
            self,
            payload: bytes,
            content_type: str,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            include_body: bool,
        ) -> None:
            self._headers(status, content_type, len(payload))
            if include_body:
                self.wfile.write(payload)

        def _json(
            self,
            value: object,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            include_body: bool,
        ) -> None:
            payload = (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
            self._bytes(
                payload,
                "application/json; charset=utf-8",
                status=status,
                include_body=include_body,
            )

        def _route(self, *, include_body: bool) -> None:
            route = urlsplit(self.path)
            if route.query or route.fragment:
                self._json(
                    {"error": "query strings are not supported"},
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return
            if route.path in ("/", "/index.html"):
                self._bytes(
                    index_path.read_bytes(),
                    "text/html; charset=utf-8",
                    include_body=include_body,
                )
                return
            if route.path == "/runtime_scene.js":
                if not runtime_scene_path.is_file():
                    self._json(
                        {"error": "runtime scene renderer is unavailable"},
                        status=HTTPStatus.NOT_FOUND,
                        include_body=include_body,
                    )
                    return
                self._bytes(
                    runtime_scene_path.read_bytes(),
                    "text/javascript; charset=utf-8",
                    include_body=include_body,
                )
                return
            if route.path == "/api/health":
                self._json(
                    {
                        "ok": True,
                        "schema": SCHEMA,
                        "bind": LOOPBACK,
                        "control_enabled": control_backend is not None,
                    },
                    include_body=include_body,
                )
                return
            if route.path == "/api/control":
                if control_backend is None:
                    self._json(
                        {"error": "planning-only control is not enabled"},
                        status=HTTPStatus.NOT_FOUND,
                        include_body=include_body,
                    )
                    return
                try:
                    status = control_backend.status()
                except Exception as error:  # pragma: no cover - defensive server boundary
                    self._json(
                        {"error": f"control status unavailable: {error}"},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                        include_body=include_body,
                    )
                    return
                self._json(status, include_body=include_body)
                return
            if route.path == "/api/bundle":
                try:
                    document = load_bundle(current_bundle_path())
                except BundleError as error:
                    self._json(
                        {"error": str(error)},
                        status=HTTPStatus.CONFLICT,
                        include_body=include_body,
                    )
                    return
                self._json(document, include_body=include_body)
                return
            prefix = "/artifact/"
            if route.path.startswith(prefix):
                key = unquote(route.path[len(prefix):])
                if not key or "/" in key or "\\" in key or key in {".", ".."}:
                    self._json(
                        {"error": "invalid artifact key"},
                        status=HTTPStatus.NOT_FOUND,
                        include_body=include_body,
                    )
                    return
                try:
                    resolved_bundle = current_bundle_path()
                    document = load_bundle(resolved_bundle)
                    artifact = _artifact_path(resolved_bundle, document, key)
                except (BundleError, FileNotFoundError) as error:
                    self._json(
                        {"error": str(error)},
                        status=HTTPStatus.NOT_FOUND,
                        include_body=include_body,
                    )
                    return
                content_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
                self._bytes(
                    artifact.read_bytes(),
                    content_type,
                    include_body=include_body,
                )
                return
            self._json(
                {"error": "not found"},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            self._route(include_body=True)

        def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            self._route(include_body=False)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            route = urlsplit(self.path)
            if control_backend is not None and route.path == "/api/runs" and not route.query:
                origin = self.headers.get("Origin")
                expected_origin = f"http://{LOOPBACK}:{self.server.server_port}"
                if origin not in (None, expected_origin):
                    self._json(
                        {"error": "cross-origin run requests are forbidden"},
                        status=HTTPStatus.FORBIDDEN,
                        include_body=True,
                    )
                    return
                if self.headers.get(CONTROL_ACTION_HEADER) != CONTROL_ACTION_VALUE:
                    self._json(
                        {"error": "explicit planning-only action header is required"},
                        status=HTTPStatus.FORBIDDEN,
                        include_body=True,
                    )
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    content_length = -1
                if content_length != 0:
                    self._json(
                        {"error": "run endpoint accepts no request body or command arguments"},
                        status=HTTPStatus.BAD_REQUEST,
                        include_body=True,
                    )
                    return
                try:
                    result = control_backend.start()
                except Exception as error:  # pragma: no cover - defensive server boundary
                    self._json(
                        {"error": f"could not start planning-only run: {error}"},
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                        include_body=True,
                    )
                    return
                started = result.get("started") is True
                self._json(
                    result,
                    status=HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT,
                    include_body=True,
                )
                return
            self._json(
                {"error": "dashboard is read-only"},
                status=HTTPStatus.METHOD_NOT_ALLOWED,
                include_body=True,
            )

        def log_message(self, format: str, *args: object) -> None:
            # High-rate ETag polling is expected and previously produced tens
            # of thousands of journal lines per hour.  Keep POSTs, unexpected
            # routes and errors visible while suppressing routine read traffic.
            route = urlsplit(self.path).path
            quiet = (
                route in {
                    "/api/runtime",
                    "/api/camera/latest.jpg",
                    "/api/depth/latest.jpg",
                    "/api/cloud/latest.bin",
                    "/api/cloud/latest.json",
                    "/api/home/status",
                    "/api/grasp/status",
                    "/api/sessions/status",
                    "/api/components/status",
                }
                or route.startswith("/api/perception/live/")
            )
            status = str(args[1]) if len(args) > 1 else ""
            if self.command in {"GET", "HEAD"} and quiet and status in {"200", "304", "409", "503"}:
                return
            print(f"dashboard {self.address_string()} - {format % args}", flush=True)

    return DashboardHandler


def create_server(
    bundle_path: Path,
    *,
    port: int = 8766,
    index_path: Path | None = None,
    control_backend: Any | None = None,
    follow_bundle_symlink: bool = False,
) -> LoopbackHTTPServer:
    """Create, but do not start, the fixed-loopback dashboard server."""

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if index_path is None:
        root = Path(__file__).resolve().parents[2]
        resolved_index = root / "web" / "debug_dashboard" / "index.html"
    else:
        resolved_index = index_path
    handler = make_handler(
        bundle_path,
        resolved_index,
        control_backend=control_backend,
        follow_bundle_symlink=follow_bundle_symlink,
    )
    return LoopbackHTTPServer((LOOPBACK, port), handler)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--index", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the bundle and exit without opening a listener",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    bundle_path = args.bundle.expanduser().resolve()
    load_bundle(bundle_path)
    if args.check:
        print(json.dumps({"ok": True, "bundle": str(bundle_path), "schema": SCHEMA}))
        return 0
    server = create_server(bundle_path, port=args.port, index_path=args.index)
    host, port = server.server_address[:2]
    print(f"Z-Manip read-only dashboard: http://{host}:{port}/")
    print(f"bundle: {bundle_path}")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("dashboard stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
