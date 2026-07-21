#!/usr/bin/env python3
"""Serve one offline PiPER joint-zero report on IPv4 loopback only.

The viewer reads one bounded JSON file.  It has no ROS, CAN, robot SDK,
actuator transport, report-generation, or subprocess integration.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SCHEMA = "z_manip.piper_joint_zero_calibration.v1"
LOOPBACK = "127.0.0.1"
MAX_REPORT_BYTES = 2 * 1024 * 1024
SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; "
    "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'none'"
)


class ReportError(ValueError):
    """The report cannot be displayed under the read-only contract."""


def load_report(path: Path) -> dict[str, Any]:
    """Read one bounded report and require its fail-closed safety evidence."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReportError(f"joint-zero report does not exist: {resolved}")
    if resolved.stat().st_size > MAX_REPORT_BYTES:
        raise ReportError("joint-zero report exceeds the 2 MiB display limit")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read joint-zero report: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ReportError("unsupported PiPER joint-zero calibration report")
    if (
        document.get("read_only") is not True
        or document.get("motion_commands_published") != 0
        or document.get("urdf_modified") is not False
    ):
        raise ReportError(
            "report lacks read-only, zero-motion, unchanged-URDF provenance",
        )
    return document


class LoopbackHTTPServer(ThreadingHTTPServer):
    """Small local server with deterministic shutdown behavior."""

    daemon_threads = True
    allow_reuse_address = True


def make_handler(report_path: Path, index_path: Path) -> type[BaseHTTPRequestHandler]:
    """Close a request handler over one fixed report and HTML document."""

    report_path = report_path.expanduser().resolve()
    index_path = index_path.expanduser().resolve()
    load_report(report_path)
    if not index_path.is_file():
        raise FileNotFoundError(f"joint-zero dashboard HTML does not exist: {index_path}")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ZManipJointZeroDashboard/1"

        def _write(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus,
            include_body: bool,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", SECURITY_POLICY)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.end_headers()
            if include_body:
                self.wfile.write(payload)

        def _json(
            self,
            value: object,
            status: HTTPStatus = HTTPStatus.OK,
            include_body: bool = True,
        ) -> None:
            payload = (json.dumps(value, separators=(",", ":")) + "\n").encode("utf-8")
            self._write(
                payload,
                "application/json; charset=utf-8",
                status,
                include_body,
            )

        def _route(self, include_body: bool) -> None:
            route = urlsplit(self.path)
            if route.query or route.fragment:
                self._json(
                    {"error": "queries are not supported"},
                    HTTPStatus.BAD_REQUEST,
                    include_body,
                )
            elif route.path in {"/", "/index.html"}:
                self._write(
                    index_path.read_bytes(),
                    "text/html; charset=utf-8",
                    HTTPStatus.OK,
                    include_body,
                )
            elif route.path == "/api/report":
                try:
                    self._json(load_report(report_path), include_body=include_body)
                except ReportError as error:
                    self._json(
                        {"error": str(error)},
                        HTTPStatus.CONFLICT,
                        include_body,
                    )
            elif route.path == "/api/health":
                self._json(
                    {"ok": True, "schema": SCHEMA, "bind": LOOPBACK},
                    include_body=include_body,
                )
            else:
                self._json(
                    {"error": "not found"},
                    HTTPStatus.NOT_FOUND,
                    include_body,
                )

        def do_GET(self) -> None:  # noqa: N802 - HTTP handler contract
            self._route(True)

        def do_HEAD(self) -> None:  # noqa: N802 - HTTP handler contract
            self._route(False)

        def do_POST(self) -> None:  # noqa: N802 - HTTP handler contract
            self._json(
                {"error": "dashboard is read-only"},
                HTTPStatus.METHOD_NOT_ALLOWED,
            )

        def log_message(self, format: str, *args: object) -> None:
            print(f"joint-zero-dashboard {self.address_string()} - {format % args}")

    return Handler


def create_server(
    report_path: Path,
    *,
    port: int = 8770,
    index_path: Path | None = None,
) -> LoopbackHTTPServer:
    """Create the fixed-loopback server without starting it."""

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    root = Path(__file__).resolve().parents[2]
    page = index_path or root / "web" / "joint_zero_dashboard" / "index.html"
    return LoopbackHTTPServer(
        (LOOPBACK, port),
        make_handler(report_path, page),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = load_report(args.report)
    if args.check:
        print(json.dumps({
            "ok": True,
            "ready_for_manual_review": report.get("ready_for_manual_review"),
            "schema": SCHEMA,
        }))
        return 0
    server = create_server(args.report, port=args.port, index_path=args.index)
    print(
        "PiPER joint-zero read-only dashboard: "
        f"http://{server.server_address[0]}:{server.server_address[1]}/",
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("joint-zero dashboard stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
