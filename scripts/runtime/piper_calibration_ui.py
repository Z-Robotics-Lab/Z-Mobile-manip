#!/usr/bin/env python3
"""Loopback-only, read-only PiPER hand-eye calibration workbench.

The workbench subscribes to wrist RGB/CameraInfo, visualizes ChArUco
detections, and exposes two fixed operations: capture one synchronized
camera/passive-CAN sample and solve the offline hand-eye dataset.  It has no ROS
publisher, robot SDK, SocketCAN handle, motion topic, or arbitrary command API.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np


LOOPBACK = "127.0.0.1"
SCHEMA = "z_manip.calibration_workbench.v1"
STREAM_BOUNDARY = b"z-manip-frame"
MAX_OBSERVATIONS = 240
MAX_CLOCK_SKEW_NS = 250_000_000
CSP = (
    "default-src 'self'; img-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


def _load_python_source(path: Path, module_name: str) -> Any:
    loader = importlib.machinery.SourceFileLoader(module_name, str(path))
    specification = importlib.util.spec_from_loader(module_name, loader)
    if specification is None:
        raise RuntimeError(f"cannot load tool: {path}")
    module = importlib.util.module_from_spec(specification)
    loader.exec_module(module)
    return module


def _stamp_ns(message: object) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _euler_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if sy > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(-float(rotation[2, 0]), sy)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(-float(rotation[2, 0]), sy)
        yaw = 0.0
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def reset_sources(dataset: Path, calibration: Path) -> list[Path]:
    """Return only calibration-session artifacts eligible for archival reset."""

    root = dataset.parent.resolve()
    if calibration.parent.resolve() != root:
        raise ValueError("dataset and calibration output must share one directory")
    candidates = [dataset, calibration, *sorted(root.glob("sample-*"))]
    sources: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved_parent = candidate.resolve().parent if candidate.is_file() else candidate.parent.resolve()
        if resolved_parent != root or candidate.is_symlink():
            raise ValueError(f"unsafe reset source: {candidate}")
        sources.append(candidate)
    return sources


def archive_calibration_session(dataset: Path, calibration: Path) -> Path | None:
    """Move one calibration session into a timestamped recoverable archive."""

    sources = reset_sources(dataset, calibration)
    if not sources:
        return None
    root = dataset.parent.resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = root / "archive" / f"reset-{stamp}-{time.time_ns() % 1_000_000_000:09d}"
    archive.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, str]] = []
    for source in sources:
        destination = archive / source.name
        shutil.move(str(source), str(destination))
        moved.append({"source": str(source), "archive": str(destination)})
    manifest = {
        "schema": "z_manip.calibration_reset_archive.v1",
        "created_unix_ns": time.time_ns(),
        "recoverable": True,
        "moved": moved,
    }
    (archive / "reset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive


@dataclass(frozen=True)
class Observation:
    source_stamp_ns: int
    camera_sample: dict[str, object]
    annotated_bgr: np.ndarray


class CalibrationState:
    """Thread-safe live image, observation, and calibration state."""

    def __init__(self, args: argparse.Namespace, charuco_tool: Any) -> None:
        self.args = args
        self.tool = charuco_tool
        board_spec = charuco_tool.load_board_metadata(args.board_metadata)
        self.board, self.dictionary = charuco_tool.make_board(**board_spec)
        self.board_spec = board_spec
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.jpeg: bytes | None = None
        self.last_frame_monotonic = 0.0
        self.detection: dict[str, object] = {
            "found": False,
            "accepted": False,
            "error": "waiting for an exact RGB/CameraInfo pair",
        }
        self.observations: deque[Observation] = deque(maxlen=MAX_OBSERVATIONS)
        self.busy_operation: str | None = None
        self.operation_message = "Ready for live preview"
        self.operation_error = ""

    def _draw_banner(
        self,
        image: np.ndarray,
        text: str,
        *,
        accepted: bool,
    ) -> np.ndarray:
        rendered = image.copy()
        color = (54, 190, 120) if accepted else (40, 150, 245)
        cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 50), (18, 24, 31), -1)
        cv2.circle(rendered, (22, 25), 7, color, -1)
        cv2.putText(
            rendered,
            text[:90],
            (40, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 247, 250),
            2,
            cv2.LINE_AA,
        )
        return rendered

    def update_pair(self, image_bgr: np.ndarray, camera_info: object, stamp_ns: int, frame_id: str) -> None:
        camera_matrix = np.asarray(camera_info.k, dtype=float).reshape(3, 3)
        distortion = np.asarray(camera_info.d, dtype=float)
        if distortion.size == 0:
            distortion = np.zeros(5)
        result: dict[str, object] | None = None
        error = "board not detected"
        try:
            result = self.tool.detect_board_pose(
                image_bgr,
                camera_matrix,
                distortion,
                self.board,
                self.dictionary,
                min_corners=4,
            )
        except ValueError as exc:
            error = str(exc)

        observation: Observation | None = None
        if result is None:
            detection = {"found": False, "accepted": False, "error": error}
            rendered = self._draw_banner(image_bgr, error, accepted=False)
        else:
            corners = int(result["charuco_corner_count"])
            markers = int(result["marker_count"])
            rmse = float(result["reprojection_rmse_px"])
            camera_from_target = np.asarray(result["camera_from_target"], dtype=float)
            distance_m = float(np.linalg.norm(camera_from_target[:3, 3]))
            accepted = corners >= self.args.min_corners and rmse <= self.args.max_rmse_px
            if corners < self.args.min_corners:
                error = f"only {corners} corners; need {self.args.min_corners}"
            elif rmse > self.args.max_rmse_px:
                error = f"RMSE {rmse:.2f}px exceeds {self.args.max_rmse_px:.2f}px"
            else:
                error = ""
            detection = {
                "found": True,
                "accepted": accepted,
                "error": error,
                "marker_count": markers,
                "corner_count": corners,
                "reprojection_rmse_px": rmse,
                "distance_m": distance_m,
                "source_stamp_ns": stamp_ns,
                "camera_frame": frame_id,
            }
            label = (
                f"READY  corners={corners}  RMSE={rmse:.2f}px  distance={distance_m:.2f}m"
                if accepted
                else error
            )
            rendered = self._draw_banner(
                np.asarray(result["annotated"]),
                label,
                accepted=accepted,
            )
            camera_sample: dict[str, object] = {
                "schema": "z_manip.charuco_camera_sample.v1",
                "read_only": True,
                "valid": accepted,
                "source_stamp_ns": stamp_ns,
                "camera_frame": frame_id,
                "target_frame": "charuco_board",
                "camera_from_target": camera_from_target.tolist(),
                "marker_count": markers,
                "charuco_corner_count": corners,
                "reprojection_rmse_px": rmse,
                "image_size": [int(camera_info.width), int(camera_info.height)],
                "board": {
                    "dictionary": self.board_spec["dictionary_name"],
                    "squares_x": self.board_spec["squares_x"],
                    "squares_y": self.board_spec["squares_y"],
                    "square_length_m": self.board_spec["square_length_m"],
                    "marker_length_m": self.board_spec["marker_length_m"],
                },
            }
            if accepted:
                observation = Observation(stamp_ns, camera_sample, rendered.copy())

        ok, encoded = cv2.imencode(".jpg", rendered, (cv2.IMWRITE_JPEG_QUALITY, 88))
        if not ok:
            return
        with self.frame_ready:
            self.jpeg = encoded.tobytes()
            self.last_frame_monotonic = time.monotonic()
            self.detection = detection
            if observation is not None:
                self.observations.append(observation)
            self.frame_ready.notify_all()

    def _dataset(self) -> dict[str, object] | None:
        try:
            document = json.loads(self.args.dataset.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return document if isinstance(document, dict) else None

    def _sample_summaries(self) -> list[dict[str, object]]:
        dataset = self._dataset()
        samples = dataset.get("samples", []) if dataset else []
        summaries: list[dict[str, object]] = []
        if not isinstance(samples, list):
            return summaries
        for index, sample in enumerate(samples):
            try:
                transform = np.asarray(sample["camera_from_target"], dtype=float)
                roll, pitch, yaw = _euler_deg(transform[:3, :3])
                translation = transform[:3, 3]
                safety = sample["safety_evidence"]
                summaries.append({
                    "index": index + 1,
                    "source_stamp_ns": int(sample["source_stamp_ns"]),
                    "roll_deg": round(roll, 2),
                    "pitch_deg": round(pitch, 2),
                    "yaw_deg": round(yaw, 2),
                    "distance_m": round(float(np.linalg.norm(translation)), 4),
                    "max_joint_range_rad": float(safety["max_joint_range_rad"]),
                    "planning_limit_violations": safety.get(
                        "planning_limit_violations",
                        [],
                    ),
                })
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        return summaries

    def _calibration_report(self) -> dict[str, object] | None:
        try:
            document = json.loads(self.args.calibration.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return document if isinstance(document, dict) else None

    def status(self) -> dict[str, object]:
        with self.lock:
            detection = dict(self.detection)
            busy = self.busy_operation
            message = self.operation_message
            error = self.operation_error
            live = self.jpeg is not None and time.monotonic() - self.last_frame_monotonic < 2.5
        samples = self._sample_summaries()
        calibration = self._calibration_report()
        return {
            "schema": SCHEMA,
            "live": live,
            "read_only": True,
            "motion_commands_published": 0,
            "detection": detection,
            "busy": busy,
            "operation_message": message,
            "operation_error": error,
            "sample_count": len(samples),
            "target_sample_count": 12,
            "minimum_sample_count": 8,
            "samples": samples,
            "calibration": calibration,
            "reset_available": bool(reset_sources(self.args.dataset, self.args.calibration)),
            "capture_only": bool(self.args.capture_only),
        }

    def _set_operation(self, busy: str | None, message: str, error: str = "") -> None:
        with self.lock:
            self.busy_operation = busy
            self.operation_message = message
            self.operation_error = error

    def begin_capture(self) -> tuple[bool, str]:
        with self.lock:
            if self.busy_operation is not None:
                return False, f"busy with {self.busy_operation}"
            if not bool(self.detection.get("accepted")):
                return False, "current board detection does not pass the quality gate"
            self.busy_operation = "capture"
            self.operation_message = "Collecting 8 s passive joint feedback…"
            self.operation_error = ""
        threading.Thread(target=self._capture_worker, daemon=True).start()
        return True, "capture started"

    def _ssh_base(self) -> list[str]:
        return [
            "ssh",
            "-i", str(self.args.ssh_key),
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.args.known_hosts}",
            "-o", "ConnectTimeout=5",
            self.args.nuc_host,
        ]

    def _capture_worker(self) -> None:
        try:
            command = self._ssh_base() + [
                "sudo", "-n", "/usr/local/sbin/z-manip-piper-passive-can-gate", "can0", "8",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-600:]
                raise RuntimeError(f"passive joint gate failed: {detail}")
            report_command = self._ssh_base() + ["cat", "/tmp/piper_passive_probe_report.json"]
            report_result = subprocess.run(
                report_command,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if report_result.returncode != 0:
                raise RuntimeError("could not retrieve the passive joint report")
            joint_report = json.loads(report_result.stdout)
            start = int(joint_report["observation_start_unix_ns"])
            end = int(joint_report["observation_end_unix_ns"])
            midpoint = (start + end) // 2
            with self.lock:
                eligible = [
                    observation
                    for observation in self.observations
                    if start - MAX_CLOCK_SKEW_NS
                    <= observation.source_stamp_ns
                    <= end + MAX_CLOCK_SKEW_NS
                ]
            if not eligible:
                raise RuntimeError(
                    "no quality-qualified camera observation overlapped the passive joint window; "
                    "hold the board and arm still, then retry"
                )
            observation = min(
                eligible,
                key=lambda candidate: abs(candidate.source_stamp_ns - midpoint),
            )
            existing = self._sample_summaries()
            sample_number = len(existing) + 1
            sample_dir = self.args.dataset.parent / (
                f"sample-{sample_number:02d}-{observation.source_stamp_ns}"
            )
            sample_dir.mkdir(parents=True, exist_ok=False)
            camera_path = sample_dir / "camera_sample.json"
            joint_path = sample_dir / "joint_report.json"
            image_path = sample_dir / "charuco_detection.jpg"
            camera_path.write_text(
                json.dumps(observation.camera_sample, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            joint_path.write_text(
                json.dumps(joint_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cv2.imwrite(str(image_path), observation.annotated_bgr)
            append_command = [
                sys.executable,
                str(self.args.sample_tool),
                "--camera-sample", str(camera_path),
                "--joint-report", str(joint_path),
                "--urdf", str(self.args.urdf),
                "--dataset", str(self.args.dataset),
            ]
            appended = subprocess.run(
                append_command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if appended.returncode != 0:
                lines = (appended.stderr or appended.stdout).strip().splitlines()
                detail = lines[-1].removeprefix("ValueError: ") if lines else "unknown failure"
                raise RuntimeError(f"sample safety gate rejected capture: {detail}")
            append_result = json.loads(appended.stdout)
            violations = append_result.get("planning_limit_violations", [])
            message = f"Sample {sample_number} accepted; move to a new pose"
            if isinstance(violations, list) and violations:
                warning = "; ".join(
                    f"J{item['joint_index']}={float(item['position_deg']):+.2f}deg "
                    f"({float(item['excess_deg']):.2f}deg {item['direction']} URDF)"
                    for item in violations
                )
                message += (
                    f". Planning-limit warning: {warning}. "
                    "Calibration kept this real stationary feedback; automatic planning stays limited by URDF."
                )
            self._set_operation(None, message)
        except Exception as exc:  # Boundary: turn all worker failures into UI state.
            self._set_operation(None, "Capture rejected", str(exc))

    def begin_solve(self) -> tuple[bool, str]:
        if self.args.capture_only:
            return False, "capture-only mode does not run the hand-eye solver"
        with self.lock:
            if self.busy_operation is not None:
                return False, f"busy with {self.busy_operation}"
            if len(self._sample_summaries()) < 8:
                return False, "at least 8 accepted samples are required"
            self.busy_operation = "solve"
            self.operation_message = "Solving hand-eye calibration offline…"
            self.operation_error = ""
        threading.Thread(target=self._solve_worker, daemon=True).start()
        return True, "solve started"

    def begin_reset(self) -> tuple[bool, str]:
        with self.lock:
            if self.busy_operation is not None:
                return False, f"busy with {self.busy_operation}"
            self.busy_operation = "reset"
            self.operation_message = "Archiving the current calibration session…"
            self.operation_error = ""
        try:
            archive = archive_calibration_session(self.args.dataset, self.args.calibration)
            if archive is None:
                self._set_operation(None, "Calibration session is already empty")
                return True, "already empty"
            self._set_operation(
                None,
                f"Previous session archived as {archive.name}; ready for sample 1",
            )
            return True, "calibration session reset"
        except Exception as exc:
            self._set_operation(None, "Reset failed", str(exc))
            return False, str(exc)

    def _solve_worker(self) -> None:
        try:
            command = [
                sys.executable,
                str(self.args.calibrate_tool),
                "--samples", str(self.args.dataset),
                "--output", str(self.args.calibration),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=40, check=False)
            report = self._calibration_report()
            if report is None:
                detail = (completed.stderr or completed.stdout).strip()[-800:]
                raise RuntimeError(f"calibration solver did not produce a report: {detail}")
            if report.get("calibrated") is not True:
                quality = report.get("quality", {})
                raise RuntimeError(
                    "quality gate failed; add more diverse poses. "
                    f"translation RMSE={quality.get('translation_rmse_m')}, "
                    f"rotation RMSE={quality.get('rotation_rmse_rad')}"
                )
            self._set_operation(None, "Calibration verified and saved")
        except Exception as exc:
            self._set_operation(None, "Calibration not accepted", str(exc))


class CameraNode:
    """Exact-stamp ROS subscriber with no publishers."""

    def __init__(self, state: CalibrationState, args: argparse.Namespace) -> None:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.rclpy = rclpy
        self.node = Node("piper_calibration_workbench_read_only")
        self.bridge = CvBridge()
        self.state = state
        self.lock = threading.Lock()
        self.images: dict[int, object] = {}
        self.infos: dict[int, object] = {}
        self.processed: deque[int] = deque(maxlen=180)
        self.node.create_subscription(Image, args.image_topic, self._image, qos_profile_sensor_data)
        self.node.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self._camera_info,
            qos_profile_sensor_data,
        )

    def _trim(self, cache: dict[int, object]) -> None:
        while len(cache) > 90:
            cache.pop(next(iter(cache)))

    def _image(self, message: object) -> None:
        stamp = _stamp_ns(message)
        with self.lock:
            self.images[stamp] = message
            self._trim(self.images)
        self._process(stamp)

    def _camera_info(self, message: object) -> None:
        stamp = _stamp_ns(message)
        with self.lock:
            self.infos[stamp] = message
            self._trim(self.infos)
        self._process(stamp)

    def _process(self, stamp: int) -> None:
        with self.lock:
            if stamp in self.processed or stamp not in self.images or stamp not in self.infos:
                return
            image = self.images.pop(stamp)
            info = self.infos.pop(stamp)
            self.processed.append(stamp)
        image_bgr = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        self.state.update_pair(image_bgr, info, stamp, image.header.frame_id)

    def spin(self) -> None:
        self.rclpy.spin(self.node)

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


class LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(state: CalibrationState, index_path: Path) -> type[BaseHTTPRequestHandler]:
    index_bytes = index_path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ZManipCalibrationWorkbench/1"

        def _headers(self, status: HTTPStatus, content_type: str, length: int | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

        def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = (json.dumps(value, separators=(",", ":")) + "\n").encode()
            self._headers(status, "application/json; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path)
            if path.query or path.fragment:
                self._json({"error": "query strings are not supported"}, HTTPStatus.BAD_REQUEST)
            elif path.path in ("/", "/index.html"):
                self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(index_bytes))
                self.wfile.write(index_bytes)
            elif path.path == "/api/status":
                self._json(state.status())
            elif path.path == "/api/stream.mjpg":
                self._stream()
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _stream(self) -> None:
            self._headers(
                HTTPStatus.OK,
                f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY.decode()}",
            )
            try:
                while True:
                    with state.frame_ready:
                        state.frame_ready.wait(timeout=1.0)
                        frame = state.jpeg
                    if frame is None:
                        continue
                    self.wfile.write(b"--" + STREAM_BOUNDARY + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path)
            if path.query or path.fragment:
                self._json({"error": "query strings are not supported"}, HTTPStatus.BAD_REQUEST)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                self.rfile.read(min(length, 4096))
            if path.path == "/api/capture":
                accepted, message = state.begin_capture()
            elif path.path == "/api/solve":
                accepted, message = state.begin_solve()
            elif path.path == "/api/reset":
                accepted, message = state.begin_reset()
            else:
                self._json({"error": "read-only workbench operation not found"}, HTTPStatus.NOT_FOUND)
                return
            status = HTTPStatus.ACCEPTED if accepted else HTTPStatus.CONFLICT
            self._json({"accepted": accepted, "message": message}, status)

        def log_message(self, format: str, *args: object) -> None:
            print(f"calibration-ui {self.address_string()} - {format % args}")

    return Handler


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--board-metadata", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--charuco-tool", type=Path, required=True)
    parser.add_argument("--sample-tool", type=Path, required=True)
    parser.add_argument("--calibrate-tool", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--nuc-host", default="yusenzlabnuc@192.168.3.8")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--max-rmse-px", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="collect synchronized samples but disable the hand-eye solve endpoint",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    if args.min_corners < 4 or args.max_rmse_px <= 0.0:
        parser.error("invalid detection quality limits")
    for path in (
        args.index,
        args.board_metadata,
        args.urdf,
        args.charuco_tool,
        args.sample_tool,
        args.calibrate_tool,
        args.ssh_key,
        args.known_hosts,
    ):
        if not path.is_file():
            parser.error(f"required file is missing: {path}")
    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    return args


def main() -> int:
    args = _arguments()
    charuco_tool = _load_python_source(args.charuco_tool, "z_manip_charuco_runtime")
    state = CalibrationState(args, charuco_tool)
    import rclpy

    rclpy.init()
    camera = CameraNode(state, args)
    ros_thread = threading.Thread(target=camera.spin, name="calibration-ros", daemon=True)
    ros_thread.start()
    server = LoopbackServer((LOOPBACK, args.port), make_handler(state, args.index))
    print(f"PiPER calibration workbench: http://{LOOPBACK}:{args.port}/")
    print("safety: no ROS publishers, no robot SDK, no CAN TX, loopback HTTP only")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
