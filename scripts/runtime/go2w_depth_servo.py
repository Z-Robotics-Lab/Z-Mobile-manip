#!/usr/bin/env python3
"""Minimal EdgeTAM-to-Go2W depth visual-servo runtime.

The node subscribes to EdgeTAM's selected 3-D target and publishes bounded
body velocity commands.  ``shadow`` mode computes and reports the command but
never publishes it.  ``live`` mode publishes on the existing guarded command
path; this module contains no Unitree/WebRTC transport.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import statistics
import threading
import time
from typing import Any

from z_manip.control.visual_servo import VisualServoConfig, VisualServoController


STATUS_SCHEMA = "z_manip.depth_servo_status.v1"


@dataclass(frozen=True)
class DepthServoSettings:
    mode: str = "shadow"
    desired_depth_m: float = 0.50
    # PiPER can solve the final centimetres.  The legged base only needs to
    # enter a coarse near-field corridor; demanding camera-perfect alignment
    # makes body sway repeatedly reset the handoff window.
    depth_tolerance_m: float = 0.01
    lateral_tolerance_m: float = 0.12
    settle_time_s: float = 0.10
    handoff_depth_m: float = 0.52
    handoff_bearing_rad: float = math.radians(20.0)
    linear_gain: float = 0.65
    yaw_gain: float = 0.70
    # Go2W's low-speed gait is inconsistent around 0.05--0.10 m/s: the API
    # accepts the command while the body can stop making forward progress.
    # Cruise briskly in the far field and keep a gait-maintaining floor until
    # the coarse handoff cone is reached.
    min_forward_mps: float = 0.10
    max_forward_mps: float = 0.18
    max_reverse_mps: float = 0.05
    max_yaw_rps: float = 0.12
    rotate_only_bearing_rad: float = math.radians(25.0)
    yaw_deadband_rad: float = math.radians(6.0)
    target_timeout_s: float = 0.25
    tracking_loss_grace_s: float = 0.75
    target_filter_window: int = 5
    target_filter_alpha: float = 0.55
    max_target_jump_m: float = 0.20

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "live"}:
            raise ValueError("mode must be shadow or live")
        if not math.isfinite(self.target_timeout_s) or self.target_timeout_s <= 0.0:
            raise ValueError("target timeout must be finite and positive")
        if not math.isfinite(self.tracking_loss_grace_s) or self.tracking_loss_grace_s < self.target_timeout_s:
            raise ValueError("tracking-loss grace must be at least the target timeout")
        if self.target_filter_window < 1:
            raise ValueError("target filter window must be positive")
        if not 0.0 < self.target_filter_alpha <= 1.0:
            raise ValueError("target filter alpha must be in (0, 1]")
        if not math.isfinite(self.max_target_jump_m) or self.max_target_jump_m <= 0.0:
            raise ValueError("maximum target jump must be finite and positive")
        if not math.isfinite(self.handoff_depth_m) or self.handoff_depth_m <= 0.0:
            raise ValueError("handoff depth must be finite and positive")
        if (
            not math.isfinite(self.min_forward_mps)
            or not 0.0 < self.min_forward_mps <= self.max_forward_mps
        ):
            raise ValueError("minimum forward speed must be in (0, max_forward_mps]")
        if (
            not math.isfinite(self.handoff_bearing_rad)
            or not 0.0 < self.handoff_bearing_rad < math.pi / 2.0
        ):
            raise ValueError("handoff bearing must be in (0, pi/2)")


@dataclass(frozen=True)
class DepthServoOutput:
    phase: str
    proposed_linear_x: float
    proposed_angular_z: float
    published_linear_x: float
    published_angular_z: float
    depth_error_m: float | None
    yaw_error_rad: float | None
    target_age_s: float | None
    done: bool = False


class DepthServoCore:
    """ROS-free state for deterministic testing and a thin ROS adapter."""

    def __init__(self, settings: DepthServoSettings) -> None:
        self.settings = settings
        self.controller = VisualServoController(VisualServoConfig(
            desired_depth_m=settings.desired_depth_m,
            depth_tolerance_m=settings.depth_tolerance_m,
            lateral_tolerance_m=settings.lateral_tolerance_m,
            settle_time_s=settings.settle_time_s,
            linear_gain=settings.linear_gain,
            yaw_gain=settings.yaw_gain,
            max_forward_mps=settings.max_forward_mps,
            max_reverse_mps=settings.max_reverse_mps,
            max_yaw_rps=settings.max_yaw_rps,
            rotate_only_bearing_rad=settings.rotate_only_bearing_rad,
            yaw_deadband_rad=settings.yaw_deadband_rad,
        ))
        self._target: tuple[float, float, float] | None = None
        self._raw_target: tuple[float, float, float] | None = None
        self._target_received_s: float | None = None
        self._samples: deque[tuple[float, float, float]] = deque(
            maxlen=settings.target_filter_window,
        )
        self._accepted_observations = 0
        self._rejected_observations = 0
        self._done = False

    @property
    def target(self) -> tuple[float, float, float] | None:
        return self._target

    @property
    def camera_geometry(self) -> dict[str, float] | None:
        """Return camera-frame 3-D metrics without inventing base-frame data."""

        if self._target is None:
            return None
        x_m, y_m, z_m = self._target
        return {
            "camera_range_m": math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m),
            "camera_elevation_rad": math.atan2(-y_m, z_m),
        }

    @property
    def filter_stats(self) -> dict[str, int | float | None]:
        return {
            "window_samples": len(self._samples),
            "accepted": self._accepted_observations,
            "rejected_outliers": self._rejected_observations,
            "raw_x_m": None if self._raw_target is None else self._raw_target[0],
            "raw_y_m": None if self._raw_target is None else self._raw_target[1],
            "raw_z_m": None if self._raw_target is None else self._raw_target[2],
        }

    def observe_target(
        self,
        *,
        x_m: float,
        z_m: float,
        stamp_s: float,
        y_m: float = 0.0,
    ) -> bool:
        """Observe a complete optical-frame target centroid.

        ``y_m`` defaults to zero only for backward-compatible callers.  The
        ROS adapter always supplies the measured optical y coordinate.
        """

        values = (float(x_m), float(y_m), float(z_m), float(stamp_s))
        if not all(math.isfinite(value) for value in values) or z_m <= 0.0:
            return False
        raw = (float(x_m), float(y_m), float(z_m))
        self._raw_target = raw
        if self._target is not None:
            jump_m = math.sqrt(sum(
                (raw[index] - self._target[index]) ** 2 for index in range(3)
            ))
            if jump_m > self.settings.max_target_jump_m:
                self._rejected_observations += 1
                return False
        self._samples.append(raw)
        median = (
            statistics.median(sample[0] for sample in self._samples),
            statistics.median(sample[1] for sample in self._samples),
            statistics.median(sample[2] for sample in self._samples),
        )
        if self._target is None:
            self._target = median
        else:
            alpha = self.settings.target_filter_alpha
            self._target = (
                alpha * median[0] + (1.0 - alpha) * self._target[0],
                alpha * median[1] + (1.0 - alpha) * self._target[1],
                alpha * median[2] + (1.0 - alpha) * self._target[2],
            )
        self._target_received_s = float(stamp_s)
        self._accepted_observations += 1
        return True

    def reset(self) -> None:
        self._target = None
        self._raw_target = None
        self._target_received_s = None
        self._samples.clear()
        self._accepted_observations = 0
        self._rejected_observations = 0
        self._done = False
        self.controller.reset()

    def _zero(self, phase: str, age_s: float | None) -> DepthServoOutput:
        self.controller.reset()
        return DepthServoOutput(
            phase=phase,
            proposed_linear_x=0.0,
            proposed_angular_z=0.0,
            published_linear_x=0.0,
            published_angular_z=0.0,
            depth_error_m=None,
            yaw_error_rad=None,
            target_age_s=age_s,
            done=self._done,
        )

    def tick(self, *, now_s: float, tracking: bool | None) -> DepthServoOutput:
        now = float(now_s)
        if self._done:
            return self._zero("reached", 0.0)
        if self._target is None or self._target_received_s is None:
            return self._zero("waiting_target", None)
        age_s = max(0.0, now - self._target_received_s)
        if tracking is not True or age_s > self.settings.target_timeout_s:
            phase = (
                "reacquiring"
                if age_s <= self.settings.tracking_loss_grace_s
                else "tracking_lost"
            )
            return self._zero(phase, age_s)
        x_m, y_m, z_m = self._target
        yaw_error = math.atan2(x_m, z_m)
        # A Go2W body pose is not a precision fixture: one footstep can move
        # the camera by several centimetres and degrees.  Stop the base as
        # soon as the object enters the arm's coarse near-field cone, latch
        # that decision, and let fresh perception + IK solve the final pose.
        # This is intentionally one-sided in depth; we never ask the base to
        # back away after it has entered the manipulation workspace.
        if (
            z_m <= self.settings.handoff_depth_m
            and abs(yaw_error) <= self.settings.handoff_bearing_rad
        ):
            self._done = True
            return DepthServoOutput(
                phase="reached",
                proposed_linear_x=0.0,
                proposed_angular_z=0.0,
                published_linear_x=0.0,
                published_angular_z=0.0,
                depth_error_m=z_m - self.settings.desired_depth_m,
                yaw_error_rad=yaw_error,
                target_age_s=age_s,
                done=True,
            )
        # This first mobile-manipulation flow is approach-only: once the target
        # is at or inside the requested standoff band, never reverse away from
        # it.  Continue yaw centering, settle, then hand off to manipulation.
        control_z_m = max(z_m, self.settings.desired_depth_m)
        command = self.controller.update((x_m, y_m, control_z_m), stamp_s=now)
        linear_x = command.linear_x
        # Keep Go2W above its observed low-speed dead zone while it is still
        # outside the manipulation handoff. If it is already near but not
        # roughly aligned, rotate without advancing past the target.
        if linear_x > 0.0 and z_m > self.settings.handoff_depth_m:
            linear_x = max(linear_x, self.settings.min_forward_mps)
        elif z_m <= self.settings.handoff_depth_m:
            linear_x = 0.0
        phase = "approach"
        if command.converged:
            self._done = True
            phase = "reached"
        elif linear_x == 0.0 and command.angular_z == 0.0:
            phase = "settling"
        live = self.settings.mode == "live"
        return DepthServoOutput(
            phase=phase,
            proposed_linear_x=linear_x,
            proposed_angular_z=command.angular_z,
            published_linear_x=linear_x if live else 0.0,
            published_angular_z=command.angular_z if live else 0.0,
            depth_error_m=z_m - self.settings.desired_depth_m,
            yaw_error_rad=command.yaw_error_rad,
            target_age_s=age_s,
            done=self._done,
        )


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, document: dict[str, Any]) -> None:
    """Append compact bounded diagnostics without ever storing camera data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 2_000_000:
        rotated = path.with_suffix(path.suffix + ".1")
        rotated.unlink(missing_ok=True)
        os.replace(path, rotated)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--target-topic", default="/track_3d/selected_target_pointcloud")
    parser.add_argument("--tracking-topic", default="/track_3d/is_tracking")
    parser.add_argument("--velocity-topic", default="/cmd_vel")
    parser.add_argument("--desired-depth-m", type=float, default=0.50)
    parser.add_argument("--handoff-depth-m", type=float, default=0.52)
    parser.add_argument("--handoff-bearing-deg", type=float, default=20.0)
    parser.add_argument("--min-forward-mps", type=float, default=0.10)
    parser.add_argument("--max-forward-mps", type=float, default=0.18)
    parser.add_argument("--max-yaw-rps", type=float, default=0.12)
    parser.add_argument("--target-timeout-s", type=float, default=0.25)
    parser.add_argument("--tracking-loss-grace-s", type=float, default=0.75)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    return parser.parse_args()


def _run_ros(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Bool

    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0.0:
        raise ValueError("rate must be finite and positive")
    settings = DepthServoSettings(
        mode=args.mode,
        desired_depth_m=args.desired_depth_m,
        handoff_depth_m=args.handoff_depth_m,
        handoff_bearing_rad=math.radians(args.handoff_bearing_deg),
        min_forward_mps=args.min_forward_mps,
        max_forward_mps=args.max_forward_mps,
        max_yaw_rps=args.max_yaw_rps,
        target_timeout_s=args.target_timeout_s,
        tracking_loss_grace_s=args.tracking_loss_grace_s,
    )

    class DepthServoNode(Node):
        def __init__(self) -> None:
            super().__init__("z_manip_depth_servo")
            self.core = DepthServoCore(settings)
            self.tracking: bool | None = None
            self.last_source_stamp_ns: int | None = None
            self.last_source_frame: str | None = None
            self.last_output = self.core.tick(now_s=time.monotonic(), tracking=False)
            self.last_trace_phase: str | None = None
            self.last_trace_s = 0.0
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            self.publisher = self.create_publisher(TwistStamped, args.velocity_topic, 1)
            self.create_subscription(PointCloud2, args.target_topic, self._target, qos)
            self.create_subscription(Bool, args.tracking_topic, self._tracking, qos)
            self.create_timer(1.0 / args.rate_hz, self._tick)
            self._write_status("starting")

        def _target(self, message: PointCloud2) -> None:
            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            for point in point_cloud2.read_points(
                message,
                field_names=("x", "y", "z"),
                skip_nans=True,
            ):
                x_m, y_m, z_m = float(point[0]), float(point[1]), float(point[2])
                if all(math.isfinite(value) for value in (x_m, y_m, z_m)) and z_m > 0.0:
                    xs.append(x_m)
                    ys.append(y_m)
                    zs.append(z_m)
                if len(xs) >= 5000:
                    break
            if not xs:
                return
            accepted = self.core.observe_target(
                x_m=statistics.median(xs),
                y_m=statistics.median(ys),
                z_m=statistics.median(zs),
                stamp_s=time.monotonic(),
            )
            if accepted:
                self.last_source_frame = str(message.header.frame_id or "") or None
                self.last_source_stamp_ns = (
                    int(message.header.stamp.sec) * 1_000_000_000
                    + int(message.header.stamp.nanosec)
                )

        def _tracking(self, message: Bool) -> None:
            self.tracking = bool(message.data)

        def _publish(self, linear_x: float, angular_z: float) -> None:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "base_link"
            message.twist.linear.x = float(linear_x)
            message.twist.angular.z = float(angular_z)
            self.publisher.publish(message)

        def _write_status(self, state: str | None = None, *, running: bool = True) -> None:
            target = self.core.target
            document = {
                "schema": STATUS_SCHEMA,
                "running": running,
                "mode": settings.mode,
                "phase": state or self.last_output.phase,
                "tracking": self.tracking,
                "target": None if target is None else {
                    "x_m": target[0],
                    "y_m": target[1],
                    "z_m": target[2],
                    "frame_id": self.last_source_frame,
                },
                "geometry": self.core.camera_geometry,
                "source_stamp_ns": self.last_source_stamp_ns,
                "output": asdict(self.last_output),
                "filter": self.core.filter_stats,
                "trace_file": None if args.trace_file is None else str(args.trace_file),
                "pid": os.getpid(),
                "updated_unix_ns": time.time_ns(),
            }
            _atomic_json(args.status_file, document)
            now_s = time.monotonic()
            if args.trace_file is not None and (
                self.last_output.phase != self.last_trace_phase
                or now_s - self.last_trace_s >= 1.0
            ):
                _append_jsonl(args.trace_file, {
                    "schema": "z_manip.depth_servo_trace.v1",
                    "updated_unix_ns": document["updated_unix_ns"],
                    "mode": settings.mode,
                    "phase": document["phase"],
                    "tracking": self.tracking,
                    "target": document["target"],
                    "source_stamp_ns": self.last_source_stamp_ns,
                    "output": document["output"],
                    "filter": document["filter"],
                })
                self.last_trace_phase = self.last_output.phase
                self.last_trace_s = now_s

        def _tick(self) -> None:
            self.last_output = self.core.tick(
                now_s=time.monotonic(),
                tracking=self.tracking,
            )
            if settings.mode == "live":
                self._publish(
                    self.last_output.published_linear_x,
                    self.last_output.published_angular_z,
                )
            self._write_status()

        def stop(self, phase: str = "stopped") -> None:
            if settings.mode == "live":
                for _ in range(3):
                    self._publish(0.0, 0.0)
            self.last_output = DepthServoOutput(
                phase=phase,
                proposed_linear_x=0.0,
                proposed_angular_z=0.0,
                published_linear_x=0.0,
                published_angular_z=0.0,
                depth_error_m=None,
                yaw_error_rad=None,
                target_age_s=None,
                done=False,
            )
            self._write_status(phase, running=False)

    rclpy.init()
    node = DepthServoNode()
    stopped = threading.Event()
    stop_published = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stopped.set()
        # Publish the final zero while the ROS context is still valid.  Calling
        # shutdown first made the finally block raise and could leave the
        # transport relying only on its watchdog stop.
        if not stop_published.is_set():
            node.stop("stopped")
            stop_published.set()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        rclpy.spin(node)
    finally:
        if not stop_published.is_set():
            node.stop("stopped" if stopped.is_set() else "exited")
            stop_published.set()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    return _run_ros(_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
