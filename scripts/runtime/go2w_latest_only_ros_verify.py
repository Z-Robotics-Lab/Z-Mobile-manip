#!/usr/bin/env python3
"""Verify latest-only RGB-D scheduling in an isolated, actuator-free ROS domain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from z_manip_edgetam.node import EdgeTamAdapter


CAMERA_HZ = 15.0
INFERENCE_HZ = 12.8
VERIFY_DURATION_S = 12.0
MAX_ALLOWED_LAG_S = 0.25


class _UnusedServiceClient:
    """Stand in for the HTTP client; worker updates are replaced below."""


class _SyntheticRgbd(Node):
    """Publish exact-stamp RGB, aligned depth, and intrinsics at camera rate."""

    def __init__(self) -> None:
        super().__init__('latest_only_synthetic_rgbd')
        self.color_pub = self.create_publisher(
            Image,
            '/camera/color/image_raw',
            qos_profile_sensor_data,
        )
        self.depth_pub = self.create_publisher(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            qos_profile_sensor_data,
        )
        self.info_pub = self.create_publisher(
            CameraInfo,
            '/camera/color/camera_info',
            qos_profile_sensor_data,
        )
        self.latest_stamp_ns = 0
        self.publish_count = 0
        self._color = np.zeros((48, 64, 3), dtype=np.uint8)
        self._color[:, :, 1] = 96
        self._depth = np.full((48, 64), 750, dtype=np.uint16)
        self.create_timer(1.0 / CAMERA_HZ, self._publish)

    def _header(self, stamp_ns: int):
        header = Image().header
        header.stamp.sec = stamp_ns // 1_000_000_000
        header.stamp.nanosec = stamp_ns % 1_000_000_000
        header.frame_id = 'latest_only_verify_optical_frame'
        return header

    def _publish(self) -> None:
        stamp_ns = self.get_clock().now().nanoseconds
        header = self._header(stamp_ns)
        color = Image()
        color.header = header
        color.height = 48
        color.width = 64
        color.encoding = 'bgr8'
        color.is_bigendian = False
        color.step = 64 * 3
        color.data = self._color.tobytes()
        depth = Image()
        depth.header = header
        depth.height = 48
        depth.width = 64
        depth.encoding = '16UC1'
        depth.is_bigendian = False
        depth.step = 64 * 2
        depth.data = self._depth.tobytes()
        info = CameraInfo()
        info.header = header
        info.height = 48
        info.width = 64
        info.k = [60.0, 0.0, 31.5, 0.0, 60.0, 23.5, 0.0, 0.0, 1.0]
        self.color_pub.publish(color)
        self.depth_pub.publish(depth)
        self.info_pub.publish(info)
        self.latest_stamp_ns = stamp_ns
        self.publish_count += 1


def _percentile(values: list[float], percentile: float) -> float:
    return round(float(np.percentile(np.asarray(values), percentile)), 6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    domain_text = os.environ.get('ROS_DOMAIN_ID', '')
    if not domain_text.isdigit() or int(domain_text) < 100:
        raise RuntimeError(
            'latest-only verification requires an isolated ROS_DOMAIN_ID >= 100',
        )
    rclpy.init()
    adapter = EdgeTamAdapter(service_client=_UnusedServiceClient())
    adapter.set_parameters([
        Parameter('max_pending_frames', value=1),
        Parameter('sync_timeout_s', value=2.0),
        Parameter('result_timeout_s', value=60.0),
    ])
    # This harness verifies scheduling, not fault timers. Cancelling the local
    # watchdog prevents its intentional pre-first-frame timeout from becoming
    # part of the measurement.
    adapter._watchdog.cancel()
    source = _SyntheticRgbd()
    samples_lock = threading.Lock()
    processed_stamps: list[int] = []
    completion_lags_s: list[float] = []

    def slow_update(command) -> None:
        time.sleep(1.0 / INFERENCE_HZ)
        if command.frame is None:
            return
        latest_stamp_ns = source.latest_stamp_ns
        with samples_lock:
            processed_stamps.append(command.frame.stamp_ns)
            completion_lags_s.append(
                max(0, latest_stamp_ns - command.frame.stamp_ns) * 1e-9,
            )

    adapter._run_update = slow_update
    with adapter._state_lock:
        adapter._generation = 1
        adapter._accept_frames = True
        adapter._tracking = True
        adapter._active_seed_id = 'latest-only-isolated-verification'
        adapter._seed_stamp_ns = 1
        adapter._tracking_started_ros_s = adapter._now_s()
        adapter._last_result_ros_s = adapter._now_s()
        adapter._commands.clear()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(adapter)
    executor.add_node(source)
    started = time.monotonic()
    max_pending = 0
    try:
        while time.monotonic() - started < VERIFY_DURATION_S:
            executor.spin_once(timeout_sec=0.02)
            with adapter._state_lock:
                pending = sum(
                    command.kind == 'frame'
                    for command in adapter._commands
                )
            max_pending = max(max_pending, pending)
        time.sleep(2.0 / INFERENCE_HZ)
    finally:
        executor.remove_node(source)
        executor.remove_node(adapter)
        source.destroy_node()
        adapter.destroy_node()
        executor.shutdown(timeout_sec=2.0)
        rclpy.shutdown()

    with samples_lock:
        lags = list(completion_lags_s)
        stamps = list(processed_stamps)
    ordered = all(newer > older for older, newer in zip(stamps, stamps[1:]))
    max_lag_s = max(lags, default=float('inf'))
    report = {
        'read_only': True,
        'isolated_ros_domain': int(domain_text),
        'network_required': False,
        'camera_hz': CAMERA_HZ,
        'inference_hz': INFERENCE_HZ,
        'duration_s': VERIFY_DURATION_S,
        'published_frames': source.publish_count,
        'processed_frames': len(stamps),
        'max_pending_frames': max_pending,
        'strictly_increasing_processed_stamps': ordered,
        'max_source_lag_s': round(max_lag_s, 6),
        'p95_source_lag_s': _percentile(lags, 95.0) if lags else None,
        'max_allowed_lag_s': MAX_ALLOWED_LAG_S,
    }
    report['passed'] = bool(
        source.publish_count >= 150
        and len(stamps) >= 120
        and max_pending <= 1
        and ordered
        and max_lag_s <= MAX_ALLOWED_LAG_S
    )
    payload = json.dumps(report, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
