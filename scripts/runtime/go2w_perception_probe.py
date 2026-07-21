#!/usr/bin/env python3
"""Read-only health probe for the real Go2W RGB-D/perception graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool


def _stamp(message: object) -> tuple[int, int]:
    header = getattr(message, "header")
    return header.stamp.sec, header.stamp.nanosec


def _stamp_ns(message: object) -> int:
    seconds, nanoseconds = _stamp(message)
    return seconds * 1_000_000_000 + nanoseconds


def _rate(values: list[float]) -> float:
    return (
        (len(values) - 1) / (values[-1] - values[0])
        if len(values) > 1 and values[-1] > values[0]
        else 0.0
    )


def _gap_summary(values: list[float], empty_gap_s: float) -> dict[str, float]:
    gaps = sorted(later - earlier for earlier, later in zip(values, values[1:]))
    p99_index = min(len(gaps) - 1, int(0.99 * len(gaps))) if gaps else 0
    return {
        "max": round(gaps[-1], 3) if gaps else empty_gap_s,
        "p99": round(gaps[p99_index], 3) if gaps else empty_gap_s,
    }


def _lag_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "over_0_25_s": 0,
            "over_0_50_s": 0,
            "over_1_00_s": 0,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return round(ordered[index], 6)

    return {
        "samples": len(ordered),
        "max": round(ordered[-1], 6),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "over_0_25_s": sum(value > 0.25 for value in ordered),
        "over_0_50_s": sum(value > 0.50 for value in ordered),
        "over_1_00_s": sum(value > 1.00 for value in ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--min-hz", type=float, default=5.0)
    parser.add_argument("--min-perception-hz", type=float, default=3.0)
    parser.add_argument("--max-gap", type=float, default=2.0)
    parser.add_argument("--max-result-lag", type=float, default=1.5)
    parser.add_argument("--require-perception", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.duration <= 1.0
        or args.min_hz <= 0.0
        or args.min_perception_hz <= 0.0
        or args.max_gap <= 0.0
        or args.max_result_lag <= 0.0
    ):
        parser.error(
            "duration must exceed 1 s; rates, max-gap, and max-result-lag "
            "must be positive",
        )

    rclpy.init()
    node = Node("go2w_perception_read_only_probe")
    arrivals: dict[str, list[float]] = {"color": [], "depth": [], "info": []}
    stamps: dict[str, set[tuple[int, int]]] = {
        "color": set(),
        "depth": set(),
        "info": set(),
    }
    result_arrivals: dict[str, list[float]] = {
        "overlay": [],
        "mask": [],
        "cloud": [],
    }
    result_stamps: dict[str, set[tuple[int, int]]] = {
        "overlay": set(),
        "mask": set(),
        "cloud": set(),
    }
    result_lags: dict[str, list[float]] = {
        "overlay": [],
        "mask": [],
        "cloud": [],
    }
    latest_info_stamp_ns: int | None = None
    perception_valid: bool | None = None

    def sensor_callback(name: str):
        def callback(message: object) -> None:
            nonlocal latest_info_stamp_ns
            arrivals[name].append(time.monotonic())
            stamps[name].add(_stamp(message))
            if name == "info":
                latest_info_stamp_ns = _stamp_ns(message)

        return callback

    def result_callback(name: str):
        def callback(message: object) -> None:
            stamp = _stamp(message)
            result_arrivals[name].append(time.monotonic())
            result_stamps[name].add(stamp)
            if latest_info_stamp_ns is not None:
                lag_ns = max(0, latest_info_stamp_ns - _stamp_ns(message))
                result_lags[name].append(lag_ns * 1e-9)

        return callback

    def valid_callback(message: Bool) -> None:
        nonlocal perception_valid
        perception_valid = bool(message.data)

    subscriptions = [
        node.create_subscription(
            Image,
            "/camera/color/image_raw",
            sensor_callback("color"),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            Image,
            "/camera/aligned_depth_to_color/image_raw",
            sensor_callback("depth"),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            sensor_callback("info"),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            Bool,
            "/z_manip/perception/valid",
            valid_callback,
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            Image,
            "/z_manip/perception/overlay",
            result_callback("overlay"),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            Image,
            "/z_manip/perception/target_mask",
            result_callback("mask"),
            qos_profile_sensor_data,
        ),
        node.create_subscription(
            PointCloud2,
            "/z_manip/perception/target_pointcloud",
            result_callback("cloud"),
            qos_profile_sensor_data,
        ),
    ]
    _ = subscriptions
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)

    rates = {name: _rate(values) for name, values in arrivals.items()}
    gap_stats = {
        name: _gap_summary(values, args.duration)
        for name, values in arrivals.items()
    }
    result_rates = {
        name: _rate(values)
        for name, values in result_arrivals.items()
    }
    result_gap_stats = {
        name: _gap_summary(values, args.duration)
        for name, values in result_arrivals.items()
    }
    exact = stamps["color"] & stamps["depth"] & stamps["info"]
    rgb_depth = stamps["color"] & stamps["depth"]
    exact_result_bundle = (
        result_stamps["overlay"]
        & result_stamps["mask"]
        & result_stamps["cloud"]
    )
    lag_stats = {
        name: _lag_summary(values)
        for name, values in result_lags.items()
    }
    report = {
        "read_only": True,
        "duration_s": args.duration,
        "counts": {name: len(values) for name, values in arrivals.items()},
        "hz": {name: round(value, 2) for name, value in rates.items()},
        "arrival_gap_s": gap_stats,
        "exact_rgb_depth_info_stamps": len(exact),
        "rgb_depth_match_fraction": round(
            len(rgb_depth) / max(1, len(stamps["color"])),
            3,
        ),
        "perception_valid": perception_valid,
        "perception_result_counts": {
            name: len(values)
            for name, values in result_arrivals.items()
        },
        "perception_result_hz": {
            name: round(value, 2)
            for name, value in result_rates.items()
        },
        "perception_result_arrival_gap_s": result_gap_stats,
        "exact_perception_bundle_stamps": len(exact_result_bundle),
        "perception_result_lag_s": lag_stats,
    }
    rgbd_healthy = (
        all(value >= args.min_hz for value in rates.values())
        and all(stats["max"] <= args.max_gap for stats in gap_stats.values())
        and len(exact) >= 3
        and report["rgb_depth_match_fraction"] >= 0.2
    )
    result_max_lags = [
        summary["max"]
        for summary in lag_stats.values()
        if isinstance(summary["max"], float)
    ]
    perception_healthy = (
        perception_valid is True
        and all(value >= args.min_perception_hz for value in result_rates.values())
        and all(stats["max"] <= args.max_gap for stats in result_gap_stats.values())
        and len(exact_result_bundle) >= 3
        and len(result_max_lags) == len(result_lags)
        and max(result_max_lags, default=float("inf")) <= args.max_result_lag
    )
    healthy = rgbd_healthy and (
        perception_healthy if args.require_perception else True
    )
    report["rgbd_healthy"] = rgbd_healthy
    report["perception_healthy"] = perception_healthy
    report["require_perception"] = args.require_perception
    report["healthy"] = healthy
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
