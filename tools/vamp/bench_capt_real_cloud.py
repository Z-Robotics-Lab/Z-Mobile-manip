#!/usr/bin/env python3
"""Benchmark VAMP CAPT whole-arm collision checking against a REAL recorded scene cloud.

Baseline this must beat: the current capsule/KD-tree architecture measures
1.76 ms per whole-arm state check (568 states/s) on a 26,454-point scene cloud
recorded from this robot.

Frame note (important, read before trusting the numbers): the recorded cloud
`scene_collision_points.npy` is stored in `camera_color_optical_frame`, and the
D435 is mounted eye-in-hand on the PiPER wrist, so recovering the true arm-base
pose of the cloud needs the wrist FK at capture time -- which is not stored
alongside the cloud. For a *timing* benchmark what matters is that the cloud
genuinely overlaps the arm's swept workspace, because a cloud outside the
workspace makes every query early-exit and reports an unrealistically fast
number. This script therefore recentres the recorded cloud into the arm's
workspace (preserving its true shape, density and point count) and reports how
many points actually land within reach. The geometry is representative; it is
not the exact recorded camera pose.

Usage:
    python3 bench_capt_real_cloud.py [robot] [--cloud PATH]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import vamp

DEFAULT_CLOUD = (
    "/home/yusenzlabpc/Z-Robotics-Lab/artifacts/go2w_real/latest/scene_collision_points.npy"
)
BASELINE_MS = 1.76  # current capsule/KD-tree whole-arm state check


def summarise(name: str, values_ms: np.ndarray) -> str:
    return (
        f"{name}: median {np.median(values_ms):.4f} ms  "
        f"mean {values_ms.mean():.4f} ms  "
        f"p95 {np.percentile(values_ms, 95):.4f} ms"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", nargs="?", default="ur5")
    ap.add_argument("--cloud", default=DEFAULT_CLOUD)
    ap.add_argument("--states", type=int, default=100_000)
    ap.add_argument("--reach", type=float, default=None)
    ap.add_argument("--origin", type=float, nargs=3, default=None)
    args = ap.parse_args()

    mod = getattr(vamp, args.robot)
    dim = mod.dimension()
    r_min, r_max = mod.min_max_radii()

    origin = args.origin
    if origin is None:
        origin = vamp.constants.ROBOT_FIRST_JOINT_LOCATIONS.get(args.robot, [0.0, 0.0, 0.0])
    origin = np.asarray(origin, dtype=np.float64)

    reach = args.reach
    if reach is None:
        reach = vamp.constants.ROBOT_MAX_RADII.get(args.robot, 1.2)

    pts = np.load(args.cloud).astype(np.float32)
    n_raw = len(pts)

    print("=" * 72)
    print(f"cloud      : {args.cloud}")
    print(f"             {pts.shape} {pts.dtype}")
    print(f"robot      : {args.robot}  dim={dim}  spheres={mod.n_spheres()}")
    print(f"             collision-sphere radii  min={r_min:.4f} m  max={r_max:.4f} m")
    print(f"reach/cull : {reach} m about origin {origin.tolist()}")
    print("=" * 72)

    # Recentre the recorded cloud into the arm's workspace. This preserves the
    # cloud's real shape/density/count; it only moves it so that the CAPT is
    # actually exercised rather than trivially missed.
    centroid = pts.mean(axis=0)
    shifted = (pts - centroid).astype(np.float32) + origin.astype(np.float32)

    d = np.linalg.norm(shifted - origin, axis=1)
    print(f"points within reach ({reach} m): {(d <= reach).sum()} / {n_raw}")
    print(f"radial distance from origin: min {d.min():.3f}  median {np.median(d):.3f}  max {d.max():.3f} m")

    bbox_lo = origin - reach
    bbox_hi = origin + reach

    # --- CAPT filter + build, timed by VAMP itself (nanoseconds) ---
    cloud_list = shifted.tolist()
    filter_times, build_times, kept = [], [], None
    for _ in range(10):
        filtered_pc, t_filter_ns = vamp.filter_pointcloud(
            cloud_list,
            vamp.POINT_RADIUS,
            float(reach),
            origin.tolist(),
            bbox_lo.tolist(),
            bbox_hi.tolist(),
            True,
        )
        env_tmp = vamp.Environment()
        t_build_ns = env_tmp.add_pointcloud(filtered_pc, r_min, r_max, vamp.POINT_RADIUS)
        filter_times.append(t_filter_ns / 1e6)
        build_times.append(t_build_ns / 1e6)
        kept = len(filtered_pc)

    print()
    print(f"filter: {n_raw} -> {kept} points  ({summarise('time', np.array(filter_times))})")
    print(f"build : {summarise('CAPT build', np.array(build_times))}")

    # Rebuild one environment to query against.
    filtered_pc, _ = vamp.filter_pointcloud(
        cloud_list,
        vamp.POINT_RADIUS,
        float(reach),
        origin.tolist(),
        bbox_lo.tolist(),
        bbox_hi.tolist(),
        True,
    )
    env = vamp.Environment()
    env.add_pointcloud(filtered_pc, r_min, r_max, vamp.POINT_RADIUS)

    # --- whole-arm state checks ---
    lo = np.array(mod.lower_bounds(), dtype=np.float32)
    hi = np.array(mod.upper_bounds(), dtype=np.float32)
    rng = np.random.default_rng(0)
    N = args.states
    configs = rng.uniform(lo, hi, size=(N, dim)).astype(np.float32)

    for c in configs[:1000]:  # warm up
        mod.validate(c, env)

    t0 = time.perf_counter()
    n_valid = 0
    for c in configs:
        n_valid += mod.validate(c, env)
    t_tot = time.perf_counter() - t0

    per_state_us = t_tot / N * 1e6
    print()
    print("-" * 72)
    print(f"whole-arm state checks: {N:,} states against {kept:,}-point CAPT")
    print(f"  total            : {t_tot * 1e3:.2f} ms")
    print(f"  per state        : {per_state_us:.3f} us  ({per_state_us * 1000:.1f} ns)")
    print(f"  throughput       : {N / t_tot:,.0f} states/s")
    print(f"  collision-free   : {n_valid}/{N} ({100.0 * n_valid / N:.1f}%)")
    print()
    print(f"  BASELINE (capsule/KD-tree): {BASELINE_MS} ms/state = {1000.0 / BASELINE_MS:.0f} states/s")
    print(f"  SPEEDUP                   : {BASELINE_MS * 1000.0 / per_state_us:,.0f}x")
    print("-" * 72)
    print()
    print("NOTE: the per-state figure above includes Python call overhead for")
    print("      vamp.<robot>.validate(). VAMP's own planners call the C++ checker")
    print("      directly, so in-planner checks are faster than this number.")


if __name__ == "__main__":
    main()
