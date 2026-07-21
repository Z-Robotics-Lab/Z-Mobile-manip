#!/usr/bin/env python3
"""Move PiPER to one server-owned wrist-camera search view.

The default invocation is a dry run and cannot import the robot SDK.  Live use
requires ``--execute`` and accepts only a finite view index plus bounded speed;
joint targets are derived locally from the measured software Home.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
import piper_staged_grasp_executor as executor

try:  # The remote wrapper copies the pure policy beside this executable.
    from wrist_search import BoundedWristSearch, WristSearchConfig  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Local development imports the installed package.
    from z_manip.control.wrist_search import BoundedWristSearch, WristSearchConfig


MAX_SEARCH_SPEED_PERCENT = 12
MAX_NON_WRIST_ERROR_RAD = math.radians(2.0)
MAX_VIEW_START_ERROR_RAD = math.radians(2.0)
STREAM_HZ = 30.0


def load_home(path: Path) -> np.ndarray:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema") != "z_manip.piper_software_home.v1"
        or document.get("capture_zero_can_tx_verified") is not True
    ):
        raise executor.SafetyError("wrist search requires measured zero-TX Home")
    home = executor._finite_vector(document.get("joint_radians"), "Home joints")
    if np.any(home < executor.JOINT_LIMITS_RAD[:, 0]) or np.any(home > executor.JOINT_LIMITS_RAD[:, 1]):
        raise executor.SafetyError("wrist-search Home lies outside joint limits")
    return home


def fixed_view_targets(home: np.ndarray) -> tuple[np.ndarray, ...]:
    search = BoundedWristSearch(WristSearchConfig())
    targets = []
    for view in search.views:
        target = home.copy()
        target[search.config.yaw_joint_index] += view.yaw_offset_rad
        target[search.config.pitch_joint_index] += view.pitch_offset_rad
        if np.any(target < executor.JOINT_LIMITS_RAD[:, 0]) or np.any(target > executor.JOINT_LIMITS_RAD[:, 1]):
            raise executor.SafetyError(f"fixed wrist view {view.index} exceeds PiPER limits")
        targets.append(target)
    return tuple(targets)


def smooth_path(start: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a reference-speed path; the executor applies speed scaling once."""
    delta = float(np.max(np.abs(target - start)))
    # The quintic blend peaks at 1.875x its average velocity.  Size the
    # reference timeline from that peak so adjacent 30 Hz commands stay small.
    duration = max(0.8, 1.875 * delta / math.radians(30.0))
    count = max(2, int(math.ceil(duration * STREAM_HZ)) + 1)
    times = np.linspace(0.0, duration, count)
    tau = times / duration
    blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    path = start[None, :] + blend[:, None] * (target - start)[None, :]
    return path, times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--view-index", type=int, required=True)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v188", choices=("default", "v183", "v188", "v189"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.speed_percent <= MAX_SEARCH_SPEED_PERCENT:
        raise executor.SafetyError(
            f"wrist-search speed must be within 1-{MAX_SEARCH_SPEED_PERCENT} percent",
        )
    home = load_home(args.home)
    targets = fixed_view_targets(home)
    if not 0 <= args.view_index < len(targets):
        raise executor.SafetyError("wrist-search view index is outside the fixed grid")
    target = targets[args.view_index]
    if not args.execute:
        print(json.dumps({
            "schema": "z_manip.piper_wrist_search_view.v1",
            "phase": "dry_run",
            "view_index": args.view_index,
            "target_joints_rad": target.tolist(),
            "commands_sent": 0,
        }))
        return 0

    robot = None
    try:
        robot, _effector = executor.connect_real_arm(args.channel, args.firmware)
        current, stamp = executor.wait_for_initial_arm_feedback(robot)
        current, _ = executor.wait_for_fresh_joint_feedback(
            robot,
            after_timestamp=stamp,
            timeout_s=1.5,
        )
        wrist_indices = (WristSearchConfig().yaw_joint_index, WristSearchConfig().pitch_joint_index)
        non_wrist = [index for index in range(6) if index not in wrist_indices]
        if float(np.max(np.abs(current[non_wrist] - home[non_wrist]))) > MAX_NON_WRIST_ERROR_RAD:
            raise executor.SafetyError("wrist search may start only from the measured Home arm posture")
        nearest = min(float(np.max(np.abs(current - candidate))) for candidate in targets)
        if nearest > MAX_VIEW_START_ERROR_RAD:
            raise executor.SafetyError("live joints do not match Home or a fixed wrist-search view")
        path, times = smooth_path(current, target)
        guard = executor.CommandGuard()
        executor.enter_can_joint_control(robot, guard, timeout_s=5.0)
        final = executor.execute_timed_joint_path(
            robot,
            path,
            times,
            guard,
            speed_percent=args.speed_percent,
            segment_timeout_s=12.0,
            start_tolerance_rad=executor.DEFAULT_START_TOLERANCE_RAD,
            feedback_tolerance_rad=executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
            reference_speed_percent=MAX_SEARCH_SPEED_PERCENT,
        )
        print(json.dumps({
            "schema": "z_manip.piper_wrist_search_view.v1",
            "phase": "complete",
            "view_index": args.view_index,
            "final_joints_rad": final.tolist(),
            "error_deg": np.degrees(final - target).tolist(),
            "samples": len(path),
            "duration_s": float(times[-1]),
        }), flush=True)
        return 0
    finally:
        executor.disconnect_quietly(robot)


if __name__ == "__main__":
    raise SystemExit(main())
