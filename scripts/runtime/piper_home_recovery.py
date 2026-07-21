#!/usr/bin/env python3
"""Bounded low-speed recovery to a measured PiPER software Home."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import piper_staged_grasp_executor as executor


def load_home(path: Path) -> np.ndarray:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "z_manip.piper_software_home.v1":
        raise executor.SafetyError("invalid PiPER Home schema")
    if document.get("capture_zero_can_tx_verified") is not True:
        raise executor.SafetyError("Home was not captured with zero CAN TX evidence")
    radians = executor._finite_vector(document.get("joint_radians"), "Home joints")
    degrees = executor._finite_vector(document.get("joint_degrees"), "Home degrees")
    if float(np.max(np.abs(np.degrees(radians) - degrees))) > 1e-3:
        raise executor.SafetyError("Home degree/radian values disagree")
    low = executor.JOINT_LIMITS_RAD[:, 0]
    high = executor.JOINT_LIMITS_RAD[:, 1]
    if np.any(radians < low) or np.any(radians > high):
        raise executor.SafetyError("Home lies outside PiPER joint limits")
    return radians


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", default="v188", choices=("default", "v183", "v188", "v189"))
    parser.add_argument("--speed-percent", type=int, default=2)
    parser.add_argument("--max-recovery-deg", type=float, default=16.0)
    parser.add_argument("--max-step-deg", type=float, default=5.0)
    parser.add_argument("--clear-electronic-estop", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    target = load_home(args.home)
    if not 1 <= args.speed_percent <= executor.MAX_SPEED_PERCENT:
        raise executor.SafetyError(
            f"Home recovery speed must be within 1-{executor.MAX_SPEED_PERCENT} percent",
        )
    if not 0.5 <= args.max_step_deg <= 5.0:
        raise executor.SafetyError("Home recovery step must be within 0.5-5 degrees")
    if not 1.0 <= args.max_recovery_deg <= 20.0:
        raise executor.SafetyError("Home recovery envelope must be within 1-20 degrees")
    if not args.execute:
        print(json.dumps({
            "schema": "z_manip.piper_home_recovery.v1",
            "dry_run": True,
            "home_deg": np.degrees(target).tolist(),
            "commands_sent": 0,
        }))
        return 0

    robot = None
    try:
        robot, effector = executor.connect_real_arm(args.channel, args.firmware)
        status_message = None
        status_deadline = time.monotonic() + 2.0
        while status_message is None and time.monotonic() < status_deadline:
            status_message = robot.get_arm_status()
            if status_message is None:
                time.sleep(0.02)
        if status_message is None:
            raise executor.SafetyError("initial arm status feedback is unavailable")
        status = getattr(status_message, "msg", None)
        arm_status = int(getattr(status, "arm_status", 0xFF))
        error_code = int(getattr(status, "err_code", 0xFFFF))
        if arm_status == 1 and error_code == 0 and args.clear_electronic_estop:
            robot.reset()
            reset_started = time.monotonic()
            settle_not_before = reset_started + 6.0
            deadline = reset_started + 10.0
            while time.monotonic() < deadline:
                reset_message = robot.get_arm_status()
                reset_status = getattr(reset_message, "msg", None)
                reset_arm = int(getattr(reset_status, "arm_status", 0xFF))
                reset_error = int(getattr(reset_status, "err_code", 0xFFFF))
                if (
                    time.monotonic() >= settle_not_before
                    and reset_arm == 0
                    and reset_error == 0
                ):
                    break
                if reset_arm not in (0, 1, 5) or reset_error not in (0, 0x003F):
                    raise executor.SafetyError(
                        f"unexpected reset fault: arm_status={reset_arm}, "
                        f"err_code=0x{reset_error:04X}",
                    )
                time.sleep(0.05)
            else:
                raise executor.SafetyError("electronic e-stop reset did not reach NORMAL")
        current, stamp = executor.wait_for_initial_arm_feedback(robot)
        current, _ = executor.wait_for_fresh_joint_feedback(
            robot,
            after_timestamp=stamp,
            timeout_s=1.5,
        )
        delta = float(np.max(np.abs(target - current)))
        if delta > math.radians(args.max_recovery_deg):
            raise executor.SafetyError(
                f"Home recovery delta {math.degrees(delta):.3f}deg exceeds "
                f"{args.max_recovery_deg:.3f}deg envelope",
            )
        print(json.dumps({
            "phase": "start",
            "current_deg": np.degrees(current).tolist(),
            "home_deg": np.degrees(target).tolist(),
            "max_delta_deg": math.degrees(delta),
        }), flush=True)
        guard = executor.CommandGuard()
        # Home is a recovery action and commonly runs immediately after boot or
        # a controller reset.  Allow the PiPER feedback/control-mode handshake
        # enough time to settle; the helper continuously holds the measured
        # pose once CAN control is confirmed.
        executor.enter_can_joint_control(robot, guard, timeout_s=5.0)
        steps = max(1, int(math.ceil(delta / math.radians(args.max_step_deg))))
        path = np.linspace(current, target, steps + 1)
        final = executor.execute_joint_path(
            robot,
            path,
            guard,
            speed_percent=args.speed_percent,
            segment_timeout_s=12.0,
            start_tolerance_rad=executor.DEFAULT_START_TOLERANCE_RAD,
            feedback_tolerance_rad=executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
        )
        # Home is the recovery boundary for the complete arm, not only J1-J6.
        # Keep the arm torqued at Home while restoring the gripper's independent
        # driver enable bit.  Preserve its measured aperture so an object is
        # never dropped merely by requesting Home.
        guard.path_motion_started = False
        gripper = executor.restore_gripper_enable_at_current_aperture(
            effector,
            guard,
        )
        print(json.dumps({
            "schema": "z_manip.piper_home_recovery.v1",
            "phase": "complete",
            "final_deg": np.degrees(final).tolist(),
            "error_deg": np.degrees(final - target).tolist(),
            "steps": steps,
            "gripper_ready": True,
            "gripper_aperture_m": gripper.aperture_m,
        }), flush=True)
        return 0
    finally:
        executor.disconnect_quietly(robot)


if __name__ == "__main__":
    raise SystemExit(main())
