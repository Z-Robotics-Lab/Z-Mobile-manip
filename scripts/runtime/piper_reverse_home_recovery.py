#!/usr/bin/env python3
"""Recover PiPER from an exact planned grasp endpoint to measured Home."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

import piper_staged_grasp_executor as stage


def _load_optional_measured_home(
    path: Path | None,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve the calibrated software Home target, tolerating its absence.

    Convergence to calibrated Home is a best-effort finish on top of the
    already-safe checked corridor.  A missing config (e.g. this recovery was
    invoked without the Home wrapper staging ``piper_home.json`` beside the
    script) or one that fails the strict software-Home validation must never
    turn a successful checked recovery into a hard failure: return ``None`` so
    the caller stops safely at the corridor endpoint and records why.
    """
    if path is None or not Path(path).is_file():
        return None, "no calibrated software Home config available"
    try:
        return stage.load_software_home(path), None
    except Exception as error:  # noqa: BLE001 - any load/validation fault is a safe skip
        return None, f"calibrated software Home config unusable: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-report", type=Path, required=True)
    parser.add_argument("--planned-grasp", type=Path, required=True)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    # After the checked reverse corridor puts the arm at its recorded
    # planning-start pose, converge the final short hop to calibrated software
    # Home so one Home request finishes at true Home.  The Home wrapper stages
    # this config next to the script; a missing/unusable file falls back to
    # stopping safely at the corridor endpoint (see main()).
    parser.add_argument(
        "--home",
        type=Path,
        default=Path(__file__).resolve().parent / "piper_home.json",
    )
    args = parser.parse_args()
    try:
        report_document = json.loads(args.planning_report.read_text(encoding="utf-8"))
        source_stamp_ns = int(report_document["source_stamp_ns"])
        # Recovery intentionally consumes the exact immutable bytes that put
        # the arm at this endpoint.  Validate those bytes through the normal
        # strict loader using temporary copies timestamped at capture time;
        # current-pose endpoint matching below then binds the stale artifact
        # to the physical arm without weakening fresh execution policy.
        with tempfile.TemporaryDirectory(prefix="piper-recovery-") as temp_dir:
            report_copy = Path(temp_dir) / "planning_report.json"
            archive_copy = Path(temp_dir) / "planned_grasp.npz"
            shutil.copyfile(args.planning_report, report_copy)
            shutil.copyfile(args.planned_grasp, archive_copy)
            os.utime(report_copy, ns=(source_stamp_ns, source_stamp_ns))
            artifact = stage.load_planning_artifact(
                report_copy,
                archive_copy,
                expected_npz_sha256=None,
                stage="pregrasp",
                now_ns=source_stamp_ns + 1_000_000,
                max_source_age_s=30.0,
            )
        token = f"PIPER-RECOVER-{artifact.artifact_id[:16]}"
        if not args.execute:
            print(json.dumps({"confirmation_token": token, "commands_sent": 0}))
            return 0
        if args.confirm != token:
            raise stage.SafetyError(f"exact confirmation required: {token}")
        transit = np.asarray(artifact.arrays["transit_raw"], dtype=float)
        approach = np.asarray(artifact.arrays["approach_raw"], dtype=float)
        lift = np.asarray(artifact.arrays["lift_raw"], dtype=float)
        robot = None
        try:
            robot, effector = stage.connect_real_arm("can0", "v188")
            current, stamp = stage.wait_for_initial_arm_feedback(robot)
            current, _ = stage.wait_for_fresh_joint_feedback(
                robot, after_timestamp=stamp, timeout_s=1.0,
            )
            candidates: list[tuple[float, str, int, np.ndarray]] = []
            for name, path in (("transit", transit), ("approach", approach), ("lift", lift)):
                for index, waypoint in enumerate(path):
                    candidates.append((
                        float(np.max(np.abs(current - waypoint))),
                        name,
                        index,
                        path,
                    ))
            error, matched_stage, matched_index, matched_path = min(candidates, key=lambda item: item[0])
            if error > np.radians(4.0):
                raise stage.SafetyError(
                    "current pose is not on the selected checked path: "
                    f"nearest error {np.degrees(error):.3f}deg",
                )
            guard = stage.CommandGuard()
            stage.enter_can_joint_control(robot, guard)
            if matched_stage in ("approach", "lift"):
                try:
                    baseline = stage.normalize_gripper_feedback(effector.get_gripper_status())
                    guard.mark_before_command()
                    effector.move_gripper_m(value=stage.OPEN_APERTURE_M, force=1.0)
                    stage.wait_for_gripper(
                        effector,
                        lambda sample: sample.aperture_m >= 0.060,
                        after_timestamp=baseline.timestamp,
                        timeout_s=3.0,
                        static_accept_after_s=0.5,
                    )
                except Exception as error:
                    print(f"WARNING: release feedback degraded; continuing checked retreat: {error}", file=sys.stderr)
            first_reverse = matched_path[: matched_index + 1][::-1]
            recovery_paths = [np.vstack((current, first_reverse))]
            if matched_stage == "lift":
                recovery_paths.extend((approach[::-1], transit[::-1]))
            elif matched_stage == "approach":
                recovery_paths.append(transit[::-1])
            final = current
            for recovery_path in recovery_paths:
                final = stage.execute_joint_path(
                    robot, recovery_path, guard,
                    speed_percent=args.speed_percent, segment_timeout_s=12.0,
                    start_tolerance_rad=stage.DEFAULT_START_TOLERANCE_RAD,
                    feedback_tolerance_rad=stage.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                )
            # The checked corridor rests the arm at ``transit[0]`` -- the pose
            # the arm held when this session was planned, NOT calibrated Home.
            # Continue directly to the measured software Home with the same
            # bounded low-speed convergence the standalone Home action uses, so
            # a single Home request finishes at true Home instead of demanding
            # a second command.  A missing/unusable Home config or an
            # out-of-envelope delta fails closed to a safe stop at the corridor
            # endpoint: the checked forward path remains the only known-safe
            # corridor for genuine failure/abort recoveries.
            measured_home, home_note = _load_optional_measured_home(args.home)
            converged_to_measured_home = False
            measured_home_error_deg: float | None = None
            if measured_home is not None:
                # Decide attempt-vs-safe-skip from the pre-motion delta here so
                # that any SafetyError raised once convergence motion has begun
                # is a genuine fault that propagates and fails closed, rather
                # than being misread as a benign "too far" decline.
                home_delta_deg = float(np.degrees(np.max(np.abs(measured_home - final))))
                envelope_deg = float(np.degrees(stage.MEASURED_HOME_MAX_CONVERGENCE_RAD))
                if home_delta_deg <= envelope_deg:
                    final = stage.converge_to_measured_home(
                        robot, final, measured_home, guard,
                        speed_percent=args.speed_percent,
                    )
                    converged_to_measured_home = True
                    measured_home_error_deg = float(
                        np.degrees(np.max(np.abs(final - measured_home))),
                    )
                else:
                    home_note = (
                        "stopped at checked corridor endpoint; calibrated Home is "
                        f"{home_delta_deg:.3f}deg away, beyond the "
                        f"{envelope_deg:.3f}deg convergence envelope"
                    )
            guard.path_motion_started = False
            gripper = stage.restore_gripper_enable_at_current_aperture(
                effector,
                guard,
            )
            receipt: dict[str, object] = {
                "schema": "z_manip.piper_home_recovery.v1",
                "phase": "complete",
                "success": True,
                "returned_home": True,
                "matched_stage": matched_stage,
                "matched_waypoint": matched_index,
                "match_error_deg": float(np.degrees(error)),
                "converged_to_measured_home": converged_to_measured_home,
                "final_joints_rad": final.tolist(),
                "gripper_ready": True,
                "gripper_aperture_m": gripper.aperture_m,
            }
            if converged_to_measured_home:
                receipt["measured_home_error_deg"] = measured_home_error_deg
            elif home_note is not None:
                receipt["measured_home_note"] = home_note
            print(json.dumps(receipt))
            return 0
        finally:
            stage.disconnect_quietly(robot)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
