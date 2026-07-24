#!/usr/bin/env python3
"""Fail-closed, explicitly authorized PiPER staged grasp execution.

The module deliberately keeps artifact validation and execution policy separate
from transport creation.  Importing it, requesting ``--help``, or running the
default dry-run path cannot import ``pyAgxArm`` or open SocketCAN.

Each real stage requires both ``--execute`` and an artifact-bound confirmation
token printed by the corresponding dry run.  ``approach_close`` additionally
requires a successful pregrasp receipt and a new plan whose capture, file
timestamp, and start joints all post-date/match that receipt.  ``lift`` requires
the successful, non-empty-grasp receipt from ``approach_close`` and uses a
different confirmation token.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


STAGES = ("pregrasp", "approach_close", "lift")
JOINT_LIMITS_RAD = np.radians(np.asarray((
    (-150.0, 150.0),
    (0.0, 180.0),
    (-170.0, 0.0),
    (-100.0, 100.0),
    (-70.0, 70.0),
    (-120.0, 120.0),
), dtype=float))
PATH_KEYS = ("transit", "approach", "lift")
RAW_PATH_KEYS = {
    "transit": "transit_raw",
    "approach": "approach_raw",
    "lift": "lift_raw",
}
SOURCE_AGE_HARD_CAP_S = {
    # A complete mobile handoff can spend tens of seconds in close-range
    # perception, IK and path generation after the base has stopped.  The
    # artifact is hash-bound and its own mtime is checked below, so the old
    # 30-second cap could reject a newly generated, internally consistent
    # plan before execution even began.  Keep a bounded two-minute window;
    # genuinely old artifacts still fail closed.
    "pregrasp": 120.0,
    # Approach reuses the complete artifact captured and planned at Home.
    # The D435 close-range blind zone makes a post-pregrasp depth refresh
    # unavailable, so receipt chaining—not a new frame—authorizes this stage.
    "approach_close": 180.0,
    "lift": 180.0,
}
RECEIPT_MAX_AGE_S = 180.0
OPEN_APERTURE_M = 0.07
# The close/lift grasp predicates require a measured aperture at or below
# OPEN_APERTURE_M - 0.005 (= 0.065 m).  A rigid object wider than that holds
# the fingers above the gate forever, so planning must reject such widths up
# front instead of guaranteeing an execution-time SafetyError; 3 mm of
# measurement/compliance headroom below the gate.
MAX_PLANNED_GRASP_WIDTH_M = 0.062
GRIPPER_CLOSE_STEPS = 8
GRIPPER_CLOSE_INTERVAL_S = 0.20
# Historic full ramp: OPEN_APERTURE_M -> ~0.026 m over 8 steps == ~5.5 mm of
# aperture per 0.20 s step.  Distance-proportional step counts preserve that
# exact maximum closing rate while a shorter remaining gap (after pre-close
# staging) no longer pays the full 1.6 s ramp.
GRIPPER_CLOSE_STEP_DISTANCE_M = 0.0055
# Direct-grasp close staging: after the arm settles at the grasp pose, close
# quickly to required_width + this clearance (3 mm per finger above the object
# face, so contact is impossible), then run the slow evidence ramp only over
# the remaining few millimetres.  The staging sweep happens at the settled
# grasp pose over a sub-range of the exact aperture interval today's full slow
# ramp already sweeps at that same pose, so it adds zero new swept volume; the
# contact-speed profile below the clearance is unchanged.  Mid-descent staging
# was deliberately rejected: the planner validates the descent corridor with
# OPEN fingers plus a closed-finger audit of the final state only
# (ros2/z_manip_task/z_manip_task/planning.py approach_path_valid), so a
# narrower in-flight aperture is not corridor-covered evidence.
GRIPPER_PRECLOSE_CLEARANCE_M = 0.006
# Minimum measurable gripper motor load that proves an object is between the
# fingers when the aperture shows no stall gap above the close target.  Live
# false positive (run mobile-handoff-grasp-1784868281926528476, 18.1mm
# charger): an EMPTY gripper converged to 14.6mm against a 14.1mm command --
# 0.5mm of servo scatter, no object -- while reporting -0.18N, so the former
# 0.08N rescue read friction/current noise as a hold and the arm lifted air.
# The force floor must sit comfortably above that empty-close noise reading.
# Contact evidence therefore rests primarily on the stall gap: any real object
# (even the soft 2026-07-23 bottle: aperture 57.8mm vs commanded 56.7mm,
# -0.114N) blocks the jaws measurably ABOVE the commanded target, because the
# command deliberately sits squeeze_margin (4mm) below the object width.
EMPTY_CLOSE_RESCUE_FORCE_N = 0.30
GRIPPER_POST_CLOSE_SETTLE_S = 0.50
DEFAULT_SPEED_PERCENT = 5
MAX_SPEED_PERCENT = 50
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_PATH_WAYPOINTS = 5000
DEFAULT_START_TOLERANCE_RAD = math.radians(1.0)
DEFAULT_FEEDBACK_TOLERANCE_RAD = math.radians(1.0)
DEFAULT_MAX_SEGMENT_RAD = math.radians(8.0)
# The planner's collision sampler emits a dense polyline.  Sending every
# sample as a separate point-to-point move makes PiPER stop and restart at
# each sample.  Coalesce only points that lie on the exact same checked joint
# space edge, while retaining intermediate targets for long low-speed moves.
EXECUTION_COLLINEAR_TOLERANCE_RAD = 1e-6
EXECUTION_MAX_COALESCED_SEGMENT_RAD = math.radians(180.0)
# Lift artifacts already contain the planner's 50 Hz rest-to-rest quintic
# trajectory.  Stream that trajectory instead of turning its raw IK vertices
# into independent point-to-point moves (which forces the controller to stop
# at every vertex).  Fifteen percent is the speed at which the recorded timing
# was first validated; lower requested speeds stretch, never compress, it.
LIFT_STREAM_REFERENCE_SPEED_PERCENT = 15
# Direct grasp streams the approach->contact descent from the SAME retimed
# quintic the artifact already carries, exactly like the lift, instead of
# stepping raw IK vertices point-to-point.  Streaming (a) commands the final
# grasp target unconditionally so the tool is driven onto the exact planned
# contact pose -- the stepped path skips a final waypoint already within the
# feedback tolerance, which stops the tool up to that tolerance short of the
# grasp and closes the gripper there ("slightly off / barely missing") -- and
# (b) removes the stop-and-go halt at the pregrasp standoff (now a via).  The
# reference speed is deliberately gentle so the tool creeps into contact.
APPROACH_STREAM_REFERENCE_SPEED_PERCENT = 15
# Transit streams from the artifact's retimed quintic for the same reason: the
# stepped point-to-point loop decelerated PiPER to a full stop at EVERY raw
# transit vertex and then dwelt >=3 feedback samples before the next target,
# which read as visible stop-and-go segmentation in demo recordings.  All three
# stream chains share one reference speed so the wall-time stretch factor is
# identical across transit/approach/lift and no junction shows a rate step.
TRANSIT_STREAM_REFERENCE_SPEED_PERCENT = 15
# 0.30 s at the 50 Hz stream keeps the arm within a bounded 15-sample drift
# on a non-realtime host; the previous 0.15 s tripped on ordinary scheduler
# jitter and its failure path unloaded a holding arm.
LIFT_STREAM_MAX_SCHEDULE_LAG_S = 0.30
# A late host RE-ANCHORS the remaining schedule instead of aborting: aborting
# mid-lift strands a held object, while shifting the schedule only stretches
# the planned velocity profile (never compresses it) and still commands one
# fresh-fault-checked target per tick.  The cumulative shift is bounded; a
# host stalled beyond this budget still aborts.  Live NUC evidence 2026-07-23:
# four consecutive mobile handoffs grasped and then died at this gate ~1.5s
# into the lift stream.
LIFT_STREAM_MAX_TOTAL_RESYNC_S = 3.0
RESAMPLED_PATH_TOLERANCE_RAD = 1e-5
# The physical J3 encoder currently reports +0.005515 rad at the URDF's
# nominal zero stop.  Keep the reconciliation gate narrowly above that
# measured offset while remaining below 0.35 degrees.
MAX_START_RECONCILIATION_RAD = 0.006
PROJECTION_MATCH_TOLERANCE_RAD = math.radians(0.01)
JOINT_LIMIT_TOLERANCE_RAD = 1e-5
# Bounded low-speed convergence from a checked-corridor endpoint to the
# calibrated software Home.  These mirror the standalone Home action exactly as
# the operator's Home wrapper invokes it (piper_home_recovery.py with
# --max-recovery-deg 20 --max-step-deg 5): a checked path leaves the arm at its
# recorded planning-start pose, and the final short hop to calibrated Home must
# never sweep a large unplanned joint-space chord.  Deltas beyond the envelope
# fail closed so the caller can stop safely at the corridor endpoint instead;
# the step cap keeps every segment inside PiPER's controller deadband budget.
MEASURED_HOME_MAX_CONVERGENCE_RAD = math.radians(20.0)
MEASURED_HOME_MAX_STEP_RAD = math.radians(5.0)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SafetyError(RuntimeError):
    """Raised whenever a fail-closed execution gate is not satisfied."""


@dataclass(frozen=True)
class PlanningArtifact:
    """Immutable in-memory planning artifact selected for one invocation."""

    report: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    report_path: Path
    npz_path: Path
    report_sha256: str
    npz_sha256: str
    artifact_id: str
    report_mtime_ns: int
    source_stamp_ns: int
    start_joints_rad: np.ndarray
    measured_joints_rad: np.ndarray
    start_projection_rad: np.ndarray
    requires_start_reconciliation: bool
    required_width_m: float


@dataclass(frozen=True)
class PriorReceipt:
    """Verified result from the immediately preceding stage."""

    document: Mapping[str, Any]
    path: Path
    sha256: str
    final_joints_rad: np.ndarray
    finished_unix_ns: int


@dataclass(frozen=True)
class GripperFeedback:
    """Normalized gripper status used by pure verification functions."""

    aperture_m: float
    force_n: float
    timestamp: float
    mode: str
    healthy: bool
    enabled: bool
    homed: bool


@dataclass
class CommandGuard:
    """Track whether an actuator command may have reached the bus."""

    started: bool = False
    path_motion_started: bool = False
    holding_load: bool = False

    def mark_before_command(self) -> None:
        """Mark before calling an SDK command, including calls that may raise."""
        self.started = True

    def mark_before_path_motion(self) -> None:
        """Mark only an actual planned joint-path command."""
        self.started = True
        self.path_motion_started = True


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    digest = str(value).strip().lower()
    if SHA256_RE.fullmatch(digest) is None:
        raise SafetyError(f"{field} must be an exact 64-hex SHA-256 digest")
    return digest


def _finite_vector(value: object, field: str, length: int = 6) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise SafetyError(f"{field} must be a finite {length}-vector") from error
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise SafetyError(f"{field} must be a finite {length}-vector")
    return vector


def _integer_ns(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise SafetyError(f"{field} must be a positive integer nanosecond stamp")
    try:
        stamp = int(value)
    except (TypeError, ValueError) as error:
        raise SafetyError(
            f"{field} must be a positive integer nanosecond stamp",
        ) from error
    if stamp <= 0:
        raise SafetyError(f"{field} must be a positive integer nanosecond stamp")
    return stamp


def age_seconds(stamp_ns: int, now_ns: int) -> float:
    """Return age while rejecting timestamps materially in the future."""
    age = (int(now_ns) - int(stamp_ns)) / 1e9
    if age < -1.0:
        raise SafetyError(f"timestamp is {-age:.3f}s in the future")
    return max(0.0, age)


def close_target_m(
    required_width_m: float,
    *,
    squeeze_margin_m: float = 0.004,
) -> float:
    """Compute a conservative close command from the planned object width."""
    width = float(required_width_m)
    margin = float(squeeze_margin_m)
    if not math.isfinite(width) or not 0.003 <= width <= MAX_PLANNED_GRASP_WIDTH_M:
        raise SafetyError(
            "required_width_m is outside the 0.003-0.062m range "
            "(the execution grasp gate needs a measured aperture <= 0.065m)"
        )
    if not math.isfinite(margin) or not 0.001 <= margin <= 0.010:
        raise SafetyError("squeeze margin is outside the 0.001-0.010m range")
    return max(0.002, width - margin)


def minimum_nonempty_aperture_m(required_width_m: float) -> float:
    """Minimum measured opening accepted as evidence against an empty grasp."""
    width = float(required_width_m)
    if not math.isfinite(width) or width <= 0.0:
        raise SafetyError("required_width_m must be finite and positive")
    return max(0.004, min(width * 0.5, width - 0.002))


def verify_nonempty_grasp(
    feedback: GripperFeedback,
    required_width_m: float,
    *,
    minimum_force_n: float = 0.20,
    commanded_close_target_m: float | None = None,
) -> None:
    """Require healthy width-mode feedback, non-zero aperture, and force."""
    if not all(math.isfinite(value) for value in (
        feedback.aperture_m,
        feedback.force_n,
        feedback.timestamp,
    )):
        raise SafetyError("gripper aperture/force/timestamp must be finite")
    if not feedback.healthy or not feedback.enabled:
        raise SafetyError("gripper feedback reports an unhealthy/unready driver")
    if feedback.mode != "width":
        raise SafetyError("gripper is not reporting width mode")
    threshold = minimum_nonempty_aperture_m(required_width_m)
    if feedback.aperture_m < threshold:
        raise SafetyError(
            f"empty-grasp aperture: {feedback.aperture_m:.4f}m < {threshold:.4f}m",
        )
    if minimum_force_n > 0.0 and feedback.force_n < minimum_force_n:
        raise SafetyError(
            f"insufficient grasp force: {feedback.force_n:.3f}N "
            f"< {minimum_force_n:.3f}N",
        )
    if (
        commanded_close_target_m is not None
        and feedback.aperture_m <= float(commanded_close_target_m) + 0.001
        and abs(feedback.force_n) < EMPTY_CLOSE_RESCUE_FORCE_N
    ):
        # No stall gap above the commanded target AND no motor load clearly
        # above empty-close noise: nothing is between the fingers.  The stall
        # gap is the primary contact evidence -- the close command sits
        # squeeze_margin (4mm) below the planned object width, so even the
        # softest real hold observed live blocked the jaws 1.1mm above the
        # command (2026-07-23 bottle: 57.8mm vs 56.7mm, -0.114N), while an
        # EMPTY gripper converges within servo scatter (~0.5mm) of the
        # command (run mobile-handoff-grasp-1784868281926528476: 14.6mm vs
        # 14.1mm, -0.18N, then lifted air).  The force branch only rescues
        # readings far above that observed noise; ambiguity reports empty so
        # the pipeline retries legibly instead of lifting air.
        raise SafetyError(
            "gripper reached the empty close target without a contact gap "
            "or grasp force",
        )


def verify_gripper_ready(feedback: GripperFeedback) -> None:
    """Reject gripper faults, disabled state, or non-width mode.

    PiPER firmware S-V1.8-8 leaves ``homing_status`` false even after a
    successful, feedback-verified width move.  Position, force, enable and
    fault feedback remain authoritative for this firmware.
    """
    if not feedback.healthy or not feedback.enabled:
        raise SafetyError("gripper feedback reports an unhealthy/unready driver")
    if feedback.mode != "width":
        raise SafetyError("gripper is not reporting width mode")


def verify_gripper_safe_to_enable(feedback: GripperFeedback) -> None:
    """Allow a fault-free rebooted gripper to receive its enable command.

    PiPER reports ``driver_enable_status=false`` after a controller reboot.
    ``move_gripper_m`` is itself the width-mode enable command (CAN status
    code 0x01), so requiring the enable bit before sending that command makes
    a cold start impossible.  Fault bits and the reported control mode remain
    hard gates; the command's resulting feedback must still pass
    :func:`verify_gripper_ready` before any planned arm motion begins.
    """
    if not feedback.healthy:
        raise SafetyError("gripper feedback reports a hardware fault")
    if feedback.mode != "width":
        raise SafetyError("gripper is not reporting width mode")


def restore_gripper_enable_at_current_aperture(
    effector: Any,
    guard: CommandGuard,
    *,
    force_n: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> GripperFeedback:
    """Make a rebooted gripper ready without changing its measured opening."""
    baseline = normalize_gripper_feedback(effector.get_gripper_status())
    verify_gripper_safe_to_enable(baseline)
    if baseline.enabled:
        verify_gripper_ready(baseline)
        return baseline
    if not 0.0 <= baseline.aperture_m <= 0.100:
        raise SafetyError("gripper aperture is outside the recoverable 0-0.100m range")
    guard.mark_before_command()
    effector.move_gripper_m(value=baseline.aperture_m, force=force_n)
    recovered = wait_for_gripper(
        effector,
        lambda sample: abs(sample.aperture_m - baseline.aperture_m) <= 0.002,
        after_timestamp=baseline.timestamp,
        timeout_s=4.0,
        static_accept_after_s=0.5,
        monotonic=monotonic,
        sleep=sleep,
    )
    verify_gripper_ready(recovered)
    return recovered


def confirmation_token(
    stage: str,
    artifact_id: str,
    prior_receipt_sha256: str | None = None,
) -> str:
    """Return an exact stage/artifact/receipt-bound operator token."""
    if stage not in STAGES:
        raise SafetyError(f"unsupported stage: {stage}")
    artifact = _require_sha256(artifact_id, "artifact_id")
    suffix = ""
    if prior_receipt_sha256 is not None:
        prior = _require_sha256(prior_receipt_sha256, "prior receipt SHA-256")
        suffix = f"-{prior[:10]}"
    return f"PIPER-{stage.replace('_', '-').upper()}-{artifact[:16]}{suffix}"


def require_execution_authorization(
    *,
    execute: bool,
    supplied_token: str | None,
    expected_token: str,
) -> None:
    """Fail unless real execution has both switches and an exact token."""
    if not execute:
        return
    if supplied_token != expected_token:
        raise SafetyError(
            "real execution requires --execute and the exact dry-run "
            f"confirmation token: {expected_token}",
        )


def _load_npz(npz_bytes: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise SafetyError(f"cannot safely load planned_grasp.npz: {error}") from error


def _validate_path(
    arrays: Mapping[str, np.ndarray],
    key: str,
    *,
    max_segment_rad: float,
) -> np.ndarray:
    if key not in arrays or f"{key}_times_s" not in arrays:
        raise SafetyError(f"planned_grasp.npz lacks {key} path/timestamps")
    path = np.asarray(arrays[key], dtype=float)
    times = np.asarray(arrays[f"{key}_times_s"], dtype=float)
    if path.ndim != 2 or path.shape[1:] != (6,) or len(path) < 2:
        raise SafetyError(f"{key} must contain at least two 6-axis waypoints")
    if len(path) > MAX_PATH_WAYPOINTS:
        raise SafetyError(f"{key} exceeds the {MAX_PATH_WAYPOINTS}-waypoint safety cap")
    if times.shape != (len(path),):
        raise SafetyError(f"{key}_times_s must align one-to-one with waypoints")
    if not np.isfinite(path).all() or not np.isfinite(times).all():
        raise SafetyError(f"{key} contains non-finite values")
    if times[0] < -1e-9 or np.any(np.diff(times) <= 0.0):
        raise SafetyError(f"{key}_times_s must be nonnegative and strictly increasing")
    low = JOINT_LIMITS_RAD[:, 0] - JOINT_LIMIT_TOLERANCE_RAD
    high = JOINT_LIMITS_RAD[:, 1] + JOINT_LIMIT_TOLERANCE_RAD
    if np.any(path < low) or np.any(path > high):
        raise SafetyError(f"{key} contains a waypoint outside PiPER joint limits")
    largest = float(np.max(np.abs(np.diff(path, axis=0))))
    if largest > max_segment_rad:
        raise SafetyError(
            f"{key} segment jump {largest:.5f}rad exceeds "
            f"{max_segment_rad:.5f}rad",
        )
    return path


def _validate_raw_path(
    arrays: Mapping[str, np.ndarray],
    key: str,
    *,
    max_segment_rad: float,
) -> np.ndarray:
    """Validate and command-densify a collision-checked raw polyline.

    The planner collision-checks every complete raw joint-space edge.  Those
    edges can legitimately be wider than the executor's per-command limit,
    so preserve the exact checked polyline while inserting points on each
    already-validated edge.  This never shortcuts or changes its geometry.
    """
    raw_key = RAW_PATH_KEYS[key]
    if raw_key not in arrays:
        raise SafetyError(f"planned_grasp.npz lacks collision-checked {raw_key}")
    path = np.asarray(arrays[raw_key], dtype=float)
    if path.ndim != 2 or path.shape[1:] != (6,) or len(path) < 2:
        raise SafetyError(f"{raw_key} must contain at least two 6-axis waypoints")
    if len(path) > MAX_PATH_WAYPOINTS:
        raise SafetyError(f"{raw_key} exceeds the {MAX_PATH_WAYPOINTS}-waypoint safety cap")
    if not np.isfinite(path).all():
        raise SafetyError(f"{raw_key} contains non-finite values")
    low = JOINT_LIMITS_RAD[:, 0] - JOINT_LIMIT_TOLERANCE_RAD
    high = JOINT_LIMITS_RAD[:, 1] + JOINT_LIMIT_TOLERANCE_RAD
    if np.any(path < low) or np.any(path > high):
        raise SafetyError(f"{raw_key} contains a waypoint outside PiPER joint limits")
    densified = [path[0].copy()]
    for first, second in zip(path, path[1:]):
        largest = float(np.max(np.abs(second - first)))
        steps = max(1, int(math.ceil(largest / max_segment_rad)))
        for step in range(1, steps + 1):
            alpha = step / steps
            densified.append(first + alpha * (second - first))
    result = np.asarray(densified, dtype=float)
    if len(result) > MAX_PATH_WAYPOINTS:
        raise SafetyError(
            f"densified {raw_key} exceeds the {MAX_PATH_WAYPOINTS}-waypoint safety cap",
        )
    return result


def load_planning_artifact(
    report_path: Path,
    npz_path: Path,
    *,
    expected_npz_sha256: str | None,
    stage: str,
    now_ns: int,
    max_source_age_s: float,
    require_fresh_source: bool = True,
    start_tolerance_rad: float = DEFAULT_START_TOLERANCE_RAD,
    max_segment_rad: float = DEFAULT_MAX_SEGMENT_RAD,
) -> PlanningArtifact:
    """Load, hash, and completely validate a staged execution artifact."""
    if stage not in STAGES:
        raise SafetyError(f"unsupported stage: {stage}")
    if not isinstance(require_fresh_source, bool):
        raise SafetyError("require_fresh_source must be boolean")
    if require_fresh_source and not 0.0 < max_source_age_s <= SOURCE_AGE_HARD_CAP_S[stage]:
        raise SafetyError(
            f"max source age for {stage} must be within "
            f"(0, {SOURCE_AGE_HARD_CAP_S[stage]:.0f}]s",
        )
    if not 0.0 < start_tolerance_rad <= math.radians(2.0):
        raise SafetyError("start tolerance must be within (0, 2deg]")
    if not 0.0 < max_segment_rad <= math.radians(10.0):
        raise SafetyError("maximum segment size must be within (0, 10deg]")

    try:
        if report_path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise SafetyError("planning_report.json exceeds the artifact size cap")
        if npz_path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise SafetyError("planned_grasp.npz exceeds the artifact size cap")
        report_bytes = report_path.read_bytes()
        npz_bytes = npz_path.read_bytes()
        report_mtime_ns = report_path.stat().st_mtime_ns
    except OSError as error:
        raise SafetyError(f"cannot read planning artifact: {error}") from error
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafetyError(f"invalid planning_report.json: {error}") from error
    if not isinstance(report, dict):
        raise SafetyError("planning_report.json must contain one JSON object")

    report_digest = sha256_bytes(report_bytes)
    npz_digest = sha256_bytes(npz_bytes)
    embedded_digest = report.get("planned_grasp_sha256")
    trusted_digest = expected_npz_sha256 or embedded_digest
    if trusted_digest is None:
        raise SafetyError(
            "no NPZ trust anchor: add planned_grasp_sha256 to the report or pass "
            "--expected-npz-sha256",
        )
    trusted_digest = _require_sha256(trusted_digest, "expected NPZ SHA-256")
    if embedded_digest is not None:
        embedded_digest = _require_sha256(
            embedded_digest,
            "planning_report.planned_grasp_sha256",
        )
        if embedded_digest != trusted_digest:
            raise SafetyError("embedded and explicitly expected NPZ hashes disagree")
    if npz_digest != trusted_digest:
        raise SafetyError("planned_grasp.npz SHA-256 mismatch")

    if report.get("plan_valid") is not True:
        raise SafetyError("planning_report.plan_valid is not exactly true")
    if report.get("read_only") is not True or report.get("planning_only") is not True:
        raise SafetyError("planning artifact lacks read_only/planning_only provenance")
    if report.get("motion_commands_published") != 0:
        raise SafetyError("planning artifact reports non-zero motion commands")
    reconciliation = report.get("execution_start_requires_limit_reconciliation")
    if not isinstance(reconciliation, bool):
        raise SafetyError("execution_start_requires_limit_reconciliation must be boolean")

    source_stamp_ns = _integer_ns(report.get("source_stamp_ns"), "source_stamp_ns")
    source_age = age_seconds(source_stamp_ns, now_ns)
    if require_fresh_source and source_age > max_source_age_s:
        raise SafetyError(
            f"perception source is stale: {source_age:.3f}s > {max_source_age_s:.3f}s",
        )
    file_age = age_seconds(report_mtime_ns, now_ns)
    if require_fresh_source and file_age > max_source_age_s:
        raise SafetyError(
            f"planning report file is stale: {file_age:.3f}s > {max_source_age_s:.3f}s",
        )

    arrays = _load_npz(npz_bytes)
    start = _finite_vector(
        report.get("planning_start_joints_rad"),
        "planning_start_joints_rad",
    )
    report_current = _finite_vector(
        report.get("current_joints_rad"),
        "current_joints_rad",
    )
    archive_current = _finite_vector(arrays.get("current_joints"), "npz.current_joints")
    report_measured = _finite_vector(
        report.get("measured_joints_rad"),
        "measured_joints_rad",
    )
    archive_measured = _finite_vector(
        arrays.get("measured_joints"),
        "npz.measured_joints",
    )
    projection = _finite_vector(
        report.get("start_limit_projection_rad"),
        "start_limit_projection_rad",
    )
    if float(np.max(np.abs(report_current - report_measured))) > PROJECTION_MATCH_TOLERANCE_RAD:
        raise SafetyError("report current joints do not match measured joints")
    if float(np.max(np.abs(report_measured - archive_measured))) > PROJECTION_MATCH_TOLERANCE_RAD:
        raise SafetyError("report and NPZ measured joints disagree")
    if float(np.max(np.abs(start - archive_current))) > start_tolerance_rad:
        raise SafetyError("planning start does not match NPZ current joints")
    computed_projection = start - report_measured
    if float(np.max(np.abs(projection - computed_projection))) > PROJECTION_MATCH_TOLERANCE_RAD:
        raise SafetyError("reported start-limit projection is inconsistent")
    projection_size = float(np.max(np.abs(projection)))
    if reconciliation:
        if stage != "pregrasp":
            raise SafetyError("start-limit reconciliation is allowed only for pregrasp")
        if not 0.0 < projection_size <= MAX_START_RECONCILIATION_RAD:
            raise SafetyError(
                "start-limit reconciliation exceeds the "
                f"{MAX_START_RECONCILIATION_RAD:.3f}rad safety cap",
            )
        low = JOINT_LIMITS_RAD[:, 0]
        high = JOINT_LIMITS_RAD[:, 1]
        if (
            np.any(start < low - JOINT_LIMIT_TOLERANCE_RAD)
            or np.any(start > high + JOINT_LIMIT_TOLERANCE_RAD)
        ):
            raise SafetyError("projected planning start is still outside joint limits")
        if (
            np.any(report_measured < low - MAX_START_RECONCILIATION_RAD)
            or np.any(report_measured > high + MAX_START_RECONCILIATION_RAD)
        ):
            raise SafetyError("measured start is too far outside joint limits")
    else:
        if projection_size > PROJECTION_MATCH_TOLERANCE_RAD:
            raise SafetyError("non-reconciled plan contains a material start projection")
        if float(np.max(np.abs(start - report_measured))) > start_tolerance_rad:
            raise SafetyError("planning start does not match measured joints")

    paths = {
        key: _validate_path(arrays, key, max_segment_rad=max_segment_rad)
        for key in PATH_KEYS
    }
    if report.get("raw_paths_collision_validated") is not True:
        raise SafetyError("planning report lacks raw_paths_collision_validated=true")
    raw_paths = {
        key: _validate_raw_path(arrays, key, max_segment_rad=max_segment_rad)
        for key in PATH_KEYS
    }
    for key in PATH_KEYS:
        expected_count = report.get(f"{key}_raw_waypoints")
        recorded_raw = np.asarray(arrays[RAW_PATH_KEYS[key]], dtype=float)
        if isinstance(expected_count, bool) or expected_count != len(recorded_raw):
            raise SafetyError(f"{key}_raw_waypoints does not match the NPZ raw polyline")
        if float(np.max(np.abs(raw_paths[key][0] - paths[key][0]))) > 1e-5:
            raise SafetyError(f"{key} raw/resampled start mismatch")
        if float(np.max(np.abs(raw_paths[key][-1] - paths[key][-1]))) > 1e-5:
            raise SafetyError(f"{key} raw/resampled end mismatch")
    if float(np.max(np.abs(paths["transit"][0] - start))) > start_tolerance_rad:
        raise SafetyError("transit does not start at planning_start_joints_rad")
    if float(np.max(np.abs(paths["approach"][0] - paths["transit"][-1]))) > 1e-5:
        raise SafetyError("transit-to-approach path discontinuity")
    if float(np.max(np.abs(paths["lift"][0] - paths["approach"][-1]))) > 1e-5:
        raise SafetyError("approach-to-lift path discontinuity")
    if float(np.max(np.abs(raw_paths["approach"][0] - raw_paths["transit"][-1]))) > 1e-5:
        raise SafetyError("raw transit-to-approach path discontinuity")
    if float(np.max(np.abs(raw_paths["lift"][0] - raw_paths["approach"][-1]))) > 1e-5:
        raise SafetyError("raw approach-to-lift path discontinuity")

    try:
        required_width = float(report.get("required_width_m", math.nan))
    except (TypeError, ValueError) as error:
        raise SafetyError("required_width_m must be numeric") from error
    close_target_m(required_width)
    artifact_id = sha256_bytes(report_bytes + b"\0" + npz_bytes)
    return PlanningArtifact(
        report=report,
        arrays=arrays,
        report_path=report_path,
        npz_path=npz_path,
        report_sha256=report_digest,
        npz_sha256=npz_digest,
        artifact_id=artifact_id,
        report_mtime_ns=report_mtime_ns,
        source_stamp_ns=source_stamp_ns,
        start_joints_rad=start,
        measured_joints_rad=report_measured,
        start_projection_rad=projection,
        requires_start_reconciliation=reconciliation,
        required_width_m=required_width,
    )


def load_prior_receipt(
    path: Path,
    *,
    expected_stage: str,
    now_ns: int,
    max_age_s: float = RECEIPT_MAX_AGE_S,
) -> PriorReceipt:
    """Load and validate an immutable prior-stage receipt."""
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafetyError(f"cannot load prior-stage receipt: {error}") from error
    if not isinstance(document, dict):
        raise SafetyError("prior-stage receipt must be one JSON object")
    if document.get("schema") != "z_manip.piper_stage_receipt.v1":
        raise SafetyError("unsupported prior-stage receipt schema")
    if document.get("stage") != expected_stage or document.get("success") is not True:
        raise SafetyError(f"a successful {expected_stage} receipt is required")
    _require_sha256(document.get("artifact_id"), "receipt.artifact_id")
    finished = _integer_ns(document.get("finished_unix_ns"), "receipt.finished_unix_ns")
    receipt_age = age_seconds(finished, now_ns)
    if receipt_age > max_age_s:
        raise SafetyError(f"prior-stage receipt is stale: {receipt_age:.3f}s > {max_age_s:.3f}s")
    final_joints = _finite_vector(document.get("final_joints_rad"), "receipt.final_joints_rad")
    return PriorReceipt(
        document=document,
        path=path,
        sha256=sha256_bytes(payload),
        final_joints_rad=final_joints,
        finished_unix_ns=finished,
    )


def validate_stage_context(
    artifact: PlanningArtifact,
    stage: str,
    prior: PriorReceipt | None,
    *,
    start_tolerance_rad: float = DEFAULT_START_TOLERANCE_RAD,
) -> np.ndarray:
    """Validate receipt chaining and return the path for ``stage``."""
    if stage == "pregrasp":
        if prior is not None:
            raise SafetyError("pregrasp must not consume a prior-stage receipt")
        transit = np.asarray(artifact.arrays["transit_raw"], dtype=float)
        if artifact.requires_start_reconciliation:
            transit = np.vstack((artifact.measured_joints_rad, transit))
        return coalesce_collinear_execution_path(transit)

    if prior is None:
        raise SafetyError(f"{stage} requires a prior-stage receipt")
    if stage == "approach_close":
        if prior.document.get("stage") != "pregrasp":
            raise SafetyError("approach_close requires a pregrasp receipt")
        if prior.document.get("artifact_id") != artifact.artifact_id:
            raise SafetyError(
                "approach artifact differs from the completed pregrasp artifact",
            )
        # A D435 has no dependable depth in the close-range pregrasp blind
        # zone.  The complete transit/approach/lift sequence is therefore
        # planned once from Home.  Receipt chaining proves that the arm
        # reached this artifact's pregrasp before its already-checked
        # approach edge can execute; no impossible close-range replan is
        # required.
        planned_pregrasp = np.asarray(artifact.arrays["transit_raw"], dtype=float)[-1]
        start_error = float(np.max(np.abs(planned_pregrasp - prior.final_joints_rad)))
        if start_error > start_tolerance_rad:
            raise SafetyError("approach start does not match completed pregrasp")
        approach = np.asarray(artifact.arrays["approach_raw"], dtype=float)
        if float(np.max(np.abs(approach[0] - prior.final_joints_rad))) > start_tolerance_rad:
            raise SafetyError("approach path does not start at completed pregrasp")
        return coalesce_collinear_execution_path(approach)

    if stage == "lift":
        if prior.document.get("stage") != "approach_close":
            raise SafetyError("lift requires an approach_close receipt")
        if prior.document.get("artifact_id") != artifact.artifact_id:
            raise SafetyError("lift artifact differs from the completed approach artifact")
        gripper = prior.document.get("gripper")
        if not isinstance(gripper, dict) or gripper.get("nonempty_verified") is not True:
            raise SafetyError("lift requires prior non-empty-grasp verification")
        try:
            commanded_target = float(gripper.get("commanded_target_m", math.nan))
            feedback = GripperFeedback(
                aperture_m=float(gripper.get("aperture_m", math.nan)),
                force_n=float(gripper.get("force_n", math.nan)),
                timestamp=float(gripper.get("timestamp", math.nan)),
                mode=str(gripper.get("mode", "")),
                healthy=bool(gripper.get("healthy")),
                enabled=bool(gripper.get("enabled")),
                homed=bool(gripper.get("homed")),
            )
        except (TypeError, ValueError) as error:
            raise SafetyError("lift receipt gripper fields are malformed") from error
        expected_target = close_target_m(artifact.required_width_m)
        if not math.isfinite(commanded_target) or abs(commanded_target - expected_target) > 1e-6:
            raise SafetyError("lift receipt close target does not match required_width_m")
        verify_nonempty_grasp(
            feedback,
            artifact.required_width_m,
            # PiPER's force estimate is not a reliable contact discriminator
            # for light objects.  Aperture/contact-gap evidence is already
            # recorded by the completed approach stage and remains the
            # authoritative non-empty-grasp check here.
            minimum_force_n=0.0,
            commanded_close_target_m=commanded_target,
        )
        lift = np.asarray(artifact.arrays["lift_raw"], dtype=float)
        if float(np.max(np.abs(lift[0] - prior.final_joints_rad))) > start_tolerance_rad:
            raise SafetyError("lift path does not start at completed grasp pose")
        return coalesce_collinear_execution_path(lift)
    raise SafetyError(f"unsupported stage: {stage}")


def coalesce_collinear_execution_path(
    path: np.ndarray,
    *,
    tolerance_rad: float = EXECUTION_COLLINEAR_TOLERANCE_RAD,
    max_segment_rad: float = EXECUTION_MAX_COALESCED_SEGMENT_RAD,
) -> np.ndarray:
    """Remove stop-and-go samples without changing the validated polyline.

    An interior waypoint is removable only when it lies on the forward line
    segment between the last retained waypoint and a later waypoint.  Long
    edges are then subdivided so low-speed moves retain bounded completion
    times.  Corners are never shortcut and no new geometric edge is created.
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (6,) or len(points) < 2:
        raise SafetyError("execution path must contain at least two 6-DOF waypoints")
    if not np.all(np.isfinite(points)):
        raise SafetyError("execution path contains non-finite values")
    if tolerance_rad < 0.0 or max_segment_rad <= 0.0:
        raise ValueError("execution coalescing tolerances must be positive")

    retained = [points[0]]
    anchor = points[0]
    for index in range(1, len(points) - 1):
        middle = points[index]
        later = points[index + 1]
        direction = later - anchor
        norm_squared = float(np.dot(direction, direction))
        removable = False
        if norm_squared > 1e-18:
            fraction = float(np.dot(middle - anchor, direction) / norm_squared)
            projection = anchor + fraction * direction
            removable = (
                -1e-12 <= fraction <= 1.0 + 1e-12
                and float(np.max(np.abs(middle - projection))) <= tolerance_rad
            )
        if not removable:
            retained.append(middle)
            anchor = middle
    retained.append(points[-1])

    bounded = [np.asarray(retained[0], dtype=float)]
    for target in retained[1:]:
        start = bounded[-1]
        step = float(np.max(np.abs(target - start)))
        pieces = max(1, int(math.ceil(step / max_segment_rad)))
        for fraction in range(1, pieces + 1):
            bounded.append(start + (target - start) * (fraction / pieces))
    return np.asarray(bounded, dtype=float)


def validate_resampled_path_on_raw_polyline(
    path: np.ndarray,
    raw_path: np.ndarray,
    *,
    tolerance_rad: float = RESAMPLED_PATH_TOLERANCE_RAD,
) -> None:
    """Prove that each streamed edge stays on the checked raw polyline.

    The planning artifact contains both the collision-checked raw IK polyline
    and a dynamically retimed dense trajectory.  Hash and endpoint checks are
    not enough: a malformed dense trajectory could shortcut a raw corner.
    This ordered projection check requires every dense edge to remain on one
    raw edge and requires every raw corner to be visited in order.
    """
    dense = np.asarray(path, dtype=float)
    raw = np.asarray(raw_path, dtype=float)
    if (
        dense.ndim != 2
        or raw.ndim != 2
        or dense.shape[1:] != (6,)
        or raw.shape[1:] != (6,)
        or len(dense) < 2
        or len(raw) < 2
    ):
        raise SafetyError("resampled/raw execution paths must be finite (N, 6) arrays")
    if not np.isfinite(dense).all() or not np.isfinite(raw).all():
        raise SafetyError("resampled/raw execution paths contain non-finite values")
    if not math.isfinite(tolerance_rad) or tolerance_rad <= 0.0:
        raise ValueError("resampled path tolerance must be finite and positive")

    # Consecutive duplicate IK poses do not define an edge.  Removing only
    # exact-near duplicates preserves every real collision-checked corner.
    raw_vertices = [raw[0]]
    for point in raw[1:]:
        if float(np.max(np.abs(point - raw_vertices[-1]))) > tolerance_rad:
            raw_vertices.append(point)
    raw = np.asarray(raw_vertices, dtype=float)
    if len(raw) < 2:
        raise SafetyError("raw execution path contains no motion")
    if (
        float(np.max(np.abs(dense[0] - raw[0]))) > tolerance_rad
        or float(np.max(np.abs(dense[-1] - raw[-1]))) > tolerance_rad
    ):
        raise SafetyError("resampled execution path endpoints differ from raw polyline")

    def progress_on_edge(point: np.ndarray, edge: int) -> float | None:
        first = raw[edge]
        delta = raw[edge + 1] - first
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-18:
            return None
        progress = float(np.dot(point - first, delta) / denominator)
        projection = first + progress * delta
        if (
            -tolerance_rad <= progress <= 1.0 + tolerance_rad
            and float(np.max(np.abs(point - projection))) <= tolerance_rad
        ):
            return min(1.0, max(0.0, progress))
        return None

    edge = 0
    progress = progress_on_edge(dense[0], edge)
    if progress is None:
        raise SafetyError("resampled execution path does not start on raw polyline")
    previous = dense[0]
    for point in dense[1:]:
        next_progress = progress_on_edge(point, edge)
        if next_progress is not None and next_progress + tolerance_rad >= progress:
            progress = next_progress
            previous = point
            continue
        # Advancing to the next checked edge is legal only after the dense
        # path explicitly visited the shared raw vertex.  This rejects corner
        # shortcuts while allowing a vertex to belong to either adjacent edge.
        if (
            edge + 1 >= len(raw) - 1
            or float(np.max(np.abs(previous - raw[edge + 1]))) > tolerance_rad
        ):
            raise SafetyError("resampled execution path shortcuts a raw polyline corner")
        edge += 1
        progress = progress_on_edge(point, edge)
        if progress is None:
            raise SafetyError("resampled execution path leaves the raw polyline")
        previous = point
    if edge != len(raw) - 2 or progress < 1.0 - tolerance_rad:
        raise SafetyError("resampled execution path omits a raw polyline edge")


def timed_stage_path(
    artifact: PlanningArtifact,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one validated dynamic trajectory tied to checked raw geometry."""
    if key not in PATH_KEYS:
        raise SafetyError(f"unsupported timed execution path: {key}")
    path = np.asarray(artifact.arrays[key], dtype=float)
    times = np.asarray(artifact.arrays[f"{key}_times_s"], dtype=float)
    raw = np.asarray(artifact.arrays[RAW_PATH_KEYS[key]], dtype=float)
    validate_resampled_path_on_raw_polyline(path, raw)
    return path, times


def _feedback_timestamp(message: object, field: str) -> float:
    try:
        value = float(getattr(message, "timestamp"))
    except (AttributeError, TypeError, ValueError) as error:
        raise SafetyError(f"{field} feedback lacks a numeric SDK timestamp") from error
    if not math.isfinite(value):
        raise SafetyError(f"{field} feedback timestamp is not finite")
    return value


def read_joint_feedback(robot: Any) -> tuple[np.ndarray, float]:
    """Normalize one six-axis SDK feedback message."""
    message = robot.get_joint_angles()
    if message is None:
        raise SafetyError("joint feedback is unavailable")
    joints = _finite_vector(getattr(message, "msg", None), "joint feedback")
    return joints, _feedback_timestamp(message, "joint")


def check_arm_status(robot: Any, *, require_idle: bool = True) -> tuple[int, float]:
    """Reject arm faults, errors, and unexpected motion."""
    message = robot.get_arm_status()
    if message is None:
        raise SafetyError("arm status feedback is unavailable")
    status = getattr(message, "msg", None)
    arm_status = int(getattr(status, "arm_status", 0xFF))
    motion_status = int(getattr(status, "motion_status", 0xFF))
    error_code = int(getattr(status, "err_code", 0xFFFF))
    if arm_status != 0 or error_code != 0:
        raise SafetyError(
            f"arm fault: arm_status={arm_status}, err_code=0x{error_code:04X}",
        )
    if require_idle and motion_status != 0:
        raise SafetyError(f"arm is not idle: motion_status={motion_status}")
    return motion_status, _feedback_timestamp(message, "arm status")


def wait_for_initial_arm_feedback(
    robot: Any,
    *,
    timeout_s: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, float]:
    """Wait briefly for the SDK receive thread without masking real faults."""
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        try:
            # PiPER S-V1.8-8 can retain motion_status=1 after a completed move.
            # Fault bits remain authoritative; stationarity is established
            # below from distinct joint samples instead of this stale flag.
            check_arm_status(robot, require_idle=False)
            return read_joint_feedback(robot)
        except SafetyError as error:
            if "feedback is unavailable" not in str(error):
                raise
        sleep(0.02)
    raise SafetyError("timed out waiting for initial arm and joint feedback")


def normalize_gripper_feedback(message: object) -> GripperFeedback:
    """Normalize one pyAgxArm gripper feedback object."""
    if message is None:
        raise SafetyError("gripper feedback is unavailable")
    payload = getattr(message, "msg", None)
    try:
        aperture = float(getattr(payload, "value"))
        force = float(getattr(payload, "force"))
        mode = str(getattr(payload, "mode"))
    except (AttributeError, TypeError, ValueError) as error:
        raise SafetyError("gripper feedback fields are unavailable") from error
    if not math.isfinite(aperture) or not math.isfinite(force):
        raise SafetyError("gripper aperture/force is not finite")
    foc = getattr(payload, "foc_status", None)
    fault_names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "sensor_status",
        "driver_error_status",
    )
    healthy = foc is not None and not any(bool(getattr(foc, name, True)) for name in fault_names)
    return GripperFeedback(
        aperture_m=aperture,
        force_n=force,
        timestamp=_feedback_timestamp(message, "gripper"),
        mode=mode,
        healthy=healthy,
        enabled=bool(getattr(foc, "driver_enable_status", False)),
        homed=bool(getattr(foc, "homing_status", False)),
    )


def wait_for_gripper(
    effector: Any,
    predicate: Callable[[GripperFeedback], bool],
    *,
    after_timestamp: float,
    timeout_s: float,
    stable_samples: int = 3,
    static_accept_after_s: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> GripperFeedback:
    """Wait for healthy feedback, tolerating S-V1.8-8's static SDK stamp."""
    started = monotonic()
    deadline = monotonic() + timeout_s
    accepted = 0
    last_timestamp = after_timestamp
    latest: GripperFeedback | None = None
    while monotonic() < deadline:
        latest = normalize_gripper_feedback(effector.get_gripper_status())
        if latest.timestamp <= last_timestamp:
            if not latest.healthy or not latest.enabled or latest.mode != "width":
                raise SafetyError(
                    "gripper reports a fault, disabled state, or wrong mode",
                )
            if (
                static_accept_after_s is not None
                and monotonic() - started >= static_accept_after_s
                and predicate(latest)
            ):
                return latest
            sleep(0.02)
            continue
        last_timestamp = latest.timestamp
        if not latest.healthy or not latest.enabled or latest.mode != "width":
            raise SafetyError(
                "gripper reports a fault, disabled state, or wrong mode",
            )
        accepted = accepted + 1 if predicate(latest) else 0
        if accepted >= stable_samples:
            return latest
        sleep(0.02)
    detail = "no usable gripper sample"
    if latest is not None:
        detail = (
            f"last aperture={latest.aperture_m:.4f}m, "
            f"force={latest.force_n:.3f}N, stamp={latest.timestamp:.6f}, "
            f"healthy={latest.healthy}, enabled={latest.enabled}, mode={latest.mode}"
        )
    raise SafetyError(
        f"timed out waiting for stable gripper feedback ({detail})",
    )


def wait_for_motion(
    robot: Any,
    target_rad: np.ndarray,
    *,
    after_timestamp: float,
    after_status_timestamp: float,
    timeout_s: float,
    tolerance_rad: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Wait for fresh joint feedback and an idle, fault-free target gate."""
    deadline = monotonic() + timeout_s
    saw_fresh_feedback = False
    saw_fresh_status = False
    stable_in_tolerance = 0
    last_joint_stamp = after_timestamp
    while monotonic() < deadline:
        _, status_stamp = check_arm_status(robot, require_idle=False)
        actual, stamp = read_joint_feedback(robot)
        saw_fresh_feedback = saw_fresh_feedback or stamp > after_timestamp
        saw_fresh_status = saw_fresh_status or status_stamp > after_status_timestamp
        if stamp > last_joint_stamp:
            last_joint_stamp = stamp
            if float(np.max(np.abs(actual - target_rad))) <= tolerance_rad:
                stable_in_tolerance += 1
            else:
                stable_in_tolerance = 0
        if saw_fresh_feedback and saw_fresh_status and stable_in_tolerance >= 3:
            return actual
        sleep(0.02)
    raise SafetyError("motion timed out or did not produce fresh in-tolerance feedback")


def wait_for_fresh_joint_feedback(
    robot: Any,
    *,
    after_timestamp: float,
    timeout_s: float,
    stable_samples: int = 3,
    stable_tolerance_rad: float = math.radians(0.10),
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, float]:
    """Require distinct, stationary samples before the first command."""
    deadline = monotonic() + timeout_s
    accepted = 0
    previous: np.ndarray | None = None
    last_stamp = after_timestamp
    while monotonic() < deadline:
        joints, stamp = read_joint_feedback(robot)
        if stamp <= last_stamp:
            sleep(0.02)
            continue
        last_stamp = stamp
        if previous is not None and float(np.max(np.abs(joints - previous))) <= stable_tolerance_rad:
            accepted += 1
        else:
            accepted = 1
        previous = joints
        if accepted >= stable_samples:
            return joints, stamp
        sleep(0.02)
    raise SafetyError("timed out waiting for fresh, stationary joint feedback")


def enable_arm(
    robot: Any,
    guard: CommandGuard,
    *,
    timeout_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Enable with a bounded acknowledgement loop."""
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        guard.mark_before_command()
        if robot.enable():
            return
        sleep(0.05)
    raise SafetyError("arm enable timed out")


def enter_can_joint_control(
    robot: Any,
    guard: CommandGuard,
    *,
    timeout_s: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Enter CAN control, enable torque, and hold before gripper motion."""
    hold_joints, hold_stamp = read_joint_feedback(robot)
    # A PiPER that has just booted or recovered from an electronic stop can
    # acknowledge motor enable while ignoring a mode packet sent immediately
    # beforehand.  Enable first, then keep requesting CAN-J mode until fresh
    # arm-status feedback confirms it.  This also avoids leaving the arm
    # disabled while waiting for a control-mode transition.
    enable_arm(
        robot,
        guard,
        timeout_s=3.0,
        monotonic=monotonic,
        sleep=sleep,
    )
    deadline = monotonic() + timeout_s
    next_mode_request = -math.inf
    last_ctrl_mode = -1
    last_arm_status = 0xFF
    last_error_code = 0xFFFF
    while monotonic() < deadline:
        now = monotonic()
        if now >= next_mode_request:
            guard.mark_before_command()
            robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
            next_mode_request = now + 0.20
        message = robot.get_arm_status()
        if message is None:
            sleep(0.02)
            continue
        status = getattr(message, "msg", None)
        arm_status = int(getattr(status, "arm_status", 0xFF))
        error_code = int(getattr(status, "err_code", 0xFFFF))
        ctrl_mode = int(getattr(status, "ctrl_mode", -1))
        last_ctrl_mode = ctrl_mode
        last_arm_status = arm_status
        last_error_code = error_code
        if arm_status != 0 or error_code != 0:
            raise SafetyError(
                f"arm fault while entering CAN control: arm_status={arm_status}, "
                f"err_code=0x{error_code:04X}",
            )
        if ctrl_mode == 1:
            guard.mark_before_command()
            robot.set_speed_percent(1)
            guard.mark_before_command()
            robot.move_j([float(value) for value in hold_joints])
            stable, _ = wait_for_fresh_joint_feedback(
                robot,
                after_timestamp=hold_stamp,
                timeout_s=1.0,
                monotonic=monotonic,
                sleep=sleep,
            )
            if float(np.max(np.abs(stable - hold_joints))) > DEFAULT_FEEDBACK_TOLERANCE_RAD:
                raise SafetyError("arm drifted while establishing pre-gripper hold")
            return
        sleep(0.02)
    raise SafetyError(
        "arm did not enter CAN control mode "
        f"(ctrl_mode={last_ctrl_mode}, arm_status={last_arm_status}, "
        f"err_code=0x{last_error_code:04X})"
    )


def verify_warm_can_control(robot: Any) -> None:
    """Fault-gate a transport that established CAN-J control earlier in-process.

    A warm stage boundary (full-grasp executor, consecutive stages on one open
    connection) leaves the arm servo-holding the previous stage's verified
    endpoint in CAN joint mode.  Re-running the cold enable/mode/hold handshake
    there only inserts dwell (speed-1 hold ``move_j`` plus fresh-feedback
    waits) between stages that read as stop-and-go.  Fault bits and the
    reported control mode remain hard gates; any deviation fails closed and
    the caller must use the full cold handshake instead.
    """
    message = robot.get_arm_status()
    if message is None:
        raise SafetyError("arm status feedback is unavailable")
    status = getattr(message, "msg", None)
    arm_status = int(getattr(status, "arm_status", 0xFF))
    error_code = int(getattr(status, "err_code", 0xFFFF))
    ctrl_mode = int(getattr(status, "ctrl_mode", -1))
    if arm_status != 0 or error_code != 0 or ctrl_mode != 1:
        raise SafetyError(
            "warm stage handoff requires fault-free CAN joint control: "
            f"arm_status={arm_status}, err_code=0x{error_code:04X}, "
            f"ctrl_mode={ctrl_mode}",
        )


def _command_joint_target(
    robot: Any,
    target: np.ndarray,
    guard: CommandGuard,
) -> None:
    """Mark and transmit one already-authorized joint target."""
    guard.mark_before_path_motion()
    robot.move_j([float(value) for value in target])


def execute_joint_path(
    robot: Any,
    path: np.ndarray,
    guard: CommandGuard,
    *,
    speed_percent: int,
    segment_timeout_s: float,
    start_tolerance_rad: float,
    feedback_tolerance_rad: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Execute one already validated path with gates around every segment."""
    if not 1 <= speed_percent <= MAX_SPEED_PERCENT:
        raise SafetyError(f"speed percent must be within 1-{MAX_SPEED_PERCENT}")
    check_arm_status(robot, require_idle=False)
    actual, _ = read_joint_feedback(robot)
    if float(np.max(np.abs(actual - path[0]))) > start_tolerance_rad:
        raise SafetyError("live joints do not match this stage's path start")

    guard.mark_before_command()
    robot.set_speed_percent(speed_percent)
    enable_arm(robot, guard, timeout_s=3.0, monotonic=monotonic, sleep=sleep)
    previous = path[0]
    for index, target in enumerate(path[1:], start=1):
        _, before_status_stamp = check_arm_status(robot, require_idle=False)
        actual, before_stamp = read_joint_feedback(robot)
        if float(np.max(np.abs(actual - previous))) > feedback_tolerance_rad:
            raise SafetyError(f"segment {index} pre-gate does not match prior waypoint")
        if float(np.max(np.abs(actual - target))) <= feedback_tolerance_rad:
            # PiPER can leave motion_status asserted for commands smaller than
            # its controller deadband.  Do not transmit a target that fresh
            # feedback already satisfies.
            previous = target
            continue
        _command_joint_target(robot, target, guard)
        # ``move_j`` delegates one joint-space edge to PiPER's internal smooth
        # trajectory generator.  A long low-speed edge legitimately takes
        # longer than the fixed watchdog used for short approach segments.
        # Scale only the wait budget; the commanded speed remains exactly the
        # bounded user setting.
        max_delta_rad = float(np.max(np.abs(target - actual)))
        conservative_rate_rad_s = math.radians(90.0) * speed_percent / 100.0
        motion_timeout_s = min(
            120.0,
            max(
                segment_timeout_s,
                3.0 + 1.5 * max_delta_rad / conservative_rate_rad_s,
            ),
        )
        actual = wait_for_motion(
            robot,
            target,
            after_timestamp=before_stamp,
            after_status_timestamp=before_status_stamp,
            timeout_s=motion_timeout_s,
            tolerance_rad=feedback_tolerance_rad,
            monotonic=monotonic,
            sleep=sleep,
        )
        previous = target
    return actual


def execute_timed_joint_path(
    robot: Any,
    path: np.ndarray,
    times_s: np.ndarray,
    guard: CommandGuard,
    *,
    speed_percent: int,
    segment_timeout_s: float,
    start_tolerance_rad: float,
    feedback_tolerance_rad: float,
    reference_speed_percent: int = LIFT_STREAM_REFERENCE_SPEED_PERCENT,
    max_schedule_lag_s: float = LIFT_STREAM_MAX_SCHEDULE_LAG_S,
    max_total_resync_s: float = LIFT_STREAM_MAX_TOTAL_RESYNC_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Stream one validated timed path without stop-and-go waypoint gates.

    Targets retain the planner's bounded quintic velocity/acceleration profile.
    Requested speeds below the validated reference stretch the schedule; higher
    speeds never compress it.  A delayed host re-anchors the remaining
    schedule (bounded by ``max_total_resync_s``) rather than bursting stale
    targets onto CAN or abandoning a held object mid-lift, and the final
    target still requires fresh feedback.
    """
    positions = np.asarray(path, dtype=float)
    times = np.asarray(times_s, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (6,)
        or len(positions) < 2
        or times.shape != (len(positions),)
        or not np.isfinite(positions).all()
        or not np.isfinite(times).all()
        or abs(float(times[0])) > 1e-9
        or np.any(np.diff(times) <= 0.0)
    ):
        raise SafetyError("timed execution path must have finite increasing samples")
    if not 1 <= speed_percent <= MAX_SPEED_PERCENT:
        raise SafetyError(f"speed percent must be within 1-{MAX_SPEED_PERCENT}")
    if not 1 <= reference_speed_percent <= MAX_SPEED_PERCENT:
        raise ValueError("reference speed percent is outside the executor range")
    if not math.isfinite(max_schedule_lag_s) or max_schedule_lag_s <= 0.0:
        raise ValueError("maximum schedule lag must be finite and positive")
    if not math.isfinite(max_total_resync_s) or max_total_resync_s < 0.0:
        raise ValueError("maximum total resync must be finite and non-negative")

    check_arm_status(robot, require_idle=False)
    actual, _ = read_joint_feedback(robot)
    if float(np.max(np.abs(actual - positions[0]))) > start_tolerance_rad:
        raise SafetyError("live joints do not match this timed path start")

    guard.mark_before_command()
    robot.set_speed_percent(speed_percent)
    enable_arm(robot, guard, timeout_s=3.0, monotonic=monotonic, sleep=sleep)
    time_scale = max(1.0, reference_speed_percent / float(speed_percent))
    started = monotonic()
    before_final_stamp = -math.inf
    before_final_status_stamp = -math.inf
    actual_before_final = actual
    resync_total_s = 0.0
    resync_count = 0
    for index, target in enumerate(positions[1:], start=1):
        due = started + float(times[index]) * time_scale
        remaining = due - monotonic()
        if remaining > 0.0:
            sleep(remaining)
        lag = monotonic() - due
        if lag > max_schedule_lag_s:
            # Re-anchor: shift the whole remaining schedule by the observed
            # lag so this target is commanded now and later targets keep
            # their planned relative spacing.  No stale burst is possible --
            # each loop iteration still commands exactly one target after a
            # fresh fault check -- and the profile is only stretched in wall
            # time.  Aborting here mid-lift would strand a held object; only
            # a host stalled beyond the cumulative budget aborts.
            resync_total_s += lag
            resync_count += 1
            if resync_total_s > max_total_resync_s:
                raise SafetyError(
                    f"timed path schedule lag {lag:.3f}s exceeds "
                    f"{max_schedule_lag_s:.3f}s and the cumulative resync "
                    f"budget {max_total_resync_s:.1f}s is exhausted "
                    f"({resync_total_s:.3f}s over {resync_count} resyncs)",
                )
            started += lag
            print(
                f"lift stream resync {resync_count}: lag {lag:.3f}s, "
                f"cumulative {resync_total_s:.3f}s of "
                f"{max_total_resync_s:.1f}s",
                flush=True,
            )
        # Fault feedback remains authoritative during streaming.  The
        # per-waypoint idle/in-tolerance gate is intentionally absent: it was
        # the source of a full stop at every lift IK vertex.
        _, status_stamp = check_arm_status(robot, require_idle=False)
        if index == len(positions) - 1:
            actual_before_final, before_final_stamp = read_joint_feedback(robot)
            before_final_status_stamp = status_stamp
        _command_joint_target(robot, target, guard)

    max_delta_rad = float(np.max(np.abs(positions[-1] - actual_before_final)))
    conservative_rate_rad_s = math.radians(90.0) * speed_percent / 100.0
    completion_timeout_s = min(
        120.0,
        max(
            segment_timeout_s,
            3.0 + 1.5 * max_delta_rad / conservative_rate_rad_s,
        ),
    )
    return wait_for_motion(
        robot,
        positions[-1],
        after_timestamp=before_final_stamp,
        after_status_timestamp=before_final_status_stamp,
        timeout_s=completion_timeout_s,
        tolerance_rad=feedback_tolerance_rad,
        monotonic=monotonic,
        sleep=sleep,
    )


def load_software_home(path: Path) -> np.ndarray:
    """Load and validate a captured PiPER software Home pose.

    Mirrors the standalone Home action's loader so the calibrated Home reached
    by a checked-corridor recovery is the exact same evidence-bound pose the
    dedicated Home action would drive to.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "z_manip.piper_software_home.v1":
        raise SafetyError("invalid PiPER software Home schema")
    if document.get("capture_zero_can_tx_verified") is not True:
        raise SafetyError("software Home was not captured with zero CAN TX evidence")
    radians = _finite_vector(document.get("joint_radians"), "software Home joints")
    degrees = _finite_vector(document.get("joint_degrees"), "software Home degrees")
    if float(np.max(np.abs(np.degrees(radians) - degrees))) > 1e-3:
        raise SafetyError("software Home degree/radian values disagree")
    low = JOINT_LIMITS_RAD[:, 0]
    high = JOINT_LIMITS_RAD[:, 1]
    if np.any(radians < low) or np.any(radians > high):
        raise SafetyError("software Home lies outside PiPER joint limits")
    return radians


def converge_to_measured_home(
    robot: Any,
    current: np.ndarray,
    home: np.ndarray,
    guard: CommandGuard,
    *,
    speed_percent: int,
    segment_timeout_s: float = 12.0,
    max_convergence_rad: float = MEASURED_HOME_MAX_CONVERGENCE_RAD,
    max_step_rad: float = MEASURED_HOME_MAX_STEP_RAD,
) -> np.ndarray:
    """Bounded low-speed linear convergence from a checked endpoint to Home.

    This is the exact recovery motion the standalone Home action uses: it
    never sweeps a large unplanned chord (a delta beyond the envelope fails
    closed so the caller can stop safely at the checked-corridor endpoint) and
    it steps the interpolation so every segment stays inside PiPER's controller
    deadband.  Reusing this from the reverse-home recovery lets a single Home
    request finish at calibrated Home instead of stranding the arm at its
    recorded planning-start pose.
    """
    current = _finite_vector(current, "convergence start")
    home = _finite_vector(home, "measured Home")
    delta = float(np.max(np.abs(home - current)))
    if delta > max_convergence_rad:
        raise SafetyError(
            f"measured Home delta {math.degrees(delta):.3f}deg exceeds "
            f"{math.degrees(max_convergence_rad):.3f}deg convergence envelope",
        )
    steps = max(1, int(math.ceil(delta / max_step_rad)))
    path = np.linspace(current, home, steps + 1)
    return execute_joint_path(
        robot,
        path,
        guard,
        speed_percent=speed_percent,
        segment_timeout_s=segment_timeout_s,
        start_tolerance_rad=DEFAULT_START_TOLERANCE_RAD,
        feedback_tolerance_rad=DEFAULT_FEEDBACK_TOLERANCE_RAD,
    )


def slow_close_step_count(
    start_aperture_m: float,
    target_aperture_m: float,
    *,
    step_distance_m: float = GRIPPER_CLOSE_STEP_DISTANCE_M,
    max_steps: int = GRIPPER_CLOSE_STEPS,
) -> int:
    """Distance-proportional slow-close step count at the historic ramp rate.

    The legacy ramp always spent ``GRIPPER_CLOSE_STEPS`` intervals regardless
    of the remaining gap, so a pre-staged close still dwelt the full 1.6 s.
    Scaling the count by distance keeps the identical worst-case aperture rate
    (one ``step_distance_m`` per interval) and merely stops charging time for
    distance that no longer exists.
    """
    start = float(start_aperture_m)
    target = float(target_aperture_m)
    if not math.isfinite(start) or not math.isfinite(target) or start < target:
        raise SafetyError("slow close step count requires a decreasing aperture")
    if not math.isfinite(step_distance_m) or step_distance_m <= 0.0 or max_steps < 2:
        raise ValueError("slow close step sizing must be positive with >=2 steps")
    return int(min(max_steps, max(2, math.ceil((start - target) / step_distance_m))))


def command_slow_gripper_close(
    effector: Any,
    guard: CommandGuard,
    *,
    start_aperture_m: float,
    target_aperture_m: float,
    force_n: float,
    steps: int = GRIPPER_CLOSE_STEPS,
    interval_s: float = GRIPPER_CLOSE_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Ramp the position target so initial object contact is not impulsive."""
    start = float(start_aperture_m)
    target = float(target_aperture_m)
    if (
        not math.isfinite(start)
        or not math.isfinite(target)
        or target < 0.0
        or start < target
    ):
        raise SafetyError("slow gripper close requires a valid decreasing aperture")
    if steps < 2 or not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("slow gripper close needs at least two positive-time steps")
    for index in range(1, steps + 1):
        aperture = start + (target - start) * (index / steps)
        guard.mark_before_command()
        effector.move_gripper_m(value=aperture, force=force_n)
        if index < steps:
            sleep(interval_s)


def emergency_stop_after_failure(robot: Any, guard: CommandGuard) -> None:
    """Send one best-effort electronic e-stop after a command-path failure."""
    # A gripper/status failure while the arm is holding must not release
    # torque: PiPER's electronic e-stop unloads this tabletop installation.
    # Reserve it for failures after a real planned joint move was transmitted.
    if guard.holding_load:
        # A loaded arm must fail toward holding: the electronic e-stop unloads
        # this installation and drops the object mid-air.  Stop commanding and
        # leave the last servo target torqued for operator recovery.
        print(
            "CRITICAL: command fault while holding load; "
            "arm left torqued at its last target",
            file=sys.stderr,
        )
        return
    if not guard.path_motion_started:
        return
    try:
        robot.electronic_emergency_stop()
    except Exception as error:  # pragma: no cover - terminal fallback only
        print(f"CRITICAL: electronic emergency stop failed: {error}", file=sys.stderr)


def execute_stage(
    robot: Any,
    effector: Any,
    artifact: PlanningArtifact,
    stage: str,
    path: np.ndarray,
    *,
    speed_percent: int = DEFAULT_SPEED_PERCENT,
    segment_timeout_s: float = 8.0,
    start_tolerance_rad: float = DEFAULT_START_TOLERANCE_RAD,
    feedback_tolerance_rad: float = DEFAULT_FEEDBACK_TOLERANCE_RAD,
    gripper_force_n: float = 1.0,
    direct_approach: bool = True,
    warm_start: bool = False,
    timing_out: dict[str, object] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, GripperFeedback | None]:
    """Execute exactly one stage; any post-command failure triggers e-stop.

    ``direct_approach`` (default) streams the transit and ``approach_close``
    descents from the artifact's retimed quintics as continuous moves (the
    approach ends on the exact grasp pose); ``direct_approach=False`` restores
    the legacy stepped, stop-and-go behavior everywhere.

    ``warm_start`` (full-grasp executor only, consecutive stages on one open
    connection) replaces the cold CAN enable/mode/hold handshake with a strict
    fault/mode gate (:func:`verify_warm_can_control`) so stage boundaries stop
    dwelling.  Standalone per-stage CLI invocations are always cold.

    ``timing_out``, when provided, receives boundary-dwell instrumentation:
    ``pre_motion_s`` (stage entry to first path command), ``motion_s`` and
    ``post_motion_s`` (path end to stage return, e.g. the gripper close
    phase), plus ``warm_start``.  Purely additive evidence for receipts.
    """
    guard = CommandGuard()
    gripper_result: GripperFeedback | None = None
    entry_time = monotonic()

    def _record(key: str, value: object) -> None:
        if timing_out is not None:
            timing_out[key] = (
                round(float(value), 4) if isinstance(value, float) else value
            )

    _record("warm_start", bool(warm_start))
    try:
        _, initial_joint_stamp = wait_for_initial_arm_feedback(
            robot,
            monotonic=monotonic,
            sleep=sleep,
        )
        live_start, _ = wait_for_fresh_joint_feedback(
            robot,
            after_timestamp=initial_joint_stamp,
            timeout_s=1.0,
            monotonic=monotonic,
            sleep=sleep,
        )
        if float(np.max(np.abs(live_start - path[0]))) > start_tolerance_rad:
            raise SafetyError("live joints do not match the authorized stage start")

        if stage == "pregrasp":
            enter_can_joint_control(
                robot,
                guard,
                monotonic=monotonic,
                sleep=sleep,
            )
            baseline = normalize_gripper_feedback(effector.get_gripper_status())
            verify_gripper_safe_to_enable(baseline)
            guard.mark_before_command()
            effector.move_gripper_m(value=OPEN_APERTURE_M, force=gripper_force_n)
            gripper_result = wait_for_gripper(
                effector,
                lambda sample: abs(sample.aperture_m - OPEN_APERTURE_M) <= 0.002,
                after_timestamp=baseline.timestamp,
                timeout_s=4.0,
                static_accept_after_s=0.5,
                monotonic=monotonic,
                sleep=sleep,
            )
            transit_path = path
            motion_started = monotonic()
            _record("pre_motion_s", motion_started - entry_time)
            if artifact.requires_start_reconciliation:
                execute_joint_path(
                    robot,
                    path[:2],
                    guard,
                    speed_percent=min(speed_percent, 2),
                    segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=start_tolerance_rad,
                    feedback_tolerance_rad=feedback_tolerance_rad,
                    monotonic=monotonic,
                    sleep=sleep,
                )
                transit_path = path[1:]
            raw_transit = np.asarray(artifact.arrays["transit_raw"], dtype=float)
            transit_has_motion = (
                float(np.max(np.abs(raw_transit - raw_transit[0])))
                > RESAMPLED_PATH_TOLERANCE_RAD
            )
            if (
                direct_approach
                and not artifact.requires_start_reconciliation
                and transit_has_motion
            ):
                # Stream the retimed transit exactly like the lift: one
                # continuous chain instead of a point-to-point move + >=3
                # settled-feedback samples at EVERY raw vertex, which was the
                # dominant stop-and-go segmentation in demo recordings.  The
                # dense trajectory is proven to ride the collision-checked raw
                # transit polyline inside ``timed_stage_path``.
                timed_transit, transit_times_s = timed_stage_path(artifact, "transit")
                if (
                    float(np.max(np.abs(timed_transit[0] - transit_path[0]))) > 1e-5
                    or float(np.max(np.abs(timed_transit[-1] - transit_path[-1]))) > 1e-5
                ):
                    raise SafetyError(
                        "timed transit path differs from authorized raw transit",
                    )
                final_joints = execute_timed_joint_path(
                    robot,
                    timed_transit,
                    transit_times_s,
                    guard,
                    speed_percent=speed_percent,
                    segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=start_tolerance_rad,
                    feedback_tolerance_rad=feedback_tolerance_rad,
                    reference_speed_percent=TRANSIT_STREAM_REFERENCE_SPEED_PERCENT,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            else:
                final_joints = execute_joint_path(
                    robot,
                    transit_path,
                    guard,
                    speed_percent=speed_percent,
                    segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=start_tolerance_rad,
                    feedback_tolerance_rad=feedback_tolerance_rad,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            motion_ended = monotonic()
            _record("motion_s", motion_ended - motion_started)
            _record("post_motion_s", monotonic() - motion_ended)
            return final_joints, gripper_result

        if stage == "approach_close":
            if warm_start:
                verify_warm_can_control(robot)
            else:
                enter_can_joint_control(
                    robot,
                    guard,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            initial_gripper = normalize_gripper_feedback(effector.get_gripper_status())
            verify_gripper_safe_to_enable(initial_gripper)
            # A new pyAgxArm connection may expose a valid cached gripper
            # sample without advancing its feedback timestamp.  Reassert the
            # already-authorized open target while holding current joints so
            # the stage gets fresh driver feedback before any approach move.
            guard.mark_before_command()
            effector.move_gripper_m(value=OPEN_APERTURE_M, force=gripper_force_n)
            open_feedback = wait_for_gripper(
                effector,
                lambda sample: (
                    sample.aperture_m >= artifact.required_width_m + 0.004
                ),
                after_timestamp=initial_gripper.timestamp,
                timeout_s=4.0,
                static_accept_after_s=0.5,
                monotonic=monotonic,
                sleep=sleep,
            )
            verify_gripper_ready(open_feedback)
            motion_started = monotonic()
            _record("pre_motion_s", motion_started - entry_time)
            if direct_approach:
                # Direct grasp: stream the retimed approach as one continuous
                # descent into contact.  Unlike the stepped path, the streamer
                # commands the FINAL grasp target unconditionally, so the tool
                # is driven onto the exact planned contact pose instead of
                # settling up to one feedback tolerance short of it (the stepped
                # path skips a final waypoint already within tolerance and the
                # gripper would then close at that short pose).  The pregrasp
                # standoff is a via, not a halt, and the quintic decelerates
                # into contact.  The dense trajectory is proven to ride the same
                # collision-checked raw polyline inside ``timed_stage_path``.
                timed_approach, approach_times_s = timed_stage_path(artifact, "approach")
                if (
                    float(np.max(np.abs(timed_approach[0] - path[0]))) > 1e-5
                    or float(np.max(np.abs(timed_approach[-1] - path[-1]))) > 1e-5
                ):
                    raise SafetyError(
                        "timed approach path differs from authorized raw approach",
                    )
                final_joints = execute_timed_joint_path(
                    robot,
                    timed_approach,
                    approach_times_s,
                    guard,
                    speed_percent=speed_percent,
                    segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=start_tolerance_rad,
                    feedback_tolerance_rad=feedback_tolerance_rad,
                    reference_speed_percent=APPROACH_STREAM_REFERENCE_SPEED_PERCENT,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            else:
                final_joints = execute_joint_path(
                    robot,
                    path,
                    guard,
                    speed_percent=speed_percent,
                    segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=start_tolerance_rad,
                    feedback_tolerance_rad=feedback_tolerance_rad,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            # The checked joint path completed.  A subsequent gripper-status
            # failure must keep this pose torqued instead of electronic-stop
            # unloading the arm; path failures themselves still e-stop.
            guard.path_motion_started = False
            motion_ended = monotonic()
            _record("motion_s", motion_ended - motion_started)
            baseline = normalize_gripper_feedback(effector.get_gripper_status())
            verify_gripper_ready(baseline)
            target = close_target_m(artifact.required_width_m)
            preclose_target = artifact.required_width_m + GRIPPER_PRECLOSE_CLEARANCE_M
            if (
                direct_approach
                and baseline.aperture_m > preclose_target + 0.004
                and preclose_target < OPEN_APERTURE_M - 0.005
            ):
                # Close staging at the settled grasp pose: sweep the
                # contact-impossible part of the aperture range (open down to
                # width + 6 mm clearance) at native gripper speed.  This is a
                # sub-range of the exact sweep the full slow ramp already
                # performed at this same pose, so no new swept volume; the
                # slow evidence ramp below the clearance is unchanged.
                guard.mark_before_command()
                effector.move_gripper_m(value=preclose_target, force=gripper_force_n)
                baseline = wait_for_gripper(
                    effector,
                    lambda sample: sample.aperture_m <= preclose_target + 0.002,
                    after_timestamp=baseline.timestamp,
                    timeout_s=3.0,
                    static_accept_after_s=0.5,
                    monotonic=monotonic,
                    sleep=sleep,
                )
                verify_gripper_ready(baseline)
            close_steps = (
                slow_close_step_count(baseline.aperture_m, target)
                if direct_approach
                else GRIPPER_CLOSE_STEPS
            )
            command_slow_gripper_close(
                effector,
                guard,
                start_aperture_m=baseline.aperture_m,
                target_aperture_m=target,
                force_n=gripper_force_n,
                steps=close_steps,
                sleep=sleep,
            )
            gripper_result = wait_for_gripper(
                effector,
                lambda sample: (
                    sample.aperture_m
                    >= minimum_nonempty_aperture_m(artifact.required_width_m)
                    and sample.aperture_m <= OPEN_APERTURE_M - 0.005
                ),
                after_timestamp=baseline.timestamp,
                timeout_s=4.0,
                static_accept_after_s=0.5,
                monotonic=monotonic,
                sleep=sleep,
            )
            verify_nonempty_grasp(
                gripper_result,
                artifact.required_width_m,
                minimum_force_n=0.0,
                commanded_close_target_m=target,
            )
            # Hold the force-limited final aperture briefly before lift so a
            # newly contacted object can settle between both fingers.
            sleep(GRIPPER_POST_CLOSE_SETTLE_S)
            _record("post_motion_s", monotonic() - motion_ended)
            return final_joints, gripper_result

        if stage == "lift":
            if warm_start:
                verify_warm_can_control(robot)
            else:
                enter_can_joint_control(
                    robot,
                    guard,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            baseline = normalize_gripper_feedback(effector.get_gripper_status())
            target = close_target_m(artifact.required_width_m)
            guard.mark_before_command()
            effector.move_gripper_m(value=target, force=gripper_force_n)
            gripper_result = wait_for_gripper(
                effector,
                lambda sample: (
                    sample.aperture_m
                    >= minimum_nonempty_aperture_m(artifact.required_width_m)
                    and sample.aperture_m <= OPEN_APERTURE_M - 0.005
                ),
                after_timestamp=baseline.timestamp,
                timeout_s=4.0,
                static_accept_after_s=0.5,
                monotonic=monotonic,
                sleep=sleep,
            )
            verify_nonempty_grasp(
                gripper_result,
                artifact.required_width_m,
                minimum_force_n=0.0,
                commanded_close_target_m=close_target_m(artifact.required_width_m),
            )
            guard.holding_load = True
            timed_lift, lift_times_s = timed_stage_path(artifact, "lift")
            if (
                float(np.max(np.abs(timed_lift[0] - path[0]))) > 1e-5
                or float(np.max(np.abs(timed_lift[-1] - path[-1]))) > 1e-5
            ):
                raise SafetyError("timed lift path differs from authorized raw lift")
            motion_started = monotonic()
            _record("pre_motion_s", motion_started - entry_time)
            final_joints = execute_timed_joint_path(
                robot,
                timed_lift,
                lift_times_s,
                guard,
                speed_percent=speed_percent,
                segment_timeout_s=segment_timeout_s,
                start_tolerance_rad=start_tolerance_rad,
                feedback_tolerance_rad=feedback_tolerance_rad,
                monotonic=monotonic,
                sleep=sleep,
            )
            guard.path_motion_started = False
            motion_ended = monotonic()
            _record("motion_s", motion_ended - motion_started)
            before_final_gripper = normalize_gripper_feedback(
                effector.get_gripper_status(),
            ).timestamp
            guard.mark_before_command()
            effector.move_gripper_m(value=target, force=gripper_force_n)
            gripper_result = wait_for_gripper(
                effector,
                lambda sample: (
                    sample.aperture_m
                    >= minimum_nonempty_aperture_m(artifact.required_width_m)
                    and sample.aperture_m <= OPEN_APERTURE_M - 0.005
                ),
                after_timestamp=before_final_gripper,
                timeout_s=2.0,
                static_accept_after_s=0.5,
                monotonic=monotonic,
                sleep=sleep,
            )
            verify_nonempty_grasp(
                gripper_result,
                artifact.required_width_m,
                minimum_force_n=0.0,
                commanded_close_target_m=close_target_m(artifact.required_width_m),
            )
            _record("post_motion_s", monotonic() - motion_ended)
            return final_joints, gripper_result
        raise SafetyError(f"unsupported stage: {stage}")
    except BaseException:
        emergency_stop_after_failure(robot, guard)
        raise


def build_receipt(
    *,
    artifact: PlanningArtifact,
    stage: str,
    prior: PriorReceipt | None,
    started_unix_ns: int,
    finished_unix_ns: int,
    final_joints_rad: Sequence[float],
    gripper: GripperFeedback | None,
    approach_execution: str = "streamed",
    stage_timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stage receipt only after all feedback gates have passed.

    ``approach_execution`` attests which grasp-descent mode ran: ``"streamed"``
    (direct grasp, the default) or ``"stepped"`` (the legacy staged fallback).
    It is only meaningful for, and only recorded on, the ``approach_close``
    stage; the field is additive and never renames an existing receipt key.

    ``stage_timing`` is the additive boundary-dwell instrumentation captured by
    :func:`execute_stage` (``pre_motion_s``/``motion_s``/``post_motion_s``/
    ``warm_start``) so smoothness improvements are measurable from receipts.
    """
    result: dict[str, Any] = {
        "schema": "z_manip.piper_stage_receipt.v1",
        "stage": stage,
        "success": True,
        "artifact_id": artifact.artifact_id,
        "planning_report_sha256": artifact.report_sha256,
        "planned_grasp_sha256": artifact.npz_sha256,
        "prior_receipt_sha256": None if prior is None else prior.sha256,
        "started_unix_ns": int(started_unix_ns),
        "finished_unix_ns": int(finished_unix_ns),
        "final_joints_rad": _finite_vector(final_joints_rad, "final joints").tolist(),
        "motion_feedback_verified": True,
        "start_limit_reconciliation_performed": bool(
            artifact.requires_start_reconciliation and stage == "pregrasp"
        ),
    }
    if stage == "approach_close":
        result["approach_execution"] = str(approach_execution)
    if stage_timing is not None:
        result["stage_timing"] = dict(stage_timing)
    if gripper is not None:
        result["gripper"] = {
            "aperture_m": gripper.aperture_m,
            "force_n": gripper.force_n,
            "timestamp": gripper.timestamp,
            "mode": gripper.mode,
            "healthy": gripper.healthy,
            "enabled": gripper.enabled,
            "homed": gripper.homed,
            "nonempty_verified": stage in ("approach_close", "lift"),
            "commanded_target_m": (
                close_target_m(artifact.required_width_m)
                if stage in ("approach_close", "lift")
                else None
            ),
        }
    return result


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically persist a successful execution receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SafetyError(f"refusing to overwrite an existing stage receipt: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def connect_real_arm(channel: str, firmware: str) -> tuple[Any, Any]:
    """Import and connect pyAgxArm only after explicit authorization."""
    from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config

    firmware_options = {
        "default": PiperFW.DEFAULT,
        "v183": PiperFW.V183,
        "v188": PiperFW.V188,
        "v189": PiperFW.V189,
    }
    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=firmware_options[firmware],
        interface="socketcan",
        channel=channel,
        receive_own_messages=False,
        local_loopback=False,
    )
    robot = AgxArmFactory.create_arm(config)
    try:
        robot.connect()
        effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        return robot, effector
    except BaseException:
        disconnect_quietly(robot)
        raise


def disconnect_quietly(robot: Any | None) -> None:
    """Release SDK resources without masking the primary result."""
    if robot is None:
        return
    try:
        robot.disconnect()
    except Exception:
        pass


def _parse_current_joints(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    return _finite_vector(parts, "--current-joints-rad")


def build_parser() -> argparse.ArgumentParser:
    """Build the fail-closed command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate or explicitly execute one PiPER grasp stage",
    )
    parser.add_argument("--planning-report", type=Path, required=True)
    parser.add_argument("--planned-grasp", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--expected-npz-sha256")
    parser.add_argument("--max-source-age-s", type=float, default=15.0)
    parser.add_argument(
        "--current-joints-rad",
        help="optional dry-run live snapshot as six CSV radians",
    )
    parser.add_argument("--pregrasp-receipt", type=Path)
    parser.add_argument("--approach-receipt", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--speed-percent", type=int, default=DEFAULT_SPEED_PERCENT)
    parser.add_argument("--segment-timeout-s", type=float, default=8.0)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", choices=("default", "v183", "v188", "v189"), default="v188")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", help="exact artifact-bound token printed by dry run")
    parser.add_argument(
        "--staged-approach-stop",
        action="store_true",
        help=(
            "legacy fallback: stop at the pregrasp standoff and step the "
            "approach_close descent instead of streaming it directly into "
            "contact (the accurate default)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate in dry-run mode or execute exactly one authorized stage."""
    args = build_parser().parse_args(argv)
    now_ns = time.time_ns()
    try:
        artifact = load_planning_artifact(
            args.planning_report,
            args.planned_grasp,
            expected_npz_sha256=args.expected_npz_sha256,
            stage=args.stage,
            now_ns=now_ns,
            max_source_age_s=args.max_source_age_s,
        )
        prior: PriorReceipt | None = None
        if args.stage == "approach_close":
            if args.pregrasp_receipt is None:
                raise SafetyError("approach_close requires --pregrasp-receipt")
            prior = load_prior_receipt(
                args.pregrasp_receipt,
                expected_stage="pregrasp",
                now_ns=now_ns,
            )
        elif args.stage == "lift":
            if args.approach_receipt is None:
                raise SafetyError("lift requires --approach-receipt")
            prior = load_prior_receipt(
                args.approach_receipt,
                expected_stage="approach_close",
                now_ns=now_ns,
            )
        path = validate_stage_context(artifact, args.stage, prior)
        token = confirmation_token(
            args.stage,
            artifact.artifact_id,
            None if prior is None else prior.sha256,
        )
        require_execution_authorization(
            execute=args.execute,
            supplied_token=args.confirm,
            expected_token=token,
        )
        dry_current = _parse_current_joints(args.current_joints_rad)
        if dry_current is not None:
            mismatch = float(np.max(np.abs(dry_current - path[0])))
            if mismatch > DEFAULT_START_TOLERANCE_RAD:
                raise SafetyError(
                    f"provided current joints miss stage start by {mismatch:.5f}rad",
                )

        summary = {
            "schema": "z_manip.piper_stage_dry_run.v1",
            "stage": args.stage,
            "dry_run": not args.execute,
            "artifact_id": artifact.artifact_id,
            "planning_report_sha256": artifact.report_sha256,
            "planned_grasp_sha256": artifact.npz_sha256,
            "source_age_s": age_seconds(artifact.source_stamp_ns, now_ns),
            "path_waypoints": len(path),
            "required_width_m": artifact.required_width_m,
            "close_target_m": close_target_m(artifact.required_width_m),
            "prior_receipt_sha256": None if prior is None else prior.sha256,
            "confirmation_token": token,
            "live_start_checked": dry_current is not None,
            "commands_sent": 0,
        }
        if not args.execute:
            print(json.dumps(summary, indent=2))
            return 0

        if not Path(f"/sys/class/net/{args.channel}").exists():
            raise SafetyError(f"SocketCAN interface does not exist: {args.channel}")
        if not 1 <= args.speed_percent <= MAX_SPEED_PERCENT:
            raise SafetyError(f"--speed-percent must be within 1-{MAX_SPEED_PERCENT}")
        if not 0.5 <= args.segment_timeout_s <= 15.0:
            raise SafetyError("--segment-timeout-s must be within 0.5-15.0")
        if not 0.2 <= args.gripper_force_n <= 2.0:
            raise SafetyError("--gripper-force-n must be within 0.2-2.0N")

        robot: Any | None = None
        try:
            robot, effector = connect_real_arm(args.channel, args.firmware)
            started_ns = time.time_ns()
            direct_approach = not args.staged_approach_stop
            stage_timing: dict[str, Any] = {}
            final_joints, gripper = execute_stage(
                robot,
                effector,
                artifact,
                args.stage,
                path,
                speed_percent=args.speed_percent,
                segment_timeout_s=args.segment_timeout_s,
                gripper_force_n=args.gripper_force_n,
                direct_approach=direct_approach,
                timing_out=stage_timing,
            )
            finished_ns = time.time_ns()
            receipt = build_receipt(
                artifact=artifact,
                stage=args.stage,
                prior=prior,
                started_unix_ns=started_ns,
                finished_unix_ns=finished_ns,
                final_joints_rad=final_joints,
                gripper=gripper,
                approach_execution="streamed" if direct_approach else "stepped",
                stage_timing=stage_timing,
            )
            output = args.receipt_output or args.planning_report.with_name(
                f"execution_{args.stage}_receipt.json",
            )
            atomic_write_json(output, receipt)
            actual_summary = {
                **summary,
                "dry_run": False,
                "move_j_commands_sent": len(path) - 1,
                "gripper_commands_sent": 0 if args.stage == "lift" else 1,
                "receipt": str(output),
            }
            actual_summary.pop("commands_sent", None)
            print(json.dumps(actual_summary, indent=2))
            return 0
        finally:
            disconnect_quietly(robot)
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130
    except (SafetyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"ERROR: runtime execution failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
