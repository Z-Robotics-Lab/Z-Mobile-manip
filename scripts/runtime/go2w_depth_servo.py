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
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import statistics
import threading
import time
from typing import Any
import uuid

import numpy as np

from z_manip.control.reactive_servo import (
    ArmViewIntent,
    BaseMotionIntent,
    PostureIntent,
    ReactivePhase,
    ReactiveServoConfig,
    ReactiveServoDecision,
    ReactiveTargetController,
    TargetGeometry,
)
from z_manip.control.go2w_posture import PosturePhase
from z_manip.control.servo_phase import (
    HANDOFF_TERMINAL_PHASES,
    LOSS_STAIR_PHASES,
    ServoPhase,
)
from z_manip.control.visual_servo import VisualServoConfig, VisualServoController
from z_manip.control.whole_body_runtime import (
    WholeBodyRuntimeCommand,
    WholeBodyRuntimeController,
)


STATUS_SCHEMA = "z_manip.depth_servo_status.v1"
POSTURE_SETTLE_TICKS = 5
POSTURE_ANGLE_RATE_SETTLED_RPS = math.radians(0.75)
ARM_RATE_SETTLED_RPS = math.radians(0.75)
ARM_TARGET_ERROR_SETTLED_RAD = math.radians(1.0)
ARM_STATUS_TIMEOUT_S = 0.50
ARM_INTENT_TTL_NS = 250_000_000
ARM_INTENT_SCHEMA = "z_manip.piper_reactive_view_intent.v1"
ARM_STATUS_SCHEMA = "z_manip.piper_reactive_view_status.v1"
# Read-only close-range IK feasibility channel.
#
# THERE IS NO PRODUCER IN THIS REPOSITORY.  ``grep -rn "reactive/ik_probe"``
# returns exactly two hits and both are in this file (this constant and the
# subscription).  ``DepthServoCore._ik_feasible`` is therefore permanently
# ``None``, so ``ReactivePhase.HANDOFF_READY``, ``DepthServoCore._done`` via
# that branch, the ``reached`` presentation remap and the ``fallback.done``
# arm of the whole-body settle branch are all unreachable in the deployed
# configuration.
#
# THIS IS DELIBERATE AND IS KEPT, not deleted, because:
#
#   1. The gate is fail-closed by construction.  ``_ik_probe_state`` returns
#      ``None`` on absence, staleness or a malformed payload, and both
#      HANDOFF_READY sites require ``ik_feasible is True``.  Deleting the gate
#      would mean declaring a manipulation handoff on distance and posture
#      alone -- the exact fail-open this hardening programme exists to remove.
#   2. The handoff is NOT dead: it terminates one level earlier, at
#      HANDOFF_PROBE.  ``needs_ik_probe`` is set whenever the corridor is
#      reached with ``ik_feasible is None``, and the supervisor's
#      ``_runtime_requests_handoff`` accepts ``needs_ik_probe is True`` and
#      ``reactive_phase in {handoff_probe, handoff_ready}``.  So the deployed
#      terminal is HANDOFF_PROBE -> the supervisor stops the servo -> the
#      close-range grasp transaction opens.  HANDOFF_READY is the *upgrade*
#      path for when a certifier exists.
#
# TO MAKE HANDOFF_READY REACHABLE, a producer must publish
# ``std_msgs/String`` JSON on ``/z_manip/reactive/ik_probe`` with
# ``{"schema": IK_PROBE_SCHEMA, "feasible": <bool>}`` at better than
# ``IK_PROBE_TIMEOUT_S`` (1.0 s), where ``feasible`` is a real pregrasp+grasp
# IK solve against the CURRENT measured base/arm state.  The natural owner is
# the resident close-range planning runner in
# ``ros2/z_manip_place`` / the mobile-handoff grasp path, which already solves
# that IK after the servo stops.  Publishing anything weaker (e.g. a reach
# heuristic) would convert this fail-closed gate into a fail-open one.
IK_PROBE_SCHEMA = "z_manip.reactive_ik_probe.v1"
IK_PROBE_TIMEOUT_S = 1.0

# Bounded life for the one-way handoff latch.  The latch is set by a SINGLE
# tick and, before this bound, had no reset edge and no expiry: once set, the
# node returned early from every subsequent ``_tick`` forever.  The contract is
# that the 5 Hz supervisor observes the handoff within one loop iteration
# (~50 ms) and stops this process.  20 s without that means nobody is watching,
# so the latch escalates to an explicit ``handoff_abandoned`` terminal.  The
# exit is ALWAYS a stop, never a resume: an un-latch that let the base drive
# again mid-grasp would be a fail-open.
HANDOFF_LATCH_TIMEOUT_S = 20.0


# Measured PC-to-NUC NTP midpoint offset is ~0.31s (see the reactive-view
# executor's MAX_FUTURE_SKEW rationale); this allowance keeps normally-fresh
# cross-host stamps from reading as stale.
CLOCK_SKEW_ALLOWANCE_S = 0.50


@dataclass(frozen=True)
class DepthServoSettings:
    mode: str = "shadow"
    desired_depth_m: float = 0.50
    # PiPER can solve the final centimetres.  The legged base only needs to
    # enter a coarse near-field corridor; demanding camera-perfect alignment
    # makes body sway repeatedly reset the handoff window.
    depth_tolerance_m: float = 0.01
    lateral_tolerance_m: float = 0.12
    # Keep the target to one side of the chassis/lidar stack at handoff.  The
    # sign is latched once per tracking session from the cheapest current
    # side; near-centre starts use ``preferred_side_sign`` deterministically.
    side_lateral_offset_m: float = 0.13
    side_lock_deadband_m: float = 0.025
    side_handoff_tolerance_m: float = 0.07
    preferred_side_sign: int = 1
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
    yaw_deadband_rad: float = math.radians(10.0)
    max_yaw_step_rps: float = 0.015
    target_timeout_s: float = 0.40
    # Receipt time is not data time: with network queuing a bundle can arrive
    # "fresh" while its capture stamp is a second old, and the servo then
    # steers on an old world (live 2026-07-23 evening: 1.2s LAN RTT tilted the
    # camera off a tracked target).  Observations whose capture-stamp age
    # exceeds this limit are rejected outright; the existing receipt-based
    # timeout then holds/stops the base safely.  The allowance absorbs the
    # measured PC/NUC NTP skew (~0.3s) plus the normal FFS+relay latency.
    max_target_capture_age_s: float = 0.70
    tracking_hold_s: float = 0.80
    tracking_loss_grace_s: float = 2.75
    handoff_settle_s: float = 0.30
    target_filter_window: int = 5
    target_filter_alpha: float = 0.55
    max_target_jump_m: float = 0.20
    outlier_rebase_samples: int = 3
    outlier_rebase_spread_m: float = 0.05
    base_frame: str = "base_link"
    arm_base_frame: str = "piper_base_link"
    # Formerly ``transform_timeout_s``, one letter away from the launcher's
    # ``--runtime-transform-timeout-s`` (which is a completely different
    # quantity: the maximum age of the subscribe-only runtime-observer JSON).
    # This one bounds how stale the DERIVED TargetGeometry may be.
    #
    # ``_transforms_received_s`` and ``_target_received_s`` are both assigned
    # ``float(stamp_s)`` from the SAME camera callback (see ``observe_target``:
    # the target stamp is written unconditionally, the transform stamp only on
    # the ``transforms_available`` branch, and the not-available branch nulls
    # the geometry).  Geometry non-None therefore IMPLIES the two stamps are
    # equal, so this budget and ``target_timeout_s`` measure the same physical
    # age.  Shipping 0.25 against a 0.40 target timeout at a ~0.133 s bundle
    # cadence meant TWO frame periods (0.265 s) hard-zeroed the base with a
    # reason string that blamed TF while printing the target age.  They are
    # now pinned equal, and ``__post_init__`` forbids the inversion.
    geometry_staleness_timeout_s: float = 0.40
    # Explicit test-only compatibility seam. Deployed ROS construction always
    # leaves this false: missing transforms must never fall back to optical z.
    allow_legacy_optical_depth_for_tests: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"shadow", "live"}:
            raise ValueError("mode must be shadow or live")
        if not math.isfinite(self.target_timeout_s) or self.target_timeout_s <= 0.0:
            raise ValueError("target timeout must be finite and positive")
        if (
            not math.isfinite(self.max_target_capture_age_s)
            or self.max_target_capture_age_s <= 0.0
        ):
            raise ValueError("maximum target capture age must be positive")
        if not math.isfinite(self.tracking_loss_grace_s) or self.tracking_loss_grace_s < self.target_timeout_s:
            raise ValueError("tracking-loss grace must be at least the target timeout")
        if (
            not math.isfinite(self.tracking_hold_s)
            or not 0.0 <= self.tracking_hold_s < self.tracking_loss_grace_s
        ):
            raise ValueError("tracking hold must be non-negative and below loss grace")
        if not math.isfinite(self.handoff_settle_s) or self.handoff_settle_s <= 0.0:
            raise ValueError("handoff settle time must be finite and positive")
        if not math.isfinite(self.max_yaw_step_rps) or self.max_yaw_step_rps <= 0.0:
            raise ValueError("maximum yaw step must be finite and positive")
        if self.target_filter_window < 1:
            raise ValueError("target filter window must be positive")
        if not 0.0 < self.target_filter_alpha <= 1.0:
            raise ValueError("target filter alpha must be in (0, 1]")
        if not math.isfinite(self.max_target_jump_m) or self.max_target_jump_m <= 0.0:
            raise ValueError("maximum target jump must be finite and positive")
        if self.outlier_rebase_samples < 2:
            raise ValueError("outlier rebase requires at least two samples")
        if (
            not math.isfinite(self.outlier_rebase_spread_m)
            or self.outlier_rebase_spread_m <= 0.0
        ):
            raise ValueError("outlier rebase spread must be finite and positive")
        if not self.base_frame.strip() or not self.arm_base_frame.strip():
            raise ValueError("base and arm-base frames must be non-empty")
        if (
            not math.isfinite(self.geometry_staleness_timeout_s)
            or self.geometry_staleness_timeout_s <= 0.0
        ):
            raise ValueError("geometry staleness timeout must be finite and positive")
        # ORDERING INVARIANT, same shape as the tracking-loss-grace check above.
        # ``transform_age_s`` IS ``target_age_s`` whenever the geometry exists
        # (both are written from one camera callback), so a geometry budget
        # BELOW the target budget can only ever fire early: it turns a single
        # dropped camera frame into a hard base zero wearing a TF error
        # message.  The geometry can never be fresher than the observation it
        # was derived from, so its budget can never legitimately be tighter.
        if self.geometry_staleness_timeout_s < self.target_timeout_s:
            raise ValueError(
                "geometry staleness timeout must be at least the target timeout",
            )
        if not math.isfinite(self.handoff_depth_m) or self.handoff_depth_m <= 0.0:
            raise ValueError("handoff depth must be finite and positive")
        if not math.isfinite(self.side_lateral_offset_m) or self.side_lateral_offset_m <= 0.0:
            raise ValueError("side lateral offset must be finite and positive")
        if not math.isfinite(self.side_lock_deadband_m) or self.side_lock_deadband_m < 0.0:
            raise ValueError("side-lock deadband must be finite and nonnegative")
        if not math.isfinite(self.side_handoff_tolerance_m) or self.side_handoff_tolerance_m <= 0.0:
            raise ValueError("side handoff tolerance must be finite and positive")
        if self.preferred_side_sign not in {-1, 1}:
            raise ValueError("preferred side sign must be -1 or 1")
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
    reason: str = ""
    reactive_phase: str | None = None
    needs_ik_probe: bool = False


# ``HANDOFF_TERMINAL_PHASES`` and ``LOSS_STAIR_PHASES`` are imported from
# ``z_manip.control.servo_phase``: they used to be hand-written here and
# hand-written AGAIN in go2w_reactive_supervision.py and inline in the
# whole-body branch below, with memberships that silently disagreed.
# LOSS_STAIR_PHASES: phases in which the servo has lost or frozen its live
# track.  A tracked-bundle arrival that immediately follows one of these is a
# loss-phase exit: the inter-arrival interval spans the loss dwell and must not
# feed the view-update damping EMA (a multi-second stall would otherwise poison
# the arm-view period).


# ``reactive_phase`` sentinel for rows the whole-body branch owns.  It is
# deliberately NOT a ``ServoPhase``: it never reaches the top-level ``phase``
# field, and it exists so a consumer can tell "the reactive controller chose
# this" from "the QP chose this".
WHOLE_BODY_REACTIVE_PHASE = "whole_body"


# The tracked-target bundle cadence is the measured FFS rate (~7.5-8.1 Hz, i.e.
# ~0.123-0.133 s/bundle).  Only inter-arrival intervals inside this band update
# the view-update damping period; gaps outside it are process pauses,
# reacquisition, or jitter and are ignored so the damper tracks the true rate.
VIEW_UPDATE_PERIOD_MIN_INTERVAL_S = 0.10
VIEW_UPDATE_PERIOD_MAX_INTERVAL_S = 0.20


# --- /track_3d/is_tracking freshness -----------------------------------------
#
# WHY THIS SUBSCRIPTION IS LATCHED AND WHY THE FLAG MUST STILL EXPIRE.
#
# The publisher (ros2/z_manip_edgetam/.../node.py, ``tracking_state_qos``)
# offers RELIABLE + TRANSIENT_LOCAL, and the VLM bridge subscribes latched to
# match.  This servo shared one VOLATILE profile with every other topic, so a
# freshly started servo received NO sample at all until EdgeTAM published its
# next observation.  In that window ``self.tracking`` is None, ``fresh_tracking``
# is False, and ReactiveTargetController._lost() sees its own ``_last_geometry``
# still None and returns SEARCH_REQUIRED with ZERO grace -- while DepthServoCore
# is holding a perfectly good geometry the decision cannot see.  In the recorded
# corpus (461 trace rows, artifacts/go2w_real/latest/depth-servo.trace.jsonl{,.1})
# EVERY row at ``bundle_count == 1`` is ``phase=search_required`` with
# ``tracking=null``: 12 of 12.
#
# Latching alone would be a fail-open: ``std_msgs/Bool`` is unstamped, so the
# durability cache can hand the servo a ``True`` from a session that ended
# before this process existed and nothing in the message can date it.  Receipt
# time is therefore recorded alongside the flag and the flag expires against it
# (``_tracking_flag_ttl_s`` / ``_aged_tracking_flag``).  Landing the QoS change
# without the expiry is the fail-open half of a two-part fix.
TRACKING_FLAG_STALE_PERIODS = 3.0
# Fallback ruler before ``view_update_period_s`` has two in-band arrivals to
# measure: the shipped FFS bundle cadence (~7.5 Hz).
TRACKING_FLAG_NOMINAL_PERIOD_S = 0.133


# Per-lookup ceiling for the tf2 FALLBACK path in the PointCloud2 callback.
#
# ``_target_transforms`` performs TWO sequential blocking ``lookup_transform``
# calls, each with ``Duration(seconds=...)``, on the node's SINGLE-THREADED
# executor -- the same thread that runs the 20 Hz control tick.  Sizing that
# blocking wait from ``geometry_staleness_timeout_s`` coupled two unrelated
# things: widening the staleness budget from 0.25 to 0.40 would have raised
# worst-case in-callback blocking from 0.50 s to 0.80 s, i.e. sixteen missed
# ticks, as a side effect of relaxing a freshness comparison.  Capping here
# holds today's worst case at exactly 0.50 s while decoupling the two.  A
# failed lookup is fail-closed (transform_error -> geometry None -> zero), so
# the cap can only ever stop the robot sooner, never later.
TF_LOOKUP_TIMEOUT_S = 0.25


# Trace cadence floors.  A non-terminal (actively approaching/recovering) servo
# is sampled at 5 Hz so diagnostics are not quantized into a phantom 1 Hz
# cadence; a terminal/parked servo changes slowly, so 1 Hz keeps the trace
# bounded.  Phase transitions always emit a row regardless of the floor.
TRACE_MOTION_MIN_INTERVAL_S = 0.20
TRACE_TERMINAL_MIN_INTERVAL_S = 1.0


def _latch_handoff_output(
    latched: DepthServoOutput | None,
    candidate: DepthServoOutput,
) -> DepthServoOutput | None:
    """Keep the base stopped once close-range planning has been requested.

    A handoff is a transaction boundary, not another visual-servo sample.  In
    real Go2W traces body sway made the next RGB-D sample leave the handoff
    corridor before the 5 Hz supervisor observed it.  Returning to approach
    at that point both moves the base again and loses the only signal that
    starts fresh close-range perception/planning.
    """

    if latched is not None:
        return latched
    if not (
        candidate.phase in HANDOFF_TERMINAL_PHASES
        or candidate.needs_ik_probe
        or candidate.reactive_phase
        in {ReactivePhase.HANDOFF_PROBE.value, ReactivePhase.HANDOFF_READY.value}
    ):
        return None
    return replace(
        candidate,
        proposed_linear_x=0.0,
        proposed_angular_z=0.0,
        published_linear_x=0.0,
        published_angular_z=0.0,
    )


def _handoff_latch_output(
    latched: DepthServoOutput | None,
    *,
    latched_since_s: float | None,
    now_s: float,
    timeout_s: float = HANDOFF_LATCH_TIMEOUT_S,
) -> DepthServoOutput | None:
    """The explicit reset edge for the one-way handoff latch.

    ``_latch_handoff_output`` is set by a single tick.  It had NO reset and NO
    expiry, so a servo whose supervisor never stopped it republished the same
    latched status at 20 Hz indefinitely, with ``needs_ik_probe`` still true --
    an advancing heartbeat and a permanent request for a transaction nobody
    was going to open.

    The edge is deliberately ONE-WAY-TO-STOP.  Data-driven un-latching (say,
    tracking recovering after a dropout) was rejected: the wrist camera is
    routinely occluded by the arm during a close-range handoff, so that rule
    would let the base drive again mid-grasp.  Expiry therefore produces
    ``handoff_abandoned``: still exactly (0, 0), but with ``needs_ik_probe``
    and ``reactive_phase`` CLEARED so ``_runtime_requests_handoff`` no longer
    reads it as a request to open a grasp transaction, and with a phase whose
    table row carries ``deadline_s = 0.0`` and ``on_expiry =
    stop_and_degrade`` so the supervisor terminates it on first sight.
    """

    if latched is None:
        return None
    if latched.phase == ServoPhase.HANDOFF_ABANDONED.value:
        return latched
    if (
        latched_since_s is None
        or not math.isfinite(timeout_s)
        or timeout_s <= 0.0
        or not math.isfinite(now_s)
    ):
        return latched
    held_s = now_s - latched_since_s
    if held_s < timeout_s:
        return latched
    return replace(
        latched,
        phase=ServoPhase.HANDOFF_ABANDONED.value,
        proposed_linear_x=0.0,
        proposed_angular_z=0.0,
        published_linear_x=0.0,
        published_angular_z=0.0,
        done=True,
        needs_ik_probe=False,
        reactive_phase=None,
        reason=(
            f"close-range handoff latched for {held_s:.1f}s without the "
            f"supervisor stopping this servo (bound {timeout_s:.1f}s); the "
            "base stays at zero and the supervisor must terminate or degrade"
        ),
    )


def _abandoned_reactive_status(
    reactive: dict[str, Any] | None,
    *,
    phase: str,
) -> dict[str, Any] | None:
    """Strip the handoff REQUEST out of the reactive block once abandoned.

    ``_handoff_latch_output`` clears ``needs_ik_probe``/``reactive_phase`` on
    the OUTPUT, but the status document's ``reactive`` block is rendered from
    ``DepthServoCore.reactive_status``, and during the latch ``_tick`` returns
    before ``core.tick`` runs -- so the core's last decision stays frozen with
    ``needs_ik_probe: True`` and ``phase: handoff_probe``.

    ``DepthServoRunner._runtime_requests_handoff`` reads that block as a
    fallback, so an abandoned latch still answered "yes, open a close-range
    grasp transaction".  That was safe only because ``_supervise`` happens to
    evaluate ``supervision.timed_out`` before the handoff check -- a
    live-motion guarantee resting on statement order in a different file.
    Clear it at the source instead, so the answer does not depend on who asks
    first.
    """

    if reactive is None or phase != ServoPhase.HANDOFF_ABANDONED.value:
        return reactive
    return {
        **reactive,
        "phase": ServoPhase.HANDOFF_ABANDONED.value,
        "needs_ik_probe": False,
        "handoff_ready": False,
    }


def _whole_body_posture_rate_converged(
    command: WholeBodyRuntimeCommand,
) -> bool:
    """Return true only when the QP no longer requests meaningful body motion."""

    try:
        intent = command.document["intent"]
        roll_rate = float(intent["body_roll_rps"])
        pitch_rate = float(intent["body_pitch_rps"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(value) for value in (roll_rate, pitch_rate))
        and abs(roll_rate) <= POSTURE_ANGLE_RATE_SETTLED_RPS
        and abs(pitch_rate) <= POSTURE_ANGLE_RATE_SETTLED_RPS
    )


def _whole_body_arm_rate_converged(command: WholeBodyRuntimeCommand) -> bool:
    rates = tuple(float(value) for value in command.arm_joint_velocity_rps)
    return (
        len(rates) == 6
        and all(math.isfinite(value) for value in rates)
        and max(abs(value) for value in rates) <= ARM_RATE_SETTLED_RPS
    )


def _arm_feedback_state(
    document: dict[str, Any] | None,
    *,
    age_s: float,
    required_seq: int | None,
) -> tuple[bool, bool, bool, str]:
    """Reduce the measured PiPER executor status to ready/reached gates."""

    if document is None or not math.isfinite(age_s) or age_s > ARM_STATUS_TIMEOUT_S:
        return False, False, False, "PiPER reactive executor status unavailable or stale"
    if document.get("schema") != ARM_STATUS_SCHEMA:
        return False, False, False, "PiPER reactive executor status schema is invalid"
    if document.get("owner") != "piper_reactive_view_executor":
        return False, False, True, "PiPER reactive CAN owner is not confirmed"
    fault = document.get("fault")
    stopped = document.get("stop_latched") is True
    blocked = stopped or fault not in (None, "")
    ready = document.get("ready") is True and not blocked
    try:
        accepted_seq = int(document.get("accepted_seq", -1))
        max_error_rad = float(document["max_error_rad"])
        feedback_age_s = float(document.get("feedback_age_s", 0.0))
    except (KeyError, TypeError, ValueError):
        return ready, False, blocked, "PiPER measured target evidence is incomplete"
    acknowledged = required_seq is None or accepted_seq >= required_seq
    reached = (
        ready
        and acknowledged
        and math.isfinite(max_error_rad)
        and math.isfinite(feedback_age_s)
        and 0.0 <= feedback_age_s <= ARM_STATUS_TIMEOUT_S
        and max_error_rad <= ARM_TARGET_ERROR_SETTLED_RAD
    )
    if blocked:
        detail = str(fault or "PiPER reactive executor is stop-latched")
    elif not acknowledged:
        detail = f"waiting for PiPER intent seq {required_seq}; accepted {accepted_seq}"
    elif reached:
        detail = f"PiPER measured target reached ({max_error_rad:.4f} rad max error)"
    else:
        detail = f"PiPER target error {max_error_rad:.4f} rad"
    return ready, reached, blocked, detail


def _arm_view_intent_document(
    command: WholeBodyRuntimeCommand,
    *,
    seq: int,
    now_unix_ns: int,
    target_source_timestamp_ns: int | None,
) -> dict[str, Any]:
    rates = tuple(float(value) for value in command.arm_joint_velocity_rps)
    if len(rates) != 6 or not all(math.isfinite(value) for value in rates):
        raise ValueError("whole-body arm intent must contain six finite velocities")
    if seq < 0 or now_unix_ns <= 0:
        raise ValueError("arm intent sequence and timestamp are invalid")
    return {
        "schema": ARM_INTENT_SCHEMA,
        "seq": int(seq),
        # This timestamp is the command-generation time used by the NUC lease.
        "source_timestamp_ns": int(now_unix_ns),
        "deadline_unix_ns": int(now_unix_ns) + ARM_INTENT_TTL_NS,
        # Keep the synchronized perception stamp separately for traceability;
        # it may use the ROS clock and must not be used as a lease clock.
        "target_source_timestamp_ns": target_source_timestamp_ns,
        "joint_velocity_rps": list(rates),
    }


def _rigid_transform_matrix(
    translation_xyz: tuple[float, float, float],
    quaternion_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    """Build a target-from-source transform from a ROS-style transform."""

    translation = np.asarray(translation_xyz, dtype=float)
    quaternion = np.asarray(quaternion_xyzw, dtype=float)
    if (
        translation.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(translation).all()
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError("transform components must be finite xyz and xyzw values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("transform quaternion must have non-zero norm")
    x, y, z, w = quaternion / norm
    rotation = np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _validated_matrix(value: object, *, name: str) -> np.ndarray:
    """Accept only a finite right-handed rigid 4x4 transform."""

    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-4):
        raise ValueError(f"{name} rotation is not right-handed")
    return matrix


def _runtime_state_transforms(
    path: Path,
    *,
    source_frame: str,
    base_frame: str,
    arm_base_frame: str,
    now_unix_ns: int,
    max_age_s: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read fresh verified model transforms from the subscribe-only observer.

    This is the deployed fallback for ROS graphs that publish RealSense TF but
    not the combined Go2W/PiPER model frames.  It is deliberately stricter
    than ordinary JSON loading: an old artifact, synthetic calibration, frame
    mismatch, or malformed rigid transform stops the base.
    """

    artifact = path.expanduser().resolve()
    if not artifact.is_file():
        raise ValueError(f"runtime observer state is missing: {artifact}")
    if artifact.stat().st_size > 2_000_000:
        raise ValueError("runtime observer state exceeds the bounded size")
    document = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != "z_manip.runtime_state.v1":
        raise ValueError("runtime observer state schema is invalid")
    transforms = document.get("kinematic_transforms")
    if not isinstance(transforms, dict):
        raise ValueError("runtime observer has no kinematic transforms")
    if (
        transforms.get("schema") != "z_manip.kinematic_transforms.v1"
        or transforms.get("verified") is not True
        or transforms.get("calibration_synthetic") is not False
    ):
        raise ValueError("runtime kinematic transform evidence is not verified")
    if str(transforms.get("camera_frame", "")) != source_frame:
        raise ValueError("runtime camera frame does not match the target cloud")
    if str(transforms.get("platform_base_frame", "")) != base_frame:
        raise ValueError("runtime platform frame does not match the servo base frame")
    if str(transforms.get("arm_base_frame", "")) != arm_base_frame:
        raise ValueError("runtime arm frame does not match the servo arm frame")
    timestamp_ns = int(transforms.get("source_timestamp_ns", 0))
    age_s = (int(now_unix_ns) - timestamp_ns) / 1e9
    if age_s < -0.50 or age_s > max_age_s:
        raise ValueError(f"runtime kinematic transforms are stale ({age_s:.3f}s)")
    base = _validated_matrix(
        transforms.get("platform_base_from_camera"),
        name="platform_base_from_camera",
    )
    arm = _validated_matrix(
        transforms.get("arm_base_from_camera"),
        name="arm_base_from_camera",
    )
    return base, arm, timestamp_ns


def _euler_body_unavailable(document: dict[str, Any] | None) -> bool:
    """Return true when the posture adapter positively reports no Euler DOF.

    The Go2W ai-w wheeled-sport service answers Euler(1007) with RPC 3203 for
    the whole epoch, so body pitch/roll can never move the wrist-camera view.
    Only positive evidence (``euler`` false with a known unavailable state)
    disables body posture; a missing/ambiguous capability keeps the legacy
    body-posture behaviour.
    """

    capabilities = (
        document.get("capabilities") if isinstance(document, dict) else None
    )
    return bool(
        isinstance(capabilities, dict)
        and capabilities.get("euler") is False
        and capabilities.get("euler_state") in {
            "UNKNOWN",
            "UNSUPPORTED_FOR_EPOCH",
            "TRANSIENT_FAULT",
        }
    )


def _ik_probe_state(
    document: dict[str, Any] | None,
    *,
    age_s: float,
    timeout_s: float = IK_PROBE_TIMEOUT_S,
) -> bool | None:
    """Reduce a resident close-range IK-probe report to the handoff contract.

    Returns ``True``/``False`` only for a fresh, schema-valid explicit verdict;
    any absence, staleness, or malformed payload returns ``None`` so the
    reactive controller keeps requesting the probe (fail-closed HANDOFF_PROBE)
    rather than declaring a handoff on missing evidence.
    """

    if document is None or not math.isfinite(age_s) or age_s > timeout_s:
        return None
    if document.get("schema") != IK_PROBE_SCHEMA:
        return None
    feasible = document.get("feasible")
    if feasible is True:
        return True
    if feasible is False:
        return False
    return None


def _posture_feedback_state(
    document: dict[str, Any] | None,
    *,
    age_s: float,
    timeout_s: float = 0.75,
) -> tuple[bool, bool, bool, str]:
    """Reduce posture-adapter telemetry to the reactive-controller contract.

    Shadow verification is deliberately distinct from measured settling: it
    is useful diagnostic evidence, but can never unlock manipulation handoff.
    """

    if document is None or not math.isfinite(age_s) or age_s > timeout_s:
        return False, False, False, "posture status unavailable or stale"
    if document.get("schema") != "z_manip.go2w_posture_status.v1":
        return False, False, False, "posture status schema is invalid"
    phase = str(document.get("phase", ""))
    mode = str(document.get("mode", ""))
    detail = str(document.get("detail", ""))
    stop_latched = document.get("stop_latched") is True
    capabilities = document.get("capabilities")
    euler_supported = (
        isinstance(capabilities, dict)
        and capabilities.get("euler") is True
        and capabilities.get("euler_state") == "SUPPORTED_OBSERVED"
    )
    euler_unavailable = _euler_body_unavailable(document)
    # PosturePhase, NOT ServoPhase.  "stopping" has no PosturePhase member;
    # it is a defensive spelling for an adapter mid-teardown.
    blocked = stop_latched or phase in {
        PosturePhase.BLOCKED.value,
        PosturePhase.FAULT.value,
        PosturePhase.STOPPED.value,
        "stopping",
    }
    feedback = document.get("feedback")
    feedback_fresh = (
        isinstance(feedback, dict) and feedback.get("fresh") is True
    )
    if euler_unavailable and mode == "live" and feedback_fresh and not stop_latched:
        return (
            True,
            False,
            False,
            detail or "Euler unavailable; body posture bypassed for base + arm fallback",
        )
    settled = (
        mode == "live"
        # PosturePhase, NOT ServoPhase: the posture adapter has its own
        # vocabulary that happens to reuse the string "reached".  Naming the
        # enum keeps the two namespaces from being confused again.
        and phase == PosturePhase.REACHED.value
        and feedback_fresh
        and not stop_latched
        and euler_supported
        and _posture_ack_matches_target(document)
    )
    shadow_verified = mode == "shadow" and phase == "shadow" and not blocked
    return settled, blocked, shadow_verified, detail


def _posture_ack_matches_target(document: dict[str, Any]) -> bool:
    """Reject legacy `reached` states without a same-generation code-0 ACK."""

    command = document.get("command")
    if not isinstance(command, dict):
        return False
    target_generation = command.get("posture_generation")
    ack_generation = command.get("euler_ack_generation")
    ack_code = command.get("euler_ack_code")
    return bool(
        isinstance(target_generation, int)
        and not isinstance(target_generation, bool)
        and target_generation >= 0
        and isinstance(ack_generation, int)
        and not isinstance(ack_generation, bool)
        and ack_generation == target_generation
        and isinstance(ack_code, int)
        and not isinstance(ack_code, bool)
        and ack_code == 0
    )


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
        self.reactive = ReactiveTargetController(ReactiveServoConfig(
            desired_planar_standoff_m=settings.desired_depth_m,
            posture_entry_planar_m=max(settings.handoff_depth_m + 0.20, 0.80),
            handoff_planar_max_m=settings.handoff_depth_m,
            handoff_lateral_tolerance_m=settings.side_handoff_tolerance_m,
            camera_handoff_depth_m=settings.handoff_depth_m,
            camera_handoff_planar_slack_m=0.15,
            linear_gain=settings.linear_gain,
            yaw_gain=settings.yaw_gain,
            max_forward_mps=settings.max_forward_mps,
            max_yaw_rps=settings.max_yaw_rps,
            yaw_deadband_rad=settings.yaw_deadband_rad,
            tracking_loss_grace_s=settings.tracking_loss_grace_s,
            tracking_hold_s=settings.tracking_hold_s,
            handoff_settle_s=settings.handoff_settle_s,
        ))
        self._target: tuple[float, float, float] | None = None
        self._raw_target: tuple[float, float, float] | None = None
        self._target_received_s: float | None = None
        self._samples: deque[tuple[float, float, float]] = deque(
            maxlen=settings.target_filter_window,
        )
        self._outlier_samples: deque[tuple[float, float, float]] = deque(
            maxlen=settings.outlier_rebase_samples,
        )
        self._accepted_observations = 0
        self._rejected_observations = 0
        self._stale_data_rejections = 0
        self._rebases = 0
        self._geometry: TargetGeometry | None = None
        self._transforms_received_s: float | None = None
        self._transform_error: str | None = "no synchronized target transforms"
        self._ik_feasible: bool | None = None
        self._last_decision: ReactiveServoDecision | None = None
        self._done = False
        self._side_sign: int | None = None

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
    def geometry(self) -> TargetGeometry | None:
        return self._geometry

    @property
    def desired_target_lateral_m(self) -> float:
        if self._side_sign is None:
            return 0.0
        return self._side_sign * self.settings.side_lateral_offset_m

    @property
    def side_lateral_error_m(self) -> float | None:
        if self._geometry is None or self._side_sign is None:
            return None
        return self._geometry.base_xyz_m[1] - self.desired_target_lateral_m

    def _latch_side(self, geometry: TargetGeometry) -> None:
        if self._side_sign is not None:
            return
        lateral_m = geometry.base_xyz_m[1]
        if lateral_m > self.settings.side_lock_deadband_m:
            self._side_sign = 1
        elif lateral_m < -self.settings.side_lock_deadband_m:
            self._side_sign = -1
        else:
            self._side_sign = self.settings.preferred_side_sign

    @property
    def reactive_status(self) -> dict[str, Any] | None:
        decision = self._last_decision
        if decision is None:
            return None
        return {
            "phase": decision.phase.value,
            "reason": decision.reason,
            "handoff_ready": decision.handoff_ready,
            "needs_ik_probe": decision.needs_ik_probe,
            "ik_feasible": self._ik_feasible,
            "side": (
                None if self._side_sign is None else (
                    "left" if self._side_sign > 0 else "right"
                )
            ),
            "desired_target_lateral_m": self.desired_target_lateral_m,
            "lateral_error_m": self.side_lateral_error_m,
            "posture": asdict(decision.posture),
            "arm_view": {
                **asdict(decision.arm_view),
                "mode": decision.arm_view.mode.value,
            },
        }

    @property
    def transform_status(self) -> dict[str, Any]:
        return {
            "valid": self._geometry is not None,
            "error": self._transform_error,
            "received_monotonic_s": self._transforms_received_s,
        }

    def set_ik_probe_result(self, feasible: bool | None) -> None:
        """Record a downstream read-only IK probe result for handoff gating."""

        self._ik_feasible = None if feasible is None else bool(feasible)

    @property
    def filter_stats(self) -> dict[str, int | float | None]:
        return {
            "window_samples": len(self._samples),
            "accepted": self._accepted_observations,
            "rejected_outliers": self._rejected_observations,
            "stale_data_rejections": self._stale_data_rejections,
            "outlier_cluster_samples": len(self._outlier_samples),
            "rebases": self._rebases,
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
        T_base_camera: np.ndarray | None = None,
        T_arm_camera: np.ndarray | None = None,
        transform_error: str | None = None,
        capture_age_s: float | None = None,
    ) -> bool:
        """Observe a complete optical-frame target centroid.

        ``y_m`` defaults to zero only for backward-compatible callers.  The
        ROS adapter always supplies the measured optical y coordinate.
        ``capture_age_s`` is the skew-adjusted age of the CAPTURE stamp; data
        older than ``max_target_capture_age_s`` is rejected so a congested
        network cannot make the servo steer on an old world.
        """
        if (
            capture_age_s is not None
            and math.isfinite(capture_age_s)
            and capture_age_s > self.settings.max_target_capture_age_s
        ):
            self._rejected_observations += 1
            self._stale_data_rejections += 1
            return False

        values = (float(x_m), float(y_m), float(z_m), float(stamp_s))
        if not all(math.isfinite(value) for value in values) or z_m <= 0.0:
            return False
        transforms_available = T_base_camera is not None and T_arm_camera is not None
        # A fresh camera observation accompanied by failed TF must stop motion
        # instead of silently retaining older valid target geometry.
        if not transforms_available:
            self._geometry = None
            self._transforms_received_s = None
            self._transform_error = (
                transform_error or "synchronized transforms unavailable"
            )
        raw = (float(x_m), float(y_m), float(z_m))
        self._raw_target = raw
        if self._target is not None:
            jump_m = math.sqrt(sum(
                (raw[index] - self._target[index]) ** 2 for index in range(3)
            ))
            if jump_m > self.settings.max_target_jump_m:
                self._outlier_samples.append(raw)
                self._rejected_observations += 1
                if len(self._outlier_samples) < self.settings.outlier_rebase_samples:
                    return False
                cluster_median = tuple(
                    statistics.median(sample[index] for sample in self._outlier_samples)
                    for index in range(3)
                )
                cluster_spread = max(
                    math.sqrt(sum(
                        (sample[index] - cluster_median[index]) ** 2
                        for index in range(3)
                    ))
                    for sample in self._outlier_samples
                )
                if cluster_spread > self.settings.outlier_rebase_spread_m:
                    return False
                # A coherent replacement cluster is a real target relocation,
                # not isolated depth noise. Rebase the filter so the old EMA
                # cannot reject the new stable track forever.
                self._samples.clear()
                self._samples.extend(self._outlier_samples)
                self._outlier_samples.clear()
                self._target = None
                self._rebases += 1
            else:
                self._outlier_samples.clear()
        if self._target is not None or not self._samples:
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
        if transforms_available:
            self._geometry = TargetGeometry.from_camera(
                self._target,
                T_base_camera=T_base_camera,
                T_arm_camera=T_arm_camera,
            )
            self._transforms_received_s = float(stamp_s)
            self._transform_error = None
            self._latch_side(self._geometry)
        return True

    def reset(self) -> None:
        self._target = None
        self._raw_target = None
        self._target_received_s = None
        self._samples.clear()
        self._outlier_samples.clear()
        self._accepted_observations = 0
        self._rejected_observations = 0
        self._stale_data_rejections = 0
        self._rebases = 0
        self._geometry = None
        self._transforms_received_s = None
        self._transform_error = "no synchronized target transforms"
        self._ik_feasible = None
        self._last_decision = None
        self._done = False
        self._side_sign = None
        self.controller.reset()
        self.reactive.reset()

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
            reason=self._transform_error or "",
        )

    def _reactive_tick(
        self,
        *,
        now_s: float,
        age_s: float,
        tracking: bool | None,
        body_settled: bool,
        posture_blocked: bool,
        posture_shadow_verified: bool,
        posture_detail: str,
        body_posture_actionable: bool = True,
    ) -> DepthServoOutput:
        fresh_tracking = (
            tracking is True and age_s <= self.settings.target_timeout_s
        )
        transform_age_s = (
            None
            if self._transforms_received_s is None
            else max(0.0, now_s - self._transforms_received_s)
        )
        # The ``self._geometry is not None`` term is the load-bearing one and
        # must stay: live 2026-07-24, raw_y swept +/-0.67 m sinusoidally while
        # the source stamp froze and the arm chased its own motion.  The AGE
        # term is a duplicate of ``target_timeout_s`` (the two receipt stamps
        # are written from one callback), which is why the settings validator
        # now forbids it from being the tighter of the two: it may confirm the
        # target-freshness verdict, never pre-empt it.
        transform_fresh = (
            self._geometry is not None
            and transform_age_s is not None
            and transform_age_s <= self.settings.geometry_staleness_timeout_s
        )
        # A transform is synchronized to a target observation.  Once the
        # tracker stops publishing targets, both timestamps necessarily age
        # together.  Classify that condition as target/tracking loss first;
        # otherwise a terminal EdgeTAM loss is misleadingly reported as a TF
        # outage even while the runtime observer continues publishing fresh
        # kinematic transforms.  A genuinely fresh target with missing/stale
        # geometry remains fail-closed below.
        if fresh_tracking and not transform_fresh:
            reason = self._transform_error or (
                f"synchronized transforms are stale ({transform_age_s:.3f}s)"
                if transform_age_s is not None
                else "synchronized transforms unavailable"
            )
            self._last_decision = ReactiveServoDecision(
                phase=ReactivePhase.TRANSFORM_UNAVAILABLE,
                base=BaseMotionIntent(),
                posture=PostureIntent(),
                arm_view=ArmViewIntent(),
                geometry=None,
                reason=reason,
            )
            output = self._zero(ServoPhase.TRANSFORM_UNAVAILABLE.value, age_s)
            return DepthServoOutput(
                **{
                    **asdict(output),
                    "reason": reason,
                    "reactive_phase": ReactivePhase.TRANSFORM_UNAVAILABLE.value,
                },
            )
        if posture_blocked and self._last_decision is not None and (
            self._last_decision.phase is ReactivePhase.POSTURE_ADJUST
        ):
            output = self._zero(ServoPhase.POSTURE_BLOCKED.value, age_s)
            return DepthServoOutput(
                **{
                    **asdict(output),
                    "reason": posture_detail or "posture adapter blocked the intent",
                    "reactive_phase": ReactivePhase.POSTURE_ADJUST.value,
                },
            )
        decision = self.reactive.update(
            self._geometry if fresh_tracking and transform_fresh else None,
            now_s=now_s,
            tracking=fresh_tracking,
            # The depth-servo runtime does not own posture hardware. A posture
            # adapter may later feed measured settling; exposing intents here
            # must never manufacture an active body command.
            body_settled=body_settled,
            ik_feasible=self._ik_feasible,
            desired_target_lateral_m=self.desired_target_lateral_m,
            body_posture_actionable=body_posture_actionable,
        )
        self._last_decision = decision
        if (
            not fresh_tracking
            and decision.phase is ReactivePhase.SEARCH_REQUIRED
        ):
            # A terminal loss ends the side-approach session.  A subsequent
            # reacquisition must choose its side from the new target geometry.
            self._side_sign = None
        phase = decision.phase.value
        if decision.phase is ReactivePhase.HANDOFF_READY:
            self._done = True
            phase = ServoPhase.REACHED.value
        elif decision.phase is ReactivePhase.BASE_APPROACH:
            phase = ServoPhase.APPROACH.value
        elif (
            posture_shadow_verified
            and decision.phase is ReactivePhase.POSTURE_ADJUST
        ):
            phase = ServoPhase.POSTURE_SHADOW_VERIFIED.value
        linear_x = decision.base.linear_x_mps
        angular_z = decision.base.angular_z_rps
        live = self.settings.mode == "live"
        geometry = decision.geometry
        depth_error = None
        yaw_error = None
        if geometry is not None:
            depth_error = (
                geometry.base_planar_distance_m
                - self.settings.desired_depth_m
            )
            yaw_error = math.atan2(
                geometry.base_xyz_m[1] - self.desired_target_lateral_m,
                max(geometry.base_xyz_m[0], 0.05),
            )
        return DepthServoOutput(
            phase=phase,
            proposed_linear_x=linear_x,
            proposed_angular_z=angular_z,
            published_linear_x=linear_x if live else 0.0,
            published_angular_z=angular_z if live else 0.0,
            depth_error_m=depth_error,
            yaw_error_rad=yaw_error,
            target_age_s=age_s,
            done=self._done,
            reason=decision.reason,
            reactive_phase=decision.phase.value,
            needs_ik_probe=decision.needs_ik_probe,
        )

    def tick(
        self,
        *,
        now_s: float,
        tracking: bool | None,
        body_settled: bool = False,
        posture_blocked: bool = False,
        posture_shadow_verified: bool = False,
        posture_detail: str = "",
        body_posture_actionable: bool = True,
    ) -> DepthServoOutput:
        now = float(now_s)
        if self._done:
            return self._zero(ServoPhase.REACHED.value, 0.0)
        if self._target is None or self._target_received_s is None:
            return self._zero(ServoPhase.WAITING_TARGET.value, None)
        age_s = max(0.0, now - self._target_received_s)
        if not self.settings.allow_legacy_optical_depth_for_tests:
            return self._reactive_tick(
                now_s=now,
                age_s=age_s,
                tracking=tracking,
                body_settled=body_settled,
                posture_blocked=posture_blocked,
                posture_shadow_verified=posture_shadow_verified,
                posture_detail=posture_detail,
                body_posture_actionable=body_posture_actionable,
            )
        if tracking is not True or age_s > self.settings.target_timeout_s:
            phase = (
                ServoPhase.REACQUIRE.value
                if age_s <= self.settings.tracking_loss_grace_s
                else ServoPhase.TRACKING_LOST.value
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
                phase=ServoPhase.REACHED.value,
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
        phase = ServoPhase.APPROACH.value
        if command.converged:
            self._done = True
            phase = ServoPhase.REACHED.value
        elif linear_x == 0.0 and command.angular_z == 0.0:
            phase = ServoPhase.SETTLING.value
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
    """Publish ``document`` at ``path`` through a genuinely unique temp file.

    The old scratch name was ``.{name}.{os.getpid()}.tmp``.  The servo is
    ALWAYS pid 1 inside its container, so that expression is the compile-time
    constant ``.depth-servo.json.1.tmp`` -- and pid namespaces do not make the
    HOST directory unique.  The status directory is bind-mounted read-write
    into every runtime container, so a second servo/replay container (also pid
    1) writes the identical scratch path: one process truncates the other's
    half-written file and ``os.replace`` then publishes a torn document that
    every reader parses with ``json.JSONDecodeError -> {}``, i.e. as "the servo
    is not running".  ``uuid4`` is per-write and namespace-independent.

    ``write_text`` (not ``tempfile.mkstemp``) is kept deliberately: mkstemp
    forces 0600 and the status file is read by other uids on this deployment.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        # A unique name no longer self-cleans by being overwritten next tick,
        # so a failed write must remove its own scratch file or a 20 Hz servo
        # papers the status directory with orphans.
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _append_jsonl(path: Path, document: dict[str, Any]) -> None:
    """Append compact bounded diagnostics without ever storing camera data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 2_000_000:
        rotated = path.with_suffix(path.suffix + ".1")
        rotated.unlink(missing_ok=True)
        os.replace(path, rotated)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")


def _view_period_update(
    *,
    previous_period_s: float | None,
    last_arrival_s: float | None,
    arrival_s: float,
    loss_phase_exit: bool,
) -> float | None:
    """Return the view-update damping EMA after a fresh bundle arrival.

    The period is only advanced by genuine cadence intervals: the first bundle
    (``last_arrival_s is None``), a loss-phase exit, and any interval outside
    the measured FFS band leave the previous period untouched so a stall never
    inflates the arm-view damper.
    """

    if last_arrival_s is None or loss_phase_exit:
        return previous_period_s
    interval_s = arrival_s - last_arrival_s
    if not (
        VIEW_UPDATE_PERIOD_MIN_INTERVAL_S
        <= interval_s
        <= VIEW_UPDATE_PERIOD_MAX_INTERVAL_S
    ):
        return previous_period_s
    if previous_period_s is None:
        return interval_s
    # Light 0.3 weight: a single jittered frame cannot swing the damping cap.
    return 0.7 * previous_period_s + 0.3 * interval_s


def _tracking_flag_ttl_s(
    *,
    view_update_period_s: float | None,
    target_timeout_s: float,
) -> float:
    """Return how long a received ``/track_3d/is_tracking`` sample stays valid.

    EdgeTAM publishes the flag from ``_publish_observation``, in the SAME call
    that publishes the selected-target cloud, so the flag cadence IS the bundle
    cadence and ``view_update_period_s`` (the measured, band-clamped EMA of
    that cadence) is the right ruler.  Three periods is one lost sample plus
    two of jitter; at the shipped ~0.133 s cadence that is ~0.40 s.

    Floored at ``target_timeout_s`` on purpose.  This TTL exists to reject a
    LATCHED HISTORICAL sample -- something seconds to minutes old, because a
    live publisher would have refreshed it within one frame -- not to police
    steady-state jitter.  Letting it drop below the target-freshness budget
    that already governs the very same decision would manufacture new stops
    without catching anything the target timer does not already catch.
    """

    period_s = view_update_period_s
    if period_s is None or not math.isfinite(period_s) or period_s <= 0.0:
        period_s = TRACKING_FLAG_NOMINAL_PERIOD_S
    return max(float(target_timeout_s), TRACKING_FLAG_STALE_PERIODS * period_s)


def _aged_tracking_flag(
    flag: bool | None,
    *,
    age_s: float | None,
    ttl_s: float,
) -> bool | None:
    """Degrade a stale tracking flag to UNKNOWN (``None``).

    MANDATORY COMPANION TO TRANSIENT_LOCAL DURABILITY on this subscription.
    ``std_msgs/Bool`` has no header, so a sample handed to a late-joining
    subscriber by the publisher's durability cache is byte-identical to one
    published this instant.  Subscribing TRANSIENT_LOCAL without this ages
    nothing and the servo can act on a ``True`` from a tracking session that
    ended before the process started -- a fail-open.  Receipt time is the only
    clock available, so receipt time is what expires.

    UNKNOWN, not ``False``: every consumer already spells the positive test
    (``tracking is True``), so ``None`` and ``False`` are equivalent at the
    gates while ``None`` is the honest report in the status document.
    """

    if flag is None:
        return None
    if age_s is None or not math.isfinite(age_s) or age_s > ttl_s:
        return None
    return bool(flag)


def _trace_min_interval_s(*, terminal: bool) -> float:
    """Trace-row cadence floor: 1 Hz when parked, 5 Hz during live motion."""

    return TRACE_TERMINAL_MIN_INTERVAL_S if terminal else TRACE_MOTION_MIN_INTERVAL_S


def _trace_row(
    document: dict[str, Any],
    *,
    view_update_period_s: float | None,
    bundle_count: int,
) -> dict[str, Any]:
    """Build a compact trace row from an already-written status document.

    ``view_update_period_s`` (the live FFS-rate damping period) and the
    monotonic ``bundle_count`` are promoted to first-class fields so a trace can
    be read for cadence health without cross-referencing the status file.
    """

    return {
        "schema": "z_manip.depth_servo_trace.v1",
        "updated_unix_ns": document["updated_unix_ns"],
        "mode": document["mode"],
        "phase": document["phase"],
        "tracking": document["tracking"],
        "target": document["target"],
        "source_stamp_ns": document["source_stamp_ns"],
        "view_update_period_s": view_update_period_s,
        "bundle_count": bundle_count,
        "output": document["output"],
        "filter": document["filter"],
        "posture_status": document["posture_status"],
        "arm_view_status": document["arm_view_status"],
        "whole_body": document["whole_body"],
    }


def _tick_should_skip(*, stop_requested: bool, ros_ok: bool) -> bool:
    """A servo tick after a stop request or a torn-down context must no-op.

    The ROS timer can fire the tick after SIGTERM latched the stop or after the
    rcl context began tearing down; publishing then would crash-loop the
    shutdown path instead of letting it settle on the transport watchdog stop.
    """

    return stop_requested or not ros_ok


def _shutdown_phase(*, stop_requested: bool) -> str:
    """Terminal phase to publish once the spin loop has returned.

    Split out of the shutdown block so the STOPPED/EXITED distinction -- the
    difference between "an operator or the supervisor asked us to stop" and
    "the executor fell out from under us" -- is testable without rclpy.
    """

    return (
        ServoPhase.STOPPED.value if stop_requested else ServoPhase.EXITED.value
    )


#: How long the spin loop may block before re-checking the stop flag.
#:
#: The SIGTERM handler is not allowed to touch the ROS context (see the
#: shutdown block), so it cannot wake a blocking wait; and installing a Python
#: handler over SIGINT/SIGTERM displaces the C-level handler rclpy.init()
#: registered to trigger every wait set's guard condition.  A bounded wait is
#: what turns "the handler sets a flag" into "the loop notices".  One 20 Hz
#: tick period: the final zero-Twist is delayed by at most this much, ~9 mm of
#: travel at the 0.18 m/s ceiling, and the transport keeps its own watchdog.
SHUTDOWN_POLL_INTERVAL_S = 0.05


#: Every ``DepthServoSettings`` field that participates in the target
#: freshness / tracking-loss stair, mapped to the CLI flag that sets it.
#:
#: The launcher pins every servo knob explicitly, which means a default this
#: table does NOT name is a value no operator can see or change -- and the
#: settings-vs-launcher test can only assert on flags the launcher already
#: passes.  That is exactly how ``--transform-timeout-s`` shipped unpinned at
#: 0.25 under a 0.40 target timeout: commit c208a6d's launcher test covered
#: hold/grace/target-timeout because those three appeared in the launcher, and
#: was structurally incapable of noticing the fourth.  tests/ walks THIS table
#: and every duration field of the dataclass, so a new stair knob fails the
#: suite until it is either pinned or explicitly classified as not-stair.
STAIR_SETTING_FLAGS: dict[str, str] = {
    "target_timeout_s": "--target-timeout-s",
    "max_target_capture_age_s": "--max-target-capture-age-s",
    "tracking_hold_s": "--tracking-hold-s",
    "tracking_loss_grace_s": "--tracking-loss-grace-s",
    "geometry_staleness_timeout_s": "--geometry-staleness-timeout-s",
    "handoff_settle_s": "--handoff-settle-s",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--target-topic", default="/track_3d/selected_target_pointcloud")
    parser.add_argument("--tracking-topic", default="/track_3d/is_tracking")
    parser.add_argument("--velocity-topic", default="/cmd_vel")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--arm-base-frame", default="piper_base_link")
    parser.add_argument(
        "--runtime-state",
        type=Path,
        help="fresh subscribe-only runtime-observer.json transform source",
    )
    parser.add_argument("--runtime-transform-timeout-s", type=float, default=0.50)
    parser.add_argument("--desired-depth-m", type=float, default=0.50)
    parser.add_argument("--handoff-depth-m", type=float, default=0.52)
    parser.add_argument("--handoff-bearing-deg", type=float, default=20.0)
    parser.add_argument("--min-forward-mps", type=float, default=0.10)
    parser.add_argument("--max-forward-mps", type=float, default=0.18)
    parser.add_argument("--max-yaw-rps", type=float, default=0.12)
    parser.add_argument("--yaw-deadband-deg", type=float, default=10.0)
    parser.add_argument("--max-yaw-step-rps", type=float, default=0.015)
    parser.add_argument("--target-timeout-s", type=float, default=0.40)
    parser.add_argument("--max-target-capture-age-s", type=float, default=0.70)
    parser.add_argument("--tracking-hold-s", type=float, default=0.80)
    parser.add_argument("--tracking-loss-grace-s", type=float, default=2.75)
    parser.add_argument("--handoff-settle-s", type=float, default=0.30)
    # RENAMED from ``--transform-timeout-s``.  That spelling sat one word away
    # from ``--runtime-transform-timeout-s`` above while meaning something
    # entirely different (derived-geometry staleness, not runtime-observer
    # document age), and the launcher passed only the latter -- so an operator
    # reading go2w_depth_servo.sh saw "0.50" and had no way to know a second,
    # tighter, invisible 0.25 s budget was also gating the base.
    parser.add_argument("--geometry-staleness-timeout-s", type=float, default=0.40)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--whole-body", choices=("off", "casadi"), default="casadi")
    parser.add_argument("--whole-body-urdf", type=Path)
    parser.add_argument("--whole-body-calibration", type=Path)
    parser.add_argument("--whole-body-collision-model", type=Path)
    return parser


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _settings_from_args(args: argparse.Namespace) -> DepthServoSettings:
    """Build the settings exactly as the deployed runtime does.

    Extracted out of ``_run_ros`` so the argparse-to-settings mapping -- the
    only place a launcher flag becomes a control budget -- can be exercised
    without rclpy, pinocchio, or casadi on the test host.  Inlined, the
    ``--geometry-staleness-timeout-s``/``--runtime-transform-timeout-s``
    confusion lived behind an import no test could perform.
    """

    return DepthServoSettings(
        mode=args.mode,
        desired_depth_m=args.desired_depth_m,
        handoff_depth_m=args.handoff_depth_m,
        handoff_bearing_rad=math.radians(args.handoff_bearing_deg),
        min_forward_mps=args.min_forward_mps,
        max_forward_mps=args.max_forward_mps,
        max_yaw_rps=args.max_yaw_rps,
        yaw_deadband_rad=math.radians(args.yaw_deadband_deg),
        max_yaw_step_rps=args.max_yaw_step_rps,
        target_timeout_s=args.target_timeout_s,
        max_target_capture_age_s=args.max_target_capture_age_s,
        tracking_hold_s=args.tracking_hold_s,
        tracking_loss_grace_s=args.tracking_loss_grace_s,
        handoff_settle_s=args.handoff_settle_s,
        base_frame=args.base_frame,
        arm_base_frame=args.arm_base_frame,
        geometry_staleness_timeout_s=args.geometry_staleness_timeout_s,
    )


def _run_ros(args: argparse.Namespace) -> int:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Bool, String
    from tf2_ros import Buffer, TransformException, TransformListener

    if not math.isfinite(args.rate_hz) or args.rate_hz <= 0.0:
        raise ValueError("rate must be finite and positive")
    if (
        not math.isfinite(args.runtime_transform_timeout_s)
        or args.runtime_transform_timeout_s <= 0.0
    ):
        raise ValueError("runtime transform timeout must be finite and positive")
    settings = _settings_from_args(args)

    class DepthServoNode(Node):
        def __init__(self) -> None:
            super().__init__("z_manip_depth_servo")
            # Latched by the signal handler; the timer-driven tick checks it (and
            # rclpy.ok()) at its top so no publish races a context teardown.
            self.stop_event = threading.Event()
            self.core = DepthServoCore(settings)
            # RAW last-received flag.  Nothing may read this directly: every
            # gate must go through ``self.tracking`` (the aged view), or a
            # latched historical TRANSIENT_LOCAL sample becomes an eternal
            # "yes, we are tracking".  See TRACKING_FLAG_STALE_PERIODS.
            self.tracking_raw: bool | None = None
            self.tracking_received_s: float | None = None
            self.last_source_stamp_ns: int | None = None
            self.last_source_frame: str | None = None
            # Live measurement of the tracked-target bundle period, used by the
            # whole-body view damper to keep the arm-view loop discrete-stable at
            # the measured FFS rate (~7.5-8 Hz) without raising any sensor rate.
            # Only genuine new bundles (advancing source stamp) count, and the
            # arrival is timed on the skew-free local monotonic clock.
            self.last_bundle_monotonic_s: float | None = None
            self.view_update_period_s: float | None = None
            # Monotonic count of genuine new tracked-target bundles (advancing
            # source stamp), promoted to a first-class trace field for cadence
            # health without cross-referencing the status file.
            self.bundle_count = 0
            self.last_transform_error: str | None = "no target transforms received"
            self.last_transform_success_s: float | None = None
            self.last_transform_source: str | None = None
            self.last_transform_stamps_ns: dict[str, int | None] = {
                settings.base_frame: None,
                settings.arm_base_frame: None,
            }
            self.posture_status: dict[str, Any] | None = None
            self.posture_status_received_s: float | None = None
            self.last_posture_intent: tuple[float, float] | None = None
            self.last_posture_intent_s = 0.0
            self.arm_view_status: dict[str, Any] | None = None
            self.arm_view_status_received_s: float | None = None
            self.ik_probe_status: dict[str, Any] | None = None
            self.ik_probe_status_received_s: float | None = None
            self.arm_view_intent_seq = 0
            self.last_arm_view_intent: dict[str, Any] | None = None
            self.whole_body: WholeBodyRuntimeController | None = None
            self.whole_body_command: WholeBodyRuntimeCommand | None = None
            self.whole_body_error: str | None = None
            self.whole_body_handoff_settle_cycles = 0
            self.last_conditioned_yaw_rps = 0.0
            self.handoff_latched_output: DepthServoOutput | None = None
            self.handoff_latched_since_s: float | None = None
            if args.whole_body == "casadi":
                if (
                    args.whole_body_urdf is None
                    or args.whole_body_calibration is None
                    or args.whole_body_collision_model is None
                ):
                    self.whole_body_error = (
                        "CasADi whole-body controller requires URDF, calibration, "
                        "and collision model"
                    )
                else:
                    try:
                        self.whole_body = WholeBodyRuntimeController(
                            urdf_path=args.whole_body_urdf,
                            calibration_path=args.whole_body_calibration,
                            collision_model_path=args.whole_body_collision_model,
                            desired_standoff_m=settings.desired_depth_m,
                        )
                    except Exception as error:
                        self.whole_body_error = f"whole-body initialization failed: {error}"
            self.last_output = self.core.tick(now_s=time.monotonic(), tracking=False)
            self.last_trace_phase: str | None = None
            self.last_trace_s = 0.0
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            # MUST match the EdgeTAM publisher's ``tracking_state_qos``
            # (ros2/z_manip_edgetam/z_manip_edgetam/node.py).  A VOLATILE
            # reader against a TRANSIENT_LOCAL writer still connects -- QoS
            # durability is request<=offered -- it just silently forfeits the
            # latched sample, which is the one a freshly started servo needs.
            # The VLM bridge already subscribes latched; this servo was the
            # only reader of this topic that did not.
            tracking_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.publisher = self.create_publisher(TwistStamped, args.velocity_topic, 1)
            self.posture_intent_publisher = self.create_publisher(
                String,
                "/z_manip/reactive/posture_intent",
                qos,
            )
            self.arm_view_intent_publisher = self.create_publisher(
                String,
                "/z_manip/reactive/arm_view_intent",
                qos,
            )
            self.create_subscription(PointCloud2, args.target_topic, self._target, qos)
            self.create_subscription(
                Bool,
                args.tracking_topic,
                self._tracking,
                tracking_qos,
            )
            self.create_subscription(
                String,
                "/go2w/posture_state",
                self._posture_state,
                qos,
            )
            self.create_subscription(
                String,
                "/z_manip/reactive/arm_view_status",
                self._arm_view_state,
                qos,
            )
            self.create_subscription(
                String,
                "/z_manip/reactive/ik_probe",
                self._ik_probe_state_message,
                qos,
            )
            self.create_timer(1.0 / args.rate_hz, self._tick)
            self._write_status(ServoPhase.STARTING.value)

        @staticmethod
        def _matrix(transform_stamped: Any) -> np.ndarray:
            transform = transform_stamped.transform
            return _rigid_transform_matrix(
                (
                    float(transform.translation.x),
                    float(transform.translation.y),
                    float(transform.translation.z),
                ),
                (
                    float(transform.rotation.x),
                    float(transform.rotation.y),
                    float(transform.rotation.z),
                    float(transform.rotation.w),
                ),
            )

        @staticmethod
        def _stamp_ns(transform_stamped: Any) -> int:
            stamp = transform_stamped.header.stamp
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        def _target_transforms(
            self,
            *,
            source_frame: str,
            source_stamp: Any,
        ) -> tuple[np.ndarray, np.ndarray]:
            if not source_frame:
                raise ValueError("target point cloud has an empty frame_id")
            runtime_error: Exception | None = None
            # The combined robot frames are intentionally reconstructed by
            # the subscribe-only observer on this deployment. Prefer that
            # fresh local artifact over waiting for two TF timeouts on every
            # camera callback; TF remains a valid fallback for simulations or
            # a future robot_state_publisher deployment.
            if args.runtime_state is not None:
                try:
                    base_matrix, arm_matrix, stamp_ns = _runtime_state_transforms(
                        args.runtime_state,
                        source_frame=source_frame,
                        base_frame=settings.base_frame,
                        arm_base_frame=settings.arm_base_frame,
                        now_unix_ns=time.time_ns(),
                        max_age_s=args.runtime_transform_timeout_s,
                    )
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    runtime_error = error
                else:
                    self.last_transform_stamps_ns = {
                        settings.base_frame: stamp_ns,
                        settings.arm_base_frame: stamp_ns,
                    }
                    self.last_transform_success_s = time.monotonic()
                    self.last_transform_source = "runtime_observer_kinematics"
                    self.last_transform_error = None
                    return base_matrix, arm_matrix
            query_time = Time.from_msg(source_stamp)
            # Sized by TF_LOOKUP_TIMEOUT_S, NOT by the geometry staleness
            # budget: this is a blocking wait taken TWICE on the single
            # -threaded executor that also runs the control tick, so it must
            # not grow just because a freshness comparison was relaxed.
            timeout = Duration(
                seconds=min(
                    settings.geometry_staleness_timeout_s,
                    TF_LOOKUP_TIMEOUT_S,
                ),
            )
            try:
                base = self.tf_buffer.lookup_transform(
                    settings.base_frame,
                    source_frame,
                    query_time,
                    timeout=timeout,
                )
                arm = self.tf_buffer.lookup_transform(
                    settings.arm_base_frame,
                    source_frame,
                    query_time,
                    timeout=timeout,
                )
                self.last_transform_stamps_ns = {
                    settings.base_frame: self._stamp_ns(base),
                    settings.arm_base_frame: self._stamp_ns(arm),
                }
                self.last_transform_success_s = time.monotonic()
                self.last_transform_source = "tf2"
                self.last_transform_error = None
                return self._matrix(base), self._matrix(arm)
            except TransformException as tf_error:
                raise ValueError(
                    f"runtime model failed ({runtime_error}); TF lookup failed ({tf_error})",
                ) from tf_error

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
            source_frame = str(message.header.frame_id or "")
            transform_error: str | None = None
            T_base_camera: np.ndarray | None = None
            T_arm_camera: np.ndarray | None = None
            try:
                T_base_camera, T_arm_camera = self._target_transforms(
                    source_frame=source_frame,
                    source_stamp=message.header.stamp,
                )
            except (TransformException, ValueError) as error:
                transform_error = (
                    f"TF {source_frame or '<empty>'} -> "
                    f"({settings.base_frame}, {settings.arm_base_frame}) unavailable: {error}"
                )
                self.last_transform_error = transform_error
            # Receipt freshness alone cannot see network queuing: judge the
            # data by its capture stamp, minus the measured PC/NUC NTP skew
            # allowance, so a congested link degrades to hold/stop instead of
            # steering on an old world.
            capture_stamp_s = (
                float(message.header.stamp.sec)
                + float(message.header.stamp.nanosec) * 1e-9
            )
            capture_age_s = max(
                0.0,
                time.time() - capture_stamp_s - CLOCK_SKEW_ALLOWANCE_S,
            )
            accepted = self.core.observe_target(
                x_m=statistics.median(xs),
                y_m=statistics.median(ys),
                z_m=statistics.median(zs),
                stamp_s=time.monotonic(),
                T_base_camera=T_base_camera,
                T_arm_camera=T_arm_camera,
                transform_error=transform_error,
                capture_age_s=capture_age_s,
            )
            if accepted:
                self.last_source_frame = source_frame or None
                new_source_stamp_ns = (
                    int(message.header.stamp.sec) * 1_000_000_000
                    + int(message.header.stamp.nanosec)
                )
                if new_source_stamp_ns != self.last_source_stamp_ns:
                    arrival_s = time.monotonic()
                    self.bundle_count += 1
                    # A bundle that lands right after a loss-stair phase spans the
                    # loss dwell; skip its interval so the stall cannot inflate
                    # the view-update damping period (only clamped in-band
                    # cadence intervals advance the EMA).
                    self.view_update_period_s = _view_period_update(
                        previous_period_s=self.view_update_period_s,
                        last_arrival_s=self.last_bundle_monotonic_s,
                        arrival_s=arrival_s,
                        loss_phase_exit=self.last_output.phase in LOSS_STAIR_PHASES,
                    )
                    self.last_bundle_monotonic_s = arrival_s
                self.last_source_stamp_ns = new_source_stamp_ns

        def _tracking(self, message: Bool) -> None:
            self.tracking_raw = bool(message.data)
            # ``std_msgs/Bool`` is unstamped, so RECEIPT is the only clock this
            # flag will ever have.  Recording it is what makes the latched
            # TRANSIENT_LOCAL subscription above safe rather than fail-open.
            self.tracking_received_s = time.monotonic()

        @property
        def tracking(self) -> bool | None:
            """The tracking flag as of NOW: UNKNOWN once it outlives its TTL.

            Read-only on purpose.  Every consumer (the control tick, the
            whole-body gate, the published status document) goes through this
            one accessor so none of them can be left reading the un-aged flag
            when the next one is added.
            """

            age_s = (
                None
                if self.tracking_received_s is None
                else max(0.0, time.monotonic() - self.tracking_received_s)
            )
            return _aged_tracking_flag(
                self.tracking_raw,
                age_s=age_s,
                ttl_s=_tracking_flag_ttl_s(
                    view_update_period_s=self.view_update_period_s,
                    target_timeout_s=settings.target_timeout_s,
                ),
            )

        def _posture_state(self, message: String) -> None:
            try:
                document = json.loads(message.data)
            except json.JSONDecodeError:
                return
            if (
                isinstance(document, dict)
                and document.get("schema") == "z_manip.go2w_posture_status.v1"
            ):
                self.posture_status = document
                self.posture_status_received_s = time.monotonic()

        def _arm_view_state(self, message: String) -> None:
            try:
                document = json.loads(message.data)
            except json.JSONDecodeError:
                return
            if (
                isinstance(document, dict)
                and document.get("schema") == ARM_STATUS_SCHEMA
            ):
                self.arm_view_status = document
                self.arm_view_status_received_s = time.monotonic()

        def _ik_probe_state_message(self, message: String) -> None:
            try:
                document = json.loads(message.data)
            except json.JSONDecodeError:
                return
            if (
                isinstance(document, dict)
                and document.get("schema") == IK_PROBE_SCHEMA
            ):
                self.ik_probe_status = document
                self.ik_probe_status_received_s = time.monotonic()

        def _posture_feedback(self) -> tuple[bool, bool, bool, str]:
            age_s = (
                math.inf
                if self.posture_status_received_s is None
                else time.monotonic() - self.posture_status_received_s
            )
            return _posture_feedback_state(
                self.posture_status,
                age_s=age_s,
            )

        def _posture_body_actionable(self) -> bool:
            # Only the epoch-stable verdict may disable body posture for the
            # whole session: UNKNOWN and TRANSIENT_FAULT are by definition
            # recoverable, so latching them here would permanently strand a
            # capable platform on the arm-only path after one blip.  Absent or
            # ambiguous evidence keeps the legacy body-posture behaviour.
            document = self.posture_status
            if not isinstance(document, dict):
                return True
            capabilities = document.get("capabilities")
            if not isinstance(capabilities, dict):
                return True
            return not (
                capabilities.get("euler") is False
                and capabilities.get("euler_state") == "UNSUPPORTED_FOR_EPOCH"
            )

        def _ik_probe_feedback(self) -> bool | None:
            age_s = (
                math.inf
                if self.ik_probe_status_received_s is None
                else time.monotonic() - self.ik_probe_status_received_s
            )
            return _ik_probe_state(self.ik_probe_status, age_s=age_s)

        def _arm_feedback(self) -> tuple[bool, bool, bool, str]:
            age_s = (
                math.inf
                if self.arm_view_status_received_s is None
                else time.monotonic() - self.arm_view_status_received_s
            )
            required_seq = (
                None
                if self.last_arm_view_intent is None
                else int(self.last_arm_view_intent["seq"])
            )
            return _arm_feedback_state(
                self.arm_view_status,
                age_s=age_s,
                required_seq=required_seq,
            )

        def _publish_posture_intent(self, *, blocked: bool = False) -> None:
            if blocked:
                return
            roll = 0.0
            yaw = 0.0
            if self.whole_body_command is not None:
                # In live mode a QP intent is forwarded only when every
                # measured-state gate used by the solve was fresh. Shadow
                # remains completely transport-free.
                if not self.whole_body_command.executable:
                    return
                transport = self.whole_body_command.document.get("transport")
                if (
                    isinstance(transport, dict)
                    and transport.get("body_enabled") is False
                ):
                    capabilities = (
                        self.posture_status.get("capabilities")
                        if isinstance(self.posture_status, dict)
                        else None
                    )
                    # UNKNOWN is not a permanent denial: forward one bounded
                    # (normally near-level) target so the NUC's sole WebRTC
                    # owner can obtain the robot's same-epoch return code.
                    # Confirmed unsupported/fault states remain blocked.
                    if not (
                        isinstance(capabilities, dict)
                        and capabilities.get("euler_state") == "UNKNOWN"
                    ):
                        return
                roll = self.whole_body_command.body_roll_target_rad
                target = (roll, self.whole_body_command.body_pitch_target_rad)
            else:
                if not self._posture_body_actionable():
                    # Defense in depth: never relay a body-posture intent the
                    # motion service rejects for the whole epoch -- each retry
                    # contends on the single WebRTC owner for nothing.
                    return
                status = self.core.reactive_status
                if status is None or status.get("phase") not in {
                    ReactivePhase.POSTURE_ADJUST.value,
                    ReactivePhase.VIEW_RECOVERY.value,
                }:
                    return
                posture = status.get("posture")
                if not isinstance(posture, dict):
                    return
                roll = float(posture.get("roll_delta_rad", 0.0))
                target = (roll, float(posture.get("pitch_delta_rad", 0.0)))
            now_s = time.monotonic()
            if (
                self.last_posture_intent == target
                and now_s - self.last_posture_intent_s < 0.25
            ):
                return
            message = String()
            message.data = json.dumps(
                {
                    "schema": "z_manip.go2w_posture_intent.v1",
                    "roll_delta_rad": roll,
                    "pitch_delta_rad": target[1],
                    "yaw_delta_rad": yaw,
                },
                separators=(",", ":"),
                allow_nan=False,
            )
            self._publish_guarded(self.posture_intent_publisher, message)
            self.last_posture_intent = target
            self.last_posture_intent_s = now_s

        def _publish_arm_view_intent(self, *, blocked: bool = False) -> None:
            command = self.whole_body_command
            if blocked or settings.mode != "live" or command is None:
                return
            transport = command.document.get("transport")
            if (
                not command.executable
                or not isinstance(transport, dict)
                or transport.get("arm_enabled") is not True
            ):
                return
            _ready, reached, _blocked, _detail = self._arm_feedback()
            # Once the optimizer asks for negligible arm motion and the NUC
            # has measured the last target, stop advancing sequence numbers.
            # This gives the handoff gate a stable acknowledged window while
            # the executor's own stale-intent behavior continues to hold pose.
            if _whole_body_arm_rate_converged(command) and reached:
                return
            sequence = self.arm_view_intent_seq
            if self.last_arm_view_intent is not None:
                previous = int(self.last_arm_view_intent["seq"])
                accepted = -1
                if isinstance(self.arm_view_status, dict):
                    try:
                        accepted = int(self.arm_view_status.get("accepted_seq", -1))
                    except (TypeError, ValueError):
                        accepted = -1
                # Retransmit a missed sequence with a fresh lease instead of
                # outrunning a 20 Hz executor with an ever-growing backlog.
                sequence = previous if accepted < previous else sequence
            now_ns = time.time_ns()
            document = _arm_view_intent_document(
                command,
                seq=sequence,
                now_unix_ns=now_ns,
                target_source_timestamp_ns=self.last_source_stamp_ns,
            )
            message = String()
            message.data = json.dumps(document, separators=(",", ":"), allow_nan=False)
            self._publish_guarded(self.arm_view_intent_publisher, message)
            self.last_arm_view_intent = document
            self.arm_view_intent_seq = max(self.arm_view_intent_seq, sequence + 1)

        def _publish_guarded(self, publisher: Any, message: Any) -> None:
            """Publish, tolerating ONLY the rcl-context teardown race.

            A stop/restart can tear the rcl context down between the
            ``rclpy.ok()`` gate and the publish itself, raising RCLError
            ("publisher's context is invalid", publisher.c:423).  That is a
            normal shutdown race, not a command fault: swallow it so teardown
            never crash-loops mid-approach.  A still-valid context means a
            genuine fault -- re-raise it rather than hiding it.

            EVERY publisher on this node must go through here.  Only the
            velocity publisher was guarded originally; the posture-intent and
            arm-view-intent publishers published raw and took the whole servo
            down during teardown (live 2026-07-28: 5 RCLError crashes in one
            session -- 3 posture, 1 arm-view, 1 velocity).  Each crash killed
            the approach ~0.4 s after spawn, so planning_control saw
            ``search_required`` with one bundle received and spent its
            reacquisition budget on wrist searches.  It presented as "detects
            the target but will not drive to it", i.e. a perception fault, and
            was nothing of the kind.
            """

            if not rclpy.ok():
                return
            try:
                publisher.publish(message)
            except Exception:  # noqa: BLE001
                if rclpy.ok():
                    raise

        def _publish(self, linear_x: float, angular_z: float) -> None:
            if not rclpy.ok():
                return
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "base_link"
            message.twist.linear.x = float(linear_x)
            message.twist.angular.z = float(angular_z)
            self._publish_guarded(self.publisher, message)

        def _write_status(self, state: str | None = None, *, running: bool = True) -> None:
            target = self.core.target
            geometry = self.core.geometry
            transform_received_s = self.core.transform_status[
                "received_monotonic_s"
            ]
            geometry_age_s = (
                None
                if transform_received_s is None
                else max(0.0, time.monotonic() - transform_received_s)
            )
            lookup_age_s = (
                None
                if self.last_transform_success_s is None
                else max(0.0, time.monotonic() - self.last_transform_success_s)
            )
            transform_fresh = (
                geometry is not None
                and geometry_age_s is not None
                and geometry_age_s <= settings.geometry_staleness_timeout_s
            )
            posture_age_s = (
                None
                if self.posture_status_received_s is None
                else max(0.0, time.monotonic() - self.posture_status_received_s)
            )
            arm_view_age_s = (
                None
                if self.arm_view_status_received_s is None
                else max(0.0, time.monotonic() - self.arm_view_status_received_s)
            )
            ik_probe_age_s = (
                None
                if self.ik_probe_status_received_s is None
                else max(0.0, time.monotonic() - self.ik_probe_status_received_s)
            )
            published_phase = state or self.last_output.phase
            document = {
                "schema": STATUS_SCHEMA,
                "running": running,
                "mode": settings.mode,
                "phase": published_phase,
                "tracking": self.tracking,
                "target": None if target is None else {
                    "x_m": target[0],
                    "y_m": target[1],
                    "z_m": target[2],
                    "frame_id": self.last_source_frame,
                },
                "geometry": (
                    asdict(geometry)
                    if geometry is not None
                    else self.core.camera_geometry
                ),
                "reactive": _abandoned_reactive_status(
                    self.core.reactive_status,
                    phase=published_phase,
                ),
                "transforms": {
                    "valid": transform_fresh,
                    "error": (
                        self.last_transform_error
                        or self.core.transform_status["error"]
                        or (
                            "synchronized transforms are stale"
                            if not transform_fresh else None
                        )
                    ),
                    "geometry_age_s": geometry_age_s,
                    "lookup_age_s": lookup_age_s,
                    "source_frame": self.last_source_frame,
                    "base_frame": settings.base_frame,
                    "arm_base_frame": settings.arm_base_frame,
                    "stamps_ns": self.last_transform_stamps_ns,
                    "source": self.last_transform_source,
                },
                "posture_status": {
                    "age_s": posture_age_s,
                    "document": self.posture_status,
                    "last_intent": None if self.last_posture_intent is None else {
                        "roll_delta_rad": self.last_posture_intent[0],
                        "pitch_delta_rad": self.last_posture_intent[1],
                    },
                },
                "arm_view_status": {
                    "age_s": arm_view_age_s,
                    "document": self.arm_view_status,
                    "last_intent": self.last_arm_view_intent,
                },
                "ik_probe": {
                    "age_s": ik_probe_age_s,
                    "feasible": self._ik_probe_feedback(),
                    "document": self.ik_probe_status,
                },
                "whole_body": (
                    {
                        "enabled": args.whole_body == "casadi",
                        "ready": self.whole_body is not None,
                        "error": self.whole_body_error,
                        "command": None,
                    }
                    if self.whole_body_command is None
                    else {
                        "enabled": True,
                        "ready": True,
                        "error": self.whole_body_error,
                        "command": self.whole_body_command.document,
                    }
                ),
                "source_stamp_ns": self.last_source_stamp_ns,
                "output": asdict(self.last_output),
                "filter": self.core.filter_stats,
                "trace_file": None if args.trace_file is None else str(args.trace_file),
                "pid": os.getpid(),
                "updated_unix_ns": time.time_ns(),
            }
            _atomic_json(args.status_file, document)
            now_s = time.monotonic()
            # A parked/terminal servo changes slowly (1 Hz floor); active motion
            # is sampled at 5 Hz so diagnostics keep their true cadence instead
            # of a >=1 s/row throttle manufacturing a phantom 1 Hz rate.
            trace_terminal = (
                self.handoff_latched_output is not None or self.last_output.done
            )
            if args.trace_file is not None and (
                self.last_output.phase != self.last_trace_phase
                or now_s - self.last_trace_s
                >= _trace_min_interval_s(terminal=trace_terminal)
            ):
                _append_jsonl(args.trace_file, _trace_row(
                    document,
                    view_update_period_s=self.view_update_period_s,
                    bundle_count=self.bundle_count,
                ))
                self.last_trace_phase = self.last_output.phase
                self.last_trace_s = now_s

        def _whole_body_output(
            self,
            fallback: DepthServoOutput,
            *,
            posture_settled: bool,
        ) -> DepthServoOutput:
            self.whole_body_command = None
            # The reactive layer deliberately hands control to the close-range
            # planner before the wrist D435 enters its blind zone.  Do not run
            # another whole-body posture/QP step here: that would extend the
            # camera after the handoff decision and invalidate the last usable
            # RGB-D capture.
            if (
                fallback.needs_ik_probe
                or fallback.reactive_phase
                in {
                    ReactivePhase.HANDOFF_SETTLE.value,
                    ReactivePhase.HANDOFF_PROBE.value,
                    ReactivePhase.HANDOFF_READY.value,
                }
            ):
                self.whole_body_handoff_settle_cycles = 0
                if self.whole_body is not None:
                    self.whole_body.reset()
                return fallback
            if self.whole_body is None:
                if args.whole_body == "casadi":
                    return DepthServoOutput(
                        phase=ServoPhase.WHOLE_BODY_BLOCKED.value,
                        proposed_linear_x=0.0,
                        proposed_angular_z=0.0,
                        published_linear_x=0.0,
                        published_angular_z=0.0,
                        depth_error_m=fallback.depth_error_m,
                        yaw_error_rad=fallback.yaw_error_rad,
                        target_age_s=fallback.target_age_s,
                        reason=self.whole_body_error or "whole-body controller unavailable",
                        reactive_phase=fallback.reactive_phase,
                    )
                return fallback
            geometry = self.core.geometry
            target = self.core.target
            if (
                geometry is None
                or target is None
                or self.tracking is not True
                # The loss stair MUST win over the whole-body branch: when the
                # core freezes on stale data (tracking_hold) or demands
                # recovery, solving with the retained multi-second-old target
                # kept the arm swinging through a 6s wifi stall (live
                # 2026-07-24: raw_y swept +/-0.67m sinusoidally while the
                # source stamp froze -- the arm was chasing its own motion).
                or fallback.phase in LOSS_STAIR_PHASES
                # Belt and braces: never solve on a target older than the
                # capture-freshness budget, whatever phase the core reports.
                or (
                    fallback.target_age_s is not None
                    and math.isfinite(fallback.target_age_s)
                    and fallback.target_age_s
                    > settings.max_target_capture_age_s
                )
            ):
                self.whole_body_handoff_settle_cycles = 0
                return fallback
            inside_handoff = (
                geometry.base_planar_distance_m <= settings.handoff_depth_m
                and self.core.side_lateral_error_m is not None
                and abs(self.core.side_lateral_error_m)
                <= settings.side_handoff_tolerance_m
            )
            if args.runtime_state is None:
                self.whole_body_error = "whole-body controller requires runtime state"
                return fallback
            try:
                command = self.whole_body.solve(
                    camera_target_xyz_m=target,
                    posture_status=self.posture_status,
                    arm_view_status=self.arm_view_status,
                    runtime_state_path=args.runtime_state,
                    mode=settings.mode,
                    freeze_base=inside_handoff,
                    desired_target_lateral_in_body_m=(
                        self.core.desired_target_lateral_m
                    ),
                    view_update_period_s=self.view_update_period_s,
                )
            except Exception as error:
                self.whole_body_error = f"whole-body solve failed: {error}"
                return DepthServoOutput(
                    phase=ServoPhase.WHOLE_BODY_BLOCKED.value,
                    proposed_linear_x=0.0,
                    proposed_angular_z=0.0,
                    published_linear_x=0.0,
                    published_angular_z=0.0,
                    depth_error_m=geometry.base_planar_distance_m - settings.desired_depth_m,
                    yaw_error_rad=geometry.base_bearing_rad,
                    target_age_s=fallback.target_age_s,
                    reason=self.whole_body_error,
                    reactive_phase=WHOLE_BODY_REACTIVE_PHASE,
                )
            self.whole_body_command = command
            self.whole_body_error = None
            posture_rate_converged = _whole_body_posture_rate_converged(command)
            arm_rate_converged = _whole_body_arm_rate_converged(command)
            arm_ready, arm_reached, arm_blocked, arm_detail = self._arm_feedback()
            if inside_handoff:
                if (
                    command.executable
                    and posture_settled
                    and posture_rate_converged
                    and arm_ready
                    and arm_reached
                    and arm_rate_converged
                    and not arm_blocked
                ):
                    self.whole_body_handoff_settle_cycles += 1
                else:
                    self.whole_body_handoff_settle_cycles = 0

                if self.whole_body_handoff_settle_cycles >= POSTURE_SETTLE_TICKS:
                    # Preserve the reactive controller's explicit IK probe.
                    # Distance and posture alone are not sufficient handoff
                    # evidence for manipulation.
                    if fallback.needs_ik_probe:
                        return DepthServoOutput(
                            **{
                                **asdict(fallback),
                                "reason": "body loop converged; waiting for close-range IK probe",
                            },
                        )
                    if fallback.done:
                        self.whole_body.reset()
                        return DepthServoOutput(
                            phase=ServoPhase.REACHED.value,
                            proposed_linear_x=0.0,
                            proposed_angular_z=0.0,
                            published_linear_x=0.0,
                            published_angular_z=0.0,
                            depth_error_m=(
                                geometry.base_planar_distance_m
                                - settings.desired_depth_m
                            ),
                            yaw_error_rad=geometry.base_bearing_rad,
                            target_age_s=fallback.target_age_s,
                            done=True,
                            reason=(
                                "measured Euler and PiPER view loops plus close-range "
                                "IK handoff converged"
                            ),
                            reactive_phase=ServoPhase.HANDOFF_READY.value,
                        )

                return DepthServoOutput(
                    phase=(
                        ServoPhase.WHOLE_BODY_POSTURE.value
                        if command.executable
                        else ServoPhase.WHOLE_BODY_SHADOW.value
                    ),
                    proposed_linear_x=0.0,
                    proposed_angular_z=0.0,
                    published_linear_x=0.0,
                    published_angular_z=0.0,
                    depth_error_m=(
                        geometry.base_planar_distance_m - settings.desired_depth_m
                    ),
                    yaw_error_rad=geometry.base_bearing_rad,
                    target_age_s=fallback.target_age_s,
                    done=False,
                    reason=(
                        "base parked; closing measured Euler and PiPER view loops "
                        f"({self.whole_body_handoff_settle_cycles}/"
                        f"{POSTURE_SETTLE_TICKS} stable ticks; {arm_detail})"
                        if command.executable
                        else "whole-body posture intent gated by stale measured state"
                    ),
                    reactive_phase=ServoPhase.POSTURE_ADJUST.value,
                )

            self.whole_body_handoff_settle_cycles = 0
            linear = float(np.clip(command.base_forward_mps, 0.0, settings.max_forward_mps))
            # Maintain Go2W's gait above its observed dead zone while outside
            # the handoff; CasADi still chooses whether forward motion helps.
            if linear > 1e-3:
                linear = max(linear, settings.min_forward_mps)
            yaw_target = float(np.clip(
                command.base_yaw_rps,
                -settings.max_yaw_rps,
                settings.max_yaw_rps,
            ))
            side_error_m = self.core.side_lateral_error_m
            steering_bearing = (
                geometry.base_bearing_rad
                if side_error_m is None
                else math.atan2(side_error_m, max(geometry.base_xyz_m[0], 0.05))
            )
            # A stepping Go2W produces large single-frame bearing spikes.  The
            # QP may legitimately alternate signs while the chassis rocks,
            # but forwarding those reversals makes the robot pivot in place
            # and repeatedly loses the target.  Use the measured side-work
            # error, a deliberately broad deadband, and a per-tick slew limit.
            if abs(steering_bearing) <= settings.yaw_deadband_rad:
                yaw_target = 0.0
            yaw_delta = float(np.clip(
                yaw_target - self.last_conditioned_yaw_rps,
                -settings.max_yaw_step_rps,
                settings.max_yaw_step_rps,
            ))
            yaw = self.last_conditioned_yaw_rps + yaw_delta
            if abs(yaw) < 1e-6:
                yaw = 0.0
            self.last_conditioned_yaw_rps = yaw
            executable = command.executable
            return DepthServoOutput(
                phase=(
                    ServoPhase.WHOLE_BODY_APPROACH.value
                    if executable
                    else ServoPhase.WHOLE_BODY_SHADOW.value
                ),
                proposed_linear_x=linear,
                proposed_angular_z=yaw,
                published_linear_x=linear if executable else 0.0,
                published_angular_z=yaw if executable else 0.0,
                depth_error_m=geometry.base_planar_distance_m - settings.desired_depth_m,
                yaw_error_rad=steering_bearing,
                target_age_s=fallback.target_age_s,
                reason=(
                    "Pinocchio/CasADi coupled base-body intent"
                    if executable
                    else "Pinocchio/CasADi shadow intent; measured live gates not satisfied"
                ),
                reactive_phase=WHOLE_BODY_REACTIVE_PHASE,
            )

        def _tick(self) -> None:
            # A timer tick after the stop was latched or the rcl context began
            # tearing down must no-op: publishing here would crash-loop teardown
            # instead of settling on the transport watchdog stop.
            if _tick_should_skip(
                stop_requested=self.stop_event.is_set(),
                ros_ok=rclpy.ok(),
            ):
                return
            # Handoff is one-way for this process.  Keep publishing a hard zero
            # and preserve the terminal status until the 5 Hz supervisor has
            # stopped this launcher and opened the fresh grasp transaction.
            if self.handoff_latched_output is not None:
                # The latch has an explicit, bounded exit; see
                # ``_handoff_latch_output``.  The exit is always a stop.
                self.handoff_latched_output = _handoff_latch_output(
                    self.handoff_latched_output,
                    latched_since_s=self.handoff_latched_since_s,
                    now_s=time.monotonic(),
                )
                self.last_output = self.handoff_latched_output
                if settings.mode == "live":
                    self._publish(0.0, 0.0)
                self._write_status()
                return
            settled, blocked, shadow_verified, detail = self._posture_feedback()
            # Feed the read-only close-range IK verdict before the FSM step so a
            # resident planning runner can certify HANDOFF_READY on the same
            # tick; absent a producer the probe is None and the controller stays
            # fail-closed in HANDOFF_PROBE.
            self.core.set_ik_probe_result(self._ik_probe_feedback())
            fallback = self.core.tick(
                now_s=time.monotonic(),
                tracking=self.tracking,
                body_settled=settled,
                posture_blocked=blocked,
                posture_shadow_verified=shadow_verified,
                posture_detail=detail,
                body_posture_actionable=self._posture_body_actionable(),
            )
            candidate = self._whole_body_output(
                fallback,
                posture_settled=settled,
            )
            handoff = _latch_handoff_output(None, candidate)
            if handoff is not None:
                self.handoff_latched_output = handoff
                self.handoff_latched_since_s = time.monotonic()
                self.last_output = handoff
                if settings.mode == "live":
                    self._publish(0.0, 0.0)
                self._write_status()
                return
            self.last_output = candidate
            self._publish_posture_intent(blocked=blocked)
            _arm_ready, _arm_reached, arm_blocked, _arm_detail = self._arm_feedback()
            self._publish_arm_view_intent(blocked=arm_blocked)
            if settings.mode == "live":
                self._publish(
                    self.last_output.published_linear_x,
                    self.last_output.published_angular_z,
                )
            self._write_status()

        def stop(self, phase: str = ServoPhase.STOPPED.value) -> None:
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

    def request_stop(_signum: int, _frame: object) -> None:
        """SIGNAL-HANDLER CONTRACT: SET FLAGS.  DO NOTHING ELSE.  EVER.

        This handler used to publish three zero Twists, write the final status
        document and call ``rclpy.shutdown()`` -- all of it re-entering rcl
        from a signal delivered while ``rclpy.spin()`` was on the stack and
        mid-way through its own use of those same objects.  One recorded log
        carries nine tracebacks from that single teardown: 5x RCLError from
        publisher.c:423 (publish on a context being destroyed), 2x
        InvalidHandle, 1x ``AttributeError: 'NoneType' object has no attribute
        'trigger'`` (a guard condition freed underneath the executor), and 1x
        FileNotFoundError (the status write racing its own scratch file).

        Commit 6a3f75d wrapped the publishes in guards.  That removed the
        printed text and left the re-entrancy: the handler was still doing
        work on rcl objects the spin thread owned.  No try/except can fix
        that, because the bug is not the exception -- it is the work.  The
        publishes, the final status write, ``destroy_node`` and ``shutdown``
        now all happen below, on the main thread, after the loop has returned
        and nothing is inside rcl.
        """

        stopped.set()
        # Read by ``_tick_should_skip`` so an already-queued timer callback
        # no-ops instead of publishing into a teardown.
        node.stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        # NOT ``rclpy.spin(node)``: that blocks in an unbounded wait, and the
        # two ``signal.signal`` calls above have displaced the C-level handler
        # rclpy.init() installed to break that wait.  A handler that only sets
        # a flag therefore needs a loop that only waits in bounded steps.
        while rclpy.ok() and not stopped.is_set():
            executor.spin_once(timeout_sec=SHUTDOWN_POLL_INTERVAL_S)
    finally:
        # Main thread; the loop has returned; the context is still valid.
        node.stop(_shutdown_phase(stop_requested=stopped.is_set()))
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def main() -> int:
    return _run_ros(_arguments())


if __name__ == "__main__":
    raise SystemExit(main())
