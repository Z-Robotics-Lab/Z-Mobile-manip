#!/usr/bin/env python3
"""Execute one complete Home-planned PiPER pick-and-return transaction.

The arm connection remains open for the entire sequence so torque is never
released between pregrasp, approach/close, lift, place-back, retreat and Home.
Every outbound edge is either the collision-checked planned path or the exact
reverse of that same checked edge.  The object is released at the original
grasp pose before the reverse approach and transit return Home.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

import piper_staged_grasp_executor as stage_executor


# The mobile find->approach->handoff workflow may need more than 30 seconds
# between its close-range capture and execution while IK/path planning runs.
# This remains bounded by the staged executor's hard cap and both the source
# timestamp and planning-report mtime are validated before transport opens.
MAX_FULL_SOURCE_AGE_S = 120.0
MAX_CONTINUATION_AGE_S = 12 * 60 * 60.0
WORKFLOW_PHASES = ("full", "pick-hold", "return-home-holding", "place-back")


def _write_executor_start_receipt(
    receipt_dir: Path,
    *,
    artifact: stage_executor.PlanningArtifact,
    workflow_phase: str,
    planning_session_id: str,
    started_unix_ns: int,
    started_monotonic_ns: int,
) -> dict[str, object]:
    """Persist proof that the real PiPER transport opened, before motion.

    This receipt is deliberately written only after ``connect_real_arm`` has
    returned successfully.  Starting the SSH wrapper or its worker thread is
    not executor-start evidence.
    """

    document: dict[str, object] = {
        "schema": "z_manip.piper_executor_start_receipt.v1",
        "event": "transport_opened",
        "artifact_id": artifact.artifact_id,
        "planning_report_sha256": artifact.report_sha256,
        "planned_grasp_sha256": artifact.npz_sha256,
        "workflow_phase": workflow_phase,
        "planning_session_id": planning_session_id,
        "executor_started_unix_ns": started_unix_ns,
        "executor_started_monotonic_ns": started_monotonic_ns,
        "monotonic_clock_domain": "nuc_piper_executor_process",
        "transport": "piper_can",
        "transport_opened": True,
        "commands_sent": 0,
        "motion_started": False,
    }
    stage_executor.atomic_write_json(
        receipt_dir / "executor-start-receipt.json",
        document,
    )
    return document


def full_token(artifact_id: str) -> str:
    artifact_id = stage_executor._require_sha256(artifact_id, "artifact_id")
    return f"PIPER-FULL-{artifact_id[:16]}"


def _receipt(
    *,
    artifact: stage_executor.PlanningArtifact,
    stage: str,
    prior: stage_executor.PriorReceipt | None,
    started_ns: int,
    final_joints: np.ndarray,
    gripper: stage_executor.GripperFeedback | None,
    receipt_dir: Path,
    approach_execution: str = "streamed",
) -> stage_executor.PriorReceipt:
    document = stage_executor.build_receipt(
        artifact=artifact,
        stage=stage,
        prior=prior,
        started_unix_ns=started_ns,
        finished_unix_ns=time.time_ns(),
        final_joints_rad=final_joints,
        gripper=gripper,
        approach_execution=approach_execution,
    )
    path = receipt_dir / f"{stage}-receipt.json"
    stage_executor.atomic_write_json(path, document)
    return stage_executor.load_prior_receipt(
        path,
        expected_stage=stage,
        now_ns=time.time_ns(),
    )


def _workflow_state(
    receipt_dir: Path,
    *,
    artifact: stage_executor.PlanningArtifact,
    phase: str,
    final_joints: np.ndarray,
    holding_object: bool,
    at_home: bool,
    planning_session_id: str,
    prior_workflow_sha256: str | None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "z_manip.piper_grasp_workflow_state.v1",
        "artifact_id": artifact.artifact_id,
        "planning_session_id": planning_session_id,
        "prior_workflow_sha256": prior_workflow_sha256,
        "phase": phase,
        "holding_object": holding_object,
        "at_home": at_home,
        "final_joints_rad": [float(value) for value in final_joints],
        "finished_unix_ns": time.time_ns(),
    }
    stage_executor.atomic_write_json(receipt_dir / "workflow-state.json", document)
    return document


def _load_workflow_state(
    prior_receipt_dir: Path | None,
    *,
    artifact: stage_executor.PlanningArtifact,
    expected_phase: str | tuple[str, ...],
    planning_session_id: str,
) -> dict[str, object]:
    expected_phases = (expected_phase,) if isinstance(expected_phase, str) else expected_phase
    expected_label = " or ".join(expected_phases)
    if prior_receipt_dir is None:
        raise stage_executor.SafetyError(f"{expected_label} requires a prior workflow receipt")
    path = prior_receipt_dir / "workflow-state.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise stage_executor.SafetyError(f"cannot load prior workflow state: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != "z_manip.piper_grasp_workflow_state.v1":
        raise stage_executor.SafetyError("unsupported prior workflow state")
    if document.get("artifact_id") != artifact.artifact_id:
        raise stage_executor.SafetyError("prior workflow state belongs to a different planning artifact")
    if document.get("planning_session_id") != planning_session_id:
        raise stage_executor.SafetyError("prior workflow state belongs to a different planning session")
    if document.get("phase") not in expected_phases:
        raise stage_executor.SafetyError(
            f"workflow phase must be {expected_label}, got {document.get('phase')}",
        )
    if document.get("holding_object") is not True:
        raise stage_executor.SafetyError("workflow continuation requires a held object")
    finished = document.get("finished_unix_ns")
    if isinstance(finished, bool) or not isinstance(finished, int):
        raise stage_executor.SafetyError("prior workflow state has no valid completion timestamp")
    age_s = stage_executor.age_seconds(finished, time.time_ns())
    if age_s > MAX_CONTINUATION_AGE_S:
        raise stage_executor.SafetyError("prior workflow state is stale")
    document["_receipt_sha256"] = stage_executor.sha256_bytes(path.read_bytes())
    return document


def _open_gripper(
    robot: object,
    effector: object,
    guard: stage_executor.CommandGuard,
    *,
    gripper_force_n: float,
) -> None:
    stage_executor.enter_can_joint_control(robot, guard)
    baseline = stage_executor.normalize_gripper_feedback(effector.get_gripper_status())
    guard.mark_before_command()
    effector.move_gripper_m(value=stage_executor.OPEN_APERTURE_M, force=gripper_force_n)
    opened = stage_executor.wait_for_gripper(
        effector,
        lambda sample: sample.aperture_m >= 0.060,
        after_timestamp=baseline.timestamp,
        timeout_s=3.0,
        static_accept_after_s=0.5,
    )
    stage_executor.verify_gripper_ready(opened)


def _verify_holding_object(
    robot: object,
    effector: object,
    guard: stage_executor.CommandGuard,
    artifact: stage_executor.PlanningArtifact,
    *,
    gripper_force_n: float,
) -> None:
    """Re-establish enable without changing aperture, then prove nonempty grip."""
    # Hold all six joints before any gripper recovery command.  PiPER exposes
    # arm and gripper enable separately; recovering the gripper must never
    # transiently leave the loaded arm without joint control.
    stage_executor.enter_can_joint_control(robot, guard)
    feedback = stage_executor.restore_gripper_enable_at_current_aperture(
        effector,
        guard,
        force_n=gripper_force_n,
    )
    # Mirror the pick-path calls: force alone cannot prove a hold (a true soft
    # hold measured -0.114N, below the 0.20N default this call silently
    # inherited, failing every carry/place-back continuation leg), so rely on
    # the stall-gap discriminator against the commanded close width instead.
    stage_executor.verify_nonempty_grasp(
        feedback,
        artifact.required_width_m,
        minimum_force_n=0.0,
        commanded_close_target_m=stage_executor.close_target_m(
            artifact.required_width_m,
        ),
    )


def _place_back_from_home_route(
    evidence: stage_executor.HoldingReturnEvidence,
) -> str | None:
    """Which corridor can set the object down again once it is Home?

    Both routes begin with the loaded run back out along the transit, so
    ``return_transit`` is mandatory for either.  After that:

    ``approach``  descend pregrasp -> grasp directly.  That edge sweeps exactly
                  the states ``reverse_approach`` attests (a joint-space edge
                  covers the same configurations whichever way it is driven).
    ``carry``     retrace the direct carry edge back to the lift top, then
                  lower down the lift column.  Those are the states ``carry``
                  and ``lift`` attest.

    Both end at the grasp pose with the fingers still shut, which is where the
    release happens.  ``None`` means the object cannot be put down at Home at
    all, which is the one state the workflow must never enter: there is no
    gripper-open endpoint anywhere in the operator's control surface, and
    ``piper_reverse_home_recovery`` only opens the gripper when the arm matches
    an ``approach``/``lift`` waypoint -- at Home it matches ``transit`` and
    keeps holding.  An arm that reaches Home without this is torqued shut on
    the object with no software action able to release it.
    """

    if not evidence.return_transit_valid:
        return None
    if evidence.segments.get("reverse_approach"):
        return "approach"
    if evidence.carry_valid and evidence.segments.get("lift"):
        return "carry"
    return None


def _checked_holding_return(
    artifact: stage_executor.PlanningArtifact,
) -> np.ndarray | None:
    """Authorize the loaded return Home and choose which corridor to drive.

    The outbound transit and approach are collision-checked with an EMPTY
    gripper: the object is an obstacle to avoid, not a payload bolted to the
    tool.  Driving those arrays backwards while HOLDING the object sweeps a
    strictly larger volume -- the object itself, out beyond the fingers --
    across the Go2W platform on the way back to Home.  Reversing a path is not
    revalidating it, so that return was being commanded on evidence collected
    for a different robot configuration.

    Nothing here is executed on unproven geometry:

    * the run Home (pregrasp -> Home) is driven on every route, so its
      attached-object verdict is mandatory;
    * the direct carry is preferred when it validated, because it also deletes
      the put-down;
    * the reverse-lift + reverse-approach replay is retained as the fallback,
      since it is the known-good recovery corridor -- but only once it has
      passed the LOADED check, never as an unchecked default;
    * if neither corridor is proven, no motion is commanded at all and the
      object stays where it is, for the operator's existing recovery actions;
    * and the arm never commits to Home unless a place-back corridor is proven
      TOO.  Going Home is one half of a round trip: the durable phase becomes
      ``holding_at_home``, from which the only way to release the object is
      Place Back.  Authorizing the outbound half on evidence the inbound half
      will reject strands the object held at Home with no software action able
      to set it down.

    Returns the carry polyline to drive, or ``None`` to drive the checked
    legacy replay.
    """

    evidence = stage_executor.holding_return_evidence(artifact)
    if not evidence.return_transit_valid:
        raise stage_executor.SafetyError(
            "the run Home was rejected with the object attached; refusing to "
            "carry it back along an unvalidated corridor",
        )
    if not evidence.carry_valid and not evidence.legacy_corridor_valid:
        raise stage_executor.SafetyError(
            "no return corridor is valid with the object attached "
            f"(carry rejected: {evidence.carry_rejection_reason or 'collision'}); "
            "leaving the object held for operator recovery",
        )
    if _place_back_from_home_route(evidence) is None:
        raise stage_executor.SafetyError(
            "no place-back corridor from Home is valid with the object "
            "attached; refusing to carry it to a pose it could not be set "
            "down from (leaving the object held for operator recovery)",
        )
    if evidence.carry_valid:
        assert evidence.carry_raw is not None
        return stage_executor.coalesce_collinear_execution_path(evidence.carry_raw)
    return None


def _checked_place_back_from_home(
    artifact: stage_executor.PlanningArtifact,
) -> np.ndarray | None:
    """Authorize and choose the place-back legs driven with the object held.

    From ``holding_at_home`` the arm runs back out along the transit and then
    lowers the object onto the grasp pose before it releases.  Every one of
    those legs is loaded, so every one needs an attached-object verdict.  Two
    descents are authorized -- see ``_place_back_from_home_route`` -- and
    whichever is proven is driven, so the evidence that permits going Home is
    exactly the evidence that permits coming back.

    From ``holding_at_lift`` the only loaded motion is the reverse lift, which
    the planner already checks with the payload attached; that limb is
    deliberately not gated here so a failed Return Home can always put the
    object back down.

    Returns the carry polyline to retrace (lift-top descent), or ``None`` to
    descend the approach directly.
    """

    evidence = stage_executor.holding_return_evidence(artifact)
    route = _place_back_from_home_route(evidence)
    if route is None:
        raise stage_executor.SafetyError(
            "place-back from Home requires a descent corridor that is valid "
            "with the object attached",
        )
    if route == "approach":
        return None
    assert evidence.carry_raw is not None
    return stage_executor.coalesce_collinear_execution_path(evidence.carry_raw)


def _require_carry_start(
    prior_state: dict[str, object],
    carry: np.ndarray,
) -> None:
    """Prove the arm physically ended the lift where the carry starts."""

    recorded = np.asarray(prior_state["final_joints_rad"], dtype=float)
    error = float(np.max(np.abs(recorded - np.asarray(carry, dtype=float)[0])))
    if error > stage_executor.DEFAULT_START_TOLERANCE_RAD:
        raise stage_executor.SafetyError(
            "recorded lift top does not match the planned carry start",
        )


def execute_full_grasp(
    robot: object,
    effector: object,
    artifact: stage_executor.PlanningArtifact,
    *,
    receipt_dir: Path,
    speed_percent: int,
    segment_timeout_s: float,
    gripper_force_n: float,
    lift_hold_s: float,
    executor_start: tuple[int, int] | None = None,
    planning_session_id: str = "",
    direct_approach: bool = True,
) -> dict[str, object]:
    """Run the complete pick, visible lift, place-back and checked return."""

    receipt_dir.mkdir(parents=True, exist_ok=False)
    if executor_start is not None:
        _write_executor_start_receipt(
            receipt_dir,
            artifact=artifact,
            workflow_phase="full",
            planning_session_id=planning_session_id,
            started_unix_ns=executor_start[0],
            started_monotonic_ns=executor_start[1],
        )
    pregrasp_path = stage_executor.validate_stage_context(
        artifact,
        "pregrasp",
        None,
    )
    started = time.time_ns()
    final, gripper = stage_executor.execute_stage(
        robot,
        effector,
        artifact,
        "pregrasp",
        pregrasp_path,
        speed_percent=speed_percent,
        segment_timeout_s=segment_timeout_s,
        gripper_force_n=gripper_force_n,
    )
    pregrasp_receipt = _receipt(
        artifact=artifact,
        stage="pregrasp",
        prior=None,
        started_ns=started,
        final_joints=final,
        gripper=gripper,
        receipt_dir=receipt_dir,
    )

    approach_path = stage_executor.validate_stage_context(
        artifact,
        "approach_close",
        pregrasp_receipt,
    )
    started = time.time_ns()
    final, gripper = stage_executor.execute_stage(
        robot,
        effector,
        artifact,
        "approach_close",
        approach_path,
        speed_percent=speed_percent,
        segment_timeout_s=segment_timeout_s,
        gripper_force_n=gripper_force_n,
        direct_approach=direct_approach,
    )
    approach_receipt = _receipt(
        artifact=artifact,
        stage="approach_close",
        prior=pregrasp_receipt,
        started_ns=started,
        final_joints=final,
        gripper=gripper,
        receipt_dir=receipt_dir,
        approach_execution="streamed" if direct_approach else "stepped",
    )

    lift_path = stage_executor.validate_stage_context(
        artifact,
        "lift",
        approach_receipt,
    )
    started = time.time_ns()
    final, gripper = stage_executor.execute_stage(
        robot,
        effector,
        artifact,
        "lift",
        lift_path,
        speed_percent=speed_percent,
        segment_timeout_s=segment_timeout_s,
        gripper_force_n=gripper_force_n,
    )
    lift_receipt = _receipt(
        artifact=artifact,
        stage="lift",
        prior=approach_receipt,
        started_ns=started,
        final_joints=final,
        gripper=gripper,
        receipt_dir=receipt_dir,
    )
    if lift_hold_s > 0.0:
        time.sleep(lift_hold_s)

    # Return along exact collision-checked edges.  Lower the held object to
    # its original grasp pose, release it there, then retreat open.
    guard = stage_executor.CommandGuard()
    try:
        timed_lift, lift_times_s = stage_executor.timed_stage_path(
            artifact,
            "lift",
        )
        reverse_lift_times_s = float(lift_times_s[-1]) - lift_times_s[::-1]
        final = stage_executor.execute_timed_joint_path(
            robot,
            np.asarray(timed_lift[::-1], dtype=float),
            np.asarray(reverse_lift_times_s, dtype=float),
            guard,
            speed_percent=speed_percent,
            segment_timeout_s=segment_timeout_s,
            start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
            feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
        )
        guard.path_motion_started = False
        stage_executor.enter_can_joint_control(robot, guard)
        try:
            baseline = stage_executor.normalize_gripper_feedback(effector.get_gripper_status())
            guard.mark_before_command()
            effector.move_gripper_m(value=stage_executor.OPEN_APERTURE_M, force=gripper_force_n)
            opened = stage_executor.wait_for_gripper(
                effector,
                lambda sample: sample.aperture_m >= 0.060,
                after_timestamp=baseline.timestamp,
                timeout_s=3.0,
                static_accept_after_s=0.5,
            )
            stage_executor.verify_gripper_ready(opened)
        except Exception as error:
            print(
                f"WARNING: release feedback degraded; continuing checked Home return: {error}",
                file=sys.stderr,
            )
        final = stage_executor.execute_joint_path(
            robot,
            np.asarray(approach_path[::-1], dtype=float),
            guard,
            speed_percent=speed_percent,
            segment_timeout_s=segment_timeout_s,
            start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
            feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
        )
        final = stage_executor.execute_joint_path(
            robot,
            np.asarray(pregrasp_path[::-1], dtype=float),
            guard,
            speed_percent=speed_percent,
            segment_timeout_s=segment_timeout_s,
            start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
            feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
        )
    except BaseException:
        stage_executor.emergency_stop_after_failure(robot, guard)
        raise
    return {
        "schema": "z_manip.piper_full_grasp_receipt.v1",
        "success": True,
        "artifact_id": artifact.artifact_id,
        "speed_percent": speed_percent,
        "lift_contact_verified": True,
        "object_released_at_grasp_pose": True,
        "returned_home_on_checked_reverse_path": True,
        "final_joints_rad": [float(value) for value in final],
        "stage_receipts": {
            "pregrasp": str(pregrasp_receipt.path),
            "approach_close": str(approach_receipt.path),
            "lift": str(lift_receipt.path),
        },
        "finished_unix_ns": time.time_ns(),
    }


def execute_workflow_phase(
    robot: object,
    effector: object,
    artifact: stage_executor.PlanningArtifact,
    *,
    workflow_phase: str,
    planning_session_id: str,
    receipt_dir: Path,
    prior_receipt_dir: Path | None,
    speed_percent: int,
    segment_timeout_s: float,
    gripper_force_n: float,
    executor_start: tuple[int, int] | None = None,
    direct_approach: bool = True,
) -> dict[str, object]:
    """Execute one durable pick/hold/return/place-back workflow transition."""
    if workflow_phase == "pick-hold":
        receipt_dir.mkdir(parents=True, exist_ok=False)
        if executor_start is not None:
            _write_executor_start_receipt(
                receipt_dir,
                artifact=artifact,
                workflow_phase=workflow_phase,
                planning_session_id=planning_session_id,
                started_unix_ns=executor_start[0],
                started_monotonic_ns=executor_start[1],
            )
        pregrasp_path = stage_executor.validate_stage_context(artifact, "pregrasp", None)
        started = time.time_ns()
        final, gripper = stage_executor.execute_stage(
            robot, effector, artifact, "pregrasp", pregrasp_path,
            speed_percent=speed_percent,
            segment_timeout_s=segment_timeout_s,
            gripper_force_n=gripper_force_n,
        )
        pregrasp_receipt = _receipt(
            artifact=artifact, stage="pregrasp", prior=None, started_ns=started,
            final_joints=final, gripper=gripper, receipt_dir=receipt_dir,
        )
        approach_path = stage_executor.validate_stage_context(
            artifact, "approach_close", pregrasp_receipt,
        )
        started = time.time_ns()
        final, gripper = stage_executor.execute_stage(
            robot, effector, artifact, "approach_close", approach_path,
            speed_percent=speed_percent,
            segment_timeout_s=segment_timeout_s,
            gripper_force_n=gripper_force_n,
            direct_approach=direct_approach,
        )
        approach_receipt = _receipt(
            artifact=artifact, stage="approach_close", prior=pregrasp_receipt,
            started_ns=started, final_joints=final, gripper=gripper,
            receipt_dir=receipt_dir,
            approach_execution="streamed" if direct_approach else "stepped",
        )
        lift_path = stage_executor.validate_stage_context(
            artifact, "lift", approach_receipt,
        )
        started = time.time_ns()
        final, gripper = stage_executor.execute_stage(
            robot, effector, artifact, "lift", lift_path,
            speed_percent=speed_percent,
            segment_timeout_s=segment_timeout_s,
            gripper_force_n=gripper_force_n,
        )
        _receipt(
            artifact=artifact, stage="lift", prior=approach_receipt,
            started_ns=started, final_joints=final, gripper=gripper,
            receipt_dir=receipt_dir,
        )
        state = _workflow_state(
            receipt_dir, artifact=artifact, phase="holding_at_lift",
            final_joints=final, holding_object=True, at_home=False,
            planning_session_id=planning_session_id, prior_workflow_sha256=None,
        )
    else:
        expected: str | tuple[str, ...] = (
            "holding_at_lift"
            if workflow_phase == "return-home-holding"
            else ("holding_at_lift", "holding_at_home")
        )
        prior_state = _load_workflow_state(
            prior_receipt_dir, artifact=artifact, expected_phase=expected,
            planning_session_id=planning_session_id,
        )
        receipt_dir.mkdir(parents=True, exist_ok=False)
        if executor_start is not None:
            _write_executor_start_receipt(
                receipt_dir,
                artifact=artifact,
                workflow_phase=workflow_phase,
                planning_session_id=planning_session_id,
                started_unix_ns=executor_start[0],
                started_monotonic_ns=executor_start[1],
            )
        pregrasp_path = stage_executor.validate_stage_context(artifact, "pregrasp", None)
        approach_path = np.asarray(artifact.arrays["approach_raw"], dtype=float)
        timed_lift, lift_times_s = stage_executor.timed_stage_path(artifact, "lift")
        # Authorize the loaded corridor BEFORE the transport is asked for
        # anything.  An artifact whose while-holding legs were never checked
        # with the object attached must not reach a single move_j.
        carry: np.ndarray | None = None
        place_back_carry: np.ndarray | None = None
        if workflow_phase == "return-home-holding":
            carry = _checked_holding_return(artifact)
        elif prior_state["phase"] == "holding_at_home":
            place_back_carry = _checked_place_back_from_home(artifact)
        guard = stage_executor.CommandGuard()
        try:
            _verify_holding_object(
                robot, effector, guard, artifact, gripper_force_n=gripper_force_n,
            )
            if workflow_phase == "return-home-holding":
                if carry is not None:
                    # One continuous move_j from the lift top straight to the
                    # pregrasp.  The reverse lift it replaces existed only so
                    # the reverse approach could start at the grasp pose, and
                    # its whole visible effect was to put the object back down
                    # on the spot it was picked from before carrying it home.
                    _require_carry_start(prior_state, carry)
                    final = stage_executor.execute_joint_path(
                        robot, np.asarray(carry, dtype=float), guard,
                        speed_percent=speed_percent,
                        segment_timeout_s=segment_timeout_s,
                        start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                        feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                    )
                else:
                    reverse_lift_times_s = float(lift_times_s[-1]) - lift_times_s[::-1]
                    final = stage_executor.execute_timed_joint_path(
                        robot, np.asarray(timed_lift[::-1], dtype=float),
                        np.asarray(reverse_lift_times_s, dtype=float), guard,
                        speed_percent=speed_percent,
                        segment_timeout_s=segment_timeout_s,
                        start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                        feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                    )
                    guard.path_motion_started = False
                    final = stage_executor.execute_joint_path(
                        robot, np.asarray(approach_path[::-1], dtype=float), guard,
                        speed_percent=speed_percent, segment_timeout_s=segment_timeout_s,
                        start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                        feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                    )
                final = stage_executor.execute_joint_path(
                    robot, np.asarray(pregrasp_path[::-1], dtype=float), guard,
                    speed_percent=speed_percent, segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                    feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                )
                state = _workflow_state(
                    receipt_dir, artifact=artifact, phase="holding_at_home",
                    final_joints=final, holding_object=True, at_home=True,
                    planning_session_id=planning_session_id,
                    prior_workflow_sha256=str(prior_state["_receipt_sha256"]),
                )
            else:
                if prior_state["phase"] == "holding_at_lift":
                    # Recovery path: Return Home may fail before moving.  In
                    # that case lower the held object along the already
                    # executed lift trajectory directly back to its grasp
                    # pose, instead of stranding it or forcing a Home detour.
                    reverse_lift_times_s = float(lift_times_s[-1]) - lift_times_s[::-1]
                    final = stage_executor.execute_timed_joint_path(
                        robot, np.asarray(timed_lift[::-1], dtype=float),
                        np.asarray(reverse_lift_times_s, dtype=float), guard,
                        speed_percent=speed_percent,
                        segment_timeout_s=segment_timeout_s,
                        start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                        feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                    )
                else:
                    final = stage_executor.execute_joint_path(
                        robot, pregrasp_path, guard, speed_percent=speed_percent,
                        segment_timeout_s=segment_timeout_s,
                        start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                        feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                    )
                    if place_back_carry is not None:
                        # The reverse-approach descent was not proven with the
                        # object attached, but the carry corridor the arm came
                        # home on was.  Retrace it -- pregrasp -> lift top --
                        # and lower down the (also checked) lift column.  This
                        # is what makes the return a round trip: the object can
                        # always be set back down from Home on the same
                        # evidence that let it be carried there.
                        final = stage_executor.execute_joint_path(
                            robot,
                            np.asarray(place_back_carry[::-1], dtype=float),
                            guard, speed_percent=speed_percent,
                            segment_timeout_s=segment_timeout_s,
                            start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                            feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                        )
                        reverse_lift_times_s = (
                            float(lift_times_s[-1]) - lift_times_s[::-1]
                        )
                        final = stage_executor.execute_timed_joint_path(
                            robot, np.asarray(timed_lift[::-1], dtype=float),
                            np.asarray(reverse_lift_times_s, dtype=float), guard,
                            speed_percent=speed_percent,
                            segment_timeout_s=segment_timeout_s,
                            start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                            feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                        )
                    else:
                        final = stage_executor.execute_joint_path(
                            robot, approach_path, guard, speed_percent=speed_percent,
                            segment_timeout_s=segment_timeout_s,
                            start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                            feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                        )
                guard.path_motion_started = False
                _open_gripper(robot, effector, guard, gripper_force_n=gripper_force_n)
                final = stage_executor.execute_joint_path(
                    robot, np.asarray(approach_path[::-1], dtype=float), guard,
                    speed_percent=speed_percent, segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                    feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                )
                final = stage_executor.execute_joint_path(
                    robot, np.asarray(pregrasp_path[::-1], dtype=float), guard,
                    speed_percent=speed_percent, segment_timeout_s=segment_timeout_s,
                    start_tolerance_rad=stage_executor.DEFAULT_START_TOLERANCE_RAD,
                    feedback_tolerance_rad=stage_executor.DEFAULT_FEEDBACK_TOLERANCE_RAD,
                )
                state = _workflow_state(
                    receipt_dir, artifact=artifact, phase="placed_back_at_home",
                    final_joints=final, holding_object=False, at_home=True,
                    planning_session_id=planning_session_id,
                    prior_workflow_sha256=str(prior_state["_receipt_sha256"]),
                )
        except BaseException:
            stage_executor.emergency_stop_after_failure(robot, guard)
            raise
    return {
        "schema": "z_manip.piper_grasp_workflow_phase_receipt.v1",
        "success": True,
        "artifact_id": artifact.artifact_id,
        "workflow": state,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-report", type=Path, required=True)
    parser.add_argument("--planned-grasp", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--segment-timeout-s", type=float, default=12.0)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument("--lift-hold-s", type=float, default=2.0)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--firmware", choices=("default", "v183", "v188", "v189"), default="v188")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--workflow-phase", choices=WORKFLOW_PHASES, default="full")
    parser.add_argument("--prior-receipt-dir", type=Path)
    parser.add_argument("--planning-session-id")
    parser.add_argument(
        "--staged-approach-stop",
        action="store_true",
        help=(
            "legacy fallback: halt at the pregrasp standoff and step the grasp "
            "descent instead of streaming directly into contact (the accurate "
            "default)"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not 1 <= args.speed_percent <= stage_executor.MAX_SPEED_PERCENT:
            raise stage_executor.SafetyError(
                f"--speed-percent must be within 1-{stage_executor.MAX_SPEED_PERCENT}",
            )
        if args.workflow_phase != "full":
            if not isinstance(args.planning_session_id, str) or not args.planning_session_id:
                raise stage_executor.SafetyError("workflow phases require --planning-session-id")
            if len(args.planning_session_id) > 128:
                raise stage_executor.SafetyError("planning session id is too long")
        if not 0.0 <= args.lift_hold_s <= 5.0:
            raise stage_executor.SafetyError("--lift-hold-s must be within 0-5s")
        artifact = stage_executor.load_planning_artifact(
            args.planning_report,
            args.planned_grasp,
            expected_npz_sha256=None,
            stage="pregrasp",
            now_ns=time.time_ns(),
            max_source_age_s=(
                MAX_FULL_SOURCE_AGE_S
                if args.workflow_phase in {"full", "pick-hold"}
                else MAX_CONTINUATION_AGE_S
            ),
            # Continuations deliberately reuse the exact artifact executed by
            # Pick & Hold.  The workflow receipt binds its hash, session and
            # allowed continuation age; demanding a new camera frame here
            # would strand a held object after the 30-second live-data limit.
            require_fresh_source=args.workflow_phase in {"full", "pick-hold"},
        )
        token = full_token(artifact.artifact_id)
        dry_run = {
            "schema": "z_manip.piper_full_grasp_dry_run.v1",
            "dry_run": not args.execute,
            "artifact_id": artifact.artifact_id,
            "confirmation_token": token,
            "speed_percent": args.speed_percent,
            "sequence": [
                "open",
                "transit",
                "approach_close",
                "lift",
                "lower_to_grasp",
                "release",
                "reverse_approach",
                "reverse_transit_home",
            ],
            "commands_sent": 0,
        }
        if not args.execute:
            print(json.dumps(dry_run, indent=2))
            return 0
        if args.confirm != token:
            raise stage_executor.SafetyError(
                f"real full grasp requires exact confirmation token: {token}",
            )
        robot = None
        try:
            robot, effector = stage_executor.connect_real_arm(args.channel, args.firmware)
            # Both clocks are sampled at the first trustworthy real-executor
            # boundary: the CAN transport has opened, but no stage command has
            # yet been emitted.
            executor_start = (time.time_ns(), time.monotonic_ns())
            direct_approach = not args.staged_approach_stop
            if args.workflow_phase == "full":
                result = execute_full_grasp(
                    robot, effector, artifact, receipt_dir=args.receipt_dir,
                    speed_percent=args.speed_percent,
                    segment_timeout_s=args.segment_timeout_s,
                    gripper_force_n=args.gripper_force_n,
                    lift_hold_s=args.lift_hold_s,
                    executor_start=executor_start,
                    planning_session_id=args.planning_session_id or "",
                    direct_approach=direct_approach,
                )
            else:
                result = execute_workflow_phase(
                    robot, effector, artifact, workflow_phase=args.workflow_phase,
                    planning_session_id=args.planning_session_id,
                    receipt_dir=args.receipt_dir,
                    prior_receipt_dir=args.prior_receipt_dir,
                    speed_percent=args.speed_percent,
                    segment_timeout_s=args.segment_timeout_s,
                    gripper_force_n=args.gripper_force_n,
                    executor_start=executor_start,
                    direct_approach=direct_approach,
                )
            print(json.dumps(result, indent=2))
            return 0
        finally:
            stage_executor.disconnect_quietly(robot)
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
