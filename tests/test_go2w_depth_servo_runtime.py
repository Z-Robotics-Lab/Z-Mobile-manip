from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from z_manip.control.servo_phase import ServoPhase  # noqa: E402
from z_manip.control.reactive_servo import ReactivePhase  # noqa: E402

SCRIPT = ROOT / "scripts" / "runtime" / "go2w_depth_servo.py"
LAUNCHER = ROOT / "scripts" / "runtime" / "go2w_depth_servo.sh"
SPEC = importlib.util.spec_from_file_location("go2w_depth_servo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVO
SPEC.loader.exec_module(SERVO)


def _core(*, mode: str = "live"):
    return SERVO.DepthServoCore(
        SERVO.DepthServoSettings(
            mode=mode,
            desired_depth_m=0.50,
            depth_tolerance_m=0.01,
            lateral_tolerance_m=0.12,
            settle_time_s=0.10,
            handoff_depth_m=0.52,
            handoff_bearing_rad=0.3490658503988659,
            yaw_gain=0.70,
            min_forward_mps=0.10,
            max_forward_mps=0.18,
            max_reverse_mps=0.05,
            max_yaw_rps=0.12,
            rotate_only_bearing_rad=0.4363323129985824,
            yaw_deadband_rad=0.10471975511965978,
            target_timeout_s=0.25,
            tracking_hold_s=0.55,
            tracking_loss_grace_s=0.75,
            allow_legacy_optical_depth_for_tests=True,
        )
    )


def _reactive_core(
    *,
    mode: str = "live",
    target_timeout_s: float = 0.25,
    geometry_staleness_timeout_s: float | None = None,
):
    return SERVO.DepthServoCore(
        SERVO.DepthServoSettings(
            mode=mode,
            desired_depth_m=0.50,
            handoff_depth_m=0.62,
            target_timeout_s=target_timeout_s,
            tracking_hold_s=0.55,
            tracking_loss_grace_s=max(0.75, target_timeout_s),
            # Defaults to the target timeout, the tightest value the settings
            # validator now permits: ``_transforms_received_s`` and
            # ``_target_received_s`` are written from one camera callback, so a
            # geometry budget below the target budget can only ever fire early.
            geometry_staleness_timeout_s=(
                target_timeout_s
                if geometry_staleness_timeout_s is None
                else geometry_staleness_timeout_s
            ),
        )
    )


def _observe_in_frames(
    core,
    *,
    camera_xyz,
    base_xyz,
    arm_xyz,
    stamp_s,
):
    import numpy as np

    camera = np.asarray(camera_xyz, dtype=float)
    base_from_camera = np.eye(4)
    base_from_camera[:3, 3] = np.asarray(base_xyz) - camera
    arm_from_camera = np.eye(4)
    arm_from_camera[:3, 3] = np.asarray(arm_xyz) - camera
    return core.observe_target(
        x_m=float(camera[0]),
        y_m=float(camera[1]),
        z_m=float(camera[2]),
        stamp_s=stamp_s,
        T_base_camera=base_from_camera,
        T_arm_camera=arm_from_camera,
    )


def _runtime_transform_artifact(path: Path, *, stamp_ns: int) -> None:
    base = np.eye(4)
    base[:3, 3] = (0.06, 0.0, 0.067)
    arm = np.eye(4)
    path.write_text(json.dumps({
        "schema": "z_manip.runtime_state.v1",
        "kinematic_transforms": {
            "schema": "z_manip.kinematic_transforms.v1",
            "verified": True,
            "calibration_synthetic": False,
            "source_timestamp_ns": stamp_ns,
            "camera_frame": "camera_color_optical_frame",
            "platform_base_frame": "base_link",
            "arm_base_frame": "piper_base_link",
            "platform_base_from_camera": base.tolist(),
            "arm_base_from_camera": arm.tolist(),
        },
    }), encoding="utf-8")


def test_runtime_observer_kinematic_transform_is_accepted_when_fresh(tmp_path):
    artifact = tmp_path / "runtime-observer.json"
    now_ns = 1_700_000_000_000_000_000
    _runtime_transform_artifact(artifact, stamp_ns=now_ns - 100_000_000)

    base, arm, stamp = SERVO._runtime_state_transforms(
        artifact,
        source_frame="camera_color_optical_frame",
        base_frame="base_link",
        arm_base_frame="piper_base_link",
        now_unix_ns=now_ns,
        max_age_s=0.5,
    )

    assert stamp == now_ns - 100_000_000
    assert base[:3, 3] == pytest.approx((0.06, 0.0, 0.067))
    assert arm == pytest.approx(np.eye(4))


def test_runtime_observer_transform_rejects_stale_or_wrong_frame(tmp_path):
    artifact = tmp_path / "runtime-observer.json"
    now_ns = 1_700_000_000_000_000_000
    _runtime_transform_artifact(artifact, stamp_ns=now_ns - 600_000_000)
    with pytest.raises(ValueError, match="stale"):
        SERVO._runtime_state_transforms(
            artifact,
            source_frame="camera_color_optical_frame",
            base_frame="base_link",
            arm_base_frame="piper_base_link",
            now_unix_ns=now_ns,
            max_age_s=0.5,
        )
    _runtime_transform_artifact(artifact, stamp_ns=now_ns)
    with pytest.raises(ValueError, match="camera frame"):
        SERVO._runtime_state_transforms(
            artifact,
            source_frame="camera_depth_optical_frame",
            base_frame="base_link",
            arm_base_frame="piper_base_link",
            now_unix_ns=now_ns,
            max_age_s=0.5,
        )


def test_live_mode_drives_toward_a_fresh_target():
    core = _core()
    core.observe_target(x_m=0.10, z_m=1.00, stamp_s=1.0)

    output = core.tick(now_s=1.05, tracking=True)

    assert output.phase == "approach"
    assert 0.0 < output.proposed_linear_x <= 0.18
    assert output.published_linear_x == output.proposed_linear_x
    assert output.published_angular_z == output.proposed_angular_z


def test_shadow_mode_calculates_but_never_publishes_motion():
    core = _core(mode="shadow")
    core.observe_target(x_m=-0.15, z_m=0.90, stamp_s=2.0)

    output = core.tick(now_s=2.05, tracking=True)

    assert output.phase == "approach"
    assert output.proposed_linear_x > 0.0
    assert output.proposed_angular_z > 0.0
    assert output.published_linear_x == 0.0
    assert output.published_angular_z == 0.0


@pytest.mark.parametrize("tracking", [False, None])
def test_missing_or_unlocked_target_immediately_stops(tracking):
    core = _core()
    core.observe_target(x_m=0.0, z_m=1.0, stamp_s=3.0)

    output = core.tick(now_s=3.05, tracking=tracking)

    assert output.phase == "reacquiring"
    assert output.published_linear_x == 0.0
    assert output.published_angular_z == 0.0


def test_stale_target_immediately_stops():
    core = _core()
    core.observe_target(x_m=0.0, z_m=1.0, stamp_s=4.0)

    output = core.tick(now_s=4.30, tracking=True)

    assert output.phase == "reacquiring"
    assert output.published_linear_x == 0.0
    assert output.published_angular_z == 0.0


def test_loss_grace_never_blindly_moves_and_then_becomes_terminal():
    core = _core()
    core.observe_target(x_m=0.0, z_m=1.0, stamp_s=4.0)

    grace = core.tick(now_s=4.40, tracking=False)
    lost = core.tick(now_s=4.80, tracking=False)

    assert grace.phase == "reacquiring"
    assert grace.published_linear_x == 0.0
    assert lost.phase == "tracking_lost"
    assert lost.published_linear_x == 0.0


def test_target_filter_rejects_one_large_depth_jump():
    core = _core()
    assert core.observe_target(x_m=0.01, z_m=0.90, stamp_s=1.0)
    assert not core.observe_target(x_m=0.40, z_m=0.30, stamp_s=1.05)

    output = core.tick(now_s=1.10, tracking=True)

    assert output.phase == "approach"
    assert core.target == pytest.approx((0.01, 0.0, 0.90))
    assert core.filter_stats["rejected_outliers"] == 1


def test_target_filter_reduces_alternating_depth_noise():
    core = _core()
    for index, depth in enumerate((0.90, 0.94, 0.88, 0.93, 0.89)):
        assert core.observe_target(x_m=0.02, z_m=depth, stamp_s=1.0 + index * 0.05)

    assert core.target is not None
    assert 0.89 <= core.target[2] <= 0.92
    assert core.filter_stats["window_samples"] == 5


def test_target_filter_preserves_vertical_coordinate_and_3d_geometry():
    core = _core()
    for index, y_m in enumerate((0.20, 0.22, 0.18, 0.21, 0.19)):
        assert core.observe_target(
            x_m=0.10,
            y_m=y_m,
            z_m=0.80,
            stamp_s=1.0 + index * 0.05,
        )

    assert core.target is not None
    assert core.target == pytest.approx((0.10, 0.20, 0.80), abs=0.01)
    geometry = core.camera_geometry
    assert geometry is not None
    assert geometry["camera_range_m"] == pytest.approx(
        (0.10 ** 2 + core.target[1] ** 2 + 0.80 ** 2) ** 0.5,
    )
    assert geometry["camera_elevation_rad"] < 0.0


def test_target_jump_filter_uses_full_3d_euclidean_distance():
    core = _core()
    assert core.observe_target(x_m=0.0, y_m=0.0, z_m=0.80, stamp_s=1.0)

    assert not core.observe_target(x_m=0.0, y_m=0.25, z_m=0.80, stamp_s=1.1)
    assert core.filter_stats["rejected_outliers"] == 1


def test_persistent_coherent_outlier_cluster_rebases_stale_filter():
    core = _core()
    assert core.observe_target(x_m=0.0, y_m=0.0, z_m=0.80, stamp_s=1.0)

    assert not core.observe_target(x_m=0.01, y_m=0.0, z_m=0.55, stamp_s=1.1)
    assert not core.observe_target(x_m=0.00, y_m=0.01, z_m=0.54, stamp_s=1.2)
    assert core.observe_target(x_m=-0.01, y_m=0.0, z_m=0.56, stamp_s=1.3)

    assert core.target == pytest.approx((0.0, 0.0, 0.55), abs=0.011)
    assert core.filter_stats["rebases"] == 1
    assert core.filter_stats["outlier_cluster_samples"] == 0


def test_incoherent_outliers_never_rebase_the_filter():
    core = _core()
    assert core.observe_target(x_m=0.0, y_m=0.0, z_m=0.80, stamp_s=1.0)

    assert not core.observe_target(x_m=0.30, y_m=0.0, z_m=0.40, stamp_s=1.1)
    assert not core.observe_target(x_m=-0.30, y_m=0.0, z_m=0.40, stamp_s=1.2)
    assert not core.observe_target(x_m=0.0, y_m=0.30, z_m=0.40, stamp_s=1.3)

    assert core.target == pytest.approx((0.0, 0.0, 0.80))
    assert core.filter_stats["rebases"] == 0


def test_legged_handoff_accepts_coarse_near_field_alignment_immediately():
    core = _core()
    core.observe_target(x_m=0.09, z_m=0.515, stamp_s=5.0)
    reached = core.tick(now_s=5.0, tracking=True)

    assert reached.phase == "reached"
    assert reached.done is True
    assert reached.published_linear_x == 0.0
    assert reached.published_angular_z == 0.0


def test_near_field_handoff_latches_before_post_step_rebound():
    core = _core()
    core.observe_target(x_m=-0.03, z_m=0.515, stamp_s=7.0)
    assert core.tick(now_s=7.0, tracking=True).phase == "reached"

    # Once handed off, later body-sway depth cannot restart base motion.
    core.observe_target(x_m=0.03, z_m=0.62, stamp_s=7.20)
    latched = core.tick(now_s=7.20, tracking=True)
    assert latched.phase == "reached"
    assert latched.published_linear_x == latched.published_angular_z == 0.0


def test_target_already_inside_55cm_never_commands_reverse_motion():
    core = _core()
    core.observe_target(x_m=0.01, z_m=0.40, stamp_s=6.0)

    output = core.tick(now_s=6.05, tracking=True)

    assert output.phase == "reached"
    assert output.published_linear_x == 0.0
    assert output.published_angular_z == 0.0


def test_approach_keeps_go2w_above_observed_low_speed_dead_zone():
    core = _core()
    core.observe_target(x_m=0.0, z_m=0.57, stamp_s=8.0)

    output = core.tick(now_s=8.01, tracking=True)

    assert output.phase == "approach"
    assert output.proposed_linear_x == 0.10
    assert output.published_linear_x == 0.10


def test_far_field_approach_uses_brisk_cruise_limit():
    core = _core()
    core.observe_target(x_m=0.0, z_m=1.30, stamp_s=9.0)

    output = core.tick(now_s=9.01, tracking=True)

    assert output.proposed_linear_x == 0.18


def test_deployed_core_missing_tf_is_explicitly_zero_speed():
    core = _reactive_core()
    core.observe_target(
        x_m=0.0,
        y_m=0.1,
        z_m=0.90,
        stamp_s=1.0,
        transform_error="base_link TF unavailable",
    )

    output = core.tick(now_s=1.05, tracking=True)

    assert output.phase == "transform_unavailable"
    assert output.published_linear_x == output.published_angular_z == 0.0
    assert output.reactive_phase == "transform_unavailable"
    assert "base_link TF unavailable" in output.reason


def test_reactive_runtime_uses_transformed_ground_plane_range_not_optical_z():
    core = _reactive_core()
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 0.45),
        base_xyz=(0.90, 0.0, -0.10),
        arm_xyz=(0.75, 0.0, 0.10),
        stamp_s=2.0,
    )

    output = core.tick(now_s=2.05, tracking=True)

    assert output.phase == "approach"
    assert output.reactive_phase == "base_approach"
    assert output.proposed_linear_x > 0.0
    assert output.depth_error_m == pytest.approx(0.40)
    assert core.geometry is not None
    assert core.geometry.base_planar_distance_m == pytest.approx(0.90)


def test_reactive_runtime_stops_for_downstream_ik_probe_in_3d_corridor():
    core = _reactive_core()
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 0.55),
        base_xyz=(0.55, 0.13, -0.10),
        arm_xyz=(0.50, 0.0, 0.10),
        stamp_s=3.0,
    )

    settling = core.tick(now_s=3.05, tracking=True)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 0.55),
        base_xyz=(0.55, 0.13, -0.10),
        arm_xyz=(0.50, 0.0, 0.10),
        stamp_s=3.35,
    )
    probe = core.tick(now_s=3.36, tracking=True)

    assert settling.phase == "handoff_settle"
    assert settling.published_linear_x == settling.published_angular_z == 0.0
    assert probe.phase == "handoff_probe"
    assert probe.needs_ik_probe
    assert probe.published_linear_x == probe.published_angular_z == 0.0
    assert core.reactive_status is not None
    assert core.reactive_status["needs_ik_probe"] is True
    assert core.reactive_status["side"] == "left"
    assert core.reactive_status["desired_target_lateral_m"] == pytest.approx(0.13)

    core.set_ik_probe_result(True)
    reached = core.tick(now_s=3.37, tracking=True)
    assert reached.phase == "reached"
    assert reached.done


def test_runtime_handoffs_at_wrist_near_field_before_base_52cm():
    core = SERVO.DepthServoCore(SERVO.DepthServoSettings(
        mode="live",
        desired_depth_m=0.50,
        handoff_depth_m=0.52,
        handoff_settle_s=0.30,
    ))
    sample = {
        "camera_xyz": (0.018, 0.037, 0.476),
        "base_xyz": (0.600, 0.130, 0.074),
        "arm_xyz": (0.480, 0.000, 0.074),
    }
    assert _observe_in_frames(core, stamp_s=10.0, **sample)
    settling = core.tick(now_s=10.01, tracking=True)
    assert _observe_in_frames(core, stamp_s=10.35, **sample)
    probe = core.tick(now_s=10.36, tracking=True)

    assert core.geometry is not None
    assert core.geometry.base_planar_distance_m > 0.52
    assert settling.phase == "handoff_settle"
    assert probe.phase == "handoff_probe"
    assert probe.needs_ik_probe
    assert probe.published_linear_x == probe.published_angular_z == 0.0


def test_handoff_is_latched_across_a_later_body_sway_sample():
    probe = SERVO.DepthServoOutput(
        phase="handoff_probe",
        proposed_linear_x=0.0,
        proposed_angular_z=0.0,
        published_linear_x=0.0,
        published_angular_z=0.0,
        depth_error_m=0.02,
        yaw_error_rad=0.03,
        target_age_s=0.01,
        reactive_phase="handoff_probe",
        needs_ik_probe=True,
    )
    body_sway = SERVO.DepthServoOutput(
        phase="approach",
        proposed_linear_x=0.10,
        proposed_angular_z=-0.05,
        published_linear_x=0.10,
        published_angular_z=-0.05,
        depth_error_m=0.08,
        yaw_error_rad=-0.12,
        target_age_s=0.01,
        reactive_phase="base_approach",
    )

    latched = SERVO._latch_handoff_output(None, probe)
    assert latched is not None
    replayed = SERVO._latch_handoff_output(latched, body_sway)

    assert replayed is latched
    assert replayed.phase == "handoff_probe"
    assert replayed.published_linear_x == replayed.published_angular_z == 0.0


def test_side_choice_is_latched_until_terminal_tracking_loss():
    core = _reactive_core(target_timeout_s=0.25)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 0.80),
        base_xyz=(0.90, -0.20, -0.10),
        arm_xyz=(0.75, 0.0, 0.10),
        stamp_s=1.0,
    )
    assert core.desired_target_lateral_m == pytest.approx(-0.13)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 0.78),
        base_xyz=(0.88, 0.20, -0.10),
        arm_xyz=(0.73, 0.0, 0.10),
        stamp_s=1.1,
    )
    assert core.desired_target_lateral_m == pytest.approx(-0.13)

    core.tick(now_s=2.0, tracking=False)
    assert core.desired_target_lateral_m == 0.0


def test_stale_synchronized_transform_never_reuses_old_geometry_for_motion():
    """Old geometry stops the base -- via the TARGET timer, not a second one.

    REWRITTEN, and the rewrite is the point.  This test used to build a core
    with ``target_timeout_s=1.0, transform_timeout_s=0.25`` and assert that an
    observation 0.30 s old produced ``transform_unavailable``.  That
    combination is now rejected at construction: it is the R4 inversion
    itself, written down as an expectation.  ``_transforms_received_s`` and
    ``_target_received_s`` are both ``float(stamp_s)`` from one camera
    callback, so the old assertion was really "a duplicate of the target timer,
    set four times tighter, fires first" -- and on the shipped robot that meant
    two dropped camera frames zeroed the base with a TF error message.

    The protection the name promises is still real and still tested: once the
    observation ages past the target budget the base is zero and the geometry
    is not reused for motion.  It is the TARGET timer that says so.
    """

    core = _reactive_core(target_timeout_s=0.25)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 0.80),
        base_xyz=(0.90, 0.0, -0.10),
        arm_xyz=(0.75, 0.0, 0.10),
        stamp_s=4.0,
    )
    output = core.tick(now_s=4.30, tracking=True)

    assert output.published_linear_x == output.published_angular_z == 0.0
    assert output.phase in SERVO.LOSS_STAIR_PHASES
    assert output.phase != "transform_unavailable"


def test_tracking_loss_with_stale_tf_reports_tracker_recovery_not_tf_outage():
    """Not a TF outage -- and, since this stage, not a loss either.

    EXPECTATION CHANGED, deliberately.  It asserted ``search_required``.  The
    core here has taken ONE observation and then ticked with
    ``tracking=False``, so ``ReactiveTargetController.update`` is handed
    ``None`` and its own ``_last_geometry`` has never been set: the controller
    has never held a 3-D target in this session.  That is the
    ``bundle_count == 1`` startup state, which is 12 of 12 rows in the recorded
    corpus, and answering it with the loss stair's most severe verdict is what
    made the supervisor sweep the wrist away from a freshly seeded target
    (symptom B).  It now answers ``acquiring``, which carries its own table
    row, its own finite deadline and a terminal stationary expiry.

    What the test's NAME promises is unchanged and still pinned: the phase is
    not ``transform_unavailable``, and the base is stopped either way.
    """

    core = _reactive_core(target_timeout_s=1.0)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.20, 0.80),
        base_xyz=(0.75, 0.0, -0.30),
        arm_xyz=(0.60, 0.0, -0.15),
        stamp_s=5.0,
    )

    output = core.tick(now_s=5.30, tracking=False)

    assert output.phase == ServoPhase.ACQUIRING.value
    assert output.phase != ServoPhase.TRANSFORM_UNAVAILABLE.value
    assert output.published_linear_x == output.published_angular_z == 0.0
    assert core.reactive_status is not None
    assert core.reactive_status["phase"] == ServoPhase.ACQUIRING.value
    assert "3-D target" in output.reason


def test_stale_target_with_tracking_true_is_tracking_loss_not_tf_outage():
    core = _reactive_core(target_timeout_s=0.25)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.20, 0.80),
        base_xyz=(0.75, 0.0, -0.30),
        arm_xyz=(0.60, 0.0, -0.15),
        stamp_s=6.0,
    )
    assert core.tick(now_s=6.05, tracking=True).phase == "posture_adjust"

    output = core.tick(now_s=6.30, tracking=True)

    assert output.phase == "tracking_hold"
    assert output.published_linear_x == output.published_angular_z == 0.0
    assert output.reactive_phase == "tracking_hold"

    recovery = core.tick(now_s=6.70, tracking=True)
    assert recovery.phase == "view_recovery"


def test_ros_style_quaternion_transform_builder_rotates_and_translates():
    matrix = SERVO._rigid_transform_matrix(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
    )
    transformed = matrix @ (1.0, 0.0, 0.0, 1.0)

    assert transformed == pytest.approx((1.0, 3.0, 3.0, 1.0))


def test_live_posture_reached_requires_fresh_feedback_to_settle():
    document = {
        "schema": "z_manip.go2w_posture_status.v1",
        "mode": "live",
        "phase": "reached",
        "stop_latched": False,
        "feedback": {"fresh": True, "source": "sport_state"},
        "capabilities": {
            "euler": True,
            "euler_state": "SUPPORTED_OBSERVED",
        },
        "command": {
            "posture_generation": 7,
            "euler_ack_generation": 7,
            "euler_ack_code": 0,
        },
        "detail": "measured pose reached",
    }

    settled, blocked, shadow, detail = SERVO._posture_feedback_state(
        document,
        age_s=0.10,
    )

    assert settled is True
    assert blocked is False
    assert shadow is False
    assert detail == "measured pose reached"


def test_old_reached_status_with_euler_3203_never_unlocks_handoff():
    document = {
        "schema": "z_manip.go2w_posture_status.v1",
        "mode": "live",
        "phase": "reached",
        "stop_latched": False,
        "feedback": {"fresh": True, "source": "sport_state"},
        "capabilities": {
            "euler": True,
            "euler_state": "SUPPORTED_OBSERVED",
        },
        "command": {
            "posture_generation": 7,
            "euler_ack_generation": 7,
            "euler_ack_code": 3203,
            "codes": {"Euler": 3203},
        },
        "detail": "legacy runtime incorrectly reported reached",
    }

    settled, blocked, shadow, _ = SERVO._posture_feedback_state(
        document,
        age_s=0.10,
    )

    assert settled is False
    assert blocked is False
    assert shadow is False

    document["command"]["euler_ack_code"] = False
    assert SERVO._posture_ack_matches_target(document) is False


def test_shadow_posture_is_diagnostic_and_never_counts_as_settled():
    document = {
        "schema": "z_manip.go2w_posture_status.v1",
        "mode": "shadow",
        "phase": "shadow",
        "stop_latched": False,
        "feedback": {"fresh": True},
    }

    settled, blocked, shadow, _ = SERVO._posture_feedback_state(
        document,
        age_s=0.10,
    )

    assert settled is False
    assert blocked is False
    assert shadow is True


def test_explicit_euler_not_implemented_uses_nonblocking_base_arm_fallback():
    document = {
        "schema": "z_manip.go2w_posture_status.v1",
        "mode": "live",
        "phase": "unsupported",
        "stop_latched": False,
        "feedback": {"fresh": True},
        "capabilities": {
            "euler": False,
            "euler_state": "UNSUPPORTED_FOR_EPOCH",
        },
        "detail": "Euler 1007 returned RPC 3203",
    }

    settled, blocked, shadow, detail = SERVO._posture_feedback_state(
        document,
        age_s=0.10,
    )

    assert settled is True
    assert blocked is False
    assert shadow is False
    assert "3203" in detail


def test_euler_body_unavailable_needs_positive_capability_evidence():
    unavailable = {
        "capabilities": {"euler": False, "euler_state": "UNSUPPORTED_FOR_EPOCH"},
    }
    supported = {
        "capabilities": {"euler": True, "euler_state": "SUPPORTED_OBSERVED"},
    }

    assert SERVO._euler_body_unavailable(unavailable) is True
    assert SERVO._euler_body_unavailable(supported) is False
    assert SERVO._euler_body_unavailable(None) is False
    assert SERVO._euler_body_unavailable({"mode": "live"}) is False


def test_ik_probe_reducer_is_fail_closed_on_absence_or_staleness():
    fresh_true = {"schema": SERVO.IK_PROBE_SCHEMA, "feasible": True}
    fresh_false = {"schema": SERVO.IK_PROBE_SCHEMA, "feasible": False}

    assert SERVO._ik_probe_state(fresh_true, age_s=0.10) is True
    assert SERVO._ik_probe_state(fresh_false, age_s=0.10) is False
    # Stale, wrong schema, malformed verdict, and absence all stay unresolved
    # so the controller keeps requesting the probe instead of handing off.
    assert SERVO._ik_probe_state(fresh_true, age_s=5.0) is None
    assert SERVO._ik_probe_state({"schema": "other", "feasible": True}, age_s=0.1) is None
    assert SERVO._ik_probe_state({"schema": SERVO.IK_PROBE_SCHEMA}, age_s=0.1) is None
    assert SERVO._ik_probe_state(None, age_s=0.1) is None


def test_unactionable_body_posture_core_skips_posture_and_approaches():
    def _run(actionable):
        core = _reactive_core()
        assert _observe_in_frames(
            core,
            camera_xyz=(0.0, 0.30, 0.65),
            base_xyz=(0.72, 0.0, -0.35),
            arm_xyz=(0.60, 0.0, -0.20),
            stamp_s=2.0,
        )
        output = core.tick(
            now_s=2.05,
            tracking=True,
            body_settled=True,
            body_posture_actionable=actionable,
        )
        return core, output

    _, trapped = _run(True)
    core, skipped = _run(False)

    assert trapped.phase == "posture_adjust"
    assert skipped.phase == "approach"
    assert skipped.published_linear_x > 0.0
    # The IK probe is unwired here, so the status exposes an unresolved verdict
    # for the dashboard without forcing a handoff.
    assert core.reactive_status is not None
    assert core.reactive_status["ik_feasible"] is None


def test_whole_body_posture_convergence_uses_velocity_not_tiny_pose_step():
    command = SERVO.WholeBodyRuntimeCommand(
        base_forward_mps=0.0,
        base_yaw_rps=0.0,
        body_height_target_m=None,
        body_roll_target_rad=0.0,
        body_pitch_target_rad=math.radians(0.8),
        arm_joint_velocity_rps=(0.0,) * 6,
        executable=True,
        document={
            "intent": {
                "body_roll_rps": 0.0,
                "body_pitch_rps": math.radians(1.5),
            }
        },
    )

    assert not SERVO._whole_body_posture_rate_converged(command)
    command.document["intent"].update({
        "body_pitch_rps": math.radians(0.3),
    })
    assert SERVO._whole_body_posture_rate_converged(command)


def test_arm_intent_has_a_bounded_wall_clock_lease_and_synchronized_source():
    command = SERVO.WholeBodyRuntimeCommand(
        base_forward_mps=0.0,
        base_yaw_rps=0.0,
        body_height_target_m=None,
        body_roll_target_rad=0.0,
        body_pitch_target_rad=0.0,
        arm_joint_velocity_rps=(0.01, -0.02, 0.03, -0.04, 0.05, -0.06),
        executable=True,
        document={"intent": {"body_roll_rps": 0.0, "body_pitch_rps": 0.0}},
    )

    intent = SERVO._arm_view_intent_document(
        command,
        seq=7,
        now_unix_ns=1_700_000_000_000_000_000,
        target_source_timestamp_ns=1_699_999_999_900_000_000,
    )

    assert intent["schema"] == "z_manip.piper_reactive_view_intent.v1"
    assert intent["seq"] == 7
    assert intent["deadline_unix_ns"] - intent["source_timestamp_ns"] == 250_000_000
    assert intent["target_source_timestamp_ns"] == 1_699_999_999_900_000_000
    assert intent["joint_velocity_rps"] == pytest.approx(command.arm_joint_velocity_rps)


def test_arm_handoff_requires_fresh_acknowledged_measured_target():
    document = {
        "schema": "z_manip.piper_reactive_view_status.v1",
        "owner": "piper_reactive_view_executor",
        "ready": True,
        "stop_latched": False,
        "fault": None,
        "accepted_seq": 8,
        "max_error_rad": math.radians(0.5),
        "feedback_age_s": 0.02,
    }

    ready, reached, blocked, detail = SERVO._arm_feedback_state(
        document,
        age_s=0.05,
        required_seq=8,
    )
    assert ready is True
    assert reached is True
    assert blocked is False
    assert "reached" in detail

    _, old_reached, _, old_detail = SERVO._arm_feedback_state(
        document,
        age_s=0.05,
        required_seq=9,
    )
    assert old_reached is False
    assert "waiting" in old_detail


def test_arm_stop_latch_or_large_measured_error_blocks_handoff():
    document = {
        "schema": "z_manip.piper_reactive_view_status.v1",
        "owner": "piper_reactive_view_executor",
        "ready": True,
        "stop_latched": False,
        "fault": None,
        "accepted_seq": 4,
        "max_error_rad": math.radians(2.0),
        "feedback_age_s": 0.01,
    }
    ready, reached, blocked, _ = SERVO._arm_feedback_state(
        document,
        age_s=0.01,
        required_seq=4,
    )
    assert ready is True
    assert reached is False
    assert blocked is False

    document["stop_latched"] = True
    ready, reached, blocked, _ = SERVO._arm_feedback_state(
        document,
        age_s=0.01,
        required_seq=4,
    )
    assert ready is False
    assert reached is False
    assert blocked is True


@pytest.mark.parametrize(
    ("document", "age_s"),
    [
        (
            {
                "schema": "z_manip.go2w_posture_status.v1",
                "mode": "live",
                "phase": "reached",
                "stop_latched": False,
                "feedback": {"fresh": True},
            },
            1.0,
        ),
        (
            {
                "schema": "z_manip.go2w_posture_status.v1",
                "mode": "live",
                "phase": "stopped",
                "stop_latched": True,
                "feedback": {"fresh": True},
            },
            0.1,
        ),
    ],
)
def test_stale_or_stop_latched_posture_never_unlocks_handoff(document, age_s):
    settled, blocked, shadow, _ = SERVO._posture_feedback_state(
        document,
        age_s=age_s,
    )

    assert settled is False
    assert shadow is False
    if age_s <= 0.75:
        assert blocked is True


def test_launcher_uses_fixed_cyclonedds_runtime_for_pc_to_nuc_commands():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in launcher
    assert "CYCLONEDDS_URI=file:///config/cyclonedds.xml" in launcher
    assert "cyclonedds-go2w-pc.xml" in launcher
    assert "--network host" in launcher
    assert "z-manip-go2w-depth-servo" in launcher
    assert "--velocity-topic /cmd_vel" in launcher
    assert "--max-yaw-rps 0.12" in launcher
    assert "--min-forward-mps 0.10" in launcher
    assert "--max-forward-mps 0.18" in launcher
    assert "--handoff-depth-m 0.52" in launcher
    assert "--handoff-bearing-deg 20" in launcher
    assert "configs/piper_collision_capsules.json" in launcher
    assert ":/robot/piper_collision_capsules.json:ro" in launcher
    assert "--whole-body-collision-model /robot/piper_collision_capsules.json" in launcher


def test_launcher_loss_stair_matches_python_defaults():
    """The launcher pins every servo knob explicitly, so widening the loss
    stair defaults in go2w_depth_servo.py is silently negated unless the
    launcher moves with it (this drift shipped once: defaults widened to
    0.80/2.75 while the launcher still passed 0.55/1.25).  Tie them together
    and re-check the settings validator's ordering at the launcher's values."""

    launcher = LAUNCHER.read_text(encoding="utf-8")

    def _flag(name: str) -> float:
        match = re.search(rf"--{name} (\S+)", launcher)
        assert match is not None, f"launcher no longer passes --{name}"
        return float(match.group(1))

    hold = _flag("tracking-hold-s")
    grace = _flag("tracking-loss-grace-s")
    target_timeout = _flag("target-timeout-s")

    defaults = {
        field.name: field.default
        for field in dataclasses.fields(SERVO.DepthServoSettings)
    }
    assert hold == defaults["tracking_hold_s"]
    assert grace == defaults["tracking_loss_grace_s"]

    assert 0.0 <= hold < grace
    assert grace >= target_timeout


def test_stale_capture_data_is_rejected_even_when_received_fresh():
    """Receipt freshness must not launder queued data: a bundle that arrives
    now but was CAPTURED a second ago (network bufferbloat, live incident
    2026-07-23: 1.2s LAN RTT) would make the servo steer the camera on an old
    world.  Such observations are rejected; the receipt-based timeout then
    holds the base safely."""

    core = _reactive_core(target_timeout_s=0.40)
    fresh = core.observe_target(
        x_m=0.0, y_m=0.0, z_m=1.5,
        stamp_s=1.0,
        capture_age_s=0.20,
    )
    assert fresh is True

    stale = core.observe_target(
        x_m=0.0, y_m=0.0, z_m=1.5,
        stamp_s=1.1,
        capture_age_s=1.2,
    )
    assert stale is False
    status = core.status() if hasattr(core, "status") else None
    # The rejection is visible for diagnosis.
    assert core._stale_data_rejections == 1


def test_whole_body_branch_defers_to_the_loss_stair_and_capture_freshness():
    """The whole-body branch must never override a loss-stair freeze.

    Live 2026-07-24: during a 6s wifi stall the core reported tracking_hold
    (frozen stale target) but the whole-body branch kept solving with the
    retained multi-second-old target and rewrote the phase to
    whole_body_approach -- the arm swung sinusoidally chasing its own motion.
    The branch's exclusion set must contain every loss-stair phase and a
    capture-age guard so stale data can only ever freeze, never steer.

    This used to assert the six phase names were QUOTED inside the branch,
    which is exactly the duplication that let the emitted string
    ("reacquire") drift from the consumer spelling ("reacquiring") with no
    test noticing.  The branch now names the shared ``LOSS_STAIR_PHASES``
    set, so assert the set membership and the guard, not the literals.
    """

    from pathlib import Path

    for phase in (
        "tracking_hold",
        "view_recovery",
        "search_required",
        "transform_unavailable",
        "tracking_lost",
        "waiting_target",
        "reacquiring",
        "posture_blocked",
    ):
        assert phase in SERVO.LOSS_STAIR_PHASES, phase

    source = Path(SERVO.__file__).read_text(encoding="utf-8")
    marker = source.index("The loss stair MUST win over the whole-body branch")
    window = source[marker:marker + 1400]
    assert "fallback.phase in LOSS_STAIR_PHASES" in window
    assert "max_target_capture_age_s" in window
    assert "fallback.target_age_s" in window


# ---------------------------------------------------------------------------
# WS-C: grace widen, trace resolution, teardown robustness, diagnostics.
# ---------------------------------------------------------------------------


def test_default_stair_widens_grace_and_hold_within_validators():
    """The no-arg defaults carry the widened loss stair and stay valid.

    The orchestrator spawns the servo with no stair args, so the file defaults
    are authoritative; grace 1.25->2.75 and hold 0.55->0.80 must satisfy the
    hold<grace and grace>=target_timeout validators.
    """

    settings = SERVO.DepthServoSettings()
    assert settings.tracking_hold_s == 0.80
    assert settings.tracking_loss_grace_s == 2.75
    assert settings.tracking_hold_s < settings.tracking_loss_grace_s
    assert settings.tracking_loss_grace_s >= settings.target_timeout_s


def test_argparse_stair_defaults_match_widened_grace(monkeypatch):
    """A no-stair-arg CLI spawn resolves the widened grace/hold defaults."""

    monkeypatch.setattr(sys, "argv", ["go2w_depth_servo.py", "--status-file", "/tmp/s.json"])
    args = SERVO._arguments()
    assert args.tracking_hold_s == 0.80
    assert args.tracking_loss_grace_s == 2.75


def test_max_target_capture_age_default_is_pinned_at_070():
    """Hard contract: the capture-freshness budget must remain 0.70 s."""

    assert SERVO.DepthServoSettings().max_target_capture_age_s == 0.70


def test_filter_stats_exposes_stale_data_rejections():
    """Stale-capture rejections are a first-class diagnostics field."""

    core = _reactive_core(target_timeout_s=0.40)
    assert core.filter_stats["stale_data_rejections"] == 0
    assert core.observe_target(x_m=0.0, y_m=0.0, z_m=1.5, stamp_s=1.0, capture_age_s=0.20)
    assert not core.observe_target(
        x_m=0.0, y_m=0.0, z_m=1.5, stamp_s=1.1, capture_age_s=1.2
    )
    assert core.filter_stats["stale_data_rejections"] == 1
    # A stale-capture rejection is also tallied in the overall rejection count
    # (pre-existing behaviour); stale_data_rejections isolates the freshness
    # cause from geometric-outlier rejections.
    assert core.filter_stats["rejected_outliers"] == 1
    core.reset()
    assert core.filter_stats["stale_data_rejections"] == 0


def test_view_period_update_accepts_only_in_band_cadence():
    """Only intervals inside the measured FFS band advance the EMA."""

    # First bundle (no prior arrival) leaves the seed period untouched.
    assert SERVO._view_period_update(
        previous_period_s=0.13, last_arrival_s=None, arrival_s=0.5, loss_phase_exit=False
    ) == 0.13
    # An in-band interval seeds a None period directly.
    assert SERVO._view_period_update(
        previous_period_s=None, last_arrival_s=1.00, arrival_s=1.13, loss_phase_exit=False
    ) == pytest.approx(0.13)
    # Band edges (0.10 and 0.20) are inclusive.
    assert SERVO._view_period_update(
        previous_period_s=None, last_arrival_s=0.0, arrival_s=0.10, loss_phase_exit=False
    ) == pytest.approx(0.10)
    assert SERVO._view_period_update(
        previous_period_s=None, last_arrival_s=0.0, arrival_s=0.20, loss_phase_exit=False
    ) == pytest.approx(0.20)


def test_view_period_update_rejects_out_of_band_and_loss_exit_intervals():
    """Gaps, jitter, and loss-phase exits never poison the damping period."""

    # Sub-band jitter and super-band gaps both leave the period unchanged.
    for arrival_s in (0.09, 0.25, 2.0):
        assert SERVO._view_period_update(
            previous_period_s=0.13,
            last_arrival_s=0.0,
            arrival_s=arrival_s,
            loss_phase_exit=False,
        ) == 0.13
    # An in-band interval that is a loss-phase exit is still skipped: the
    # interval spans the loss dwell and must not feed the damper.
    assert SERVO._view_period_update(
        previous_period_s=0.13,
        last_arrival_s=0.0,
        arrival_s=0.13,
        loss_phase_exit=True,
    ) == 0.13


def test_view_period_update_blends_with_light_weight():
    """An accepted in-band interval blends into the EMA at a 0.3 weight."""

    assert SERVO._view_period_update(
        previous_period_s=0.10, last_arrival_s=0.0, arrival_s=0.15, loss_phase_exit=False
    ) == pytest.approx(0.7 * 0.10 + 0.3 * 0.15)


def test_loss_stair_phase_set_matches_the_loss_stair():
    """Every loss/hold phase must be recognised as a loss-phase exit source."""

    for phase in (
        "waiting_target",
        "transform_unavailable",
        "tracking_lost",
        "reacquiring",
        "posture_blocked",
        "tracking_hold",
        "view_recovery",
        "search_required",
    ):
        assert phase in SERVO.LOSS_STAIR_PHASES
    # A live-motion phase is not a loss-phase exit.
    assert "approach" not in SERVO.LOSS_STAIR_PHASES
    assert "whole_body_approach" not in SERVO.LOSS_STAIR_PHASES


def test_trace_cadence_floor_is_5hz_in_motion_and_1hz_when_parked():
    """Non-terminal motion samples at 5 Hz; a parked/terminal servo at 1 Hz."""

    assert SERVO._trace_min_interval_s(terminal=False) == 0.20
    assert SERVO._trace_min_interval_s(terminal=True) == 1.0
    # The 5 Hz floor is a genuine drop from the retired >=1 s/row throttle.
    assert (
        SERVO._trace_min_interval_s(terminal=False)
        < SERVO._trace_min_interval_s(terminal=True)
    )


def test_trace_row_promotes_view_period_and_bundle_count():
    """view_update_period_s and the monotonic bundle counter are first-class."""

    document = {
        "updated_unix_ns": 1_700_000_000_000_000_000,
        "mode": "live",
        "phase": "whole_body_approach",
        "tracking": True,
        "target": {"x_m": 0.1, "y_m": 0.0, "z_m": 0.9, "frame_id": "cam"},
        "source_stamp_ns": 42,
        "output": {"phase": "whole_body_approach"},
        "filter": {"stale_data_rejections": 3},
        "posture_status": {"age_s": 0.1},
        "arm_view_status": {"age_s": 0.2},
        "whole_body": {"enabled": True},
    }

    row = SERVO._trace_row(document, view_update_period_s=0.132, bundle_count=57)

    assert row["schema"] == "z_manip.depth_servo_trace.v1"
    assert row["view_update_period_s"] == pytest.approx(0.132)
    assert row["bundle_count"] == 57
    # The status document fields are copied through verbatim.
    for key in (
        "updated_unix_ns",
        "mode",
        "phase",
        "tracking",
        "target",
        "source_stamp_ns",
        "output",
        "filter",
        "posture_status",
        "arm_view_status",
        "whole_body",
    ):
        assert row[key] == document[key]


def test_trace_row_carries_a_none_view_period_before_the_first_interval():
    """Before any cadence measurement the trace still emits the field as null."""

    document = {
        "updated_unix_ns": 1,
        "mode": "shadow",
        "phase": "waiting_target",
        "tracking": None,
        "target": None,
        "source_stamp_ns": None,
        "output": {},
        "filter": {},
        "posture_status": {},
        "arm_view_status": {},
        "whole_body": {},
    }
    row = SERVO._trace_row(document, view_update_period_s=None, bundle_count=0)
    assert "view_update_period_s" in row
    assert row["view_update_period_s"] is None
    assert row["bundle_count"] == 0


def test_tick_should_skip_on_stop_request_or_dead_context():
    """A tick after stop or a torn-down rcl context must no-op."""

    assert SERVO._tick_should_skip(stop_requested=True, ros_ok=True) is True
    assert SERVO._tick_should_skip(stop_requested=False, ros_ok=False) is True
    assert SERVO._tick_should_skip(stop_requested=True, ros_ok=False) is True
    assert SERVO._tick_should_skip(stop_requested=False, ros_ok=True) is False


def test_tick_and_teardown_wire_the_stop_event_and_context_guards():
    """The ROS node (uninstantiable without rclpy) wires the teardown guards.

    Proven by source inspection, mirroring the loss-stair guard test above.
    """

    source = Path(SERVO.__file__).read_text(encoding="utf-8")
    # The stop Event exists and the signal handler latches it.
    assert "self.stop_event = threading.Event()" in source
    assert "node.stop_event.set()" in source
    # _tick consults the stop Event + rclpy.ok() at its very top.
    tick_marker = source.index("def _tick(self) -> None:")
    tick_window = source[tick_marker:tick_marker + 400]
    assert "_tick_should_skip(" in tick_window
    assert "self.stop_event.is_set()" in tick_window
    assert "ros_ok=rclpy.ok()" in tick_window


def test_publish_swallows_shutdown_race_but_reraises_a_live_fault():
    """_publish_guarded gates on rclpy.ok() and only swallows a torn-down context.

    A still-valid context re-raises (fail-loud), matching the sibling passive
    joint-state bridge convention.
    """

    # Located by AST, not by an exact ``def`` line: the signature grew a
    # keyword-only ``final_stop`` (the stop mute's escape hatch) and the old
    # substring match broke on it, which is a brittle-test failure, not a
    # behaviour failure.
    source = Path(SERVO.__file__).read_text(encoding="utf-8")
    pub_window = ast.unparse(_function_ast("_publish_guarded"))
    # The context gate now lives in _publish_suppressed, which checks ros_ok
    # first; see test_a_tick_already_past_its_skip_check_is_muted_by_the_stop_latch.
    assert "ros_ok=rclpy.ok()" in pub_window
    assert "publisher.publish(message)" in pub_window
    assert "except Exception:" in pub_window
    # Genuine faults on a live context are re-raised, not hidden.
    assert "if rclpy.ok():" in pub_window
    assert "raise" in pub_window
    assert "def _publish_guarded(" in source


def test_every_publisher_goes_through_the_shutdown_guard():
    """The guard is worthless if a publisher can bypass it -- and one did.

    Only the velocity publisher was guarded originally.  The posture-intent and
    arm-view-intent publishers published raw, so an RCLError during teardown
    ("publisher's context is invalid") propagated out of the timer callback and
    killed the servo (live 2026-07-28: 5 crashes in one session).  The approach
    died ~0.4 s after spawn, planning_control saw ``search_required`` with a
    single bundle received, and burned its reacquisition budget on wrist
    searches -- presenting as a perception failure it was not.

    So assert the invariant on the WHOLE file: the only bare ``.publish(`` call
    allowed is the one inside the guard itself.
    """

    source = Path(SERVO.__file__).read_text(encoding="utf-8")
    bare = [
        line.strip()
        for line in source.splitlines()
        if ".publish(" in line and "_publish_guarded(" not in line
    ]
    # The guard's own call site is the single permitted bare publish.
    assert bare == ["publisher.publish(message)"], (
        f"publisher(s) bypassing the shutdown guard: {bare}"
    )


# ---------------------------------------------------------------------------
# STAGE 2 -- the servo process.
#
# R2 the tracking flag never arrives, R4 a duplicated staleness timer wearing a
# TF error message, R8 the servo kills itself on the way out.
# ---------------------------------------------------------------------------


EDGETAM_NODE = (
    ROOT / "ros2" / "z_manip_edgetam" / "z_manip_edgetam" / "node.py"
)


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _latched_qos_names(tree: ast.Module) -> set[str]:
    """Names bound to a ``QoSProfile(..., durability=...TRANSIENT_LOCAL)``."""

    latched: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "QoSProfile"
        ):
            continue
        for keyword in value.keywords:
            if keyword.arg != "durability":
                continue
            if (
                isinstance(keyword.value, ast.Attribute)
                and keyword.value.attr == "TRANSIENT_LOCAL"
            ):
                latched.add(target.id)
    return latched


def _subscription_qos_name(tree: ast.Module, *, topic_attr: str) -> str:
    """QoS argument name of the ``create_subscription`` on ``args.<attr>``."""

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_subscription"
            and len(node.args) >= 4
        ):
            continue
        topic = node.args[1]
        if not (
            isinstance(topic, ast.Attribute)
            and topic.attr == topic_attr
            and isinstance(topic.value, ast.Name)
            and topic.value.id == "args"
        ):
            continue
        qos = node.args[3]
        assert isinstance(qos, ast.Name), ast.dump(qos)
        return qos.id
    raise AssertionError(f"no create_subscription on args.{topic_attr}")


def _publisher_qos_name(tree: ast.Module, *, topic_parameter: str) -> str:
    """QoS argument name of the ``create_publisher`` on ``_topic('<param>')``."""

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_publisher"
            and len(node.args) >= 3
        ):
            continue
        topic = node.args[1]
        if not (
            isinstance(topic, ast.Call)
            and isinstance(topic.func, ast.Attribute)
            and topic.func.attr == "_topic"
            and len(topic.args) == 1
            and isinstance(topic.args[0], ast.Constant)
            and topic.args[0].value == topic_parameter
        ):
            continue
        qos = node.args[2]
        assert isinstance(qos, ast.Name), ast.dump(qos)
        return qos.id
    raise AssertionError(f"no create_publisher on _topic({topic_parameter!r})")


def test_tracking_subscription_durability_matches_the_edgetam_publisher():
    """R2(a).  ``/track_3d/is_tracking`` must be read LATCHED, like it is sent.

    The publisher offers RELIABLE + TRANSIENT_LOCAL and its README states the
    contract outright: "A downstream controller that starts after a failure
    therefore observes the latest false state".  The VLM bridge subscribes
    latched.  This servo shared one VOLATILE ``qos`` object with five other
    topics, and a VOLATILE reader against a TRANSIENT_LOCAL writer still
    MATCHES (durability is request<=offered) -- it just silently forfeits the
    latched sample, with no warning anywhere.

    Consequence, recorded: in every trace row at ``bundle_count == 1``
    (artifacts/go2w_real/latest/depth-servo.trace.jsonl{,.1}, 12 of 12) the
    servo reports ``phase=search_required`` with ``tracking=null``.  With no
    flag, ``fresh_tracking`` is False on the first bundle, so DepthServoCore
    passes ``None`` geometry into ReactiveTargetController.update(); its
    ``_lost()`` finds its OWN ``_last_geometry`` still None and returns
    SEARCH_REQUIRED with zero grace -- while the core is holding a perfectly
    good geometry that decision cannot see.  That is symptom B's first half.

    Asserted across BOTH files so drift on either side fails: this is exactly
    the shape of defect the phase-vocabulary audit found, one contract written
    twice with nothing forcing the copies to agree.
    """

    servo = _module_ast(Path(SERVO.__file__))
    edgetam = _module_ast(EDGETAM_NODE)

    servo_qos = _subscription_qos_name(servo, topic_attr="tracking_topic")
    assert servo_qos in _latched_qos_names(servo), (
        f"the servo subscribes to /track_3d/is_tracking with {servo_qos!r}, "
        "which is not a TRANSIENT_LOCAL QoSProfile; a freshly started servo "
        "will never receive the publisher's latched sample"
    )

    publisher_qos = _publisher_qos_name(edgetam, topic_parameter="tracking_topic")
    assert publisher_qos in _latched_qos_names(edgetam), (
        "the EdgeTAM publisher stopped offering TRANSIENT_LOCAL; the servo's "
        "latched subscription is now pointless and R2 needs re-deciding"
    )

    # The shared VOLATILE profile must not be what carries this subscription.
    assert servo_qos != _subscription_qos_name(servo, topic_attr="target_topic")


def test_tracking_flag_expires_when_the_publisher_goes_quiet():
    """A flag we stop hearing refreshed degrades to UNKNOWN, never to True.

    RENAMED, and the rename is the point.  This was
    ``..._so_a_historical_true_cannot_drive``, which claimed a property the
    function does not have: ``_tracking`` stamps ``tracking_received_s`` at
    CALLBACK time, and a TRANSIENT_LOCAL historical sample is delivered to a
    late-joining reader at MATCH time, so its receipt age is ~0 by
    construction.  The ``age_s=90.0`` row below is a value the deployed path
    cannot produce for a latched sample; it is kept only as a monotonicity
    check, not as evidence for a scenario the runtime can generate.

    What bounds a retained ``True`` is that the selected-target cloud is NOT
    latched, so it can never pair with stale geometry.  What THIS covers is the
    case receipt time can genuinely see: a publisher that matched and then went
    quiet.  Fail-closed in the only direction that matters -- it turns True
    into None and can never turn None or False into True.
    """

    ttl_s = 0.40

    # Fresh: acted on, in both polarities.
    assert SERVO._aged_tracking_flag(True, age_s=0.0, ttl_s=ttl_s) is True
    assert SERVO._aged_tracking_flag(True, age_s=0.39, ttl_s=ttl_s) is True
    assert SERVO._aged_tracking_flag(False, age_s=0.39, ttl_s=ttl_s) is False

    # Past the TTL: UNKNOWN, not True.  ``None`` and not ``False`` because
    # every gate spells the positive test (``tracking is True``), so they are
    # equivalent at the gates while ``None`` is the honest status report.
    assert SERVO._aged_tracking_flag(True, age_s=0.41, ttl_s=ttl_s) is None
    assert SERVO._aged_tracking_flag(True, age_s=90.0, ttl_s=ttl_s) is None
    assert SERVO._aged_tracking_flag(False, age_s=0.41, ttl_s=ttl_s) is None

    # Never received, or received at an unusable time: also UNKNOWN.
    assert SERVO._aged_tracking_flag(None, age_s=0.0, ttl_s=ttl_s) is None
    assert SERVO._aged_tracking_flag(True, age_s=None, ttl_s=ttl_s) is None
    assert SERVO._aged_tracking_flag(True, age_s=math.nan, ttl_s=ttl_s) is None


def test_tracking_flag_ttl_is_measured_but_never_tighter_than_target_timeout():
    """R2(b) sizing.  Three measured bundle periods, floored at the target budget.

    EdgeTAM publishes the flag from ``_publish_observation``, in the same call
    as the selected-target cloud, so the flag cadence IS the bundle cadence and
    ``view_update_period_s`` is the right ruler.

    The floor is the safety property.  This TTL exists to reject a sample that
    is seconds-to-minutes old; it must never become a NEW, tighter stop
    condition on the same decision ``target_timeout_s`` already governs, or
    R2(b) starts costing approaches instead of protecting them.
    """

    target_timeout_s = 0.40

    # Before any cadence has been measured, the nominal FFS period is used.
    assert SERVO._tracking_flag_ttl_s(
        view_update_period_s=None,
        target_timeout_s=target_timeout_s,
    ) == pytest.approx(
        max(
            target_timeout_s,
            SERVO.TRACKING_FLAG_STALE_PERIODS * SERVO.TRACKING_FLAG_NOMINAL_PERIOD_S,
        )
    )

    # A slower measured cadence stretches the TTL with it.
    assert SERVO._tracking_flag_ttl_s(
        view_update_period_s=SERVO.VIEW_UPDATE_PERIOD_MAX_INTERVAL_S,
        target_timeout_s=target_timeout_s,
    ) == pytest.approx(0.60)

    # THE FLOOR.  ``view_update_period_s`` is band-clamped to [0.10, 0.20], so
    # walk the whole reachable range plus the degenerate inputs.
    degenerate = [None, 0.0, -1.0, math.nan, math.inf]
    measured = [0.10, 0.12, 0.133, 0.15, 0.20]
    for period in degenerate + measured:
        ttl_s = SERVO._tracking_flag_ttl_s(
            view_update_period_s=period,
            target_timeout_s=target_timeout_s,
        )
        assert ttl_s >= target_timeout_s, (
            f"period {period!r} yields a {ttl_s}s tracking-flag TTL, tighter "
            f"than the {target_timeout_s}s target timeout that already gates "
            "the same decision; R2(b) must not manufacture new stops"
        )


def test_one_dropped_camera_frame_no_longer_zeroes_the_base_blaming_tf():
    """R4.  The geometry budget was a duplicate of the target budget, set tighter.

    ``observe_target`` writes ``self._target_received_s = float(stamp_s)`` and,
    on the transforms-available branch, ``self._transforms_received_s =
    float(stamp_s)`` -- the same value, from one camera callback; the
    not-available branch nulls the geometry outright.  Geometry non-None
    therefore IMPLIES the two receipt stamps are equal, so ``transform_age_s``
    is IDENTICALLY ``target_age_s``.

    Shipped, that duplicate was compared against 0.25 s while the real target
    budget was 0.40 s, at a ~0.133 s bundle cadence.  TWO frame periods
    (0.265 s) therefore hard-zeroed the base and reported
    ``transform_unavailable`` -- a TF outage -- while printing the target age
    and while the runtime observer was publishing perfectly fresh kinematics.

    Constructed the way the LAUNCHER constructs it: pin the flags
    go2w_depth_servo.sh actually passed, and leave the geometry budget at its
    default, because the launcher never passed one.
    """

    settings = SERVO.DepthServoSettings(
        mode="live",
        target_timeout_s=0.40,
        tracking_hold_s=0.80,
        tracking_loss_grace_s=2.75,
        handoff_settle_s=0.30,
    )
    core = SERVO.DepthServoCore(settings)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 1.10),
        base_xyz=(1.20, 0.0, -0.10),
        arm_xyz=(1.05, 0.0, 0.10),
        stamp_s=10.0,
    )

    # One dropped frame: the next bundle lands two cadence periods later.
    output = core.tick(now_s=10.0 + 2 * 0.133, tracking=True)

    assert output.phase != "transform_unavailable", (
        f"a {2 * 0.133:.3f}s gap -- one dropped camera frame, well inside the "
        f"{settings.target_timeout_s}s target budget -- still reports a TF "
        f"outage: {output.reason!r}"
    )
    assert output.target_age_s == pytest.approx(0.266)


def test_geometry_staleness_budget_may_not_be_tighter_than_the_target_budget():
    """R4's structural fix: the inversion is now unconstructible.

    Same shape as the existing ``tracking_loss_grace_s >= target_timeout_s``
    check.  Equality is legal and is what ships -- the geometry cannot be
    fresher than the observation it was derived from, so its budget can
    confirm the target-freshness verdict but must never pre-empt it.
    """

    with pytest.raises(ValueError, match="geometry staleness timeout"):
        SERVO.DepthServoSettings(
            target_timeout_s=0.40,
            geometry_staleness_timeout_s=0.25,
        )

    equal = SERVO.DepthServoSettings(
        target_timeout_s=0.40,
        geometry_staleness_timeout_s=0.40,
    )
    assert equal.geometry_staleness_timeout_s == equal.target_timeout_s

    defaults = SERVO.DepthServoSettings()
    assert defaults.geometry_staleness_timeout_s >= defaults.target_timeout_s


def test_blocked_geometry_still_fails_closed_on_a_missing_transform():
    """R4 explicitly does NOT delete the guard.

    ``transform_unavailable`` sits in the whole-body bypass set whose comment
    records live 2026-07-24: raw_y swept +/-0.67 m sinusoidally while the
    source stamp froze and the arm chased its own motion.  The AGE term was the
    duplicate; the ``self._geometry is None`` term is the protection, and a
    fresh observation whose TF lookup failed must still zero the base.
    """

    core = _reactive_core(target_timeout_s=0.40)
    core.observe_target(
        x_m=0.0,
        y_m=0.1,
        z_m=0.90,
        stamp_s=20.0,
        transform_error="base_link TF unavailable",
    )

    output = core.tick(now_s=20.01, tracking=True)

    assert output.phase == "transform_unavailable"
    assert output.published_linear_x == output.published_angular_z == 0.0
    assert core.geometry is None


def test_every_stair_setting_is_pinned_by_the_launcher():
    """The test commit c208a6d could not write, and why it missed R4.

    That test asserted the launcher's loss-stair values matched the Python
    defaults -- but it could only ever assert on flags the launcher ALREADY
    passed, so it covered hold/grace/target-timeout and was structurally
    incapable of noticing the fourth budget, which had no flag in the launcher
    at all and sat at an invisible 0.25 s.

    This one walks the mapping from the servo module and every duration field
    of the settings dataclass, so a stair knob that nobody pins fails here
    instead of on the robot.
    """

    launcher = LAUNCHER.read_text(encoding="utf-8")
    defaults = {
        field.name: field.default
        for field in dataclasses.fields(SERVO.DepthServoSettings)
    }

    # Duration fields deliberately outside the freshness/loss stair.  Adding a
    # new ``*_s`` field forces a decision here rather than allowing silence.
    not_stair = {
        # VisualServoController convergence dwell on the legacy optical path,
        # which deployed construction never enables
        # (allow_legacy_optical_depth_for_tests stays False).
        "settle_time_s",
    }
    duration_fields = {name for name in defaults if name.endswith("_s")}
    unclassified = duration_fields - set(SERVO.STAIR_SETTING_FLAGS) - not_stair
    assert not unclassified, (
        f"new duration setting(s) {sorted(unclassified)} are neither pinned in "
        "STAIR_SETTING_FLAGS nor explicitly exempted; classify them or an "
        "operator reading the launcher cannot see them"
    )

    for name, flag in SERVO.STAIR_SETTING_FLAGS.items():
        assert name in defaults, f"STAIR_SETTING_FLAGS names a dead field {name!r}"
        match = re.search(rf"{re.escape(flag)} (\S+)", launcher)
        assert match is not None, (
            f"go2w_depth_servo.sh does not pass {flag}, so the {name!r} budget "
            "is invisible to whoever reads the launcher"
        )
        assert float(match.group(1)) == pytest.approx(defaults[name]), (
            f"launcher pins {flag} at {match.group(1)} but the Python default "
            f"for {name} is {defaults[name]}"
        )

    # And the launcher's ACTUAL command line must parse and land those values.
    # Asserting the flag text alone would still miss a spelling that argparse
    # rejects, or one that argparse accepts into a field nothing reads.
    settings = SERVO._settings_from_args(SERVO._arguments(_launcher_argv()))
    for name in SERVO.STAIR_SETTING_FLAGS:
        assert getattr(settings, name) == pytest.approx(defaults[name])


def _launcher_argv() -> list[str]:
    """The servo argv go2w_depth_servo.sh builds, with shell values stubbed."""

    launcher = LAUNCHER.read_text(encoding="utf-8")
    start = launcher.index('python3 "$SCRIPT_DIR/go2w_depth_servo.py"')
    lines: list[str] = []
    for line in launcher[start:].splitlines():
        lines.append(line)
        if not line.rstrip().endswith("\\"):
            break
    tokens = " ".join(lines).replace("\\", "").split()[2:]
    # ``--mode`` is a choices= flag, so its stub has to be a legal mode.
    return [
        ("live" if token == '"$MODE"' else "/tmp/launcher-stub")
        if "$" in token
        else token
        for token in tokens
    ]


def test_launcher_command_line_parses_into_the_shipped_settings():
    """The whole launcher argv, not a hand-picked subset of its flags.

    Every previous launcher test read the file as TEXT and asserted substrings.
    That cannot catch a flag argparse would reject, and it cannot catch a flag
    argparse accepts into a field ``_settings_from_args`` never reads.
    """

    settings = SERVO._settings_from_args(SERVO._arguments(_launcher_argv()))

    assert settings.mode == "live"
    assert settings.target_timeout_s == 0.40
    assert settings.geometry_staleness_timeout_s == 0.40
    # THE R4 ORDERING, evaluated on what the robot is actually launched with.
    assert settings.geometry_staleness_timeout_s >= settings.target_timeout_s


def test_tf_lookup_blocking_wait_is_capped_independently_of_the_staleness_budget():
    """R4's fourth-consumer trap: one number sized two unrelated things.

    ``_target_transforms`` takes TWO sequential blocking ``lookup_transform``
    calls on the node's single-threaded executor -- the same thread as the
    20 Hz control tick.  Sizing that wait from the geometry staleness budget
    meant relaxing a freshness COMPARISON from 0.25 to 0.40 would have raised
    worst-case in-callback BLOCKING from 0.50 s to 0.80 s, sixteen missed
    ticks, as an invisible side effect.
    """

    settings = SERVO.DepthServoSettings()
    per_lookup_s = min(settings.geometry_staleness_timeout_s, SERVO.TF_LOOKUP_TIMEOUT_S)

    assert per_lookup_s == pytest.approx(0.25)
    assert 2 * per_lookup_s <= 0.50, (
        "worst-case in-callback blocking on the single-threaded executor rose "
        "above the 0.50s it was before the staleness budget was widened"
    )

    source = Path(SERVO.__file__).read_text(encoding="utf-8")
    window = source[source.index("query_time = Time.from_msg"):][:600]
    assert "TF_LOOKUP_TIMEOUT_S" in window, (
        "the tf2 lookup Duration is no longer capped by its own constant"
    )


def _signal_handler_ast() -> ast.FunctionDef:
    for node in ast.walk(_module_ast(Path(SERVO.__file__))):
        if isinstance(node, ast.FunctionDef) and node.name == "request_stop":
            return node
    raise AssertionError("the SIGTERM/SIGINT handler is gone")


def test_signal_handler_only_sets_flags_and_never_re_enters_rcl():
    """R8.  The pattern to remove is doing WORK in the handler.

    The handler ran ``node.stop()`` -- three zero-Twist publishes plus a status
    file write -- and then ``rclpy.shutdown()``, from a signal delivered while
    ``rclpy.spin()`` was on the stack and midway through using those same
    objects.  One recorded log carries nine tracebacks from that single
    teardown: 5x RCLError at publisher.c:423, 2x InvalidHandle, 1x
    ``AttributeError: 'NoneType' object has no attribute 'trigger'`` (a guard
    condition freed under the executor), 1x FileNotFoundError.

    Commit 6a3f75d guarded the publishes.  That removed the printed text and
    left the re-entrancy: the handler was still touching rcl objects the spin
    thread owned.  No try/except can fix that, because the exception is not the
    bug -- the work is.  So this asserts the SHAPE, not the symptom.
    """

    handler = _signal_handler_ast()
    calls = [
        node for node in ast.walk(handler) if isinstance(node, ast.Call)
    ]
    offenders = [
        ast.unparse(call)
        for call in calls
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "set")
    ]
    assert not offenders, (
        "the signal handler does work again instead of only latching flags: "
        f"{offenders}"
    )
    assert calls, "the handler no longer latches anything"


def test_shutdown_work_happens_on_the_main_thread_after_the_spin_loop():
    """R8's other half: the work has to actually still happen, just later.

    Removing it from the handler is only correct if the final zero-Twist, the
    final status write, ``destroy_node`` and ``shutdown`` all run once the loop
    has returned -- and if a handler that can no longer wake a blocking wait is
    paired with a loop that only waits in bounded steps.  ``rclpy.spin()``
    blocks unboundedly, and installing Python handlers over SIGINT/SIGTERM
    displaces the C-level handler ``rclpy.init()`` registered to break it.
    """

    source = Path(SERVO.__file__).read_text(encoding="utf-8")
    shutdown = source[source.index("    rclpy.init()"):]

    assert "while rclpy.ok() and not stopped.is_set():" in shutdown
    assert "executor.spin_once(timeout_sec=SHUTDOWN_POLL_INTERVAL_S)" in shutdown
    code = [
        line for line in shutdown.splitlines() if not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if "rclpy.spin(node)" in line], (
        "an unbounded blocking spin cannot be woken by a flag-only handler"
    )
    for expected in (
        "node.stop(_shutdown_phase(stop_requested=stopped.is_set()))",
        "node.destroy_node()",
        "rclpy.shutdown()",
    ):
        assert expected in shutdown.split("finally:", 1)[1], expected

    # A poll interval no coarser than one control tick keeps the delay to the
    # final zero-Twist inside a tick period.
    assert 0.0 < SERVO.SHUTDOWN_POLL_INTERVAL_S <= 0.05
    assert SERVO._shutdown_phase(stop_requested=True) == "stopped"
    assert SERVO._shutdown_phase(stop_requested=False) == "exited"


def _innermost_function_names(predicate) -> list[str]:
    """Names of the innermost functions containing a node matching *predicate*."""

    hits: list[str] = []

    def visit(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            if predicate(child):
                hits.append(current)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
            else:
                visit(child, current)

    visit(_module_ast(Path(SERVO.__file__)), "<module>")
    return hits


def _function_ast(name: str) -> ast.FunctionDef:
    for node in ast.walk(_module_ast(Path(SERVO.__file__))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone")


def test_a_tick_already_past_its_skip_check_is_muted_by_the_stop_latch():
    """R8 REGRESSION.  SIGTERM must mute an IN-FLIGHT tick, not just a queued one.

    The bug this pins: R8 reduced the signal handler to two flag sets, which is
    right, but the inline ``rclpy.shutdown()`` it deleted had been doing a
    second, unadvertised job -- flipping ``rclpy.ok()`` false, which every
    publish site gates on.  Removing it removed the mute.  ``_tick`` checks
    ``_tick_should_skip`` ONCE, as its first statement; Python delivers signals
    on the main thread at a bytecode boundary and ``_tick`` runs on the main
    thread under the executor.  So a SIGTERM landing after that check let the
    tick run to completion and publish the full commanded base velocity (up to
    --max-forward-mps 0.18) plus a posture intent and an arm-view intent, all
    after the stop was requested -- and the delay to the final zero was bounded
    by the remaining tick (whole-body solve + status write), not by
    SHUTDOWN_POLL_INTERVAL_S.

    Sequence, in order, with the values the runtime actually holds.
    """

    # 1. The tick starts.  Nothing has been requested; it proceeds.
    assert SERVO._tick_should_skip(stop_requested=False, ros_ok=True) is False

    # 2. SIGTERM lands mid-tick.  The handler sets the flags and returns; it is
    #    forbidden from touching rcl, so rclpy.ok() STAYS TRUE.
    stop_requested, ros_ok = True, True

    # 3. The tick resumes and reaches its terminal publishes.  Every one of
    #    them must now be dropped.
    assert SERVO._publish_suppressed(
        stop_requested=stop_requested, ros_ok=ros_ok, final_stop=False
    ) is True, (
        "a tick already past its skip check can still publish a motion command "
        "after SIGTERM; the stop latch is not read at the publish sites"
    )

    # 4. ...but the teardown's own zero-Twist must still go out.  A blanket
    #    stop_event gate would have silenced it and left the base to the
    #    transport watchdog, which is the worse bug.
    assert SERVO._publish_suppressed(
        stop_requested=True, ros_ok=True, final_stop=True
    ) is False, "the terminal stop can no longer publish its zero Twist"

    # 5. A dead context wins over everything, final_stop included.
    assert SERVO._publish_suppressed(
        stop_requested=False, ros_ok=False, final_stop=True
    ) is True
    assert SERVO._publish_suppressed(
        stop_requested=False, ros_ok=False, final_stop=False
    ) is True

    # 6. Steady state is unchanged.
    assert SERVO._publish_suppressed(
        stop_requested=False, ros_ok=True, final_stop=False
    ) is False


def test_every_publish_site_gates_on_the_stop_latch_not_only_on_rclpy_ok():
    """The mute has to be SPELLED OUT, because that is how it got deleted.

    This repo's characteristic failure: the value that decides the transition
    lives in a file -- here, a function -- the transition never names.  The
    publish sites named only ``rclpy.ok()``; ``stop_event`` was read solely at
    the top of ``_tick``.  Asserting the source shape is the only check
    available (rclpy is not importable here, so the node cannot be built), and
    it is the check that would have caught the regression.
    """

    for name in ("_publish", "_publish_guarded"):
        body = ast.unparse(_function_ast(name))
        assert "_publish_suppressed(" in body, (
            f"{name} no longer routes its gate through _publish_suppressed"
        )
        assert "self.stop_event.is_set()" in body, (
            f"{name} does not consult the stop latch; a tick already past "
            "_tick_should_skip can publish after SIGTERM"
        )
        assert "final_stop" in body, f"{name} lost the terminal-stop escape hatch"

    # And nothing bypasses the guard: the raw .publish() call exists once, and
    # only inside _publish_guarded.
    raw = _innermost_function_names(
        lambda node: isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "publish"
    )
    assert raw == ["_publish_guarded"], (
        f"a publisher bypasses the guarded/muted path: {raw}"
    )


def test_only_the_terminal_stop_may_publish_through_the_stop_mute():
    """``final_stop=True`` is an escape hatch; exactly one caller may use it.

    If a tick path ever passes it, the mute is decorative again.
    """

    callers = _innermost_function_names(
        lambda node: isinstance(node, ast.keyword)
        and node.arg == "final_stop"
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
    )
    assert sorted(set(callers)) == ["stop"], (
        "final_stop=True is passed outside the terminal stop: "
        f"{sorted(set(callers))}"
    )
    # ...and the teardown really does go through that one caller.
    source = Path(SERVO.__file__).read_text(encoding="utf-8")
    finally_block = source.split("    rclpy.init()", 1)[1].split("finally:", 1)[1]
    assert "node.stop(_shutdown_phase(stop_requested=stopped.is_set()))" in finally_block
    assert "executor.shutdown()" in finally_block, (
        "the executor's guard condition is never released before the context goes"
    )


def test_atomic_status_write_scratch_path_is_unique_not_the_container_pid(
    tmp_path, monkeypatch
):
    """R8's second half.  ``.{name}.{getpid()}.tmp`` is a compile-time constant here.

    The servo is ALWAYS pid 1 inside its container, so that expression always
    renders ``.depth-servo.json.1.tmp`` -- and a pid namespace does not make
    the HOST path unique.  The status directory is bind-mounted read-write into
    every runtime container, so a second servo or replay container (also pid 1)
    writes the identical scratch path on the same host directory: one truncates
    the other's half-written file and ``os.replace`` publishes a torn document,
    which every reader parses as ``JSONDecodeError -> {}``, i.e. "the servo is
    not running".
    """

    status = tmp_path / "depth-servo.json"
    seen: list[str] = []
    real_replace = os.replace

    def _record(source, destination):
        seen.append(Path(source).name)
        real_replace(source, destination)

    monkeypatch.setattr(SERVO.os, "replace", _record)
    # Same process, therefore the same pid: uniqueness may not come from it.
    SERVO._atomic_json(status, {"a": 1})
    SERVO._atomic_json(status, {"a": 2})

    assert len(set(seen)) == 2, (
        f"two writes from one pid reused the scratch path {seen[0]!r}; a "
        "second container is also pid 1 and would collide on it"
    )
    assert all(name != ".depth-servo.json.1.tmp" for name in seen)
    assert json.loads(status.read_text(encoding="utf-8")) == {"a": 2}
    # A unique name no longer self-cleans by being overwritten, so nothing may
    # be left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["depth-servo.json"]


def test_atomic_status_write_removes_its_scratch_file_when_the_publish_fails(
    tmp_path, monkeypatch
):
    """The cost of uniqueness, paid back.

    A constant scratch name self-cleaned: the next tick overwrote whatever the
    last failure left behind.  A per-write name does not, and this runs at
    20 Hz, so a repeating publish failure (a full or read-only status volume --
    the same directory a magnetometer-log runaway filled to zero bytes free
    once already) would paper the directory with orphans on top of it.  The
    failure must be injected AFTER the scratch file exists, which is exactly
    where the old code had nothing.
    """

    status = tmp_path / "depth-servo.json"

    def _explode(source, destination):
        raise OSError("read-only file system")

    monkeypatch.setattr(SERVO.os, "replace", _explode)
    with pytest.raises(OSError):
        SERVO._atomic_json(status, {"a": 1})

    assert list(tmp_path.iterdir()) == [], (
        f"scratch file left behind at 20 Hz: {[p.name for p in tmp_path.iterdir()]}"
    )


# ---------------------------------------------------------------------------
# R5 hazard -- extending VIEW_RECOVERY triples the life of a phase whose
# decision carries an "active wrist sweep".  What actually bounds it?
# ---------------------------------------------------------------------------


def test_no_arm_motion_is_commanded_while_the_servo_is_on_the_loss_stair():
    """WHAT BOUNDS THE SWEEP'S GEOMETRY: nothing executes it.

    R5 lets ``view_recovery`` live for the servo's full configured
    ``tracking_loss_grace_s`` (2.75 s) instead of being cut at
    ``tracking_hold_s`` (0.80 s).  ``ReactiveTargetController._lost`` returns
    ``ArmViewIntent(mode=SEARCH)`` from that branch, and ``view_recovery``
    bypasses the whole-body branch and therefore ``FixedSelfCollisionGuard``
    entirely -- and the recorded gripper-to-LiDAR margin reached 0.6 mm.  A
    2.75 s UNGUARDED sweep would not be free.

    It is not a sweep.  The chain, pinned here in all three links:

    1. ``_whole_body_output`` sets ``self.whole_body_command = None`` on entry
       and returns the fallback unchanged for any ``LOSS_STAIR_PHASES`` member;
       ``view_recovery`` is a member.
    2. ``_publish_arm_view_intent`` returns immediately when
       ``self.whole_body_command is None``.  It is the ONLY writer of the
       joint-velocity intent topic the PiPER executor obeys.
    3. The executor holds pose once the last intent ages past
       ``MAX_INTENT_AGE_S``.

    So the SEARCH mode reaches the status document (and
    ``ownership_snapshot`` labels it ``intent_only``) and reaches no actuator.
    If someone later publishes an arm intent from the loss stair, this test
    fails and the 2.75 s window has to be re-argued.
    """

    assert ServoPhase.VIEW_RECOVERY.value in SERVO.LOSS_STAIR_PHASES
    assert ServoPhase.ACQUIRING.value in SERVO.LOSS_STAIR_PHASES

    whole_body = ast.unparse(_function_ast("_whole_body_output"))
    assert "self.whole_body_command = None" in whole_body
    assert "fallback.phase in LOSS_STAIR_PHASES" in whole_body

    publish_intent = ast.unparse(_function_ast("_publish_arm_view_intent"))
    assert "command = self.whole_body_command" in publish_intent
    assert "command is None" in publish_intent
    guard_index = publish_intent.index("command is None")
    publish_index = publish_intent.index("_publish_guarded")
    assert guard_index < publish_index, (
        "the arm-view intent is published before the whole-body-command guard; "
        "a loss-stair phase can now command the wrist"
    )

    # The intent topic has exactly one writer.
    writers = _innermost_function_names(
        lambda node: isinstance(node, ast.Attribute)
        and node.attr == "arm_view_intent_publisher"
    )
    # ``__init__`` creates it; only ``_publish_arm_view_intent`` uses it.
    assert set(writers) == {"__init__", "_publish_arm_view_intent"}, writers

    executor = ast.parse(
        (ROOT / "scripts" / "runtime" / "piper_reactive_view_executor.py")
        .read_text(encoding="utf-8")
    )
    limits = {
        node.targets[0].id: node.value
        for node in ast.walk(executor)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert "MAX_INTENT_AGE_S" in limits
    assert float(ast.literal_eval(limits["MAX_INTENT_AGE_S"])) <= 1.0, (
        "the executor's intent lease grew; a stale SEARCH intent could now "
        "outlive the loss-stair tick that produced it"
    )


def test_the_status_document_reports_the_servos_own_loss_stair_limits():
    """R5: the supervisor must read ONE number, not write a second one.

    ``go2w_depth_servo.sh`` passes ``--tracking-hold-s 0.80`` and
    ``--tracking-loss-grace-s 2.75``.  The supervisor used to act on the phase
    NAME ``view_recovery``, which the servo enters at 0.80 s, so the 2.75 s was
    unreachable dead config.  It now derives its bound from
    ``status["limits"]["tracking_loss_grace_s"]``, so this key is a contract.
    """

    write_status = ast.unparse(_function_ast("_write_status"))
    assert "'limits'" in write_status
    for field in (
        "tracking_loss_grace_s",
        "tracking_hold_s",
        "target_timeout_s",
    ):
        assert f"settings.{field}" in write_status, (
            f"the status document stopped reporting {field}; the supervisor's "
            "derived deadline silently falls back to a hard-coded constant"
        )

    # The reader on the other side names the same path.
    supervision = ast.parse(
        (ROOT / "scripts" / "runtime" / "go2w_reactive_supervision.py")
        .read_text(encoding="utf-8")
    )
    reader = next(
        node
        for node in ast.walk(supervision)
        if isinstance(node, ast.FunctionDef)
        and node.name == "servo_tracking_loss_grace_s"
    )
    source = ast.unparse(reader)
    assert "'limits'" in source and "'tracking_loss_grace_s'" in source


def test_acquiring_is_emitted_before_the_first_bundle_not_search_required():
    """Symptom B, at the source: the first tick of every session.

    In the recorded corpus every row at ``bundle_count == 1`` is
    ``phase=search_required`` with ``tracking=null`` (12 of 12,
    artifacts/go2w_real/latest/depth-servo.trace.jsonl{,.1}).  The controller
    had never been handed a geometry, so ``_lost()`` fell into its
    ``_last_geometry is None`` branch and answered with the loss stair's most
    severe verdict.
    """

    core = _reactive_core(target_timeout_s=0.40)
    # One good bundle has landed -- the core HAS a geometry -- but the latched
    # ``/track_3d/is_tracking`` flag has not arrived, so ``tracking`` is None
    # and the geometry never reaches the controller.  That is the recorded row
    # exactly.
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.10, 0.80),
        base_xyz=(0.85, 0.0, -0.20),
        arm_xyz=(0.70, 0.0, 0.05),
        stamp_s=1.0,
    )

    output = core.tick(now_s=1.10, tracking=None)

    assert output.phase == ServoPhase.ACQUIRING.value
    assert output.phase != ServoPhase.SEARCH_REQUIRED.value
    assert output.published_linear_x == output.published_angular_z == 0.0
    assert output.proposed_linear_x == output.proposed_angular_z == 0.0
    # The wrist holds the pose perception seeded the tracker from; the old
    # answer asked it to SEARCH, i.e. to sweep off that very target.
    assert core.reactive_status["arm_view"]["mode"] == "hold"
    # The controller's own phase, not just the presentation remap.
    assert core.reactive_status["phase"] == ReactivePhase.ACQUIRING.value
