from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
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
            tracking_loss_grace_s=0.75,
            allow_legacy_optical_depth_for_tests=True,
        )
    )


def _reactive_core(
    *,
    mode: str = "live",
    target_timeout_s: float = 0.25,
    transform_timeout_s: float = 0.25,
):
    return SERVO.DepthServoCore(
        SERVO.DepthServoSettings(
            mode=mode,
            desired_depth_m=0.50,
            handoff_depth_m=0.62,
            target_timeout_s=target_timeout_s,
            tracking_loss_grace_s=max(0.75, target_timeout_s),
            transform_timeout_s=transform_timeout_s,
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
    core = _reactive_core(target_timeout_s=1.0, transform_timeout_s=0.25)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.0, 0.80),
        base_xyz=(0.90, 0.0, -0.10),
        arm_xyz=(0.75, 0.0, 0.10),
        stamp_s=4.0,
    )
    output = core.tick(now_s=4.30, tracking=True)

    assert output.phase == "transform_unavailable"
    assert output.published_linear_x == output.published_angular_z == 0.0
    assert "stale" in output.reason


def test_tracking_loss_with_stale_tf_reports_tracker_recovery_not_tf_outage():
    core = _reactive_core(target_timeout_s=1.0, transform_timeout_s=0.25)
    assert _observe_in_frames(
        core,
        camera_xyz=(0.0, 0.20, 0.80),
        base_xyz=(0.75, 0.0, -0.30),
        arm_xyz=(0.60, 0.0, -0.15),
        stamp_s=5.0,
    )

    output = core.tick(now_s=5.30, tracking=False)

    assert output.phase == "search_required"
    assert output.published_linear_x == output.published_angular_z == 0.0
    assert core.reactive_status is not None
    assert core.reactive_status["phase"] == "search_required"
    assert "search" in output.reason


def test_stale_target_with_tracking_true_is_tracking_loss_not_tf_outage():
    core = _reactive_core(target_timeout_s=0.25, transform_timeout_s=0.25)
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
