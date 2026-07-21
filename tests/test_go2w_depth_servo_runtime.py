from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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
        )
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
