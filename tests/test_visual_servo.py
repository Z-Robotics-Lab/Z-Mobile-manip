import math

import pytest

from z_manip.control.visual_servo import VisualServoConfig, VisualServoController


def test_servo_commands_are_bounded_and_turn_toward_target():
    controller = VisualServoController(VisualServoConfig())

    command = controller.update((0.35, 0.0, 1.8), stamp_s=1.0)

    assert 0.0 < command.linear_x <= controller.config.max_forward_mps
    assert -controller.config.max_yaw_rps <= command.angular_z < 0.0
    assert not command.converged


def test_lost_target_immediately_commands_zero():
    controller = VisualServoController(VisualServoConfig())
    controller.update((-0.2, 0.0, 1.4), stamp_s=1.0)

    command = controller.update(None, stamp_s=1.1)

    assert command.linear_x == 0.0
    assert command.angular_z == 0.0
    assert not command.converged


def test_convergence_requires_continuous_sim_time_window():
    config = VisualServoConfig(settle_time_s=2.0)
    controller = VisualServoController(config)
    target = (0.01, 0.0, config.desired_depth_m + 0.01)

    assert not controller.update(target, stamp_s=4.0).converged
    assert not controller.update(target, stamp_s=5.9).converged
    assert controller.update(target, stamp_s=6.01).converged

    controller.update((0.2, 0.0, 1.0), stamp_s=6.1)
    assert not controller.update(target, stamp_s=7.0).converged


def test_convergence_accepts_exact_decimal_tick_boundary():
    config = VisualServoConfig(settle_time_s=0.6)
    controller = VisualServoController(config)
    target = (0.01, 0.0, config.desired_depth_m)

    controller.update(target, stamp_s=177.16)
    elapsed = 177.76 - 177.16

    assert elapsed < config.settle_time_s
    assert controller.update(target, stamp_s=177.76).converged


def test_convergence_hysteresis_retains_window_across_depth_noise():
    config = VisualServoConfig(
        desired_depth_m=0.50,
        depth_tolerance_m=0.06,
        lateral_tolerance_m=0.03,
        depth_exit_hysteresis_m=0.01,
        lateral_exit_hysteresis_m=0.005,
        settle_time_s=0.6,
    )
    controller = VisualServoController(config)

    assert not controller.update((0.029, 0.0, 0.559), stamp_s=1.0).converged
    noisy = controller.update((0.034, 0.0, 0.564), stamp_s=1.3)
    converged = controller.update((0.032, 0.0, 0.563), stamp_s=1.6)

    assert noisy.linear_x == noisy.angular_z == 0.0
    assert converged.converged


def test_exit_hysteresis_cannot_start_or_survive_outside_its_envelope():
    config = VisualServoConfig(
        desired_depth_m=0.50,
        depth_tolerance_m=0.06,
        depth_exit_hysteresis_m=0.01,
        settle_time_s=0.6,
    )
    controller = VisualServoController(config)

    outside_entry = controller.update((0.0, 0.0, 0.565), stamp_s=1.0)
    controller.update((0.0, 0.0, 0.559), stamp_s=1.1)
    outside_exit = controller.update((0.0, 0.0, 0.571), stamp_s=1.2)
    restarted = controller.update((0.0, 0.0, 0.559), stamp_s=1.7)

    assert outside_entry.linear_x > 0.0
    assert outside_exit.linear_x > 0.0
    assert not restarted.converged


def test_invalid_coordinates_fail_closed():
    controller = VisualServoController(VisualServoConfig())
    command = controller.update((math.nan, 0.0, 1.0), stamp_s=1.0)
    assert command.linear_x == command.angular_z == 0.0


def test_reachability_selected_depth_overrides_deployment_default():
    controller = VisualServoController(VisualServoConfig(desired_depth_m=0.55))
    command = controller.update(
        (0.0, 0.0, 0.40),
        stamp_s=1.0,
        desired_depth_m=0.35,
    )
    assert command.linear_x > 0.0


def test_rotate_only_bearing_is_configured_and_blocks_blind_arc():
    config = VisualServoConfig(
        rotate_only_bearing_rad=math.radians(10.0),
        max_yaw_rps=0.12,
    )
    controller = VisualServoController(config)

    rotate = controller.update((0.25, 0.0, 0.80), stamp_s=1.0)
    approach = controller.update((0.10, 0.0, 0.80), stamp_s=1.1)

    assert rotate.linear_x == 0.0
    assert rotate.angular_z == pytest.approx(-0.12)
    assert approach.linear_x > 0.0


def test_yaw_deadband_does_not_chase_legged_body_sway():
    config = VisualServoConfig(
        yaw_deadband_rad=math.radians(6.0),
        rotate_only_bearing_rad=math.radians(25.0),
    )
    controller = VisualServoController(config)

    right_sway = controller.update((0.04, 0.0, 0.60), stamp_s=1.0)
    left_sway = controller.update((-0.04, 0.0, 0.60), stamp_s=1.1)
    coarse_turn = controller.update((0.15, 0.0, 0.60), stamp_s=1.2)

    assert right_sway.linear_x > 0.0 and right_sway.angular_z == 0.0
    assert left_sway.linear_x > 0.0 and left_sway.angular_z == 0.0
    assert coarse_turn.linear_x > 0.0 and coarse_turn.angular_z < 0.0


@pytest.mark.parametrize(
    "changes",
    (
        {"max_yaw_rps": 0.0},
        {"rotate_only_bearing_rad": math.nan},
        {"rotate_only_bearing_rad": math.pi / 2.0},
        {"depth_exit_hysteresis_m": -0.001},
        {"depth_exit_hysteresis_m": 0.051},
        {"lateral_exit_hysteresis_m": 0.036},
        {"yaw_deadband_rad": -0.001},
    ),
)
def test_visual_servo_rejects_invalid_motion_limits(changes):
    with pytest.raises(ValueError):
        VisualServoConfig(**changes)
