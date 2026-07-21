import math

from z_manip.control.approach import (
    ApproachInput,
    ApproachPhase,
    TwoStageApproachConfig,
    TwoStageApproachController,
    VelocityOwner,
)
from z_manip.control.visual_servo import VisualServoConfig


def _input(
    stamp_s,
    *,
    nav_range_m=2.0,
    position=(0.0, 0.0, 2.0),
    locked=True,
    nav_speed=0.0,
    nav_yaw_rate=0.0,
    roll=0.0,
    pitch=0.0,
):
    return ApproachInput(
        stamp_s=stamp_s,
        approximate_range_m=nav_range_m,
        target_position_camera=position,
        tracker_locked=locked,
        navigation_speed_mps=nav_speed,
        navigation_yaw_rate_rps=nav_yaw_rate,
        base_roll_rad=roll,
        base_pitch_rad=pitch,
    )


def _controller():
    return TwoStageApproachController(TwoStageApproachConfig(
        near_stage_threshold_m=1.4,
        tracker_lock_time_s=0.25,
        handoff_quiet_time_s=0.20,
        track_loss_timeout_s=0.3,
        timeout_s=20.0,
        visual_servo=VisualServoConfig(
            desired_depth_m=0.55,
            settle_time_s=0.2,
            depth_tolerance_m=0.04,
            lateral_tolerance_m=0.03,
        ),
    ))


def test_approximate_navigation_hands_off_exclusively_to_visual_servo():
    controller = _controller()
    far = controller.update(_input(0.0, nav_range_m=2.2, position=(0.0, 0.0, 2.2)))
    assert far.phase == ApproachPhase.FAR_NAV
    assert far.owner == VelocityOwner.NAVIGATION

    controller.update(_input(1.0, nav_range_m=1.2, position=(0.08, 0.0, 1.2)))
    handoff = controller.update(
        _input(1.3, nav_range_m=1.2, position=(0.08, 0.0, 1.2), nav_speed=0.08),
    )
    assert handoff.phase == ApproachPhase.HANDOFF
    assert handoff.owner == VelocityOwner.NONE
    assert handoff.cancel_navigation
    assert handoff.servo.linear_x == handoff.servo.angular_z == 0.0

    controller.update(
        _input(1.4, nav_range_m=1.2, position=(0.08, 0.0, 1.2), nav_speed=0.0),
    )
    near = controller.update(
        _input(1.65, nav_range_m=1.2, position=(0.08, 0.0, 1.2), nav_speed=0.0),
    )
    assert near.phase == ApproachPhase.VISUAL_SERVO
    assert near.owner == VelocityOwner.MANIP_SERVO
    assert near.servo.linear_x > 0.0


def test_handoff_checks_linear_speed_and_yaw_rate_in_separate_units():
    controller = _controller()
    target = (0.08, 0.0, 1.2)
    controller.update(_input(1.0, nav_range_m=1.2, position=target))
    controller.update(_input(1.3, nav_range_m=1.2, position=target))

    turning = controller.update(_input(
        1.4,
        nav_range_m=1.2,
        position=target,
        nav_speed=0.01,
        nav_yaw_rate=0.08,
    ))
    assert turning.phase == ApproachPhase.HANDOFF
    assert turning.servo.linear_x == turning.servo.angular_z == 0.0

    controller.update(_input(
        1.5,
        nav_range_m=1.2,
        position=target,
        nav_speed=0.01,
        nav_yaw_rate=0.01,
    ))
    near = controller.update(_input(
        1.75,
        nav_range_m=1.2,
        position=target,
        nav_speed=0.01,
        nav_yaw_rate=0.01,
    ))
    assert near.phase == ApproachPhase.VISUAL_SERVO
    assert near.owner == VelocityOwner.MANIP_SERVO


def test_handoff_and_visual_servo_have_independent_bounded_deadlines():
    controller = TwoStageApproachController(TwoStageApproachConfig(
        tracker_lock_time_s=0.1,
        handoff_quiet_time_s=0.3,
        handoff_timeout_s=100.0,
        timeout_s=0.5,
    ))
    target = (0.08, 0.0, 1.2)
    controller.update(_input(100.0, nav_range_m=1.2, position=target))
    handoff = controller.update(
        _input(100.2, nav_range_m=1.2, position=target),
    )
    assert handoff.phase == ApproachPhase.HANDOFF

    controller.update(_input(
        160.31,
        nav_range_m=1.2,
        position=target,
    ))
    servo = controller.update(_input(
        160.61,
        nav_range_m=1.2,
        position=target,
    ))
    assert servo.phase == ApproachPhase.VISUAL_SERVO

    still_running = controller.update(_input(
        161.10,
        nav_range_m=1.2,
        position=target,
    ))
    assert still_running.phase == ApproachPhase.VISUAL_SERVO
    timed_out = controller.update(_input(
        161.12,
        nav_range_m=1.2,
        position=target,
    ))
    assert timed_out.phase == ApproachPhase.FAILED
    assert timed_out.reason == "visual servo timeout"


def test_handoff_deadline_remains_bounded():
    controller = TwoStageApproachController(TwoStageApproachConfig(
        tracker_lock_time_s=0.1,
        handoff_quiet_time_s=0.3,
        handoff_timeout_s=0.5,
        timeout_s=60.0,
    ))
    target = (0.08, 0.0, 1.2)
    controller.update(_input(0.0, nav_range_m=1.2, position=target))
    controller.update(_input(0.2, nav_range_m=1.2, position=target))
    timed_out = controller.update(_input(
        0.71,
        nav_range_m=1.2,
        position=target,
        nav_speed=0.1,
    ))
    assert timed_out.phase == ApproachPhase.FAILED
    assert timed_out.reason == "approach handoff timeout"
    sticky = controller.update(_input(
        100.0,
        nav_range_m=1.2,
        position=target,
    ))
    assert sticky.phase == ApproachPhase.FAILED
    assert sticky.owner == VelocityOwner.NONE
    assert not sticky.cancel_navigation


def test_near_approach_never_moves_without_persistent_track():
    controller = _controller()
    output = controller.update(
        _input(0.0, nav_range_m=1.0, position=None, locked=False),
    )
    assert output.owner == VelocityOwner.NAVIGATION
    assert output.phase == ApproachPhase.FAR_NAV

    controller.update(_input(1.0, nav_range_m=1.0, position=(0.0, 0.0, 1.0)))
    controller.update(_input(1.3, nav_range_m=1.0, position=(0.0, 0.0, 1.0)))
    controller.update(_input(1.4, nav_range_m=1.0, position=(0.0, 0.0, 1.0)))
    controller.update(_input(1.7, nav_range_m=1.0, position=(0.0, 0.0, 1.0)))
    lost = controller.update(_input(1.8, nav_range_m=1.0, position=None, locked=False))
    assert lost.owner == VelocityOwner.MANIP_SERVO
    assert lost.servo.linear_x == lost.servo.angular_z == 0.0
    failed = controller.update(_input(2.2, nav_range_m=1.0, position=None, locked=False))
    assert failed.phase == ApproachPhase.FAILED
    assert failed.owner == VelocityOwner.NONE


def test_visual_convergence_is_tracker_based_and_uses_sim_time():
    controller = _controller()
    target = (0.01, 0.0, 0.56)
    controller.update(_input(0.0, nav_range_m=0.56, position=target))
    controller.update(_input(0.3, nav_range_m=0.56, position=target))
    controller.update(_input(0.4, nav_range_m=0.56, position=target))
    controller.update(_input(0.65, nav_range_m=0.56, position=target))
    settling = controller.update(_input(0.7, nav_range_m=0.56, position=target))
    done = controller.update(_input(0.91, nav_range_m=0.56, position=target))
    assert settling.phase == ApproachPhase.VISUAL_SERVO
    assert done.phase == ApproachPhase.COMPLETE
    assert done.owner == VelocityOwner.NONE
    sticky = controller.update(_input(100.0, nav_range_m=0.56, position=target))
    assert sticky.phase == ApproachPhase.COMPLETE
    assert sticky.owner == VelocityOwner.NONE
    assert not sticky.cancel_navigation


def test_attitude_gate_stops_navigation_and_arm_approach():
    controller = _controller()
    output = controller.update(_input(
        0.0,
        nav_range_m=2.0,
        pitch=math.radians(15.0),
    ))
    assert output.phase == ApproachPhase.FAILED
    assert output.owner == VelocityOwner.NONE
    assert "attitude" in output.reason
