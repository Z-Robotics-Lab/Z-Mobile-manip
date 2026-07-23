import math

import numpy as np
import pytest

from z_manip.control.reactive_servo import (
    ArmViewMode,
    PostureIntent,
    ReactivePhase,
    ReactiveServoConfig,
    ReactiveTargetController,
    TargetGeometry,
    transform_point,
)


IDENTITY = np.eye(4)


def _geometry(
    camera_xyz=(0.0, 0.0, 0.70),
    *,
    base_xyz=(0.70, 0.0, -0.10),
    arm_xyz=(0.55, 0.0, 0.10),
):
    base_from_camera = np.eye(4)
    base_from_camera[:3, 3] = np.asarray(base_xyz) - np.asarray(camera_xyz)
    arm_from_camera = np.eye(4)
    arm_from_camera[:3, 3] = np.asarray(arm_xyz) - np.asarray(camera_xyz)
    return TargetGeometry.from_camera(
        camera_xyz,
        T_base_camera=base_from_camera,
        T_arm_camera=arm_from_camera,
    )


def test_geometry_keeps_full_xyz_and_distinguishes_planar_and_3d_range():
    geometry = _geometry(
        camera_xyz=(0.10, 0.20, 0.80),
        base_xyz=(0.60, 0.80, -0.40),
        arm_xyz=(0.30, 0.40, 0.50),
    )

    assert geometry.camera_xyz_m == pytest.approx((0.10, 0.20, 0.80))
    assert geometry.camera_range_m == pytest.approx(math.sqrt(0.69))
    assert geometry.camera_elevation_rad == pytest.approx(math.atan2(-0.20, 0.80))
    assert geometry.base_planar_distance_m == pytest.approx(1.0)
    assert geometry.base_range_m == pytest.approx(math.sqrt(1.16))
    assert geometry.target_height_m == pytest.approx(-0.40)
    assert geometry.arm_range_m == pytest.approx(math.sqrt(0.50))


def test_transform_point_requires_explicit_valid_homogeneous_transform():
    transform = np.eye(4)
    transform[:3, 3] = (1.0, -2.0, 0.5)
    assert transform_point(transform, (0.1, 0.2, 0.3)) == pytest.approx(
        (1.1, -1.8, 0.8),
    )
    malformed = np.eye(4)
    malformed[3, 3] = 0.0
    with pytest.raises(ValueError):
        transform_point(malformed, (0.0, 0.0, 1.0))


def test_base_approach_uses_ground_plane_euclidean_distance_not_camera_depth():
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, 0.0, 0.45),
        base_xyz=(0.90, 0.0, -0.10),
        arm_xyz=(0.75, 0.0, 0.10),
    )

    decision = controller.update(
        geometry,
        now_s=1.0,
        tracking=True,
        body_settled=True,
    )

    # Optical depth is already below the old 0.52 m cutoff, but ground-plane
    # range is still 0.90 m, so the base must continue its coarse approach.
    assert decision.phase is ReactivePhase.BASE_APPROACH
    assert decision.base.linear_x_mps > 0.0
    assert "ground-plane" in decision.reason


def test_high_target_approach_slows_when_camera_elevation_is_at_risk():
    controller = ReactiveTargetController()
    # A high window-sill target with the base still far (planar 1.9 m) and body
    # posture NOT actionable (the Go2W SPORT service rejects Euler), so the FSM
    # stays in BASE_APPROACH -- the branch that drove the recorded loss.  The
    # camera elevation error (~24 deg) exceeds the 16 deg soft limit.
    high = _geometry(
        camera_xyz=(0.0, -0.52, 1.80),
        base_xyz=(1.90, 0.0, 0.25),
        arm_xyz=(1.75, 0.0, 0.35),
    )
    slowed = controller.update(
        high,
        now_s=1.0,
        tracking=True,
        body_settled=True,
        body_posture_actionable=False,
    )
    assert slowed.phase is ReactivePhase.BASE_APPROACH

    # A level target at the same range is not at elevation risk and advances at
    # the full commanded speed.
    controller.reset()
    level = _geometry(
        camera_xyz=(0.0, 0.0, 1.80),
        base_xyz=(1.90, 0.0, 0.25),
        arm_xyz=(1.75, 0.0, 0.35),
    )
    full = controller.update(
        level,
        now_s=1.0,
        tracking=True,
        body_settled=True,
        body_posture_actionable=False,
    )
    assert full.phase is ReactivePhase.BASE_APPROACH
    assert 0.0 < slowed.base.linear_x_mps < full.base.linear_x_mps


def test_elevation_speed_floor_of_zero_halts_approach_past_the_hard_limit():
    controller = ReactiveTargetController(
        ReactiveServoConfig(elevation_approach_speed_floor=0.0)
    )
    # Elevation error well beyond the 26 deg hard limit ramps the forward
    # command all the way to zero: hold the base rather than drive the target
    # out of the wrist-camera view.
    too_high = _geometry(
        camera_xyz=(0.0, -1.20, 1.80),
        base_xyz=(1.90, 0.0, 0.25),
        arm_xyz=(1.75, 0.0, 0.35),
    )
    decision = controller.update(
        too_high,
        now_s=1.0,
        tracking=True,
        body_settled=True,
        body_posture_actionable=False,
    )
    assert decision.phase is ReactivePhase.BASE_APPROACH
    assert decision.base.linear_x_mps == 0.0


def test_elevation_speed_floor_is_validated():
    with pytest.raises(ValueError):
        ReactiveServoConfig(elevation_approach_speed_floor=1.5)


def test_side_setpoint_steers_off_centre_and_gates_handoff():
    controller = ReactiveTargetController()
    centred = _geometry(
        camera_xyz=(0.0, 0.0, 0.55),
        base_xyz=(0.55, 0.0, -0.10),
        arm_xyz=(0.50, 0.0, 0.10),
    )

    approach = controller.update(
        centred,
        now_s=1.0,
        tracking=True,
        body_settled=True,
        ik_feasible=True,
        desired_target_lateral_m=0.13,
    )
    assert approach.phase is ReactivePhase.BASE_APPROACH
    assert approach.base.angular_z_rps < 0.0

    aligned = _geometry(
        camera_xyz=(0.0, 0.0, 0.55),
        base_xyz=(0.55, 0.13, -0.10),
        arm_xyz=(0.50, 0.0, 0.10),
    )
    handoff = controller.update(
        aligned,
        now_s=1.1,
        tracking=True,
        body_settled=True,
        ik_feasible=True,
        desired_target_lateral_m=0.13,
    )
    assert handoff.phase is ReactivePhase.HANDOFF_READY


def test_low_target_requests_body_and_arm_view_adjustment_before_handoff():
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, 0.30, 0.65),
        base_xyz=(0.72, 0.0, -0.35),
        arm_xyz=(0.60, 0.0, -0.20),
    )

    decision = controller.update(
        geometry,
        now_s=2.0,
        tracking=True,
        body_settled=True,
        ik_feasible=True,
    )

    assert decision.phase is ReactivePhase.POSTURE_ADJUST
    assert decision.base.linear_x_mps == 0.0
    assert decision.posture.body_height_delta_m < 0.0
    assert decision.posture.pitch_delta_rad < 0.0
    assert decision.arm_view.mode is ArmViewMode.LOOK_DOWN


def test_high_target_requests_look_up_and_raise_intent():
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, -0.25, 0.70),
        base_xyz=(0.75, 0.0, 0.20),
        arm_xyz=(0.58, 0.0, 0.35),
    )

    decision = controller.update(
        geometry,
        now_s=2.0,
        tracking=True,
        body_settled=True,
    )

    assert decision.phase is ReactivePhase.POSTURE_ADJUST
    assert decision.posture.body_height_delta_m > 0.0
    assert decision.posture.pitch_delta_rad > 0.0
    assert decision.arm_view.mode is ArmViewMode.LOOK_UP


def test_unactionable_body_posture_skips_latch_and_keeps_approaching():
    # When body attitude cannot move the view (e.g. Go2W ai-w rejects Euler),
    # a view-at-risk target outside the handoff corridor must not latch the
    # controller in POSTURE_ADJUST; it keeps approaching while the wrist arm
    # view corrects elevation.
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, 0.30, 0.65),
        base_xyz=(0.72, 0.0, -0.35),
        arm_xyz=(0.60, 0.0, -0.20),
    )

    latched = controller.update(
        geometry, now_s=2.0, tracking=True, body_settled=True,
    )
    assert latched.phase is ReactivePhase.POSTURE_ADJUST

    controller.reset()
    skipped = controller.update(
        geometry,
        now_s=2.0,
        tracking=True,
        body_settled=True,
        body_posture_actionable=False,
    )

    assert skipped.phase is ReactivePhase.BASE_APPROACH
    assert skipped.base.linear_x_mps > 0.0
    assert skipped.posture == PostureIntent()
    assert skipped.arm_view.mode is ArmViewMode.LOOK_DOWN


def test_unactionable_body_posture_reaches_handoff_probe_then_ready():
    # Inside the corridor with a view still at risk, the actionable path traps
    # in POSTURE_ADJUST, while the unactionable path proceeds to the explicit
    # IK probe and only declares HANDOFF_READY once IK is feasible.
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, -0.20, 0.55),
        base_xyz=(0.55, 0.13, 0.0),
        arm_xyz=(0.50, 0.0, 0.10),
    )

    trapped = controller.update(
        geometry,
        now_s=3.0,
        tracking=True,
        body_settled=True,
        desired_target_lateral_m=0.13,
    )
    assert trapped.phase is ReactivePhase.POSTURE_ADJUST

    controller.reset()
    probe = controller.update(
        geometry,
        now_s=3.0,
        tracking=True,
        body_settled=True,
        ik_feasible=None,
        desired_target_lateral_m=0.13,
        body_posture_actionable=False,
    )
    assert probe.phase is ReactivePhase.HANDOFF_PROBE
    assert probe.needs_ik_probe is True
    assert probe.handoff_ready is False

    ready = controller.update(
        geometry,
        now_s=3.05,
        tracking=True,
        body_settled=True,
        ik_feasible=True,
        desired_target_lateral_m=0.13,
        body_posture_actionable=False,
    )
    assert ready.phase is ReactivePhase.HANDOFF_READY
    assert ready.handoff_ready is True


def test_posture_motion_requires_measured_settle_and_stable_reacquisition():
    config = ReactiveServoConfig(reacquire_stable_s=0.25)
    controller = ReactiveTargetController(config)
    geometry = _geometry(
        camera_xyz=(0.0, 0.25, 0.70),
        base_xyz=(0.75, 0.0, -0.30),
        arm_xyz=(0.60, 0.0, -0.15),
    )
    controller.update(
        geometry, now_s=1.0, tracking=True, body_settled=True,
    )

    moving = controller.update(
        geometry, now_s=1.1, tracking=True, body_settled=False,
    )
    reacquire = controller.update(
        geometry, now_s=1.2, tracking=True, body_settled=True,
    )
    stable = controller.update(
        geometry, now_s=1.46, tracking=True, body_settled=True,
    )

    assert moving.phase is ReactivePhase.POSTURE_ADJUST
    assert reacquire.phase is ReactivePhase.REACQUIRE
    # The same low-view geometry requests another posture increment only after
    # the old observation window was rebuilt; it cannot silently hand off.
    assert stable.phase is ReactivePhase.POSTURE_ADJUST


def test_short_tracking_loss_holds_base_and_recovers_last_viewing_ray():
    controller = ReactiveTargetController(ReactiveServoConfig(tracking_loss_grace_s=0.75))
    geometry = _geometry()
    controller.update(geometry, now_s=4.0, tracking=True, body_settled=True)

    decision = controller.update(
        None,
        now_s=4.40,
        tracking=False,
        body_settled=True,
    )

    assert decision.phase is ReactivePhase.VIEW_RECOVERY
    assert decision.base.linear_x_mps == decision.base.angular_z_rps == 0.0
    assert decision.arm_view.mode is ArmViewMode.SEARCH
    assert decision.geometry is geometry


def test_stepping_base_tracking_hole_freezes_without_triggering_recovery():
    controller = ReactiveTargetController(ReactiveServoConfig(
        tracking_hold_s=0.40,
        tracking_loss_grace_s=0.90,
    ))
    geometry = _geometry()
    controller.update(geometry, now_s=4.0, tracking=True, body_settled=True)

    decision = controller.update(
        None,
        now_s=4.25,
        tracking=False,
        body_settled=True,
    )

    assert decision.phase is ReactivePhase.TRACKING_HOLD
    assert decision.base.linear_x_mps == decision.base.angular_z_rps == 0.0
    assert decision.posture == PostureIntent()
    assert decision.arm_view.mode is ArmViewMode.HOLD
    assert decision.geometry is geometry


def test_persistent_tracking_loss_requests_bounded_search_without_base_motion():
    controller = ReactiveTargetController(ReactiveServoConfig(tracking_loss_grace_s=0.50))
    geometry = _geometry()
    controller.update(geometry, now_s=4.0, tracking=True, body_settled=True)

    decision = controller.update(
        None,
        now_s=4.60,
        tracking=False,
        body_settled=True,
    )

    assert decision.phase is ReactivePhase.SEARCH_REQUIRED
    assert decision.base.linear_x_mps == decision.base.angular_z_rps == 0.0
    assert "bounded" in decision.reason


def test_handoff_requires_3d_arm_corridor_and_explicit_ik_probe():
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, 0.0, 0.55),
        base_xyz=(0.55, 0.0, -0.10),
        arm_xyz=(0.50, 0.0, 0.10),
    )

    unknown = controller.update(
        geometry,
        now_s=5.0,
        tracking=True,
        body_settled=True,
        ik_feasible=None,
    )
    rejected = controller.update(
        geometry,
        now_s=5.1,
        tracking=True,
        body_settled=True,
        ik_feasible=False,
    )
    accepted = controller.update(
        geometry,
        now_s=5.2,
        tracking=True,
        body_settled=True,
        ik_feasible=True,
    )

    assert unknown.phase is rejected.phase is ReactivePhase.HANDOFF_PROBE
    assert unknown.needs_ik_probe
    assert not rejected.needs_ik_probe
    assert unknown.base.linear_x_mps == rejected.base.linear_x_mps == 0.0
    assert not unknown.handoff_ready and not rejected.handoff_ready
    assert accepted.phase is ReactivePhase.HANDOFF_READY
    assert accepted.handoff_ready
    assert accepted.base.linear_x_mps == accepted.base.angular_z_rps == 0.0


def test_near_field_handoff_precedes_another_posture_increment():
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, 0.18, 0.40),
        base_xyz=(0.58, 0.0, -0.28),
        arm_xyz=(0.54, 0.0, -0.14),
    )

    decision = controller.update(
        geometry,
        now_s=6.0,
        tracking=True,
        body_settled=True,
        ik_feasible=None,
    )

    assert decision.phase is ReactivePhase.HANDOFF_PROBE
    assert decision.needs_ik_probe
    assert decision.base.linear_x_mps == decision.base.angular_z_rps == 0.0
    assert decision.posture.body_height_delta_m == 0.0
    assert decision.posture.pitch_delta_rad == 0.0
    assert decision.arm_view.mode is ArmViewMode.HOLD
    assert "near-field" in decision.reason


def test_wrist_camera_near_field_handoffs_before_base_reaches_same_distance():
    """A wrist-mounted D435 reaches 52 cm before base_link does."""

    controller = ReactiveTargetController(ReactiveServoConfig(
        handoff_planar_max_m=0.52,
        camera_handoff_depth_m=0.52,
        camera_handoff_planar_slack_m=0.15,
    ))
    geometry = _geometry(
        camera_xyz=(0.018, 0.037, 0.476),
        base_xyz=(0.600, 0.130, 0.074),
        arm_xyz=(0.480, 0.000, 0.074),
    )

    decision = controller.update(
        geometry,
        now_s=6.0,
        tracking=True,
        body_settled=False,
        ik_feasible=None,
        desired_target_lateral_m=0.13,
    )

    assert geometry.base_planar_distance_m > 0.52
    assert decision.phase is ReactivePhase.HANDOFF_PROBE
    assert decision.needs_ik_probe
    assert decision.base.linear_x_mps == decision.base.angular_z_rps == 0.0
    assert decision.arm_view.mode is ArmViewMode.HOLD


def test_hard_camera_floor_stops_view_motion_outside_soft_handoff_corridor():
    controller = ReactiveTargetController()
    geometry = _geometry(
        camera_xyz=(0.0, 0.0, 0.36),
        base_xyz=(0.82, 0.0, -0.10),
        arm_xyz=(0.78, 0.0, 0.10),
    )

    decision = controller.update(
        geometry,
        now_s=6.0,
        tracking=True,
        body_settled=True,
    )

    assert decision.phase is ReactivePhase.HANDOFF_PROBE
    assert decision.arm_view.mode is ArmViewMode.HOLD
    assert decision.base.linear_x_mps == 0.0
    assert "hard near-field floor" in decision.reason


def test_from_frames_accepts_external_tf_results_without_reinterpreting_axes():
    geometry = TargetGeometry.from_frames(
        (0.1, 0.2, 0.8),
        base_xyz_m=(0.6, -0.8, -0.3),
        arm_xyz_m=(0.3, 0.4, 0.5),
    )

    assert geometry.base_planar_distance_m == pytest.approx(1.0)
    assert geometry.base_bearing_rad == pytest.approx(math.atan2(-0.8, 0.6))
    assert geometry.target_height_m == pytest.approx(-0.3)
    assert geometry.arm_range_m == pytest.approx(math.sqrt(0.5))


def test_configuration_rejects_overlapping_or_inverted_corridors():
    with pytest.raises(ValueError):
        ReactiveServoConfig(posture_entry_planar_m=0.60, handoff_planar_max_m=0.62)
    with pytest.raises(ValueError):
        ReactiveServoConfig(
            camera_elevation_soft_limit_rad=0.5,
            camera_elevation_hard_limit_rad=0.4,
        )
    with pytest.raises(ValueError):
        ReactiveServoConfig(
            camera_handoff_depth_m=0.38,
            camera_hard_min_depth_m=0.40,
        )
