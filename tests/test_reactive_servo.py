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
    ViewMeasurementDampingConfig,
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


# ---------------------------------------------------------------------------
# Measurement-aware arm-view damping.
#
# The Fast-FoundationStereo depth switch dropped the tracked-target bundle rate
# to a measured ~7.5-8 Hz and grew capture-to-use latency to ~200-250 ms.  The
# wrist arm-view inner loop (the whole-body QP nulling the image error over its
# fixed 0.20 s horizon) keeps integrating a frozen perception error between
# bundles.  With the loop delay roughly tripled, the *unchanged* gain crosses
# the discrete stability boundary and the camera over-pitches past a high
# target.  These tests validate the damping law offline against a replayed
# plant model before any live motion.
# ---------------------------------------------------------------------------

# QP small-signal view gain: the optimizer nulls the image error over its fixed
# 0.20 s horizon, so the inner proportional gain is 1 / 0.20 = 5 /s regardless
# of the perception rate.  MAX_QDOT mirrors the NUC executor's 12 deg/s cap.
_QP_HORIZON_S = 0.20
_QP_VIEW_GAIN_PER_S = 1.0 / _QP_HORIZON_S
_MAX_QDOT_RPS = math.radians(12.0)


def _view_loop_spectral_radius(*, effective_gain_per_s, update_period_s, latency_s):
    """Dominant closed-loop pole of the delayed sampled arm-view loop.

    Small-signal model near centre: the camera pitch integrates the commanded
    rate; every ``update_period`` the controller resamples an error that is
    ``latency`` seconds stale and holds (ZOH) the command between updates::

        e[n+1] = e[n] - gain * update_period * e[n - m],  m = round(latency / P)

    ``|z|_max > 1`` means the frozen-error loop diverges.
    """

    delay_samples = max(0, round(latency_s / update_period_s))
    loop_gain = effective_gain_per_s * update_period_s
    coefficients = [0.0] * (delay_samples + 2)
    coefficients[0] = 1.0
    coefficients[1] = -1.0
    coefficients[-1] = loop_gain
    return float(np.max(np.abs(np.roots(coefficients))))


def _simulate_view_loop(
    *,
    initial_error_rad,
    update_period_s,
    latency_s,
    damping=None,
    dt_s=0.002,
    horizon_s=10.0,
):
    """Replay the wrist-view loop against a delayed ZOH plant with MAX_QDOT.

    ``damping`` None reproduces the pre-fix aggressive QP (rate saturates at
    MAX_QDOT).  A ``ViewMeasurementDampingConfig`` scales the commanded view
    rate exactly as the wired whole-body adapter does.  Returns the error
    trajectory (rad), the peak overshoot fraction past centre, the residual
    oscillation amplitude over the final 3 s, and whether it converged and
    stayed within 5 % of the initial error.
    """

    steps = int(horizon_s / dt_s)
    error = float(initial_error_rad)
    history = [(-1.0, error)]
    command_rate = 0.0
    next_sample_s = 0.0
    trajectory = [error]

    def observed(now_s):
        stale_s = now_s - latency_s
        for stamp_s, value in reversed(history):
            if stamp_s <= stale_s:
                return value
        return float(initial_error_rad)

    for index in range(steps):
        now_s = index * dt_s
        if now_s >= next_sample_s:
            error_obs = observed(now_s)
            raw_rate = max(
                -_MAX_QDOT_RPS,
                min(_MAX_QDOT_RPS, _QP_VIEW_GAIN_PER_S * error_obs),
            )
            if damping is not None:
                raw_rate *= damping.rate_scale(
                    view_error_rad=error_obs,
                    commanded_rate_rps=raw_rate,
                    update_period_s=update_period_s,
                )
            command_rate = raw_rate
            next_sample_s += update_period_s
        error -= command_rate * dt_s
        history.append((now_s, error))
        trajectory.append(error)

    peak_reverse = min(0.0, min(trajectory))
    overshoot_fraction = abs(peak_reverse) / abs(initial_error_rad)
    tail = trajectory[int((horizon_s - 3.0) / dt_s):]
    oscillation = (max(tail) - min(tail)) / 2.0
    tolerance = 0.05 * abs(initial_error_rad)
    converged = all(abs(value) <= tolerance for value in tail)
    return trajectory, overshoot_fraction, oscillation, converged


def test_view_damping_config_validation():
    ViewMeasurementDampingConfig()  # defaults are valid
    with pytest.raises(ValueError):
        ViewMeasurementDampingConfig(stability_alpha=0.0)
    with pytest.raises(ValueError):
        ViewMeasurementDampingConfig(stability_alpha=1.0)
    with pytest.raises(ValueError):
        ViewMeasurementDampingConfig(nominal_latency_s=0.0)
    with pytest.raises(ValueError):
        ViewMeasurementDampingConfig(min_rate_scale=1.5)


def test_view_damping_passes_through_low_latency_and_bounds_high_latency():
    config = ViewMeasurementDampingConfig(stability_alpha=0.5, nominal_latency_s=0.25)
    error = math.radians(26.0)

    # A brisk pre-FFS loop leaves an already-small command untouched.
    fast = config.rate_scale(
        view_error_rad=error,
        commanded_rate_rps=config.rate_cap_rps(error, update_period_s=0.033) * 0.5,
        update_period_s=0.033,
    )
    assert fast == pytest.approx(1.0)

    # At the measured FFS regime a saturated command is scaled below unity, and
    # the resulting rate never exceeds the measurement-aware cap.
    saturated = _MAX_QDOT_RPS
    scale = config.rate_scale(
        view_error_rad=math.radians(3.0),
        commanded_rate_rps=saturated,
        update_period_s=0.133,
    )
    assert 0.0 < scale < 1.0
    capped = saturated * scale
    assert capped <= config.rate_cap_rps(
        math.radians(3.0), update_period_s=0.133
    ) + 1e-9

    # Far from centre the cap is large, so full authority is retained to keep a
    # climbing target in frame during approach.
    far = config.rate_scale(
        view_error_rad=math.radians(26.0),
        commanded_rate_rps=saturated,
        update_period_s=0.133,
    )
    assert far == pytest.approx(1.0)


def test_old_gain_diverges_but_new_law_is_stable_across_the_ffs_regime():
    # Same QP gain, only the loop delay changes: pre-FFS is well damped, the
    # measured FFS rate/latency pushes the identical gain past the unit circle.
    pre_ffs = _view_loop_spectral_radius(
        effective_gain_per_s=_QP_VIEW_GAIN_PER_S,
        update_period_s=0.033,
        latency_s=0.03,
    )
    ffs = _view_loop_spectral_radius(
        effective_gain_per_s=_QP_VIEW_GAIN_PER_S,
        update_period_s=0.133,
        latency_s=0.25,
    )
    assert pre_ffs < 1.0
    assert ffs > 1.0  # the recorded regression: frozen-error loop diverges

    # The damping law reduces the effective small-signal gain to
    # alpha / (update_period + latency); check it stays comfortably inside the
    # unit circle across the whole 5-10 Hz / 200-350 ms envelope.
    config = ViewMeasurementDampingConfig(stability_alpha=0.5)
    for update_period_s, latency_s in [
        (0.200, 0.35),
        (0.133, 0.25),
        (0.125, 0.20),
        (0.100, 0.20),
        (0.133, 0.35),
    ]:
        effective_gain = config.stability_alpha / (update_period_s + latency_s)
        spectral_radius = _view_loop_spectral_radius(
            effective_gain_per_s=effective_gain,
            update_period_s=update_period_s,
            latency_s=latency_s,
        )
        assert spectral_radius < 0.9


def test_view_loop_simulation_old_oscillates_and_new_converges_on_a_step():
    step = math.radians(26.0)

    # Pre-fix aggressive QP at the measured FFS regime: sustained oscillation
    # that never settles within tolerance (the target leaves frame).
    _traj, old_overshoot, old_oscillation, old_converged = _simulate_view_loop(
        initial_error_rad=step,
        update_period_s=0.133,
        latency_s=0.25,
        damping=None,
    )
    assert not old_converged
    assert old_oscillation > math.radians(1.0)

    # Measurement-aware damping: monotonic convergence with bounded overshoot.
    config = ViewMeasurementDampingConfig(stability_alpha=0.5)
    _traj, new_overshoot, new_oscillation, new_converged = _simulate_view_loop(
        initial_error_rad=step,
        update_period_s=0.133,
        latency_s=0.25,
        damping=config,
    )
    assert new_converged
    assert new_overshoot < 0.20  # well under the 20 % overshoot budget
    assert new_oscillation < math.radians(0.5)


def test_view_damping_can_be_disabled_for_a_transparent_passthrough():
    disabled = ViewMeasurementDampingConfig(enabled=False)
    assert disabled.rate_scale(
        view_error_rad=math.radians(3.0),
        commanded_rate_rps=_MAX_QDOT_RPS,
        update_period_s=0.133,
    ) == pytest.approx(1.0)
