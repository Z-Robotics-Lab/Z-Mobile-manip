import math

import numpy as np
import pytest

from z_manip.control.reactive_servo import (
    ArmViewMode,
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
