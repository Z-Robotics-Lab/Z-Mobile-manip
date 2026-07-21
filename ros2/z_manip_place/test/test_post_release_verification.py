"""Tests for fail-closed observed post-release placement evidence."""

from dataclasses import replace
import json

import numpy as np
import pytest

from z_manip.planning import NormalizedPlacementRegion
from z_manip_place.core import (
    capture_observed_region_geometry,
    capture_planned_object_geometry,
    ObservedPerceptionIdentity,
    ObservedPlacementRegionGeometry,
    PlaceExecutionCorrelation,
    PlacementContractError,
    PlacementOrientationVerification,
    PlannedObjectGeometry,
    PostReleaseObservation,
    PostReleasePlacementVerifier,
    PostReleaseVerificationConfig,
)


OWNER = {
    'request_id': 'task-request-7',
    'producer_epoch': 'bridge-epoch-4',
    'generation': 3,
    'frame_id': 'wrist_camera_optical_frame',
}
PLANNING_STAMP_NS = 9_900_000_000
RELEASE_STAMP_NS = 10_000_000_000


def _identity(stamp_ns: int, **changes) -> ObservedPerceptionIdentity:
    values = dict(OWNER)
    values.update(changes)
    return ObservedPerceptionIdentity(stamp_ns=stamp_ns, **values)


def _region() -> ObservedPlacementRegionGeometry:
    axis = np.linspace(-0.20, 0.20, 41)
    uu, vv = np.meshgrid(axis, axis)
    return ObservedPlacementRegionGeometry(
        frame_id='base',
        plane_origin=np.zeros(3),
        plane_normal=np.array((0.0, 0.0, 1.0)),
        tangent_u=np.array((1.0, 0.0, 0.0)),
        tangent_v=np.array((0.0, 1.0, 0.0)),
        support_coordinates=np.column_stack((uu.ravel(), vv.ravel())),
    )


def _config(**changes) -> PostReleaseVerificationConfig:
    values = {
        'min_stable_duration_s': 0.50,
        'min_samples': 3,
        'min_target_points': 40,
        'min_current_support_points': 24,
        'max_geometry_samples': 2048,
        'gripper_probe_radius_m': 0.0,
    }
    values.update(changes)
    return PostReleaseVerificationConfig(**values)


def _verifier(
    *,
    planned_object=None,
    observation_start_stamp_ns=RELEASE_STAMP_NS,
    **config_changes,
) -> PostReleasePlacementVerifier:
    verifier = PostReleasePlacementVerifier(_config(**config_changes))
    verifier.arm(
        goal_id='place-3-9900000000',
        identity=_identity(PLANNING_STAMP_NS),
        region=_region(),
        planned_object=(
            _planned_object() if planned_object is None else planned_object
        ),
        expected_joint_names=tuple(f'joint_{index}' for index in range(6)),
    )
    verifier.begin_release(
        gripper_command_id=12,
        gripper_source_stamp_ns=8_200_000_000,
        acknowledgement_stamp_ns=RELEASE_STAMP_NS,
        observation_start_stamp_ns=observation_start_stamp_ns,
    )
    return verifier


def _organized() -> np.ndarray:
    rows, columns = np.indices((80, 100))
    return np.stack((
        (columns - 50.0) * 0.005,
        (rows - 40.0) * 0.005,
        np.zeros_like(columns, dtype=float),
    ), axis=2)


def _target(shift=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray([
        (u, v)
        for v in range(25, 35)
        for u in range(40, 46)
    ], dtype=float)
    index = np.arange(len(pixels))
    angle = 2.0 * np.pi * (index % 10) / 10.0
    radius = 0.020 + (index // 10) * 0.0015
    points = np.column_stack((
        radius * np.cos(angle),
        radius * np.sin(angle),
        0.005 + (index // 10) * 0.045,
    ))
    return points + np.asarray(shift, dtype=float), pixels


def _planned_object(
    *,
    reference_points=None,
    expected_pose=None,
    verification=None,
) -> PlannedObjectGeometry:
    points, _ = _target()
    reference = points - np.array((0.0, 0.0, 0.05))
    pose = np.eye(4)
    pose[:3, 3] = (0.0, 0.0, 0.05)
    return PlannedObjectGeometry(
        frame_id='base',
        expected_pose=pose if expected_pose is None else expected_pose,
        reference_points_object=(
            reference if reference_points is None else reference_points
        ),
        support_normal=np.array((0.0, 0.0, 1.0)),
        verification=(
            PlacementOrientationVerification()
            if verification is None
            else verification
        ),
    )


def _partial_axial_reference(view_arc_degrees: float) -> np.ndarray:
    half_arc = np.deg2rad(view_arc_degrees) / 2.0
    angles = np.linspace(-half_arc, half_arc, 20)
    heights = np.array((-0.08, -0.07, -0.05, 0.0, 0.04, 0.08))
    return np.asarray([
        (0.025 * np.cos(angle), 0.025 * np.sin(angle), height)
        for angle in angles
        for height in heights
    ])


def _asymmetric_full_reference() -> np.ndarray:
    return np.asarray([
        (x, y, z)
        for x in (-0.06, 0.0, 0.06)
        for y in (-0.025, -0.01, 0.005, 0.025)
        for z in (-0.045, -0.03, -0.01, 0.02, 0.055)
    ])


def _observation(
    stamp_ns: int,
    *,
    shift=(0.0, 0.0, 0.0),
    organized=None,
    target_points=None,
    target_pixels=None,
    probes=None,
    identity_changes=None,
    status_stamp_ns=None,
    rgb_stamp_ns=None,
    depth_stamp_ns=None,
    target_stamp_ns=None,
    joint_stamp_ns=None,
    execution_status_received_ns=None,
    joint_names=None,
    planning_from_tool_fk=None,
    planning_from_tool_tf=None,
    gripper_source_stamp_ns=8_200_000_000,
    now_ns=None,
    command_id=12,
    aperture_m=0.075,
) -> PostReleaseObservation:
    default_points, default_pixels = _target(shift)
    organized_value = _organized() if organized is None else np.asarray(
        organized,
        dtype=float,
    ).copy()
    selected_points = default_points if target_points is None else target_points
    selected_pixels = default_pixels if target_pixels is None else target_pixels
    if organized is None:
        integer_pixels = np.rint(selected_pixels).astype(int)
        organized_value[
            integer_pixels[:, 1],
            integer_pixels[:, 0],
        ] = selected_points
    return PostReleaseObservation(
        identity=_identity(
            stamp_ns if status_stamp_ns is None else status_stamp_ns,
            **(identity_changes or {}),
        ),
        rgb_stamp_ns=stamp_ns if rgb_stamp_ns is None else rgb_stamp_ns,
        depth_stamp_ns=stamp_ns if depth_stamp_ns is None else depth_stamp_ns,
        target_stamp_ns=(
            stamp_ns if target_stamp_ns is None else target_stamp_ns
        ),
        joint_stamp_ns=stamp_ns if joint_stamp_ns is None else joint_stamp_ns,
        execution_status_received_ns=(
            stamp_ns + 5_000_000
            if execution_status_received_ns is None
            else execution_status_received_ns
        ),
        now_ns=stamp_ns + 10_000_000 if now_ns is None else now_ns,
        geometry_frame_id='base',
        organized_points=organized_value,
        target_points=selected_points,
        target_pixels_uv=selected_pixels,
        gripper_probe_points=(
            np.array(((0.25, 0.25, 0.20),)) if probes is None else probes
        ),
        joint_names=(
            tuple(f'joint_{index}' for index in range(6))
            if joint_names is None
            else tuple(joint_names)
        ),
        joint_positions=np.linspace(-0.2, 0.3, 6),
        planning_from_tool_fk=(
            np.eye(4)
            if planning_from_tool_fk is None
            else planning_from_tool_fk
        ),
        planning_from_tool_tf=(
            np.eye(4)
            if planning_from_tool_tf is None
            else planning_from_tool_tf
        ),
        gripper_command_id=command_id,
        gripper_source_stamp_ns=gripper_source_stamp_ns,
        gripper_aperture_m=aperture_m,
    )


def test_capture_region_freezes_only_selected_observed_plane_geometry():
    organized = _organized()
    organized[20:30, 20:30, 2] = 0.12
    geometry = capture_observed_region_geometry(
        organized,
        NormalizedPlacementRegion(
            (0.10, 0.10, 0.90, 0.90),
            ((0.10, 0.10, 0.30, 0.40),),
        ),
        frame_id='base',
        plane_origin=np.zeros(3),
        plane_normal=np.array((0.0, 0.0, 1.0)),
        tangent_u=np.array((1.0, 0.0, 0.0)),
        tangent_v=np.array((0.0, 1.0, 0.0)),
        plane_distance_tolerance_m=0.01,
        min_points=80,
        max_points=500,
    )

    assert geometry.frame_id == 'base'
    assert geometry.support_coordinates.shape == (500, 2)
    assert np.max(np.abs(geometry.support_coordinates[:, 0])) <= 0.20
    with pytest.raises(PlacementContractError, match='too few'):
        capture_observed_region_geometry(
            np.full((4, 4, 3), np.nan),
            NormalizedPlacementRegion((0.0, 0.0, 1.0, 1.0)),
            frame_id='base',
            plane_origin=np.zeros(3),
            plane_normal=np.array((0.0, 0.0, 1.0)),
            tangent_u=np.array((1.0, 0.0, 0.0)),
            tangent_v=np.array((0.0, 1.0, 0.0)),
            plane_distance_tolerance_m=0.01,
            min_points=3,
            max_points=10,
        )


def _arm_execution(correlation: PlaceExecutionCorrelation) -> None:
    correlation.arm(
        goal_id='place-3-9900000000',
        executor_epoch='executor-epoch-a',
        trajectory_contract_id='place-3-9900000000',
        trajectory_command_highwater=3,
        gripper_command_highwater=7,
        trajectory_source_highwater_ns=300,
        gripper_source_highwater_ns=350,
    )


def _trajectory(
    correlation: PlaceExecutionCorrelation,
    status: str,
    segment: str,
    command_id: int,
    source_stamp_ns: int,
    **changes,
) -> None:
    values = {
        'executor_epoch': 'executor-epoch-a',
        'trajectory_contract_id': 'place-3-9900000000',
    }
    values.update(changes)
    correlation.observe_trajectory(
        status=status,
        segment=segment,
        command_id=command_id,
        source_stamp_ns=source_stamp_ns,
        **values,
    )


def test_place_execution_requires_ordered_transaction_acknowledgements():
    correlation = PlaceExecutionCorrelation()
    _arm_execution(correlation)
    with pytest.raises(PlacementContractError, match='start with active'):
        _trajectory(correlation, 'succeeded', 'place_approach', 4, 400)
    assert correlation.state == 'invalid'

    _arm_execution(correlation)
    _trajectory(correlation, 'active', 'place_approach', 4, 400)
    with pytest.raises(PlacementContractError, match='command_id changed'):
        _trajectory(correlation, 'succeeded', 'place_approach', 5, 400)

    _arm_execution(correlation)
    with pytest.raises(PlacementContractError, match='must start with active'):
        _trajectory(correlation, 'active', 'place_retreat', 6, 600)

    _arm_execution(correlation)
    with pytest.raises(PlacementContractError, match='high-water'):
        _trajectory(correlation, 'active', 'place_approach', 3, 300)

    _arm_execution(correlation)
    with pytest.raises(PlacementContractError, match='different contract'):
        _trajectory(
            correlation,
            'active',
            'place_approach',
            4,
            400,
            trajectory_contract_id='other-place-goal',
        )


def test_place_execution_binds_release_between_approach_and_retreat():
    correlation = PlaceExecutionCorrelation()
    _arm_execution(correlation)
    _trajectory(correlation, 'active', 'place_approach', 4, 400)
    _trajectory(correlation, 'succeeded', 'place_approach', 4, 400)
    assert correlation.ready_for_release
    assert not correlation.is_new_release(
        7,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=350,
    )
    correlation.observe_gripper_command(
        12,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=500,
    )
    assert correlation.is_new_release(
        12,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=500,
    )
    correlation.observe_release(
        12,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=500,
    )
    _trajectory(correlation, 'active', 'place_retreat', 5, 600)
    _trajectory(correlation, 'succeeded', 'place_retreat', 5, 600)

    assert correlation.complete
    assert correlation.approach_command_id == 4
    assert correlation.release_gripper_command_id == 12
    assert correlation.retreat_command_id == 5


def test_place_execution_ignores_retained_status_without_replaying_progress():
    correlation = PlaceExecutionCorrelation()
    _arm_execution(correlation)
    _trajectory(correlation, 'active', 'place_approach', 4, 400)
    _trajectory(correlation, 'succeeded', 'place_approach', 4, 400)
    _trajectory(correlation, 'succeeded', 'place_approach', 4, 400)
    assert correlation.ready_for_release
    correlation.observe_gripper_command(
        8,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=500,
    )
    correlation.observe_release(
        8,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=500,
    )
    _trajectory(correlation, 'succeeded', 'place_approach', 4, 400)
    assert correlation.state == 'await_retreat_active'


def test_place_execution_rejects_same_command_for_retreat_and_epoch_change():
    correlation = PlaceExecutionCorrelation()
    _arm_execution(correlation)
    _trajectory(correlation, 'active', 'place_approach', 4, 400)
    _trajectory(correlation, 'succeeded', 'place_approach', 4, 400)
    correlation.observe_gripper_command(
        8,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=500,
    )
    correlation.observe_release(
        8,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=500,
    )
    with pytest.raises(PlacementContractError, match='did not follow approach'):
        _trajectory(correlation, 'active', 'place_retreat', 4, 600)

    _arm_execution(correlation)
    with pytest.raises(PlacementContractError, match='epoch changed'):
        _trajectory(
            correlation,
            'active',
            'place_approach',
            4,
            400,
            executor_epoch='executor-epoch-b',
        )

    _arm_execution(correlation)
    _trajectory(correlation, 'active', 'place_approach', 4, 400)
    correlation.observe_gripper_command(
        8,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=450,
    )
    _trajectory(correlation, 'succeeded', 'place_approach', 4, 400)
    assert not correlation.is_new_release(
        8,
        executor_epoch='executor-epoch-a',
        source_stamp_ns=450,
    )


def test_multiple_fresh_frames_publish_acceptance_compatible_verified_contract():
    verifier = _verifier()
    assert verifier.observe(_observation(10_100_000_000)) is None
    assert verifier.observe(_observation(10_350_000_000)) is None
    result = verifier.observe(_observation(10_650_000_000))

    assert result is not None
    payload = result.to_payload()
    assert payload['schema'] == 'z_manip.post_release_verification.v2'
    assert payload['state'] == 'verified'
    assert payload['result'] == 'post_release_target_stable_in_region'
    assert payload['observation_source'] == 'synchronized_rgbd_pointcloud'
    assert payload['goal_id'] == 'place-3-9900000000'
    assert payload['place_goal_id'] == payload['goal_id']
    assert payload['release_gripper_command_id'] == 12
    assert payload['request_id'] == OWNER['request_id']
    assert payload['producer_epoch'] == OWNER['producer_epoch']
    assert payload['generation'] == OWNER['generation']
    assert payload['frame_id'] == OWNER['frame_id']
    assert payload['sample_count'] == 3
    assert payload['target_point_count'] == 60
    assert payload['stable_duration_s'] == pytest.approx(0.55)
    assert payload['max_target_motion_m'] <= 0.025
    assert payload['region_support_fraction'] >= 0.80
    assert payload['target_gripper_clearance_m'] >= 0.04
    assert payload['observation_start_stamp_ns'] == RELEASE_STAMP_NS
    assert payload['first_status_stamp_ns'] == 10_100_000_000
    assert payload['last_status_stamp_ns'] == 10_650_000_000
    assert payload['first_rgb_stamp_ns'] == 10_100_000_000
    assert payload['first_depth_stamp_ns'] == 10_100_000_000
    assert payload['first_target_stamp_ns'] == 10_100_000_000
    assert payload['last_rgb_stamp_ns'] == 10_650_000_000
    assert payload['last_depth_stamp_ns'] == 10_650_000_000
    assert payload['last_target_stamp_ns'] == 10_650_000_000
    assert payload['target_depth_correspondence_max_error_m'] == 0.0
    assert payload['object_orientation_mode'] == 'axial'
    assert payload['object_position_error_m'] == pytest.approx(0.0)
    assert payload['object_upright_error_rad'] == pytest.approx(0.0)
    assert payload['rejected_sample_count'] == 0
    assert payload['rejected_sample_reasons'] == ()
    assert verifier.state == 'verified'


@pytest.mark.parametrize('stale_source', ['rgb', 'depth', 'target', 'status'])
def test_every_observation_source_must_strictly_follow_retreat(stale_source):
    stamp_ns = RELEASE_STAMP_NS + 10_000_000
    changes = {
        'rgb': {'rgb_stamp_ns': RELEASE_STAMP_NS},
        'depth': {'depth_stamp_ns': RELEASE_STAMP_NS},
        'target': {'target_stamp_ns': RELEASE_STAMP_NS},
        'status': {'status_stamp_ns': RELEASE_STAMP_NS},
    }[stale_source]

    result = _verifier().observe(_observation(stamp_ns, **changes))

    assert result is not None and result.state == 'failed'
    assert 'must all be newer than observation start' in result.failure


def test_target_pixels_require_unique_measured_xyz_correspondence():
    points, pixels = _target()
    duplicate_pixels = pixels.copy()
    duplicate_pixels[1] = duplicate_pixels[0]
    duplicate_verifier = _verifier()
    duplicate = duplicate_verifier.observe(_observation(
        10_100_000_000,
        target_points=points,
        target_pixels=duplicate_pixels,
    ))
    assert duplicate is None
    assert 'pixels are not unique' in duplicate_verifier.rejected_sample_reasons[-1]

    organized = _organized()
    integer_pixels = pixels.astype(int)
    organized[integer_pixels[:, 1], integer_pixels[:, 0]] = points
    organized[integer_pixels[0, 1], integer_pixels[0, 0]] = np.nan
    missing_depth_verifier = _verifier()
    missing_depth = missing_depth_verifier.observe(_observation(
        10_100_000_000,
        target_points=points,
        target_pixels=pixels,
        organized=organized,
    ))
    assert missing_depth is None
    assert (
        'lack measured organized depth'
        in missing_depth_verifier.rejected_sample_reasons[-1]
    )

    organized[integer_pixels[0, 1], integer_pixels[0, 0]] = points[0]
    mismatched = points.copy()
    mismatched[0, 0] += 0.03
    mismatch_verifier = _verifier()
    mismatch = mismatch_verifier.observe(_observation(
        10_100_000_000,
        target_points=mismatched,
        target_pixels=pixels,
        organized=organized,
    ))
    assert mismatch is None
    assert (
        'xyz does not match organized depth'
        in mismatch_verifier.rejected_sample_reasons[-1]
    )


def test_axial_spin_is_accepted_but_upright_violation_fails_closed():
    points, pixels = _target()
    center = np.mean(points, axis=0)
    verifier = _verifier()
    result = None
    for stamp_ns, angle in (
        (10_100_000_000, 0.10),
        (10_350_000_000, 0.37),
        (10_650_000_000, 0.72),
    ):
        spin = np.array((
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ))
        spun = (points - center) @ spin.T + center
        result = verifier.observe(_observation(
            stamp_ns,
            target_points=spun,
            target_pixels=pixels,
        ))
    assert result is not None and result.state == 'verified'
    assert result.object_orientation_mode == 'axial'

    roll = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
    lying = (points - center) @ roll.T + center
    upright_verifier = _verifier(
        object_orientation_tolerance_rad=2.0,
        object_position_tolerance_m=0.20,
        max_bottom_height_m=0.20,
    )
    failed = upright_verifier.observe(_observation(
        10_100_000_000,
        target_points=lying,
        target_pixels=pixels,
    ))
    assert failed is None
    assert (
        'violates requested upright orientation'
        in upright_verifier.rejected_sample_reasons[-1]
    )


def test_signed_upright_profile_accepts_normal_and_rejects_inverted_geometry():
    planned = _planned_object()
    points, _ = _target()
    verifier = PostReleasePlacementVerifier(_config())

    metrics = verifier.validate_observed_object_pose(planned, points)
    assert metrics[1] == pytest.approx(0.0)
    assert metrics[7] == pytest.approx((0.0, 0.0, 1.0))

    center = np.mean(points, axis=0)
    inversion = np.diag((1.0, -1.0, -1.0))
    inverted = (points - center) @ inversion.T + center
    with pytest.raises(PlacementContractError, match='orientation'):
        verifier.validate_observed_object_pose(planned, inverted)

    inverted_pose = np.asarray(planned.expected_pose).copy()
    inverted_pose[:3, :3] = inversion
    with pytest.raises(PlacementContractError, match='planned object pose'):
        verifier.validate_observed_object_pose(
            _planned_object(expected_pose=inverted_pose),
            points,
        )


def test_upright_direction_must_be_observable_in_frozen_and_current_geometry():
    angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    heights = np.linspace(-0.10, 0.10, 6)
    symmetric = np.asarray([
        (0.025 * np.cos(angle), 0.025 * np.sin(angle), height)
        for height in heights
        for angle in angles
    ])
    with pytest.raises(PlacementContractError, match='not observable'):
        _verifier(planned_object=_planned_object(reference_points=symmetric))

    planned = _planned_object()
    expected_points, _ = _target()
    point_angles = 2.0 * np.pi * (np.arange(len(expected_points)) % 10) / 10.0
    unobservable_observation = np.column_stack((
        0.025 * np.cos(point_angles),
        0.025 * np.sin(point_angles),
        expected_points[:, 2],
    ))
    verifier = PostReleasePlacementVerifier(_config())
    with pytest.raises(PlacementContractError, match='not observable'):
        verifier.validate_observed_object_pose(
            planned,
            unobservable_observation,
        )


def test_signed_upright_profile_preserves_full_orientation_observability():
    reference = _asymmetric_full_reference()
    planned = _planned_object(
        reference_points=reference,
        verification=PlacementOrientationVerification(
            require_upright=True,
            orientation_symmetry='none',
        ),
    )
    observed = reference + np.asarray(planned.expected_pose)[:3, 3]
    verifier = PostReleasePlacementVerifier(_config())

    metrics = verifier.validate_observed_object_pose(planned, observed)

    assert metrics[1] == pytest.approx(0.0)
    assert metrics[6] == pytest.approx(np.eye(3))
    assert metrics[7] == pytest.approx((0.0, 0.0, 1.0))


def test_partial_view_axial_geometry_accepts_continuous_spin():
    reference = _partial_axial_reference(265.0)
    eigenvalues = np.linalg.eigvalsh(np.cov(reference.T, bias=True))
    transverse_ratio = eigenvalues[1] / eigenvalues[0]
    assert 1.84 < transverse_ratio < 1.90
    expected_pose = np.eye(4)
    expected_pose[:3, 3] = (0.0, 0.0, 0.08)
    planned = PlannedObjectGeometry(
        frame_id='base',
        expected_pose=expected_pose,
        reference_points_object=reference,
        support_normal=np.array((0.0, 0.0, 1.0)),
        verification=PlacementOrientationVerification(
            require_upright=True,
            upright_axis_object=(0.0, 0.0, 1.0),
            orientation_symmetry='axial',
            symmetry_axis_object=(0.0, 0.0, 1.0),
        ),
    )
    angle = 0.47
    spin = np.array((
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    ))
    observed = reference @ spin.T + expected_pose[:3, 3]
    verifier = PostReleasePlacementVerifier(_config(
        max_axial_transverse_ratio=1.90,
    ))

    position_error, orientation_error, *_ = (
        verifier.validate_observed_object_pose(planned, observed)
    )

    assert position_error < 0.004
    assert orientation_error == pytest.approx(0.0)


def test_partial_view_axial_geometry_rejects_ratio_above_boundary():
    reference = _partial_axial_reference(260.0)
    eigenvalues = np.linalg.eigvalsh(np.cov(reference.T, bias=True))
    transverse_ratio = eigenvalues[1] / eigenvalues[0]
    assert transverse_ratio >= 1.90
    expected_pose = np.eye(4)
    expected_pose[:3, 3] = (0.0, 0.0, 0.08)
    planned = PlannedObjectGeometry(
        frame_id='base',
        expected_pose=expected_pose,
        reference_points_object=reference,
        support_normal=np.array((0.0, 0.0, 1.0)),
        verification=PlacementOrientationVerification(
            require_upright=True,
            upright_axis_object=(0.0, 0.0, 1.0),
            orientation_symmetry='axial',
            symmetry_axis_object=(0.0, 0.0, 1.0),
        ),
    )
    verifier = PostReleasePlacementVerifier(_config(
        max_axial_transverse_ratio=1.90,
    ))

    with pytest.raises(PlacementContractError, match='transverse variances'):
        verifier.validate_observed_object_pose(
            planned,
            reference + expected_pose[:3, 3],
        )


def test_side_grasp_bottle_frame_is_observed_not_assumed_from_tool_axes():
    reference, pixels = _target()
    reference = reference - np.array((0.0, 0.0, 0.05))
    expected_pose = np.eye(4)
    expected_pose[:3, 3] = (0.0, 0.0, 0.05)
    planned = capture_planned_object_geometry(
        reference,
        frame_id='base',
        expected_object_pose=expected_pose,
        support_normal=(0.0, 0.0, 1.0),
        verification=PlacementOrientationVerification(
            require_upright=True,
            upright_axis_object=(0.0, 0.0, 1.0),
            orientation_symmetry='axial',
            symmetry_axis_object=(0.0, 0.0, 1.0),
        ),
        min_points=40,
        max_points=100,
    )
    assert planned.reference_points_object == pytest.approx(reference)

    verifier = _verifier(planned_object=planned)
    for stamp_ns in (10_100_000_000, 10_350_000_000):
        assert verifier.observe(_observation(
            stamp_ns,
            target_points=reference + expected_pose[:3, 3],
            target_pixels=pixels,
        )) is None
    result = verifier.observe(_observation(
        10_650_000_000,
        target_points=reference + expected_pose[:3, 3],
        target_pixels=pixels,
    ))
    assert result is not None and result.state == 'verified'


def test_explicit_non_upright_asymmetric_box_pose_is_verified_in_full_3d():
    reference = np.asarray([
        (x, y, z)
        for x in (-0.04, 0.0, 0.04)
        for y in (-0.025, -0.008, 0.008, 0.025)
        for z in (-0.012, -0.006, 0.0, 0.006, 0.012)
    ])
    pose = np.eye(4)
    pose[:3, :3] = np.array((
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ))
    pose[:3, 3] = (0.0, 0.0, 0.04)
    planned = _planned_object(
        reference_points=reference,
        expected_pose=pose,
        verification=PlacementOrientationVerification(
            require_upright=False,
            upright_axis_object=(0.0, 0.0, 1.0),
            orientation_symmetry='none',
        ),
    )
    observed = reference @ pose[:3, :3].T + pose[:3, 3]
    pixels = _target()[1][:len(reference)]
    verifier = _verifier(planned_object=planned)
    for stamp_ns in (10_100_000_000, 10_350_000_000):
        assert verifier.observe(_observation(
            stamp_ns,
            target_points=observed,
            target_pixels=pixels,
        )) is None
    result = verifier.observe(_observation(
        10_650_000_000,
        target_points=observed,
        target_pixels=pixels,
    ))
    assert result is not None and result.state == 'verified'
    assert result.object_orientation_mode == 'full'
    assert result.object_upright_error_rad == pytest.approx(np.pi / 2.0)


def test_triaxial_box_cannot_claim_axial_symmetry():
    reference = np.asarray([
        (x, y, z)
        for x in (-0.04, 0.0, 0.04)
        for y in (-0.025, -0.008, 0.008, 0.025)
        for z in (-0.012, -0.006, 0.0, 0.006, 0.012)
    ])
    planned = _planned_object(
        reference_points=reference,
        verification=PlacementOrientationVerification(
            require_upright=True,
            upright_axis_object=(0.0, 0.0, 1.0),
            orientation_symmetry='axial',
            symmetry_axis_object=(0.0, 0.0, 1.0),
        ),
    )
    with pytest.raises(PlacementContractError, match='transverse variances'):
        _verifier(planned_object=planned)


def test_current_observation_detects_attachment_slip_against_frozen_model():
    reference = np.asarray([
        (x, y, z)
        for x in (-0.04, 0.0, 0.04)
        for y in (-0.025, -0.008, 0.008, 0.025)
        for z in (-0.012, -0.006, 0.0, 0.006, 0.012)
    ])
    expected_pose = np.eye(4)
    expected_pose[:3, 3] = (0.1, -0.2, 0.3)
    planned = PlannedObjectGeometry(
        frame_id='base',
        expected_pose=expected_pose,
        reference_points_object=reference,
        support_normal=np.array((0.0, 0.0, 1.0)),
        verification=PlacementOrientationVerification(
            require_upright=False,
            orientation_symmetry='none',
        ),
    )
    quarter_turn = np.array((
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ))
    slipped = reference @ quarter_turn.T + expected_pose[:3, 3]
    verifier = PostReleasePlacementVerifier(_config())

    with pytest.raises(PlacementContractError, match='orientation'):
        verifier.validate_observed_object_pose(planned, slipped)

    metrics = verifier.validate_observed_object_pose(
        planned,
        reference + expected_pose[:3, 3],
    )
    assert metrics[0] == pytest.approx(0.0)
    assert metrics[1] == pytest.approx(0.0)


def test_fully_degenerate_object_orientation_is_rejected_before_release():
    index = np.arange(60, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / len(index)
    azimuth = index * np.pi * (3.0 - np.sqrt(5.0))
    radius = np.sqrt(1.0 - z * z)
    isotropic = 0.03 * np.column_stack((
        radius * np.cos(azimuth),
        radius * np.sin(azimuth),
        z,
    ))
    with pytest.raises(PlacementContractError, match='fully degenerate'):
        _verifier(planned_object=_planned_object(reference_points=isotropic))


@pytest.mark.parametrize('identity_changes', [
    {'request_id': 'different-request'},
    {'producer_epoch': 'different-producer'},
    {'generation': 4},
    {'frame_id': 'different_camera'},
])
def test_mixed_perception_ownership_fails_closed(identity_changes):
    verifier = _verifier()
    result = verifier.observe(_observation(
        10_100_000_000,
        identity_changes=identity_changes,
    ))

    assert result is not None
    assert result.state == 'failed'
    assert 'mixed request/producer/generation/frame' in result.failure
    assert result.sample_count == 0


@pytest.mark.parametrize(('change', 'failure'), [
    ({'rgb_stamp_ns': 10_050_000_000}, 'not synchronized'),
    ({'target_stamp_ns': 10_100_000_001}, 'do not match exactly'),
    ({'joint_stamp_ns': 9_800_000_000}, 'predates observation start'),
    ({'now_ns': 10_600_000_000}, 'observation is stale'),
    ({'command_id': 13}, 'gripper command ID changed'),
    ({'aperture_m': 0.02}, 'gripper is not measured open'),
])
def test_mixed_stale_or_wrong_robot_feedback_fails_closed(change, failure):
    verifier = _verifier()
    result = verifier.observe(_observation(10_100_000_000, **change))

    assert result is not None
    assert result.state == 'failed'
    assert failure in result.failure


def test_joint_names_fk_and_gripper_source_are_strictly_correlated():
    wrong_names = _verifier().observe(_observation(
        10_100_000_000,
        joint_names=tuple(f'other_{index}' for index in range(6)),
    ))
    assert wrong_names is not None
    assert 'measured arm state is malformed' in wrong_names.failure

    shifted_fk = np.eye(4)
    shifted_fk[0, 3] = 0.05
    wrong_fk = _verifier().observe(_observation(
        10_100_000_000,
        planning_from_tool_fk=shifted_fk,
    ))
    assert wrong_fk is not None
    assert 'joints disagree with stamped tool TF' in wrong_fk.failure

    wrong_source = _verifier().observe(_observation(
        10_100_000_000,
        gripper_source_stamp_ns=8_200_000_001,
    ))
    assert wrong_source is not None
    assert 'gripper source identity changed' in wrong_source.failure


def test_bad_geometry_resets_dwell_and_records_bounded_diagnostics():
    cases = []
    points, pixels = _target()
    occluded = np.full((80, 100, 3), np.nan)
    integer_pixels = pixels.astype(int)
    occluded[integer_pixels[:, 1], integer_pixels[:, 0]] = points
    cases.append(_observation(
        10_100_000_000,
        target_points=points[:10],
        target_pixels=pixels[:10],
    ))
    cases.append(_observation(
        10_100_000_000,
        organized=occluded,
    ))
    cases.append(_observation(
        10_100_000_000,
        shift=(0.35, 0.0, 0.0),
    ))
    cases.append(_observation(
        10_100_000_000,
        probes=points[[0]],
    ))

    failures = []
    for observation in cases:
        verifier = _verifier()
        result = verifier.observe(observation)
        assert result is None
        assert verifier.samples == []
        assert verifier.rejected_sample_count == 1
        failures.append(verifier.rejected_sample_reasons[-1])
    assert 'insufficient finite points' in failures[0]
    assert 'support is occluded' in failures[1]
    assert 'support inside selected region is insufficient' in failures[2]
    assert 'gripper has not cleared target' in failures[3]


def test_good_bad_good_geometry_restarts_dwell_then_can_verify():
    verifier = _verifier(observation_timeout_s=1.2)
    assert verifier.observe(_observation(10_100_000_000)) is None
    occluded = np.full((80, 100, 3), np.nan)
    points, pixels = _target()
    integer_pixels = pixels.astype(int)
    occluded[integer_pixels[:, 1], integer_pixels[:, 0]] = points
    assert verifier.observe(_observation(
        10_200_000_000,
        organized=occluded,
    )) is None
    assert verifier.samples == []
    assert verifier.rejected_sample_count == 1
    assert verifier.observe(_observation(10_300_000_000)) is None
    assert verifier.observe(_observation(10_550_000_000)) is None
    result = verifier.observe(_observation(10_850_000_000))

    assert result is not None and result.state == 'verified'
    assert result.first_observation_stamp_ns == 10_300_000_000
    assert result.rejected_sample_count == 1


def test_persistent_bad_geometry_fails_only_at_valid_observation_timeout():
    verifier = _verifier(
        observation_timeout_s=0.60,
        max_rejection_diagnostics=1,
    )
    points, pixels = _target()
    duplicate_pixels = pixels.copy()
    duplicate_pixels[1] = duplicate_pixels[0]
    for stamp_ns in (10_100_000_000, 10_300_000_000):
        assert verifier.observe(_observation(
            stamp_ns,
            target_points=points,
            target_pixels=duplicate_pixels,
        )) is None
    assert verifier.rejected_sample_count == 2
    assert len(verifier.rejected_sample_reasons) == 1

    result = verifier.tick(10_600_000_001)

    assert result is not None and result.state == 'failed'
    assert 'timed out or is occluded' in result.failure
    assert result.rejected_sample_count == 2
    assert len(result.rejected_sample_reasons) == 1


def test_motion_and_frame_gap_restart_the_continuous_stability_window():
    verifier = _verifier()
    assert verifier.observe(_observation(10_100_000_000)) is None
    assert verifier.observe(_observation(10_300_000_000)) is None
    assert verifier.observe(_observation(
        10_400_000_000,
        shift=(0.03, 0.0, 0.0),
    )) is None
    assert len(verifier.samples) == 1
    assert verifier.observe(_observation(
        10_500_000_000,
        shift=(0.03, 0.0, 0.0),
    )) is None
    assert verifier.observe(_observation(
        10_750_000_000,
        shift=(0.03, 0.0, 0.0),
    )) is None
    result = verifier.observe(_observation(
        11_000_000_000,
        shift=(0.03, 0.0, 0.0),
    ))

    assert result is not None and result.state == 'verified'
    assert result.sample_count == 4
    assert result.first_observation_stamp_ns == 10_400_000_000

    gap_verifier = _verifier()
    gap_verifier.observe(_observation(10_100_000_000))
    gap_verifier.observe(_observation(10_450_000_000))
    assert len(gap_verifier.samples) == 1


def test_fixed_center_axis_swing_restarts_orientation_stability_window():
    points, pixels = _target()
    center = np.mean(points, axis=0)
    verifier = _verifier(
        object_orientation_tolerance_rad=0.35,
        upright_tolerance_rad=0.35,
        max_object_orientation_motion_rad=0.10,
    )

    for stamp_ns, angle in (
        (10_100_000_000, -0.08),
        (10_350_000_000, 0.08),
        (10_650_000_000, -0.08),
    ):
        swing = np.array((
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle)),
            (0.0, np.sin(angle), np.cos(angle)),
        ))
        swung = (points - center) @ swing.T + center
        assert verifier.observe(_observation(
            stamp_ns,
            target_points=swung,
            target_pixels=pixels,
        )) is None

    assert len(verifier.samples) == 1
    assert verifier.samples[0].stamp_ns == 10_650_000_000


def test_fixed_center_full_orientation_yaw_restarts_stability_window():
    reference = _asymmetric_full_reference()
    planned = _planned_object(
        reference_points=reference,
        verification=PlacementOrientationVerification(
            require_upright=True,
            orientation_symmetry='none',
        ),
    )
    translation = np.asarray(planned.expected_pose)[:3, 3]
    points = reference + translation
    pixels = _target()[1]
    center = np.mean(points, axis=0)
    verifier = _verifier(
        planned_object=planned,
        max_object_orientation_motion_rad=0.10,
    )

    for stamp_ns, angle in (
        (10_100_000_000, -0.08),
        (10_350_000_000, 0.08),
        (10_650_000_000, -0.08),
    ):
        yaw = np.array((
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        ))
        rotated = (points - center) @ yaw.T + center
        assert verifier.observe(_observation(
            stamp_ns,
            target_points=rotated,
            target_pixels=pixels,
        )) is None

    assert len(verifier.samples) == 1
    assert verifier.samples[0].stamp_ns == 10_650_000_000


def test_duplicate_is_not_a_sample_but_source_or_ros_clock_rollback_fails():
    verifier = _verifier()
    observation = _observation(10_200_000_000)
    assert verifier.observe(observation) is None
    assert verifier.observe(replace(
        observation,
        now_ns=10_220_000_000,
        execution_status_received_ns=10_215_000_000,
    )) is None
    assert len(verifier.samples) == 1
    result = verifier.observe(_observation(
        10_150_000_000,
        now_ns=10_230_000_000,
        execution_status_received_ns=10_225_000_000,
    ))
    assert result is not None
    assert 'source clock moved backwards' in result.failure

    clock_verifier = _verifier()
    result = clock_verifier.tick(RELEASE_STAMP_NS - 1)
    assert result is not None
    assert 'clock moved backwards' in result.failure


def test_missing_post_release_target_times_out_as_occlusion():
    verifier = _verifier(observation_timeout_s=0.60)
    assert verifier.tick(RELEASE_STAMP_NS + 500_000_000) is None
    result = verifier.tick(RELEASE_STAMP_NS + 600_000_001)

    assert result is not None
    assert result.state == 'failed'
    assert 'timed out or is occluded' in result.failure
    assert result.to_payload()['sample_count'] == 0


def test_contract_rejects_invalid_thresholds_and_nonorthogonal_region_basis():
    with pytest.raises(ValueError, match='support fraction'):
        _config(min_region_support_fraction=0.0)
    with pytest.raises(ValueError, match='geometry margins'):
        _config(max_plane_penetration_m=float('nan'))
    with pytest.raises(ValueError, match='axial transverse'):
        _config(max_axial_transverse_ratio=float('nan'))
    with pytest.raises(ValueError, match='metric and timing'):
        _config(min_signed_upright_profile_asymmetry=0.0)
    with pytest.raises(ValueError, match='profile alignment'):
        _config(min_signed_upright_profile_alignment=0.0)
    with pytest.raises(ValueError, match='metric and timing'):
        _config(max_object_orientation_motion_rad=0.0)
    with pytest.raises(PlacementContractError, match='orthonormal'):
        ObservedPlacementRegionGeometry(
            frame_id='base',
            plane_origin=np.zeros(3),
            plane_normal=np.array((0.0, 0.0, 1.0)),
            tangent_u=np.array((1.0, 0.0, 0.0)),
            tangent_v=np.array((1.0, 0.0, 0.0)),
            support_coordinates=np.zeros((3, 2)),
        )


def test_terminal_payload_contains_no_nonfinite_json_values():
    verifier = _verifier()
    verifier.observe(_observation(10_100_000_000))
    verifier.observe(_observation(10_350_000_000))
    result = verifier.observe(_observation(10_650_000_000))

    assert result is not None
    json.dumps(result.to_payload(), allow_nan=False)
