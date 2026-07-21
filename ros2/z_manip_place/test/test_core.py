"""Tests for platform-neutral observed placement ROS contracts."""

import json

import numpy as np
import pytest

from z_manip.models.planner import PlanningError
from z_manip.planning import ObservedPlacementConfig, ObservedPlacementPlanner
from z_manip_place.core import (
    backproject_depth,
    combine_trajectory_segments,
    EvaluatedPlacementMotion,
    parse_region_request,
    PlacementContractError,
    PlacementCoordinator,
    PlacementPerceptionSnapshot,
    RawTrajectorySegment,
)


def _request_payload(**changes):
    reference = [
        [x, y, z]
        for x in (-0.03, 0.0, 0.03)
        for y in (-0.025, -0.008, 0.008, 0.025)
        for z in (-0.04, 0.0, 0.04)
    ]
    reference.extend([
        [-0.015, -0.016, z]
        for z in (-0.03, -0.01, 0.01, 0.03)
    ])
    payload = {
        'schema_version': 2,
        'goal_id': 'place-12',
        'stamp_ns': 10_000_000_000,
        'image_frame': 'camera_optical',
        'request_id': 'place-observation-12',
        'producer_epoch': 'place-producer-3',
        'executor_epoch': 'executor-epoch-3',
        'generation': 12,
        'region_xyxy': [0.0, 0.0, 1.0, 1.0],
        'avoid_xyxy': [],
        'constraints': {'min_clearance_m': 0.01},
        'object_extent_m': [0.06, 0.05, 0.08],
        'tool_from_object': np.eye(4).tolist(),
        'object_reference_points_object': reference,
        'object_reference_identity': {
            'request_id': 'grasp-observation-7',
            'producer_epoch': 'grasp-producer-2',
            'generation': 7,
            'observation_stamp_ns': 9_000_000_000,
            'frame_id': 'camera_optical',
        },
        'verification': {
            'require_upright': True,
            'upright_axis_object': [0.0, 0.0, 1.0],
            'orientation_symmetry': 'axial',
            'symmetry_axis_object': [0.0, 0.0, 1.0],
        },
    }
    payload.update(changes)
    return json.dumps(payload)


def _segments():
    return (
        RawTrajectorySegment(
            'transit', ('j2', 'j1'),
            np.array(((0.0, 0.0), (0.2, 0.1))), np.array((0.0, 0.4)),
        ),
        RawTrajectorySegment(
            'approach', ('j1', 'j2'),
            np.array(((0.1, 0.2), (0.15, 0.25))), np.array((0.0, 0.2)),
        ),
        RawTrajectorySegment(
            'retreat', ('j2', 'j1'),
            np.array(((0.25, 0.15), (0.1, 0.0))), np.array((0.0, 0.3)),
        ),
    )


def _snapshot(
    stamp_ns=10_000_000_000,
    *,
    depth_offset_ns=0,
    camera_info_offset_ns=0,
    scene_offset_ns=4_000_000,
    image_frame='camera_optical',
):
    coordinates = np.linspace(-0.20, 0.20, 25)
    xx, yy = np.meshgrid(coordinates, coordinates)
    organized = np.stack((xx, yy, np.zeros_like(xx)), axis=2)
    return PlacementPerceptionSnapshot(
        rgb_stamp_ns=stamp_ns,
        depth_stamp_ns=stamp_ns + depth_offset_ns,
        camera_info_stamp_ns=stamp_ns + camera_info_offset_ns,
        scene_stamp_ns=stamp_ns + scene_offset_ns,
        joint_stamp_ns=stamp_ns + scene_offset_ns,
        image_frame=image_frame,
        frame_id='base',
        organized_points=organized,
        scene_points=organized.reshape(-1, 3),
        gravity=np.array((0.0, 0.0, -1.0)),
        joint_names=('gripper', 'j2', 'j1'),
        joint_positions=np.array((0.02, 0.0, 0.0)),
    )


def _trajectory(goal_id='place-12'):
    return combine_trajectory_segments(
        goal_id=goal_id,
        frame_id='base',
        expected_joint_names=('j1', 'j2'),
        start_positions=np.zeros(2),
        segments=_segments(),
        continuity_tolerance_rad=0.01,
    )


def test_region_json_is_versioned_normalized_and_rejects_unknown_ground_truth():
    request = parse_region_request(_request_payload(
        avoid_xyxy=[[0.4, 0.4, 0.6, 0.6]],
        constraints={
            'min_clearance_m': 0.03,
            'preferred_yaw_rad': 1.2,
            'yaw_tolerance_rad': 0.4,
        },
    ))
    assert request.goal_id == 'place-12'
    assert request.executor_epoch == 'executor-epoch-3'
    assert request.region.avoid_xyxy == ((0.4, 0.4, 0.6, 0.6),)
    assert request.constraints.preferred_yaw_rad == pytest.approx(1.2)
    assert request.verification.require_upright is True
    assert request.verification.orientation_symmetry == 'axial'
    assert request.observation_identity.request_id == 'place-observation-12'
    assert request.object_reference_identity.request_id == 'grasp-observation-7'
    assert request.object_reference_identity.stamp_ns < request.stamp_ns
    assert request.object_reference_points_object.shape == (40, 3)
    assert not request.object_reference_points_object.flags.writeable
    assert not request.object_extent_m.flags.writeable
    assert not request.tool_from_object.flags.writeable

    payload = json.loads(_request_payload())
    payload['sim_object_pose'] = [1.0, 2.0, 3.0]
    with pytest.raises(PlacementContractError, match='unknown'):
        parse_region_request(json.dumps(payload))
    with pytest.raises(PlacementContractError, match='schema_version'):
        parse_region_request(_request_payload(schema_version=1))
    with pytest.raises(PlacementContractError, match='JSON integer'):
        parse_region_request(_request_payload(stamp_ns=True))
    payload = json.loads(_request_payload())
    del payload['verification']
    with pytest.raises(PlacementContractError, match='missing'):
        parse_region_request(json.dumps(payload))
    payload = json.loads(_request_payload())
    del payload['executor_epoch']
    with pytest.raises(PlacementContractError, match='missing'):
        parse_region_request(json.dumps(payload))
    with pytest.raises(PlacementContractError, match='identity'):
        parse_region_request(_request_payload(executor_epoch=''))
    payload = json.loads(_request_payload())
    payload['verification'] = {}
    with pytest.raises(PlacementContractError, match='verification keys mismatch'):
        parse_region_request(json.dumps(payload))


def test_region_json_rejects_swapped_or_malformed_reference_identity():
    swapped = json.loads(_request_payload())
    swapped['object_reference_identity'] = {
        'request_id': swapped['request_id'],
        'producer_epoch': swapped['producer_epoch'],
        'generation': swapped['generation'],
        'observation_stamp_ns': swapped['stamp_ns'],
        'frame_id': swapped['image_frame'],
    }
    with pytest.raises(PlacementContractError, match='must predate'):
        parse_region_request(json.dumps(swapped))

    malformed = json.loads(_request_payload())
    malformed['object_reference_identity']['unexpected'] = 'ground-truth-owner'
    with pytest.raises(PlacementContractError, match='keys must be exactly'):
        parse_region_request(json.dumps(malformed))

    tampered_extent = json.loads(_request_payload())
    tampered_extent['object_extent_m'][0] += 0.001
    with pytest.raises(PlacementContractError, match='extent does not match'):
        parse_region_request(json.dumps(tampered_extent))

    boolean_extent = json.loads(_request_payload())
    boolean_extent['object_extent_m'][0] = True
    with pytest.raises(PlacementContractError, match='cannot contain JSON booleans'):
        parse_region_request(json.dumps(boolean_extent))

    nonfinite = _request_payload().replace('0.06', 'NaN', 1)
    with pytest.raises(PlacementContractError, match='non-finite'):
        parse_region_request(nonfinite)


def test_aligned_depth_backprojects_through_measured_tf_and_keeps_image_shape():
    depth = np.array(((1.0, 0.0), (2.0, 1.5)), dtype=float)
    matrix = np.array(((100.0, 0.0, 0.5), (0.0, 100.0, 0.5), (0.0, 0.0, 1.0)))
    transform = np.eye(4)
    transform[:3, 3] = (0.3, -0.1, 0.2)

    points = backproject_depth(
        depth, matrix, transform, min_depth_m=0.2, max_depth_m=1.8,
    )

    assert points.shape == (2, 2, 3)
    assert np.all(np.isnan(points[0, 1]))
    assert np.all(np.isnan(points[1, 0]))
    assert points[0, 0, 2] == pytest.approx(1.2)
    assert points[1, 1, 0] > 0.3


def test_trajectory_contract_reorders_joints_and_joins_all_three_phases():
    trajectory = _trajectory()

    assert trajectory.joint_names == ('j1', 'j2')
    assert trajectory.points[1].positions == pytest.approx((0.1, 0.2))
    assert [name for name, _index in trajectory.phase_start_indices] == [
        'transit', 'approach', 'retreat',
    ]
    assert all(
        second.time_from_start_s > first.time_from_start_s
        for first, second in zip(trajectory.points, trajectory.points[1:])
    )
    assert trajectory.points[-1].phase == 'retreat'


def test_trajectory_contract_fails_on_phase_discontinuity_or_wrong_joint_set():
    broken = list(_segments())
    broken[1] = RawTrajectorySegment(
        'approach', ('j1', 'j2'),
        np.array(((1.0, 1.0), (1.1, 1.1))), np.array((0.0, 0.2)),
    )
    with pytest.raises(PlacementContractError, match='prior state'):
        combine_trajectory_segments(
            goal_id='place-12', frame_id='base', expected_joint_names=('j1', 'j2'),
            start_positions=np.zeros(2), segments=broken,
            continuity_tolerance_rad=0.01,
        )
    wrong = list(_segments())
    wrong[2] = RawTrajectorySegment(
        'retreat', ('j1', 'j3'), np.zeros((2, 2)), np.array((0.0, 0.1)),
    )
    with pytest.raises(PlacementContractError, match='configured arm'):
        combine_trajectory_segments(
            goal_id='place-12', frame_id='base', expected_joint_names=('j1', 'j2'),
            start_positions=np.zeros(2), segments=wrong,
            continuity_tolerance_rad=0.01,
        )


def test_coordinator_uses_one_snapshot_and_records_motion_candidate_audits():
    planner = ObservedPlacementPlanner(ObservedPlacementConfig(
        ransac_iterations=80,
        min_plane_points=50,
        sample_spacing_m=0.05,
        support_neighbor_radius_m=0.03,
        footprint_samples_per_axis=3,
        yaw_samples=2,
        max_geometric_candidates=8,
    ))
    coordinator = PlacementCoordinator(
        planner,
        expected_joint_names=('j1', 'j2'),
        max_sync_skew_s=0.05,
        max_snapshot_age_s=0.25,
    )
    calls = 0

    def evaluate(candidate, current):
        nonlocal calls
        calls += 1
        np.testing.assert_allclose(current, (0.0, 0.0))
        if calls == 1:
            raise PlanningError('fake continuous collision rejection')
        return EvaluatedPlacementMotion(
            score=float(candidate.support_position[0]), trajectory=_trajectory(),
        )

    output = coordinator.plan(
        parse_region_request(_request_payload()),
        _snapshot(),
        now_ns=10_100_000_000,
        evaluate=evaluate,
    )

    assert output.goal_id == 'place-12'
    assert len(output.candidates) == calls
    assert not output.candidates[0].feasible
    assert any(audit.feasible for audit in output.candidates)
    assert output.trajectory.points[-1].phase == 'retreat'


def test_coordinator_fails_closed_on_stale_or_unsynchronized_inputs():
    planner = ObservedPlacementPlanner(ObservedPlacementConfig(
        ransac_iterations=10, min_plane_points=20,
    ))
    coordinator = PlacementCoordinator(
        planner,
        expected_joint_names=('j1', 'j2'),
        max_sync_skew_s=0.02,
        max_snapshot_age_s=0.10,
    )
    calls = 0

    def evaluate(*_args):
        nonlocal calls
        calls += 1
        return EvaluatedPlacementMotion(0.0, _trajectory())

    with pytest.raises(PlacementContractError, match='must match exactly'):
        coordinator.plan(
            parse_region_request(_request_payload()),
            _snapshot(depth_offset_ns=1),
            now_ns=10_010_000_000,
            evaluate=evaluate,
        )
    with pytest.raises(PlacementContractError, match='must match exactly'):
        coordinator.plan(
            parse_region_request(_request_payload()),
            _snapshot(camera_info_offset_ns=1),
            now_ns=10_010_000_000,
            evaluate=evaluate,
        )
    with pytest.raises(PlacementContractError, match='image frame'):
        coordinator.plan(
            parse_region_request(_request_payload()),
            _snapshot(image_frame='old_camera_frame'),
            now_ns=10_010_000_000,
            evaluate=evaluate,
        )
    with pytest.raises(PlacementContractError, match='not synchronized'):
        coordinator.plan(
            parse_region_request(_request_payload()),
            _snapshot(scene_offset_ns=100_000_000),
            now_ns=10_110_000_000,
            evaluate=evaluate,
        )
    with pytest.raises(PlacementContractError, match='stale'):
        coordinator.plan(
            parse_region_request(_request_payload()),
            _snapshot(),
            now_ns=10_500_000_000,
            evaluate=evaluate,
        )
    assert calls == 0
