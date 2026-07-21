"""Strict task-side contract for observed post-release placement evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np


POST_RELEASE_VERIFICATION_SCHEMA = 'z_manip.post_release_verification.v2'
POST_RELEASE_VERIFICATION_RESULT = 'post_release_target_stable_in_region'
POST_RELEASE_FAILURE_RESULT = 'post_release_verification_failed'
POST_RELEASE_OBSERVATION_SOURCE = 'synchronized_rgbd_pointcloud'


class PostReleaseVerificationError(ValueError):
    """Raised when placement evidence is malformed or not task-correlated."""


@dataclass(frozen=True)
class PlacementObservationIdentity:
    """Exact perception owner and source frame used to plan one placement."""

    goal_id: str
    request_id: str
    producer_epoch: str
    generation: int
    frame_id: str
    planning_observation_stamp_ns: int
    require_upright: bool

    def __post_init__(self) -> None:
        for label, value in (
            ('goal_id', self.goal_id),
            ('request_id', self.request_id),
            ('producer_epoch', self.producer_epoch),
            ('frame_id', self.frame_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise PostReleaseVerificationError(f'{label} must be non-empty')
        _positive_int(self.generation, 'generation')
        _positive_int(
            self.planning_observation_stamp_ns,
            'planning_observation_stamp_ns',
        )
        if not isinstance(self.require_upright, bool):
            raise PostReleaseVerificationError('require_upright must be boolean')


@dataclass(frozen=True)
class PostReleaseVerificationPolicy:
    """Task-owned minimum evidence quality and bounded wait contract."""

    timeout_s: float = 2.0
    wall_timeout_s: float = 12.0
    min_stable_duration_s: float = 0.50
    min_samples: int = 3
    min_target_points: int = 24
    max_target_motion_m: float = 0.025
    min_region_support_fraction: float = 0.80
    min_gripper_clearance_m: float = 0.04
    max_rgbd_target_skew_s: float = 0.025
    max_joint_target_skew_s: float = 0.12
    max_target_depth_correspondence_m: float = 0.012
    max_object_position_error_m: float = 0.04
    max_object_orientation_error_rad: float = 0.35
    max_object_upright_error_rad: float = 0.26
    min_object_registration_inlier_fraction: float = 0.55
    max_object_registration_rms_m: float = 0.025

    def __post_init__(self) -> None:
        positive_floats = (
            ('timeout_s', self.timeout_s),
            ('wall_timeout_s', self.wall_timeout_s),
            ('min_stable_duration_s', self.min_stable_duration_s),
            ('min_gripper_clearance_m', self.min_gripper_clearance_m),
            ('max_rgbd_target_skew_s', self.max_rgbd_target_skew_s),
            ('max_joint_target_skew_s', self.max_joint_target_skew_s),
            (
                'max_target_depth_correspondence_m',
                self.max_target_depth_correspondence_m,
            ),
            ('max_object_position_error_m', self.max_object_position_error_m),
            (
                'max_object_orientation_error_rad',
                self.max_object_orientation_error_rad,
            ),
            ('max_object_upright_error_rad', self.max_object_upright_error_rad),
            ('max_object_registration_rms_m', self.max_object_registration_rms_m),
        )
        for label, value in positive_floats:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PostReleaseVerificationError(f'{label} must be numeric')
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise PostReleaseVerificationError(f'{label} must be finite and positive')
        for label, value in (
            ('min_samples', self.min_samples),
            ('min_target_points', self.min_target_points),
        ):
            _positive_int(value, label)
        motion = _finite_float(self.max_target_motion_m, 'max_target_motion_m')
        if motion < 0.0:
            raise PostReleaseVerificationError(
                'max_target_motion_m cannot be negative',
            )
        support = _finite_float(
            self.min_region_support_fraction,
            'min_region_support_fraction',
        )
        if not 0.0 < support <= 1.0:
            raise PostReleaseVerificationError(
                'min_region_support_fraction must be in (0, 1]',
            )
        registration = _finite_float(
            self.min_object_registration_inlier_fraction,
            'min_object_registration_inlier_fraction',
        )
        if not 0.0 < registration <= 1.0:
            raise PostReleaseVerificationError(
                'min_object_registration_inlier_fraction must be in (0, 1]',
            )
        if self.timeout_s <= self.min_stable_duration_s:
            raise PostReleaseVerificationError(
                'timeout_s must exceed the required stable duration',
            )


@dataclass(frozen=True)
class PostReleaseVerificationEvidence:
    """One fully parsed terminal verifier result."""

    verified: bool
    failure: str
    goal_id: str
    release_gripper_command_id: int
    request_id: str
    producer_epoch: str
    generation: int
    frame_id: str
    planning_observation_stamp_ns: int
    release_ack_stamp_ns: int
    observation_start_stamp_ns: int
    first_observation_stamp_ns: int
    last_observation_stamp_ns: int
    sample_count: int
    target_point_count: int
    stable_duration_s: float
    max_target_motion_m: float
    region_support_fraction: float
    target_gripper_clearance_m: float
    target_depth_correspondence_max_error_m: float
    object_position_error_m: float
    object_orientation_error_rad: float
    object_upright_error_rad: float
    object_registration_inlier_fraction: float
    object_registration_rms_m: float
    object_orientation_mode: str
    planned_object_pose: tuple[tuple[float, ...], ...]
    observed_object_center_m: tuple[float, float, float]
    payload: dict[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PostReleaseVerificationError(
                f'post-release payload repeats field {key!r}',
            )
        value[key] = item
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PostReleaseVerificationError(f'{label} must be a positive integer')
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PostReleaseVerificationError(
            f'{label} must be a nonnegative integer',
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostReleaseVerificationError(f'{label} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise PostReleaseVerificationError(f'{label} must be finite')
    return result


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PostReleaseVerificationError(f'{key} must be a non-empty string')
    return value


def validate_place_trajectory_perception_identity(
    contract: dict[str, Any],
    expected: PlacementObservationIdentity,
) -> None:
    """Require a planned trajectory to echo its exact perception owner."""
    if not isinstance(contract, dict):
        raise PostReleaseVerificationError(
            'placement trajectory contract must be an object',
        )
    if not isinstance(expected, PlacementObservationIdentity):
        raise PostReleaseVerificationError('placement expectation is invalid')
    observed = (
        _required_string(contract, 'goal_id'),
        _required_string(contract, 'request_id'),
        _required_string(contract, 'producer_epoch'),
        _positive_int(contract.get('generation'), 'generation'),
        _positive_int(
            contract.get('observation_stamp_ns'),
            'observation_stamp_ns',
        ),
        _required_string(contract, 'observation_frame_id'),
    )
    wanted = (
        expected.goal_id,
        expected.request_id,
        expected.producer_epoch,
        expected.generation,
        expected.planning_observation_stamp_ns,
        expected.frame_id,
    )
    if observed != wanted:
        raise PostReleaseVerificationError(
            'placement trajectory perception identity mismatch',
        )


_POST_RELEASE_V2_FIELDS = frozenset({
    'schema', 'state', 'result', 'failure', 'observation_source',
    'goal_id', 'place_goal_id', 'release_gripper_command_id', 'request_id',
    'producer_epoch', 'generation', 'frame_id', 'geometry_frame_id',
    'planning_observation_stamp_ns', 'release_ack_stamp_ns',
    'observation_start_stamp_ns', 'first_observation_stamp_ns',
    'last_observation_stamp_ns', 'first_status_stamp_ns',
    'last_status_stamp_ns', 'first_rgb_stamp_ns', 'first_depth_stamp_ns',
    'first_target_stamp_ns', 'last_rgb_stamp_ns', 'last_depth_stamp_ns',
    'last_target_stamp_ns', 'last_joint_stamp_ns',
    'last_execution_status_received_ns', 'sample_count', 'target_point_count',
    'stable_duration_s', 'max_target_motion_m', 'region_support_fraction',
    'target_gripper_clearance_m', 'target_depth_correspondence_max_error_m',
    'object_position_error_m', 'object_orientation_error_rad',
    'object_upright_error_rad', 'object_registration_inlier_fraction',
    'object_registration_rms_m', 'object_orientation_mode',
    'planned_object_pose', 'observed_object_center_m',
    'rejected_sample_count', 'rejected_sample_reasons',
})


def _proper_transform(value: object, label: str) -> tuple[tuple[float, ...], ...]:
    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise PostReleaseVerificationError(f'{label} must be numeric') from error
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise PostReleaseVerificationError(f'{label} must be a finite 4x4 matrix')
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise PostReleaseVerificationError(f'{label} homogeneous row is invalid')
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise PostReleaseVerificationError(f'{label} rotation is not proper SE(3)')
    return tuple(tuple(float(item) for item in row) for row in transform)


def _finite_three_vector(value: object, label: str) -> tuple[float, float, float]:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise PostReleaseVerificationError(f'{label} must be numeric') from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise PostReleaseVerificationError(f'{label} must be a finite three-vector')
    return tuple(float(item) for item in vector)


def _nonnegative_metric(payload: dict[str, Any], key: str) -> float:
    value = _finite_float(payload.get(key), key)
    if value < 0.0:
        raise PostReleaseVerificationError(f'{key} cannot be negative')
    return value


def parse_post_release_verification(
    raw: str,
    *,
    expected: PlacementObservationIdentity,
    expected_release_gripper_command_id: int,
    policy: PostReleaseVerificationPolicy,
) -> PostReleaseVerificationEvidence:
    """Parse exact v2 terminal evidence and require full task ownership."""
    if not isinstance(expected, PlacementObservationIdentity):
        raise PostReleaseVerificationError('placement expectation is invalid')
    release_id = _nonnegative_int(
        expected_release_gripper_command_id,
        'expected_release_gripper_command_id',
    )
    if not isinstance(policy, PostReleaseVerificationPolicy):
        raise PostReleaseVerificationError('post-release policy is invalid')
    try:
        payload = json.loads(str(raw), object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as error:
        raise PostReleaseVerificationError(
            f'post-release payload is not valid JSON: {error}',
        ) from error
    if not isinstance(payload, dict):
        raise PostReleaseVerificationError('post-release payload must be an object')
    if set(payload) != _POST_RELEASE_V2_FIELDS:
        raise PostReleaseVerificationError(
            'post-release v2 fields are incomplete or unknown',
        )
    if payload.get('schema') != POST_RELEASE_VERIFICATION_SCHEMA:
        raise PostReleaseVerificationError('post-release schema is unsupported')
    if payload.get('observation_source') != POST_RELEASE_OBSERVATION_SOURCE:
        raise PostReleaseVerificationError('post-release observation source is invalid')

    observed_goal = _required_string(payload, 'goal_id')
    observed_place_goal = _required_string(payload, 'place_goal_id')
    observed_request = _required_string(payload, 'request_id')
    observed_epoch = _required_string(payload, 'producer_epoch')
    observed_frame = _required_string(payload, 'frame_id')
    _required_string(payload, 'geometry_frame_id')
    observed_generation = _positive_int(payload.get('generation'), 'generation')
    observed_release = _nonnegative_int(
        payload.get('release_gripper_command_id'),
        'release_gripper_command_id',
    )
    planning_stamp = _positive_int(
        payload.get('planning_observation_stamp_ns'),
        'planning_observation_stamp_ns',
    )
    observed_identity = (
        observed_goal, observed_place_goal, observed_request, observed_epoch,
        observed_generation, observed_frame, planning_stamp, observed_release,
    )
    expected_identity = (
        expected.goal_id, expected.goal_id, expected.request_id,
        expected.producer_epoch, expected.generation, expected.frame_id,
        expected.planning_observation_stamp_ns, release_id,
    )
    if observed_identity != expected_identity:
        raise PostReleaseVerificationError(
            'post-release goal/release/perception identity mismatch',
        )

    state = _required_string(payload, 'state')
    result = _required_string(payload, 'result')
    failure = payload.get('failure')
    if not isinstance(failure, str):
        raise PostReleaseVerificationError('post-release failure must be a string')
    release_ack = _positive_int(payload.get('release_ack_stamp_ns'), 'release_ack_stamp_ns')
    observation_start = _positive_int(
        payload.get('observation_start_stamp_ns'),
        'observation_start_stamp_ns',
    )
    if release_ack <= planning_stamp:
        raise PostReleaseVerificationError(
            'release acknowledgement must follow the planning observation',
        )
    if observation_start < release_ack:
        raise PostReleaseVerificationError(
            'post-release observation window predates release acknowledgement',
        )

    timestamp_keys = (
        'first_observation_stamp_ns', 'last_observation_stamp_ns',
        'first_status_stamp_ns', 'last_status_stamp_ns', 'first_rgb_stamp_ns',
        'first_depth_stamp_ns', 'first_target_stamp_ns', 'last_rgb_stamp_ns',
        'last_depth_stamp_ns', 'last_target_stamp_ns', 'last_joint_stamp_ns',
        'last_execution_status_received_ns',
    )
    stamps = {
        key: _nonnegative_int(payload.get(key), key)
        for key in timestamp_keys
    }
    sample_count = _nonnegative_int(payload.get('sample_count'), 'sample_count')
    target_points = _nonnegative_int(
        payload.get('target_point_count'),
        'target_point_count',
    )
    rejected_count = _nonnegative_int(
        payload.get('rejected_sample_count'),
        'rejected_sample_count',
    )
    rejected_reasons = payload.get('rejected_sample_reasons')
    if (
        not isinstance(rejected_reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in rejected_reasons)
        or len(rejected_reasons) > rejected_count
    ):
        raise PostReleaseVerificationError('rejected sample diagnostics are malformed')

    stable_duration = _nonnegative_metric(payload, 'stable_duration_s')
    max_motion = _nonnegative_metric(payload, 'max_target_motion_m')
    support = _nonnegative_metric(payload, 'region_support_fraction')
    clearance = _nonnegative_metric(payload, 'target_gripper_clearance_m')
    depth_error = _nonnegative_metric(
        payload,
        'target_depth_correspondence_max_error_m',
    )
    position_error = _nonnegative_metric(payload, 'object_position_error_m')
    orientation_error = _nonnegative_metric(
        payload,
        'object_orientation_error_rad',
    )
    upright_error = _nonnegative_metric(payload, 'object_upright_error_rad')
    registration_inlier = _nonnegative_metric(
        payload,
        'object_registration_inlier_fraction',
    )
    registration_rms = _nonnegative_metric(payload, 'object_registration_rms_m')
    if support > 1.0 or registration_inlier > 1.0:
        raise PostReleaseVerificationError('post-release fractions must be in [0, 1]')
    orientation_mode = _required_string(payload, 'object_orientation_mode')
    if orientation_mode not in {'full', 'axial'}:
        raise PostReleaseVerificationError('object_orientation_mode is unsupported')
    planned_pose = _proper_transform(payload.get('planned_object_pose'), 'planned_object_pose')
    observed_center = _finite_three_vector(
        payload.get('observed_object_center_m'),
        'observed_object_center_m',
    )

    evidence_arguments = {
        'goal_id': observed_goal,
        'release_gripper_command_id': observed_release,
        'request_id': observed_request,
        'producer_epoch': observed_epoch,
        'generation': observed_generation,
        'frame_id': observed_frame,
        'planning_observation_stamp_ns': planning_stamp,
        'release_ack_stamp_ns': release_ack,
        'observation_start_stamp_ns': observation_start,
        'first_observation_stamp_ns': stamps['first_observation_stamp_ns'],
        'last_observation_stamp_ns': stamps['last_observation_stamp_ns'],
        'sample_count': sample_count,
        'target_point_count': target_points,
        'stable_duration_s': stable_duration,
        'max_target_motion_m': max_motion,
        'region_support_fraction': support,
        'target_gripper_clearance_m': clearance,
        'target_depth_correspondence_max_error_m': depth_error,
        'object_position_error_m': position_error,
        'object_orientation_error_rad': orientation_error,
        'object_upright_error_rad': upright_error,
        'object_registration_inlier_fraction': registration_inlier,
        'object_registration_rms_m': registration_rms,
        'object_orientation_mode': orientation_mode,
        'planned_object_pose': planned_pose,
        'observed_object_center_m': observed_center,
        'payload': payload,
    }
    if state == 'failed':
        if result != POST_RELEASE_FAILURE_RESULT or not failure.strip():
            raise PostReleaseVerificationError('post-release failure result is malformed')
        return PostReleaseVerificationEvidence(
            verified=False,
            failure=failure,
            **evidence_arguments,
        )
    if state != 'verified' or result != POST_RELEASE_VERIFICATION_RESULT or failure:
        raise PostReleaseVerificationError('post-release terminal state/result is invalid')

    first_stamp = _positive_int(
        stamps['first_observation_stamp_ns'],
        'first_observation_stamp_ns',
    )
    last_stamp = _positive_int(
        stamps['last_observation_stamp_ns'],
        'last_observation_stamp_ns',
    )
    first_rgb = _positive_int(stamps['first_rgb_stamp_ns'], 'first_rgb_stamp_ns')
    first_depth = _positive_int(
        stamps['first_depth_stamp_ns'],
        'first_depth_stamp_ns',
    )
    first_target = _positive_int(
        stamps['first_target_stamp_ns'],
        'first_target_stamp_ns',
    )
    last_rgb = _positive_int(stamps['last_rgb_stamp_ns'], 'last_rgb_stamp_ns')
    last_depth = _positive_int(stamps['last_depth_stamp_ns'], 'last_depth_stamp_ns')
    last_target = _positive_int(stamps['last_target_stamp_ns'], 'last_target_stamp_ns')
    last_joint = _positive_int(stamps['last_joint_stamp_ns'], 'last_joint_stamp_ns')
    last_execution = _positive_int(
        stamps['last_execution_status_received_ns'],
        'last_execution_status_received_ns',
    )
    if (
        first_stamp <= observation_start
        or last_stamp <= first_stamp
        or stamps['first_status_stamp_ns'] != first_stamp
        or stamps['last_status_stamp_ns'] != last_stamp
    ):
        raise PostReleaseVerificationError(
            'post-release observations must strictly follow retreat',
        )
    if first_target != first_stamp or last_target != last_stamp:
        raise PostReleaseVerificationError(
            'target and observation stamps must match exactly',
        )
    rgbd_skew_ns = int(policy.max_rgbd_target_skew_s * 1e9)
    if (
        min(first_rgb, first_depth, first_target) <= observation_start
        or max(first_rgb, first_depth, first_target)
        - min(first_rgb, first_depth, first_target) > rgbd_skew_ns
    ):
        raise PostReleaseVerificationError(
            'first post-release RGB-D/target sample is not retreat-bounded',
        )
    if (
        max(last_rgb, last_depth, last_target)
        - min(last_rgb, last_depth, last_target) > rgbd_skew_ns
    ):
        raise PostReleaseVerificationError(
            'post-release RGB-D/target stamps exceed the synchronization bound',
        )
    if abs(last_joint - last_target) > int(policy.max_joint_target_skew_s * 1e9):
        raise PostReleaseVerificationError(
            'post-release arm state exceeds the synchronization bound',
        )
    if last_execution < release_ack:
        raise PostReleaseVerificationError(
            'post-release gripper feedback predates release',
        )

    stamp_duration = (last_stamp - first_stamp) * 1e-9
    metric_failure = (
        sample_count < policy.min_samples
        or target_points < policy.min_target_points
        or stable_duration < policy.min_stable_duration_s
        or stamp_duration < policy.min_stable_duration_s
        or abs(stable_duration - stamp_duration) > 1e-6
        or max_motion > policy.max_target_motion_m
        or support < policy.min_region_support_fraction
        or clearance < policy.min_gripper_clearance_m
        or depth_error > policy.max_target_depth_correspondence_m
        or position_error > policy.max_object_position_error_m
        or orientation_error > policy.max_object_orientation_error_rad
        or (
            expected.require_upright
            and upright_error > policy.max_object_upright_error_rad
        )
        or registration_inlier < policy.min_object_registration_inlier_fraction
        or registration_rms > policy.max_object_registration_rms_m
    )
    if metric_failure:
        raise PostReleaseVerificationError(
            'post-release geometric evidence does not meet task policy',
        )
    return PostReleaseVerificationEvidence(
        verified=True,
        failure='',
        **evidence_arguments,
    )
