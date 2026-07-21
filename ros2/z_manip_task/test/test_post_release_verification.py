"""Task-side post-release evidence correlation and quality tests."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from z_manip_task.post_release_verification import (
    parse_post_release_verification,
    PlacementObservationIdentity,
    PostReleaseVerificationError,
    PostReleaseVerificationPolicy,
    validate_place_trajectory_perception_identity,
)


def _identity(*, require_upright: bool = True) -> PlacementObservationIdentity:
    return PlacementObservationIdentity(
        goal_id='place-7-1000000000',
        request_id='request-7',
        producer_epoch='epoch-a',
        generation=7,
        frame_id='camera_color_optical_frame',
        planning_observation_stamp_ns=1_000_000_000,
        require_upright=require_upright,
    )


def _payload() -> dict[str, object]:
    return {
        'schema': 'z_manip.post_release_verification.v2',
        'state': 'verified',
        'result': 'post_release_target_stable_in_region',
        'failure': '',
        'observation_source': 'synchronized_rgbd_pointcloud',
        'goal_id': 'place-7-1000000000',
        'place_goal_id': 'place-7-1000000000',
        'release_gripper_command_id': 12,
        'request_id': 'request-7',
        'producer_epoch': 'epoch-a',
        'generation': 7,
        'frame_id': 'camera_color_optical_frame',
        'geometry_frame_id': 'piper_link',
        'planning_observation_stamp_ns': 1_000_000_000,
        'release_ack_stamp_ns': 2_000_000_000,
        'observation_start_stamp_ns': 2_050_000_000,
        'first_observation_stamp_ns': 2_100_000_000,
        'last_observation_stamp_ns': 2_600_000_000,
        'first_status_stamp_ns': 2_100_000_000,
        'last_status_stamp_ns': 2_600_000_000,
        'first_rgb_stamp_ns': 2_100_000_000,
        'first_depth_stamp_ns': 2_100_000_000,
        'first_target_stamp_ns': 2_100_000_000,
        'last_rgb_stamp_ns': 2_600_000_000,
        'last_depth_stamp_ns': 2_600_000_000,
        'last_target_stamp_ns': 2_600_000_000,
        'last_joint_stamp_ns': 2_590_000_000,
        'last_execution_status_received_ns': 2_550_000_000,
        'sample_count': 6,
        'target_point_count': 32,
        'stable_duration_s': 0.5,
        'max_target_motion_m': 0.01,
        'region_support_fraction': 0.9,
        'target_gripper_clearance_m': 0.05,
        'target_depth_correspondence_max_error_m': 0.008,
        'object_position_error_m': 0.02,
        'object_orientation_error_rad': 0.15,
        'object_upright_error_rad': 0.10,
        'object_registration_inlier_fraction': 0.80,
        'object_registration_rms_m': 0.012,
        'object_orientation_mode': 'axial',
        'planned_object_pose': [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.7],
            [0.0, 0.0, 0.0, 1.0],
        ],
        'observed_object_center_m': [0.51, 0.0, 0.71],
        'rejected_sample_count': 0,
        'rejected_sample_reasons': [],
    }


def _parse(payload: dict[str, object], *, require_upright: bool = True):
    return parse_post_release_verification(
        json.dumps(payload),
        expected=_identity(require_upright=require_upright),
        expected_release_gripper_command_id=12,
        policy=PostReleaseVerificationPolicy(),
    )


def _trajectory_contract() -> dict[str, object]:
    identity = _identity()
    return {
        'goal_id': identity.goal_id,
        'request_id': identity.request_id,
        'producer_epoch': identity.producer_epoch,
        'generation': identity.generation,
        'observation_stamp_ns': identity.planning_observation_stamp_ns,
        'observation_frame_id': identity.frame_id,
    }


def test_exact_verified_geometry_evidence_is_accepted() -> None:
    evidence = _parse(_payload())

    assert evidence.verified
    assert evidence.goal_id == _identity().goal_id
    assert evidence.release_gripper_command_id == 12
    assert evidence.sample_count == 6
    assert evidence.object_orientation_mode == 'axial'


def test_old_or_incomplete_schema_cannot_claim_verified_placement() -> None:
    payload = _payload()
    payload['schema'] = 'z_manip.post_release_verification.v1'
    with pytest.raises(PostReleaseVerificationError, match='unsupported'):
        _parse(payload)

    payload = _payload()
    payload.pop('planned_object_pose')
    with pytest.raises(PostReleaseVerificationError, match='fields'):
        _parse(payload)


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('object_position_error_m', float('nan')),
        ('object_registration_rms_m', float('inf')),
        ('observed_object_center_m', [0.0, float('nan'), 0.0]),
        ('planned_object_pose', [[1.0, 0.0], [0.0, 1.0]]),
    ),
)
def test_nonfinite_or_non_se3_pose_evidence_is_rejected(
    field: str,
    value: object,
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(PostReleaseVerificationError):
        _parse(payload)


def test_non_upright_contract_does_not_invent_an_upright_requirement() -> None:
    payload = _payload()
    payload['object_upright_error_rad'] = 1.2

    evidence = _parse(payload, require_upright=False)

    assert evidence.verified


def test_place_trajectory_must_echo_exact_planning_perception_identity() -> None:
    validate_place_trajectory_perception_identity(
        _trajectory_contract(),
        _identity(),
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('goal_id', 'other-goal'),
        ('request_id', 'other-request'),
        ('producer_epoch', 'other-epoch'),
        ('generation', 8),
        ('observation_stamp_ns', 1_000_000_001),
        ('observation_frame_id', 'other-frame'),
    ),
)
def test_place_trajectory_rejects_mixed_perception_identity(
    field: str,
    value: object,
) -> None:
    contract = _trajectory_contract()
    contract[field] = value

    with pytest.raises(PostReleaseVerificationError, match='identity mismatch'):
        validate_place_trajectory_perception_identity(contract, _identity())


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('goal_id', 'another-goal'),
        ('place_goal_id', 'another-goal'),
        ('release_gripper_command_id', 13),
        ('request_id', 'another-request'),
        ('producer_epoch', 'epoch-b'),
        ('generation', 8),
        ('frame_id', 'another-camera'),
        ('planning_observation_stamp_ns', 1_000_000_001),
    ),
)
def test_every_task_ownership_field_is_exact(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(PostReleaseVerificationError, match='identity mismatch'):
        _parse(payload)


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('sample_count', 2),
        ('target_point_count', 23),
        ('stable_duration_s', 0.49),
        ('max_target_motion_m', 0.026),
        ('region_support_fraction', 0.79),
        ('target_gripper_clearance_m', 0.039),
        ('target_depth_correspondence_max_error_m', 0.013),
        ('object_position_error_m', 0.041),
        ('object_orientation_error_rad', 0.351),
        ('object_upright_error_rad', 0.261),
        ('object_registration_inlier_fraction', 0.549),
        ('object_registration_rms_m', 0.026),
    ),
)
def test_geometry_below_task_policy_is_rejected(field: str, value: object) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(PostReleaseVerificationError, match='geometric evidence'):
        _parse(payload)


def test_source_stamp_and_state_skew_are_bounded() -> None:
    payload = _payload()
    payload['last_rgb_stamp_ns'] = 2_574_999_999
    with pytest.raises(PostReleaseVerificationError, match='synchronization bound'):
        _parse(payload)

    payload = _payload()
    payload['last_joint_stamp_ns'] = 2_479_999_999
    with pytest.raises(PostReleaseVerificationError, match='arm state'):
        _parse(payload)


def test_stable_window_starts_only_after_release_and_retreat() -> None:
    payload = _payload()
    payload['observation_start_stamp_ns'] = 1_999_999_999
    with pytest.raises(PostReleaseVerificationError, match='predates release'):
        _parse(payload)

    payload = _payload()
    payload['first_observation_stamp_ns'] = 2_050_000_000
    payload['first_target_stamp_ns'] = 2_050_000_000
    with pytest.raises(PostReleaseVerificationError, match='follow retreat'):
        _parse(payload)

    payload = _payload()
    payload['first_rgb_stamp_ns'] = 2_049_999_999
    with pytest.raises(PostReleaseVerificationError, match='retreat-bounded'):
        _parse(payload)


def test_failed_correlated_verifier_result_is_parsed_for_fail_closed_runtime() -> None:
    payload = _payload()
    payload.update({
        'state': 'failed',
        'result': 'post_release_verification_failed',
        'failure': 'target observation timed out',
        'first_observation_stamp_ns': 0,
        'last_observation_stamp_ns': 0,
        'sample_count': 0,
        'target_point_count': 0,
        'stable_duration_s': 0.0,
        'max_target_motion_m': 0.0,
        'region_support_fraction': 0.0,
        'target_gripper_clearance_m': 0.0,
    })

    evidence = _parse(payload)

    assert not evidence.verified
    assert evidence.failure == 'target observation timed out'


def test_duplicate_json_fields_are_rejected() -> None:
    raw = json.dumps(_payload())[:-1] + ',"goal_id":"shadow"}'

    with pytest.raises(PostReleaseVerificationError, match='repeats field'):
        parse_post_release_verification(
            raw,
            expected=_identity(),
            expected_release_gripper_command_id=12,
            policy=PostReleaseVerificationPolicy(),
        )


@pytest.mark.parametrize('value', (True, 7.0, -1, '12'))
def test_release_command_id_requires_canonical_nonnegative_integer(
    value: object,
) -> None:
    payload = _payload()
    payload['release_gripper_command_id'] = value

    with pytest.raises(PostReleaseVerificationError):
        _parse(payload)


def test_policy_requires_timeout_longer_than_observation_dwell() -> None:
    with pytest.raises(PostReleaseVerificationError, match='must exceed'):
        PostReleaseVerificationPolicy(
            timeout_s=0.5,
            min_stable_duration_s=0.5,
        )


def test_runtime_caches_verified_result_until_measured_retreat_succeeds() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.core import RuntimePhase, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        _post_release_verification_cb = (
            MobileManipulationRuntime._post_release_verification_cb
        )
        _complete_post_release_verification = (
            MobileManipulationRuntime._complete_post_release_verification
        )

        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.phase = RuntimePhase.PLACE_RETREAT
            self._place_observation_identity = _identity()
            self._post_release_release_command_id = 12
            self._post_release_verification_policy = (
                PostReleaseVerificationPolicy()
            )
            self._post_release_pending_evidence = None
            self._post_release_verified_evidence = None
            self._post_release_verification_started_at_s = None
            self._post_release_verification_started_wall_s = None
            self._post_release_verification_last_tick_s = None
            self._task = SimpleNamespace(
                stage=SimpleNamespace(value='already_complete'),
            )
            self.status_count = 0

        def _apply_safety(self, _action) -> None:
            return

        def _publish_status(self, *, force: bool) -> None:
            assert force
            self.status_count += 1

    harness = Harness()
    MobileManipulationRuntime._post_release_verification_cb(
        harness,
        String(data=json.dumps(_payload())),
    )

    assert harness._core.phase is RuntimePhase.PLACE_RETREAT
    assert harness._post_release_pending_evidence is not None
    harness._core.phase = RuntimePhase.POST_RELEASE_VERIFICATION
    MobileManipulationRuntime._begin_post_release_verification(harness, 3.0)
    assert harness._core.phase is RuntimePhase.COMPLETE
    assert harness._post_release_verified_evidence is not None


def test_place_identity_preserves_epoch_sized_integer_nanoseconds() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    exact_stamp_ns = 1_783_915_238_470_000_000
    rounded_from_float = int(round(float(exact_stamp_ns * 1e-9) * 1e9))
    assert rounded_from_float != exact_stamp_ns

    harness = SimpleNamespace(
        _bound_perception_request_id='request-7',
        _bound_perception_producer_epoch='epoch-a',
        _bound_perception_generation=7,
        _target_frame_id='camera_color_optical_frame',
        _valid_perception_request_id='request-7',
        _valid_perception_producer_epoch='epoch-a',
        _valid_perception_generation=7,
        _valid_observation_stamp_ns=exact_stamp_ns,
        _valid_observation_frame_id='camera_color_optical_frame',
        _affordance_request_id='request-7',
        _affordance_producer_epoch='epoch-a',
        _affordance_generation=7,
        _required_perception_request_id='request-7',
        _required_perception_generation=7,
        _required_affordance_generation=7,
        _carried_object_geometry=SimpleNamespace(require_upright=True),
    )

    identity = MobileManipulationRuntime._capture_place_observation_identity(
        harness,
        SimpleNamespace(stamp_s=exact_stamp_ns * 1e-9),
        'place-7-1783915238470000000',
    )

    assert identity.planning_observation_stamp_ns == exact_stamp_ns


def test_runtime_fails_closed_on_correlated_verifier_failure() -> None:
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_task.core import RuntimePhase, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    payload = _payload()
    payload.update({
        'state': 'failed',
        'result': 'post_release_verification_failed',
        'failure': 'target occluded',
        'first_observation_stamp_ns': 0,
        'last_observation_stamp_ns': 0,
        'sample_count': 0,
        'target_point_count': 0,
        'stable_duration_s': 0.0,
        'max_target_motion_m': 0.0,
        'region_support_fraction': 0.0,
        'target_gripper_clearance_m': 0.0,
    })

    class Harness:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.phase = RuntimePhase.POST_RELEASE_VERIFICATION
            self._place_observation_identity = _identity()
            self._post_release_release_command_id = 12
            self._post_release_verification_policy = (
                PostReleaseVerificationPolicy()
            )

        def _apply_safety(self, _action) -> None:
            return

        def _publish_status(self, *, force: bool) -> None:
            assert force

    harness = Harness()
    MobileManipulationRuntime._post_release_verification_cb(
        harness,
        String(data=json.dumps(payload)),
    )

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'target occluded' in harness._core.failure_reason


def test_runtime_uses_independent_ros_and_wall_timeouts() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.core import RuntimePhase, RuntimeSafetyCore
    from z_manip_task.node import MobileManipulationRuntime

    class Harness:
        def __init__(self) -> None:
            self._core = RuntimeSafetyCore()
            self._core.phase = RuntimePhase.POST_RELEASE_VERIFICATION
            self._post_release_verification_policy = (
                PostReleaseVerificationPolicy(
                    timeout_s=2.0,
                    wall_timeout_s=12.0,
                )
            )
            self._post_release_verification_started_at_s = 10.0
            self._post_release_verification_started_wall_s = (
                time.monotonic() - 12.1
            )
            self._post_release_verification_last_tick_s = 10.0

        def _apply_safety(self, _action) -> None:
            return

    harness = Harness()
    MobileManipulationRuntime._post_release_verification_tick(harness, 10.1)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'timed out' in harness._core.failure_reason
