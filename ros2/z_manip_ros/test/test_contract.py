from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from z_manip_ros.contract import (
    ContractPhase,
    ExactObservationBundler,
    FailureCode,
    normalized_xyxy_to_pixel_box,
    expand_pixel_box,
    parse_grounding_request,
    parse_tracker_failure_report,
    parse_tracker_frame_manifest,
    TrackingContract,
)


def test_versioned_grounding_request_preserves_exact_task_identity():
    request = parse_grounding_request(json.dumps({
        'schema': 'z_manip.grounding_request.v1',
        'request_id': 'task-request-42',
        'instruction': 'pick the mustard bottle',
    }))

    assert request.request_id == 'task-request-42'
    assert request.instruction == 'pick the mustard bottle'
    assert request.scope == 'grasp_only'


def test_versioned_grounding_request_v2_requires_an_explicit_scope():
    request = parse_grounding_request(json.dumps({
        'schema': 'z_manip.grounding_request.v2',
        'request_id': 'task-request-43',
        'instruction': 'pick and place the mustard bottle',
        'scope': 'grasp_for_place',
    }))

    assert request.scope == 'grasp_for_place'


def test_plain_text_grounding_request_requires_explicit_legacy_identity():
    with pytest.raises(ValueError, match='versioned envelope'):
        parse_grounding_request('pick the mug')

    request = parse_grounding_request(
        'pick the mug',
        legacy_request_id='legacy-local-1',
    )
    assert request.request_id == 'legacy-local-1'
    assert request.instruction == 'pick the mug'


@pytest.mark.parametrize(
    'document',
    (
        {'schema': 'z_manip.grounding_request.v1', 'instruction': 'pick'},
        {
            'schema': 'z_manip.grounding_request.v1',
            'request_id': 'contains whitespace',
            'instruction': 'pick',
        },
        {
            'schema': 'z_manip.grounding_request.v0',
            'request_id': 'request-1',
            'instruction': 'pick',
        },
        {
            'schema': 'z_manip.grounding_request.v2',
            'request_id': 'request-2',
            'instruction': 'pick',
        },
        {
            'schema': 'z_manip.grounding_request.v2',
            'request_id': 'request-3',
            'instruction': 'pick',
            'scope': 'unknown',
        },
    ),
)
def test_grounding_request_rejects_ambiguous_identity(document):
    with pytest.raises(ValueError):
        parse_grounding_request(json.dumps(document))


def _waiting_tracker() -> TrackingContract:
    contract = TrackingContract(data_timeout_s=0.4, min_cloud_points=20)
    generation = contract.request(
        'pick the mug',
        now_s=0.0,
        request_id='test-task-request',
    )
    contract.grounding_started(generation, now_s=0.1)
    contract.grounding_succeeded(
        generation,
        target_label='mug body',
        confidence=0.9,
        now_s=0.2,
    )
    return contract


def _acquire(contract: TrackingContract, *, now_s: float = 0.3) -> None:
    contract.detections(['7'], now_s=now_s)
    contract.selected_target('7', now_s=now_s)
    contract.selected_cloud(80, now_s=now_s)
    contract.tracker_status(True, now_s=now_s)


def test_contract_requires_all_persistent_tracker_outputs_before_valid():
    contract = _waiting_tracker()
    contract.tracker_status(False, now_s=0.25)  # normal before first acquisition
    contract.tracker_status(True, now_s=0.3)
    contract.detections(['7'], now_s=0.3)
    contract.selected_target('7', now_s=0.3)
    assert contract.snapshot.phase is ContractPhase.WAITING_TRACKER

    contract.selected_cloud(80, now_s=0.3)
    assert contract.snapshot.valid
    assert contract.snapshot.track_id == '7'


@pytest.mark.parametrize(
    ('update', 'reason'),
    [
        (lambda state: state.tracker_status(False, now_s=0.31), FailureCode.TRACKER_REPORTED_LOSS),
        (lambda state: state.detections([], now_s=0.31), FailureCode.EMPTY_DETECTIONS),
        (lambda state: state.selected_target('8', now_s=0.31), FailureCode.TRACK_ID_CHANGED),
        (lambda state: state.selected_cloud(3, now_s=0.31), FailureCode.CLOUD_TOO_SMALL),
    ],
)
def test_acquired_track_fails_closed_immediately(update, reason):
    contract = _waiting_tracker()
    _acquire(contract)
    update(contract)
    assert contract.snapshot.phase is ContractPhase.FAILED
    assert contract.snapshot.failure is reason
    assert not contract.snapshot.valid


def test_seed_correlated_tracker_failure_is_terminal_before_first_true():
    contract = _waiting_tracker()

    contract.tracker_failed(now_s=0.25)

    assert contract.snapshot.phase is ContractPhase.FAILED
    assert contract.snapshot.failure is FailureCode.TRACKER_REPORTED_LOSS


def test_tracker_failure_report_preserves_bounded_seed_and_reason():
    report = parse_tracker_failure_report(json.dumps({
        'schema': 'z_manip.tracker_failure.v1',
        'seed_id': 'z-manip-seed:2',
        'seed_stamp_ns': 1234,
        'reason_code': 'mask_continuity',
        'reason': 'EdgeTAM mask continuity broke (IoU 0.010)',
        'replay_candidates': 25,
        'replay_selected': 8,
        'replay_span_ns': 9_500_000_000,
        'acquisition_live_updates': 1,
    }))

    assert report.seed_id == 'z-manip-seed:2'
    assert report.seed_stamp_ns == 1234
    assert report.reason_code == 'mask_continuity'
    assert report.replay_candidates == 25
    assert report.replay_selected == 8
    assert report.replay_span_ns == 9_500_000_000
    assert report.acquisition_live_updates == 1


def test_tracker_failure_report_rejects_seed_identity_alias_by_truncation():
    with pytest.raises(ValueError, match='identity exceeds'):
        parse_tracker_failure_report(json.dumps(_failure_document(
            seed_id='z-manip-seed:' + ('x' * 300),
        )))


def _failure_document(**overrides):
    document = {
        'schema': 'z_manip.tracker_failure.v1',
        'seed_id': 'z-manip-seed:2',
        'seed_stamp_ns': 1234,
        'reason_code': 'mask_continuity',
        'reason': 'lost',
        'replay_candidates': 25,
        'replay_selected': 8,
        'replay_span_ns': 9_500_000_000,
        'acquisition_live_updates': 1,
    }
    document.update(overrides)
    return document


@pytest.mark.parametrize('stamp', [None, True, -1, '1234'])
def test_tracker_failure_report_rejects_ambiguous_seed_timestamp(stamp):
    with pytest.raises(ValueError, match='timestamp'):
        parse_tracker_failure_report(json.dumps(_failure_document(
            seed_stamp_ns=stamp,
        )))


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('seed_stamp_ns', 1 << 63),
        ('replay_candidates', 1_000_001),
        ('replay_selected', 1_000_001),
        ('replay_span_ns', 86_400_000_000_001),
        ('acquisition_live_updates', 1_000_001),
        ('replay_candidates', 10 ** 400),
    ],
)
def test_tracker_failure_report_rejects_unbounded_integers(field, value):
    with pytest.raises(ValueError):
        parse_tracker_failure_report(json.dumps(_failure_document(**{field: value})))


def test_exact_bundler_never_mixes_stamps_or_frames():
    bundler = ExactObservationBundler(max_pending_bundles=8)
    bundler.reset(generation=3, seed_stamp_ns=100)

    assert bundler.add(
        'detections', generation=3, stamp_ns=200, frame_id='camera', payload='d200',
    ) is None
    assert bundler.add(
        'target', generation=3, stamp_ns=201, frame_id='camera', payload='t201',
    ) is None
    assert bundler.add(
        'cloud', generation=3, stamp_ns=200, frame_id='other', payload='c-other',
    ) is None
    assert bundler.add(
        'target', generation=3, stamp_ns=200, frame_id='camera', payload='t200',
    ) is None
    assert bundler.add(
        'cloud', generation=3, stamp_ns=200, frame_id='camera', payload='c200',
    ) is None
    bundle = bundler.add(
        'manifest', generation=3, stamp_ns=200, frame_id='camera', payload='m200',
    )

    assert bundle is not None
    assert bundle.stamp_ns == 200
    assert bundle.frame_id == 'camera'
    assert (bundle.detections, bundle.target, bundle.cloud, bundle.manifest) == (
        'd200', 't200', 'c200', 'm200',
    )


def test_exact_bundler_reset_rejects_old_epoch_and_preseed_frames():
    bundler = ExactObservationBundler(max_pending_bundles=2)
    bundler.reset(generation=1, seed_stamp_ns=100)
    assert bundler.add(
        'detections', generation=1, stamp_ns=200, frame_id='camera', payload='old',
    ) is None

    bundler.reset(generation=2, seed_stamp_ns=300)

    assert bundler.add(
        'target', generation=1, stamp_ns=200, frame_id='camera', payload='old',
    ) is None
    assert bundler.add(
        'cloud', generation=2, stamp_ns=300, frame_id='camera', payload='seed',
    ) is None


def test_atomic_observation_bundle_requires_matching_identity():
    contract = _waiting_tracker()
    contract.tracker_status(True, now_s=0.25)

    contract.observation_bundle(
        ['track-a'],
        selected_track_id='track-b',
        point_count=80,
        now_s=0.3,
    )

    assert contract.snapshot.phase is ContractPhase.FAILED
    assert contract.snapshot.failure is FailureCode.TARGET_MISSING


def test_atomic_observation_bundle_promotes_one_exact_frame():
    contract = _waiting_tracker()
    contract.tracker_status(True, now_s=0.25)

    contract.observation_bundle(
        ['track-a'],
        selected_track_id='track-a',
        point_count=80,
        now_s=0.3,
    )

    assert contract.snapshot.valid
    assert contract.snapshot.track_id == 'track-a'


def test_tracker_frame_manifest_preserves_exact_seed_and_track_epoch():
    manifest = parse_tracker_frame_manifest(json.dumps({
        'schema': 'z_manip.tracker_frame.v1',
        'seed_id': 'z-manip-seed:3',
        'seed_stamp_ns': 100,
        'adapter_generation': 7,
        'result_stamp_ns': 200,
        'frame_id': 'camera_color_optical_frame',
        'session_id': 'ros-session',
        'track_id': 'track-a',
    }))

    assert manifest.seed_id == 'z-manip-seed:3'
    assert manifest.seed_stamp_ns == 100
    assert manifest.adapter_generation == 7
    assert manifest.result_stamp_ns == 200
    assert manifest.track_id == 'track-a'


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('seed_stamp_ns', 1 << 63),
        ('adapter_generation', 1 << 63),
        ('result_stamp_ns', 1 << 63),
        ('result_stamp_ns', 100),
    ],
)
def test_tracker_frame_manifest_rejects_unbounded_or_nonforward_fields(field, value):
    document = {
        'schema': 'z_manip.tracker_frame.v1',
        'seed_id': 'z-manip-seed:3',
        'seed_stamp_ns': 100,
        'adapter_generation': 7,
        'result_stamp_ns': 200,
        'frame_id': 'camera_color_optical_frame',
        'session_id': 'ros-session',
        'track_id': 'track-a',
    }
    document[field] = value

    with pytest.raises(ValueError):
        parse_tracker_frame_manifest(json.dumps(document))


def test_contract_fails_if_any_persistent_stream_goes_stale():
    contract = _waiting_tracker()
    _acquire(contract)
    contract.tick(now_s=0.71)
    assert contract.snapshot.failure is FailureCode.TRACKER_DATA_STALE


def test_contract_fails_closed_on_ros_time_rollback():
    contract = _waiting_tracker()
    _acquire(contract, now_s=10.0)

    contract.tick(now_s=9.0)

    assert contract.snapshot.phase is ContractPhase.FAILED
    assert contract.snapshot.failure is FailureCode.CLOCK_ROLLBACK


@pytest.mark.parametrize(
    'report_loss',
    [
        lambda contract: contract.tracker_status(False, now_s=9.0),
        lambda contract: contract.tracker_failed(now_s=9.0),
    ],
)
def test_tracker_loss_does_not_overwrite_clock_rollback(report_loss):
    contract = _waiting_tracker()
    _acquire(contract, now_s=10.0)

    report_loss(contract)

    assert contract.snapshot.phase is ContractPhase.FAILED
    assert contract.snapshot.failure is FailureCode.CLOCK_ROLLBACK


def test_new_request_explicitly_rebases_clock_after_rollback():
    contract = _waiting_tracker()
    _acquire(contract, now_s=10.0)
    contract.tick(now_s=9.0)
    previous_generation = contract.generation

    generation = contract.request('pick another mug', now_s=1.0)

    assert generation == previous_generation + 1
    assert contract.snapshot.phase is ContractPhase.WAITING_FRAME
    assert contract.snapshot.failure is FailureCode.NONE


def test_generation_rejects_a_late_vlm_result_from_an_old_request():
    contract = TrackingContract()
    old = contract.request('first object', now_s=0.0)
    contract.grounding_started(old, now_s=0.1)
    current = contract.request('second object', now_s=0.2)

    assert current == old + 1
    assert not contract.grounding_succeeded(
        old,
        target_label='wrong old object',
        confidence=0.99,
        now_s=0.3,
    )
    assert contract.snapshot.phase is ContractPhase.WAITING_FRAME
    assert contract.snapshot.instruction == 'second object'


def test_reset_advances_epoch_and_rejects_inflight_grounding_result():
    contract = TrackingContract()
    generation = contract.request(
        'object being canceled',
        now_s=0.0,
        request_id='task-request-before-reset',
    )
    contract.grounding_started(generation, now_s=0.1)
    assert contract.snapshot.request_id == 'task-request-before-reset'

    reset_generation = contract.reset()

    assert reset_generation == generation + 1
    assert contract.snapshot.phase is ContractPhase.IDLE
    assert contract.snapshot.request_id == ''
    assert not contract.grounding_succeeded(
        generation,
        target_label='stale object',
        confidence=0.99,
        now_s=0.2,
    )
    assert contract.snapshot.phase is ContractPhase.IDLE

    next_generation = contract.request('new object', now_s=0.3)
    assert next_generation == reset_generation + 1
    assert contract.grounding_started(next_generation, now_s=0.4)
    assert contract.grounding_succeeded(
        next_generation,
        target_label='new object',
        confidence=0.9,
        now_s=0.5,
    )
    assert contract.snapshot.phase is ContractPhase.WAITING_TRACKER


def test_bridge_reset_drains_but_never_reads_stale_running_future():
    pytest.importorskip('rclpy')
    from z_manip_ros.vlm_edgetam_bridge import VlmEdgeTamBridge

    class StaleFuture:
        def cancel(self) -> bool:
            return False

        def done(self) -> bool:
            return True

        def result(self):
            raise AssertionError('stale VLM result was read after reset')

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    class Harness:
        _cancel_pending_grounding = VlmEdgeTamBridge._cancel_pending_grounding
        _reset_tracker_reacquire = VlmEdgeTamBridge._reset_tracker_reacquire

        def __init__(self) -> None:
            from z_manip_ros.vlm_edgetam_bridge import (
                _FrozenCoarseNavAuthorizationGate,
            )

            self._lock = threading.RLock()
            self._contract = TrackingContract()
            generation = self._contract.request('cancel this', now_s=0.0)
            self._contract.grounding_started(generation, now_s=0.1)
            self._future = StaleFuture()
            self._future_cancel_event = threading.Event()
            self._future_generation = generation
            self._future_image = object()
            self._future_grounding_scope = 'grasp_only'
            self._grounding_scope = 'grasp_only'
            self._expected_edge_seed_id = 'old-seed'
            self._expected_edge_seed_stamp_ns = 123
            self._tracker_failure_detail = 'old failure'
            self._tracker_reacquire_attempts = 1
            self._tracker_reacquire_due_monotonic_s = 1.0
            self._tracker_reacquire_deadline_monotonic_s = 2.0
            self._tracker_reacquire_instruction = 'cancel this'
            self._tracker_reacquire_request_id = 'old-request'
            self._tracker_reacquire_state = 'scheduled'
            self._coarse_nav_authorization = (
                _FrozenCoarseNavAuthorizationGate(0.30)
            )
            self.seed_commands = []
            self.clear_calls = 0
            self.zero_calls = 0
            self.contract_publish_calls = 0
            self.restart_calls = 0

        def _clear_tracker_messages(self) -> None:
            self.clear_calls += 1

        def _publish_zero_velocity(self) -> None:
            self.zero_calls += 1

        def _publish_contract(self) -> None:
            self.contract_publish_calls += 1

        def _publish_seed_command(self, action: str) -> None:
            self.seed_commands.append(action)

        def _maybe_start_grounding(self, _now: float) -> None:
            self.restart_calls += 1

    harness = Harness()
    old_generation = harness._future_generation

    VlmEdgeTamBridge._reset_cb(harness, None)
    VlmEdgeTamBridge._poll_grounding(harness, 0.2)

    assert harness._contract.generation == old_generation + 1
    assert harness._contract.snapshot.phase is ContractPhase.IDLE
    assert harness._future is None
    assert harness._future_image is None
    assert harness.seed_commands == ['cancel']
    assert harness.clear_calls == 1
    assert harness.zero_calls == 1
    assert harness.contract_publish_calls == 1
    assert harness._future_cancel_event is None
    assert harness.restart_calls == 0
    assert harness._tracker_reacquire_attempts == 0
    assert harness._tracker_reacquire_state == 'idle'


def test_bridge_request_identity_is_idempotent_and_task_owned():
    pytest.importorskip('rclpy')
    from std_msgs.msg import String
    from z_manip_ros.vlm_edgetam_bridge import VlmEdgeTamBridge

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    class Harness:
        _reset_tracker_reacquire = VlmEdgeTamBridge._reset_tracker_reacquire

        def __init__(self) -> None:
            from z_manip_ros.vlm_edgetam_bridge import (
                _FrozenCoarseNavAuthorizationGate,
            )

            self._lock = threading.RLock()
            self._contract = TrackingContract()
            self._expected_edge_seed_id = ''
            self._expected_edge_seed_stamp_ns = None
            self._tracker_failure_detail = ''
            self._tracker_reacquire_attempts = 0
            self._tracker_reacquire_due_monotonic_s = None
            self._tracker_reacquire_deadline_monotonic_s = None
            self._tracker_reacquire_instruction = ''
            self._tracker_reacquire_request_id = ''
            self._tracker_reacquire_state = 'idle'
            self._coarse_nav_authorization = (
                _FrozenCoarseNavAuthorizationGate(0.30)
            )
            self._producer_epoch = 'producer-a'
            self._current_seed_request = None
            self._seed_images = {}
            self._seed_offer_manifests = {}
            self._latest_seed_image = None
            self._latest_seed_image_at = None
            self._seed_request_pub = Publisher()
            self.clear_calls = 0
            self.cancel_calls = 0
            self.start_calls = 0
            self.publish_calls = 0
            self.warnings = []

        @staticmethod
        def _now_s() -> float:
            return 1.0

        @staticmethod
        def _monotonic_s() -> float:
            return 2.0

        @staticmethod
        def get_clock():
            return SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=1_000_000_000),
            )

        def _clear_tracker_messages(self) -> None:
            self.clear_calls += 1

        def _cancel_pending_grounding(self) -> None:
            self.cancel_calls += 1

        def _publish_zero_velocity(self) -> None:
            return

        def _maybe_start_grounding(self, _now: float) -> None:
            self.start_calls += 1

        def _publish_contract(self) -> None:
            self.publish_calls += 1

        def _clear_seed_offer_join(self) -> None:
            VlmEdgeTamBridge._clear_seed_offer_join(self)

        def _publish_seed_command(self, action: str):
            return VlmEdgeTamBridge._publish_seed_command(self, action)

        def get_logger(self):
            return SimpleNamespace(warn=lambda message: self.warnings.append(message))

    def message(
        request_id: str,
        instruction: str,
        scope: str = 'grasp_for_place',
    ) -> String:
        return String(data=json.dumps({
            'schema': 'z_manip.grounding_request.v2',
            'request_id': request_id,
            'instruction': instruction,
            'scope': scope,
        }))

    harness = Harness()
    VlmEdgeTamBridge._request_cb(harness, message('task-a', 'pick the mug'))
    generation = harness._contract.generation
    assert harness._contract.request_id == 'task-a'
    assert harness._contract.instruction == 'pick the mug'
    assert harness._grounding_scope == 'grasp_for_place'
    assert harness._tracker_reacquire_state == 'idle'

    VlmEdgeTamBridge._request_cb(harness, message('task-a', 'pick the mug'))
    assert harness._contract.generation == generation
    assert harness.clear_calls == 1
    assert len(harness._seed_request_pub.messages) == 1
    first_seed_request = json.loads(harness._seed_request_pub.messages[0].data)
    assert first_seed_request['action'] == 'arm'
    assert first_seed_request['request_id'] == 'task-a'
    assert len(first_seed_request['nonce']) == 32

    VlmEdgeTamBridge._request_cb(harness, message('task-a', 'pick something else'))
    assert harness._contract.generation == generation
    assert harness.warnings

    VlmEdgeTamBridge._request_cb(
        harness,
        message('task-a', 'pick the mug', 'place_support'),
    )
    assert harness._grounding_scope == 'grasp_for_place'

    VlmEdgeTamBridge._request_cb(harness, message('task-b', 'pick the bottle'))
    assert harness._contract.generation == generation + 1
    assert harness._contract.request_id == 'task-b'
    assert harness._contract.instruction == 'pick the bottle'
    assert harness.clear_calls == 2
    assert len(harness._seed_request_pub.messages) == 2
    replacement_seed_request = json.loads(harness._seed_request_pub.messages[-1].data)
    assert replacement_seed_request['request_id'] == 'task-b'
    assert replacement_seed_request['grounding_generation'] == generation + 1
    assert replacement_seed_request['nonce'] != first_seed_request['nonce']


def test_affordance_echoes_request_and_producer_identity():
    pytest.importorskip('rclpy')
    from z_manip_ros.vlm_edgetam_bridge import VlmEdgeTamBridge

    box = SimpleNamespace(x1=0.1, y1=0.2, x2=0.4, y2=0.8)
    result = SimpleNamespace(
        model='test/model',
        target_label='mustard bottle',
        target_bbox=box,
        confidence=0.9,
        grasp_part_label=None,
        grasp_part_bbox=None,
        avoid_regions=(),
        preferred_approach_camera=(0.1, 0.0, 1.0),
        placement_region_label=None,
        placement_region_bbox=None,
        placement_avoid_regions=(),
        placement_verification=None,
        constraints=('avoid cap',),
    )
    header = SimpleNamespace(
        frame_id='wrist_camera_optical_frame',
        stamp=SimpleNamespace(sec=3, nanosec=4),
    )

    value = json.loads(VlmEdgeTamBridge._affordance_json(
        2,
        result,
        header,
        request_id='task-request-2',
        producer_epoch='bridge-epoch-9',
        grounding_scope='grasp_only',
    ))

    assert value['schema'] == 'z_manip.affordance.v2'
    assert value['generation'] == 2
    assert value['request_id'] == 'task-request-2'
    assert value['producer_epoch'] == 'bridge-epoch-9'
    assert value['grounding_scope'] == 'grasp_only'
    assert value['source_image'] == {
        'frame_id': 'wrist_camera_optical_frame',
        'stamp_ns': 3_000_000_004,
    }


def test_running_vlm_cancel_is_queue_bounded_and_latest_request_starts_promptly():
    pytest.importorskip('rclpy')
    from z_manip.perception.vlm_affordance import VLMCancellationError
    from z_manip_ros.vlm_edgetam_bridge import (
        _GroundingSeedImage,
        _SeedRequestIdentity,
        VlmEdgeTamBridge,
    )

    first_started = threading.Event()
    first_cancel_seen = threading.Event()
    allow_first_teardown = threading.Event()
    replacement_started = threading.Event()

    class CancellableVLM:
        def __init__(self) -> None:
            self.requests = []

        def locate_and_reason(
            self,
            _jpeg,
            instruction,
            *,
            grounding_scope,
            cancel_event,
        ):
            self.requests.append((instruction, grounding_scope))
            if len(self.requests) == 1:
                assert grounding_scope == 'grasp_only'
                first_started.set()
                assert cancel_event.wait(timeout=0.5)
                first_cancel_seen.set()
                assert allow_first_teardown.wait(timeout=0.5)
            else:
                assert grounding_scope == 'place_support'
                replacement_started.set()
                assert cancel_event.wait(timeout=0.5)
            raise VLMCancellationError('test cancellation')

    class Harness:
        _cancel_pending_grounding = VlmEdgeTamBridge._cancel_pending_grounding
        _maybe_start_grounding = VlmEdgeTamBridge._maybe_start_grounding
        _poll_grounding = VlmEdgeTamBridge._poll_grounding
        _contract_allows_async_commit = (
            VlmEdgeTamBridge._contract_allows_async_commit
        )

        def __init__(self) -> None:
            self._contract = TrackingContract(frame_wait_timeout_s=2.0)
            self._worker = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix='test_openrouter_vlm',
            )
            self._future = None
            self._future_cancel_event = None
            self._future_generation = 0
            self._future_image = None
            self._future_grounding_scope = None
            self._grounding_scope = 'grasp_only'
            self._current_seed_request = None
            self._latest_seed_image = None
            self._latest_seed_image_at = 0.0
            self._vlm = CancellableVLM()
            self.failures = []

        def set_seed(self, generation: int) -> None:
            nonce = f'{generation:032x}'
            self._current_seed_request = _SeedRequestIdentity(
                self._contract.request_id,
                'producer-a',
                generation,
                nonce,
                0,
            )
            self._latest_seed_image = _GroundingSeedImage(
                header=object(),
                width=640,
                height=480,
                jpeg=b'jpeg',
                offer_token=f'z-manip-seed:{generation}:{nonce}',
                adapter_generation=generation,
                request_nonce=nonce,
                request_id=self._contract.request_id,
                producer_epoch='producer-a',
                grounding_generation=generation,
            )

        @staticmethod
        def get_parameter(name):
            return SimpleNamespace(value={
                'max_camera_age_s': 0.5,
            }[name])

        def get_logger(self):
            return SimpleNamespace(error=lambda message: self.failures.append(message))

        def _handle_new_failure(self):
            raise AssertionError('grounding setup unexpectedly failed')

    harness = Harness()
    first_generation = harness._contract.request('first request', now_s=0.0)
    harness.set_seed(first_generation)
    harness._maybe_start_grounding(0.0)
    assert first_started.wait(timeout=0.5)
    running_future = harness._future
    assert running_future is not None and not running_future.done()

    harness._grounding_scope = 'place_support'
    for index in range(5):
        generation = harness._contract.request(
            f'latest request {index}',
            now_s=0.1 + index * 0.01,
        )
        harness.set_seed(generation)
        harness._cancel_pending_grounding()
        harness._maybe_start_grounding(0.1 + index * 0.01)

    assert first_cancel_seen.wait(timeout=0.5)
    assert harness._future is running_future
    assert harness._future_generation == first_generation
    assert harness._vlm.requests == [('first request', 'grasp_only')]

    replacement_requested_at = time.monotonic()
    allow_first_teardown.set()
    while not running_future.done() and time.monotonic() - replacement_requested_at < 0.5:
        time.sleep(0.005)
    assert running_future.done()
    with pytest.raises(VLMCancellationError):
        running_future.result()
    harness._poll_grounding(0.3)

    assert replacement_started.wait(timeout=0.5)
    assert time.monotonic() - replacement_requested_at < 0.5
    assert harness._vlm.requests == [
        ('first request', 'grasp_only'),
        ('latest request 4', 'place_support'),
    ]
    assert harness._contract.phase is ContractPhase.GROUNDING

    harness._contract.reset()
    harness._cancel_pending_grounding()
    replacement = harness._future
    assert replacement is not None
    harness._worker.shutdown(wait=True, cancel_futures=True)
    with pytest.raises(VLMCancellationError):
        replacement.result()
    assert not any(
        thread.name.startswith('test_openrouter_vlm')
        for thread in threading.enumerate()
    )


def test_perception_status_exposes_only_an_exact_validated_observation_key():
    pytest.importorskip('rclpy')
    from z_manip_ros.vlm_edgetam_bridge import VlmEdgeTamBridge

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    class Clock:
        @staticmethod
        def now():
            return SimpleNamespace(to_msg=lambda: object())

    class Harness:
        _header_key = staticmethod(VlmEdgeTamBridge._header_key)
        _status_observation_key = VlmEdgeTamBridge._status_observation_key
        _verified_observation_key = VlmEdgeTamBridge._verified_observation_key
        _relay_if_valid = VlmEdgeTamBridge._relay_if_valid
        _publish_contract = VlmEdgeTamBridge._publish_contract

        def __init__(self) -> None:
            self._contract = _waiting_tracker()
            _acquire(self._contract)
            header = SimpleNamespace(
                stamp=SimpleNamespace(sec=2, nanosec=3),
                frame_id='wrist_camera_optical_frame',
            )
            self._latest_detections = SimpleNamespace(header=header)
            self._latest_target = SimpleNamespace(header=header)
            self._latest_cloud = SimpleNamespace(header=header)
            self._latest_observation_key = (2_000_000_003, 'wrist_camera_optical_frame')
            self._bundle_serial = 1
            self._relayed_bundle_serial = 0
            self._tracker_failure_detail = ''
            self._tracker_reacquire_attempts = 0
            self._tracker_reacquire_state = 'idle'
            self._producer_epoch = 'test-bridge-epoch'
            self._grounding_scope = 'grasp_only'
            self._valid_pub = Publisher()
            self._status_pub = Publisher()
            self._tracked_2d_pub = Publisher()
            self._target_3d_pub = Publisher()
            self._target_cloud_pub = Publisher()
            self.failure_resets = 0

        @staticmethod
        def get_clock():
            return Clock()

        @staticmethod
        def _now_s():
            return 2.1

        def _handle_new_failure(self):
            self.failure_resets += 1
            self._latest_detections = None
            self._latest_target = None
            self._latest_cloud = None
            self._latest_observation_key = None

    harness = Harness()
    harness._publish_contract()
    values = {
        item.key: item.value
        for item in harness._status_pub.messages[-1].status[0].values
    }
    assert values['observation_stamp_ns'] == '2000000003'
    assert values['observation_frame_id'] == 'wrist_camera_optical_frame'
    assert values['request_id'] == 'test-task-request'
    assert values['producer_epoch'] == 'test-bridge-epoch'
    assert values['valid'] == 'true'
    harness._relay_if_valid()
    assert len(harness._tracked_2d_pub.messages) == 1
    assert len(harness._target_3d_pub.messages) == 1
    assert len(harness._target_cloud_pub.messages) == 1

    harness._latest_target.header = SimpleNamespace(
        stamp=SimpleNamespace(sec=2, nanosec=4),
        frame_id='wrist_camera_optical_frame',
    )
    harness._bundle_serial = 2
    harness._relay_if_valid()
    assert len(harness._tracked_2d_pub.messages) == 1
    assert len(harness._target_3d_pub.messages) == 1
    assert len(harness._target_cloud_pub.messages) == 1
    harness._publish_contract()
    values = {
        item.key: item.value
        for item in harness._status_pub.messages[-1].status[0].values
    }
    assert values['observation_stamp_ns'] == ''
    assert values['observation_frame_id'] == ''
    assert values['valid'] == 'false'
    assert harness._contract.phase is ContractPhase.FAILED
    assert harness.failure_resets == 1
    assert harness._valid_pub.messages[-1].data is False


def test_normalized_box_uses_runtime_image_size_without_scene_constants():
    box = normalized_xyxy_to_pixel_box((0.1, 0.2, 0.5, 0.8), 848, 480)
    assert box.center_x == pytest.approx(254.4)
    assert box.center_y == pytest.approx(240.0)
    assert box.size_x == pytest.approx(339.2)
    assert box.size_y == pytest.approx(288.0)


def test_seed_box_expansion_uses_bounded_margin_and_clips_to_image():
    box = normalized_xyxy_to_pixel_box((0.9, 0.8, 1.0, 1.0), 640, 480)
    expanded = expand_pixel_box(box, 640, 480, 0.12)
    assert expanded.center_x == pytest.approx(604.16)
    assert expanded.center_y == pytest.approx(426.24)
    assert expanded.size_x == pytest.approx(71.68)
    assert expanded.size_y == pytest.approx(107.52)


def test_deployment_files_pin_the_no_heuristic_topic_contract():
    package = Path(__file__).resolve().parents[1]
    node = (package / 'z_manip_ros' / 'vlm_edgetam_bridge.py').read_text()
    contract_source = (package / 'z_manip_ros' / 'contract.py').read_text()
    launch = (package / 'launch' / 'perception.launch.py').read_text()
    config = (package / 'config' / 'perception.yaml').read_text()
    manifest = (package / 'package.xml').read_text()
    readme = (package / 'README.md').read_text()

    assert 'OpenRouterVLM' in node
    assert 'cv2.imdecode' in node
    assert 'z_manip.grounding_request.v1' in contract_source
    assert "KeyValue(key='request_id'" in node
    assert "key='instruction_sha256'" in node
    assert "KeyValue(key='producer_epoch'" in node
    assert 'ColorDepthTracker' not in node
    for topic in (
        '/camera/color/image_raw',
        '/camera/aligned_depth_to_color/image_raw',
        '/camera/color/camera_info',
        '/track_3d/seed_request',
        '/track_3d/exact_seed_image',
        '/track_3d/seed_offer_manifest',
        '/track_3d/init_bbox',
        '/track_3d/failure',
        '/track_3d/frame_manifest',
        '/track_3d/detections_2d',
        '/track_3d/selected_target_pointcloud',
    ):
        assert topic in launch or topic in config
    assert 'OPENROUTER_API_KEY' in config
    assert 'api_key:' not in config.lower()
    assert 'vlm_model_timeouts_s: [8.5]' in config
    assert 'vlm_local_grounding_url: "http://127.0.0.1:8771"' in config
    assert 'vlm_local_grounding_timeout_s: 1.25' in config
    assert 'seed_bbox_padding_fraction: 0.12' in config
    assert config.count('relative_0_1000') == 1
    assert 'vlm_model_bbox_coordinate_spaces' in node
    assert 'vlm_provider_retries: 1' in config
    assert 'vlm_timeout_retries: 0' in config
    assert 'vlm_hedge_delay_s: 0.05' in config
    assert 'qwen/qwen3-vl-8b-instruct:nitro' in config
    assert 'qwen/qwen3-vl-235b-a22b-instruct:nitro' not in config
    assert 'tracker_data_timeout_s: 1.0' in config
    assert 'vlm_max_semantic_conflict_coverage_ratio' in config
    assert '_record_vlm_attempt' in node
    assert '/z_manip/visual_search/active' in config
    assert "'stop_cmd_topic': '/safety_cmd_vel'" in node
    assert 'stop_cmd_topic: /safety_cmd_vel' in config
    assert '/local_movement_cmd_vel' not in node
    assert '/local_movement_cmd_vel' not in config
    assert 'motion_override_timeout_s' in config
    assert 'edge_seed_image_topic: /track_3d/exact_seed_image' in config
    assert 'edge_seed_request_topic: /track_3d/seed_request' in config
    assert 'edge_seed_offer_manifest_topic: /track_3d/seed_offer_manifest' in config
    assert "topic('edge_seed_image_topic')" in node
    assert 'DurabilityPolicy.TRANSIENT_LOCAL' in node
    assert 'seed_id = image_meta.offer_token' in node
    assert "'color_topic': '/camera/color/image_raw'" not in node
    assert '_motion_override_is_fresh' in node
    assert "FindPackageShare('z_manip_edgetam')" in launch
    assert "'edgetam.launch.py'" in launch
    assert "DeclareLaunchArgument('start_edge_tam', default_value='true')" in launch
    assert "'tracker_config'" in launch
    assert "'tracker_service_url'" in launch
    assert "default_value='http://127.0.0.1:8092'" in launch
    assert 'tracker_package' not in launch
    assert 'tracker_executable' not in launch
    assert '<exec_depend>curl</exec_depend>' in manifest
    assert '<exec_depend>z_manip_edgetam</exec_depend>' in manifest
    assert '<exec_depend>python3-numpy</exec_depend>' in manifest
    assert '`start_edge_tam:=false`' in readme
    assert '/track_3d/frame_manifest' in readme
