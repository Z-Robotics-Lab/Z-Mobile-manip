"""Focused safety tests for the task-to-perception bridge boundaries."""

from contextlib import nullcontext
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from z_manip_ros.contract import ContractPhase, FailureCode
from z_manip_ros.vlm_edgetam_bridge import (
    _FrozenCoarseNavAuthorizationGate,
    _parse_seed_offer_manifest,
    _SeedRequestIdentity,
    VlmEdgeTamBridge,
)


REQUEST_ID = 'request-a'
PRODUCER_EPOCH = 'producer-a'
GENERATION = 7


def _authorization(**overrides: object) -> str:
    value = {
        'schema': 'z_manip.frozen_coarse_nav_authorization.v1',
        'active': True,
        'request_id': REQUEST_ID,
        'producer_epoch': PRODUCER_EPOCH,
        'generation': GENERATION,
        'observation_serial': 11,
        'nav_goal_id': 'work-goal-a',
    }
    value.update(overrides)
    return json.dumps(value)


def _update(
    gate: _FrozenCoarseNavAuthorizationGate,
    payload: str,
    *,
    now: float = 10.0,
    request_id: str = REQUEST_ID,
    producer_epoch: str = PRODUCER_EPOCH,
    generation: int = GENERATION,
) -> bool:
    return gate.update(
        payload,
        received_monotonic_s=now,
        request_id=request_id,
        producer_epoch=producer_epoch,
        generation=generation,
    )


def _fresh(
    gate: _FrozenCoarseNavAuthorizationGate,
    *,
    now: float,
    request_id: str = REQUEST_ID,
    producer_epoch: str = PRODUCER_EPOCH,
    generation: int = GENERATION,
) -> bool:
    return gate.is_fresh(
        now_monotonic_s=now,
        request_id=request_id,
        producer_epoch=producer_epoch,
        generation=generation,
    )


def test_frozen_nav_authorization_uses_receive_monotonic_timeout() -> None:
    gate = _FrozenCoarseNavAuthorizationGate(0.30)

    assert not _fresh(gate, now=9.9)
    assert _update(gate, _authorization(), now=10.0)
    assert _fresh(gate, now=10.299999)
    assert not _fresh(gate, now=10.3001)
    assert not _fresh(gate, now=10.1)


@pytest.mark.parametrize(
    'payload',
    (
        'not-json',
        _authorization(active=1),
        _authorization(observation_serial=True),
        _authorization(nav_goal_id='contains whitespace'),
        _authorization(request_id='trailing-space '),
        _authorization(unexpected='field'),
        (
            '{"schema":"z_manip.frozen_coarse_nav_authorization.v1",'
            '"active":true,"active":false,"request_id":"request-a",'
            '"producer_epoch":"producer-a","generation":7,'
            '"observation_serial":11,"nav_goal_id":"work-goal-a"}'
        ),
    ),
)
def test_malformed_authorization_revokes_a_previously_fresh_gate(payload: str) -> None:
    gate = _FrozenCoarseNavAuthorizationGate(0.30)
    assert _update(gate, _authorization(), now=5.0)

    with pytest.raises(ValueError):
        _update(gate, payload, now=5.1)

    assert not _fresh(gate, now=5.1)


@pytest.mark.parametrize(
    'overrides',
    (
        {'request_id': 'other-request'},
        {'producer_epoch': 'other-producer'},
        {'generation': GENERATION + 1},
    ),
)
def test_active_request_identity_mismatch_revokes_authorization(overrides) -> None:
    gate = _FrozenCoarseNavAuthorizationGate(0.30)
    assert _update(gate, _authorization(), now=5.0)

    assert not _update(gate, _authorization(**overrides), now=5.1)
    assert not _fresh(gate, now=5.1)


@pytest.mark.parametrize(
    'overrides',
    (
        {'observation_serial': 12},
        {'nav_goal_id': 'work-goal-b'},
    ),
)
def test_work_goal_identity_cannot_change_while_authorization_is_active(
    overrides,
) -> None:
    gate = _FrozenCoarseNavAuthorizationGate(0.30)
    assert _update(gate, _authorization(), now=5.0)

    assert not _update(gate, _authorization(**overrides), now=5.1)
    assert not _fresh(gate, now=5.1)


def test_inactive_heartbeat_and_monotonic_rollback_revoke_gate() -> None:
    gate = _FrozenCoarseNavAuthorizationGate(0.30)
    assert _update(gate, _authorization(), now=5.0)
    assert not _update(gate, _authorization(active=False), now=5.1)
    assert not _fresh(gate, now=5.1)

    assert _update(gate, _authorization(), now=6.0)
    assert not _update(gate, _authorization(), now=5.9)
    assert not _fresh(gate, now=6.0)


@pytest.mark.parametrize('timeout', (0.0, 0.350001, float('inf'), float('nan')))
def test_authorization_timeout_has_a_hard_safety_bound(timeout: float) -> None:
    with pytest.raises(ValueError):
        _FrozenCoarseNavAuthorizationGate(timeout)


def _release_harness(phase: ContractPhase, failure: FailureCode):
    snapshot = SimpleNamespace(
        phase=phase,
        failure=failure,
        request_id=REQUEST_ID,
        generation=GENERATION,
    )
    gate = _FrozenCoarseNavAuthorizationGate(0.30)
    assert _update(gate, _authorization(), now=10.0)
    return SimpleNamespace(
        _contract=SimpleNamespace(snapshot=snapshot),
        _coarse_nav_authorization=gate,
        _producer_epoch=PRODUCER_EPOCH,
        _monotonic_s=lambda: 10.1,
    )


def test_authorization_releases_only_failed_tracker_health_hold() -> None:
    tracking = _release_harness(
        ContractPhase.TRACKING,
        FailureCode.TRACKER_REPORTED_LOSS,
    )
    grounding_failure = _release_harness(
        ContractPhase.FAILED,
        FailureCode.GROUNDING_FAILED,
    )
    tracker_failure = _release_harness(
        ContractPhase.FAILED,
        FailureCode.TRACKER_REPORTED_LOSS,
    )

    assert not VlmEdgeTamBridge._frozen_coarse_nav_authorization_releases_hold(
        tracking,
    )
    assert not VlmEdgeTamBridge._frozen_coarse_nav_authorization_releases_hold(
        grounding_failure,
    )
    assert VlmEdgeTamBridge._frozen_coarse_nav_authorization_releases_hold(
        tracker_failure,
    )


class _HealthContract:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.phase = snapshot.phase

    def tick(self, *, now_s: float) -> None:
        del now_s


def test_fresh_authorization_suppresses_only_repeated_health_zero() -> None:
    snapshot = SimpleNamespace(
        phase=ContractPhase.FAILED,
        failure=FailureCode.TRACKER_REPORTED_LOSS,
        request_id=REQUEST_ID,
        generation=GENERATION,
    )
    published_zeros = []
    harness = SimpleNamespace(
        _lock=nullcontext(),
        _contract=_HealthContract(snapshot),
        _poll_grounding=lambda _now: None,
        _handle_new_failure=lambda: None,
        get_parameter=lambda _name: SimpleNamespace(value=True),
        _motion_override_is_fresh=lambda _now: False,
        _frozen_coarse_nav_authorization_releases_hold=lambda: True,
        _publish_zero_velocity=lambda: published_zeros.append(True),
        _relay_if_valid=lambda: None,
        _publish_contract=lambda: None,
        _now_s=lambda: 100.0,
    )

    VlmEdgeTamBridge._health_cb(harness)

    assert published_zeros == []
    harness._frozen_coarse_nav_authorization_releases_hold = lambda: False
    VlmEdgeTamBridge._health_cb(harness)
    assert published_zeros == [True]


def test_health_ticks_contract_before_polling_a_ready_future() -> None:
    events: list[str] = []

    class Contract:
        phase = ContractPhase.IDLE

        def tick(self, *, now_s: float) -> None:
            del now_s
            events.append('tick')

    harness = SimpleNamespace(
        _lock=nullcontext(),
        _contract=Contract(),
        _poll_grounding=lambda _now: events.append('poll'),
        _handle_new_failure=lambda: events.append('failure'),
        get_parameter=lambda _name: SimpleNamespace(value=True),
        _motion_override_is_fresh=lambda _now: False,
        _frozen_coarse_nav_authorization_releases_hold=lambda: False,
        _publish_zero_velocity=lambda: events.append('zero'),
        _relay_if_valid=lambda: None,
        _publish_contract=lambda: None,
        _now_s=lambda: 100.0,
    )

    VlmEdgeTamBridge._health_cb(harness)

    assert events == ['tick', 'poll']


def test_authorization_never_suppresses_the_initial_failure_zero() -> None:
    effects = []
    harness = SimpleNamespace(
        _tracker_failure_detail='mask continuity',
        _contract=SimpleNamespace(
            failure=FailureCode.TRACKER_REPORTED_LOSS,
        ),
        _cancel_pending_grounding=lambda: effects.append('cancel_grounding'),
        _clear_tracker_messages=lambda: effects.append('clear_tracker'),
        _publish_seed_command=lambda action: effects.append(action),
        _publish_zero_velocity=lambda: effects.append('zero'),
        get_logger=lambda: SimpleNamespace(
            error=lambda _detail: effects.append('log_failure'),
        ),
    )

    VlmEdgeTamBridge._handle_new_failure(harness)

    assert effects == [
        'cancel_grounding',
        'clear_tracker',
        'cancel',
        'zero',
        'log_failure',
    ]


def test_first_manifest_binds_session_after_seed_offer_prebinds_generation() -> None:
    class Contract:
        phase = ContractPhase.WAITING_TRACKER
        generation = GENERATION

        def tracker_failed(self, *, now_s: float) -> None:
            raise AssertionError(f'valid first manifest failed at {now_s}')

    seed_id = 'z-manip-seed:12:' + ('a' * 32)
    manifest = json.dumps({
        'schema': 'z_manip.tracker_frame.v1',
        'seed_id': seed_id,
        'seed_stamp_ns': 2_001,
        'adapter_generation': 12,
        'result_stamp_ns': 2_101,
        'frame_id': 'camera_color_optical_frame',
        'session_id': 'session-a',
        'track_id': 'track-a',
    })
    additions: list[dict[str, object]] = []
    finishes: list[ContractPhase] = []
    harness = SimpleNamespace(
        _lock=nullcontext(),
        _contract=Contract(),
        _expected_edge_seed_id=seed_id,
        _expected_edge_seed_stamp_ns=2_001,
        _expected_adapter_generation=12,
        _expected_edge_session_id='',
        _expected_edge_track_id='',
        _tracker_failure_detail='',
        _observation_bundler=SimpleNamespace(
            add=lambda kind, **kwargs: additions.append(
                {'kind': kind, **kwargs},
            ),
        ),
        _now_s=lambda: 3.0,
        _finish_tracker_update=finishes.append,
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
    )

    VlmEdgeTamBridge._frame_manifest_cb(
        harness,
        SimpleNamespace(data=manifest),
    )

    assert harness._expected_adapter_generation == 12
    assert harness._expected_edge_session_id == 'session-a'
    assert harness._expected_edge_track_id == 'track-a'
    assert additions[0]['kind'] == 'manifest'
    assert finishes == []


def _offer_manifest(**overrides: object) -> str:
    value = {
        'schema': 'z_manip.seed_offer.v1',
        'request_id': REQUEST_ID,
        'producer_epoch': PRODUCER_EPOCH,
        'grounding_generation': GENERATION,
        'request_nonce': '1' * 32,
        'adapter_generation': 12,
        'offer_token': 'z-manip-seed:12:' + ('a' * 32),
        'stamp_ns': 2_001,
        'frame_id': 'camera_color_optical_frame',
        'width': 3,
        'height': 2,
    }
    value.update(overrides)
    return json.dumps(value)


def _offer_harness():
    starts: list[float] = []
    warnings: list[str] = []
    request = _SeedRequestIdentity(
        REQUEST_ID,
        PRODUCER_EPOCH,
        GENERATION,
        '1' * 32,
        2_000,
    )
    contract = SimpleNamespace(
        phase=ContractPhase.WAITING_FRAME,
        tick=lambda **_kwargs: None,
    )
    harness = SimpleNamespace(
        _lock=nullcontext(),
        _contract=contract,
        _current_seed_request=request,
        _seed_images={},
        _seed_offer_manifests={},
        _latest_seed_image=None,
        _latest_seed_image_at=None,
        _now_s=lambda: 3.0,
        _header_key=VlmEdgeTamBridge._header_key,
        _maybe_start_grounding=starts.append,
        _handle_new_failure=lambda: None,
        _publish_contract=lambda: None,
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
    )
    harness._contract_allows_async_commit = lambda now: (
        VlmEdgeTamBridge._contract_allows_async_commit(harness, now)
    )
    harness._try_accept_seed_offer = lambda token, now: (
        VlmEdgeTamBridge._try_accept_seed_offer(harness, token, now)
    )
    return harness, starts, warnings


def _offer_image(*, token: str | None = None):
    chosen = token or 'z-manip-seed:12:' + ('a' * 32)
    ok, encoded = cv2.imencode('.jpg', np.zeros((2, 3, 3), dtype=np.uint8))
    assert ok
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=0, nanosec=2_001),
            frame_id='camera_color_optical_frame',
        ),
        format=f'jpeg; z_manip_seed_offer={chosen}',
        data=encoded.tobytes(),
    )


@pytest.mark.parametrize('order', ('image_first', 'manifest_first'))
def test_seed_image_and_manifest_join_in_either_dds_order(order: str) -> None:
    harness, starts, warnings = _offer_harness()
    image = _offer_image()
    manifest = SimpleNamespace(data=_offer_manifest())

    callbacks = (
        (
            lambda: VlmEdgeTamBridge._image_cb(harness, image),
            lambda: VlmEdgeTamBridge._seed_offer_manifest_cb(harness, manifest),
        )
        if order == 'image_first'
        else (
            lambda: VlmEdgeTamBridge._seed_offer_manifest_cb(harness, manifest),
            lambda: VlmEdgeTamBridge._image_cb(harness, image),
        )
    )
    callbacks[0]()
    assert harness._latest_seed_image is None
    callbacks[1]()

    assert warnings == []
    assert starts == [3.0]
    assert harness._latest_seed_image.offer_token.startswith('z-manip-seed:12:')
    assert harness._latest_seed_image.adapter_generation == 12
    assert harness._latest_seed_image.request_nonce == '1' * 32


def test_contract_timeout_wins_before_seed_offer_admission() -> None:
    harness, starts, _warnings = _offer_harness()
    effects: list[str] = []

    class ExpiringContract:
        phase = ContractPhase.WAITING_FRAME

        def tick(self, *, now_s: float) -> None:
            assert now_s == 3.0
            effects.append('tick')
            self.phase = ContractPhase.FAILED

    token = 'z-manip-seed:12:' + ('a' * 32)
    harness._contract = ExpiringContract()
    harness._handle_new_failure = lambda: effects.append('failure')
    harness._publish_contract = lambda: effects.append('publish')
    harness._seed_images[token] = _offer_image()
    harness._seed_offer_manifests[token] = _parse_seed_offer_manifest(
        _offer_manifest(),
    )

    VlmEdgeTamBridge._try_accept_seed_offer(harness, token, 3.0)

    assert effects == ['tick', 'failure', 'publish']
    assert starts == []
    assert harness._latest_seed_image is None


def test_contract_timeout_during_jpeg_decode_wins_before_offer_commit(
    monkeypatch,
) -> None:
    harness, starts, _warnings = _offer_harness()
    effects: list[str] = []

    class ExpiringContract:
        phase = ContractPhase.WAITING_FRAME

        def tick(self, *, now_s: float) -> None:
            effects.append(f'tick:{now_s}')
            if now_s >= 4.0:
                self.phase = ContractPhase.FAILED

    real_imdecode = cv2.imdecode

    def decode(*args, **kwargs):
        effects.append('decode')
        return real_imdecode(*args, **kwargs)

    token = 'z-manip-seed:12:' + ('a' * 32)
    harness._contract = ExpiringContract()
    harness._now_s = lambda: 4.0
    harness._handle_new_failure = lambda: effects.append('failure')
    harness._publish_contract = lambda: effects.append('publish')
    harness._seed_images[token] = _offer_image()
    harness._seed_offer_manifests[token] = _parse_seed_offer_manifest(
        _offer_manifest(),
    )
    monkeypatch.setattr(cv2, 'imdecode', decode)

    VlmEdgeTamBridge._try_accept_seed_offer(harness, token, 3.0)

    assert effects == ['tick:3.0', 'decode', 'tick:4.0', 'failure', 'publish']
    assert starts == []
    assert harness._latest_seed_image is None


def test_contract_timeout_wins_before_ready_future_result_is_read() -> None:
    effects: list[str] = []

    class ExpiringContract:
        phase = ContractPhase.GROUNDING

        def tick(self, *, now_s: float) -> None:
            assert now_s == 4.0
            effects.append('tick')
            self.phase = ContractPhase.FAILED

    class ReadyFuture:
        @staticmethod
        def done() -> bool:
            return True

        @staticmethod
        def result():
            raise AssertionError('timed-out future result was read')

    harness = SimpleNamespace(
        _contract=ExpiringContract(),
        _future=ReadyFuture(),
        _handle_new_failure=lambda: effects.append('failure'),
        _publish_contract=lambda: effects.append('publish'),
    )
    harness._contract_allows_async_commit = lambda now: (
        VlmEdgeTamBridge._contract_allows_async_commit(harness, now)
    )

    VlmEdgeTamBridge._poll_grounding(harness, 4.0)

    assert effects == ['tick', 'failure', 'publish']


def test_contract_timeout_during_future_parsing_wins_before_bbox_commit() -> None:
    effects: list[str] = []

    class ExpiringContract:
        phase = ContractPhase.GROUNDING
        generation = GENERATION

        def tick(self, *, now_s: float) -> None:
            effects.append(f'tick:{now_s}')
            if now_s >= 4.0:
                self.phase = ContractPhase.FAILED

        @staticmethod
        def grounding_succeeded(*_args, **_kwargs):
            raise AssertionError('timed-out grounding result was committed')

    result = SimpleNamespace(
        target_bbox=SimpleNamespace(x1=0.1, y1=0.2, x2=0.4, y2=0.8),
        target_label='mustard bottle',
        confidence=0.9,
    )

    class ReadyFuture:
        @staticmethod
        def done() -> bool:
            return True

        @staticmethod
        def result():
            effects.append('result')
            return result

    request = _SeedRequestIdentity(
        REQUEST_ID,
        PRODUCER_EPOCH,
        GENERATION,
        '1' * 32,
        2_000,
    )
    image = SimpleNamespace(
        header=object(),
        width=640,
        height=480,
        offer_token='z-manip-seed:12:' + ('a' * 32),
        adapter_generation=12,
        request_nonce=request.nonce,
        request_id=request.request_id,
        producer_epoch=request.producer_epoch,
        grounding_generation=request.grounding_generation,
    )
    bbox_messages = []
    affordance_messages = []
    harness = SimpleNamespace(
        _contract=ExpiringContract(),
        _future=ReadyFuture(),
        _future_cancel_event=object(),
        _future_generation=GENERATION,
        _future_image=image,
        _future_grounding_scope='grasp_only',
        _grounding_scope='grasp_only',
        _current_seed_request=request,
        _now_s=lambda: 4.0,
        _handle_new_failure=lambda: effects.append('failure'),
        _publish_contract=lambda: effects.append('publish'),
        _bbox_pub=SimpleNamespace(publish=bbox_messages.append),
        _affordance_pub=SimpleNamespace(publish=affordance_messages.append),
    )
    harness._contract_allows_async_commit = lambda now: (
        VlmEdgeTamBridge._contract_allows_async_commit(harness, now)
    )

    VlmEdgeTamBridge._poll_grounding(harness, 3.0)

    assert effects == ['tick:3.0', 'result', 'tick:4.0', 'failure', 'publish']
    assert bbox_messages == []
    assert affordance_messages == []


def test_stale_replacement_offer_rearms_same_task_after_old_future_drains() -> None:
    published = []
    warnings = []

    class Publisher:
        def publish(self, message) -> None:
            published.append(json.loads(message.data))

    class DoneFuture:
        @staticmethod
        def done() -> bool:
            return True

        @staticmethod
        def result():
            raise AssertionError('stale future result was read')

    snapshot = SimpleNamespace(
        request_id=REQUEST_ID,
        generation=GENERATION,
    )
    contract = SimpleNamespace(
        phase=ContractPhase.WAITING_FRAME,
        generation=GENERATION,
        snapshot=snapshot,
        tick=lambda **_kwargs: None,
    )
    old_request = _SeedRequestIdentity(
        REQUEST_ID,
        PRODUCER_EPOCH,
        GENERATION,
        '1' * 32,
        2_000,
    )
    harness = SimpleNamespace(
        _contract=contract,
        _producer_epoch=PRODUCER_EPOCH,
        _current_seed_request=old_request,
        _seed_images={},
        _seed_offer_manifests={},
        _latest_seed_image=object(),
        _latest_seed_image_at=0.0,
        _seed_request_pub=Publisher(),
        _future=DoneFuture(),
        _future_cancel_event=object(),
        _future_generation=GENERATION - 1,
        _future_image=None,
        _future_grounding_scope='grasp_only',
        _grounding_scope='grasp_only',
        get_parameter=lambda name: SimpleNamespace(value={
            'max_camera_age_s': 0.5,
        }[name]),
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=3_000),
        ),
        get_logger=lambda: SimpleNamespace(
            warn=warnings.append,
            error=lambda _message: None,
        ),
        _handle_new_failure=lambda: None,
        _publish_contract=lambda: None,
    )
    harness._clear_seed_offer_join = lambda: (
        VlmEdgeTamBridge._clear_seed_offer_join(harness)
    )
    harness._publish_seed_command = lambda action: (
        VlmEdgeTamBridge._publish_seed_command(harness, action)
    )
    harness._rearm_current_seed_request = lambda: (
        VlmEdgeTamBridge._rearm_current_seed_request(harness)
    )
    harness._maybe_start_grounding = lambda now: (
        VlmEdgeTamBridge._maybe_start_grounding(harness, now)
    )
    harness._contract_allows_async_commit = lambda now: (
        VlmEdgeTamBridge._contract_allows_async_commit(harness, now)
    )

    VlmEdgeTamBridge._poll_grounding(harness, 1.0)

    assert contract.snapshot is snapshot
    assert contract.generation == GENERATION
    assert contract.phase is ContractPhase.WAITING_FRAME
    assert len(published) == 1
    replacement = published[0]
    assert replacement['action'] == 'arm'
    assert replacement['request_id'] == REQUEST_ID
    assert replacement['producer_epoch'] == PRODUCER_EPOCH
    assert replacement['grounding_generation'] == GENERATION
    assert replacement['nonce'] != old_request.nonce
    assert harness._current_seed_request.nonce == replacement['nonce']
    assert warnings


def test_stale_nonce_and_dimension_mismatch_never_start_grounding() -> None:
    stale, stale_starts, _warnings = _offer_harness()
    VlmEdgeTamBridge._image_cb(stale, _offer_image())
    VlmEdgeTamBridge._seed_offer_manifest_cb(
        stale,
        SimpleNamespace(data=_offer_manifest(request_nonce='2' * 32)),
    )
    assert stale_starts == []
    assert stale._latest_seed_image is None

    mismatch, mismatch_starts, _warnings = _offer_harness()
    VlmEdgeTamBridge._image_cb(mismatch, _offer_image())
    VlmEdgeTamBridge._seed_offer_manifest_cb(
        mismatch,
        SimpleNamespace(data=_offer_manifest(width=4)),
    )
    assert mismatch_starts == []
    assert mismatch._latest_seed_image is None


def test_seed_offer_manifest_parser_rejects_duplicate_fields() -> None:
    payload = _offer_manifest().replace(
        '"width": 3',
        '"width": 3, "width": 4',
    )
    with pytest.raises(ValueError):
        _parse_seed_offer_manifest(payload)

    with pytest.raises(ValueError):
        _parse_seed_offer_manifest(_offer_manifest(adapter_generation=13))
