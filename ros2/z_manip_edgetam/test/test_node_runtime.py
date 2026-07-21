"""Focused runtime tests for queue freshness and generation publication order."""

from collections import deque, OrderedDict
import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from z_manip_edgetam.core import (
    AcquisitionGate,
    CameraIntrinsics,
    ReseedRegistrationConfig,
    RgbdFrame,
)


def _provide_test_vision_messages() -> None:
    """Provide import-only message stubs when the host lacks vision_msgs."""
    try:
        importlib.import_module('vision_msgs.msg')
        return
    except ModuleNotFoundError:
        pass
    messages = ModuleType('vision_msgs.msg')
    for name in (
        'Detection2D',
        'Detection2DArray',
        'Detection3D',
        'ObjectHypothesis',
        'ObjectHypothesisWithPose',
    ):
        setattr(messages, name, type(name, (), {}))
    package = ModuleType('vision_msgs')
    package.msg = messages
    sys.modules['vision_msgs'] = package
    sys.modules['vision_msgs.msg'] = messages


_provide_test_vision_messages()
node_module = importlib.import_module('z_manip_edgetam.node')
EdgeTamAdapter = node_module.EdgeTamAdapter
Command = node_module._Command
CachedRgb = node_module._CachedRgb
SeedOffer = node_module._SeedOffer
SeedRequest = node_module._SeedRequest
parse_seed_request = node_module._parse_seed_request
ExactTimeSynchronizer = node_module._ExactTimeSynchronizer


class _FakeFilter:
    def __init__(self) -> None:
        self.callback = None
        self.args: tuple[object, ...] = ()

    def registerCallback(self, callback, *args):
        self.callback = callback
        self.args = args
        return callback

    def emit(self, message: object) -> None:
        assert self.callback is not None
        self.callback(message, *self.args)


def _stamped_message(stamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
        sec=stamp_ns // 1_000_000_000,
        nanosec=stamp_ns % 1_000_000_000,
    )))


def test_exact_synchronizer_joins_out_of_order_stamps() -> None:
    filters = [_FakeFilter() for _ in range(3)]
    received: list[tuple[object, ...]] = []
    synchronizer = ExactTimeSynchronizer(filters, 3)
    synchronizer.registerCallback(lambda *messages: received.append(messages))

    filters[0].emit(_stamped_message(2))
    filters[1].emit(_stamped_message(1))
    filters[2].emit(_stamped_message(1))
    filters[0].emit(_stamped_message(1))
    filters[1].emit(_stamped_message(2))
    filters[2].emit(_stamped_message(2))

    assert len(received) == 2
    assert [message.header.stamp.nanosec for message in received[0]] == [1, 1, 1]
    assert [message.header.stamp.nanosec for message in received[1]] == [2, 2, 2]


def test_exact_synchronizer_callback_exception_does_not_wedge() -> None:
    filters = [_FakeFilter() for _ in range(3)]
    synchronizer = ExactTimeSynchronizer(filters, 3)
    synchronizer.registerCallback(
        lambda *_messages: (_ for _ in ()).throw(ValueError('test failure')),
    )
    filters[0].emit(_stamped_message(1))
    filters[1].emit(_stamped_message(1))
    with pytest.raises(ValueError, match='test failure'):
        filters[2].emit(_stamped_message(1))

    received: list[tuple[object, ...]] = []
    synchronizer.registerCallback(lambda *messages: received.append(messages))
    for source in filters:
        source.emit(_stamped_message(2))
    assert len(received) == 1


def test_exact_synchronizer_drops_invalid_stamp_without_wedging() -> None:
    filters = [_FakeFilter() for _ in range(3)]
    received: list[tuple[object, ...]] = []
    synchronizer = ExactTimeSynchronizer(filters, 3)
    synchronizer.registerCallback(lambda *messages: received.append(messages))

    filters[0].emit(SimpleNamespace())
    for source in filters:
        source.emit(_stamped_message(3))
    assert len(received) == 1


def _seed_request(
    *,
    action: str = 'arm',
    generation: int = 4,
    nonce: str = '1' * 32,
    floor_ns: int = 5,
) -> SeedRequest:
    return SeedRequest(
        action=action,
        request_id='request-a',
        producer_epoch='producer-a',
        grounding_generation=generation,
        nonce=nonce,
        source_stamp_floor_ns=floor_ns,
    )


def _frame(stamp_ns: int) -> RgbdFrame:
    return RgbdFrame(
        stamp_ns=stamp_ns,
        frame_id='camera_color_optical_frame',
        image_jpeg=b'\xff\xd8test\xff\xd9',
        width=2,
        height=2,
        depth_m=np.ones((2, 2), dtype=np.float32),
        intrinsics=CameraIntrinsics(2.0, 2.0, 0.5, 0.5),
    )


def _parameter_reader(values: dict[str, object]):
    def get_parameter(name: str) -> SimpleNamespace:
        return SimpleNamespace(value=values[name])

    return get_parameter


def test_exact_sync_publishes_one_seed_offer_and_pins_it_past_cache_eviction() -> None:
    frames = iter((_frame(5), _frame(10), _frame(20)))
    offers: list[object] = []
    lock = threading.RLock()
    adapter = SimpleNamespace(
        _state_lock=lock,
        _cache=OrderedDict(),
        _seed_offer_armed=True,
        _pending_seed_request=_seed_request(),
        _seed_request_deadline_steady_s=2.0,
        _seed_offer=None,
        _generation=4,
        _accept_frames=False,
        _last_sync_stamp_ns=None,
        _last_sync_ros_s=None,
        _make_frame=lambda *_messages: next(frames),
        _now_s=lambda: 1.0,
        _steady_now_s=lambda: 1.0,
        get_parameter=_parameter_reader({'frame_cache_size': 1}),
        _publish_seed_offer=offers.append,
        _publish_seed_status=lambda *_args, **_kwargs: None,
        _fail_if_armed=lambda _reason: None,
        _fail_closed=lambda _reason, **_kwargs: None,
    )
    EdgeTamAdapter._synchronized_cb(adapter, object(), object(), object())
    EdgeTamAdapter._synchronized_cb(adapter, object(), object(), object())
    EdgeTamAdapter._synchronized_cb(adapter, object(), object(), object())

    assert len(offers) == 1
    assert list(adapter._cache) == [20]
    assert adapter._seed_offer is offers[0]
    assert adapter._seed_offer.frame.stamp_ns == 10
    assert adapter._seed_offer.request.nonce == '1' * 32
    assert adapter._seed_offer.adapter_generation == 4
    assert adapter._seed_offer.token.startswith('z-manip-seed:4:')
    assert not adapter._seed_offer_armed


def test_seed_ignores_an_exact_cache_frame_that_was_never_offered() -> None:
    diagnostics: list[tuple[str, str]] = []
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0),
        frame_id='camera_color_optical_frame',
    )
    message = SimpleNamespace(
        header=header,
        detections=[SimpleNamespace(id='seed', header=header)],
    )
    adapter = SimpleNamespace(
        _state_lock=threading.RLock(),
        _seed_offer=None,
        _active_seed_id='active-token',
        _cache=OrderedDict({
            1_000_000_000: CachedRgb(
                stamp_ns=1_000_000_000,
                frame_id=header.frame_id,
                image_jpeg=b'jpeg',
                width=2,
                height=2,
            ),
        }),
        _stamp_ns=EdgeTamAdapter._stamp_ns,
        _ignore_seed_bbox=lambda event, detail, **_kwargs: diagnostics.append(
            (event, detail),
        ),
    )

    EdgeTamAdapter._seed_cb(adapter, message)

    assert diagnostics == [('unmatched_bbox', 'no matching seed offer is outstanding')]
    assert adapter._active_seed_id == 'active-token'


def _seed_request_json(**overrides: object) -> str:
    value = {
        'schema': 'z_manip.seed_request.v1',
        'action': 'arm',
        'request_id': 'request-a',
        'producer_epoch': 'producer-a',
        'grounding_generation': 4,
        'nonce': '1' * 32,
        'source_stamp_floor_ns': 5,
    }
    value.update(overrides)
    import json
    return json.dumps(value)


@pytest.mark.parametrize(
    'payload',
    (
        'not-json',
        _seed_request_json(action='restart'),
        _seed_request_json(nonce='short'),
        _seed_request_json(grounding_generation=True),
        _seed_request_json(extra='field'),
        (
            '{"schema":"z_manip.seed_request.v1","action":"arm",'
            '"action":"cancel","request_id":"request-a",'
            '"producer_epoch":"producer-a","grounding_generation":4,'
            '"nonce":"11111111111111111111111111111111",'
            '"source_stamp_floor_ns":5}'
        ),
    ),
)
def test_seed_request_parser_rejects_ambiguous_envelopes(payload: str) -> None:
    with pytest.raises(ValueError):
        parse_seed_request(payload)


def test_duplicate_seed_request_republishes_offer_without_reset() -> None:
    request = _seed_request()
    offer = SeedOffer(
        request=request,
        adapter_generation=4,
        token='z-manip-seed:4:' + ('d' * 32),
        frame=CachedRgb(10, 'camera_color_optical_frame', b'jpeg', 2, 2),
        deadline_steady_s=10.0,
    )
    replays = []
    statuses = []
    adapter = SimpleNamespace(
        _state_lock=threading.RLock(),
        _seed_owner_producer_epoch=request.producer_epoch,
        _last_seed_request_identity=request.identity,
        _seed_offer=offer,
        _generation=4,
        _publish_seed_offer=replays.append,
        _publish_seed_status=lambda event, *_args, **_kwargs: statuses.append(event),
        _reset_session=lambda **_kwargs: pytest.fail('duplicate request reset session'),
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
    )

    EdgeTamAdapter._seed_request_cb(
        adapter,
        SimpleNamespace(data=_seed_request_json()),
    )

    assert replays == [offer]
    assert statuses == ['duplicate_offer_republished']
    assert adapter._generation == 4


def test_first_arm_owns_adapter_and_foreign_epoch_cannot_replace_or_cancel() -> None:
    transitions = []
    statuses = []
    adapter = SimpleNamespace(
        _state_lock=threading.RLock(),
        _seed_owner_producer_epoch=None,
        _last_seed_request_identity=None,
        _seed_offer=None,
        _generation=4,
        _reset_session=lambda **kwargs: transitions.append(kwargs.get('seed_request')),
        _publish_seed_offer=lambda _offer: None,
        _publish_seed_status=lambda event, *_args, **_kwargs: statuses.append(event),
        get_logger=lambda: SimpleNamespace(warn=lambda _message: None),
    )

    EdgeTamAdapter._seed_request_cb(
        adapter,
        SimpleNamespace(data=_seed_request_json(
            action='cancel',
            producer_epoch='producer-old',
        )),
    )
    assert adapter._seed_owner_producer_epoch is None

    EdgeTamAdapter._seed_request_cb(
        adapter,
        SimpleNamespace(data=_seed_request_json()),
    )
    owner_request = transitions[-1]
    assert adapter._seed_owner_producer_epoch == 'producer-a'

    for action in ('arm', 'cancel'):
        EdgeTamAdapter._seed_request_cb(
            adapter,
            SimpleNamespace(data=_seed_request_json(
                action=action,
                producer_epoch='producer-new',
                nonce='2' * 32,
            )),
        )

    assert transitions == [owner_request]
    assert statuses == [
        'unowned_cancel_ignored',
        'armed',
        'foreign_producer_ignored',
        'foreign_producer_ignored',
    ]

    EdgeTamAdapter._seed_request_cb(
        adapter,
        SimpleNamespace(data=_seed_request_json(nonce='3' * 32)),
    )
    assert len(transitions) == 2
    assert transitions[-1].producer_epoch == 'producer-a'
    assert transitions[-1].nonce == '3' * 32


def test_acquisition_queue_continually_replaces_old_pending_frames() -> None:
    adapter = SimpleNamespace(
        _generation=4,
        _tracking=False,
        _commands=deque([Command('init', 4)]),
        _active_seed_id='z-manip-seed:2',
        get_parameter=_parameter_reader({
            'max_acquisition_pending_frames': 1,
            'max_pending_frames': 30,
        }),
    )

    for stamp_ns in (10, 20, 30):
        assert EdgeTamAdapter._enqueue_frame_locked(
            adapter,
            _frame(stamp_ns),
        ) is None

    commands = tuple(adapter._commands)
    assert [command.kind for command in commands] == ['init', 'frame']
    assert commands[-1].frame.stamp_ns == 30


def test_tracking_queue_continually_replaces_old_pending_frames() -> None:
    adapter = SimpleNamespace(
        _generation=4,
        _tracking=True,
        _commands=deque(),
        _active_seed_id='z-manip-seed:2',
        get_parameter=_parameter_reader({
            'max_acquisition_pending_frames': 1,
            'max_pending_frames': 1,
        }),
    )

    for stamp_ns in (10, 20, 30):
        assert EdgeTamAdapter._enqueue_frame_locked(
            adapter,
            _frame(stamp_ns),
        ) is None

    commands = tuple(adapter._commands)
    assert [command.kind for command in commands] == ['frame']
    assert commands[-1].frame.stamp_ns == 30


def test_latest_only_queue_bounds_lag_when_camera_is_faster_than_inference() -> None:
    adapter = SimpleNamespace(
        _generation=4,
        _tracking=True,
        _commands=deque(),
        _active_seed_id='z-manip-seed:2',
        get_parameter=_parameter_reader({
            'max_acquisition_pending_frames': 1,
            'max_pending_frames': 1,
        }),
    )
    camera_period_s = 1.0 / 15.0
    inference_period_s = 1.0 / 12.8
    next_camera_s = 0.0
    completion_lags_s: list[float] = []

    # Ten simulated minutes reproduce the measured 15 Hz camera / 12.8 Hz
    # serialized-inference mismatch without wall-clock sleeps.
    for service_index in range(int(600.0 / inference_period_s)):
        service_start_s = service_index * inference_period_s
        while next_camera_s <= service_start_s:
            stamp_ns = round(next_camera_s * 1e9)
            assert EdgeTamAdapter._enqueue_frame_locked(
                adapter,
                _frame(stamp_ns),
            ) is None
            assert sum(command.kind == 'frame' for command in adapter._commands) <= 1
            next_camera_s += camera_period_s
        if adapter._commands:
            command = adapter._commands.popleft()
            assert command.frame is not None
            completion_lags_s.append(
                service_start_s
                + inference_period_s
                - command.frame.stamp_ns * 1e-9,
            )

    assert len(completion_lags_s) > 7_000
    assert max(completion_lags_s) < 0.15


def test_observation_manifest_precedes_large_pointcloud_publication() -> None:
    events: list[str] = []

    def publisher(name: str) -> SimpleNamespace:
        return SimpleNamespace(publish=lambda _message: events.append(name))

    adapter = SimpleNamespace(
        _detections_pub=publisher('detections'),
        _target_pub=publisher('target'),
        _frame_manifest_pub=publisher('manifest'),
        _cloud_pub=publisher('cloud'),
        _mask_pub=publisher('mask'),
        _overlay_pub=publisher('overlay'),
        _scene_cloud_pub=publisher('scene_cloud'),
        _publish_tracking=lambda value: events.append(f'tracking:{value}'),
    )
    messages = SimpleNamespace(
        detections=object(),
        target=object(),
        manifest=object(),
        cloud=object(),
        mask=object(),
        overlay=object(),
        scene_cloud=object(),
    )

    EdgeTamAdapter._publish_observation(adapter, messages)

    assert events[:4] == ['detections', 'target', 'manifest', 'cloud']
    assert events[-1] == 'tracking:True'


def test_stale_result_fails_without_counting_toward_acquisition() -> None:
    frame = _frame(1_000_000_000)
    failures: list[tuple[str, dict[str, object]]] = []
    publications: list[int] = []
    tracker = SimpleNamespace(
        update=lambda _frame: SimpleNamespace(stamp_ns=frame.stamp_ns),
        reset=lambda: None,
    )
    adapter = SimpleNamespace(
        _tracker=tracker,
        _state_lock=threading.RLock(),
        _generation=7,
        _accept_frames=True,
        _last_sync_stamp_ns=1_500_000_001,
        _tracking=False,
        _acquisition_gate=AcquisitionGate(3),
        _last_result_ros_s=None,
        _replay_candidate_count=20,
        _replay_selected_count=8,
        _replay_span_ns=9_000_000_000,
        _active_seed_id='z-manip-seed:7',
        get_parameter=_parameter_reader({'max_result_stamp_lag_s': 0.5}),
        _now_s=lambda: 1.0,
        _publish_observation=lambda observation, _frame: publications.append(
            observation.stamp_ns,
        ),
        _fail_closed=lambda reason, **kwargs: failures.append((reason, kwargs)),
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
    )
    adapter._result_lag_failure_locked = lambda stamp_ns: (
        EdgeTamAdapter._result_lag_failure_locked(adapter, stamp_ns)
    )
    command = Command(
        'frame',
        7,
        frame=frame,
        seed_id='z-manip-seed:7',
    )

    EdgeTamAdapter._run_update(adapter, command)

    assert not publications
    assert adapter._acquisition_gate.accepted_updates == 0
    assert len(failures) == 1
    assert failures[0][1]['reason_code'] == 'result_stamp_lag'


def test_pending_mask_anomaly_does_not_publish_or_advance_acquisition() -> None:
    frame = _frame(1_000_000_000)
    failures: list[tuple[str, dict[str, object]]] = []
    publications: list[int] = []
    adapter = SimpleNamespace(
        _tracker=SimpleNamespace(update=lambda _frame: None, reset=lambda: None),
        _state_lock=threading.RLock(),
        _generation=7,
        _accept_frames=True,
        _last_sync_stamp_ns=frame.stamp_ns,
        _tracking=False,
        _acquisition_gate=AcquisitionGate(3),
        _last_result_ros_s=None,
        _active_seed_id='z-manip-seed:7',
        _seed_stamp_ns=500_000_000,
        get_parameter=_parameter_reader({'max_result_stamp_lag_s': 0.5}),
        _now_s=lambda: 1.0,
        _publish_observation=lambda observation, _frame: publications.append(
            observation.stamp_ns,
        ),
        _fail_closed=lambda reason, **kwargs: failures.append((reason, kwargs)),
    )
    adapter._result_lag_failure_locked = lambda stamp_ns: (
        EdgeTamAdapter._result_lag_failure_locked(adapter, stamp_ns)
    )

    EdgeTamAdapter._run_update(
        adapter,
        Command('frame', 7, frame=frame, seed_id='z-manip-seed:7'),
    )

    assert not publications
    assert not failures
    assert adapter._acquisition_gate.accepted_updates == 0
    assert adapter._last_result_ros_s == 1.0


def test_sensor_stamp_advance_during_build_blocks_stale_commit() -> None:
    frame = _frame(1_000_000_000)
    build_started = threading.Event()
    allow_build = threading.Event()
    publications: list[str] = []
    failures: list[tuple[str, dict[str, object]]] = []

    def build(_observation, _frame, _token):
        build_started.set()
        assert allow_build.wait(timeout=2.0)
        return object()

    adapter = SimpleNamespace(
        _tracker=SimpleNamespace(
            update=lambda _frame: SimpleNamespace(stamp_ns=frame.stamp_ns),
            reset=lambda: None,
        ),
        _state_lock=threading.RLock(),
        _generation=8,
        _accept_frames=True,
        _last_sync_stamp_ns=frame.stamp_ns,
        _tracking=False,
        _acquisition_gate=AcquisitionGate(1),
        _last_result_ros_s=None,
        _replay_candidate_count=20,
        _replay_selected_count=8,
        _replay_span_ns=9_000_000_000,
        _active_seed_id='z-manip-seed:8',
        _seed_stamp_ns=500_000_000,
        get_parameter=_parameter_reader({'max_result_stamp_lag_s': 0.5}),
        _now_s=lambda: 1.0,
        _build_observation_messages=build,
        _publish_observation=lambda _messages: publications.append('true'),
        _fail_closed=lambda reason, **kwargs: failures.append((reason, kwargs)),
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
    )
    adapter._result_lag_failure_locked = lambda stamp_ns: (
        EdgeTamAdapter._result_lag_failure_locked(adapter, stamp_ns)
    )
    command = Command(
        'frame',
        8,
        frame=frame,
        seed_id='z-manip-seed:8',
    )
    update_thread = threading.Thread(
        target=EdgeTamAdapter._run_update,
        args=(adapter, command),
    )

    update_thread.start()
    assert build_started.wait(timeout=2.0)
    with adapter._state_lock:
        adapter._last_sync_stamp_ns = 1_500_000_001
    allow_build.set()
    update_thread.join(timeout=2.0)

    assert not update_thread.is_alive()
    assert not publications
    assert len(failures) == 1
    assert failures[0][1]['reason_code'] == 'result_stamp_lag'


@pytest.mark.parametrize(
    ('last_sync_ros_s', 'last_result_ros_s'),
    [
        (6.0, 4.9),
        (4.9, 6.0),
    ],
)
def test_watchdog_fails_closed_on_ros_time_rollback(
    last_sync_ros_s: float,
    last_result_ros_s: float,
) -> None:
    failures: list[tuple[str, dict[str, object]]] = []
    adapter = SimpleNamespace(
        _now_s=lambda: 5.0,
        _steady_now_s=lambda: 5.0,
        _state_lock=threading.RLock(),
        _accept_frames=True,
        _generation=12,
        _pending_seed_request=None,
        _seed_offer=None,
        _seed_request_started_steady_s=1.0,
        _seed_request_deadline_steady_s=10.0,
        _last_sync_ros_s=last_sync_ros_s,
        _last_result_ros_s=last_result_ros_s,
        _tracking_started_ros_s=4.0,
        get_parameter=_parameter_reader({
            'sync_timeout_s': 0.5,
            'result_timeout_s': 11.0,
        }),
        _fail_closed=lambda reason, **kwargs: failures.append((reason, kwargs)),
    )

    EdgeTamAdapter._watchdog_cb(adapter)

    assert len(failures) == 1
    assert failures[0][1] == {
        'generation': 12,
        'reason_code': 'clock_rollback',
    }


def test_synchronized_callback_guard_never_wedges_message_filter_lock() -> None:
    errors: list[str] = []
    failures: list[tuple[str, dict[str, object]]] = []

    def fail_callback(*_messages: object) -> None:
        raise ValueError('bad frame')

    adapter = SimpleNamespace(
        _synchronized_cb=fail_callback,
        get_logger=lambda: SimpleNamespace(error=errors.append),
        _fail_closed=lambda reason, **kwargs: failures.append((reason, kwargs)),
    )

    EdgeTamAdapter._synchronized_cb_guarded(adapter, object(), object(), object())

    assert errors and 'ValueError' in errors[0]
    assert failures == [(
        'exact-time RGB-D callback failed (ValueError)',
        {'reason_code': 'rgbd_callback_exception'},
    )]


def test_synchronized_callback_guard_swallows_secondary_failure() -> None:
    adapter = SimpleNamespace(
        _synchronized_cb=lambda *_messages: (_ for _ in ()).throw(ValueError()),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
        _fail_closed=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )

    EdgeTamAdapter._synchronized_cb_guarded(adapter, object(), object(), object())


def test_synchronized_enqueue_records_receipt_and_bounds_backlog() -> None:
    lock = threading.RLock()
    condition = threading.Condition()
    failures: list[str] = []
    adapter = SimpleNamespace(
        _state_lock=lock,
        _last_sync_ros_s=None,
        _rgbd_condition=condition,
        _rgbd_messages=deque(),
        _stop_rgbd_worker=False,
        _now_s=lambda: 8.5,
        get_parameter=_parameter_reader({'sync_processing_queue_size': 2}),
        _fail_if_armed=failures.append,
    )

    for stamp_ns in (1, 2, 3):
        messages = tuple(_stamped_message(stamp_ns) for _ in range(3))
        EdgeTamAdapter._enqueue_synchronized_cb(adapter, *messages)

    assert adapter._last_sync_ros_s == 8.5
    assert len(adapter._rgbd_messages) == 2
    assert adapter._rgbd_messages[0][0].header.stamp.nanosec == 2
    assert adapter._rgbd_messages[1][0].header.stamp.nanosec == 3
    assert not failures


def test_synchronized_enqueue_rejects_mismatched_stamps() -> None:
    failures: list[str] = []
    adapter = SimpleNamespace(
        _state_lock=threading.RLock(),
        _last_sync_ros_s=None,
        _rgbd_condition=threading.Condition(),
        _rgbd_messages=deque(),
        _stop_rgbd_worker=False,
        _now_s=lambda: 8.5,
        get_parameter=_parameter_reader({'sync_processing_queue_size': 2}),
        _fail_if_armed=failures.append,
    )

    EdgeTamAdapter._enqueue_synchronized_cb(
        adapter,
        _stamped_message(1),
        _stamped_message(1),
        _stamped_message(2),
    )

    assert adapter._last_sync_ros_s is None
    assert not adapter._rgbd_messages
    assert failures == ['exact-time RGB-D enqueue received invalid stamps']


def test_rgbd_worker_processes_only_freshest_backlogged_frame() -> None:
    condition = threading.Condition()
    received: list[tuple[object, ...]] = []
    older = tuple(_stamped_message(1) for _ in range(3))
    newest = tuple(_stamped_message(2) for _ in range(3))
    adapter = SimpleNamespace(
        _rgbd_condition=condition,
        _rgbd_messages=deque((older, newest)),
        _stop_rgbd_worker=False,
    )

    def process(*messages: object) -> None:
        received.append(messages)
        with condition:
            adapter._stop_rgbd_worker = True
            condition.notify_all()

    adapter._synchronized_cb_guarded = process

    EdgeTamAdapter._rgbd_worker_loop(adapter)

    assert received == [newest]
    assert not adapter._rgbd_messages


def test_recreate_rgbd_subscriptions_replaces_stale_readers() -> None:
    destroyed: list[object] = []
    created: list[bool] = []
    warnings: list[str] = []
    old_readers = tuple(
        SimpleNamespace(sub=object())
        for _ in range(3)
    )
    adapter = SimpleNamespace(
        _sync_subscribers=old_readers,
        _synchronizer=object(),
        _rgbd_condition=threading.Condition(),
        _rgbd_messages=deque(((object(), object(), object()),)),
        _state_lock=threading.RLock(),
        _last_sync_ros_s=5.0,
        _depth_filter=SimpleNamespace(reset=lambda: None),
        destroy_subscription=lambda subscription: destroyed.append(subscription),
        _create_rgbd_subscriptions=lambda: created.append(True),
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
    )

    EdgeTamAdapter._recreate_rgbd_subscriptions(adapter)

    assert destroyed == [reader.sub for reader in old_readers]
    assert created == [True]
    assert adapter._sync_subscribers == ()
    assert adapter._synchronizer is None
    assert not adapter._rgbd_messages
    assert adapter._last_sync_ros_s is None
    assert warnings == [
        'recreated RGB-D DDS readers after synchronization timeout',
    ]


def test_watchdog_recreates_rgbd_readers_before_sync_timeout_failure() -> None:
    events: list[str] = []
    adapter = SimpleNamespace(
        _now_s=lambda: 5.0,
        _steady_now_s=lambda: 5.0,
        _state_lock=threading.RLock(),
        _accept_frames=True,
        _generation=12,
        _pending_seed_request=None,
        _seed_offer=None,
        _seed_request_started_steady_s=1.0,
        _seed_request_deadline_steady_s=10.0,
        _last_sync_ros_s=4.0,
        _last_result_ros_s=4.9,
        _tracking_started_ros_s=4.0,
        get_parameter=_parameter_reader({
            'sync_timeout_s': 0.5,
            'result_timeout_s': 11.0,
        }),
        _recreate_rgbd_subscriptions=lambda: events.append('recreate'),
        _fail_closed=lambda _reason, **_kwargs: events.append('fail_closed'),
        get_logger=lambda: SimpleNamespace(error=lambda _message: None),
    )

    EdgeTamAdapter._watchdog_cb(adapter)

    assert events == ['recreate', 'fail_closed']


def test_reset_cannot_overtake_inflight_generation_publication() -> None:
    frame = _frame(2_000_000_000)
    events: list[str] = []
    publish_started = threading.Event()
    allow_publish = threading.Event()
    reset_attempted = threading.Event()
    reset_done = threading.Event()

    def publish(_messages: object) -> None:
        events.append('publish_started')
        publish_started.set()
        assert allow_publish.wait(timeout=2.0)
        events.append('true')

    adapter = SimpleNamespace(
        _tracker=SimpleNamespace(
            update=lambda _frame: SimpleNamespace(stamp_ns=frame.stamp_ns),
            reset=lambda: None,
        ),
        _state_lock=threading.RLock(),
        _generation=9,
        _accept_frames=True,
        _last_sync_stamp_ns=frame.stamp_ns,
        _tracking=False,
        _acquisition_gate=AcquisitionGate(1),
        _last_result_ros_s=None,
        _replay_candidate_count=30,
        _replay_selected_count=8,
        _replay_span_ns=10_000_000_000,
        _active_seed_id='z-manip-seed:9',
        _seed_stamp_ns=1_000_000_000,
        get_parameter=_parameter_reader({'max_result_stamp_lag_s': 0.5}),
        _now_s=lambda: 2.0,
        _build_observation_messages=lambda _observation, _frame, _token: object(),
        _publish_observation=publish,
        _fail_closed=lambda _reason, **_kwargs: None,
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
    )
    adapter._result_lag_failure_locked = lambda stamp_ns: (
        EdgeTamAdapter._result_lag_failure_locked(adapter, stamp_ns)
    )
    command = Command(
        'frame',
        9,
        frame=frame,
        seed_id='z-manip-seed:9',
    )

    update_thread = threading.Thread(
        target=EdgeTamAdapter._run_update,
        args=(adapter, command),
    )

    def reset_generation() -> None:
        reset_attempted.set()
        with adapter._state_lock:
            adapter._generation += 1
        events.append('false')
        reset_done.set()

    update_thread.start()
    assert publish_started.wait(timeout=2.0)
    reset_thread = threading.Thread(target=reset_generation)
    reset_thread.start()
    assert reset_attempted.wait(timeout=2.0)
    assert not reset_done.is_set()

    allow_publish.set()
    update_thread.join(timeout=2.0)
    reset_thread.join(timeout=2.0)

    assert not update_thread.is_alive()
    assert not reset_thread.is_alive()
    assert events == ['publish_started', 'true', 'false']


def _transition_adapter(publish_tracking, publish_observation) -> SimpleNamespace:
    lock = threading.RLock()
    frame = _frame(2_000_000_000)
    adapter = SimpleNamespace(
        _tracker=SimpleNamespace(
            update=lambda _frame: SimpleNamespace(stamp_ns=frame.stamp_ns),
            reset=lambda: None,
        ),
        _state_lock=lock,
        _worker_condition=threading.Condition(lock),
        _commands=deque(),
        _cache=OrderedDict(),
        _seed_offer_armed=False,
        _pending_seed_request=None,
        _seed_owner_producer_epoch='producer-a',
        _last_seed_request_identity=None,
        _seed_request_started_steady_s=1.0,
        _seed_request_deadline_steady_s=10.0,
        _seed_offer=None,
        _generation=1,
        _accept_frames=True,
        _tracking=False,
        _active_seed_id='old-seed',
        _acquisition_gate=AcquisitionGate(1),
        _replay_candidate_count=0,
        _replay_selected_count=0,
        _replay_span_ns=0,
        _seed_stamp_ns=1_000_000_000,
        _last_sync_stamp_ns=frame.stamp_ns,
        _tracking_started_ros_s=1.0,
        _last_result_ros_s=None,
        _publish_tracking=publish_tracking,
        _publish_observation=publish_observation,
        _failure_pub=SimpleNamespace(publish=lambda _message: None),
        _now_s=lambda: 2.0,
        _steady_now_s=lambda: 2.0,
        _stamp_ns=EdgeTamAdapter._stamp_ns,
        _build_observation_messages=lambda _observation, _frame, _token: object(),
        get_parameter=_parameter_reader({
            'max_result_stamp_lag_s': 0.5,
        }),
        get_logger=lambda: SimpleNamespace(
            info=lambda _message: None,
            warn=lambda _message: None,
            error=lambda _message: None,
        ),
    )
    adapter._result_lag_failure_locked = lambda stamp_ns: (
        EdgeTamAdapter._result_lag_failure_locked(adapter, stamp_ns)
    )
    adapter._seed_offer_state_locked = lambda offer: (
        EdgeTamAdapter._seed_offer_state_locked(adapter, offer)
    )
    adapter._reject_noncurrent_seed_offer = lambda offer, state, **kwargs: (
        EdgeTamAdapter._reject_noncurrent_seed_offer(
            adapter,
            offer,
            state,
            **kwargs,
        )
    )
    return adapter


@pytest.mark.parametrize(
    ('seed_id', 'expected_event'),
    (('old-seed', 'duplicate_bbox'), ('stale-token', 'unmatched_bbox')),
)
def test_stale_or_duplicate_bbox_cannot_stop_active_tracking(
    seed_id: str,
    expected_event: str,
) -> None:
    tracking_events: list[bool] = []
    diagnostics: list[str] = []
    adapter = _transition_adapter(tracking_events.append, lambda _messages: None)
    adapter._tracking = True
    adapter._ignore_seed_bbox = (
        lambda event, _detail, **_kwargs: diagnostics.append(event)
    )
    generation = adapter._generation

    EdgeTamAdapter._seed_cb(
        adapter,
        SimpleNamespace(detections=[SimpleNamespace(id=seed_id)]),
    )

    assert diagnostics == [expected_event]
    assert adapter._generation == generation
    assert adapter._tracking
    assert adapter._accept_frames
    assert adapter._active_seed_id == 'old-seed'
    assert tracking_events == []


def _matching_seed_message(token: str, *, stamp_ns: int = 1_000_000_000):
    header = SimpleNamespace(
        stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        ),
        frame_id='camera_color_optical_frame',
    )
    detection = SimpleNamespace(
        id=token,
        header=header,
        bbox=SimpleNamespace(
            center=SimpleNamespace(
                theta=0.0,
                position=SimpleNamespace(x=1.0, y=1.0),
            ),
            size_x=1.0,
            size_y=1.0,
        ),
        results=[],
    )
    return SimpleNamespace(header=header, detections=[detection])


def test_matching_malformed_bbox_keeps_offer_pinned_for_retry() -> None:
    adapter = _transition_adapter(lambda _value: None, lambda _messages: None)
    adapter._accept_frames = False
    request = _seed_request(generation=adapter._generation)
    token = 'z-manip-seed:1:' + ('e' * 32)
    frame = CachedRgb(
        stamp_ns=1_000_000_000,
        frame_id='camera_color_optical_frame',
        image_jpeg=b'jpeg',
        width=2,
        height=2,
    )
    offer = SeedOffer(request, adapter._generation, token, frame, 10.0)
    adapter._pending_seed_request = request
    adapter._seed_offer = offer
    diagnostics: list[str] = []
    adapter._ignore_seed_bbox = (
        lambda event, _detail, **_kwargs: diagnostics.append(event)
    )
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0),
        frame_id=frame.frame_id,
    )
    detection = SimpleNamespace(
        id=token,
        header=header,
        bbox=SimpleNamespace(center=SimpleNamespace(theta=1.0)),
    )

    EdgeTamAdapter._seed_cb(
        adapter,
        SimpleNamespace(header=header, detections=[detection]),
    )

    assert diagnostics == ['invalid_bbox']
    assert adapter._seed_offer is offer
    assert adapter._pending_seed_request is request


def test_bbox_already_past_steady_deadline_releases_pin_without_init() -> None:
    statuses: list[str] = []
    adapter = _transition_adapter(lambda _value: None, lambda _messages: None)
    adapter._accept_frames = False
    adapter._steady_now_s = lambda: 10.1
    adapter._publish_seed_status = (
        lambda event, *_args, **_kwargs: statuses.append(event)
    )
    adapter._reset_session = (
        lambda **kwargs: EdgeTamAdapter._reset_session(adapter, **kwargs)
    )
    request = _seed_request(generation=adapter._generation)
    token = 'z-manip-seed:1:' + ('6' * 32)
    frame = CachedRgb(
        1_000_000_000,
        'camera_color_optical_frame',
        b'jpeg',
        2,
        2,
    )
    adapter._pending_seed_request = request
    adapter._seed_offer = SeedOffer(
        request,
        adapter._generation,
        token,
        frame,
        10.0,
    )
    adapter._cache[frame.stamp_ns] = frame

    EdgeTamAdapter._seed_cb(adapter, _matching_seed_message(token))

    assert statuses == ['expired']
    assert adapter._seed_offer is None
    assert adapter._pending_seed_request is None
    assert [command.kind for command in adapter._commands] == ['reset']
    assert not adapter._accept_frames


def test_bbox_crossing_deadline_during_registration_cannot_commit(
    monkeypatch,
) -> None:
    statuses: list[str] = []
    registration_calls = []
    steady_times = iter((2.0, 10.1))
    adapter = _transition_adapter(lambda _value: None, lambda _messages: None)
    adapter._accept_frames = False
    adapter._steady_now_s = lambda: next(steady_times)
    adapter._publish_seed_status = (
        lambda event, *_args, **_kwargs: statuses.append(event)
    )
    adapter._reset_session = (
        lambda **kwargs: EdgeTamAdapter._reset_session(adapter, **kwargs)
    )
    request = _seed_request(generation=adapter._generation)
    token = 'z-manip-seed:1:' + ('7' * 32)
    seed = CachedRgb(
        1_000_000_000,
        'camera_color_optical_frame',
        b'seed',
        2,
        2,
    )
    latest = CachedRgb(
        1_100_000_000,
        seed.frame_id,
        b'latest',
        2,
        2,
    )
    adapter._pending_seed_request = request
    adapter._seed_offer = SeedOffer(
        request,
        adapter._generation,
        token,
        seed,
        10.0,
    )
    adapter._cache[seed.stamp_ns] = seed
    adapter._cache[latest.stamp_ns] = latest
    adapter._reseed_registration_config = object()

    def register(*_args, **_kwargs):
        registration_calls.append(True)
        return SimpleNamespace(bbox_xyxy=(0, 0, 1, 1))

    monkeypatch.setattr(node_module, 'register_seed_bbox_to_latest', register)

    EdgeTamAdapter._seed_cb(adapter, _matching_seed_message(token))

    assert registration_calls == [True]
    assert statuses == ['expired']
    assert [command.kind for command in adapter._commands] == ['reset']
    assert not adapter._accept_frames


def test_short_span_featureless_reseed_falls_back_to_exact_offered_frame(
    monkeypatch,
) -> None:
    failures: list[tuple[str, dict[str, object]]] = []
    tracking_events: list[bool] = []
    adapter = _transition_adapter(tracking_events.append, lambda _messages: None)
    adapter._accept_frames = False
    adapter._allow_short_span_seed_fallback = True
    adapter._short_span_seed_fallback_max_s = 0.15
    adapter._short_span_seed_fallback_max_frames = 1
    adapter._reseed_registration_config = object()
    adapter._fail_closed = lambda reason, **kwargs: failures.append((reason, kwargs))
    request = _seed_request(generation=adapter._generation)
    token = 'z-manip-seed:1:' + ('d' * 32)
    seed = CachedRgb(
        1_000_000_000,
        'camera_color_optical_frame',
        b'featureless-seed',
        2,
        2,
    )
    latest = CachedRgb(
        1_067_000_000,
        seed.frame_id,
        b'featureless-latest',
        2,
        2,
    )
    adapter._pending_seed_request = request
    adapter._seed_offer = SeedOffer(
        request,
        adapter._generation,
        token,
        seed,
        10.0,
    )
    adapter._cache[seed.stamp_ns] = seed
    adapter._cache[latest.stamp_ns] = latest

    def reject_registration(*_args, **_kwargs):
        raise node_module.TrackerFailure(
            'global reseed has too few seed features (0 < 20)',
            reason_code='seed_reseed_registration',
        )

    monkeypatch.setattr(node_module, 'register_seed_bbox_to_latest', reject_registration)

    EdgeTamAdapter._seed_cb(adapter, _matching_seed_message(token))

    assert failures == []
    assert tracking_events == [False]
    assert adapter._accept_frames
    assert adapter._replay_candidate_count == 1
    assert adapter._replay_selected_count == 0
    assert adapter._replay_span_ns == 67_000_000
    assert len(adapter._commands) == 1
    command = adapter._commands[0]
    assert command.kind == 'init'
    assert command.frame is seed


def test_short_span_reseed_fallback_remains_bounded(monkeypatch) -> None:
    failures: list[tuple[str, dict[str, object]]] = []
    adapter = _transition_adapter(lambda _value: None, lambda _messages: None)
    adapter._accept_frames = False
    adapter._allow_short_span_seed_fallback = True
    adapter._short_span_seed_fallback_max_s = 0.15
    adapter._short_span_seed_fallback_max_frames = 1
    adapter._reseed_registration_config = object()
    adapter._fail_closed = lambda reason, **kwargs: failures.append((reason, kwargs))
    request = _seed_request(generation=adapter._generation)
    token = 'z-manip-seed:1:' + ('0' * 32)
    seed = CachedRgb(1_000_000_000, 'camera_color_optical_frame', b'seed', 2, 2)
    latest = CachedRgb(1_300_000_000, seed.frame_id, b'latest', 2, 2)
    adapter._pending_seed_request = request
    adapter._seed_offer = SeedOffer(request, adapter._generation, token, seed, 10.0)
    adapter._cache[seed.stamp_ns] = seed
    adapter._cache[latest.stamp_ns] = latest

    monkeypatch.setattr(
        node_module,
        'register_seed_bbox_to_latest',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            node_module.TrackerFailure(
                'featureless',
                reason_code='seed_reseed_registration',
            ),
        ),
    )

    EdgeTamAdapter._seed_cb(adapter, _matching_seed_message(token))

    assert len(failures) == 1
    assert failures[0][1]['reason_code'] == 'seed_reseed_registration'
    assert adapter._commands == deque()


def test_seed_offer_replacement_during_bbox_validation_is_ignored() -> None:
    adapter = _transition_adapter(lambda _value: None, lambda _messages: None)
    adapter._accept_frames = False
    request = _seed_request(generation=adapter._generation)
    token = 'z-manip-seed:1:' + ('8' * 32)
    frame = CachedRgb(
        stamp_ns=1_000_000_000,
        frame_id='camera_color_optical_frame',
        image_jpeg=b'jpeg',
        width=2,
        height=2,
    )
    offer = SeedOffer(request, adapter._generation, token, frame, 10.0)
    adapter._pending_seed_request = request
    adapter._seed_offer = offer
    diagnostics: list[str] = []
    adapter._ignore_seed_bbox = (
        lambda event, _detail, **_kwargs: diagnostics.append(event)
    )
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0),
        frame_id=frame.frame_id,
    )
    detection = SimpleNamespace(
        id=token,
        header=header,
        bbox=SimpleNamespace(
            center=SimpleNamespace(
                theta=0.0,
                position=SimpleNamespace(x=1.0, y=1.0),
            ),
            size_x=1.0,
            size_y=1.0,
        ),
        results=[],
    )
    stamp_calls = 0

    def racing_stamp(message_header) -> int:
        nonlocal stamp_calls
        stamp_calls += 1
        stamp_ns = EdgeTamAdapter._stamp_ns(message_header)
        if stamp_calls == 2:
            with adapter._state_lock:
                adapter._generation += 1
                adapter._seed_offer = None
                adapter._pending_seed_request = None
        return stamp_ns

    adapter._stamp_ns = racing_stamp

    EdgeTamAdapter._seed_cb(
        adapter,
        SimpleNamespace(header=header, detections=[detection]),
    )

    assert diagnostics == ['stale_bbox']
    assert adapter._commands == deque()
    assert not adapter._accept_frames


def test_seed_watchdog_releases_pin_while_ros_clock_is_frozen() -> None:
    adapter = _transition_adapter(lambda _value: None, lambda _messages: None)
    adapter._accept_frames = False
    request = _seed_request(generation=adapter._generation)
    offer = SeedOffer(
        request,
        adapter._generation,
        'z-manip-seed:1:' + ('f' * 32),
        CachedRgb(10, 'camera_color_optical_frame', b'jpeg', 2, 2),
        10.0,
    )
    adapter._pending_seed_request = request
    adapter._seed_offer = offer
    adapter._seed_offer_armed = False
    adapter._seed_request_started_steady_s = 1.0
    adapter._seed_request_deadline_steady_s = 10.0
    adapter._now_s = lambda: 4.0
    adapter._steady_now_s = lambda: 11.0
    statuses: list[str] = []
    adapter._publish_seed_status = (
        lambda event, *_args, **_kwargs: statuses.append(event)
    )
    adapter._reset_session = (
        lambda **kwargs: EdgeTamAdapter._reset_session(adapter, **kwargs)
    )

    EdgeTamAdapter._watchdog_cb(adapter)

    assert statuses == ['expired']
    assert adapter._seed_offer is None
    assert adapter._pending_seed_request is None
    assert not adapter._seed_offer_armed
    assert adapter._seed_request_started_steady_s is None
    assert adapter._seed_request_deadline_steady_s is None
    assert adapter._seed_owner_producer_epoch == 'producer-a'


def _assert_false_transition_precedes_new_true(
    adapter: SimpleNamespace,
    transition,
    *,
    prepare_new_generation: bool,
    events: list[str],
    false_started: threading.Event,
    allow_false: threading.Event,
) -> None:
    frame = _frame(2_000_000_000)
    transition_thread = threading.Thread(target=transition)
    transition_thread.start()
    assert false_started.wait(timeout=2.0)
    acquired = adapter._state_lock.acquire(blocking=False)
    if acquired:
        adapter._state_lock.release()
    assert not acquired

    def publish_new_generation() -> None:
        if prepare_new_generation:
            with adapter._state_lock:
                adapter._generation += 1
                adapter._accept_frames = True
                adapter._tracking = False
                adapter._active_seed_id = 'new-seed'
                adapter._seed_stamp_ns = 1_500_000_000
                adapter._acquisition_gate.reset()
        command = Command(
            'frame',
            adapter._generation,
            frame=frame,
            seed_id=adapter._active_seed_id,
        )
        EdgeTamAdapter._run_update(adapter, command)

    update_thread = threading.Thread(target=publish_new_generation)
    update_thread.start()
    assert 'true' not in events

    allow_false.set()
    transition_thread.join(timeout=2.0)
    update_thread.join(timeout=2.0)

    assert not transition_thread.is_alive()
    assert not update_thread.is_alive()
    assert events == ['false_started', 'false', 'true']


def _blocking_false_publisher(
    events: list[str],
    false_started: threading.Event,
    allow_false: threading.Event,
):
    def publish(value: bool) -> None:
        assert value is False
        events.append('false_started')
        false_started.set()
        assert allow_false.wait(timeout=2.0)
        events.append('false')

    return publish


def test_seed_false_cannot_land_after_same_generation_true() -> None:
    events: list[str] = []
    false_started = threading.Event()
    allow_false = threading.Event()
    adapter = _transition_adapter(
        _blocking_false_publisher(events, false_started, allow_false),
        lambda _messages: events.append('true'),
    )
    seed_stamp_ns = 1_000_000_000
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0),
        frame_id='camera_color_optical_frame',
    )
    request = _seed_request(generation=adapter._generation)
    token = 'z-manip-seed:1:' + ('a' * 32)
    detection = SimpleNamespace(
        id=token,
        header=header,
        bbox=SimpleNamespace(
            center=SimpleNamespace(
                theta=0.0,
                position=SimpleNamespace(x=1.0, y=1.0),
            ),
            size_x=1.0,
            size_y=1.0,
        ),
        results=[],
    )
    message = SimpleNamespace(header=header, detections=[detection])
    offered = CachedRgb(
        stamp_ns=seed_stamp_ns,
        frame_id=header.frame_id,
        image_jpeg=b'\xff\xd8test\xff\xd9',
        width=2,
        height=2,
    )
    adapter._cache[seed_stamp_ns] = offered
    adapter._pending_seed_request = request
    adapter._seed_offer = SeedOffer(
        request=request,
        adapter_generation=adapter._generation,
        token=token,
        frame=offered,
        deadline_steady_s=10.0,
    )

    _assert_false_transition_precedes_new_true(
        adapter,
        lambda: EdgeTamAdapter._seed_cb(adapter, message),
        prepare_new_generation=False,
        events=events,
        false_started=false_started,
        allow_false=allow_false,
    )


def test_seed_callback_reseeds_latest_frame_across_fixed_133_frame_backlog() -> None:
    failures: list[tuple[str, dict[str, object]]] = []
    tracking_events: list[bool] = []
    seed_stamp_ns = 1_000_000_000
    rng = np.random.default_rng(23)
    seed_image = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    transform = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    latest_image = cv2.warpAffine(
        seed_image,
        transform,
        (seed_image.shape[1], seed_image.shape[0]),
        borderMode=cv2.BORDER_REFLECT101,
    )

    def jpeg(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            '.jpg',
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        assert ok
        return encoded.tobytes()

    seed_jpeg = jpeg(seed_image)
    latest_jpeg = jpeg(latest_image)
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=1, nanosec=0),
        frame_id='camera_color_optical_frame',
    )
    request = _seed_request(generation=1)
    token = 'z-manip-seed:1:' + ('b' * 32)
    detection = SimpleNamespace(
        id=token,
        header=header,
        bbox=SimpleNamespace(
            center=SimpleNamespace(
                theta=0.0,
                position=SimpleNamespace(x=120.0, y=92.0),
            ),
            size_x=56.0,
            size_y=68.0,
        ),
        results=[],
    )
    cache = OrderedDict()
    cache[seed_stamp_ns] = CachedRgb(
        stamp_ns=seed_stamp_ns,
        frame_id=header.frame_id,
        image_jpeg=seed_jpeg,
        width=240,
        height=180,
    )
    for index in range(133):
        stamp_ns = seed_stamp_ns + (index + 1) * 100_000_000
        cache[stamp_ns] = CachedRgb(
            stamp_ns=stamp_ns,
            frame_id=header.frame_id,
            image_jpeg=latest_jpeg,
            width=240,
            height=180,
        )
    lock = threading.RLock()
    adapter = SimpleNamespace(
        _state_lock=lock,
        _worker_condition=threading.Condition(lock),
        _commands=deque(),
        _cache=cache,
        _seed_offer_armed=False,
        _pending_seed_request=request,
        _seed_request_started_steady_s=1.0,
        _seed_request_deadline_steady_s=10.0,
        _seed_offer=SeedOffer(
            request=request,
            adapter_generation=1,
            token=token,
            frame=cache[seed_stamp_ns],
            deadline_steady_s=10.0,
        ),
        _generation=1,
        _accept_frames=False,
        _tracking=False,
        _active_seed_id='',
        _acquisition_gate=AcquisitionGate(3),
        _replay_candidate_count=0,
        _replay_selected_count=0,
        _replay_span_ns=0,
        _seed_stamp_ns=None,
        _tracking_started_ros_s=None,
        _last_result_ros_s=None,
        _reseed_registration_config=ReseedRegistrationConfig(
            min_global_tracks=16,
            min_global_inliers=12,
            min_roi_tracks=6,
            min_roi_inliers=5,
        ),
        _stamp_ns=EdgeTamAdapter._stamp_ns,
        _now_s=lambda: 2.0,
        _steady_now_s=lambda: 2.0,
        _publish_tracking=tracking_events.append,
        _fail_closed=lambda reason, **kwargs: failures.append((reason, kwargs)),
        _publish_seed_status=lambda *_args, **_kwargs: None,
        get_logger=lambda: SimpleNamespace(
            info=lambda _message: None,
            warn=lambda _message: None,
        ),
    )
    adapter._seed_offer_state_locked = lambda offer: (
        EdgeTamAdapter._seed_offer_state_locked(adapter, offer)
    )

    EdgeTamAdapter._seed_cb(
        adapter,
        SimpleNamespace(header=header, detections=[detection]),
    )

    assert failures == []
    assert tracking_events == [False]
    assert adapter._replay_candidate_count == 133
    assert adapter._replay_selected_count == 1
    assert adapter._replay_span_ns == 13_300_000_000
    assert adapter._seed_stamp_ns == seed_stamp_ns
    assert adapter._seed_offer is None
    assert len(adapter._commands) == 1
    command = adapter._commands[0]
    assert command.kind == 'init'
    assert command.seed_id == token
    assert command.frame.stamp_ns == seed_stamp_ns + 13_300_000_000
    assert np.allclose(command.bbox_xyxy, (96, 55, 152, 123), atol=1)


def test_reset_false_cannot_land_after_new_generation_true() -> None:
    events: list[str] = []
    false_started = threading.Event()
    allow_false = threading.Event()
    adapter = _transition_adapter(
        _blocking_false_publisher(events, false_started, allow_false),
        lambda _messages: events.append('true'),
    )
    request = _seed_request(generation=adapter._generation)
    adapter._pending_seed_request = request
    adapter._seed_request_started_steady_s = 1.0
    adapter._seed_request_deadline_steady_s = 10.0
    adapter._seed_offer = SeedOffer(
        request=request,
        adapter_generation=adapter._generation,
        token='z-manip-seed:1:' + ('c' * 32),
        frame=CachedRgb(
            stamp_ns=500_000_000,
            frame_id='camera_color_optical_frame',
            image_jpeg=b'old-offer',
            width=2,
            height=2,
        ),
        deadline_steady_s=10.0,
    )

    _assert_false_transition_precedes_new_true(
        adapter,
        lambda: EdgeTamAdapter._reset_session(adapter),
        prepare_new_generation=True,
        events=events,
        false_started=false_started,
        allow_false=allow_false,
    )
    assert adapter._seed_offer is None
    assert not adapter._seed_offer_armed
    assert adapter._pending_seed_request is None
    assert adapter._seed_request_started_steady_s is None
    assert adapter._seed_request_deadline_steady_s is None


def test_fail_closed_false_cannot_land_after_new_generation_true() -> None:
    events: list[str] = []
    false_started = threading.Event()
    allow_false = threading.Event()
    adapter = _transition_adapter(
        _blocking_false_publisher(events, false_started, allow_false),
        lambda _messages: events.append('true'),
    )
    request = _seed_request(generation=adapter._generation)
    adapter._pending_seed_request = request
    adapter._seed_request_started_steady_s = 1.0
    adapter._seed_request_deadline_steady_s = 10.0
    adapter._seed_offer_armed = True
    adapter._seed_offer = SeedOffer(
        request,
        adapter._generation,
        'z-manip-seed:1:' + ('9' * 32),
        CachedRgb(10, 'camera_color_optical_frame', b'jpeg', 2, 2),
        10.0,
    )

    _assert_false_transition_precedes_new_true(
        adapter,
        lambda: EdgeTamAdapter._fail_closed(
            adapter,
            'forced test failure',
            reason_code='test_failure',
        ),
        prepare_new_generation=True,
        events=events,
        false_started=false_started,
        allow_false=allow_false,
    )
    assert adapter._seed_offer is None
    assert adapter._pending_seed_request is None
    assert not adapter._seed_offer_armed
    assert adapter._seed_request_started_steady_s is None
    assert adapter._seed_request_deadline_steady_s is None
