"""ROS-runtime boundary tests for executor transaction parsing."""

from collections import OrderedDict
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip('rclpy')

from builtin_interfaces.msg import Time as TimeMessage  # noqa: E402,I100
from z_manip.trajectory_digest import (  # noqa: E402
    canonical_joint_trajectory_sha256,
)

from z_manip_place.core import (  # noqa: E402
    ObservedPerceptionIdentity,
    PlacementContractError,
)
from z_manip_place.node import (  # noqa: E402
    _ExecutionFeedback,
    ObservedPlacementNode,
)
from z_manip_place.transaction import (  # noqa: E402
    PlacementTransactionLifecycle,
    PlaceTransactionControl,
    PlaceTransactionError,
    PlaceTransactionIdentity,
)


def _image_source(stamp_ns: int, frame_id: str):
    return SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        ),
        frame_id=frame_id,
    ))


def test_rgbd_source_key_requires_exact_stamp_and_frame_for_all_three_inputs():
    color = _image_source(81_230_000_001, 'wrist_camera_optical_frame')
    depth = _image_source(81_230_000_001, 'wrist_camera_optical_frame')
    info = _image_source(81_230_000_001, 'wrist_camera_optical_frame')

    assert ObservedPlacementNode._exact_rgbd_source_key(
        color,
        depth,
        info,
    ) == (81_230_000_001, 'wrist_camera_optical_frame')

    depth.header.stamp.nanosec += 1
    with pytest.raises(PlacementContractError, match='stamps must match exactly'):
        ObservedPlacementNode._exact_rgbd_source_key(color, depth, info)
    depth.header.stamp.nanosec -= 1
    info.header.frame_id = 'stale_camera_optical_frame'
    with pytest.raises(PlacementContractError, match='frames must match exactly'):
        ObservedPlacementNode._exact_rgbd_source_key(color, depth, info)


def test_rgbd_source_key_rejects_zero_stamp_instead_of_caching_timeless_data():
    sources = [_image_source(0, 'camera') for _ in range(3)]
    with pytest.raises(PlacementContractError, match='stamps must match exactly'):
        ObservedPlacementNode._exact_rgbd_source_key(*sources)


def test_planning_sources_share_the_region_request_key_without_latest_fallback():
    identity = object()
    request = SimpleNamespace(
        stamp_ns=81_230_000_001,
        image_frame='wrist_camera_optical_frame',
        observation_identity=identity,
    )
    rgbd = SimpleNamespace(
        rgb_stamp_ns=request.stamp_ns,
        depth_stamp_ns=request.stamp_ns,
        camera_info_stamp_ns=request.stamp_ns,
        source_frame=request.image_frame,
    )
    target = SimpleNamespace(
        stamp_ns=request.stamp_ns,
        source_frame=request.image_frame,
    )

    ObservedPlacementNode._validate_exact_planning_sources(
        request,
        identity,
        rgbd,
        target,
    )

    rgbd.camera_info_stamp_ns -= 1
    with pytest.raises(PlacementContractError, match='exact RGB-D source key'):
        ObservedPlacementNode._validate_exact_planning_sources(
            request,
            identity,
            rgbd,
            target,
        )
    rgbd.camera_info_stamp_ns += 1
    target.stamp_ns -= 1
    with pytest.raises(PlacementContractError, match='target model'):
        ObservedPlacementNode._validate_exact_planning_sources(
            request,
            identity,
            rgbd,
            target,
        )
    target.stamp_ns += 1
    with pytest.raises(PlacementContractError, match='owner is not exact'):
        ObservedPlacementNode._validate_exact_planning_sources(
            request,
            object(),
            rgbd,
            target,
        )


def test_planning_rgbd_cache_is_bounded_and_poisons_same_key_collisions():
    cache = OrderedDict()
    conflicts = OrderedDict()

    def rgbd(stamp_ns, value):
        return SimpleNamespace(
            rgb_stamp_ns=stamp_ns,
            depth_stamp_ns=stamp_ns,
            camera_info_stamp_ns=stamp_ns,
            source_frame='camera',
            organized_points=np.full((2, 2, 3), value, dtype=np.float32),
        )

    first_key = (100, 'camera')
    second_key = (200, 'camera')
    ObservedPlacementNode._cache_exact_planning_rgbd(
        cache,
        conflicts,
        first_key,
        rgbd(100, 1.0),
        limit=1,
    )
    ObservedPlacementNode._cache_exact_planning_rgbd(
        cache,
        conflicts,
        second_key,
        rgbd(200, 2.0),
        limit=1,
    )
    assert tuple(cache) == (second_key,)
    assert first_key not in cache

    with pytest.raises(PlacementContractError, match='multiple RGB-D payloads'):
        ObservedPlacementNode._cache_exact_planning_rgbd(
            cache,
            conflicts,
            second_key,
            rgbd(200, 3.0),
            limit=1,
        )
    assert second_key not in cache
    assert second_key in conflicts

    ObservedPlacementNode._cache_exact_planning_rgbd(
        cache,
        conflicts,
        second_key,
        rgbd(200, 2.0),
        limit=1,
    )
    assert second_key not in cache

    with pytest.raises(PlacementContractError, match='positive integer'):
        ObservedPlacementNode._cache_exact_planning_rgbd(
            cache,
            conflicts,
            first_key,
            rgbd(100, 1.0),
            limit=0,
        )


def test_executor_status_parser_accepts_exact_go2w_transaction_fields():
    status, fields = ObservedPlacementNode._execution_fields(
        'succeeded;owner=trajectory;command_id=4;segment=place_approach;'
        'trajectory_contract_id=place-goal-7;executor_epoch=epoch-a;'
        'trajectory_received_at=81.230000;gripper=accepted:0.0750;'
        'gripper_command_id=2;gripper_received_at=82.110000;aperture=0.0745',
    )

    assert status == 'succeeded'
    assert fields['trajectory_contract_id'] == 'place-goal-7'
    assert fields['executor_epoch'] == 'epoch-a'
    assert ObservedPlacementNode._execution_source_stamp_ns(
        fields,
        'trajectory_received_at',
    ) == 81_230_000_000
    assert ObservedPlacementNode._execution_source_stamp_ns(
        fields,
        'gripper_received_at',
    ) == 82_110_000_000


def test_executor_status_parser_rejects_duplicate_or_nonfinite_source():
    with pytest.raises(PlacementContractError, match='repeats field'):
        ObservedPlacementNode._execution_fields(
            'active;executor_epoch=a;executor_epoch=b',
        )
    with pytest.raises(PlacementContractError, match='finite and nonnegative'):
        ObservedPlacementNode._execution_source_stamp_ns(
            {'trajectory_received_at': 'nan'},
            'trajectory_received_at',
        )


def _completed_carry_execution():
    return _ExecutionFeedback(
        received_ns=8_000_000_000,
        executor_epoch='executor-a',
        trajectory_status='succeeded',
        trajectory_owner='trajectory',
        trajectory_segment='carry',
        trajectory_command_id=7,
        trajectory_source_stamp_ns=7_000_000_000,
        trajectory_contract_id='none',
        gripper_command_id=3,
        gripper_source_stamp_ns=6_500_000_000,
        aperture_m=0.018,
    )


def test_node_abort_resets_only_exact_armed_transaction_and_unblocks_retry():
    from std_msgs.msg import String

    class Resettable:
        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls += 1

    class Harness:
        def __init__(self):
            self._lock = __import__('threading').Lock()
            self._transaction = PlacementTransactionLifecycle(
                ros_timeout_s=3.0,
                wall_timeout_s=5.0,
            )
            token = self._transaction.begin(PlaceTransactionIdentity(
                'place-7',
                'executor-a',
            ), now_ros_ns=10_000_000_000, now_wall_s=20.0)
            self._transaction.start_planning(token)
            self._transaction.arm(
                token,
                now_ros_ns=10_000_000_000,
                now_wall_s=20.0,
            )
            self._request = object()
            self._request_identity = object()
            self._request_transaction_token = token
            self._planning_identity = object()
            self._post_release = Resettable()
            self._place_execution = Resettable()
            self._pending_release = object()
            self._verification_rgbd = {1: 1}
            self._verification_targets = {1: 1}
            self._verification_identities = {1: 1}
            self._verification_identity_conflicts = {1: 1}
            self._verification_joints = {1: 1}
            self.status = []

        def _publish_status(self, state, detail):
            self.status.append((state, detail))

    harness = Harness()
    foreign = {
        'schema': 'z_manip.place_transaction_control.v1',
        'action': 'abort',
        'goal_id': 'place-7',
        'executor_epoch': 'executor-b',
    }
    ObservedPlacementNode._on_transaction_control(
        harness,
        String(data=json.dumps(foreign)),
    )
    assert harness._transaction.armed
    assert harness._post_release.calls == 0

    foreign['executor_epoch'] = 'executor-a'
    ObservedPlacementNode._on_transaction_control(
        harness,
        String(data=json.dumps(foreign)),
    )
    assert not harness._transaction.active
    assert harness._request is None
    assert harness._planning_identity is None
    assert harness._post_release.calls == 1
    assert harness._place_execution.calls == 1
    assert harness.status[-1][0] == 'transaction_aborted'

    retry = harness._transaction.begin(PlaceTransactionIdentity(
        'place-8',
        'executor-a',
    ), now_ros_ns=11_000_000_000, now_wall_s=21.0)
    assert harness._transaction.matches(retry)


@pytest.mark.parametrize('state', ('pending', 'planning'))
def test_steady_watchdog_correlates_and_resets_prearm_timeout(
    state,
    monkeypatch,
):
    class Resettable:
        def reset(self):
            return None

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Harness:
        def __init__(self):
            self._lock = threading.Lock()
            self._transaction = PlacementTransactionLifecycle(
                ros_timeout_s=30.0,
                wall_timeout_s=5.0,
            )
            token = self._transaction.begin(
                PlaceTransactionIdentity('place-7', 'executor-a'),
                now_ros_ns=100,
                now_wall_s=10.0,
            )
            if state == 'planning':
                self._transaction.start_planning(token)
            self._request = object()
            self._request_identity = object()
            self._request_transaction_token = token
            self._planning_identity = None
            self._post_release = Resettable()
            self._place_execution = Resettable()
            self._pending_release = None
            self._verification_rgbd = {}
            self._verification_targets = {}
            self._verification_identities = {}
            self._verification_identity_conflicts = {}
            self._verification_joints = {}
            self._status_publisher = Publisher()

        @staticmethod
        def get_clock():
            return SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=100),
            )

        def _publish_terminal_failure(self, failure):
            ObservedPlacementNode._publish_terminal_failure(self, failure)

    monkeypatch.setattr('z_manip_place.node.time.monotonic', lambda: 15.0)
    harness = Harness()
    ObservedPlacementNode._transaction_watchdog_tick(harness)

    assert not harness._transaction.active
    assert harness._request is None
    assert len(harness._status_publisher.messages) == 1
    payload = json.loads(harness._status_publisher.messages[0].data)
    assert payload['goal_id'] == 'place-7'
    assert payload['executor_epoch'] == 'executor-a'
    assert payload['reason'] == (
        f'placement transaction {state} wall-time deadline exceeded'
    )


def test_aborted_worker_exits_before_retry_without_clearing_new_worker():
    """One canceled generation cannot stack clients or own its retry."""

    class Harness:
        def __init__(self):
            self._lock = threading.Lock()
            self._transaction = PlacementTransactionLifecycle(
                ros_timeout_s=3.0,
                wall_timeout_s=5.0,
            )
            self._scene = object()
            self._joints = object()
            self._execution_feedback = SimpleNamespace(
                executor_epoch='executor-a',
                trajectory_status='succeeded',
                trajectory_owner='trajectory',
                trajectory_segment='carry',
                trajectory_command_id=7,
                trajectory_source_stamp_ns=7_000_000_000,
                trajectory_contract_id='none',
                gripper_command_id=3,
                gripper_source_stamp_ns=6_500_000_000,
            )
            self._planning_rgbd = {}
            self._planning_rgbd_conflicts = {}
            self._verification_targets = {}
            self._verification_identities = {}
            self._verification_identity_conflicts = {}
            self._planning_identity = None
            self._worker = None
            self._workers = {}
            self.started = {}
            self.release = {}

        def stage(self, goal_id, stamp_ns):
            token = self._transaction.begin(PlaceTransactionIdentity(
                goal_id,
                'executor-a',
            ), now_ros_ns=stamp_ns, now_wall_s=float(stamp_ns))
            identity = object()
            key = (stamp_ns, 'camera')
            self._request = SimpleNamespace(
                stamp_ns=stamp_ns,
                image_frame='camera',
                observation_identity=identity,
                executor_epoch='executor-a',
            )
            self._request_identity = identity
            self._request_transaction_token = token
            self._planning_rgbd[key] = SimpleNamespace(
                rgb_stamp_ns=stamp_ns,
                depth_stamp_ns=stamp_ns,
                camera_info_stamp_ns=stamp_ns,
                source_frame='camera',
            )
            self._verification_targets[key] = SimpleNamespace(
                stamp_ns=stamp_ns,
                source_frame='camera',
            )
            self._verification_identities[key] = identity
            self.started[token.generation] = threading.Event()
            self.release[token.generation] = threading.Event()
            return token

        @staticmethod
        def _validate_exact_planning_sources(*_args):
            return None

        def _plan_worker(self, inputs):
            token = inputs.token
            self.started[token.generation].set()
            self.release[token.generation].wait(timeout=2.0)
            current = threading.current_thread()
            with self._lock:
                ObservedPlacementNode._release_planning_worker_locked(
                    self,
                    token,
                    current,
                )

    harness = Harness()
    old_token = harness.stage('place-old', 100)
    accepted, _reason = ObservedPlacementNode._start_plan(harness)
    assert accepted
    assert harness.started[old_token.generation].wait(timeout=1.0)
    old_worker = harness._worker

    assert harness._transaction.abort(PlaceTransactionControl(
        'abort',
        old_token.identity,
    ))
    harness._planning_identity = None
    new_token = harness.stage('place-new', 200)
    accepted, reason = ObservedPlacementNode._start_plan(harness)
    assert not accepted
    assert reason == 'previous placement worker cancellation is pending'
    assert old_worker.is_alive()
    assert tuple(harness._workers) == (old_token.generation,)

    harness.release[old_token.generation].set()
    old_worker.join(timeout=1.0)
    assert not old_worker.is_alive()

    accepted, _reason = ObservedPlacementNode._start_plan(harness)
    assert accepted
    assert harness.started[new_token.generation].wait(timeout=1.0)
    new_worker = harness._worker
    assert new_worker is not old_worker
    assert harness._workers[new_token.generation] is new_worker
    with pytest.raises(PlaceTransactionError, match='cannot arm'):
        harness._transaction.arm(
            old_token,
            now_ros_ns=300,
            now_wall_s=1.0,
        )
    assert harness._transaction.matches(new_token)

    harness.release[new_token.generation].set()
    new_worker.join(timeout=1.0)
    assert not new_worker.is_alive()
    assert harness._worker is None
    assert not harness._workers


@pytest.mark.parametrize(
    'delayed_input',
    ('identity', 'rgbd', 'target', 'scene', 'joints', 'execution'),
)
def test_request_keyed_assembler_retries_each_late_input_exactly_once(
    delayed_input,
):
    stamp_ns = 81_230_000_001
    frame_id = 'wrist_camera_optical_frame'
    identity = SimpleNamespace(
        stamp_ns=stamp_ns,
        frame_id=frame_id,
    )
    rgbd = SimpleNamespace(
        rgb_stamp_ns=stamp_ns,
        depth_stamp_ns=stamp_ns,
        camera_info_stamp_ns=stamp_ns,
        source_frame=frame_id,
    )
    target = SimpleNamespace(stamp_ns=stamp_ns, source_frame=frame_id)

    class Harness:
        def __init__(self):
            self._lock = threading.Lock()
            self._transaction = PlacementTransactionLifecycle(
                ros_timeout_s=3.0,
                wall_timeout_s=5.0,
            )
            token = self._transaction.begin(
                PlaceTransactionIdentity('place-7', 'executor-a'),
                now_ros_ns=stamp_ns,
                now_wall_s=10.0,
            )
            self._request = SimpleNamespace(
                stamp_ns=stamp_ns,
                image_frame=frame_id,
                observation_identity=identity,
                executor_epoch='executor-a',
            )
            self._request_transaction_token = token
            self._request_identity = identity
            self._scene = object()
            self._joints = object()
            self._execution_feedback = _completed_carry_execution()
            key = (stamp_ns, frame_id)
            self._planning_rgbd = {key: rgbd}
            self._planning_rgbd_conflicts = {}
            self._verification_targets = {key: target}
            self._verification_identities = {key: identity}
            self._verification_identity_conflicts = {}
            self._planning_identity = None
            self._worker = None
            self._workers = {}
            self.started = 0
            self.worker_done = threading.Event()
            if delayed_input == 'identity':
                self._request_identity = None
                self._verification_identities.clear()
            elif delayed_input == 'rgbd':
                self._planning_rgbd.clear()
            elif delayed_input == 'target':
                self._verification_targets.clear()
            elif delayed_input == 'execution':
                self._execution_feedback = None
            else:
                setattr(self, f'_{delayed_input}', None)

        def _plan_worker(self, inputs):
            self.started += 1
            with self._lock:
                ObservedPlacementNode._release_planning_worker_locked(
                    self,
                    inputs.token,
                    threading.current_thread(),
                )
            self.worker_done.set()

    harness = Harness()
    accepted, _reason = ObservedPlacementNode._start_plan(harness)
    assert not accepted

    key = (stamp_ns, frame_id)
    if delayed_input == 'identity':
        harness._verification_identities[key] = identity
    elif delayed_input == 'rgbd':
        harness._planning_rgbd[key] = rgbd
    elif delayed_input == 'target':
        harness._verification_targets[key] = target
    elif delayed_input == 'scene':
        harness._scene = object()
    elif delayed_input == 'joints':
        harness._joints = object()
    else:
        harness._execution_feedback = _completed_carry_execution()

    accepted, _reason = ObservedPlacementNode._start_plan(harness)
    assert accepted
    assert harness.worker_done.wait(timeout=1.0)
    accepted, _reason = ObservedPlacementNode._start_plan(harness)
    assert not accepted
    assert harness.started == 1


def test_completed_carry_snapshot_is_frozen_into_place_contract_v2():
    execution = ObservedPlacementNode._validate_completed_carry_execution(
        _completed_carry_execution(),
    )
    output = SimpleNamespace(
        goal_id='place-7-1000000000',
        trajectory=SimpleNamespace(
            frame_id='base_link',
            joint_names=('joint1', 'joint2'),
            phase_start_indices=(
                ('transit', 0), ('approach', 2), ('retreat', 4),
            ),
            points=(object(),) * 6,
        ),
    )
    identity = ObservedPerceptionIdentity(
        request_id='request-a',
        producer_epoch='perception-a',
        generation=4,
        stamp_ns=5_000_000_000,
        frame_id='wrist_camera_optical_frame',
    )

    payload = ObservedPlacementNode._place_contract_payload(
        output,
        identity,
        execution,
        trajectory_topic='/z_manip/place/trajectory',
        trajectory_frame_id='base_link',
        trajectory_digest_sha256='a' * 64,
    )

    assert payload['schema'] == 'z_manip.place_contract.v2'
    assert payload['schema_version'] == 2
    assert payload['goal_id'] == 'place-7-1000000000'
    assert payload['trajectory_contract_id'] == payload['goal_id']
    assert payload['executor_epoch'] == 'executor-a'
    assert payload['trajectory_command_highwater'] == 7
    assert payload['trajectory_source_highwater_ns'] == 7_000_000_000
    assert payload['gripper_command_highwater'] == 3
    assert payload['gripper_source_highwater_ns'] == 6_500_000_000
    assert payload['trajectory_digest_sha256'] == 'a' * 64


def test_published_place_contract_digests_the_actual_ros_trajectory():
    """Bind contract metadata to the exact separately published ROS message."""
    publication_order = []

    class Publisher:
        def __init__(self, name, topic_name=''):
            self.name = name
            self.topic_name = topic_name
            self.messages = []

        def publish(self, message):
            publication_order.append(self.name)
            self.messages.append(message)

    pose = np.eye(4, dtype=float)
    pose[:3, 3] = (0.4, -0.2, 0.7)
    candidate = SimpleNamespace(
        preplace_pose=pose.copy(),
        place_pose=pose.copy(),
        retreat_pose=pose.copy(),
    )
    output = SimpleNamespace(
        goal_id='place-7-1000000000',
        candidates=(SimpleNamespace(candidate=candidate, feasible=True),),
        result=SimpleNamespace(
            candidate=candidate,
            score=1.25,
            plane=SimpleNamespace(inlier_ratio=0.92, rms_error_m=0.004),
        ),
        trajectory=SimpleNamespace(
            frame_id='planner-intermediate-frame',
            joint_names=('joint1', 'joint2'),
            phase_start_indices=(
                ('transit', 0), ('approach', 1), ('retreat', 2),
            ),
            points=(
                SimpleNamespace(positions=(0.0, 0.1), time_from_start_s=0.2),
                SimpleNamespace(positions=(0.2, 0.3), time_from_start_s=0.5),
                SimpleNamespace(positions=(0.4, 0.5), time_from_start_s=0.9),
            ),
        ),
    )
    identity = ObservedPerceptionIdentity(
        request_id='request-a',
        producer_epoch='perception-a',
        generation=4,
        stamp_ns=5_000_000_000,
        frame_id='wrist_camera_optical_frame',
    )
    harness = SimpleNamespace(
        _planning_frame='base_link',
        _candidate_publisher=Publisher('candidates'),
        _selected_publisher=Publisher('selected'),
        _trajectory_publisher=Publisher(
            'trajectory',
            '/resolved/z_manip/place/trajectory',
        ),
        _contract_publisher=Publisher('contract'),
        _place_contract_payload=ObservedPlacementNode._place_contract_payload,
        _publish_status=lambda *_args, **_kwargs: None,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(
                to_msg=lambda: TimeMessage(sec=12, nanosec=345),
            ),
        ),
    )

    ObservedPlacementNode._publish_output(
        harness,
        output,
        identity,
        _completed_carry_execution(),
    )

    trajectory = harness._trajectory_publisher.messages[0]
    contract = json.loads(harness._contract_publisher.messages[0].data)
    expected_digest = canonical_joint_trajectory_sha256(
        frame_id=trajectory.header.frame_id,
        header_stamp=trajectory.header.stamp,
        joint_names=trajectory.joint_names,
        points=trajectory.points,
    )
    assert publication_order[-2:] == ['trajectory', 'contract']
    assert trajectory.header.frame_id == 'base_link'
    assert contract['frame_id'] == trajectory.header.frame_id
    assert contract['trajectory_topic'] == (
        '/resolved/z_manip/place/trajectory'
    )
    assert contract['trajectory_digest_sha256'] == expected_digest


@pytest.mark.parametrize(
    ('field', 'value'),
    (
        ('trajectory_status', 'active'),
        ('trajectory_owner', 'trajectory_hold'),
        ('trajectory_segment', 'lift'),
        ('trajectory_contract_id', 'place-7-1000000000'),
        ('executor_epoch', ''),
        ('trajectory_command_id', 0),
        ('trajectory_source_stamp_ns', -1),
        ('gripper_command_id', 0),
        ('gripper_source_stamp_ns', -1),
    ),
)
def test_place_arm_rejects_every_non_completed_carry_snapshot_field(field, value):
    execution = replace(_completed_carry_execution(), **{field: value})

    with pytest.raises(PlacementContractError, match='completed carry'):
        ObservedPlacementNode._validate_completed_carry_execution(execution)


def test_place_arm_rejects_missing_executor_snapshot():
    with pytest.raises(PlacementContractError, match='unavailable'):
        ObservedPlacementNode._validate_completed_carry_execution(None)
