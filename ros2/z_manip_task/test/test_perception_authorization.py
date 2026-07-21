"""Generation and frame authorization tests for semantic grounding."""

import importlib
from itertools import permutations
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip('rclpy')
diagnostic_msgs = importlib.import_module('diagnostic_msgs.msg')
std_msgs = importlib.import_module('std_msgs.msg')
task_core = importlib.import_module('z_manip_task.core')
task_node = importlib.import_module('z_manip_task.node')
DiagnosticArray = diagnostic_msgs.DiagnosticArray
DiagnosticStatus = diagnostic_msgs.DiagnosticStatus
KeyValue = diagnostic_msgs.KeyValue
Bool = std_msgs.Bool
String = std_msgs.String
ObservationSerialGate = task_core.ObservationSerialGate
ExecutionOcclusionDecision = task_core.ExecutionOcclusionDecision
ExecutionOcclusionGate = task_core.ExecutionOcclusionGate
RuntimePhase = task_core.RuntimePhase
RuntimeSafetyCore = task_core.RuntimeSafetyCore
MobileManipulationRuntime = task_node.MobileManipulationRuntime


STAMP_NS = 10_123_456_789
FRAME_ID = 'wrist_depth_optical_frame'
REQUEST_ID = 'task-request-7'
PRODUCER_EPOCH = 'bridge-epoch-a'
EXECUTOR_EPOCH = 'executor-epoch-a'


def _carry_execution_status():
    return task_core.parse_execution_status(
        'succeeded;owner=trajectory;segment=carry;command_id=7;'
        'trajectory_contract_id=none;'
        f'executor_epoch={EXECUTOR_EPOCH};'
        'trajectory_token=trajectory-carry;'
        'trajectory_received_at=7.000000;'
        'gripper_command_id=3;gripper_received_at=6.500000',
    )


class _Resettable:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls += 1


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


def _header(stamp_ns: int = STAMP_NS, frame_id: str = FRAME_ID):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=stamp_ns // 1_000_000_000,
            nanosec=stamp_ns % 1_000_000_000,
        ),
        frame_id=frame_id,
    )


def _status(
    generation: int,
    *,
    request_id: str = REQUEST_ID,
    producer_epoch: str = PRODUCER_EPOCH,
    grounding_scope: str | None = 'grasp_for_place',
    phase: str = 'tracking',
    stamp_ns: int | None = STAMP_NS,
    frame_id: str | None = FRAME_ID,
    valid: str = 'true',
    failure: str = 'none',
    level: int = DiagnosticStatus.OK,
) -> DiagnosticArray:
    values = [
        KeyValue(key='schema', value='z_manip.perception_status.v1'),
        KeyValue(key='generation', value=str(generation)),
        KeyValue(key='request_id', value=request_id),
        KeyValue(key='producer_epoch', value=producer_epoch),
        KeyValue(key='valid', value=valid),
        KeyValue(key='failure', value=failure),
    ]
    if grounding_scope is not None:
        values.append(KeyValue(key='grounding_scope', value=grounding_scope))
    if stamp_ns is not None:
        values.append(KeyValue(key='observation_stamp_ns', value=str(stamp_ns)))
    if frame_id is not None:
        values.append(KeyValue(key='observation_frame_id', value=frame_id))
    status = DiagnosticStatus(level=level, message=phase, values=values)
    return DiagnosticArray(status=[status])


class _Harness:
    _revoke_perception_success = MobileManipulationRuntime._revoke_perception_success
    _clear_perception_authorization = (
        MobileManipulationRuntime._clear_perception_authorization
    )
    _reject_perception_generation_advance = (
        MobileManipulationRuntime._reject_perception_generation_advance
    )
    _clear_semantic_observation_cache = (
        MobileManipulationRuntime._clear_semantic_observation_cache
    )
    _grounding_observation_authorized = (
        MobileManipulationRuntime._grounding_observation_authorized
    )
    _begin_execution_occlusion_loss = (
        MobileManipulationRuntime._begin_execution_occlusion_loss
    )
    _has_frozen_coarse_nav_contract = (
        MobileManipulationRuntime._has_frozen_coarse_nav_contract
    )
    _publish_grounding_request = MobileManipulationRuntime._publish_grounding_request
    _request_semantic_reground = MobileManipulationRuntime._request_semantic_reground

    def __init__(self, *, required_generation: int = 7) -> None:
        self._lock = threading.RLock()
        self._core = RuntimeSafetyCore()
        self._core.begin('pick the observed bottle')
        self._core.mark_pose_settled()
        self._serial_gate = ObservationSerialGate(
            sync_slop_s=0.12,
            max_age_s=0.35,
        )
        self._perception_valid = False
        self._valid_seen_at = None
        self._perception_generation = required_generation - 1
        self._required_perception_generation = None
        self._required_perception_request_id = REQUEST_ID
        self._required_grounding_scope = 'grasp_for_place'
        self._bound_perception_request_id = None
        self._bound_perception_producer_epoch = None
        self._bound_perception_generation = None
        self._valid_perception_request_id = None
        self._valid_perception_producer_epoch = None
        self._valid_perception_generation = None
        self._valid_observation_stamp_ns = None
        self._valid_observation_frame_id = ''
        self._affordance = None
        self._affordance_generation = required_generation - 1
        self._affordance_request_id = ''
        self._affordance_producer_epoch = ''
        self._required_affordance_generation = None
        self._handled_perception_failure = None
        self._execution_occlusion = ExecutionOcclusionGate()
        self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
            False,
            'execution occlusion is not armed',
        )
        self._execution_occlusion_loss_detail = ''
        self._coarse_nav_perception_loss_detail = ''
        self._target_camera = None
        self._target_piper = None
        self._target_stamp = 0.0
        self._target_stamp_ns = 0
        self._target_frame_id = ''
        self._target_cloud = None
        self._target_uv = None
        self._target_cloud_stamp = 0.0
        self._target_cloud_stamp_ns = 0
        self._target_cloud_frame_id = ''
        self._scene_cloud = None
        self._scene_cloud_stamp = 0.0
        self._scene_cloud_stamp_ns = 0
        self._scene_cloud_frame_id = ''
        self._camera_origin_piper = None
        self._camera_rotation_piper = None
        self.actions = []
        self.recoveries = []
        self.invalidations = 0
        self.warnings = []

    def _now_s(self) -> float:
        return STAMP_NS * 1e-9 + 0.05

    def get_parameter(self, name: str):
        return SimpleNamespace(value={
            'sync_slop_s': 0.12,
            'max_perception_age_s': 0.35,
            'semantic_reground_timeout_s': 2.0,
            'place_mode': 'observed_place',
            'visual_servo_image_margin_ratio': 0.02,
        }[name])

    def _topic_value(self, parameter: str) -> str:
        return str(self.get_parameter(parameter).value)

    def _lookup_piper_from(self, _frame_id, _stamp):
        return np.eye(4)

    @staticmethod
    def _read_cloud(_msg, *, need_uv: bool):
        points = np.array([[0.1, 0.0, 0.5], [0.11, 0.01, 0.51]])
        uv = np.array([[10.0, 20.0], [11.0, 21.0]]) if need_uv else None
        return points, uv

    def _apply_safety(self, action) -> None:
        self.actions.append(action)

    def _recover_precontact(self, kind, detail: str) -> bool:
        self.recoveries.append((kind, detail))
        return False

    def _invalidate_async_work(self) -> None:
        self.invalidations += 1

    def get_logger(self):
        return SimpleNamespace(warning=self.warnings.append)


def _target_message(stamp_ns: int = STAMP_NS, frame_id: str = FRAME_ID):
    return SimpleNamespace(
        header=_header(stamp_ns, frame_id),
        bbox=SimpleNamespace(
            center=SimpleNamespace(
                position=SimpleNamespace(x=0.1, y=0.0, z=0.5),
            ),
        ),
    )


def _cloud_message(stamp_ns: int = STAMP_NS, frame_id: str = FRAME_ID):
    return SimpleNamespace(header=_header(stamp_ns, frame_id))


def _apply_event(harness: _Harness, event: str, generation: int = 7) -> None:
    if event == 'status':
        MobileManipulationRuntime._perception_status_cb(
            harness,
            _status(generation),
        )
    elif event == 'affordance':
        MobileManipulationRuntime._affordance_cb(harness, String(data=json.dumps({
            'schema': 'z_manip.affordance.v2',
            'generation': generation,
            'request_id': REQUEST_ID,
            'producer_epoch': PRODUCER_EPOCH,
            'grounding_scope': 'grasp_for_place',
            'target': {'bbox_xyxy_normalized': [0.2, 0.2, 0.4, 0.7]},
            'placement_region': None,
            'placement_avoid_regions': [],
            'placement_verification': {
                'require_upright': True,
                'upright_axis': 'principal_long',
                'orientation_symmetry': 'axial',
                'symmetry_axis': 'principal_long',
            },
        })))
    elif event == 'target':
        MobileManipulationRuntime._target_cb(harness, _target_message())
    elif event == 'target_cloud':
        MobileManipulationRuntime._target_cloud_cb(harness, _cloud_message())
    elif event == 'scene_cloud':
        MobileManipulationRuntime._scene_cloud_cb(harness, _cloud_message())
    else:
        raise AssertionError(f'unknown event {event}')


def _authorized(harness: _Harness) -> bool:
    synchronized = harness._serial_gate.snapshot(harness._now_s())
    return harness._grounding_observation_authorized(synchronized)


@pytest.mark.parametrize(
    'order',
    tuple(permutations((
        'status', 'affordance', 'target', 'target_cloud', 'scene_cloud',
    ))),
)
def test_all_callback_orders_require_the_complete_exact_bundle(order) -> None:
    harness = _Harness()

    for index, event in enumerate(order):
        _apply_event(harness, event)
        assert _authorized(harness) is (index == len(order) - 1)

    assert harness._bound_perception_generation == 7
    assert harness._valid_perception_generation == 7


def test_neither_generationless_bool_value_can_mutate_status_authority() -> None:
    harness = _Harness()

    MobileManipulationRuntime._valid_cb(harness, Bool(data=True))
    assert not harness._perception_valid
    assert harness._bound_perception_generation is None

    for event in ('status', 'affordance', 'target', 'target_cloud', 'scene_cloud'):
        _apply_event(harness, event)
    assert _authorized(harness)

    MobileManipulationRuntime._valid_cb(harness, Bool(data=False))
    assert _authorized(harness)
    assert harness._bound_perception_generation == 7


def test_unrelated_generation_cannot_bind_or_poison_bridge_restart() -> None:
    harness = _Harness(required_generation=7)
    harness._perception_generation = 500

    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(
            999_999,
            request_id='some-other-task',
            phase='tracking',
        ),
    )
    assert harness._bound_perception_generation is None
    assert harness._perception_generation == 500
    assert harness.actions == []

    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(
            1,
            phase='waiting_tracker',
            stamp_ns=None,
            frame_id=None,
            valid='false',
            level=DiagnosticStatus.WARN,
        ),
    )
    assert harness._bound_perception_generation == 1
    assert harness._required_perception_generation == 1
    assert harness._required_affordance_generation == 1
    assert harness._perception_generation == 1

    for event in ('affordance', 'target', 'target_cloud', 'scene_cloud', 'status'):
        _apply_event(harness, event, generation=1)
    assert _authorized(harness)


def test_only_the_bound_generation_can_trigger_failure_recovery() -> None:
    harness = _Harness(required_generation=7)

    stale = _status(
        6,
        request_id='another-task-request',
        phase='failed',
        stamp_ns=None,
        frame_id=None,
        valid='false',
        level=DiagnosticStatus.ERROR,
    )
    MobileManipulationRuntime._perception_status_cb(harness, stale)
    assert harness.recoveries == []
    assert harness._core.phase is RuntimePhase.GROUNDING

    current = _status(
        8,
        phase='failed',
        stamp_ns=None,
        frame_id=None,
        valid='false',
        level=DiagnosticStatus.ERROR,
    )
    MobileManipulationRuntime._perception_status_cb(harness, current)
    assert harness._bound_perception_generation == 8
    assert len(harness.recoveries) == 1
    assert 'perception generation 8 failed' in harness._core.failure_reason

    MobileManipulationRuntime._perception_status_cb(harness, current)
    assert len(harness.recoveries) == 1


def _freeze_coarse_nav_contract(harness: _Harness, *, serial: int = 1) -> None:
    harness._core.phase = RuntimePhase.COARSE_NAV
    harness._core.prospective_serial = serial
    harness._task = SimpleNamespace(stage=SimpleNamespace(value='coarse_nav'))
    harness._work_pose = {
        'goal_id': 'work-observed-1',
        'map_goal_xy': [1.0, -0.2],
        'map_goal_yaw_rad': 0.1,
        'map_frame': 'map',
        'source': {
            'request_id': REQUEST_ID,
            'producer_epoch': PRODUCER_EPOCH,
            'generation': 7,
            'observation_serial': serial,
        },
    }


def test_bound_tracker_loss_is_deferred_for_frozen_map_work_pose() -> None:
    harness = _complete_harness()
    _freeze_coarse_nav_contract(harness)
    work_pose = harness._work_pose

    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(
            7,
            phase='failed',
            stamp_ns=None,
            frame_id=None,
            valid='false',
            failure='tracker_reported_loss',
            level=DiagnosticStatus.ERROR,
        ),
    )

    assert harness._core.phase is RuntimePhase.COARSE_NAV
    assert harness._work_pose is work_pose
    assert not harness._perception_valid
    assert harness.recoveries == []
    assert harness.actions == []
    assert 'tracker_reported_loss' in harness._coarse_nav_perception_loss_detail
    assert len(harness.warnings) == 1


@pytest.mark.parametrize(
    'failure',
    ('clock_rollback', 'grounding_failed', 'camera_frame_timeout'),
)
def test_nontracker_failure_is_never_deferred_by_frozen_navigation(
    failure,
) -> None:
    harness = _complete_harness()
    _freeze_coarse_nav_contract(harness)

    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(
            7,
            phase='failed',
            stamp_ns=None,
            frame_id=None,
            valid='false',
            failure=failure,
            level=DiagnosticStatus.ERROR,
        ),
    )

    assert len(harness.recoveries) == 1
    assert harness._coarse_nav_perception_loss_detail == ''


def test_frozen_map_work_pose_publishes_exact_revocable_heartbeat() -> None:
    harness = _complete_harness()
    _freeze_coarse_nav_contract(harness)
    harness._frozen_coarse_nav_authorization_pub = _Publisher()
    harness._frozen_coarse_nav_authorization_identity = None

    MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
        harness,
        True,
    )
    active = json.loads(
        harness._frozen_coarse_nav_authorization_pub.messages[-1].data,
    )
    assert active == {
        'schema': 'z_manip.frozen_coarse_nav_authorization.v1',
        'active': True,
        'request_id': REQUEST_ID,
        'producer_epoch': PRODUCER_EPOCH,
        'generation': 7,
        'observation_serial': 1,
        'nav_goal_id': 'work-observed-1',
    }

    harness._core.phase = RuntimePhase.NEAR_GROUNDING
    MobileManipulationRuntime._frozen_coarse_nav_authorization_tick(harness)
    inactive = json.loads(
        harness._frozen_coarse_nav_authorization_pub.messages[-1].data,
    )
    assert inactive == {**active, 'active': False}
    assert harness._frozen_coarse_nav_authorization_identity is None


def test_terminal_release_revokes_authorization_even_after_one_shot_cleanup() -> None:
    identity = {
        'schema': 'z_manip.frozen_coarse_nav_authorization.v1',
        'request_id': REQUEST_ID,
        'producer_epoch': PRODUCER_EPOCH,
        'generation': 7,
        'observation_serial': 1,
        'nav_goal_id': 'work-observed-1',
    }
    publisher = _Publisher()
    harness = SimpleNamespace(
        _terminal_ownership_released=True,
        _frozen_coarse_nav_authorization_pub=publisher,
        _frozen_coarse_nav_authorization_identity=identity,
    )

    MobileManipulationRuntime._release_terminal_ownership(harness)

    assert json.loads(publisher.messages[-1].data) == {
        **identity,
        'active': False,
    }
    assert harness._frozen_coarse_nav_authorization_identity is None


@pytest.mark.parametrize('invalid_serial', (0, -1))
def test_nonpositive_serial_cannot_authorize_frozen_navigation(
    invalid_serial,
) -> None:
    harness = _complete_harness()
    _freeze_coarse_nav_contract(harness, serial=invalid_serial)
    harness._frozen_coarse_nav_authorization_pub = _Publisher()
    harness._frozen_coarse_nav_authorization_identity = None

    assert not harness._has_frozen_coarse_nav_contract()
    with pytest.raises(RuntimeError, match='frozen nav contract'):
        MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
            harness,
            True,
        )
    assert harness._frozen_coarse_nav_authorization_pub.messages == []


@pytest.mark.parametrize(
    'mutation',
    ('legacy_goal', 'missing_source', 'serial_mismatch', 'task_stage_mismatch'),
)
def test_tracker_loss_without_strict_frozen_contract_still_recovers(mutation) -> None:
    harness = _complete_harness()
    _freeze_coarse_nav_contract(harness)
    if mutation == 'legacy_goal':
        harness._work_pose = None
    elif mutation == 'missing_source':
        harness._work_pose.pop('source')
    elif mutation == 'serial_mismatch':
        harness._work_pose['source']['observation_serial'] = 2
    else:
        harness._task.stage.value = 'search'

    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(
            7,
            phase='failed',
            stamp_ns=None,
            frame_id=None,
            valid='false',
            failure='tracker_reported_loss',
            level=DiagnosticStatus.ERROR,
        ),
    )

    assert len(harness.recoveries) == 1
    assert harness._coarse_nav_perception_loss_detail == ''


def test_status_from_another_request_or_producer_cannot_steal_ownership() -> None:
    harness = _complete_harness()

    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(
            8,
            request_id='some-other-task',
        ),
    )
    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(1, producer_epoch='stale-or-restarted-bridge'),
    )

    assert harness.invalidations == 0
    assert _authorized(harness)
    assert harness._core.phase is RuntimePhase.GROUNDING


@pytest.mark.parametrize('grounding_scope', (None, 'place_support'))
def test_matching_status_must_echo_the_exact_grounding_scope(
    grounding_scope,
) -> None:
    harness = _Harness()

    MobileManipulationRuntime._perception_status_cb(
        harness,
        _status(7, grounding_scope=grounding_scope),
    )

    assert harness._bound_perception_generation is None
    assert len(harness.actions) == 1
    assert 'grounding scope' in harness._core.failure_reason


def test_unrelated_affordance_cannot_poison_generation_or_semantics() -> None:
    harness = _Harness()

    MobileManipulationRuntime._affordance_cb(harness, String(data=json.dumps({
        'schema': 'z_manip.affordance.v2',
        'request_id': 'another-task',
        'producer_epoch': 'another-bridge',
        'generation': 1_000_000,
        'target': {'bbox_xyxy_normalized': [0.0, 0.0, 1.0, 1.0]},
    })))

    assert harness._affordance is None
    assert harness._affordance_generation == 6
    assert harness.actions == []


@pytest.mark.parametrize(
    ('field', 'invalid_value'),
    (
        ('grasp_part', {'label': 'body'}),
        ('avoid_regions', [{'label': 'handle'}]),
        ('preferred_approach_camera', [0.0, 0.0, 1.0]),
        ('placement_avoid_regions', {'not': 'an array'}),
    ),
)
def test_place_support_affordance_rejects_cross_stage_fields(
    field,
    invalid_value,
) -> None:
    harness = _Harness()
    harness._required_grounding_scope = 'place_support'
    harness._carried_object_geometry = object()
    value = {
        'schema': 'z_manip.affordance.v2',
        'request_id': REQUEST_ID,
        'producer_epoch': PRODUCER_EPOCH,
        'generation': 7,
        'grounding_scope': 'place_support',
        'target': {'bbox_xyxy_normalized': [0.2, 0.2, 0.4, 0.7]},
        'grasp_part': None,
        'avoid_regions': [],
        'preferred_approach_camera': None,
        'placement_region': {
            'label': 'empty shelf area',
            'bbox_xyxy_normalized': [0.5, 0.2, 0.8, 0.6],
        },
        'placement_avoid_regions': [],
        'placement_verification': None,
    }
    value[field] = invalid_value

    MobileManipulationRuntime._affordance_cb(
        harness,
        String(data=json.dumps(value)),
    )

    assert harness._affordance is None
    assert len(harness.actions) == 1
    assert 'place_support fields' in harness._core.failure_reason


def test_generation_change_within_exact_request_fails_closed() -> None:
    harness = _complete_harness()

    MobileManipulationRuntime._perception_status_cb(harness, _status(8))

    assert harness.invalidations == 1
    assert not _authorized(harness)
    assert harness._core.phase is RuntimePhase.FAILED
    assert 'generation changed' in harness._core.failure_reason


@pytest.mark.parametrize(
    ('attribute', 'value'),
    (
        ('_affordance_generation', 6),
        ('_affordance_request_id', 'another-request'),
        ('_affordance_producer_epoch', 'another-bridge'),
        ('_valid_perception_request_id', 'another-request'),
        ('_valid_perception_producer_epoch', 'another-bridge'),
        ('_valid_perception_generation', 6),
        ('_target_stamp_ns', STAMP_NS + 1),
        ('_target_cloud_stamp_ns', STAMP_NS + 1),
        ('_scene_cloud_stamp_ns', STAMP_NS + 1),
        ('_target_frame_id', 'other_frame'),
        ('_target_cloud_frame_id', 'other_frame'),
        ('_scene_cloud_frame_id', 'other_frame'),
    ),
)
def test_generation_stamp_and_frame_must_each_match_exactly(attribute, value) -> None:
    harness = _Harness()
    for event in ('status', 'affordance', 'target', 'target_cloud', 'scene_cloud'):
        _apply_event(harness, event)
    assert _authorized(harness)

    setattr(harness, attribute, value)
    assert not _authorized(harness)


@pytest.mark.parametrize(
    'status',
    (
        _status(7, stamp_ns=None),
        _status(7, frame_id=None),
        _status(7, stamp_ns=0),
        _status(7, valid='false'),
        _status(7, level=DiagnosticStatus.WARN),
    ),
)
def test_malformed_tracking_status_cannot_authorize(status) -> None:
    harness = _Harness()

    MobileManipulationRuntime._perception_status_cb(harness, status)

    assert not harness._perception_valid
    assert harness._valid_perception_generation is None
    assert len(harness.actions) == 1
    assert 'exact observation identity' in harness._core.failure_reason


def test_matching_status_without_producer_epoch_fails_closed() -> None:
    harness = _Harness()
    status = _status(7)
    status.status[0].values = [
        item for item in status.status[0].values
        if item.key != 'producer_epoch'
    ]

    MobileManipulationRuntime._perception_status_cb(harness, status)

    assert not harness._perception_valid
    assert len(harness.actions) == 1
    assert 'ownership identity' in harness._core.failure_reason


def _complete_harness() -> _Harness:
    harness = _Harness()
    for event in ('status', 'affordance', 'target', 'target_cloud', 'scene_cloud'):
        _apply_event(harness, event)
    assert _authorized(harness)
    return harness


def test_initial_grounding_transition_uses_generation_authorization() -> None:
    harness = _complete_harness()
    harness._future = None
    harness._visual_search = _Resettable()
    harness._visual_search_error_rad = 1.0
    harness._visual_search_reason = 'old'
    harness._visual_search_edge_direction = 0
    harness.started = []
    harness._horizontal_edge_direction = lambda **_kwargs: 0
    harness._vertical_edge_direction = lambda **_kwargs: 0
    harness._semantic_observation = lambda serial, stamp: (serial, stamp)
    harness._start_planning = lambda kind, observation: harness.started.append(
        (kind, observation),
    )

    synchronized = harness._serial_gate.snapshot(harness._now_s())
    harness._target_stamp_ns += 1
    MobileManipulationRuntime._grounding_tick(harness, synchronized)
    assert harness._core.phase is RuntimePhase.GROUNDING
    assert harness.started == []

    harness._target_stamp_ns = STAMP_NS
    MobileManipulationRuntime._grounding_tick(harness, synchronized)
    assert harness._core.phase is RuntimePhase.STANDOFF
    assert harness.started[0][0] == 'standoff'


@pytest.mark.parametrize(
    ('phase', 'expected'),
    (
        (RuntimePhase.NEAR_GROUNDING, RuntimePhase.VISUAL_SERVO),
        (RuntimePhase.FINAL_GROUNDING, RuntimePhase.PLANNING),
        (RuntimePhase.PLACE_GROUNDING, RuntimePhase.PLACE_PLANNING),
    ),
)
def test_every_reground_transition_uses_generation_authorization(
    phase: RuntimePhase,
    expected: RuntimePhase,
) -> None:
    harness = _complete_harness()
    harness._core.phase = phase
    harness._core.required_replan_serial = 1
    harness._reground_started_at = harness._now_s() - 0.1
    harness._reground_last_tick_at = harness._reground_started_at
    harness._approach = _Resettable()
    harness._future = None
    harness.started = []
    harness.place_requests = 0
    harness._semantic_observation = lambda serial, stamp: (serial, stamp)
    harness._start_planning = lambda kind, observation: harness.started.append(
        (kind, observation),
    )
    harness._execution_status = _carry_execution_status()
    harness._horizontal_edge_direction = lambda **_kwargs: 0
    harness._vertical_edge_direction = lambda **_kwargs: 0

    def publish_place_request(_synchronized) -> None:
        harness._core.place_request_sent(
            place_contract_id='place-test-goal',
            executor_state=harness._execution_status,
        )
        harness.place_requests += 1

    harness._publish_place_request = publish_place_request
    harness._place_planning_started_at = None
    synchronized = harness._serial_gate.snapshot(harness._now_s())

    harness._scene_cloud_frame_id = 'stale_frame'
    MobileManipulationRuntime._reground_tick(
        harness,
        harness._now_s(),
        synchronized,
    )
    assert harness._core.phase is phase
    assert harness.started == []
    assert harness.place_requests == 0

    harness._scene_cloud_frame_id = FRAME_ID
    MobileManipulationRuntime._reground_tick(
        harness,
        harness._now_s(),
        synchronized,
    )
    assert harness._core.phase is expected
    if phase is RuntimePhase.FINAL_GROUNDING:
        assert harness.started[0][0] == 'pregrasp'
    if phase is RuntimePhase.PLACE_GROUNDING:
        assert harness.place_requests == 1


def test_near_view_reprojects_work_target_with_measured_camera_extrinsics() -> None:
    harness = SimpleNamespace(
        _work_pose={
            'predicted_target_position_piper': [0.4, -0.1, 0.7],
            'desired_camera_depth_m': 0.6,
        },
        _camera_origin_piper=np.array([0.1, -0.1, 0.2]),
        _camera_rotation_piper=np.eye(3),
        _desired_depth=0.6,
        _config=SimpleNamespace(
            standoff=SimpleNamespace(
                min_camera_depth_m=0.32,
                max_camera_depth_m=0.75,
            ),
        ),
    )

    MobileManipulationRuntime._refresh_near_view_desired_depth(harness)

    assert harness._desired_depth == pytest.approx(0.5)
    assert harness._work_pose['desired_camera_depth_m'] == pytest.approx(0.5)
    assert harness._work_pose['near_view_raw_camera_depth_m'] == pytest.approx(0.5)


def test_near_view_reprojection_rejects_target_behind_measured_camera() -> None:
    harness = SimpleNamespace(
        _work_pose={
            'predicted_target_position_piper': [0.0, 0.0, -0.1],
        },
        _camera_origin_piper=np.zeros(3),
        _camera_rotation_piper=np.eye(3),
        _desired_depth=0.6,
        _config=SimpleNamespace(
            standoff=SimpleNamespace(
                min_camera_depth_m=0.32,
                max_camera_depth_m=0.75,
            ),
        ),
    )

    with pytest.raises(ValueError, match='desired depth'):
        MobileManipulationRuntime._refresh_near_view_desired_depth(harness)

    assert harness._desired_depth == pytest.approx(0.6)


def test_near_reground_recenters_edge_target_before_visual_servo() -> None:
    harness = _complete_harness()
    harness._core.phase = RuntimePhase.NEAR_GROUNDING
    harness._reground_started_at = harness._now_s() - 0.1
    harness._reground_last_tick_at = harness._reground_started_at
    harness._approach = _Resettable()
    harness._horizontal_edge_direction = lambda **_kwargs: 1
    harness._vertical_edge_direction = lambda **_kwargs: 0
    harness._visual_search_edge_direction = 0
    harness._publish_status = lambda: None
    harness._recover_precontact = lambda kind, detail: (
        harness.recoveries.append((kind, detail)) or True
    )
    synchronized = harness._serial_gate.snapshot(harness._now_s())

    MobileManipulationRuntime._reground_tick(
        harness,
        harness._now_s(),
        synchronized,
    )

    assert harness._core.phase is RuntimePhase.NEAR_GROUNDING
    assert harness._approach.calls == 0
    assert harness._visual_search_edge_direction == 1
    assert harness.recoveries[0][0] is task_node.FailureKind.NOT_FOUND


def test_near_reground_rejects_vertically_clipped_target_before_servo() -> None:
    harness = _complete_harness()
    harness._core.phase = RuntimePhase.NEAR_GROUNDING
    harness._reground_started_at = harness._now_s() - 0.1
    harness._reground_last_tick_at = harness._reground_started_at
    harness._approach = _Resettable()
    harness._horizontal_edge_direction = lambda **_kwargs: 0
    harness._vertical_edge_direction = lambda **_kwargs: -1
    harness._visual_search_edge_direction = 1
    harness._publish_status = lambda: None
    harness._recover_precontact = lambda kind, detail: (
        harness.recoveries.append((kind, detail)) or True
    )
    synchronized = harness._serial_gate.snapshot(harness._now_s())

    MobileManipulationRuntime._reground_tick(
        harness,
        harness._now_s(),
        synchronized,
    )

    assert harness._core.phase is RuntimePhase.NEAR_GROUNDING
    assert harness._approach.calls == 0
    assert harness._visual_search_edge_direction == 0
    assert (
        harness.recoveries[0][0]
        is task_node.FailureKind.VISUAL_APPROACH_FAILED
    )
    assert 'vertical' in harness.recoveries[0][1]


@pytest.mark.parametrize(
    ('top_edge', 'expected_phase', 'expected_recoveries'),
    (
        (0.034, RuntimePhase.VISUAL_SERVO, 0),
        (0.019, RuntimePhase.NEAR_GROUNDING, 1),
    ),
)
def test_near_reground_uses_servo_hard_margin_not_reentry_margin(
    top_edge: float,
    expected_phase: RuntimePhase,
    expected_recoveries: int,
) -> None:
    harness = _complete_harness()
    harness._core.phase = RuntimePhase.NEAR_GROUNDING
    harness._reground_started_at = harness._now_s() - 0.1
    harness._reground_last_tick_at = harness._reground_started_at
    harness._approach = _Resettable()
    harness._visual_search_edge_direction = 0
    harness._publish_status = lambda: None
    harness._recover_precontact = lambda kind, detail: (
        harness.recoveries.append((kind, detail)) or True
    )
    harness._affordance['target']['bbox_xyxy_normalized'] = [
        0.390,
        top_edge,
        0.467,
        0.388,
    ]
    harness._horizontal_edge_direction = lambda **kwargs: (
        MobileManipulationRuntime._horizontal_edge_direction(harness, **kwargs)
    )
    harness._vertical_edge_direction = lambda **kwargs: (
        MobileManipulationRuntime._vertical_edge_direction(harness, **kwargs)
    )

    MobileManipulationRuntime._reground_tick(
        harness,
        harness._now_s(),
        harness._serial_gate.snapshot(harness._now_s()),
    )

    assert harness._core.phase is expected_phase
    assert len(harness.recoveries) == expected_recoveries
    assert harness._approach.calls == (1 if expected_recoveries == 0 else 0)
    if expected_recoveries:
        assert (
            harness.recoveries[0][0]
            is task_node.FailureKind.VISUAL_APPROACH_FAILED
        )
    else:
        assert task_core.vertical_edge_direction(
            harness._affordance['target']['bbox_xyxy_normalized'],
            margin_ratio=0.06,
        ) == 1


def test_initial_grounding_vertical_clip_uses_coarse_nav_recovery() -> None:
    harness = _complete_harness()
    harness._future = None
    harness._horizontal_edge_direction = lambda **_kwargs: 0
    harness._vertical_edge_direction = lambda **_kwargs: -1
    harness._visual_search_edge_direction = 1
    harness._publish_status = lambda: None
    harness._recover_precontact = lambda kind, detail: (
        harness.recoveries.append((kind, detail)) or True
    )

    MobileManipulationRuntime._grounding_tick(
        harness,
        harness._serial_gate.snapshot(harness._now_s()),
    )

    assert harness._visual_search_edge_direction == 0
    assert (
        harness.recoveries[0][0]
        is task_node.FailureKind.VISUAL_APPROACH_FAILED
    )
    assert 'vertical' in harness.recoveries[0][1]


def test_place_payload_keeps_place_owner_distinct_from_grasp_reference() -> None:
    reference_identity = {
        'request_id': 'grasp-request',
        'producer_epoch': 'bridge-before-carry',
        'generation': 3,
        'observation_stamp_ns': STAMP_NS - 1_000_000,
        'frame_id': FRAME_ID,
    }
    geometry = SimpleNamespace(
        object_extent_m=np.asarray((0.05, 0.04, 0.12)),
        tool_from_object=np.eye(4),
        reference_points_object=np.arange(120, dtype=float).reshape(40, 3) * 1e-3,
        identity=SimpleNamespace(to_payload=lambda: dict(reference_identity)),
        verification_payload=lambda: {
            'require_upright': True,
            'upright_axis_object': [0.0, 0.0, 1.0],
            'orientation_symmetry': 'none',
        },
    )
    place_identity = task_node.PlacementObservationIdentity(
        goal_id='placeholder',
        request_id='place-request',
        producer_epoch='bridge-after-carry',
        generation=9,
        frame_id=FRAME_ID,
        planning_observation_stamp_ns=STAMP_NS,
        require_upright=True,
    )
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.PLACE_GROUNDING
    harness = SimpleNamespace(
        _core=core,
        _execution_status=_carry_execution_status(),
        _affordance={
            'placement_region': {'bbox_xyxy_normalized': [0.1, 0.2, 0.8, 0.9]},
            'placement_avoid_regions': [],
        },
        _target_cloud=np.ones((40, 3)),
        _carried_object_geometry=geometry,
        _target_frame_id=FRAME_ID,
        _valid_observation_stamp_ns=STAMP_NS,
        _affordance_generation=9,
        _place_programs={},
        _place_region_pub=_Publisher(),
        _nav_speed=0.0,
        _now_s=lambda: 10.0,
        _arm_is_still=lambda _now: True,
    )
    harness._capture_place_observation_identity = lambda _sync, goal_id: (
        task_node.PlacementObservationIdentity(
            goal_id=goal_id,
            request_id=place_identity.request_id,
            producer_epoch=place_identity.producer_epoch,
            generation=place_identity.generation,
            frame_id=place_identity.frame_id,
            planning_observation_stamp_ns=(
                place_identity.planning_observation_stamp_ns
            ),
            require_upright=place_identity.require_upright,
        )
    )

    MobileManipulationRuntime._publish_place_request(harness, object())

    payload = json.loads(harness._place_region_pub.messages[-1].data)
    assert (
        payload['request_id'],
        payload['producer_epoch'],
        payload['generation'],
    ) == ('place-request', 'bridge-after-carry', 9)
    assert payload['object_reference_identity'] == reference_identity
    assert payload['request_id'] != payload['object_reference_identity']['request_id']
    assert payload['producer_epoch'] != payload['object_reference_identity']['producer_epoch']
    assert payload['generation'] != payload['object_reference_identity']['generation']


def test_grounding_request_clears_geometry_and_binds_only_a_future_generation() -> None:
    harness = _complete_harness()
    harness._perception_generation = 9
    harness._grounding_pub = _Publisher()

    MobileManipulationRuntime._publish_grounding_request(harness)

    assert harness._required_perception_generation is None
    assert harness._required_affordance_generation is None
    assert harness._required_perception_request_id is not None
    assert harness._required_perception_request_id != REQUEST_ID
    assert harness._bound_perception_generation is None
    assert harness._bound_perception_request_id is None
    assert harness._bound_perception_producer_epoch is None
    assert harness._valid_perception_generation is None
    assert not harness._perception_valid
    assert harness._affordance is None
    assert harness._target_camera is None
    assert harness._target_cloud is None
    assert harness._scene_cloud is None
    assert harness._target_stamp_ns == 0
    assert harness._target_cloud_stamp_ns == 0
    assert harness._scene_cloud_stamp_ns == 0
    assert harness._serial_gate.snapshot(harness._now_s()) is None
    assert len(harness._grounding_pub.messages) == 1
    request = json.loads(harness._grounding_pub.messages[0].data)
    assert request == {
        'schema': 'z_manip.grounding_request.v2',
        'request_id': harness._required_perception_request_id,
        'instruction': 'pick the observed bottle',
        'scope': 'grasp_for_place',
    }


@pytest.mark.parametrize(
    ('phase', 'place_mode', 'expected_scope'),
    (
        (RuntimePhase.GROUNDING, 'carry_only', 'grasp_only'),
        (RuntimePhase.GROUNDING, 'observed_place', 'grasp_for_place'),
        (RuntimePhase.PLACE_GROUNDING, 'observed_place', 'place_support'),
    ),
)
def test_grounding_request_scope_is_explicitly_bound_to_task_stage(
    phase: RuntimePhase,
    place_mode: str,
    expected_scope: str,
) -> None:
    harness = _complete_harness()
    harness._core.phase = phase
    harness._grounding_pub = _Publisher()
    harness._topic_value = lambda _parameter: place_mode

    MobileManipulationRuntime._publish_grounding_request(harness)

    request = json.loads(harness._grounding_pub.messages[-1].data)
    assert request['scope'] == expected_scope
    assert harness._required_grounding_scope == expected_scope


def test_servo_completion_starts_fresh_reground_without_old_gate_snapshot() -> None:
    harness = _Harness()
    harness._core.mark_standoff(1)
    harness._core.mark_coarse_ready()
    harness._core.mark_near_grounded(1)
    assert harness._serial_gate.snapshot(harness._now_s()) is None

    harness._approach = SimpleNamespace(update=lambda _value: SimpleNamespace(
        phase=task_node.ApproachPhase.COMPLETE,
        cancel_navigation=False,
        servo=SimpleNamespace(linear_x=0.0, angular_z=0.0),
        reason='',
    ))
    harness._target_camera = np.array([0.0, 0.0, 0.5])
    harness._nav_speed = 0.0
    harness._roll = 0.0
    harness._pitch = 0.0
    harness._desired_depth = 0.5
    harness._perception_valid = True
    harness._config = SimpleNamespace(
        robot=SimpleNamespace(platform_base_frame='base_link'),
    )
    harness._cancel_nav_pub = _Publisher()
    harness._velocity_pub = _Publisher()
    harness.get_clock = lambda: task_node.Clock()
    zero_commands = []
    harness._publish_zero = lambda: zero_commands.append(True)
    stage_results = []
    harness._task = SimpleNamespace(
        stage=SimpleNamespace(value='visual_approach'),
        apply=stage_results.append,
    )
    harness._grounding_pub = _Publisher()
    harness._tracked_target_edge_directions = lambda: (0, 0)
    handoff_stationary = [False]
    harness._coarse_nav_arrival_is_stationary = (
        lambda _now: handoff_stationary[0]
    )
    harness._coarse_nav_arrival_started_at_s = 10.0
    harness._coarse_nav_arrival_stable_since_s = None
    harness._coarse_nav_arrival_stable_start_odom_stamp_ns = None
    harness._coarse_nav_arrival_last_odom_sequence = None
    harness._coarse_nav_arrival_last_odom_stamp_ns = None

    MobileManipulationRuntime._visual_servo(harness, harness._now_s())

    assert zero_commands == [True]
    assert harness._core.phase is RuntimePhase.VISUAL_SERVO
    assert stage_results == []
    assert harness._grounding_pub.messages == []

    handoff_stationary[0] = True
    MobileManipulationRuntime._visual_servo(harness, harness._now_s() + 0.4)

    assert zero_commands == [True, True]
    assert harness._core.phase is RuntimePhase.FINAL_GROUNDING
    assert harness._core.required_replan_serial == 1
    assert len(stage_results) == 1
    assert len(harness._grounding_pub.messages) == 1
    assert harness._required_perception_request_id != REQUEST_ID
    assert harness._serial_gate.snapshot(harness._now_s()) is None
    assert harness._coarse_nav_arrival_started_at_s is None
    assert harness.actions == []


def _vertical_recovery_harness():
    harness = _Harness()
    harness._core.mark_standoff(1)
    harness._core.mark_coarse_ready()
    harness._core.mark_near_grounded(1)
    harness._visual_search = SimpleNamespace(
        config=SimpleNamespace(stationary_wait_timeout_s=1.0),
    )
    harness._visual_servo_vertical_stationarity = (
        task_core.ContinuousMotionQuietWindow(
            quiet_window_s=0.20,
            max_odom_gap_s=0.15,
            max_linear_speed_mps=0.035,
            max_angular_speed_rps=0.05,
        )
    )
    harness._visual_servo_vertical_recovery_started_at_s = None
    harness._visual_servo_vertical_minimum_cloud_stamp_ns = None
    harness._visual_servo_vertical_safe_start_cloud_stamp_ns = None
    harness._visual_servo_vertical_safe_last_cloud_stamp_ns = None
    harness._odom_sequence = 5
    harness._odom_stamp_ns = 10_000_000_000
    harness._odom_seen_at = 10.0
    harness._max_perception_age_s = 0.15
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_000_000_000,
    )
    visual_servo = _Resettable()
    harness._approach = SimpleNamespace(
        visual_servo=visual_servo,
        update=lambda _value: pytest.fail('edge guard must run before control'),
    )
    zero_commands = []
    harness._publish_zero = lambda: zero_commands.append(True)
    harness.get_parameter = lambda name: SimpleNamespace(value={
        'visual_servo_image_margin_ratio': 0.02,
        'visual_search_vertical_margin_ratio': 0.06,
        'max_perception_age_s': harness._max_perception_age_s,
    }[name])
    return harness, visual_servo, zero_commands


def _set_vertical_recovery_observation(
    harness,
    *,
    observation_stamp_ns: int,
    target_stamp_ns: int | None = None,
    cloud_stamp_ns: int | None = None,
    valid_frame_id: str = FRAME_ID,
    target_frame_id: str | None = None,
    cloud_frame_id: str | None = None,
) -> None:
    target_stamp = (
        observation_stamp_ns if target_stamp_ns is None else target_stamp_ns
    )
    cloud_stamp = (
        observation_stamp_ns if cloud_stamp_ns is None else cloud_stamp_ns
    )
    target_frame = valid_frame_id if target_frame_id is None else target_frame_id
    cloud_frame = valid_frame_id if cloud_frame_id is None else cloud_frame_id
    harness._perception_valid = True
    harness._valid_observation_stamp_ns = observation_stamp_ns
    harness._valid_observation_frame_id = valid_frame_id
    harness._target_stamp_ns = target_stamp
    harness._target_frame_id = target_frame
    harness._target_cloud_stamp_ns = cloud_stamp
    harness._target_cloud_frame_id = cloud_frame
    harness._target_uv = np.array(((200.0, 100.0), (300.0, 200.0)))


def _record_vertical_recovery_odom(
    harness,
    *,
    received_at_s: float,
    source_stamp_ns: int,
    linear_speed_mps: float,
    angular_speed_rps: float,
) -> None:
    harness._odom_sequence += 1
    harness._odom_stamp_ns = source_stamp_ns
    harness._odom_seen_at = received_at_s
    MobileManipulationRuntime._record_visual_servo_vertical_motion_sample(
        harness,
        received_at_s=received_at_s,
        odom_sequence=harness._odom_sequence,
        odom_stamp_ns=source_stamp_ns,
        linear_speed_mps=linear_speed_mps,
        angular_speed_rps=angular_speed_rps,
    )


def test_visual_servo_vertical_edge_holds_zero_until_fresh_safe_and_quiet() -> None:
    harness, visual_servo, zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )

    MobileManipulationRuntime._visual_servo(harness, 10.0)

    assert zero_commands == [True]
    assert harness._core.phase is RuntimePhase.VISUAL_SERVO
    assert harness.recoveries == []
    assert visual_servo.calls == 1
    assert MobileManipulationRuntime._visual_servo_vertical_recovery_state_complete(
        harness,
    )

    # A fresh safe mask alone cannot resume while measured motion is above the
    # platform limits.
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_100_000_000,
    )
    _record_vertical_recovery_odom(
        harness,
        received_at_s=10.1,
        source_stamp_ns=10_100_000_000,
        linear_speed_mps=0.08,
        angular_speed_rps=0.04,
    )
    MobileManipulationRuntime._visual_servo(harness, 10.1)
    assert harness._visual_servo_vertical_recovery_started_at_s == 10.0

    # Motion reset the odometry quiet window. Both source clocks must now cover
    # one complete quiet window after the target is safely inside the margin.
    for now_s, stamp_ns in (
        (10.2, 10_200_000_000),
        (10.3, 10_300_000_000),
        (10.4, 10_400_000_000),
    ):
        _set_vertical_recovery_observation(
            harness,
            observation_stamp_ns=stamp_ns,
        )
        _record_vertical_recovery_odom(
            harness,
            received_at_s=now_s,
            source_stamp_ns=stamp_ns,
            linear_speed_mps=0.01,
            angular_speed_rps=0.02,
        )
        MobileManipulationRuntime._visual_servo(harness, now_s)

    assert harness._visual_servo_vertical_recovery_started_at_s is None
    assert harness.recoveries == []
    assert len(zero_commands) == 5


def test_vertical_recovery_arms_while_exact_bundle_topics_are_split() -> None:
    harness, visual_servo, zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )
    # The authenticated tracking status advanced before either geometry topic.
    # This is a normal cross-topic DDS ordering, not a failed observation.
    harness._valid_observation_stamp_ns = 10_100_000_000

    MobileManipulationRuntime._visual_servo(harness, 10.1)

    assert zero_commands == [True]
    assert visual_servo.calls == 1
    assert harness.recoveries == []
    assert harness._visual_servo_vertical_recovery_started_at_s == 10.1
    assert (
        harness._visual_servo_vertical_minimum_cloud_stamp_ns
        == 10_100_000_000
    )

    for now_s, stamp_ns in (
        (10.2, 10_200_000_000),
        (10.3, 10_300_000_000),
        (10.4, 10_400_000_000),
    ):
        _set_vertical_recovery_observation(
            harness,
            observation_stamp_ns=stamp_ns,
        )
        _record_vertical_recovery_odom(
            harness,
            received_at_s=now_s,
            source_stamp_ns=stamp_ns,
            linear_speed_mps=0.01,
            angular_speed_rps=0.01,
        )
        MobileManipulationRuntime._visual_servo(harness, now_s)

    assert harness._visual_servo_vertical_recovery_started_at_s is None
    assert harness.recoveries == []


def test_vertical_recovery_fences_partial_source_stamps() -> None:
    harness, _visual_servo, _zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )
    harness._target_stamp_ns = 10_100_000_000
    harness._target_cloud_stamp_ns = 10_200_000_000

    MobileManipulationRuntime._visual_servo(harness, 10.2)

    assert (
        harness._visual_servo_vertical_minimum_cloud_stamp_ns
        == 10_200_000_000
    )
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_200_000_000,
    )
    _record_vertical_recovery_odom(
        harness,
        received_at_s=10.2,
        source_stamp_ns=10_200_000_000,
        linear_speed_mps=0.01,
        angular_speed_rps=0.01,
    )
    MobileManipulationRuntime._visual_servo(harness, 10.2)

    assert harness._visual_servo_vertical_safe_start_cloud_stamp_ns is None
    assert harness._visual_servo_vertical_safe_last_cloud_stamp_ns is None
    assert harness._visual_servo_vertical_recovery_started_at_s == 10.2
    assert harness.recoveries == []


def test_visual_servo_vertical_recovery_timeout_uses_coarse_nav_budget() -> None:
    harness, _visual_servo, zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **_kwargs: (0, -1)
    harness._recover_precontact = lambda kind, detail: (
        harness.recoveries.append((kind, detail)) or True
    )

    MobileManipulationRuntime._visual_servo(harness, 10.0)
    MobileManipulationRuntime._visual_servo(harness, 11.01)

    assert len(zero_commands) == 2
    assert harness._visual_servo_vertical_recovery_started_at_s is None
    assert (
        harness.recoveries[0][0]
        is task_node.FailureKind.VISUAL_APPROACH_FAILED
    )
    assert 'timed out' in harness.recoveries[0][1]


def test_vertical_recovery_does_not_advance_from_new_detection_on_stale_cloud(
) -> None:
    harness, _visual_servo, _zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )

    MobileManipulationRuntime._visual_servo(harness, 10.0)
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_100_000_000,
        cloud_stamp_ns=10_000_000_000,
    )
    _record_vertical_recovery_odom(
        harness,
        received_at_s=10.1,
        source_stamp_ns=10_100_000_000,
        linear_speed_mps=0.01,
        angular_speed_rps=0.01,
    )
    MobileManipulationRuntime._visual_servo(harness, 10.1)

    assert harness._visual_servo_vertical_recovery_started_at_s == 10.0
    assert harness._visual_servo_vertical_safe_start_cloud_stamp_ns is None
    assert harness._visual_servo_vertical_safe_last_cloud_stamp_ns is None
    assert harness.recoveries == []


def test_vertical_recovery_rejects_current_observation_identity_mismatch() -> None:
    harness, _visual_servo, _zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )

    MobileManipulationRuntime._visual_servo(harness, 10.0)
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_100_000_000,
        cloud_frame_id='unexpected_depth_frame',
    )
    MobileManipulationRuntime._visual_servo(harness, 10.1)

    assert harness._visual_servo_vertical_recovery_started_at_s == 10.0
    assert harness._visual_servo_vertical_safe_start_cloud_stamp_ns is None
    assert harness._visual_servo_vertical_safe_last_cloud_stamp_ns is None
    assert harness.recoveries == []


def test_vertical_recovery_restarts_safe_window_after_cloud_gap() -> None:
    harness, _visual_servo, _zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )

    MobileManipulationRuntime._visual_servo(harness, 10.0)
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_100_000_000,
    )
    MobileManipulationRuntime._visual_servo(harness, 10.1)
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_300_000_000,
    )
    MobileManipulationRuntime._visual_servo(harness, 10.3)

    assert (
        harness._visual_servo_vertical_safe_start_cloud_stamp_ns
        == 10_300_000_000
    )
    assert (
        harness._visual_servo_vertical_safe_last_cloud_stamp_ns
        == 10_300_000_000
    )
    assert harness._visual_servo_vertical_recovery_started_at_s == 10.0


def test_vertical_recovery_does_not_reuse_ready_odometry_after_freeze() -> None:
    harness, _visual_servo, _zero_commands = _vertical_recovery_harness()
    harness._max_perception_age_s = 0.35
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )

    MobileManipulationRuntime._visual_servo(harness, 10.0)
    for now_s, stamp_ns in (
        (10.1, 10_100_000_000),
        (10.2, 10_200_000_000),
        (10.3, 10_300_000_000),
    ):
        _set_vertical_recovery_observation(
            harness,
            observation_stamp_ns=stamp_ns,
        )
        MobileManipulationRuntime._visual_servo(harness, now_s)
    for now_s, stamp_ns in (
        (10.1, 10_100_000_000),
        (10.2, 10_200_000_000),
        (10.3, 10_300_000_000),
    ):
        _record_vertical_recovery_odom(
            harness,
            received_at_s=now_s,
            source_stamp_ns=stamp_ns,
            linear_speed_mps=0.01,
            angular_speed_rps=0.01,
        )
    assert harness._visual_servo_vertical_stationarity.ready(
        odom_sequence=harness._odom_sequence,
        odom_stamp_ns=harness._odom_stamp_ns,
        odom_seen_at_s=harness._odom_seen_at,
    )

    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=10_400_000_000,
    )
    MobileManipulationRuntime._visual_servo(harness, 10.46)

    assert harness._visual_servo_vertical_recovery_started_at_s == 10.0
    assert harness.recoveries == []


def test_vertical_recovery_timeout_wins_over_late_ready_evidence() -> None:
    harness, _visual_servo, _zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **kwargs: (
        (0, -1) if not kwargs else (0, 0)
    )
    harness._recover_precontact = lambda kind, detail: (
        harness.recoveries.append((kind, detail)) or True
    )

    MobileManipulationRuntime._visual_servo(harness, 10.0)
    for now_s, stamp_ns in (
        (10.7, 10_700_000_000),
        (10.8, 10_800_000_000),
        (10.9, 10_900_000_000),
    ):
        _set_vertical_recovery_observation(
            harness,
            observation_stamp_ns=stamp_ns,
        )
        MobileManipulationRuntime._visual_servo(harness, now_s)
        _record_vertical_recovery_odom(
            harness,
            received_at_s=now_s,
            source_stamp_ns=stamp_ns,
            linear_speed_mps=0.01,
            angular_speed_rps=0.01,
        )
    _record_vertical_recovery_odom(
        harness,
        received_at_s=11.0,
        source_stamp_ns=11_000_000_000,
        linear_speed_mps=0.01,
        angular_speed_rps=0.01,
    )
    _set_vertical_recovery_observation(
        harness,
        observation_stamp_ns=11_000_000_000,
    )
    MobileManipulationRuntime._visual_servo(harness, 11.0)

    assert harness._visual_servo_vertical_recovery_started_at_s is None
    assert (
        harness.recoveries[0][0]
        is task_node.FailureKind.VISUAL_APPROACH_FAILED
    )
    assert 'timed out' in harness.recoveries[0][1]


def test_vertical_recovery_exemption_requires_visual_servo_phase() -> None:
    harness, _visual_servo, _zero_commands = _vertical_recovery_harness()
    harness._tracked_target_edge_directions = lambda **_kwargs: (0, -1)

    MobileManipulationRuntime._visual_servo(harness, 10.0)
    assert MobileManipulationRuntime._visual_servo_vertical_recovery_state_complete(
        harness,
    )

    harness._core.phase = RuntimePhase.NEAR_GROUNDING
    assert not MobileManipulationRuntime._visual_servo_vertical_recovery_state_complete(
        harness,
    )


def test_live_mask_edge_gate_uses_pointcloud_image_coordinates() -> None:
    harness = SimpleNamespace(
        _target_uv=np.array(((220.0, 365.0), (265.0, 479.0))),
        _image_size=(848, 480),
        get_parameter=lambda name: SimpleNamespace(value={
            'visual_servo_image_margin_ratio': 0.02,
        }[name]),
    )

    assert MobileManipulationRuntime._tracked_target_edge_directions(harness) == (
        0,
        -1,
    )

    harness._target_uv = np.array(((520.0, 220.0), (580.0, 380.0)))
    assert MobileManipulationRuntime._tracked_target_edge_directions(harness) == (
        0,
        0,
    )


def test_live_mask_edge_gate_requires_image_coordinates() -> None:
    harness = SimpleNamespace(
        _target_uv=None,
        _image_size=(848, 480),
    )

    with pytest.raises(
        ValueError,
        match='pointcloud lacks u/v image coordinates',
    ):
        MobileManipulationRuntime._tracked_target_edge_directions(harness)


def test_target_pointcloud_contract_requires_uv_fields(monkeypatch) -> None:
    message = SimpleNamespace(
        fields=[SimpleNamespace(name=name) for name in ('x', 'y', 'z')],
    )
    monkeypatch.setattr(
        task_node.point_cloud2,
        'read_points',
        lambda *_args, **_kwargs: pytest.fail(
            'malformed target cloud must be rejected before decoding',
        ),
    )

    with pytest.raises(ValueError, match='missing required u/v fields'):
        MobileManipulationRuntime._read_cloud(message, need_uv=True)
