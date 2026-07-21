import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest


rclpy = pytest.importorskip('rclpy')

from std_msgs.msg import Bool, String  # noqa: E402

from z_manip.orchestration.mobile_manipulation import (  # noqa: E402
    MobileManipulationStateMachine,
)
from z_manip.planning.work_pose import WorkPoseDiagnostics  # noqa: E402
from z_manip_task.core import (  # noqa: E402
    ObservationSerialGate,
    RuntimePhase,
    RuntimeSafetyCore,
    TaskGenerationGuard,
)
from z_manip_task.node import (  # noqa: E402
    _compose_se2,
    _PlanningBaseAnchor,
    _PlanningObservationChanged,
    _PlanningObservationIdentity,
    _PlanningObservationWait,
    _relative_se2,
    MobileManipulationRuntime,
)


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, value):
        self.errors.append(str(value))


def _diagnostics():
    return WorkPoseDiagnostics(
        sampled_hypotheses=3,
        geometric_candidates=2,
        ranked_candidates=2,
        exact_evaluations=1,
        feasible_candidates=1,
        rejection_counts=(),
        failures=(),
        sample_budget_exhausted=False,
        exact_budget_exhausted=False,
    )


class _CompletedFuture:
    def __init__(self, result):
        self.value = result
        self.result_calls = 0
        self.cancel_calls = 0

    @staticmethod
    def done():
        return True

    def result(self):
        self.result_calls += 1
        return self.value

    def cancel(self):
        self.cancel_calls += 1
        return False


class _CompletedWorkPoseHarness:
    _validate_completed_work_pose_observation = (
        MobileManipulationRuntime._validate_completed_work_pose_observation
    )
    _defer_completed_work_pose_observation = (
        MobileManipulationRuntime._defer_completed_work_pose_observation
    )
    _grounding_observation_authorized = (
        MobileManipulationRuntime._grounding_observation_authorized
    )
    _maybe_finish_coarse_nav = MobileManipulationRuntime._maybe_finish_coarse_nav

    def __init__(self):
        self.now = 10.05
        self.request_id = 'request-4'
        self.producer_epoch = 'producer-a'
        self.generation = 3
        self.frame_id = 'wrist_camera'
        self.stamp_ns = 10_000_000_000
        self.initial_target = np.array((0.1, -0.2, 1.4))
        result = SimpleNamespace(
            relative_base_pose=np.array((1.0, 0.0, math.pi / 2.0)),
            desired_camera_depth_m=0.54,
            predicted_target_position_piper=np.array((0.66, 0.0, 0.05)),
            selection_mode='kinematic_precheck',
            kinematic_precheck_feasible=True,
            diagnostics=_diagnostics(),
            rejected_precheck_diagnostics=None,
        )
        self.future = _CompletedFuture(result)
        self._future = self.future
        self._future_kind = 'standoff'
        self._future_serial = 9
        self._task_generation = TaskGenerationGuard()
        self._future_generation = self._task_generation.current
        self._future_cancel_event = threading.Event()
        self._future_base_anchor = _PlanningBaseAnchor(
            (0.0, 0.0, 0.0), 7, 9_900_000_000,
        )
        self._future_observation_identity = _PlanningObservationIdentity(
            self.request_id,
            self.producer_epoch,
            self.generation,
            self.stamp_ns,
            self.frame_id,
            tuple(self.initial_target),
        )
        self._future_observation_wait = None

        self._required_perception_request_id = self.request_id
        self._required_perception_generation = self.generation
        self._required_affordance_generation = self.generation
        self._bound_perception_request_id = self.request_id
        self._bound_perception_producer_epoch = self.producer_epoch
        self._bound_perception_generation = self.generation
        self._valid_perception_request_id = self.request_id
        self._valid_perception_producer_epoch = self.producer_epoch
        self._valid_perception_generation = self.generation
        self._valid_observation_stamp_ns = self.stamp_ns
        self._valid_observation_frame_id = self.frame_id
        self._perception_valid = True
        self._affordance = {'target': {}}
        self._affordance_request_id = self.request_id
        self._affordance_producer_epoch = self.producer_epoch
        self._affordance_generation = self.generation

        self._target_camera = self.initial_target.copy()
        self._target_frame_id = self.frame_id
        self._target_stamp_ns = self.stamp_ns
        self._target_cloud = np.ones((3, 3))
        self._target_cloud_frame_id = self.frame_id
        self._target_cloud_stamp_ns = self.stamp_ns
        self._scene_cloud = np.ones((4, 3))
        self._scene_cloud_frame_id = self.frame_id
        self._scene_cloud_stamp_ns = self.stamp_ns
        self._serial_gate = ObservationSerialGate(sync_slop_s=1e-6, max_age_s=0.35)
        for stream in ('target', 'target_cloud', 'scene_cloud'):
            self._serial_gate.update(stream, self.stamp_ns * 1e-9)

        self._position_xy = (0.0, 0.0)
        self._yaw = 0.0
        self._desired_depth = None
        self._approximate_displacement = None
        self._coarse_nav_ready = False
        self._work_pose = None
        self._work_pose_history_map = []
        self._task = MobileManipulationStateMachine()
        self._core = RuntimeSafetyCore()
        self._core.instruction = 'pick the mustard bottle'
        self._core.phase = RuntimePhase.STANDOFF
        self.recoveries = []
        self.safety_actions = []

    def _now_s(self):
        return self.now

    @staticmethod
    def get_parameter(name):
        values = {
            'perception_loss_timeout_s': 0.60,
            'work_pose_target_drift_tolerance_m': 0.06,
            'work_pose_anchor_translation_tolerance_m': 0.05,
            'work_pose_anchor_yaw_tolerance_rad': 0.05,
            'platform_odometry_parent_frame': 'map',
        }
        return SimpleNamespace(value=values[name])

    def _recover_precontact(self, kind, detail):
        self.recoveries.append((kind, detail))
        return True

    def _apply_safety(self, action):
        self.safety_actions.append(action)

    def begin_partial_bundle(self, stamp_ns, target=None):
        self.now = stamp_ns * 1e-9 + 0.05
        self._valid_observation_stamp_ns = stamp_ns
        self._target_stamp_ns = stamp_ns
        self._target_camera = (
            self.initial_target + np.array((0.001, 0.0, 0.0))
            if target is None
            else np.asarray(target, dtype=float)
        )
        self._serial_gate.update('target', stamp_ns * 1e-9)

    def complete_partial_bundle(self, stamp_ns):
        self._target_cloud_stamp_ns = stamp_ns
        self._scene_cloud_stamp_ns = stamp_ns
        self._serial_gate.update('target_cloud', stamp_ns * 1e-9)
        self._serial_gate.update('scene_cloud', stamp_ns * 1e-9)


def test_se2_map_composition_and_relative_inverse_include_yaw():
    anchor = np.array((10.0, 20.0, math.pi / 2.0))
    relative = np.array((1.0, -0.5, math.pi / 3.0))

    goal = _compose_se2(anchor, relative)

    assert goal == pytest.approx((10.5, 21.0, 5.0 * math.pi / 6.0))
    assert _relative_se2(anchor, goal) == pytest.approx(relative)


def test_completed_work_pose_plan_is_anchored_and_correlated_for_navigation():
    result = SimpleNamespace(
        relative_base_pose=np.array((1.0, 0.0, math.pi / 2.0)),
        desired_camera_depth_m=0.54,
        predicted_target_position_piper=np.array((0.66, 0.0, 0.05)),
        selection_mode='kinematic_precheck',
        kinematic_precheck_feasible=True,
        diagnostics=_diagnostics(),
        rejected_precheck_diagnostics=None,
    )

    class _Future:
        def done(self):
            return True

        def result(self):
            return result

    class _Harness:
        def __init__(self):
            self._future = _Future()
            self._future_kind = 'standoff'
            self._future_serial = 9
            self._task_generation = TaskGenerationGuard()
            self._future_generation = self._task_generation.current
            self._future_cancel_event = threading.Event()
            self._future_base_anchor = _PlanningBaseAnchor(
                (10.0, 20.0, math.pi / 2.0), 7, 1234,
            )
            self._future_observation_identity = _PlanningObservationIdentity(
                'request-4', 'producer-a', 3, 9876, 'wrist_camera',
                (0.1, -0.2, 1.4),
            )
            self._position_xy = (10.0, 20.0)
            self._yaw = math.pi / 2.0
            self._desired_depth = None
            self._approximate_displacement = None
            self._coarse_nav_ready = False
            self._work_pose = None
            self._work_pose_history_map = []
            self._bound_perception_request_id = 'request-4'
            self._bound_perception_producer_epoch = 'producer-a'
            self._bound_perception_generation = 3
            self._valid_observation_stamp_ns = 9999
            self._task = MobileManipulationStateMachine()
            self._core = RuntimeSafetyCore()
            self._core.instruction = 'pick the mustard bottle'
            self._core.phase = RuntimePhase.STANDOFF

        def get_parameter(self, name):
            values = {
                'work_pose_anchor_translation_tolerance_m': 0.05,
                'work_pose_anchor_yaw_tolerance_rad': 0.05,
                'platform_odometry_parent_frame': 'map',
            }
            return SimpleNamespace(value=values[name])

        @staticmethod
        def _now_s():
            return 5.0

        def _validate_completed_work_pose_observation(self, identity):
            assert identity is self._future_observation_identity_snapshot

        def _maybe_finish_coarse_nav(self):
            MobileManipulationRuntime._maybe_finish_coarse_nav(self)

        def _recover_precontact(self, *_args):
            raise AssertionError('valid work-pose result must not recover')

        def _apply_safety(self, _action):
            raise AssertionError('valid work-pose result must not fail')

    harness = _Harness()
    harness._future_observation_identity_snapshot = harness._future_observation_identity

    MobileManipulationRuntime._poll_planning(harness)

    assert harness._core.phase is RuntimePhase.COARSE_NAV
    assert harness._task.stage.value == 'coarse_nav'
    assert harness._work_pose['map_goal_xy'] == pytest.approx((10.0, 21.0))
    assert abs(harness._work_pose['map_goal_yaw_rad']) == pytest.approx(math.pi)
    assert harness._work_pose['source']['observation_serial'] == 9
    assert harness._work_pose['source']['request_id'] == 'request-4'
    assert harness._work_pose['source']['observation_stamp_ns'] == 9876
    assert harness._work_pose['source']['observation_frame_id'] == 'wrist_camera'
    assert len(harness._work_pose_history_map) == 0
    assert not harness._navigation_goal_acknowledged
    assert not harness._coarse_nav_ready


def test_completed_work_pose_waits_for_partial_next_bundle_then_commits_once():
    harness = _CompletedWorkPoseHarness()
    next_stamp_ns = harness.stamp_ns + 100_000_000
    future = harness._future
    anchor = harness._future_base_anchor
    identity = harness._future_observation_identity
    harness.begin_partial_bundle(next_stamp_ns)

    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is future
    assert harness._future_base_anchor is anchor
    assert harness._future_observation_identity is identity
    wait = harness._future_observation_wait
    assert isinstance(wait, _PlanningObservationWait)
    assert wait.started_at_s == pytest.approx(10.15)
    assert wait.deadline_s == pytest.approx(10.75)
    assert harness._core.phase is RuntimePhase.STANDOFF
    assert harness._work_pose is None
    assert harness.recoveries == []

    harness.complete_partial_bundle(next_stamp_ns)
    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is None
    assert harness._future_observation_wait is None
    assert harness._core.phase is RuntimePhase.COARSE_NAV
    assert harness._work_pose['source']['observation_serial'] == 9
    assert harness._work_pose['source']['observation_stamp_ns'] == harness.stamp_ns
    assert future.result_calls == 2
    assert harness.recoveries == []

    MobileManipulationRuntime._poll_planning(harness)
    assert future.result_calls == 2


def test_completed_work_pose_partial_bundle_timeout_recovers_exactly_once():
    harness = _CompletedWorkPoseHarness()
    harness.begin_partial_bundle(harness.stamp_ns + 100_000_000)

    MobileManipulationRuntime._poll_planning(harness)
    wait = harness._future_observation_wait
    assert isinstance(wait, _PlanningObservationWait)
    assert wait.started_at_s == pytest.approx(10.15)
    assert wait.deadline_s == pytest.approx(10.75)

    harness.now = 10.749
    MobileManipulationRuntime._poll_planning(harness)
    assert harness._future is harness.future
    assert harness._future_observation_wait is wait
    assert harness.recoveries == []

    harness.now = wait.deadline_s
    MobileManipulationRuntime._poll_planning(harness)
    assert harness._future is None
    assert harness._future_observation_wait is None
    assert len(harness.recoveries) == 1
    assert harness.recoveries[0][0].value == 'target_lost'
    assert 'timed out after 0.600s' in harness.recoveries[0][1]

    MobileManipulationRuntime._poll_planning(harness)
    assert len(harness.recoveries) == 1


def test_completed_work_pose_late_valid_bundle_cannot_bypass_fixed_wait():
    harness = _CompletedWorkPoseHarness()
    harness.begin_partial_bundle(harness.stamp_ns + 100_000_000)
    MobileManipulationRuntime._poll_planning(harness)
    wait = harness._future_observation_wait
    assert isinstance(wait, _PlanningObservationWait)

    late_stamp_ns = int(round((wait.deadline_s + 0.05) * 1e9))
    harness.begin_partial_bundle(late_stamp_ns)
    harness.complete_partial_bundle(late_stamp_ns)
    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is None
    assert harness._future_observation_wait is None
    assert len(harness.recoveries) == 1
    assert harness.recoveries[0][0].value == 'target_lost'
    assert 'timed out after 0.600s' in harness.recoveries[0][1]

    MobileManipulationRuntime._poll_planning(harness)
    assert len(harness.recoveries) == 1


@pytest.mark.parametrize(
    ('field', 'value', 'detail'),
    (
        ('_bound_perception_request_id', 'request-other', 'ownership'),
        ('_bound_perception_producer_epoch', 'producer-other', 'ownership'),
        ('_bound_perception_generation', 4, 'generation'),
        ('_valid_observation_frame_id', 'another_camera', 'frame'),
    ),
)
def test_completed_work_pose_identity_change_rejects_without_wait(
    field,
    value,
    detail,
):
    harness = _CompletedWorkPoseHarness()
    harness.begin_partial_bundle(harness.stamp_ns + 100_000_000)
    setattr(harness, field, value)

    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is None
    assert harness._future_observation_wait is None
    assert len(harness.recoveries) == 1
    assert harness.recoveries[0][0].value == 'target_lost'
    assert detail in harness.recoveries[0][1]


def test_completed_work_pose_coherent_target_drift_rejects_immediately():
    harness = _CompletedWorkPoseHarness()
    next_stamp_ns = harness.stamp_ns + 100_000_000
    harness.begin_partial_bundle(
        next_stamp_ns,
        harness.initial_target + np.array((0.061, 0.0, 0.0)),
    )
    harness.complete_partial_bundle(next_stamp_ns)

    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is None
    assert harness._future_observation_wait is None
    assert len(harness.recoveries) == 1
    assert 'target moved' in harness.recoveries[0][1]
    assert 'drift=0.061m' in harness.recoveries[0][1]


def test_completed_work_pose_pending_wait_is_cleared_by_async_invalidation():
    harness = _CompletedWorkPoseHarness()
    harness.begin_partial_bundle(harness.stamp_ns + 100_000_000)
    MobileManipulationRuntime._poll_planning(harness)
    generation = harness._task_generation.current

    MobileManipulationRuntime._invalidate_async_work(harness)

    assert harness._future_cancel_event is None
    assert harness._future is None
    assert harness._future_base_anchor is None
    assert harness._future_observation_identity is None
    assert harness._future_observation_wait is None
    assert harness.future.cancel_calls == 1
    assert harness._task_generation.current == generation + 1


def test_completed_work_pose_pending_wait_rejects_clock_rollback():
    harness = _CompletedWorkPoseHarness()
    harness.begin_partial_bundle(harness.stamp_ns + 100_000_000)
    MobileManipulationRuntime._poll_planning(harness)
    harness.now = 10.149

    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is None
    assert len(harness.recoveries) == 1
    assert 'clock moved backwards' in harness.recoveries[0][1]


def test_completed_work_pose_valid_bundle_rejects_clock_rollback_once():
    harness = _CompletedWorkPoseHarness()
    next_stamp_ns = harness.stamp_ns + 100_000_000
    harness.begin_partial_bundle(next_stamp_ns)
    MobileManipulationRuntime._poll_planning(harness)
    wait = harness._future_observation_wait
    assert isinstance(wait, _PlanningObservationWait)
    harness.complete_partial_bundle(next_stamp_ns)
    harness.now = wait.started_at_s - 0.001

    MobileManipulationRuntime._poll_planning(harness)

    assert harness._future is None
    assert harness._future_observation_wait is None
    assert len(harness.recoveries) == 1
    assert 'clock moved backwards' in harness.recoveries[0][1]

    MobileManipulationRuntime._poll_planning(harness)
    assert len(harness.recoveries) == 1


def test_explicit_work_pose_requires_matching_navigation_status_not_legacy_bool():
    source = {
        'request_id': 'request-4',
        'producer_epoch': 'producer-a',
        'generation': 3,
        'odom_sequence': 4,
        'odom_stamp_ns': 11_800_000_000,
    }

    class _Harness:
        def __init__(self):
            self._lock = threading.RLock()
            self._core = RuntimeSafetyCore()
            self._core.phase = RuntimePhase.COARSE_NAV
            self._work_pose = {
                'goal_id': 'work-9',
                'map_goal_xy': [1.25, -0.75],
                'map_goal_yaw_rad': 0.2,
                'map_frame': 'map',
                'anchor_map_pose': [1.0, -0.75, 0.0],
                'source': source,
            }
            self._coarse_nav_ready = False
            self._position_xy = (1.25, -0.75)
            self._odom_seen_at = 11.9
            self._odom_stamp_ns = 11_900_000_000
            self._odom_sequence = 5
            self._navigation_status_seen_s = None
            self._navigation_status_goal_id = ''
            self._navigation_status_phase = ''
            self._navigation_goal_acknowledged = False
            self._navigation_ack_position_xy = None
            self._navigation_ack_odom_sequence = None
            self._navigation_ack_odom_stamp_ns = None
            self._navigation_history_recorded = False
            self._work_pose_history_map = []
            self._logger = _Logger()

        def get_logger(self):
            return self._logger

        @staticmethod
        def _now_s():
            return 12.0

        def get_parameter(self, name):
            values = {
                'work_pose_goal_tolerance_m': 0.2,
                'work_pose_odom_max_age_s': 0.5,
                'work_pose_history_min_displacement_m': 0.03,
            }
            return SimpleNamespace(value=values[name])

        def _validate_navigation_ready_pose(self, now):
            MobileManipulationRuntime._validate_navigation_ready_pose(self, now)

    harness = _Harness()
    MobileManipulationRuntime._coarse_ready_cb(harness, Bool(data=True))
    assert not harness._coarse_nav_ready

    stale = {
        'schema': 'z_manip.navigation_status.v1',
        'phase': 'ready',
        'task_key': 'work-old',
        'goal_id': 'work-old',
        'map_goal_xy': [1.25, -0.75],
        'map_frame': 'map',
        'coarse_goal_check': 'xy_only',
        'goal_reset_acknowledged': True,
        'work_pose_source': source,
    }
    MobileManipulationRuntime._navigation_status_cb(
        harness, String(data=json.dumps(stale)),
    )
    assert not harness._coarse_nav_ready

    matched = dict(
        stale,
        task_key='work-9',
        goal_id='work-9',
        phase='navigating',
    )
    MobileManipulationRuntime._navigation_status_cb(
        harness, String(data=json.dumps(matched)),
    )
    assert harness._navigation_goal_acknowledged
    assert not harness._coarse_nav_ready
    assert harness._work_pose_history_map == []

    ready = dict(matched, phase='ready')
    MobileManipulationRuntime._navigation_status_cb(
        harness, String(data=json.dumps(ready)),
    )
    assert not harness._coarse_nav_ready
    assert harness._logger.errors == []

    harness._odom_sequence += 1
    harness._odom_stamp_ns = 11_950_000_000
    harness._position_xy = (1.30, -0.75)
    MobileManipulationRuntime._navigation_status_cb(
        harness, String(data=json.dumps(ready)),
    )
    assert harness._coarse_nav_ready
    assert harness._navigation_history_recorded
    assert len(harness._work_pose_history_map) == 1
    assert harness._logger.errors == []


def test_completed_work_pose_rejects_target_drift_from_frozen_observation():
    identity = _PlanningObservationIdentity(
        'request-4', 'producer-a', 3, 9876, 'wrist_camera', (0.1, -0.2, 1.4),
    )

    class _Harness:
        _required_perception_request_id = 'request-4'
        _required_perception_generation = 3
        _required_affordance_generation = 3
        _bound_perception_request_id = 'request-4'
        _bound_perception_producer_epoch = 'producer-a'
        _bound_perception_generation = 3
        _affordance_request_id = 'request-4'
        _affordance_producer_epoch = 'producer-a'
        _affordance_generation = 3
        _valid_perception_request_id = 'request-4'
        _valid_perception_producer_epoch = 'producer-a'
        _valid_perception_generation = 3
        _valid_observation_stamp_ns = 9999
        _valid_observation_frame_id = 'wrist_camera'
        _target_frame_id = 'wrist_camera'
        _target_cloud_frame_id = 'wrist_camera'
        _scene_cloud_frame_id = 'wrist_camera'
        _target_camera = np.array((0.1, -0.2, 1.48))
        _serial_gate = SimpleNamespace(snapshot=lambda _now: object())

        @staticmethod
        def _now_s():
            return 12.0

        @staticmethod
        def _grounding_observation_authorized(_snapshot):
            return True

        @staticmethod
        def get_parameter(_name):
            return SimpleNamespace(value=0.06)

    with pytest.raises(_PlanningObservationChanged, match='target moved'):
        MobileManipulationRuntime._validate_completed_work_pose_observation(
            _Harness(), identity,
        )


def test_near_depth_cannot_bypass_an_explicit_work_pose_goal():
    class _Harness:
        def __init__(self):
            self._core = RuntimeSafetyCore()
            self._core.phase = RuntimePhase.COARSE_NAV
            self._work_pose = {'goal_id': 'work-9'}
            self._target_camera = np.array((0.0, 0.0, 0.8))
            self._coarse_nav_ready = False
            self._config = SimpleNamespace(
                approach=SimpleNamespace(near_stage_threshold_m=1.4),
            )

    harness = _Harness()
    MobileManipulationRuntime._maybe_finish_coarse_nav(harness)
    assert harness._core.phase is RuntimePhase.COARSE_NAV


def test_coarse_nav_handoff_settles_then_stops_before_fresh_grounding():
    events = []

    class _Task:
        def __init__(self):
            self.stage = SimpleNamespace(value='coarse_nav')

        def apply(self, _result):
            events.append('task_success')
            self.stage.value = 'visual_approach'

    class _Harness:
        def __init__(self):
            self._core = RuntimeSafetyCore()
            self._core.phase = RuntimePhase.COARSE_NAV
            self._task = _Task()
            self._work_pose = {'goal_id': 'work-9'}
            self._target_camera = None
            self._coarse_nav_ready = True
            self._coarse_nav_perception_loss_detail = 'deferred tracker loss'
            self._coarse_nav_arrival_started_at_s = None
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_last_odom_sequence = None
            self._coarse_nav_arrival_last_odom_stamp_ns = None
            self._odom_seen_at = 12.0
            self._odom_sequence = 0
            self._odom_stamp_ns = None
            self._base_linear_speed_mps = 0.0
            self._base_angular_speed_rps = 0.0
            self.now = 12.0
            self._config = SimpleNamespace(
                approach=SimpleNamespace(near_stage_threshold_m=1.4),
            )

        def _apply_safety(self, action):
            assert action.stop_base
            events.append(
                'stop_and_cancel' if action.cancel_navigation else 'stop',
            )

        def _now_s(self):
            return self.now

        def get_parameter(self, name):
            return SimpleNamespace(value={
                'coarse_nav_arrival_settle_s': 0.35,
                'coarse_nav_arrival_stop_timeout_s': 3.0,
                'coarse_nav_arrival_max_linear_speed_mps': 0.035,
                'coarse_nav_arrival_max_angular_speed_rps': 0.05,
                'coarse_nav_arrival_max_xy_excursion_m': 0.01,
                'coarse_nav_arrival_max_yaw_excursion_rad': 0.01,
                'coarse_nav_arrival_max_odom_gap_s': 0.15,
                'work_pose_odom_max_age_s': 0.5,
                'near_view_pose': '',
            }[name])

        def _topic_value(self, name):
            return str(self.get_parameter(name).value)

        _coarse_nav_arrival_is_stationary = (
            MobileManipulationRuntime._coarse_nav_arrival_is_stationary
        )

        @staticmethod
        def _recover_precontact(*_args):
            raise AssertionError('stationary handoff must not recover')

        def _request_semantic_reground(self, now):
            assert now == self.now
            events.append('fresh_reground')

    harness = _Harness()
    MobileManipulationRuntime._maybe_finish_coarse_nav(harness)
    assert events == ['stop']
    assert harness._core.phase is RuntimePhase.COARSE_NAV

    for sequence, sample_at in enumerate(
        (12.05, 12.14, 12.23, 12.32, 12.41),
        start=1,
    ):
        harness._odom_sequence = sequence
        harness._odom_stamp_ns = int(round(sample_at * 1e9))
        harness._odom_seen_at = sample_at
        MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
            harness,
            received_at_s=sample_at,
            odom_sequence=sequence,
            odom_stamp_ns=harness._odom_stamp_ns,
            linear_speed_mps=0.0,
            angular_speed_rps=0.0,
            position_xy=(0.0, 0.0),
            yaw_rad=0.0,
        )
    harness.now = 12.41
    MobileManipulationRuntime._maybe_finish_coarse_nav(harness)

    assert events == [
        'stop',
        'stop',
        'stop_and_cancel',
        'task_success',
        'fresh_reground',
    ]
    assert harness._core.phase is RuntimePhase.NEAR_GROUNDING
    assert harness._coarse_nav_perception_loss_detail == ''
    assert harness._coarse_nav_arrival_started_at_s is None
    assert harness._coarse_nav_arrival_stable_since_s is None
    assert harness._coarse_nav_arrival_stable_start_odom_stamp_ns is None
    assert harness._coarse_nav_arrival_last_odom_sequence is None
    assert harness._coarse_nav_arrival_last_odom_stamp_ns is None


def test_near_view_retires_old_track_and_regrounds_only_after_settle():
    events = []
    core = RuntimeSafetyCore()
    core.phase = RuntimePhase.NEAR_GROUNDING
    harness = SimpleNamespace(
        _core=core,
        _near_view_pose_name='',
        _near_view_settle_started_at=None,
        _near_view_settle_last_tick_at=None,
        _near_view_settle_until=None,
        _near_view_deadline_s=None,
        _near_view_joint_sequence_floor=None,
        _near_view_joint_target=None,
        _near_view_joint_error_rad=None,
        _near_view_achieved_pose_name='',
        _joint_sequence=4,
        _joint_state=np.zeros(6),
        _joint_history=[],
        _topic_value=lambda name: (
            'MANIP_LOOKOUT' if name == 'near_view_pose' else ''
        ),
        get_parameter=lambda name: SimpleNamespace(value={
            'near_view_settle_s': 0.30,
            'near_view_timeout_s': 1.0,
            'near_view_joint_positions': [0.0, 1.0, -0.71, 0.0, 0.0, 0.0],
            'near_view_joint_tolerance_rad': 0.05,
        }[name]),
        _invalidate_perception_session=lambda: events.append('reset_track'),
        _named_pose_pub=SimpleNamespace(
            publish=lambda msg: events.append(f'pose:{msg.data}'),
        ),
        _publish_zero=lambda: events.append('zero'),
        _request_semantic_reground=lambda now: events.append(f'reground:{now:.2f}'),
        _recover_precontact=lambda *_args: False,
        _apply_safety=lambda _action: events.append('failed'),
        _arm_is_still=lambda _now: True,
    )

    MobileManipulationRuntime._begin_near_view_settle(harness, 10.0)

    assert events == ['reset_track', 'pose:MANIP_LOOKOUT']
    assert harness._near_view_pose_name == 'MANIP_LOOKOUT'
    assert harness._near_view_settle_until == pytest.approx(10.30)

    MobileManipulationRuntime._near_view_settle_tick(harness, 10.20)
    assert events[-1] == 'zero'
    assert not any(event.startswith('reground:') for event in events)

    harness._joint_history.append(SimpleNamespace(
        sequence=5,
        positions=np.array([0.0, 1.0, -0.71, 0.0, 0.0, 0.0]),
    ))
    MobileManipulationRuntime._near_view_settle_tick(harness, 10.31)
    assert events[-2:] == ['zero', 'reground:10.31']
    assert harness._near_view_pose_name == ''
    assert harness._near_view_settle_until is None
    assert harness._near_view_achieved_pose_name == 'MANIP_LOOKOUT'


def test_coarse_nav_arrival_requires_continuous_fresh_stillness():
    values = {
        'coarse_nav_arrival_settle_s': 0.35,
        'coarse_nav_arrival_stop_timeout_s': 3.0,
        'coarse_nav_arrival_max_linear_speed_mps': 0.035,
        'coarse_nav_arrival_max_angular_speed_rps': 0.05,
        'coarse_nav_arrival_max_xy_excursion_m': 0.01,
        'coarse_nav_arrival_max_yaw_excursion_rad': 0.01,
        'coarse_nav_arrival_max_odom_gap_s': 0.15,
        'work_pose_odom_max_age_s': 0.5,
    }
    harness = SimpleNamespace(
        _coarse_nav_arrival_started_at_s=None,
        _coarse_nav_arrival_stable_since_s=None,
        _coarse_nav_arrival_stable_start_odom_stamp_ns=None,
        _coarse_nav_arrival_last_odom_sequence=None,
        _coarse_nav_arrival_last_odom_stamp_ns=None,
        _odom_seen_at=10.0,
        _odom_sequence=0,
        _odom_stamp_ns=None,
        _base_linear_speed_mps=0.0,
        _base_angular_speed_rps=0.0,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )

    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.0,
    )
    harness._odom_seen_at = 10.1
    harness._odom_sequence = 1
    harness._odom_stamp_ns = 10_100_000_000
    MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
        harness,
        received_at_s=10.1,
        odom_sequence=1,
        odom_stamp_ns=harness._odom_stamp_ns,
        linear_speed_mps=0.04,
        angular_speed_rps=0.0,
        position_xy=(0.0, 0.0),
        yaw_rad=0.0,
    )
    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.1,
    )
    assert harness._coarse_nav_arrival_stable_since_s is None

    for sequence, sample_at in enumerate(
        (10.2, 10.3, 10.4, 10.5, 10.6),
        start=2,
    ):
        harness._odom_seen_at = sample_at
        harness._odom_sequence = sequence
        harness._odom_stamp_ns = int(round(sample_at * 1e9))
        MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
            harness,
            received_at_s=sample_at,
            odom_sequence=sequence,
            odom_stamp_ns=harness._odom_stamp_ns,
            linear_speed_mps=0.0,
            angular_speed_rps=0.0,
            position_xy=(0.0, 0.0),
            yaw_rad=0.0,
        )
    assert MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.6,
    )


def test_coarse_nav_arrival_profile_accepts_bounded_gait_twist_noise():
    values = {
        'coarse_nav_arrival_settle_s': 0.35,
        'coarse_nav_arrival_stop_timeout_s': 3.0,
        'coarse_nav_arrival_max_linear_speed_mps': 0.05,
        'coarse_nav_arrival_max_angular_speed_rps': 0.05,
        'coarse_nav_arrival_max_xy_excursion_m': 0.01,
        'coarse_nav_arrival_max_yaw_excursion_rad': 0.01,
        'coarse_nav_arrival_max_odom_gap_s': 0.15,
        'work_pose_odom_max_age_s': 0.5,
    }
    harness = SimpleNamespace(
        _coarse_nav_arrival_started_at_s=None,
        _coarse_nav_arrival_stable_since_s=None,
        _coarse_nav_arrival_stable_start_odom_stamp_ns=None,
        _coarse_nav_arrival_last_odom_sequence=None,
        _coarse_nav_arrival_last_odom_stamp_ns=None,
        _odom_seen_at=10.0,
        _odom_sequence=0,
        _odom_stamp_ns=None,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )

    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.0,
    )
    for sequence, sample_at in enumerate(
        (
            10.04, 10.08, 10.12, 10.16, 10.20,
            10.24, 10.28, 10.32, 10.36, 10.40,
        ),
        start=1,
    ):
        harness._odom_seen_at = sample_at
        harness._odom_sequence = sequence
        harness._odom_stamp_ns = int(round(sample_at * 1e9))
        MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
            harness,
            received_at_s=sample_at,
            odom_sequence=sequence,
            odom_stamp_ns=harness._odom_stamp_ns,
            linear_speed_mps=0.049 if sequence % 2 else 0.03,
            angular_speed_rps=0.02,
            position_xy=(0.0, 0.0),
            yaw_rad=0.0,
        )
    assert MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.40,
    )

    harness._odom_seen_at = 10.44
    harness._odom_sequence = 11
    harness._odom_stamp_ns = 10_440_000_000
    MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
        harness,
        received_at_s=10.44,
        odom_sequence=11,
        odom_stamp_ns=harness._odom_stamp_ns,
        linear_speed_mps=0.051,
        angular_speed_rps=0.02,
        position_xy=(0.0, 0.0),
        yaw_rad=0.0,
    )
    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.44,
    )


def test_coarse_nav_arrival_uses_bounded_se2_excursion_for_legged_policy():
    values = {
        'coarse_nav_arrival_settle_s': 0.35,
        'coarse_nav_arrival_stop_timeout_s': 3.0,
        'coarse_nav_arrival_max_linear_speed_mps': 0.075,
        'coarse_nav_arrival_max_angular_speed_rps': 0.05,
        'coarse_nav_arrival_max_xy_excursion_m': 0.01,
        'coarse_nav_arrival_max_yaw_excursion_rad': 0.01,
        'coarse_nav_arrival_max_odom_gap_s': 0.15,
        'work_pose_odom_max_age_s': 0.5,
    }

    def harness(start):
        return SimpleNamespace(
            _coarse_nav_arrival_started_at_s=None,
            _coarse_nav_arrival_stable_since_s=None,
            _coarse_nav_arrival_stable_start_odom_stamp_ns=None,
            _coarse_nav_arrival_last_odom_sequence=None,
            _coarse_nav_arrival_last_odom_stamp_ns=None,
            _coarse_nav_arrival_anchor_xy=None,
            _coarse_nav_arrival_anchor_yaw_rad=None,
            _odom_seen_at=start,
            _odom_sequence=0,
            _odom_stamp_ns=None,
            get_parameter=lambda name: SimpleNamespace(value=values[name]),
        )

    parked = harness(10.0)
    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        parked, 10.0,
    )
    for sequence, sample_at in enumerate(
        (10.04, 10.08, 10.12, 10.16, 10.20, 10.24, 10.28, 10.32, 10.36, 10.40),
        start=1,
    ):
        parked._odom_seen_at = sample_at
        parked._odom_sequence = sequence
        parked._odom_stamp_ns = int(round(sample_at * 1e9))
        MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
            parked,
            received_at_s=sample_at,
            odom_sequence=sequence,
            odom_stamp_ns=parked._odom_stamp_ns,
            linear_speed_mps=0.07 if sequence % 2 else 0.03,
            angular_speed_rps=0.02,
            position_xy=(0.003 * math.sin(sequence), 0.003 * math.cos(sequence)),
            yaw_rad=0.004 * math.sin(sequence),
        )
    assert MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        parked, 10.40,
    )

    moving = harness(20.0)
    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        moving, 20.0,
    )
    for sequence, sample_at in enumerate(
        (20.04, 20.08, 20.12, 20.16, 20.20, 20.24, 20.28, 20.32, 20.36, 20.40),
        start=1,
    ):
        moving._odom_seen_at = sample_at
        moving._odom_sequence = sequence
        moving._odom_stamp_ns = int(round(sample_at * 1e9))
        MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
            moving,
            received_at_s=sample_at,
            odom_sequence=sequence,
            odom_stamp_ns=moving._odom_stamp_ns,
            linear_speed_mps=0.04,
            angular_speed_rps=0.0,
            position_xy=(0.04 * (sample_at - 20.0), 0.0),
            yaw_rad=0.0,
        )
    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        moving, 20.40,
    )


def test_coarse_nav_arrival_timeout_fails_closed_on_persistent_motion():
    values = {
        'coarse_nav_arrival_settle_s': 0.35,
        'coarse_nav_arrival_stop_timeout_s': 3.0,
        'coarse_nav_arrival_max_linear_speed_mps': 0.035,
        'coarse_nav_arrival_max_angular_speed_rps': 0.05,
        'coarse_nav_arrival_max_xy_excursion_m': 0.01,
        'coarse_nav_arrival_max_yaw_excursion_rad': 0.01,
        'coarse_nav_arrival_max_odom_gap_s': 0.15,
        'work_pose_odom_max_age_s': 0.5,
    }
    harness = SimpleNamespace(
        _coarse_nav_arrival_started_at_s=None,
        _coarse_nav_arrival_stable_since_s=None,
        _coarse_nav_arrival_stable_start_odom_stamp_ns=None,
        _coarse_nav_arrival_last_odom_sequence=None,
        _coarse_nav_arrival_last_odom_stamp_ns=None,
        _odom_seen_at=10.0,
        _odom_sequence=0,
        _odom_stamp_ns=None,
        _base_linear_speed_mps=0.1,
        _base_angular_speed_rps=0.0,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )

    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.0,
    )
    harness._odom_seen_at = 13.01
    with pytest.raises(TimeoutError, match='stationary'):
        MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
            harness, 13.01,
        )


def test_coarse_nav_arrival_cannot_reuse_one_cached_zero_speed_sample():
    values = {
        'coarse_nav_arrival_settle_s': 0.35,
        'coarse_nav_arrival_stop_timeout_s': 3.0,
        'coarse_nav_arrival_max_linear_speed_mps': 0.035,
        'coarse_nav_arrival_max_angular_speed_rps': 0.05,
        'coarse_nav_arrival_max_xy_excursion_m': 0.01,
        'coarse_nav_arrival_max_yaw_excursion_rad': 0.01,
        'coarse_nav_arrival_max_odom_gap_s': 0.15,
        'work_pose_odom_max_age_s': 0.5,
    }
    harness = SimpleNamespace(
        _coarse_nav_arrival_started_at_s=None,
        _coarse_nav_arrival_stable_since_s=None,
        _coarse_nav_arrival_stable_start_odom_stamp_ns=None,
        _coarse_nav_arrival_last_odom_sequence=None,
        _coarse_nav_arrival_last_odom_stamp_ns=None,
        _odom_seen_at=10.0,
        _odom_sequence=1,
        _odom_stamp_ns=10_000_000_000,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )

    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.0,
    )
    assert not MobileManipulationRuntime._coarse_nav_arrival_is_stationary(
        harness, 10.36,
    )


def test_navigation_ready_requires_fresh_measured_xy_at_the_active_goal():
    harness = SimpleNamespace(
        _work_pose={
            'map_goal_xy': [1.0, -0.5],
            'source': {
                'odom_sequence': 2,
                'odom_stamp_ns': 9_800_000_000,
            },
        },
        _position_xy=(0.6, -0.5),
        _odom_seen_at=9.9,
        _odom_stamp_ns=9_900_000_000,
        _odom_sequence=3,
        _navigation_goal_acknowledged=True,
        _navigation_ack_odom_sequence=2,
        _navigation_ack_odom_stamp_ns=9_850_000_000,
        get_parameter=lambda name: SimpleNamespace(value={
            'work_pose_odom_max_age_s': 0.5,
            'work_pose_goal_tolerance_m': 0.2,
        }[name]),
    )

    with pytest.raises(ValueError, match='outside the work-pose XY tolerance'):
        MobileManipulationRuntime._validate_navigation_ready_pose(harness, 10.0)


def test_navigation_ready_accepts_bounded_braking_handoff_envelope():
    harness = SimpleNamespace(
        _work_pose={
            'map_goal_xy': [1.0, 0.0],
            'source': {
                'odom_sequence': 2,
                'odom_stamp_ns': 9_800_000_000,
            },
        },
        _position_xy=(0.705, 0.0),
        _odom_seen_at=9.9,
        _odom_stamp_ns=9_900_000_000,
        _odom_sequence=3,
        _navigation_goal_acknowledged=True,
        _navigation_ack_odom_sequence=2,
        _navigation_ack_odom_stamp_ns=9_850_000_000,
        get_parameter=lambda name: SimpleNamespace(value={
            'work_pose_odom_max_age_s': 0.5,
            'work_pose_goal_tolerance_m': 0.33,
        }[name]),
    )

    MobileManipulationRuntime._validate_navigation_ready_pose(harness, 10.0)

    harness._position_xy = (0.669, 0.0)
    with pytest.raises(ValueError, match='outside the work-pose XY tolerance'):
        MobileManipulationRuntime._validate_navigation_ready_pose(harness, 10.0)


def test_navigation_ready_rejects_stale_source_and_missing_post_plan_sample():
    source = {
        'odom_sequence': 3,
        'odom_stamp_ns': 9_900_000_000,
    }
    harness = SimpleNamespace(
        _work_pose={'map_goal_xy': [1.0, -0.5], 'source': source},
        _position_xy=(1.0, -0.5),
        _odom_seen_at=10.0,
        _odom_stamp_ns=9_900_000_000,
        _odom_sequence=3,
        _navigation_goal_acknowledged=True,
        _navigation_ack_odom_sequence=3,
        _navigation_ack_odom_stamp_ns=9_900_000_000,
        get_parameter=lambda name: SimpleNamespace(value={
            'work_pose_odom_max_age_s': 0.5,
            'work_pose_goal_tolerance_m': 0.2,
        }[name]),
    )

    with pytest.raises(ValueError, match='no post-work-pose'):
        MobileManipulationRuntime._validate_navigation_ready_pose(harness, 10.0)

    harness._odom_sequence = 4
    harness._odom_stamp_ns = 9_950_000_000
    MobileManipulationRuntime._validate_navigation_ready_pose(harness, 10.0)

    harness._odom_seen_at = 10.5
    with pytest.raises(ValueError, match='source stamp is stale'):
        MobileManipulationRuntime._validate_navigation_ready_pose(harness, 10.6)


def test_work_pose_history_requires_navigation_ack_and_measured_motion():
    harness = SimpleNamespace(
        _work_pose={
            'anchor_map_pose': [0.0, 0.0, 0.0],
            'map_goal_xy': [1.0, 0.0],
            'map_goal_yaw_rad': 0.0,
        },
        _navigation_goal_acknowledged=False,
        _navigation_ack_position_xy=None,
        _navigation_ack_odom_sequence=None,
        _navigation_ack_odom_stamp_ns=None,
        _navigation_history_recorded=False,
        _position_xy=(0.5, 0.0),
        _odom_sequence=4,
        _odom_stamp_ns=10_000_000_000,
        _work_pose_history_map=[],
        get_parameter=lambda _name: SimpleNamespace(value=0.03),
    )

    MobileManipulationRuntime._record_navigation_attempt_if_moved(harness)
    assert harness._work_pose_history_map == []

    harness._navigation_goal_acknowledged = True
    harness._navigation_ack_position_xy = (0.5, 0.0)
    harness._navigation_ack_odom_sequence = 4
    harness._navigation_ack_odom_stamp_ns = 10_000_000_000
    MobileManipulationRuntime._record_navigation_attempt_if_moved(harness)
    assert harness._work_pose_history_map == []

    harness._odom_sequence = 5
    harness._odom_stamp_ns = 10_100_000_000
    harness._position_xy = (0.54, 0.0)
    MobileManipulationRuntime._record_navigation_attempt_if_moved(harness)
    assert len(harness._work_pose_history_map) == 1
    assert harness._navigation_history_recorded


def test_correlated_navigation_status_watchdog_fails_closed():
    class _Harness:
        _work_pose = {'goal_id': 'work-9'}
        _work_pose_created_at_s = 1.0
        _navigation_status_seen_s = None

        def __init__(self):
            self.recovery = None

        @staticmethod
        def get_parameter(_name):
            return SimpleNamespace(value=3.0)

        def _recover_precontact(self, kind, detail):
            self.recovery = (kind, detail)
            return True

        def _apply_safety(self, _action):
            raise AssertionError('accepted recovery must own the failure')

    harness = _Harness()
    assert not MobileManipulationRuntime._coarse_navigation_status_alive(
        harness, 4.1,
    )
    assert harness.recovery[0].value == 'nav_blocked'
    assert 'work-9' in harness.recovery[1]
