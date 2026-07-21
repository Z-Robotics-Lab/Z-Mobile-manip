"""ROS-message boundary tests that do not create a ROS graph."""

import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip('rclpy')

from rclpy.qos import DurabilityPolicy, ReliabilityPolicy  # noqa: E402
from rclpy.time import Time  # noqa: E402
from z_manip_navigation.core import (  # noqa: E402
    CoarseNavigationCore,
    NavigationTaskRequest,
    NavPhase,
)
from z_manip_navigation.node import CoarseNavigationNode  # noqa: E402


class _Publisher:
    def __init__(self, name=None, events=None):
        self.messages = []
        self.name = name
        self.events = events

    def publish(self, message):
        self.messages.append(message)
        if self.events is not None:
            self.events.append((self.name, message))


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)


class _NodeBoundary:
    """Minimum state needed to exercise node goal lifecycle methods."""

    _activate_task = CoarseNavigationNode._activate_task
    _deactivate_task = CoarseNavigationNode._deactivate_task
    _retire_task_key = CoarseNavigationNode._retire_task_key
    _goal_reached_cb = CoarseNavigationNode._goal_reached_cb
    _tick = CoarseNavigationNode._tick
    _publish_waypoint = CoarseNavigationNode._publish_waypoint
    _publish_status = CoarseNavigationNode._publish_status

    def __init__(self):
        self._core = CoarseNavigationCore()
        self._task_instruction = ''
        self._task_key = ''
        self._goal_id = ''
        self._work_pose_map_frame = ''
        self._explicit_goal_xy = None
        self._map_goal_yaw_rad = None
        self._work_pose_source = None
        self._retired_task_keys = []
        self._suggested_displacement = None
        self._goal_reached = False
        self._goal_false_seen = False
        self._goal_reset_pending = False
        self._goal_reset_requested_at = None
        self._pending_waypoint_xy = None
        self._last_waypoint_at = 10.0
        self._perception_valid = True
        self._target_stamp = 12.0
        self._navigation_healthy = True
        self._odometry_healthy = True
        self._target_depth = 1.0
        self._target_xy = np.array([2.0, 0.0])
        self._base_xy = np.array([0.0, 0.0])
        self._base_speed = 0.0
        self._odom_stamp_s = 12.0
        self._last_status = ''
        self._last_status_at = None
        self._time = 12.0
        self._events = []
        self._cancel_pub = _Publisher('cancel', self._events)
        self._ready_pub = _Publisher('ready', self._events)
        self._status_pub = _Publisher('status', self._events)
        self._waypoint_pub = _Publisher('waypoint', self._events)
        self._grounding_pub = _Publisher('grounding', self._events)
        self._logger = _Logger()
        self._lock = threading.RLock()

    def _now_s(self):
        return self._time

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: Time(nanoseconds=int(round(self._time * 1e9))),
        )

    @staticmethod
    def get_parameter(name):
        return SimpleNamespace(value={
            'status_heartbeat_s': 0.5,
            'goal_reset_ack_timeout_s': 2.0,
            'waypoint_refresh_s': 1.0,
        }[name])

    @staticmethod
    def _topic(name):
        assert name == 'map_frame'
        return 'map'


class _IoBoundary:
    """Capture the QoS contracts created by the ROS adapter."""

    _setup_io = CoarseNavigationNode._setup_io

    def __init__(self):
        self.subscriptions = []

    @staticmethod
    def _topic(name):
        return f'/{name}'

    @staticmethod
    def create_publisher(_message_type, _topic, _qos):
        return _Publisher()

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions.append((message_type, topic, callback, qos))

    def _task_status_cb(self, _message):
        pass

    def _valid_cb(self, _message):
        pass

    def _target_cb(self, _message):
        pass

    def _odom_cb(self, _message):
        pass

    def _health_cb(self, _message):
        pass

    def _goal_reached_cb(self, _message):
        pass


class _OdomBoundary:
    """Minimum state for exercising the platform-odometry contract."""

    _odom_cb = CoarseNavigationNode._odom_cb

    def __init__(self):
        self._lock = threading.RLock()
        self._base_xy = np.array((9.0, 9.0))
        self._base_speed = 0.0
        self._odom_stamp_s = None
        self._odometry_healthy = True
        self._logger = _Logger()

    @staticmethod
    def _topic(name):
        return {
            'map_frame': 'map',
            'platform_base_frame': 'base_link',
        }[name]

    @staticmethod
    def get_parameter(name):
        assert name == 'angular_speed_weight_m'
        return SimpleNamespace(value=0.30)

    def get_logger(self):
        return self._logger


def _request(goal_id, xy=(1.0, 2.0), yaw=0.25, map_frame='map'):
    return NavigationTaskRequest(
        instruction='pick mustard',
        task_key=goal_id,
        goal_id=goal_id,
        map_frame=map_frame,
        map_goal_xy=np.asarray(xy, dtype=float),
        map_goal_yaw_rad=yaw,
        suggested_displacement_m=None,
        source={'epoch': 2, 'generation': int(goal_id[-1])},
    )


def _legacy_request(task_key='legacy:1', displacement=1.2):
    return NavigationTaskRequest(
        instruction='pick mustard',
        task_key=task_key,
        goal_id=None,
        map_frame=None,
        map_goal_xy=None,
        map_goal_yaw_rad=None,
        suggested_displacement_m=displacement,
    )


def _odom(*, parent='map', child='base_link', x=1.25, y=-0.75, stamp_s=1.0):
    from nav_msgs.msg import Odometry

    message = Odometry()
    message.header.frame_id = parent
    message.header.stamp.sec = int(stamp_s)
    message.header.stamp.nanosec = int(round((stamp_s - int(stamp_s)) * 1e9))
    message.child_frame_id = child
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.twist.twist.linear.x = 0.03
    message.twist.twist.linear.y = 0.04
    message.twist.twist.angular.z = 0.10
    return message


def test_platform_odometry_uses_map_base_link_origin_and_finite_twist():
    node = _OdomBoundary()

    node._odom_cb(_odom())

    np.testing.assert_allclose(node._base_xy, (1.25, -0.75))
    assert node._base_speed == pytest.approx(math.sqrt(0.03**2 + 0.04**2 + 0.03**2))
    assert node._odom_stamp_s == pytest.approx(1.0)
    assert node._odometry_healthy
    assert not node._logger.errors


@pytest.mark.parametrize(
    ('parent', 'child'),
    (('odom', 'base_link'), ('map', 'sensor')),
)
def test_platform_odometry_rejects_wrong_frames_and_drops_stale_pose(parent, child):
    node = _OdomBoundary()

    node._odom_cb(_odom(parent=parent, child=child))

    assert node._base_xy is None
    assert math.isinf(node._base_speed)
    assert not node._odometry_healthy
    assert 'frame mismatch' in node._logger.errors[-1]


def test_platform_odometry_rejects_non_finite_values_and_drops_stale_pose():
    node = _OdomBoundary()

    node._odom_cb(_odom(x=float('nan')))

    assert node._base_xy is None
    assert math.isinf(node._base_speed)
    assert not node._odometry_healthy
    assert 'non-finite' in node._logger.errors[-1]


def test_platform_odometry_ignores_non_increasing_stamp_without_regressing_pose():
    node = _OdomBoundary()
    node._odom_cb(_odom(x=1.0, stamp_s=2.0))

    node._odom_cb(_odom(x=9.0, stamp_s=1.9))

    np.testing.assert_allclose(node._base_xy, (1.0, -0.75))
    assert node._odom_stamp_s == pytest.approx(2.0)
    assert node._odometry_healthy
    assert node._logger.warnings


def test_new_goal_id_resets_ready_and_retired_goal_cannot_revive():
    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    assert node._core.phase is NavPhase.WAIT_OBSERVATION
    assert node._core.task_key == 'goal-1'
    assert node._ready_pub.messages[-1].data is False

    node._core.phase = NavPhase.READY
    node._activate_task(_request('goal-2', xy=(3.0, -1.0)))
    assert node._core.phase is NavPhase.WAIT_OBSERVATION
    assert node._core.task_key == 'goal-2'
    assert node._goal_id == 'goal-2'
    assert 'goal-1' in node._retired_task_keys
    assert node._cancel_pub.messages[-1].data is True

    node._activate_task(_request('goal-1'))
    assert node._core.task_key == 'goal-2'
    assert node._logger.warnings


def test_same_goal_id_cannot_mutate_work_pose_contract():
    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    with pytest.raises(ValueError, match='changed without a new goal_id'):
        node._activate_task(_request('goal-1', xy=(1.1, 2.0)))
    np.testing.assert_array_equal(node._explicit_goal_xy, [1.0, 2.0])


def test_explicit_work_pose_frame_must_match_navigation_map_frame():
    node = _NodeBoundary()
    with pytest.raises(ValueError, match='map_frame'):
        node._activate_task(_request('goal-1', map_frame='odom'))
    assert node._core.phase is NavPhase.IDLE
    assert node._task_key == ''


def test_task_status_subscription_is_reliable_and_transient_local():
    node = _IoBoundary()
    node._setup_io()
    task_status = next(
        entry for entry in node.subscriptions
        if entry[1] == '/task_status_topic'
    )
    qos = task_status[3]
    assert qos.reliability is ReliabilityPolicy.RELIABLE
    assert qos.durability is DurabilityPolicy.TRANSIENT_LOCAL


def test_new_goal_id_requires_a_fresh_goal_reached_edge():
    from std_msgs.msg import Bool

    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    assert node._goal_reset_pending
    assert node._cancel_pub.messages[-1].data is True
    node._goal_reached_cb(Bool(data=True))
    assert not node._goal_reached
    node._goal_reached_cb(Bool(data=False))
    assert not node._goal_reset_pending
    node._goal_reached_cb(Bool(data=True))
    assert node._goal_reached

    node._activate_task(_request('goal-2'))
    node._goal_reached_cb(Bool(data=True))
    assert not node._goal_reached


def test_waypoint_is_held_until_local_planner_reset_ack_and_timeout_fails():
    from std_msgs.msg import Bool

    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    node._tick()
    waiting = json.loads(node._status_pub.messages[-1].data)
    assert waiting['phase'] == 'wait_observation'
    assert not waiting['goal_reset_acknowledged']
    assert 'reset acknowledgement' in waiting['reason']
    assert not node._waypoint_pub.messages

    node._events.append(('false_callback', None))
    node._goal_reached_cb(Bool(data=False))
    assert not node._goal_reset_pending
    assert node._goal_false_seen
    node._tick()
    assert len(node._waypoint_pub.messages) == 1
    assert node._waypoint_pub.messages[-1].point.x == pytest.approx(1.0)
    assert node._waypoint_pub.messages[-1].point.y == pytest.approx(2.0)
    ordering = [
        name for name, _message in node._events
        if name in {'cancel', 'false_callback', 'waypoint'}
    ]
    assert ordering == ['cancel', 'false_callback', 'waypoint']

    timed_out = _NodeBoundary()
    timed_out._activate_task(_request('goal-1'))
    timed_out._time += 2.01
    timed_out._tick()
    assert timed_out._core.phase is NavPhase.FAILED
    assert timed_out._cancel_pub.messages[-1].data is True
    failed = json.loads(timed_out._status_pub.messages[-1].data)
    assert 'reset acknowledgement timed out' in failed['reason']


def test_replan_repeats_cancel_ack_before_republishing_waypoint():
    from std_msgs.msg import Bool

    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    node._goal_reached_cb(Bool(data=False))
    node._tick()
    assert len(node._waypoint_pub.messages) == 1

    node._time += 0.1
    node._odom_stamp_s = node._time
    node._tick()
    node._time += 8.01
    node._odom_stamp_s = node._time
    node._tick()
    assert node._goal_reset_pending
    assert node._pending_waypoint_xy is not None
    assert len(node._waypoint_pub.messages) == 1
    assert node._cancel_pub.messages[-1].data is True

    node._events.append(('false_callback', None))
    node._goal_reached_cb(Bool(data=False))
    node._tick()
    assert len(node._waypoint_pub.messages) == 2
    assert node._waypoint_pub.messages[-1].point.x == pytest.approx(1.0)
    assert node._waypoint_pub.messages[-1].point.y == pytest.approx(2.0)
    assert node._pending_waypoint_xy is None
    ordering = [
        name for name, _message in node._events
        if name in {'cancel', 'false_callback', 'waypoint'}
    ]
    assert ordering[-3:] == ['cancel', 'false_callback', 'waypoint']


def test_waypoint_refresh_does_not_reset_current_reached_evidence():
    node = _NodeBoundary()
    node._goal_false_seen = True
    node._goal_reached = True

    node._publish_waypoint(np.array([1.0, 2.0]), 12.0, new_goal=False)

    assert node._goal_false_seen
    assert node._goal_reached


def test_explicit_work_pose_is_not_periodically_republished():
    """A duplicate PointStamped must not clear localPlanner's reached latch."""
    from std_msgs.msg import Bool

    node = _NodeBoundary()
    request = _request('goal-1')
    node._activate_task(request)
    node._goal_reached_cb(Bool(data=False))
    node._tick()
    assert node._core.uses_explicit_goal
    assert len(node._waypoint_pub.messages) == 1

    for _ in range(2):
        node._activate_task(request)
        node._time += 1.01
        node._tick()

    assert len(node._waypoint_pub.messages) == 1
    assert not node._goal_reset_pending
    assert node._pending_waypoint_xy is None


def test_legacy_goal_retains_periodic_waypoint_refresh():
    """Observed-target navigation still refreshes its moving map-ray goal."""
    from std_msgs.msg import Bool

    node = _NodeBoundary()
    node._target_depth = 2.0
    node._activate_task(_legacy_request())
    node._goal_reached_cb(Bool(data=False))
    node._tick()
    assert not node._core.uses_explicit_goal
    assert len(node._waypoint_pub.messages) == 1

    node._time += 1.01
    node._target_stamp = node._time
    node._tick()

    assert len(node._waypoint_pub.messages) == 2
    assert not node._goal_reset_pending
    assert node._pending_waypoint_xy is None


def test_fast_reached_after_dispatch_is_accepted_without_another_false_tick():
    from std_msgs.msg import Bool

    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    node._goal_reached_cb(Bool(data=False))
    node._tick()
    assert len(node._waypoint_pub.messages) == 1

    node._base_xy = np.array([1.0, 2.0])
    node._goal_reached_cb(Bool(data=True))
    node._time += 0.01
    node._tick()
    assert node._core.phase is NavPhase.NAVIGATING
    node._time += 0.36
    node._tick()

    assert node._core.phase is NavPhase.READY
    assert node._ready_pub.messages[-1].data is True
    assert node._cancel_pub.messages[-1].data is True


def test_ready_rebound_clears_latch_before_republishing_same_goal():
    from std_msgs.msg import Bool

    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    node._goal_reached_cb(Bool(data=False))
    node._tick()
    node._base_xy = np.array([1.0, 2.0])
    node._time += 0.01
    node._odom_stamp_s = node._time
    node._tick()
    node._time += 0.36
    node._odom_stamp_s = node._time
    node._tick()
    assert node._core.phase is NavPhase.READY

    event_floor = len(node._events)
    node._base_xy = np.array([0.669, 2.0])
    node._base_speed = 0.08
    node._time += 0.05
    node._odom_stamp_s = node._time
    node._tick()

    assert node._core.phase is NavPhase.NAVIGATING
    assert node._core.task_key == 'goal-1'
    assert node._goal_id == 'goal-1'
    assert node._ready_pub.messages[-1].data is False
    assert node._goal_reset_pending
    np.testing.assert_array_equal(node._pending_waypoint_xy, [1.0, 2.0])
    transition_events = [name for name, _message in node._events[event_floor:]]
    assert transition_events.index('ready') < transition_events.index('cancel')

    node._goal_reached_cb(Bool(data=False))
    node._tick()
    assert len(node._waypoint_pub.messages) == 2
    assert node._waypoint_pub.messages[-1].point.x == pytest.approx(1.0)
    assert node._waypoint_pub.messages[-1].point.y == pytest.approx(2.0)


def test_deactivation_retires_goal_and_clears_latched_ready():
    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    node._core.phase = NavPhase.READY
    node._deactivate_task()
    assert node._core.phase is NavPhase.IDLE
    assert node._task_key == ''
    assert node._goal_id == ''
    assert node._explicit_goal_xy is None
    assert node._ready_pub.messages[-1].data is False
    assert 'goal-1' in node._retired_task_keys


def test_navigation_status_echoes_correlated_work_pose_identity():
    node = _NodeBoundary()
    node._activate_task(_request('goal-1', xy=(1.4, -0.6), yaw=-0.3))
    node._core.goal_xy = np.array([1.4, -0.6])
    node._publish_status('waiting for fresh target observation')
    status = json.loads(node._status_pub.messages[-1].data)
    assert status['task_key'] == 'goal-1'
    assert status['goal_id'] == 'goal-1'
    assert status['map_frame'] == 'map'
    assert status['map_goal_xy'] == [1.4, -0.6]
    assert status['map_goal_yaw_rad'] == -0.3
    assert status['coarse_goal_check'] == 'xy_only'
    assert status['work_pose_source'] == {'epoch': 2, 'generation': 1}


def test_unchanged_navigation_status_has_bounded_heartbeat():
    node = _NodeBoundary()
    node._activate_task(_request('goal-1'))
    node._publish_status('navigating')
    assert len(node._status_pub.messages) == 1
    node._time += 0.49
    node._publish_status('navigating')
    assert len(node._status_pub.messages) == 1
    node._time += 0.02
    node._publish_status('navigating')
    assert len(node._status_pub.messages) == 2
