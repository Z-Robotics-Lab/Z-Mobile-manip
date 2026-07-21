"""ROS 2 adapter for observed-target coarse navigation."""

from __future__ import annotations

import json
import math
import threading
from typing import Any

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D

from .core import (
    CoarseNavigationCore,
    NavigationConfig,
    NavigationTaskRequest,
    NavInput,
    NavPhase,
    parse_task_navigation_request,
)


def _stamp_s(header: Any) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def _transform_point(transform: Any, point: np.ndarray) -> np.ndarray:
    q = transform.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1e-9:
        raise ValueError('TF quaternion is degenerate')
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    translation = np.array([
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ])
    return rotation @ point + translation


class CoarseNavigationNode(Node):
    """Generate map waypoints from SLAM and persistent target perception."""

    def __init__(self) -> None:
        """Create parameterized policy, TF buffer, and ROS graph boundary."""
        super().__init__('z_manip_coarse_navigation')
        self._declare_parameters()
        self._core = CoarseNavigationCore(NavigationConfig(
            near_target_depth_m=float(self.get_parameter('near_target_depth_m').value),
            still_speed_mps=float(self.get_parameter('still_speed_mps').value),
            still_settle_s=float(self.get_parameter('still_settle_s').value),
            target_timeout_s=float(self.get_parameter('target_timeout_s').value),
            observation_wait_timeout_s=float(
                self.get_parameter('observation_wait_timeout_s').value,
            ),
            navigation_timeout_s=float(self.get_parameter('navigation_timeout_s').value),
            stall_timeout_s=float(self.get_parameter('stall_timeout_s').value),
            progress_min_net_decrease_m=float(
                self.get_parameter('progress_min_net_decrease_m').value,
            ),
            progress_min_slope_mps=float(
                self.get_parameter('progress_min_slope_mps').value,
            ),
            odometry_timeout_s=float(
                self.get_parameter('odometry_timeout_s').value,
            ),
            min_displacement_m=float(self.get_parameter('min_displacement_m').value),
            max_displacement_m=float(self.get_parameter('max_displacement_m').value),
            goal_update_threshold_m=float(
                self.get_parameter('goal_update_threshold_m').value,
            ),
            explicit_goal_tolerance_m=float(
                self.get_parameter('explicit_goal_tolerance_m').value,
            ),
            explicit_goal_handoff_hysteresis_m=float(self.get_parameter(
                'explicit_goal_handoff_hysteresis_m',
            ).value),
            max_reacquisitions=int(self.get_parameter('max_reacquisitions').value),
            max_replans=int(self.get_parameter('max_replans').value),
        ))
        self._lock = threading.RLock()
        # Bind tf2's jump callback to this node's ROS clock so a restarted
        # simulator clears transforms from the previous clock epoch.
        self._tf_buffer = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._task_instruction = ''
        self._task_key = ''
        self._goal_id = ''
        self._work_pose_map_frame = ''
        self._explicit_goal_xy: np.ndarray | None = None
        self._map_goal_yaw_rad: float | None = None
        self._work_pose_source: dict[str, Any] | None = None
        self._retired_task_keys: list[str] = []
        self._suggested_displacement: float | None = None
        self._perception_valid = False
        self._target_stamp: float | None = None
        self._target_depth: float | None = None
        self._target_xy: np.ndarray | None = None
        self._base_xy: np.ndarray | None = None
        self._base_speed = float('inf')
        self._odom_stamp_s: float | None = None
        self._navigation_healthy = True
        self._odometry_healthy = False
        self._goal_reached = False
        self._goal_false_seen = False
        self._goal_reset_pending = False
        self._goal_reset_requested_at: float | None = None
        self._pending_waypoint_xy: np.ndarray | None = None
        self._last_waypoint_at: float | None = None
        self._last_status = ''
        self._last_status_at: float | None = None
        self._setup_io()
        self.create_timer(float(self.get_parameter('control_period_s').value), self._tick)
        self.get_logger().info('ready: observed-target coarse navigation')

    def _declare_parameters(self) -> None:
        defaults = {
            'map_frame': 'map',
            'task_status_topic': '/z_manip/task/status',
            'perception_valid_topic': '/z_manip/perception/valid',
            'target_topic': '/z_manip/perception/target_3d',
            'odometry_topic': '/odom_base_link',
            'platform_base_frame': 'base_link',
            'navigation_health_topic': '/state_estimation_health',
            'goal_reached_topic': '/goal_reached',
            'waypoint_topic': '/way_point',
            'cancel_topic': '/cancel_goal',
            'coarse_ready_topic': '/z_manip/navigation/coarse_ready',
            'grounding_request_topic': '/z_manip/grounding/request',
            'status_topic': '/z_manip/navigation/status',
            'near_target_depth_m': 1.4,
            'still_speed_mps': 0.035,
            'still_settle_s': 0.35,
            'target_timeout_s': 0.55,
            'observation_wait_timeout_s': 12.0,
            'navigation_timeout_s': 90.0,
            'stall_timeout_s': 8.0,
            'progress_min_net_decrease_m': 0.005,
            'progress_min_slope_mps': 0.001,
            'odometry_timeout_s': 0.50,
            'min_displacement_m': 0.10,
            'max_displacement_m': 3.0,
            'goal_update_threshold_m': 0.20,
            'explicit_goal_tolerance_m': 0.25,
            'explicit_goal_handoff_hysteresis_m': 0.08,
            'max_reacquisitions': 2,
            'max_replans': 3,
            'waypoint_refresh_s': 1.0,
            'goal_reset_ack_timeout_s': 2.0,
            'angular_speed_weight_m': 0.30,
            'tf_timeout_s': 0.12,
            'tf_latest_tolerance_s': 0.10,
            'control_period_s': 0.05,
            'status_heartbeat_s': 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _topic(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _setup_io(self) -> None:
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._waypoint_pub = self.create_publisher(
            PointStamped, self._topic('waypoint_topic'), reliable,
        )
        self._cancel_pub = self.create_publisher(
            Bool, self._topic('cancel_topic'), reliable,
        )
        self._ready_pub = self.create_publisher(
            Bool, self._topic('coarse_ready_topic'), latched,
        )
        self._grounding_pub = self.create_publisher(
            String, self._topic('grounding_request_topic'), reliable,
        )
        self._status_pub = self.create_publisher(
            String, self._topic('status_topic'), latched,
        )
        self.create_subscription(
            String, self._topic('task_status_topic'), self._task_status_cb, latched,
        )
        self.create_subscription(
            Bool, self._topic('perception_valid_topic'), self._valid_cb, reliable,
        )
        self.create_subscription(
            Detection3D, self._topic('target_topic'), self._target_cb, reliable,
        )
        self.create_subscription(
            Odometry, self._topic('odometry_topic'), self._odom_cb, reliable,
        )
        self.create_subscription(
            Bool, self._topic('navigation_health_topic'), self._health_cb, reliable,
        )
        self.create_subscription(
            Bool, self._topic('goal_reached_topic'), self._goal_reached_cb, reliable,
        )
        self._ready_pub.publish(Bool(data=False))

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _lookup_map_transform(self, source_frame: str, stamp: Any) -> Any:
        """
        Use exact sensor time, with a bounded latest-TF fallback.

        Isaac's sensor and TF publishers can differ by a few milliseconds. A
        future extrapolation within the configured tolerance is safe for coarse
        XY navigation because the downstream visual servo closes the final gap;
        larger or stale discrepancies remain a hard perception failure.
        """
        timeout = Duration(seconds=float(self.get_parameter('tf_timeout_s').value))
        try:
            return self._tf_buffer.lookup_transform(
                self._topic('map_frame'), source_frame,
                Time.from_msg(stamp), timeout=timeout,
            )
        except TransformException:
            latest = self._tf_buffer.lookup_transform(
                self._topic('map_frame'), source_frame, Time(), timeout=timeout,
            )
            age = self._now_s() - (
                float(stamp.sec) + float(stamp.nanosec) * 1e-9
            )
            tolerance = float(self.get_parameter('tf_latest_tolerance_s').value)
            if not math.isfinite(age) or abs(age) > tolerance:
                raise
            return latest

    def _task_status_cb(self, msg: String) -> None:
        with self._lock:
            try:
                value = json.loads(msg.data)
                request = parse_task_navigation_request(value)
                if request is None:
                    self._deactivate_task()
                else:
                    self._activate_task(request)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self.get_logger().error(f'task status rejected: {error}')

    def _activate_task(self, request: NavigationTaskRequest) -> None:
        """Activate a new immutable goal contract or refresh the active one."""
        if request.uses_explicit_goal:
            configured_frame = self._topic('map_frame')
            if request.map_frame != configured_frame:
                raise ValueError(
                    'work_pose map_frame does not match navigation map_frame',
                )
        if request.task_key in self._retired_task_keys:
            self.get_logger().warning(
                f'ignoring retired navigation goal {request.task_key!r}',
            )
            return
        if request.task_key == self._task_key:
            if request.instruction != self._task_instruction:
                raise ValueError('active goal_id was reused for another instruction')
            if request.uses_explicit_goal != (self._explicit_goal_xy is not None):
                raise ValueError('active navigation goal mode changed')
            if request.uses_explicit_goal:
                assert request.map_goal_xy is not None
                if (
                    not np.array_equal(request.map_goal_xy, self._explicit_goal_xy)
                    or request.map_goal_yaw_rad != self._map_goal_yaw_rad
                    or request.source != self._work_pose_source
                ):
                    raise ValueError('active work_pose changed without a new goal_id')
            else:
                self._suggested_displacement = request.suggested_displacement_m
            return

        if self._task_key:
            self._retire_task_key(self._task_key)
        self._task_instruction = request.instruction
        self._task_key = request.task_key
        self._goal_id = request.goal_id or ''
        self._work_pose_map_frame = request.map_frame or ''
        self._explicit_goal_xy = (
            None if request.map_goal_xy is None else request.map_goal_xy.copy()
        )
        self._map_goal_yaw_rad = request.map_goal_yaw_rad
        self._work_pose_source = (
            None if request.source is None else dict(request.source)
        )
        self._suggested_displacement = request.suggested_displacement_m
        now = self._now_s()
        self._core.begin(request.instruction, request.task_key, stamp_s=now)
        self._ready_pub.publish(Bool(data=False))
        self._goal_reached = False
        self._goal_false_seen = False
        self._goal_reset_pending = True
        self._goal_reset_requested_at = now
        self._pending_waypoint_xy = None
        self._last_waypoint_at = None
        # Establish a causal reset edge before publishing this task's first
        # waypoint. The upstream planner does not publish false when it merely
        # receives a new waypoint, but it does acknowledge an explicit cancel.
        self._cancel_pub.publish(Bool(data=True))

    def _retire_task_key(self, task_key: str) -> None:
        """Remember recent completed goal IDs so delayed status cannot revive them."""
        if not task_key or task_key in self._retired_task_keys:
            return
        self._retired_task_keys.append(task_key)
        del self._retired_task_keys[:-128]

    def _deactivate_task(self) -> None:
        """Cancel and clear the active task when orchestration leaves coarse nav."""
        if not self._task_key:
            return
        if self._core.phase in (
            NavPhase.WAIT_OBSERVATION, NavPhase.NAVIGATING, NavPhase.REACQUIRE,
        ):
            self._cancel_pub.publish(Bool(data=True))
        self._retire_task_key(self._task_key)
        self._core.reset()
        self._ready_pub.publish(Bool(data=False))
        self._task_instruction = ''
        self._task_key = ''
        self._goal_id = ''
        self._work_pose_map_frame = ''
        self._explicit_goal_xy = None
        self._map_goal_yaw_rad = None
        self._work_pose_source = None
        self._suggested_displacement = None
        self._goal_reached = False
        self._goal_false_seen = False
        self._goal_reset_pending = False
        self._goal_reset_requested_at = None
        self._pending_waypoint_xy = None
        self._last_waypoint_at = None

    def _valid_cb(self, msg: Bool) -> None:
        with self._lock:
            self._perception_valid = bool(msg.data)

    def _target_cb(self, msg: Detection3D) -> None:
        with self._lock:
            try:
                raw = np.array([
                    msg.bbox.center.position.x,
                    msg.bbox.center.position.y,
                    msg.bbox.center.position.z,
                ], dtype=float)
                if not np.all(np.isfinite(raw)) or raw[2] <= 0.0:
                    raise ValueError('target center is invalid')
                transform = self._lookup_map_transform(
                    msg.header.frame_id, msg.header.stamp,
                )
                mapped = _transform_point(transform.transform, raw)
                self._target_xy = mapped[:2]
                self._target_depth = float(raw[2])
                self._target_stamp = _stamp_s(msg.header)
            except (TransformException, ValueError) as error:
                self._perception_valid = False
                self.get_logger().error(f'target transform failed: {error}')

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            try:
                expected_parent = self._topic('map_frame')
                expected_child = self._topic('platform_base_frame')
                if (
                    msg.header.frame_id != expected_parent
                    or msg.child_frame_id != expected_child
                ):
                    raise ValueError(
                        'platform odometry frame mismatch: '
                        f'expected {expected_parent!r}->{expected_child!r}, got '
                        f'{msg.header.frame_id!r}->{msg.child_frame_id!r}',
                    )
                raw = np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ], dtype=float)
                if not np.all(np.isfinite(raw)):
                    raise ValueError('platform odometry position is non-finite')
                odom_stamp_s = _stamp_s(msg.header)
                if not math.isfinite(odom_stamp_s) or odom_stamp_s < 0.0:
                    raise ValueError('platform odometry stamp is invalid')
                if (
                    self._odom_stamp_s is not None
                    and odom_stamp_s <= self._odom_stamp_s
                ):
                    self.get_logger().warning(
                        'ignored non-increasing platform odometry stamp: '
                        f'{odom_stamp_s:.9f} <= {self._odom_stamp_s:.9f}',
                    )
                    return
                self._base_xy = raw[:2]
                linear = msg.twist.twist.linear
                angular = msg.twist.twist.angular
                angular_weight = float(self.get_parameter('angular_speed_weight_m').value)
                velocity = np.array((linear.x, linear.y, angular.z), dtype=float)
                if not np.all(np.isfinite(velocity)):
                    raise ValueError('platform odometry velocity is non-finite')
                self._base_speed = math.sqrt(
                    velocity[0] * velocity[0] + velocity[1] * velocity[1]
                    + (angular_weight * velocity[2]) ** 2,
                )
                self._odom_stamp_s = odom_stamp_s
                self._odometry_healthy = True
            except (TypeError, ValueError) as error:
                self._base_xy = None
                self._base_speed = float('inf')
                self._odom_stamp_s = None
                self._odometry_healthy = False
                self.get_logger().error(f'platform odometry rejected: {error}')

    def _health_cb(self, msg: Bool) -> None:
        with self._lock:
            self._navigation_healthy = bool(msg.data)

    def _goal_reached_cb(self, msg: Bool) -> None:
        with self._lock:
            if not msg.data:
                self._goal_false_seen = True
                self._goal_reached = False
                self._goal_reset_pending = False
            elif self._goal_false_seen:
                self._goal_reached = True

    def _tick(self) -> None:
        with self._lock:
            now = self._now_s()
            if self._goal_reset_pending:
                requested_at = self._goal_reset_requested_at
                timeout = float(self.get_parameter(
                    'goal_reset_ack_timeout_s',
                ).value)
                elapsed = (
                    float('inf') if requested_at is None
                    else now - requested_at
                )
                if (
                    not math.isfinite(timeout)
                    or timeout <= 0.0
                    or not math.isfinite(elapsed)
                    or elapsed < 0.0
                    or elapsed > timeout
                ):
                    decision = self._core.fail(
                        'local planner goal reset acknowledgement timed out',
                    )
                    self._goal_reset_pending = False
                    self._cancel_pub.publish(Bool(data=True))
                    self._publish_status(decision.reason)
                    return
                self._publish_status(
                    'waiting for local planner goal reset acknowledgement',
                )
                return
            if self._pending_waypoint_xy is not None:
                waypoint = self._pending_waypoint_xy.copy()
                self._pending_waypoint_xy = None
                self._publish_waypoint(waypoint, now, new_goal=True)
                self._publish_status(
                    'waypoint published after local planner reset acknowledgement',
                )
                return
            previous_phase = self._core.phase
            decision = self._core.update(NavInput(
                stamp_s=now,
                perception_valid=self._perception_valid,
                target_stamp_s=self._target_stamp,
                target_depth_m=self._target_depth,
                base_xy=self._base_xy,
                target_xy=self._target_xy,
                suggested_displacement_m=self._suggested_displacement,
                base_speed_mps=self._base_speed,
                odom_stamp_s=self._odom_stamp_s,
                navigation_healthy=(
                    self._navigation_healthy and self._odometry_healthy
                ),
                goal_reached=self._goal_reached,
                explicit_goal_xy=self._explicit_goal_xy,
            ))
            if (
                previous_phase is NavPhase.READY
                and self._core.phase is not NavPhase.READY
            ):
                self._ready_pub.publish(Bool(data=False))
            if decision.cancel_navigation:
                self._cancel_pub.publish(Bool(data=True))
            if decision.request_reacquire and self._task_instruction:
                self._grounding_pub.publish(String(data=self._task_instruction))
            if decision.waypoint_xy is not None:
                if self._last_waypoint_at is None:
                    self._publish_waypoint(decision.waypoint_xy, now, new_goal=True)
                else:
                    self._pending_waypoint_xy = decision.waypoint_xy.copy()
                    self._goal_reached = False
                    self._goal_false_seen = False
                    self._goal_reset_pending = True
                    self._goal_reset_requested_at = now
                    self._cancel_pub.publish(Bool(data=True))
            elif (
                self._core.phase is NavPhase.NAVIGATING
                # localPlanner treats every PointStamped as a new goal and
                # clears its goalReached latch.  Refreshing an immutable work
                # pose can therefore erase the reached edge between terrain
                # updates and drive the platform straight through the goal.
                # Explicit goals are republished only by the causal
                # cancel/false-ack replan path above.
                and not self._core.uses_explicit_goal
                and self._core.goal_xy is not None
                and (
                    self._last_waypoint_at is None
                    or now - self._last_waypoint_at
                    >= float(self.get_parameter('waypoint_refresh_s').value)
                )
            ):
                self._publish_waypoint(self._core.goal_xy, now, new_goal=False)
            if decision.coarse_ready:
                self._ready_pub.publish(Bool(data=True))
            self._publish_status(decision.reason)

    def _publish_waypoint(self, xy: np.ndarray, now: float, *, new_goal: bool) -> None:
        message = PointStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._topic('map_frame')
        message.point.x = float(xy[0])
        message.point.y = float(xy[1])
        self._waypoint_pub.publish(message)
        self._last_waypoint_at = now
        if new_goal:
            self._goal_reached = False
            self._core.arm_current_goal()

    def _publish_status(self, reason: str) -> None:
        now = self._now_s()
        value = json.dumps({
            'schema': 'z_manip.navigation_status.v1',
            'phase': self._core.phase.value,
            'task_key': self._core.task_key,
            'goal_id': self._goal_id or None,
            'map_frame': self._work_pose_map_frame or self._topic('map_frame'),
            'perception_valid': self._perception_valid,
            'navigation_healthy': (
                self._navigation_healthy and self._odometry_healthy
            ),
            'goal_xy': None if self._core.goal_xy is None else self._core.goal_xy.tolist(),
            'map_goal_xy': (
                None if self._explicit_goal_xy is None
                else self._explicit_goal_xy.tolist()
            ),
            'map_goal_yaw_rad': self._map_goal_yaw_rad,
            'coarse_goal_check': 'xy_only',
            'goal_reset_acknowledged': (
                self._goal_false_seen and not self._goal_reset_pending
            ),
            'work_pose_source': self._work_pose_source,
            'target_depth_m': self._target_depth,
            'suggested_displacement_m': self._suggested_displacement,
            'progress': {
                'window_duration_s': self._core.progress_window_duration_s,
                'net_decrease_m': self._core.progress_net_decrease_m,
                'slope_mps': self._core.progress_slope_mps,
                'odom_age_s': self._core.progress_odom_age_s,
            },
            'replans': self._core.replan_count,
            'reacquisitions': self._core.reacquisition_count,
            'reason': reason or self._core.failure_reason,
        }, separators=(',', ':'))
        heartbeat_s = float(self.get_parameter('status_heartbeat_s').value)
        heartbeat_due = (
            self._last_status_at is None
            or now < self._last_status_at
            or now - self._last_status_at >= heartbeat_s
        )
        if value != self._last_status or heartbeat_due:
            self._status_pub.publish(String(data=value))
            self._last_status = value
            self._last_status_at = now


def main(args: list[str] | None = None) -> None:
    """Run the observed-target coarse navigation node."""
    rclpy.init(args=args)
    node = CoarseNavigationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
