"""ROS 2 adapter for synchronized, observed placement planning."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

from builtin_interfaces.msg import Duration as DurationMessage
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Point, PoseArray
import message_filters
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from z_manip.collision import RobotCollisionModel
from z_manip.kinematics import KinematicChain
from z_manip.planning import ObservedPlacementConfig, ObservedPlacementPlanner
from z_manip.planning_control import PlanningControl
from z_manip.trajectory_digest import canonical_joint_trajectory_sha256

from .attached_collision import (
    AttachedCollisionAuditConfig,
    AttachedObjectPathAuditor,
)
from .core import (
    backproject_depth,
    capture_observed_region_geometry,
    capture_planned_object_geometry,
    ObservedPerceptionIdentity,
    parse_region_request,
    PlaceExecutionCorrelation,
    PlacementContractError,
    PlacementCoordinator,
    PlacementPerceptionSnapshot,
    PlacementRegionRequest,
    PostReleaseObservation,
    PostReleasePlacementVerifier,
    PostReleaseVerificationConfig,
    PostReleaseVerificationResult,
)
from .moveit_evaluator import _pose, MoveItPlacementConfig, MoveItPlacementEvaluator
from .transaction import (
    parse_region_transaction_identity,
    parse_transaction_control,
    PlacementTransactionLifecycle,
    PlaceTerminalFailure,
    PlaceTransactionError,
    PlaceTransactionIdentity,
    PlaceTransactionToken,
)


@dataclass(frozen=True, eq=False)
class _RgbdCache:
    rgb_stamp_ns: int
    depth_stamp_ns: int
    camera_info_stamp_ns: int
    source_frame: str
    organized_points: np.ndarray


@dataclass(frozen=True, eq=False)
class _SceneCache:
    stamp_ns: int
    points: np.ndarray


@dataclass(frozen=True, eq=False)
class _JointCache:
    stamp_ns: int
    names: tuple[str, ...]
    positions: np.ndarray
    planning_from_kinematic_base: np.ndarray
    planning_from_tool_fk: np.ndarray


@dataclass(frozen=True, eq=False)
class _TargetCache:
    stamp_ns: int
    source_frame: str
    points: np.ndarray
    pixels_uv: np.ndarray
    gripper_probe_points: np.ndarray
    planning_from_tool: np.ndarray


@dataclass(frozen=True)
class _ExecutionFeedback:
    received_ns: int
    executor_epoch: str
    trajectory_status: str
    trajectory_owner: str
    trajectory_segment: str
    trajectory_command_id: int
    trajectory_source_stamp_ns: int
    trajectory_contract_id: str
    gripper_command_id: int
    gripper_source_stamp_ns: int
    aperture_m: float


@dataclass(frozen=True, eq=False)
class _PlacementPlanInputs:
    """One exact request-keyed snapshot consumed by a single worker."""

    request: PlacementRegionRequest
    identity: ObservedPerceptionIdentity
    rgbd: _RgbdCache
    scene: _SceneCache
    joints: _JointCache
    target: _TargetCache
    execution: _ExecutionFeedback
    token: PlaceTransactionToken


@dataclass(frozen=True)
class _PendingRelease:
    acknowledgement_stamp_ns: int
    gripper_command_id: int
    gripper_source_stamp_ns: int


def _stamp_ns(stamp: object) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _transform_matrix(transform: object) -> np.ndarray:
    translation = transform.transform.translation
    quaternion = transform.transform.rotation
    q = np.asarray((quaternion.x, quaternion.y, quaternion.z, quaternion.w), dtype=float)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-9:
        raise PlacementContractError('TF rotation quaternion is invalid')
    x, y, z, w = q / norm
    rotation = np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)),
    ))
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = (translation.x, translation.y, translation.z)
    return result


class ObservedPlacementNode(Node):
    """Synchronize observed inputs and publish only fully evaluated placement."""

    def __init__(self) -> None:
        super().__init__('z_manip_observed_placement')
        self._declare_parameters()
        self._planning_frame = str(self.get_parameter('planning_frame').value)
        self._tool_link = str(self.get_parameter('tool_link').value)
        self._kinematic_base_link = str(
            self.get_parameter('kinematic_base_link').value,
        )
        self._joint_names = tuple(self.get_parameter('arm_joint_names').value)
        robot_description_file = Path(str(
            self.get_parameter('robot_description_file').value,
        )).expanduser()
        if not robot_description_file.is_file():
            raise ValueError(
                'robot_description_file must name a readable URDF for FK validation',
            )
        self._kinematic_chain = KinematicChain.from_urdf(
            robot_description_file,
            self._kinematic_base_link,
            self._tool_link,
        )
        if self._kinematic_chain.joint_names != self._joint_names:
            raise ValueError(
                'URDF kinematic joint order does not match arm_joint_names',
            )
        collision_model_file = Path(str(
            self.get_parameter('collision_model_file').value,
        )).expanduser()
        if not collision_model_file.is_file():
            raise ValueError(
                'collision_model_file must name a readable robot model',
            )
        collision_model_data = json.loads(
            collision_model_file.read_text(encoding='utf-8'),
        )
        if not isinstance(collision_model_data, dict):
            raise ValueError('collision_model_file must contain a JSON object')
        self._collision_model = RobotCollisionModel.from_mapping(
            collision_model_data,
        )
        self._attached_collision_config_value = self._attached_collision_config()
        self._gravity = np.asarray(self.get_parameter('gravity').value, dtype=float)
        self._planner_config_value = self._planner_config()
        self._moveit_config_value = self._moveit_config()
        self._max_sync_skew_s = float(
            self.get_parameter('max_sync_skew_s').value,
        )
        self._max_snapshot_age_s = float(
            self.get_parameter('max_snapshot_age_s').value,
        )
        self._motion_plan_service = str(
            self.get_parameter('motion_plan_service').value,
        )
        self._cartesian_path_service = str(
            self.get_parameter('cartesian_path_service').value,
        )
        self._bridge = CvBridge()
        # Bind tf2's jump callback to this node's ROS clock so a restarted
        # simulator clears transforms from the previous clock epoch.
        self._tf_buffer = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._lock = threading.Lock()
        self._planning_rgbd: OrderedDict[
            tuple[int, str], _RgbdCache
        ] = OrderedDict()
        self._planning_rgbd_conflicts: OrderedDict[
            tuple[int, str], None
        ] = OrderedDict()
        self._scene: _SceneCache | None = None
        self._joints: _JointCache | None = None
        self._request: PlacementRegionRequest | None = None
        self._request_identity: ObservedPerceptionIdentity | None = None
        self._request_transaction_token: PlaceTransactionToken | None = None
        self._planning_identity: ObservedPerceptionIdentity | None = None
        self._worker: threading.Thread | None = None
        self._workers: dict[int, threading.Thread] = {}
        self._transaction = PlacementTransactionLifecycle(
            ros_timeout_s=float(
                self.get_parameter('transaction_ros_timeout_s').value,
            ),
            wall_timeout_s=float(
                self.get_parameter('transaction_wall_timeout_s').value,
            ),
        )
        self._post_release = PostReleasePlacementVerifier(
            self._post_release_config(),
        )
        self._place_execution = PlaceExecutionCorrelation()
        self._pending_release: _PendingRelease | None = None
        self._execution_feedback: _ExecutionFeedback | None = None
        self._verification_rgbd: OrderedDict[
            tuple[int, str], _RgbdCache
        ] = OrderedDict()
        self._verification_targets: OrderedDict[
            tuple[int, str], _TargetCache
        ] = OrderedDict()
        self._verification_identities: OrderedDict[
            tuple[int, str], ObservedPerceptionIdentity
        ] = OrderedDict()
        self._verification_identity_conflicts: OrderedDict[
            tuple[int, str], None
        ] = OrderedDict()
        self._verification_joints: OrderedDict[int, _JointCache] = OrderedDict()
        raw_probe_points = np.asarray(
            self.get_parameter('gripper_probe_points_tool_flat').value,
            dtype=float,
        )
        if (
            raw_probe_points.ndim != 1
            or len(raw_probe_points) == 0
            or len(raw_probe_points) % 3
            or not np.all(np.isfinite(raw_probe_points))
        ):
            raise ValueError(
                'gripper_probe_points_tool_flat must contain finite xyz triples',
            )
        self._gripper_probe_points_tool = raw_probe_points.reshape(-1, 3)

        self._status_publisher = self.create_publisher(
            String, str(self.get_parameter('status_topic').value), 10,
        )
        self._candidate_publisher = self.create_publisher(
            MarkerArray, str(self.get_parameter('candidate_markers_topic').value), 5,
        )
        self._selected_publisher = self.create_publisher(
            PoseArray, str(self.get_parameter('selected_poses_topic').value), 5,
        )
        self._trajectory_publisher = self.create_publisher(
            JointTrajectory, str(self.get_parameter('trajectory_topic').value), 5,
        )
        self._contract_publisher = self.create_publisher(
            String, str(self.get_parameter('trajectory_contract_topic').value), 5,
        )
        post_release_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._post_release_publisher = self.create_publisher(
            String,
            str(self.get_parameter('post_release_verification_topic').value),
            post_release_qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('scene_cloud_topic').value),
            self._on_scene,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('joint_state_topic').value),
            self._on_joints,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('region_request_topic').value),
            self._on_region,
            10,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter('target_cloud_topic').value),
            self._on_target_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            DiagnosticArray,
            str(self.get_parameter('perception_status_topic').value),
            self._on_perception_status,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('execution_status_topic').value),
            self._on_execution_status,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter('transaction_control_topic').value),
            self._on_transaction_control,
            10,
        )
        color = message_filters.Subscriber(
            self, Image, str(self.get_parameter('color_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        depth = message_filters.Subscriber(
            self, Image, str(self.get_parameter('depth_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        info = message_filters.Subscriber(
            self, CameraInfo, str(self.get_parameter('camera_info_topic').value),
            qos_profile=qos_profile_sensor_data,
        )
        self._rgbd_sync = message_filters.TimeSynchronizer(
            (color, depth, info),
            queue_size=int(self.get_parameter('sync_queue_size').value),
        )
        self._rgbd_sync.registerCallback(self._on_rgbd)
        self.create_service(
            Trigger,
            str(self.get_parameter('plan_service').value),
            self._on_plan_service,
        )
        self.create_timer(0.10, self._maybe_auto_plan)
        self.create_timer(0.10, self._post_release_tick)
        watchdog_period = float(
            self.get_parameter('transaction_watchdog_period_s').value,
        )
        if not math.isfinite(watchdog_period) or not 0.0 < watchdog_period <= 0.25:
            raise ValueError(
                'transaction_watchdog_period_s must be finite and in (0, 0.25]',
            )
        self._transaction_watchdog_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(
            watchdog_period,
            self._transaction_watchdog_tick,
            clock=self._transaction_watchdog_clock,
        )
        self._publish_status('waiting', 'waiting for synchronized observed inputs')

    def _declare_parameters(self) -> None:
        topics = {
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'scene_cloud_topic': '/z_manip/perception/scene_pointcloud',
            'joint_state_topic': '/piper/state',
            'region_request_topic': '/z_manip/place/region_request',
            'candidate_markers_topic': '/z_manip/place/candidates',
            'selected_poses_topic': '/z_manip/place/selected_poses',
            'trajectory_topic': '/z_manip/place/trajectory',
            'trajectory_contract_topic': '/z_manip/place/trajectory_contract',
            'post_release_verification_topic': (
                '/z_manip/place/post_release_verification'
            ),
            'status_topic': '/z_manip/place/status',
            'plan_service': '/z_manip/place/plan',
            'target_cloud_topic': '/z_manip/perception/target_pointcloud',
            'perception_status_topic': '/z_manip/perception/status',
            'execution_status_topic': '/piper/execution_status',
            'transaction_control_topic': '/z_manip/place/transaction_control',
            'motion_plan_service': '/plan_kinematic_path',
            'cartesian_path_service': '/compute_cartesian_path',
        }
        for name, value in topics.items():
            self.declare_parameter(name, value)
        self.declare_parameter('planning_frame', 'base')
        self.declare_parameter('robot_description_file', '')
        self.declare_parameter('collision_model_file', '')
        self.declare_parameter('kinematic_base_link', 'piper_base_link')
        self.declare_parameter('planning_group', 'piper_arm')
        self.declare_parameter('tool_link', 'piper_gripper_base')
        self.declare_parameter(
            'arm_joint_names', [f'piper_joint{index}' for index in range(1, 7)],
        )
        self.declare_parameter('joint_velocity_limits', [1.0] * 6)
        self.declare_parameter('gravity', [0.0, 0.0, -1.0])
        self.declare_parameter('auto_plan_on_region', True)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('planning_rgbd_cache_size', 20)
        self.declare_parameter('max_sync_skew_s', 0.08)
        self.declare_parameter('max_snapshot_age_s', 0.35)
        self.declare_parameter('tf_timeout_s', 0.08)
        self.declare_parameter('min_depth_m', 0.20)
        self.declare_parameter('max_depth_m', 3.0)
        self.declare_parameter('transaction_ros_timeout_s', 30.0)
        self.declare_parameter('transaction_wall_timeout_s', 60.0)
        self.declare_parameter('transaction_watchdog_period_s', 0.05)
        defaults = ObservedPlacementConfig()
        for name in (
            'ransac_iterations', 'min_plane_points', 'max_ransac_points',
            'footprint_samples_per_axis', 'yaw_samples',
            'max_geometric_candidates', 'seed',
        ):
            self.declare_parameter(name, int(getattr(defaults, name)))
        for name in (
            'ransac_distance_m', 'max_plane_rms_m', 'min_plane_inlier_ratio',
            'max_surface_tilt_rad', 'sample_spacing_m',
            'support_neighbor_radius_m', 'boundary_margin_m',
            'plane_exclusion_m', 'obstacle_height_margin_m',
            'tool_clearance_radius_m', 'preplace_distance_m',
            'retreat_distance_m', 'support_score_weight',
            'clearance_score_weight', 'centrality_score_weight',
        ):
            self.declare_parameter(name, float(getattr(defaults, name)))
        self.declare_parameter('planning_pipeline', 'ompl')
        self.declare_parameter('planner_id', 'RRTConnect')
        self.declare_parameter('planning_attempts', 3)
        self.declare_parameter('allowed_planning_time_s', 3.0)
        self.declare_parameter('moveit_service_wait_timeout_s', 1.0)
        self.declare_parameter('moveit_response_timeout_s', 5.0)
        self.declare_parameter('position_tolerance_m', 0.008)
        self.declare_parameter('orientation_tolerance_rad', 0.06)
        self.declare_parameter('cartesian_step_m', 0.008)
        self.declare_parameter('cartesian_jump_threshold', 0.0)
        self.declare_parameter('min_cartesian_fraction', 0.995)
        self.declare_parameter('min_waypoint_duration_s', 0.04)
        self.declare_parameter('continuity_tolerance_rad', 0.04)
        self.declare_parameter('attached_collision_clearance_m', 0.02)
        self.declare_parameter('attached_collision_point_radius_m', 0.005)
        self.declare_parameter('attached_collision_segment_joint_step_rad', 0.025)
        self.declare_parameter('attached_collision_min_scene_points', 32)
        self.declare_parameter('attached_collision_max_attached_points', 512)
        self.declare_parameter('attached_collision_extent_samples_per_axis', 5)
        self.declare_parameter(
            'attached_collision_carried_object_scene_exclusion_m',
            0.012,
        )
        self.declare_parameter('post_release_min_stable_duration_s', 0.50)
        self.declare_parameter('post_release_min_samples', 3)
        self.declare_parameter('post_release_min_target_points', 40)
        self.declare_parameter('post_release_min_current_support_points', 24)
        self.declare_parameter('post_release_max_target_motion_m', 0.025)
        self.declare_parameter('post_release_min_region_support_fraction', 0.80)
        self.declare_parameter('post_release_min_gripper_clearance_m', 0.04)
        self.declare_parameter('post_release_min_gripper_aperture_m', 0.065)
        self.declare_parameter('post_release_max_sync_skew_s', 0.025)
        self.declare_parameter('post_release_max_observation_age_s', 0.35)
        self.declare_parameter('post_release_max_state_skew_s', 0.12)
        self.declare_parameter('post_release_observation_timeout_s', 1.0)
        self.declare_parameter('post_release_max_observation_gap_s', 0.30)
        self.declare_parameter('post_release_plane_distance_tolerance_m', 0.015)
        self.declare_parameter('post_release_region_neighbor_radius_m', 0.04)
        self.declare_parameter('post_release_support_neighbor_radius_m', 0.05)
        self.declare_parameter('post_release_bottom_band_m', 0.025)
        self.declare_parameter('post_release_max_bottom_height_m', 0.035)
        self.declare_parameter('post_release_max_plane_penetration_m', 0.012)
        self.declare_parameter('post_release_target_mask_dilation_px', 2)
        self.declare_parameter('post_release_max_geometry_samples', 4096)
        self.declare_parameter('post_release_gripper_probe_radius_m', 0.031)
        self.declare_parameter(
            'post_release_target_depth_correspondence_tolerance_m',
            0.012,
        )
        self.declare_parameter('post_release_object_position_tolerance_m', 0.04)
        self.declare_parameter(
            'post_release_object_orientation_tolerance_rad',
            0.35,
        )
        self.declare_parameter('post_release_upright_tolerance_rad', 0.26)
        self.declare_parameter('post_release_orientation_degeneracy_ratio', 1.20)
        self.declare_parameter('post_release_max_axial_transverse_ratio', 1.90)
        self.declare_parameter(
            'post_release_max_symmetry_axis_alignment_error_rad',
            0.12,
        )
        self.declare_parameter(
            'post_release_min_signed_upright_profile_asymmetry',
            0.10,
        )
        self.declare_parameter(
            'post_release_min_signed_upright_profile_alignment',
            0.60,
        )
        self.declare_parameter(
            'post_release_max_object_orientation_motion_rad',
            0.10,
        )
        self.declare_parameter('post_release_registration_distance_m', 0.035)
        self.declare_parameter(
            'post_release_min_registration_inlier_fraction',
            0.55,
        )
        self.declare_parameter('post_release_max_registration_rms_m', 0.025)
        self.declare_parameter('post_release_min_object_reference_points', 40)
        self.declare_parameter('post_release_fk_position_tolerance_m', 0.015)
        self.declare_parameter('post_release_fk_orientation_tolerance_rad', 0.10)
        self.declare_parameter('post_release_max_rejection_diagnostics', 8)
        self.declare_parameter('post_release_region_capture_min_points', 80)
        self.declare_parameter('post_release_cache_size', 6)
        self.declare_parameter('post_release_joint_cache_size', 128)
        self.declare_parameter('gripper_probe_points_tool_flat', [
            -0.00475, -0.042, 0.0315,
            -0.00475, 0.042, 0.0315,
            0.0, 0.04825, 0.072,
            0.0, 0.04825, 0.110,
            0.0, -0.04825, 0.072,
            0.0, -0.04825, 0.110,
        ])

    def _planner_config(self) -> ObservedPlacementConfig:
        defaults = ObservedPlacementConfig()
        values = {
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
            if self.has_parameter(name)
        }
        values['max_sync_skew_s'] = float(
            self.get_parameter('max_sync_skew_s').value,
        )
        return ObservedPlacementConfig(**values)

    def _moveit_config(self) -> MoveItPlacementConfig:
        def value(name: str) -> object:
            return self.get_parameter(name).value

        return MoveItPlacementConfig(
            planning_frame=self._planning_frame,
            planning_group=str(value('planning_group')),
            tool_link=str(value('tool_link')),
            joint_names=self._joint_names,
            joint_velocity_limits=tuple(float(item) for item in value(
                'joint_velocity_limits',
            )),
            planning_pipeline=str(value('planning_pipeline')),
            planner_id=str(value('planner_id')),
            planning_attempts=int(value('planning_attempts')),
            allowed_planning_time_s=float(value('allowed_planning_time_s')),
            service_wait_timeout_s=float(value('moveit_service_wait_timeout_s')),
            response_timeout_s=float(value('moveit_response_timeout_s')),
            position_tolerance_m=float(value('position_tolerance_m')),
            orientation_tolerance_rad=float(value('orientation_tolerance_rad')),
            cartesian_step_m=float(value('cartesian_step_m')),
            cartesian_jump_threshold=float(value('cartesian_jump_threshold')),
            min_cartesian_fraction=float(value('min_cartesian_fraction')),
            min_waypoint_duration_s=float(value('min_waypoint_duration_s')),
            continuity_tolerance_rad=float(value('continuity_tolerance_rad')),
        )

    def _attached_collision_config(self) -> AttachedCollisionAuditConfig:
        def value(name: str) -> object:
            return self.get_parameter(f'attached_collision_{name}').value

        return AttachedCollisionAuditConfig(
            clearance_m=float(value('clearance_m')),
            point_radius_m=float(value('point_radius_m')),
            segment_joint_step_rad=float(value('segment_joint_step_rad')),
            min_scene_points=int(value('min_scene_points')),
            max_attached_points=int(value('max_attached_points')),
            extent_samples_per_axis=int(value('extent_samples_per_axis')),
            carried_object_scene_exclusion_m=float(value(
                'carried_object_scene_exclusion_m',
            )),
        )

    def _post_release_config(self) -> PostReleaseVerificationConfig:
        def value(name: str) -> object:
            return self.get_parameter(f'post_release_{name}').value

        return PostReleaseVerificationConfig(
            min_stable_duration_s=float(value('min_stable_duration_s')),
            min_samples=int(value('min_samples')),
            min_target_points=int(value('min_target_points')),
            min_current_support_points=int(value('min_current_support_points')),
            max_target_motion_m=float(value('max_target_motion_m')),
            min_region_support_fraction=float(value(
                'min_region_support_fraction',
            )),
            min_gripper_clearance_m=float(value('min_gripper_clearance_m')),
            min_gripper_aperture_m=float(value('min_gripper_aperture_m')),
            max_sync_skew_s=float(value('max_sync_skew_s')),
            max_observation_age_s=float(value('max_observation_age_s')),
            max_state_skew_s=float(value('max_state_skew_s')),
            observation_timeout_s=float(value('observation_timeout_s')),
            max_observation_gap_s=float(value('max_observation_gap_s')),
            plane_distance_tolerance_m=float(value(
                'plane_distance_tolerance_m',
            )),
            region_neighbor_radius_m=float(value('region_neighbor_radius_m')),
            support_neighbor_radius_m=float(value('support_neighbor_radius_m')),
            bottom_band_m=float(value('bottom_band_m')),
            max_bottom_height_m=float(value('max_bottom_height_m')),
            max_plane_penetration_m=float(value('max_plane_penetration_m')),
            target_mask_dilation_px=int(value('target_mask_dilation_px')),
            max_geometry_samples=int(value('max_geometry_samples')),
            gripper_probe_radius_m=float(value('gripper_probe_radius_m')),
            target_depth_correspondence_tolerance_m=float(value(
                'target_depth_correspondence_tolerance_m',
            )),
            object_position_tolerance_m=float(value(
                'object_position_tolerance_m',
            )),
            object_orientation_tolerance_rad=float(value(
                'object_orientation_tolerance_rad',
            )),
            upright_tolerance_rad=float(value('upright_tolerance_rad')),
            orientation_degeneracy_ratio=float(value(
                'orientation_degeneracy_ratio',
            )),
            max_axial_transverse_ratio=float(value(
                'max_axial_transverse_ratio',
            )),
            max_symmetry_axis_alignment_error_rad=float(value(
                'max_symmetry_axis_alignment_error_rad',
            )),
            min_signed_upright_profile_asymmetry=float(value(
                'min_signed_upright_profile_asymmetry',
            )),
            min_signed_upright_profile_alignment=float(value(
                'min_signed_upright_profile_alignment',
            )),
            max_object_orientation_motion_rad=float(value(
                'max_object_orientation_motion_rad',
            )),
            registration_distance_m=float(value('registration_distance_m')),
            min_registration_inlier_fraction=float(value(
                'min_registration_inlier_fraction',
            )),
            max_registration_rms_m=float(value('max_registration_rms_m')),
            min_object_reference_points=int(value(
                'min_object_reference_points',
            )),
            fk_position_tolerance_m=float(value('fk_position_tolerance_m')),
            fk_orientation_tolerance_rad=float(value(
                'fk_orientation_tolerance_rad',
            )),
            max_rejection_diagnostics=int(value('max_rejection_diagnostics')),
        )

    def _new_planning_backend(self):
        """Build generation-isolated planner state for one worker."""
        attached_collision = AttachedObjectPathAuditor(
            chain=self._kinematic_chain,
            collision_model=self._collision_model,
            config=self._attached_collision_config_value,
        )
        evaluator = MoveItPlacementEvaluator(
            self,
            self._moveit_config_value,
            motion_service=self._motion_plan_service,
            cartesian_service=self._cartesian_path_service,
            attached_collision_auditor=attached_collision,
        )
        coordinator = PlacementCoordinator(
            ObservedPlacementPlanner(self._planner_config_value),
            expected_joint_names=self._joint_names,
            max_sync_skew_s=self._max_sync_skew_s,
            max_snapshot_age_s=self._max_snapshot_age_s,
        )
        return coordinator, evaluator, attached_collision

    def _planning_cancelled(self, token: PlaceTransactionToken) -> bool:
        with self._lock:
            return not self._transaction.matches(token)

    def _release_planning_worker_locked(
        self,
        token: PlaceTransactionToken,
        worker: threading.Thread,
    ) -> None:
        """Release only the registry entries owned by one worker generation."""
        if self._workers.get(token.generation) is worker:
            self._workers.pop(token.generation, None)
        if self._worker is worker:
            self._worker = None

    def _lookup_transform(self, source_frame: str, stamp: object) -> np.ndarray:
        transform = self._tf_buffer.lookup_transform(
            self._planning_frame,
            source_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=float(self.get_parameter('tf_timeout_s').value)),
        )
        return _transform_matrix(transform)

    def _cache_put(
        self,
        cache: OrderedDict,
        key: object,
        value: object,
        *,
        limit_parameter: str = 'post_release_cache_size',
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        limit = int(self.get_parameter(limit_parameter).value)
        while len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def _exact_rgbd_source_key(
        color: Image,
        depth: Image,
        info: CameraInfo,
    ) -> tuple[int, str]:
        """Return the shared source key or reject any mixed RGB-D frame."""
        frames = (
            str(color.header.frame_id),
            str(depth.header.frame_id),
            str(info.header.frame_id),
        )
        if any(not frame.strip() for frame in frames) or len(set(frames)) != 1:
            raise PlacementContractError(
                'aligned color, depth, and camera-info frames must match exactly',
            )
        stamps = (
            _stamp_ns(color.header.stamp),
            _stamp_ns(depth.header.stamp),
            _stamp_ns(info.header.stamp),
        )
        if any(stamp <= 0 for stamp in stamps) or len(set(stamps)) != 1:
            raise PlacementContractError(
                'aligned color, depth, and camera-info stamps must match exactly',
            )
        return stamps[0], frames[0]

    @staticmethod
    def _validate_exact_planning_sources(
        request: PlacementRegionRequest,
        identity: ObservedPerceptionIdentity,
        rgbd: _RgbdCache,
        target: _TargetCache,
    ) -> None:
        """Bind every place-time perception input to the schema-v2 key."""
        key = (request.stamp_ns, request.image_frame)
        if identity != request.observation_identity:
            raise PlacementContractError(
                'placement request perception owner is not exact',
            )
        if (
            rgbd.rgb_stamp_ns,
            rgbd.depth_stamp_ns,
            rgbd.camera_info_stamp_ns,
            rgbd.source_frame,
        ) != (
            request.stamp_ns,
            request.stamp_ns,
            request.stamp_ns,
            request.image_frame,
        ):
            raise PlacementContractError(
                'placement request does not match exact RGB-D source key',
            )
        if (target.stamp_ns, target.source_frame) != key:
            raise PlacementContractError(
                'placement target model does not match exact request source key',
            )

    @staticmethod
    def _cache_exact_planning_rgbd(
        cache: OrderedDict[tuple[int, str], _RgbdCache],
        conflicts: OrderedDict[tuple[int, str], None],
        key: tuple[int, str],
        rgbd: _RgbdCache,
        *,
        limit: int,
    ) -> None:
        """Insert one immutable keyed frame and poison source-key collisions."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise PlacementContractError(
                'planning RGB-D cache size must be a positive integer',
            )
        if (
            rgbd.rgb_stamp_ns,
            rgbd.depth_stamp_ns,
            rgbd.camera_info_stamp_ns,
            rgbd.source_frame,
        ) != (key[0], key[0], key[0], key[1]):
            raise PlacementContractError(
                'planning RGB-D cache value does not match its exact source key',
            )
        if key in conflicts:
            return
        existing = cache.get(key)
        if existing is not None and not np.array_equal(
            existing.organized_points,
            rgbd.organized_points,
            equal_nan=True,
        ):
            cache.pop(key, None)
            conflicts[key] = None
            conflicts.move_to_end(key)
            while len(conflicts) > limit:
                conflicts.popitem(last=False)
            raise PlacementContractError(
                'multiple RGB-D payloads claimed one exact source key',
            )
        cache[key] = rgbd
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _reset_transaction_state_locked(self) -> None:
        """Clear every place-owned latch so a later request can retry."""
        self._transaction.reset()
        self._request = None
        self._request_identity = None
        self._request_transaction_token = None
        self._planning_identity = None
        self._post_release.reset()
        self._place_execution.reset()
        self._pending_release = None
        self._verification_rgbd.clear()
        self._verification_targets.clear()
        self._verification_identities.clear()
        self._verification_identity_conflicts.clear()
        self._verification_joints.clear()

    def _terminal_failure_locked(
        self,
        reason: str,
        *,
        expected: PlaceTransactionIdentity | None = None,
    ) -> PlaceTerminalFailure | None:
        """Atomically close only the expected transaction and retain its receipt."""
        identity = self._transaction.identity
        if identity is None or (expected is not None and identity != expected):
            return None
        detail = str(reason).strip() or 'unspecified placement transaction failure'
        failure = PlaceTerminalFailure(identity, detail[:1024])
        self._reset_transaction_state_locked()
        return failure

    def _publish_terminal_failure(
        self,
        failure: PlaceTerminalFailure | None,
    ) -> None:
        if failure is None:
            return
        self._status_publisher.publish(String(data=failure.to_json()))

    def _on_transaction_control(self, message: String) -> None:
        """Abort an exact goal/epoch and invalidate any late planning worker."""
        try:
            control = parse_transaction_control(message.data)
        except PlaceTransactionError as error:
            self._publish_status('invalid_transaction_control', str(error))
            return
        with self._lock:
            matched = self._transaction.abort(control)
            if matched:
                self._request = None
                self._request_identity = None
                self._request_transaction_token = None
                self._planning_identity = None
                self._post_release.reset()
                self._place_execution.reset()
                self._pending_release = None
                self._verification_rgbd.clear()
                self._verification_targets.clear()
                self._verification_identities.clear()
                self._verification_identity_conflicts.clear()
                self._verification_joints.clear()
        if matched:
            self._publish_status(
                'transaction_aborted',
                json.dumps(control.to_payload(), sort_keys=True),
            )

    def _transaction_watchdog_tick(self) -> None:
        """Use a steady timer so paused simulation still reaches a finite terminal."""
        failure = None
        with self._lock:
            identity = self._transaction.identity
            reason = self._transaction.watchdog_reason(
                now_ros_ns=self.get_clock().now().nanoseconds,
                now_wall_s=time.monotonic(),
            )
            if reason is not None and identity is not None:
                failure = PlaceTerminalFailure(identity, reason)
                # watchdog_reason already invalidated the lifecycle token.
                self._request = None
                self._request_identity = None
                self._request_transaction_token = None
                self._planning_identity = None
                self._post_release.reset()
                self._place_execution.reset()
                self._pending_release = None
                self._verification_rgbd.clear()
                self._verification_targets.clear()
                self._verification_identities.clear()
                self._verification_identity_conflicts.clear()
                self._verification_joints.clear()
        self._publish_terminal_failure(failure)

    def _publish_post_release_result(
        self,
        result: PostReleaseVerificationResult | None,
    ) -> None:
        if result is None:
            return
        payload = result.to_payload()
        failure = None
        with self._lock:
            identity = self._transaction.identity
            if identity is None or identity.goal_id != result.goal_id:
                return
            if result.state == 'failed':
                failure = self._terminal_failure_locked(
                    result.failure,
                    expected=identity,
                )
            elif result.state == 'verified':
                self._reset_transaction_state_locked()
        self._post_release_publisher.publish(String(data=json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )))
        state = (
            'post_release_verified'
            if result.state == 'verified'
            else 'post_release_verification_failed'
        )
        self._publish_status(
            state,
            json.dumps(payload, sort_keys=True, allow_nan=False),
        )
        self._publish_terminal_failure(failure)

    def _post_release_tick(self) -> None:
        with self._lock:
            result = self._post_release.tick(
                self.get_clock().now().nanoseconds,
            )
        self._publish_post_release_result(result)

    @staticmethod
    def _execution_fields(raw: str) -> tuple[str, dict[str, str]]:
        fields = [item.strip() for item in str(raw).split(';') if item.strip()]
        if not fields:
            raise PlacementContractError('execution status is empty')
        values: dict[str, str] = {}
        for item in fields[1:]:
            key, separator, value = item.partition('=')
            normalized_key = key.strip()
            if not separator or not normalized_key:
                raise PlacementContractError('execution status field is malformed')
            if normalized_key in values:
                raise PlacementContractError(
                    f'execution status repeats field {normalized_key!r}',
                )
            values[normalized_key] = value.strip()
        return fields[0], values

    @staticmethod
    def _execution_source_stamp_ns(
        values: dict[str, str],
        name: str,
    ) -> int | None:
        raw = values.get(name)
        if raw is None or raw.lower() in {'', 'none'}:
            return None
        seconds = float(raw)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise PlacementContractError(f'{name} must be finite and nonnegative')
        stamp = int(round(seconds * 1e9))
        if not 0 <= stamp < 2**63:
            raise PlacementContractError(f'{name} exceeds the supported range')
        return stamp

    @staticmethod
    def _validate_completed_carry_execution(
        execution: _ExecutionFeedback | None,
    ) -> _ExecutionFeedback:
        if execution is None:
            raise PlacementContractError(
                'executor high-water snapshot is unavailable',
            )
        if (
            execution.trajectory_status != 'succeeded'
            or execution.trajectory_owner != 'trajectory'
            or execution.trajectory_segment != 'carry'
            or execution.trajectory_contract_id != 'none'
            or not execution.executor_epoch
            or execution.trajectory_command_id <= 0
            or execution.trajectory_source_stamp_ns < 0
            or execution.gripper_command_id <= 0
            or execution.gripper_source_stamp_ns < 0
        ):
            raise PlacementContractError(
                'executor snapshot is not the completed carry transaction',
            )
        return execution

    def _on_execution_status(self, message: String) -> None:
        result = None
        terminal_failure = None
        try:
            trajectory, values = self._execution_fields(message.data)
            now_ns = self.get_clock().now().nanoseconds
            epoch = values.get('executor_epoch', '').strip()
            if not epoch or len(epoch) > 256:
                raise PlacementContractError('executor_epoch is missing')
            command_id = (
                None
                if values.get('gripper_command_id') in {None, '', 'none'}
                else int(values['gripper_command_id'])
            )
            trajectory_command_id = (
                None
                if values.get('command_id') in {None, '', 'none'}
                else int(values['command_id'])
            )
            trajectory_source_ns = self._execution_source_stamp_ns(
                values,
                'trajectory_received_at',
            )
            gripper_source_ns = self._execution_source_stamp_ns(
                values,
                'gripper_received_at',
            )
            aperture = (
                None
                if values.get('aperture') in {None, '', 'none'}
                else float(values['aperture'])
            )
            if command_id is not None and command_id < 0:
                raise PlacementContractError('gripper command ID is negative')
            if trajectory_command_id is not None and trajectory_command_id < 0:
                raise PlacementContractError('trajectory command ID is negative')
            if aperture is not None and (
                not math.isfinite(aperture) or aperture < 0.0
            ):
                raise PlacementContractError('measured gripper aperture is invalid')
            if (
                command_id is not None
                and (
                    (command_id == 0 and gripper_source_ns is not None)
                    or (command_id > 0 and gripper_source_ns is None)
                )
            ):
                raise PlacementContractError(
                    'gripper command and source identity must appear together',
                )
            if (
                trajectory_command_id is not None
                and (
                    (trajectory_command_id == 0 and trajectory_source_ns is not None)
                    or (
                        trajectory_command_id > 0
                        and trajectory_source_ns is None
                    )
                )
            ):
                raise PlacementContractError(
                    'trajectory command and source identity must appear together',
                )
            with self._lock:
                segment = values.get('segment', '')
                is_place_trajectory = (
                    values.get('owner') == 'trajectory'
                    and segment in {'place_approach', 'place_retreat'}
                )
                contract_id = values.get('trajectory_contract_id', '').strip()
                if len(contract_id) > 256:
                    raise PlacementContractError(
                        'trajectory_contract_id exceeds the supported length',
                    )
                previous = self._execution_feedback
                if previous is not None and previous.executor_epoch != epoch:
                    if self._post_release.armed:
                        raise PlacementContractError(
                            'executor epoch changed during placement execution',
                        )
                    previous = None
                if (
                    previous is not None
                    and trajectory_command_id is not None
                ):
                    candidate_source = (
                        0
                        if trajectory_command_id == 0
                        else trajectory_source_ns
                    )
                    assert candidate_source is not None
                    if (
                        trajectory_command_id < previous.trajectory_command_id
                        or candidate_source < previous.trajectory_source_stamp_ns
                    ):
                        raise PlacementContractError(
                            'trajectory execution high-water moved backwards',
                        )
                    if (
                        trajectory_command_id == previous.trajectory_command_id
                        and candidate_source
                        != previous.trajectory_source_stamp_ns
                    ):
                        raise PlacementContractError(
                            'trajectory source changed without a new command',
                        )
                if (
                    previous is not None
                    and command_id is not None
                ):
                    candidate_source = (
                        0 if command_id == 0 else gripper_source_ns
                    )
                    assert candidate_source is not None
                    if (
                        command_id < previous.gripper_command_id
                        or candidate_source < previous.gripper_source_stamp_ns
                    ):
                        raise PlacementContractError(
                            'gripper execution high-water moved backwards',
                        )
                    if (
                        command_id == previous.gripper_command_id
                        and candidate_source
                        != previous.gripper_source_stamp_ns
                    ):
                        raise PlacementContractError(
                            'gripper source changed without a new command',
                        )
                if (
                    command_id is not None
                    and command_id > 0
                    and self._post_release.armed
                ):
                    assert gripper_source_ns is not None
                    self._place_execution.observe_gripper_command(
                        command_id,
                        executor_epoch=epoch,
                        source_stamp_ns=gripper_source_ns,
                    )
                if is_place_trajectory and self._post_release.armed:
                    if (
                        trajectory_command_id is None
                        or trajectory_source_ns is None
                        or not contract_id
                    ):
                        raise PlacementContractError(
                            'place trajectory status lacks transaction identity',
                        )
                    self._place_execution.observe_trajectory(
                        status=trajectory,
                        segment=segment,
                        command_id=trajectory_command_id,
                        executor_epoch=epoch,
                        trajectory_contract_id=contract_id,
                        source_stamp_ns=trajectory_source_ns,
                    )
                self._execution_feedback = _ExecutionFeedback(
                    received_ns=(
                        now_ns
                        if command_id is not None and aperture is not None
                        else (0 if previous is None else previous.received_ns)
                    ),
                    executor_epoch=epoch,
                    trajectory_status=(
                        '' if previous is None else previous.trajectory_status
                    ) if trajectory_command_id is None else trajectory,
                    trajectory_owner=(
                        '' if previous is None else previous.trajectory_owner
                    ) if trajectory_command_id is None else values.get('owner', ''),
                    trajectory_segment=(
                        '' if previous is None else previous.trajectory_segment
                    ) if trajectory_command_id is None else values.get('segment', ''),
                    trajectory_command_id=(
                        0 if previous is None else previous.trajectory_command_id
                    ) if trajectory_command_id is None else trajectory_command_id,
                    trajectory_source_stamp_ns=(
                        0
                        if previous is None
                        else previous.trajectory_source_stamp_ns
                    ) if trajectory_source_ns is None else trajectory_source_ns,
                    trajectory_contract_id=(
                        '' if previous is None else previous.trajectory_contract_id
                    ) if not contract_id else contract_id,
                    gripper_command_id=(
                        0 if previous is None else previous.gripper_command_id
                    ) if command_id is None else command_id,
                    gripper_source_stamp_ns=(
                        0 if previous is None else previous.gripper_source_stamp_ns
                    ) if gripper_source_ns is None else gripper_source_ns,
                    aperture_m=(
                        0.0 if previous is None else previous.aperture_m
                    ) if aperture is None else aperture,
                )
                accepted_raw = values.get('gripper', '')
                accepted_aperture = None
                if accepted_raw.startswith('accepted:'):
                    accepted_aperture = float(accepted_raw.split(':', 1)[1])
                minimum = self._post_release.config.min_gripper_aperture_m
                if (
                    self._post_release.state == 'armed'
                    and self._place_execution.ready_for_release
                    and command_id is not None
                    and command_id > 0
                    and gripper_source_ns is not None
                    and aperture is not None
                    and accepted_aperture is not None
                    and math.isfinite(accepted_aperture)
                    and accepted_aperture >= minimum
                    and aperture >= minimum
                    and self._place_execution.is_new_release(
                        command_id,
                        executor_epoch=epoch,
                        source_stamp_ns=gripper_source_ns,
                    )
                    and self._pending_release is None
                ):
                    self._place_execution.observe_release(
                        command_id,
                        executor_epoch=epoch,
                        source_stamp_ns=gripper_source_ns,
                    )
                    self._pending_release = _PendingRelease(
                        acknowledgement_stamp_ns=now_ns,
                        gripper_command_id=command_id,
                        gripper_source_stamp_ns=gripper_source_ns,
                    )
                    self._verification_rgbd.clear()
                    self._verification_targets.clear()
                    self._verification_identities.clear()
                    self._verification_identity_conflicts.clear()
                    self._verification_joints.clear()
                    self._publish_status(
                        'post_release_waiting_retreat',
                        json.dumps({
                            'goal_id': self._post_release.goal_id,
                            'release_gripper_command_id': command_id,
                        }, sort_keys=True),
                    )
                if (
                    self._post_release.state == 'armed'
                    and self._place_execution.complete
                    and self._pending_release is not None
                ):
                    pending = self._pending_release
                    self._pending_release = None
                    self._post_release.begin_release(
                        gripper_command_id=pending.gripper_command_id,
                        gripper_source_stamp_ns=(
                            pending.gripper_source_stamp_ns
                        ),
                        acknowledgement_stamp_ns=(
                            pending.acknowledgement_stamp_ns
                        ),
                        observation_start_stamp_ns=now_ns,
                    )
                    self._verification_rgbd.clear()
                    self._verification_targets.clear()
                    self._verification_identities.clear()
                    self._verification_identity_conflicts.clear()
                    self._verification_joints.clear()
                    self._publish_status(
                        'post_release_observing',
                        json.dumps({
                            'goal_id': self._post_release.goal_id,
                            'release_gripper_command_id': (
                                pending.gripper_command_id
                            ),
                        }, sort_keys=True),
                    )
                elif self._post_release.observing and (
                    command_id is not None
                    and command_id
                    != self._post_release.release_gripper_command_id
                ):
                    result = self._post_release.fail(
                        'release gripper command ID changed during verification',
                    )
                if result is None:
                    result = self._drain_post_release_observations()
        except (PlacementContractError, TypeError, ValueError) as error:
            detail = f'execution feedback invalid: {error}'
            with self._lock:
                result = self._post_release.fail(detail)
                if result is None and self._transaction.active:
                    terminal_failure = self._terminal_failure_locked(detail)
            self._publish_status('invalid_post_release_execution_chain', detail)
        self._publish_post_release_result(result)
        self._publish_terminal_failure(terminal_failure)
        self._maybe_auto_plan()

    def _on_perception_status(self, message: DiagnosticArray) -> None:
        results: list[PostReleaseVerificationResult] = []
        for status in message.status:
            values = {item.key: item.value for item in status.values}
            if values.get('schema') != 'z_manip.perception_status.v1':
                continue
            request_id = values.get('request_id', '').strip()
            producer_epoch = values.get('producer_epoch', '').strip()
            try:
                generation = int(values.get('generation', ''))
            except ValueError:
                generation = 0
            with self._lock:
                expected = self._post_release.identity
                same_request = bool(
                    expected is not None
                    and request_id == expected.request_id
                )
                if self._post_release.observing and same_request and (
                    producer_epoch != expected.producer_epoch
                    or generation != expected.generation
                ):
                    result = self._post_release.fail(
                        'post-release perception producer or generation changed',
                    )
                    if result is not None:
                        results.append(result)
                    continue
                public_valid = (
                    status.level == DiagnosticStatus.OK
                    and status.message == 'tracking'
                    and values.get('valid') == 'true'
                )
                if (
                    self._post_release.observing
                    and same_request
                    and public_valid
                    and values.get('observation_frame_id', '').strip()
                    != expected.frame_id
                ):
                    result = self._post_release.fail(
                        'post-release perception observation frame changed',
                    )
                    if result is not None:
                        results.append(result)
                    continue
                if self._post_release.observing and same_request and not public_valid:
                    result = self._post_release.reject_sample(
                        'post-release target perception is invalid or occluded',
                        now_ns=self.get_clock().now().nanoseconds,
                    )
                    if result is not None:
                        results.append(result)
                    continue
                if not public_valid:
                    continue
                try:
                    identity = ObservedPerceptionIdentity(
                        request_id=request_id,
                        producer_epoch=producer_epoch,
                        generation=generation,
                        stamp_ns=int(values.get('observation_stamp_ns', '')),
                        frame_id=values.get('observation_frame_id', '').strip(),
                    )
                except (PlacementContractError, TypeError, ValueError):
                    if self._post_release.observing and same_request:
                        result = self._post_release.fail(
                            'post-release perception identity is malformed',
                        )
                        if result is not None:
                            results.append(result)
                    continue
                key = (identity.stamp_ns, identity.frame_id)
                planning_identity = self._planning_identity
                if (
                    planning_identity is not None
                    and key == (
                        planning_identity.stamp_ns,
                        planning_identity.frame_id,
                    )
                    and identity != planning_identity
                ):
                    self._cache_put(
                        self._verification_identity_conflicts,
                        key,
                        None,
                    )
                    self._publish_status(
                        'invalid_perception_identity',
                        'planning observation owner changed during planning',
                    )
                    continue
                existing = self._verification_identities.get(key)
                if existing is not None and existing != identity:
                    self._verification_identities.pop(key, None)
                    self._cache_put(
                        self._verification_identity_conflicts,
                        key,
                        None,
                    )
                    if self._post_release.observing:
                        result = self._post_release.fail(
                            'post-release perception identity collision',
                        )
                        if result is not None:
                            results.append(result)
                    self._publish_status(
                        'invalid_perception_identity',
                        'multiple owners claimed one observation source key',
                    )
                    continue
                if key in self._verification_identity_conflicts:
                    continue
                self._cache_put(self._verification_identities, key, identity)
                if (
                    self._request is not None
                    and self._request_identity is None
                    and self._request.stamp_ns == identity.stamp_ns
                    and self._request.image_frame == identity.frame_id
                ):
                    if identity == self._request.observation_identity:
                        self._request_identity = identity
                    else:
                        self._cache_put(
                            self._verification_identity_conflicts,
                            key,
                            None,
                        )
                        self._publish_status(
                            'invalid_perception_identity',
                            'placement request owner does not match perception owner',
                        )
                result = self._try_post_release_observation(key)
                if result is not None:
                    results.append(result)
        for result in results:
            self._publish_post_release_result(result)
        self._maybe_auto_plan()

    @staticmethod
    def _read_target_cloud(message: PointCloud2) -> tuple[np.ndarray, np.ndarray]:
        available = {field.name for field in message.fields}
        fields = ('x', 'y', 'z', 'u', 'v')
        if not set(fields).issubset(available):
            raise PlacementContractError(
                'target point cloud must contain exact x/y/z/u/v correspondence',
            )
        values = np.asarray(point_cloud2.read_points(
            message,
            field_names=fields,
            skip_nans=True,
        ))
        if values.dtype.names:
            dense = np.column_stack([
                np.asarray(values[name], dtype=float).reshape(-1)
                for name in fields
            ])
        else:
            dense = np.asarray(values.tolist(), dtype=float)
        if dense.ndim != 2 or dense.shape[1] != len(fields):
            raise PlacementContractError('target point cloud conversion is malformed')
        return dense[:, :3], dense[:, 3:5]

    def _on_target_cloud(self, message: PointCloud2) -> None:
        result = None
        try:
            points, pixels_uv = self._read_target_cloud(message)
            transform = self._lookup_transform(
                message.header.frame_id,
                message.header.stamp,
            )
            points = points @ transform[:3, :3].T + transform[:3, 3]
            planning_from_tool = self._lookup_transform(
                self._tool_link,
                message.header.stamp,
            )
            probes = (
                self._gripper_probe_points_tool
                @ planning_from_tool[:3, :3].T
                + planning_from_tool[:3, 3]
            )
            stamp_ns = _stamp_ns(message.header.stamp)
            key = (stamp_ns, str(message.header.frame_id))
            target = _TargetCache(
                stamp_ns=stamp_ns,
                source_frame=str(message.header.frame_id),
                points=points,
                pixels_uv=pixels_uv,
                gripper_probe_points=probes,
                planning_from_tool=planning_from_tool,
            )
            with self._lock:
                self._cache_put(self._verification_targets, key, target)
                result = self._try_post_release_observation(key)
        except (
            PlacementContractError,
            TransformException,
            TypeError,
            ValueError,
        ) as error:
            with self._lock:
                result = self._post_release.reject_sample(
                    f'post-release target geometry invalid: {error}',
                    now_ns=self.get_clock().now().nanoseconds,
                )
        self._publish_post_release_result(result)
        self._maybe_auto_plan()

    def _try_post_release_observation(
        self,
        key: tuple[int, str],
    ) -> PostReleaseVerificationResult | None:
        if not self._post_release.observing:
            return None
        release_stamp = self._post_release.release_ack_stamp_ns
        observation_start = self._post_release.observation_start_stamp_ns
        if (
            release_stamp is None
            or observation_start is None
            or key[0] <= observation_start
        ):
            self._verification_rgbd.pop(key, None)
            self._verification_targets.pop(key, None)
            self._verification_identities.pop(key, None)
            return None
        rgbd = self._verification_rgbd.get(key)
        target = self._verification_targets.get(key)
        identity = self._verification_identities.get(key)
        joints = None
        fresh_joints = [
            value
            for value in self._verification_joints.values()
            if value.stamp_ns > observation_start
        ]
        if fresh_joints:
            joints = min(
                fresh_joints,
                key=lambda item: abs(item.stamp_ns - key[0]),
            )
        execution = self._execution_feedback
        if any(value is None for value in (
            rgbd,
            target,
            identity,
            joints,
            execution,
        )):
            return None
        observation = PostReleaseObservation(
            identity=identity,
            rgb_stamp_ns=rgbd.rgb_stamp_ns,
            depth_stamp_ns=rgbd.depth_stamp_ns,
            target_stamp_ns=target.stamp_ns,
            joint_stamp_ns=joints.stamp_ns,
            execution_status_received_ns=execution.received_ns,
            now_ns=self.get_clock().now().nanoseconds,
            geometry_frame_id=self._planning_frame,
            organized_points=rgbd.organized_points,
            target_points=target.points,
            target_pixels_uv=target.pixels_uv,
            gripper_probe_points=target.gripper_probe_points,
            joint_names=joints.names,
            joint_positions=joints.positions,
            planning_from_tool_fk=joints.planning_from_tool_fk,
            planning_from_tool_tf=target.planning_from_tool,
            gripper_command_id=execution.gripper_command_id,
            gripper_source_stamp_ns=execution.gripper_source_stamp_ns,
            gripper_aperture_m=execution.aperture_m,
        )
        self._verification_rgbd.pop(key, None)
        self._verification_targets.pop(key, None)
        self._verification_identities.pop(key, None)
        return self._post_release.observe(observation)

    def _drain_post_release_observations(
        self,
    ) -> PostReleaseVerificationResult | None:
        while self._post_release.observing:
            ready = (
                set(self._verification_rgbd)
                & set(self._verification_targets)
                & set(self._verification_identities)
            )
            if not ready:
                return None
            key = min(ready)
            result = self._try_post_release_observation(key)
            if result is not None:
                return result
            if key in ready and (
                key in self._verification_rgbd
                and key in self._verification_targets
                and key in self._verification_identities
            ):
                return None
        return None

    def _on_rgbd(self, color: Image, depth: Image, info: CameraInfo) -> None:
        result = None
        try:
            source_key = self._exact_rgbd_source_key(color, depth, info)
            color_array = self._bridge.imgmsg_to_cv2(color, desired_encoding='rgb8')
            depth_array = self._bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
            if depth.encoding in ('16UC1', 'mono16'):
                depth_m = np.asarray(depth_array, dtype=float) * 0.001
            elif depth.encoding == '32FC1':
                depth_m = np.asarray(depth_array, dtype=float)
            else:
                raise PlacementContractError(
                    f'unsupported aligned depth encoding {depth.encoding}',
                )
            dimensions_match = (
                color_array.shape[:2] == depth_m.shape
                and info.width == depth.width
                and info.height == depth.height
            )
            if not dimensions_match:
                raise PlacementContractError('aligned RGB/depth/camera-info dimensions differ')
            transform = self._lookup_transform(depth.header.frame_id, depth.header.stamp)
            organized = backproject_depth(
                depth_m,
                np.asarray(info.k, dtype=float).reshape(3, 3),
                transform,
                min_depth_m=float(self.get_parameter('min_depth_m').value),
                max_depth_m=float(self.get_parameter('max_depth_m').value),
            )
            with self._lock:
                rgbd = _RgbdCache(
                    _stamp_ns(color.header.stamp),
                    _stamp_ns(depth.header.stamp),
                    _stamp_ns(info.header.stamp),
                    depth.header.frame_id,
                    organized.astype(np.float32),
                )
                self._cache_exact_planning_rgbd(
                    self._planning_rgbd,
                    self._planning_rgbd_conflicts,
                    source_key,
                    rgbd,
                    limit=int(self.get_parameter(
                        'planning_rgbd_cache_size',
                    ).value),
                )
                if self._post_release.observing:
                    key = source_key
                    verification_rgbd = _RgbdCache(
                        rgbd.rgb_stamp_ns,
                        rgbd.depth_stamp_ns,
                        rgbd.camera_info_stamp_ns,
                        rgbd.source_frame,
                        rgbd.organized_points.astype(np.float32),
                    )
                    self._cache_put(
                        self._verification_rgbd,
                        key,
                        verification_rgbd,
                    )
                    result = self._try_post_release_observation(key)
        except (
            CvBridgeError, PlacementContractError, TransformException, ValueError,
        ) as error:
            with self._lock:
                result = self._post_release.reject_sample(
                    f'post-release RGB-D invalid: {error}',
                    now_ns=self.get_clock().now().nanoseconds,
                )
            self._publish_status('invalid_rgbd', str(error))
        self._publish_post_release_result(result)
        self._maybe_auto_plan()

    def _on_scene(self, message: PointCloud2) -> None:
        try:
            points = point_cloud2.read_points_numpy(
                message, field_names=('x', 'y', 'z'), skip_nans=True,
            )
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            transform = self._lookup_transform(message.header.frame_id, message.header.stamp)
            points = points @ transform[:3, :3].T + transform[:3, 3]
            with self._lock:
                self._scene = _SceneCache(_stamp_ns(message.header.stamp), points)
        except (TransformException, ValueError) as error:
            with self._lock:
                self._scene = None
            self._publish_status('invalid_scene_cloud', str(error))
        self._maybe_auto_plan()

    def _on_joints(self, message: JointState) -> None:
        result = None
        try:
            positions = np.asarray(message.position, dtype=float)
            names = tuple(message.name)
            if (
                positions.shape != (len(names),)
                or not np.all(np.isfinite(positions))
                or len(set(names)) != len(names)
            ):
                raise PlacementContractError('joint state arrays are malformed')
            mapping = dict(zip(names, positions))
            missing = set(self._joint_names) - set(mapping)
            if missing:
                raise PlacementContractError(
                    f'joint state is missing configured joints: {sorted(missing)}',
                )
            ordered_positions = np.asarray([
                mapping[name] for name in self._joint_names
            ], dtype=float)
            kinematic_base_from_tool = self._kinematic_chain.forward(
                ordered_positions,
            )
            planning_from_kinematic_base = self._lookup_transform(
                self._kinematic_base_link,
                message.header.stamp,
            )
            planning_from_tool_fk = (
                planning_from_kinematic_base @ kinematic_base_from_tool
            )
            with self._lock:
                self._joints = _JointCache(
                    _stamp_ns(message.header.stamp),
                    self._joint_names,
                    ordered_positions,
                    planning_from_kinematic_base,
                    planning_from_tool_fk,
                )
                if self._post_release.observing:
                    self._cache_put(
                        self._verification_joints,
                        self._joints.stamp_ns,
                        self._joints,
                        limit_parameter='post_release_joint_cache_size',
                    )
                result = self._drain_post_release_observations()
        except (PlacementContractError, TransformException, ValueError) as error:
            with self._lock:
                self._joints = None
                result = self._post_release.fail(
                    f'post-release arm state invalid: {error}',
                )
            self._publish_status('invalid_joint_state', str(error))
        self._publish_post_release_result(result)
        self._maybe_auto_plan()

    def _on_region(self, message: String) -> None:
        busy = False
        terminal_failure = None
        try:
            request = parse_region_request(message.data)
            with self._lock:
                busy = bool(
                    self._planning_identity is not None
                    or self._post_release.armed
                    or self._transaction.active
                )
                if busy:
                    raise PlacementContractError(
                        'an active placement transaction cannot be superseded',
                    )
                key = (request.stamp_ns, request.image_frame)
                if key in self._planning_rgbd_conflicts:
                    raise PlacementContractError(
                        'placement observation has conflicting RGB-D payloads',
                    )
                if key in self._verification_identity_conflicts:
                    raise PlacementContractError(
                        'placement observation has conflicting perception owners',
                    )
                cached_identity = self._verification_identities.get(key)
                if (
                    cached_identity is not None
                    and cached_identity != request.observation_identity
                ):
                    raise PlacementContractError(
                        'placement request owner does not match perception owner',
                    )
                transaction_token = self._transaction.begin(
                    PlaceTransactionIdentity(
                        request.goal_id,
                        request.executor_epoch,
                    ),
                    now_ros_ns=self.get_clock().now().nanoseconds,
                    now_wall_s=time.monotonic(),
                )
                self._request = request
                self._request_identity = cached_identity
                self._request_transaction_token = transaction_token
            self._publish_status('request_received', request.goal_id)
            if bool(self.get_parameter('auto_plan_on_region').value):
                self._start_plan()
        except PlacementContractError as error:
            with self._lock:
                if not busy:
                    self._request = None
                    self._request_identity = None
                    self._request_transaction_token = None
                    try:
                        failed_identity = parse_region_transaction_identity(
                            message.data,
                        )
                    except PlaceTransactionError:
                        failed_identity = None
                    if failed_identity is not None:
                        terminal_failure = PlaceTerminalFailure(
                            failed_identity,
                            f'invalid placement request: {error}'[:1024],
                        )
            self._publish_status('invalid_region_request', str(error))
            self._publish_terminal_failure(terminal_failure)

    def _on_plan_service(self, _request: Trigger.Request, response: Trigger.Response):
        accepted, reason = self._start_plan()
        response.success = accepted
        response.message = reason
        return response

    def _maybe_auto_plan(self) -> None:
        if bool(self.get_parameter('auto_plan_on_region').value):
            self._start_plan()

    @staticmethod
    def _assemble_plan_inputs_locked(self: object) -> _PlacementPlanInputs:
        """Assemble only the exact current request key, without latest fallback."""
        request = self._request
        token = self._request_transaction_token
        missing = [
            name
            for name, value in (
                ('request', request),
                ('scene', self._scene),
                ('joints', self._joints),
                ('execution', self._execution_feedback),
                ('transaction', token),
            )
            if value is None
        ]
        if missing:
            raise PlacementContractError(
                f'placement inputs are incomplete: {", ".join(missing)}',
            )
        assert request is not None
        assert token is not None
        if not self._transaction.matches(token) or self._transaction.state != 'pending':
            raise PlacementContractError(
                'placement request does not own the pending transaction',
            )
        key = (request.stamp_ns, request.image_frame)
        if (
            key in self._planning_rgbd_conflicts
            or key in self._verification_identity_conflicts
        ):
            raise PlacementContractError(
                'placement source key has conflicting producers',
            )
        identity = self._request_identity
        if identity is None:
            cached_identity = self._verification_identities.get(key)
            if cached_identity == request.observation_identity:
                identity = cached_identity
                self._request_identity = identity
        rgbd = self._planning_rgbd.get(key)
        target = self._verification_targets.get(key)
        missing = [
            name
            for name, value in (
                ('identity', identity),
                ('RGB-D', rgbd),
                ('target', target),
            )
            if value is None
        ]
        if missing:
            raise PlacementContractError(
                'exact placement perception snapshot is incomplete: '
                + ', '.join(missing),
            )
        assert identity is not None
        assert rgbd is not None
        assert target is not None
        ObservedPlacementNode._validate_exact_planning_sources(
            request,
            identity,
            rgbd,
            target,
        )
        execution = ObservedPlacementNode._validate_completed_carry_execution(
            self._execution_feedback,
        )
        if execution.executor_epoch != request.executor_epoch:
            raise PlacementContractError(
                'completed carry executor epoch does not match placement request',
            )
        return _PlacementPlanInputs(
            request=request,
            identity=identity,
            rgbd=rgbd,
            scene=self._scene,
            joints=self._joints,
            target=target,
            execution=execution,
            token=token,
        )

    def _start_plan(self) -> tuple[bool, str]:
        with self._lock:
            for generation, worker in tuple(self._workers.items()):
                if not worker.is_alive():
                    self._workers.pop(generation, None)
                    if self._worker is worker:
                        self._worker = None
            if self._workers:
                return False, 'previous placement worker cancellation is pending'
            try:
                inputs = ObservedPlacementNode._assemble_plan_inputs_locked(self)
            except PlacementContractError as error:
                return False, str(error)
            self._request = None
            self._request_identity = None
            self._request_transaction_token = None
            self._planning_rgbd.pop(
                (inputs.request.stamp_ns, inputs.request.image_frame),
                None,
            )
            self._planning_identity = inputs.identity
            self._transaction.start_planning(inputs.token)
            worker = threading.Thread(
                target=self._plan_worker,
                args=(inputs,),
                daemon=True,
            )
            self._worker = worker
            self._workers[inputs.token.generation] = worker
            worker.start()
        return True, 'placement planning accepted'

    def _plan_worker(
        self,
        inputs: _PlacementPlanInputs,
    ) -> None:
        request = inputs.request
        identity = inputs.identity
        rgbd = inputs.rgbd
        scene = inputs.scene
        joints = inputs.joints
        target = inputs.target
        transaction_token = inputs.token
        coordinator = None
        evaluator = None
        attached_collision = None
        try:
            with self._lock:
                if not self._transaction.matches(transaction_token):
                    return
                started_wall_s = self._transaction.started_wall_s
                if started_wall_s is None:
                    raise PlacementContractError(
                        'placement transaction wall deadline is unavailable',
                    )
                planning_deadline_s = (
                    started_wall_s + self._transaction.wall_timeout_s
                )
            control = PlanningControl(
                deadline_s=planning_deadline_s,
                cancel_check=lambda: self._planning_cancelled(transaction_token),
            )
            coordinator, evaluator, attached_collision = (
                self._new_planning_backend()
            )
            control.checkpoint('placement worker startup')
            self._validate_exact_planning_sources(request, identity, rgbd, target)
            snapshot = PlacementPerceptionSnapshot(
                rgb_stamp_ns=rgbd.rgb_stamp_ns,
                depth_stamp_ns=rgbd.depth_stamp_ns,
                camera_info_stamp_ns=rgbd.camera_info_stamp_ns,
                scene_stamp_ns=scene.stamp_ns,
                joint_stamp_ns=joints.stamp_ns,
                image_frame=rgbd.source_frame,
                frame_id=self._planning_frame,
                organized_points=rgbd.organized_points,
                scene_points=scene.points,
                gravity=self._gravity,
                joint_names=joints.names,
                joint_positions=joints.positions,
            )
            expected_current_pose = (
                target.planning_from_tool @ request.tool_from_object
            )
            current_upright = (
                expected_current_pose[:3, :3]
                @ np.asarray(request.verification.upright_axis_object, dtype=float)
            )
            current_model = capture_planned_object_geometry(
                request.object_reference_points_object,
                frame_id=snapshot.frame_id,
                expected_object_pose=expected_current_pose,
                support_normal=current_upright,
                verification=request.verification,
                min_points=self._post_release.config.min_object_reference_points,
                max_points=512,
            )
            self._post_release.validate_observed_object_pose(
                current_model,
                target.points,
            )
            attached_collision.bind_snapshot(
                scene_points=snapshot.scene_points,
                scene_stamp_s=snapshot.scene_stamp_ns / 1e9,
                planning_from_kinematic_base=(
                    joints.planning_from_kinematic_base
                ),
                attachment_joints=joints.positions,
                object_reference_points_object=(
                    request.object_reference_points_object
                ),
                object_extent_m=request.object_extent_m,
                tool_from_object=request.tool_from_object,
            )
            evaluator.set_goal_id(request.goal_id)
            output = coordinator.plan(
                request,
                snapshot,
                now_ns=self.get_clock().now().nanoseconds,
                evaluate=lambda candidate, current: evaluator.evaluate(
                    candidate,
                    current,
                    control=control,
                ),
                control=control,
            )
            plane = output.result.plane
            region_geometry = capture_observed_region_geometry(
                snapshot.organized_points,
                request.region,
                frame_id=snapshot.frame_id,
                plane_origin=plane.origin,
                plane_normal=plane.normal,
                tangent_u=plane.tangent_u,
                tangent_v=plane.tangent_v,
                plane_distance_tolerance_m=(
                    self._post_release.config.plane_distance_tolerance_m
                ),
                min_points=int(self.get_parameter(
                    'post_release_region_capture_min_points',
                ).value),
                max_points=self._post_release.config.max_geometry_samples,
            )
            if (
                target.stamp_ns != request.stamp_ns
                or target.source_frame != request.image_frame
            ):
                raise PlacementContractError(
                    'planned object model identity does not match placement request',
                )
            planned_object = capture_planned_object_geometry(
                request.object_reference_points_object,
                frame_id=snapshot.frame_id,
                expected_object_pose=output.result.candidate.object_pose,
                support_normal=plane.normal,
                verification=request.verification,
                min_points=self._post_release.config.min_object_reference_points,
                max_points=512,
            )
            with self._lock:
                key = (identity.stamp_ns, identity.frame_id)
                if (
                    self._planning_identity != identity
                    or key in self._verification_identity_conflicts
                ):
                    raise PlacementContractError(
                        'placement perception ownership changed during planning',
                    )
                latest_execution = self._validate_completed_carry_execution(
                    self._execution_feedback,
                )
                execution = inputs.execution
                immutable_execution_fields = (
                    'executor_epoch',
                    'trajectory_status',
                    'trajectory_owner',
                    'trajectory_segment',
                    'trajectory_command_id',
                    'trajectory_source_stamp_ns',
                    'trajectory_contract_id',
                    'gripper_command_id',
                    'gripper_source_stamp_ns',
                )
                if any(
                    getattr(latest_execution, field) != getattr(execution, field)
                    for field in immutable_execution_fields
                ):
                    raise PlacementContractError(
                        'completed carry executor identity changed during planning',
                    )
                self._transaction.arm(
                    transaction_token,
                    now_ros_ns=self.get_clock().now().nanoseconds,
                    now_wall_s=time.monotonic(),
                )
                self._post_release.arm(
                    goal_id=request.goal_id,
                    identity=identity,
                    region=region_geometry,
                    planned_object=planned_object,
                    expected_joint_names=self._joint_names,
                )
                self._place_execution.arm(
                    goal_id=request.goal_id,
                    executor_epoch=execution.executor_epoch,
                    trajectory_contract_id=request.goal_id,
                    trajectory_command_highwater=(
                        execution.trajectory_command_id
                    ),
                    gripper_command_highwater=execution.gripper_command_id,
                    trajectory_source_highwater_ns=(
                        execution.trajectory_source_stamp_ns
                    ),
                    gripper_source_highwater_ns=(
                        execution.gripper_source_stamp_ns
                    ),
                )
                self._pending_release = None
                self._planning_identity = None
                self._verification_rgbd.clear()
                self._verification_targets.clear()
                self._verification_identities.clear()
                self._verification_joints.clear()
                # Publish while the transaction lock is held: an abort either
                # invalidates the worker before this point or happens after all
                # plan outputs, never between the token check and publication.
                self._publish_output(output, identity, execution)
        except Exception as error:  # A service/TF failure must remain fail-closed.
            reason = f'{type(error).__name__}: {error}'
            failure = None
            with self._lock:
                if self._transaction.matches(transaction_token):
                    failure = self._terminal_failure_locked(
                        reason,
                        expected=transaction_token.identity,
                    )
            if failure is not None:
                self._publish_status('planning_failed', reason)
                self._publish_terminal_failure(failure)
        finally:
            if attached_collision is not None:
                attached_collision.clear_snapshot()
            if evaluator is not None:
                self.destroy_client(evaluator.motion_client)
                self.destroy_client(evaluator.cartesian_client)
            current = threading.current_thread()
            with self._lock:
                self._release_planning_worker_locked(
                    transaction_token,
                    current,
                )
            self._maybe_auto_plan()

    @staticmethod
    def _place_contract_payload(
        output: object,
        identity: ObservedPerceptionIdentity,
        execution: _ExecutionFeedback,
        *,
        trajectory_topic: str,
        trajectory_frame_id: str,
        trajectory_digest_sha256: str,
    ) -> dict[str, object]:
        return {
            'schema': 'z_manip.place_contract.v2',
            'schema_version': 2,
            'goal_id': output.goal_id,
            'frame_id': trajectory_frame_id,
            'joint_names': output.trajectory.joint_names,
            'phase_start_indices': dict(output.trajectory.phase_start_indices),
            'point_count': len(output.trajectory.points),
            'trajectory_topic': trajectory_topic,
            'trajectory_digest_sha256': trajectory_digest_sha256,
            'request_id': identity.request_id,
            'producer_epoch': identity.producer_epoch,
            'generation': identity.generation,
            'observation_stamp_ns': identity.stamp_ns,
            'observation_frame_id': identity.frame_id,
            'executor_epoch': execution.executor_epoch,
            'trajectory_contract_id': output.goal_id,
            'trajectory_command_highwater': execution.trajectory_command_id,
            'trajectory_source_highwater_ns': (
                execution.trajectory_source_stamp_ns
            ),
            'gripper_command_highwater': execution.gripper_command_id,
            'gripper_source_highwater_ns': execution.gripper_source_stamp_ns,
        }

    def _publish_output(
        self,
        output: object,
        identity: ObservedPerceptionIdentity,
        execution: _ExecutionFeedback,
    ) -> None:
        now = self.get_clock().now().to_msg()
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for index, audit in enumerate(output.candidates):
            marker = Marker()
            marker.header.frame_id = self._planning_frame
            marker.header.stamp = now
            marker.ns = 'feasible' if audit.feasible else 'rejected'
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.points = [
                Point(
                    x=float(audit.candidate.preplace_pose[0, 3]),
                    y=float(audit.candidate.preplace_pose[1, 3]),
                    z=float(audit.candidate.preplace_pose[2, 3]),
                ),
                Point(
                    x=float(audit.candidate.place_pose[0, 3]),
                    y=float(audit.candidate.place_pose[1, 3]),
                    z=float(audit.candidate.place_pose[2, 3]),
                ),
            ]
            marker.scale.x = 0.008
            marker.scale.y = 0.016
            marker.scale.z = 0.022
            marker.color.a = 0.75
            marker.color.r = 0.9 if not audit.feasible else 0.1
            marker.color.g = 0.8 if audit.feasible else 0.1
            marker.color.b = 0.15
            markers.markers.append(marker)
        self._candidate_publisher.publish(markers)

        selected = PoseArray()
        selected.header.frame_id = self._planning_frame
        selected.header.stamp = now
        candidate = output.result.candidate
        selected.poses = [
            _pose(candidate.preplace_pose),
            _pose(candidate.place_pose),
            _pose(candidate.retreat_pose),
        ]
        self._selected_publisher.publish(selected)

        trajectory = JointTrajectory()
        trajectory.header.frame_id = self._planning_frame
        trajectory.header.stamp = now
        trajectory.joint_names = list(output.trajectory.joint_names)
        for contract_point in output.trajectory.points:
            point = JointTrajectoryPoint()
            point.positions = list(contract_point.positions)
            total_nanoseconds = int(round(contract_point.time_from_start_s * 1e9))
            point.time_from_start = DurationMessage(
                sec=total_nanoseconds // 1_000_000_000,
                nanosec=total_nanoseconds % 1_000_000_000,
            )
            trajectory.points.append(point)
        trajectory_digest = canonical_joint_trajectory_sha256(
            frame_id=trajectory.header.frame_id,
            header_stamp=trajectory.header.stamp,
            joint_names=trajectory.joint_names,
            points=trajectory.points,
        )
        contract = String()
        contract.data = json.dumps(self._place_contract_payload(
            output,
            identity,
            execution,
            trajectory_topic=self._trajectory_publisher.topic_name,
            trajectory_frame_id=trajectory.header.frame_id,
            trajectory_digest_sha256=trajectory_digest,
        ), sort_keys=True, allow_nan=False)
        self._trajectory_publisher.publish(trajectory)
        self._contract_publisher.publish(contract)
        self._publish_status('planned', json.dumps({
            'goal_id': output.goal_id,
            'score': output.result.score,
            'evaluated_candidates': len(output.candidates),
            'feasible_candidates': sum(audit.feasible for audit in output.candidates),
            'plane_inlier_ratio': output.result.plane.inlier_ratio,
            'plane_rms_m': output.result.plane.rms_error_m,
            'trajectory_points': len(output.trajectory.points),
        }, sort_keys=True, allow_nan=False))

    def _publish_status(self, state: str, detail: str) -> None:
        message = String()
        message.data = json.dumps(
            {'state': state, 'detail': detail},
            sort_keys=True,
            allow_nan=False,
        )
        self._status_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    """Run the placement node with service-capable concurrent callbacks."""
    rclpy.init(args=args)
    node = ObservedPlacementNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
