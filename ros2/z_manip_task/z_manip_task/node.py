"""ROS 2 online mobile manipulation runtime."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any
import uuid

from builtin_interfaces.msg import Duration as DurationMsg
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
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
from sensor_msgs.msg import CameraInfo, JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, ColorRGBA, Empty, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vision_msgs.msg import Detection3D
from visualization_msgs.msg import Marker, MarkerArray

from z_manip.configuration import load_stack_config
from z_manip.control.approach import (
    ApproachInput,
    ApproachPhase,
    TwoStageApproachController,
)
from z_manip.kinematics import fixed_transform_from_urdf
from z_manip.orchestration.mobile_manipulation import (
    FailureKind,
    MobileManipulationStateMachine,
    Stage,
    StageResult,
)
from z_manip.perception.rgbd import filter_object_cloud
from z_manip.planning.time_parameterization import TimedJointTrajectory
from z_manip.planning_control import (
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
)
from z_manip.verification.grasp import (
    GraspVerificationConfig,
    GraspVerifier,
    VerificationResult,
    VerificationSample,
    VerificationState,
)

from .core import (
    base_twist_speed_magnitudes,
    BoundedYawSearch,
    ContinuousMotionQuietWindow,
    ExecutionOcclusionConfig,
    ExecutionOcclusionDecision,
    ExecutionOcclusionGate,
    ExecutionState,
    grasp_close_aperture,
    horizontal_edge_direction,
    ObservationSerialGate,
    parse_execution_status,
    parse_place_contract,
    PostureSafetyGate,
    PostureState,
    RuntimePhase,
    RuntimeSafetyCore,
    SafetyAction,
    split_placement_trajectory,
    TaskGenerationGuard,
    terminal_result,
    trajectory_segment_frame_id,
    validate_grasp_aperture_contract,
    validate_place_trajectory_content,
    validate_platform_odometry_frames,
    validate_position_hold_frame_contract,
    vertical_edge_direction,
    VisualSearchConfig,
    wrap_angle,
)
from .grasp_verification import establish_baseline_before_lift
from .object_geometry import (
    CarriedObjectGeometry,
    CarriedObjectObservationIdentity,
    estimate_carried_object_geometry,
    ObjectGeometryConfig,
    ObjectGeometryError,
    parse_placement_verification,
    PlacementVerificationSemantics,
)
from .place_transaction import (
    parse_terminal_place_status,
    PlaceTransactionProtocolError,
    transaction_abort_json,
)
from .planning import (
    GraspCompletionProgram,
    OnlinePlanner,
    PerceptionObservation,
    PregraspTransitProgram,
    select_semantic_target_points,
)
from .post_release_verification import (
    parse_post_release_verification,
    PlacementObservationIdentity,
    PostReleaseVerificationError,
    PostReleaseVerificationEvidence,
    PostReleaseVerificationPolicy,
    validate_place_trajectory_perception_identity,
)


_DEFERRED_COARSE_NAV_TRACKER_FAILURES = frozenset({
    'tracker_acquisition_timeout',
    'tracker_reported_loss',
    'empty_detections',
    'track_id_changed',
    'selected_target_missing',
    'selected_cloud_too_small',
    'tracker_data_stale',
})


@dataclass(frozen=True)
class _VisualSearchSettleReference:
    """Measured search target retained through the stationary settle period."""

    position_anchor_xy: tuple[float, float]
    target_yaw_rad: float
    started_at_s: float
    stop_started_at_s: float
    minimum_odom_sequence: int
    minimum_odom_stamp_ns: int
    correction_deadline_s: float
    absolute_deadline_s: float
    stationary_deadline_s: float
    reacquire_count: int


@dataclass(frozen=True)
class _JointFeedback:
    """One arm sample with independent receipt and ROS source clocks."""

    received_at_s: float
    source_stamp_ns: int
    sequence: int
    positions: np.ndarray


@dataclass(frozen=True)
class _PlanningBaseAnchor:
    """Map-frame platform pose tied to the perception used by a planner job."""

    pose_map: tuple[float, float, float]
    odom_sequence: int
    odom_stamp_ns: int


@dataclass(frozen=True)
class _PlanningObservationIdentity:
    """Exact task-owned observation consumed by an asynchronous planner job."""

    request_id: str
    producer_epoch: str
    generation: int
    stamp_ns: int
    frame_id: str
    target_position_camera: tuple[float, float, float]


@dataclass(frozen=True)
class _PlanningObservationWait:
    """Fixed ROS-time window for one completed result to regain exact geometry."""

    started_at_s: float
    deadline_s: float


@dataclass(frozen=True, eq=False)
class _PregraspHandoff:
    """Executor and perception identity frozen at pregrasp completion."""

    observation_serial: int
    observation_identity: _PlanningObservationIdentity
    endpoint_joints: np.ndarray
    executor_epoch: str
    command_id: int
    trajectory_received_at: float
    completed_at_s: float
    completion_source_stamp_ns: int
    deadline_s: float
    minimum_joint_sequence: int
    minimum_joint_stamp_ns: int


@dataclass(frozen=True, eq=False)
class _ApproachPlanningAnchor:
    """Fresh observation and measured joint pair used by stage-two planning."""

    observation_identity: _PlanningObservationIdentity
    observation_serial: int
    joint_sequence: int
    joint_stamp_ns: int
    joint_positions: np.ndarray
    target_geometry: _TargetGeometrySignature


@dataclass(frozen=True)
class _ApproachExecutionJointFence:
    """Joint feedback watermark frozen when stage-two planning completes."""

    deadline_s: float
    minimum_joint_sequence: int
    minimum_joint_stamp_ns: int


@dataclass(frozen=True)
class _PregraspDispatchFence:
    """Post-validation feedback watermarks required before pregrasp transit."""

    deadline_s: float
    minimum_joint_sequence: int
    minimum_joint_stamp_ns: int
    minimum_odom_sequence: int
    minimum_odom_stamp_ns: int


@dataclass(frozen=True, eq=False)
class _TargetGeometrySignature:
    """Robust target shape and observable orientation in the PiPER frame."""

    center_piper: np.ndarray
    principal_extents_m: np.ndarray
    principal_variances: np.ndarray
    principal_axes_piper: np.ndarray
    retained_point_count: int


class _ApproachJointFeedbackPending(ValueError):
    """The completed stage-two plan is waiting for a newer arm sample."""


def _readonly_finite_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f'{label} must be a finite array with shape {shape}')
    result = array.copy()
    result.setflags(write=False)
    return result


def _target_geometry_signature(
    points: object,
    *,
    min_points: int,
    trim_mad_scale: float,
    extent_percentile: float,
) -> _TargetGeometrySignature:
    """Build a robust, ground-truth-free signature from a semantic target cloud."""
    cloud = np.asarray(points, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError('target geometry cloud must have shape (N, 3)')
    if (
        isinstance(min_points, bool)
        or int(min_points) < 6
        or not math.isfinite(float(trim_mad_scale))
        or trim_mad_scale <= 0.0
        or not math.isfinite(float(extent_percentile))
        or not 0.0 < extent_percentile < 25.0
    ):
        raise ValueError('target geometry signature configuration is invalid')
    finite = cloud[np.all(np.isfinite(cloud), axis=1)]
    finite = np.unique(finite, axis=0)
    if len(finite) < int(min_points):
        raise ValueError(
            f'target geometry has {len(finite)} finite unique points; '
            f'need {int(min_points)}',
        )
    initial_center = np.median(finite, axis=0)
    radii = np.linalg.norm(finite - initial_center, axis=1)
    median_radius = float(np.median(radii))
    radial_mad = float(np.median(np.abs(radii - median_radius)))
    cutoff = median_radius + float(trim_mad_scale) * max(
        1.4826 * radial_mad,
        1e-6,
    )
    retained = finite[radii <= cutoff]
    if len(retained) < int(min_points):
        raise ValueError(
            f'target geometry retained {len(retained)} inliers; '
            f'need {int(min_points)}',
        )
    center = np.median(retained, axis=0)
    centered = retained - center
    covariance = centered.T @ centered / float(len(retained))
    variances, axes = np.linalg.eigh(covariance)
    order = np.argsort(variances)[::-1]
    variances = np.maximum(variances[order], 0.0)
    axes = axes[:, order]
    projected = centered @ axes
    lower = np.percentile(projected, extent_percentile, axis=0)
    upper = np.percentile(projected, 100.0 - extent_percentile, axis=0)
    extents = upper - lower
    if (
        not np.all(np.isfinite(extents))
        or float(np.max(extents)) <= 1e-6
        or float(np.max(variances)) <= 1e-12
    ):
        raise ValueError('target geometry is degenerate')
    return _TargetGeometrySignature(
        center_piper=_readonly_finite_array(center, (3,), 'target center'),
        principal_extents_m=_readonly_finite_array(
            extents,
            (3,),
            'target principal extents',
        ),
        principal_variances=_readonly_finite_array(
            variances,
            (3,),
            'target principal variances',
        ),
        principal_axes_piper=_readonly_finite_array(
            axes,
            (3, 3),
            'target principal axes',
        ),
        retained_point_count=int(len(retained)),
    )


def _validate_target_geometry_change(
    reference: _TargetGeometrySignature,
    current: _TargetGeometrySignature,
    *,
    max_center_drift_m: float,
    max_extent_change_m: float,
    max_extent_ratio: float,
    axis_separation_ratio: float,
    max_orientation_change_rad: float,
) -> None:
    """Reject grasp-relevant target motion while ignoring unobservable symmetry."""
    limits = (
        max_center_drift_m,
        max_extent_change_m,
        max_extent_ratio,
        axis_separation_ratio,
        max_orientation_change_rad,
    )
    if (
        not all(math.isfinite(float(value)) for value in limits)
        or max_center_drift_m <= 0.0
        or max_extent_change_m <= 0.0
        or max_extent_ratio <= 1.0
        or axis_separation_ratio <= 1.0
        or not 0.0 < max_orientation_change_rad < math.pi
    ):
        raise ValueError('target geometry comparison configuration is invalid')
    center_drift = float(np.linalg.norm(
        current.center_piper - reference.center_piper,
    ))
    if center_drift > max_center_drift_m:
        raise ValueError(
            f'target geometry center drifted {center_drift:.4f}m '
            f'(limit {max_center_drift_m:.4f}m)',
        )
    reference_extents = np.sort(reference.principal_extents_m)[::-1]
    current_extents = np.sort(current.principal_extents_m)[::-1]
    extent_delta = np.abs(current_extents - reference_extents)
    extent_ratio = np.maximum(current_extents, reference_extents) / np.maximum(
        np.minimum(current_extents, reference_extents),
        max_extent_change_m,
    )
    changed = (extent_delta > max_extent_change_m) | (
        extent_ratio > max_extent_ratio
    )
    if np.any(changed):
        axis = int(np.argmax(changed))
        raise ValueError(
            'target geometry extent changed on principal axis '
            f'{axis}: delta={extent_delta[axis]:.4f}m, '
            f'ratio={extent_ratio[axis]:.3f}',
        )

    def observable_axes(signature: _TargetGeometrySignature) -> set[int]:
        variance = signature.principal_variances
        observable: set[int] = set()
        if variance[0] / max(float(variance[1]), 1e-12) >= axis_separation_ratio:
            observable.add(0)
        if variance[1] / max(float(variance[2]), 1e-12) >= axis_separation_ratio:
            observable.add(2)
        if 0 in observable and 2 in observable:
            observable.add(1)
        return observable

    for axis in sorted(observable_axes(reference) & observable_axes(current)):
        alignment = abs(float(np.dot(
            reference.principal_axes_piper[:, axis],
            current.principal_axes_piper[:, axis],
        )))
        angle = math.acos(float(np.clip(alignment, 0.0, 1.0)))
        if angle > max_orientation_change_rad:
            raise ValueError(
                f'target observable principal axis {axis} rotated '
                f'{angle:.4f}rad (limit {max_orientation_change_rad:.4f}rad)',
            )


def _fresh_approach_joint_positions(
    feedback: _JointFeedback,
    anchor: _ApproachPlanningAnchor,
    *,
    now_s: float,
    maximum_age_s: float,
    dof: int,
    minimum_sequence: int | None = None,
    minimum_source_stamp_ns: int | None = None,
) -> np.ndarray:
    """Validate arm state newer than planning and optional result watermarks."""
    positions = np.asarray(feedback.positions, dtype=float)
    receipt_age = float(now_s) - float(feedback.received_at_s)
    source_age = float(now_s) - float(feedback.source_stamp_ns) * 1e-9
    if (
        not math.isfinite(float(now_s))
        or not math.isfinite(float(maximum_age_s))
        or maximum_age_s <= 0.0
        or positions.shape != (int(dof),)
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError('approach joint feedback contract is invalid')
    if (minimum_sequence is None) != (minimum_source_stamp_ns is None):
        raise ValueError('approach result joint watermark is incomplete')
    if minimum_sequence is not None:
        if (
            isinstance(minimum_sequence, bool)
            or isinstance(minimum_source_stamp_ns, bool)
            or int(minimum_sequence) < anchor.joint_sequence
            or int(minimum_source_stamp_ns) < anchor.joint_stamp_ns
        ):
            raise ValueError('approach result joint watermark is invalid')
        sequence_floor = int(minimum_sequence)
        source_floor = int(minimum_source_stamp_ns)
    else:
        sequence_floor = anchor.joint_sequence
        source_floor = anchor.joint_stamp_ns
    if (
        feedback.sequence <= sequence_floor
        or feedback.source_stamp_ns <= source_floor
    ):
        raise _ApproachJointFeedbackPending(
            'approach joint feedback did not advance after the active watermark',
        )
    if not (
        0.0 <= receipt_age <= maximum_age_s
        and 0.0 <= source_age <= maximum_age_s
    ):
        raise _ApproachJointFeedbackPending('approach joint feedback is stale')
    return positions.copy()


class _PlanningObservationChanged(RuntimeError):
    """The live target no longer supports a completed asynchronous result."""


class _PlanningObservationPending(RuntimeError):
    """The immutable identity is intact but its exact live bundle is incomplete."""


def _se2_pose(value: object, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError(f'{label} must be a finite [x, y, yaw] vector')
    result = pose.copy()
    result[2] = wrap_angle(float(result[2]))
    return result


def _compose_se2(parent_pose: object, relative_pose: object) -> np.ndarray:
    """Compose a body-frame relative platform pose into the map frame."""
    parent = _se2_pose(parent_pose, 'parent SE(2) pose')
    relative = _se2_pose(relative_pose, 'relative SE(2) pose')
    cosine = math.cos(float(parent[2]))
    sine = math.sin(float(parent[2]))
    return np.array((
        parent[0] + cosine * relative[0] - sine * relative[1],
        parent[1] + sine * relative[0] + cosine * relative[1],
        wrap_angle(float(parent[2] + relative[2])),
    ))


def _relative_se2(parent_pose: object, map_pose: object) -> np.ndarray:
    """Express one map-frame platform pose relative to another platform pose."""
    parent = _se2_pose(parent_pose, 'parent SE(2) pose')
    goal = _se2_pose(map_pose, 'map SE(2) pose')
    delta = goal[:2] - parent[:2]
    cosine = math.cos(float(parent[2]))
    sine = math.sin(float(parent[2]))
    return np.array((
        cosine * delta[0] + sine * delta[1],
        -sine * delta[0] + cosine * delta[1],
        wrap_angle(float(goal[2] - parent[2])),
    ))


def _work_pose_diagnostics_payload(value: object | None) -> dict[str, object] | None:
    """Convert bounded optimizer evidence into a stable JSON status payload."""
    if value is None:
        return None
    counts = {
        getattr(code, 'value', str(code)): int(count)
        for code, count in getattr(value, 'rejection_counts', ())
    }
    return {
        'sampled_hypotheses': int(value.sampled_hypotheses),
        'geometric_candidates': int(value.geometric_candidates),
        'ranked_candidates': int(value.ranked_candidates),
        'exact_evaluations': int(value.exact_evaluations),
        'feasible_candidates': int(value.feasible_candidates),
        'rejection_counts': counts,
        'sample_budget_exhausted': bool(value.sample_budget_exhausted),
        'exact_budget_exhausted': bool(value.exact_budget_exhausted),
    }


def _stamp_s(header: Any) -> float:
    return float(_stamp_ns(header)) * 1e-9


def _stamp_ns(header: Any) -> int:
    sec = int(header.stamp.sec)
    nanosec = int(header.stamp.nanosec)
    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError('observation stamp is invalid')
    return sec * 1_000_000_000 + nanosec


def _matrix_from_transform(transform: Any) -> np.ndarray:
    q = transform.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1e-9:
        raise ValueError('TF quaternion is degenerate')
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    result = np.eye(4)
    result[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    result[:3, 3] = (transform.translation.x, transform.translation.y, transform.translation.z)
    return result


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _normalized_quaternion(q: Any) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in (q.x, q.y, q.z, q.w))
    if not all(math.isfinite(value) for value in values):
        raise ValueError('state-estimation quaternion is non-finite')
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        raise ValueError('state-estimation quaternion is degenerate')
    return tuple(value / norm for value in values)


def _quaternion_roll_pitch(q: Any) -> tuple[float, float]:
    x, y, z, w = _normalized_quaternion(q)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def _quaternion_yaw(q: Any) -> float:
    x, y, z, w = _normalized_quaternion(q)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class MobileManipulationRuntime(Node):
    """Execute a text-requested task using only synchronized ROS perception."""

    def __init__(self) -> None:
        """Load external models and create the fail-closed ROS boundary."""
        super().__init__('z_manip_task_runtime')
        self._declare_parameters()
        stack_path = str(self.get_parameter('stack_config_path').value)
        if not stack_path:
            raise ValueError('stack_config_path is required')
        self._config = load_stack_config(stack_path)
        validate_position_hold_frame_contract(
            str(self.get_parameter('platform_odometry_child_frame').value),
            self._config.robot.platform_base_frame,
        )
        self._planner = OnlinePlanner(self._config)
        validate_grasp_aperture_contract(
            candidate_min_m=self._config.grasp_plan.min_width_m,
            candidate_max_m=self._config.grasp_plan.max_width_m,
            open_aperture_m=float(self.get_parameter('open_aperture_m').value),
            squeeze_m=float(self.get_parameter('grasp_squeeze_m').value),
            command_min_m=float(self.get_parameter('gripper_min_aperture_m').value),
            command_max_m=float(self.get_parameter('gripper_max_aperture_m').value),
            contact_margin_m=float(self.get_parameter('grasp_contact_margin_m').value),
        )
        mount_parent_t_piper = fixed_transform_from_urdf(
            self._config.robot.urdf_path,
            self._config.robot.mount_parent_link,
            self._config.robot.base_link,
        )
        self._piper_t_platform = np.linalg.inv(mount_parent_t_piper)
        # Bind Jazzy's jump callback to this node's simulation clock.
        self._tf_buffer = Buffer(node=self)
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='z-manip-plan')
        self._future: Future[Any] | None = None
        self._future_kind = ''
        self._future_serial = 0
        self._future_generation: int | None = None
        self._future_cancel_event: threading.Event | None = None
        self._future_base_anchor: _PlanningBaseAnchor | None = None
        self._future_observation_identity: _PlanningObservationIdentity | None = None
        self._future_observation_wait: _PlanningObservationWait | None = None
        self._task_generation = TaskGenerationGuard()
        self._core = RuntimeSafetyCore()
        self._task = MobileManipulationStateMachine(self._config.retry_budget)
        self._approach = TwoStageApproachController(self._config.approach)
        self._posture_guard = PostureSafetyGate(
            max_roll_rad=self._config.approach.max_roll_rad,
            max_pitch_rad=self._config.approach.max_pitch_rad,
            max_age_s=float(self.get_parameter('posture_state_max_age_s').value),
            acquisition_timeout_s=float(
                self.get_parameter('posture_state_acquisition_timeout_s').value,
            ),
        )
        self._coarse_nav_posture_violation_started_at_s: float | None = None
        self._visual_search = BoundedYawSearch(VisualSearchConfig(
            yaw_step_rad=float(self.get_parameter('visual_search_yaw_step_rad').value),
            max_yaw_offset_rad=float(
                self.get_parameter('visual_search_max_yaw_offset_rad').value,
            ),
            yaw_tolerance_rad=float(
                self.get_parameter('visual_search_yaw_tolerance_rad').value,
            ),
            settle_yaw_tolerance_rad=float(
                self.get_parameter('visual_search_settle_yaw_tolerance_rad').value,
            ),
            position_heading_reacquire_tolerance_rad=float(self.get_parameter(
                'visual_search_position_heading_reacquire_tolerance_rad',
            ).value),
            yaw_gain=float(self.get_parameter('visual_search_yaw_gain').value),
            max_yaw_rate_rps=float(
                self.get_parameter('visual_search_max_yaw_rate_rps').value,
            ),
            min_yaw_rate_rps=float(
                self.get_parameter('visual_search_min_yaw_rate_rps').value,
            ),
            turn_timeout_s=float(
                self.get_parameter('visual_search_turn_timeout_s').value,
            ),
            max_turn_timeout_s=float(
                self.get_parameter('visual_search_max_turn_timeout_s').value,
            ),
            max_planar_drift_m=float(
                self.get_parameter('visual_search_max_planar_drift_m').value,
            ),
            position_hold_deadband_m=float(
                self.get_parameter('visual_search_position_hold_deadband_m').value,
            ),
            position_completion_tolerance_m=float(
                self.get_parameter(
                    'visual_search_position_completion_tolerance_m',
                ).value,
            ),
            moving_rebound_reacquire_m=float(
                self.get_parameter(
                    'visual_search_moving_rebound_reacquire_m',
                ).value,
            ),
            position_hold_gain_s_inv=float(
                self.get_parameter('visual_search_position_hold_gain_s_inv').value,
            ),
            max_position_hold_speed_mps=float(
                self.get_parameter('visual_search_max_position_hold_speed_mps').value,
            ),
            position_hold_slowdown_radius_m=float(self.get_parameter(
                'visual_search_position_hold_slowdown_radius_m',
            ).value),
            min_position_hold_speed_mps=float(self.get_parameter(
                'visual_search_min_position_hold_speed_mps',
            ).value),
            position_hold_timeout_s=float(
                self.get_parameter('visual_search_position_hold_timeout_s').value,
            ),
            settle_max_linear_speed_mps=float(
                self.get_parameter(
                    'visual_search_settle_max_linear_speed_mps',
                ).value,
            ),
            settle_max_angular_speed_rps=float(
                self.get_parameter(
                    'visual_search_settle_max_angular_speed_rps',
                ).value,
            ),
            stationary_wait_timeout_s=float(
                self.get_parameter(
                    'visual_search_stationary_wait_timeout_s',
                ).value,
            ),
            stationary_quiet_window_s=float(
                self.get_parameter(
                    'visual_search_stationary_quiet_window_s',
                ).value,
            ),
            stationary_max_odom_gap_s=float(
                self.get_parameter(
                    'visual_search_stationary_max_odom_gap_s',
                ).value,
            ),
            settle_reacquire_budget_s=float(
                self.get_parameter(
                    'visual_search_settle_reacquire_budget_s',
                ).value,
            ),
            # Allow one delayed control callback after the nominal deadline
            # while keeping every turn on a finite, deterministic budget.
            deadline_grace_s=(
                2.0 * float(self.get_parameter('control_period_s').value)
            ),
        ))
        self._visual_search_stationarity = ContinuousMotionQuietWindow(
            quiet_window_s=self._visual_search.config.stationary_quiet_window_s,
            max_odom_gap_s=self._visual_search.config.stationary_max_odom_gap_s,
            max_linear_speed_mps=(
                self._visual_search.config.settle_max_linear_speed_mps
            ),
            max_angular_speed_rps=(
                self._visual_search.config.settle_max_angular_speed_rps
            ),
        )
        self._visual_servo_vertical_stationarity = ContinuousMotionQuietWindow(
            quiet_window_s=self._visual_search.config.stationary_quiet_window_s,
            max_odom_gap_s=self._visual_search.config.stationary_max_odom_gap_s,
            max_linear_speed_mps=(
                self._visual_search.config.settle_max_linear_speed_mps
            ),
            max_angular_speed_rps=(
                self._visual_search.config.settle_max_angular_speed_rps
            ),
        )
        self._verifier = GraspVerifier()
        self._execution_occlusion = ExecutionOcclusionGate(
            ExecutionOcclusionConfig(
                max_duration_s=float(self.get_parameter(
                    'execution_occlusion_max_duration_s',
                ).value),
                joint_state_max_age_s=float(self.get_parameter(
                    'execution_occlusion_joint_state_max_age_s',
                ).value),
                execution_status_max_age_s=float(self.get_parameter(
                    'execution_occlusion_status_max_age_s',
                ).value),
                command_ack_timeout_s=float(self.get_parameter(
                    'execution_occlusion_command_ack_timeout_s',
                ).value),
                near_contact_joint_tolerance_rad=float(self.get_parameter(
                    'execution_occlusion_near_contact_tolerance_rad',
                ).value),
                lift_path_joint_tolerance_rad=float(self.get_parameter(
                    'execution_occlusion_lift_path_tolerance_rad',
                ).value),
                max_path_regression_samples=int(self.get_parameter(
                    'execution_occlusion_max_path_regression_samples',
                ).value),
            ),
        )
        self._execution_occlusion_verification_reacquire_timeout_s = float(
            self.get_parameter(
                'execution_occlusion_verification_reacquire_timeout_s',
            ).value,
        )
        if (
            not math.isfinite(
                self._execution_occlusion_verification_reacquire_timeout_s,
            )
            or self._execution_occlusion_verification_reacquire_timeout_s <= 0.0
            or self._execution_occlusion_verification_reacquire_timeout_s
            > self._execution_occlusion.config.max_duration_s
        ):
            raise ValueError(
                'occlusion verification reacquisition timeout must be positive '
                'and no longer than the prediction window',
            )
        self._post_release_verification_policy = PostReleaseVerificationPolicy(
            timeout_s=float(self.get_parameter(
                'post_release_verification_timeout_s',
            ).value),
            wall_timeout_s=float(self.get_parameter(
                'post_release_verification_wall_timeout_s',
            ).value),
            min_stable_duration_s=float(self.get_parameter(
                'post_release_min_stable_duration_s',
            ).value),
            min_samples=int(self.get_parameter(
                'post_release_min_samples',
            ).value),
            min_target_points=int(self.get_parameter(
                'post_release_min_target_points',
            ).value),
            max_target_motion_m=float(self.get_parameter(
                'post_release_max_target_motion_m',
            ).value),
            min_region_support_fraction=float(self.get_parameter(
                'post_release_min_region_support_fraction',
            ).value),
            min_gripper_clearance_m=float(self.get_parameter(
                'post_release_min_gripper_clearance_m',
            ).value),
            max_rgbd_target_skew_s=float(self.get_parameter(
                'post_release_max_rgbd_target_skew_s',
            ).value),
            max_joint_target_skew_s=float(self.get_parameter(
                'post_release_max_joint_target_skew_s',
            ).value),
            max_target_depth_correspondence_m=float(self.get_parameter(
                'post_release_max_target_depth_correspondence_m',
            ).value),
            max_object_position_error_m=float(self.get_parameter(
                'post_release_max_object_position_error_m',
            ).value),
            max_object_orientation_error_rad=float(self.get_parameter(
                'post_release_max_object_orientation_error_rad',
            ).value),
            max_object_upright_error_rad=float(self.get_parameter(
                'post_release_max_object_upright_error_rad',
            ).value),
            min_object_registration_inlier_fraction=float(self.get_parameter(
                'post_release_min_object_registration_inlier_fraction',
            ).value),
            max_object_registration_rms_m=float(self.get_parameter(
                'post_release_max_object_registration_rms_m',
            ).value),
        )
        self._object_geometry_config = ObjectGeometryConfig(
            min_points=int(self.get_parameter(
                'carried_object_min_points',
            ).value),
            trim_mad_scale=float(self.get_parameter(
                'carried_object_trim_mad_scale',
            ).value),
            extent_percentile=float(self.get_parameter(
                'carried_object_extent_percentile',
            ).value),
            min_extent_m=float(self.get_parameter(
                'carried_object_min_extent_m',
            ).value),
            min_axis_separation_ratio=float(self.get_parameter(
                'carried_object_min_axis_separation_ratio',
            ).value),
            max_axial_transverse_ratio=float(self.get_parameter(
                'carried_object_max_axial_transverse_ratio',
            ).value),
            max_reference_points=int(self.get_parameter(
                'carried_object_max_reference_points',
            ).value),
        )
        self._serial_gate = ObservationSerialGate(
            sync_slop_s=float(self.get_parameter('sync_slop_s').value),
            max_age_s=float(self.get_parameter('max_perception_age_s').value),
        )
        self._perception_valid = False
        self._valid_seen_at: float | None = None
        self._target_camera: np.ndarray | None = None
        self._target_piper: np.ndarray | None = None
        self._target_stamp = 0.0
        self._target_stamp_ns = 0
        self._target_frame_id = ''
        self._target_cloud: np.ndarray | None = None
        self._target_uv: np.ndarray | None = None
        self._target_cloud_stamp = 0.0
        self._target_cloud_stamp_ns = 0
        self._target_cloud_frame_id = ''
        self._scene_cloud: np.ndarray | None = None
        self._scene_cloud_stamp = 0.0
        self._scene_cloud_stamp_ns = 0
        self._scene_cloud_frame_id = ''
        self._camera_origin_piper: np.ndarray | None = None
        self._camera_rotation_piper: np.ndarray | None = None
        self._affordance: dict[str, Any] | None = None
        self._affordance_placement_semantics: (
            PlacementVerificationSemantics | None
        ) = None
        self._affordance_generation = 0
        self._affordance_request_id = ''
        self._affordance_producer_epoch = ''
        self._required_affordance_generation: int | None = None
        self._perception_generation = 0
        self._required_perception_generation: int | None = None
        self._required_perception_request_id: str | None = None
        self._required_grounding_scope: str | None = None
        self._bound_perception_request_id: str | None = None
        self._bound_perception_producer_epoch: str | None = None
        self._bound_perception_generation: int | None = None
        self._valid_perception_request_id: str | None = None
        self._valid_perception_producer_epoch: str | None = None
        self._valid_perception_generation: int | None = None
        self._valid_observation_stamp_ns: int | None = None
        self._valid_observation_frame_id = ''
        self._handled_perception_failure: tuple[str, str, int, str] | None = None
        self._reground_started_at: float | None = None
        self._reground_last_tick_at: float | None = None
        self._near_view_pose_name = ''
        self._near_view_achieved_pose_name = ''
        self._near_view_settle_started_at: float | None = None
        self._near_view_settle_last_tick_at: float | None = None
        self._near_view_settle_until: float | None = None
        self._near_view_deadline_s: float | None = None
        self._near_view_joint_sequence_floor: int | None = None
        self._near_view_joint_target: np.ndarray | None = None
        self._near_view_joint_error_rad: float | None = None
        self._image_size: tuple[int, int] | None = None
        self._joint_state: np.ndarray | None = None
        self._joint_history: deque[_JointFeedback] = deque(maxlen=40)
        self._joint_stamp_ns: int | None = None
        self._joint_sequence = 0
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw: float | None = None
        self._position_xy: tuple[float, float] | None = None
        self._odom_seen_at: float | None = None
        self._odom_stamp_ns: int | None = None
        self._odom_payload: tuple[float, ...] | None = None
        self._odom_sequence = 0
        self._base_linear_speed_mps: float | None = None
        self._base_angular_speed_rps: float | None = None
        self._base_yaw_rate_rps: float | None = None
        self._nav_speed = float('inf')
        self._coarse_nav_ready = False
        self._work_pose: dict[str, object] | None = None
        self._work_pose_history_map: list[np.ndarray] = []
        self._work_pose_created_at_s: float | None = None
        self._navigation_status_seen_s: float | None = None
        self._navigation_status_goal_id = ''
        self._navigation_status_phase = ''
        self._navigation_goal_acknowledged = False
        self._navigation_ack_position_xy: tuple[float, float] | None = None
        self._navigation_ack_odom_sequence: int | None = None
        self._navigation_ack_odom_stamp_ns: int | None = None
        self._navigation_history_recorded = False
        self._coarse_nav_perception_loss_detail = ''
        self._coarse_nav_arrival_started_at_s: float | None = None
        self._coarse_nav_arrival_stable_since_s: float | None = None
        self._coarse_nav_arrival_stable_start_odom_stamp_ns: int | None = None
        self._coarse_nav_arrival_last_odom_sequence: int | None = None
        self._coarse_nav_arrival_last_odom_stamp_ns: int | None = None
        self._coarse_nav_arrival_anchor_xy: tuple[float, float] | None = None
        self._coarse_nav_arrival_anchor_yaw_rad: float | None = None
        self._frozen_coarse_nav_authorization_identity: dict[str, object] | None = None
        self._execution_status = None
        self._execution_status_seen_s: float | None = None
        self._latest_gripper_command_id = 0
        self._expected_gripper_command_id: int | None = None
        self._gripper_command_sent_s: float | None = None
        self._gripper_feedback: deque[tuple[float, int, float]] = deque(maxlen=32)
        self._trajectory_deadline_s: float | None = None
        self._pregrasp_program: PregraspTransitProgram | None = None
        self._pregrasp_planning_identity: _PlanningObservationIdentity | None = None
        self._pregrasp_dispatch_fence: _PregraspDispatchFence | None = None
        self._pregrasp_handoff: _PregraspHandoff | None = None
        self._pregrasp_stable_joint_sequence: int | None = None
        self._pregrasp_stable_joint_stamp_ns: int | None = None
        self._pregrasp_joint_error_rad: float | None = None
        self._approach_planning_anchor: _ApproachPlanningAnchor | None = None
        self._approach_execution_joint_fence: (
            _ApproachExecutionJointFence | None
        ) = None
        self._program: GraspCompletionProgram | None = None
        self._carry_program: Any | None = None
        self._place_programs: dict[str, TimedJointTrajectory] = {}
        self._place_trajectory: JointTrajectory | None = None
        self._place_contract: dict[str, Any] | None = None
        self._place_goal_id = ''
        self._place_transaction_requested = False
        self._place_transaction_abort_sent = False
        self._place_planning_started_at: float | None = None
        self._place_planning_started_wall_s: float | None = None
        self._release_started_at: float | None = None
        self._place_observation_identity: PlacementObservationIdentity | None = None
        self._carried_object_geometry: CarriedObjectGeometry | None = None
        self._carried_object_observation_stamp_ns: int | None = None
        self._post_release_release_command_id: int | None = None
        self._post_release_pending_evidence: (
            PostReleaseVerificationEvidence | None
        ) = None
        self._post_release_verification_started_at_s: float | None = None
        self._post_release_verification_started_wall_s: float | None = None
        self._post_release_verification_last_tick_s: float | None = None
        self._post_release_verified_evidence: (
            PostReleaseVerificationEvidence | None
        ) = None
        self._desired_depth: float | None = None
        self._approximate_displacement: float | None = None
        self._selection_mode = ''
        self._pose_settle_until: float | None = None
        self._pose_settle_started_at: float | None = None
        self._pose_settle_last_tick_at: float | None = None
        self._visual_search_settle_reference: _VisualSearchSettleReference | None = None
        self._terminal_ownership_released = False
        self._lookout_pending = False
        self._visual_search_pending = False
        self._visual_search_edge_direction = 0
        self._visual_search_error_rad: float | None = None
        self._visual_search_reason = ''
        self._visual_servo_vertical_recovery_started_at_s: float | None = None
        self._visual_servo_vertical_minimum_cloud_stamp_ns: int | None = None
        self._visual_servo_vertical_safe_start_cloud_stamp_ns: int | None = None
        self._visual_servo_vertical_safe_last_cloud_stamp_ns: int | None = None
        self._closing_started_at: float | None = None
        self._verification_started_at: float | None = None
        self._commanded_close_aperture: float | None = None
        self._execution_occlusion_target_piper: np.ndarray | None = None
        self._execution_occlusion_target_cloud: np.ndarray | None = None
        self._execution_occlusion_scene_cloud: np.ndarray | None = None
        self._execution_occlusion_loss_detail = ''
        self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
            False,
            'execution occlusion is not armed',
        )
        self._last_status_json = ''
        self._setup_io()
        self._clear_debug_plan()
        self.create_timer(float(self.get_parameter('control_period_s').value), self._tick)
        authorization_period = float(
            self.get_parameter('frozen_coarse_nav_authorization_period_s').value,
        )
        if (
            not math.isfinite(authorization_period)
            or not 0.0 < authorization_period <= 0.10
        ):
            raise ValueError(
                'frozen coarse-navigation authorization period must be in '
                '(0, 0.10] seconds',
            )
        self._frozen_coarse_nav_authorization_clock = Clock(
            clock_type=ClockType.STEADY_TIME,
        )
        self.create_timer(
            authorization_period,
            self._frozen_coarse_nav_authorization_tick,
            clock=self._frozen_coarse_nav_authorization_clock,
        )
        self.create_timer(
            authorization_period,
            self._post_release_wall_timeout_tick,
            clock=self._frozen_coarse_nav_authorization_clock,
        )
        self.create_timer(
            authorization_period,
            self._place_planning_wall_timeout_tick,
            clock=self._frozen_coarse_nav_authorization_clock,
        )
        self.get_logger().info(
            'ready: perception-only task runtime; no object truth subscriptions',
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'stack_config_path': '',
            'task_topic': '/z_manip/task/request',
            'task_cancel_topic': '/z_manip/task/cancel',
            'place_region_request_topic': '/z_manip/place/region_request',
            'place_trajectory_topic': '/z_manip/place/trajectory',
            'place_contract_topic': '/z_manip/place/trajectory_contract',
            'place_status_topic': '/z_manip/place/status',
            'place_transaction_control_topic': (
                '/z_manip/place/transaction_control'
            ),
            'post_release_verification_topic': (
                '/z_manip/place/post_release_verification'
            ),
            'perception_valid_topic': '/z_manip/perception/valid',
            'target_topic': '/z_manip/perception/target_3d',
            'target_cloud_topic': '/z_manip/perception/target_pointcloud',
            'scene_cloud_topic': '/z_manip/perception/scene_pointcloud',
            'affordance_topic': '/z_manip/perception/affordance',
            'perception_status_topic': '/z_manip/perception/status',
            'frozen_coarse_nav_authorization_topic': (
                '/z_manip/coarse_nav/perception_loss_authorization'
            ),
            'frozen_coarse_nav_authorization_period_s': 0.05,
            'visual_search_active_topic': '/z_manip/visual_search/active',
            'grounding_request_topic': '/z_manip/grounding/request',
            'grounding_reset_topic': '/z_manip/grounding/reset',
            # Keep the legacy parameter name, but consume platform-base odometry.
            'state_estimation_topic': '/odom_base_link',
            'platform_odometry_parent_frame': 'map',
            'platform_odometry_child_frame': 'base_link',
            'coarse_nav_ready_topic': '/z_manip/navigation/coarse_ready',
            'navigation_status_topic': '/z_manip/navigation/status',
            'execution_status_topic': '/piper/execution_status',
            'cancel_goal_topic': '/cancel_goal',
            'named_pose_topic': '/piper/named_pose',
            'arm_cancel_topic': '/piper/cancel',
            'task_status_topic': '/z_manip/task/status',
            'debug_markers_topic': '/z_manip/debug/markers',
            'debug_path_topic': '/z_manip/debug/arm_path',
            'place_mode': 'observed_place',
            'lookout_pose': 'LOOKOUT',
            'near_view_pose': '',
            'near_view_settle_s': 3.0,
            'near_view_timeout_s': 12.0,
            'near_view_joint_positions': [],
            'near_view_joint_tolerance_rad': 0.05,
            'carry_joint_positions': [0.0, 1.0, -0.71, 0.0, 0.0, 0.0],
            'open_aperture_m': 0.070,
            'close_aperture_m': 0.014,
            'gripper_min_aperture_m': 0.0,
            'gripper_max_aperture_m': 0.075,
            'grasp_squeeze_m': 0.006,
            'gripper_settle_s': 0.65,
            'grasp_contact_margin_m': 0.0015,
            'verification_timeout_s': 4.0,
            'verification_joint_state_max_age_s': 0.35,
            'semantic_reground_timeout_s': 105.0,
            'execution_ack_margin_s': 2.0,
            'max_trajectory_start_error_rad': 0.04,
            'gripper_stable_samples': 3,
            'gripper_stable_tolerance_m': 0.0008,
            'release_settle_s': 0.65,
            'release_min_aperture_m': 0.065,
            'place_planning_timeout_s': 12.0,
            'place_planning_wall_timeout_s': 30.0,
            'post_release_verification_timeout_s': 2.0,
            'post_release_verification_wall_timeout_s': 12.0,
            'post_release_min_stable_duration_s': 0.50,
            'post_release_min_samples': 3,
            'post_release_min_target_points': 24,
            'post_release_max_target_motion_m': 0.025,
            'post_release_min_region_support_fraction': 0.80,
            'post_release_min_gripper_clearance_m': 0.04,
            'post_release_max_rgbd_target_skew_s': 0.025,
            'post_release_max_joint_target_skew_s': 0.12,
            'post_release_max_target_depth_correspondence_m': 0.012,
            'post_release_max_object_position_error_m': 0.04,
            'post_release_max_object_orientation_error_rad': 0.35,
            'post_release_max_object_upright_error_rad': 0.26,
            'post_release_min_object_registration_inlier_fraction': 0.55,
            'post_release_max_object_registration_rms_m': 0.025,
            'carried_object_min_points': 40,
            'carried_object_trim_mad_scale': 4.5,
            'carried_object_extent_percentile': 2.0,
            'carried_object_min_extent_m': 0.008,
            'carried_object_min_axis_separation_ratio': 1.20,
            'carried_object_max_axial_transverse_ratio': 1.90,
            'carried_object_max_reference_points': 512,
            'carried_object_max_joint_target_skew_s': 0.12,
            'standoff_planning_budget_s': 20.0,
            'work_pose_odom_max_age_s': 0.50,
            'work_pose_anchor_translation_tolerance_m': 0.05,
            'work_pose_anchor_yaw_tolerance_rad': 0.0523598776,
            'work_pose_target_drift_tolerance_m': 0.06,
            'work_pose_goal_tolerance_m': 0.33,
            'work_pose_history_min_displacement_m': 0.03,
            'coarse_navigation_status_timeout_s': 3.0,
            'coarse_nav_arrival_settle_s': 0.35,
            'coarse_nav_arrival_stop_timeout_s': 3.0,
            'coarse_nav_arrival_max_linear_speed_mps': 0.050,
            'coarse_nav_arrival_max_angular_speed_rps': 0.05,
            'coarse_nav_arrival_max_xy_excursion_m': 0.010,
            'coarse_nav_arrival_max_yaw_excursion_rad': 0.010,
            'coarse_nav_arrival_max_odom_gap_s': 0.15,
            'coarse_nav_posture_violation_dwell_s': 0.15,
            'grasp_planning_budget_s': 15.0,
            'approach_planning_budget_s': 15.0,
            'pregrasp_dispatch_feedback_wait_timeout_s': 1.0,
            'pregrasp_reobserve_timeout_s': 8.0,
            'pregrasp_joint_state_max_age_s': 0.25,
            'pregrasp_joint_tolerance_rad': 0.05,
            'pregrasp_max_observation_joint_skew_s': 0.12,
            'approach_planning_target_drift_tolerance_m': 0.025,
            'approach_planning_geometry_trim_mad_scale': 4.5,
            'approach_planning_geometry_extent_percentile': 2.0,
            'approach_planning_geometry_max_extent_change_m': 0.008,
            'approach_planning_geometry_max_extent_ratio': 1.25,
            'approach_planning_geometry_axis_separation_ratio': 1.25,
            'approach_planning_geometry_max_orientation_change_rad': 0.3490658504,
            'approach_execution_joint_state_max_age_s': 0.25,
            'approach_execution_joint_wait_timeout_s': 1.0,
            'execution_status_max_age_s': 0.50,
            'carry_planning_budget_s': 8.0,
            'arm_still_window_s': 0.20,
            'arm_still_tolerance_rad': 0.01,
            'lookout_settle_s': 3.0,
            'visual_search_settle_s': 0.75,
            'visual_search_yaw_step_rad': 0.3490658504,
            'visual_search_max_yaw_offset_rad': 1.0471975512,
            'visual_search_yaw_tolerance_rad': 0.0174532925,
            'visual_search_settle_yaw_tolerance_rad': 0.0349065850,
            'visual_search_position_heading_reacquire_tolerance_rad': 0.0698131701,
            'visual_search_yaw_gain': 1.5,
            'visual_search_max_yaw_rate_rps': 0.30,
            'visual_search_min_yaw_rate_rps': 0.0,
            'visual_search_turn_timeout_s': 8.0,
            'visual_search_max_turn_timeout_s': 30.0,
            'visual_search_max_planar_drift_m': 0.15,
            'visual_search_position_hold_deadband_m': 0.01,
            'visual_search_position_completion_tolerance_m': 0.06,
            'visual_search_moving_rebound_reacquire_m': 0.10,
            'visual_search_position_hold_gain_s_inv': 1.0,
            'visual_search_max_position_hold_speed_mps': 0.05,
            'visual_search_position_hold_slowdown_radius_m': 0.0,
            'visual_search_min_position_hold_speed_mps': 0.0,
            'visual_search_position_hold_timeout_s': 4.0,
            'visual_search_settle_max_linear_speed_mps': 0.035,
            'visual_search_settle_max_angular_speed_rps': 0.05,
            'visual_search_stationary_wait_timeout_s': 0.0,
            'visual_search_stationary_quiet_window_s': 0.35,
            'visual_search_stationary_max_odom_gap_s': 0.15,
            'visual_search_settle_reacquire_budget_s': 2.0,
            'visual_search_odom_timeout_s': 0.50,
            'posture_state_max_age_s': 0.50,
            'posture_state_acquisition_timeout_s': 1.0,
            'visual_search_horizontal_margin_ratio': 0.08,
            'visual_search_vertical_margin_ratio': 0.06,
            'visual_servo_image_margin_ratio': 0.02,
            'sync_slop_s': 1e-6,
            'max_perception_age_s': 0.35,
            'perception_loss_timeout_s': 0.60,
            'execution_occlusion_max_duration_s': 3.0,
            'execution_occlusion_joint_state_max_age_s': 0.25,
            'execution_occlusion_status_max_age_s': 0.30,
            'execution_occlusion_command_ack_timeout_s': 0.40,
            'execution_occlusion_near_contact_tolerance_rad': 0.05,
            'execution_occlusion_lift_path_tolerance_rad': 0.08,
            'execution_occlusion_max_path_regression_samples': 3,
            'execution_occlusion_verification_reacquire_timeout_s': 0.75,
            'control_period_s': 0.05,
            'tf_timeout_s': 0.12,
            'semantic_min_points': 40,
        }
        for name, value in defaults.items():
            descriptor = None
            if name == 'near_view_joint_positions':
                # ROS infers an empty YAML sequence as BYTE_ARRAY. This
                # platform-optional vector must accept a deployment's finite
                # DOUBLE_ARRAY override before its runtime shape validation.
                descriptor = ParameterDescriptor(dynamic_typing=True)
            self.declare_parameter(name, value, descriptor)

    def _topic(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _setup_io(self) -> None:
        reliable = 10
        latched_debug = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._grounding_pub = self.create_publisher(
            String, self._topic('grounding_request_topic'), reliable,
        )
        self._grounding_reset_pub = self.create_publisher(
            Empty, self._topic('grounding_reset_topic'), reliable,
        )
        self._visual_search_active_pub = self.create_publisher(
            Bool, self._topic('visual_search_active_topic'), latched_debug,
        )
        self._frozen_coarse_nav_authorization_pub = self.create_publisher(
            String,
            self._topic('frozen_coarse_nav_authorization_topic'),
            reliable,
        )
        self._velocity_pub = self.create_publisher(
            TwistStamped, self._config.topics.local_velocity, reliable,
        )
        self._cancel_nav_pub = self.create_publisher(
            Bool, self._topic('cancel_goal_topic'), reliable,
        )
        self._named_pose_pub = self.create_publisher(
            String, self._topic('named_pose_topic'), reliable,
        )
        self._trajectory_pub = self.create_publisher(
            JointTrajectory, self._config.topics.arm_trajectory, reliable,
        )
        self._gripper_pub = self.create_publisher(
            Float32, self._config.topics.gripper_aperture, reliable,
        )
        self._arm_cancel_pub = self.create_publisher(
            Bool, self._topic('arm_cancel_topic'), reliable,
        )
        self._status_pub = self.create_publisher(
            String, self._topic('task_status_topic'), latched_debug,
        )
        self._markers_pub = self.create_publisher(
            MarkerArray, self._topic('debug_markers_topic'), latched_debug,
        )
        self._path_pub = self.create_publisher(
            Path, self._topic('debug_path_topic'), latched_debug,
        )
        self._place_region_pub = self.create_publisher(
            String, self._topic('place_region_request_topic'), reliable,
        )
        self._place_transaction_control_pub = self.create_publisher(
            String,
            self._topic('place_transaction_control_topic'),
            reliable,
        )
        self.create_subscription(String, self._topic('task_topic'), self._task_cb, reliable)
        self.create_subscription(
            Bool, self._topic('task_cancel_topic'), self._task_cancel_cb, reliable,
        )
        self._place_trajectory_subscription = self.create_subscription(
            JointTrajectory,
            self._topic('place_trajectory_topic'),
            self._place_trajectory_cb,
            reliable,
        )
        self.create_subscription(
            String, self._topic('place_contract_topic'), self._place_contract_cb, reliable,
        )
        self.create_subscription(
            String, self._topic('place_status_topic'), self._place_status_cb, reliable,
        )
        self.create_subscription(
            String,
            self._topic('post_release_verification_topic'),
            self._post_release_verification_cb,
            reliable,
        )
        self.create_subscription(
            Bool, self._topic('perception_valid_topic'), self._valid_cb, reliable,
        )
        self.create_subscription(
            Detection3D, self._topic('target_topic'), self._target_cb, reliable,
        )
        self.create_subscription(
            PointCloud2, self._topic('target_cloud_topic'), self._target_cloud_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2, self._topic('scene_cloud_topic'), self._scene_cloud_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, self._topic('affordance_topic'), self._affordance_cb, reliable,
        )
        self.create_subscription(
            DiagnosticArray,
            self._topic('perception_status_topic'),
            self._perception_status_cb,
            latched_debug,
        )
        self.create_subscription(
            CameraInfo, self._config.topics.camera_info, self._camera_info_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState, self._config.topics.joint_state, self._joint_cb, reliable,
        )
        self.create_subscription(
            Odometry, self._topic('state_estimation_topic'), self._odom_cb, reliable,
        )
        self.create_subscription(
            Bool, self._topic('coarse_nav_ready_topic'), self._coarse_ready_cb, reliable,
        )
        self.create_subscription(
            String,
            self._topic('navigation_status_topic'),
            self._navigation_status_cb,
            latched_debug,
        )
        self.create_subscription(
            String, self._topic('execution_status_topic'), self._execution_cb, reliable,
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _fail_posture(self, reason: str) -> None:
        """Atomically invalidate task work and cancel every motion owner."""
        self._coarse_nav_posture_violation_started_at_s = None
        MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
            self,
            False,
        )
        self._visual_search_settle_reference = None
        MobileManipulationRuntime._clear_visual_search_stationarity(
            self,
            'posture_failure',
        )
        MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
            self,
            'posture_failure',
        )
        self._pose_settle_started_at = None
        self._pose_settle_last_tick_at = None
        if not self._core.active:
            return
        self._approach.reset()
        self._visual_search.reset()
        self._verifier.reset()
        self._pregrasp_program = None
        MobileManipulationRuntime._clear_pregrasp_handoff(self)
        self._program = None
        self._carry_program = None
        self._place_programs.clear()
        self._place_trajectory = None
        self._place_contract = None
        self._place_planning_started_at = None
        self._place_planning_started_wall_s = None
        self._release_started_at = None
        MobileManipulationRuntime._reset_post_release_verification(self)
        self._closing_started_at = None
        self._verification_started_at = None
        self._trajectory_deadline_s = None
        self._expected_gripper_command_id = None
        self._gripper_command_sent_s = None
        self._lookout_pending = False
        MobileManipulationRuntime._clear_near_view_settle(self)
        self._pose_settle_until = None
        self._visual_search_pending = False
        detail = f'posture safety violation: {reason}'
        action = self._core.posture_invalid(detail)
        self._release_terminal_ownership()
        self._apply_safety(action)
        self._publish_status(force=True)

    def _guard_active_posture(self, now: float) -> bool:
        """Hold startup motion or fail an active task on unsafe posture state."""
        if not self._core.active:
            self._coarse_nav_posture_violation_started_at_s = None
            return True
        try:
            assessment = self._posture_guard.assess(now)
        except RuntimeError as error:
            self._fail_posture(str(error))
            return False
        if assessment.state is PostureState.WAITING:
            self._coarse_nav_posture_violation_started_at_s = None
            # Arm and navigation cancellation are issued when task ownership
            # begins; keep the task-local velocity channel at zero while the
            # first state sample is acquired.
            self._apply_safety(SafetyAction(stop_base=True))
            return False
        if not assessment.safe:
            is_coarse_nav_limit = (
                self._core.phase is RuntimePhase.COARSE_NAV
                and assessment.reason.startswith('base posture limit exceeded:')
            )
            if is_coarse_nav_limit:
                dwell_s = float(
                    self.get_parameter(
                        'coarse_nav_posture_violation_dwell_s',
                    ).value,
                )
                if not math.isfinite(dwell_s) or dwell_s <= 0.0:
                    self._coarse_nav_posture_violation_started_at_s = None
                    self._fail_posture(
                        'coarse navigation posture violation dwell must be '
                        'finite and positive',
                    )
                    return False
                started_at = self._coarse_nav_posture_violation_started_at_s
                if started_at is None:
                    self._coarse_nav_posture_violation_started_at_s = float(now)
                    return False
                elapsed_s = float(now) - started_at
                if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
                    self._coarse_nav_posture_violation_started_at_s = None
                    self._fail_posture(
                        'coarse navigation posture violation dwell clock is invalid',
                    )
                    return False
                if elapsed_s < dwell_s:
                    # Keep the navigation owner alive, but do not let the task
                    # advance into near-field or arm motion while out of bounds.
                    return False
            self._coarse_nav_posture_violation_started_at_s = None
            self._fail_posture(assessment.reason)
            return False
        self._coarse_nav_posture_violation_started_at_s = None
        return True

    def _dispatch_pending_lookout(self, now: float) -> None:
        """Move to LOOKOUT only after the posture gate authorizes motion."""
        if not self._lookout_pending:
            return
        if self._core.phase is not RuntimePhase.POSE_SETTLE or not self._core.active:
            self._lookout_pending = False
            self._pose_settle_started_at = None
            self._pose_settle_last_tick_at = None
            self._pose_settle_until = None
            return
        self._lookout_pending = False
        self._pose_settle_started_at = now
        self._pose_settle_last_tick_at = now
        self._pose_settle_until = (
            now + float(self.get_parameter('lookout_settle_s').value)
        )
        self._named_pose_pub.publish(String(data=self._topic_value('lookout_pose')))

    def _clear_near_view_settle(self) -> None:
        self._near_view_pose_name = ''
        self._near_view_settle_started_at = None
        self._near_view_settle_last_tick_at = None
        self._near_view_settle_until = None
        self._near_view_deadline_s = None
        self._near_view_joint_sequence_floor = None
        self._near_view_joint_target = None
        self._near_view_joint_error_rad = None

    def _begin_near_view_settle(self, now: float) -> None:
        """Retire the moving-base track before changing the wrist viewpoint."""
        pose = self._topic_value('near_view_pose').strip()
        if not pose:
            self._request_semantic_reground(now)
            return
        settle_s = float(self.get_parameter('near_view_settle_s').value)
        timeout_s = float(self.get_parameter('near_view_timeout_s').value)
        tolerance = float(
            self.get_parameter('near_view_joint_tolerance_rad').value,
        )
        target = np.asarray(
            self.get_parameter('near_view_joint_positions').value,
            dtype=float,
        )
        if (
            not math.isfinite(float(now))
            or not math.isfinite(settle_s)
            or settle_s <= 0.0
            or not math.isfinite(timeout_s)
            or timeout_s <= settle_s
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or self._joint_state is None
            or target.shape != self._joint_state.shape
            or not np.all(np.isfinite(target))
        ):
            raise ValueError(
                'near-view pose requires bounded timing and a finite joint target',
            )
        self._invalidate_perception_session()
        self._near_view_achieved_pose_name = ''
        self._near_view_pose_name = pose
        self._near_view_settle_started_at = float(now)
        self._near_view_settle_last_tick_at = float(now)
        self._near_view_settle_until = float(now) + settle_s
        self._near_view_deadline_s = float(now) + timeout_s
        self._near_view_joint_sequence_floor = int(self._joint_sequence)
        self._near_view_joint_target = target.copy()
        self._near_view_joint_error_rad = None
        self._named_pose_pub.publish(String(data=pose))

    def _near_view_settle_tick(self, now: float) -> None:
        """Hold the stopped base until the configured wrist view has settled."""
        started = self._near_view_settle_started_at
        last_tick = self._near_view_settle_last_tick_at
        settle_until = self._near_view_settle_until
        deadline = self._near_view_deadline_s
        sequence_floor = self._near_view_joint_sequence_floor
        target = self._near_view_joint_target
        self._publish_zero()
        if (
            self._core.phase is not RuntimePhase.NEAR_GROUNDING
            or not self._near_view_pose_name
            or started is None
            or last_tick is None
            or settle_until is None
            or deadline is None
            or sequence_floor is None
            or target is None
            or not all(math.isfinite(value) for value in (
                float(now), started, last_tick, settle_until, deadline,
            ))
            or settle_until <= started
            or deadline <= settle_until
            or last_tick < started
        ):
            detail = 'near-view settle timing contract is unavailable or invalid'
            MobileManipulationRuntime._clear_near_view_settle(self)
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if float(now) < last_tick:
            detail = 'near-view settle clock moved backwards'
            MobileManipulationRuntime._clear_near_view_settle(self)
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        self._near_view_settle_last_tick_at = float(now)
        if float(now) < settle_until:
            return
        if float(now) > deadline:
            detail = 'near-view joint convergence deadline expired'
            MobileManipulationRuntime._clear_near_view_settle(self)
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if not self._joint_history:
            return
        latest = self._joint_history[-1]
        if latest.sequence <= sequence_floor:
            return
        positions = np.asarray(latest.positions, dtype=float)
        if positions.shape != target.shape or not np.all(np.isfinite(positions)):
            detail = 'near-view joint feedback is malformed'
            MobileManipulationRuntime._clear_near_view_settle(self)
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        error = float(np.max(np.abs(positions - target)))
        self._near_view_joint_error_rad = error
        tolerance = float(
            self.get_parameter('near_view_joint_tolerance_rad').value,
        )
        if error > tolerance or not self._arm_is_still(float(now)):
            return
        achieved_pose = self._near_view_pose_name
        MobileManipulationRuntime._clear_near_view_settle(self)
        self._near_view_achieved_pose_name = achieved_pose
        self._request_semantic_reground(float(now))

    def _lookup_piper_from(self, source_frame: str, stamp: Any) -> np.ndarray:
        transform = self._tf_buffer.lookup_transform(
            self._config.robot.platform_base_frame,
            source_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=float(self.get_parameter('tf_timeout_s').value)),
        )
        platform_t_source = _matrix_from_transform(transform.transform)
        return self._piper_t_platform @ platform_t_source

    def _task_cb(self, msg: String) -> None:
        with self._lock:
            MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
                self,
                False,
            )
            now = self._now_s()
            try:
                self._core.begin(msg.data)
                self._posture_guard.begin(now)
            except ValueError as error:
                self.get_logger().error(str(error))
                return
            self._coarse_nav_posture_violation_started_at_s = None
            MobileManipulationRuntime._clear_debug_plan(self)
            self._invalidate_async_work()
            self._invalidate_perception_session()
            self._terminal_ownership_released = False
            self._task = MobileManipulationStateMachine(self._config.retry_budget)
            self._approach.reset()
            self._visual_search.reset()
            self._verifier.reset()
            self._reset_execution_occlusion()
            self._serial_gate = ObservationSerialGate(
                sync_slop_s=float(self.get_parameter('sync_slop_s').value),
                max_age_s=float(self.get_parameter('max_perception_age_s').value),
            )
            self._perception_valid = False
            self._valid_seen_at = None
            self._target_camera = None
            self._target_piper = None
            self._target_frame_id = ''
            self._target_cloud = None
            self._target_uv = None
            self._scene_cloud = None
            self._joint_history.clear()
            self._joint_state = None
            self._camera_origin_piper = None
            self._camera_rotation_piper = None
            self._affordance = None
            self._affordance_placement_semantics = None
            self._required_affordance_generation = None
            self._required_perception_generation = None
            self._required_grounding_scope = None
            self._handled_perception_failure = None
            self._reground_started_at = None
            self._reground_last_tick_at = None
            MobileManipulationRuntime._clear_near_view_settle(self)
            self._near_view_achieved_pose_name = ''
            self._pregrasp_program = None
            MobileManipulationRuntime._clear_pregrasp_handoff(self)
            self._program = None
            self._carry_program = None
            self._place_programs.clear()
            self._place_trajectory = None
            self._place_contract = None
            self._place_goal_id = ''
            self._place_transaction_requested = False
            self._place_transaction_abort_sent = False
            self._place_planning_started_at = None
            self._place_planning_started_wall_s = None
            self._release_started_at = None
            self._carried_object_geometry = None
            self._carried_object_observation_stamp_ns = None
            MobileManipulationRuntime._reset_post_release_verification(self)
            self._desired_depth = None
            self._approximate_displacement = None
            self._work_pose = None
            self._work_pose_created_at_s = None
            self._navigation_status_seen_s = None
            self._navigation_status_goal_id = ''
            self._navigation_status_phase = ''
            self._navigation_goal_acknowledged = False
            self._navigation_history_recorded = False
            self._coarse_nav_perception_loss_detail = ''
            self._coarse_nav_arrival_started_at_s = None
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_last_odom_sequence = None
            self._coarse_nav_arrival_last_odom_stamp_ns = None
            if hasattr(self, '_work_pose_history_map'):
                self._work_pose_history_map.clear()
            else:
                self._work_pose_history_map = []
            self._closing_started_at = None
            self._verification_started_at = None
            self._commanded_close_aperture = None
            self._expected_gripper_command_id = None
            self._gripper_command_sent_s = None
            self._gripper_feedback.clear()
            self._trajectory_deadline_s = None
            self._coarse_nav_ready = False
            self._pose_settle_until = None
            self._pose_settle_started_at = None
            self._pose_settle_last_tick_at = None
            self._visual_search_settle_reference = None
            MobileManipulationRuntime._clear_visual_search_stationarity(
                self,
                'task_reset',
            )
            MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
                self,
                'task_reset',
            )
            self._lookout_pending = True
            self._visual_search_pending = False
            self._visual_search_edge_direction = 0
            self._visual_search_error_rad = None
            self._visual_search_reason = ''
            self._selection_mode = ''
            self._visual_search_active_pub.publish(Bool(data=False))
            self._apply_safety(SafetyAction(
                stop_base=True,
                cancel_navigation=True,
                cancel_arm=True,
            ))
            if self._guard_active_posture(now):
                self._dispatch_pending_lookout(now)
            self._publish_status(force=True)

    def _invalidate_async_work(self) -> None:
        """Cancel the tracked future and invalidate any result already in flight."""
        if self._future_cancel_event is not None:
            self._future_cancel_event.set()
        self._task_generation.advance()
        if self._future is not None:
            self._future.cancel()
        self._pregrasp_dispatch_fence = None
        MobileManipulationRuntime._clear_planning_future_state(self)

    def _clear_planning_future_state(self) -> None:
        """Drop one planner result and every immutable contract retained with it."""
        self._future = None
        self._future_kind = ''
        self._future_serial = 0
        self._future_generation = None
        self._future_cancel_event = None
        self._future_base_anchor = None
        self._future_observation_identity = None
        self._future_observation_wait = None

    def _clear_pregrasp_handoff(self) -> None:
        """Drop every watermark that could authorize a stale stage-two plan."""
        self._pregrasp_planning_identity = None
        self._pregrasp_dispatch_fence = None
        self._pregrasp_handoff = None
        self._pregrasp_stable_joint_sequence = None
        self._pregrasp_stable_joint_stamp_ns = None
        self._pregrasp_joint_error_rad = None
        self._approach_planning_anchor = None
        self._approach_execution_joint_fence = None

    def _invalidate_perception_session(self) -> None:
        """Invalidate task-owned perception before an ownership boundary."""
        MobileManipulationRuntime._clear_perception_authorization(self)
        self._required_perception_request_id = None
        self._required_perception_generation = None
        self._required_affordance_generation = None
        self._required_grounding_scope = None
        self._grounding_reset_pub.publish(Empty())

    def _clear_perception_authorization(self) -> None:
        """Revoke status-derived permission without changing the active request."""
        MobileManipulationRuntime._revoke_perception_success(self)
        self._bound_perception_request_id = None
        self._bound_perception_producer_epoch = None
        self._bound_perception_generation = None

    def _revoke_perception_success(self) -> None:
        """Clear the currently valid status while retaining its request binding."""
        self._perception_valid = False
        self._valid_seen_at = None
        self._valid_perception_request_id = None
        self._valid_perception_producer_epoch = None
        self._valid_perception_generation = None
        self._valid_observation_stamp_ns = None
        self._valid_observation_frame_id = ''

    def _reject_perception_generation_advance(self) -> None:
        """Cancel stale work if one exact producer/request changes generation."""
        self._revoke_perception_success()
        self._invalidate_async_work()
        if self._core.active:
            self._apply_safety(self._core.perception_invalid(
                'perception generation changed within the task-owned request',
            ))

    def _clear_semantic_observation_cache(self) -> None:
        """Drop geometry and semantics before requesting a different identity."""
        self._serial_gate = ObservationSerialGate(
            sync_slop_s=float(self.get_parameter('sync_slop_s').value),
            max_age_s=float(self.get_parameter('max_perception_age_s').value),
        )
        self._clear_perception_authorization()
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
        self._affordance = None
        self._affordance_placement_semantics = None
        self._affordance_generation = 0
        self._affordance_request_id = ''
        self._affordance_producer_epoch = ''

    def _reset_execution_occlusion(self) -> None:
        """Discard every prediction and its copied exact perception snapshot."""
        self._execution_occlusion.reset()
        self._execution_occlusion_target_piper = None
        self._execution_occlusion_target_cloud = None
        self._execution_occlusion_scene_cloud = None
        self._execution_occlusion_loss_detail = ''
        self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
            False,
            'execution occlusion is not armed',
        )

    def _cache_execution_occlusion_geometry(self) -> None:
        """Copy the last exact target and scene so raw callbacks cannot replace it."""
        if (
            self._target_piper is None
            or self._target_cloud is None
            or self._scene_cloud is None
        ):
            raise ValueError('exact near-contact geometry is incomplete')
        self._execution_occlusion_target_piper = self._target_piper.copy()
        self._execution_occlusion_target_cloud = self._target_cloud.copy()
        self._execution_occlusion_scene_cloud = self._scene_cloud.copy()

    def _arm_execution_occlusion(self, now: float) -> None:
        """Arm prediction after exact geometry and measured approach completion."""
        synchronized = self._serial_gate.snapshot(now)
        exact_authorized = self._grounding_observation_authorized(synchronized)
        if synchronized is None:
            raise ValueError('near-contact perception is not synchronized')
        if self._program is None:
            raise ValueError('near-contact motion program is unavailable')
        if not getattr(self, '_joint_history', None):
            raise ValueError('near-contact joint feedback is unavailable')
        joint_feedback = self._joint_history[-1]
        request_id = self._bound_perception_request_id or ''
        producer_epoch = self._bound_perception_producer_epoch or ''
        generation = self._bound_perception_generation or 0
        if self._valid_observation_stamp_ns is None:
            raise ValueError('near-contact observation stamp is unavailable')
        self._execution_occlusion.arm_near_contact(
            now_s=now,
            exact_authorized=exact_authorized,
            request_id=request_id,
            producer_epoch=producer_epoch,
            generation=generation,
            observation_serial=synchronized.serial,
            observation_stamp_ns=self._valid_observation_stamp_ns,
            observation_frame_id=self._valid_observation_frame_id,
            measured_joints=joint_feedback.positions,
            approach_endpoint_joints=self._program.approach.positions[-1],
            joint_seen_at_s=joint_feedback.received_at_s,
            joint_source_stamp_ns=joint_feedback.source_stamp_ns,
            joint_sequence=joint_feedback.sequence,
        )
        self._cache_execution_occlusion_geometry()

    def _execution_occlusion_decision(
        self,
        now: float,
        *,
        phase: RuntimePhase | None = None,
        lift_execution_completed: bool = False,
        allow_loss_watermark_sample: bool = False,
    ) -> ExecutionOcclusionDecision:
        """Evaluate prediction from current joint, gripper, and executor evidence."""
        if self._joint_history:
            joint_feedback = self._joint_history[-1]
        else:
            joint_feedback = _JointFeedback(
                float('-inf'),
                0,
                0,
                np.asarray((), dtype=float),
            )
        status = self._execution_status
        close_acknowledged = bool(
            status is not None
            and self._expected_gripper_command_id is not None
            and status.gripper_command_id == self._expected_gripper_command_id
            and status.accepted_gripper_aperture_m is not None
            and self._commanded_close_aperture is not None
            and abs(
                status.accepted_gripper_aperture_m
                - self._commanded_close_aperture
            ) <= 1e-5
        )
        lift_execution_active = bool(
            status is not None
            and status.trajectory == 'active'
            and status.owner == 'trajectory'
            and status.segment == 'lift'
            and status.command_id == self._core.expected_command_id
            and self._core.execution_segment == 'lift'
            and self._core.execution_seen_active
        )
        lift_path = (
            () if self._program is None else self._program.lift.positions
        )
        decision = self._execution_occlusion.evaluate(
            now_s=now,
            phase=self._core.phase if phase is None else phase,
            measured_joints=joint_feedback.positions,
            joint_seen_at_s=joint_feedback.received_at_s,
            joint_source_stamp_ns=joint_feedback.source_stamp_ns,
            joint_sequence=joint_feedback.sequence,
            close_command_sent_at_s=self._gripper_command_sent_s,
            close_acknowledged=close_acknowledged,
            execution_status_seen_at_s=self._execution_status_seen_s,
            lift_path=lift_path,
            lift_execution_active=lift_execution_active,
            lift_execution_completed=lift_execution_completed,
            allow_loss_watermark_sample=allow_loss_watermark_sample,
        )
        self._execution_occlusion_last_decision = decision
        return decision

    def _begin_execution_occlusion_loss(
        self,
        now: float,
        detail: str,
        *,
        phase: RuntimePhase | None = None,
        lift_execution_completed: bool = False,
    ) -> bool:
        """Start prediction only when the current execution evidence allows it."""
        if not getattr(self, '_joint_history', None):
            self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
                False,
                'joint feedback is unavailable at perception loss',
            )
            return False
        joint_feedback = self._joint_history[-1]
        try:
            self._execution_occlusion.mark_loss(
                now,
                joint_source_stamp_ns=joint_feedback.source_stamp_ns,
                joint_sequence=joint_feedback.sequence,
            )
        except (TypeError, ValueError) as error:
            self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
                False,
                str(error),
            )
            return False
        decision = self._execution_occlusion_decision(
            now,
            phase=phase,
            lift_execution_completed=lift_execution_completed,
            allow_loss_watermark_sample=True,
        )
        if not decision.allowed:
            return False
        self._execution_occlusion_loss_detail = str(detail)
        return True

    def _execution_occlusion_allows_loss(
        self,
        now: float,
        detail: str,
        *,
        phase: RuntimePhase | None = None,
        lift_execution_completed: bool = False,
    ) -> bool:
        """Continue an active prediction, or attempt to begin one exactly once."""
        if not self._execution_occlusion.loss_active:
            return self._begin_execution_occlusion_loss(
                now,
                detail,
                phase=phase,
                lift_execution_completed=lift_execution_completed,
            )
        decision = self._execution_occlusion_decision(
            now,
            phase=phase,
            lift_execution_completed=lift_execution_completed,
        )
        return decision.allowed

    def _retain_execution_occlusion_observation(
        self,
        synchronized: Any,
    ) -> bool:
        """Advance retained geometry only from the current exact live bundle."""
        try:
            if self._valid_observation_stamp_ns is None:
                raise ValueError('exact observation stamp is unavailable')
            target_piper = self._target_piper.copy()
            target_cloud = self._target_cloud.copy()
            scene_cloud = self._scene_cloud.copy()
            advanced = self._execution_occlusion.retain_exact_observation(
                request_id=self._bound_perception_request_id or '',
                producer_epoch=self._bound_perception_producer_epoch or '',
                generation=self._bound_perception_generation or 0,
                observation_serial=synchronized.serial,
                observation_stamp_ns=self._valid_observation_stamp_ns,
                observation_frame_id=self._valid_observation_frame_id,
            )
            if advanced:
                self._execution_occlusion_target_piper = target_piper
                self._execution_occlusion_target_cloud = target_cloud
                self._execution_occlusion_scene_cloud = scene_cloud
        except (AttributeError, TypeError, ValueError) as error:
            self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
                False,
                str(error),
            )
            return False
        self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
            True,
            '',
            mode='live_tracking',
        )
        return True

    def _restore_execution_occlusion_tracking(
        self,
        now: float,
        synchronized: Any | None = None,
    ) -> bool:
        """End prediction only after a newly exact-authorized live observation."""
        if not self._execution_occlusion.loss_active:
            return True
        observation = (
            self._serial_gate.snapshot(now)
            if synchronized is None
            else synchronized
        )
        if not self._grounding_observation_authorized(observation):
            return False
        try:
            if self._valid_observation_stamp_ns is None:
                raise ValueError('restored observation stamp is unavailable')
            target_piper = self._target_piper.copy()
            target_cloud = self._target_cloud.copy()
            scene_cloud = self._scene_cloud.copy()
            restored = self._execution_occlusion.tracking_restored(
                now,
                request_id=self._bound_perception_request_id or '',
                producer_epoch=self._bound_perception_producer_epoch or '',
                generation=self._bound_perception_generation or 0,
                observation_serial=observation.serial,
                observation_stamp_ns=self._valid_observation_stamp_ns,
                observation_frame_id=self._valid_observation_frame_id,
            )
        except (TypeError, ValueError) as error:
            self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
                False,
                str(error),
            )
            self._apply_safety(self._core.perception_invalid(
                f'execution occlusion recovery rejected: {error}',
            ))
            return False
        if not restored:
            self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
                False,
                'live perception did not advance beyond the loss watermark',
            )
            return False
        self._execution_occlusion_target_piper = target_piper
        self._execution_occlusion_target_cloud = target_cloud
        self._execution_occlusion_scene_cloud = scene_cloud
        self._execution_occlusion_loss_detail = ''
        self._execution_occlusion_last_decision = ExecutionOcclusionDecision(
            True,
            '',
            mode='live_tracking',
        )
        return True

    def _execution_perception_admitted(
        self,
        now: float,
        phase: RuntimePhase,
        *,
        lift_execution_completed: bool = False,
    ) -> bool:
        """Admit one execution step from exact live data or bounded prediction."""
        synchronized = self._serial_gate.snapshot(now)
        exact = self._grounding_observation_authorized(synchronized)
        if self._execution_occlusion.loss_active:
            if exact and self._restore_execution_occlusion_tracking(
                now,
                synchronized,
            ):
                return True
            return self._execution_occlusion_decision(
                now,
                phase=phase,
                lift_execution_completed=lift_execution_completed,
            ).allowed
        if exact:
            assert synchronized is not None
            return self._retain_execution_occlusion_observation(synchronized)
        return self._execution_occlusion_allows_loss(
            now,
            'execution perception is not synchronized and exact-authorized',
            phase=phase,
            lift_execution_completed=lift_execution_completed,
        )

    def _lift_completion_identity_matches(
        self,
        status: ExecutionState,
    ) -> bool:
        """Capture exact lift completion identity before the core clears it."""
        return bool(
            self._core.phase is RuntimePhase.LIFT
            and status.trajectory == 'succeeded'
            and status.owner == 'trajectory'
            and status.segment == 'lift'
            and status.command_id is not None
            and status.command_id == self._core.expected_command_id
            and self._core.execution_segment == 'lift'
            and self._core.execution_seen_active
        )

    def _planning_control(self, kind: str) -> PlanningControl:
        """Create one cooperatively cancellable wall-clock budget per request."""
        parameter = {
            'standoff': 'standoff_planning_budget_s',
            'pregrasp': 'grasp_planning_budget_s',
            'pregrasp_validation': 'grasp_planning_budget_s',
            'approach': 'approach_planning_budget_s',
            'carry': 'carry_planning_budget_s',
        }.get(kind)
        if parameter is None:
            raise ValueError(f'unsupported planning kind: {kind}')
        budget_s = float(self.get_parameter(parameter).value)
        if not math.isfinite(budget_s) or budget_s <= 0.0:
            raise ValueError(f'{parameter} must be finite and positive')
        cancel_event = threading.Event()
        self._future_cancel_event = cancel_event
        return PlanningControl(
            deadline_s=time.monotonic() + budget_s,
            cancel_check=cancel_event.is_set,
        )

    def _task_cancel_cb(self, msg: Bool) -> None:
        """Stop all task-owned motion and enter a terminal canceled state."""
        if not bool(msg.data):
            return
        with self._lock:
            MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
                self,
                False,
            )
            self._release_terminal_ownership()
            self._approach.reset()
            self._visual_search.reset()
            self._verifier.reset()
            self._pregrasp_program = None
            MobileManipulationRuntime._clear_pregrasp_handoff(self)
            self._program = None
            self._carry_program = None
            self._place_programs.clear()
            self._place_trajectory = None
            self._place_contract = None
            self._place_goal_id = ''
            self._place_planning_started_at = None
            self._place_planning_started_wall_s = None
            self._release_started_at = None
            self._carried_object_geometry = None
            self._carried_object_observation_stamp_ns = None
            MobileManipulationRuntime._reset_post_release_verification(self)
            self._desired_depth = None
            self._approximate_displacement = None
            self._work_pose = None
            self._work_pose_created_at_s = None
            self._navigation_status_seen_s = None
            self._navigation_status_goal_id = ''
            self._navigation_status_phase = ''
            self._navigation_goal_acknowledged = False
            self._navigation_history_recorded = False
            self._coarse_nav_perception_loss_detail = ''
            self._coarse_nav_arrival_started_at_s = None
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_last_odom_sequence = None
            self._coarse_nav_arrival_last_odom_stamp_ns = None
            if hasattr(self, '_work_pose_history_map'):
                self._work_pose_history_map.clear()
            else:
                self._work_pose_history_map = []
            self._closing_started_at = None
            self._verification_started_at = None
            self._commanded_close_aperture = None
            self._expected_gripper_command_id = None
            self._gripper_command_sent_s = None
            self._gripper_feedback.clear()
            self._trajectory_deadline_s = None
            self._coarse_nav_ready = False
            self._pose_settle_until = None
            self._pose_settle_started_at = None
            self._pose_settle_last_tick_at = None
            self._visual_search_settle_reference = None
            MobileManipulationRuntime._clear_visual_search_stationarity(
                self,
                'task_cancel',
            )
            MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
                self,
                'task_cancel',
            )
            self._lookout_pending = False
            self._visual_search_pending = False
            self._visual_search_edge_direction = 0
            self._visual_search_error_rad = None
            self._visual_search_reason = ''
            self._required_affordance_generation = None
            self._required_perception_generation = None
            self._reground_started_at = None
            self._reground_last_tick_at = None
            MobileManipulationRuntime._clear_near_view_settle(self)
            self._apply_safety(self._core.cancel())
            self._publish_status(force=True)

    def _topic_value(self, parameter: str) -> str:
        return str(self.get_parameter(parameter).value)

    def _reset_post_release_verification(self) -> None:
        """Drop every identity and terminal result from the preceding task."""
        self._place_observation_identity = None
        self._post_release_release_command_id = None
        self._post_release_pending_evidence = None
        self._post_release_verification_started_at_s = None
        self._post_release_verification_started_wall_s = None
        self._post_release_verification_last_tick_s = None
        self._post_release_verified_evidence = None

    def _place_trajectory_cb(self, msg: JointTrajectory) -> None:
        with self._lock:
            if self._core.phase is not RuntimePhase.PLACE_PLANNING:
                return
            self._place_trajectory = msg
            self._maybe_accept_place_plan()

    def _place_contract_cb(self, msg: String) -> None:
        with self._lock:
            if self._core.phase is not RuntimePhase.PLACE_PLANNING:
                return
            try:
                value = parse_place_contract(msg.data)
                self._place_contract = value
                self._maybe_accept_place_plan()
            except (TypeError, ValueError) as error:
                self._apply_safety(self._core.fail(
                    f'placement trajectory contract rejected: {error}',
                ))

    def _place_status_cb(self, msg: String) -> None:
        with self._lock:
            if self._core.phase not in {
                RuntimePhase.PLACE_PLANNING,
                RuntimePhase.PLACE_TRANSIT,
                RuntimePhase.PLACE_APPROACH,
                RuntimePhase.RELEASING,
                RuntimePhase.PLACE_RETREAT,
                RuntimePhase.POST_RELEASE_VERIFICATION,
            }:
                return
            try:
                failure = parse_terminal_place_status(msg.data)
            except PlaceTransactionProtocolError as error:
                self.get_logger().warning(
                    f'rejected malformed placement status: {error}',
                )
                return
            if failure is None:
                return
            if (
                failure.goal_id != self._place_goal_id
                or failure.executor_epoch != self._core.place_executor_epoch
            ):
                return
            self._apply_safety(self._core.fail(
                f'placement transaction failed: {failure.reason}',
            ))
            self._publish_status(force=True)

    def _post_release_verification_cb(self, msg: String) -> None:
        """Accept only terminal geometry evidence owned by this exact task."""
        with self._lock:
            if self._core.phase not in (
                RuntimePhase.PLACE_RETREAT,
                RuntimePhase.POST_RELEASE_VERIFICATION,
            ):
                return
            identity = self._place_observation_identity
            release_id = self._post_release_release_command_id
            if identity is None or release_id is None:
                self._apply_safety(self._core.fail(
                    'post-release verifier result has no active expectation',
                ))
                self._publish_status(force=True)
                return
            try:
                evidence = parse_post_release_verification(
                    msg.data,
                    expected=identity,
                    expected_release_gripper_command_id=release_id,
                    policy=self._post_release_verification_policy,
                )
            except PostReleaseVerificationError as error:
                self._apply_safety(self._core.fail(
                    f'post-release verification rejected: {error}',
                ))
                self._publish_status(force=True)
                return
            if not evidence.verified:
                self._apply_safety(self._core.fail(
                    'post-release verification failed: '
                    f'{evidence.failure}',
                ))
                self._publish_status(force=True)
                return
            if self._core.phase is RuntimePhase.PLACE_RETREAT:
                self._post_release_pending_evidence = evidence
            else:
                self._complete_post_release_verification(evidence)
            self._publish_status(force=True)

    def _begin_post_release_verification(self, now_s: float) -> None:
        """Start the bounded wait after measured retreat execution succeeds."""
        if (
            self._core.phase is not RuntimePhase.POST_RELEASE_VERIFICATION
            or self._place_observation_identity is None
            or self._post_release_release_command_id is None
            or not math.isfinite(now_s)
        ):
            self._apply_safety(self._core.fail(
                'post-release verification start contract is incomplete',
            ))
            return
        self._post_release_verification_started_at_s = now_s
        self._post_release_verification_last_tick_s = now_s
        self._post_release_verification_started_wall_s = time.monotonic()
        pending = self._post_release_pending_evidence
        self._post_release_pending_evidence = None
        if pending is not None:
            self._complete_post_release_verification(pending)

    def _complete_post_release_verification(
        self,
        evidence: PostReleaseVerificationEvidence,
    ) -> None:
        """Publish success only after retreat and exact observed-place proof."""
        if self._core.phase is not RuntimePhase.POST_RELEASE_VERIFICATION:
            self._apply_safety(self._core.fail(
                'post-release evidence arrived outside its completion phase',
            ))
            return
        self._post_release_verified_evidence = evidence
        self._post_release_verification_started_at_s = None
        self._post_release_verification_started_wall_s = None
        self._post_release_verification_last_tick_s = None
        self._core.post_release_verification_complete()
        if self._task.stage.value == 'execute_place':
            self._task.apply(StageResult.success(evidence.payload))
        self._apply_safety(SafetyAction())

    def _post_release_verification_tick(self, now_s: float) -> None:
        """Fail on ROS-time rollback or either finite timeout budget."""
        started = self._post_release_verification_started_at_s
        wall_started = self._post_release_verification_started_wall_s
        previous = self._post_release_verification_last_tick_s
        if (
            started is None
            or wall_started is None
            or previous is None
            or not math.isfinite(now_s)
            or not math.isfinite(started)
            or not math.isfinite(previous)
            or now_s < previous
        ):
            self._apply_safety(self._core.fail(
                'post-release verification clock contract is invalid',
            ))
            return
        self._post_release_verification_last_tick_s = now_s
        timeout = self._post_release_verification_policy.timeout_s
        if (
            now_s - started > timeout
            or time.monotonic() - wall_started
            > self._post_release_verification_policy.wall_timeout_s
        ):
            self._apply_safety(self._core.fail(
                'post-release geometry verification timed out',
            ))

    def _post_release_wall_timeout_tick(self) -> None:
        """Enforce the hard wait bound even while simulation time is paused."""
        with self._lock:
            if self._core.phase is not RuntimePhase.POST_RELEASE_VERIFICATION:
                return
            started = self._post_release_verification_started_wall_s
            now = time.monotonic()
            if (
                started is None
                or not math.isfinite(started)
                or now < started
            ):
                self._apply_safety(self._core.fail(
                    'post-release wall-clock contract is invalid',
                ))
                self._publish_status(force=True)
                return
            if (
                now - started
                > self._post_release_verification_policy.wall_timeout_s
            ):
                self._apply_safety(self._core.fail(
                    'post-release geometry verification wall timeout',
                ))
                self._publish_status(force=True)

    def _place_planning_wall_timeout_tick(self) -> None:
        """Bound placement planning while a paused simulator freezes ROS time."""
        with self._lock:
            if self._core.phase is not RuntimePhase.PLACE_PLANNING:
                return
            started = self._place_planning_started_wall_s
            now = time.monotonic()
            timeout = float(
                self.get_parameter('place_planning_wall_timeout_s').value,
            )
            if (
                started is None
                or not math.isfinite(started)
                or not math.isfinite(now)
                or now < started
                or not math.isfinite(timeout)
                or timeout <= 0.0
            ):
                self._apply_safety(self._core.fail(
                    'placement planning wall-clock contract is invalid',
                ))
                self._publish_status(force=True)
                return
            if now - started >= timeout:
                self._apply_safety(self._core.fail(
                    'placement planning wall timeout',
                ))
                self._publish_status(force=True)

    @staticmethod
    def _duration_s(value: Any) -> float:
        return float(value.sec) + float(value.nanosec) * 1e-9

    def _maybe_accept_place_plan(self) -> None:
        if self._place_trajectory is None or self._place_contract is None:
            return
        if not self._guard_active_posture(self._now_s()):
            return
        try:
            message = self._place_trajectory
            contract = self._place_contract
            identity = self._place_observation_identity
            if identity is None:
                raise ValueError('placement perception identity is unavailable')
            validate_place_trajectory_perception_identity(contract, identity)
            validate_place_trajectory_content(
                contract,
                message,
                expected_topic=self._place_trajectory_subscription.topic_name,
            )
            if int(contract.get('point_count', -1)) != len(message.points):
                raise ValueError('placement point count does not match its contract')
            if tuple(contract.get('joint_names', ())) != tuple(message.joint_names):
                raise ValueError('placement joint names do not match its contract')
            if tuple(message.joint_names) != tuple(self._planner.chain.joint_names):
                raise ValueError('placement trajectory does not match the active arm chain')
            positions = tuple(tuple(point.positions) for point in message.points)
            times = tuple(self._duration_s(point.time_from_start) for point in message.points)
            starts = contract.get('phase_start_indices')
            if not isinstance(starts, dict):
                raise ValueError('placement phase indices are missing')
            segments = split_placement_trajectory(positions, times, starts)
            self._place_programs = {
                segment.name: TimedJointTrajectory(
                    np.asarray(segment.positions, dtype=float),
                    np.asarray(segment.times_s, dtype=float),
                )
                for segment in segments
            }
            self._core.place_plan_ready(contract)
            if self._task.stage.value == 'plan_place':
                self._task.apply(StageResult.success(contract))
            self._publish_program_segment('place_transit')
        except (TypeError, ValueError) as error:
            self._apply_safety(self._core.fail(f'placement plan rejected: {error}'))

    def _valid_cb(self, msg: Bool) -> None:
        with self._lock:
            # Neither Bool value carries generation or frame identity. A stale
            # false can cross a newer Diagnostic status on DDS just as a stale
            # true can, so only the versioned status may mutate authorization.
            if bool(msg.data):
                return
            return

    def _target_cb(self, msg: Detection3D) -> None:
        with self._lock:
            try:
                piper_t_source = self._lookup_piper_from(msg.header.frame_id, msg.header.stamp)
                raw = np.array([
                    msg.bbox.center.position.x,
                    msg.bbox.center.position.y,
                    msg.bbox.center.position.z,
                ], dtype=float)
                if not np.all(np.isfinite(raw)) or raw[2] <= 0.0:
                    raise ValueError('target center is invalid in camera frame')
                self._target_camera = raw
                self._target_frame_id = msg.header.frame_id
                self._target_piper = _transform_points(piper_t_source, raw[None, :])[0]
                self._camera_origin_piper = piper_t_source[:3, 3].copy()
                self._camera_rotation_piper = piper_t_source[:3, :3].copy()
                self._target_stamp_ns = _stamp_ns(msg.header)
                self._target_stamp = float(self._target_stamp_ns) * 1e-9
                self._serial_gate.update('target', self._target_stamp)
            except (TransformException, ValueError, KeyError) as error:
                self._apply_safety(self._core.perception_invalid(
                    f'target transform failed: {error}',
                ))

    @staticmethod
    def _read_cloud(msg: PointCloud2, *, need_uv: bool) -> tuple[np.ndarray, np.ndarray | None]:
        available = {field.name for field in msg.fields}
        xyz_fields = ('x', 'y', 'z')
        if not set(xyz_fields).issubset(available):
            raise ValueError('PointCloud2 is missing x/y/z fields')
        if need_uv and not {'u', 'v'}.issubset(available):
            raise ValueError('target PointCloud2 is missing required u/v fields')
        fields = xyz_fields + (('u', 'v') if need_uv else ())
        values = np.asarray(point_cloud2.read_points(msg, field_names=fields, skip_nans=True))
        if values.dtype.names:
            columns = [np.asarray(values[name], dtype=float).reshape(-1) for name in fields]
            dense = np.column_stack(columns)
        else:
            dense = np.asarray(values.tolist(), dtype=float)
        if dense.ndim != 2 or dense.shape[1] != len(fields):
            raise ValueError('PointCloud2 conversion returned an invalid shape')
        if len(dense) == 0:
            raise ValueError('PointCloud2 contains no finite points')
        uv = dense[:, 3:5].copy() if len(fields) == 5 else None
        return dense[:, :3].copy(), uv

    def _target_cloud_cb(self, msg: PointCloud2) -> None:
        with self._lock:
            try:
                points, uv = self._read_cloud(msg, need_uv=True)
                piper_t_source = self._lookup_piper_from(msg.header.frame_id, msg.header.stamp)
                self._target_cloud = _transform_points(piper_t_source, points)
                self._target_uv = uv
                self._target_cloud_stamp_ns = _stamp_ns(msg.header)
                self._target_cloud_stamp = float(self._target_cloud_stamp_ns) * 1e-9
                self._target_cloud_frame_id = str(msg.header.frame_id)
                self._serial_gate.update('target_cloud', self._target_cloud_stamp)
            except (TransformException, ValueError, KeyError) as error:
                self._apply_safety(self._core.perception_invalid(f'target cloud failed: {error}'))

    def _scene_cloud_cb(self, msg: PointCloud2) -> None:
        with self._lock:
            try:
                points, _ = self._read_cloud(msg, need_uv=False)
                piper_t_source = self._lookup_piper_from(msg.header.frame_id, msg.header.stamp)
                self._scene_cloud = _transform_points(piper_t_source, points)
                self._scene_cloud_stamp_ns = _stamp_ns(msg.header)
                self._scene_cloud_stamp = float(self._scene_cloud_stamp_ns) * 1e-9
                self._scene_cloud_frame_id = str(msg.header.frame_id)
                self._serial_gate.update('scene_cloud', self._scene_cloud_stamp)
            except (TransformException, ValueError, KeyError) as error:
                self._apply_safety(self._core.perception_invalid(f'scene cloud failed: {error}'))

    def _affordance_cb(self, msg: String) -> None:
        with self._lock:
            try:
                value = json.loads(msg.data)
                if not isinstance(value, dict) or value.get('schema') != 'z_manip.affordance.v2':
                    return
                request_id = value.get('request_id')
                required_request_id = self._required_perception_request_id
                if (
                    not isinstance(request_id, str)
                    or required_request_id is None
                    or request_id != required_request_id
                ):
                    return
                grounding_scope = value.get('grounding_scope')
                required_scope = self._required_grounding_scope
                if (
                    not isinstance(grounding_scope, str)
                    or required_scope is None
                    or grounding_scope != required_scope
                ):
                    raise ValueError('affordance grounding scope does not match its request')
                producer_epoch = value.get('producer_epoch')
                if (
                    not isinstance(producer_epoch, str)
                    or not producer_epoch
                    or len(producer_epoch) > 128
                ):
                    raise ValueError('affordance ownership identity is invalid')
                generation = value.get('generation')
                if (
                    isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation <= 0
                ):
                    raise ValueError('affordance generation must be an integer')
                bound_epoch = self._bound_perception_producer_epoch
                if bound_epoch is not None and producer_epoch != bound_epoch:
                    return
                if (
                    request_id == self._affordance_request_id
                    and producer_epoch == self._affordance_producer_epoch
                    and generation < self._affordance_generation
                ):
                    return
                raw_verification = value.get('placement_verification')
                semantics = (
                    None
                    if raw_verification is None
                    else parse_placement_verification(raw_verification)
                )
                placement_region = value.get('placement_region')
                placement_avoids = value.get('placement_avoid_regions')
                if grounding_scope == 'grasp_for_place' and semantics is None:
                    raise ValueError(
                        'grasp_for_place lacks object verification semantics',
                    )
                if grounding_scope in {'grasp_only', 'grasp_for_place'} and (
                    placement_region is not None or placement_avoids != []
                ):
                    raise ValueError('grasp grounding returned placement geometry')
                if grounding_scope == 'grasp_only' and semantics is not None:
                    raise ValueError('grasp_only returned placement verification')
                if grounding_scope == 'place_support' and (
                    not isinstance(placement_region, dict)
                    or not isinstance(placement_avoids, list)
                    or value.get('grasp_part') is not None
                    or value.get('avoid_regions') != []
                    or value.get('preferred_approach_camera') is not None
                    or semantics is not None
                    or getattr(self, '_carried_object_geometry', None) is None
                ):
                    raise ValueError(
                        'place_support fields or frozen carried-object geometry are invalid',
                    )
                self._affordance = value
                self._affordance_placement_semantics = semantics
                self._affordance_generation = generation
                self._affordance_request_id = request_id
                self._affordance_producer_epoch = producer_epoch
            except json.JSONDecodeError:
                return
            except ValueError as error:
                self._apply_safety(self._core.perception_invalid(f'affordance invalid: {error}'))

    def _perception_status_cb(self, msg: DiagnosticArray) -> None:
        """Authorize only the task's exact request and one bridge producer epoch."""
        with self._lock:
            for status in msg.status:
                values = {item.key: item.value for item in status.values}
                if values.get('schema') != 'z_manip.perception_status.v1':
                    continue
                request_id = values.get('request_id', '').strip()
                required_request_id = self._required_perception_request_id
                # Ownership is checked before generation parsing. Unrelated
                # publishers therefore cannot poison any task-local floor.
                if required_request_id is None or request_id != required_request_id:
                    continue
                grounding_scope = values.get('grounding_scope', '').strip()
                required_scope = self._required_grounding_scope
                if required_scope is None or grounding_scope != required_scope:
                    self._apply_safety(self._core.perception_invalid(
                        'perception status grounding scope does not match its request',
                    ))
                    return
                try:
                    producer_epoch = values['producer_epoch'].strip()
                    generation = int(values['generation'])
                    if (
                        not producer_epoch
                        or len(producer_epoch) > 128
                        or generation <= 0
                    ):
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    self._apply_safety(self._core.perception_invalid(
                        'perception status ownership identity is invalid',
                    ))
                    return

                phase = str(status.message).strip()
                if phase not in {
                    'waiting_frame', 'grounding', 'waiting_tracker',
                    'tracking', 'failed',
                }:
                    self._apply_safety(self._core.perception_invalid(
                        'perception status phase is invalid',
                    ))
                    return

                if self._bound_perception_request_id is None:
                    self._bound_perception_request_id = request_id
                    self._bound_perception_producer_epoch = producer_epoch
                    self._bound_perception_generation = generation
                    self._required_perception_generation = generation
                    self._required_affordance_generation = generation
                elif (
                    request_id != self._bound_perception_request_id
                    or producer_epoch != self._bound_perception_producer_epoch
                ):
                    return
                elif generation != self._bound_perception_generation:
                    self._reject_perception_generation_advance()
                    return
                self._perception_generation = generation

                if (
                    self._execution_occlusion.armed
                    and not self._execution_occlusion.loss_active
                    and self._core.phase in (
                        RuntimePhase.CLOSING,
                        RuntimePhase.LIFT,
                        RuntimePhase.VERIFY,
                    )
                ):
                    retain_now = self._now_s()
                    retained = self._serial_gate.snapshot(retain_now)
                    if (
                        self._grounding_observation_authorized(retained)
                        and not self._retain_execution_occlusion_observation(
                            retained,
                        )
                    ):
                        self._apply_safety(self._core.perception_invalid(
                            'pre-loss exact observation retention failed: '
                            f'{self._execution_occlusion_last_decision.reason}',
                        ))
                        return
                self._revoke_perception_success()
                if phase == 'tracking':
                    try:
                        stamp_ns = int(values['observation_stamp_ns'])
                        frame_id = values['observation_frame_id'].strip()
                        if (
                            status.level != DiagnosticStatus.OK
                            or values.get('valid') != 'true'
                            or stamp_ns <= 0
                            or not frame_id
                        ):
                            raise ValueError
                    except (KeyError, TypeError, ValueError):
                        self._apply_safety(self._core.perception_invalid(
                            'valid perception status lacks exact observation identity',
                        ))
                        return
                    self._perception_valid = True
                    self._valid_seen_at = self._now_s()
                    self._valid_perception_request_id = request_id
                    self._valid_perception_producer_epoch = producer_epoch
                    self._valid_perception_generation = generation
                    self._valid_observation_stamp_ns = stamp_ns
                    self._valid_observation_frame_id = frame_id
                    if self._execution_occlusion.loss_active:
                        self._restore_execution_occlusion_tracking(
                            self._valid_seen_at,
                        )
                    return

                if phase != 'failed':
                    return
                failure = values.get('failure', '').strip() or 'unspecified_failure'
                failure_key = (request_id, producer_epoch, generation, failure)
                if failure_key == self._handled_perception_failure:
                    return
                if self._core.phase in (
                    RuntimePhase.IDLE,
                    RuntimePhase.PICK_COMPLETE,
                    RuntimePhase.COMPLETE,
                    RuntimePhase.CANCELED,
                    RuntimePhase.FAILED,
                ):
                    return
                self._handled_perception_failure = failure_key
                grounding_failures = {
                    'camera_frame_timeout',
                    'grounding_timeout',
                    'grounding_failed',
                }
                kind = (
                    FailureKind.NOT_FOUND
                    if self._core.phase is RuntimePhase.GROUNDING
                    and failure in grounding_failures
                    else FailureKind.TARGET_LOST
                )
                detail = f'perception generation {generation} failed: {failure}'
                if (
                    failure in _DEFERRED_COARSE_NAV_TRACKER_FAILURES
                    and MobileManipulationRuntime._has_frozen_coarse_nav_contract(
                        self,
                    )
                ):
                    self._coarse_nav_perception_loss_detail = detail
                    self.get_logger().warning(
                        f'{detail}; continuing the frozen map work pose and '
                        'requiring fresh semantic grounding after arrival',
                    )
                    return
                if self._begin_execution_occlusion_loss(self._now_s(), detail):
                    return
                if not self._recover_precontact(kind, detail):
                    self._apply_safety(self._core.perception_invalid(detail))
                return

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if msg.width > 0 and msg.height > 0:
            with self._lock:
                self._image_size = (int(msg.width), int(msg.height))

    def _joint_cb(self, msg: JointState) -> None:
        with self._lock:
            now = self._now_s()
            try:
                if (
                    len(msg.name) != len(msg.position)
                    or len(set(msg.name)) != len(msg.name)
                ):
                    raise ValueError('joint names and positions are not one-to-one')
                index = dict(zip(msg.name, msg.position))
                values = np.asarray([index[name] for name in self._planner.chain.joint_names])
                source_stamp_ns = _stamp_ns(msg.header)
                if source_stamp_ns <= 0:
                    raise ValueError('joint source stamp is zero')
            except (KeyError, TypeError, ValueError) as error:
                if self._core.active:
                    self._apply_safety(self._core.fail(
                        f'joint state source is invalid: {error}',
                    ))
                return
            if not np.all(np.isfinite(values)):
                if self._core.active:
                    self._apply_safety(self._core.fail(
                        'joint state contains a non-finite position',
                    ))
                return
            previous_stamp_ns = self._joint_stamp_ns
            if (
                previous_stamp_ns is not None
                and source_stamp_ns < previous_stamp_ns
                and self._core.active
            ):
                self._apply_safety(self._core.fail(
                    'joint state source time moved backwards',
                ))
                return
            if previous_stamp_ns == source_stamp_ns:
                if (
                    self._core.active
                    and self._joint_state is not None
                    and not np.array_equal(values, self._joint_state)
                ):
                    self._apply_safety(self._core.fail(
                        'joint state payload changed at the same source stamp',
                    ))
                return
            self._joint_sequence += 1
            self._joint_stamp_ns = source_stamp_ns
            self._joint_state = values
            self._joint_history.append(_JointFeedback(
                received_at_s=now,
                source_stamp_ns=source_stamp_ns,
                sequence=self._joint_sequence,
                positions=values.copy(),
            ))

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            now = self._now_s()
            try:
                validate_platform_odometry_frames(
                    msg.header.frame_id,
                    msg.child_frame_id,
                    expected_parent_frame=str(self.get_parameter(
                        'platform_odometry_parent_frame',
                    ).value),
                    expected_child_frame=str(self.get_parameter(
                        'platform_odometry_child_frame',
                    ).value),
                )
                source_stamp_ns = _stamp_ns(msg.header)
                if source_stamp_ns <= 0:
                    raise ValueError('platform odometry source stamp is zero')
                roll, pitch = _quaternion_roll_pitch(
                    msg.pose.pose.orientation,
                )
                yaw = _quaternion_yaw(msg.pose.pose.orientation)
                position = msg.pose.pose.position
                position_components = tuple(float(value) for value in (
                    position.x, position.y, position.z,
                ))
                if not all(math.isfinite(value) for value in position_components):
                    raise ValueError('platform odometry position is non-finite')
                position_xy = position_components[:2]
                orientation_components = _normalized_quaternion(
                    msg.pose.pose.orientation,
                )
                linear = msg.twist.twist.linear
                angular = msg.twist.twist.angular
                linear_components = tuple(float(value) for value in (
                    linear.x, linear.y, linear.z,
                ))
                angular_components = tuple(float(value) for value in (
                    angular.x, angular.y, angular.z,
                ))
                base_linear_speed, base_angular_speed = (
                    base_twist_speed_magnitudes(
                        linear_components,
                        angular_components,
                    )
                )
                navigation_components = (
                    linear_components[0],
                    linear_components[1],
                )
                if not all(math.isfinite(value) for value in navigation_components):
                    raise ValueError('platform odometry navigation twist is non-finite')
                nav_speed = math.sqrt(sum(
                    value * value for value in navigation_components
                ))
                payload = (
                    *position_components,
                    *orientation_components,
                    *linear_components,
                    *angular_components,
                )
            except (TypeError, ValueError) as error:
                self._roll = float('nan')
                self._pitch = float('nan')
                self._yaw = None
                self._position_xy = None
                self._base_linear_speed_mps = None
                self._base_angular_speed_rps = None
                self._base_yaw_rate_rps = None
                self._nav_speed = float('inf')
                self._posture_guard.update(
                    self._roll,
                    self._pitch,
                    seen_at_s=now,
                )
                if self._core.active:
                    self._fail_posture(str(error))
                return
            previous_stamp_ns = self._odom_stamp_ns
            if previous_stamp_ns is not None and source_stamp_ns < previous_stamp_ns:
                if self._core.active:
                    self._fail_posture(
                        'platform odometry source time moved backwards',
                    )
                return
            if previous_stamp_ns == source_stamp_ns:
                if (
                    self._core.active
                    and self._odom_payload is not None
                    and payload != self._odom_payload
                ):
                    self._fail_posture(
                        'platform odometry payload changed at the same source stamp',
                    )
                return
            self._roll, self._pitch = roll, pitch
            self._yaw = yaw
            self._position_xy = position_xy
            self._odom_sequence += 1
            self._odom_stamp_ns = source_stamp_ns
            self._odom_payload = payload
            self._odom_seen_at = now
            MobileManipulationRuntime._record_navigation_attempt_if_moved(self)
            self._posture_guard.update(self._roll, self._pitch, seen_at_s=now)
            self._base_linear_speed_mps = base_linear_speed
            self._base_angular_speed_rps = base_angular_speed
            self._base_yaw_rate_rps = abs(angular_components[2])
            self._nav_speed = nav_speed
            MobileManipulationRuntime._record_coarse_nav_arrival_motion_sample(
                self,
                received_at_s=now,
                odom_sequence=self._odom_sequence,
                odom_stamp_ns=source_stamp_ns,
                linear_speed_mps=nav_speed,
                angular_speed_rps=abs(angular_components[2]),
                position_xy=position_xy,
                yaw_rad=yaw,
            )
            try:
                MobileManipulationRuntime._record_visual_search_settle_motion_sample(
                    self,
                    received_at_s=now,
                    odom_sequence=self._odom_sequence,
                    odom_stamp_ns=source_stamp_ns,
                    linear_speed_mps=base_linear_speed,
                    angular_speed_rps=base_angular_speed,
                )
            except ValueError as error:
                if self._core.active:
                    MobileManipulationRuntime._fail_pose_settle(
                        self,
                        f'visual search quiet-window odometry failed: {error}',
                    )
                return
            try:
                MobileManipulationRuntime._record_visual_servo_vertical_motion_sample(
                    self,
                    received_at_s=now,
                    odom_sequence=self._odom_sequence,
                    odom_stamp_ns=source_stamp_ns,
                    linear_speed_mps=base_linear_speed,
                    angular_speed_rps=base_angular_speed,
                )
            except ValueError as error:
                if self._core.phase is RuntimePhase.VISUAL_SERVO:
                    detail = (
                        'visual-servo vertical recovery odometry failed: '
                        f'{error}'
                    )
                    MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
                        self,
                        'invalid_odometry',
                    )
                    if not self._recover_precontact(
                        FailureKind.VISUAL_APPROACH_FAILED,
                        detail,
                    ):
                        self._apply_safety(self._core.fail(detail))
                return
            if self._guard_active_posture(now):
                self._dispatch_pending_lookout(now)

    def _coarse_ready_cb(self, msg: Bool) -> None:
        with self._lock:
            # The Bool channel is retained for old navigation adapters only.
            # Explicit work poses require the correlated String status below.
            if self._work_pose is None:
                self._coarse_nav_ready = bool(msg.data)

    def _record_navigation_attempt_if_moved(self) -> None:
        """Blacklist a work pose only after ACK plus measured base motion."""
        if (
            getattr(self, '_work_pose', None) is None
            or not getattr(self, '_navigation_goal_acknowledged', False)
            or getattr(self, '_navigation_history_recorded', False)
            or self._position_xy is None
            or self._odom_stamp_ns is None
            or self._navigation_ack_position_xy is None
            or self._navigation_ack_odom_sequence is None
            or self._navigation_ack_odom_stamp_ns is None
        ):
            return
        if (
            self._odom_sequence <= self._navigation_ack_odom_sequence
            or self._odom_stamp_ns <= self._navigation_ack_odom_stamp_ns
        ):
            return
        displacement = float(np.linalg.norm(
            np.asarray(self._position_xy, dtype=float)
            - np.asarray(self._navigation_ack_position_xy, dtype=float),
        ))
        minimum = float(self.get_parameter(
            'work_pose_history_min_displacement_m',
        ).value)
        if not math.isfinite(displacement) or displacement < minimum:
            return
        goal = np.array((
            *np.asarray(self._work_pose['map_goal_xy'], dtype=float),
            float(self._work_pose['map_goal_yaw_rad']),
        ))
        self._work_pose_history_map.append(goal)
        del self._work_pose_history_map[:-32]
        self._navigation_history_recorded = True

    def _validate_navigation_ready_pose(self, now: float) -> None:
        """Independently verify fresh measured XY before accepting READY."""
        if (
            self._work_pose is None
            or self._position_xy is None
            or self._odom_seen_at is None
            or self._odom_stamp_ns is None
        ):
            raise ValueError('navigation READY lacks platform odometry')
        age = now - self._odom_seen_at
        max_age = float(self.get_parameter('work_pose_odom_max_age_s').value)
        if not math.isfinite(age) or age < 0.0 or age > max_age:
            raise ValueError(f'navigation READY odometry is stale: age={age:.3f}s')
        source_age = now - float(self._odom_stamp_ns) * 1e-9
        if (
            not math.isfinite(source_age)
            or source_age < 0.0
            or source_age > max_age
        ):
            raise ValueError(
                'navigation READY odometry source stamp is stale: '
                f'age={source_age:.3f}s',
            )
        source = self._work_pose.get('source')
        if not isinstance(source, dict):
            raise ValueError('navigation READY work-pose source is unavailable')
        try:
            minimum_sequence = int(source['odom_sequence'])
            minimum_stamp_ns = int(source['odom_stamp_ns'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                'navigation READY work-pose odometry identity is invalid',
            ) from error
        if (
            self._odom_sequence <= minimum_sequence
            or self._odom_stamp_ns <= minimum_stamp_ns
        ):
            raise ValueError(
                'navigation READY has no post-work-pose odometry sample',
            )
        if (
            not self._navigation_goal_acknowledged
            or self._navigation_ack_odom_sequence is None
            or self._navigation_ack_odom_stamp_ns is None
        ):
            raise ValueError(
                'navigation READY lacks a waypoint dispatch acknowledgement',
            )
        if (
            self._odom_sequence <= self._navigation_ack_odom_sequence
            or self._odom_stamp_ns <= self._navigation_ack_odom_stamp_ns
        ):
            raise ValueError(
                'navigation READY has no post-dispatch odometry sample',
            )
        expected = np.asarray(self._work_pose['map_goal_xy'], dtype=float)
        error = float(np.linalg.norm(
            np.asarray(self._position_xy, dtype=float) - expected,
        ))
        tolerance = float(self.get_parameter('work_pose_goal_tolerance_m').value)
        if not math.isfinite(error) or error > tolerance:
            raise ValueError(
                'navigation READY is outside the work-pose XY tolerance: '
                f'error={error:.3f}m, tolerance={tolerance:.3f}m',
            )

    def _coarse_navigation_status_alive(self, now: float) -> bool:
        """Bound discovery loss and a silent navigation adapter by goal ID."""
        if self._work_pose is None:
            return True
        created = self._work_pose_created_at_s
        seen = self._navigation_status_seen_s
        reference = seen if seen is not None else created
        timeout = float(self.get_parameter(
            'coarse_navigation_status_timeout_s',
        ).value)
        if (
            reference is not None
            and math.isfinite(reference)
            and math.isfinite(now)
            and 0.0 <= now - reference <= timeout
        ):
            return True
        detail = (
            'correlated coarse-navigation status timed out for '
            f'goal {self._work_pose["goal_id"]}'
        )
        if not self._recover_precontact(FailureKind.NAV_BLOCKED, detail):
            self._apply_safety(self._core.fail(detail))
        return False

    def _has_frozen_coarse_nav_contract(self) -> bool:
        """Return whether navigation owns one immutable observed map goal."""
        if (
            self._core.phase is not RuntimePhase.COARSE_NAV
            or getattr(
                getattr(getattr(self, '_task', None), 'stage', None),
                'value',
                '',
            ) != 'coarse_nav'
            or not isinstance(self._work_pose, dict)
        ):
            return False
        source = self._work_pose.get('source')
        prospective_serial = self._core.prospective_serial
        try:
            goal_id = self._work_pose['goal_id']
            map_frame = self._work_pose['map_frame']
            map_goal_xy = np.asarray(self._work_pose['map_goal_xy'], dtype=float)
            map_goal_yaw = float(self._work_pose['map_goal_yaw_rad'])
            source_serial = int(source['observation_serial'])
            source_generation = int(source['generation'])
            source_request_id = source['request_id']
            source_producer_epoch = source['producer_epoch']
        except (KeyError, TypeError, ValueError):
            return False
        bound_request_id = getattr(self, '_bound_perception_request_id', None)
        required_request_id = getattr(self, '_required_perception_request_id', None)
        bound_producer_epoch = getattr(
            self,
            '_bound_perception_producer_epoch',
            None,
        )
        bound_generation = getattr(self, '_bound_perception_generation', None)
        required_generation = getattr(self, '_required_perception_generation', None)
        return bool(
            isinstance(goal_id, str)
            and bool(goal_id)
            and isinstance(map_frame, str)
            and bool(map_frame)
            and map_goal_xy.shape == (2,)
            and np.all(np.isfinite(map_goal_xy))
            and math.isfinite(map_goal_yaw)
            and prospective_serial is not None
            and source_serial > 0
            and source_serial == prospective_serial
            and isinstance(source_request_id, str)
            and bool(source_request_id)
            and source_request_id == bound_request_id
            and source_request_id == required_request_id
            and isinstance(source_producer_epoch, str)
            and bool(source_producer_epoch)
            and source_producer_epoch == bound_producer_epoch
            and source_generation > 0
            and source_generation == bound_generation
            and source_generation == required_generation
        )

    def _publish_frozen_coarse_nav_authorization(self, active: bool) -> None:
        """Publish or revoke one exact task-owned frozen-goal heartbeat."""
        publisher = getattr(
            self,
            '_frozen_coarse_nav_authorization_pub',
            None,
        )
        if publisher is None:
            # Lightweight unit harnesses do not construct ROS publishers.
            return
        if active:
            if not MobileManipulationRuntime._has_frozen_coarse_nav_contract(self):
                raise RuntimeError(
                    'cannot authorize perception loss without a frozen nav contract',
                )
            source = self._work_pose['source']
            identity: dict[str, object] = {
                'schema': 'z_manip.frozen_coarse_nav_authorization.v1',
                'request_id': str(source['request_id']),
                'producer_epoch': str(source['producer_epoch']),
                'generation': int(source['generation']),
                'observation_serial': int(source['observation_serial']),
                'nav_goal_id': str(self._work_pose['goal_id']),
            }
            previous = self._frozen_coarse_nav_authorization_identity
            if previous is not None and previous != identity:
                publisher.publish(String(data=json.dumps(
                    {**previous, 'active': False},
                    separators=(',', ':'),
                )))
            self._frozen_coarse_nav_authorization_identity = identity
            publisher.publish(String(data=json.dumps(
                {**identity, 'active': True},
                separators=(',', ':'),
            )))
            return
        identity = self._frozen_coarse_nav_authorization_identity
        self._frozen_coarse_nav_authorization_identity = None
        if identity is not None:
            publisher.publish(String(data=json.dumps(
                {**identity, 'active': False},
                separators=(',', ':'),
            )))

    def _frozen_coarse_nav_authorization_tick(self) -> None:
        """Keep the fail-closed perception-loss handoff alive at >=10 Hz."""
        with self._lock:
            MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
                self,
                MobileManipulationRuntime._has_frozen_coarse_nav_contract(self),
            )

    def _navigation_status_cb(self, msg: String) -> None:
        """Accept navigation completion only for the active immutable goal ID."""
        with self._lock:
            if self._work_pose is None or self._core.phase is not RuntimePhase.COARSE_NAV:
                return
            try:
                value = json.loads(msg.data)
                if value.get('schema') != 'z_manip.navigation_status.v1':
                    raise ValueError('unsupported navigation status schema')
                goal_id = str(value.get('goal_id') or '')
                expected_id = str(self._work_pose['goal_id'])
                if goal_id != expected_id or str(value.get('task_key') or '') != expected_id:
                    return
                goal_xy = np.asarray(value.get('map_goal_xy'), dtype=float)
                expected_xy = np.asarray(self._work_pose['map_goal_xy'], dtype=float)
                if (
                    goal_xy.shape != (2,)
                    or not np.all(np.isfinite(goal_xy))
                    or not np.allclose(goal_xy, expected_xy, atol=1e-6, rtol=0.0)
                ):
                    raise ValueError('navigation status map goal differs from active work pose')
                if value.get('work_pose_source') != self._work_pose['source']:
                    raise ValueError('navigation status source differs from active work pose')
                if str(value.get('map_frame') or '') != str(self._work_pose['map_frame']):
                    raise ValueError('navigation status map frame differs from active work pose')
                if value.get('coarse_goal_check') != 'xy_only':
                    raise ValueError('navigation status must declare the coarse XY contract')
                phase = str(value.get('phase') or '')
                if phase not in {
                    'wait_observation', 'navigating', 'ready', 'failed',
                }:
                    raise ValueError('navigation status phase is invalid')
                reset_acknowledged = value.get('goal_reset_acknowledged')
                if not isinstance(reset_acknowledged, bool):
                    raise ValueError(
                        'navigation status reset acknowledgement is invalid',
                    )
                now = self._now_s()
                self._navigation_status_seen_s = now
                self._navigation_status_goal_id = goal_id
                self._navigation_status_phase = phase
                if (
                    phase in {'navigating', 'ready'}
                    and reset_acknowledged
                    and not self._navigation_goal_acknowledged
                    and self._position_xy is not None
                    and self._odom_stamp_ns is not None
                ):
                    self._navigation_goal_acknowledged = True
                    self._navigation_ack_position_xy = tuple(
                        float(value) for value in self._position_xy
                    )
                    self._navigation_ack_odom_sequence = int(self._odom_sequence)
                    self._navigation_ack_odom_stamp_ns = int(self._odom_stamp_ns)
                if self._navigation_goal_acknowledged:
                    MobileManipulationRuntime._record_navigation_attempt_if_moved(self)
                if phase == 'failed':
                    detail = str(value.get('reason') or 'coarse navigation failed')
                    if not self._recover_precontact(FailureKind.NAV_BLOCKED, detail):
                        self._apply_safety(self._core.fail(detail))
                    self._publish_status()
                    return
                if phase == 'ready':
                    try:
                        self._validate_navigation_ready_pose(now)
                    except ValueError as error:
                        if 'no post-dispatch odometry sample' in str(error):
                            self._coarse_nav_ready = False
                            self._coarse_nav_arrival_started_at_s = None
                            self._coarse_nav_arrival_stable_since_s = None
                            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
                            self._coarse_nav_arrival_last_odom_sequence = None
                            self._coarse_nav_arrival_last_odom_stamp_ns = None
                            return
                        detail = f'correlated navigation READY rejected: {error}'
                        if not self._recover_precontact(
                            FailureKind.NAV_BLOCKED,
                            detail,
                        ):
                            self._apply_safety(self._core.fail(detail))
                        self._publish_status()
                        return
                    self._coarse_nav_ready = True
                else:
                    self._coarse_nav_ready = False
                    self._coarse_nav_arrival_started_at_s = None
                    self._coarse_nav_arrival_stable_since_s = None
                    self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
                    self._coarse_nav_arrival_last_odom_sequence = None
                    self._coarse_nav_arrival_last_odom_stamp_ns = None
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                self.get_logger().error(f'navigation status rejected: {error}')

    def _begin_pregrasp_reobserve(
        self,
        status: ExecutionState,
        now: float,
    ) -> None:
        """Freeze executor high-water and wait for measured pregrasp arrival."""
        program = self._pregrasp_program
        identity = self._pregrasp_planning_identity
        received_at = status.trajectory_received_at
        timeout = float(self.get_parameter('pregrasp_reobserve_timeout_s').value)
        completion_source_stamp_ns = int(round(float(now) * 1e9))
        if (
            program is None
            or identity is None
            or status.command_id is None
            or status.command_id <= 0
            or not status.executor_epoch
            or received_at is None
            or not math.isfinite(received_at)
            or received_at < 0.0
            or not math.isfinite(float(now))
            or completion_source_stamp_ns <= 0
            or not math.isfinite(timeout)
            or timeout <= 0.0
            or self._joint_stamp_ns is None
            or self._joint_stamp_ns <= 0
        ):
            raise ValueError(
                'pregrasp completion lacks executor, planner, or joint identity',
            )
        endpoint = np.asarray(program.transit.positions[-1], dtype=float).copy()
        if (
            endpoint.shape != (self._planner.chain.dof,)
            or not np.all(np.isfinite(endpoint))
        ):
            raise ValueError('pregrasp endpoint is invalid')
        endpoint.setflags(write=False)
        self._pregrasp_handoff = _PregraspHandoff(
            observation_serial=program.observation_serial,
            observation_identity=identity,
            endpoint_joints=endpoint,
            executor_epoch=status.executor_epoch,
            command_id=status.command_id,
            trajectory_received_at=received_at,
            completed_at_s=float(now),
            completion_source_stamp_ns=completion_source_stamp_ns,
            deadline_s=float(now) + timeout,
            minimum_joint_sequence=int(self._joint_sequence),
            minimum_joint_stamp_ns=int(self._joint_stamp_ns),
        )
        self._pregrasp_stable_joint_sequence = None
        self._pregrasp_stable_joint_stamp_ns = None
        self._pregrasp_joint_error_rad = None
        self._approach_planning_anchor = None
        self._publish_zero()

    def _execution_cb(self, msg: String) -> None:
        with self._lock:
            if (
                self._core.active
                and not self._guard_active_posture(self._now_s())
            ):
                return
            try:
                status = parse_execution_status(msg.data)
            except ValueError as error:
                if self._core.phase is RuntimePhase.CANCELED:
                    return
                self._apply_safety(self._core.fail(f'execution status invalid: {error}'))
                return
            lift_completion_evidence = (
                MobileManipulationRuntime._lift_completion_identity_matches(
                    self,
                    status,
                )
            )
            self._execution_status = status
            seen_at = self._now_s()
            self._execution_status_seen_s = seen_at
            if status.gripper_command_id is not None:
                self._latest_gripper_command_id = max(
                    self._latest_gripper_command_id,
                    status.gripper_command_id,
                )
                if status.aperture_m is not None:
                    self._gripper_feedback.append((
                        seen_at,
                        status.gripper_command_id,
                        status.aperture_m,
                    ))
            previous = self._core.phase
            self._apply_safety(self._core.execution_update(status))
            if previous is RuntimePhase.CANCELED:
                return
            if self._core.phase is RuntimePhase.FAILED:
                return
            if previous is not self._core.phase:
                self._trajectory_deadline_s = None
            if (
                previous is RuntimePhase.TRANSIT
                and self._core.phase is RuntimePhase.PREGRASP_REOBSERVE
            ):
                try:
                    self._begin_pregrasp_reobserve(status, seen_at)
                except (TypeError, ValueError) as error:
                    self._apply_safety(self._core.fail(
                        f'pregrasp handoff rejected: {error}',
                    ))
                    return
            elif previous is RuntimePhase.APPROACH and self._core.phase is RuntimePhase.CLOSING:
                try:
                    self._arm_execution_occlusion(seen_at)
                    self._commanded_close_aperture = self._grasp_close_aperture()
                except ValueError as error:
                    self._apply_safety(self._core.fail(
                        f'near-contact execution rejected: {error}',
                    ))
                    return
                self._expected_gripper_command_id = self._latest_gripper_command_id + 1
                self._gripper_command_sent_s = self._now_s()
                self._gripper_feedback.clear()
                self._gripper_pub.publish(Float32(
                    data=self._commanded_close_aperture,
                ))
                self._closing_started_at = self._now_s()
            elif previous is RuntimePhase.LIFT and self._core.phase is RuntimePhase.VERIFY:
                if not self._execution_perception_admitted(
                    seen_at,
                    RuntimePhase.LIFT,
                    lift_execution_completed=lift_completion_evidence,
                ):
                    reason = self._execution_occlusion_last_decision.reason
                    self._apply_safety(self._core.fail(
                        'lift completion perception rejected: '
                        f'{reason or "execution evidence is unavailable"}',
                    ))
                    return
                try:
                    self._execution_occlusion.note_lift_completed(seen_at)
                except (TypeError, ValueError) as error:
                    self._apply_safety(self._core.fail(
                        f'lift completion evidence rejected: {error}',
                    ))
                    return
                self._verification_started_at = self._now_s()
                if self._task.stage.value == 'execute_grasp':
                    self._task.apply(StageResult.success())
            elif (
                previous is RuntimePhase.CARRY
                and self._core.phase in (
                    RuntimePhase.PICK_COMPLETE,
                    RuntimePhase.PLACE_GROUNDING,
                )
                and self._task.stage.value == 'carry'
            ):
                self._task.apply(StageResult.success())
                if self._core.phase is RuntimePhase.PLACE_GROUNDING:
                    self._request_semantic_reground(self._now_s())
            elif (
                previous is RuntimePhase.PLACE_TRANSIT
                and self._core.phase is RuntimePhase.PLACE_APPROACH
            ):
                self._publish_program_segment('place_approach')
            elif (
                previous is RuntimePhase.PLACE_APPROACH
                and self._core.phase is RuntimePhase.RELEASING
            ):
                self._start_release(self._now_s())
            elif (
                previous is RuntimePhase.PLACE_RETREAT
                and self._core.phase is RuntimePhase.POST_RELEASE_VERIFICATION
            ):
                self._begin_post_release_verification(seen_at)

    def _semantic_observation(self, serial: int, stamp_s: float) -> PerceptionObservation:
        required = (
            self._target_cloud,
            self._scene_cloud,
            self._target_camera,
            self._camera_origin_piper,
            self._camera_rotation_piper,
            self._joint_state,
            self._affordance,
            self._image_size,
        )
        if any(value is None for value in required):
            raise ValueError('synchronized observation is incomplete')
        assert self._target_cloud is not None
        assert self._image_size is not None
        assert self._affordance is not None
        selection = select_semantic_target_points(
            self._target_cloud,
            self._target_uv,
            self._affordance,
            image_width=self._image_size[0],
            image_height=self._image_size[1],
            min_points=int(self.get_parameter('semantic_min_points').value),
        )
        collision_target = filter_object_cloud(
            self._target_cloud,
            viewpoint=self._camera_origin_piper,
            min_points=int(self.get_parameter('semantic_min_points').value),
        )
        self._selection_mode = selection.mode
        affordance = copy.deepcopy(self._affordance)
        direction = affordance.get('preferred_approach_camera')
        if direction is not None:
            vector = np.asarray(direction, dtype=float)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError('preferred VLM approach is invalid')
            assert self._camera_rotation_piper is not None
            affordance['preferred_approach'] = (
                self._camera_rotation_piper @ vector
            ).tolist()
        return PerceptionObservation(
            serial=serial,
            stamp_s=stamp_s,
            target_points=selection.points.copy(),
            target_collision_points=collision_target.copy(),
            scene_points=self._scene_cloud.copy(),
            target_position_camera=self._target_camera.copy(),
            camera_origin_piper=self._camera_origin_piper.copy(),
            camera_rotation_piper=self._camera_rotation_piper.copy(),
            affordance=affordance,
        )

    def _capture_planning_observation_identity(
        self,
        observation: PerceptionObservation,
    ) -> _PlanningObservationIdentity:
        """Freeze the source identity and target used by a work-pose job."""
        request_id = self._bound_perception_request_id
        producer_epoch = self._bound_perception_producer_epoch
        generation = self._bound_perception_generation
        stamp_ns = self._valid_observation_stamp_ns
        frame_id = self._valid_observation_frame_id
        target = np.asarray(observation.target_position_camera, dtype=float)
        if (
            request_id is None
            or producer_epoch is None
            or generation is None
            or stamp_ns is None
            or stamp_ns <= 0
            or not frame_id
            or target.shape != (3,)
            or not np.all(np.isfinite(target))
        ):
            raise ValueError('work-pose observation identity is incomplete')
        return _PlanningObservationIdentity(
            request_id=request_id,
            producer_epoch=producer_epoch,
            generation=generation,
            stamp_ns=stamp_ns,
            frame_id=frame_id,
            target_position_camera=tuple(float(value) for value in target),
        )

    def _validate_completed_work_pose_observation(
        self,
        identity: _PlanningObservationIdentity,
    ) -> None:
        """Distinguish an immutable identity change from an incomplete live bundle."""
        ownership = (
            self._bound_perception_request_id == identity.request_id
            and self._bound_perception_producer_epoch == identity.producer_epoch
            and self._bound_perception_generation == identity.generation
            and self._required_perception_request_id == identity.request_id
            and self._required_perception_generation == identity.generation
            and self._required_affordance_generation == identity.generation
            and self._affordance_request_id == identity.request_id
            and self._affordance_producer_epoch == identity.producer_epoch
            and self._affordance_generation == identity.generation
        )
        if not ownership:
            raise _PlanningObservationChanged(
                'work-pose perception ownership or generation changed while planning',
            )

        valid_identity = (
            self._valid_perception_request_id,
            self._valid_perception_producer_epoch,
            self._valid_perception_generation,
        )
        if all(value is not None for value in valid_identity) and valid_identity != (
            identity.request_id,
            identity.producer_epoch,
            identity.generation,
        ):
            raise _PlanningObservationChanged(
                'work-pose valid perception identity changed while planning',
            )
        current_frame = self._valid_observation_frame_id
        if current_frame and current_frame != identity.frame_id:
            raise _PlanningObservationChanged(
                'work-pose observation frame changed while planning',
            )
        geometry_frames = (
            self._target_frame_id,
            self._target_cloud_frame_id,
            self._scene_cloud_frame_id,
        )
        if any(frame and frame != identity.frame_id for frame in geometry_frames):
            raise _PlanningObservationChanged(
                'work-pose geometry frame changed while planning',
            )

        current_stamp_ns = self._valid_observation_stamp_ns
        if current_stamp_ns is not None and current_stamp_ns < identity.stamp_ns:
            raise _PlanningObservationChanged(
                'work-pose observation time moved backwards while planning',
            )
        synchronized = self._serial_gate.snapshot(self._now_s())
        if not self._grounding_observation_authorized(synchronized):
            raise _PlanningObservationPending(
                'work-pose result is waiting for a fresh exact-authorized observation',
            )
        current_target = self._target_camera
        if (
            self._valid_perception_request_id != identity.request_id
            or self._valid_perception_producer_epoch != identity.producer_epoch
            or self._valid_perception_generation != identity.generation
            or current_stamp_ns is None
            or current_stamp_ns < identity.stamp_ns
            or self._valid_observation_frame_id != identity.frame_id
            or current_target is None
        ):
            raise _PlanningObservationChanged(
                'work-pose perception ownership or frame changed while planning',
            )
        current = np.asarray(current_target, dtype=float)
        frozen = np.asarray(identity.target_position_camera, dtype=float)
        drift = float(np.linalg.norm(current - frozen))
        tolerance = float(self.get_parameter(
            'work_pose_target_drift_tolerance_m',
        ).value)
        if not math.isfinite(drift) or drift > tolerance:
            raise _PlanningObservationChanged(
                'target moved while work-pose planning was active: '
                f'drift={drift:.3f}m, tolerance={tolerance:.3f}m',
            )

    def _validate_grasp_planning_observation(
        self,
        identity: _PlanningObservationIdentity,
        *,
        target_geometry: _TargetGeometrySignature | None = None,
    ) -> None:
        """Require the same semantic owner and a live target near the plan input."""
        ownership = (
            self._bound_perception_request_id == identity.request_id
            and self._bound_perception_producer_epoch == identity.producer_epoch
            and self._bound_perception_generation == identity.generation
            and self._required_perception_request_id == identity.request_id
            and self._affordance_request_id == identity.request_id
            and self._affordance_producer_epoch == identity.producer_epoch
            and self._affordance_generation == identity.generation
        )
        if not ownership:
            raise _PlanningObservationChanged(
                'grasp-planning perception ownership changed',
            )
        synchronized = self._serial_gate.snapshot(self._now_s())
        if not self._grounding_observation_authorized(synchronized):
            raise _PlanningObservationPending(
                'grasp plan is waiting for an exact-authorized live bundle',
            )
        if (
            self._valid_observation_stamp_ns is None
            or self._valid_observation_stamp_ns < identity.stamp_ns
            or self._valid_observation_frame_id != identity.frame_id
            or self._target_camera is None
        ):
            raise _PlanningObservationChanged(
                'grasp-planning observation time or frame changed',
            )
        if (
            target_geometry is not None
            and self._valid_observation_stamp_ns == identity.stamp_ns
        ):
            raise _PlanningObservationPending(
                'approach plan is waiting for a strictly newer exact RGB-D bundle',
            )
        current = np.asarray(self._target_camera, dtype=float)
        frozen = np.asarray(identity.target_position_camera, dtype=float)
        drift = float(np.linalg.norm(current - frozen))
        tolerance = float(self.get_parameter(
            'approach_planning_target_drift_tolerance_m',
        ).value)
        if (
            not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not math.isfinite(drift)
            or drift > tolerance
        ):
            raise _PlanningObservationChanged(
                'target moved while grasp planning was active: '
                f'drift={drift:.3f}m, tolerance={tolerance:.3f}m',
            )
        if target_geometry is not None:
            if self._target_cloud is None:
                raise _PlanningObservationChanged(
                    'approach result lost its exact semantic target cloud',
                )
            try:
                current_geometry = _target_geometry_signature(
                    self._target_cloud,
                    min_points=int(self.get_parameter('semantic_min_points').value),
                    trim_mad_scale=float(self.get_parameter(
                        'approach_planning_geometry_trim_mad_scale',
                    ).value),
                    extent_percentile=float(self.get_parameter(
                        'approach_planning_geometry_extent_percentile',
                    ).value),
                )
                _validate_target_geometry_change(
                    target_geometry,
                    current_geometry,
                    max_center_drift_m=tolerance,
                    max_extent_change_m=float(self.get_parameter(
                        'approach_planning_geometry_max_extent_change_m',
                    ).value),
                    max_extent_ratio=float(self.get_parameter(
                        'approach_planning_geometry_max_extent_ratio',
                    ).value),
                    axis_separation_ratio=float(self.get_parameter(
                        'approach_planning_geometry_axis_separation_ratio',
                    ).value),
                    max_orientation_change_rad=float(self.get_parameter(
                        'approach_planning_geometry_max_orientation_change_rad',
                    ).value),
                )
            except ValueError as error:
                raise _PlanningObservationChanged(
                    f'target geometry changed while approach planning: {error}',
                ) from error

    def _defer_completed_work_pose_observation(self, detail: str) -> None:
        """Retain a completed result for one fixed exact-observation grace window."""
        now = float(self._now_s())
        if not math.isfinite(now):
            raise _PlanningObservationChanged(
                'work-pose exact-observation wait clock is invalid',
            )
        wait = getattr(self, '_future_observation_wait', None)
        if wait is None:
            timeout = float(self.get_parameter('perception_loss_timeout_s').value)
            deadline = now + timeout
            if (
                not math.isfinite(timeout)
                or timeout <= 0.0
                or not math.isfinite(deadline)
            ):
                raise ValueError(
                    'perception_loss_timeout_s must bound the work-pose result wait',
                )
            self._future_observation_wait = _PlanningObservationWait(now, deadline)
            return
        if (
            not math.isfinite(wait.started_at_s)
            or not math.isfinite(wait.deadline_s)
            or wait.deadline_s <= wait.started_at_s
            or now < wait.started_at_s
        ):
            raise _PlanningObservationChanged(
                'work-pose exact-observation wait clock moved backwards',
            )
        if now < wait.deadline_s:
            return
        raise _PlanningObservationChanged(
            'work-pose result exact-observation wait timed out after '
            f'{wait.deadline_s - wait.started_at_s:.3f}s: {detail}',
        )

    def _start_planning(
        self,
        kind: str,
        observation: PerceptionObservation,
        *,
        planning_joints: np.ndarray | None = None,
    ) -> None:
        if planning_joints is None:
            if self._joint_state is None:
                raise ValueError('planning requires current joint feedback')
            joints = self._joint_state.copy()
        else:
            joints = np.asarray(planning_joints, dtype=float).copy()
            if (
                joints.shape != (self._planner.chain.dof,)
                or not np.all(np.isfinite(joints))
            ):
                raise ValueError('planning joint snapshot is invalid')
        control = self._planning_control(kind)
        self._future_kind = kind
        self._future_serial = observation.serial
        self._future_generation = self._task_generation.current
        self._future_base_anchor = None
        self._future_observation_identity = None
        self._future_observation_wait = None
        if kind == 'standoff':
            now = self._now_s()
            if (
                self._position_xy is None
                or self._yaw is None
                or self._odom_stamp_ns is None
                or self._odom_seen_at is None
            ):
                raise ValueError('work-pose planning requires platform odometry')
            odom_age = now - self._odom_seen_at
            max_odom_age = float(self.get_parameter('work_pose_odom_max_age_s').value)
            if (
                not math.isfinite(odom_age)
                or odom_age < 0.0
                or odom_age > max_odom_age
            ):
                raise ValueError(
                    f'work-pose odometry is stale: age={odom_age:.3f}s',
                )
            source_age = now - float(self._odom_stamp_ns) * 1e-9
            if (
                not math.isfinite(source_age)
                or source_age < 0.0
                or source_age > max_odom_age
            ):
                raise ValueError(
                    'work-pose odometry source stamp is stale: '
                    f'age={source_age:.3f}s',
                )
            anchor_pose = _se2_pose(
                (*self._position_xy, self._yaw),
                'work-pose planning anchor',
            )
            self._future_base_anchor = _PlanningBaseAnchor(
                pose_map=tuple(float(value) for value in anchor_pose),
                odom_sequence=self._odom_sequence,
                odom_stamp_ns=self._odom_stamp_ns,
            )
            self._future_observation_identity = (
                self._capture_planning_observation_identity(observation)
            )
            history = tuple(
                _relative_se2(anchor_pose, attempted)
                for attempted in self._work_pose_history_map
            )
            self._future = self._worker.submit(
                self._planner.prospective_standoff,
                observation,
                joints,
                control,
                history_relative_base_poses=history,
            )
        elif kind == 'pregrasp':
            self._future_observation_identity = (
                self._capture_planning_observation_identity(observation)
            )
            self._future = self._worker.submit(
                self._planner.pregrasp_program,
                observation,
                joints,
                control,
            )
        elif kind == 'approach':
            if self._pregrasp_program is None:
                raise ValueError('approach planning requires a pregrasp program')
            self._future_observation_identity = (
                self._capture_planning_observation_identity(observation)
            )
            self._future = self._worker.submit(
                self._planner.grasp_completion_program,
                self._pregrasp_program,
                observation,
                joints,
                control,
            )
        else:
            raise ValueError(f'unsupported planning kind: {kind}')

    def _start_carry_planning(self, stamp_s: float) -> None:
        if self._joint_state is None or self._scene_cloud is None or self._target_cloud is None:
            raise ValueError('carry planning requires current joints and perception clouds')
        goal = np.asarray(
            self.get_parameter('carry_joint_positions').value,
            dtype=float,
        )
        if goal.shape != (self._planner.chain.dof,) or not np.all(np.isfinite(goal)):
            raise ValueError('carry_joint_positions must match the active arm chain')
        control = self._planning_control('carry')
        self._future_kind = 'carry'
        self._future_serial = self._core.planned_serial or 0
        self._future_generation = self._task_generation.current
        self._future_base_anchor = None
        self._future_observation_identity = None
        self._future_observation_wait = None
        self._future = self._worker.submit(
            self._planner.joint_motion,
            current_joints=self._joint_state.copy(),
            goal_joints=goal,
            scene_points=self._scene_cloud.copy(),
            target_points=self._target_cloud.copy(),
            stamp_s=float(stamp_s),
            control=control,
        )

    def _start_pregrasp_transit_validation(
        self,
        observation_identity: _PlanningObservationIdentity,
        serial: int,
    ) -> None:
        """Revalidate transit while keeping state callbacks live."""
        program = self._pregrasp_program
        if program is None or self._joint_state is None:
            raise ValueError(
                'pregrasp transit validation requires a plan and arm state',
            )
        now = self._now_s()
        synchronized = self._serial_gate.snapshot(now)
        if (
            not self._grounding_observation_authorized(synchronized)
            or self._scene_cloud is None
            or self._target_cloud is None
        ):
            raise ValueError(
                'pregrasp transit validation requires fresh exact perception',
            )
        timed = program.transit
        if (
            np.max(np.abs(self._joint_state - timed.positions[0]))
            > float(self.get_parameter('max_trajectory_start_error_rad').value)
        ):
            raise ValueError(
                'measured arm differs from the pregrasp transit start',
            )
        assert synchronized is not None
        control = self._planning_control('pregrasp_validation')
        self._future_kind = 'pregrasp_validation'
        self._future_serial = int(serial)
        self._future_generation = self._task_generation.current
        self._future_base_anchor = None
        self._future_observation_identity = observation_identity
        self._future_observation_wait = None
        self._future = self._worker.submit(
            self._planner.validate_path,
            timed.positions,
            scene_points=self._scene_cloud.copy(),
            target_points=self._target_cloud.copy(),
            stamp_s=synchronized.stamp_s,
            segment_name='transit',
            control=control,
        )

    def _freeze_pregrasp_dispatch_fence(self) -> _PregraspDispatchFence:
        """Freeze state watermarks after asynchronous validation."""
        validated_at_s = self._now_s()
        timeout_s = float(self.get_parameter(
            'pregrasp_dispatch_feedback_wait_timeout_s',
        ).value)
        joint_stamp_ns = self._joint_stamp_ns
        odom_stamp_ns = self._odom_stamp_ns
        if (
            not math.isfinite(validated_at_s)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
            or joint_stamp_ns is None
            or joint_stamp_ns <= 0
            or odom_stamp_ns is None
            or odom_stamp_ns <= 0
            or self._joint_sequence <= 0
            or self._odom_sequence <= 0
        ):
            raise ValueError(
                'pregrasp dispatch feedback wait contract is unavailable '
                'or invalid',
            )
        deadline_s = validated_at_s + timeout_s
        if not math.isfinite(deadline_s) or deadline_s <= validated_at_s:
            raise ValueError('pregrasp dispatch feedback deadline is invalid')
        return _PregraspDispatchFence(
            deadline_s=deadline_s,
            minimum_joint_sequence=int(self._joint_sequence),
            minimum_joint_stamp_ns=int(joint_stamp_ns),
            minimum_odom_sequence=int(self._odom_sequence),
            minimum_odom_stamp_ns=int(odom_stamp_ns),
        )

    def _poll_planning(self) -> None:
        if self._future is None or not self._future.done():
            return
        future = self._future
        kind = self._future_kind
        serial = self._future_serial
        generation = self._future_generation
        base_anchor = getattr(self, '_future_base_anchor', None)
        observation_identity = getattr(self, '_future_observation_identity', None)
        if not self._task_generation.accepts(generation):
            MobileManipulationRuntime._clear_planning_future_state(self)
            return
        try:
            result = future.result()
            if kind == 'standoff':
                if base_anchor is None:
                    raise ValueError('work-pose result lost its odometry anchor')
                if observation_identity is None:
                    raise ValueError('work-pose result lost its observation identity')
                try:
                    self._validate_completed_work_pose_observation(
                        observation_identity,
                    )
                except _PlanningObservationPending as error:
                    self._defer_completed_work_pose_observation(str(error))
                    return
                if getattr(self, '_future_observation_wait', None) is not None:
                    self._defer_completed_work_pose_observation(
                        'the exact-authorized observation arrived after its '
                        'fixed work-pose result wait',
                    )
            elif kind in {'pregrasp', 'pregrasp_validation', 'approach'}:
                if observation_identity is None:
                    raise ValueError(
                        f'{kind} result lost its observation identity',
                    )
                try:
                    approach_anchor = (
                        self._approach_planning_anchor
                        if kind == 'approach'
                        else None
                    )
                    if kind == 'approach' and approach_anchor is None:
                        raise _PlanningObservationChanged(
                            'approach result lost its fresh handoff anchor',
                        )
                    self._validate_grasp_planning_observation(
                        observation_identity,
                        target_geometry=(
                            approach_anchor.target_geometry
                            if approach_anchor is not None
                            else None
                        ),
                    )
                except _PlanningObservationPending as error:
                    now = self._now_s()
                    wait = self._future_observation_wait
                    if wait is None:
                        timeout = float(self.get_parameter(
                            'perception_loss_timeout_s',
                        ).value)
                        self._future_observation_wait = _PlanningObservationWait(
                            now,
                            now + timeout,
                        )
                        return
                    if now <= wait.deadline_s:
                        return
                    raise _PlanningObservationChanged(
                        f'{kind} result exact-observation wait timed out: {error}',
                    ) from error
                wait = self._future_observation_wait
                if wait is not None and self._now_s() > wait.deadline_s:
                    raise _PlanningObservationChanged(
                        f'{kind} result exact-observation deadline expired',
                    )
            MobileManipulationRuntime._clear_planning_future_state(self)
            if kind == 'carry':
                self._carry_program = result
                self._publish_program_segment('carry')
            elif kind == 'standoff':
                assert base_anchor is not None
                assert observation_identity is not None
                if self._position_xy is None or self._yaw is None:
                    raise ValueError('work-pose result requires current platform odometry')
                current_map_pose = _se2_pose(
                    (*self._position_xy, self._yaw),
                    'current work-pose platform pose',
                )
                anchor_map_pose = _se2_pose(
                    base_anchor.pose_map,
                    'work-pose result anchor',
                )
                drift = _relative_se2(anchor_map_pose, current_map_pose)
                max_translation_drift = float(self.get_parameter(
                    'work_pose_anchor_translation_tolerance_m',
                ).value)
                max_yaw_drift = float(self.get_parameter(
                    'work_pose_anchor_yaw_tolerance_rad',
                ).value)
                if (
                    np.linalg.norm(drift[:2]) > max_translation_drift
                    or abs(float(drift[2])) > max_yaw_drift
                ):
                    raise ValueError(
                        'platform moved while work-pose planning was active: '
                        f'dxy={np.linalg.norm(drift[:2]):.3f}m, '
                        f'dyaw={drift[2]:.3f}rad',
                    )
                relative_pose = _se2_pose(
                    result.relative_base_pose,
                    'selected relative work pose',
                )
                map_goal = _compose_se2(anchor_map_pose, relative_pose)
                self._desired_depth = float(result.desired_camera_depth_m)
                self._approximate_displacement = float(np.linalg.norm(relative_pose[:2]))
                self._coarse_nav_ready = False
                source = {
                    'request_id': observation_identity.request_id,
                    'producer_epoch': observation_identity.producer_epoch,
                    'generation': observation_identity.generation,
                    'observation_serial': serial,
                    'observation_stamp_ns': observation_identity.stamp_ns,
                    'observation_frame_id': observation_identity.frame_id,
                    'odom_sequence': base_anchor.odom_sequence,
                    'odom_stamp_ns': base_anchor.odom_stamp_ns,
                }
                self._work_pose = {
                    'goal_id': f'work-{uuid.uuid4().hex}',
                    'relative_base_pose': relative_pose.tolist(),
                    'map_goal_xy': map_goal[:2].tolist(),
                    'map_goal_yaw_rad': float(map_goal[2]),
                    'map_frame': str(self.get_parameter(
                        'platform_odometry_parent_frame',
                    ).value),
                    'anchor_map_pose': anchor_map_pose.tolist(),
                    'predicted_target_position_piper': np.asarray(
                        result.predicted_target_position_piper,
                        dtype=float,
                    ).tolist(),
                    'desired_camera_depth_m': self._desired_depth,
                    'selection_mode': str(result.selection_mode),
                    'kinematic_precheck_feasible': bool(
                        result.kinematic_precheck_feasible,
                    ),
                    'diagnostics': _work_pose_diagnostics_payload(
                        result.diagnostics,
                    ),
                    'rejected_precheck_diagnostics': (
                        _work_pose_diagnostics_payload(
                            result.rejected_precheck_diagnostics,
                        )
                    ),
                    'source': source,
                }
                self._work_pose_created_at_s = self._now_s()
                self._navigation_status_seen_s = None
                self._navigation_status_goal_id = ''
                self._navigation_status_phase = ''
                self._navigation_goal_acknowledged = False
                self._navigation_ack_position_xy = None
                self._navigation_ack_odom_sequence = None
                self._navigation_ack_odom_stamp_ns = None
                self._navigation_history_recorded = False
                self._coarse_nav_arrival_started_at_s = None
                self._coarse_nav_arrival_stable_since_s = None
                self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
                self._coarse_nav_arrival_last_odom_sequence = None
                self._coarse_nav_arrival_last_odom_stamp_ns = None
                if self._task.stage.value == 'search':
                    self._task.apply(StageResult.success())
                self._core.mark_standoff(serial)
                MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
                    self,
                    True,
                )
                self._maybe_finish_coarse_nav()
            elif kind == 'pregrasp':
                if not isinstance(result, PregraspTransitProgram):
                    raise TypeError('pregrasp planner returned the wrong program type')
                assert observation_identity is not None
                if self._core.phase is not RuntimePhase.PLANNING:
                    raise RuntimeError(
                        f'pregrasp result arrived while {self._core.phase.value}',
                    )
                self._pregrasp_program = result
                self._pregrasp_planning_identity = observation_identity
                self._program = None
                self._start_pregrasp_transit_validation(
                    observation_identity,
                    serial,
                )
            elif kind == 'pregrasp_validation':
                if result is not True:
                    raise ValueError(
                        'newest perceived scene invalidates the '
                        'pregrasp transit',
                    )
                if (
                    self._core.phase is not RuntimePhase.PLANNING
                    or self._pregrasp_program is None
                    or self._pregrasp_planning_identity != observation_identity
                ):
                    raise RuntimeError(
                        'pregrasp validation result lost its active '
                        'planning owner',
                    )
                self._pregrasp_dispatch_fence = (
                    self._freeze_pregrasp_dispatch_fence()
                )
            elif kind == 'approach':
                if not isinstance(result, GraspCompletionProgram):
                    raise TypeError('approach planner returned the wrong program type')
                anchor = self._approach_planning_anchor
                if (
                    anchor is None
                    or observation_identity != anchor.observation_identity
                    or serial != anchor.observation_serial
                ):
                    raise ValueError('approach result lost its fresh handoff anchor')
                if self._core.phase is not RuntimePhase.APPROACH_PLANNING:
                    raise RuntimeError(
                        f'approach result arrived while {self._core.phase.value}',
                    )
                wait_timeout = float(self.get_parameter(
                    'approach_execution_joint_wait_timeout_s',
                ).value)
                now = self._now_s()
                if (
                    not math.isfinite(wait_timeout)
                    or wait_timeout <= 0.0
                    or not math.isfinite(now)
                ):
                    raise ValueError(
                        'approach execution joint wait timeout is invalid',
                    )
                self._program = result
                if self._joint_history:
                    latest_joint = self._joint_history[-1]
                    minimum_sequence = max(
                        anchor.joint_sequence,
                        int(latest_joint.sequence),
                    )
                    minimum_stamp_ns = max(
                        anchor.joint_stamp_ns,
                        int(latest_joint.source_stamp_ns),
                    )
                else:
                    minimum_sequence = anchor.joint_sequence
                    minimum_stamp_ns = anchor.joint_stamp_ns
                self._approach_execution_joint_fence = (
                    _ApproachExecutionJointFence(
                        deadline_s=now + wait_timeout,
                        minimum_joint_sequence=minimum_sequence,
                        minimum_joint_stamp_ns=minimum_stamp_ns,
                    )
                )
            else:
                raise RuntimeError(f'unhandled planning result kind: {kind}')
        except PlanningCancelled:
            MobileManipulationRuntime._clear_planning_future_state(self)
            return
        except _PlanningObservationChanged as error:
            MobileManipulationRuntime._clear_planning_future_state(self)
            detail = str(error)
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
        except PlanningDeadlineExceeded as error:
            MobileManipulationRuntime._clear_planning_future_state(self)
            detail = f'{kind} planning budget exhausted: {error}'
            failure = (
                FailureKind.PLAN_BLOCKED
                if kind in {'pregrasp', 'pregrasp_validation', 'approach'}
                else FailureKind.IK_UNREACHABLE
            )
            if kind == 'carry' or not self._recover_precontact(failure, detail):
                self._apply_safety(self._core.fail(detail))
        except Exception as error:
            MobileManipulationRuntime._clear_planning_future_state(self)
            detail = f'{kind} planning failed: {type(error).__name__}: {error}'
            failure = (
                FailureKind.PLAN_BLOCKED
                if kind in {'pregrasp', 'pregrasp_validation', 'approach'}
                else FailureKind.IK_UNREACHABLE
            )
            if kind == 'carry' or not self._recover_precontact(failure, detail):
                self._apply_safety(self._core.fail(detail))

    def _tick(self) -> None:
        with self._lock:
            now = self._now_s()
            terminal = (
                RuntimePhase.IDLE,
                RuntimePhase.PICK_COMPLETE,
                RuntimePhase.COMPLETE,
                RuntimePhase.CANCELED,
                RuntimePhase.FAILED,
            )
            if self._core.phase in terminal:
                self._publish_status()
                return
            if not self._guard_active_posture(now):
                self._publish_status()
                return
            if self._lookout_pending:
                self._dispatch_pending_lookout(now)
                self._publish_status()
                return
            self._poll_planning()
            if self._core.phase in terminal:
                self._publish_status()
                return
            # A completed planner future can schedule bounded recovery into
            # POSE_SETTLE during this tick. Establish the LOOKOUT timing
            # contract before the phase validator below observes that state.
            if self._lookout_pending:
                self._dispatch_pending_lookout(now)
                self._publish_status()
                return
            if (
                self._core.phase is RuntimePhase.PLANNING
                and self._pregrasp_dispatch_fence is not None
            ):
                self._pregrasp_result_execution_tick(self._now_s())
                self._publish_status()
                return
            if (
                self._core.phase in (
                    RuntimePhase.TRANSIT,
                    RuntimePhase.APPROACH,
                    RuntimePhase.LIFT,
                    RuntimePhase.CARRY,
                    RuntimePhase.PLACE_TRANSIT,
                    RuntimePhase.PLACE_APPROACH,
                    RuntimePhase.PLACE_RETREAT,
                )
                and self._trajectory_deadline_s is not None
                and now > self._trajectory_deadline_s
            ):
                self._apply_safety(self._core.fail('arm trajectory execution timed out'))
            if self._core.phase is RuntimePhase.POSE_SETTLE:
                visual_search_settle = (
                    getattr(self, '_visual_search_settle_reference', None)
                    is not None
                )
                if visual_search_settle:
                    # Keep the task-local mux lease through the fixed stop dwell.
                    # The bridge may still be failed while search recovery settles.
                    self._visual_search_active_pub.publish(Bool(data=True))
                if (
                    visual_search_settle
                    and bool(getattr(
                        getattr(self, '_visual_search', None),
                        'active',
                        False,
                    ))
                ):
                    self._visual_search_settle_correction_tick(now)
                    self._publish_status()
                    return
                # Retain task-local velocity ownership throughout both the
                # fixed dwell and any bounded platform-stationary wait.
                self._publish_zero()
                settle_started = self._pose_settle_started_at
                settle_last_tick = self._pose_settle_last_tick_at
                settle_until = self._pose_settle_until
                if (
                    settle_started is None
                    or settle_last_tick is None
                    or settle_until is None
                    or not math.isfinite(now)
                    or not math.isfinite(settle_started)
                    or not math.isfinite(settle_last_tick)
                    or not math.isfinite(settle_until)
                    or settle_last_tick < settle_started
                    or settle_until <= settle_started
                ):
                    self._apply_safety(self._core.fail(
                        'pose settle timing contract is unavailable or invalid',
                    ))
                elif now < settle_last_tick:
                    self._apply_safety(self._core.fail(
                        'pose settle clock moved backwards',
                    ))
                else:
                    self._pose_settle_last_tick_at = now
                    if MobileManipulationRuntime._monitor_visual_search_settle_rebound(
                        self,
                        now,
                    ):
                        self._publish_status()
                        return
                    if now >= settle_until:
                        if self._visual_search_pending:
                            self._begin_visual_search(now)
                        else:
                            self._finish_pose_settle(now)
                self._publish_status()
                return
            if self._core.phase is RuntimePhase.VISUAL_SEARCH:
                self._visual_search_tick(now)
                self._publish_status()
                return
            synchronized = self._serial_gate.snapshot(now)
            perception_exempt = self._core.phase in (
                RuntimePhase.GROUNDING,
                RuntimePhase.STANDOFF,
                RuntimePhase.NEAR_GROUNDING,
                RuntimePhase.FINAL_GROUNDING,
                RuntimePhase.PREGRASP_REOBSERVE,
                RuntimePhase.APPROACH_PLANNING,
                RuntimePhase.PLACE_GROUNDING,
                RuntimePhase.POST_RELEASE_VERIFICATION,
            ) or (
                MobileManipulationRuntime._has_frozen_coarse_nav_contract(self)
            ) or (
                MobileManipulationRuntime._visual_servo_vertical_recovery_state_complete(
                    self,
                )
            )
            predicted_loss_allowed = False
            execution_perception_phase = self._core.phase in (
                RuntimePhase.CLOSING,
                RuntimePhase.LIFT,
                RuntimePhase.VERIFY,
            )
            exact_execution_observation = bool(
                execution_perception_phase
                and self._grounding_observation_authorized(synchronized)
            )
            if execution_perception_phase and self._execution_occlusion.loss_active:
                restored = bool(
                    exact_execution_observation
                    and self._restore_execution_occlusion_tracking(
                        now,
                        synchronized,
                    )
                )
                if self._core.phase in terminal:
                    self._publish_status()
                    return
                if not restored:
                    decision = self._execution_occlusion_decision(now)
                    if not decision.allowed:
                        detail = (
                            'execution perception loss rejected: '
                            f'{decision.reason}'
                        )
                        self._apply_safety(self._core.perception_invalid(detail))
                        self._publish_status()
                        return
                    predicted_loss_allowed = True
            elif execution_perception_phase:
                if exact_execution_observation:
                    assert synchronized is not None
                    if not self._retain_execution_occlusion_observation(
                        synchronized,
                    ):
                        detail = (
                            'execution perception authorization rejected: '
                            f'{self._execution_occlusion_last_decision.reason}'
                        )
                        self._apply_safety(self._core.perception_invalid(detail))
                        self._publish_status()
                        return
                else:
                    detail = (
                        'execution perception is not synchronized and '
                        'exact-authorized'
                    )
                    if not self._execution_occlusion_allows_loss(now, detail):
                        decision_detail = (
                            self._execution_occlusion_last_decision.reason
                        )
                        failure_detail = (
                            detail
                            if not decision_detail
                            else f'{detail}: {decision_detail}'
                        )
                        self._apply_safety(self._core.perception_invalid(
                            failure_detail,
                        ))
                        self._publish_status()
                        return
                    predicted_loss_allowed = True
            if not perception_exempt and not self._perception_valid:
                detail = 'perception is not valid'
                if (
                    not predicted_loss_allowed
                    and not self._execution_occlusion_allows_loss(now, detail)
                ):
                    decision_detail = self._execution_occlusion_last_decision.reason
                    failure_detail = (
                        detail
                        if not decision_detail
                        else f'{detail}: {decision_detail}'
                    )
                    if not self._recover_precontact(
                        FailureKind.TARGET_LOST,
                        failure_detail,
                    ):
                        self._apply_safety(self._core.perception_invalid(
                            failure_detail,
                        ))
                    self._publish_status()
                    return
                predicted_loss_allowed = True
            if (
                not perception_exempt
                and self._valid_seen_at is not None
                and now - self._valid_seen_at
                > float(self.get_parameter('perception_loss_timeout_s').value)
            ):
                detail = 'perception validity is stale'
                self._revoke_perception_success()
                if (
                    not predicted_loss_allowed
                    and not self._execution_occlusion_allows_loss(now, detail)
                ):
                    decision_detail = self._execution_occlusion_last_decision.reason
                    failure_detail = (
                        detail
                        if not decision_detail
                        else f'{detail}: {decision_detail}'
                    )
                    if not self._recover_precontact(
                        FailureKind.TARGET_LOST,
                        failure_detail,
                    ):
                        self._apply_safety(self._core.perception_invalid(
                            failure_detail,
                        ))
                    self._publish_status()
                    return
            if self._core.phase is RuntimePhase.GROUNDING:
                self._grounding_tick(synchronized)
            elif self._core.phase is RuntimePhase.VISUAL_SERVO:
                self._visual_servo(now)
            elif (
                self._core.phase is RuntimePhase.NEAR_GROUNDING
                and self._near_view_settle_until is not None
            ):
                self._near_view_settle_tick(now)
            elif self._core.phase in (
                RuntimePhase.NEAR_GROUNDING,
                RuntimePhase.FINAL_GROUNDING,
                RuntimePhase.PLACE_GROUNDING,
            ):
                self._reground_tick(now, synchronized)
            elif self._core.phase is RuntimePhase.COARSE_NAV:
                if self._coarse_navigation_status_alive(now):
                    self._maybe_finish_coarse_nav()
            elif self._core.phase is RuntimePhase.WAIT_FRESH_OBSERVATION:
                self._wait_fresh_observation_tick(synchronized)
            elif self._core.phase is RuntimePhase.PREGRASP_REOBSERVE:
                self._pregrasp_reobserve_tick(now, synchronized)
            elif self._core.phase is RuntimePhase.APPROACH_PLANNING:
                self._approach_result_execution_tick(now)
            elif self._core.phase is RuntimePhase.CLOSING:
                self._closing_tick(now)
            elif self._core.phase is RuntimePhase.RELEASING:
                self._release_tick(now)
            elif self._core.phase is RuntimePhase.VERIFY:
                self._verification_tick(now)
            elif self._core.phase is RuntimePhase.POST_RELEASE_VERIFICATION:
                self._post_release_verification_tick(now)
            elif (
                self._core.phase is RuntimePhase.PLACE_PLANNING
                and self._place_planning_started_at is not None
                and now - self._place_planning_started_at
                > float(self.get_parameter('place_planning_timeout_s').value)
            ):
                self._apply_safety(self._core.fail('placement planning timed out'))
            self._publish_status()

    def _visual_servo_vertical_recovery_state_complete(self) -> bool:
        """Require an armed, phase-owned stop before granting exemptions."""
        started_at = getattr(
            self,
            '_visual_servo_vertical_recovery_started_at_s',
            None,
        )
        minimum_cloud_stamp_ns = getattr(
            self,
            '_visual_servo_vertical_minimum_cloud_stamp_ns',
            None,
        )
        stationarity = getattr(
            self,
            '_visual_servo_vertical_stationarity',
            None,
        )
        if (
            getattr(getattr(self, '_core', None), 'phase', None)
            is not RuntimePhase.VISUAL_SERVO
            or started_at is None
            or isinstance(started_at, bool)
            or minimum_cloud_stamp_ns is None
            or isinstance(minimum_cloud_stamp_ns, bool)
            or stationarity is None
        ):
            return False
        try:
            started = float(started_at)
            minimum_cloud_stamp = int(minimum_cloud_stamp_ns)
            stop_received_at = float(stationarity.stop_received_at_s)
            minimum_odom_sequence = int(stationarity.minimum_odom_sequence)
            minimum_odom_stamp_ns = int(stationarity.minimum_odom_stamp_ns)
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(started)
            and math.isfinite(stop_received_at)
            and stop_received_at == started
            and minimum_cloud_stamp > 0
            and minimum_odom_sequence > 0
            and minimum_odom_stamp_ns > 0
        )

    def _fail_visual_servo_vertical_recovery(self, detail: str) -> None:
        MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
            self,
            'failed',
        )
        if not self._recover_precontact(
            FailureKind.VISUAL_APPROACH_FAILED,
            detail,
        ):
            self._apply_safety(self._core.fail(detail))

    def _begin_visual_servo_vertical_recovery(self, now: float) -> None:
        """Stop, then wait for a newer exact bundle after vertical clipping."""
        timeout_s = self._visual_search.config.stationary_wait_timeout_s
        valid_stamp_ns = self._valid_observation_stamp_ns
        valid_frame_id = self._valid_observation_frame_id
        target_stamp_ns = self._target_stamp_ns
        target_cloud_stamp_ns = self._target_cloud_stamp_ns
        if (
            not math.isfinite(float(now))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
            or self._odom_stamp_ns is None
            or self._odom_stamp_ns <= 0
            or self._odom_sequence <= 0
            or not self._perception_valid
            or valid_stamp_ns is None
            or valid_stamp_ns <= 0
            or not valid_frame_id
            or self._target_uv is None
            or isinstance(target_stamp_ns, bool)
            or not isinstance(target_stamp_ns, int)
            or target_stamp_ns <= 0
            or isinstance(target_cloud_stamp_ns, bool)
            or not isinstance(target_cloud_stamp_ns, int)
            or target_cloud_stamp_ns <= 0
            or not self._target_frame_id
            or not self._target_cloud_frame_id
        ):
            raise ValueError(
                'vertical recovery requires a positive timeout, valid '
                'odometry, and authorized target image evidence',
            )
        # The status, Detection3D, and target PointCloud2 are independent DDS
        # samples.  A control tick may observe a legitimate bundle while only
        # part of its topics have arrived.  Stop immediately and fence against
        # every source stamp already seen; the recovery tick still requires a
        # strictly newer, exact-authorized target/cloud pair before resuming.
        minimum_cloud_stamp_ns = max(
            int(valid_stamp_ns),
            target_stamp_ns,
            target_cloud_stamp_ns,
        )
        self._visual_servo_vertical_recovery_started_at_s = float(now)
        self._visual_servo_vertical_minimum_cloud_stamp_ns = (
            minimum_cloud_stamp_ns
        )
        self._visual_servo_vertical_safe_start_cloud_stamp_ns = None
        self._visual_servo_vertical_safe_last_cloud_stamp_ns = None
        self._visual_servo_vertical_stationarity.reset(
            stop_received_at_s=float(now),
            minimum_odom_sequence=int(self._odom_sequence),
            minimum_odom_stamp_ns=int(self._odom_stamp_ns),
        )
        controller = getattr(self._approach, 'visual_servo', None)
        if controller is not None:
            controller.reset()
        self._publish_zero()

    def _visual_servo_vertical_recovery_tick(self, now: float) -> None:
        """Resume visual servo only after fresh safe pixels and quiet odometry."""
        self._publish_zero()
        started_at = self._visual_servo_vertical_recovery_started_at_s
        minimum_cloud_stamp_ns = (
            self._visual_servo_vertical_minimum_cloud_stamp_ns
        )
        timeout_s = self._visual_search.config.stationary_wait_timeout_s
        if (
            not MobileManipulationRuntime._visual_servo_vertical_recovery_state_complete(
                self,
            )
            or started_at is None
            or minimum_cloud_stamp_ns is None
            or not math.isfinite(float(now))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
            or float(now) < started_at
        ):
            MobileManipulationRuntime._fail_visual_servo_vertical_recovery(
                self,
                'visual-servo vertical recovery timing contract is invalid',
            )
            return

        elapsed_s = float(now) - started_at
        if elapsed_s >= timeout_s:
            MobileManipulationRuntime._fail_visual_servo_vertical_recovery(
                self,
                'visual-servo vertical image recovery timed out after '
                f'{timeout_s:.2f}s',
            )
            return

        try:
            outer_margin = float(
                self.get_parameter('visual_servo_image_margin_ratio').value,
            )
            reentry_margin = float(
                self.get_parameter('visual_search_vertical_margin_ratio').value,
            )
            max_cloud_gap_s = float(
                self.get_parameter('max_perception_age_s').value,
            )
            if (
                not math.isfinite(outer_margin)
                or not math.isfinite(reentry_margin)
                or not math.isfinite(max_cloud_gap_s)
                or not 0.0 <= outer_margin < reentry_margin < 0.5
                or max_cloud_gap_s <= 0.0
            ):
                raise ValueError(
                    'vertical reentry margin and cloud gap must be valid',
                )
        except ValueError as error:
            MobileManipulationRuntime._fail_visual_servo_vertical_recovery(
                self,
                f'visual-servo vertical recovery image guard failed: {error}',
            )
            return

        valid_stamp_ns = self._valid_observation_stamp_ns
        valid_frame_id = self._valid_observation_frame_id
        cloud_stamp_ns = self._target_cloud_stamp_ns
        exact_observation = bool(
            self._perception_valid
            and valid_stamp_ns is not None
            and valid_stamp_ns > 0
            and bool(valid_frame_id)
            and self._target_uv is not None
            and self._target_stamp_ns == valid_stamp_ns
            and cloud_stamp_ns == valid_stamp_ns
            and self._target_frame_id == valid_frame_id
            and self._target_cloud_frame_id == valid_frame_id
        )
        if not exact_observation:
            self._visual_servo_vertical_safe_start_cloud_stamp_ns = None
            self._visual_servo_vertical_safe_last_cloud_stamp_ns = None
            return

        freshness_epsilon_s = 1e-9
        cloud_age_s = float(now) - cloud_stamp_ns * 1e-9
        if not -freshness_epsilon_s <= cloud_age_s <= max_cloud_gap_s:
            self._visual_servo_vertical_safe_start_cloud_stamp_ns = None
            self._visual_servo_vertical_safe_last_cloud_stamp_ns = None
            return

        try:
            _horizontal_edge, vertical_edge = (
                self._tracked_target_edge_directions(
                    margin_ratio=reentry_margin,
                )
            )
        except ValueError as error:
            MobileManipulationRuntime._fail_visual_servo_vertical_recovery(
                self,
                f'visual-servo vertical recovery image guard failed: {error}',
            )
            return

        if vertical_edge:
            self._visual_servo_vertical_safe_start_cloud_stamp_ns = None
            self._visual_servo_vertical_safe_last_cloud_stamp_ns = None
        else:
            cloud_stamp_ns = int(cloud_stamp_ns)
            last_stamp_ns = (
                self._visual_servo_vertical_safe_last_cloud_stamp_ns
            )
            if last_stamp_ns is not None and cloud_stamp_ns < last_stamp_ns:
                MobileManipulationRuntime._fail_visual_servo_vertical_recovery(
                    self,
                    'visual-servo vertical recovery cloud time moved backwards',
                )
                return
            if cloud_stamp_ns > minimum_cloud_stamp_ns:
                if last_stamp_ns is None:
                    self._visual_servo_vertical_safe_start_cloud_stamp_ns = (
                        cloud_stamp_ns
                    )
                    self._visual_servo_vertical_safe_last_cloud_stamp_ns = (
                        cloud_stamp_ns
                    )
                elif cloud_stamp_ns > last_stamp_ns:
                    cloud_gap_s = (cloud_stamp_ns - last_stamp_ns) * 1e-9
                    if cloud_gap_s > max_cloud_gap_s:
                        self._visual_servo_vertical_safe_start_cloud_stamp_ns = (
                            cloud_stamp_ns
                        )
                    self._visual_servo_vertical_safe_last_cloud_stamp_ns = (
                        cloud_stamp_ns
                    )

        safe_start_stamp_ns = (
            self._visual_servo_vertical_safe_start_cloud_stamp_ns
        )
        safe_last_stamp_ns = self._visual_servo_vertical_safe_last_cloud_stamp_ns
        safe_duration_s = (
            0.0
            if safe_start_stamp_ns is None or safe_last_stamp_ns is None
            else (safe_last_stamp_ns - safe_start_stamp_ns) * 1e-9
        )
        stationarity = self._visual_servo_vertical_stationarity
        max_odom_gap_s = stationarity.max_odom_gap_s
        odometry_ready = False
        if self._odom_stamp_ns is not None and self._odom_seen_at is not None:
            odom_receipt_age_s = float(now) - float(self._odom_seen_at)
            odom_source_age_s = float(now) - self._odom_stamp_ns * 1e-9
            odometry_ready = bool(
                -freshness_epsilon_s
                <= odom_receipt_age_s
                <= max_odom_gap_s
                and -freshness_epsilon_s
                <= odom_source_age_s
                <= max_odom_gap_s
                and stationarity.ready(
                    odom_sequence=self._odom_sequence,
                    odom_stamp_ns=self._odom_stamp_ns,
                    odom_seen_at_s=self._odom_seen_at,
                )
            )
        if (
            not vertical_edge
            and safe_duration_s + 1e-9 >= stationarity.quiet_window_s
            and odometry_ready
        ):
            MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
                self,
                'recovered',
            )
            return

    def _visual_servo(self, now: float) -> None:
        if getattr(
            self,
            '_visual_servo_vertical_recovery_started_at_s',
            None,
        ) is not None:
            MobileManipulationRuntime._visual_servo_vertical_recovery_tick(
                self,
                now,
            )
            return
        try:
            horizontal_edge, vertical_edge = self._tracked_target_edge_directions()
        except ValueError as error:
            detail = f'visual-servo image guard rejected target: {error}'
            self._publish_zero()
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if vertical_edge:
            try:
                MobileManipulationRuntime._begin_visual_servo_vertical_recovery(
                    self,
                    now,
                )
            except ValueError as error:
                MobileManipulationRuntime._fail_visual_servo_vertical_recovery(
                    self,
                    f'visual-servo vertical recovery setup failed: {error}',
                )
            return
        if horizontal_edge:
            detail = (
                'visual-servo target reached the horizontal image safety margin'
            )
            self._publish_zero()
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        output = self._approach.update(ApproachInput(
            stamp_s=now,
            approximate_range_m=(
                None if self._target_camera is None
                else float(self._target_camera[2])
            ),
            target_position_camera=self._target_camera,
            tracker_locked=self._perception_valid,
            navigation_speed_mps=self._nav_speed,
            navigation_yaw_rate_rps=(
                float('inf')
                if getattr(self, '_base_yaw_rate_rps', None) is None
                else self._base_yaw_rate_rps
            ),
            base_roll_rad=self._roll,
            base_pitch_rad=self._pitch,
            desired_depth_m=self._desired_depth,
        ))
        if output.cancel_navigation:
            self._cancel_nav_pub.publish(Bool(data=True))
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self._config.robot.platform_base_frame
        command.twist.linear.x = output.servo.linear_x
        command.twist.angular.z = output.servo.angular_z
        self._velocity_pub.publish(command)
        if output.phase is ApproachPhase.FAILED:
            detail = output.reason or 'visual approach failed'
            if not self._recover_precontact(FailureKind.VISUAL_APPROACH_FAILED, detail):
                self._apply_safety(self._core.fail(detail))
        elif output.phase is ApproachPhase.COMPLETE:
            self._publish_zero()
            try:
                stationary = self._coarse_nav_arrival_is_stationary(now)
            except (TimeoutError, ValueError) as error:
                detail = f'visual-servo handoff settle failed: {error}'
                if not self._recover_precontact(
                    FailureKind.VISUAL_APPROACH_FAILED,
                    detail,
                ):
                    self._apply_safety(self._core.fail(detail))
                return
            if not stationary:
                return
            self._coarse_nav_arrival_started_at_s = None
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_last_odom_sequence = None
            self._coarse_nav_arrival_last_odom_stamp_ns = None
            self._core.mark_servo_complete_for_reground()
            if self._task.stage.value == 'visual_approach':
                self._task.apply(StageResult.success())
            self._request_semantic_reground(now)

    def _coarse_nav_arrival_limits(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        settle_s = float(self.get_parameter('coarse_nav_arrival_settle_s').value)
        timeout_s = float(
            self.get_parameter('coarse_nav_arrival_stop_timeout_s').value,
        )
        max_linear = float(
            self.get_parameter('coarse_nav_arrival_max_linear_speed_mps').value,
        )
        max_angular = float(
            self.get_parameter('coarse_nav_arrival_max_angular_speed_rps').value,
        )
        max_xy_excursion = float(
            self.get_parameter('coarse_nav_arrival_max_xy_excursion_m').value,
        )
        max_yaw_excursion = float(
            self.get_parameter('coarse_nav_arrival_max_yaw_excursion_rad').value,
        )
        odom_max_age = float(self.get_parameter('work_pose_odom_max_age_s').value)
        max_odom_gap = float(
            self.get_parameter('coarse_nav_arrival_max_odom_gap_s').value,
        )
        if (
            not all(math.isfinite(value) and value > 0.0 for value in (
                settle_s,
                timeout_s,
                max_linear,
                max_angular,
                max_xy_excursion,
                max_yaw_excursion,
                odom_max_age,
                max_odom_gap,
            ))
            or timeout_s <= settle_s
            or max_odom_gap >= settle_s
            or max_yaw_excursion >= math.pi
        ):
            raise ValueError('coarse-navigation arrival settle limits are invalid')
        return (
            settle_s,
            timeout_s,
            max_linear,
            max_angular,
            max_xy_excursion,
            max_yaw_excursion,
            odom_max_age,
            max_odom_gap,
        )

    def _record_coarse_nav_arrival_motion_sample(
        self,
        *,
        received_at_s: float,
        odom_sequence: int,
        odom_stamp_ns: int,
        linear_speed_mps: float,
        angular_speed_rps: float,
        position_xy: tuple[float, float],
        yaw_rad: float,
    ) -> None:
        """Accumulate consecutive post-stop samples inside one SE(2) envelope."""
        started = getattr(self, '_coarse_nav_arrival_started_at_s', None)
        if started is None or received_at_s <= started:
            return
        try:
            (
                _settle_s,
                _timeout_s,
                max_linear,
                max_angular,
                max_xy_excursion,
                max_yaw_excursion,
                _odom_max_age,
                max_odom_gap,
            ) = MobileManipulationRuntime._coarse_nav_arrival_limits(self)
        except ValueError:
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_anchor_xy = None
            self._coarse_nav_arrival_anchor_yaw_rad = None
            return
        try:
            position = tuple(float(value) for value in position_xy)
            yaw = float(yaw_rad)
        except (TypeError, ValueError):
            position = ()
            yaw = float('nan')
        if (
            not math.isfinite(received_at_s)
            or isinstance(odom_sequence, bool)
            or odom_sequence <= 0
            or isinstance(odom_stamp_ns, bool)
            or odom_stamp_ns <= 0
            or not math.isfinite(linear_speed_mps)
            or not math.isfinite(angular_speed_rps)
            or len(position) != 2
            or not all(math.isfinite(value) for value in position)
            or not math.isfinite(yaw)
        ):
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_anchor_xy = None
            self._coarse_nav_arrival_anchor_yaw_rad = None
            return

        previous_sequence = self._coarse_nav_arrival_last_odom_sequence
        previous_stamp_ns = self._coarse_nav_arrival_last_odom_stamp_ns
        discontinuous = bool(
            previous_sequence is not None
            and (
                odom_sequence != previous_sequence + 1
                or previous_stamp_ns is None
                or odom_stamp_ns <= previous_stamp_ns
                or (odom_stamp_ns - previous_stamp_ns) * 1e-9 > max_odom_gap
            )
        )
        self._coarse_nav_arrival_last_odom_sequence = int(odom_sequence)
        self._coarse_nav_arrival_last_odom_stamp_ns = int(odom_stamp_ns)
        below_hard_motion_limit = bool(
            0.0 <= linear_speed_mps <= max_linear
            and 0.0 <= angular_speed_rps <= max_angular
        )
        anchor_xy = self._coarse_nav_arrival_anchor_xy
        anchor_yaw = self._coarse_nav_arrival_anchor_yaw_rad
        inside_pose_envelope = bool(
            anchor_xy is not None
            and anchor_yaw is not None
            and math.hypot(
                position[0] - anchor_xy[0],
                position[1] - anchor_xy[1],
            ) <= max_xy_excursion
            and abs(wrap_angle(yaw - anchor_yaw)) <= max_yaw_excursion
        )
        if discontinuous or not below_hard_motion_limit or not inside_pose_envelope:
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_anchor_xy = None
            self._coarse_nav_arrival_anchor_yaw_rad = None
        if not below_hard_motion_limit:
            return
        if self._coarse_nav_arrival_stable_since_s is None:
            self._coarse_nav_arrival_stable_since_s = float(received_at_s)
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = int(odom_stamp_ns)
            self._coarse_nav_arrival_anchor_xy = (position[0], position[1])
            self._coarse_nav_arrival_anchor_yaw_rad = yaw

    def _record_visual_search_settle_motion_sample(
        self,
        *,
        received_at_s: float,
        odom_sequence: int,
        odom_stamp_ns: int,
        linear_speed_mps: float,
        angular_speed_rps: float,
    ) -> None:
        """Feed only zero-command epochs into the continuous stillness gate."""
        if (
            getattr(self, '_visual_search_settle_reference', None) is None
            or bool(getattr(getattr(self, '_visual_search', None), 'active', False))
        ):
            return
        self._visual_search_stationarity.observe(
            received_at_s=received_at_s,
            odom_sequence=odom_sequence,
            odom_stamp_ns=odom_stamp_ns,
            linear_speed_mps=linear_speed_mps,
            angular_speed_rps=angular_speed_rps,
        )

    def _clear_visual_servo_vertical_recovery(self, reason: str) -> None:
        """Discard every sample owned by a previous vertical-edge stop."""
        self._visual_servo_vertical_recovery_started_at_s = None
        self._visual_servo_vertical_minimum_cloud_stamp_ns = None
        self._visual_servo_vertical_safe_start_cloud_stamp_ns = None
        self._visual_servo_vertical_safe_last_cloud_stamp_ns = None
        stationarity = getattr(
            self,
            '_visual_servo_vertical_stationarity',
            None,
        )
        if stationarity is not None:
            stationarity.clear(reason)

    def _record_visual_servo_vertical_motion_sample(
        self,
        *,
        received_at_s: float,
        odom_sequence: int,
        odom_stamp_ns: int,
        linear_speed_mps: float,
        angular_speed_rps: float,
    ) -> None:
        """Feed only post-edge odometry into the visual-servo stillness gate."""
        if not MobileManipulationRuntime._visual_servo_vertical_recovery_state_complete(
            self,
        ):
            return
        self._visual_servo_vertical_stationarity.observe(
            received_at_s=received_at_s,
            odom_sequence=odom_sequence,
            odom_stamp_ns=odom_stamp_ns,
            linear_speed_mps=linear_speed_mps,
            angular_speed_rps=angular_speed_rps,
        )

    def _coarse_nav_arrival_is_stationary(self, now: float) -> bool:
        """Require a bounded interval of fresh, measured platform stillness."""
        (
            settle_s,
            timeout_s,
            _max_linear,
            _max_angular,
            _max_xy_excursion,
            _max_yaw_excursion,
            odom_max_age,
            _max_odom_gap,
        ) = MobileManipulationRuntime._coarse_nav_arrival_limits(self)
        if not math.isfinite(now):
            raise ValueError('coarse-navigation arrival time is invalid')
        started = self._coarse_nav_arrival_started_at_s
        if started is None:
            started = now
            self._coarse_nav_arrival_started_at_s = now
            self._coarse_nav_arrival_stable_since_s = None
            self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
            self._coarse_nav_arrival_last_odom_sequence = None
            self._coarse_nav_arrival_last_odom_stamp_ns = None
            self._coarse_nav_arrival_anchor_xy = None
            self._coarse_nav_arrival_anchor_yaw_rad = None
        elif now < started:
            raise ValueError('coarse-navigation arrival clock moved backwards')

        odom_seen = self._odom_seen_at
        stable_since = self._coarse_nav_arrival_stable_since_s
        stable_stamp_ns = self._coarse_nav_arrival_stable_start_odom_stamp_ns
        last_sequence = self._coarse_nav_arrival_last_odom_sequence
        last_stamp_ns = self._coarse_nav_arrival_last_odom_stamp_ns
        fresh_window = bool(
            odom_seen is not None
            and math.isfinite(odom_seen)
            and 0.0 <= now - odom_seen <= odom_max_age
            and stable_since is not None
            and stable_stamp_ns is not None
            and last_sequence is not None
            and last_sequence == self._odom_sequence
            and last_stamp_ns is not None
            and last_stamp_ns == self._odom_stamp_ns
        )
        if fresh_window:
            if odom_seen < stable_since or last_stamp_ns < stable_stamp_ns:
                raise ValueError('coarse-navigation stable clock moved backwards')
            if (
                odom_seen - stable_since >= settle_s
                and (last_stamp_ns - stable_stamp_ns) * 1e-9 >= settle_s
            ):
                return True
        if now - started > timeout_s:
            raise TimeoutError(
                'platform did not become stationary before the arrival deadline',
            )
        return False

    def _maybe_finish_coarse_nav(self) -> None:
        if self._core.phase is not RuntimePhase.COARSE_NAV:
            return
        observed_near = (
            self._work_pose is None
            and
            self._target_camera is not None
            and float(self._target_camera[2])
            <= self._config.approach.near_stage_threshold_m
        )
        if not (self._coarse_nav_ready or observed_near):
            return
        now = self._now_s()
        # Hold a higher-priority zero command while the navigation owner is
        # still alive. Cancel only after measured inertia has settled so the
        # adapter cannot report a cancellation failure during this handoff.
        self._apply_safety(SafetyAction(
            stop_base=True,
        ))
        try:
            stationary = self._coarse_nav_arrival_is_stationary(now)
        except (TimeoutError, ValueError) as error:
            detail = f'coarse-navigation arrival settle failed: {error}'
            if not self._recover_precontact(FailureKind.NAV_BLOCKED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if not stationary:
            return
        MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
            self,
            False,
        )
        self._apply_safety(SafetyAction(
            stop_base=True,
            cancel_navigation=True,
        ))
        if self._task.stage.value == 'coarse_nav':
            self._task.apply(StageResult.success())
        self._core.mark_coarse_ready()
        self._coarse_nav_perception_loss_detail = ''
        self._coarse_nav_arrival_started_at_s = None
        self._coarse_nav_arrival_stable_since_s = None
        self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
        self._coarse_nav_arrival_last_odom_sequence = None
        self._coarse_nav_arrival_last_odom_stamp_ns = None
        self._coarse_nav_arrival_anchor_xy = None
        self._coarse_nav_arrival_anchor_yaw_rad = None
        try:
            MobileManipulationRuntime._begin_near_view_settle(self, now)
        except ValueError as error:
            detail = f'near-view setup failed: {error}'
            if not self._recover_precontact(
                FailureKind.VISUAL_APPROACH_FAILED,
                detail,
            ):
                self._apply_safety(self._core.fail(detail))

    def _grounding_observation_authorized(self, synchronized: Any | None) -> bool:
        """Accept one task request, producer epoch, generation, and source frame."""
        request_id = self._required_perception_request_id
        producer_epoch = self._bound_perception_producer_epoch
        required = self._required_perception_generation
        generation = self._bound_perception_generation
        if (
            synchronized is None
            or request_id is None
            or self._bound_perception_request_id != request_id
            or producer_epoch is None
            or required is None
            or generation is None
            or generation != required
            or not self._perception_valid
            or self._valid_perception_request_id != request_id
            or self._valid_perception_producer_epoch != producer_epoch
            or self._valid_perception_generation != generation
            or self._affordance is None
            or self._affordance_request_id != request_id
            or self._affordance_producer_epoch != producer_epoch
            or self._affordance_generation != generation
            or self._valid_observation_stamp_ns is None
            or not self._valid_observation_frame_id
        ):
            return False
        geometry = (
            (self._target_stamp_ns, self._target_frame_id, self._target_camera),
            (
                self._target_cloud_stamp_ns,
                self._target_cloud_frame_id,
                self._target_cloud,
            ),
            (
                self._scene_cloud_stamp_ns,
                self._scene_cloud_frame_id,
                self._scene_cloud,
            ),
        )
        return all(
            value is not None
            and stamp_ns == self._valid_observation_stamp_ns
            and frame_id == self._valid_observation_frame_id
            for stamp_ns, frame_id, value in geometry
        )

    def _refresh_near_view_desired_depth(self) -> None:
        """Project the frozen work target through the measured near-view camera."""
        work_pose = getattr(self, '_work_pose', None)
        if work_pose is None:
            return
        if not isinstance(work_pose, dict):
            raise ValueError('work pose is malformed')
        predicted_target = np.asarray(
            work_pose.get('predicted_target_position_piper'),
            dtype=float,
        )
        camera_origin = np.asarray(self._camera_origin_piper, dtype=float)
        camera_rotation = np.asarray(self._camera_rotation_piper, dtype=float)
        if (
            predicted_target.shape != (3,)
            or camera_origin.shape != (3,)
            or camera_rotation.shape != (3, 3)
            or not np.all(np.isfinite(predicted_target))
            or not np.all(np.isfinite(camera_origin))
            or not np.all(np.isfinite(camera_rotation))
            or not np.allclose(
                camera_rotation.T @ camera_rotation,
                np.eye(3),
                atol=1e-3,
            )
            or np.linalg.det(camera_rotation) <= 0.0
        ):
            raise ValueError('near-view camera projection is invalid')
        predicted_camera = camera_rotation.T @ (
            predicted_target - camera_origin
        )
        raw_depth = float(predicted_camera[2])
        minimum = float(self._config.standoff.min_camera_depth_m)
        maximum = float(self._config.standoff.max_camera_depth_m)
        if (
            not math.isfinite(raw_depth)
            or raw_depth <= 0.0
            or not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum <= 0.0
            or maximum < minimum
        ):
            raise ValueError('near-view desired depth is invalid')
        desired_depth = float(np.clip(raw_depth, minimum, maximum))
        self._desired_depth = desired_depth
        work_pose['desired_camera_depth_m'] = desired_depth
        work_pose['near_view_raw_camera_depth_m'] = raw_depth

    def _grounding_tick(self, synchronized: Any | None) -> None:
        """Start the prospective plan from one generation-correlated observation."""
        if (
            not self._grounding_observation_authorized(synchronized)
            or self._future is not None
        ):
            return
        try:
            edge_direction = self._horizontal_edge_direction()
            vertical_direction = self._vertical_edge_direction()
            if vertical_direction:
                self._visual_search_edge_direction = 0
                if not self._recover_precontact(
                    FailureKind.VISUAL_APPROACH_FAILED,
                    'grounded target is outside the vertical safe field of view',
                ):
                    self._apply_safety(self._core.fail(
                        'target viewpoint recovery was rejected',
                    ))
                self._publish_status()
                return
            if edge_direction:
                self._visual_search_edge_direction = edge_direction
                if not self._recover_precontact(
                    FailureKind.NOT_FOUND,
                    'grounded target is outside the horizontal safe field of view',
                ):
                    self._apply_safety(self._core.fail(
                        'target recenter recovery was rejected',
                    ))
                self._publish_status()
                return
            assert synchronized is not None
            observation = self._semantic_observation(
                synchronized.serial, synchronized.stamp_s,
            )
            self._visual_search.reset()
            self._visual_search_error_rad = None
            self._visual_search_reason = ''
            self._core.phase = RuntimePhase.STANDOFF
            self._start_planning('standoff', observation)
        except Exception as error:
            if not self._recover_precontact(
                FailureKind.TARGET_LOST,
                f'observation rejected: {error}',
            ):
                self._apply_safety(self._core.fail(
                    f'observation rejected: {error}',
                ))

    def _wait_fresh_observation_tick(self, synchronized: Any | None) -> None:
        """Plan again only from a fresh frame authorized by the same request."""
        if (
            not self._grounding_observation_authorized(synchronized)
            or self._future is not None
        ):
            return
        assert synchronized is not None
        required = self._core.required_replan_serial or 1
        if synchronized.serial < required:
            return
        try:
            observation = self._semantic_observation(
                synchronized.serial,
                synchronized.stamp_s,
            )
            self._core.begin_replan(synchronized.serial)
            self._start_planning('pregrasp', observation)
        except Exception as error:
            if not self._recover_precontact(
                FailureKind.TARGET_LOST,
                f'post-servo observation rejected: {error}',
            ):
                self._apply_safety(self._core.fail(
                    f'post-servo observation rejected: {error}',
                ))

    def _publish_grounding_request(self) -> None:
        """Publish a unique task-owned request without assuming producer state."""
        request_id = uuid.uuid4().hex
        if self._core.phase is RuntimePhase.PLACE_GROUNDING:
            grounding_scope = 'place_support'
        elif self._topic_value('place_mode') == 'carry_only':
            grounding_scope = 'grasp_only'
        else:
            grounding_scope = 'grasp_for_place'
        self._clear_semantic_observation_cache()
        self._required_perception_request_id = request_id
        self._required_grounding_scope = grounding_scope
        self._required_perception_generation = None
        self._required_affordance_generation = None
        self._handled_perception_failure = None
        self._grounding_pub.publish(String(data=json.dumps({
            'schema': 'z_manip.grounding_request.v2',
            'request_id': request_id,
            'instruction': self._core.instruction,
            'scope': grounding_scope,
        }, separators=(',', ':'))))

    def _request_semantic_reground(self, now: float) -> None:
        """Invalidate the old semantic anchor and request a new VLM generation."""
        MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
            self,
            False,
        )
        self._reground_started_at = float(now)
        self._reground_last_tick_at = float(now)
        self._publish_grounding_request()

    def _horizontal_edge_direction(
        self,
        *,
        margin_ratio: float | None = None,
    ) -> int:
        """Return the odometry-yaw direction needed to move a box off an edge."""
        if self._affordance is None:
            raise ValueError('target affordance is unavailable')
        target = self._affordance.get('target')
        box = target.get('bbox_xyxy_normalized') if isinstance(target, dict) else None
        if not isinstance(box, list):
            raise ValueError('target affordance bbox is malformed')
        margin = float(
            self.get_parameter('visual_search_horizontal_margin_ratio').value
            if margin_ratio is None
            else margin_ratio
        )
        return horizontal_edge_direction(box, margin_ratio=margin)

    def _vertical_edge_direction(
        self,
        *,
        margin_ratio: float | None = None,
    ) -> int:
        """Return the camera-pitch direction required by a clipped VLM box."""
        if self._affordance is None:
            raise ValueError('target affordance is unavailable')
        target = self._affordance.get('target')
        box = target.get('bbox_xyxy_normalized') if isinstance(target, dict) else None
        if not isinstance(box, list):
            raise ValueError('target affordance bbox is malformed')
        margin = float(
            self.get_parameter('visual_search_vertical_margin_ratio').value
            if margin_ratio is None
            else margin_ratio
        )
        return vertical_edge_direction(box, margin_ratio=margin)

    def _tracked_target_edge_directions(
        self,
        *,
        margin_ratio: float | None = None,
    ) -> tuple[int, int]:
        """Return live mask edge violations in horizontal and vertical axes."""
        if self._target_uv is None:
            raise ValueError(
                'tracked target pointcloud lacks u/v image coordinates',
            )
        if self._image_size is None:
            raise ValueError('camera image size is unavailable')
        uv = np.asarray(self._target_uv, dtype=float)
        width, height = self._image_size
        if (
            uv.ndim != 2
            or uv.shape[1] != 2
            or len(uv) == 0
            or not np.all(np.isfinite(uv))
            or width <= 0
            or height <= 0
        ):
            raise ValueError('tracked target image coordinates are invalid')
        u_min, v_min = np.min(uv, axis=0)
        u_max, v_max = np.max(uv, axis=0)
        box = (
            float(u_min) / float(width),
            float(v_min) / float(height),
            (float(u_max) + 1.0) / float(width),
            (float(v_max) + 1.0) / float(height),
        )
        margin = float(
            self.get_parameter('visual_servo_image_margin_ratio').value
            if margin_ratio is None
            else margin_ratio
        )
        return (
            horizontal_edge_direction(box, margin_ratio=margin),
            vertical_edge_direction(box, margin_ratio=margin),
        )

    def _begin_visual_search(self, now: float) -> None:
        self._pose_settle_until = None
        self._pose_settle_started_at = None
        self._pose_settle_last_tick_at = None
        self._visual_search_settle_reference = None
        MobileManipulationRuntime._clear_visual_search_stationarity(
            self,
            'new_search_view',
        )
        if (
            self._yaw is None
            or self._position_xy is None
            or self._odom_seen_at is None
        ):
            self._apply_safety(self._core.fail('visual search has no odometry pose'))
            return
        max_age = float(self.get_parameter('visual_search_odom_timeout_s').value)
        if not math.isfinite(max_age) or max_age <= 0.0:
            self._apply_safety(self._core.fail('visual search odometry timeout is invalid'))
            return
        if now - self._odom_seen_at > max_age:
            self._apply_safety(self._core.fail('visual search odometry yaw is stale'))
            return
        try:
            started = self._visual_search.start(
                self._yaw,
                now_s=now,
                current_position_xy=self._position_xy,
                image_edge_direction=self._visual_search_edge_direction,
            )
        except (RuntimeError, ValueError) as error:
            self._apply_safety(self._core.fail(f'visual search setup failed: {error}'))
            return
        if not started:
            self._apply_safety(self._core.fail(
                'visual search exhausted its configured yaw coverage',
            ))
            return
        self._core.begin_visual_search()
        self._visual_search_pending = False
        self._visual_search_error_rad = None
        self._visual_search_active_pub.publish(Bool(data=True))

    def _visual_search_tick(self, now: float) -> None:
        if (
            self._yaw is None
            or self._position_xy is None
            or self._odom_seen_at is None
        ):
            self._apply_safety(self._core.fail('visual search lost odometry pose'))
            return
        max_age = float(self.get_parameter('visual_search_odom_timeout_s').value)
        if now - self._odom_seen_at > max_age:
            self._apply_safety(self._core.fail('visual search odometry yaw became stale'))
            return
        try:
            update = self._visual_search.update(
                self._yaw,
                now_s=now,
                current_position_xy=self._position_xy,
                measured_angular_speed_rps=getattr(
                    self,
                    '_base_yaw_rate_rps',
                    None,
                ),
            )
        except (RuntimeError, ValueError) as error:
            self._apply_safety(self._core.fail(f'visual search update failed: {error}'))
            return
        self._visual_search_error_rad = update.error_rad
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self._config.robot.platform_base_frame
        command.twist.linear.x = update.linear_x
        command.twist.linear.y = update.linear_y
        command.twist.angular.z = update.angular_z
        self._visual_search_active_pub.publish(Bool(data=True))
        self._velocity_pub.publish(command)
        if update.timed_out:
            self._visual_search_active_pub.publish(Bool(data=False))
            timeout_phase = update.timeout_phase or 'unknown'
            timeout_s = (
                self._visual_search.config.position_hold_timeout_s
                if timeout_phase == 'position_hold'
                else self._visual_search.allocated_timeout_s
            )
            self._apply_safety(self._core.fail(
                f'visual search {timeout_phase} timed out after '
                f'{timeout_s:.2f}s with {update.error_rad:.4f}rad yaw error and '
                f'{update.planar_drift_m:.4f}m planar error',
            ))
            return
        if update.drift_exceeded:
            self._visual_search_active_pub.publish(Bool(data=False))
            self._apply_safety(self._core.fail(
                f'visual search planar drift {update.planar_drift_m:.3f}m exceeds '
                f'{self._visual_search.config.max_planar_drift_m:.3f}m',
            ))
            return
        if update.complete:
            self._publish_zero()
            anchor = self._visual_search.position_anchor_xy
            target_yaw = self._visual_search.target_yaw_rad
            if (
                anchor is None
                or target_yaw is None
                or not math.isfinite(float(target_yaw))
                or self._odom_stamp_ns is None
            ):
                self._apply_safety(self._core.fail(
                    'visual search completed without a finite settle reference',
                ))
                return
            settle = float(self.get_parameter('visual_search_settle_s').value)
            if not math.isfinite(settle) or settle <= 0.0:
                self._apply_safety(self._core.fail('visual search settle time is invalid'))
                return
            control_period = float(self.get_parameter('control_period_s').value)
            if not math.isfinite(control_period) or control_period <= 0.0:
                self._apply_safety(self._core.fail(
                    'visual search settle control period is invalid',
                ))
                return
            reacquire_budget = self._visual_search.config.settle_reacquire_budget_s
            stationary_budget = (
                settle
                + self._visual_search.config.stationary_wait_timeout_s
                + control_period
            )
            if (
                not math.isfinite(stationary_budget)
                or stationary_budget
                < self._visual_search.config.stationary_quiet_window_s
            ):
                self._apply_safety(self._core.fail(
                    'visual search stationary budget cannot contain its quiet window',
                ))
                return
            correction_deadline = float(now) + reacquire_budget
            stationary_deadline = (
                float(now) + stationary_budget
            )
            absolute_deadline = correction_deadline + stationary_budget
            if not all(math.isfinite(value) for value in (
                correction_deadline,
                stationary_deadline,
                absolute_deadline,
            )):
                self._apply_safety(self._core.fail(
                    'visual search settle deadlines are invalid',
                ))
                return
            self._core.mark_visual_search_complete()
            self._visual_search_edge_direction = 0
            self._visual_search_settle_reference = _VisualSearchSettleReference(
                position_anchor_xy=(float(anchor[0]), float(anchor[1])),
                target_yaw_rad=float(target_yaw),
                started_at_s=float(now),
                stop_started_at_s=float(now),
                minimum_odom_sequence=int(self._odom_sequence),
                minimum_odom_stamp_ns=int(self._odom_stamp_ns),
                correction_deadline_s=correction_deadline,
                absolute_deadline_s=absolute_deadline,
                stationary_deadline_s=stationary_deadline,
                reacquire_count=0,
            )
            try:
                self._visual_search_stationarity.reset(
                    stop_received_at_s=float(now),
                    minimum_odom_sequence=int(self._odom_sequence),
                    minimum_odom_stamp_ns=int(self._odom_stamp_ns),
                )
            except ValueError as error:
                MobileManipulationRuntime._fail_pose_settle(
                    self,
                    f'visual search quiet-window setup failed: {error}',
                )
                return
            self._pose_settle_started_at = now
            self._pose_settle_last_tick_at = now
            self._pose_settle_until = now + settle

    def _clear_pose_settle_state(self) -> None:
        """Clear every timer and retained observation for one settle."""
        self._pose_settle_until = None
        self._pose_settle_started_at = None
        self._pose_settle_last_tick_at = None
        self._visual_search_settle_reference = None
        MobileManipulationRuntime._clear_visual_search_stationarity(
            self,
            'pose_settle_cleared',
        )

    def _clear_visual_search_stationarity(self, reason: str) -> None:
        """Clear the optional quiet-window state in runtime and unit harnesses."""
        stationarity = getattr(self, '_visual_search_stationarity', None)
        if stationarity is not None:
            stationarity.clear(reason)

    def _fail_pose_settle(self, reason: str) -> None:
        """Fail closed after atomically clearing the settle substate."""
        MobileManipulationRuntime._clear_pose_settle_state(self)
        self._apply_safety(self._core.fail(reason))

    def _visual_search_settle_correction_deadline(
        self,
        reference: _VisualSearchSettleReference,
    ) -> tuple[float, float, float]:
        """Validate and return the immutable pose-correction deadline."""
        settle = float(self.get_parameter('visual_search_settle_s').value)
        control_period = float(self.get_parameter('control_period_s').value)
        correction_deadline = float(reference.correction_deadline_s)
        if (
            not math.isfinite(settle)
            or settle <= 0.0
            or not math.isfinite(control_period)
            or control_period <= 0.0
            or not math.isfinite(correction_deadline)
            or correction_deadline < reference.started_at_s
            or not math.isfinite(reference.absolute_deadline_s)
            or reference.absolute_deadline_s <= correction_deadline
            or reference.stationary_deadline_s > reference.absolute_deadline_s
        ):
            raise ValueError('visual search settle correction budget is invalid')
        return correction_deadline, settle, control_period

    def _reacquire_visual_search_settle(
        self,
        now: float,
        reference: _VisualSearchSettleReference,
        yaw_error: float,
    ) -> None:
        """Re-arm one retained viewpoint inside its immutable correction budget."""
        if self._yaw is None or self._position_xy is None:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle cannot reacquire without a measured pose',
            )
            return
        try:
            correction_deadline, _settle, _control_period = (
                MobileManipulationRuntime._visual_search_settle_correction_deadline(
                    self,
                    reference,
                )
            )
            if float(now) >= correction_deadline:
                raise ValueError(
                    'visual search settle pose correction deadline expired '
                    'before the reserved stationary dwell',
                )
            self._visual_search.reacquire_current_target(
                self._yaw,
                now_s=now,
                current_position_xy=self._position_xy,
                deadline_s=correction_deadline,
            )
        except (RuntimeError, ValueError) as error:
            MobileManipulationRuntime._fail_pose_settle(self, str(error))
            return
        self._visual_search_settle_reference = _VisualSearchSettleReference(
            position_anchor_xy=reference.position_anchor_xy,
            target_yaw_rad=reference.target_yaw_rad,
            started_at_s=reference.started_at_s,
            stop_started_at_s=reference.stop_started_at_s,
            minimum_odom_sequence=reference.minimum_odom_sequence,
            minimum_odom_stamp_ns=reference.minimum_odom_stamp_ns,
            correction_deadline_s=reference.correction_deadline_s,
            absolute_deadline_s=reference.absolute_deadline_s,
            stationary_deadline_s=reference.stationary_deadline_s,
            reacquire_count=reference.reacquire_count + 1,
        )
        self._visual_search_stationarity.clear('pose_reacquire')
        self._visual_search_error_rad = float(yaw_error)
        self._visual_search_active_pub.publish(Bool(data=True))

    def _monitor_visual_search_settle_rebound(self, now: float) -> bool:
        """Interrupt the fixed dwell when measured XY recoil crosses hysteresis."""
        reference = self._visual_search_settle_reference
        if (
            reference is None
            or bool(getattr(self._visual_search, 'active', False))
        ):
            return False
        if self._position_xy is None or self._yaw is None:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle lost its measured rebound pose',
            )
            return True
        drift = math.hypot(
            self._position_xy[0] - reference.position_anchor_xy[0],
            self._position_xy[1] - reference.position_anchor_xy[1],
        )
        self._visual_search.planar_drift_m = float(drift)
        maximum = self._visual_search.config.max_planar_drift_m
        if not math.isfinite(drift) or drift > maximum:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                f'visual search settle planar drift {drift:.3f}m exceeds '
                f'{maximum:.3f}m',
            )
            return True
        if drift < self._visual_search.config.moving_rebound_reacquire_m:
            return False
        yaw_error = wrap_angle(reference.target_yaw_rad - self._yaw)
        MobileManipulationRuntime._reacquire_visual_search_settle(
            self,
            now,
            reference,
            yaw_error,
        )
        return True

    def _visual_search_settle_correction_tick(self, now: float) -> None:
        """Close parking rebound on the retained target inside its hard deadline."""
        reference = self._visual_search_settle_reference
        settle_started = self._pose_settle_started_at
        settle_last_tick = self._pose_settle_last_tick_at
        if (
            reference is None
            or self._yaw is None
            or self._position_xy is None
            or self._odom_seen_at is None
            or self._odom_stamp_ns is None
            or settle_started is None
            or settle_last_tick is None
            or not bool(getattr(self._visual_search, 'active', False))
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle correction lost timing, odometry, or target state',
            )
            return
        try:
            correction_deadline, settle, control_period = (
                MobileManipulationRuntime._visual_search_settle_correction_deadline(
                    self,
                    reference,
                )
            )
        except ValueError as error:
            MobileManipulationRuntime._fail_pose_settle(self, str(error))
            return
        if (
            not math.isfinite(float(now))
            or not math.isfinite(float(settle_started))
            or not math.isfinite(float(settle_last_tick))
            or float(settle_last_tick) < float(settle_started)
            or float(now) < float(settle_last_tick)
            or float(now) < reference.started_at_s
            or not math.isfinite(float(self._yaw))
            or not all(math.isfinite(float(value)) for value in self._position_xy)
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle correction pose or clock is invalid',
            )
            return
        if float(now) >= correction_deadline:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle pose correction deadline expired before '
                'the reserved stationary dwell',
            )
            return
        max_age = float(self.get_parameter('visual_search_odom_timeout_s').value)
        odom_age = float(now) - self._odom_seen_at
        if (
            not math.isfinite(max_age)
            or max_age <= 0.0
            or not math.isfinite(odom_age)
            or odom_age < 0.0
            or odom_age > max_age
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle correction odometry receipt is stale',
            )
            return
        self._pose_settle_last_tick_at = float(now)
        try:
            update = self._visual_search.update(
                self._yaw,
                now_s=now,
                current_position_xy=self._position_xy,
                measured_angular_speed_rps=getattr(
                    self,
                    '_base_yaw_rate_rps',
                    None,
                ),
            )
        except (RuntimeError, ValueError) as error:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                f'visual search settle correction failed: {error}',
            )
            return
        self._visual_search_error_rad = update.error_rad
        if update.timed_out:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle pose correction timed out inside its '
                f'fixed deadline with {update.error_rad:.4f}rad error',
            )
            return
        if update.drift_exceeded:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                f'visual search settle correction planar drift '
                f'{update.planar_drift_m:.3f}m exceeds '
                f'{self._visual_search.config.max_planar_drift_m:.3f}m',
            )
            return
        if update.complete:
            self._publish_zero()
            settle_until = float(now) + settle
            stationary_deadline = min(
                reference.absolute_deadline_s,
                float(now)
                + settle
                + self._visual_search.config.stationary_wait_timeout_s
                + control_period,
            )
            if settle_until > stationary_deadline + 1e-9:
                MobileManipulationRuntime._fail_pose_settle(
                    self,
                    'visual search settle pose correction left no bounded time '
                    'for a fresh stationary sample',
                )
                return
            self._visual_search_settle_reference = _VisualSearchSettleReference(
                position_anchor_xy=reference.position_anchor_xy,
                target_yaw_rad=reference.target_yaw_rad,
                started_at_s=reference.started_at_s,
                stop_started_at_s=float(now),
                minimum_odom_sequence=int(self._odom_sequence),
                minimum_odom_stamp_ns=int(self._odom_stamp_ns),
                correction_deadline_s=reference.correction_deadline_s,
                absolute_deadline_s=reference.absolute_deadline_s,
                stationary_deadline_s=stationary_deadline,
                reacquire_count=reference.reacquire_count,
            )
            try:
                self._visual_search_stationarity.reset(
                    stop_received_at_s=float(now),
                    minimum_odom_sequence=int(self._odom_sequence),
                    minimum_odom_stamp_ns=int(self._odom_stamp_ns),
                )
            except ValueError as error:
                MobileManipulationRuntime._fail_pose_settle(
                    self,
                    f'visual search quiet-window reset failed: {error}',
                )
                return
            self._pose_settle_started_at = float(now)
            self._pose_settle_last_tick_at = float(now)
            self._pose_settle_until = settle_until
            return

        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self._config.robot.platform_base_frame
        command.twist.linear.x = update.linear_x
        command.twist.linear.y = update.linear_y
        command.twist.angular.z = update.angular_z
        self._visual_search_active_pub.publish(Bool(data=True))
        self._velocity_pub.publish(command)

    def _finish_pose_settle(self, now: float) -> None:
        """Complete LOOKOUT or validate the final visual-search settle pose."""
        reference = self._visual_search_settle_reference
        if reference is None:
            MobileManipulationRuntime._clear_pose_settle_state(self)
            self._core.mark_pose_settled()
            self._publish_grounding_request()
            return
        if (
            self._position_xy is None
            or self._yaw is None
            or self._odom_seen_at is None
            or self._odom_stamp_ns is None
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle lost odometry pose',
            )
            return
        settle_until = self._pose_settle_until
        if (
            not math.isfinite(float(now))
            or not math.isfinite(reference.started_at_s)
            or not math.isfinite(reference.stop_started_at_s)
            or not math.isfinite(reference.correction_deadline_s)
            or not math.isfinite(reference.absolute_deadline_s)
            or not math.isfinite(reference.stationary_deadline_s)
            or settle_until is None
            or not math.isfinite(settle_until)
            or reference.stationary_deadline_s < settle_until
            or reference.stationary_deadline_s > reference.absolute_deadline_s
            or reference.correction_deadline_s < reference.started_at_s
            or reference.absolute_deadline_s <= reference.correction_deadline_s
            or float(now) < reference.started_at_s
            or float(now) < reference.stop_started_at_s
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle clock moved backwards or timing is invalid',
            )
            return
        if (
            self._odom_sequence <= reference.minimum_odom_sequence
            or self._odom_stamp_ns <= reference.minimum_odom_stamp_ns
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle has no post-stop odometry sample',
            )
            return
        max_age = float(self.get_parameter('visual_search_odom_timeout_s').value)
        if not math.isfinite(max_age) or max_age <= 0.0:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle odometry timeout is invalid',
            )
            return
        odom_age = float(now) - self._odom_seen_at
        if not math.isfinite(odom_age) or odom_age < 0.0 or odom_age > max_age:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle odometry receipt is stale',
            )
            return
        source_age = float(now) - float(self._odom_stamp_ns) * 1e-9
        if (
            not math.isfinite(source_age)
            or source_age < 0.0
            or source_age > max_age
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle odometry source stamp is stale',
            )
            return
        final_drift = math.hypot(
            self._position_xy[0] - reference.position_anchor_xy[0],
            self._position_xy[1] - reference.position_anchor_xy[1],
        )
        max_drift = self._visual_search.config.max_planar_drift_m
        if not math.isfinite(final_drift) or final_drift > max_drift:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                f'visual search settle planar drift {final_drift:.3f}m exceeds '
                f'{max_drift:.3f}m',
            )
            return
        if (
            self._base_linear_speed_mps is None
            or self._base_angular_speed_rps is None
            or not math.isfinite(self._base_linear_speed_mps)
            or not math.isfinite(self._base_angular_speed_rps)
        ):
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search settle odometry speed is unavailable or non-finite',
            )
            return
        moving = bool(
            self._base_linear_speed_mps
            > self._visual_search.config.settle_max_linear_speed_mps
            or self._base_angular_speed_rps
            > self._visual_search.config.settle_max_angular_speed_rps
        )
        final_yaw_error = wrap_angle(reference.target_yaw_rad - self._yaw)
        yaw_outside_completion = (
            not math.isfinite(final_yaw_error)
            or abs(final_yaw_error)
            > self._visual_search.config.settle_yaw_tolerance_rad
        )
        position_outside_completion = (
            final_drift
            > self._visual_search.config.position_completion_tolerance_m
        )
        quiet_ready = self._visual_search_stationarity.ready(
            odom_sequence=self._odom_sequence,
            odom_stamp_ns=self._odom_stamp_ns,
            odom_seen_at_s=self._odom_seen_at,
        )
        if float(now) > reference.absolute_deadline_s:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search absolute settle deadline expired',
            )
            return
        defer_position_reacquire = False
        moderate_moving_rebound = bool(
            position_outside_completion
            and not yaw_outside_completion
            and not quiet_ready
            and final_drift
            <= self._visual_search.config.moving_rebound_reacquire_m
            and float(now) < reference.absolute_deadline_s
        )
        if moderate_moving_rebound:
            correction_span = max(
                0.0,
                reference.correction_deadline_s - reference.stop_started_at_s,
            )
            rebound_wait_budget = min(
                self._visual_search.config.stationary_wait_timeout_s,
                0.5 * correction_span,
            )
            rebound_wait_deadline = (
                reference.stop_started_at_s + rebound_wait_budget
            )
            defer_position_reacquire = float(now) < rebound_wait_deadline
        if defer_position_reacquire:
            return
        if yaw_outside_completion or position_outside_completion:
            MobileManipulationRuntime._reacquire_visual_search_settle(
                self,
                now,
                reference,
                final_yaw_error,
            )
            return
        if not quiet_ready and float(now) < reference.absolute_deadline_s:
            return
        if not quiet_ready and moving:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search absolute settle deadline expired while platform is moving: '
                f'linear={self._base_linear_speed_mps:.4f}m/s '
                f'(limit {self._visual_search.config.settle_max_linear_speed_mps:.4f}), '
                f'angular={self._base_angular_speed_rps:.4f}rad/s '
                f'(limit {self._visual_search.config.settle_max_angular_speed_rps:.4f})',
            )
            return
        if not quiet_ready:
            MobileManipulationRuntime._fail_pose_settle(
                self,
                'visual search absolute settle deadline expired before a continuous '
                f'{self._visual_search.config.stationary_quiet_window_s:.3f}s '
                'quiet window; observed '
                f'{self._visual_search_stationarity.stable_duration_s:.3f}s',
            )
            return
        self._visual_search_active_pub.publish(Bool(data=False))
        MobileManipulationRuntime._clear_pose_settle_state(self)
        self._core.mark_pose_settled()
        self._publish_grounding_request()

    def _reground_tick(self, now: float, synchronized: Any | None) -> None:
        if self._reground_started_at is None:
            self._apply_safety(self._core.fail('semantic re-grounding was not requested'))
            return
        started_at = float(self._reground_started_at)
        last_tick_at = self._reground_last_tick_at
        timeout_s = float(self.get_parameter('semantic_reground_timeout_s').value)
        if (
            last_tick_at is None
            or not math.isfinite(float(now))
            or not math.isfinite(started_at)
            or not math.isfinite(float(last_tick_at))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
            or float(last_tick_at) < started_at
        ):
            self._apply_safety(self._core.fail(
                'semantic re-grounding timing contract is invalid',
            ))
            return
        if float(now) < float(last_tick_at):
            self._apply_safety(self._core.fail(
                'semantic re-grounding clock moved backwards',
            ))
            return
        self._reground_last_tick_at = float(now)
        if float(now) - started_at > timeout_s:
            if not self._recover_precontact(
                FailureKind.TARGET_LOST,
                'semantic re-grounding timed out',
            ):
                self._apply_safety(self._core.fail('semantic re-grounding timed out'))
            return
        if not self._grounding_observation_authorized(synchronized):
            return
        assert synchronized is not None
        if self._core.phase is RuntimePhase.NEAR_GROUNDING:
            try:
                # Admit the freshly grounded target at the same hard image
                # boundary enforced before every servo command.  The wider
                # search margin remains the re-entry boundary after an edge
                # stop, preserving hysteresis without rejecting safe pixels.
                servo_margin = float(
                    self.get_parameter(
                        'visual_servo_image_margin_ratio',
                    ).value,
                )
                edge_direction = self._horizontal_edge_direction(
                    margin_ratio=servo_margin,
                )
                vertical_direction = self._vertical_edge_direction(
                    margin_ratio=servo_margin,
                )
            except (TypeError, ValueError) as error:
                detail = f'near-field edge gate rejected observation: {error}'
                if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                    self._apply_safety(self._core.fail(detail))
                return
            if vertical_direction:
                self._visual_search_edge_direction = 0
                if not self._recover_precontact(
                    FailureKind.VISUAL_APPROACH_FAILED,
                    'near-field target is outside the vertical safe field of view',
                ):
                    self._apply_safety(self._core.fail(
                        'near-field target viewpoint recovery was rejected',
                    ))
                self._publish_status()
                return
            if edge_direction:
                self._visual_search_edge_direction = edge_direction
                if not self._recover_precontact(
                    FailureKind.NOT_FOUND,
                    'near-field target is outside the horizontal safe field of view',
                ):
                    self._apply_safety(self._core.fail(
                        'near-field target recenter recovery was rejected',
                    ))
                self._publish_status()
                return
            try:
                MobileManipulationRuntime._refresh_near_view_desired_depth(self)
            except (TypeError, ValueError) as error:
                detail = f'near-view servo target projection failed: {error}'
                if not self._recover_precontact(
                    FailureKind.VISUAL_APPROACH_FAILED,
                    detail,
                ):
                    self._apply_safety(self._core.fail(detail))
                return
            self._core.mark_near_grounded(synchronized.serial)
            self._approach.reset()
            return
        if self._core.phase is RuntimePhase.PLACE_GROUNDING:
            try:
                self._publish_place_request(synchronized)
                self._place_planning_started_at = now
                self._place_planning_started_wall_s = time.monotonic()
            except (RuntimeError, TypeError, ValueError) as error:
                self._apply_safety(self._core.fail(
                    f'observed placement request rejected: {error}',
                ))
            return
        required_serial = self._core.required_replan_serial or 1
        if synchronized.serial < required_serial or self._future is not None:
            return
        try:
            observation = self._semantic_observation(
                synchronized.serial,
                synchronized.stamp_s,
            )
            self._core.begin_replan(synchronized.serial)
            self._start_planning('pregrasp', observation)
        except Exception as error:
            detail = f'final semantic observation rejected: {error}'
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))

    def _pregrasp_reobserve_tick(
        self,
        now: float,
        synchronized: Any | None,
    ) -> None:
        """Wait for measured arrival, then pair newer RGB-D with arm feedback."""
        self._publish_zero()
        handoff = self._pregrasp_handoff
        if (
            handoff is None
            or self._core.phase is not RuntimePhase.PREGRASP_REOBSERVE
            or not math.isfinite(float(now))
            or float(now) < handoff.completed_at_s
            or handoff.deadline_s <= handoff.completed_at_s
        ):
            detail = 'pregrasp re-observation contract is unavailable or invalid'
            if not self._recover_precontact(FailureKind.EXECUTION_FAILED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if float(now) > handoff.deadline_s:
            detail = (
                'pregrasp did not produce measured stable joints and a newer '
                'exact RGB-D bundle before its deadline'
            )
            if not self._recover_precontact(FailureKind.EXECUTION_FAILED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if not self._joint_history:
            return
        latest = self._joint_history[-1]
        maximum_age = float(
            self.get_parameter('pregrasp_joint_state_max_age_s').value,
        )
        tolerance = float(
            self.get_parameter('pregrasp_joint_tolerance_rad').value,
        )
        joint_age = float(now) - latest.received_at_s
        joint_source_age = float(now) - float(latest.source_stamp_ns) * 1e-9
        if (
            not math.isfinite(maximum_age)
            or maximum_age <= 0.0
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or latest.positions.shape != handoff.endpoint_joints.shape
            or not np.all(np.isfinite(latest.positions))
        ):
            detail = 'pregrasp joint feedback contract is invalid'
            if not self._recover_precontact(FailureKind.EXECUTION_FAILED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if (
            latest.sequence <= handoff.minimum_joint_sequence
            or latest.source_stamp_ns <= handoff.minimum_joint_stamp_ns
            or latest.source_stamp_ns <= handoff.completion_source_stamp_ns
            or not 0.0 <= joint_age <= maximum_age
            or not 0.0 <= joint_source_age <= maximum_age
        ):
            return
        error = float(np.max(np.abs(
            latest.positions - handoff.endpoint_joints,
        )))
        self._pregrasp_joint_error_rad = error
        if error > tolerance or not self._arm_is_still(float(now)):
            self._pregrasp_stable_joint_sequence = None
            self._pregrasp_stable_joint_stamp_ns = None
            return
        if self._pregrasp_stable_joint_sequence is None:
            self._pregrasp_stable_joint_sequence = latest.sequence
            self._pregrasp_stable_joint_stamp_ns = latest.source_stamp_ns
            return
        stable_stamp_ns = self._pregrasp_stable_joint_stamp_ns
        if stable_stamp_ns is None:
            return
        if not self._grounding_observation_authorized(synchronized):
            return
        assert synchronized is not None
        observation_stamp_ns = self._valid_observation_stamp_ns
        if (
            synchronized.serial <= handoff.observation_serial
            or observation_stamp_ns is None
            or observation_stamp_ns <= stable_stamp_ns
        ):
            return
        identity = handoff.observation_identity
        if (
            self._bound_perception_request_id != identity.request_id
            or self._bound_perception_producer_epoch != identity.producer_epoch
            or self._bound_perception_generation != identity.generation
            or self._valid_observation_frame_id != identity.frame_id
        ):
            detail = 'pregrasp perception owner, generation, or frame changed'
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        eligible = tuple(
            item
            for item in self._joint_history
            if item.sequence >= self._pregrasp_stable_joint_sequence
        )
        if not eligible:
            return
        matched = min(
            eligible,
            key=lambda item: abs(item.source_stamp_ns - observation_stamp_ns),
        )
        maximum_skew = float(self.get_parameter(
            'pregrasp_max_observation_joint_skew_s',
        ).value)
        source_skew = abs(
            matched.source_stamp_ns - observation_stamp_ns,
        ) * 1e-9
        matched_error = float(np.max(np.abs(
            matched.positions - handoff.endpoint_joints,
        )))
        if (
            not math.isfinite(maximum_skew)
            or maximum_skew <= 0.0
            or source_skew > maximum_skew
            or matched_error > tolerance
        ):
            return
        try:
            observation = self._semantic_observation(
                synchronized.serial,
                synchronized.stamp_s,
            )
            fresh_identity = self._capture_planning_observation_identity(
                observation,
            )
            joint_positions = np.asarray(
                matched.positions,
                dtype=float,
            ).copy()
            joint_positions.setflags(write=False)
            target_geometry = _target_geometry_signature(
                observation.target_collision_points,
                min_points=int(self.get_parameter('semantic_min_points').value),
                trim_mad_scale=float(self.get_parameter(
                    'approach_planning_geometry_trim_mad_scale',
                ).value),
                extent_percentile=float(self.get_parameter(
                    'approach_planning_geometry_extent_percentile',
                ).value),
            )
            self._core.begin_approach_replan(synchronized.serial)
            self._approach_planning_anchor = _ApproachPlanningAnchor(
                observation_identity=fresh_identity,
                observation_serial=synchronized.serial,
                joint_sequence=matched.sequence,
                joint_stamp_ns=matched.source_stamp_ns,
                joint_positions=joint_positions,
                target_geometry=target_geometry,
            )
            self._start_planning(
                'approach',
                observation,
                planning_joints=joint_positions,
            )
        except Exception as error:
            detail = f'post-pregrasp observation rejected: {error}'
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))

    def _pregrasp_result_execution_tick(self, now: float) -> None:
        """Wait for post-validation platform and arm samples before transit."""
        self._publish_zero()
        fence = self._pregrasp_dispatch_fence
        program = self._pregrasp_program
        identity = self._pregrasp_planning_identity
        if (
            fence is None
            or program is None
            or identity is None
            or self._core.phase is not RuntimePhase.PLANNING
            or not math.isfinite(float(now))
            or not math.isfinite(float(fence.deadline_s))
        ):
            detail = (
                'completed pregrasp plan lost its dispatch feedback '
                'contract'
            )
            if not self._recover_precontact(FailureKind.PLAN_BLOCKED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if float(now) > fence.deadline_s:
            detail = (
                'completed pregrasp plan did not receive fresh '
                'post-validation '
                'odometry and joint feedback before its deadline'
            )
            if not self._recover_precontact(FailureKind.PLAN_BLOCKED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if (
            not self._joint_history
            or self._odom_seen_at is None
            or self._odom_stamp_ns is None
        ):
            return
        latest_joint = self._joint_history[-1]
        if (
            latest_joint.sequence <= fence.minimum_joint_sequence
            or latest_joint.source_stamp_ns <= fence.minimum_joint_stamp_ns
            or self._odom_sequence <= fence.minimum_odom_sequence
            or self._odom_stamp_ns <= fence.minimum_odom_stamp_ns
        ):
            return

        joint_max_age_s = float(
            self.get_parameter('pregrasp_joint_state_max_age_s').value,
        )
        odom_max_age_s = float(
            self.get_parameter('posture_state_max_age_s').value,
        )
        joint_receipt_age_s = float(now) - latest_joint.received_at_s
        joint_source_age_s = (
            float(now) - float(latest_joint.source_stamp_ns) * 1e-9
        )
        odom_receipt_age_s = float(now) - float(self._odom_seen_at)
        odom_source_age_s = float(now) - float(self._odom_stamp_ns) * 1e-9
        positions = np.asarray(latest_joint.positions, dtype=float)
        if (
            not math.isfinite(joint_max_age_s)
            or joint_max_age_s <= 0.0
            or not math.isfinite(odom_max_age_s)
            or odom_max_age_s <= 0.0
            or positions.shape != (self._planner.chain.dof,)
            or not np.all(np.isfinite(positions))
        ):
            detail = 'pregrasp dispatch feedback contract is invalid'
            if not self._recover_precontact(FailureKind.PLAN_BLOCKED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if not (
            0.0 <= joint_receipt_age_s <= joint_max_age_s
            and 0.0 <= joint_source_age_s <= joint_max_age_s
            and 0.0 <= odom_receipt_age_s <= odom_max_age_s
            and 0.0 <= odom_source_age_s <= odom_max_age_s
        ):
            return
        if not self._arm_is_still(float(now)):
            return
        if (
            np.max(np.abs(positions - program.transit.positions[0]))
            > float(self.get_parameter('max_trajectory_start_error_rad').value)
        ):
            detail = 'measured arm changed after pregrasp transit validation'
            if not self._recover_precontact(FailureKind.PLAN_BLOCKED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        try:
            self._validate_grasp_planning_observation(identity)
        except _PlanningObservationPending:
            return
        except _PlanningObservationChanged as error:
            detail = str(error)
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if not self._guard_active_posture(float(now)):
            return
        if self._task.stage not in (Stage.OBSERVE_GRASP, Stage.PLAN_GRASP):
            self._apply_safety(self._core.fail(
                'completed pregrasp plan is not owned by the '
                'observe_grasp or plan_grasp stage',
            ))
            return
        try:
            if self._task.stage is Stage.OBSERVE_GRASP:
                self._task.apply(StageResult.success())
            self._core.plan_ready()
        except RuntimeError as error:
            self._apply_safety(self._core.fail(
                f'pregrasp dispatch stage transition failed: {error}',
            ))
            return
        self._pregrasp_dispatch_fence = None
        self._gripper_pub.publish(Float32(data=float(
            self.get_parameter('open_aperture_m').value,
        )))
        self._publish_program_segment('transit', path_prevalidated=True)

    def _approach_result_execution_tick(self, now: float) -> None:
        """Wait boundedly for post-plan feedback, then publish contact motion."""
        self._publish_zero()
        if self._program is None:
            return
        anchor = self._approach_planning_anchor
        joint_fence = self._approach_execution_joint_fence
        if (
            anchor is None
            or joint_fence is None
            or not math.isfinite(float(now))
            or not math.isfinite(float(joint_fence.deadline_s))
        ):
            detail = 'completed approach plan lost its execution handoff contract'
            if not self._recover_precontact(FailureKind.PLAN_BLOCKED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if float(now) > joint_fence.deadline_s:
            detail = (
                'completed approach plan did not receive fresh post-planning '
                'joint feedback before its deadline'
            )
            if not self._recover_precontact(FailureKind.PLAN_BLOCKED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if not self._joint_history:
            return
        try:
            _fresh_approach_joint_positions(
                self._joint_history[-1],
                anchor,
                now_s=now,
                maximum_age_s=float(self.get_parameter(
                    'approach_execution_joint_state_max_age_s',
                ).value),
                dof=self._planner.chain.dof,
                minimum_sequence=joint_fence.minimum_joint_sequence,
                minimum_source_stamp_ns=joint_fence.minimum_joint_stamp_ns,
            )
        except _ApproachJointFeedbackPending:
            return
        except ValueError as error:
            detail = f'completed approach plan joint contract is invalid: {error}'
            if not self._recover_precontact(FailureKind.EXECUTION_FAILED, detail):
                self._apply_safety(self._core.fail(detail))
            return
        try:
            self._validate_grasp_planning_observation(
                anchor.observation_identity,
                target_geometry=anchor.target_geometry,
            )
        except _PlanningObservationPending:
            return
        except _PlanningObservationChanged as error:
            detail = str(error)
            if not self._recover_precontact(FailureKind.TARGET_LOST, detail):
                self._apply_safety(self._core.fail(detail))
            return
        if self._task.stage.value != 'plan_grasp':
            self._apply_safety(self._core.fail(
                'completed approach plan is not owned by the plan_grasp stage',
            ))
            return
        self._core.approach_plan_ready()
        self._publish_program_segment('approach')
        if self._core.phase is not RuntimePhase.APPROACH:
            return
        try:
            self._task.apply(StageResult.success(self._program))
        except RuntimeError as error:
            self._apply_safety(self._core.fail(
                f'approach execution stage transition failed: {error}',
            ))
            return
        self._approach_execution_joint_fence = None
        self._publish_debug_plan()

    def _arm_is_still(self, now: float) -> bool:
        window = float(self.get_parameter('arm_still_window_s').value)
        tolerance = float(self.get_parameter('arm_still_tolerance_rad').value)
        if (
            not math.isfinite(float(now))
            or not math.isfinite(window)
            or window <= 0.0
            or not math.isfinite(tolerance)
            or tolerance <= 0.0
            or not self._joint_history
        ):
            return False
        latest = self._joint_history[-1]
        latest_age_s = float(now) - float(latest.received_at_s)
        latest_source_age_s = (
            float(now) - float(latest.source_stamp_ns) * 1e-9
        )
        if (
            not math.isfinite(latest_age_s)
            or latest_age_s < 0.0
            or latest_age_s > window
            or not math.isfinite(latest_source_age_s)
            or latest_source_age_s < 0.0
            or latest_source_age_s > window
        ):
            return False
        window_ns = int(round(window * 1e9))
        minimum_span_ns = int(round(0.75 * window * 1e9))
        samples = [
            item for item in self._joint_history
            if 0 <= latest.source_stamp_ns - item.source_stamp_ns <= window_ns + 1
        ]
        if (
            len(samples) < 2
            or samples[-1].source_stamp_ns - samples[0].source_stamp_ns
            < minimum_span_ns
        ):
            return False
        positions = np.stack([item.positions for item in samples])
        excursion = float(np.max(np.ptp(positions, axis=0)))
        return excursion <= tolerance

    def _capture_carried_object_geometry(
        self,
        now: float,
        synchronized: Any,
    ) -> None:
        """Freeze object axes and tool attachment from the exact grasp observation."""
        semantics = self._affordance_placement_semantics
        stamp_ns = self._valid_observation_stamp_ns
        request_id = self._bound_perception_request_id
        producer_epoch = self._bound_perception_producer_epoch
        generation = self._bound_perception_generation
        if (
            semantics is None
            or self._target_cloud is None
            or not self._joint_history
            or stamp_ns is None
            or stamp_ns <= 0
            or request_id is None
            or producer_epoch is None
            or generation is None
            or generation <= 0
            or self._target_cloud_stamp_ns != stamp_ns
            or not self._grounding_observation_authorized(synchronized)
        ):
            raise ValueError(
                'carried-object geometry lacks exact tracked semantics or points',
            )
        joint_feedback = self._joint_history[-1]
        skew_limit = float(self.get_parameter(
            'carried_object_max_joint_target_skew_s',
        ).value)
        joint_age = now - joint_feedback.received_at_s
        source_age = now - float(joint_feedback.source_stamp_ns) * 1e-9
        source_skew = abs(joint_feedback.source_stamp_ns - stamp_ns) * 1e-9
        if (
            not math.isfinite(skew_limit)
            or skew_limit <= 0.0
            or not 0.0 <= joint_age <= skew_limit
            or not 0.0 <= source_age <= skew_limit
            or source_skew > skew_limit
        ):
            raise ValueError(
                'carried-object FK is not synchronized with the tracked cloud',
            )
        try:
            base_t_tool = self._planner.chain.forward(joint_feedback.positions)
            identity = CarriedObjectObservationIdentity(
                request_id=request_id,
                producer_epoch=producer_epoch,
                generation=generation,
                observation_stamp_ns=stamp_ns,
                frame_id=self._target_frame_id,
            )
            geometry = estimate_carried_object_geometry(
                self._target_cloud,
                base_t_tool,
                semantics,
                identity,
                config=self._object_geometry_config,
            )
        except (ObjectGeometryError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                f'carried-object frame is unobservable: {error}',
            ) from error
        self._carried_object_geometry = geometry
        self._carried_object_observation_stamp_ns = stamp_ns

    def _capture_place_observation_identity(
        self,
        synchronized: Any,
        goal_id: str,
    ) -> PlacementObservationIdentity:
        """Freeze the exact perception owner consumed by placement planning."""
        stamp_ns = self._valid_observation_stamp_ns
        request_id = self._bound_perception_request_id
        producer_epoch = self._bound_perception_producer_epoch
        generation = self._bound_perception_generation
        expected = (
            request_id,
            producer_epoch,
            generation,
            stamp_ns,
            self._target_frame_id,
        )
        observed = (
            self._valid_perception_request_id,
            self._valid_perception_producer_epoch,
            self._valid_perception_generation,
            self._valid_observation_stamp_ns,
            self._valid_observation_frame_id,
        )
        affordance = (
            self._affordance_request_id,
            self._affordance_producer_epoch,
            self._affordance_generation,
        )
        carried_geometry = self._carried_object_geometry
        if (
            synchronized is None
            or carried_geometry is None
            or request_id is None
            or producer_epoch is None
            or generation is None
            or generation <= 0
            or stamp_ns is None
            or stamp_ns <= 0
            or expected != observed
            or affordance != (request_id, producer_epoch, generation)
            or self._required_perception_request_id != request_id
            or self._required_perception_generation != generation
            or self._required_affordance_generation != generation
        ):
            raise ValueError(
                'placement observation identity is not exact-authorized',
            )
        return PlacementObservationIdentity(
            goal_id=goal_id,
            request_id=request_id,
            producer_epoch=producer_epoch,
            generation=generation,
            frame_id=self._target_frame_id,
            planning_observation_stamp_ns=stamp_ns,
            require_upright=carried_geometry.require_upright,
        )

    def _publish_place_request(self, synchronized: Any) -> None:
        if (
            self._affordance is None
            or self._target_cloud is None
            or self._carried_object_geometry is None
            or not self._target_frame_id
        ):
            raise ValueError('placement request is missing observed object geometry')
        region = self._affordance.get('placement_region')
        if not isinstance(region, dict):
            raise ValueError('VLM did not identify a visible placement region')
        box = np.asarray(region.get('bbox_xyxy_normalized'), dtype=float)
        if (
            box.shape != (4,)
            or not np.all(np.isfinite(box))
            or np.any(box < 0.0)
            or np.any(box > 1.0)
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError('VLM placement region is invalid')
        now = self._now_s()
        if self._nav_speed > 0.035 or not self._arm_is_still(now):
            raise ValueError('base and arm must be measured stationary for placement observation')

        geometry = self._carried_object_geometry
        avoids = []
        for item in self._affordance.get('placement_avoid_regions', []):
            if not isinstance(item, dict):
                raise ValueError('VLM placement avoid region is malformed')
            avoid = np.asarray(item.get('bbox_xyxy_normalized'), dtype=float)
            if avoid.shape != (4,) or not np.all(np.isfinite(avoid)):
                raise ValueError('VLM placement avoid region is invalid')
            avoids.append(avoid.tolist())

        stamp_ns = self._valid_observation_stamp_ns
        if stamp_ns is None or stamp_ns <= 0:
            raise ValueError('placement observation stamp is unavailable')
        self._place_goal_id = (
            f'place-{self._affordance_generation}-{stamp_ns}'
        )
        self._place_transaction_requested = False
        self._place_transaction_abort_sent = False
        identity = self._capture_place_observation_identity(
            synchronized,
            self._place_goal_id,
        )
        MobileManipulationRuntime._reset_post_release_verification(self)
        self._place_observation_identity = identity
        self._place_trajectory = None
        self._place_contract = None
        self._place_programs.clear()
        payload = {
            'schema_version': 2,
            'goal_id': self._place_goal_id,
            'stamp_ns': stamp_ns,
            'image_frame': self._target_frame_id,
            'request_id': identity.request_id,
            'producer_epoch': identity.producer_epoch,
            'generation': identity.generation,
            'region_xyxy': box.tolist(),
            'avoid_xyxy': avoids,
            'constraints': {
                'min_clearance_m': 0.03,
                'min_support_fraction': 1.0,
            },
            'object_extent_m': geometry.object_extent_m.tolist(),
            'tool_from_object': geometry.tool_from_object.tolist(),
            'object_reference_points_object': (
                geometry.reference_points_object.tolist()
            ),
            'object_reference_identity': geometry.identity.to_payload(),
            'verification': geometry.verification_payload(),
        }
        self._core.place_request_sent(
            place_contract_id=self._place_goal_id,
            executor_state=self._execution_status,
        )
        payload['executor_epoch'] = self._core.place_executor_epoch
        self._place_region_pub.publish(String(data=json.dumps(
            payload,
            separators=(',', ':'),
        )))
        self._place_transaction_requested = True

    def _start_release(self, now: float) -> None:
        if not self._guard_active_posture(now):
            return
        previous_gripper_id = self._core.place_approach_gripper_command_id
        if previous_gripper_id is None:
            self._apply_safety(self._core.fail(
                'release command lacks the completed approach gripper identity',
            ))
            return
        open_aperture = float(self.get_parameter('open_aperture_m').value)
        self._expected_gripper_command_id = previous_gripper_id + 1
        self._gripper_command_sent_s = now
        self._gripper_feedback.clear()
        self._release_started_at = now
        self._gripper_pub.publish(Float32(data=open_aperture))

    def _release_tick(self, now: float) -> None:
        if self._release_started_at is None or self._gripper_command_sent_s is None:
            self._apply_safety(self._core.fail('release phase has no command time'))
            return
        if now - self._release_started_at < float(self.get_parameter('release_settle_s').value):
            return
        status = self._execution_status
        expected = self._expected_gripper_command_id
        open_aperture = float(self.get_parameter('open_aperture_m').value)
        if (
            status is None
            or expected is None
            or status.gripper_command_id != expected
            or status.accepted_gripper_aperture_m is None
            or status.gripper_received_at is None
            or self._core.place_release_gripper_command_id != expected
            or self._core.place_release_gripper_received_at
            != status.gripper_received_at
            or abs(status.accepted_gripper_aperture_m - open_aperture) > 1e-5
        ):
            self._apply_safety(self._core.fail('gripper release acknowledgement mismatch'))
            return
        stable_samples = int(self.get_parameter('gripper_stable_samples').value)
        samples = [
            aperture
            for stamp, command_id, aperture in self._gripper_feedback
            if stamp > self._gripper_command_sent_s and command_id == expected
        ]
        if len(samples) < stable_samples:
            return
        recent = samples[-stable_samples:]
        if (
            max(recent) - min(recent)
            > float(self.get_parameter('gripper_stable_tolerance_m').value)
            or recent[-1] < float(self.get_parameter('release_min_aperture_m').value)
        ):
            return
        if self._place_observation_identity is None:
            self._apply_safety(self._core.fail(
                'post-release placement identity is unavailable',
            ))
            return
        self._post_release_release_command_id = expected
        self._post_release_pending_evidence = None
        self._post_release_verified_evidence = None
        self._core.release_complete()
        self._publish_program_segment('place_retreat')

    def _closing_tick(self, now: float) -> None:
        if self._closing_started_at is None:
            self._apply_safety(self._core.fail('closing phase has no start time'))
            return
        if now - self._closing_started_at < float(self.get_parameter('gripper_settle_s').value):
            return
        if self._execution_status is None or self._execution_status.aperture_m is None:
            self._apply_safety(self._core.fail('gripper aperture feedback unavailable'))
            return
        if (
            self._expected_gripper_command_id is None
            or self._gripper_command_sent_s is None
            or self._execution_status_seen_s is None
            or self._execution_status_seen_s <= self._gripper_command_sent_s
            or self._execution_status.gripper_command_id
            != self._expected_gripper_command_id
        ):
            self._apply_safety(self._core.fail('gripper close acknowledgement identity mismatch'))
            return
        if (
            self._execution_status.accepted_gripper_aperture_m is None
            or self._commanded_close_aperture is None
            or abs(
                self._execution_status.accepted_gripper_aperture_m
                - self._commanded_close_aperture
            ) > 1e-5
        ):
            self._apply_safety(self._core.fail('gripper close command was not accepted'))
            return
        stable_samples = int(self.get_parameter('gripper_stable_samples').value)
        samples = [
            aperture
            for stamp, command_id, aperture in self._gripper_feedback
            if stamp > self._gripper_command_sent_s
            and command_id == self._expected_gripper_command_id
        ]
        if len(samples) < stable_samples:
            return
        recent = samples[-stable_samples:]
        if max(recent) - min(recent) > float(
            self.get_parameter('gripper_stable_tolerance_m').value,
        ):
            return
        open_aperture = float(self.get_parameter('open_aperture_m').value)
        contact_margin = float(self.get_parameter('grasp_contact_margin_m').value)
        held_min = max(
            0.001,
            float(self._commanded_close_aperture or 0.0) + contact_margin,
        )
        held_max = open_aperture - 2.0 * contact_margin
        if held_min >= held_max:
            self._apply_safety(self._core.fail('grasp aperture has no valid contact interval'))
            return
        try:
            self._execution_occlusion.confirm_contact(now)
        except (TypeError, ValueError) as error:
            self._apply_safety(self._core.fail(
                f'measured contact evidence rejected: {error}',
            ))
            return
        predicted_baseline = self._execution_occlusion.loss_active
        if predicted_baseline:
            decision = self._execution_occlusion_decision(
                now,
                phase=RuntimePhase.CLOSING,
            )
            if not decision.allowed:
                self._apply_safety(self._core.fail(
                    f'pre-lift occlusion rejected: {decision.reason}',
                ))
                return
        self._verifier = GraspVerifier(GraspVerificationConfig(
            min_held_aperture_m=held_min,
            max_held_aperture_m=held_max,
            track_loss_timeout_s=(
                self._execution_occlusion_verification_reacquire_timeout_s
            ),
        ))
        try:
            establish_baseline_before_lift(
                lambda: self._sample_grasp_verifier(
                    now,
                    require_fresh_target_lock=not predicted_baseline,
                    allow_predicted_target_baseline=predicted_baseline,
                ),
                self._begin_lift_motion,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            self._apply_safety(self._core.fail(
                f'pre-lift grasp verification rejected: {error}',
            ))

    def _begin_lift_motion(self) -> None:
        self._core.close_complete()
        self._publish_program_segment('lift')

    def _grasp_close_aperture(self) -> float:
        assert self._program is not None
        return grasp_close_aperture(
            self._program.required_width_m,
            fallback_m=float(self.get_parameter('close_aperture_m').value),
            squeeze_m=float(self.get_parameter('grasp_squeeze_m').value),
            minimum_m=float(self.get_parameter('gripper_min_aperture_m').value),
            maximum_m=float(self.get_parameter('gripper_max_aperture_m').value),
        )

    def _sample_grasp_verifier(
        self,
        now: float,
        *,
        require_fresh_target_lock: bool,
        allow_predicted_target_baseline: bool = False,
    ) -> VerificationResult:
        if not math.isfinite(now):
            raise ValueError('verification time is not finite')
        if require_fresh_target_lock and allow_predicted_target_baseline:
            raise ValueError('a fresh target lock cannot use a predicted baseline')
        if not self._joint_history:
            raise ValueError('current joint feedback is unavailable')
        joint_feedback = self._joint_history[-1]
        joint_seen_at = joint_feedback.received_at_s
        joint_state = joint_feedback.positions
        max_joint_age = float(
            self.get_parameter('verification_joint_state_max_age_s').value,
        )
        joint_age = now - joint_seen_at
        joint_source_age = now - float(joint_feedback.source_stamp_ns) * 1e-9
        if (
            not math.isfinite(max_joint_age)
            or max_joint_age <= 0.0
            or not 0.0 <= joint_age <= max_joint_age
            or not 0.0 <= joint_source_age <= max_joint_age
        ):
            raise ValueError('current joint feedback is stale')
        status = self._execution_status
        if status is None or status.aperture_m is None:
            raise ValueError('gripper aperture feedback is unavailable')

        validity_age = (
            float('inf')
            if self._valid_seen_at is None
            else now - self._valid_seen_at
        )
        validity_limit = float(self.get_parameter('perception_loss_timeout_s').value)
        synchronized = self._serial_gate.snapshot(now)
        tracker_locked = bool(
            self._perception_valid
            and not self._execution_occlusion.loss_active
            and math.isfinite(validity_limit)
            and validity_limit > 0.0
            and 0.0 <= validity_age <= validity_limit
            and synchronized is not None
            and self._target_piper is not None
            and self._grounding_observation_authorized(synchronized)
        )
        if require_fresh_target_lock and not tracker_locked:
            raise ValueError('fresh synchronized target lock is unavailable')

        target_centroid = (
            self._target_piper.copy() if tracker_locked else None
        )
        sample_tracker_locked = tracker_locked
        if not tracker_locked and allow_predicted_target_baseline:
            decision = self._execution_occlusion_decision(
                now,
                phase=RuntimePhase.CLOSING,
            )
            if (
                not self._execution_occlusion.loss_active
                or not self._execution_occlusion.contact_confirmed
                or not decision.allowed
                or self._execution_occlusion_target_piper is None
            ):
                reason = decision.reason or 'exact near-contact target is unavailable'
                raise ValueError(f'predicted grasp baseline rejected: {reason}')
            target_centroid = self._execution_occlusion_target_piper.copy()
            sample_tracker_locked = True

        try:
            ee_position = self._planner.chain.forward(joint_state)[:3, 3]
        except Exception as error:
            raise ValueError(f'current joint FK failed: {error}') from error
        return self._verifier.update(VerificationSample(
            stamp_s=now,
            gripper_aperture_m=status.aperture_m,
            ee_position_base=ee_position,
            target_centroid_base=target_centroid,
            tracker_locked=sample_tracker_locked,
        ))

    def _verification_tick(self, now: float) -> None:
        if self._verification_started_at is None:
            self._apply_safety(self._core.fail('verification phase has no start time'))
            return
        if (
            now - self._verification_started_at
            > float(self.get_parameter('verification_timeout_s').value)
        ):
            self._apply_safety(self._core.fail('grasp verification timed out'))
            return
        try:
            result = self._sample_grasp_verifier(
                now,
                require_fresh_target_lock=False,
            )
        except (TypeError, ValueError) as error:
            self._apply_safety(self._core.fail(
                f'grasp verification sample rejected: {error}',
            ))
            return
        if result.state is VerificationState.FAILED:
            if not self._task.terminal:
                self._task.apply(StageResult.failure(FailureKind.VERIFY_FAILED, result.reason))
            self._apply_safety(self._core.fail(f'grasp verification failed: {result.reason}'))
        elif result.state is VerificationState.SUCCESS and self._core.phase is RuntimePhase.VERIFY:
            synchronized = self._serial_gate.snapshot(now)
            if (
                not self._grounding_observation_authorized(synchronized)
                or self._future is not None
            ):
                self._apply_safety(self._core.fail(
                    'carry planning requires a fresh exact-authorized observation',
                ))
                return
            carry_only = self._topic_value('place_mode') == 'carry_only'
            if not carry_only:
                try:
                    assert synchronized is not None
                    self._capture_carried_object_geometry(now, synchronized)
                except (TypeError, ValueError) as error:
                    self._apply_safety(self._core.fail(
                        f'carried-object observation rejected: {error}',
                    ))
                    return
            if self._task.stage.value == 'verify_grasp':
                self._task.apply(StageResult.success(result))
            self._core.verification_complete(carry_only=carry_only)
            self._reset_execution_occlusion()
            assert synchronized is not None
            try:
                self._start_carry_planning(synchronized.stamp_s)
            except (TypeError, ValueError) as error:
                self._apply_safety(self._core.fail(f'carry planning failed: {error}'))

    def _publish_program_segment(
        self,
        name: str,
        *,
        path_prevalidated: bool = False,
    ) -> None:
        """Publish one segment, optionally consuming staged transit proof."""
        now = self._now_s()
        if not self._core.active or not self._guard_active_posture(now):
            return
        if path_prevalidated and name != 'transit':
            self._apply_safety(self._core.fail(
                'only pregrasp transit may consume staged path validation',
            ))
            return
        is_place = name.startswith('place_')
        if name == 'transit' and self._pregrasp_program is None:
            self._apply_safety(self._core.fail('pregrasp program is unavailable'))
            return
        if (
            name != 'transit'
            and self._program is None
            and name != 'carry'
            and not is_place
        ):
            self._apply_safety(self._core.fail('motion program is unavailable'))
            return
        timed = (
            self._place_programs.get(name)
            if is_place
            else self._carry_program
            if name == 'carry'
            else self._pregrasp_program.transit
            if name == 'transit' and self._pregrasp_program is not None
            else getattr(self._program, name)
        )
        if timed is None:
            self._apply_safety(self._core.fail(f'{name} motion is unavailable'))
            return
        synchronized = self._serial_gate.snapshot(now)
        if name == 'lift' and not self._execution_perception_admitted(
            now,
            RuntimePhase.CLOSING,
        ):
            reason = self._execution_occlusion_last_decision.reason
            self._apply_safety(self._core.fail(
                'cannot execute lift: execution perception rejected: '
                f'{reason or "execution evidence is unavailable"}',
            ))
            return
        use_cached_occlusion = bool(
            name == 'lift' and self._execution_occlusion.loss_active
        )
        if self._joint_state is None and name != 'approach':
            self._apply_safety(self._core.fail(
                f'cannot execute {name}: current joint feedback is unavailable',
            ))
            return
        measured_joint_state = self._joint_state
        approach_joint_fence: _ApproachExecutionJointFence | None = None
        if name == 'approach':
            anchor = self._approach_planning_anchor
            approach_joint_fence = self._approach_execution_joint_fence
            if (
                anchor is None
                or approach_joint_fence is None
                or not self._joint_history
            ):
                self._apply_safety(self._core.fail(
                    'cannot execute approach: fresh post-planning joint '
                    'feedback is unavailable',
                ))
                return
            try:
                measured_joint_state = _fresh_approach_joint_positions(
                    self._joint_history[-1],
                    anchor,
                    now_s=now,
                    maximum_age_s=float(self.get_parameter(
                        'approach_execution_joint_state_max_age_s',
                    ).value),
                    dof=self._planner.chain.dof,
                    minimum_sequence=(
                        approach_joint_fence.minimum_joint_sequence
                    ),
                    minimum_source_stamp_ns=(
                        approach_joint_fence.minimum_joint_stamp_ns
                    ),
                )
            except (TypeError, ValueError) as error:
                self._apply_safety(self._core.fail(
                    f'cannot execute approach: {error}',
                ))
                return
        elif name == 'lift':
            if not self._joint_history:
                self._apply_safety(self._core.fail(
                    'cannot execute lift: current joint feedback is unavailable',
                ))
                return
            joint_feedback = self._joint_history[-1]
            joint_age = now - joint_feedback.received_at_s
            joint_source_age = (
                now - float(joint_feedback.source_stamp_ns) * 1e-9
            )
            maximum_age = self._execution_occlusion.config.joint_state_max_age_s
            if not (
                0.0 <= joint_age <= maximum_age
                and 0.0 <= joint_source_age <= maximum_age
            ):
                self._apply_safety(self._core.fail(
                    'cannot execute lift: current joint feedback is stale',
                ))
                return
            measured_joint_state = np.asarray(
                joint_feedback.positions,
                dtype=float,
            )
        if use_cached_occlusion:
            decision = self._execution_occlusion_decision(
                now,
                phase=RuntimePhase.CLOSING,
            )
            if (
                not decision.allowed
                or self._execution_occlusion_scene_cloud is None
                or self._execution_occlusion_target_cloud is None
                or self._execution_occlusion.observation_stamp_s is None
            ):
                reason = decision.reason or 'cached exact geometry is unavailable'
                self._apply_safety(self._core.fail(
                    f'cannot execute lift during occlusion: {reason}',
                ))
                return
            scene_cloud = self._execution_occlusion_scene_cloud
            target_cloud = self._execution_occlusion_target_cloud
            observation_stamp_s = self._execution_occlusion.observation_stamp_s
        else:
            if (
                not self._grounding_observation_authorized(synchronized)
                or self._scene_cloud is None
                or self._target_cloud is None
            ):
                self._apply_safety(self._core.fail(
                    f'cannot execute {name}: newest perception is not '
                    'fresh, synchronized, and exact-authorized',
                ))
                return
            scene_cloud = self._scene_cloud
            target_cloud = self._target_cloud
            assert synchronized is not None
            observation_stamp_s = synchronized.stamp_s
            if name == 'lift':
                try:
                    self._cache_execution_occlusion_geometry()
                except ValueError as error:
                    self._apply_safety(self._core.fail(
                        f'cannot retain exact lift geometry: {error}',
                    ))
                    return
        assert measured_joint_state is not None
        if (
            np.max(np.abs(measured_joint_state - timed.positions[0]))
            > float(self.get_parameter('max_trajectory_start_error_rad').value)
        ):
            self._apply_safety(self._core.fail(
                f'cannot execute {name}: measured arm differs from trajectory start',
            ))
            return
        path_validation_wall_s = 0.0
        path_valid = bool(path_prevalidated)
        if not path_prevalidated:
            path_validation_started_wall_s = time.monotonic()
            path_valid = self._planner.validate_path(
                timed.positions,
                scene_points=scene_cloud,
                target_points=target_cloud,
                stamp_s=observation_stamp_s,
                segment_name=name,
                attachment_joints=(
                    measured_joint_state.copy()
                    if name in (
                        'lift', 'carry', 'place_transit', 'place_approach',
                    )
                    else None
                ),
                required_width_m=(
                    getattr(self._program, 'required_width_m', None)
                    if name in ('approach', 'lift')
                    else None
                ),
            )
            path_validation_wall_s = (
                time.monotonic() - path_validation_started_wall_s
            )
        if (
            name == 'approach'
            and path_validation_wall_s > float(self.get_parameter(
                'approach_execution_joint_state_max_age_s',
            ).value)
        ):
            self._apply_safety(self._core.fail(
                'cannot execute approach: collision validation outlived the '
                'joint freshness budget',
            ))
            return
        if not path_valid:
            self._apply_safety(self._core.fail(
                f'cannot execute {name}: newest perceived scene invalidates the path',
            ))
            return
        publish_now = self._now_s()
        if name == 'approach':
            anchor = self._approach_planning_anchor
            assert anchor is not None and approach_joint_fence is not None
            try:
                self._validate_grasp_planning_observation(
                    anchor.observation_identity,
                    target_geometry=anchor.target_geometry,
                )
                publish_now = self._now_s()
                measured_joint_state = _fresh_approach_joint_positions(
                    self._joint_history[-1],
                    anchor,
                    now_s=publish_now,
                    maximum_age_s=float(self.get_parameter(
                        'approach_execution_joint_state_max_age_s',
                    ).value),
                    dof=self._planner.chain.dof,
                    minimum_sequence=(
                        approach_joint_fence.minimum_joint_sequence
                    ),
                    minimum_source_stamp_ns=(
                        approach_joint_fence.minimum_joint_stamp_ns
                    ),
                )
            except (_PlanningObservationPending, _PlanningObservationChanged) as error:
                self._apply_safety(self._core.fail(
                    f'cannot execute approach: final perception fence failed: {error}',
                ))
                return
            except (TypeError, ValueError) as error:
                self._apply_safety(self._core.fail(
                    f'cannot execute approach: final joint fence failed: {error}',
                ))
                return
            if (
                np.max(np.abs(measured_joint_state - timed.positions[0]))
                > float(self.get_parameter(
                    'max_trajectory_start_error_rad',
                ).value)
            ):
                self._apply_safety(self._core.fail(
                    'cannot execute approach: measured arm changed during '
                    'collision validation',
                ))
                return
        if not self._guard_active_posture(publish_now):
            return
        executor_epoch: str | None = None
        if not is_place:
            status = self._execution_status
            status_seen_s = self._execution_status_seen_s
            maximum_status_age = float(self.get_parameter(
                'execution_status_max_age_s',
            ).value)
            status_age = (
                math.inf
                if status_seen_s is None
                else publish_now - float(status_seen_s)
            )
            if (
                status is None
                or not status.executor_epoch
                or not math.isfinite(maximum_status_age)
                or maximum_status_age <= 0.0
                or not 0.0 <= status_age <= maximum_status_age
            ):
                self._apply_safety(self._core.fail(
                    f'cannot execute {name}: executor epoch status is unavailable '
                    'or stale',
                ))
                return
            executor_epoch = status.executor_epoch
        if name == 'lift':
            try:
                self._execution_occlusion.note_lift_sent(publish_now)
            except (TypeError, ValueError) as error:
                self._apply_safety(self._core.fail(
                    f'lift execution evidence rejected: {error}',
                ))
                return
        message = JointTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        execution_token = f'trajectory-{uuid.uuid4().hex}'
        try:
            message.header.frame_id = trajectory_segment_frame_id(
                name,
                self._place_goal_id if is_place else None,
                execution_token=execution_token,
            )
        except ValueError as error:
            self._apply_safety(self._core.fail(
                f'cannot execute {name}: {error}',
            ))
            return
        message.joint_names = list(self._planner.chain.joint_names)
        for positions, seconds in zip(timed.positions, timed.times_s):
            point = JointTrajectoryPoint()
            point.positions = positions.tolist()
            whole = int(seconds)
            point.time_from_start = DurationMsg(
                sec=whole,
                nanosec=int(round((float(seconds) - whole) * 1e9)),
            )
            message.points.append(point)
        try:
            if is_place:
                self._core.trajectory_sent(
                    name,
                    place_contract_id=self._place_goal_id,
                    executor_state=self._execution_status,
                    trajectory_token=execution_token,
                )
            else:
                self._core.trajectory_sent(
                    name,
                    executor_epoch=executor_epoch,
                    published_at_s=publish_now,
                    trajectory_token=execution_token,
                )
        except (RuntimeError, TypeError, ValueError) as error:
            self._apply_safety(self._core.fail(
                f'cannot execute {name}: executor transaction rejected: {error}',
            ))
            return
        self._trajectory_deadline_s = (
            publish_now
            + float(timed.times_s[-1])
            + float(self.get_parameter('execution_ack_margin_s').value)
        )
        self._trajectory_pub.publish(message)

    def _publish_zero(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._config.robot.platform_base_frame
        self._velocity_pub.publish(message)

    def _recover_precontact(self, kind: FailureKind, detail: str) -> bool:
        MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
            self,
            False,
        )
        precontact = {
            RuntimePhase.POSE_SETTLE,
            RuntimePhase.VISUAL_SEARCH,
            RuntimePhase.GROUNDING,
            RuntimePhase.STANDOFF,
            RuntimePhase.COARSE_NAV,
            RuntimePhase.NEAR_GROUNDING,
            RuntimePhase.VISUAL_SERVO,
            RuntimePhase.FINAL_GROUNDING,
            RuntimePhase.WAIT_FRESH_OBSERVATION,
            RuntimePhase.PLANNING,
            RuntimePhase.PREGRASP_REOBSERVE,
            RuntimePhase.APPROACH_PLANNING,
        }
        if self._core.phase not in precontact or self._task.terminal:
            return False
        try:
            transition = self._task.apply(StageResult.failure(kind, detail))
        except RuntimeError:
            return False
        self._visual_search_settle_reference = None
        MobileManipulationRuntime._clear_visual_search_stationarity(
            self,
            'bounded_recovery',
        )
        MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
            self,
            'bounded_recovery',
        )
        self._pose_settle_started_at = None
        self._pose_settle_last_tick_at = None
        self._reset_execution_occlusion()
        self._coarse_nav_perception_loss_detail = ''
        self._coarse_nav_arrival_started_at_s = None
        self._coarse_nav_arrival_stable_since_s = None
        self._coarse_nav_arrival_stable_start_odom_stamp_ns = None
        self._coarse_nav_arrival_last_odom_sequence = None
        self._coarse_nav_arrival_last_odom_stamp_ns = None
        if transition.current is Stage.FAILED:
            action = self._core.fail(
                transition.reason or self._task.failure_reason or detail,
            )
            self._release_terminal_ownership()
            self._apply_safety(action)
            return True

        # Every accepted nonterminal recovery invalidates in-flight SciPy/RRT
        # work before selecting its next bounded state.
        self._invalidate_async_work()
        self._apply_safety(SafetyAction(
            stop_base=True,
            cancel_navigation=True,
            cancel_arm=True,
            reason=f'bounded recovery: {kind.value}: {detail}',
        ))
        self._pregrasp_program = None
        MobileManipulationRuntime._clear_pregrasp_handoff(self)
        self._program = None
        self._carry_program = None
        self._approach.reset()
        self._coarse_nav_ready = False
        self._reground_started_at = None
        self._reground_last_tick_at = None
        MobileManipulationRuntime._clear_near_view_settle(self)

        if transition.current is Stage.PLAN_GRASP:
            if (
                self._bound_perception_request_id is None
                or self._bound_perception_producer_epoch is None
                or self._bound_perception_generation is None
            ):
                self._apply_safety(self._core.fail(
                    'fresh-observation retry lost its perception ownership binding',
                ))
                return True
            self._required_perception_request_id = self._bound_perception_request_id
            self._required_perception_generation = self._bound_perception_generation
            self._required_affordance_generation = self._bound_perception_generation
            latest = self._serial_gate.snapshot(self._now_s())
            after = (
                latest.serial
                if latest is not None
                else self._core.planned_serial
                or self._core.prospective_serial
                or 0
            )
            self._core.request_reobservation(after)
            return True

        self._work_pose = None
        self._work_pose_created_at_s = None
        self._navigation_status_seen_s = None
        self._navigation_status_goal_id = ''
        self._navigation_status_phase = ''
        self._navigation_goal_acknowledged = False
        self._navigation_history_recorded = False
        self._desired_depth = None
        self._approximate_displacement = None
        self._required_affordance_generation = None
        self._required_perception_generation = None
        self._required_perception_request_id = None
        self._required_grounding_scope = None
        self._clear_perception_authorization()
        self._core.restart_grounding()
        self._visual_search_pending = transition.current is Stage.SEARCH
        self._visual_search_reason = detail if self._visual_search_pending else ''
        if not self._visual_search_pending:
            self._visual_search_edge_direction = 0
        self._serial_gate = ObservationSerialGate(
            sync_slop_s=float(self.get_parameter('sync_slop_s').value),
            max_age_s=float(self.get_parameter('max_perception_age_s').value),
        )
        self._perception_valid = False
        self._valid_seen_at = None
        self._target_camera = None
        self._target_piper = None
        self._target_cloud = None
        self._target_uv = None
        self._scene_cloud = None
        self._camera_origin_piper = None
        self._camera_rotation_piper = None
        self._affordance = None
        self._pose_settle_until = None
        self._pose_settle_started_at = None
        self._pose_settle_last_tick_at = None
        self._visual_search_settle_reference = None
        MobileManipulationRuntime._clear_visual_search_stationarity(
            self,
            'recovery_lookout',
        )
        self._lookout_pending = True
        return True

    def _publish_place_abort_once(self) -> None:
        """Release place-node ownership once, before clearing its exact identity."""
        if (
            not getattr(self, '_place_transaction_requested', False)
            or getattr(self, '_place_transaction_abort_sent', False)
        ):
            return
        goal_id = getattr(self, '_place_goal_id', '')
        core = getattr(self, '_core', None)
        executor_epoch = '' if core is None else core.place_executor_epoch
        publisher = getattr(self, '_place_transaction_control_pub', None)
        if not goal_id or not executor_epoch or publisher is None:
            return
        self._place_transaction_abort_sent = True
        publisher.publish(String(data=transaction_abort_json(
            goal_id=goal_id,
            executor_epoch=executor_epoch,
        )))

    def _release_terminal_ownership(self) -> None:
        """Once per task, stop asynchronous work and reset owned perception."""
        self._coarse_nav_posture_violation_started_at_s = None
        MobileManipulationRuntime._publish_frozen_coarse_nav_authorization(
            self,
            False,
        )
        if getattr(self, '_terminal_ownership_released', False):
            return
        MobileManipulationRuntime._publish_place_abort_once(self)
        MobileManipulationRuntime._clear_debug_plan(self)
        self._terminal_ownership_released = True
        self._lookout_pending = False
        MobileManipulationRuntime._clear_near_view_settle(self)
        self._visual_search_pending = False
        self._pose_settle_until = None
        self._pose_settle_started_at = None
        self._pose_settle_last_tick_at = None
        self._visual_search_settle_reference = None
        MobileManipulationRuntime._clear_visual_search_stationarity(
            self,
            'terminal_release',
        )
        MobileManipulationRuntime._clear_visual_servo_vertical_recovery(
            self,
            'terminal_release',
        )
        self._reset_execution_occlusion()
        self._invalidate_async_work()
        self._invalidate_perception_session()

    def _apply_safety(self, action: SafetyAction) -> None:
        if self._core.phase in (
            RuntimePhase.PICK_COMPLETE,
            RuntimePhase.COMPLETE,
            RuntimePhase.CANCELED,
            RuntimePhase.FAILED,
        ):
            self._release_terminal_ownership()
        if action.stop_base:
            self._visual_search_active_pub.publish(Bool(data=False))
            self._publish_zero()
        if action.cancel_navigation:
            self._cancel_nav_pub.publish(Bool(data=True))
        if action.cancel_arm:
            self._arm_cancel_pub.publish(Bool(data=True))
        if action.reason:
            self.get_logger().error(action.reason)

    def _publish_status(self, *, force: bool = False) -> None:
        pregrasp_handoff = getattr(self, '_pregrasp_handoff', None)
        approach_anchor = getattr(self, '_approach_planning_anchor', None)
        value = json.dumps({
            'schema': 'z_manip.task_status.v1',
            'phase': self._core.phase.value,
            'task_stage': self._task.stage.value,
            'instruction': self._core.instruction,
            'perception_valid': self._perception_valid,
            'prospective_serial': self._core.prospective_serial,
            'required_replan_serial': self._core.required_replan_serial,
            'planned_serial': self._core.planned_serial,
            'pregrasp_plan_available': (
                getattr(self, '_pregrasp_program', None) is not None
            ),
            'approach_plan_available': getattr(self, '_program', None) is not None,
            'pregrasp_handoff': None if pregrasp_handoff is None else {
                'observation_serial': pregrasp_handoff.observation_serial,
                'executor_epoch': pregrasp_handoff.executor_epoch,
                'command_id': pregrasp_handoff.command_id,
                'trajectory_received_at': (
                    pregrasp_handoff.trajectory_received_at
                ),
                'completed_at_s': pregrasp_handoff.completed_at_s,
                'deadline_s': pregrasp_handoff.deadline_s,
                'minimum_joint_sequence': (
                    pregrasp_handoff.minimum_joint_sequence
                ),
                'minimum_joint_stamp_ns': (
                    pregrasp_handoff.minimum_joint_stamp_ns
                ),
                'stable_joint_sequence': (
                    self._pregrasp_stable_joint_sequence
                ),
                'stable_joint_stamp_ns': self._pregrasp_stable_joint_stamp_ns,
                'joint_error_rad': self._pregrasp_joint_error_rad,
            },
            'approach_planning_anchor': None if approach_anchor is None else {
                'observation_serial': approach_anchor.observation_serial,
                'observation_stamp_ns': (
                    approach_anchor.observation_identity.stamp_ns
                ),
                'joint_sequence': approach_anchor.joint_sequence,
                'joint_stamp_ns': approach_anchor.joint_stamp_ns,
            },
            'desired_camera_depth_m': self._desired_depth,
            'coarse_nav_ready': self._coarse_nav_ready,
            'prospective_base_displacement_m': self._approximate_displacement,
            'work_pose': self._work_pose,
            'work_pose_attempt_count': len(self._work_pose_history_map),
            'navigation_status_goal_id': self._navigation_status_goal_id or None,
            'navigation_status_phase': self._navigation_status_phase or None,
            'navigation_status_seen_s': self._navigation_status_seen_s,
            'navigation_goal_acknowledged': self._navigation_goal_acknowledged,
            'navigation_history_recorded': self._navigation_history_recorded,
            'coarse_nav_perception_loss_detail': (
                self._coarse_nav_perception_loss_detail
            ),
            'coarse_nav_arrival_started_at_s': (
                self._coarse_nav_arrival_started_at_s
            ),
            'coarse_nav_arrival_stable_since_s': (
                self._coarse_nav_arrival_stable_since_s
            ),
            'coarse_nav_arrival_stable_start_odom_stamp_ns': (
                self._coarse_nav_arrival_stable_start_odom_stamp_ns
            ),
            'coarse_nav_arrival_last_odom_sequence': (
                self._coarse_nav_arrival_last_odom_sequence
            ),
            'coarse_nav_arrival_last_odom_stamp_ns': (
                self._coarse_nav_arrival_last_odom_stamp_ns
            ),
            'near_view_pose': self._near_view_pose_name or None,
            'near_view_achieved_pose': (
                self._near_view_achieved_pose_name or None
            ),
            'near_view_settle_started_at_s': self._near_view_settle_started_at,
            'near_view_settle_last_tick_at_s': (
                self._near_view_settle_last_tick_at
            ),
            'near_view_settle_until_s': self._near_view_settle_until,
            'near_view_deadline_s': self._near_view_deadline_s,
            'near_view_joint_sequence_floor': (
                self._near_view_joint_sequence_floor
            ),
            'near_view_joint_error_rad': self._near_view_joint_error_rad,
            'frozen_coarse_nav_authorization_active': (
                self._frozen_coarse_nav_authorization_identity is not None
            ),
            'commanded_close_aperture_m': self._commanded_close_aperture,
            'semantic_point_selection': self._selection_mode,
            'affordance_generation': self._affordance_generation,
            'required_affordance_generation': self._required_affordance_generation,
            'perception_generation': self._perception_generation,
            'required_perception_generation': self._required_perception_generation,
            'required_perception_request_id': self._required_perception_request_id,
            'required_grounding_scope': self._required_grounding_scope,
            'bound_perception_request_id': self._bound_perception_request_id,
            'bound_perception_producer_epoch': self._bound_perception_producer_epoch,
            'bound_perception_generation': self._bound_perception_generation,
            'valid_perception_request_id': self._valid_perception_request_id,
            'valid_perception_producer_epoch': self._valid_perception_producer_epoch,
            'valid_perception_generation': self._valid_perception_generation,
            'valid_observation_stamp_ns': self._valid_observation_stamp_ns,
            'valid_observation_frame_id': self._valid_observation_frame_id,
            'execution_occlusion': {
                'armed': self._execution_occlusion.armed,
                'loss_active': self._execution_occlusion.loss_active,
                'contact_confirmed': self._execution_occlusion.contact_confirmed,
                'armed_at_s': self._execution_occlusion.armed_at_s,
                'loss_at_s': self._execution_occlusion.loss_at_s,
                'lift_sent_at_s': self._execution_occlusion.lift_sent_at_s,
                'lift_completed_at_s': (
                    self._execution_occlusion.lift_completed_at_s
                ),
                'request_id': self._execution_occlusion.request_id,
                'producer_epoch': self._execution_occlusion.producer_epoch,
                'generation': self._execution_occlusion.generation,
                'observation_serial': (
                    self._execution_occlusion.observation_serial
                ),
                'observation_stamp_ns': (
                    self._execution_occlusion.observation_stamp_ns
                ),
                'observation_frame_id': (
                    self._execution_occlusion.observation_frame_id
                ),
                'observation_stamp_s': (
                    self._execution_occlusion.observation_stamp_s
                ),
                'loss_observation_serial': (
                    self._execution_occlusion.loss_observation_serial
                ),
                'loss_observation_stamp_ns': (
                    self._execution_occlusion.loss_observation_stamp_ns
                ),
                'joint_sequence': self._execution_occlusion.joint_sequence,
                'joint_source_stamp_ns': (
                    self._execution_occlusion.joint_source_stamp_ns
                ),
                'loss_joint_sequence': (
                    self._execution_occlusion.loss_joint_sequence
                ),
                'loss_joint_source_stamp_ns': (
                    self._execution_occlusion.loss_joint_source_stamp_ns
                ),
                'mode': self._execution_occlusion_last_decision.mode,
                'allowed': self._execution_occlusion_last_decision.allowed,
                'reason': self._execution_occlusion_last_decision.reason,
                'lift_path_index': (
                    self._execution_occlusion_last_decision.path_index
                ),
                'loss_detail': self._execution_occlusion_loss_detail,
            },
            'retry_counters': dict(self._task.counters),
            'visual_search': {
                'pending': self._visual_search_pending,
                'active': self._visual_search.active,
                'attempt': self._visual_search.attempt,
                'target_offset_rad': self._visual_search.target_offset_rad,
                'yaw_error_rad': self._visual_search_error_rad,
                'allocated_timeout_s': self._visual_search.allocated_timeout_s,
                'position_hold_timeout_s': (
                    self._visual_search.config.position_hold_timeout_s
                ),
                'position_completion_tolerance_m': (
                    self._visual_search.config.position_completion_tolerance_m
                ),
                'moving_rebound_reacquire_m': (
                    self._visual_search.config.moving_rebound_reacquire_m
                ),
                'position_hold_started_at_s': (
                    self._visual_search.position_hold_started_at_s
                ),
                'last_update_at_s': self._visual_search.last_update_at_s,
                'pose_settle_started_at_s': self._pose_settle_started_at,
                'pose_settle_last_tick_at_s': self._pose_settle_last_tick_at,
                'measured_odom_sequence': self._odom_sequence,
                'measured_odom_stamp_ns': self._odom_stamp_ns,
                'settle_anchor_xy_m': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.position_anchor_xy
                ),
                'settle_target_yaw_rad': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.target_yaw_rad
                ),
                'settle_minimum_odom_sequence': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.minimum_odom_sequence
                ),
                'settle_minimum_odom_stamp_ns': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.minimum_odom_stamp_ns
                ),
                'settle_stationary_deadline_s': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.stationary_deadline_s
                ),
                'settle_stop_started_at_s': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.stop_started_at_s
                ),
                'settle_correction_deadline_s': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.correction_deadline_s
                ),
                'settle_absolute_deadline_s': (
                    None
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.absolute_deadline_s
                ),
                'settle_reacquire_count': (
                    0
                    if self._visual_search_settle_reference is None
                    else self._visual_search_settle_reference.reacquire_count
                ),
                'stationary_wait_timeout_s': (
                    self._visual_search.config.stationary_wait_timeout_s
                ),
                'stationary_quiet_window_s': (
                    self._visual_search.config.stationary_quiet_window_s
                ),
                'stationary_max_odom_gap_s': (
                    self._visual_search.config.stationary_max_odom_gap_s
                ),
                'settle_reacquire_budget_s': (
                    self._visual_search.config.settle_reacquire_budget_s
                ),
                'settle_stable_since_s': (
                    self._visual_search_stationarity.stable_since_received_at_s
                ),
                'settle_stable_start_odom_stamp_ns': (
                    self._visual_search_stationarity.stable_start_odom_stamp_ns
                ),
                'settle_stable_duration_s': (
                    self._visual_search_stationarity.stable_duration_s
                ),
                'settle_last_odom_sequence': (
                    self._visual_search_stationarity.last_odom_sequence
                ),
                'settle_last_odom_stamp_ns': (
                    self._visual_search_stationarity.last_odom_stamp_ns
                ),
                'settle_last_reset_reason': (
                    self._visual_search_stationarity.last_reset_reason
                ),
                'deadline_grace_s': self._visual_search.config.deadline_grace_s,
                'measured_base_linear_speed_mps': self._base_linear_speed_mps,
                'measured_base_angular_speed_rps': self._base_angular_speed_rps,
                'measured_base_yaw_rate_rps': getattr(
                    self,
                    '_base_yaw_rate_rps',
                    None,
                ),
                'settle_max_linear_speed_mps': (
                    self._visual_search.config.settle_max_linear_speed_mps
                ),
                'settle_max_angular_speed_rps': (
                    self._visual_search.config.settle_max_angular_speed_rps
                ),
                'position_heading_reacquire_tolerance_rad': (
                    self._visual_search.config
                    .position_heading_reacquire_tolerance_rad
                ),
                'planar_drift_m': self._visual_search.planar_drift_m,
                'position_anchor_frame': str(self.get_parameter(
                    'platform_odometry_parent_frame',
                ).value),
                'position_anchor_xy_m': self._visual_search.position_anchor_xy,
                'position_error_base_xy_m': self._visual_search.position_error_base_xy,
                'linear_command_base_xy_mps': (
                    self._visual_search.linear_command_base_xy
                ),
                'edge_direction': self._visual_search_edge_direction,
                'reason': self._visual_search_reason,
            },
            'place_mode': self._topic_value('place_mode'),
            'post_release_verification': {
                'expected_goal_id': (
                    None
                    if self._place_observation_identity is None
                    else self._place_observation_identity.goal_id
                ),
                'expected_release_gripper_command_id': (
                    self._post_release_release_command_id
                ),
                'expected_request_id': (
                    None
                    if self._place_observation_identity is None
                    else self._place_observation_identity.request_id
                ),
                'expected_producer_epoch': (
                    None
                    if self._place_observation_identity is None
                    else self._place_observation_identity.producer_epoch
                ),
                'expected_generation': (
                    None
                    if self._place_observation_identity is None
                    else self._place_observation_identity.generation
                ),
                'expected_frame_id': (
                    None
                    if self._place_observation_identity is None
                    else self._place_observation_identity.frame_id
                ),
                'expected_planning_observation_stamp_ns': (
                    None
                    if self._place_observation_identity is None
                    else self._place_observation_identity
                    .planning_observation_stamp_ns
                ),
                'verified': self._post_release_verified_evidence is not None,
                'sample_count': (
                    None
                    if self._post_release_verified_evidence is None
                    else self._post_release_verified_evidence.sample_count
                ),
                'stable_duration_s': (
                    None
                    if self._post_release_verified_evidence is None
                    else self._post_release_verified_evidence.stable_duration_s
                ),
            },
            'result': terminal_result(self._core.phase),
            'place_goal_id': self._place_goal_id,
            'place_plan_available': bool(self._place_programs),
            'failure': self._core.failure_reason,
        }, separators=(',', ':'))
        if force or value != self._last_status_json:
            self._status_pub.publish(String(data=value))
            self._last_status_json = value

    def _publish_debug_plan(self) -> None:
        if self._program is None or self._pregrasp_program is None:
            return
        now = self.get_clock().now().to_msg()
        frame = self._config.robot.platform_base_frame
        platform_t_piper = np.linalg.inv(self._piper_t_platform)
        markers = MarkerArray()
        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        poses = (
            ('pregrasp', self._program.pregrasp_pose, (0.1, 0.7, 1.0, 0.9)),
            ('grasp', self._program.grasp_pose, (0.1, 1.0, 0.2, 1.0)),
        )
        for index, (label, pose, color) in enumerate(poses):
            pose = platform_t_piper @ pose
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = frame
            marker.ns = 'z_manip_grasp'
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            start = pose[:3, 3]
            end = start + 0.12 * pose[:3, 2]
            marker.points = [Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
                             Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))]
            marker.scale.x = 0.015
            marker.scale.y = 0.03
            marker.scale.z = 0.04
            marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
            marker.text = label
            markers.markers.append(marker)
        self._markers_pub.publish(markers)

        path = Path()
        path.header.stamp = now
        path.header.frame_id = frame
        all_joints = np.vstack((
            self._pregrasp_program.transit.positions,
            self._program.approach.positions[1:],
            self._program.lift.positions[1:],
        ))
        stride = max(1, len(all_joints) // 180)
        for joints in all_joints[::stride]:
            transform = platform_t_piper @ self._planner.chain.forward(joints)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = transform[:3, 3]
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)

    def _clear_debug_plan(self) -> None:
        """Replace transient-local planning visuals with an explicit empty state."""
        marker_publisher = getattr(self, '_markers_pub', None)
        path_publisher = getattr(self, '_path_pub', None)
        config = getattr(self, '_config', None)
        if marker_publisher is None or path_publisher is None or config is None:
            return
        now = self.get_clock().now().to_msg()
        frame = config.robot.platform_base_frame
        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        markers = MarkerArray()
        markers.markers.append(clear)
        marker_publisher.publish(markers)
        path = Path()
        path.header.stamp = now
        path.header.frame_id = frame
        path_publisher.publish(path)

    def destroy_node(self) -> bool:
        """Stop both actuator channels and terminate the planning worker."""
        self._release_terminal_ownership()
        self._publish_zero()
        self._arm_cancel_pub.publish(Bool(data=True))
        self._worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Run the task runtime."""
    rclpy.init(args=args)
    node = MobileManipulationRuntime()
    # Exact-stamp lookups may wait while the TransformListener's reentrant
    # callback group consumes an already queued transform.
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        rclpy.spin(node, executor=executor)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
