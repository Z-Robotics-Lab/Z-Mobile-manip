"""Exact-time ROS 2 adapter for the external EdgeTAM HTTP service."""

from __future__ import annotations

from collections import deque, OrderedDict
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any
import uuid

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Empty, Header, String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

from .core import (
    AcquisitionGate,
    CameraIntrinsics,
    center_box_to_half_open,
    FailClosedTracker,
    MotionAdaptiveDepthFilter,
    project_scene_depth,
    register_seed_bbox_to_latest,
    ReseedRegistrationConfig,
    RgbdFrame,
    ServiceClient,
    TrackerFailure,
    TrackingObservation,
)


_SEED_REQUEST_SCHEMA = 'z_manip.seed_request.v1'
_SEED_OFFER_SCHEMA = 'z_manip.seed_offer.v1'
_SEED_STATUS_SCHEMA = 'z_manip.seed_status.v1'
_SEED_IMAGE_FORMAT_PREFIX = 'jpeg; z_manip_seed_offer='


class _ExactTimeSynchronizer:
    """Bounded exact-stamp join that never calls user code under its queue lock."""

    def __init__(self, filters: list[object], queue_size: int) -> None:
        if queue_size < 1 or not filters:
            raise ValueError('exact synchronizer requires filters and a positive queue')
        self.queue_size = int(queue_size)
        self.queues: list[dict[int, object]] = [{} for _ in filters]
        self._queue_lock = threading.Lock()
        self._dispatch_lock = threading.RLock()
        self._callback: Any = None
        self.input_connections = [
            source.registerCallback(self.add, index)
            for index, source in enumerate(filters)
        ]

    def registerCallback(self, callback: Any) -> Any:  # noqa: N802 - ROS API parity
        self._callback = callback
        return callback

    @staticmethod
    def _stamp_ns(message: object) -> int | None:
        try:
            stamp = message.header.stamp
            sec = stamp.sec
            nanosec = stamp.nanosec
        except AttributeError:
            return None
        if (
            isinstance(sec, bool)
            or isinstance(nanosec, bool)
            or not isinstance(sec, int)
            or not isinstance(nanosec, int)
            or sec < 0
            or not 0 <= nanosec < 1_000_000_000
        ):
            return None
        return sec * 1_000_000_000 + nanosec

    def add(self, message: object, queue_index: int) -> None:
        stamp_ns = self._stamp_ns(message)
        if stamp_ns is None:
            return
        messages: list[object] | None = None
        callback: Any = None
        # Serialize delivery order, but use a separate queue lock so malformed
        # input or callback exceptions cannot permanently wedge queue mutation.
        with self._dispatch_lock:
            with self._queue_lock:
                queue = self.queues[queue_index]
                queue[stamp_ns] = message
                while len(queue) > self.queue_size:
                    del queue[min(queue)]
                common = set(self.queues[0])
                for candidate in self.queues[1:]:
                    common.intersection_update(candidate)
                if common:
                    matched_stamp = min(common)
                    messages = [candidate[matched_stamp] for candidate in self.queues]
                    for candidate in self.queues:
                        for queued_stamp in tuple(candidate):
                            if queued_stamp <= matched_stamp:
                                del candidate[queued_stamp]
                    callback = self._callback
            if messages is not None and callback is not None:
                callback(*messages)


def _bounded_identity(value: object, label: str, *, max_length: int = 128) -> str:
    """Validate one printable, whitespace-free cross-node identity."""
    if not isinstance(value, str):
        raise ValueError(f'{label} must be a string')
    clean = value.strip()
    if (
        not clean
        or clean != value
        or len(clean) > max_length
        or any(ord(character) < 0x21 or ord(character) > 0x7e for character in clean)
    ):
        raise ValueError(f'{label} is invalid')
    return clean


def _parse_seed_request(payload: str) -> _SeedRequest:
    """Parse one complete bridge-to-adapter seed transaction command."""
    if not isinstance(payload, str) or not payload or len(payload) > 4096:
        raise ValueError('seed request is not a bounded string')

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('seed request has duplicate fields')
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (TypeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('seed request is not valid JSON') from error
    required = {
        'schema',
        'action',
        'request_id',
        'producer_epoch',
        'grounding_generation',
        'nonce',
        'source_stamp_floor_ns',
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get('schema') != _SEED_REQUEST_SCHEMA
        or value.get('action') not in {'arm', 'cancel'}
    ):
        raise ValueError('unsupported seed request envelope')
    integers = (
        value.get('grounding_generation'),
        value.get('source_stamp_floor_ns'),
    )
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 <= item <= (1 << 63) - 1
        for item in integers
    ):
        raise ValueError('seed request integers are invalid')
    nonce = _bounded_identity(value['nonce'], 'nonce')
    if len(nonce) != 32 or any(character not in '0123456789abcdef' for character in nonce):
        raise ValueError('nonce must be a lowercase UUID hex value')
    return _SeedRequest(
        action=value['action'],
        request_id=_bounded_identity(value['request_id'], 'request_id'),
        producer_epoch=_bounded_identity(value['producer_epoch'], 'producer_epoch'),
        grounding_generation=value['grounding_generation'],
        nonce=nonce,
        source_stamp_floor_ns=value['source_stamp_floor_ns'],
    )


@dataclass(frozen=True)
class _CachedRgb:
    stamp_ns: int
    frame_id: str
    image_jpeg: bytes
    width: int
    height: int


@dataclass(frozen=True)
class _SeedRequest:
    action: str
    request_id: str
    producer_epoch: str
    grounding_generation: int
    nonce: str
    source_stamp_floor_ns: int

    @property
    def identity(self) -> tuple[str, str, str, int, str, int]:
        return (
            self.action,
            self.request_id,
            self.producer_epoch,
            self.grounding_generation,
            self.nonce,
            self.source_stamp_floor_ns,
        )


@dataclass(frozen=True)
class _SeedOffer:
    request: _SeedRequest
    adapter_generation: int
    token: str
    frame: _CachedRgb
    deadline_steady_s: float


@dataclass(frozen=True)
class _Command:
    kind: str
    generation: int
    frame: RgbdFrame | _CachedRgb | None = None
    bbox_xyxy: tuple[int, int, int, int] | None = None
    label: str = ''
    seed_id: str = ''


@dataclass(frozen=True)
class _PublicationToken:
    generation: int
    seed_id: str
    seed_stamp_ns: int
    newly_ready: bool


@dataclass(frozen=True)
class _ObservationMessages:
    detections: Detection2DArray
    target: Detection3D
    cloud: PointCloud2
    manifest: String
    mask: Image
    overlay: Image
    scene_cloud: PointCloud2


class EdgeTamAdapter(Node):
    """Publish a controller-safe 2-D/3-D view of one persistent EdgeTAM track."""

    def __init__(self, *, service_client: ServiceClient | None = None) -> None:
        super().__init__('z_manip_edgetam')
        self._declare_parameters()
        self._cv = CvBridge()
        self._depth_filter = MotionAdaptiveDepthFilter(
            window_size=int(self.get_parameter('depth_filter_window_size').value),
            min_valid_fraction=float(
                self.get_parameter('depth_filter_min_valid_fraction').value,
            ),
            max_mad_m=float(self.get_parameter('depth_filter_max_mad_m').value),
            motion_threshold_m=float(
                self.get_parameter('depth_filter_motion_threshold_m').value,
            ),
            global_motion_fraction=float(
                self.get_parameter('depth_filter_global_motion_fraction').value,
            ),
            min_motion_pixels=int(
                self.get_parameter('depth_filter_min_motion_pixels').value,
            ),
            max_gap_s=float(self.get_parameter('depth_filter_max_gap_s').value),
        )
        self._state_lock = threading.RLock()
        self._worker_condition = threading.Condition(self._state_lock)
        self._commands: deque[_Command] = deque()
        self._cache: OrderedDict[int, _CachedRgb] = OrderedDict()
        self._seed_offer_armed = False
        self._pending_seed_request: _SeedRequest | None = None
        self._seed_owner_producer_epoch: str | None = None
        self._last_seed_request_identity: tuple[str, str, str, int, str, int] | None = None
        self._seed_request_started_steady_s: float | None = None
        self._seed_request_deadline_steady_s: float | None = None
        self._seed_offer: _SeedOffer | None = None
        self._generation = 0
        self._accept_frames = False
        self._tracking = False
        self._active_seed_id = ''
        self._acquisition_gate = AcquisitionGate(
            int(self.get_parameter('min_acquisition_live_updates').value),
        )
        self._replay_candidate_count = 0
        self._replay_selected_count = 0
        self._replay_span_ns = 0
        self._seed_stamp_ns: int | None = None
        self._last_sync_stamp_ns: int | None = None
        self._last_sync_ros_s: float | None = None
        self._tracking_started_ros_s: float | None = None
        self._last_result_ros_s: float | None = None
        self._stop_worker = False
        self._rgbd_condition = threading.Condition()
        self._rgbd_messages: deque[tuple[Image, Image, CameraInfo]] = deque()
        self._stop_rgbd_worker = False
        self._reseed_registration_config = self._registration_config()
        self._allow_short_span_seed_fallback = bool(
            self.get_parameter('allow_short_span_seed_fallback').value,
        )
        self._short_span_seed_fallback_max_s = float(
            self.get_parameter('short_span_seed_fallback_max_s').value,
        )
        self._short_span_seed_fallback_max_frames = int(
            self.get_parameter('short_span_seed_fallback_max_frames').value,
        )

        if service_client is None:
            from z_manip.perception.edgetam_service_client import EdgeTamServiceClient

            service_client = EdgeTamServiceClient(
                base_url=str(self.get_parameter('service_url').value),
                request_timeout_s=float(self.get_parameter('service_timeout_s').value),
                session_idle_timeout_s=float(
                    self.get_parameter('service_session_idle_timeout_s').value,
                ),
                min_mask_pixels=int(self.get_parameter('min_mask_pixels').value),
                min_score=float(self.get_parameter('min_score').value),
            )
        self._tracker = FailClosedTracker(
            service_client,
            min_depth_m=float(self.get_parameter('min_depth_m').value),
            max_depth_m=float(self.get_parameter('max_depth_m').value),
            min_points=int(self.get_parameter('min_cloud_points').value),
            max_points=int(self.get_parameter('max_cloud_points').value),
            min_mask_iou=float(self.get_parameter('min_mask_iou').value),
            hard_min_mask_iou=float(
                self.get_parameter('hard_min_mask_iou').value,
            ),
            min_mask_area_ratio=float(
                self.get_parameter('min_mask_area_ratio').value,
            ),
            max_mask_displacement_ratio=float(
                self.get_parameter('max_mask_displacement_ratio').value,
            ),
            min_mask_overlap_ratio=float(
                self.get_parameter('min_mask_overlap_ratio').value,
            ),
            min_mask_bbox_iou=float(
                self.get_parameter('min_mask_bbox_iou').value,
            ),
            max_soft_continuity_frames=int(
                self.get_parameter('max_soft_continuity_frames').value,
            ),
            min_soft_depth_mask_retention=float(
                self.get_parameter('min_soft_depth_mask_retention').value,
            ),
            allow_motion_reanchor=bool(
                self.get_parameter('allow_motion_reanchor').value,
            ),
            min_motion_reanchor_area_ratio=float(
                self.get_parameter('min_motion_reanchor_area_ratio').value,
            ),
            max_motion_reanchor_displacement_ratio=float(
                self.get_parameter(
                    'max_motion_reanchor_displacement_ratio',
                ).value,
            ),
            max_contained_collapse_recovery_frames=int(
                self.get_parameter(
                    'max_contained_collapse_recovery_frames',
                ).value,
            ),
            max_centroid_speed_mps=float(
                self.get_parameter('max_centroid_speed_mps').value,
            ),
            max_mask_area_ratio=float(
                self.get_parameter('max_mask_area_ratio').value,
            ),
            max_rejected_mask_ratio=float(
                self.get_parameter('max_rejected_mask_ratio').value,
            ),
            max_largest_rejected_to_selected_ratio=float(
                self.get_parameter(
                    'max_largest_rejected_to_selected_ratio',
                ).value,
            ),
            cluster_max_depth_jump_m=float(
                self.get_parameter('cluster_max_depth_jump_m').value,
            ),
            cluster_max_depth_jump_ratio=float(
                self.get_parameter('cluster_max_depth_jump_ratio').value,
            ),
        )

        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        tracking_state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        seed_offer_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._seed_image_pub = self.create_publisher(
            CompressedImage,
            self._topic('seed_image_topic'),
            seed_offer_qos,
        )
        self._seed_offer_manifest_pub = self.create_publisher(
            String,
            self._topic('seed_offer_manifest_topic'),
            seed_offer_qos,
        )
        self._seed_status_pub = self.create_publisher(
            String,
            self._topic('seed_status_topic'),
            reliable,
        )
        self._tracking_pub = self.create_publisher(
            Bool,
            self._topic('tracking_topic'),
            tracking_state_qos,
        )
        self._failure_pub = self.create_publisher(
            String,
            self._topic('failure_topic'),
            reliable,
        )
        self._frame_manifest_pub = self.create_publisher(
            String,
            self._topic('frame_manifest_topic'),
            reliable,
        )
        self._detections_pub = self.create_publisher(
            Detection2DArray,
            self._topic('detections_topic'),
            reliable,
        )
        self._target_pub = self.create_publisher(
            Detection3D,
            self._topic('selected_target_topic'),
            reliable,
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2,
            self._topic('selected_cloud_topic'),
            reliable,
        )
        self._mask_pub = self.create_publisher(
            Image,
            self._topic('mask_topic'),
            qos_profile_sensor_data,
        )
        self._overlay_pub = self.create_publisher(
            Image,
            self._topic('overlay_topic'),
            qos_profile_sensor_data,
        )
        self._scene_cloud_pub = self.create_publisher(
            PointCloud2,
            self._topic('scene_cloud_topic'),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Detection2DArray,
            self._topic('init_bbox_topic'),
            self._seed_cb,
            reliable,
        )
        self.create_subscription(
            String,
            self._topic('seed_request_topic'),
            self._seed_request_cb,
            seed_offer_qos,
        )
        self.create_subscription(
            Empty,
            self._topic('reset_topic'),
            self._reset_cb,
            reliable,
        )

        self._create_rgbd_subscriptions()

        self._rgbd_worker = threading.Thread(
            target=self._rgbd_worker_loop,
            name='z-manip-rgbd-preprocess',
            daemon=True,
        )
        self._rgbd_worker.start()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name='z-manip-edgetam-http',
            daemon=True,
        )
        self._worker.start()
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._watchdog = self.create_timer(
            float(self.get_parameter('watchdog_period_s').value),
            self._watchdog_cb,
            clock=self._steady_clock,
        )
        self._publish_tracking(False)
        self.get_logger().info(
            'ready: exact-time RGB-D -> external EdgeTAM persistent target',
        )

    def _declare_parameters(self) -> None:
        defaults: dict[str, Any] = {
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'init_bbox_topic': '/track_3d/init_bbox',
            'reset_topic': '/track_3d/reset',
            'seed_request_topic': '/track_3d/seed_request',
            'seed_image_topic': '/track_3d/exact_seed_image',
            'seed_offer_manifest_topic': '/track_3d/seed_offer_manifest',
            'seed_status_topic': '/track_3d/seed_status',
            'tracking_topic': '/track_3d/is_tracking',
            'failure_topic': '/track_3d/failure',
            'frame_manifest_topic': '/track_3d/frame_manifest',
            'detections_topic': '/track_3d/detections_2d',
            'selected_target_topic': '/track_3d/selected_target_3d',
            'selected_cloud_topic': '/track_3d/selected_target_pointcloud',
            'mask_topic': '/z_manip/perception/target_mask',
            'overlay_topic': '/z_manip/perception/overlay',
            'scene_cloud_topic': '/z_manip/perception/scene_pointcloud',
            'service_url': 'http://127.0.0.1:8092',
            'service_timeout_s': 5.0,
            'service_session_idle_timeout_s': 30.0,
            'min_mask_pixels': 24,
            'min_score': 0.35,
            'min_mask_iou': 0.15,
            'hard_min_mask_iou': 0.03,
            'min_mask_area_ratio': 0.35,
            'max_mask_displacement_ratio': 0.65,
            'min_mask_overlap_ratio': 0.50,
            'min_mask_bbox_iou': 0.10,
            'max_soft_continuity_frames': 2,
            'min_soft_depth_mask_retention': 0.35,
            'allow_motion_reanchor': False,
            'min_motion_reanchor_area_ratio': 0.60,
            'max_motion_reanchor_displacement_ratio': 1.25,
            'max_contained_collapse_recovery_frames': 2,
            'max_centroid_speed_mps': 2.0,
            'max_mask_area_ratio': 0.35,
            'max_rejected_mask_ratio': 0.08,
            'max_largest_rejected_to_selected_ratio': 0.20,
            'cluster_max_depth_jump_m': 0.06,
            'cluster_max_depth_jump_ratio': 0.03,
            'sync_queue_size': 20,
            'sync_processing_queue_size': 3,
            'frame_cache_size': 900,
            'reseed_roi_expansion_ratio': 0.75,
            'reseed_max_features': 500,
            'reseed_feature_quality': 0.01,
            'reseed_feature_min_distance_px': 4.0,
            'reseed_lk_window_px': 21,
            'reseed_lk_max_level': 3,
            'reseed_max_forward_backward_error_px': 1.5,
            'reseed_ransac_reproj_threshold_px': 2.0,
            'reseed_min_global_tracks': 20,
            'reseed_min_roi_tracks': 6,
            'reseed_min_global_inliers': 15,
            'reseed_min_roi_inliers': 5,
            'reseed_min_inlier_ratio': 0.60,
            'reseed_max_reprojection_rms_px': 1.75,
            'reseed_max_rotation_rad': 0.20,
            'reseed_max_scale_deviation': 0.12,
            'reseed_max_translation_ratio': 0.20,
            'reseed_max_global_roi_center_delta_ratio': 0.025,
            'reseed_max_global_roi_rotation_delta_rad': 0.04,
            'reseed_max_global_roi_scale_delta': 0.05,
            'reseed_min_bbox_retained_ratio': 0.80,
            # A local detector reply can arrive one camera frame after its
            # exact offered image.  Feature-poor scenes (plain boxes, walls)
            # cannot satisfy global LK registration even though the elapsed
            # time is too short for a meaningful target displacement.  When
            # explicitly enabled, initialise on the exact offered frame and
            # let the normal tracker consume the next live RGB-D frame.
            'allow_short_span_seed_fallback': False,
            'short_span_seed_fallback_max_s': 0.15,
            'short_span_seed_fallback_max_frames': 1,
            'min_acquisition_live_updates': 3,
            'max_acquisition_pending_frames': 1,
            # Serialized EdgeTAM inference can be slower than the camera. Keep
            # only the freshest queued update so latency cannot grow merely
            # because input Hz is higher than inference Hz.
            'max_pending_frames': 1,
            'sync_timeout_s': 0.5,
            'result_timeout_s': 11.0,
            'seed_offer_timeout_s': 105.0,
            'max_result_stamp_lag_s': 0.5,
            'watchdog_period_s': 0.05,
            'jpeg_quality': 85,
            'depth_16u_scale_m': 0.001,
            'depth_32f_scale_m': 1.0,
            'depth_filter_window_size': 5,
            'depth_filter_min_valid_fraction': 0.6,
            'depth_filter_max_mad_m': 0.006,
            'depth_filter_motion_threshold_m': 0.012,
            'depth_filter_global_motion_fraction': 0.15,
            'depth_filter_min_motion_pixels': 24,
            'depth_filter_max_gap_s': 0.5,
            'min_depth_m': 0.28,
            'max_depth_m': 2.5,
            'min_cloud_points': 24,
            'max_cloud_points': 20_000,
            'scene_cloud_stride': 3,
            'scene_cloud_max_points': 60_000,
            'target_mask_dilation_px': 3,
            'require_matching_frame_id': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        positive = (
            'service_timeout_s',
            'service_session_idle_timeout_s',
            'sync_timeout_s',
            'result_timeout_s',
            'seed_offer_timeout_s',
            'max_result_stamp_lag_s',
            'watchdog_period_s',
            'depth_16u_scale_m',
            'depth_32f_scale_m',
            'cluster_max_depth_jump_m',
            'depth_filter_max_mad_m',
            'depth_filter_motion_threshold_m',
            'depth_filter_global_motion_fraction',
            'depth_filter_max_gap_s',
            'short_span_seed_fallback_max_s',
        )
        if any(
            not math.isfinite(float(self.get_parameter(name).value))
            or float(self.get_parameter(name).value) <= 0.0
            for name in positive
        ):
            raise ValueError('EdgeTAM timeouts and depth scales must be positive')
        cluster_ratio = float(
            self.get_parameter('cluster_max_depth_jump_ratio').value,
        )
        if not math.isfinite(cluster_ratio) or cluster_ratio < 0.0:
            raise ValueError('cluster_max_depth_jump_ratio cannot be negative')
        integers = (
            'min_mask_pixels',
            'sync_queue_size',
            'sync_processing_queue_size',
            'frame_cache_size',
            'min_acquisition_live_updates',
            'max_acquisition_pending_frames',
            'max_pending_frames',
            'max_contained_collapse_recovery_frames',
            'min_cloud_points',
            'max_cloud_points',
            'scene_cloud_stride',
            'scene_cloud_max_points',
            'depth_filter_window_size',
            'depth_filter_min_motion_pixels',
            'short_span_seed_fallback_max_frames',
        )
        if any(int(self.get_parameter(name).value) < 1 for name in integers):
            raise ValueError('EdgeTAM queue and point limits must be positive')
        if int(
            self.get_parameter('max_contained_collapse_recovery_frames').value,
        ) > 2:
            raise ValueError(
                'max_contained_collapse_recovery_frames cannot exceed two',
            )
        if int(self.get_parameter('target_mask_dilation_px').value) < 0:
            raise ValueError('target_mask_dilation_px cannot be negative')
        filter_fraction = float(
            self.get_parameter('depth_filter_min_valid_fraction').value,
        )
        filter_motion_fraction = float(
            self.get_parameter('depth_filter_global_motion_fraction').value,
        )
        if not (
            math.isfinite(filter_fraction)
            and 0.0 < filter_fraction <= 1.0
            and 0.0 < filter_motion_fraction <= 1.0
        ):
            raise ValueError('depth-filter fractions must be in (0, 1]')
        min_iou = float(self.get_parameter('min_mask_iou').value)
        hard_min_iou = float(self.get_parameter('hard_min_mask_iou').value)
        min_area_ratio = float(self.get_parameter('min_mask_area_ratio').value)
        max_displacement = float(
            self.get_parameter('max_mask_displacement_ratio').value,
        )
        min_overlap_ratio = float(
            self.get_parameter('min_mask_overlap_ratio').value,
        )
        min_bbox_iou = float(self.get_parameter('min_mask_bbox_iou').value)
        min_depth_retention = float(
            self.get_parameter('min_soft_depth_mask_retention').value,
        )
        min_motion_reanchor_area = float(
            self.get_parameter('min_motion_reanchor_area_ratio').value,
        )
        max_motion_reanchor_displacement = float(
            self.get_parameter(
                'max_motion_reanchor_displacement_ratio',
            ).value,
        )
        max_soft_frames = int(
            self.get_parameter('max_soft_continuity_frames').value,
        )
        max_speed = float(self.get_parameter('max_centroid_speed_mps').value)
        if not (
            math.isfinite(min_iou) and 0.0 < min_iou <= 1.0
            and math.isfinite(hard_min_iou)
            and 0.0 <= hard_min_iou < min_iou
            and math.isfinite(min_area_ratio)
            and 0.0 < min_area_ratio <= 1.0
            and math.isfinite(max_displacement)
            and max_displacement > 0.0
            and math.isfinite(min_overlap_ratio)
            and 0.0 < min_overlap_ratio <= 1.0
            and math.isfinite(min_bbox_iou)
            and 0.0 < min_bbox_iou <= 1.0
            and max_soft_frames >= 0
            and math.isfinite(min_depth_retention)
            and 0.0 < min_depth_retention <= 1.0
            and math.isfinite(min_motion_reanchor_area)
            and 0.0 < min_motion_reanchor_area <= 1.0
            and math.isfinite(max_motion_reanchor_displacement)
            and max_motion_reanchor_displacement > 0.0
            and math.isfinite(max_speed) and max_speed > 0.0
        ):
            raise ValueError('tracking continuity thresholds are invalid')
        quality = int(self.get_parameter('jpeg_quality').value)
        if not 1 <= quality <= 100:
            raise ValueError('jpeg_quality must be in [1, 100]')

    def _registration_config(self) -> ReseedRegistrationConfig:
        """Build the immutable latest-frame reseed contract from ROS parameters."""
        return ReseedRegistrationConfig(
            roi_expansion_ratio=float(
                self.get_parameter('reseed_roi_expansion_ratio').value,
            ),
            max_features=int(self.get_parameter('reseed_max_features').value),
            feature_quality=float(
                self.get_parameter('reseed_feature_quality').value,
            ),
            feature_min_distance_px=float(
                self.get_parameter('reseed_feature_min_distance_px').value,
            ),
            lk_window_px=int(self.get_parameter('reseed_lk_window_px').value),
            lk_max_level=int(self.get_parameter('reseed_lk_max_level').value),
            max_forward_backward_error_px=float(
                self.get_parameter(
                    'reseed_max_forward_backward_error_px',
                ).value,
            ),
            ransac_reproj_threshold_px=float(
                self.get_parameter(
                    'reseed_ransac_reproj_threshold_px',
                ).value,
            ),
            min_global_tracks=int(
                self.get_parameter('reseed_min_global_tracks').value,
            ),
            min_roi_tracks=int(
                self.get_parameter('reseed_min_roi_tracks').value,
            ),
            min_global_inliers=int(
                self.get_parameter('reseed_min_global_inliers').value,
            ),
            min_roi_inliers=int(
                self.get_parameter('reseed_min_roi_inliers').value,
            ),
            min_inlier_ratio=float(
                self.get_parameter('reseed_min_inlier_ratio').value,
            ),
            max_reprojection_rms_px=float(
                self.get_parameter('reseed_max_reprojection_rms_px').value,
            ),
            max_rotation_rad=float(
                self.get_parameter('reseed_max_rotation_rad').value,
            ),
            max_scale_deviation=float(
                self.get_parameter('reseed_max_scale_deviation').value,
            ),
            max_translation_ratio=float(
                self.get_parameter('reseed_max_translation_ratio').value,
            ),
            max_global_roi_center_delta_ratio=float(
                self.get_parameter(
                    'reseed_max_global_roi_center_delta_ratio',
                ).value,
            ),
            max_global_roi_rotation_delta_rad=float(
                self.get_parameter(
                    'reseed_max_global_roi_rotation_delta_rad',
                ).value,
            ),
            max_global_roi_scale_delta=float(
                self.get_parameter('reseed_max_global_roi_scale_delta').value,
            ),
            min_bbox_retained_ratio=float(
                self.get_parameter('reseed_min_bbox_retained_ratio').value,
            ),
        )

    def _topic(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _create_rgbd_subscriptions(self) -> None:
        """Create one fresh exact-time RGB-D reader set."""
        color_sub = message_filters.Subscriber(
            self,
            Image,
            self._topic('color_topic'),
            qos_profile=qos_profile_sensor_data,
        )
        depth_sub = message_filters.Subscriber(
            self,
            Image,
            self._topic('depth_topic'),
            qos_profile=qos_profile_sensor_data,
        )
        info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            self._topic('camera_info_topic'),
            qos_profile=qos_profile_sensor_data,
        )
        self._sync_subscribers = (color_sub, depth_sub, info_sub)
        self._synchronizer = _ExactTimeSynchronizer(
            list(self._sync_subscribers),
            int(self.get_parameter('sync_queue_size').value),
        )
        # Keep ROS subscription callbacks bounded.  Image decoding, depth
        # conversion and point-cloud preparation happen on a dedicated worker,
        # which also collapses backlog to the freshest exact RGB-D triple.
        self._synchronizer.registerCallback(self._enqueue_synchronized_cb)

    def _recreate_rgbd_subscriptions(self) -> None:
        """Replace stale DDS readers after a proven exact-input timeout."""
        old_subscribers = tuple(getattr(self, '_sync_subscribers', ()))
        self._sync_subscribers = ()
        self._synchronizer = None
        for subscriber in old_subscribers:
            self.destroy_subscription(subscriber.sub)
        with self._rgbd_condition:
            self._rgbd_messages.clear()
        self._depth_filter.reset()
        with self._state_lock:
            self._last_sync_ros_s = None
        self._create_rgbd_subscriptions()
        self.get_logger().warn(
            'recreated RGB-D DDS readers after synchronization timeout',
        )

    def _publish_seed_status(
        self,
        event: str,
        detail: str,
        *,
        offer: _SeedOffer | None = None,
        request: _SeedRequest | None = None,
        adapter_generation: int | None = None,
    ) -> None:
        """Publish bounded diagnostics without changing tracker state."""
        if request is None and offer is not None:
            request = offer.request
        generation = (
            offer.adapter_generation
            if offer is not None
            else self._generation if adapter_generation is None else adapter_generation
        )
        message = String(data=json.dumps(
            {
                'schema': _SEED_STATUS_SCHEMA,
                'event': str(event).replace('\n', ' ').strip()[:128],
                'detail': str(detail).replace('\n', ' ').strip()[:256],
                'adapter_generation': int(generation),
                'offer_token': '' if offer is None else offer.token,
                'request_nonce': '' if request is None else request.nonce,
            },
            separators=(',', ':'),
        ))
        self._seed_status_pub.publish(message)

    def _publish_seed_offer(self, offer: _SeedOffer) -> None:
        """Publish the pinned JPEG and its complete causal manifest."""
        image = CompressedImage()
        image.header = self._make_header(offer.frame.stamp_ns, offer.frame.frame_id)
        image.format = f'{_SEED_IMAGE_FORMAT_PREFIX}{offer.token}'
        image.data = offer.frame.image_jpeg
        manifest = String(data=json.dumps(
            {
                'schema': _SEED_OFFER_SCHEMA,
                'request_id': offer.request.request_id,
                'producer_epoch': offer.request.producer_epoch,
                'grounding_generation': offer.request.grounding_generation,
                'request_nonce': offer.request.nonce,
                'adapter_generation': offer.adapter_generation,
                'offer_token': offer.token,
                'stamp_ns': offer.frame.stamp_ns,
                'frame_id': offer.frame.frame_id,
                'width': offer.frame.width,
                'height': offer.frame.height,
            },
            separators=(',', ':'),
        ))
        self._seed_image_pub.publish(image)
        self._seed_offer_manifest_pub.publish(manifest)

    def _seed_request_cb(self, msg: String) -> None:
        """Apply an ordered, idempotent arm/cancel transaction command."""
        try:
            request = _parse_seed_request(msg.data)
        except ValueError as error:
            self.get_logger().warn(f'ignored invalid seed request: {error}')
            self._publish_seed_status('invalid_request', str(error))
            return
        replay: _SeedOffer | None = None
        duplicate_event = ''
        ignored_event = ''
        ignored_detail = ''
        with self._state_lock:
            owner = self._seed_owner_producer_epoch
            if owner is None and request.action != 'arm':
                ignored_event = 'unowned_cancel_ignored'
                ignored_detail = 'cancel cannot claim an unowned adapter process'
            elif owner is not None and request.producer_epoch != owner:
                ignored_event = 'foreign_producer_ignored'
                ignored_detail = (
                    'seed request producer does not own this adapter process'
                )
            elif request.identity == self._last_seed_request_identity:
                if (
                    request.action == 'arm'
                    and self._seed_offer is not None
                    and self._seed_offer.request.identity == request.identity
                ):
                    replay = self._seed_offer
                    duplicate_event = 'duplicate_offer_republished'
                else:
                    duplicate_event = 'duplicate_request_ignored'
            else:
                if owner is None:
                    self._seed_owner_producer_epoch = request.producer_epoch
                self._last_seed_request_identity = request.identity
                self._reset_session(
                    seed_request=request if request.action == 'arm' else None,
                )
        if ignored_event:
            self._publish_seed_status(
                ignored_event,
                ignored_detail,
                request=request,
            )
            self.get_logger().warn(
                f'ignored seed request from {request.producer_epoch}: '
                f'{ignored_detail}',
            )
            return
        if duplicate_event:
            if replay is not None:
                self._publish_seed_offer(replay)
            self._publish_seed_status(
                duplicate_event,
                'idempotent seed request did not reset the active session',
                offer=replay,
                request=request,
            )
            return
        self._publish_seed_status(
            'armed' if request.action == 'arm' else 'cancelled',
            'seed transaction accepted',
            request=request,
        )

    def _synchronized_cb_guarded(
        self,
        color_msg: Image,
        depth_msg: Image,
        info_msg: CameraInfo,
    ) -> None:
        try:
            self._synchronized_cb(color_msg, depth_msg, info_msg)
        except Exception as error:
            detail = f'exact-time RGB-D callback failed ({type(error).__name__})'
            try:
                self.get_logger().error(
                    f'unexpected {detail}; failing closed',
                )
                self._fail_closed(
                    detail,
                    reason_code='rgbd_callback_exception',
                )
            except Exception:
                # This guard exists specifically because TimeSynchronizer does
                # not unlock when its callback raises.  A secondary logging or
                # shutdown failure must not escape either.
                pass

    def _enqueue_synchronized_cb(
        self,
        color_msg: Image,
        depth_msg: Image,
        info_msg: CameraInfo,
    ) -> None:
        """Record exact-input freshness and hand preprocessing to a worker."""
        stamps = tuple(
            _ExactTimeSynchronizer._stamp_ns(message)
            for message in (color_msg, depth_msg, info_msg)
        )
        if None in stamps or len(set(stamps)) != 1:
            self._fail_if_armed('exact-time RGB-D enqueue received invalid stamps')
            return
        now = self._now_s()
        with self._state_lock:
            self._last_sync_ros_s = now
        limit = int(self.get_parameter('sync_processing_queue_size').value)
        with self._rgbd_condition:
            if self._stop_rgbd_worker:
                return
            while len(self._rgbd_messages) >= limit:
                self._rgbd_messages.popleft()
            self._rgbd_messages.append((color_msg, depth_msg, info_msg))
            self._rgbd_condition.notify()

    def _rgbd_worker_loop(self) -> None:
        """Preprocess only the freshest synchronized frame outside ROS callbacks."""
        while True:
            with self._rgbd_condition:
                while not self._rgbd_messages and not self._stop_rgbd_worker:
                    self._rgbd_condition.wait()
                if self._stop_rgbd_worker:
                    return
                messages = self._rgbd_messages.pop()
                self._rgbd_messages.clear()
            self._synchronized_cb_guarded(*messages)

    def _synchronized_cb(
        self,
        color_msg: Image,
        depth_msg: Image,
        info_msg: CameraInfo,
    ) -> None:
        try:
            frame = self._make_frame(color_msg, depth_msg, info_msg)
        except Exception as error:
            self._fail_if_armed(f'synchronized RGB-D rejected ({type(error).__name__})')
            return
        now = self._now_s()
        steady_now = self._steady_now_s()
        failure: str | None = None
        failure_code = 'tracker_failure'
        offer_to_publish: _SeedOffer | None = None
        with self._state_lock:
            if (
                self._last_sync_stamp_ns is not None
                and frame.stamp_ns <= self._last_sync_stamp_ns
            ):
                if self._accept_frames:
                    failure = 'synchronized RGB-D timestamp is duplicate or out of order'
                    failure_code = 'frame_order'
                else:
                    self._cache.clear()
            self._last_sync_stamp_ns = frame.stamp_ns
            self._last_sync_ros_s = now
            cached = _CachedRgb(
                stamp_ns=frame.stamp_ns,
                frame_id=frame.frame_id,
                image_jpeg=frame.image_jpeg,
                width=frame.width,
                height=frame.height,
            )
            self._cache[frame.stamp_ns] = cached
            cache_size = int(self.get_parameter('frame_cache_size').value)
            while len(self._cache) > cache_size:
                self._cache.popitem(last=False)
            request = self._pending_seed_request
            deadline = self._seed_request_deadline_steady_s
            if (
                self._seed_offer_armed
                and request is not None
                and deadline is not None
                and steady_now <= deadline
                and frame.stamp_ns > request.source_stamp_floor_ns
                and not self._accept_frames
            ):
                offer_to_publish = _SeedOffer(
                    request=request,
                    adapter_generation=self._generation,
                    token=f'z-manip-seed:{self._generation}:{uuid.uuid4().hex}',
                    frame=cached,
                    deadline_steady_s=deadline,
                )
                # The offer owns its JPEG independently of normal cache eviction.
                self._seed_offer = offer_to_publish
                self._seed_offer_armed = False
            if failure is None and self._accept_frames:
                if self._seed_stamp_ns is not None and frame.stamp_ns > self._seed_stamp_ns:
                    failure = self._enqueue_frame_locked(frame)
                    if failure is None:
                        self._worker_condition.notify()
                    else:
                        failure_code = 'update_queue_overflow'
        if offer_to_publish is not None:
            self._publish_seed_offer(offer_to_publish)
            self._publish_seed_status(
                'offered',
                'published exact admitted RGB-D seed offer',
                offer=offer_to_publish,
            )
        if failure is not None:
            self._fail_closed(failure, reason_code=failure_code)

    def _enqueue_frame_locked(self, frame: RgbdFrame) -> str | None:
        """Queue a live frame while retaining only the freshest bounded window."""
        generation = self._generation
        limit = int(
            self.get_parameter(
                'max_pending_frames'
                if self._tracking
                else 'max_acquisition_pending_frames',
            ).value,
        )
        pending = sum(
            command.kind == 'frame' and command.generation == generation
            for command in self._commands
        )
        while pending >= limit:
            for index, command in enumerate(self._commands):
                if command.kind == 'frame' and command.generation == generation:
                    del self._commands[index]
                    pending -= 1
                    break
            else:
                # Parameter validation guarantees a positive limit, so this is
                # only reachable if queue accounting is internally inconsistent.
                return 'EdgeTAM update queue accounting failed'
        self._commands.append(
            _Command(
                'frame',
                generation,
                frame=frame,
                seed_id=self._active_seed_id,
            ),
        )
        return None

    def _make_frame(
        self,
        color_msg: Image,
        depth_msg: Image,
        info_msg: CameraInfo,
    ) -> RgbdFrame:
        stamps = (
            self._stamp_ns(color_msg.header),
            self._stamp_ns(depth_msg.header),
            self._stamp_ns(info_msg.header),
        )
        if len(set(stamps)) != 1:
            raise ValueError('RGB, aligned depth, and K timestamps must be exact')
        frame_ids = (
            color_msg.header.frame_id.strip(),
            depth_msg.header.frame_id.strip(),
            info_msg.header.frame_id.strip(),
        )
        if any(not frame_id for frame_id in frame_ids):
            raise ValueError('RGB-D frame IDs must not be empty')
        if (
            bool(self.get_parameter('require_matching_frame_id').value)
            and len(set(frame_ids)) != 1
        ):
            raise ValueError('RGB, aligned depth, and K frame IDs must match')
        bgr = np.asarray(
            self._cv.imgmsg_to_cv2(color_msg, desired_encoding='bgr8'),
        )
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('color image must have three channels')
        height, width = bgr.shape[:2]
        if (
            int(depth_msg.width) != width
            or int(depth_msg.height) != height
            or int(info_msg.width) != width
            or int(info_msg.height) != height
        ):
            raise ValueError('RGB, aligned depth, and K dimensions must match')
        quality = int(self.get_parameter('jpeg_quality').value)
        ok, encoded = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            raise ValueError('JPEG encoding failed')
        depth_raw = np.asarray(
            self._cv.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'),
        )
        if depth_raw.shape != (height, width):
            raise ValueError('aligned depth must be a single-channel image')
        encoding = depth_msg.encoding.upper()
        if encoding in ('16UC1', 'MONO16') and depth_raw.dtype == np.uint16:
            scale = float(self.get_parameter('depth_16u_scale_m').value)
        elif encoding == '32FC1' and np.issubdtype(depth_raw.dtype, np.floating):
            scale = float(self.get_parameter('depth_32f_scale_m').value)
        else:
            raise ValueError(f'unsupported aligned depth encoding {depth_msg.encoding!r}')
        raw_depth_m = np.asarray(depth_raw, dtype=np.float32) * scale
        depth_m, depth_filter = self._depth_filter.update(
            raw_depth_m,
            stamp_ns=stamps[0],
        )
        depth_m.setflags(write=False)
        k = tuple(float(value) for value in info_msg.k)
        if len(k) != 9 or not all(math.isfinite(value) for value in k):
            raise ValueError('camera K must contain nine finite values')
        return RgbdFrame(
            stamp_ns=stamps[0],
            frame_id=frame_ids[0],
            image_jpeg=encoded.tobytes(),
            width=width,
            height=height,
            depth_m=depth_m,
            intrinsics=CameraIntrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5]),
            depth_filter=depth_filter,
        )

    def _ignore_seed_bbox(
        self,
        event: str,
        detail: str,
        *,
        offer: _SeedOffer | None = None,
    ) -> None:
        """Diagnose an untrusted bbox without disturbing any active session."""
        self._publish_seed_status(event, detail, offer=offer)
        self.get_logger().warn(f'ignored EdgeTAM seed bbox: {detail}')

    def _seed_offer_state_locked(self, offer: _SeedOffer) -> str:
        """Classify one offer at the exact point where a bbox may commit."""
        if (
            self._seed_offer is not offer
            or self._generation != offer.adapter_generation
            or self._pending_seed_request is None
            or self._pending_seed_request.identity != offer.request.identity
        ):
            return 'stale'
        started = self._seed_request_started_steady_s
        deadline = self._seed_request_deadline_steady_s
        now = self._steady_now_s()
        if (
            started is None
            or deadline is None
            or deadline != offer.deadline_steady_s
            or not math.isfinite(now)
            or now < started
            or now > deadline
        ):
            return 'expired'
        return 'current'

    def _reject_noncurrent_seed_offer(
        self,
        offer: _SeedOffer,
        state: str,
        *,
        detail: str,
    ) -> None:
        """Ignore replacement races and synchronously release expired pins."""
        if state != 'expired':
            self._ignore_seed_bbox('stale_bbox', detail, offer=offer)
            return
        with self._state_lock:
            if self._seed_offer is not offer:
                stale = True
                request = None
                generation = offer.adapter_generation
            else:
                stale = False
                request = offer.request
                generation = self._generation
                self._reset_session()
        if stale:
            self._ignore_seed_bbox(
                'stale_bbox',
                'seed offer changed while its expired bbox was rejected',
                offer=offer,
            )
            return
        self._publish_seed_status(
            'expired',
            detail,
            offer=offer,
            request=request,
            adapter_generation=generation,
        )
        self.get_logger().warn(f'rejected expired EdgeTAM seed bbox: {detail}')

    def _seed_cb(self, msg: Detection2DArray) -> None:
        with self._state_lock:
            offer = self._seed_offer
            active_seed_id = self._active_seed_id
        if len(msg.detections) != 1:
            self._ignore_seed_bbox(
                'invalid_bbox',
                'initialization requires exactly one detection',
                offer=offer,
            )
            return
        detection = msg.detections[0]
        try:
            seed_id = _bounded_identity(detection.id, 'seed_id', max_length=256)
        except (AttributeError, ValueError):
            self._ignore_seed_bbox(
                'invalid_bbox',
                'initialization seed identity is invalid',
                offer=offer,
            )
            return
        if offer is None:
            event = (
                'duplicate_bbox'
                if seed_id == active_seed_id and active_seed_id
                else 'unmatched_bbox'
            )
            self._ignore_seed_bbox(
                event,
                'no matching seed offer is outstanding',
            )
            return
        if seed_id != offer.token:
            self._ignore_seed_bbox(
                'unmatched_bbox',
                'bbox token does not match the current offer',
                offer=offer,
            )
            return
        try:
            stamp_ns = self._stamp_ns(msg.header)
            detection_stamp_ns = self._stamp_ns(detection.header)
        except (AttributeError, TypeError, ValueError, OverflowError):
            self._ignore_seed_bbox(
                'invalid_bbox',
                'matching bbox timestamp is invalid',
                offer=offer,
            )
            return
        cached = offer.frame
        if cached.stamp_ns != stamp_ns or detection_stamp_ns != stamp_ns:
            self._ignore_seed_bbox(
                'invalid_bbox',
                'matching bbox timestamps differ from the current offer',
                offer=offer,
            )
            return
        try:
            frame_ids = (
                msg.header.frame_id.strip(),
                detection.header.frame_id.strip(),
                cached.frame_id,
            )
        except AttributeError:
            frame_ids = ('', '', cached.frame_id)
        if any(not frame_id for frame_id in frame_ids) or len(set(frame_ids)) != 1:
            self._ignore_seed_bbox(
                'invalid_bbox',
                'matching bbox frame IDs differ from the current offer',
                offer=offer,
            )
            return
        try:
            theta = float(detection.bbox.center.theta)
            if not math.isfinite(theta) or abs(theta) > 1e-9:
                raise ValueError('EdgeTAM seed must be axis aligned')
            bbox = center_box_to_half_open(
                float(detection.bbox.center.position.x),
                float(detection.bbox.center.position.y),
                float(detection.bbox.size_x),
                float(detection.bbox.size_y),
                width=cached.width,
                height=cached.height,
            )
        except (AttributeError, TypeError, ValueError):
            self._ignore_seed_bbox(
                'invalid_bbox',
                'matching bbox geometry is invalid',
                offer=offer,
            )
            return
        label = ''
        try:
            if detection.results:
                label = str(detection.results[0].hypothesis.class_id).strip()
        except AttributeError:
            label = ''
        with self._state_lock:
            offer_state = self._seed_offer_state_locked(offer)
            if offer_state == 'current':
                replay_candidates = tuple(
                    frame
                    for frame_stamp, frame in self._cache.items()
                    if frame_stamp > stamp_ns
                )
            else:
                replay_candidates = ()
        if offer_state != 'current':
            self._reject_noncurrent_seed_offer(
                offer,
                offer_state,
                detail='seed offer changed or expired before bbox registration',
            )
            return
        replay_span_ns = (
            replay_candidates[-1].stamp_ns - stamp_ns
            if replay_candidates
            else 0
        )
        init_frame = cached
        registration = None
        registration_fallback = False
        latest_compatible = False
        try:
            if replay_candidates:
                latest = replay_candidates[-1]
                if (
                    latest.frame_id != cached.frame_id
                    or latest.width != cached.width
                    or latest.height != cached.height
                ):
                    raise TrackerFailure(
                        'latest reseed frame changed camera identity or dimensions',
                        reason_code='seed_reseed_registration',
                    )
                latest_compatible = True
                registration = register_seed_bbox_to_latest(
                    cached.image_jpeg,
                    latest.image_jpeg,
                    bbox,
                    width=cached.width,
                    height=cached.height,
                    config=self._reseed_registration_config,
                )
                init_frame = latest
                bbox = registration.bbox_xyxy
        except TrackerFailure as error:
            fallback_allowed = (
                error.reason_code == 'seed_reseed_registration'
                and latest_compatible
                and bool(getattr(self, '_allow_short_span_seed_fallback', False))
                and len(replay_candidates) <= int(
                    getattr(self, '_short_span_seed_fallback_max_frames', 0),
                )
                and replay_span_ns <= int(
                    float(
                        getattr(self, '_short_span_seed_fallback_max_s', 0.0),
                    ) * 1_000_000_000
                )
            )
            if fallback_allowed:
                # Preserve the exact seed/bbox contract.  Do not pretend that
                # registration succeeded and do not publish tracking yet;
                # acquisition still requires depth-validated live updates.
                init_frame = cached
                registration_fallback = True
                self.get_logger().warn(
                    f'short-span reseed registration fallback: '
                    f'{len(replay_candidates)} frame(s), '
                    f'{replay_span_ns * 1e-9:.3f}s; {error}',
                )
            else:
                with self._state_lock:
                    offer_state = self._seed_offer_state_locked(offer)
                    if offer_state == 'current':
                        self._replay_candidate_count = len(replay_candidates)
                        self._replay_selected_count = 1 if replay_candidates else 0
                        self._replay_span_ns = replay_span_ns
                if offer_state != 'current':
                    self._reject_noncurrent_seed_offer(
                        offer,
                        offer_state,
                        detail='seed offer changed or expired during bbox registration',
                    )
                    return
                self._fail_closed(
                    str(error),
                    generation=offer.adapter_generation,
                    reason_code=error.reason_code,
                    seed_id=seed_id,
                    seed_stamp_ns=stamp_ns,
                )
                return
        with self._state_lock:
            offer_state = self._seed_offer_state_locked(offer)
            if offer_state != 'current':
                generation = None
            else:
                generation = self._generation
                self._accept_frames = True
                self._tracking = False
                self._active_seed_id = seed_id
                self._acquisition_gate.reset()
                self._replay_candidate_count = len(replay_candidates)
                self._replay_selected_count = (
                    1 if replay_candidates and not registration_fallback else 0
                )
                self._replay_span_ns = replay_span_ns
                self._seed_stamp_ns = stamp_ns
                self._seed_offer = None
                self._seed_offer_armed = False
                self._pending_seed_request = None
                self._seed_request_started_steady_s = None
                self._seed_request_deadline_steady_s = None
                self._tracking_started_ros_s = self._now_s()
                self._last_result_ros_s = None
                self._commands.clear()
                self._commands.append(
                    _Command(
                        'init',
                        generation,
                        frame=init_frame,
                        bbox_xyxy=bbox,
                        label=label,
                        seed_id=seed_id,
                    ),
                )
                self._worker_condition.notify()
                self._publish_tracking(False)
        if offer_state != 'current':
            self._reject_noncurrent_seed_offer(
                offer,
                offer_state,
                detail='seed offer changed or expired during bbox registration',
            )
            return
        replay_span_s = replay_span_ns * 1e-9
        if registration_fallback:
            detail = 'short-span fallback kept exact seed frame'
        elif registration is None:
            detail = 'exact seed frame required no latest-frame transfer'
        else:
            detail = (
                f'latest reseed global={registration.global_inliers}/'
                f'{registration.global_tracks} ROI={registration.roi_inliers}/'
                f'{registration.roi_tracks} center_delta='
                f'{registration.center_delta_ratio:.4f}'
            )
        self.get_logger().info(
            f'accepted {seed_id}; RGB candidates={len(replay_candidates)} '
            f'latest_selected={int(bool(replay_candidates) and not registration_fallback)} '
            f'span={replay_span_s:.3f}s; {detail}',
        )

    def _reset_cb(self, _msg: Empty) -> None:
        self._reset_session()

    def _reset_session(self, *, seed_request: _SeedRequest | None = None) -> None:
        with self._state_lock:
            self._generation += 1
            self._seed_offer = None
            self._pending_seed_request = seed_request
            self._seed_offer_armed = seed_request is not None
            if seed_request is None:
                self._seed_request_started_steady_s = None
                self._seed_request_deadline_steady_s = None
            else:
                started = self._steady_now_s()
                self._seed_request_started_steady_s = started
                self._seed_request_deadline_steady_s = started + float(
                    self.get_parameter('seed_offer_timeout_s').value,
                )
            self._accept_frames = False
            self._tracking = False
            self._active_seed_id = ''
            self._acquisition_gate.reset()
            self._replay_candidate_count = 0
            self._replay_selected_count = 0
            self._replay_span_ns = 0
            self._seed_stamp_ns = None
            self._tracking_started_ros_s = None
            self._last_result_ros_s = None
            self._commands.clear()
            self._commands.append(_Command('reset', self._generation))
            self._worker_condition.notify()
            self._publish_tracking(False)

    def _fail_if_armed(self, reason: str) -> None:
        with self._state_lock:
            armed = self._accept_frames
        if armed:
            self._fail_closed(reason)

    def _fail_closed(
        self,
        reason: str,
        *,
        generation: int | None = None,
        reason_code: str = 'tracker_failure',
        seed_id: str = '',
        seed_stamp_ns: int | None = None,
    ) -> None:
        with self._state_lock:
            if generation is not None and generation != self._generation:
                return
            failed_seed_id = seed_id or self._active_seed_id
            failed_seed_stamp_ns = (
                seed_stamp_ns
                if seed_stamp_ns is not None
                else self._seed_stamp_ns
            )
            replay_candidate_count = self._replay_candidate_count
            replay_selected_count = self._replay_selected_count
            replay_span_ns = self._replay_span_ns
            acquisition_live_updates = self._acquisition_gate.accepted_updates
            self._generation += 1
            self._seed_offer = None
            self._pending_seed_request = None
            self._seed_offer_armed = False
            self._seed_request_started_steady_s = None
            self._seed_request_deadline_steady_s = None
            self._accept_frames = False
            self._tracking = False
            self._active_seed_id = ''
            self._acquisition_gate.reset()
            self._replay_candidate_count = 0
            self._replay_selected_count = 0
            self._replay_span_ns = 0
            self._seed_stamp_ns = None
            self._tracking_started_ros_s = None
            self._last_result_ros_s = None
            self._commands.clear()
            self._commands.append(_Command('reset', self._generation))
            self._worker_condition.notify()
            self._publish_tracking(False)
        safe_reason_code = (
            str(reason_code).replace('\n', ' ').strip()[:128]
            or 'tracker_failure'
        )
        if failed_seed_id and failed_seed_stamp_ns is not None:
            self._failure_pub.publish(String(data=json.dumps(
                {
                    'schema': 'z_manip.tracker_failure.v1',
                    'seed_id': failed_seed_id,
                    'seed_stamp_ns': failed_seed_stamp_ns,
                    'reason_code': safe_reason_code,
                    'reason': reason.replace('\n', ' ').strip()[:256],
                    'replay_candidates': replay_candidate_count,
                    'replay_selected': replay_selected_count,
                    'replay_span_ns': replay_span_ns,
                    'acquisition_live_updates': acquisition_live_updates,
                },
                separators=(',', ':'),
            )))
        self.get_logger().error(f'EdgeTAM fail-closed: {reason}')

    def _watchdog_cb(self) -> None:
        now = self._now_s()
        steady_now = self._steady_now_s()
        reason: str | None = None
        reason_code = 'tracker_failure'
        generation = 0
        expired_request: _SeedRequest | None = None
        expired_offer: _SeedOffer | None = None
        with self._state_lock:
            request_started = self._seed_request_started_steady_s
            request_deadline = self._seed_request_deadline_steady_s
            if (
                self._pending_seed_request is not None
                and request_started is not None
                and request_deadline is not None
                and (
                    not math.isfinite(steady_now)
                    or steady_now < request_started
                    or steady_now > request_deadline
                )
            ):
                expired_request = self._pending_seed_request
                expired_offer = self._seed_offer
                generation = self._generation
                self._reset_session()
            elif not self._accept_frames:
                return
            else:
                generation = self._generation
            if expired_request is None:
                sync_age = (
                    None
                    if self._last_sync_ros_s is None
                    else now - self._last_sync_ros_s
                )
                if sync_age is not None and sync_age < 0.0:
                    reason = 'ROS time moved backwards during RGB-D synchronization'
                    reason_code = 'clock_rollback'
                elif (
                    sync_age is None
                    or sync_age > float(self.get_parameter('sync_timeout_s').value)
                ):
                    reason = 'exact-time RGB-D synchronization timed out'
                    reason_code = 'rgbd_sync_timeout'
                else:
                    reference = (
                        self._last_result_ros_s
                        if self._last_result_ros_s is not None
                        else self._tracking_started_ros_s
                    )
                    result_age = None if reference is None else now - reference
                    if result_age is not None and result_age < 0.0:
                        reason = 'ROS time moved backwards during EdgeTAM tracking'
                        reason_code = 'clock_rollback'
                    elif (
                        result_age is not None
                        and result_age
                        > float(self.get_parameter('result_timeout_s').value)
                    ):
                        reason = 'EdgeTAM service result timed out'
                        reason_code = 'service_result_timeout'
        if expired_request is not None:
            self._publish_seed_status(
                'expired',
                'seed request watchdog expired and released the pinned frame',
                offer=expired_offer,
                request=expired_request,
                adapter_generation=generation,
            )
            return
        if reason is not None:
            if reason_code == 'rgbd_sync_timeout':
                try:
                    self._recreate_rgbd_subscriptions()
                except Exception as error:
                    self.get_logger().error(
                        'failed to recreate RGB-D DDS readers '
                        f'({type(error).__name__})',
                    )
            self._fail_closed(
                reason,
                generation=generation,
                reason_code=reason_code,
            )

    def _worker_loop(self) -> None:
        while True:
            with self._worker_condition:
                while not self._commands and not self._stop_worker:
                    self._worker_condition.wait()
                if self._stop_worker:
                    return
                command = self._commands.popleft()
                current_generation = self._generation
            if command.generation != current_generation:
                continue
            if command.kind == 'reset':
                self._tracker.reset()
                continue
            try:
                if command.kind == 'init':
                    self._run_init(command)
                elif command.kind == 'frame':
                    self._run_update(command)
                else:
                    raise RuntimeError(f'unknown worker command {command.kind!r}')
            except Exception as error:
                detail = (
                    str(error).replace('\n', ' ').strip()[:180]
                    if isinstance(error, TrackerFailure)
                    else type(error).__name__
                )
                self._fail_closed(
                    f'worker rejected {command.kind} ({type(error).__name__}): {detail}',
                    generation=command.generation,
                    reason_code=str(
                        getattr(
                            error,
                            'reason_code',
                            f'worker_{command.kind}_failed',
                        ),
                    ),
                    seed_id=command.seed_id,
                )

    def _run_init(self, command: _Command) -> None:
        if not isinstance(command.frame, _CachedRgb) or command.bbox_xyxy is None:
            raise RuntimeError('invalid initialization command')
        self._tracker.initialize(
            stamp_ns=command.frame.stamp_ns,
            image_jpeg=command.frame.image_jpeg,
            width=command.frame.width,
            height=command.frame.height,
            bbox_xyxy=command.bbox_xyxy,
            label=command.label,
        )

    def _run_update(self, command: _Command) -> None:
        if not isinstance(command.frame, RgbdFrame):
            raise RuntimeError('invalid frame command')
        observation = self._tracker.update(command.frame)
        lag_failure: str | None = None
        token: _PublicationToken | None = None
        with self._state_lock:
            if command.generation != self._generation or not self._accept_frames:
                self._tracker.reset()
                return
            result_stamp_ns = (
                command.frame.stamp_ns
                if observation is None
                else observation.stamp_ns
            )
            lag_failure = self._result_lag_failure_locked(result_stamp_ns)
            if lag_failure is None:
                self._last_result_ros_s = self._now_s()
                if observation is None:
                    return
                newly_ready = not self._tracking and self._acquisition_gate.accept()
                ready = self._tracking or newly_ready
                if ready:
                    if self._seed_stamp_ns is None or not self._active_seed_id:
                        raise RuntimeError('cannot reserve publication without a seed epoch')
                    token = _PublicationToken(
                        generation=self._generation,
                        seed_id=self._active_seed_id,
                        seed_stamp_ns=self._seed_stamp_ns,
                        newly_ready=newly_ready,
                    )
        if lag_failure is not None:
            self._fail_closed(
                lag_failure,
                generation=command.generation,
                reason_code='result_stamp_lag',
                seed_id=command.seed_id,
            )
            return
        if token is None:
            return

        messages = self._build_observation_messages(
            observation,
            command.frame,
            token,
        )
        with self._state_lock:
            if (
                token.generation != self._generation
                or token.seed_id != self._active_seed_id
                or token.seed_stamp_ns != self._seed_stamp_ns
                or not self._accept_frames
            ):
                self._tracker.reset()
                return
            lag_failure = self._result_lag_failure_locked(observation.stamp_ns)
            if lag_failure is None:
                self._tracking = True
                self._last_result_ros_s = self._now_s()
                if token.newly_ready:
                    self.get_logger().info(
                        f'acquired {self._active_seed_id}; RGB catch-up '
                        f'candidates={self._replay_candidate_count} '
                        f'selected={self._replay_selected_count} '
                        f'span={self._replay_span_ns * 1e-9:.3f}s; '
                        'validated_live_updates='
                        f'{self._acquisition_gate.accepted_updates}',
                    )
                # Generation transitions publish false with this same lock.
                self._publish_observation(messages)
        if lag_failure is not None:
            self._fail_closed(
                lag_failure,
                generation=command.generation,
                reason_code='result_stamp_lag',
                seed_id=command.seed_id,
            )

    def _result_lag_failure_locked(self, result_stamp_ns: int) -> str | None:
        latest_stamp_ns = self._last_sync_stamp_ns
        max_lag_ns = int(
            float(self.get_parameter('max_result_stamp_lag_s').value) * 1e9,
        )
        if latest_stamp_ns is None or result_stamp_ns > latest_stamp_ns:
            return 'EdgeTAM result timestamp is outside the live RGB-D timeline'
        lag_ns = latest_stamp_ns - result_stamp_ns
        if lag_ns > max_lag_ns:
            return (
                'EdgeTAM result is stale relative to live RGB-D '
                f'({lag_ns * 1e-9:.3f}s > {max_lag_ns * 1e-9:.3f}s)'
            )
        return None

    def _build_observation_messages(
        self,
        observation: TrackingObservation,
        frame: RgbdFrame,
        token: _PublicationToken,
    ) -> _ObservationMessages:
        header = self._make_header(observation.stamp_ns, observation.frame_id)
        detection = Detection2D()
        detection.header = header
        detection.id = observation.track_id
        x1, y1, x2, y2 = observation.bbox_xyxy
        detection.bbox.center.position.x = 0.5 * (x1 + x2)
        detection.bbox.center.position.y = 0.5 * (y1 + y2)
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = float(x2 - x1)
        detection.bbox.size_y = float(y2 - y1)
        detection.results.append(self._hypothesis(observation))
        detections = Detection2DArray()
        detections.header = header
        detections.detections.append(detection)

        points = observation.points_xyz
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        target = Detection3D()
        target.header = header
        target.id = observation.track_id
        target.results.append(self._hypothesis(observation))
        target.bbox.center.position.x = float(0.5 * (lower[0] + upper[0]))
        target.bbox.center.position.y = float(0.5 * (lower[1] + upper[1]))
        target.bbox.center.position.z = float(0.5 * (lower[2] + upper[2]))
        target.bbox.center.orientation.w = 1.0
        target.bbox.size.x = float(upper[0] - lower[0])
        target.bbox.size.y = float(upper[1] - lower[1])
        target.bbox.size.z = float(upper[2] - lower[2])
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='u', offset=12, datatype=PointField.UINT32, count=1),
            PointField(name='v', offset=16, datatype=PointField.UINT32, count=1),
        ]
        cloud_rows = [
            (float(point[0]), float(point[1]), float(point[2]), int(pixel[0]), int(pixel[1]))
            for point, pixel in zip(points, observation.pixels_uv)
        ]
        cloud = point_cloud2.create_cloud(header, fields, cloud_rows)

        mask_u8 = np.asarray(observation.mask, dtype=np.uint8) * 255
        mask_msg = self._cv.cv2_to_imgmsg(mask_u8, encoding='mono8')
        mask_msg.header = header
        encoded = np.frombuffer(frame.image_jpeg, dtype=np.uint8)
        overlay = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if overlay is None:
            raise RuntimeError('could not decode synchronized JPEG for overlay')
        selected = observation.mask
        green = np.zeros_like(overlay[selected])
        green[:, 1] = 255
        overlay[selected] = cv2.addWeighted(
            overlay[selected], 0.55, green, 0.45, 0.0,
        )
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
        label = observation.label or observation.track_id
        cv2.putText(
            overlay,
            f'{label}  id={observation.track_id}  {observation.score:.2f}',
            (max(0, x1), max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        overlay_msg = self._cv.cv2_to_imgmsg(overlay, encoding='bgr8')
        overlay_msg.header = header
        scene_points = project_scene_depth(
            observation.mask,
            frame.depth_m,
            frame.intrinsics,
            target_dilation_px=int(
                self.get_parameter('target_mask_dilation_px').value,
            ),
            stride=int(self.get_parameter('scene_cloud_stride').value),
            min_depth_m=float(self.get_parameter('min_depth_m').value),
            max_depth_m=float(self.get_parameter('max_depth_m').value),
            max_points=int(self.get_parameter('scene_cloud_max_points').value),
            restore_pixels_uv=observation.rejected_pixels_uv,
        )
        scene_cloud = point_cloud2.create_cloud_xyz32(header, scene_points)
        manifest = String(data=json.dumps(
            {
                'schema': 'z_manip.tracker_frame.v1',
                'seed_id': token.seed_id,
                'seed_stamp_ns': token.seed_stamp_ns,
                'adapter_generation': token.generation,
                'result_stamp_ns': observation.stamp_ns,
                'frame_id': observation.frame_id,
                'session_id': observation.session_id,
                'track_id': observation.track_id,
                'depth_filter': frame.depth_filter,
                'raw_mask_pixels': observation.mask_diagnostics.raw_pixels,
                'cleaned_mask_pixels': observation.mask_diagnostics.cleaned_pixels,
                'raw_component_count': observation.mask_diagnostics.component_count,
                'rejected_component_pixels': (
                    observation.mask_diagnostics.rejected_pixels
                ),
                'largest_rejected_component_pixels': (
                    observation.mask_diagnostics.largest_rejected_component_pixels
                ),
                'selected_overlap_pixels': (
                    observation.mask_diagnostics.selected_overlap_pixels
                ),
                'component_selection_mode': (
                    observation.mask_diagnostics.selection_mode
                ),
                'rejected_mask_ratio': (
                    observation.mask_diagnostics.rejected_ratio
                ),
                'largest_rejected_to_selected_ratio': (
                    observation.mask_diagnostics
                    .largest_rejected_to_selected_ratio
                ),
                'max_rejected_mask_ratio': float(
                    self.get_parameter('max_rejected_mask_ratio').value,
                ),
                'max_largest_rejected_to_selected_ratio': float(
                    self.get_parameter(
                        'max_largest_rejected_to_selected_ratio',
                    ).value,
                ),
            },
            separators=(',', ':'),
        ))
        return _ObservationMessages(
            detections=detections,
            target=target,
            cloud=cloud,
            manifest=manifest,
            mask=mask_msg,
            overlay=overlay_msg,
            scene_cloud=scene_cloud,
        )

    def _publish_observation(self, messages: _ObservationMessages) -> None:
        """Commit one prebuilt generation after its final locked recheck."""
        self._detections_pub.publish(messages.detections)
        self._target_pub.publish(messages.target)
        self._frame_manifest_pub.publish(messages.manifest)
        self._cloud_pub.publish(messages.cloud)
        self._mask_pub.publish(messages.mask)
        self._overlay_pub.publish(messages.overlay)
        self._scene_cloud_pub.publish(messages.scene_cloud)
        self._publish_tracking(True)

    @staticmethod
    def _hypothesis(observation: TrackingObservation) -> ObjectHypothesisWithPose:
        result = ObjectHypothesisWithPose()
        result.hypothesis = ObjectHypothesis(
            class_id=observation.label or observation.track_id,
            score=observation.score,
        )
        return result

    def _publish_tracking(self, value: bool) -> None:
        self._tracking_pub.publish(Bool(data=value))

    @staticmethod
    def _stamp_ns(header: Header) -> int:
        sec = int(header.stamp.sec)
        nanosec = int(header.stamp.nanosec)
        if sec < 0 or not 0 <= nanosec < 1_000_000_000:
            raise ValueError('ROS timestamp is invalid')
        return sec * 1_000_000_000 + nanosec

    @staticmethod
    def _make_header(stamp_ns: int, frame_id: str) -> Header:
        header = Header()
        header.stamp.sec = stamp_ns // 1_000_000_000
        header.stamp.nanosec = stamp_ns % 1_000_000_000
        header.frame_id = frame_id
        return header

    def _now_s(self) -> float:
        """Use ROS time so stream freshness follows simulation time."""
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _steady_now_s() -> float:
        """Use a process-local clock so a seed pin expires if ROS time freezes."""
        return time.monotonic()

    def destroy_node(self) -> bool:
        """Stop the HTTP worker and clear the remote session before shutdown."""
        with self._rgbd_condition:
            self._stop_rgbd_worker = True
            self._rgbd_messages.clear()
            self._rgbd_condition.notify_all()
        if hasattr(self, '_rgbd_worker'):
            self._rgbd_worker.join(timeout=3.0)
        with self._worker_condition:
            self._stop_worker = True
            self._commands.clear()
            self._worker_condition.notify_all()
        if hasattr(self, '_worker'):
            self._worker.join(timeout=3.0)
        self._tracker.reset()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Run the EdgeTAM ROS adapter."""
    rclpy.init(args=args)
    node = EdgeTamAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
