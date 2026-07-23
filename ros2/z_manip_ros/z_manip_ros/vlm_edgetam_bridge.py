"""One-shot OpenRouter grounding followed by fail-closed EdgeTAM tracking."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
import math
import threading
import time
from typing import Any
import uuid

import cv2
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TwistStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from std_msgs.msg import Bool, Empty, String
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)

from z_manip.perception.vlm_affordance import (
    AffordanceResult,
    OpenRouterVLM,
    VLMAttemptEvent,
)
from z_manip.perception.seed_gate import (
    SeedConfidenceConfig,
    SeedDepthGateConfig,
    SeedDepthMeasurement,
    evaluate_seed_depth,
    hygiene_confidence,
    median_depth_in_bbox,
)

from .contract import (
    ContractPhase,
    ExactObservationBundler,
    expand_pixel_box,
    FailureCode,
    normalized_xyxy_to_pixel_box,
    parse_grounding_request,
    parse_tracker_failure_report,
    parse_tracker_frame_manifest,
    TrackingContract,
)


_MAX_FROZEN_COARSE_NAV_AUTHORIZATION_AGE_S = 0.35
_TRACKER_FAILURES = frozenset({
    FailureCode.TRACKER_ACQUISITION_TIMEOUT,
    FailureCode.TRACKER_REPORTED_LOSS,
    FailureCode.EMPTY_DETECTIONS,
    FailureCode.TRACK_ID_CHANGED,
    FailureCode.TARGET_MISSING,
    FailureCode.CLOUD_TOO_SMALL,
    FailureCode.TRACKER_DATA_STALE,
})
_SEED_REQUEST_SCHEMA = 'z_manip.seed_request.v1'
_SEED_OFFER_SCHEMA = 'z_manip.seed_offer.v1'
_SEED_IMAGE_FORMAT_PREFIX = 'jpeg; z_manip_seed_offer='
_MAX_SEED_JPEG_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class _SeedRequestIdentity:
    request_id: str
    producer_epoch: str
    grounding_generation: int
    nonce: str
    source_stamp_floor_ns: int


@dataclass(frozen=True)
class _SeedOfferManifest:
    request_id: str
    producer_epoch: str
    grounding_generation: int
    request_nonce: str
    adapter_generation: int
    offer_token: str
    stamp_ns: int
    frame_id: str
    width: int
    height: int


@dataclass(frozen=True)
class _GroundingSeedImage:
    header: Any
    width: int
    height: int
    jpeg: bytes
    offer_token: str
    adapter_generation: int
    request_nonce: str
    request_id: str
    producer_epoch: str
    grounding_generation: int


@dataclass(frozen=True)
class _FrozenCoarseNavAuthorization:
    """One task heartbeat bound to an immutable observed navigation goal."""

    active: bool
    request_id: str
    producer_epoch: str
    generation: int
    observation_serial: int
    nav_goal_id: str

    @property
    def identity(self) -> tuple[str, str, int, int, str]:
        """Return the complete immutable authorization identity."""
        return (
            self.request_id,
            self.producer_epoch,
            self.generation,
            self.observation_serial,
            self.nav_goal_id,
        )


def _bounded_identity(value: object, label: str) -> str:
    """Validate one printable, whitespace-free cross-node identity."""
    if not isinstance(value, str):
        raise ValueError(f'{label} must be a string')
    clean = value.strip()
    if (
        not clean
        or clean != value
        or len(clean) > 128
        or any(ord(character) < 0x21 or ord(character) > 0x7e for character in clean)
    ):
        raise ValueError(f'{label} is invalid')
    return clean


def _parse_seed_offer_manifest(payload: str) -> _SeedOfferManifest:
    """Parse one adapter offer without accepting identity aliases."""
    if not isinstance(payload, str) or not payload or len(payload) > 4096:
        raise ValueError('seed offer manifest is not a bounded string')

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('seed offer manifest has duplicate fields')
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (TypeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('seed offer manifest is not valid JSON') from error
    required = {
        'schema', 'request_id', 'producer_epoch', 'grounding_generation',
        'request_nonce', 'adapter_generation', 'offer_token', 'stamp_ns',
        'frame_id', 'width', 'height',
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get('schema') != _SEED_OFFER_SCHEMA
    ):
        raise ValueError('unsupported seed offer manifest')
    nonnegative = (value.get('grounding_generation'), value.get('stamp_ns'))
    positive = (
        value.get('adapter_generation'), value.get('width'), value.get('height'),
    )
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 <= item <= (1 << 63) - 1
        for item in nonnegative
    ) or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 < item <= (1 << 63) - 1
        for item in positive
    ):
        raise ValueError('seed offer manifest integers are invalid')
    nonce = _bounded_identity(value['request_nonce'], 'request_nonce')
    if len(nonce) != 32 or any(character not in '0123456789abcdef' for character in nonce):
        raise ValueError('request_nonce must be a lowercase UUID hex value')
    offer_token = _bounded_identity(value['offer_token'], 'offer_token')
    token_prefix = f'z-manip-seed:{value["adapter_generation"]}:'
    token_nonce = offer_token[len(token_prefix):]
    if (
        not offer_token.startswith(token_prefix)
        or len(token_nonce) != 32
        or any(character not in '0123456789abcdef' for character in token_nonce)
    ):
        raise ValueError('offer_token does not encode adapter_generation')
    return _SeedOfferManifest(
        request_id=_bounded_identity(value['request_id'], 'request_id'),
        producer_epoch=_bounded_identity(value['producer_epoch'], 'producer_epoch'),
        grounding_generation=value['grounding_generation'],
        request_nonce=nonce,
        adapter_generation=value['adapter_generation'],
        offer_token=offer_token,
        stamp_ns=value['stamp_ns'],
        frame_id=_bounded_identity(value['frame_id'], 'frame_id'),
        width=value['width'],
        height=value['height'],
    )


def _seed_image_token(image_format: object) -> str:
    """Extract the offer token carried by the compressed image itself."""
    if not isinstance(image_format, str) or not image_format.startswith(
        _SEED_IMAGE_FORMAT_PREFIX,
    ):
        raise ValueError('compressed seed image format has no offer token')
    return _bounded_identity(
        image_format[len(_SEED_IMAGE_FORMAT_PREFIX):],
        'offer_token',
    )


def _parse_frozen_coarse_nav_authorization(
    payload: str,
) -> _FrozenCoarseNavAuthorization:
    """Parse the fail-closed task-to-perception navigation heartbeat."""
    if not isinstance(payload, str) or not payload or len(payload) > 4096:
        raise ValueError('coarse-navigation authorization is not a bounded string')

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('coarse-navigation authorization has duplicate fields')
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=unique_object)
    except (TypeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('coarse-navigation authorization is not valid JSON') from error
    required = {
        'schema',
        'active',
        'request_id',
        'producer_epoch',
        'generation',
        'observation_serial',
        'nav_goal_id',
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get('schema') != 'z_manip.frozen_coarse_nav_authorization.v1'
        or not isinstance(value.get('active'), bool)
    ):
        raise ValueError('unsupported coarse-navigation authorization envelope')
    integers = (value.get('generation'), value.get('observation_serial'))
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 < item <= (1 << 63) - 1
        for item in integers
    ):
        raise ValueError('coarse-navigation authorization integers are invalid')
    return _FrozenCoarseNavAuthorization(
        active=value['active'],
        request_id=_bounded_identity(value['request_id'], 'request_id'),
        producer_epoch=_bounded_identity(value['producer_epoch'], 'producer_epoch'),
        generation=value['generation'],
        observation_serial=value['observation_serial'],
        nav_goal_id=_bounded_identity(value['nav_goal_id'], 'nav_goal_id'),
    )


class _FrozenCoarseNavAuthorizationGate:
    """Permit only a fresh, stable task heartbeat for the active VLM epoch."""

    def __init__(self, timeout_s: float) -> None:
        timeout = float(timeout_s)
        if (
            not math.isfinite(timeout)
            or not 0.0 < timeout <= _MAX_FROZEN_COARSE_NAV_AUTHORIZATION_AGE_S
        ):
            raise ValueError('coarse-navigation authorization timeout is invalid')
        self.timeout_s = timeout
        self.reset()

    def reset(self) -> None:
        """Revoke all authorization and its immutable work-goal identity."""
        self._authorization: _FrozenCoarseNavAuthorization | None = None
        self._received_monotonic_s: float | None = None

    def update(
        self,
        payload: str,
        *,
        received_monotonic_s: float,
        request_id: str,
        producer_epoch: str,
        generation: int,
    ) -> bool:
        """Accept one heartbeat, revoking the gate on every invalid transition."""
        try:
            authorization = _parse_frozen_coarse_nav_authorization(payload)
            received = float(received_monotonic_s)
            if not math.isfinite(received) or received < 0.0:
                raise ValueError('authorization receipt time is invalid')
        except (TypeError, ValueError, OverflowError):
            self.reset()
            raise
        if not authorization.active:
            self.reset()
            return False
        if (
            authorization.request_id != request_id
            or authorization.producer_epoch != producer_epoch
            or authorization.generation != generation
        ):
            self.reset()
            return False
        if (
            self._authorization is not None
            and authorization.identity != self._authorization.identity
        ):
            self.reset()
            return False
        if (
            self._received_monotonic_s is not None
            and received < self._received_monotonic_s
        ):
            self.reset()
            return False
        self._authorization = authorization
        self._received_monotonic_s = received
        return True

    def is_fresh(
        self,
        *,
        now_monotonic_s: float,
        request_id: str,
        producer_epoch: str,
        generation: int,
    ) -> bool:
        """Return whether the exact active-request heartbeat is still live."""
        authorization = self._authorization
        received = self._received_monotonic_s
        try:
            now = float(now_monotonic_s)
        except (TypeError, ValueError, OverflowError):
            self.reset()
            return False
        if authorization is None or received is None:
            return False
        if not math.isfinite(now):
            self.reset()
            return False
        if (
            authorization.request_id != request_id
            or authorization.producer_epoch != producer_epoch
            or authorization.generation != generation
        ):
            self.reset()
            return False
        age = now - received
        if not 0.0 <= age <= self.timeout_s:
            self.reset()
            return False
        return True


class VlmEdgeTamBridge(Node):
    """Expose a validated perception boundary for manipulation controllers."""

    def __init__(self) -> None:
        super().__init__('vlm_edgetam_bridge')
        self._lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='openrouter_vlm')
        self._future: Future[AffordanceResult] | None = None
        self._future_cancel_event: threading.Event | None = None
        self._future_generation = 0
        self._future_image: _GroundingSeedImage | None = None
        self._future_grounding_scope: str | None = None
        self._grounding_scope = 'grasp_only'
        self._producer_epoch = uuid.uuid4().hex

        self._declare_parameters()
        models = tuple(
            str(value)
            for value in self.get_parameter('vlm_models').value
            if str(value)
        )
        base_url = str(self.get_parameter('vlm_base_url').value).strip() or None
        model_timeouts_s = tuple(
            float(value)
            for value in self.get_parameter('vlm_model_timeouts_s').value
        )
        model_bbox_coordinate_spaces = tuple(
            str(value)
            for value in self.get_parameter(
                'vlm_model_bbox_coordinate_spaces',
            ).value
        )
        self._seed_bbox_padding_fraction = float(
            self.get_parameter('seed_bbox_padding_fraction').value,
        )
        if not 0.0 <= self._seed_bbox_padding_fraction <= 1.0:
            raise ValueError('seed_bbox_padding_fraction must be within [0, 1]')
        self._seed_depth_gate_cfg = SeedDepthGateConfig(
            enabled=bool(self.get_parameter('vlm_seed_depth_gate_enabled').value),
            distant_min_z_m=float(self.get_parameter('vlm_seed_distant_min_z_m').value),
            sanity_min_z_m=float(self.get_parameter('vlm_seed_sanity_min_z_m').value),
            sanity_max_z_m=float(self.get_parameter('vlm_seed_sanity_max_z_m').value),
            min_valid_fraction=float(
                self.get_parameter('vlm_seed_depth_min_valid_fraction').value,
            ),
        )
        self._seed_confidence_cfg = SeedConfidenceConfig(
            ceiling=float(self.get_parameter('vlm_seed_confidence_ceiling').value),
            apply_ceiling=bool(
                self.get_parameter('vlm_seed_confidence_ceiling_enabled').value,
            ),
            corroboration_enabled=bool(
                self.get_parameter('vlm_seed_local_corroboration_enabled').value,
            ),
            corroboration_floor=float(
                self.get_parameter('vlm_seed_local_corroboration_floor').value,
            ),
            corroboration_min_iou=float(
                self.get_parameter('vlm_seed_local_corroboration_min_iou').value,
            ),
        )
        self._seed_depth_scale_m = float(
            self.get_parameter('vlm_seed_depth_scale_m').value,
        )
        self._seed_depth_max_join_age_s = float(
            self.get_parameter('vlm_seed_depth_max_join_age_s').value,
        )
        # Aligned depth frames cached by stamp_ns for the seed depth gate. Only
        # populated when the gate is enabled; the seed decision joins the frame
        # whose stamp matches the admitted seed image.
        self._seed_depth_frames: OrderedDict[int, np.ndarray] = OrderedDict()
        # OpenRouterVLM reads OPENROUTER_API_KEY from the process environment.
        self._vlm = OpenRouterVLM(
            models=models or None,
            base_url=base_url,
            local_grounding_url=(
                str(self.get_parameter('vlm_local_grounding_url').value).strip()
                or None
            ),
            local_grounding_timeout_s=float(
                self.get_parameter('vlm_local_grounding_timeout_s').value,
            ),
            timeout_s=float(self.get_parameter('vlm_timeout_s').value),
            model_timeouts_s=model_timeouts_s,
            model_bbox_coordinate_spaces=model_bbox_coordinate_spaces,
            min_confidence=float(self.get_parameter('vlm_min_confidence').value),
            max_target_area_ratio=float(
                self.get_parameter('vlm_max_target_area_ratio').value,
            ),
            max_semantic_conflict_coverage_ratio=float(self.get_parameter(
                'vlm_max_semantic_conflict_coverage_ratio',
            ).value),
            provider_retries=int(
                self.get_parameter('vlm_provider_retries').value,
            ),
            timeout_retries=int(
                self.get_parameter('vlm_timeout_retries').value,
            ),
            hedge_delay_s=float(
                self.get_parameter('vlm_hedge_delay_s').value,
            ),
            attempt_callback=self._record_vlm_attempt,
        )
        self._contract = TrackingContract(
            frame_wait_timeout_s=float(self.get_parameter('frame_wait_timeout_s').value),
            grounding_timeout_s=float(self.get_parameter('grounding_timeout_s').value),
            acquisition_timeout_s=float(self.get_parameter('tracker_acquisition_timeout_s').value),
            data_timeout_s=float(self.get_parameter('tracker_data_timeout_s').value),
            min_cloud_points=int(self.get_parameter('min_cloud_points').value),
        )
        self._observation_bundler = ExactObservationBundler(
            int(self.get_parameter('bundle_cache_size').value),
        )
        self._coarse_nav_authorization = _FrozenCoarseNavAuthorizationGate(
            float(self.get_parameter(
                'frozen_coarse_nav_authorization_timeout_s',
            ).value),
        )
        reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        seed_offer_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        def topic(name: str) -> str:
            return str(self.get_parameter(name).value)

        self._bbox_pub = self.create_publisher(
            Detection2DArray, topic('edge_init_bbox_topic'), reliable,
        )
        self._seed_request_pub = self.create_publisher(
            String,
            topic('edge_seed_request_topic'),
            seed_offer_qos,
        )
        self._status_pub = self.create_publisher(DiagnosticArray, topic('status_topic'), latched)
        self._valid_pub = self.create_publisher(Bool, topic('valid_topic'), latched)
        self._affordance_pub = self.create_publisher(String, topic('affordance_topic'), latched)
        self._stop_pub = self.create_publisher(TwistStamped, topic('stop_cmd_topic'), reliable)
        self._tracked_2d_pub = self.create_publisher(
            Detection2DArray, topic('validated_detections_topic'), reliable,
        )
        self._target_3d_pub = self.create_publisher(
            Detection3D, topic('validated_target_topic'), reliable,
        )
        self._target_cloud_pub = self.create_publisher(
            PointCloud2, topic('validated_cloud_topic'), reliable,
        )

        self.create_subscription(String, topic('request_topic'), self._request_cb, reliable)
        self.create_subscription(Empty, topic('reset_topic'), self._reset_cb, reliable)
        self.create_subscription(
            String,
            topic('frozen_coarse_nav_authorization_topic'),
            self._frozen_coarse_nav_authorization_cb,
            reliable,
        )
        self.create_subscription(
            CompressedImage,
            topic('edge_seed_image_topic'),
            self._image_cb,
            seed_offer_qos,
        )
        self.create_subscription(
            String,
            topic('edge_seed_offer_manifest_topic'),
            self._seed_offer_manifest_cb,
            seed_offer_qos,
        )
        self.create_subscription(
            Bool,
            topic('edge_tracking_topic'),
            self._tracking_cb,
            latched,
        )
        self.create_subscription(
            String,
            topic('edge_failure_topic'),
            self._tracker_failure_cb,
            reliable,
        )
        self.create_subscription(
            String,
            topic('edge_frame_manifest_topic'),
            self._frame_manifest_cb,
            reliable,
        )
        self.create_subscription(
            Detection2DArray, topic('edge_detections_topic'), self._detections_cb, reliable,
        )
        self.create_subscription(
            Detection3D, topic('edge_selected_target_topic'), self._selected_target_cb, reliable,
        )
        self.create_subscription(
            PointCloud2, topic('edge_selected_cloud_topic'), self._selected_cloud_cb, reliable,
        )
        if self._seed_depth_gate_cfg.enabled:
            # Aligned depth is a best-effort sensor stream; the gate abstains if
            # no frame joins the seed, so sensor-data QoS is appropriate here.
            depth_qos = QoSProfile(depth=4, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(
                Image,
                topic('vlm_seed_depth_topic'),
                self._seed_depth_cb,
                depth_qos,
            )

        self._current_seed_request: _SeedRequestIdentity | None = None
        self._seed_images: dict[str, CompressedImage] = {}
        self._seed_offer_manifests: dict[str, _SeedOfferManifest] = {}
        self._latest_seed_image: _GroundingSeedImage | None = None
        self._latest_seed_image_at: float | None = None
        self._latest_detections: Detection2DArray | None = None
        self._latest_target: Detection3D | None = None
        self._latest_cloud: PointCloud2 | None = None
        self._bundle_serial = 0
        self._relayed_bundle_serial = 0
        self._expected_edge_seed_id = ''
        self._expected_edge_seed_stamp_ns: int | None = None
        self._expected_adapter_generation: int | None = None
        self._expected_edge_session_id = ''
        self._expected_edge_track_id = ''
        self._latest_observation_key: tuple[int, str] | None = None
        self._tracker_failure_detail = ''
        self._motion_override_active = False
        self._motion_override_at: float | None = None
        # Re-grounding is bounded per operator request.  A tracker loss always
        # revokes the old observation first; this state only schedules a fresh
        # exact-frame seed after the fail-closed transition has completed.
        self._tracker_reacquire_attempts = 0
        self._tracker_reacquire_due_monotonic_s: float | None = None
        self._tracker_reacquire_deadline_monotonic_s: float | None = None
        self._tracker_reacquire_instruction = ''
        self._tracker_reacquire_request_id = ''
        self._tracker_reacquire_state = 'idle'

        self.create_subscription(
            Bool,
            topic('motion_override_topic'),
            self._motion_override_cb,
            latched,
        )

        status_period = float(self.get_parameter('status_period_s').value)
        self._health_timer = self.create_timer(status_period, self._health_cb)
        self._publish_contract()
        self.get_logger().info(
            'ready: one-shot OpenRouter grounding -> persistent EdgeTAM tracking',
        )

    def _declare_parameters(self) -> None:
        defaults: dict[str, Any] = {
            'edge_seed_image_topic': '/track_3d/exact_seed_image',
            'edge_seed_request_topic': '/track_3d/seed_request',
            'edge_seed_offer_manifest_topic': '/track_3d/seed_offer_manifest',
            'request_topic': '/z_manip/grounding/request',
            'reset_topic': '/z_manip/grounding/reset',
            'edge_init_bbox_topic': '/track_3d/init_bbox',
            'edge_tracking_topic': '/track_3d/is_tracking',
            'edge_failure_topic': '/track_3d/failure',
            'edge_frame_manifest_topic': '/track_3d/frame_manifest',
            'edge_detections_topic': '/track_3d/detections_2d',
            'edge_selected_target_topic': '/track_3d/selected_target_3d',
            'edge_selected_cloud_topic': '/track_3d/selected_target_pointcloud',
            'status_topic': '/z_manip/perception/status',
            'valid_topic': '/z_manip/perception/valid',
            'affordance_topic': '/z_manip/perception/affordance',
            'validated_detections_topic': '/z_manip/perception/tracked_detections_2d',
            'validated_target_topic': '/z_manip/perception/target_3d',
            'validated_cloud_topic': '/z_manip/perception/target_pointcloud',
            'stop_cmd_topic': '/safety_cmd_vel',
            'stop_frame_id': 'base_link',
            'motion_override_topic': '/z_manip/visual_search/active',
            'motion_override_timeout_s': 0.25,
            'frozen_coarse_nav_authorization_topic': (
                '/z_manip/coarse_nav/perception_loss_authorization'
            ),
            'frozen_coarse_nav_authorization_timeout_s': 0.30,
            'vlm_models': [
                'qwen/qwen3-vl-8b-instruct:nitro',
            ],
            'vlm_base_url': '',
            'vlm_local_grounding_url': 'http://127.0.0.1:8771',
            'vlm_local_grounding_timeout_s': 1.25,
            'vlm_timeout_s': 15.0,
            'vlm_model_timeouts_s': [8.5],
            'vlm_model_bbox_coordinate_spaces': [
                'relative_0_1000',
            ],
            'vlm_provider_retries': 1,
            'vlm_timeout_retries': 0,
            'vlm_hedge_delay_s': 0.05,
            'vlm_min_confidence': 0.15,
            'vlm_max_target_area_ratio': 0.95,
            'vlm_max_semantic_conflict_coverage_ratio': 0.95,
            # --- seed depth gate (P2-1) -------------------------------------
            # Project the admitted seed box into the aligned depth frame and
            # reject a seed whose measured range contradicts the instruction.
            # Fails open: a missing/unjoinable depth frame ABSTAINS (admits),
            # so this can never regress the pipeline into false rejections.
            'vlm_seed_depth_gate_enabled': True,
            'vlm_seed_depth_topic': '/camera/aligned_depth_to_color/image_raw',
            # Metres per raw unit for a 16UC1 depth image (mm -> m). Ignored for
            # a 32FC1 float depth image already in metres.
            'vlm_seed_depth_scale_m': 0.001,
            'vlm_seed_distant_min_z_m': 1.2,
            'vlm_seed_sanity_min_z_m': 0.25,
            'vlm_seed_sanity_max_z_m': 4.0,
            'vlm_seed_depth_min_valid_fraction': 0.10,
            'vlm_seed_depth_max_join_age_s': 0.20,
            # --- seed confidence hygiene (P2-2) -----------------------------
            'vlm_seed_confidence_ceiling_enabled': True,
            'vlm_seed_confidence_ceiling': 0.60,
            'vlm_seed_local_corroboration_enabled': False,
            'vlm_seed_local_corroboration_floor': 0.08,
            'vlm_seed_local_corroboration_min_iou': 0.10,
            'jpeg_quality': 85,
            'seed_bbox_padding_fraction': 0.12,
            'max_camera_age_s': 0.5,
            'frame_wait_timeout_s': 2.0,
            'grounding_timeout_s': 60.0,
            'tracker_acquisition_timeout_s': 8.0,
            'tracker_data_timeout_s': 1.0,
            'tracker_auto_reacquire_enabled': True,
            'tracker_auto_reacquire_max_attempts': 2,
            'tracker_auto_reacquire_backoff_s': 0.25,
            'tracker_auto_reacquire_window_s': 8.0,
            'min_cloud_points': 24,
            'bundle_cache_size': 12,
            'status_period_s': 0.1,
            'hold_stop_until_valid': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        max_attempts = int(
            self.get_parameter('tracker_auto_reacquire_max_attempts').value,
        )
        backoff_s = float(
            self.get_parameter('tracker_auto_reacquire_backoff_s').value,
        )
        window_s = float(
            self.get_parameter('tracker_auto_reacquire_window_s').value,
        )
        if max_attempts < 0 or max_attempts > 5:
            raise ValueError('tracker auto-reacquire attempts must be within [0, 5]')
        if (
            not math.isfinite(backoff_s)
            or not math.isfinite(window_s)
            or backoff_s < 0.0
            or window_s <= 0.0
            or backoff_s >= window_s
        ):
            raise ValueError('tracker auto-reacquire timing is invalid')

    def _record_vlm_attempt(self, event: VLMAttemptEvent) -> None:
        detail = f' detail={event.detail}' if event.detail else ''
        message = (
            f'VLM attempt model={event.model} attempt={event.attempt} '
            f'outcome={event.outcome} elapsed={event.elapsed_s:.3f}s{detail}'
        )
        if event.outcome in {'start', 'success'}:
            self.get_logger().info(message)
        else:
            self.get_logger().warn(message)

    def _clear_seed_offer_join(self) -> None:
        self._seed_images.clear()
        self._seed_offer_manifests.clear()
        self._latest_seed_image = None
        self._latest_seed_image_at = None

    def _contract_allows_async_commit(self, now: float) -> bool:
        """Make a contract deadline win over a concurrently ready async result."""
        before = self._contract.phase
        self._contract.tick(now_s=now)
        if self._contract.phase is not ContractPhase.FAILED:
            return True
        if before is not ContractPhase.FAILED:
            self._handle_new_failure()
            self._publish_contract()
        return False

    def _rearm_current_seed_request(self) -> bool:
        """Replace a stale offer without changing the active task generation."""
        request = self._current_seed_request
        snapshot = self._contract.snapshot
        if (
            request is None
            or self._contract.phase is not ContractPhase.WAITING_FRAME
            or request.request_id != snapshot.request_id
            or request.producer_epoch != self._producer_epoch
            or request.grounding_generation != snapshot.generation
        ):
            return False
        old_nonce = request.nonce
        replacement = self._publish_seed_command('arm')
        if replacement is None or replacement.nonce == old_nonce:
            raise RuntimeError('seed re-arm did not create a fresh nonce')
        self.get_logger().warn(
            're-armed current grounding request after its joined seed offer expired',
        )
        return True

    def _publish_seed_command(self, action: str) -> _SeedRequestIdentity | None:
        """Publish one ordered arm/cancel command on the durable seed topic."""
        if action not in {'arm', 'cancel'}:
            raise ValueError('unsupported seed command')
        nonce = uuid.uuid4().hex
        source_stamp_floor_ns = int(self.get_clock().now().nanoseconds)
        current = self._current_seed_request
        snapshot = self._contract.snapshot
        request_id = (
            current.request_id
            if action == 'cancel' and current is not None
            else snapshot.request_id or f'cancel-{nonce}'
        )
        generation = (
            current.grounding_generation
            if action == 'cancel' and current is not None
            else int(snapshot.generation)
        )
        identity = _SeedRequestIdentity(
            request_id=_bounded_identity(request_id, 'request_id'),
            producer_epoch=self._producer_epoch,
            grounding_generation=generation,
            nonce=nonce,
            source_stamp_floor_ns=source_stamp_floor_ns,
        )
        self._clear_seed_offer_join()
        self._current_seed_request = identity if action == 'arm' else None
        self._seed_request_pub.publish(String(data=json.dumps(
            {
                'schema': _SEED_REQUEST_SCHEMA,
                'action': action,
                'request_id': identity.request_id,
                'producer_epoch': identity.producer_epoch,
                'grounding_generation': identity.grounding_generation,
                'nonce': identity.nonce,
                'source_stamp_floor_ns': identity.source_stamp_floor_ns,
            },
            separators=(',', ':'),
        )))
        return self._current_seed_request

    def _request_cb(self, msg: String) -> None:
        now = self._now_s()
        with self._lock:
            try:
                request = parse_grounding_request(
                    msg.data,
                    legacy_request_id=f'legacy-{uuid.uuid4().hex}',
                )
            except ValueError as error:
                self.get_logger().warn(f'ignored invalid grounding request: {error}')
                return
            snapshot = self._contract.snapshot
            if snapshot.request_id == request.request_id:
                if (
                    snapshot.instruction != request.instruction
                    or self._grounding_scope != request.scope
                ):
                    self.get_logger().warn(
                        'ignored reused grounding request_id with different instruction or scope',
                    )
                self._publish_contract()
                return
            self._grounding_scope = request.scope
            self._reset_tracker_reacquire()
            self._contract.request(
                request.instruction,
                now_s=now,
                request_id=request.request_id,
            )
            self._coarse_nav_authorization.reset()
            self._clear_tracker_messages()
            self._expected_edge_seed_id = ''
            self._expected_edge_seed_stamp_ns = None
            self._tracker_failure_detail = ''
            self._cancel_pending_grounding()
            self._publish_seed_command('arm')
            self._publish_zero_velocity()
            self._publish_contract()

    def _reset_cb(self, _msg: Empty) -> None:
        with self._lock:
            self._reset_tracker_reacquire()
            self._publish_seed_command('cancel')
            self._contract.reset()
            self._grounding_scope = 'grasp_only'
            self._cancel_pending_grounding()
            self._clear_tracker_messages()
            self._expected_edge_seed_id = ''
            self._expected_edge_seed_stamp_ns = None
            self._tracker_failure_detail = ''
            self._coarse_nav_authorization.reset()
            self._publish_zero_velocity()
            self._publish_contract()

    def _cancel_pending_grounding(self) -> None:
        """Signal running work and drop work that has not entered the worker."""
        cancel_event = self._future_cancel_event
        if cancel_event is not None:
            cancel_event.set()
        future = self._future
        if future is None:
            self._future_cancel_event = None
            self._future_grounding_scope = None
            return
        if future.done() or future.cancel():
            self._future = None
            self._future_cancel_event = None
            self._future_image = None
            self._future_grounding_scope = None

    def _image_cb(self, msg: CompressedImage) -> None:
        """Cache one token-bearing JPEG until its strict manifest arrives."""
        now = self._now_s()
        with self._lock:
            try:
                token = _seed_image_token(msg.format)
                self._header_key(msg.header)
                jpeg = bytes(msg.data)
                if not jpeg or len(jpeg) > _MAX_SEED_JPEG_BYTES:
                    raise ValueError('compressed seed image size is invalid')
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                self.get_logger().warn(
                    f'ignored invalid grounding image identity: {type(error).__name__}',
                )
                return
            self._seed_images[token] = msg
            while len(self._seed_images) > 4:
                self._seed_images.pop(next(iter(self._seed_images)))
            self._try_accept_seed_offer(token, now)

    def _seed_offer_manifest_cb(self, msg: String) -> None:
        """Cache one causal manifest and join it with the token-bearing JPEG."""
        now = self._now_s()
        with self._lock:
            try:
                manifest = _parse_seed_offer_manifest(msg.data)
            except ValueError as error:
                self.get_logger().warn(f'ignored invalid seed offer manifest: {error}')
                return
            request = self._current_seed_request
            if (
                request is None
                or manifest.request_id != request.request_id
                or manifest.producer_epoch != request.producer_epoch
                or manifest.grounding_generation != request.grounding_generation
                or manifest.request_nonce != request.nonce
                or manifest.stamp_ns <= request.source_stamp_floor_ns
            ):
                self.get_logger().warn('ignored stale or unmatched seed offer manifest')
                return
            self._seed_offer_manifests[manifest.offer_token] = manifest
            while len(self._seed_offer_manifests) > 4:
                self._seed_offer_manifests.pop(next(iter(self._seed_offer_manifests)))
            self._try_accept_seed_offer(manifest.offer_token, now)

    def _try_accept_seed_offer(self, token: str, now: float) -> None:
        """Admit only a byte/header/dimension-exact current transaction pair."""
        if not self._contract_allows_async_commit(now):
            return
        image = self._seed_images.get(token)
        manifest = self._seed_offer_manifests.get(token)
        request = self._current_seed_request
        if (
            image is None
            or manifest is None
            or request is None
            or self._contract.phase is not ContractPhase.WAITING_FRAME
        ):
            return
        try:
            stamp_ns, frame_id = self._header_key(image.header)
            if (
                manifest.request_id != request.request_id
                or manifest.producer_epoch != request.producer_epoch
                or manifest.grounding_generation != request.grounding_generation
                or manifest.request_nonce != request.nonce
                or manifest.offer_token != token
                or manifest.stamp_ns != stamp_ns
                or manifest.frame_id != frame_id
                or stamp_ns <= request.source_stamp_floor_ns
                or manifest.width > 16384
                or manifest.height > 16384
            ):
                raise ValueError('seed image and manifest identity differ')
            jpeg = bytes(image.data)
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if (
                decoded is None
                or decoded.ndim != 3
                or decoded.shape[1] != manifest.width
                or decoded.shape[0] != manifest.height
            ):
                raise ValueError('seed image and manifest dimensions differ')
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self._seed_images.pop(token, None)
            self._seed_offer_manifests.pop(token, None)
            self.get_logger().warn(f'ignored mismatched seed offer: {error}')
            return
        fresh_now = self._now_s()
        if not self._contract_allows_async_commit(fresh_now):
            return
        self._latest_seed_image = _GroundingSeedImage(
            header=image.header,
            width=manifest.width,
            height=manifest.height,
            jpeg=jpeg,
            offer_token=token,
            adapter_generation=manifest.adapter_generation,
            request_nonce=request.nonce,
            request_id=request.request_id,
            producer_epoch=request.producer_epoch,
            grounding_generation=request.grounding_generation,
        )
        self._latest_seed_image_at = fresh_now
        self._seed_images.clear()
        self._seed_offer_manifests.clear()
        self._maybe_start_grounding(fresh_now)

    def _maybe_start_grounding(self, now: float) -> None:
        if self._contract.phase is not ContractPhase.WAITING_FRAME:
            return
        # A single referenced future is the executor's queue bound. A completed
        # stale future is drained by _poll_grounding before replacement.
        if self._future is not None:
            return
        if self._latest_seed_image is None or self._latest_seed_image_at is None:
            return
        max_age = float(self.get_parameter('max_camera_age_s').value)
        image_age = now - self._latest_seed_image_at
        if not 0.0 <= image_age <= max_age:
            if image_age > max_age:
                self._rearm_current_seed_request()
            return
        generation = self._contract.generation
        try:
            seed_image = self._latest_seed_image
            request = self._current_seed_request
            if (
                request is None
                or seed_image.request_nonce != request.nonce
                or seed_image.request_id != request.request_id
                or seed_image.producer_epoch != request.producer_epoch
                or seed_image.grounding_generation != request.grounding_generation
                or generation != request.grounding_generation
            ):
                raise ValueError('seed offer no longer matches the active request')
            self._contract.grounding_started(generation, now_s=now)
            self._future_generation = generation
            self._future_grounding_scope = self._grounding_scope
            self._future_image = seed_image
            instruction = self._contract.instruction
            cancel_event = threading.Event()
            self._future_cancel_event = cancel_event
            self._future = self._worker.submit(
                self._vlm.locate_and_reason,
                seed_image.jpeg,
                instruction,
                grounding_scope=self._grounding_scope,
                cancel_event=cancel_event,
            )
        except Exception as error:
            self._future = None
            self._future_cancel_event = None
            self._future_image = None
            self._future_grounding_scope = None
            self._contract.grounding_failed(generation)
            self.get_logger().error(f'grounding setup failed ({type(error).__name__})')
            self._handle_new_failure()

    def _seed_depth_cb(self, msg: Image) -> None:
        """Cache one aligned depth frame (metres) by stamp for the seed gate."""
        try:
            depth_m = self._decode_depth_frame(msg, self._seed_depth_scale_m)
            stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self.get_logger().warn(
                f'ignored undecodable seed depth frame: {type(error).__name__}',
            )
            return
        with self._lock:
            self._seed_depth_frames[stamp_ns] = depth_m
            self._seed_depth_frames.move_to_end(stamp_ns)
            while len(self._seed_depth_frames) > 12:
                self._seed_depth_frames.popitem(last=False)

    @staticmethod
    def _decode_depth_frame(msg: Image, scale_m: float) -> np.ndarray:
        """Decode a raw aligned depth image into a metric float32 H x W array."""
        height, width = int(msg.height), int(msg.width)
        if height <= 0 or width <= 0:
            raise ValueError('depth frame has no extent')
        encoding = str(msg.encoding)
        buffer = bytes(msg.data)
        if encoding in ('16UC1', 'mono16'):
            frame = np.frombuffer(buffer, dtype=np.uint16)
            if frame.size < height * width:
                raise ValueError('depth buffer is shorter than its declared extent')
            metric = frame[: height * width].reshape(height, width).astype(np.float32)
            return metric * float(scale_m)
        if encoding == '32FC1':
            frame = np.frombuffer(buffer, dtype=np.float32)
            if frame.size < height * width:
                raise ValueError('depth buffer is shorter than its declared extent')
            return frame[: height * width].reshape(height, width).astype(np.float32)
        raise ValueError(f'unsupported depth encoding {encoding!r}')

    def _measure_seed_depth(
        self,
        header: Any,
        bbox_xyxy_normalized: tuple[float, float, float, float],
    ) -> SeedDepthMeasurement:
        """Median depth under the seed box from the joined aligned depth frame.

        Returns an abstaining (median None) measurement when no depth frame is
        close enough in time to the seed image, so the gate fails open.
        """
        stamp_ns = (
            int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
        )
        with self._lock:
            frames = list(self._seed_depth_frames.items())
        if not frames:
            return SeedDepthMeasurement(None, 0.0, 0)
        best_stamp, best_frame = min(
            frames, key=lambda item: abs(item[0] - stamp_ns)
        )
        if abs(best_stamp - stamp_ns) > self._seed_depth_max_join_age_s * 1e9:
            return SeedDepthMeasurement(None, 0.0, 0)
        return median_depth_in_bbox(
            best_frame,
            bbox_xyxy_normalized,
            sanity_min_z_m=self._seed_depth_gate_cfg.sanity_min_z_m,
            sanity_max_z_m=self._seed_depth_gate_cfg.sanity_max_z_m,
        )

    def _poll_grounding(self, now: float) -> None:
        future = self._future
        if future is None or not future.done():
            return
        if not self._contract_allows_async_commit(now):
            return
        generation = self._future_generation
        image_meta = self._future_image
        grounding_scope = self._future_grounding_scope
        self._future = None
        self._future_cancel_event = None
        self._future_image = None
        self._future_grounding_scope = None
        if (
            generation != self._contract.generation
            or self._contract.phase is not ContractPhase.GROUNDING
            or grounding_scope != self._grounding_scope
            or image_meta is None
            or self._current_seed_request is None
            or image_meta.request_nonce != self._current_seed_request.nonce
            or image_meta.request_id != self._current_seed_request.request_id
            or image_meta.producer_epoch != self._current_seed_request.producer_epoch
            or image_meta.grounding_generation
            != self._current_seed_request.grounding_generation
        ):
            self._maybe_start_grounding(now)
            return
        try:
            result = future.result()
            if grounding_scope is None:
                raise RuntimeError('grounding image metadata or scope is unavailable')
            header = image_meta.header
            # Confidence hygiene: a remote VLM's self-report must never be
            # trusted as high certainty downstream (P2-2). Attributes are read
            # via getattr so the harness-driven contract tests, which invoke
            # this method on a lightweight stand-in, keep the legacy behaviour.
            confidence_cfg = getattr(self, '_seed_confidence_cfg', None)
            if (
                confidence_cfg is not None
                and confidence_cfg.apply_ceiling
                and not result.model.startswith('local/')
            ):
                capped = hygiene_confidence(result.confidence, confidence_cfg)
                if capped != result.confidence:
                    result = replace(result, confidence=capped)
            # Depth gate: reject a seed whose measured range contradicts the
            # instruction (e.g. a near distractor narrated as the distant
            # target). Fails open when depth is unavailable (P2-1).
            depth_cfg = getattr(self, '_seed_depth_gate_cfg', None)
            if depth_cfg is not None and depth_cfg.enabled:
                measurement = self._measure_seed_depth(
                    header,
                    (
                        result.target_bbox.x1,
                        result.target_bbox.y1,
                        result.target_bbox.x2,
                        result.target_bbox.y2,
                    ),
                )
                depth_decision = evaluate_seed_depth(
                    measurement,
                    self._contract.instruction,
                    depth_cfg,
                )
                if not depth_decision.accepted:
                    raise ValueError(
                        f'seed depth gate rejected seed ({result.model}): '
                        f'{depth_decision.reason}'
                    )
            semantic_box = normalized_xyxy_to_pixel_box(
                (
                    result.target_bbox.x1,
                    result.target_bbox.y1,
                    result.target_bbox.x2,
                    result.target_bbox.y2,
                ),
                image_meta.width,
                image_meta.height,
            )
            box = expand_pixel_box(
                semantic_box,
                image_meta.width,
                image_meta.height,
                getattr(self, '_seed_bbox_padding_fraction', 0.0),
            )
            fresh_now = self._now_s()
            if not self._contract_allows_async_commit(fresh_now):
                return
            self._contract.grounding_succeeded(
                generation,
                target_label=result.target_label,
                confidence=result.confidence,
                now_s=fresh_now,
            )
            seed_id = image_meta.offer_token
            self._expected_edge_seed_id = seed_id
            self._expected_adapter_generation = image_meta.adapter_generation
            self._expected_edge_session_id = ''
            self._expected_edge_track_id = ''
            self._expected_edge_seed_stamp_ns = (
                int(header.stamp.sec) * 1_000_000_000
                + int(header.stamp.nanosec)
            )
            self._observation_bundler.reset(
                generation=generation,
                seed_stamp_ns=self._expected_edge_seed_stamp_ns,
            )
            self._tracker_failure_detail = ''
            self._bbox_pub.publish(self._make_seed_detection(
                header,
                result,
                box,
                seed_id=seed_id,
            ))
            self._affordance_pub.publish(String(data=self._affordance_json(
                generation,
                result,
                header,
                request_id=self._contract.request_id,
                producer_epoch=self._producer_epoch,
                grounding_scope=grounding_scope,
            )))
            self.get_logger().info(
                f'grounded generation {generation}; EdgeTAM initialization published '
                f'(semantic_box={semantic_box.size_x:.1f}x{semantic_box.size_y:.1f}, '
                f'seed_box={box.size_x:.1f}x{box.size_y:.1f})',
            )
        except Exception as error:
            self._contract.grounding_failed(generation)
            self.get_logger().error(f'grounding failed ({type(error).__name__}): {error}')
            self._handle_new_failure()

    @staticmethod
    def _make_seed_detection(
        header: Any,
        result: AffordanceResult,
        box: Any,
        *,
        seed_id: str,
    ) -> Detection2DArray:
        detection = Detection2D()
        detection.header = header
        detection.id = seed_id
        detection.bbox.center.position.x = box.center_x
        detection.bbox.center.position.y = box.center_y
        detection.bbox.center.theta = 0.0
        detection.bbox.size_x = box.size_x
        detection.bbox.size_y = box.size_y
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis = ObjectHypothesis(
            class_id=result.target_label,
            score=float(result.confidence),
        )
        detection.results.append(hypothesis)
        array = Detection2DArray()
        array.header = header
        array.detections.append(detection)
        return array

    @staticmethod
    def _affordance_json(
        generation: int,
        result: AffordanceResult,
        header: Any,
        *,
        request_id: str,
        producer_epoch: str,
        grounding_scope: str,
    ) -> str:
        def box(value: Any) -> list[float] | None:
            if value is None:
                return None
            return [value.x1, value.y1, value.x2, value.y2]

        return json.dumps(
            {
                'schema': 'z_manip.affordance.v2',
                'generation': generation,
                'request_id': request_id,
                'producer_epoch': producer_epoch,
                'grounding_scope': grounding_scope,
                'source': 'openrouter_vlm_once',
                'model': result.model,
                'source_image': {
                    'frame_id': header.frame_id,
                    'stamp_ns': (
                        int(header.stamp.sec) * 1_000_000_000
                        + int(header.stamp.nanosec)
                    ),
                },
                'target': {
                    'label': result.target_label,
                    'bbox_xyxy_normalized': box(result.target_bbox),
                    'confidence': result.confidence,
                },
                'grasp_part': None if result.grasp_part_bbox is None else {
                    'label': result.grasp_part_label,
                    'bbox_xyxy_normalized': box(result.grasp_part_bbox),
                },
                'avoid_regions': [
                    {'label': region.label, 'bbox_xyxy_normalized': box(region.bbox)}
                    for region in result.avoid_regions
                ],
                'preferred_approach_camera': result.preferred_approach_camera,
                'placement_region': (
                    None if result.placement_region_bbox is None else {
                        'label': result.placement_region_label,
                        'bbox_xyxy_normalized': box(result.placement_region_bbox),
                    }
                ),
                'placement_avoid_regions': [
                    {'label': region.label, 'bbox_xyxy_normalized': box(region.bbox)}
                    for region in result.placement_avoid_regions
                ],
                'placement_verification': (
                    None if result.placement_verification is None else {
                        'require_upright': (
                            result.placement_verification.require_upright
                        ),
                        'upright_axis': result.placement_verification.upright_axis,
                        'orientation_symmetry': (
                            result.placement_verification.orientation_symmetry
                        ),
                        'symmetry_axis': (
                            result.placement_verification.symmetry_axis
                        ),
                    }
                ),
                'constraints': result.constraints,
            },
            separators=(',', ':'),
        )

    def _tracking_cb(self, msg: Bool) -> None:
        with self._lock:
            # A Bool has no seed/stamp correlation. Current manifests are the
            # sole success authority; false remains a compatibility loss signal.
            if bool(msg.data):
                return
            before = self._contract.phase
            self._contract.tracker_status(False, now_s=self._now_s())
            self._finish_tracker_update(before)

    def _tracker_failure_cb(self, msg: String) -> None:
        try:
            report = parse_tracker_failure_report(msg.data)
        except (ValueError, OverflowError):
            self.get_logger().warn('ignored a malformed tracker failure report')
            return
        with self._lock:
            if (
                report.seed_id != self._expected_edge_seed_id
                or report.seed_stamp_ns != self._expected_edge_seed_stamp_ns
            ):
                return
            before = self._contract.phase
            self._tracker_failure_detail = (
                f'{report.reason_code}: {report.reason}; catch_up='
                f'{report.replay_selected}/{report.replay_candidates}; '
                f'span={report.replay_span_ns * 1e-9:.3f}s; '
                f'live_updates={report.acquisition_live_updates}'
            )
            self._contract.tracker_failed(now_s=self._now_s())
            self._finish_tracker_update(before)

    def _frame_manifest_cb(self, msg: String) -> None:
        try:
            manifest = parse_tracker_frame_manifest(msg.data)
        except (ValueError, OverflowError):
            self.get_logger().warn('ignored a malformed tracker frame manifest')
            return
        with self._lock:
            if (
                manifest.seed_id != self._expected_edge_seed_id
                or manifest.seed_stamp_ns != self._expected_edge_seed_stamp_ns
                or self._contract.phase
                not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING)
            ):
                return
            before = self._contract.phase
            if self._expected_adapter_generation is None:
                self._expected_adapter_generation = manifest.adapter_generation
            elif manifest.adapter_generation != self._expected_adapter_generation:
                self._tracker_failure_detail = 'frame_manifest_identity_changed'
                self._contract.tracker_failed(now_s=self._now_s())
                self._finish_tracker_update(before)
                return
            if not self._expected_edge_session_id and not self._expected_edge_track_id:
                self._expected_edge_session_id = manifest.session_id
                self._expected_edge_track_id = manifest.track_id
            elif (
                not self._expected_edge_session_id
                or not self._expected_edge_track_id
                or manifest.session_id != self._expected_edge_session_id
                or manifest.track_id != self._expected_edge_track_id
            ):
                self._tracker_failure_detail = 'frame_manifest_identity_changed'
                self._contract.tracker_failed(now_s=self._now_s())
                self._finish_tracker_update(before)
                return
            try:
                bundle = self._observation_bundler.add(
                    'manifest',
                    generation=self._contract.generation,
                    stamp_ns=manifest.result_stamp_ns,
                    frame_id=manifest.frame_id,
                    payload=manifest,
                )
            except (TypeError, ValueError, OverflowError) as error:
                self._tracker_failure_detail = (
                    f'frame_manifest_invalid: {type(error).__name__}'
                )
                self._contract.tracker_failed(now_s=self._now_s())
                self._finish_tracker_update(before)
                return
            if bundle is not None:
                self._accept_complete_bundle(bundle, before=before)

    def _motion_override_cb(self, msg: Bool) -> None:
        """Yield zero-velocity ownership only to the live task search controller."""
        with self._lock:
            self._motion_override_active = bool(msg.data)
            self._motion_override_at = self._now_s()

    def _frozen_coarse_nav_authorization_cb(self, msg: String) -> None:
        """Accept only the task's live immutable work-goal heartbeat."""
        with self._lock:
            snapshot = self._contract.snapshot
            try:
                self._coarse_nav_authorization.update(
                    msg.data,
                    received_monotonic_s=self._monotonic_s(),
                    request_id=snapshot.request_id,
                    producer_epoch=self._producer_epoch,
                    generation=snapshot.generation,
                )
            except (TypeError, ValueError, OverflowError) as error:
                self.get_logger().warn(
                    'revoked malformed coarse-navigation authorization: '
                    f'{type(error).__name__}',
                )

    def _detections_cb(self, msg: Detection2DArray) -> None:
        with self._lock:
            self._buffer_observation('detections', msg)

    def _selected_target_cb(self, msg: Detection3D) -> None:
        with self._lock:
            self._buffer_observation('target', msg)

    def _selected_cloud_cb(self, msg: PointCloud2) -> None:
        with self._lock:
            self._buffer_observation('cloud', msg)

    def _buffer_observation(self, kind: str, msg: Any) -> None:
        if (
            self._expected_edge_seed_stamp_ns is None
            or self._contract.phase
            not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING)
        ):
            return
        before = self._contract.phase
        try:
            stamp_ns, frame_id = self._header_key(msg.header)
            if kind == 'detections':
                for detection in msg.detections:
                    if self._header_key(detection.header) != (stamp_ns, frame_id):
                        raise ValueError('detection header differs from its array')
            bundle = self._observation_bundler.add(
                kind,
                generation=self._contract.generation,
                stamp_ns=stamp_ns,
                frame_id=frame_id,
                payload=msg,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self._tracker_failure_detail = (
                f'observation_bundle_invalid: {type(error).__name__}'
            )
            self._contract.tracker_failed(now_s=self._now_s())
            self._finish_tracker_update(before)
            return
        if bundle is None:
            return

        self._accept_complete_bundle(bundle, before=before)

    def _accept_complete_bundle(self, bundle: Any, *, before: ContractPhase) -> None:
        detections = bundle.detections
        target = bundle.target
        cloud = bundle.cloud
        manifest = bundle.manifest
        detection_ids = tuple(
            str(item.id).strip()
            for item in detections.detections
            if str(item.id).strip()
        )
        if (
            len(detection_ids) != 1
            or detection_ids[0] != str(target.id).strip()
            or detection_ids[0] != manifest.track_id
        ):
            self._tracker_failure_detail = 'frame_manifest_track_mismatch'
            self._contract.tracker_failed(now_s=self._now_s())
            self._finish_tracker_update(before)
            return
        now = self._now_s()
        try:
            self._contract.observation_bundle(
                detection_ids,
                selected_track_id=target.id,
                point_count=int(cloud.width) * int(cloud.height),
                now_s=now,
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            self._tracker_failure_detail = (
                f'observation_bundle_invalid: {type(error).__name__}'
            )
            self._contract.tracker_failed(now_s=self._now_s())
        if self._contract.phase is not ContractPhase.FAILED:
            self._contract.tracker_status(True, now_s=now)
        if self._contract.phase is not ContractPhase.FAILED:
            self._latest_detections = detections
            self._latest_target = target
            self._latest_cloud = cloud
            self._latest_observation_key = (int(bundle.stamp_ns), str(bundle.frame_id))
            self._bundle_serial += 1
        self._finish_tracker_update(before)

    @staticmethod
    def _header_key(header: Any) -> tuple[int, str]:
        sec = int(header.stamp.sec)
        nanosec = int(header.stamp.nanosec)
        frame_id = str(header.frame_id).strip()
        if (
            sec < 0
            or not 0 <= nanosec < 1_000_000_000
            or not frame_id
            or len(frame_id) > 256
        ):
            raise ValueError('tracker observation header is invalid')
        stamp_ns = sec * 1_000_000_000 + nanosec
        if stamp_ns > (1 << 63) - 1:
            raise OverflowError('tracker observation timestamp exceeds int64')
        return stamp_ns, frame_id

    def _finish_tracker_update(self, before: ContractPhase) -> None:
        if self._contract.phase is ContractPhase.TRACKING:
            self._tracker_failure_detail = ''
            if self._tracker_reacquire_attempts:
                self._tracker_reacquire_state = 'recovered'
            self._tracker_reacquire_due_monotonic_s = None
            self._tracker_reacquire_deadline_monotonic_s = None
        if self._contract.phase is ContractPhase.FAILED and before is not ContractPhase.FAILED:
            self._handle_new_failure()
        self._relay_if_valid()
        self._publish_contract()

    def _health_cb(self) -> None:
        with self._lock:
            now = self._now_s()
            before = self._contract.phase
            self._contract.tick(now_s=now)
            if self._contract.phase is ContractPhase.FAILED and before is not ContractPhase.FAILED:
                self._handle_new_failure()
            if self._contract.phase is ContractPhase.FAILED:
                self._maybe_start_tracker_reacquire(now)
            if self._contract.phase is not ContractPhase.FAILED:
                self._poll_grounding(now)
            if (
                bool(self.get_parameter('hold_stop_until_valid').value)
                and self._contract.phase not in (ContractPhase.IDLE, ContractPhase.TRACKING)
                and not self._motion_override_is_fresh(now)
                and not self._frozen_coarse_nav_authorization_releases_hold()
            ):
                self._publish_zero_velocity()
            self._relay_if_valid()
            self._publish_contract()

    def _motion_override_is_fresh(self, now: float) -> bool:
        if not self._motion_override_active or self._motion_override_at is None:
            return False
        timeout = float(self.get_parameter('motion_override_timeout_s').value)
        age = now - self._motion_override_at
        return timeout > 0.0 and 0.0 <= age <= timeout

    def _frozen_coarse_nav_authorization_releases_hold(self) -> bool:
        """Release only repeated bridge zeros for an authorized tracker loss."""
        snapshot = self._contract.snapshot
        if (
            snapshot.phase is not ContractPhase.FAILED
            or snapshot.failure not in _TRACKER_FAILURES
        ):
            return False
        return self._coarse_nav_authorization.is_fresh(
            now_monotonic_s=self._monotonic_s(),
            request_id=snapshot.request_id,
            producer_epoch=self._producer_epoch,
            generation=snapshot.generation,
        )

    def _handle_new_failure(self) -> None:
        snapshot = self._contract.snapshot
        self._cancel_pending_grounding()
        self._clear_tracker_messages()
        self._publish_seed_command('cancel')
        self._publish_zero_velocity()
        self._schedule_tracker_reacquire(snapshot)
        suffix = f' ({self._tracker_failure_detail})' if self._tracker_failure_detail else ''
        self.get_logger().error(
            f'perception contract failed: {self._contract.failure.value}{suffix}',
        )

    def _reset_tracker_reacquire(self) -> None:
        """Forget retry authority at an explicit operator task boundary."""
        self._tracker_reacquire_attempts = 0
        self._tracker_reacquire_due_monotonic_s = None
        self._tracker_reacquire_deadline_monotonic_s = None
        self._tracker_reacquire_instruction = ''
        self._tracker_reacquire_request_id = ''
        self._tracker_reacquire_state = 'idle'

    def _schedule_tracker_reacquire(self, snapshot: Any) -> bool:
        """Schedule one bounded fresh-frame grounding after tracker loss."""
        if not bool(
            self.get_parameter('tracker_auto_reacquire_enabled').value,
        ):
            self._tracker_reacquire_state = 'disabled'
            return False
        if snapshot.failure not in _TRACKER_FAILURES:
            self._tracker_reacquire_state = 'not_tracker_loss'
            return False
        instruction = str(snapshot.instruction).strip()
        request_id = str(snapshot.request_id).strip()
        if not instruction or not request_id:
            self._tracker_reacquire_state = 'unavailable'
            return False
        limit = int(
            self.get_parameter('tracker_auto_reacquire_max_attempts').value,
        )
        if self._tracker_reacquire_attempts >= limit:
            self._tracker_reacquire_state = 'exhausted'
            self._tracker_failure_detail = (
                f'tracker_lost; reacquire_exhausted '
                f'{self._tracker_reacquire_attempts}/{limit}'
            )
            return False
        now = self._monotonic_s()
        backoff_s = float(
            self.get_parameter('tracker_auto_reacquire_backoff_s').value,
        )
        window_s = float(
            self.get_parameter('tracker_auto_reacquire_window_s').value,
        )
        self._tracker_reacquire_instruction = instruction
        self._tracker_reacquire_request_id = request_id
        self._tracker_reacquire_due_monotonic_s = now + backoff_s
        self._tracker_reacquire_deadline_monotonic_s = now + window_s
        self._tracker_reacquire_state = 'scheduled'
        self._tracker_failure_detail = (
            f'tracker_lost; reacquire_scheduled '
            f'{self._tracker_reacquire_attempts + 1}/{limit}'
        )
        return True

    def _maybe_start_tracker_reacquire(self, now_ros_s: float) -> bool:
        """Start a scheduled retry without ever reviving old target geometry."""
        due = self._tracker_reacquire_due_monotonic_s
        deadline = self._tracker_reacquire_deadline_monotonic_s
        if due is None or deadline is None:
            return False
        now = self._monotonic_s()
        if not math.isfinite(now) or now > deadline:
            self._tracker_reacquire_due_monotonic_s = None
            self._tracker_reacquire_deadline_monotonic_s = None
            self._tracker_reacquire_state = 'expired'
            self._tracker_failure_detail = 'tracker_lost; reacquire_expired'
            return False
        if now < due:
            return False
        instruction = self._tracker_reacquire_instruction
        request_id = self._tracker_reacquire_request_id
        self._tracker_reacquire_due_monotonic_s = None
        self._tracker_reacquire_deadline_monotonic_s = None
        self._tracker_reacquire_attempts += 1
        self._tracker_reacquire_state = 'grounding'
        self._contract.request(
            instruction,
            now_s=now_ros_s,
            request_id=request_id,
        )
        self._coarse_nav_authorization.reset()
        self._clear_tracker_messages()
        self._expected_edge_seed_id = ''
        self._expected_edge_seed_stamp_ns = None
        self._cancel_pending_grounding()
        self._publish_seed_command('arm')
        self._publish_zero_velocity()
        self.get_logger().warn(
            'tracker lost; started bounded fresh-frame perception reacquire '
            f'{self._tracker_reacquire_attempts}/'
            f'{int(self.get_parameter("tracker_auto_reacquire_max_attempts").value)}',
        )
        self._publish_contract()
        return True

    def _relay_if_valid(self) -> None:
        observation_stamp_ns, observation_frame_id = self._verified_observation_key()
        if (
            not self._contract.snapshot.valid
            or not observation_stamp_ns
            or not observation_frame_id
            or self._bundle_serial <= self._relayed_bundle_serial
            or self._latest_detections is None
            or self._latest_target is None
            or self._latest_cloud is None
        ):
            return
        self._tracked_2d_pub.publish(self._latest_detections)
        self._target_3d_pub.publish(self._latest_target)
        self._target_cloud_pub.publish(self._latest_cloud)
        self._relayed_bundle_serial = self._bundle_serial

    def _publish_contract(self) -> None:
        observation_stamp_ns, observation_frame_id = self._verified_observation_key()
        snapshot = self._contract.snapshot
        public_valid = bool(
            snapshot.valid and observation_stamp_ns and observation_frame_id
        )
        self._valid_pub.publish(Bool(data=public_valid))
        status = DiagnosticStatus()
        status.name = 'z_manip/perception_contract'
        status.hardware_id = 'wrist_rgbd'
        status.level = (
            DiagnosticStatus.OK if public_valid
            else DiagnosticStatus.ERROR if snapshot.phase is ContractPhase.FAILED
            else DiagnosticStatus.WARN
        )
        status.message = snapshot.phase.value
        status.values = [
            KeyValue(key='schema', value='z_manip.perception_status.v1'),
            KeyValue(
                key='source_chain',
                value='openrouter_vlm_once>edgetam_persistent_mask_depth',
            ),
            KeyValue(key='generation', value=str(snapshot.generation)),
            KeyValue(key='request_id', value=snapshot.request_id),
            KeyValue(
                key='instruction_sha256',
                value=hashlib.sha256(snapshot.instruction.encode('utf-8')).hexdigest(),
            ),
            KeyValue(key='grounding_scope', value=self._grounding_scope),
            KeyValue(key='producer_epoch', value=self._producer_epoch),
            KeyValue(key='valid', value=str(public_valid).lower()),
            KeyValue(key='target_label', value=snapshot.target_label),
            KeyValue(key='track_id', value=snapshot.track_id),
            KeyValue(key='failure', value=snapshot.failure.value),
            KeyValue(key='failure_detail', value=self._tracker_failure_detail),
            KeyValue(key='reacquire_state', value=self._tracker_reacquire_state),
            KeyValue(
                key='reacquire_attempts',
                value=str(self._tracker_reacquire_attempts),
            ),
            KeyValue(key='observation_stamp_ns', value=observation_stamp_ns),
            KeyValue(key='observation_frame_id', value=observation_frame_id),
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self._status_pub.publish(message)

    def _verified_observation_key(self) -> tuple[str, str]:
        """Fail the tracker if an internally-valid bundle loses exact identity."""
        snapshot = self._contract.snapshot
        key = self._status_observation_key(snapshot.valid)
        if not snapshot.valid or all(key):
            return key
        self._tracker_failure_detail = 'validated_observation_identity_mismatch'
        self._contract.tracker_failed(now_s=self._now_s())
        self._handle_new_failure()
        return '', ''

    def _status_observation_key(self, valid: bool) -> tuple[str, str]:
        """Return the exact validated bundle identity, or empty fail-closed fields."""
        if (
            not valid
            or self._latest_observation_key is None
            or self._latest_detections is None
            or self._latest_target is None
            or self._latest_cloud is None
        ):
            return '', ''
        try:
            keys = (
                self._header_key(self._latest_detections.header),
                self._header_key(self._latest_target.header),
                self._header_key(self._latest_cloud.header),
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return '', ''
        if not all(key == self._latest_observation_key for key in keys):
            return '', ''
        return str(self._latest_observation_key[0]), self._latest_observation_key[1]

    def _publish_zero_velocity(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter('stop_frame_id').value)
        self._stop_pub.publish(message)

    def _now_s(self) -> float:
        """Use ROS time so freshness gates scale with simulation RTF."""
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _monotonic_s() -> float:
        """Use a process-local clock for cross-node heartbeat receipt age."""
        return time.monotonic()

    def _clear_tracker_messages(self) -> None:
        self._latest_detections = None
        self._latest_target = None
        self._latest_cloud = None
        self._bundle_serial = 0
        self._relayed_bundle_serial = 0
        self._expected_adapter_generation = None
        self._expected_edge_session_id = ''
        self._expected_edge_track_id = ''
        self._latest_observation_key = None
        self._observation_bundler.reset(
            generation=self._contract.generation,
            seed_stamp_ns=None,
        )

    def destroy_node(self) -> bool:
        with self._lock:
            self._cancel_pending_grounding()
            self._publish_seed_command('cancel')
        self._worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VlmEdgeTamBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            # ros2 launch and the terminal can deliver SIGINT to the child in
            # quick succession; cleanup remains best-effort and idempotent.
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
