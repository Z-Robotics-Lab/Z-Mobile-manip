"""ROS-independent, fail-closed EdgeTAM RGB-D tracking core."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Protocol, Sequence
import uuid
import warnings

import cv2
import numpy as np


class TrackerFailure(RuntimeError):
    """Tracking data is unsafe to publish to a controller."""

    def __init__(self, message: str, *, reason_code: str = 'tracker_failure') -> None:
        super().__init__(message)
        self.reason_code = reason_code


class TrackResult(Protocol):
    """Validated result returned by the platform-neutral service client."""

    session_id: str
    track_id: str
    frame_seq: int
    image_size: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    mask: np.ndarray


class ServiceClient(Protocol):
    """Injectable subset of ``EdgeTamServiceClient`` used by this package."""

    @property
    def active(self) -> bool:
        """Return whether the client owns a live service session."""

    def init(
        self,
        image_jpeg: bytes,
        bbox_xyxy: Sequence[int],
        *,
        session_id: str,
        frame_seq: int = 0,
    ) -> TrackResult:
        """Initialize one persistent identity from an image-space box."""

    def update(self, image_jpeg: bytes, *, frame_seq: int | None = None) -> TrackResult:
        """Advance the active identity by one strictly ordered frame."""

    def reset(self) -> None:
        """Discard the active identity."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole calibration for an aligned color/depth image."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('camera intrinsics must be finite')
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError('camera focal lengths must be positive')


@dataclass(frozen=True, eq=False)
class RgbdFrame:
    """One exact-time aligned RGB-D observation in the optical frame."""

    stamp_ns: int
    frame_id: str
    image_jpeg: bytes
    width: int
    height: int
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    depth_filter: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stamp_ns, bool) or self.stamp_ns < 0:
            raise ValueError('stamp_ns must be a non-negative integer')
        if not self.frame_id.strip():
            raise ValueError('frame_id must not be empty')
        if self.width < 1 or self.height < 1:
            raise ValueError('image dimensions must be positive')
        if not isinstance(self.image_jpeg, bytes):
            raise ValueError('image_jpeg must be bytes')
        if not self.image_jpeg.startswith(b'\xff\xd8'):
            raise ValueError('image_jpeg must start with a JPEG marker')
        if not self.image_jpeg.endswith(b'\xff\xd9'):
            raise ValueError('image_jpeg must end with a JPEG marker')
        depth = np.asarray(self.depth_m)
        if depth.shape != (self.height, self.width):
            raise ValueError('depth shape must exactly match the color image')
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError('depth_m must use floating-point metres')


class MotionAdaptiveDepthFilter:
    """
    Temporal D435 depth filter shared by target and scene projection.

    Static pixels use a short robust median/MAD window. Coherent local motion
    bypasses the median immediately, and broad camera motion resets the window,
    preventing trails from an eye-in-hand camera. Isolated large depth spikes
    do not qualify as motion and are rejected by the temporal stability gate.
    This class only transforms measured depth; it has no planning or transport
    dependency.

    The per-frame cost is dominated by the windowed ``np.nanmedian`` over the
    temporal stack.  ``half_resolution`` (default on) computes that median and
    MAD on a 2x-decimated stack and restores them with nearest-neighbour
    upsampling, which measured 4-5x faster at 640x480x5 (~60-75ms -> ~15ms).
    The temporal median per sampled pixel stays exact; the only approximation
    is that a pixel borrows its 2x2 block leader's stabilized value, so on the
    spatially smooth depth this filter targets the MAD-class stability metrics
    are expected to be virtually unchanged.  Per-pixel validity counts and the
    stability gate remain full resolution, so invalid pixels are never revived
    by an upsampled neighbour.  Camera-motion detection also compares on the
    decimated grid so the tiled output is only ever tested against the exact
    pixels it stabilizes; a full-resolution comparison would misread the
    intra-block spatial gradient as scene motion.  Set
    ``half_resolution=False`` to force the exact full-resolution median.
    """

    def __init__(
        self,
        *,
        window_size: int = 5,
        min_valid_fraction: float = 0.6,
        max_mad_m: float = 0.006,
        motion_threshold_m: float = 0.012,
        global_motion_fraction: float = 0.15,
        min_motion_pixels: int = 24,
        max_gap_s: float = 0.5,
        half_resolution: bool = True,
    ) -> None:
        if isinstance(window_size, bool) or window_size < 3:
            raise ValueError('depth-filter window must contain at least three frames')
        if not 0.0 < min_valid_fraction <= 1.0:
            raise ValueError('depth-filter valid fraction must be in (0, 1]')
        numeric = (max_mad_m, motion_threshold_m, global_motion_fraction, max_gap_s)
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError('depth-filter thresholds must be positive and finite')
        if global_motion_fraction > 1.0:
            raise ValueError('global motion fraction cannot exceed one')
        if isinstance(min_motion_pixels, bool) or min_motion_pixels < 1:
            raise ValueError('minimum coherent motion area must be positive')
        if not isinstance(half_resolution, bool):
            raise ValueError('half_resolution must be boolean')
        self._half_resolution = bool(half_resolution)
        self.window_size = int(window_size)
        self.min_valid_fraction = float(min_valid_fraction)
        self.max_mad_m = float(max_mad_m)
        self.motion_threshold_m = float(motion_threshold_m)
        self.global_motion_fraction = float(global_motion_fraction)
        self.min_motion_pixels = int(min_motion_pixels)
        self.max_gap_ns = round(float(max_gap_s) * 1_000_000_000)
        self._frames: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._previous_output: np.ndarray | None = None
        self._motion_hold: np.ndarray | None = None
        self._last_stamp_ns: int | None = None

    def reset(self) -> None:
        self._frames.clear()
        self._previous_output = None
        self._motion_hold = None
        self._last_stamp_ns = None

    def _coherent_motion(self, changed: np.ndarray) -> np.ndarray:
        if not np.any(changed):
            return np.zeros(changed.shape, dtype=bool)
        mask = np.asarray(changed, dtype=np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            opened,
            connectivity=8,
        )
        coherent = np.zeros(changed.shape, dtype=bool)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= self.min_motion_pixels:
                coherent |= labels == label
        return coherent

    def _windowed_median_and_mad(
        self,
        stack: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the per-pixel temporal median and MAD over the frame stack.

        NumPy emits RuntimeWarning for pixels that are invalid throughout the
        window. Those pixels are intentionally rejected by the support mask in
        ``update``, so keep normal runtime logs quiet. With ``half_resolution``
        the median/MAD are computed on a 2x-decimated stack and restored by
        nearest-neighbour upsampling; the sampled-pixel temporal median stays
        exact while its 2x2 block borrows that value (see the class docstring).
        """
        _frames, height, width = stack.shape
        use_half = self._half_resolution and height >= 4 and width >= 4
        source = stack[:, ::2, ::2] if use_half else stack
        with warnings.catch_warnings(), np.errstate(all='ignore'):
            warnings.simplefilter('ignore', category=RuntimeWarning)
            median = np.nanmedian(source, axis=0)
            mad = np.nanmedian(np.abs(source - median[None, ...]), axis=0)
        if not use_half:
            return median, mad
        return (
            self._upsample_nearest(median, height, width),
            self._upsample_nearest(mad, height, width),
        )

    @staticmethod
    def _upsample_nearest(
        small: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Restore a 2x-decimated map to full size by nearest-neighbour tiling."""
        expanded = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        return np.asarray(expanded[:height, :width], dtype=np.float32)

    def update(
        self,
        depth_m: object,
        *,
        stamp_ns: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            raise ValueError('depth-filter stamp must be a non-negative integer')
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2 or not depth.size:
            raise ValueError('depth filter expects a non-empty 2-D image')
        reset_reason: str | None = None
        if self._frames and self._frames[-1].shape != depth.shape:
            reset_reason = 'shape_changed'
        elif self._last_stamp_ns is not None and stamp_ns <= self._last_stamp_ns:
            reset_reason = 'stamp_not_increasing'
        elif (
            self._last_stamp_ns is not None
            and stamp_ns - self._last_stamp_ns > self.max_gap_ns
        ):
            reset_reason = 'input_gap'
        if reset_reason is not None:
            self.reset()
        self._last_stamp_ns = stamp_ns

        height, width = depth.shape
        use_half = self._half_resolution and height >= 4 and width >= 4
        current_valid = np.isfinite(depth) & (depth > 0.0)
        current = np.where(current_valid, depth, np.nan).astype(np.float32)
        changed = np.zeros(depth.shape, dtype=bool)
        overlap_count = 0
        changed_fraction = 0.0
        if self._previous_output is not None:
            # With half_resolution the stabilized output tiles each decimated
            # value over its 2x2 block, so a full-resolution comparison would
            # read the intra-block spatial gradient (one pixel of surface
            # slope, easily beyond motion_threshold_m on oblique floors) as
            # persistent motion on every frame.  Comparing on the decimated
            # grid keeps the test pixel-aligned: element [2i, 2j] of the tiled
            # output is exactly the stabilized value of input pixel [2i, 2j].
            if use_half:
                depth_cmp = depth[::2, ::2]
                previous_cmp = self._previous_output[::2, ::2]
                valid_cmp = current_valid[::2, ::2]
            else:
                depth_cmp = depth
                previous_cmp = self._previous_output
                valid_cmp = current_valid
            previous_valid = np.isfinite(previous_cmp) & (previous_cmp > 0.0)
            overlap = valid_cmp & previous_valid
            overlap_count = int(np.count_nonzero(overlap))
            changed_cmp = np.zeros(depth_cmp.shape, dtype=bool)
            changed_cmp[overlap] = (
                np.abs(depth_cmp[overlap] - previous_cmp[overlap])
                > self.motion_threshold_m
            )
            if overlap_count:
                changed_fraction = (
                    float(np.count_nonzero(changed_cmp)) / float(overlap_count)
                )
            if use_half:
                # Scale the sampled overlap back to a full-resolution pixel
                # count so min_motion_pixels keeps its calibrated meaning, and
                # tile the changed mask so motion-hold bookkeeping stays at
                # full resolution.
                overlap_count *= 4
                changed = np.repeat(
                    np.repeat(changed_cmp, 2, axis=0),
                    2,
                    axis=1,
                )[:height, :width]
            else:
                changed = changed_cmp
        global_motion = bool(
            overlap_count >= self.min_motion_pixels
            and changed_fraction >= self.global_motion_fraction
        )
        coherent_motion = self._coherent_motion(changed) if not global_motion else changed

        if global_motion:
            self._frames.clear()
            self._frames.append(current)
            self._motion_hold = np.zeros(depth.shape, dtype=np.uint8)
            output = np.where(current_valid, depth, 0.0).astype(np.float32)
            mode = 'camera_motion_reset'
            stable = current_valid
            counts = current_valid.astype(np.int16)
            mad = np.zeros(depth.shape, dtype=np.float32)
        else:
            self._frames.append(current)
            stack = np.stack(tuple(self._frames), axis=0)
            observed = np.isfinite(stack) & (stack > 0.0)
            counts = np.count_nonzero(observed, axis=0)
            median, mad = self._windowed_median_and_mad(stack)
            minimum = max(2, int(math.ceil(len(self._frames) * self.min_valid_fraction)))
            stable = (
                (counts >= minimum)
                & np.isfinite(median)
                & np.isfinite(mad)
                & (mad <= self.max_mad_m)
            )
            if len(self._frames) < 3:
                output = np.where(current_valid, depth, 0.0).astype(np.float32)
                mode = 'warmup'
            else:
                output = np.zeros(depth.shape, dtype=np.float32)
                output[stable] = median[stable].astype(np.float32)
                if self._motion_hold is None or self._motion_hold.shape != depth.shape:
                    self._motion_hold = np.zeros(depth.shape, dtype=np.uint8)
                self._motion_hold[self._motion_hold > 0] -= 1
                self._motion_hold[coherent_motion] = self.window_size
                dynamic = (self._motion_hold > 0) & current_valid
                output[dynamic] = depth[dynamic]
                mode = 'local_motion' if np.any(dynamic) else 'static_temporal'

        output.setflags(write=False)
        self._previous_output = output
        finite_mad = mad[np.isfinite(mad)]
        minimum_observations = max(
            1,
            int(math.ceil(len(self._frames) * self.min_valid_fraction)),
        )
        report: dict[str, object] = {
            'method': 'motion_adaptive_temporal_median',
            'frame_count': len(self._frames),
            'window_size': self.window_size,
            'minimum_observations': minimum_observations,
            'mode': mode,
            'reset_reason': reset_reason,
            'motion_threshold_mm': self.motion_threshold_m * 1000.0,
            'global_changed_fraction': changed_fraction,
            'dynamic_pixels': int(np.count_nonzero(coherent_motion)),
            'stable_pixels': int(np.count_nonzero(stable)),
            'rejected_low_support_pixels': int(
                np.count_nonzero(counts < minimum_observations),
            ),
            'rejected_unstable_pixels': int(
                np.count_nonzero(
                    (counts >= minimum_observations)
                    & np.isfinite(mad)
                    & (mad > self.max_mad_m)
                ),
            ),
            'mad_p95_mm': (
                0.0
                if not finite_mad.size
                else float(np.percentile(finite_mad, 95) * 1000.0)
            ),
            'applied_to': ['target_pointcloud', 'scene_pointcloud'],
        }
        return output, report


@dataclass(frozen=True, eq=False)
class TrackingObservation:
    """Persistent 2-D identity and its current observed 3-D target points."""

    stamp_ns: int
    frame_id: str
    session_id: str
    track_id: str
    label: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    points_xyz: np.ndarray
    pixels_uv: np.ndarray
    rejected_pixels_uv: np.ndarray
    mask_diagnostics: MaskCleanupDiagnostics


@dataclass(frozen=True)
class MaskCleanupDiagnostics:
    """Bounded diagnostics retained while only the identity component is used."""

    raw_pixels: int
    cleaned_pixels: int
    component_count: int
    rejected_pixels: int
    largest_rejected_component_pixels: int
    selected_overlap_pixels: int
    selection_mode: str

    @property
    def rejected_ratio(self) -> float:
        """Return the fraction of raw pixels discarded by component cleanup."""
        return float(self.rejected_pixels) / float(self.raw_pixels)

    @property
    def largest_rejected_to_selected_ratio(self) -> float:
        """Return the largest competing component relative to the selected one."""
        return float(self.largest_rejected_component_pixels) / float(
            self.cleaned_pixels,
        )


@dataclass(frozen=True)
class ReseedRegistrationConfig:
    """Fail-closed limits for transferring one seed box to a latest RGB frame."""

    roi_expansion_ratio: float = 0.75
    max_features: int = 500
    feature_quality: float = 0.01
    feature_min_distance_px: float = 4.0
    lk_window_px: int = 21
    lk_max_level: int = 3
    max_forward_backward_error_px: float = 1.5
    ransac_reproj_threshold_px: float = 2.0
    min_global_tracks: int = 20
    min_roi_tracks: int = 6
    min_global_inliers: int = 15
    min_roi_inliers: int = 5
    min_inlier_ratio: float = 0.60
    max_reprojection_rms_px: float = 1.75
    max_rotation_rad: float = 0.20
    max_scale_deviation: float = 0.12
    max_translation_ratio: float = 0.20
    max_global_roi_center_delta_ratio: float = 0.025
    max_global_roi_rotation_delta_rad: float = 0.04
    max_global_roi_scale_delta: float = 0.05
    min_bbox_retained_ratio: float = 0.80

    def __post_init__(self) -> None:
        integers = {
            'max_features': self.max_features,
            'lk_window_px': self.lk_window_px,
            'lk_max_level': self.lk_max_level,
            'min_global_tracks': self.min_global_tracks,
            'min_roi_tracks': self.min_roi_tracks,
            'min_global_inliers': self.min_global_inliers,
            'min_roi_inliers': self.min_roi_inliers,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in integers.values()
        ):
            raise ValueError('reseed registration integer limits must be positive')
        if self.lk_window_px % 2 == 0:
            raise ValueError('reseed LK window must be odd')
        if self.lk_max_level > 8:
            raise ValueError('reseed LK pyramid level is unreasonably large')
        positive = (
            self.roi_expansion_ratio,
            self.feature_quality,
            self.feature_min_distance_px,
            self.max_forward_backward_error_px,
            self.ransac_reproj_threshold_px,
            self.max_reprojection_rms_px,
            self.max_rotation_rad,
            self.max_scale_deviation,
            self.max_translation_ratio,
            self.max_global_roi_center_delta_ratio,
            self.max_global_roi_rotation_delta_rad,
            self.max_global_roi_scale_delta,
            self.min_bbox_retained_ratio,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('reseed registration limits must be finite and positive')
        unit_interval = (
            self.feature_quality,
            self.min_inlier_ratio,
            self.min_bbox_retained_ratio,
        )
        if not all(math.isfinite(value) and 0.0 < value <= 1.0 for value in unit_interval):
            raise ValueError('reseed registration ratios must be in (0, 1]')
        if self.min_global_inliers > self.min_global_tracks:
            raise ValueError('global reseed inliers cannot exceed required tracks')
        if self.min_roi_inliers > self.min_roi_tracks:
            raise ValueError('ROI reseed inliers cannot exceed required tracks')


@dataclass(frozen=True)
class ReseedRegistration:
    """Auditable latest-frame box transfer backed by two geometric fits."""

    bbox_xyxy: tuple[int, int, int, int]
    global_tracks: int
    global_inliers: int
    global_inlier_ratio: float
    global_rms_px: float
    roi_tracks: int
    roi_inliers: int
    roi_inlier_ratio: float
    roi_rms_px: float
    global_rotation_rad: float
    roi_rotation_rad: float
    global_scale: float
    roi_scale: float
    center_delta_ratio: float


@dataclass(frozen=True)
class _RegistrationFit:
    matrix: np.ndarray
    tracks: int
    inliers: int
    inlier_ratio: float
    rms_px: float
    rotation_rad: float
    scale: float


class AcquisitionGate:
    """Require consecutive validated RGB-D updates before first publication."""

    def __init__(self, minimum_updates: int) -> None:
        if (
            isinstance(minimum_updates, bool)
            or not isinstance(minimum_updates, int)
            or minimum_updates < 1
        ):
            raise ValueError('minimum_updates must be a positive integer')
        self._minimum_updates = minimum_updates
        self.reset()

    @property
    def accepted_updates(self) -> int:
        """Return the number of accepted updates in the current acquisition."""
        return self._accepted_updates

    @property
    def ready(self) -> bool:
        """Return whether the minimum consecutive update count was reached."""
        return self._accepted_updates >= self._minimum_updates

    def accept(self) -> bool:
        """Record one validated update and return the resulting readiness."""
        if not self.ready:
            self._accepted_updates += 1
        return self.ready

    def reset(self) -> None:
        """Start a new acquisition window."""
        self._accepted_updates = 0


def _registration_failure(message: str) -> TrackerFailure:
    return TrackerFailure(message, reason_code='seed_reseed_registration')


def _decode_registration_gray(
    image_jpeg: bytes,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if (
        not isinstance(image_jpeg, bytes)
        or not image_jpeg.startswith(b'\xff\xd8')
        or not image_jpeg.endswith(b'\xff\xd9')
    ):
        raise _registration_failure('reseed image is not a bounded JPEG')
    encoded = np.frombuffer(image_jpeg, dtype=np.uint8)
    try:
        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise _registration_failure('reseed JPEG decoding failed') from error
    if gray is None or gray.shape != (height, width):
        raise _registration_failure('reseed JPEG dimensions are invalid')
    return np.asarray(gray, dtype=np.uint8)


def _expanded_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    expansion_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    margin_x = expansion_ratio * float(x2 - x1)
    margin_y = expansion_ratio * float(y2 - y1)
    return (
        max(0, int(math.floor(x1 - margin_x))),
        max(0, int(math.floor(y1 - margin_y))),
        min(width, int(math.ceil(x2 + margin_x))),
        min(height, int(math.ceil(y2 + margin_y))),
    )


def _track_registration_points(
    seed_gray: np.ndarray,
    latest_gray: np.ndarray,
    feature_mask: np.ndarray,
    *,
    config: ReseedRegistrationConfig,
    minimum_tracks: int,
    minimum_inliers: int,
    name: str,
) -> _RegistrationFit:
    try:
        corners = cv2.goodFeaturesToTrack(
            seed_gray,
            maxCorners=config.max_features,
            qualityLevel=config.feature_quality,
            minDistance=config.feature_min_distance_px,
            mask=feature_mask,
            blockSize=5,
            useHarrisDetector=False,
        )
    except cv2.error as error:
        raise _registration_failure(f'{name} reseed feature detection failed') from error
    if corners is None or len(corners) < minimum_tracks:
        count = 0 if corners is None else len(corners)
        raise _registration_failure(
            f'{name} reseed has too few seed features ({count} < {minimum_tracks})',
        )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    try:
        forward, forward_status, _forward_error = cv2.calcOpticalFlowPyrLK(
            seed_gray,
            latest_gray,
            corners,
            None,
            winSize=(config.lk_window_px, config.lk_window_px),
            maxLevel=config.lk_max_level,
            criteria=criteria,
        )
        if forward is None or forward_status is None:
            raise _registration_failure(f'{name} reseed forward flow is unavailable')
        backward, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
            latest_gray,
            seed_gray,
            forward,
            None,
            winSize=(config.lk_window_px, config.lk_window_px),
            maxLevel=config.lk_max_level,
            criteria=criteria,
        )
    except cv2.error as error:
        raise _registration_failure(f'{name} reseed optical flow failed') from error
    if backward is None or backward_status is None:
        raise _registration_failure(f'{name} reseed backward flow is unavailable')
    source_all = np.asarray(corners[:, 0, :], dtype=np.float64)
    destination_all = np.asarray(forward[:, 0, :], dtype=np.float64)
    backward_all = np.asarray(backward[:, 0, :], dtype=np.float64)
    forward_ok = np.asarray(forward_status).reshape(-1) == 1
    backward_ok = np.asarray(backward_status).reshape(-1) == 1
    finite = np.all(np.isfinite(destination_all), axis=1) & np.all(
        np.isfinite(backward_all),
        axis=1,
    )
    fb_error = np.linalg.norm(source_all - backward_all, axis=1)
    valid = (
        forward_ok
        & backward_ok
        & finite
        & (fb_error <= config.max_forward_backward_error_px)
    )
    source = source_all[valid]
    destination = destination_all[valid]
    if len(source) < minimum_tracks:
        raise _registration_failure(
            f'{name} reseed has too few bidirectional tracks '
            f'({len(source)} < {minimum_tracks})',
        )
    try:
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reproj_threshold_px,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
    except cv2.error as error:
        raise _registration_failure(f'{name} reseed RANSAC failed') from error
    if matrix is None or inlier_mask is None:
        raise _registration_failure(f'{name} reseed affine fit is unavailable')
    matrix = np.asarray(matrix, dtype=np.float64)
    inliers = np.asarray(inlier_mask).reshape(-1) == 1
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count) / float(len(source))
    if inlier_count < minimum_inliers or inlier_ratio < config.min_inlier_ratio:
        raise _registration_failure(
            f'{name} reseed RANSAC support is insufficient '
            f'({inlier_count}/{len(source)}, ratio {inlier_ratio:.3f})',
        )
    homogeneous = np.column_stack((source[inliers], np.ones(inlier_count)))
    predicted = homogeneous @ matrix.T
    residuals = np.linalg.norm(predicted - destination[inliers], axis=1)
    rms_px = float(math.sqrt(float(np.mean(np.square(residuals)))))
    if not math.isfinite(rms_px) or rms_px > config.max_reprojection_rms_px:
        raise _registration_failure(
            f'{name} reseed residual is unsafe ({rms_px:.3f}px)',
        )
    a, b, _tx = matrix[0]
    c, d, _ty = matrix[1]
    scale_x = math.hypot(float(a), float(c))
    scale_y = math.hypot(float(b), float(d))
    scale = 0.5 * (scale_x + scale_y)
    rotation = math.atan2(float(c), float(a))
    if (
        not math.isfinite(scale)
        or not math.isfinite(rotation)
        or np.linalg.det(matrix[:, :2]) <= 0.0
        or abs(scale_x - scale_y) > config.max_scale_deviation
        or abs(scale - 1.0) > config.max_scale_deviation
        or abs(rotation) > config.max_rotation_rad
    ):
        raise _registration_failure(
            f'{name} reseed rotation or scale is unsafe '
            f'(rotation {rotation:.4f}, scale {scale:.4f})',
        )
    return _RegistrationFit(
        matrix=matrix,
        tracks=int(len(source)),
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        rms_px=rms_px,
        rotation_rad=rotation,
        scale=scale,
    )


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return np.asarray(homogeneous @ matrix.T, dtype=np.float64)


def register_seed_bbox_to_latest(
    seed_image_jpeg: bytes,
    latest_image_jpeg: bytes,
    bbox_xyxy: Sequence[int],
    *,
    width: int,
    height: int,
    config: ReseedRegistrationConfig | None = None,
) -> ReseedRegistration:
    """Transfer a seed box only when scene and target-region motion agree."""
    limits = config or ReseedRegistrationConfig()
    if width < 2 or height < 2:
        raise _registration_failure('reseed image dimensions are too small')
    if (
        not isinstance(bbox_xyxy, (tuple, list))
        or len(bbox_xyxy) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox_xyxy)
    ):
        raise _registration_failure('reseed bbox must contain four integer pixels')
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise _registration_failure('reseed bbox is outside the seed image')
    seed_gray = _decode_registration_gray(
        seed_image_jpeg,
        width=width,
        height=height,
    )
    latest_gray = _decode_registration_gray(
        latest_image_jpeg,
        width=width,
        height=height,
    )
    expanded = _expanded_bbox(
        (x1, y1, x2, y2),
        width=width,
        height=height,
        expansion_ratio=limits.roi_expansion_ratio,
    )
    ex1, ey1, ex2, ey2 = expanded
    global_mask = np.full((height, width), 255, dtype=np.uint8)
    global_mask[ey1:ey2, ex1:ex2] = 0
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[y1:y2, x1:x2] = 255
    global_fit = _track_registration_points(
        seed_gray,
        latest_gray,
        global_mask,
        config=limits,
        minimum_tracks=limits.min_global_tracks,
        minimum_inliers=limits.min_global_inliers,
        name='global',
    )
    roi_fit = _track_registration_points(
        seed_gray,
        latest_gray,
        roi_mask,
        config=limits,
        minimum_tracks=limits.min_roi_tracks,
        minimum_inliers=limits.min_roi_inliers,
        name='ROI',
    )
    image_diagonal = max(math.hypot(width, height), 1.0)
    image_center = np.array([[0.5 * width, 0.5 * height]], dtype=np.float64)
    global_center_motion = float(np.linalg.norm(
        _transform_points(global_fit.matrix, image_center)[0] - image_center[0],
    )) / image_diagonal
    roi_center_motion = float(np.linalg.norm(
        _transform_points(roi_fit.matrix, image_center)[0] - image_center[0],
    )) / image_diagonal
    if (
        global_center_motion > limits.max_translation_ratio
        or roi_center_motion > limits.max_translation_ratio
    ):
        raise _registration_failure('reseed translation exceeds its image bound')
    bbox_center = np.array(
        [[0.5 * (x1 + x2), 0.5 * (y1 + y2)]],
        dtype=np.float64,
    )
    global_bbox_center = _transform_points(global_fit.matrix, bbox_center)[0]
    roi_bbox_center = _transform_points(roi_fit.matrix, bbox_center)[0]
    center_delta_ratio = float(np.linalg.norm(
        global_bbox_center - roi_bbox_center,
    )) / image_diagonal
    if (
        center_delta_ratio > limits.max_global_roi_center_delta_ratio
        or abs(global_fit.rotation_rad - roi_fit.rotation_rad)
        > limits.max_global_roi_rotation_delta_rad
        or abs(global_fit.scale - roi_fit.scale)
        > limits.max_global_roi_scale_delta
    ):
        raise _registration_failure(
            'reseed ROI motion is inconsistent with global scene motion',
        )
    corners = np.array(
        ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        dtype=np.float64,
    )
    mapped = _transform_points(roi_fit.matrix, corners)
    if not np.all(np.isfinite(mapped)):
        raise _registration_failure('reseed bbox transform is non-finite')
    unclamped = (
        float(np.min(mapped[:, 0])),
        float(np.min(mapped[:, 1])),
        float(np.max(mapped[:, 0])),
        float(np.max(mapped[:, 1])),
    )
    ux1, uy1, ux2, uy2 = unclamped
    raw_area = max(0.0, ux2 - ux1) * max(0.0, uy2 - uy1)
    cx1 = max(0, min(width, int(math.floor(ux1))))
    cy1 = max(0, min(height, int(math.floor(uy1))))
    cx2 = max(0, min(width, int(math.ceil(ux2))))
    cy2 = max(0, min(height, int(math.ceil(uy2))))
    retained_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    if (
        raw_area <= 0.0
        or cx1 >= cx2
        or cy1 >= cy2
        or float(retained_area) / raw_area < limits.min_bbox_retained_ratio
    ):
        raise _registration_failure('reseed bbox left the latest image')
    if not (
        ex1 <= roi_bbox_center[0] <= ex2
        and ey1 <= roi_bbox_center[1] <= ey2
    ):
        raise _registration_failure('reseed ROI moved outside its bounded search region')
    return ReseedRegistration(
        bbox_xyxy=(cx1, cy1, cx2, cy2),
        global_tracks=global_fit.tracks,
        global_inliers=global_fit.inliers,
        global_inlier_ratio=global_fit.inlier_ratio,
        global_rms_px=global_fit.rms_px,
        roi_tracks=roi_fit.tracks,
        roi_inliers=roi_fit.inliers,
        roi_inlier_ratio=roi_fit.inlier_ratio,
        roi_rms_px=roi_fit.rms_px,
        global_rotation_rad=global_fit.rotation_rad,
        roi_rotation_rad=roi_fit.rotation_rad,
        global_scale=global_fit.scale,
        roi_scale=roi_fit.scale,
        center_delta_ratio=center_delta_ratio,
    )


def _clean_identity_component(
    mask: object,
    reference_mask: object | None,
) -> tuple[np.ndarray, MaskCleanupDiagnostics]:
    """Select one 8-connected identity component before geometric gates."""
    raw = np.asarray(mask, dtype=bool)
    if raw.ndim != 2:
        raise TrackerFailure('EdgeTAM mask dimensions changed')
    raw_pixels = int(np.count_nonzero(raw))
    if raw_pixels == 0:
        raise TrackerFailure('EdgeTAM mask is empty')
    if reference_mask is None:
        reference = None
    else:
        reference = np.asarray(reference_mask, dtype=bool)
        if reference.shape != raw.shape:
            raise TrackerFailure('EdgeTAM mask dimensions changed')

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.asarray(raw, dtype=np.uint8),
        connectivity=8,
    )
    component_count = int(count - 1)
    if component_count < 1:
        raise TrackerFailure('EdgeTAM mask is empty')
    overlap_counts = np.zeros(count, dtype=np.int64)
    reference_centroid: np.ndarray | None = None
    if reference is not None and np.any(reference):
        overlap_counts = np.bincount(
            labels[reference],
            minlength=count,
        ).astype(np.int64, copy=False)
        rows, cols = np.nonzero(reference)
        reference_centroid = np.array(
            [float(cols.mean()), float(rows.mean())],
            dtype=np.float64,
        )

    def rank(label: int) -> tuple[float, ...]:
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int(overlap_counts[label])
        centroid = np.asarray(centroids[label], dtype=np.float64)
        distance = (
            0.0
            if reference_centroid is None
            else float(np.linalg.norm(centroid - reference_centroid))
        )
        if overlap > 0:
            return (1.0, float(overlap), overlap / area, -distance, float(area), -label)
        return (0.0, float(area), -distance, 0.0, 0.0, -label)

    selected_label = max(range(1, count), key=rank)
    cleaned = np.asarray(labels == selected_label, dtype=bool)
    cleaned_pixels = int(stats[selected_label, cv2.CC_STAT_AREA])
    rejected_component_areas = tuple(
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
        if label != selected_label
    )
    selected_overlap = int(overlap_counts[selected_label])
    if component_count == 1:
        selection_mode = 'single_component'
    elif selected_overlap > 0:
        selection_mode = 'reference_overlap'
    else:
        selection_mode = 'largest_fallback'
    diagnostics = MaskCleanupDiagnostics(
        raw_pixels=raw_pixels,
        cleaned_pixels=cleaned_pixels,
        component_count=component_count,
        rejected_pixels=raw_pixels - cleaned_pixels,
        largest_rejected_component_pixels=max(rejected_component_areas, default=0),
        selected_overlap_pixels=selected_overlap,
        selection_mode=selection_mode,
    )
    return cleaned, diagnostics


@dataclass(frozen=True, eq=False)
class TargetDepthProjection:
    """One depth-connected target cluster and its rejected mask pixels."""

    mask: np.ndarray
    points_xyz: np.ndarray
    pixels_uv: np.ndarray
    rejected_pixels_uv: np.ndarray


def center_box_to_half_open(
    center_x: float,
    center_y: float,
    size_x: float,
    size_y: float,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a vision center/size box to an enclosing half-open pixel box."""
    values = (center_x, center_y, size_x, size_y)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('bbox values must be finite')
    if width < 1 or height < 1 or size_x <= 0.0 or size_y <= 0.0:
        raise ValueError('bbox and image dimensions must be positive')
    x1 = max(0, min(width, math.floor(center_x - 0.5 * size_x)))
    y1 = max(0, min(height, math.floor(center_y - 0.5 * size_y)))
    x2 = max(0, min(width, math.ceil(center_x + 0.5 * size_x)))
    y2 = max(0, min(height, math.ceil(center_y + 0.5 * size_y)))
    if x1 >= x2 or y1 >= y2:
        raise ValueError('bbox is empty or outside the image')
    return int(x1), int(y1), int(x2), int(y2)


def project_mask_depth(
    mask: object,
    depth_m: object,
    intrinsics: CameraIntrinsics,
    *,
    min_depth_m: float,
    max_depth_m: float,
    min_points: int,
    max_points: int,
    cluster_max_depth_jump_m: float = 0.06,
    cluster_max_depth_jump_ratio: float = 0.03,
) -> np.ndarray:
    """Back-project the dominant measured-depth cluster in the current mask."""
    points, _pixels = project_mask_depth_with_pixels(
        mask,
        depth_m,
        intrinsics,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        min_points=min_points,
        max_points=max_points,
        cluster_max_depth_jump_m=cluster_max_depth_jump_m,
        cluster_max_depth_jump_ratio=cluster_max_depth_jump_ratio,
    )
    return points


def project_mask_depth_with_pixels(
    mask: object,
    depth_m: object,
    intrinsics: CameraIntrinsics,
    *,
    min_depth_m: float,
    max_depth_m: float,
    min_points: int,
    max_points: int,
    cluster_max_depth_jump_m: float = 0.06,
    cluster_max_depth_jump_ratio: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project the target cluster while retaining aligned source pixels."""
    projection = project_mask_depth_geometry(
        mask,
        depth_m,
        intrinsics,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        min_points=min_points,
        max_points=max_points,
        cluster_max_depth_jump_m=cluster_max_depth_jump_m,
        cluster_max_depth_jump_ratio=cluster_max_depth_jump_ratio,
    )
    return projection.points_xyz, projection.pixels_uv


def project_mask_depth_geometry(
    mask: object,
    depth_m: object,
    intrinsics: CameraIntrinsics,
    *,
    min_depth_m: float,
    max_depth_m: float,
    min_points: int,
    max_points: int,
    cluster_max_depth_jump_m: float = 0.06,
    cluster_max_depth_jump_ratio: float = 0.03,
) -> TargetDepthProjection:
    """
    Select a dominant 8-connected depth cluster from a segmentation mask.

    Spatially adjacent pixels are connected only when their measured depths
    agree within an absolute-or-relative sensor tolerance. This rejects shelf
    and wall depth leaking through a segmentation mask without using object
    models or simulator state. Rejected valid pixels remain identified so the
    ROS adapter can conservatively restore them to the collision scene.
    """
    if not (
        math.isfinite(min_depth_m)
        and math.isfinite(max_depth_m)
        and 0.0 <= min_depth_m < max_depth_m
    ):
        raise ValueError('depth bounds must be finite and increasing')
    if min_points < 1 or max_points < min_points:
        raise ValueError('point limits must be positive and ordered')
    if not (
        math.isfinite(cluster_max_depth_jump_m)
        and cluster_max_depth_jump_m > 0.0
        and math.isfinite(cluster_max_depth_jump_ratio)
        and cluster_max_depth_jump_ratio >= 0.0
    ):
        raise ValueError('depth-cluster tolerances are invalid')
    target_mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if target_mask.ndim != 2 or depth.shape != target_mask.shape:
        raise TrackerFailure('mask and aligned depth dimensions changed')
    valid = (
        target_mask
        & np.isfinite(depth)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    rows, cols = np.nonzero(valid)
    if rows.size < min_points:
        raise TrackerFailure(
            f'target depth is sparse: {rows.size} valid points, need {min_points}',
        )
    selected = _dominant_depth_component(
        valid,
        depth,
        max_depth_jump_m=float(cluster_max_depth_jump_m),
        max_depth_jump_ratio=float(cluster_max_depth_jump_ratio),
    )
    selected_rows, selected_cols = np.nonzero(selected)
    if selected_rows.size < min_points:
        raise TrackerFailure(
            'dominant target depth cluster is sparse: '
            f'{selected_rows.size} valid points, need {min_points}',
        )
    rejected = valid & ~selected
    rejected_rows, rejected_cols = np.nonzero(rejected)

    points = _backproject_pixels(
        depth,
        selected_rows,
        selected_cols,
        intrinsics,
    )
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        selected_rows = selected_rows[indices]
        selected_cols = selected_cols[indices]
    points = np.asarray(points, dtype=np.float32)
    pixels = np.column_stack((selected_cols, selected_rows)).astype(
        np.uint32,
        copy=False,
    )
    rejected_pixels = np.column_stack((rejected_cols, rejected_rows)).astype(
        np.uint32,
        copy=False,
    )
    selected.setflags(write=False)
    points.setflags(write=False)
    pixels.setflags(write=False)
    rejected_pixels.setflags(write=False)
    return TargetDepthProjection(
        mask=selected,
        points_xyz=points,
        pixels_uv=pixels,
        rejected_pixels_uv=rejected_pixels,
    )


def _dominant_depth_component(
    valid: np.ndarray,
    depth: np.ndarray,
    *,
    max_depth_jump_m: float,
    max_depth_jump_ratio: float,
) -> np.ndarray:
    """Return the largest deterministic component in an RGB-D pixel graph."""
    rows, cols = np.nonzero(valid)
    count = int(rows.size)
    pixel_ids = np.full(valid.shape, -1, dtype=np.int32)
    pixel_ids[rows, cols] = np.arange(count, dtype=np.int32)
    parents = np.arange(count, dtype=np.int32)
    sizes = np.ones(count, dtype=np.int32)

    def find(index: int) -> int:
        root = index
        while int(parents[root]) != root:
            root = int(parents[root])
        while index != root:
            parent = int(parents[index])
            parents[index] = root
            index = parent
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if (
            sizes[first_root] < sizes[second_root]
            or (
                sizes[first_root] == sizes[second_root]
                and first_root > second_root
            )
        ):
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root
        sizes[first_root] += sizes[second_root]

    height, width = valid.shape
    for row_offset, column_offset in ((0, 1), (1, -1), (1, 0), (1, 1)):
        row_a = slice(max(0, -row_offset), height - max(0, row_offset))
        row_b = slice(max(0, row_offset), height - max(0, -row_offset))
        col_a = slice(max(0, -column_offset), width - max(0, column_offset))
        col_b = slice(max(0, column_offset), width - max(0, -column_offset))
        valid_a = valid[row_a, col_a]
        valid_b = valid[row_b, col_b]
        depth_a = depth[row_a, col_a]
        depth_b = depth[row_b, col_b]
        tolerance = np.maximum(
            max_depth_jump_m,
            max_depth_jump_ratio * np.minimum(depth_a, depth_b),
        )
        connected = valid_a & valid_b & (np.abs(depth_a - depth_b) <= tolerance)
        first_ids = pixel_ids[row_a, col_a][connected]
        second_ids = pixel_ids[row_b, col_b][connected]
        for first, second in zip(first_ids.tolist(), second_ids.tolist()):
            union(int(first), int(second))

    roots = np.fromiter(
        (find(index) for index in range(count)),
        dtype=np.int32,
        count=count,
    )
    unique_roots, component_sizes = np.unique(roots, return_counts=True)
    largest_size = int(component_sizes.max())
    candidates = unique_roots[component_sizes == largest_size]
    first_members = np.full(count, count, dtype=np.int32)
    np.minimum.at(first_members, roots, np.arange(count, dtype=np.int32))
    dominant_root = int(candidates[np.argmin(first_members[candidates])])
    selected = np.zeros(valid.shape, dtype=bool)
    members = roots == dominant_root
    selected[rows[members], cols[members]] = True
    return selected


def _backproject_pixels(
    depth: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project aligned integer pixels into the optical frame."""
    z = depth[rows, cols]
    return np.column_stack(
        (
            (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx,
            (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy,
            z,
        ),
    )


def project_scene_depth(
    mask: object,
    depth_m: object,
    intrinsics: CameraIntrinsics,
    *,
    target_dilation_px: int,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
    restore_pixels_uv: object | None = None,
) -> np.ndarray:
    """Back-project the scene, restoring valid mask points rejected as target."""
    target = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if target.ndim != 2 or depth.shape != target.shape:
        raise TrackerFailure('mask and aligned scene depth dimensions changed')
    if target_dilation_px < 0 or stride < 1 or max_points < 1:
        raise ValueError('scene dilation, stride, and point limit are invalid')
    if not 0.0 <= min_depth_m < max_depth_m:
        raise ValueError('scene depth interval is invalid')
    if target_dilation_px:
        radius = int(target_dilation_px)
        padded = np.pad(target, radius, mode='constant', constant_values=False)
        dilated = np.zeros_like(target)
        height, width = target.shape
        for row_offset in range(2 * radius + 1):
            for column_offset in range(2 * radius + 1):
                dilated |= padded[
                    row_offset:row_offset + height,
                    column_offset:column_offset + width,
                ]
        target = dilated
    rows = np.arange(0, depth.shape[0], stride)
    columns = np.arange(0, depth.shape[1], stride)
    u, v = np.meshgrid(columns, rows)
    sampled = depth[np.ix_(rows, columns)]
    excluded = target[np.ix_(rows, columns)]
    valid = (
        ~excluded
        & np.isfinite(sampled)
        & (sampled >= min_depth_m)
        & (sampled <= max_depth_m)
    )
    scene_rows = v[valid].astype(np.int64, copy=False)
    scene_cols = u[valid].astype(np.int64, copy=False)
    scene_flat = scene_rows * depth.shape[1] + scene_cols

    restore_flat = np.empty(0, dtype=np.int64)
    if restore_pixels_uv is not None:
        restored = np.asarray(restore_pixels_uv)
        if restored.ndim != 2 or restored.shape[1] != 2:
            raise ValueError('restored pixels must have shape (N, 2)')
        if restored.size:
            if not np.issubdtype(restored.dtype, np.integer):
                raise ValueError('restored pixels must use integer coordinates')
            restored = restored.astype(np.int64, copy=False)
            restore_cols = restored[:, 0]
            restore_rows = restored[:, 1]
            if (
                np.any(restore_cols < 0)
                or np.any(restore_cols >= depth.shape[1])
                or np.any(restore_rows < 0)
                or np.any(restore_rows >= depth.shape[0])
            ):
                raise ValueError('restored pixels fall outside the depth image')
            restore_depth = depth[restore_rows, restore_cols]
            restore_valid = (
                np.isfinite(restore_depth)
                & (restore_depth >= min_depth_m)
                & (restore_depth <= max_depth_m)
            )
            restore_flat = np.unique(
                restore_rows[restore_valid] * depth.shape[1]
                + restore_cols[restore_valid],
            )

    if restore_flat.size:
        scene_flat = scene_flat[~np.isin(scene_flat, restore_flat)]
        if restore_flat.size >= max_points:
            indices = np.linspace(
                0,
                restore_flat.size - 1,
                max_points,
                dtype=np.int64,
            )
            restore_flat = restore_flat[indices]
            scene_flat = np.empty(0, dtype=np.int64)
        else:
            scene_budget = max_points - restore_flat.size
            if scene_flat.size > scene_budget:
                indices = np.linspace(
                    0,
                    scene_flat.size - 1,
                    scene_budget,
                    dtype=np.int64,
                )
                scene_flat = scene_flat[indices]
    elif scene_flat.size > max_points:
        indices = np.linspace(0, scene_flat.size - 1, max_points, dtype=np.int64)
        scene_flat = scene_flat[indices]

    selected_flat = np.concatenate((scene_flat, restore_flat))
    selected_rows = selected_flat // depth.shape[1]
    selected_cols = selected_flat % depth.shape[1]
    points = _backproject_pixels(
        depth,
        selected_rows,
        selected_cols,
        intrinsics,
    )
    return np.asarray(points, dtype=np.float32)


class FailClosedTracker:
    """Serialize one service identity and reject every unsafe observation."""

    def __init__(
        self,
        client: ServiceClient,
        *,
        min_depth_m: float = 0.28,
        max_depth_m: float = 2.5,
        min_points: int = 24,
        max_points: int = 20_000,
        min_mask_iou: float = 0.15,
        hard_min_mask_iou: float = 0.03,
        min_mask_area_ratio: float = 0.35,
        max_mask_displacement_ratio: float = 0.65,
        min_mask_overlap_ratio: float = 0.50,
        min_mask_bbox_iou: float = 0.10,
        max_soft_continuity_frames: int = 2,
        min_soft_depth_mask_retention: float = 0.35,
        allow_motion_reanchor: bool = False,
        min_motion_reanchor_area_ratio: float = 0.60,
        max_motion_reanchor_displacement_ratio: float = 1.25,
        max_contained_collapse_recovery_frames: int = 2,
        max_tracking_coast_frames: int = 32,
        max_centroid_speed_mps: float = 2.0,
        max_mask_area_ratio: float = 0.35,
        max_rejected_mask_ratio: float = 0.12,
        max_largest_rejected_to_selected_ratio: float = 0.20,
        cluster_max_depth_jump_m: float = 0.06,
        cluster_max_depth_jump_ratio: float = 0.03,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if min_points < 1 or max_points < min_points:
            raise ValueError('point limits must be positive and ordered')
        if not (
            math.isfinite(min_depth_m)
            and math.isfinite(max_depth_m)
            and 0.0 <= min_depth_m < max_depth_m
        ):
            raise ValueError('depth bounds must be finite and increasing')
        if not 0.0 < min_mask_iou <= 1.0:
            raise ValueError('min_mask_iou must be in (0, 1]')
        if not (
            math.isfinite(hard_min_mask_iou)
            and 0.0 <= hard_min_mask_iou < min_mask_iou
        ):
            raise ValueError('hard_min_mask_iou must be finite and below min_mask_iou')
        if not (
            math.isfinite(min_mask_area_ratio)
            and 0.0 < min_mask_area_ratio <= 1.0
        ):
            raise ValueError('min_mask_area_ratio must be finite and in (0, 1]')
        if not (
            math.isfinite(max_mask_displacement_ratio)
            and max_mask_displacement_ratio > 0.0
        ):
            raise ValueError('max_mask_displacement_ratio must be finite and positive')
        if not (
            math.isfinite(min_mask_overlap_ratio)
            and 0.0 < min_mask_overlap_ratio <= 1.0
        ):
            raise ValueError('min_mask_overlap_ratio must be finite and in (0, 1]')
        if not (
            math.isfinite(min_mask_bbox_iou)
            and 0.0 < min_mask_bbox_iou <= 1.0
        ):
            raise ValueError('min_mask_bbox_iou must be finite and in (0, 1]')
        if (
            isinstance(max_soft_continuity_frames, bool)
            or not isinstance(max_soft_continuity_frames, int)
            or max_soft_continuity_frames < 0
        ):
            raise ValueError('max_soft_continuity_frames must be a non-negative integer')
        if not (
            math.isfinite(min_soft_depth_mask_retention)
            and 0.0 < min_soft_depth_mask_retention <= 1.0
        ):
            raise ValueError(
                'min_soft_depth_mask_retention must be finite and in (0, 1]',
            )
        if not isinstance(allow_motion_reanchor, bool):
            raise ValueError('allow_motion_reanchor must be boolean')
        if not (
            math.isfinite(min_motion_reanchor_area_ratio)
            and 0.0 < min_motion_reanchor_area_ratio <= 1.0
        ):
            raise ValueError(
                'min_motion_reanchor_area_ratio must be finite and in (0, 1]',
            )
        if not (
            math.isfinite(max_motion_reanchor_displacement_ratio)
            and max_motion_reanchor_displacement_ratio > 0.0
        ):
            raise ValueError(
                'max_motion_reanchor_displacement_ratio must be finite and positive',
            )
        if (
            isinstance(max_contained_collapse_recovery_frames, bool)
            or not isinstance(max_contained_collapse_recovery_frames, int)
            or not 1 <= max_contained_collapse_recovery_frames <= 2
        ):
            raise ValueError(
                'max_contained_collapse_recovery_frames must be 1 or 2',
            )
        if (
            isinstance(max_tracking_coast_frames, bool)
            or not isinstance(max_tracking_coast_frames, int)
            or max_tracking_coast_frames < 0
        ):
            raise ValueError('max_tracking_coast_frames must be a non-negative integer')
        if not math.isfinite(max_centroid_speed_mps) or max_centroid_speed_mps <= 0.0:
            raise ValueError('max_centroid_speed_mps must be finite and positive')
        if not math.isfinite(max_mask_area_ratio) or not 0.0 < max_mask_area_ratio <= 1.0:
            raise ValueError('max_mask_area_ratio must be finite and in (0, 1]')
        if not (
            math.isfinite(max_rejected_mask_ratio)
            and 0.0 <= max_rejected_mask_ratio < 1.0
            and math.isfinite(max_largest_rejected_to_selected_ratio)
            and 0.0 <= max_largest_rejected_to_selected_ratio < 1.0
        ):
            raise ValueError('component cleanup ratios must be finite and in [0, 1)')
        if not (
            math.isfinite(cluster_max_depth_jump_m)
            and cluster_max_depth_jump_m > 0.0
            and math.isfinite(cluster_max_depth_jump_ratio)
            and cluster_max_depth_jump_ratio >= 0.0
        ):
            raise ValueError('depth-cluster tolerances are invalid')
        self._client = client
        self._min_depth_m = float(min_depth_m)
        self._max_depth_m = float(max_depth_m)
        self._min_points = int(min_points)
        self._max_points = int(max_points)
        self._min_mask_iou = float(min_mask_iou)
        self._hard_min_mask_iou = float(hard_min_mask_iou)
        self._min_mask_area_ratio = float(min_mask_area_ratio)
        self._max_mask_displacement_ratio = float(max_mask_displacement_ratio)
        self._min_mask_overlap_ratio = float(min_mask_overlap_ratio)
        self._min_mask_bbox_iou = float(min_mask_bbox_iou)
        self._max_soft_continuity_frames = int(max_soft_continuity_frames)
        self._min_soft_depth_mask_retention = float(
            min_soft_depth_mask_retention,
        )
        self._allow_motion_reanchor = allow_motion_reanchor
        self._min_motion_reanchor_area_ratio = float(
            min_motion_reanchor_area_ratio,
        )
        self._max_motion_reanchor_displacement_ratio = float(
            max_motion_reanchor_displacement_ratio,
        )
        self._max_contained_collapse_recovery_frames = int(
            max_contained_collapse_recovery_frames,
        )
        self._max_tracking_coast_frames = int(max_tracking_coast_frames)
        self._max_centroid_speed_mps = float(max_centroid_speed_mps)
        self._max_mask_area_ratio = float(max_mask_area_ratio)
        self._max_rejected_mask_ratio = float(max_rejected_mask_ratio)
        self._max_largest_rejected_to_selected_ratio = float(
            max_largest_rejected_to_selected_ratio,
        )
        self._cluster_max_depth_jump_m = float(cluster_max_depth_jump_m)
        self._cluster_max_depth_jump_ratio = float(cluster_max_depth_jump_ratio)
        self._session_id_factory = session_id_factory or (
            lambda: f'ros-{uuid.uuid4().hex}'
        )
        self._clear()

    @property
    def active(self) -> bool:
        """Return whether a validated identity is active locally."""
        return self._track_id is not None

    @property
    def last_stamp_ns(self) -> int | None:
        """Return the last image timestamp accepted by the service."""
        return self._last_stamp_ns

    @property
    def pending_mask_anomaly(self) -> bool:
        """Return whether publication is held for contained-collapse recovery."""
        return self._pending_mask_anomaly

    def initialize(
        self,
        *,
        stamp_ns: int,
        image_jpeg: bytes,
        width: int,
        height: int,
        bbox_xyxy: Sequence[int],
        label: str = '',
    ) -> TrackResult:
        """Start a new identity on the exact cached image used by grounding."""
        self.reset()
        bbox = self._validate_bbox(bbox_xyxy, width, height)
        if isinstance(stamp_ns, bool) or stamp_ns < 0:
            raise TrackerFailure('initialization timestamp is invalid')
        session_id = self._session_id_factory()
        try:
            result = self._client.init(
                image_jpeg,
                bbox,
                session_id=session_id,
                frame_seq=0,
            )
            self._validate_result(result, width=width, height=height, frame_seq=0)
            if result.session_id != session_id:
                raise TrackerFailure('EdgeTAM initialization session identity changed')
            bounded_mask, bounded_diagnostics = self._bounded_mask(
                result.mask,
                bbox,
                width,
                height,
            )
        except Exception as error:
            self._clear()
            self._best_effort_reset()
            reason_code = str(
                getattr(error, 'reason_code', 'service_init_failed'),
            ).strip() or 'service_init_failed'
            raise TrackerFailure(
                f'EdgeTAM initialization failed ({type(error).__name__})',
                reason_code=reason_code,
            ) from error
        self._session_id = result.session_id
        self._track_id = result.track_id
        self._label = label.strip()
        self._width = int(width)
        self._height = int(height)
        self._seed_bbox = bbox
        self._last_stamp_ns = int(stamp_ns)
        self._next_frame_seq = 1
        self._last_mask = bounded_mask
        self._last_mask_diagnostics = bounded_diagnostics
        self._last_raw_mask_diagnostics = self._last_mask_diagnostics
        self._continuity_anchor_mask = self._last_mask.copy()
        self._last_validated_stamp_ns = int(stamp_ns)
        return result

    def update(self, frame: RgbdFrame) -> TrackingObservation | None:
        """Update exactly once and build a mask-aligned measured target cloud."""
        try:
            result, current_mask, mask_diagnostics = self._request_update(
                stamp_ns=frame.stamp_ns,
                image_jpeg=frame.image_jpeg,
                width=frame.width,
                height=frame.height,
            )
            if getattr(result, 'coasting', False):
                # Keep the identity alive across a brief occlusion-driven gap
                # without publishing an observation.  The validated anchor,
                # centroid, and continuity references are intentionally left
                # unchanged; only the service timeline advances.
                self._register_coast()
                self._commit_service_update(frame.stamp_ns)
                return None
            continuity_metrics = self._current_continuity_metrics(current_mask)
            if continuity_metrics is not None:
                if self._is_contained_scale_collapse(current_mask):
                    self._hold_contained_collapse(
                        frame.stamp_ns,
                        mask_diagnostics,
                    )
                    return None
            projection = project_mask_depth_geometry(
                current_mask,
                frame.depth_m,
                frame.intrinsics,
                min_depth_m=self._min_depth_m,
                max_depth_m=self._max_depth_m,
                min_points=self._min_points,
                max_points=self._max_points,
                cluster_max_depth_jump_m=self._cluster_max_depth_jump_m,
                cluster_max_depth_jump_ratio=self._cluster_max_depth_jump_ratio,
            )
            points = projection.points_xyz
            pixels = projection.pixels_uv
            centroid = np.median(points, axis=0)
            centroid_continuity_validated = False
            if self._last_centroid is not None:
                if self._last_validated_stamp_ns is None:
                    raise TrackerFailure('validated centroid timestamp is unavailable')
                dt = (frame.stamp_ns - self._last_validated_stamp_ns) * 1e-9
                speed = float(np.linalg.norm(centroid - self._last_centroid) / dt)
                if speed > self._max_centroid_speed_mps:
                    raise TrackerFailure(
                        f'target centroid continuity broke ({speed:.3f} m/s)',
                        reason_code='centroid_continuity',
                    )
                centroid_continuity_validated = True
            cleaned_mask_pixels = int(np.count_nonzero(current_mask))
            depth_retention = float(np.count_nonzero(projection.mask)) / float(
                cleaned_mask_pixels,
            )
            if self._pending_mask_anomaly:
                recovery_ready = self._advance_pending_recovery(
                    current_mask,
                    continuity_metrics,
                )
                if not recovery_ready:
                    self._commit_pending_image_update(
                        frame.stamp_ns,
                        mask_diagnostics,
                    )
                    return None
            self._validate_mask_continuity(
                current_mask,
                depth_mask_retention=depth_retention,
                centroid_continuity_validated=centroid_continuity_validated,
                metrics=continuity_metrics,
            )
        except Exception as error:
            self._raise_update_failure(error)
        self._commit_image_update(
            frame.stamp_ns,
            current_mask,
            mask_diagnostics,
        )
        mask = projection.mask.copy()
        self._last_centroid = centroid.copy()
        mask.setflags(write=False)
        rows, cols = np.nonzero(mask)
        effective_bbox = (
            int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1,
        )
        return TrackingObservation(
            stamp_ns=frame.stamp_ns,
            frame_id=frame.frame_id,
            session_id=str(self._session_id),
            track_id=str(self._track_id),
            label=self._label,
            score=float(result.score),
            bbox_xyxy=effective_bbox,
            mask=mask,
            points_xyz=points,
            pixels_uv=pixels,
            rejected_pixels_uv=projection.rejected_pixels_uv,
            mask_diagnostics=mask_diagnostics,
        )

    def replay_rgb(
        self,
        *,
        stamp_ns: int,
        image_jpeg: bytes,
        width: int,
        height: int,
    ) -> bool:
        """Advance a newly seeded 2-D identity without publishing depth geometry."""
        try:
            if self._last_centroid is not None:
                raise TrackerFailure(
                    'RGB replay is only valid before the first depth observation',
                    reason_code='invalid_replay_phase',
                )
            _result, current_mask, mask_diagnostics = self._request_update(
                stamp_ns=stamp_ns,
                image_jpeg=image_jpeg,
                width=width,
                height=height,
            )
            if getattr(_result, 'coasting', False):
                # A coast during 2-D acquisition makes no forward progress but
                # keeps the identity alive and advances the service timeline.
                self._register_coast()
                self._commit_service_update(stamp_ns)
                return False
            continuity_metrics = self._current_continuity_metrics(current_mask)
            if continuity_metrics is not None and self._is_contained_scale_collapse(
                current_mask,
            ):
                self._hold_contained_collapse(stamp_ns, mask_diagnostics)
                return False
            if continuity_metrics is not None:
                self._reject_hard_mask_discontinuity(continuity_metrics)
            if self._pending_mask_anomaly and not self._advance_pending_recovery(
                current_mask,
                continuity_metrics,
            ):
                self._commit_pending_image_update(stamp_ns, mask_diagnostics)
                return False
            self._validate_mask_continuity(current_mask)
        except Exception as error:
            self._raise_update_failure(error)
        self._commit_image_update(stamp_ns, current_mask, mask_diagnostics)
        return True

    def _request_update(
        self,
        *,
        stamp_ns: int,
        image_jpeg: bytes,
        width: int,
        height: int,
    ) -> tuple[TrackResult, np.ndarray | None, MaskCleanupDiagnostics | None]:
        if not self.active or self._last_stamp_ns is None:
            raise TrackerFailure('cannot update without an initialized identity')
        if isinstance(stamp_ns, bool) or stamp_ns <= self._last_stamp_ns:
            raise TrackerFailure(
                'RGB-D frame timestamp is duplicate or out of order',
                reason_code='frame_order',
            )
        if width != self._width or height != self._height:
            raise TrackerFailure(
                'RGB-D image dimensions changed within a tracking session',
                reason_code='image_size_changed',
            )
        frame_seq = self._next_frame_seq
        result = self._client.update(image_jpeg, frame_seq=frame_seq)
        if getattr(result, 'coasting', False):
            # Session-preserving keep-alive: the service kept the identity but
            # produced no trustworthy mask this frame.  Enforce identity
            # lockstep, then defer to the caller (no mask/depth is available).
            self._validate_coast(result, width=width, height=height, frame_seq=frame_seq)
            return result, None, None
        self._validate_result(
            result,
            width=width,
            height=height,
            frame_seq=frame_seq,
        )
        if result.session_id != self._session_id:
            raise TrackerFailure('EdgeTAM session identity changed')
        if result.track_id != self._track_id:
            raise TrackerFailure('EdgeTAM target identity changed')
        raw_mask = np.asarray(result.mask, dtype=bool)
        raw_area_ratio = float(np.count_nonzero(raw_mask)) / float(width * height)
        if raw_area_ratio > self._max_mask_area_ratio:
            raise TrackerFailure(
                f'EdgeTAM raw mask area is unsafe ({raw_area_ratio:.3f} of image)',
                reason_code='mask_area_limit',
            )
        current_mask, diagnostics = _clean_identity_component(
            raw_mask,
            self._last_mask,
        )
        self._validate_component_cleanup(diagnostics)
        return result, current_mask, diagnostics

    def _register_coast(self) -> None:
        """Bound how long tracking may coast without a published observation."""
        self._coast_streak += 1
        if self._coast_streak > self._max_tracking_coast_frames:
            raise TrackerFailure(
                'EdgeTAM tracking coasted beyond the configured window '
                f'({self._coast_streak} > {self._max_tracking_coast_frames} frames)',
                reason_code='tracking_coast_exhausted',
            )

    def _validate_coast(
        self,
        result: object,
        *,
        width: int,
        height: int,
        frame_seq: int,
    ) -> None:
        """Enforce identity lockstep on a session-preserving coast frame."""
        if getattr(result, 'frame_seq', None) != frame_seq:
            raise TrackerFailure('EdgeTAM coast sequence changed')
        if tuple(getattr(result, 'image_size', ())) != (width, height):
            raise TrackerFailure('EdgeTAM coast image dimensions changed')
        session_id = getattr(result, 'session_id', '')
        track_id = getattr(result, 'track_id', '')
        if not session_id or not track_id:
            raise TrackerFailure('EdgeTAM returned an empty identity')
        if session_id != self._session_id:
            raise TrackerFailure('EdgeTAM session identity changed')
        if track_id != self._track_id:
            raise TrackerFailure('EdgeTAM target identity changed')

    def _validate_component_cleanup(
        self,
        diagnostics: MaskCleanupDiagnostics,
    ) -> None:
        """Allow only bounded speck removal, never a competing target component."""
        if (
            diagnostics.rejected_ratio > self._max_rejected_mask_ratio
            or diagnostics.largest_rejected_to_selected_ratio
            > self._max_largest_rejected_to_selected_ratio
        ):
            raise TrackerFailure(
                'EdgeTAM mask has ambiguous competing components '
                f'(rejected/raw {diagnostics.rejected_ratio:.3f} > '
                f'{self._max_rejected_mask_ratio:.3f} or largest/selected '
                f'{diagnostics.largest_rejected_to_selected_ratio:.3f} > '
                f'{self._max_largest_rejected_to_selected_ratio:.3f})',
                reason_code='mask_component_ambiguity',
            )

    def _validate_mask_continuity(
        self,
        current_mask: np.ndarray,
        *,
        depth_mask_retention: float | None = None,
        centroid_continuity_validated: bool = False,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Reject identity jumps while tolerating bounded segmentation jitter."""
        if self._last_mask is None:
            return
        if metrics is None:
            metrics = self._current_continuity_metrics(current_mask)
        if metrics is None:
            return
        if self._motion_reanchor_is_valid(
            metrics,
            depth_mask_retention=depth_mask_retention,
            centroid_continuity_validated=centroid_continuity_validated,
        ):
            # A mobile camera can move one correctly tracked object far enough
            # between serialized EdgeTAM results that adjacent masks no longer
            # overlap.  Once live depth and 3-D centroid continuity independently
            # confirm the identity, advance the anchor instead of comparing every
            # later result with the pre-turn image forever.
            self._soft_continuity_frames = 0
            self._continuity_anchor_mask = current_mask.copy()
            return
        self._reject_hard_mask_discontinuity(metrics)

        if metrics['iou'] >= self._min_mask_iou:
            self._soft_continuity_frames = 0
            self._continuity_anchor_mask = current_mask.copy()
            return

        soft_reasons: list[str] = []
        if metrics['overlap_ratio'] < self._min_mask_overlap_ratio:
            soft_reasons.append('overlap_ratio')
        if depth_mask_retention is None:
            soft_reasons.append('depth_unavailable')
        elif depth_mask_retention < self._min_soft_depth_mask_retention:
            soft_reasons.append('depth_retention')
        if not centroid_continuity_validated:
            soft_reasons.append('centroid_history_unavailable')
        if self._soft_continuity_frames >= self._max_soft_continuity_frames:
            soft_reasons.append('soft_frame_budget')
        if soft_reasons:
            self._raise_mask_continuity_failure(metrics, soft_reasons)
        self._soft_continuity_frames += 1

    def _motion_reanchor_is_valid(
        self,
        metrics: dict[str, float],
        *,
        depth_mask_retention: float | None,
        centroid_continuity_validated: bool,
    ) -> bool:
        """Accept a bounded image-plane jump backed by live 3-D evidence."""
        if not self._allow_motion_reanchor or not centroid_continuity_validated:
            return False
        if (
            depth_mask_retention is None
            or depth_mask_retention < self._min_soft_depth_mask_retention
            or metrics['area_ratio'] < self._min_motion_reanchor_area_ratio
        ):
            return False
        return (
            metrics['centroid_shift_ratio']
            <= self._max_motion_reanchor_displacement_ratio
            and metrics['bbox_shift_ratio']
            <= self._max_motion_reanchor_displacement_ratio
        )

    def _current_continuity_metrics(
        self,
        current_mask: np.ndarray,
    ) -> dict[str, float] | None:
        if self._last_mask is None:
            return None
        return self._mask_continuity_metrics(
            self._continuity_reference_mask(),
            current_mask,
        )

    def _continuity_reference_mask(self) -> np.ndarray:
        if self._last_mask is None:
            raise TrackerFailure('validated mask anchor is unavailable')
        return (
            self._continuity_anchor_mask
            if self._soft_continuity_frames > 0
            and self._continuity_anchor_mask is not None
            else self._last_mask
        )

    def _is_contained_scale_collapse(
        self,
        current_mask: np.ndarray,
    ) -> bool:
        """Recognize only a strict subset collapse, never a jump or expansion."""
        if self._last_mask is None:
            return False
        previous = np.asarray(self._last_mask, dtype=bool)
        current = np.asarray(current_mask, dtype=bool)
        previous_area = int(np.count_nonzero(previous))
        current_area = int(np.count_nonzero(current))
        return (
            float(current_area) / float(previous_area)
            < self._min_mask_area_ratio
            and current_area < previous_area
            and not np.any(current & ~previous)
        )

    def _hold_contained_collapse(
        self,
        stamp_ns: int,
        diagnostics: MaskCleanupDiagnostics,
    ) -> None:
        """Advance the service timeline while retaining the validated anchor."""
        if not self._pending_mask_anomaly:
            self._pending_mask_anomaly = True
            self._pending_anomaly_followup_frames = 0
        else:
            self._pending_anomaly_followup_frames += 1
            if (
                self._pending_anomaly_followup_frames
                >= self._max_contained_collapse_recovery_frames
            ):
                raise TrackerFailure(
                    'EdgeTAM mask scale collapse persisted through the bounded '
                    'recovery window '
                    f'(raw {diagnostics.raw_pixels}, cleaned '
                    f'{diagnostics.cleaned_pixels}, components '
                    f'{diagnostics.component_count})',
                    reason_code='mask_scale_collapse',
                )
        self._pending_recovery_streak = 0
        self._pending_recovery_mask = None
        self._commit_pending_image_update(stamp_ns, diagnostics)

    def _advance_pending_recovery(
        self,
        current_mask: np.ndarray,
        metrics: dict[str, float] | None,
    ) -> bool:
        """Require two bounded consecutive source frames before recommitting."""
        if not self._pending_mask_anomaly:
            return True
        if metrics is None or self._last_mask is None:
            raise TrackerFailure('pending mask anomaly lost its validated anchor')
        self._pending_anomaly_followup_frames += 1
        previous_metrics = self._mask_continuity_metrics(
            self._last_mask,
            current_mask,
        )
        self._reject_hard_mask_discontinuity(previous_metrics)
        nominal = previous_metrics['iou'] >= self._min_mask_iou
        if nominal and self._pending_recovery_mask is not None:
            pair_metrics = self._mask_continuity_metrics(
                self._pending_recovery_mask,
                current_mask,
            )
            self._reject_hard_mask_discontinuity(pair_metrics)
            nominal = pair_metrics['iou'] >= self._min_mask_iou

        if nominal:
            self._pending_recovery_streak += 1
            self._pending_recovery_mask = current_mask.copy()
        else:
            self._pending_recovery_streak = 0
            self._pending_recovery_mask = None

        required_streak = min(
            2,
            self._max_contained_collapse_recovery_frames,
        )
        if self._pending_recovery_streak >= required_streak:
            return True
        if (
            self._pending_anomaly_followup_frames
            >= self._max_contained_collapse_recovery_frames
        ):
            raise TrackerFailure(
                'EdgeTAM mask continuity did not recover in consecutive source '
                'frames after a contained collapse',
                reason_code='mask_scale_collapse',
            )
        return False

    def _reject_hard_mask_discontinuity(
        self,
        metrics: dict[str, float],
    ) -> None:
        if metrics['area_ratio'] < self._min_mask_area_ratio:
            expanded = metrics['current_area_ratio'] > 1.0
            direction = 'expanded' if expanded else 'collapsed'
            raise TrackerFailure(
                f'EdgeTAM mask scale {direction} '
                f"(area ratio {metrics['area_ratio']:.3f}, "
                f"IoU {metrics['iou']:.3f})",
                reason_code=(
                    'mask_scale_expansion' if expanded else 'mask_scale_collapse'
                ),
            )

        hard_reasons: list[str] = []
        if metrics['iou'] < self._hard_min_mask_iou:
            hard_reasons.append('hard_iou')
        if metrics['bbox_iou'] < self._min_mask_bbox_iou:
            hard_reasons.append('bbox_iou')
        if (
            metrics['centroid_shift_ratio'] > self._max_mask_displacement_ratio
            or metrics['bbox_shift_ratio'] > self._max_mask_displacement_ratio
        ):
            hard_reasons.append('displacement')
        if hard_reasons:
            self._raise_mask_continuity_failure(metrics, hard_reasons)

    @staticmethod
    def _raise_mask_continuity_failure(
        metrics: dict[str, float],
        reasons: Sequence[str],
    ) -> None:
        reason = ','.join(reasons)
        raise TrackerFailure(
            'EdgeTAM mask continuity broke '
            f"(IoU {metrics['iou']:.3f}, overlap {metrics['overlap_ratio']:.3f}, "
            f"area {metrics['area_ratio']:.3f}, "
            f"centroid shift {metrics['centroid_shift_ratio']:.3f}, "
            f"bbox IoU {metrics['bbox_iou']:.3f}, "
            f"bbox shift {metrics['bbox_shift_ratio']:.3f}, reason {reason})",
            reason_code='mask_continuity',
        )

    @staticmethod
    def _mask_continuity_metrics(
        previous_mask: np.ndarray,
        current_mask: np.ndarray,
    ) -> dict[str, float]:
        previous = np.asarray(previous_mask, dtype=bool)
        current = np.asarray(current_mask, dtype=bool)
        if previous.shape != current.shape or previous.ndim != 2:
            raise TrackerFailure('EdgeTAM mask dimensions changed')
        previous_rows, previous_cols = np.nonzero(previous)
        current_rows, current_cols = np.nonzero(current)
        if previous_rows.size == 0 or current_rows.size == 0:
            raise TrackerFailure('EdgeTAM mask is empty')

        previous_area = int(previous_rows.size)
        current_area = int(current_rows.size)
        intersection = int(np.count_nonzero(previous & current))
        union = previous_area + current_area - intersection
        previous_bbox = (
            int(previous_cols.min()),
            int(previous_rows.min()),
            int(previous_cols.max()) + 1,
            int(previous_rows.max()) + 1,
        )
        current_bbox = (
            int(current_cols.min()),
            int(current_rows.min()),
            int(current_cols.max()) + 1,
            int(current_rows.max()) + 1,
        )
        px1, py1, px2, py2 = previous_bbox
        cx1, cy1, cx2, cy2 = current_bbox
        bbox_intersection = max(0, min(px2, cx2) - max(px1, cx1)) * max(
            0,
            min(py2, cy2) - max(py1, cy1),
        )
        previous_bbox_area = (px2 - px1) * (py2 - py1)
        current_bbox_area = (cx2 - cx1) * (cy2 - cy1)
        bbox_union = previous_bbox_area + current_bbox_area - bbox_intersection
        scale = max(
            math.hypot(px2 - px1, py2 - py1),
            math.hypot(cx2 - cx1, cy2 - cy1),
            1.0,
        )
        centroid_shift = math.hypot(
            float(current_cols.mean() - previous_cols.mean()),
            float(current_rows.mean() - previous_rows.mean()),
        )
        bbox_shift = math.hypot(
            0.5 * float(cx1 + cx2 - px1 - px2),
            0.5 * float(cy1 + cy2 - py1 - py2),
        )
        return {
            'iou': float(intersection) / float(union),
            'overlap_ratio': float(intersection) / float(
                min(previous_area, current_area),
            ),
            'area_ratio': float(min(previous_area, current_area)) / float(
                max(previous_area, current_area),
            ),
            'current_area_ratio': float(current_area) / float(previous_area),
            'bbox_iou': float(bbox_intersection) / float(bbox_union),
            'centroid_shift_ratio': centroid_shift / scale,
            'bbox_shift_ratio': bbox_shift / scale,
        }

    def _commit_service_update(self, stamp_ns: int) -> None:
        self._last_stamp_ns = int(stamp_ns)
        self._next_frame_seq += 1

    def _commit_pending_image_update(
        self,
        stamp_ns: int,
        diagnostics: MaskCleanupDiagnostics,
    ) -> None:
        self._commit_service_update(stamp_ns)
        self._last_raw_mask_diagnostics = diagnostics

    def _commit_image_update(
        self,
        stamp_ns: int,
        current_mask: np.ndarray,
        diagnostics: MaskCleanupDiagnostics,
    ) -> None:
        self._commit_service_update(stamp_ns)
        self._last_mask = current_mask.copy()
        self._last_validated_stamp_ns = int(stamp_ns)
        self._last_mask_diagnostics = diagnostics
        self._last_raw_mask_diagnostics = diagnostics
        self._pending_mask_anomaly = False
        self._pending_anomaly_followup_frames = 0
        self._pending_recovery_streak = 0
        self._pending_recovery_mask = None
        self._coast_streak = 0

    def _raise_update_failure(self, error: Exception) -> None:
        self._clear()
        self._best_effort_reset()
        if isinstance(error, TrackerFailure):
            raise error
        reason_code = str(
            getattr(error, 'reason_code', 'service_update_failed'),
        ).strip() or 'service_update_failed'
        raise TrackerFailure(
            f'EdgeTAM update failed ({type(error).__name__})',
            reason_code=reason_code,
        ) from error

    def reset(self) -> None:
        """Fail closed locally before asking the external service to reset."""
        was_active = self.active or bool(getattr(self._client, 'active', False))
        self._clear()
        if was_active:
            self._best_effort_reset()

    def _best_effort_reset(self) -> None:
        try:
            self._client.reset()
        except Exception:
            pass

    def _clear(self) -> None:
        self._session_id: str | None = None
        self._track_id: str | None = None
        self._label = ''
        self._width = 0
        self._height = 0
        self._last_stamp_ns: int | None = None
        self._last_validated_stamp_ns: int | None = None
        self._next_frame_seq = 0
        self._last_mask: np.ndarray | None = None
        self._last_mask_diagnostics: MaskCleanupDiagnostics | None = None
        self._last_raw_mask_diagnostics: MaskCleanupDiagnostics | None = None
        self._continuity_anchor_mask: np.ndarray | None = None
        self._soft_continuity_frames = 0
        self._last_centroid: np.ndarray | None = None
        self._seed_bbox: tuple[int, int, int, int] | None = None
        self._pending_mask_anomaly = False
        self._pending_anomaly_followup_frames = 0
        self._pending_recovery_streak = 0
        self._pending_recovery_mask: np.ndarray | None = None
        self._coast_streak = 0

    def _bounded_mask(
        self,
        mask: object,
        seed_bbox: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, MaskCleanupDiagnostics]:
        """Validate the raw initialization mask before bounded speck cleanup."""
        raw = np.asarray(mask, dtype=bool)
        if raw.shape != (height, width):
            raise TrackerFailure('EdgeTAM mask dimensions changed')
        raw_area_ratio = float(np.count_nonzero(raw)) / float(width * height)
        if raw_area_ratio > self._max_mask_area_ratio:
            raise TrackerFailure(
                f'EdgeTAM raw initialization mask is unsafe '
                f'({raw_area_ratio:.3f} of image)',
                reason_code='mask_area_limit',
            )
        reference = np.zeros((height, width), dtype=bool)
        if seed_bbox is not None:
            x1, y1, x2, y2 = seed_bbox
            reference[y1:y2, x1:x2] = True
        current, diagnostics = _clean_identity_component(raw, reference)
        self._validate_component_cleanup(diagnostics)
        if seed_bbox is not None and diagnostics.selected_overlap_pixels < 1:
            raise TrackerFailure(
                'EdgeTAM initialization mask does not overlap its exact seed box',
                reason_code='mask_component_identity',
            )
        return current.copy(), diagnostics

    @staticmethod
    def _validate_bbox(
        bbox_xyxy: Sequence[int],
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
            raise TrackerFailure('initialization bbox must contain four pixels')
        if any(isinstance(value, bool) or not isinstance(value, int) for value in bbox_xyxy):
            raise TrackerFailure('initialization bbox must use integer pixels')
        x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise TrackerFailure('initialization bbox is empty or outside the image')
        return x1, y1, x2, y2

    @staticmethod
    def _validate_result(
        result: TrackResult,
        *,
        width: int,
        height: int,
        frame_seq: int,
    ) -> None:
        if result.frame_seq != frame_seq:
            raise TrackerFailure('EdgeTAM result sequence changed')
        if tuple(result.image_size) != (width, height):
            raise TrackerFailure('EdgeTAM result image dimensions changed')
        if not result.session_id or not result.track_id:
            raise TrackerFailure('EdgeTAM returned an empty identity')
        score = float(result.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise TrackerFailure('EdgeTAM returned an invalid confidence')
        mask = np.asarray(result.mask)
        if mask.shape != (height, width):
            raise TrackerFailure('EdgeTAM mask dimensions changed')
        x1, y1, x2, y2 = FailClosedTracker._validate_bbox(
            tuple(result.bbox_xyxy),
            width,
            height,
        )
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            raise TrackerFailure('EdgeTAM mask is empty')
        actual = (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()) + 1,
            int(rows.max()) + 1,
        )
        if actual != (x1, y1, x2, y2):
            raise TrackerFailure('EdgeTAM bbox is not the exact half-open mask bound')
