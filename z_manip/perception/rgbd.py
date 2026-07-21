"""RGB-D target geometry derived only from camera observations.

The functions in this module deliberately know nothing about ROS, Isaac, or an
object database. A text-conditioned detector supplies one image box, aligned
depth supplies metric geometry, and the lightweight tracker keeps the same
appearance/depth cluster locked between detector calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence
import warnings

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")


@dataclass(frozen=True)
class BoundingBox:
    """Half-open image box: ``[x1, x2) x [y1, y2)``."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def clamp(self, width: int, height: int) -> "BoundingBox":
        x1 = min(max(int(self.x1), 0), width)
        y1 = min(max(int(self.y1), 0), height)
        x2 = min(max(int(self.x2), x1), width)
        y2 = min(max(int(self.y2), y1), height)
        return BoundingBox(x1, y1, x2, y2)

    def expanded(self, scale: float, width: int, height: int) -> "BoundingBox":
        if scale < 1.0:
            raise ValueError("expansion scale must be >= 1")
        cx = 0.5 * (self.x1 + self.x2)
        cy = 0.5 * (self.y1 + self.y2)
        half_w = max(2.0, 0.5 * self.width * scale)
        half_h = max(2.0, 0.5 * self.height * scale)
        return BoundingBox(
            int(np.floor(cx - half_w)),
            int(np.floor(cy - half_h)),
            int(np.ceil(cx + half_w)),
            int(np.ceil(cy + half_h)),
        ).clamp(width, height)


@dataclass(frozen=True, eq=False)
class TargetObservation:
    label: str
    bbox: BoundingBox
    position_camera: tuple[float, float, float]
    valid_points: int
    stamp_s: float
    score: float = 1.0


def _validate_rgbd_shapes(image: np.ndarray, depth_mm: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image.shape}")
    if depth_mm.ndim != 2 or depth_mm.shape != image.shape[:2]:
        raise ValueError(
            f"aligned depth shape {depth_mm.shape} does not match RGB {image.shape[:2]}"
        )


def depth_bbox_observation(
    depth_mm: np.ndarray,
    bbox: BoundingBox,
    intrinsics: CameraIntrinsics,
    *,
    label: str,
    stamp_s: float,
    mask: Optional[np.ndarray] = None,
    min_points: int = 20,
) -> TargetObservation:
    """Back-project the robust median of valid target pixels into optical XYZ."""
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError(f"expected HxW depth image, got {depth.shape}")
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth dimensions do not match CameraInfo")
    box = bbox.clamp(intrinsics.width, intrinsics.height)
    if box.area == 0:
        raise ValueError("empty target bounding box")

    roi = depth[box.y1:box.y2, box.x1:box.x2].astype(np.float64)
    valid = np.isfinite(roi) & (roi > 0.0)
    if mask is not None:
        target_mask = np.asarray(mask, dtype=bool)
        if target_mask.shape == depth.shape:
            target_mask = target_mask[box.y1:box.y2, box.x1:box.x2]
        if target_mask.shape != roi.shape:
            raise ValueError("target mask must match the depth image or box ROI")
        valid &= target_mask
    ys, xs = np.nonzero(valid)
    if len(xs) < min_points:
        raise ValueError(f"only {len(xs)} valid target depth pixels; need {min_points}")

    depths_m = roi[valid] * 0.001
    median = float(np.median(depths_m))
    # Reject background leakage and flying pixels before taking the image centre.
    mad = float(np.median(np.abs(depths_m - median)))
    band = max(0.015, 3.5 * mad)
    keep = np.abs(depths_m - median) <= band
    xs = xs[keep] + box.x1
    ys = ys[keep] + box.y1
    depths_m = depths_m[keep]
    if len(xs) < min_points:
        raise ValueError("target depth cluster vanished after robust filtering")

    z = float(np.median(depths_m))
    u = float(np.median(xs))
    v = float(np.median(ys))
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return TargetObservation(
        label=label,
        bbox=box,
        position_camera=(x, y, z),
        valid_points=int(len(xs)),
        stamp_s=float(stamp_s),
    )


def depth_to_pointcloud(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    mask: Optional[np.ndarray] = None,
    stride: int = 1,
    min_depth_m: float = 0.28,
    max_depth_m: float = 5.0,
    transform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Back-project aligned 16UC1 depth to an observed metric point cloud."""

    depth = np.asarray(depth_mm)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth dimensions do not match CameraInfo")
    if stride < 1:
        raise ValueError("point-cloud stride must be positive")
    if not 0.0 <= min_depth_m < max_depth_m:
        raise ValueError("invalid point-cloud depth interval")
    target_mask = None
    if mask is not None:
        target_mask = np.asarray(mask, dtype=bool)
        if target_mask.shape != depth.shape:
            raise ValueError("point-cloud mask must match the depth image")

    rows = np.arange(0, intrinsics.height, stride)
    columns = np.arange(0, intrinsics.width, stride)
    u, v = np.meshgrid(columns, rows)
    sampled = depth[np.ix_(rows, columns)].astype(np.float64) * 0.001
    valid = (
        np.isfinite(sampled)
        & (sampled >= min_depth_m)
        & (sampled <= max_depth_m)
    )
    if target_mask is not None:
        valid &= target_mask[np.ix_(rows, columns)]
    z = sampled[valid]
    x = (u[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (v[valid] - intrinsics.cy) * z / intrinsics.fy
    points = np.column_stack((x, y, z))
    if transform is not None:
        target_from_camera = np.asarray(transform, dtype=float)
        if (
            target_from_camera.shape != (4, 4)
            or not np.all(np.isfinite(target_from_camera))
            or not np.allclose(
                target_from_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7,
            )
        ):
            raise ValueError("point-cloud transform must be a finite homogeneous matrix")
        points = (
            points @ target_from_camera[:3, :3].T
            + target_from_camera[:3, 3]
        )
    return points.astype(np.float32, copy=False)


def depth_to_scene_cloud(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    target_mask: np.ndarray,
    target_dilation_px: int = 2,
    stride: int = 2,
    min_depth_m: float = 0.28,
    max_depth_m: float = 5.0,
    transform: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a scene while preserving the tracked-target exclusion label.

    Keeping labels in pixel space avoids an unsafe nearest-neighbour guess at
    object silhouettes. Mask-edge depth may belong to the background, but a
    small configurable dilation intentionally reserves that contact corridor;
    the remaining shelf and neighbouring-object geometry stays in the scene.
    """

    depth = np.asarray(depth_mm)
    labels = np.asarray(target_mask, dtype=bool)
    expected = (intrinsics.height, intrinsics.width)
    if depth.shape != expected or labels.shape != expected:
        raise ValueError("scene depth and target mask must match CameraInfo")
    if target_dilation_px < 0:
        raise ValueError("target mask dilation cannot be negative")
    if target_dilation_px:
        size = 2 * int(target_dilation_px) + 1
        labels = binary_dilation(labels, structure=np.ones((size, size), dtype=bool))
    points = depth_to_pointcloud(
        depth,
        intrinsics,
        stride=stride,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        transform=transform,
    )
    rows = np.arange(0, intrinsics.height, stride)
    columns = np.arange(0, intrinsics.width, stride)
    sampled_depth = depth[np.ix_(rows, columns)].astype(np.float64) * 0.001
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= min_depth_m)
        & (sampled_depth <= max_depth_m)
    )
    aligned_labels = labels[np.ix_(rows, columns)][valid]
    if len(aligned_labels) != len(points):
        raise RuntimeError("internal RGB-D label alignment failure")
    return points, aligned_labels.astype(bool, copy=False)


def temporal_median_depth(
    frames_mm: object,
    *,
    min_valid_fraction: float = 0.6,
    max_mad_mm: float = 8.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fuse a short stationary depth burst and remove temporally unstable pixels.

    Zero depth is treated as missing. A pixel is retained only when enough input
    frames observe it and its median absolute deviation stays below the explicit
    millimetre threshold. This intentionally creates holes at flickering object
    silhouettes instead of turning depth noise into planning-scene obstacles.
    """

    values = np.asarray(frames_mm)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("temporal depth input must contain at least three 2-D frames")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("minimum valid depth fraction must be in (0, 1]")
    if not np.isfinite(max_mad_mm) or max_mad_mm <= 0.0:
        raise ValueError("maximum temporal depth MAD must be finite and positive")

    numeric = values.astype(np.float64, copy=False)
    observed = np.isfinite(numeric) & (numeric > 0.0)
    samples = np.where(observed, numeric, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(samples, axis=0)
        mad = np.nanmedian(np.abs(samples - median[None, ...]), axis=0)
    minimum = max(2, int(math.ceil(values.shape[0] * min_valid_fraction)))
    counts = np.count_nonzero(observed, axis=0)
    stable = (
        (counts >= minimum)
        & np.isfinite(median)
        & np.isfinite(mad)
        & (mad <= max_mad_mm)
    )
    filtered = np.zeros(values.shape[1:], dtype=np.uint16)
    filtered[stable] = np.clip(
        np.rint(median[stable]),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    finite_mad = mad[np.isfinite(mad)]
    diagnostics: dict[str, float | int] = {
        "frame_count": int(values.shape[0]),
        "minimum_observations": minimum,
        "stable_pixels": int(np.count_nonzero(stable)),
        "rejected_low_support_pixels": int(np.count_nonzero(counts < minimum)),
        "rejected_unstable_pixels": int(
            np.count_nonzero((counts >= minimum) & np.isfinite(mad) & (mad > max_mad_mm))
        ),
        "mad_p95_mm": (
            0.0 if not finite_mad.size else float(np.percentile(finite_mad, 95))
        ),
        "max_mad_mm": float(max_mad_mm),
    }
    return filtered, diagnostics


def target_exclusion_mask(
    scene_points: object,
    target_points: object,
    *,
    radius_m: float = 0.012,
    min_target_points: int = 20,
) -> np.ndarray:
    """Mark scene samples belonging to the tracked target cloud.

    EdgeTAM publishes the persistent target cloud even when its internal mask is
    not exposed. A nearest-neighbour radius maps that cloud back onto the full
    RGB-D scene, allowing the collision checker to permit intended gripper
    contact while retaining shelf and neighbouring-object points.
    """

    scene = np.asarray(scene_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if scene.ndim != 2 or scene.shape[1:] != (3,) or not np.all(np.isfinite(scene)):
        raise ValueError("scene cloud must be a finite (N, 3) array")
    if target.ndim != 2 or target.shape[1:] != (3,):
        raise ValueError("target cloud must have shape (N, 3)")
    target = target[np.all(np.isfinite(target), axis=1)]
    if len(target) < min_target_points:
        raise ValueError(
            f"target cloud has {len(target)} finite points; need {min_target_points}",
        )
    if radius_m <= 0.0:
        raise ValueError("target exclusion radius must be positive")
    distance, _ = cKDTree(target).query(scene, k=1, distance_upper_bound=radius_m)
    return np.isfinite(distance)


def filter_object_cloud(
    points: object,
    *,
    viewpoint: Sequence[float] = (0.0, 0.0, 0.0),
    radial_mad_scale: float = 4.0,
    min_radial_band_m: float = 0.025,
    neighbour_count: int = 12,
    neighbour_mad_scale: float = 4.0,
    min_points: int = 40,
) -> np.ndarray:
    """Remove mask-edge background and isolated depth fliers robustly.

    Segmentation boundaries often contain a few pixels from the shelf behind an
    object. The first gate keeps the dominant camera-range layer; the second
    rejects points whose local k-neighbour spacing is anomalous. No object size,
    class, or scene coordinate is assumed.
    """

    cloud = np.asarray(points, dtype=float)
    origin = np.asarray(viewpoint, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,):
        raise ValueError("object cloud must have shape (N, 3)")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("object-cloud viewpoint must be a finite three-vector")
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if len(cloud) < min_points:
        raise ValueError(f"object cloud has only {len(cloud)} finite points")
    ranges = np.linalg.norm(cloud - origin, axis=1)
    median_range = float(np.median(ranges))
    radial_mad = float(np.median(np.abs(ranges - median_range)))
    radial_band = max(min_radial_band_m, radial_mad_scale * radial_mad)
    layered = cloud[np.abs(ranges - median_range) <= radial_band]
    if len(layered) < min_points:
        raise ValueError("object cloud lost its dominant depth layer")

    k = min(max(3, int(neighbour_count)), len(layered))
    distances, _ = cKDTree(layered).query(layered, k=k)
    local_spacing = np.mean(distances[:, 1:], axis=1)
    spacing_median = float(np.median(local_spacing))
    spacing_mad = float(np.median(np.abs(local_spacing - spacing_median)))
    spacing_limit = spacing_median + max(1e-4, neighbour_mad_scale * spacing_mad)
    filtered = layered[local_spacing <= spacing_limit]
    if len(filtered) < min_points:
        raise ValueError("object cloud has too few locally supported points")
    return filtered.astype(np.float32, copy=False)


class ColorDepthTracker:
    """Deterministic short-horizon tracker for servoing between VLM detections.

    It matches a robust RGB prototype inside an expanded previous box and gates
    it by aligned depth. It intentionally fails closed when too few pixels
    survive; callers then stop the base and request a fresh detector result.
    """

    def __init__(
        self,
        *,
        color_tolerance: float = 70.0,
        depth_tolerance_mm: int = 450,
        search_scale: float = 2.0,
        min_pixels: int = 24,
    ) -> None:
        self.color_tolerance = float(color_tolerance)
        self.depth_tolerance_mm = int(depth_tolerance_mm)
        self.search_scale = float(search_scale)
        self.min_pixels = int(min_pixels)
        self._bbox: Optional[BoundingBox] = None
        self._color: Optional[np.ndarray] = None
        self._depth_mm: Optional[int] = None

    @property
    def bbox(self) -> Optional[BoundingBox]:
        return self._bbox

    @property
    def depth_mm(self) -> Optional[int]:
        return self._depth_mm

    def initialize(
        self,
        image: np.ndarray,
        depth_mm: np.ndarray,
        bbox: BoundingBox,
    ) -> BoundingBox:
        image = np.asarray(image)
        depth = np.asarray(depth_mm)
        _validate_rgbd_shapes(image, depth)
        box = bbox.clamp(image.shape[1], image.shape[0])
        if box.area == 0:
            raise ValueError("cannot initialize tracker with an empty box")
        inset_x = max(1, int(round(box.width * 0.18)))
        inset_y = max(1, int(round(box.height * 0.18)))
        core = BoundingBox(
            box.x1 + inset_x,
            box.y1 + inset_y,
            box.x2 - inset_x,
            box.y2 - inset_y,
        )
        if core.area <= 0:
            core = box
        rgb = image[core.y1:core.y2, core.x1:core.x2].reshape(-1, 3)
        dep = depth[core.y1:core.y2, core.x1:core.x2].reshape(-1)
        valid_depth = dep > 0
        if int(valid_depth.sum()) < self.min_pixels:
            raise ValueError("not enough valid depth to initialize tracker")
        self._color = np.median(rgb[valid_depth].astype(np.float32), axis=0)
        self._depth_mm = int(np.median(dep[valid_depth]))
        self._bbox = box
        return box

    def update(self, image: np.ndarray, depth_mm: np.ndarray) -> Optional[BoundingBox]:
        if self._bbox is None or self._color is None or self._depth_mm is None:
            raise RuntimeError("tracker must be initialized before update")
        image = np.asarray(image)
        depth = np.asarray(depth_mm)
        _validate_rgbd_shapes(image, depth)
        search = self._bbox.expanded(self.search_scale, image.shape[1], image.shape[0])
        rgb = image[search.y1:search.y2, search.x1:search.x2].astype(np.float32)
        dep = depth[search.y1:search.y2, search.x1:search.x2].astype(np.int32)
        color_distance = np.linalg.norm(rgb - self._color, axis=2)
        mask = (
            (color_distance <= self.color_tolerance)
            & (dep > 0)
            & (np.abs(dep - self._depth_mm) <= self.depth_tolerance_mm)
        )
        ys, xs = np.nonzero(mask)
        if len(xs) < self.min_pixels:
            self._bbox = None
            return None

        x1 = int(np.floor(np.percentile(xs, 2))) + search.x1
        x2 = int(np.ceil(np.percentile(xs, 98))) + 1 + search.x1
        y1 = int(np.floor(np.percentile(ys, 2))) + search.y1
        y2 = int(np.ceil(np.percentile(ys, 98))) + 1 + search.y1
        self._bbox = BoundingBox(x1, y1, x2, y2).clamp(image.shape[1], image.shape[0])
        matched_rgb = rgb[mask]
        matched_depth = dep[mask]
        median_color = np.median(matched_rgb, axis=0)
        self._color = 0.85 * self._color + 0.15 * median_color
        self._depth_mm = int(np.median(matched_depth))
        return self._bbox

    def reset(self) -> None:
        self._bbox = None
        self._color = None
        self._depth_mm = None
