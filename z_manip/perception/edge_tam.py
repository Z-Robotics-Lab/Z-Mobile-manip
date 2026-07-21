"""Fail-closed state contract for an external persistent EdgeTAM tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rgbd import BoundingBox


class TrackingLost(RuntimeError):
    """The target identity or observation is not currently trustworthy."""


@dataclass(frozen=True)
class EdgeTamTopics:
    """Explicit topic names used by the separately deployed EdgeTAM ROS node."""

    init_bbox: str = "/track_3d/init_bbox"
    select_id: str = "/track_3d/select_id"
    reset: str = "/track_3d/reset"
    is_tracking: str = "/track_3d/is_tracking"
    detections_2d: str = "/track_3d/detections_2d"
    selected_detection: str = "/track_3d/selected_target_3d"
    selected_pointcloud: str = "/track_3d/selected_target_pointcloud"
    selected_normal: str = "/track_3d/selected_normal_pose"

    def __post_init__(self) -> None:
        if any(not value.startswith("/") for value in self.__dict__.values()):
            raise ValueError("EdgeTAM topics must be absolute")


@dataclass(frozen=True, eq=False)
class EdgeTamObservation:
    track_id: int
    is_tracking: bool
    bbox: BoundingBox
    object_points: np.ndarray
    frame: str
    stamp_s: float
    score: float


class EdgeTamTrackBuffer:
    """Validate asynchronous tracker outputs before control consumes them."""

    def __init__(self, *, max_age_s: float = 0.35, min_points: int = 24):
        if max_age_s <= 0.0 or min_points < 1:
            raise ValueError("tracker freshness and point thresholds must be positive")
        self.max_age_s = float(max_age_s)
        self.min_points = int(min_points)
        self._requested_label: str | None = None
        self._requested_bbox: BoundingBox | None = None
        self._request_stamp_s: float | None = None
        self._track_id: int | None = None
        self._latest: EdgeTamObservation | None = None

    def begin(self, label: str, bbox: BoundingBox, *, stamp_s: float) -> None:
        if not label.strip() or bbox.area <= 0:
            raise ValueError("tracking request needs a label and non-empty bbox")
        self.reset()
        self._requested_label = label.strip()
        self._requested_bbox = bbox
        self._request_stamp_s = float(stamp_s)

    def ingest(self, observation: EdgeTamObservation) -> EdgeTamObservation:
        if self._requested_bbox is None:
            raise TrackingLost("tracker output arrived without an active request")
        if not observation.is_tracking:
            self._latest = None
            raise TrackingLost("tracker reported loss")
        if observation.stamp_s < float(self._request_stamp_s):
            raise TrackingLost("tracker output predates the current request")
        if observation.bbox.area <= 0 or not observation.frame:
            raise TrackingLost("tracker output has invalid image or frame metadata")
        points = np.asarray(observation.object_points)
        valid_points = points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(axis=1).sum()
        if not valid_points or int(valid_points) < self.min_points:
            self._latest = None
            raise TrackingLost("tracked target point cloud is missing or too thin")
        if self._track_id is None:
            self._track_id = observation.track_id
        elif observation.track_id != self._track_id:
            self._latest = None
            raise TrackingLost(
                f"track identity changed from {self._track_id} to {observation.track_id}",
            )
        self._latest = observation
        return observation

    def latest(self, *, now_s: float) -> EdgeTamObservation:
        if self._latest is None:
            raise TrackingLost("no current tracked target observation")
        age = float(now_s) - self._latest.stamp_s
        if age < -1e-6:
            raise TrackingLost("tracker timestamp is ahead of control time")
        if age > self.max_age_s:
            self._latest = None
            raise TrackingLost(f"tracked target observation is stale by {age:.3f} s")
        return self._latest

    def reset(self) -> None:
        self._requested_label = None
        self._requested_bbox = None
        self._request_stamp_s = None
        self._track_id = None
        self._latest = None
