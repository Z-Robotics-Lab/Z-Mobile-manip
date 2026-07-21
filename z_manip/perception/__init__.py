"""Platform-independent RGB-D perception utilities."""

from .rgbd import (
    BoundingBox,
    CameraIntrinsics,
    ColorDepthTracker,
    TargetObservation,
    depth_bbox_observation,
)

__all__ = [
    "BoundingBox",
    "CameraIntrinsics",
    "ColorDepthTracker",
    "TargetObservation",
    "depth_bbox_observation",
]
