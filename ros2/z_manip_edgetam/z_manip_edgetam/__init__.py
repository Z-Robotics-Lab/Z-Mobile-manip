"""ROS 2 adapter for the standalone EdgeTAM mask service."""

from .core import (
    CameraIntrinsics,
    center_box_to_half_open,
    FailClosedTracker,
    project_mask_depth,
    project_mask_depth_geometry,
    RgbdFrame,
    TrackerFailure,
    TrackingObservation,
)

__all__ = [
    'CameraIntrinsics',
    'FailClosedTracker',
    'RgbdFrame',
    'TrackerFailure',
    'TrackingObservation',
    'center_box_to_half_open',
    'project_mask_depth',
    'project_mask_depth_geometry',
]
