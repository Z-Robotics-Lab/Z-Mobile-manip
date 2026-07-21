"""Collision checking against perceived geometry."""

from .contact_phase import (
    SegmentCollisionChecker,
    TargetContactApproachResult,
    check_target_contact_approach,
)
from .gripper_aperture import (
    collision_aperture_for_grasp,
    with_parallel_gripper_aperture,
)
from .pointcloud import (
    CapsuleSpec,
    CollisionResult,
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
    SegmentCollisionResult,
    SelfCollisionConfig,
)

__all__ = [
    "CapsuleSpec",
    "CollisionResult",
    "PointCloudCollisionChecker",
    "PointCloudCollisionConfig",
    "RobotCollisionModel",
    "SegmentCollisionResult",
    "SegmentCollisionChecker",
    "SelfCollisionConfig",
    "TargetContactApproachResult",
    "check_target_contact_approach",
    "collision_aperture_for_grasp",
    "with_parallel_gripper_aperture",
]
