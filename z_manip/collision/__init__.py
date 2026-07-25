"""Collision checking against perceived geometry."""

from .contact_phase import (
    SegmentCollisionChecker,
    TargetContactApproachResult,
    check_target_contact_approach,
)
from .fixed_fixture import (
    CollisionState,
    CollisionWitness,
    FixedCapsule,
    FixedSelfCollisionGuard,
    StepDecision,
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
from .trajectory_clearance import (
    FixedFixtureStateGuard,
    FixedFixtureTrajectoryEvidence,
    SmoothClearancePenalty,
    TrajectoryCollisionWitness,
    TrajectorySegmentClearance,
    evaluate_fixed_fixture_trajectory,
    evaluate_smooth_clearance_penalty,
)

__all__ = [
    "CapsuleSpec",
    "CollisionResult",
    "CollisionState",
    "CollisionWitness",
    "FixedCapsule",
    "FixedFixtureStateGuard",
    "FixedFixtureTrajectoryEvidence",
    "FixedSelfCollisionGuard",
    "PointCloudCollisionChecker",
    "PointCloudCollisionConfig",
    "RobotCollisionModel",
    "SegmentCollisionResult",
    "SegmentCollisionChecker",
    "SelfCollisionConfig",
    "SmoothClearancePenalty",
    "StepDecision",
    "TargetContactApproachResult",
    "TrajectoryCollisionWitness",
    "TrajectorySegmentClearance",
    "check_target_contact_approach",
    "collision_aperture_for_grasp",
    "evaluate_fixed_fixture_trajectory",
    "evaluate_smooth_clearance_penalty",
    "with_parallel_gripper_aperture",
]
