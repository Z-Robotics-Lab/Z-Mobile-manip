"""Collision-aware arm motion planning and grasp-program generation."""

from .grasp_pipeline import GraspPlanConfig, GraspPlanGenerator, PlannedGrasp
from .placement import (
    NormalizedPlacementRegion,
    ObservedPlacementConfig,
    ObservedPlacementInput,
    ObservedPlacementPlanner,
    PlacementCandidate,
    PlacementConstraints,
    PlacementMotionEvaluation,
    PlannedPlacement,
    SupportPlane,
)
from .rrt_connect import JointSpaceRRTConnect, RRTConnectConfig
from .standoff import (
    ReachabilityStandoffConfig,
    ReachabilityStandoffOptimizer,
    StandoffChoice,
)
from .time_parameterization import (
    TimedJointTrajectory,
    TimeParameterizationConfig,
    retime_path,
)
from .staged_trajectory import (
    GraspStage,
    GraspTrajectorySegment,
    GraspTrajectoryTarget,
    SidePreference,
    StagedGraspRequest,
    StagedGraspTrajectory,
    StagedGraspTrajectoryBuilder,
)
from .work_pose import (
    BoundedSE2WorkPoseOptimizer,
    WorkPoseCandidate,
    WorkPoseChoice,
    WorkPoseConfig,
    WorkPoseDiagnostics,
    WorkPoseFailure,
    WorkPoseFailureCode,
    WorkPoseObservation,
    WorkPoseOptimizationError,
)

__all__ = [
    "GraspCompletionProgram",
    "GraspPlanConfig",
    "GraspPlanGenerator",
    "GraspStage",
    "GraspTrajectorySegment",
    "GraspTrajectoryTarget",
    "JointSpaceRRTConnect",
    "BoundedSE2WorkPoseOptimizer",
    "MotionProgram",
    "NormalizedPlacementRegion",
    "OnlinePlanner",
    "ObservedPlacementConfig",
    "ObservedPlacementInput",
    "ObservedPlacementPlanner",
    "PerceptionObservation",
    "PlacementCandidate",
    "PlacementConstraints",
    "PlacementMotionEvaluation",
    "PlannedGrasp",
    "PlannedPlacement",
    "PregraspTransitProgram",
    "ProspectiveWorkPose",
    "RRTConnectConfig",
    "ReachabilityStandoffConfig",
    "ReachabilityStandoffOptimizer",
    "SemanticPointSelection",
    "StandoffChoice",
    "SidePreference",
    "StagedGraspRequest",
    "StagedGraspTrajectory",
    "StagedGraspTrajectoryBuilder",
    "SupportPlane",
    "TimedJointTrajectory",
    "TimeParameterizationConfig",
    "WorkPoseCandidate",
    "WorkPoseChoice",
    "WorkPoseConfig",
    "WorkPoseDiagnostics",
    "WorkPoseFailure",
    "WorkPoseFailureCode",
    "WorkPoseObservation",
    "WorkPoseOptimizationError",
    "retime_path",
    "select_semantic_target_points",
]

_ONLINE_PLANNER_EXPORTS = frozenset({
    "GraspCompletionProgram",
    "MotionProgram",
    "OnlinePlanner",
    "PerceptionObservation",
    "PregraspTransitProgram",
    "ProspectiveWorkPose",
    "SemanticPointSelection",
    "select_semantic_target_points",
})


def __getattr__(name: str) -> object:
    # Resolved on demand because online_planner imports z_manip.configuration,
    # which imports this package: an eager import here would be circular.
    if name in _ONLINE_PLANNER_EXPORTS:
        from . import online_planner

        return getattr(online_planner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
