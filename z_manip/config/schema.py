"""Typed schema for the strict external deployment configuration.

Frozen dataclasses describing each stack-config section, including each
class's own single-object structural invariants (topics start with ``/``,
tool axes are finite orthonormal unit vectors, etc.). Cross-section checks
that span multiple independently-authored sections live in
:mod:`z_manip.config.validation`; JSON loading and orchestration live in
:mod:`z_manip.config.loading`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from z_manip.control.approach import TwoStageApproachConfig
from z_manip.control.visual_servo import VisualServoConfig
from z_manip.kinematics.robust_ik import IKConfig
from z_manip.orchestration.mobile_manipulation import RetryBudget
from z_manip.planning.grasp_pipeline import GraspPlanConfig
from z_manip.planning.rrt_connect import RRTConnectConfig
from z_manip.planning.standoff import ReachabilityStandoffConfig
from z_manip.planning.time_parameterization import TimeParameterizationConfig
from z_manip.planning.work_pose import WorkPoseConfig


@dataclass(frozen=True)
class RobotModelConfig:
    urdf_path: Path
    platform_base_frame: str
    mount_parent_link: str
    base_link: str
    tip_link: str
    acceleration_limits: tuple[float, ...]


@dataclass(frozen=True)
class TopicConfig:
    color: str
    camera_info: str
    aligned_depth: str
    joint_state: str
    arm_trajectory: str
    arm_trajectory_status: str
    gripper_aperture: str
    local_velocity: str

    def __post_init__(self) -> None:
        if any(not value.startswith("/") for value in self.__dict__.values()):
            raise ValueError("all ROS topic names must be absolute")


@dataclass(frozen=True)
class ToolGeometryConfig:
    """Measured tip-frame geometry shared by sim and real adapters.

    The axes and contact interval are expressed in ``robot.tip_link``.  The
    interval scalar is measured along ``tip_approach_axis``; the historical
    ``_z_`` field name is retained in schema v2 to keep the public setting
    unambiguous for existing deployments that prepared the migration.
    """

    tip_closing_axis: tuple[float, float, float]
    tip_approach_axis: tuple[float, float, float]
    finger_contact_z_interval_m: tuple[float, float]
    contact_tcp_z_m: float
    collision_open_aperture_m: float
    collision_grasp_margin_m: float

    def __post_init__(self) -> None:
        axes = (self.tip_closing_axis, self.tip_approach_axis)
        if any(
            len(axis) != 3 or not all(math.isfinite(value) for value in axis)
            for axis in axes
        ):
            raise ValueError("tool axes must be finite three-vectors")
        norms = tuple(math.sqrt(sum(value * value for value in axis)) for axis in axes)
        if any(not math.isclose(norm, 1.0, abs_tol=1e-6) for norm in norms):
            raise ValueError("tool axes must be unit vectors")
        dot = sum(first * second for first, second in zip(*axes))
        if not math.isclose(dot, 0.0, abs_tol=1e-6):
            raise ValueError("tool closing and approach axes must be orthogonal")
        interval = self.finger_contact_z_interval_m
        if (
            len(interval) != 2
            or not all(math.isfinite(value) for value in interval)
            or not 0.0 <= interval[0] < interval[1]
        ):
            raise ValueError("finger contact interval must be finite and increasing")
        if (
            not math.isfinite(self.contact_tcp_z_m)
            or not interval[0] < self.contact_tcp_z_m < interval[1]
        ):
            raise ValueError("contact TCP must lie inside the finger contact interval")
        if (
            not math.isfinite(self.collision_open_aperture_m)
            or self.collision_open_aperture_m <= 0.0
        ):
            raise ValueError("collision aperture must be finite and positive")
        if (
            not math.isfinite(self.collision_grasp_margin_m)
            or self.collision_grasp_margin_m < 0.0
            or self.collision_grasp_margin_m >= self.collision_open_aperture_m
        ):
            raise ValueError(
                "collision grasp margin must be finite, non-negative, and below open aperture",
            )


@dataclass(frozen=True)
class StackConfig:
    schema_version: int
    robot: RobotModelConfig
    tool_geometry: ToolGeometryConfig
    topics: TopicConfig
    visual_servo: VisualServoConfig
    approach: TwoStageApproachConfig
    ik: IKConfig
    rrt: RRTConnectConfig
    standoff: ReachabilityStandoffConfig
    work_pose: WorkPoseConfig
    grasp_plan: GraspPlanConfig
    time_parameterization: TimeParameterizationConfig
    retry_budget: RetryBudget
    collision_model_path: Path
    vlm_models: tuple[str, ...]


__all__ = [
    "RobotModelConfig",
    "StackConfig",
    "TopicConfig",
    "ToolGeometryConfig",
]
