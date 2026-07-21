"""Strict external deployment configuration for the platform-neutral stack."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from xml.etree import ElementTree

import numpy as np

from z_manip.collision import RobotCollisionModel
from z_manip.control.approach import TwoStageApproachConfig
from z_manip.control.visual_servo import VisualServoConfig
from z_manip.kinematics.robust_ik import IKConfig
from z_manip.orchestration.mobile_manipulation import RetryBudget
from z_manip.planning.grasp_pipeline import GraspPlanConfig
from z_manip.planning.rrt_connect import RRTConnectConfig
from z_manip.planning.standoff import ReachabilityStandoffConfig
from z_manip.planning.time_parameterization import TimeParameterizationConfig
from z_manip.planning.work_pose import WorkPoseConfig


_ENVIRONMENT_VALUE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_SCHEMA_VERSION = 2


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


def _resolve_environment(value: object, environ: Mapping[str, str]) -> object:
    if isinstance(value, str):
        match = _ENVIRONMENT_VALUE.fullmatch(value)
        if match:
            name = match.group(1)
            if not environ.get(name):
                raise ValueError(f"required environment variable {name} is not set")
            return environ[name]
        return value
    if isinstance(value, list):
        return [_resolve_environment(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_environment(item, environ) for key, item in value.items()}
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _float_tuple(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(float(item) for item in value)


def _float_matrix(
    value: object,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array of arrays")
    return tuple(_float_tuple(row, f"{label} row") for row in value)


def _load_collision_model(path: Path) -> RobotCollisionModel:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load collision model {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"collision model {path} must contain an object")
    try:
        return RobotCollisionModel.from_mapping(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid collision model {path}: {error}") from error


def _urdf_link_names(path: Path) -> set[str]:
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise ValueError(f"could not parse robot URDF {path}: {error}") from error
    return {
        str(element.attrib["name"])
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "link" and element.attrib.get("name")
    }


def _validate_geometry_contract(
    *,
    robot: RobotModelConfig,
    tool: ToolGeometryConfig,
    grasp: GraspPlanConfig,
    collision: RobotCollisionModel,
) -> None:
    """Cross-check the independently maintained tool and collision settings."""
    tool_from_tip = np.asarray(grasp.tool_from_tip, dtype=float)
    if tool_from_tip.shape != (4, 4) or not np.all(np.isfinite(tool_from_tip)):
        raise ValueError("grasp_plan.tool_from_tip must be a finite 4x4 transform")
    if not np.allclose(tool_from_tip[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("grasp_plan.tool_from_tip must have a homogeneous final row")
    rotation = tool_from_tip[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("grasp_plan.tool_from_tip rotation must be right-handed and orthonormal")

    closing_axis = np.asarray(tool.tip_closing_axis, dtype=float)
    approach_axis = np.asarray(tool.tip_approach_axis, dtype=float)
    if not np.allclose(rotation[:, 0], closing_axis, atol=1e-6):
        raise ValueError(
            "tool_geometry.tip_closing_axis must match the tool_from_tip tool-X axis",
        )
    if not np.allclose(rotation[:, 2], approach_axis, atol=1e-6):
        raise ValueError(
            "tool_geometry.tip_approach_axis must match the tool_from_tip tool-Z axis",
        )
    expected_tcp = approach_axis * tool.contact_tcp_z_m
    if not np.allclose(tool_from_tip[:3, 3], expected_tcp, atol=1e-6):
        raise ValueError(
            "grasp_plan.tool_from_tip translation must match the configured contact TCP",
        )
    if grasp.max_width_m > tool.collision_open_aperture_m + 1e-9:
        raise ValueError(
            "grasp_plan.max_width_m cannot exceed tool_geometry.collision_open_aperture_m",
        )

    contacts = {
        capsule.name: capsule
        for capsule in collision.capsules
        if capsule.name in collision.target_contact_capsules
    }
    if not contacts:
        raise ValueError("collision model must identify target_contact_capsules")

    # Fixed tip-frame proxies can be checked numerically without assuming a
    # robot, joint layout, or capsule name. Dynamic finger-link proxies remain
    # valid, but their open-state transform belongs to the runtime kinematics.
    fixed_contacts = tuple(
        capsule
        for capsule in contacts.values()
        if capsule.start_frame == robot.tip_link and capsule.end_frame == robot.tip_link
    )
    if len(fixed_contacts) != len(contacts):
        return
    interval_min, interval_max = tool.finger_contact_z_interval_m
    intervals_by_side: dict[int, list[tuple[float, float]]] = {-1: [], 1: []}
    for capsule in fixed_contacts:
        start = np.asarray(capsule.start_offset, dtype=float)
        end = np.asarray(capsule.end_offset, dtype=float)
        approach_projection = (float(start @ approach_axis), float(end @ approach_axis))
        proxy_min = min(approach_projection) - capsule.radius
        proxy_max = max(approach_projection) + capsule.radius
        closing_center = float((0.5 * (start + end)) @ closing_axis)
        side = -1 if closing_center < -1e-6 else 1 if closing_center > 1e-6 else 0
        if side:
            intervals_by_side[side].append((proxy_min, proxy_max))
    if not all(intervals_by_side.values()):
        raise ValueError(
            "fixed target-contact capsules must bracket the TCP on the closing axis",
        )
    for side, intervals in intervals_by_side.items():
        ordered = sorted(intervals)
        covered_min, covered_max = ordered[0]
        for next_min, next_max in ordered[1:]:
            if next_min > covered_max + 1e-6:
                raise ValueError(
                    f"target-contact capsules on side {side:+d} leave a gap in "
                    "the configured finger contact interval",
                )
            covered_max = max(covered_max, next_max)
        if covered_min > interval_min + 1e-6 or covered_max < interval_max - 1e-6:
            raise ValueError(
                f"target-contact capsules on side {side:+d} do not cover the "
                "configured finger contact interval",
            )


def _validate_collision_frames(
    robot: RobotModelConfig,
    collision: RobotCollisionModel,
) -> None:
    links = _urdf_link_names(robot.urdf_path)
    referenced = {robot.mount_parent_link, robot.base_link, robot.tip_link}
    for capsule in collision.capsules:
        referenced.update((capsule.start_frame, capsule.end_frame))
    unknown = referenced - links
    if unknown:
        raise ValueError(
            "robot/collision configuration references unknown URDF links: "
            f"{sorted(unknown)}",
        )


def load_stack_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> StackConfig:
    """Load schema v2 without silently accepting unknown constructor fields."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load stack config {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("stack config must contain an object")
    schema_version = raw.get("schema_version")
    if schema_version == 1:
        raise ValueError(
            "schema_version 1 requires migration to 2: add explicit tool_geometry "
            "and work_pose sections; safety-critical tool geometry has no implicit default",
        )
    if schema_version != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported or missing schema_version (expected {_SCHEMA_VERSION})",
        )
    values = _resolve_environment(raw, os.environ if environ is None else environ)
    assert isinstance(values, dict)
    expected_sections = {
        "schema_version", "robot", "tool_geometry", "topics", "visual_servo",
        "approach",
        "ik", "rrt", "standoff", "work_pose", "grasp_plan",
        "time_parameterization", "retry_budget",
        "collision_model", "vlm_models",
    }
    unknown = set(values) - expected_sections
    missing = expected_sections - set(values)
    if unknown or missing:
        raise ValueError(
            "stack config sections mismatch; "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}",
        )
    try:
        robot_values = _mapping(values["robot"], "robot")
        urdf = Path(str(robot_values.pop("urdf_path"))).expanduser()
        if not urdf.is_absolute():
            urdf = config_path.parent / urdf
        robot = RobotModelConfig(
            urdf_path=urdf.resolve(),
            platform_base_frame=str(robot_values.pop("platform_base_frame")),
            mount_parent_link=str(robot_values.pop("mount_parent_link")),
            base_link=str(robot_values.pop("base_link")),
            tip_link=str(robot_values.pop("tip_link")),
            acceleration_limits=_float_tuple(
                robot_values.pop("acceleration_limits"),
                "robot.acceleration_limits",
            ),
        )
        if robot_values:
            raise ValueError(f"unknown robot fields: {sorted(robot_values)}")
        if not robot.urdf_path.exists():
            raise ValueError(f"robot URDF does not exist: {robot.urdf_path}")
        if (
            not robot.platform_base_frame
            or not robot.mount_parent_link
            or not robot.base_link
            or not robot.tip_link
            or not robot.acceleration_limits
        ):
            raise ValueError("robot links and acceleration limits must be configured")
        if not all(
            math.isfinite(limit) and limit > 0.0
            for limit in robot.acceleration_limits
        ):
            raise ValueError(
                "robot acceleration limits must be finite and positive",
            )

        tool_values = _mapping(values["tool_geometry"], "tool_geometry")
        tool_geometry = ToolGeometryConfig(
            tip_closing_axis=_float_tuple(
                tool_values.pop("tip_closing_axis"),
                "tool_geometry.tip_closing_axis",
            ),
            tip_approach_axis=_float_tuple(
                tool_values.pop("tip_approach_axis"),
                "tool_geometry.tip_approach_axis",
            ),
            finger_contact_z_interval_m=_float_tuple(
                tool_values.pop("finger_contact_z_interval_m"),
                "tool_geometry.finger_contact_z_interval_m",
            ),
            contact_tcp_z_m=float(tool_values.pop("contact_tcp_z_m")),
            collision_open_aperture_m=float(tool_values.pop("collision_open_aperture_m")),
            collision_grasp_margin_m=float(tool_values.pop("collision_grasp_margin_m")),
        )
        if tool_values:
            raise ValueError(f"unknown tool_geometry fields: {sorted(tool_values)}")

        visual_servo = VisualServoConfig(**_mapping(values["visual_servo"], "visual_servo"))
        approach = TwoStageApproachConfig(
            **_mapping(values["approach"], "approach"),
            visual_servo=visual_servo,
        )
        collision = Path(str(values["collision_model"])).expanduser()
        if not collision.is_absolute():
            collision = config_path.parent / collision
        collision = collision.resolve()
        collision_model = _load_collision_model(collision)
        _validate_collision_frames(robot, collision_model)
        raw_models = values["vlm_models"]
        if (
            not isinstance(raw_models, list)
            or not raw_models
            or any(
                not isinstance(model, str) or not model.strip()
                for model in raw_models
            )
        ):
            raise ValueError(
                "vlm_models must be a non-empty array of model identifiers",
            )
        models = tuple(model.strip() for model in raw_models)

        work_pose_values = _mapping(values["work_pose"], "work_pose")
        for name in (
            "radial_distances_m",
            "target_lateral_offsets_m",
            "yaw_offsets_rad",
        ):
            if name in work_pose_values:
                work_pose_values[name] = _float_tuple(
                    work_pose_values[name],
                    f"work_pose.{name}",
                )
        work_pose = WorkPoseConfig(**work_pose_values)

        grasp_values = _mapping(values["grasp_plan"], "grasp_plan")
        if "lift_direction_base" in grasp_values:
            grasp_values["lift_direction_base"] = _float_tuple(
                grasp_values["lift_direction_base"],
                "grasp_plan.lift_direction_base",
            )
        if "tool_from_tip" in grasp_values:
            grasp_values["tool_from_tip"] = _float_matrix(
                grasp_values["tool_from_tip"],
                "grasp_plan.tool_from_tip",
            )
        grasp_plan = GraspPlanConfig(**grasp_values)
        _validate_geometry_contract(
            robot=robot,
            tool=tool_geometry,
            grasp=grasp_plan,
            collision=collision_model,
        )
        return StackConfig(
            schema_version=_SCHEMA_VERSION,
            robot=robot,
            tool_geometry=tool_geometry,
            topics=TopicConfig(**_mapping(values["topics"], "topics")),
            visual_servo=visual_servo,
            approach=approach,
            ik=IKConfig(**_mapping(values["ik"], "ik")),
            rrt=RRTConnectConfig(**_mapping(values["rrt"], "rrt")),
            standoff=ReachabilityStandoffConfig(
                **_mapping(values["standoff"], "standoff"),
            ),
            work_pose=work_pose,
            grasp_plan=grasp_plan,
            time_parameterization=TimeParameterizationConfig(
                **_mapping(values["time_parameterization"], "time_parameterization"),
            ),
            retry_budget=RetryBudget(**_mapping(values["retry_budget"], "retry_budget")),
            collision_model_path=collision,
            vlm_models=models,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid stack config {config_path}: {error}") from error
