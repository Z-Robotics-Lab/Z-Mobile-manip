"""JSON loading, parsing, and orchestration for the deployment configuration.

Reads the strict-schema JSON stack config (env-var substitution, dict-shape
coercion, collision-model loading) and assembles the typed sections defined
in :mod:`z_manip.config.schema`, invoking the cross-section checks in
:mod:`z_manip.config.validation` along the way.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Mapping

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

from .schema import RobotModelConfig, StackConfig, ToolGeometryConfig, TopicConfig
from .validation import _validate_collision_frames, _validate_geometry_contract


_ENVIRONMENT_VALUE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_SCHEMA_VERSION = 2


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


def _pop_required(mapping: dict[str, object], key: str, label: str) -> object:
    """Pop ``key`` from ``mapping``, raising a clear message if it is absent.

    ``robot`` and ``tool_geometry`` build their typed config via individual
    ``dict.pop`` calls (each field needs its own path/tuple conversion), so a
    missing field raises a bare ``KeyError('urdf_path')`` instead of the
    descriptive ``TypeError`` every other section gets for free from
    ``Cls(**mapping)``. This closes that message-quality gap.
    """
    try:
        return mapping.pop(key)
    except KeyError:
        raise ValueError(f"{label} is missing required field {key!r}") from None


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
        urdf = Path(str(_pop_required(robot_values, "urdf_path", "robot"))).expanduser()
        if not urdf.is_absolute():
            urdf = config_path.parent / urdf
        robot = RobotModelConfig(
            urdf_path=urdf.resolve(),
            platform_base_frame=str(_pop_required(robot_values, "platform_base_frame", "robot")),
            mount_parent_link=str(_pop_required(robot_values, "mount_parent_link", "robot")),
            base_link=str(_pop_required(robot_values, "base_link", "robot")),
            tip_link=str(_pop_required(robot_values, "tip_link", "robot")),
            acceleration_limits=_float_tuple(
                _pop_required(robot_values, "acceleration_limits", "robot"),
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
                _pop_required(tool_values, "tip_closing_axis", "tool_geometry"),
                "tool_geometry.tip_closing_axis",
            ),
            tip_approach_axis=_float_tuple(
                _pop_required(tool_values, "tip_approach_axis", "tool_geometry"),
                "tool_geometry.tip_approach_axis",
            ),
            finger_contact_z_interval_m=_float_tuple(
                _pop_required(tool_values, "finger_contact_z_interval_m", "tool_geometry"),
                "tool_geometry.finger_contact_z_interval_m",
            ),
            contact_tcp_z_m=float(_pop_required(tool_values, "contact_tcp_z_m", "tool_geometry")),
            collision_open_aperture_m=float(
                _pop_required(tool_values, "collision_open_aperture_m", "tool_geometry"),
            ),
            collision_grasp_margin_m=float(
                _pop_required(tool_values, "collision_grasp_margin_m", "tool_geometry"),
            ),
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
        ik_values = _mapping(values["ik"], "ik")
        if "position_error_offset_tip_m" not in ik_values:
            # Enforce the IK position tolerance at the tool CONTACT point, not
            # the tip link: the TCP sits ``contact_tcp_z_m`` along the approach
            # axis, so tip-frame orientation error levers into contact-point
            # translation that a tip-origin gate never bounds.  Derived from
            # the validated tool geometry so it cannot drift from the deployed
            # gripper; an explicit config value still wins.
            approach_axis = np.asarray(tool_geometry.tip_approach_axis, dtype=float)
            ik_values["position_error_offset_tip_m"] = tuple(
                float(component * tool_geometry.contact_tcp_z_m)
                for component in approach_axis
            )
        return StackConfig(
            schema_version=_SCHEMA_VERSION,
            robot=robot,
            tool_geometry=tool_geometry,
            topics=TopicConfig(**_mapping(values["topics"], "topics")),
            visual_servo=visual_servo,
            approach=approach,
            ik=IKConfig(**ik_values),
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


__all__ = ["load_stack_config"]
