"""Cross-section semantic validation for the deployment configuration.

Checks invariants that span multiple independently-authored config sections
(tool geometry vs. the grasp-plan transform vs. collision capsules) and an
external URDF file, as opposed to the single-object invariants that live on
each dataclass in :mod:`z_manip.config.schema`.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from z_manip.collision import RobotCollisionModel
from z_manip.planning.grasp_pipeline import GraspPlanConfig

from .schema import RobotModelConfig, ToolGeometryConfig


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
