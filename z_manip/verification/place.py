"""Freeze the geometry of the object the gripper is carrying, from observation.

Placement needs two things the pick path never recorded: where the carried
object sits in the tool frame, and how big it is.  Both are derived here from
artifacts a real run already writes - the tracked-target cloud, the hand-eye
transform, and the tool pose the gripper actually closed at - so nothing new has
to be sensed while the object is held.

The extent is measured from ONE camera view, so the occluded far face is not
observed and the raw oriented box under-reports.  An under-reported footprint is
the fail-open direction for placement: it would let the planner accept a support
patch the object does not actually fit on.  ``extent_margin_m`` is therefore
applied on every axis and both the observed and the inflated extent are carried
in the record, so a reader can never mistake one for the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CARRIED_OBJECT_GEOMETRY_SCHEMA = "z_manip.carried_object_geometry.v1"

#: Half of the depth-sensor three-sigma band plus the hand-eye position RMSE
#: measured on this deployment (0.006 m + 0.0098 m); an unobserved far face is
#: at least this much larger than the single-view oriented box.
DEFAULT_EXTENT_MARGIN_M = 0.01


@dataclass(frozen=True, eq=False)
class CarriedObjectGeometry:
    """Observed pose and size of the held object, in the tool and base frames.

    ``object_extent_m`` is what a placement footprint must use; it is
    ``observed_extent_m`` inflated by ``extent_margin_m`` on every axis.
    """

    frame: str
    tool_from_object: np.ndarray
    object_pose_base: np.ndarray
    observed_extent_m: np.ndarray
    object_extent_m: np.ndarray
    extent_margin_m: float
    observed_point_count: int
    tool_pose_source: str

    @property
    def record(self) -> dict:
        return carried_object_geometry_record(self)


def _finite_transform(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) < 0.999
    ):
        raise ValueError(f"{label} rotation is not right-handed orthonormal")
    return matrix


def derive_carried_object_geometry(
    target_points_camera: object,
    *,
    base_from_camera: object,
    tool_pose_base: object,
    tool_pose_source: str,
    frame: str,
    extent_margin_m: float = DEFAULT_EXTENT_MARGIN_M,
    min_points: int = 40,
) -> CarriedObjectGeometry:
    """Freeze the held object's tool-frame pose and footprint from one view.

    ``target_points_camera`` is the persistent tracked-target cloud recorded at
    pick time, in the camera optical frame.  ``tool_pose_base`` is the tool pose
    the gripper closed at, in ``frame``; ``tool_pose_source`` records whether it
    was the planned grasp pose or forward kinematics on measured joints, because
    the two are not interchangeable evidence.
    """

    points = np.asarray(target_points_camera, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("tracked-target cloud must have shape (N, 3)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < min_points:
        raise ValueError(
            f"tracked-target cloud has {len(points)} finite points; need {min_points}",
        )
    if not np.isfinite(extent_margin_m) or extent_margin_m < 0.0:
        raise ValueError("carried-object extent margin must be finite and non-negative")
    if not tool_pose_source:
        raise ValueError("carried-object geometry requires a named tool-pose source")
    if not frame:
        raise ValueError("carried-object geometry requires a non-empty frame")
    transform = _finite_transform(base_from_camera, "base_from_camera")
    tool_pose = _finite_transform(tool_pose_base, "tool_pose_base")

    cloud = points @ transform[:3, :3].T + transform[:3, 3]
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    covariance = centered.T @ centered / len(centered)
    _, eigenvectors = np.linalg.eigh(covariance)
    # Longest observed axis first, so the footprint axes stay comparable across
    # runs; then force right-handedness, which the placement planner requires.
    axes = eigenvectors[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] = -axes[:, 2]
    local = centered @ axes
    lower = local.min(axis=0)
    upper = local.max(axis=0)
    observed_extent = upper - lower
    if not np.all(np.isfinite(observed_extent)) or np.any(observed_extent <= 0.0):
        raise ValueError("tracked-target cloud is degenerate in at least one axis")

    object_pose = np.eye(4)
    object_pose[:3, :3] = axes
    object_pose[:3, 3] = centroid + axes @ (0.5 * (lower + upper))
    return CarriedObjectGeometry(
        frame=str(frame),
        tool_from_object=_finite_transform(
            np.linalg.inv(tool_pose) @ object_pose, "tool_from_object",
        ),
        object_pose_base=object_pose,
        observed_extent_m=observed_extent,
        object_extent_m=observed_extent + 2.0 * float(extent_margin_m),
        extent_margin_m=float(extent_margin_m),
        observed_point_count=int(len(points)),
        tool_pose_source=str(tool_pose_source),
    )


def derive_from_session_artifacts(
    target_points_camera: object,
    planning_report: dict,
    *,
    extent_margin_m: float = DEFAULT_EXTENT_MARGIN_M,
) -> CarriedObjectGeometry:
    """Derive the record from a recorded ``planning/planning_report.json``.

    The planned grasp pose is the tool pose here, not a measured one: a real
    pick can close somewhere else, so the record says so and a caller that has
    executor evidence should pass measured forward kinematics instead.
    """

    for key in ("base_from_camera", "grasp_pose", "planning_frame"):
        if key not in planning_report:
            raise ValueError(f"planning report is missing {key}")
    return derive_carried_object_geometry(
        target_points_camera,
        base_from_camera=planning_report["base_from_camera"],
        tool_pose_base=planning_report["grasp_pose"],
        tool_pose_source="planning_report.grasp_pose",
        frame=str(planning_report["planning_frame"]),
        extent_margin_m=extent_margin_m,
    )


def carried_object_geometry_record(
    geometry: CarriedObjectGeometry,
    *,
    sources: dict | None = None,
) -> dict:
    """Serialize the frozen geometry as a hash-chainable receipt payload."""

    return {
        "schema": CARRIED_OBJECT_GEOMETRY_SCHEMA,
        "frame": geometry.frame,
        "tool_from_object": [
            [float(value) for value in row] for row in geometry.tool_from_object
        ],
        "object_pose_base": [
            [float(value) for value in row] for row in geometry.object_pose_base
        ],
        "observed_extent_m": [float(value) for value in geometry.observed_extent_m],
        "object_extent_m": [float(value) for value in geometry.object_extent_m],
        "extent_margin_m": float(geometry.extent_margin_m),
        "extent_basis": "single_view_oriented_box_plus_margin",
        "extent_is_observed_lower_bound": True,
        "observed_point_count": int(geometry.observed_point_count),
        "tool_pose_source": geometry.tool_pose_source,
        "sources": dict(sources or {}),
    }


__all__ = [
    "CARRIED_OBJECT_GEOMETRY_SCHEMA",
    "DEFAULT_EXTENT_MARGIN_M",
    "CarriedObjectGeometry",
    "carried_object_geometry_record",
    "derive_carried_object_geometry",
    "derive_from_session_artifacts",
]
