"""Continuous payload-aware collision audit for placement trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from z_manip.collision import (
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
)
from z_manip.kinematics import KinematicChain
from z_manip.models.planner import PlanningError
from z_manip.planning_control import checkpoint, PlanningControl


def _transform(matrix: object, label: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f'{label} must be a finite 4x4 transform')
    rotation = values[:3, :3]
    if (
        not np.allclose(values[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(f'{label} is not a rigid transform')
    return values.copy()


def _points(points: object, label: str, *, minimum: int = 1) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or len(values) < minimum
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f'{label} must contain at least {minimum} finite XYZ points',
        )
    return np.ascontiguousarray(values)


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


@dataclass(frozen=True)
class AttachedCollisionAuditConfig:
    """Externalized uncertainty and sampling limits for payload auditing."""

    clearance_m: float = 0.02
    point_radius_m: float = 0.005
    segment_joint_step_rad: float = 0.025
    min_scene_points: int = 32
    max_attached_points: int = 512
    extent_samples_per_axis: int = 5
    carried_object_scene_exclusion_m: float = 0.012

    def __post_init__(self) -> None:
        nonnegative = (
            self.clearance_m,
            self.point_radius_m,
            self.carried_object_scene_exclusion_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError('attached collision distances must be non-negative')
        if (
            not math.isfinite(self.segment_joint_step_rad)
            or self.segment_joint_step_rad <= 0.0
        ):
            raise ValueError('attached collision joint step must be positive')
        if self.min_scene_points < 1 or self.max_attached_points < 8:
            raise ValueError('attached collision point limits are invalid')
        if self.extent_samples_per_axis < 2:
            raise ValueError('extent_samples_per_axis must be at least two')

    def pointcloud_config(self) -> PointCloudCollisionConfig:
        """Build the common collision checker configuration."""
        return PointCloudCollisionConfig(
            clearance=self.clearance_m,
            point_radius=self.point_radius_m,
            min_scene_points=self.min_scene_points,
            segment_joint_step=self.segment_joint_step_rad,
            max_attached_points=self.max_attached_points,
        )


@dataclass(frozen=True, eq=False)
class AttachedCollisionSnapshot:
    """One immutable placement snapshot expressed in the kinematic base."""

    scene_points: np.ndarray
    scene_stamp_s: float
    attachment_joints: np.ndarray
    payload_points_object: np.ndarray
    tool_from_object: np.ndarray
    kinematic_base_from_planning: np.ndarray


class AttachedObjectPathAuditor:
    """Reject a candidate unless every payload-aware phase is collision-free."""

    def __init__(
        self,
        *,
        chain: KinematicChain,
        collision_model: RobotCollisionModel,
        config: AttachedCollisionAuditConfig,
    ) -> None:
        self.chain = chain
        self.collision_model = collision_model
        self.config = config
        self._snapshot: AttachedCollisionSnapshot | None = None

    def bind_snapshot(
        self,
        *,
        scene_points: object,
        scene_stamp_s: float,
        planning_from_kinematic_base: object,
        attachment_joints: object,
        object_reference_points_object: object,
        object_extent_m: object,
        tool_from_object: object,
    ) -> None:
        """Freeze one observed scene and grasp-time payload model."""
        if not math.isfinite(scene_stamp_s) or scene_stamp_s < 0.0:
            raise ValueError('attached collision scene stamp must be non-negative')
        scene_planning = _points(
            scene_points,
            'attached collision scene',
            minimum=self.config.min_scene_points,
        )
        planning_from_base = _transform(
            planning_from_kinematic_base,
            'planning_from_kinematic_base',
        )
        base_from_planning = np.linalg.inv(planning_from_base)
        reference = _points(
            object_reference_points_object,
            'frozen object reference',
            minimum=8,
        )
        extent = np.asarray(object_extent_m, dtype=float)
        if (
            extent.shape != (3,)
            or not np.all(np.isfinite(extent))
            or np.any(extent <= 0.0)
        ):
            raise ValueError('frozen object extent must be a positive XYZ vector')
        if not np.allclose(
            np.ptp(reference, axis=0),
            extent,
            rtol=1e-6,
            atol=1e-9,
        ):
            raise ValueError('frozen object extent does not match its reference')
        attachment = np.asarray(attachment_joints, dtype=float)
        if (
            attachment.shape != (self.chain.dof,)
            or not np.all(np.isfinite(attachment))
        ):
            raise ValueError('attachment joints do not match the kinematic chain')
        tool_from_object_values = _transform(
            tool_from_object,
            'tool_from_object',
        )

        axes = tuple(
            np.linspace(
                float(np.min(reference[:, axis])),
                float(np.max(reference[:, axis])),
                self.config.extent_samples_per_axis,
            )
            for axis in range(3)
        )
        xx, yy, zz = np.meshgrid(*axes, indexing='ij')
        extent_envelope = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        payload_object = np.unique(
            np.vstack((reference, extent_envelope)),
            axis=0,
        )
        base_from_tool = self.chain.forward(attachment)
        current_payload = _apply(
            base_from_tool @ tool_from_object_values,
            payload_object,
        )
        scene_base = _apply(base_from_planning, scene_planning)
        target_mask = np.zeros(len(scene_base), dtype=bool)
        exclusion = self.config.carried_object_scene_exclusion_m
        if exclusion > 0.0:
            distance, _ = cKDTree(current_payload).query(scene_base, k=1)
            target_mask = np.asarray(distance <= exclusion, dtype=bool)
        filtered_scene = np.ascontiguousarray(scene_base[~target_mask])
        if len(filtered_scene) < self.config.min_scene_points:
            raise ValueError(
                'carried-object exclusion leaves too few scene points',
            )
        for values in (
            filtered_scene,
            attachment,
            payload_object,
            tool_from_object_values,
            base_from_planning,
        ):
            values.setflags(write=False)
        self._snapshot = AttachedCollisionSnapshot(
            scene_points=filtered_scene,
            scene_stamp_s=float(scene_stamp_s),
            attachment_joints=attachment,
            payload_points_object=payload_object,
            tool_from_object=tool_from_object_values,
            kinematic_base_from_planning=base_from_planning,
        )

    def clear_snapshot(self) -> None:
        """Prevent accidental reuse across placement transactions."""
        self._snapshot = None

    def _checker(self) -> PointCloudCollisionChecker:
        snapshot = self._snapshot
        if snapshot is None:
            raise PlanningError('attached collision snapshot is not bound')
        checker = PointCloudCollisionChecker(
            chain=self.chain,
            model=self.collision_model,
            frame_provider=self.chain.link_transforms,
            config=self.config.pointcloud_config(),
            now_fn=lambda: snapshot.scene_stamp_s,
        )
        checker.update_scene(
            snapshot.scene_points,
            stamp_s=snapshot.scene_stamp_s,
        )
        return checker

    def _ordered_positions(self, segment: object, phase: str) -> np.ndarray:
        names = tuple(getattr(segment, 'joint_names', ()))
        if (
            len(names) != len(set(names))
            or set(names) != set(self.chain.joint_names)
        ):
            raise PlanningError(
                f'attached collision {phase} joints do not match the arm',
            )
        positions = np.asarray(getattr(segment, 'positions', None), dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != len(names)
            or len(positions) < 2
            or not np.all(np.isfinite(positions))
        ):
            raise PlanningError(
                f'attached collision {phase} path is empty or malformed',
            )
        return positions[:, [names.index(name) for name in self.chain.joint_names]]

    @staticmethod
    def _phase_segments(path: np.ndarray):
        return tuple(zip(path, path[1:]))

    def _require_segments(
        self,
        checker: PointCloudCollisionChecker,
        path: np.ndarray,
        phase: str,
        control: PlanningControl | None,
    ) -> None:
        for index, (first, second) in enumerate(self._phase_segments(path)):
            checkpoint(control, f'attached collision {phase} segment')
            result = checker.check_segment(first, second, control=control)
            if not result.valid:
                kind = (
                    ''
                    if result.state_result is None or result.state_result.kind is None
                    else f' [{result.state_result.kind}]'
                )
                raise PlanningError(
                    f'attached collision {phase} segment {index}{kind}: '
                    f'{result.reason}',
                )

    def audit(
        self,
        *,
        segments: Sequence[object],
        control: PlanningControl | None = None,
    ) -> None:
        """Continuously audit transit, support approach, and released retreat."""
        checkpoint(control, 'attached collision audit')
        snapshot = self._snapshot
        if snapshot is None:
            raise PlanningError('attached collision snapshot is not bound')
        if tuple(getattr(segment, 'phase', '') for segment in segments) != (
            'transit',
            'approach',
            'retreat',
        ):
            raise PlanningError(
                'attached collision audit requires transit/approach/retreat',
            )
        transit, approach, retreat = (
            self._ordered_positions(segment, segment.phase)
            for segment in segments
        )
        if not np.allclose(
            transit[0],
            snapshot.attachment_joints,
            rtol=0.0,
            atol=1e-7,
        ):
            raise PlanningError(
                'attached collision transit does not start at the attachment state',
            )

        current_payload = _apply(
            self.chain.forward(snapshot.attachment_joints)
            @ snapshot.tool_from_object,
            snapshot.payload_points_object,
        )
        strict = self._checker()
        strict.update_attached_target(
            current_payload,
            attachment_joints=snapshot.attachment_joints,
            allowed_contact_capsules=(
                self.collision_model.target_contact_capsules
            ),
        )
        self._require_segments(strict, transit, 'transit', control)
        if len(approach) > 2:
            self._require_segments(strict, approach[:-1], 'approach', control)

        final = approach[-1]
        before_final = approach[-2]
        final_payload = _apply(
            self.chain.forward(final) @ snapshot.tool_from_object,
            snapshot.payload_points_object,
        )
        prior_payload = _apply(
            self.chain.forward(before_final) @ snapshot.tool_from_object,
            snapshot.payload_points_object,
        )
        departure = np.mean(prior_payload, axis=0) - np.mean(final_payload, axis=0)
        departure_norm = float(np.linalg.norm(departure))
        if not math.isfinite(departure_norm) or departure_norm <= 1e-9:
            raise PlanningError(
                'attached collision final approach has no measurable departure',
            )
        support_contact = self._checker()
        support_contact.update_attached_target(
            final_payload,
            attachment_joints=final,
            allowed_contact_capsules=(
                self.collision_model.target_contact_capsules
            ),
            allow_initial_scene_contact=True,
            departure_direction_base=departure / departure_norm,
        )
        reverse_result = support_contact.check_segment(
            final,
            before_final,
            control=control,
        )
        if not reverse_result.valid:
            raise PlanningError(
                'attached collision final approach support audit: '
                f'{reverse_result.reason}',
            )

        released = self._checker()
        released.update_target(
            final_payload,
            allowed_contact_capsules=(
                self.collision_model.target_contact_capsules
            ),
        )
        self._require_segments(released, retreat, 'retreat', control)


__all__ = [
    'AttachedCollisionAuditConfig',
    'AttachedCollisionSnapshot',
    'AttachedObjectPathAuditor',
]
