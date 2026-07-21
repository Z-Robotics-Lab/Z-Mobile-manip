"""URDF-mesh self-collision checks backed by Pinocchio and hpp-fcl."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from z_manip.kinematics import KinematicChain

from .pointcloud import CollisionResult


class PinocchioSelfCollisionChecker:
    """Check non-adjacent PiPER meshes from the deployed robot URDF.

    Links separated by at most two URDF joints are excluded, matching the
    usual SRDF "adjacent/never" policy and avoiding designed-in overlap around
    bearings, the wrist flange, and the parallel fingers.  Every remaining
    pair is checked with the actual collision meshes rather than arm capsules.
    """

    def __init__(self, urdf_path: str | Path, chain: KinematicChain) -> None:
        try:
            import pinocchio as pin
        except ImportError as error:  # pragma: no cover - host unit env lacks ROS bindings
            raise RuntimeError(
                "mesh self-collision requires the pinocchio Python module",
            ) from error

        self.pin = pin
        self.chain = chain
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        models = pin.buildModelsFromUrdf(
            str(self.urdf_path),
            package_dirs=[str(self.urdf_path.parent)],
            geometry_types=[pin.GeometryType.COLLISION],
        )
        full_model, full_geometry = models[0], models[1]
        active_names = set(chain.joint_names)
        locked_joint_ids = [
            joint_id
            for joint_id, name in enumerate(full_model.names)
            if joint_id and name not in active_names
        ]
        self.model, reduced_geometries = pin.buildReducedModel(
            full_model,
            [full_geometry],
            locked_joint_ids,
            pin.neutral(full_model),
        )
        self.geometry_model = reduced_geometries[0]

        parent_by_child: dict[str, str] = {}
        root = ElementTree.parse(self.urdf_path).getroot()
        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is not None and child is not None:
                parent_by_child[child.attrib["link"]] = parent.attrib["link"]

        selected: list[int] = []
        for geometry_id, geometry in enumerate(self.geometry_model.geometryObjects):
            frame_name = self.model.frames[geometry.parentFrame].name
            supporting_joint = self.model.names[geometry.parentJoint]
            if frame_name == chain.base_link or supporting_joint in active_names:
                selected.append(geometry_id)

        for offset, first_id in enumerate(selected):
            first = self.geometry_model.geometryObjects[first_id]
            first_frame = self.model.frames[first.parentFrame].name
            for second_id in selected[offset + 1:]:
                second = self.geometry_model.geometryObjects[second_id]
                second_frame = self.model.frames[second.parentFrame].name
                if self._link_distance(
                    first_frame,
                    second_frame,
                    parent_by_child,
                    maximum=2,
                ) <= 2:
                    continue
                self.geometry_model.addCollisionPair(
                    pin.CollisionPair(first_id, second_id),
                )
        if not self.geometry_model.collisionPairs:
            raise ValueError("Pinocchio mesh checker produced no collision pairs")
        self.data = self.model.createData()
        self.geometry_data = pin.GeometryData(self.geometry_model)

    @staticmethod
    def _link_distance(
        first: str,
        second: str,
        parent_by_child: dict[str, str],
        *,
        maximum: int,
    ) -> int:
        if first == second:
            return 0
        adjacency: dict[str, set[str]] = {}
        for child, parent in parent_by_child.items():
            adjacency.setdefault(child, set()).add(parent)
            adjacency.setdefault(parent, set()).add(child)
        pending = deque([(first, 0)])
        seen = {first}
        while pending:
            link, distance = pending.popleft()
            if distance >= maximum:
                continue
            for neighbor in adjacency.get(link, ()):
                if neighbor == second:
                    return distance + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append((neighbor, distance + 1))
        return maximum + 1

    @property
    def pair_count(self) -> int:
        return len(self.geometry_model.collisionPairs)

    def check_state(self, joints: object) -> CollisionResult:
        try:
            values = np.asarray(joints, dtype=float)
        except (TypeError, ValueError):
            values = np.asarray([], dtype=float)
        if values.shape != (self.chain.dof,) or not np.all(np.isfinite(values)):
            return CollisionResult(
                False,
                f"joint state must be a finite ({self.chain.dof},) vector",
                kind="kinematics",
            )
        if np.any(values < self.chain.lower_limits) or np.any(
            values > self.chain.upper_limits
        ):
            return CollisionResult(
                False,
                "joint state violates URDF limits",
                kind="kinematics",
            )
        try:
            collided = self.pin.computeCollisions(
                self.model,
                self.data,
                self.geometry_model,
                self.geometry_data,
                values,
                True,
            )
        except Exception as error:
            return CollisionResult(
                False,
                f"Pinocchio mesh collision failed: {type(error).__name__}: {error}",
                kind="kinematics",
            )
        if not collided:
            return CollisionResult(True, "mesh self-collision-free")
        for pair_index, collision in enumerate(self.geometry_data.collisionResults):
            if not collision.isCollision():
                continue
            pair = self.geometry_model.collisionPairs[pair_index]
            first = self.geometry_model.geometryObjects[pair.first].name
            second = self.geometry_model.geometryObjects[pair.second].name
            return CollisionResult(
                False,
                f"URDF meshes {first!r} and {second!r} self-collide",
                kind="self_mesh",
                capsules=(first, second),
            )
        return CollisionResult(
            False,
            "Pinocchio reported collision without a witness pair",
            kind="self_mesh",
        )


__all__ = ["PinocchioSelfCollisionChecker"]
