"""Small, dependency-light kinematic chain built directly from a URDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np


_ACTIVE_TYPES = frozenset(("revolute", "continuous", "prismatic"))


def _vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    vector = np.fromstring(value, sep=" ", dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"expected a three-vector, got {value!r}")
    return vector


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    sine, cosine = np.sin(angle), np.cos(angle)
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm as a rotation vector.

    The near-pi branch avoids division by a vanishing sine, which matters when
    numerical IK starts on the opposite wrist orientation.
    """

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.array(
        (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
         matrix[1, 0] - matrix[0, 1]),
        dtype=float,
    )
    if angle < 1e-7:
        return 0.5 * vee
    if np.pi - angle < 1e-5:
        symmetric = 0.5 * (matrix + np.eye(3))
        pivot = int(np.argmax(np.diag(symmetric)))
        axis = symmetric[:, pivot]
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            axis = eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))]
        else:
            axis /= norm
        # Resolve the otherwise arbitrary eigenspace sign continuously where
        # the skew component still carries a usable sign.
        if np.dot(axis, vee) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * np.sin(angle))) * vee


@dataclass(frozen=True)
class Joint:
    """One URDF joint on a serial path."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


class KinematicChain:
    """Ordered URDF path between two links, including intervening fixed joints."""

    def __init__(self, joints: list[Joint], base_link: str, tip_link: str):
        self.joints = tuple(joints)
        self.base_link = base_link
        self.tip_link = tip_link
        active = [joint for joint in joints if joint.joint_type in _ACTIVE_TYPES]
        self.active_joints = tuple(active)
        self.joint_names = tuple(joint.name for joint in active)
        self.lower_limits = np.asarray([joint.lower for joint in active], dtype=float)
        self.upper_limits = np.asarray([joint.upper for joint in active], dtype=float)
        self.velocity_limits = np.asarray([joint.velocity for joint in active], dtype=float)
        self.dof = len(active)
        if not self.dof:
            raise ValueError(f"chain {base_link!r} -> {tip_link!r} has no active joints")

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        base_link: str,
        tip_link: str,
    ) -> "KinematicChain":
        """Load the unique parent path from ``base_link`` to ``tip_link``."""

        root = ElementTree.parse(Path(path)).getroot()
        by_child: dict[str, Joint] = {}
        for element in root.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                continue
            parent = parent_element.attrib["link"]
            child = child_element.attrib["link"]
            origin_element = element.find("origin")
            xyz = _vector(
                None if origin_element is None else origin_element.get("xyz"),
                (0.0, 0.0, 0.0),
            )
            rpy = _vector(
                None if origin_element is None else origin_element.get("rpy"),
                (0.0, 0.0, 0.0),
            )
            origin = np.eye(4)
            origin[:3, :3] = _rotation_from_rpy(rpy)
            origin[:3, 3] = xyz

            joint_type = element.attrib.get("type", "fixed")
            axis_element = element.find("axis")
            axis = _vector(
                None if axis_element is None else axis_element.get("xyz"),
                (1.0, 0.0, 0.0),
            )
            if joint_type in _ACTIVE_TYPES:
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    raise ValueError(f"active joint {element.attrib['name']!r} has zero axis")
                axis = axis / norm
            limit_element = element.find("limit")
            if joint_type == "continuous":
                lower, upper = -np.pi, np.pi
            elif joint_type in _ACTIVE_TYPES:
                if limit_element is None:
                    raise ValueError(f"active joint {element.attrib['name']!r} has no limits")
                lower = float(limit_element.attrib["lower"])
                upper = float(limit_element.attrib["upper"])
            else:
                lower = upper = 0.0
            velocity = (
                float(limit_element.attrib["velocity"])
                if joint_type in _ACTIVE_TYPES
                and limit_element is not None
                and "velocity" in limit_element.attrib
                else float("inf")
            )
            if lower > upper:
                raise ValueError(f"joint {element.attrib['name']!r} has inverted limits")
            if velocity <= 0.0:
                raise ValueError(f"joint {element.attrib['name']!r} has invalid velocity limit")
            by_child[child] = Joint(
                name=element.attrib["name"],
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                velocity=velocity,
            )

        reverse_path: list[Joint] = []
        link = tip_link
        visited: set[str] = set()
        while link != base_link:
            if link in visited:
                raise ValueError(f"cycle while tracing URDF from {tip_link!r}")
            visited.add(link)
            try:
                joint = by_child[link]
            except KeyError as error:
                raise ValueError(
                    f"{base_link!r} is not an ancestor of {tip_link!r}",
                ) from error
            reverse_path.append(joint)
            link = joint.parent
        return cls(list(reversed(reverse_path)), base_link, tip_link)

    def _check_joints(self, joints: np.ndarray) -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.shape != (self.dof,):
            raise ValueError(f"joint vector must have shape ({self.dof},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("joint vector contains a non-finite value")
        return values

    def forward(self, joints: np.ndarray) -> np.ndarray:
        """Compute the tip transform in the base frame."""

        values = self._check_joints(joints)
        transform = np.eye(4)
        active_index = 0
        for joint in self.joints:
            transform = transform @ joint.origin
            if joint.joint_type in ("revolute", "continuous"):
                motion = np.eye(4)
                motion[:3, :3] = _axis_angle(joint.axis, values[active_index])
                transform = transform @ motion
                active_index += 1
            elif joint.joint_type == "prismatic":
                motion = np.eye(4)
                motion[:3, 3] = joint.axis * values[active_index]
                transform = transform @ motion
                active_index += 1
        return transform

    def link_transforms(self, joints: np.ndarray) -> dict[str, np.ndarray]:
        """Return every link frame on the chain expressed in ``base_link``.

        This is the geometry boundary used by collision backends. It exposes
        only URDF kinematics, so the same capsule or mesh checker works in sim,
        on recorded sensor data, and against the real robot model.
        """

        values = self._check_joints(joints)
        transform = np.eye(4)
        frames = {self.base_link: transform.copy()}
        active_index = 0
        for joint in self.joints:
            transform = transform @ joint.origin
            if joint.joint_type in ("revolute", "continuous"):
                motion = np.eye(4)
                motion[:3, :3] = _axis_angle(joint.axis, values[active_index])
                transform = transform @ motion
                active_index += 1
            elif joint.joint_type == "prismatic":
                motion = np.eye(4)
                motion[:3, 3] = joint.axis * values[active_index]
                transform = transform @ motion
                active_index += 1
            frames[joint.child] = transform.copy()
        return frames

    def jacobian(self, joints: np.ndarray) -> np.ndarray:
        """Return the 6xN base-frame geometric Jacobian at the tip."""

        values = self._check_joints(joints)
        transform = np.eye(4)
        active_index = 0
        joint_frames: list[tuple[str, np.ndarray, np.ndarray]] = []
        for joint in self.joints:
            transform = transform @ joint.origin
            if joint.joint_type in _ACTIVE_TYPES:
                position = transform[:3, 3].copy()
                axis = transform[:3, :3] @ joint.axis
                joint_frames.append((joint.joint_type, position, axis))
                if joint.joint_type in ("revolute", "continuous"):
                    motion = np.eye(4)
                    motion[:3, :3] = _axis_angle(joint.axis, values[active_index])
                else:
                    motion = np.eye(4)
                    motion[:3, 3] = joint.axis * values[active_index]
                transform = transform @ motion
                active_index += 1

        tip_position = transform[:3, 3]
        jacobian = np.zeros((6, self.dof), dtype=float)
        for index, (joint_type, position, axis) in enumerate(joint_frames):
            if joint_type in ("revolute", "continuous"):
                jacobian[:3, index] = np.cross(axis, tip_position - position)
                jacobian[3:, index] = axis
            else:
                jacobian[:3, index] = axis
        return jacobian


def fixed_transform_from_urdf(
    path: str | Path,
    base_link: str,
    tip_link: str,
) -> np.ndarray:
    """Resolve a fixed-link transform from the deployed robot description.

    Refusing an active joint prevents a nominal arm-mount transform from
    silently ignoring robot state.
    """
    root = ElementTree.parse(Path(path)).getroot()
    by_child: dict[str, tuple[str, str, np.ndarray]] = {}
    for element in root.findall("joint"):
        parent_element = element.find("parent")
        child_element = element.find("child")
        if parent_element is None or child_element is None:
            continue
        origin_element = element.find("origin")
        xyz = _vector(
            None if origin_element is None else origin_element.get("xyz"),
            (0.0, 0.0, 0.0),
        )
        rpy = _vector(
            None if origin_element is None else origin_element.get("rpy"),
            (0.0, 0.0, 0.0),
        )
        origin = np.eye(4)
        origin[:3, :3] = _rotation_from_rpy(rpy)
        origin[:3, 3] = xyz
        by_child[child_element.attrib["link"]] = (
            parent_element.attrib["link"],
            element.attrib.get("type", "fixed"),
            origin,
        )

    reverse_path: list[tuple[str, str, np.ndarray]] = []
    link = tip_link
    visited: set[str] = set()
    while link != base_link:
        if link in visited:
            raise ValueError(f"cycle while tracing URDF from {tip_link!r}")
        visited.add(link)
        try:
            joint = by_child[link]
        except KeyError as error:
            raise ValueError(
                f"{base_link!r} is not an ancestor of {tip_link!r}",
            ) from error
        if joint[1] != "fixed":
            raise ValueError(
                f"mount path {base_link!r} -> {tip_link!r} contains active joint",
            )
        reverse_path.append(joint)
        link = joint[0]
    transform = np.eye(4)
    for _parent, _joint_type, origin in reversed(reverse_path):
        transform = transform @ origin
    return transform
