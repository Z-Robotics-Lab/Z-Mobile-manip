"""Dependency-light fixed-fixture collision guard for the live PiPER owner.

The offline grasp planner checks both perception and robot geometry.  Reactive
camera-view motion happens before that planner runs, however, and therefore
needs a small final command-boundary guard of its own.  This module deliberately
checks only configured self-collision pairs involving a
``supplemental_self_collision`` capsule (for example the Go2W Mid-360 and the
wrist-mounted D435).  It has no ROS, SocketCAN, or hardware side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from z_manip.kinematics.chain import KinematicChain


@dataclass(frozen=True)
class FixedCapsule:
    name: str
    start_frame: str
    end_frame: str
    radius_m: float
    start_offset: np.ndarray
    end_offset: np.ndarray
    supplemental: bool


@dataclass(frozen=True)
class CollisionWitness:
    pair: tuple[str, str]
    distance_m: float
    threshold_m: float
    margin_m: float


@dataclass(frozen=True)
class CollisionState:
    valid: bool
    minimum_margin_m: float
    witness: CollisionWitness


@dataclass(frozen=True)
class StepDecision:
    allowed: bool
    escaping: bool
    reason: str
    witness: CollisionWitness
    current_margin_m: float
    target_margin_m: float


def _vector(value: object, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite three-vector")
    return vector


def _normalized_pair(pair: Sequence[str]) -> tuple[str, str]:
    if len(pair) != 2 or not all(isinstance(name, str) and name for name in pair):
        raise ValueError("collision pair must contain two non-empty names")
    if pair[0] == pair[1]:
        raise ValueError("collision pair cannot reference one capsule twice")
    return tuple(sorted((pair[0], pair[1])))


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    length_squared = float(delta @ delta)
    if length_squared <= 1e-20:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip(((point - start) @ delta) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + alpha * delta)))


def _segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Shortest Euclidean distance between two finite 3-D segments."""

    first = first_end - first_start
    second = second_end - second_start
    between = first_start - second_start
    aa = float(first @ first)
    bb = float(first @ second)
    cc = float(second @ second)
    dd = float(first @ between)
    ee = float(second @ between)
    epsilon = 1e-14
    if aa <= epsilon and cc <= epsilon:
        return float(np.linalg.norm(first_start - second_start))
    if aa <= epsilon:
        return _point_segment_distance(first_start, second_start, second_end)
    if cc <= epsilon:
        return _point_segment_distance(second_start, first_start, first_end)

    denominator = aa * cc - bb * bb
    s_numerator = 0.0 if denominator <= epsilon else bb * ee - cc * dd
    s_denominator = denominator
    t_numerator = aa * ee - bb * dd
    t_denominator = denominator
    if denominator <= epsilon:
        s_numerator, s_denominator = 0.0, 1.0
        t_numerator, t_denominator = ee, cc
    elif s_numerator < 0.0:
        s_numerator = 0.0
        t_numerator, t_denominator = ee, cc
    elif s_numerator > s_denominator:
        s_numerator = s_denominator
        t_numerator, t_denominator = ee + bb, cc

    if t_numerator < 0.0:
        t_numerator = 0.0
        if -dd < 0.0:
            s_numerator = 0.0
        elif -dd > aa:
            s_numerator = s_denominator
        else:
            s_numerator, s_denominator = -dd, aa
    elif t_numerator > t_denominator:
        t_numerator = t_denominator
        if -dd + bb < 0.0:
            s_numerator = 0.0
        elif -dd + bb > aa:
            s_numerator = s_denominator
        else:
            s_numerator, s_denominator = -dd + bb, aa
    first_fraction = 0.0 if abs(s_numerator) <= epsilon else s_numerator / s_denominator
    second_fraction = 0.0 if abs(t_numerator) <= epsilon else t_numerator / t_denominator
    separation = between + first_fraction * first - second_fraction * second
    return float(np.linalg.norm(separation))


class FixedSelfCollisionGuard:
    """Continuously gate small joint commands around fixed Go2W fixtures."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        model_path: str | Path,
        base_link: str = "piper_base_link",
        tip_link: str = "piper_gripper_base",
    ) -> None:
        self.chain = KinematicChain.from_urdf(urdf_path, base_link, tip_link)
        if self.chain.dof != 6:
            raise ValueError(f"fixed collision guard requires six arm joints, got {self.chain.dof}")
        raw = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.clearance_m = float(raw.get("scene_clearance_m", 0.0))
        if not math.isfinite(self.clearance_m) or self.clearance_m < 0.0:
            raise ValueError("collision clearance must be finite and non-negative")
        capsules: list[FixedCapsule] = []
        for item in raw.get("capsules", ()):
            radius = float(item["radius"])
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("collision capsule radius must be finite and positive")
            capsules.append(FixedCapsule(
                name=str(item["name"]),
                start_frame=str(item["start_frame"]),
                end_frame=str(item["end_frame"]),
                radius_m=radius,
                start_offset=_vector(item.get("start_offset", (0.0, 0.0, 0.0)), "start_offset"),
                end_offset=_vector(item.get("end_offset", (0.0, 0.0, 0.0)), "end_offset"),
                supplemental=bool(item.get("supplemental_self_collision", False)),
            ))
        if not capsules:
            raise ValueError("collision model contains no capsules")
        self.capsules = tuple(capsules)
        by_name = {capsule.name: capsule for capsule in capsules}
        if len(by_name) != len(capsules):
            raise ValueError("collision capsule names must be unique")
        requested_frames = {
            frame
            for capsule in capsules
            for frame in (capsule.start_frame, capsule.end_frame)
        }
        chain_frames = {self.chain.base_link, self.chain.tip_link}
        for joint in self.chain.joints:
            chain_frames.update((joint.parent, joint.child))
        missing_frames = requested_frames - chain_frames
        if missing_frames:
            raise ValueError(f"collision capsules reference unknown frames: {sorted(missing_frames)}")

        configured = raw.get("self_collision", {}).get("pairs", ())
        pairs = itertools.combinations(by_name, 2) if configured is None else configured
        ignored = {
            _normalized_pair(pair)
            for pair in raw.get("self_collision", {}).get("ignore_pairs", ())
        }
        parsed: list[tuple[str, str]] = []
        for pair in pairs:
            normalized = _normalized_pair(pair)
            unknown = set(normalized) - set(by_name)
            if unknown:
                raise ValueError(f"collision pair references unknown capsules: {sorted(unknown)}")
            if normalized in ignored:
                continue
            first, second = (by_name[name] for name in normalized)
            if first.supplemental or second.supplemental:
                parsed.append(normalized)
        self.pairs = tuple(dict.fromkeys(parsed))
        if not self.pairs:
            raise ValueError("collision model contains no supplemental fixed-fixture pairs")

    def _world_capsules(self, joints: object) -> Mapping[str, tuple[np.ndarray, np.ndarray, float]]:
        values = np.asarray(joints, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("joint state must be a finite six-vector")
        frames = self.chain.link_transforms(values)
        result: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for capsule in self.capsules:
            start_transform = frames[capsule.start_frame]
            end_transform = frames[capsule.end_frame]
            start = start_transform[:3, :3] @ capsule.start_offset + start_transform[:3, 3]
            end = end_transform[:3, :3] @ capsule.end_offset + end_transform[:3, 3]
            result[capsule.name] = (start, end, capsule.radius_m)
        return result

    def check_state(self, joints: object) -> CollisionState:
        world = self._world_capsules(joints)
        closest: CollisionWitness | None = None
        for first_name, second_name in self.pairs:
            first_start, first_end, first_radius = world[first_name]
            second_start, second_end, second_radius = world[second_name]
            distance = _segment_distance(first_start, first_end, second_start, second_end)
            threshold = first_radius + second_radius + self.clearance_m
            witness = CollisionWitness(
                pair=(first_name, second_name),
                distance_m=distance,
                threshold_m=threshold,
                margin_m=distance - threshold,
            )
            if closest is None or witness.margin_m < closest.margin_m:
                closest = witness
        assert closest is not None
        return CollisionState(
            valid=closest.margin_m > 0.0,
            minimum_margin_m=closest.margin_m,
            witness=closest,
        )

    def check_step(
        self,
        current_joints: object,
        target_joints: object,
        *,
        max_joint_step_rad: float = 0.01,
        escape_improvement_m: float = 1e-5,
    ) -> StepDecision:
        current = np.asarray(current_joints, dtype=float)
        target = np.asarray(target_joints, dtype=float)
        if current.shape != (6,) or target.shape != (6,):
            raise ValueError("collision step endpoints must be six-vectors")
        if not np.isfinite(current).all() or not np.isfinite(target).all():
            raise ValueError("collision step endpoints must be finite")
        if not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be finite and positive")
        current_state = self.check_state(current)
        target_state = self.check_state(target)
        sample_count = max(1, int(math.ceil(float(np.linalg.norm(target - current)) / max_joint_step_rad)))
        states = [
            self.check_state(current + alpha * (target - current))
            for alpha in np.linspace(0.0, 1.0, sample_count + 1)
        ]
        if current_state.valid:
            blocked = next((state for state in states[1:] if not state.valid), None)
            if blocked is not None:
                return StepDecision(
                    False,
                    False,
                    "reactive joint step enters a fixed-fixture collision",
                    blocked.witness,
                    current_state.minimum_margin_m,
                    target_state.minimum_margin_m,
                )
            return StepDecision(
                True,
                False,
                "collision-free reactive joint step",
                target_state.witness,
                current_state.minimum_margin_m,
                target_state.minimum_margin_m,
            )

        # A stale model or manually recovered pose can begin just inside the
        # conservative envelope.  Do not trap the arm there: admit only a
        # monotonic escape that increases the global minimum clearance.
        previous = current_state.minimum_margin_m
        for state in states[1:]:
            if state.minimum_margin_m + escape_improvement_m < previous:
                return StepDecision(
                    False,
                    False,
                    "reactive joint step deepens a fixed-fixture collision",
                    state.witness,
                    current_state.minimum_margin_m,
                    target_state.minimum_margin_m,
                )
            previous = state.minimum_margin_m
        escaping = target_state.minimum_margin_m >= (
            current_state.minimum_margin_m + escape_improvement_m
        )
        return StepDecision(
            escaping,
            escaping,
            (
                "monotonic escape from fixed-fixture envelope"
                if escaping
                else "reactive joint step does not escape fixed-fixture collision"
            ),
            target_state.witness,
            current_state.minimum_margin_m,
            target_state.minimum_margin_m,
        )
