"""Robot-agnostic, collision-aware bidirectional RRT-Connect fallback."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning_control import (
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)


class RRTTimeout(PlanningError):
    """One RRT call exhausted its recoverable relative timeout."""


class _RRTBudget:
    """Combine a recoverable local timeout with caller-owned abort control."""

    def __init__(
        self,
        timeout_s: float,
        parent: PlanningControl | None,
    ) -> None:
        self._parent = parent
        self._clock = time.monotonic if parent is None else parent.monotonic_fn
        if parent is not None:
            parent.checkpoint("RRT planning")
        self._timeout_s = float(timeout_s)
        self._deadline_s = self._now("RRT planning") + self._timeout_s

    def _now(self, operation: str) -> float:
        try:
            now = float(self._clock())
        except Exception as error:
            raise RRTTimeout(
                f"{operation} local timeout clock failed closed: "
                f"{type(error).__name__}: {error}",
            ) from error
        if not math.isfinite(now):
            raise RRTTimeout(
                f"{operation} local timeout clock returned a non-finite value",
            )
        if (
            self._parent is not None
            and self._parent.deadline_s is not None
            and now >= float(self._parent.deadline_s)
        ):
            raise PlanningDeadlineExceeded(
                f"{operation} exceeded its monotonic deadline "
                f"({float(self._parent.deadline_s):.6f} s)",
            )
        return now

    def checkpoint(self, operation: str = "RRT planning") -> None:
        if self._parent is not None:
            self._parent.checkpoint(operation)
        if self._now(operation) >= self._deadline_s:
            raise RRTTimeout(
                f"{operation} exceeded the recoverable local RRT timeout "
                f"({self._timeout_s:.3f} s)",
            )


@dataclass(frozen=True)
class RRTConnectConfig:
    step_size: float = 0.18
    collision_resolution: float = 0.035
    max_iterations: int = 4000
    goal_bias: float = 0.08
    shortcut_attempts: int = 80
    seed: int = 0

    def __post_init__(self) -> None:
        if self.step_size <= 0.0 or self.collision_resolution <= 0.0:
            raise ValueError("RRT step size and collision resolution must be positive")
        if self.max_iterations < 1 or self.shortcut_attempts < 0:
            raise ValueError("RRT iteration counts must be non-negative")
        if not 0.0 <= self.goal_bias <= 1.0:
            raise ValueError("RRT goal bias must be in [0, 1]")


class _Tree:
    def __init__(self, root: np.ndarray, *, rooted_at_start: bool):
        self.nodes = [root.copy()]
        self.parents = [-1]
        self.rooted_at_start = rooted_at_start

    def add(self, joints: np.ndarray, parent: int) -> int:
        self.nodes.append(joints.copy())
        self.parents.append(parent)
        return len(self.nodes) - 1

    def path(self, index: int) -> list[np.ndarray]:
        path = []
        while index >= 0:
            path.append(self.nodes[index])
            index = self.parents[index]
        return list(reversed(path))


class JointSpaceRRTConnect:
    """Plan within arbitrary joint limits using a caller-owned validity check.

    ``state_valid`` is the scalability boundary: deployments can supply MoveIt
    planning-scene validity, a point-cloud/capsule checker, or a hardware-model
    checker without changing RRT or task orchestration.
    """

    _TRAPPED = 0
    _ADVANCED = 1
    _REACHED = 2

    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        state_valid: Callable[[np.ndarray], bool],
        config: RRTConnectConfig | None = None,
    ) -> None:
        self.joint_names = tuple(joint_names)
        self.lower_limits = np.asarray(lower_limits, dtype=float)
        self.upper_limits = np.asarray(upper_limits, dtype=float)
        self.state_valid = state_valid
        self.config = config or RRTConnectConfig()
        expected = (len(self.joint_names),)
        if self.lower_limits.shape != expected or self.upper_limits.shape != expected:
            raise ValueError(f"joint limits must have shape {expected}")
        if np.any(self.lower_limits >= self.upper_limits):
            raise ValueError("every lower joint limit must be below its upper limit")
        self._span = self.upper_limits - self.lower_limits

    def _validate_state(
        self,
        joints: object,
        label: str,
        control: PlanningControl | None = None,
    ) -> np.ndarray:
        checkpoint(control, f"RRT {label} validation")
        values = np.asarray(joints, dtype=float)
        expected = (len(self.joint_names),)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise PlanningError(f"{label} joint state must be a finite {expected} vector")
        if np.any(values < self.lower_limits) or np.any(values > self.upper_limits):
            raise PlanningError(f"{label} joint state violates joint limits")
        valid = self.state_valid(values)
        checkpoint(control, f"RRT {label} validation")
        if not valid:
            raise PlanningError(f"{label} joint state is in collision")
        return values

    def _distance(self, first: np.ndarray, second: np.ndarray) -> float:
        # Normalizing by range prevents a wide wrist joint from dominating a
        # short prismatic or narrow-range arm joint.
        return float(np.linalg.norm((second - first) / self._span))

    def _segment_valid(
        self,
        first: np.ndarray,
        second: np.ndarray,
        control: PlanningControl | None = None,
    ) -> bool:
        distance = float(np.linalg.norm(second - first))
        steps = max(1, int(np.ceil(distance / self.config.collision_resolution)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            checkpoint(control, "RRT segment collision checking")
            valid = self.state_valid(first + alpha * (second - first))
            checkpoint(control, "RRT segment collision checking")
            if not valid:
                return False
        return True

    def segment_valid(
        self,
        first: object,
        second: object,
        *,
        control: PlanningControl | None = None,
    ) -> bool:
        """Continuously validate one in-limit joint-space segment."""

        checkpoint(control, "RRT segment validation")
        first_values = np.asarray(first, dtype=float)
        second_values = np.asarray(second, dtype=float)
        expected = (len(self.joint_names),)
        if first_values.shape != expected or second_values.shape != expected:
            return False
        if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
            return False
        if (
            np.any(first_values < self.lower_limits)
            or np.any(first_values > self.upper_limits)
            or np.any(second_values < self.lower_limits)
            or np.any(second_values > self.upper_limits)
        ):
            return False
        first_valid = self.state_valid(first_values)
        checkpoint(control, "RRT segment validation")
        return bool(
            first_valid
            and self._segment_valid(first_values, second_values, control)
        )

    def _nearest(
        self,
        tree: _Tree,
        target: np.ndarray,
        control: PlanningControl | None = None,
    ) -> int:
        distances = []
        for node in tree.nodes:
            checkpoint(control, "RRT nearest-neighbor search")
            distances.append(self._distance(node, target))
        return int(np.argmin(distances))

    def _extend(
        self,
        tree: _Tree,
        target: np.ndarray,
        control: PlanningControl | None = None,
    ) -> tuple[int | None, int]:
        checkpoint(control, "RRT tree extension")
        nearest_index = self._nearest(tree, target, control)
        nearest = tree.nodes[nearest_index]
        delta = target - nearest
        distance = float(np.linalg.norm(delta))
        if distance < 1e-12:
            return nearest_index, self._REACHED
        if distance <= self.config.step_size:
            proposed, status = target, self._REACHED
        else:
            proposed = nearest + delta * (self.config.step_size / distance)
            status = self._ADVANCED
        if not self._segment_valid(nearest, proposed, control):
            return None, self._TRAPPED
        return tree.add(proposed, nearest_index), status

    def _connect(
        self,
        tree: _Tree,
        target: np.ndarray,
        control: PlanningControl | None = None,
    ) -> tuple[int | None, int]:
        last_index = None
        while True:
            checkpoint(control, "RRT tree connection")
            index, status = self._extend(tree, target, control)
            if status == self._TRAPPED:
                return last_index, status
            last_index = index
            if status == self._REACHED:
                return last_index, status

    @staticmethod
    def _join(first: _Tree, first_index: int, second: _Tree, second_index: int) -> list[np.ndarray]:
        first_path = first.path(first_index)
        second_path = second.path(second_index)
        if first.rooted_at_start:
            start_path, goal_path = first_path, second_path
        else:
            start_path, goal_path = second_path, first_path
        return start_path + list(reversed(goal_path))[1:]

    def _shortcut(
        self,
        path: list[np.ndarray],
        rng: np.random.Generator,
        control: PlanningControl | None = None,
    ) -> list[np.ndarray]:
        result = list(path)
        for _ in range(self.config.shortcut_attempts):
            checkpoint(control, "RRT path shortcutting")
            if len(result) < 3:
                break
            first, second = sorted(rng.choice(len(result), size=2, replace=False))
            if second <= first + 1:
                continue
            if self._segment_valid(result[first], result[second], control):
                result = result[: first + 1] + result[second:]
        return result

    def _densify(
        self,
        path: list[np.ndarray],
        control: PlanningControl | None = None,
    ) -> np.ndarray:
        dense = [path[0]]
        for first, second in zip(path, path[1:]):
            checkpoint(control, "RRT path densification")
            distance = float(np.linalg.norm(second - first))
            steps = max(1, int(np.ceil(distance / self.config.collision_resolution)))
            for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
                checkpoint(control, "RRT path densification")
                dense.append(first + alpha * (second - first))
        return np.asarray(dense)

    def plan_joint(
        self,
        start_joints: object,
        goal_joints: object,
        *,
        timeout_s: float = 5.0,
        control: PlanningControl | None = None,
    ) -> JointTrajectory:
        """Return a dense, collision-checked joint path from start to goal."""

        if not math.isfinite(float(timeout_s)) or timeout_s <= 0.0:
            raise PlanningError("planning timeout must be positive")
        budget = _RRTBudget(timeout_s, control)
        start = self._validate_state(start_joints, "start", budget)
        goal = self._validate_state(goal_joints, "goal", budget)
        if self._segment_valid(start, goal, budget):
            return JointTrajectory(
                self.joint_names,
                self._densify([start, goal], budget),
            )

        rng = np.random.default_rng(self.config.seed)
        first = _Tree(start, rooted_at_start=True)
        second = _Tree(goal, rooted_at_start=False)
        for _ in range(self.config.max_iterations):
            checkpoint(budget, "RRT planning")
            if rng.random() < self.config.goal_bias:
                sample = second.nodes[0]
            else:
                sample = rng.uniform(self.lower_limits, self.upper_limits)
            first_index, status = self._extend(first, sample, budget)
            if status != self._TRAPPED and first_index is not None:
                second_index, connect_status = self._connect(
                    second,
                    first.nodes[first_index],
                    budget,
                )
                if connect_status == self._REACHED and second_index is not None:
                    path = self._join(first, first_index, second, second_index)
                    path = self._shortcut(path, rng, budget)
                    dense = self._densify(path, budget)
                    for joints in dense:
                        checkpoint(budget, "RRT final collision audit")
                        valid = self.state_valid(joints)
                        checkpoint(budget, "RRT final collision audit")
                        if not valid:
                            raise PlanningError(
                                "internal error: path failed final collision audit",
                            )
                    return JointTrajectory(self.joint_names, dense)
            first, second = second, first
        raise PlanningError(
            "no collision-free joint path found within "
            f"{self.config.max_iterations} iterations / {timeout_s:.3f} s",
        )
