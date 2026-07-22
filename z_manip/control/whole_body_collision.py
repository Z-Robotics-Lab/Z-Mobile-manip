"""Pure proactive collision selection for one whole-body arm horizon.

The optimizer proposes a short arm velocity.  This module checks the complete
joint-space edge against the injected fixed-fixture guard, then tries bounded
same-side reductions before reflecting the tool's lateral velocity to the
opposite side.  It contains no ROS, CAN, WebRTC, or actuator imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

import numpy as np


ARM_DOF = 6
_CURRENT_SIDE_SCALES = (0.75, 0.50, 0.25, 0.125)


class CollisionWitnessLike(Protocol):
    pair: tuple[str, str]
    margin_m: float


class StepDecisionLike(Protocol):
    allowed: bool
    escaping: bool
    reason: str
    witness: CollisionWitnessLike
    current_margin_m: float
    target_margin_m: float


class ArmStepCollisionGuard(Protocol):
    def check_state(self, joints: object) -> object: ...

    def check_step(
        self,
        current_joints: object,
        target_joints: object,
    ) -> StepDecisionLike: ...


def _escape_candidates(
    current: np.ndarray,
    *,
    horizon_dt_s: float,
    guard: ArmStepCollisionGuard,
    speed_rps: float = 0.10,
    probe_rad: float = 0.002,
) -> tuple[tuple[str, str, np.ndarray], ...]:
    """Return bounded directions that increase fixed-fixture clearance.

    Clearance is a non-smooth minimum over capsule pairs, so a gradient alone
    can vanish at a pair switch.  Try the normalized finite-difference
    gradient first, followed by the individual signed joint directions ordered
    by their predicted clearance.  These candidates are used only when the
    measured state already lies inside the conservative envelope.
    """

    check_state = getattr(guard, "check_state", None)
    if not callable(check_state):
        return ()
    try:
        current_margin = float(check_state(current).minimum_margin_m)
    except (AttributeError, TypeError, ValueError):
        return ()
    if not math.isfinite(current_margin) or current_margin >= 0.0:
        return ()

    gradient = np.zeros(ARM_DOF)
    axes: list[tuple[float, str, str, np.ndarray]] = []
    for index in range(ARM_DOF):
        plus = current.copy()
        minus = current.copy()
        plus[index] += probe_rad
        minus[index] -= probe_rad
        try:
            plus_margin = float(check_state(plus).minimum_margin_m)
            minus_margin = float(check_state(minus).minimum_margin_m)
        except (AttributeError, TypeError, ValueError):
            continue
        if not (math.isfinite(plus_margin) and math.isfinite(minus_margin)):
            continue
        gradient[index] = (plus_margin - minus_margin) / (2.0 * probe_rad)
        for sign, margin in ((1.0, plus_margin), (-1.0, minus_margin)):
            velocity = np.zeros(ARM_DOF)
            velocity[index] = sign * speed_rps
            axes.append((margin, f"collision_escape_joint_{index + 1}", _side(sign), velocity))

    candidates: list[tuple[str, str, np.ndarray]] = []
    maximum = float(np.max(np.abs(gradient)))
    if maximum > 1e-12:
        velocity = speed_rps * gradient / maximum
        candidates.append(("collision_escape_gradient", "clearance", velocity))
    for _margin, strategy, side, velocity in sorted(axes, key=lambda item: item[0], reverse=True):
        candidates.append((strategy, side, velocity))
    return tuple(candidates)


@dataclass(frozen=True)
class CollisionAttempt:
    strategy: str
    side: str
    allowed_by_geometry: bool
    task_improved: bool
    escaping: bool
    reason: str
    pair: tuple[str, str]
    current_margin_m: float
    target_margin_m: float

    def document(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "side": self.side,
            "allowed_by_geometry": self.allowed_by_geometry,
            "task_improved": self.task_improved,
            "escaping": self.escaping,
            "reason": self.reason,
            "pair": list(self.pair),
            "current_margin_m": self.current_margin_m,
            "target_margin_m": self.target_margin_m,
        }


@dataclass(frozen=True)
class CollisionGateSelection:
    allowed: bool
    strategy: str
    current_side: str
    selected_side: str | None
    arm_velocity_rps: np.ndarray
    attempts: tuple[CollisionAttempt, ...]

    @property
    def selected_attempt(self) -> CollisionAttempt | None:
        if not self.allowed:
            return None
        return self.attempts[-1]

    def document(self) -> dict[str, object]:
        selected = self.selected_attempt
        return {
            "checked": True,
            "allowed": self.allowed,
            "strategy": self.strategy,
            "current_side": self.current_side,
            "selected_side": self.selected_side,
            "selected_pair": None if selected is None else list(selected.pair),
            "current_margin_m": (
                None if selected is None else selected.current_margin_m
            ),
            "target_margin_m": (
                None if selected is None else selected.target_margin_m
            ),
            "attempts": [attempt.document() for attempt in self.attempts],
        }


def _vector6(value: object, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (ARM_DOF,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite six-vector")
    return result


def _side(rate: float, *, tolerance: float = 1e-9) -> str:
    if rate > tolerance:
        return "left"
    if rate < -tolerance:
        return "right"
    return "neutral"


def _opposite_lateral_velocity(
    primary: np.ndarray,
    lateral_jacobian: np.ndarray,
) -> np.ndarray | None:
    norm_squared = float(lateral_jacobian @ lateral_jacobian)
    lateral_rate = float(lateral_jacobian @ primary)
    if norm_squared <= 1e-14 or abs(lateral_rate) <= 1e-9:
        return None
    # Reflect the joint velocity across the zero lateral-tool-velocity plane.
    # Orthogonal view-task components are retained while the local lateral
    # rate changes sign.
    return primary - 2.0 * lateral_rate / norm_squared * lateral_jacobian


def _candidate_velocities(
    primary: np.ndarray,
    lateral_jacobian: np.ndarray,
) -> tuple[tuple[str, str, np.ndarray], ...]:
    primary_rate = float(lateral_jacobian @ primary)
    current_side = _side(primary_rate)
    candidates: list[tuple[str, str, np.ndarray]] = [
        ("current_side", current_side, primary),
    ]
    candidates.extend(
        (
            f"current_side_scale_{scale:.3f}",
            current_side,
            scale * primary,
        )
        for scale in _CURRENT_SIDE_SCALES
    )
    opposite = _opposite_lateral_velocity(primary, lateral_jacobian)
    if opposite is not None:
        opposite_side = _side(float(lateral_jacobian @ opposite))
        candidates.append(("opposite_side", opposite_side, opposite))
        candidates.extend(
            (
                f"opposite_side_scale_{scale:.3f}",
                opposite_side,
                scale * opposite,
            )
            for scale in _CURRENT_SIDE_SCALES
        )
    return tuple(candidates)


def select_collision_safe_arm_step(
    *,
    current_joints: object,
    primary_arm_velocity: object,
    horizon_dt_s: float,
    tool_lateral_jacobian: object,
    guard: ArmStepCollisionGuard,
    candidate_improves_task: Callable[[np.ndarray], bool],
) -> CollisionGateSelection:
    """Choose the first continuous collision-safe, task-improving arm step.

    Candidate order deliberately preserves the optimizer's current lateral
    side first.  If the measured state is already barely inside a conservative
    fixed-fixture envelope, an explicit monotonic clearance escape is allowed
    even when it temporarily worsens the visual task.  This prevents a model
    boundary from trapping the arm while retaining the final collision guard.
    """

    current = _vector6(current_joints, label="current joints")
    primary = _vector6(primary_arm_velocity, label="primary arm velocity")
    lateral = _vector6(tool_lateral_jacobian, label="tool lateral Jacobian")
    dt = float(horizon_dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("collision horizon must be finite and positive")
    current_side = _side(float(lateral @ primary))
    attempts: list[CollisionAttempt] = []
    for strategy, side, velocity in _candidate_velocities(primary, lateral):
        target = current + dt * velocity
        decision = guard.check_step(current, target)
        task_improved = False
        if decision.allowed:
            try:
                task_improved = bool(candidate_improves_task(velocity.copy()))
            except (RuntimeError, ValueError):
                task_improved = False
        attempt = CollisionAttempt(
            strategy=strategy,
            side=side,
            allowed_by_geometry=bool(decision.allowed),
            task_improved=task_improved,
            escaping=bool(decision.escaping),
            reason=str(decision.reason),
            pair=tuple(str(name) for name in decision.witness.pair),
            current_margin_m=float(decision.current_margin_m),
            target_margin_m=float(decision.target_margin_m),
        )
        attempts.append(attempt)
        if decision.allowed and task_improved:
            return CollisionGateSelection(
                allowed=True,
                strategy=strategy,
                current_side=current_side,
                selected_side=side,
                arm_velocity_rps=velocity.copy(),
                attempts=tuple(attempts),
            )

    # A task optimizer cannot be expected to move *away* from its target to
    # recover clearance.  Once ordinary task-improving candidates fail, try a
    # short bounded escape and accept it solely on monotonic geometry proof.
    for strategy, side, velocity in _escape_candidates(
        current,
        horizon_dt_s=dt,
        guard=guard,
    ):
        target = current + dt * velocity
        decision = guard.check_step(current, target)
        try:
            task_improved = bool(candidate_improves_task(velocity.copy()))
        except (RuntimeError, ValueError):
            task_improved = False
        attempt = CollisionAttempt(
            strategy=strategy,
            side=side,
            allowed_by_geometry=bool(decision.allowed),
            task_improved=task_improved,
            escaping=bool(decision.escaping),
            reason=str(decision.reason),
            pair=tuple(str(name) for name in decision.witness.pair),
            current_margin_m=float(decision.current_margin_m),
            target_margin_m=float(decision.target_margin_m),
        )
        attempts.append(attempt)
        if decision.allowed and decision.escaping:
            return CollisionGateSelection(
                allowed=True,
                strategy=strategy,
                current_side=current_side,
                selected_side=side,
                arm_velocity_rps=velocity.copy(),
                attempts=tuple(attempts),
            )
    return CollisionGateSelection(
        allowed=False,
        strategy="fail_closed",
        current_side=current_side,
        selected_side=None,
        arm_velocity_rps=np.zeros(ARM_DOF),
        attempts=tuple(attempts),
    )


__all__ = [
    "ArmStepCollisionGuard",
    "CollisionAttempt",
    "CollisionGateSelection",
    "select_collision_safe_arm_step",
]
