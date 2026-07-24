"""Bounded, deterministic multi-start inverse kinematics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .chain import KinematicChain, rotation_log
from z_manip.planning_control import (
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)


@dataclass(frozen=True)
class IKConfig:
    """Numerical IK settings independent of a specific robot model."""

    position_tolerance_m: float = 0.003
    orientation_tolerance_rad: float = 0.03
    max_iterations: int = 220
    random_seeds: int = 12
    position_scale_m: float = 0.10
    orientation_scale_rad: float = 0.35
    current_distance_weight: float = 0.015
    joint_limit_weight: float = 0.003
    solve_timeout_s: float = 0.60
    seed_timeout_s: float = 0.15
    max_feasible_solutions: int = 2
    solution_refinement_timeout_s: float = 0.05
    continuation_timeout_s: float = 0.18
    continuation_seed_timeout_s: float = 0.08
    continuation_fallback_seeds: int = 2
    # Optional, default-off anisotropic acceptance (FROZEN-SAFETY study flag).
    # When ``orientation_free_axis_tolerance_rad`` is positive, orientation
    # acceptance is split: the residual rotation *about* ``orientation_free_axis``
    # (expressed in the goal tip frame) may reach this relaxed tolerance while
    # the transverse residual — which carries finger closing-line alignment —
    # must still stay within ``orientation_tolerance_rad``.  Zero preserves the
    # historical isotropic geodesic gate exactly.
    orientation_free_axis_tolerance_rad: float = 0.0
    orientation_free_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    # Control point (tip frame) at which the POSITION tolerance is enforced.
    # The grasp contact TCP sits ``contact_tcp_z_m`` past the tip link, so a
    # tip-frame orientation residual levers into contact-point translation the
    # historical tip-position gate never saw (0.236 rad inside a loose gate put
    # the PiPER fingers 27 mm off a verified live grasp, 2026-07-24).  With the
    # offset wired to the tool contact point, an accepted solution bounds the
    # finger placement itself.  Zero preserves the historical tip gate exactly.
    position_error_offset_tip_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        positive = (
            self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.position_scale_m,
            self.orientation_scale_rad,
            self.solve_timeout_s,
            self.seed_timeout_s,
            self.solution_refinement_timeout_s,
            self.continuation_timeout_s,
            self.continuation_seed_timeout_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "IK tolerances, residual scales, and timeouts must be positive",
            )
        if self.max_iterations < 1 or self.random_seeds < 0:
            raise ValueError("IK iteration and seed counts must be non-negative")
        if self.max_feasible_solutions < 1:
            raise ValueError("IK feasible-solution count must be positive")
        if self.continuation_fallback_seeds < 0:
            raise ValueError("IK continuation fallback seed count cannot be negative")
        if (
            not np.isfinite(self.orientation_free_axis_tolerance_rad)
            or self.orientation_free_axis_tolerance_rad < 0.0
        ):
            raise ValueError(
                "orientation_free_axis_tolerance_rad must be finite and non-negative",
            )
        free_axis = np.asarray(self.orientation_free_axis, dtype=float)
        if free_axis.shape != (3,) or not np.all(np.isfinite(free_axis)) or float(
            np.linalg.norm(free_axis)
        ) <= 1e-9:
            raise ValueError(
                "orientation_free_axis must be a finite nonzero 3-vector",
            )
        offset = np.asarray(self.position_error_offset_tip_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(
                "position_error_offset_tip_m must be a finite 3-vector",
            )


@dataclass(frozen=True)
class IKSolution:
    joints: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    manipulability: float
    iterations: int
    seed_index: int
    min_joint_limit_margin: float = 0.0


class IKFailure(RuntimeError):
    """No in-limit joint configuration met the Cartesian tolerances."""


def control_point_delta(
    actual_position: np.ndarray,
    actual_rotation: np.ndarray,
    goal_position: np.ndarray,
    goal_rotation: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """Position error vector measured at a tip-frame control point.

    With a zero ``offset`` this is exactly the historical tip-origin position
    error.  With the offset at the tool contact point, a transverse orientation
    residual contributes its full lever (``theta * |offset_perp|``) to the
    measured error, while rotation *about* the offset direction contributes
    nothing — matching the physics of a parallel jaw whose TCP lies along
    tool-Z.  Shared by both IK backends so acceptance stays identical.
    """

    return (actual_position + actual_rotation @ offset) - (
        goal_position + goal_rotation @ offset
    )


def _van_der_corput(index: int, base: int) -> float:
    value, denominator = 0.0, 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _halton(index: int, dimensions: int) -> np.ndarray:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)
    if dimensions > len(primes):
        raise ValueError(f"Halton seeds support at most {len(primes)} joints")
    return np.asarray(
        [_van_der_corput(index, primes[dimension]) for dimension in range(dimensions)],
        dtype=float,
    )


class RobustIKSolver:
    """Solve a Cartesian target with bounds and diversified deterministic seeds.

    Each start uses SciPy's trust-region reflective least-squares solver with
    the URDF limits as hard bounds. Low-discrepancy seeds cover the entire joint
    range repeatably, while current, midpoint and zero configurations preserve
    continuity during normal servo-rate operation.
    """

    def __init__(self, chain: KinematicChain, config: IKConfig | None = None):
        self.chain = chain
        self.config = config or IKConfig()

    def _seeds(
        self,
        current: np.ndarray | None,
        control: PlanningControl | None = None,
    ) -> list[np.ndarray]:
        lower, upper = self.chain.lower_limits, self.chain.upper_limits
        midpoint = 0.5 * (lower + upper)
        candidates: list[np.ndarray] = []
        if current is not None:
            values = np.asarray(current, dtype=float)
            if values.shape != (self.chain.dof,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"current joints must have shape ({self.chain.dof},) and be finite",
                )
            candidates.append(np.clip(values, lower, upper))
        candidates.extend((midpoint, np.clip(np.zeros(self.chain.dof), lower, upper)))
        if current is not None:
            candidates.append(np.clip(2.0 * midpoint - candidates[0], lower, upper))
        span = upper - lower
        for index in range(1, self.config.random_seeds + 1):
            checkpoint(control, "IK seed generation")
            candidates.append(lower + _halton(index, self.chain.dof) * span)

        unique: list[np.ndarray] = []
        for candidate in candidates:
            checkpoint(control, "IK seed deduplication")
            # Keep starts infinitesimally inside the trust-region bounds.
            candidate = np.minimum(np.maximum(candidate, lower + 1e-10), upper - 1e-10)
            if not any(np.allclose(candidate, other, atol=1e-10) for other in unique):
                unique.append(candidate)
        return unique

    @staticmethod
    def _validated_goal(target: object) -> np.ndarray:
        goal = np.asarray(target, dtype=float)
        if goal.shape != (4, 4) or not np.all(np.isfinite(goal)):
            raise ValueError("target must be a finite (4, 4) transform")
        if not np.allclose(goal[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
            raise ValueError("target is not a homogeneous transform")
        if not np.allclose(goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("target rotation is not orthonormal")
        return goal

    def make_seed_pose_ranker(
        self,
        current: np.ndarray | None,
        control: PlanningControl | None = None,
    ) -> Callable[..., float]:
        """Build a cheap URDF-seed FK distance used only to order IK work."""

        checkpoint(control, "IK reachability ranker setup")
        seed_poses = np.stack([
            self.chain.forward(seed)
            for seed in self._seeds(current, control)
        ])
        seed_positions = seed_poses[:, :3, 3]
        seed_rotations = seed_poses[:, :3, :3]
        position_scale = self.config.position_scale_m
        orientation_scale = self.config.orientation_scale_rad

        def rank(
            target: object,
            *,
            control: PlanningControl | None = None,
        ) -> float:
            checkpoint(control, "IK reachability ranking")
            goal = self._validated_goal(target)
            position_cost = np.linalg.norm(
                seed_positions - goal[:3, 3],
                axis=1,
            ) / position_scale
            relative = np.matmul(
                goal[None, :3, :3],
                np.transpose(seed_rotations, (0, 2, 1)),
            )
            cosine = np.clip(
                (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
            orientation_angle = np.where(
                cosine >= 1.0 - 1e-12,
                0.0,
                np.arccos(cosine),
            )
            orientation_cost = orientation_angle / orientation_scale
            result = float(np.min(np.hypot(position_cost, orientation_cost)))
            checkpoint(control, "IK reachability ranking")
            return result

        checkpoint(control, "IK reachability ranker setup")
        return rank

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        *,
        control: PlanningControl | None = None,
    ) -> IKSolution:
        checkpoint(control, "inverse kinematics")
        goal = self._validated_goal(target)

        config = self.config
        local_control = control or PlanningControl()
        seeds = self._seeds(current, local_control)
        solve_control = local_control.limited_to(
            config.solve_timeout_s,
            "inverse kinematics solve budget",
        )
        offset = np.asarray(config.position_error_offset_tip_m, dtype=float)

        def residual_for(seed_control: PlanningControl | None):
            def residual(joints: np.ndarray) -> np.ndarray:
                # SciPy can otherwise spend the whole aggregate task budget in
                # one trust-region solve. Each seed inherits cancellation and
                # the aggregate deadline through a shorter child control.
                checkpoint(seed_control, "inverse kinematics numerical solve")
                actual = self.chain.forward(joints)
                position = control_point_delta(
                    actual[:3, 3], actual[:3, :3], goal[:3, 3], goal[:3, :3], offset,
                ) / config.position_scale_m
                orientation = rotation_log(goal[:3, :3] @ actual[:3, :3].T)
                return np.concatenate((position, orientation / config.orientation_scale_rad))

            return residual

        reference = (
            np.asarray(current, dtype=float)
            if current is not None
            else 0.5 * (self.chain.lower_limits + self.chain.upper_limits)
        )
        span = np.maximum(self.chain.upper_limits - self.chain.lower_limits, 1e-9)
        feasible: list[tuple[float, IKSolution]] = []
        best_errors: tuple[float, float] | None = None
        seed_timeouts = 0
        solve_budget_exhausted = False
        refinement_control: PlanningControl | None = None
        for seed_index, seed in enumerate(seeds):
            active_control = refinement_control or solve_control
            try:
                checkpoint(active_control, "inverse kinematics candidate search")
                seed_control = active_control.limited_to(
                    config.seed_timeout_s,
                    "inverse kinematics seed budget",
                )
            except PlanningDeadlineExceeded:
                # The caller-owned deadline always wins. Expiring the optional
                # improvement window preserves the best feasible seed already
                # found; expiring the full solve budget rejects this pose.
                checkpoint(control, "inverse kinematics candidate search")
                if refinement_control is not None:
                    break
                solve_budget_exhausted = True
                break
            try:
                result = least_squares(
                    residual_for(seed_control),
                    seed,
                    bounds=(self.chain.lower_limits, self.chain.upper_limits),
                    method="trf",
                    jac="3-point",
                    x_scale="jac",
                    ftol=1e-11,
                    xtol=1e-11,
                    gtol=1e-11,
                    max_nfev=config.max_iterations,
                )
                checkpoint(seed_control, "inverse kinematics candidate search")
            except PlanningDeadlineExceeded:
                checkpoint(control, "inverse kinematics numerical solve")
                seed_timeouts += 1
                if refinement_control is not None:
                    try:
                        checkpoint(
                            refinement_control,
                            "inverse kinematics solution refinement",
                        )
                    except PlanningDeadlineExceeded:
                        break
                try:
                    checkpoint(solve_control, "inverse kinematics solve budget")
                except PlanningDeadlineExceeded:
                    solve_budget_exhausted = True
                    break
                continue
            actual = self.chain.forward(result.x)
            position_error = float(np.linalg.norm(control_point_delta(
                actual[:3, 3], actual[:3, :3], goal[:3, 3], goal[:3, :3], offset,
            )))
            orientation_error = float(
                np.linalg.norm(rotation_log(goal[:3, :3] @ actual[:3, :3].T)),
            )
            if best_errors is None or (position_error + orientation_error) < sum(best_errors):
                best_errors = (position_error, orientation_error)
            if (
                position_error >= config.position_tolerance_m
                or orientation_error >= config.orientation_tolerance_rad
            ):
                continue

            checkpoint(control, "inverse kinematics solution scoring")
            singular_values = np.linalg.svd(self.chain.jacobian(result.x), compute_uv=False)
            manipulability = float(np.prod(np.maximum(singular_values, 0.0)))
            normalized_distance = float(np.linalg.norm((result.x - reference) / span))
            normalized_margin = np.minimum(
                (result.x - self.chain.lower_limits) / span,
                (self.chain.upper_limits - result.x) / span,
            )
            score = (
                position_error / config.position_tolerance_m
                + orientation_error / config.orientation_tolerance_rad
                + config.current_distance_weight * normalized_distance
                + config.joint_limit_weight / (float(np.min(normalized_margin)) + 1e-3)
            )
            feasible.append(
                (
                    score,
                    IKSolution(
                        joints=result.x.copy(),
                        position_error_m=position_error,
                        orientation_error_rad=orientation_error,
                        manipulability=manipulability,
                        iterations=int(result.nfev),
                        seed_index=seed_index,
                        min_joint_limit_margin=float(np.min(normalized_margin)),
                    ),
                ),
            )
            if len(feasible) >= config.max_feasible_solutions:
                break
            if refinement_control is None:
                try:
                    refinement_control = solve_control.limited_to(
                        config.solution_refinement_timeout_s,
                        "inverse kinematics solution refinement",
                    )
                except PlanningDeadlineExceeded:
                    # A valid seed outranks exhaustion of the optional local
                    # improvement window; caller-owned aborts still propagate.
                    checkpoint(control, "inverse kinematics solution refinement")
                    break
        if feasible:
            checkpoint(control, "inverse kinematics solution selection")
            return min(feasible, key=lambda item: item[0])[1]
        position, orientation = best_errors or (float("inf"), float("inf"))
        raise IKFailure(
            "no IK solution met tolerances "
            f"(best position={position:.6f} m, orientation={orientation:.6f} rad; "
            f"seed_timeouts={seed_timeouts}; "
            f"solve_budget_exhausted={str(solve_budget_exhausted).lower()})",
        )

    def solve_continuation(
        self,
        target: np.ndarray,
        current: np.ndarray,
        *,
        max_joint_step_rad: float,
        control: PlanningControl | None = None,
    ) -> IKSolution:
        """Track a nearby Cartesian target without leaving the current IK branch.

        Unlike :meth:`solve`, this method never introduces a robot-wide seed.
        The previous solution is attempted first inside a per-joint trust region.
        Only if that local solve fails are a bounded number of deterministic seeds
        sampled inside the same region. Feasible fallbacks are selected
        lexicographically by joint continuity before Cartesian residual quality.
        """

        checkpoint(control, "Cartesian IK continuation")
        goal = self._validated_goal(target)
        reference = np.asarray(current, dtype=float)
        if reference.shape != (self.chain.dof,) or not np.all(np.isfinite(reference)):
            raise ValueError(
                f"current joints must have shape ({self.chain.dof},) and be finite",
            )
        step_limit = float(max_joint_step_rad)
        if not np.isfinite(step_limit) or step_limit <= 0.0:
            raise ValueError("max_joint_step_rad must be finite and positive")

        lower = self.chain.lower_limits
        upper = self.chain.upper_limits
        if np.any(reference < lower) or np.any(reference > upper):
            raise ValueError("current joints violate the kinematic chain limits")
        config = self.config
        offset = np.asarray(config.position_error_offset_tip_m, dtype=float)

        def pose_errors(joints: np.ndarray) -> tuple[float, float]:
            actual = self.chain.forward(joints)
            return (
                float(np.linalg.norm(control_point_delta(
                    actual[:3, 3], actual[:3, :3], goal[:3, 3], goal[:3, :3], offset,
                ))),
                float(np.linalg.norm(
                    rotation_log(goal[:3, :3] @ actual[:3, :3].T),
                )),
            )

        def solution(
            joints: np.ndarray,
            position_error: float,
            orientation_error: float,
            *,
            iterations: int,
            seed_index: int,
        ) -> IKSolution:
            checkpoint(control, "Cartesian IK continuation scoring")
            singular_values = np.linalg.svd(
                self.chain.jacobian(joints),
                compute_uv=False,
            )
            span = np.maximum(upper - lower, 1e-9)
            normalized_margin = np.minimum(
                (joints - lower) / span,
                (upper - joints) / span,
            )
            return IKSolution(
                joints=joints.copy(),
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                manipulability=float(np.prod(np.maximum(singular_values, 0.0))),
                iterations=iterations,
                seed_index=seed_index,
                min_joint_limit_margin=float(np.min(normalized_margin)),
            )

        current_position_error, current_orientation_error = pose_errors(reference)
        if (
            current_position_error < config.position_tolerance_m
            and current_orientation_error < config.orientation_tolerance_rad
        ):
            # This exact-target fast path is also a branch-holding guarantee for
            # the alpha=0 sample of any Cartesian segment.
            return solution(
                reference,
                current_position_error,
                current_orientation_error,
                iterations=0,
                seed_index=0,
            )

        trust_lower = np.maximum(lower, reference - step_limit)
        trust_upper = np.minimum(upper, reference + step_limit)
        if np.any(trust_upper - trust_lower <= 2e-10):
            raise IKFailure("Cartesian IK continuation has a degenerate trust region")

        local_control = control or PlanningControl()
        continuation_control = local_control.limited_to(
            config.continuation_timeout_s,
            "Cartesian IK continuation budget",
        )

        def residual(joints: np.ndarray, seed_control: PlanningControl) -> np.ndarray:
            checkpoint(seed_control, "Cartesian IK continuation numerical solve")
            actual = self.chain.forward(joints)
            position = control_point_delta(
                actual[:3, 3], actual[:3, :3], goal[:3, 3], goal[:3, :3], offset,
            ) / config.position_scale_m
            orientation = rotation_log(goal[:3, :3] @ actual[:3, :3].T)
            return np.concatenate(
                (position, orientation / config.orientation_scale_rad),
            )

        def strict_seed(candidate: np.ndarray) -> np.ndarray:
            # SciPy requires starts strictly inside reflective bounds.
            epsilon = np.minimum(1e-10, 0.25 * (trust_upper - trust_lower))
            return np.minimum(
                np.maximum(candidate, trust_lower + epsilon),
                trust_upper - epsilon,
            )

        initial_seed = strict_seed(reference)
        fallback_seeds = [
            strict_seed(
                trust_lower
                + _halton(index, self.chain.dof) * (trust_upper - trust_lower),
            )
            for index in range(1, config.continuation_fallback_seeds + 1)
        ]
        best_errors = (current_position_error, current_orientation_error)
        seed_timeouts = 0

        def attempt(seed: np.ndarray, seed_index: int) -> IKSolution | None:
            nonlocal best_errors, seed_timeouts
            try:
                checkpoint(
                    continuation_control,
                    "Cartesian IK continuation candidate search",
                )
                seed_control = continuation_control.limited_to(
                    config.continuation_seed_timeout_s,
                    "Cartesian IK continuation seed budget",
                )
                result = least_squares(
                    lambda joints: residual(joints, seed_control),
                    seed,
                    bounds=(trust_lower, trust_upper),
                    method="trf",
                    jac="3-point",
                    x_scale="jac",
                    ftol=1e-11,
                    xtol=1e-11,
                    gtol=1e-11,
                    max_nfev=config.max_iterations,
                )
                checkpoint(seed_control, "Cartesian IK continuation candidate search")
            except PlanningDeadlineExceeded:
                # A caller deadline or cancellation always propagates. A local
                # seed timeout merely unlocks the strictly local fallback.
                checkpoint(control, "Cartesian IK continuation numerical solve")
                seed_timeouts += 1
                return None

            joints = np.asarray(result.x, dtype=float)
            if joints.shape != reference.shape or not np.all(np.isfinite(joints)):
                return None
            if (
                np.any(joints < trust_lower)
                or np.any(joints > trust_upper)
            ):
                return None
            position_error, orientation_error = pose_errors(joints)
            if sum((position_error, orientation_error)) < sum(best_errors):
                best_errors = (position_error, orientation_error)
            if (
                position_error >= config.position_tolerance_m
                or orientation_error >= config.orientation_tolerance_rad
            ):
                return None
            return solution(
                joints,
                position_error,
                orientation_error,
                iterations=int(result.nfev),
                seed_index=seed_index,
            )

        primary = attempt(initial_seed, 0)
        if primary is not None:
            return primary

        feasible_fallbacks = [
            candidate
            for seed_index, seed in enumerate(fallback_seeds, start=1)
            if (candidate := attempt(seed, seed_index)) is not None
        ]
        if feasible_fallbacks:
            span = np.maximum(upper - lower, 1e-9)

            def continuity_key(candidate: IKSolution) -> tuple[float, ...]:
                delta = candidate.joints - reference
                return (
                    float(np.max(np.abs(delta))),
                    float(np.linalg.norm(delta / span)),
                    candidate.position_error_m / config.position_tolerance_m,
                    candidate.orientation_error_rad / config.orientation_tolerance_rad,
                    -candidate.min_joint_limit_margin,
                )

            checkpoint(control, "Cartesian IK continuation solution selection")
            return min(feasible_fallbacks, key=continuity_key)

        position, orientation = best_errors
        raise IKFailure(
            "no local Cartesian IK continuation met tolerances "
            f"within max_joint_step_rad={step_limit:.6f} "
            f"(best position={position:.6f} m, orientation={orientation:.6f} rad; "
            f"seed_timeouts={seed_timeouts})",
        )
