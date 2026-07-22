"""Safe, transport-free refinement of a joint-space RRT seed.

The RRT planner owns topology: it finds a collision-free homotopy class.  This
module only improves the geometry of that existing path.  Endpoints are fixed,
joint bounds are hard constraints, and every candidate is continuously checked
through both a generic state-validity callback and the fixed-fixture collision
guard before it can replace the seed.

The optimization problem is a convex quadratic smoothing problem.  CasADi's
``qrqp`` plugin is used when requested and available; SciPy L-BFGS-B is the
deterministic reference/fallback.  Collision constraints deliberately remain a
post-solve hard acceptance test because the production fixed-fixture guard is
not symbolic.  Backtracking from the optimum to the known seed makes that
acceptance useful without weakening the safety invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from z_manip.trajectory_clearance import (
    FixedFixtureStateGuard,
    FixedFixtureTrajectoryEvidence,
    evaluate_fixed_fixture_trajectory,
)


ARM_DOF = 6


class TrajectoryRefinementUnavailable(RuntimeError):
    """The explicitly requested refinement backend is unavailable."""


class StateValidityCallback(Protocol):
    def __call__(self, joints: np.ndarray) -> bool: ...


@dataclass(frozen=True)
class GenericTrajectoryEvidence:
    """Resolution-bounded evidence for an arbitrary state-validity callback."""

    valid: bool
    state_checks: int
    failing_segment_index: int | None = None
    failing_alpha: float | None = None


@dataclass(frozen=True)
class TrajectoryRefinementConfig:
    """Weights and acceptance bounds for :func:`refine_joint_trajectory`."""

    smoothness_weight: float = 4.0
    path_length_weight: float = 1.0
    seed_trust_weight: float = 0.05
    max_joint_step_rad: float = 0.01
    minimum_objective_improvement: float = 1e-10
    fixed_margin_tolerance_m: float = 1e-9
    line_search_steps: int = 18
    max_iterations: int = 250

    def __post_init__(self) -> None:
        weights = (
            self.smoothness_weight,
            self.path_length_weight,
            self.seed_trust_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError(
                "trajectory refinement weights must be finite and nonnegative",
            )
        if self.smoothness_weight == 0.0 and self.path_length_weight == 0.0:
            raise ValueError("trajectory refinement needs a path objective")
        positive = (
            self.max_joint_step_rad,
            self.minimum_objective_improvement,
            self.fixed_margin_tolerance_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "trajectory refinement tolerances must be finite and positive",
            )
        if self.line_search_steps < 1 or self.max_iterations < 1:
            raise ValueError("trajectory refinement iteration counts must be positive")


@dataclass(frozen=True, eq=False)
class TrajectoryRefinementResult:
    """Chosen path and auditable proof that the seed was or was not replaced."""

    trajectory: np.ndarray
    accepted: bool
    backend: str
    reason: str
    objective_before: float
    objective_after: float
    blend_alpha: float
    generic_before: GenericTrajectoryEvidence
    generic_after: GenericTrajectoryEvidence
    fixed_before: FixedFixtureTrajectoryEvidence | None
    fixed_after: FixedFixtureTrajectoryEvidence | None

    def __post_init__(self) -> None:
        values = np.array(self.trajectory, dtype=float, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "trajectory", values)

    def document(self) -> dict[str, object]:
        def generic(evidence: GenericTrajectoryEvidence) -> dict[str, object]:
            return {
                "valid": evidence.valid,
                "state_checks": evidence.state_checks,
                "failing_segment_index": evidence.failing_segment_index,
                "failing_alpha": evidence.failing_alpha,
            }

        return {
            "schema": "z_manip.trajectory_refinement.v1",
            "accepted": self.accepted,
            "backend": self.backend,
            "reason": self.reason,
            "objective_before": self.objective_before,
            "objective_after": self.objective_after,
            "blend_alpha": self.blend_alpha,
            "generic_before": generic(self.generic_before),
            "generic_after": generic(self.generic_after),
            "fixed_before": (
                None if self.fixed_before is None else self.fixed_before.document()
            ),
            "fixed_after": (
                None if self.fixed_after is None else self.fixed_after.document()
            ),
        }


@dataclass(frozen=True)
class _QuadraticPathProblem:
    seed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    temporal_matrix: np.ndarray
    trust_weight: float

    @property
    def variable_count(self) -> int:
        return max(0, (len(self.seed) - 2) * ARM_DOF)

    def path(self, variables: object) -> np.ndarray:
        result = self.seed.copy()
        if self.variable_count:
            result[1:-1] = np.asarray(variables, dtype=float).reshape(-1, ARM_DOF)
        return result

    def value_and_gradient(self, variables: object) -> tuple[float, np.ndarray]:
        path = self.path(variables)
        temporal_gradient = 2.0 * self.temporal_matrix @ path
        trust_delta = path - self.seed
        value = float(
            np.sum(path * (self.temporal_matrix @ path))
            + self.trust_weight * np.sum(trust_delta * trust_delta)
        )
        gradient = temporal_gradient + 2.0 * self.trust_weight * trust_delta
        return value, gradient[1:-1].reshape(-1)


def _path(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] < 2
        or result.shape[1] != ARM_DOF
        or not np.isfinite(result).all()
    ):
        raise ValueError("RRT seed must be a finite Nx6 array with N >= 2")
    return result.copy()


def _limits(
    lower: object,
    upper: object,
    seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if low.shape != (ARM_DOF,) or high.shape != (ARM_DOF,):
        raise ValueError("joint limits must be six-vectors")
    if (
        not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(low >= high)
    ):
        raise ValueError("joint limits must be finite and ordered")
    tolerance = 1e-10
    if np.any(seed < low - tolerance) or np.any(seed > high + tolerance):
        raise ValueError("RRT seed violates joint limits")
    return low.copy(), high.copy()


def _difference_matrix(size: int, order: int) -> np.ndarray:
    return np.diff(np.eye(size), n=order, axis=0)


def _problem(
    seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: TrajectoryRefinementConfig,
) -> _QuadraticPathProblem:
    first = _difference_matrix(len(seed), 1)
    temporal = config.path_length_weight * (first.T @ first)
    if len(seed) >= 3 and config.smoothness_weight:
        second = _difference_matrix(len(seed), 2)
        temporal += config.smoothness_weight * (second.T @ second)
    return _QuadraticPathProblem(
        seed=seed,
        lower=lower,
        upper=upper,
        temporal_matrix=temporal,
        trust_weight=config.seed_trust_weight,
    )


def _solve_scipy(
    problem: _QuadraticPathProblem,
    config: TrajectoryRefinementConfig,
) -> tuple[np.ndarray, str]:
    if not problem.variable_count:
        return problem.seed.copy(), "scipy-lbfgsb"
    initial = problem.seed[1:-1].reshape(-1)
    bounds = list(
        zip(
            np.tile(problem.lower, len(problem.seed) - 2),
            np.tile(problem.upper, len(problem.seed) - 2),
        ),
    )

    def objective(value: np.ndarray) -> float:
        return problem.value_and_gradient(value)[0]

    def gradient(value: np.ndarray) -> np.ndarray:
        return problem.value_and_gradient(value)[1]

    result = minimize(
        objective,
        initial,
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": config.max_iterations, "ftol": 1e-13, "gtol": 1e-10},
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError("SciPy trajectory refinement returned non-finite values")
    # L-BFGS-B can report a line-search warning at an already optimal point.
    # The finite candidate remains usable and still faces the hard acceptance gate.
    return problem.path(result.x), "scipy-lbfgsb"


def _solve_casadi(
    problem: _QuadraticPathProblem,
    *,
    plugin: str = "qrqp",
) -> tuple[np.ndarray, str]:
    if not problem.variable_count:
        return problem.seed.copy(), f"casadi-{plugin}"
    try:
        import casadi as ca
    except ImportError as error:  # pragma: no cover - optional dependency
        raise TrajectoryRefinementUnavailable("CasADi is not installed") from error

    internal = slice(1, -1)
    temporal = problem.temporal_matrix
    q_internal = temporal[internal, internal]
    q_boundary = temporal[internal][:, (0, len(problem.seed) - 1)]
    boundary = problem.seed[[0, len(problem.seed) - 1]]
    # Row-major flattening groups the six joints at each waypoint.
    hessian = 2.0 * np.kron(
        q_internal + problem.trust_weight * np.eye(len(problem.seed) - 2),
        np.eye(ARM_DOF),
    )
    gradient = (
        2.0 * q_boundary @ boundary
        - 2.0 * problem.trust_weight * problem.seed[1:-1]
    ).reshape(-1)
    try:
        solver = ca.conic(
            "trajectory_refinement_qp",
            plugin,
            {
                "h": ca.Sparsity.dense(problem.variable_count, problem.variable_count),
                "a": ca.Sparsity(0, problem.variable_count),
            },
            {
                "print_header": False,
                "print_iter": False,
                "print_info": False,
                "print_time": False,
            },
        )
        output = solver(
            h=ca.DM(hessian),
            g=ca.DM(gradient),
            lbx=ca.DM(np.tile(problem.lower, len(problem.seed) - 2)),
            ubx=ca.DM(np.tile(problem.upper, len(problem.seed) - 2)),
            lba=ca.DM([]),
            uba=ca.DM([]),
        )
    except Exception as error:  # pragma: no cover - plugin dependent
        raise TrajectoryRefinementUnavailable(
            f"CasADi QP plugin {plugin!r} failed: {error}",
        ) from error
    variables = np.asarray(output["x"], dtype=float).reshape(-1)
    if not np.isfinite(variables).all():
        raise RuntimeError("CasADi trajectory refinement returned non-finite values")
    return problem.path(variables), f"casadi-{plugin}"


def evaluate_generic_trajectory(
    joint_trajectory: object,
    *,
    state_valid: StateValidityCallback | None,
    max_joint_step_rad: float,
) -> GenericTrajectoryEvidence:
    """Continuously sample a generic validity callback along every path edge."""

    path = _path(joint_trajectory)
    maximum_step = float(max_joint_step_rad)
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("maximum joint step must be finite and positive")
    if state_valid is None:
        return GenericTrajectoryEvidence(valid=True, state_checks=0)
    checks = 0
    for segment_index, (start, end) in enumerate(zip(path[:-1], path[1:])):
        intervals = max(
            1,
            int(math.ceil(float(np.linalg.norm(end - start)) / maximum_step)),
        )
        first_step = 0 if segment_index == 0 else 1
        for step in range(first_step, intervals + 1):
            alpha = step / intervals
            checks += 1
            if not bool(state_valid(start + alpha * (end - start))):
                return GenericTrajectoryEvidence(
                    valid=False,
                    state_checks=checks,
                    failing_segment_index=segment_index,
                    failing_alpha=float(alpha),
                )
    return GenericTrajectoryEvidence(valid=True, state_checks=checks)


def _fixed_evidence(
    path: np.ndarray,
    guard: FixedFixtureStateGuard | None,
    maximum_step: float,
) -> FixedFixtureTrajectoryEvidence | None:
    if guard is None:
        return None
    return evaluate_fixed_fixture_trajectory(
        path,
        guard=guard,
        max_joint_step_rad=maximum_step,
    )


def _fixed_not_worse(
    before: FixedFixtureTrajectoryEvidence | None,
    after: FixedFixtureTrajectoryEvidence | None,
    tolerance_m: float,
) -> bool:
    if before is None:
        return True
    assert after is not None
    return bool(
        after.valid
        and after.minimum_margin_m + tolerance_m >= before.minimum_margin_m
    )


def refine_joint_trajectory(
    rrt_seed: object,
    *,
    lower_limits: object,
    upper_limits: object,
    state_valid: StateValidityCallback | None = None,
    fixed_guard: FixedFixtureStateGuard | None = None,
    config: TrajectoryRefinementConfig | None = None,
    backend: str = "auto",
    casadi_plugin: str = "qrqp",
) -> TrajectoryRefinementResult:
    """Smooth a known RRT path without weakening any collision evidence.

    ``backend`` accepts ``"auto"``, ``"scipy"`` or ``"casadi"``.  Auto first
    tries CasADi and deterministically falls back to SciPy.  An explicitly
    requested unavailable CasADi backend raises instead of silently changing
    the requested validation environment.
    """

    settings = config or TrajectoryRefinementConfig()
    seed = _path(rrt_seed)
    lower, upper = _limits(lower_limits, upper_limits, seed)
    path_problem = _problem(seed, lower, upper, settings)
    objective_before = path_problem.value_and_gradient(seed[1:-1].reshape(-1))[0]
    generic_before = evaluate_generic_trajectory(
        seed,
        state_valid=state_valid,
        max_joint_step_rad=settings.max_joint_step_rad,
    )
    fixed_before = _fixed_evidence(seed, fixed_guard, settings.max_joint_step_rad)

    normalized_backend = str(backend).strip().lower()
    if normalized_backend not in {"auto", "scipy", "casadi"}:
        raise ValueError("trajectory refinement backend must be auto, scipy, or casadi")
    if normalized_backend in {"auto", "casadi"}:
        try:
            optimum, backend_name = _solve_casadi(path_problem, plugin=casadi_plugin)
        except TrajectoryRefinementUnavailable:
            if normalized_backend == "casadi":
                raise
            optimum, backend_name = _solve_scipy(path_problem, settings)
            backend_name += "-auto-fallback"
    else:
        optimum, backend_name = _solve_scipy(path_problem, settings)

    fallback_reason = "no smoothing candidate preserved continuous safety and clearance"
    seed_objective = objective_before
    for exponent in range(settings.line_search_steps):
        alpha = 0.5**exponent
        candidate = seed + alpha * (optimum - seed)
        candidate[0] = seed[0]
        candidate[-1] = seed[-1]
        candidate[1:-1] = np.clip(candidate[1:-1], lower, upper)
        candidate_objective = path_problem.value_and_gradient(
            candidate[1:-1].reshape(-1),
        )[0]
        if candidate_objective > seed_objective - settings.minimum_objective_improvement:
            fallback_reason = "smoothing objective did not improve"
            continue
        generic_after = evaluate_generic_trajectory(
            candidate,
            state_valid=state_valid,
            max_joint_step_rad=settings.max_joint_step_rad,
        )
        if not generic_after.valid:
            fallback_reason = "generic continuous state validation rejected refinement"
            continue
        fixed_after = _fixed_evidence(
            candidate,
            fixed_guard,
            settings.max_joint_step_rad,
        )
        if not _fixed_not_worse(
            fixed_before,
            fixed_after,
            settings.fixed_margin_tolerance_m,
        ):
            fallback_reason = "fixed-fixture clearance would decrease"
            continue
        return TrajectoryRefinementResult(
            trajectory=candidate,
            accepted=True,
            backend=backend_name,
            reason="accepted continuously validated smoothing refinement",
            objective_before=seed_objective,
            objective_after=candidate_objective,
            blend_alpha=alpha,
            generic_before=generic_before,
            generic_after=generic_after,
            fixed_before=fixed_before,
            fixed_after=fixed_after,
        )

    return TrajectoryRefinementResult(
        trajectory=seed,
        accepted=False,
        backend=backend_name,
        reason=fallback_reason,
        objective_before=seed_objective,
        objective_after=seed_objective,
        blend_alpha=0.0,
        generic_before=generic_before,
        generic_after=generic_before,
        fixed_before=fixed_before,
        fixed_after=fixed_before,
    )


__all__ = [
    "GenericTrajectoryEvidence",
    "StateValidityCallback",
    "TrajectoryRefinementConfig",
    "TrajectoryRefinementResult",
    "TrajectoryRefinementUnavailable",
    "evaluate_generic_trajectory",
    "refine_joint_trajectory",
]
