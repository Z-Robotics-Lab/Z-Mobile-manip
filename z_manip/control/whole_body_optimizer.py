"""Velocity-level whole-body optimization for offline/shadow validation.

The controller linearizes a short-horizon task around measured state, then
solves a bounded convex QP.  Pinocchio supplies FK/Jacobians through the model
seam; CasADi supplies the production shadow QP backend.  A SciPy reference
backend keeps ordinary unit tests useful on hosts without optional robotics
packages.

This module produces command *intents* only.  It has no transport imports and
cannot open ROS, WebRTC, SocketCAN, or a Unitree/PiPER connection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import minimize

from .whole_body_model import (
    ARM_DOF,
    CONTROL_DOF,
    CONTROL_NAMES,
    ReducedWholeBodyState,
    ReducedWholeBodyVelocity,
    WholeBodyKinematics,
)


WHOLE_BODY_RESULT_SCHEMA = "z_manip.whole_body_shadow_result.v1"
CAMERA_DEPTH_BELOW_MINIMUM = "CAMERA_DEPTH_BELOW_MINIMUM"


class CasadiWholeBodyUnavailable(RuntimeError):
    """The optional CasADi runtime or requested QP plugin is unavailable."""


class WholeBodyVisibilityError(ValueError):
    """The synchronized target cannot be projected into the camera image."""

    def __init__(self, *, camera_depth_m: float, minimum_depth_m: float) -> None:
        self.camera_depth_m = float(camera_depth_m)
        self.minimum_depth_m = float(minimum_depth_m)
        super().__init__(
            f"target camera depth {self.camera_depth_m:.6f} m is not above "
            f"the minimum {self.minimum_depth_m:.6f} m",
        )


def _point3(value: Sequence[float], *, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must contain exactly three finite values")
    return point


def _homogeneous_inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


@dataclass(frozen=True)
class WholeBodyTask:
    """One synchronized target and desired close-range handoff geometry."""

    target_world_xyz_m: tuple[float, float, float]
    desired_planar_standoff_m: float = 0.52
    desired_image_u: float = 0.0
    desired_image_v: float = 0.0
    desired_target_height_in_body_m: float = -0.10
    desired_target_lateral_in_body_m: float = 0.0
    tool_target_offset_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _point3(self.target_world_xyz_m, label="world target")
        _point3(self.tool_target_offset_world_m, label="tool target offset")
        finite = (
            self.desired_planar_standoff_m,
            self.desired_image_u,
            self.desired_image_v,
            self.desired_target_height_in_body_m,
            self.desired_target_lateral_in_body_m,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("whole-body task values must be finite")
        if self.desired_planar_standoff_m <= 0.0:
            raise ValueError("desired planar standoff must be positive")


@dataclass(frozen=True)
class WholeBodyOptimizerConfig:
    horizon_dt_s: float = 0.20
    finite_difference_step: float = 1e-4
    transition_start_m: float = 1.00
    handoff_planar_m: float = 0.58
    camera_min_depth_m: float = 0.10
    # Camera centering is the arm's primary job while the mobile base is
    # approaching.  A wrist camera that loses the target cannot support a
    # fresh close-range grasp, so this remains active over the full range.
    image_weight: float = 18.0
    # Ground standoff must dominate near the handoff; otherwise the arm/tool
    # residual can incorrectly pull the rolling base inside the 0.5 m zone.
    planar_weight: float = 100.0
    target_height_weight: float = 5.0
    target_lateral_weight: float = 40.0
    tool_position_weight: float = 12.0
    # Do not turn view keeping into an early reach.  The tool-to-target
    # residual becomes active only in the narrow final handoff band; before
    # that the arm follows image u/v while the base owns range closure.
    tool_transition_start_m: float = 0.64
    control_weight: float = 0.30
    smooth_weight: float = 1.50
    manipulability_weight: float = 0.08
    base_forward_min_mps: float = -0.05
    base_forward_max_mps: float = 0.18
    base_yaw_max_rps: float = 0.16
    body_roll_rate_max_rps: float = math.radians(5.0)
    body_pitch_rate_max_rps: float = math.radians(7.0)
    body_roll_abs_max_rad: float = math.radians(8.0)
    body_pitch_abs_max_rad: float = math.radians(12.0)
    arm_velocity_scale: float = 0.35

    def __post_init__(self) -> None:
        positive = (
            self.horizon_dt_s,
            self.finite_difference_step,
            self.transition_start_m,
            self.handoff_planar_m,
            self.camera_min_depth_m,
            self.image_weight,
            self.planar_weight,
            self.target_height_weight,
            self.target_lateral_weight,
            self.tool_position_weight,
            self.tool_transition_start_m,
            self.control_weight,
            self.smooth_weight,
            self.base_yaw_max_rps,
            self.body_roll_rate_max_rps,
            self.body_pitch_rate_max_rps,
            self.body_roll_abs_max_rad,
            self.body_pitch_abs_max_rad,
            self.arm_velocity_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("optimizer weights, limits, and intervals must be positive")
        if self.base_forward_min_mps >= self.base_forward_max_mps:
            raise ValueError("base forward velocity bounds are reversed")
        if self.transition_start_m <= self.handoff_planar_m:
            raise ValueError("transition start must be outside handoff distance")
        if self.tool_transition_start_m <= self.handoff_planar_m:
            raise ValueError("tool transition must be outside handoff distance")
        if self.tool_transition_start_m > self.transition_start_m:
            raise ValueError("tool transition must be inside the view transition")
        if self.manipulability_weight < 0.0 or not math.isfinite(
            self.manipulability_weight,
        ):
            raise ValueError("manipulability weight must be finite and nonnegative")


@dataclass(frozen=True)
class LinearizedWholeBodyProblem:
    hessian: np.ndarray
    gradient: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    residual: np.ndarray
    jacobian: np.ndarray
    residual_names: tuple[str, ...]
    near_weight: float
    reach_weight: float
    planar_distance_m: float
    camera_depth_m: float
    manipulability: float


@dataclass(frozen=True)
class WholeBodyOptimizationResult:
    velocity: ReducedWholeBodyVelocity
    predicted_state: ReducedWholeBodyState
    backend: str
    objective_before: float
    objective_after: float
    residual_before: tuple[float, ...]
    residual_after: tuple[float, ...]
    residual_names: tuple[str, ...]
    near_weight: float
    reach_weight: float
    planar_distance_m: float
    camera_depth_m: float
    manipulability: float
    success: bool
    reason: str
    failure_code: str | None
    minimum_camera_depth_m: float

    def status_document(self) -> dict[str, object]:
        velocity = self.velocity.as_vector()
        return {
            "schema": WHOLE_BODY_RESULT_SCHEMA,
            "mode": "shadow",
            "transport_opened": False,
            "motion_commands_sent": 0,
            "backend": self.backend,
            "success": self.success,
            "reason": self.reason,
            "failure_code": self.failure_code,
            "executable_intent": self.success,
            "objective": {
                "before": self.objective_before,
                "after": self.objective_after,
            },
            "schedule": {
                "near_weight": self.near_weight,
                "reach_weight": self.reach_weight,
                "planar_distance_m": self.planar_distance_m,
                "camera_depth_m": self.camera_depth_m,
                "minimum_camera_depth_m": self.minimum_camera_depth_m,
                "camera_depth_valid": (
                    self.camera_depth_m > self.minimum_camera_depth_m
                ),
                "arm_manipulability": self.manipulability,
            },
            "residuals": {
                name: {"before": before, "after": after}
                for name, before, after in zip(
                    self.residual_names,
                    self.residual_before,
                    self.residual_after,
                )
            },
            "intent": dict(zip(CONTROL_NAMES, (float(item) for item in velocity))),
            "predicted_state": asdict(self.predicted_state),
        }


class WholeBodyQPSolver(Protocol):
    name: str

    def solve(self, problem: LinearizedWholeBodyProblem) -> np.ndarray: ...


class ScipyReferenceBoxQP:
    """Deterministic offline reference for hosts without CasADi."""

    name = "scipy-lbfgsb-reference"

    def solve(self, problem: LinearizedWholeBodyProblem) -> np.ndarray:
        hessian = problem.hessian
        gradient = problem.gradient

        def objective(value: np.ndarray) -> float:
            return float(0.5 * value @ hessian @ value + gradient @ value)

        def derivative(value: np.ndarray) -> np.ndarray:
            return hessian @ value + gradient

        result = minimize(
            objective,
            np.clip(np.zeros(CONTROL_DOF), problem.lower, problem.upper),
            jac=derivative,
            bounds=list(zip(problem.lower, problem.upper)),
            method="L-BFGS-B",
            options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 200},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"reference whole-body QP failed: {result.message}")
        return np.asarray(result.x, dtype=float)


class CasadiBoxQP:
    """CasADi bounded QP using the bundled sparse QRQP plugin."""

    name = "casadi-qrqp"

    def __init__(self, *, plugin: str = "qrqp") -> None:
        try:
            import casadi as ca
        except ImportError as error:  # pragma: no cover - optional runtime
            raise CasadiWholeBodyUnavailable(
                "CasADi whole-body QP requires the optional casadi package",
            ) from error
        self.ca = ca
        self.plugin = plugin

    def solve(self, problem: LinearizedWholeBodyProblem) -> np.ndarray:
        ca = self.ca
        try:
            solver = ca.conic(
                "whole_body_shadow_qp",
                self.plugin,
                {
                    "h": ca.Sparsity.dense(CONTROL_DOF, CONTROL_DOF),
                    "a": ca.Sparsity(0, CONTROL_DOF),
                },
                {
                    "print_header": False,
                    "print_iter": False,
                    "print_info": False,
                    "print_time": False,
                },
            )
            output = solver(
                h=ca.DM(problem.hessian),
                g=ca.DM(problem.gradient),
                lbx=ca.DM(problem.lower),
                ubx=ca.DM(problem.upper),
                lba=ca.DM([]),
                uba=ca.DM([]),
            )
        except Exception as error:  # pragma: no cover - plugin dependent
            raise CasadiWholeBodyUnavailable(
                f"CasADi QP plugin {self.plugin!r} failed: {error}",
            ) from error
        result = np.asarray(output["x"], dtype=float).reshape(CONTROL_DOF)
        if not np.isfinite(result).all():
            raise RuntimeError("CasADi whole-body QP returned non-finite velocity")
        return result


class WholeBodyShadowOptimizer:
    """Coupled base/body/arm optimizer with continuous distance scheduling."""

    _RESIDUAL_NAMES = (
        "image_u",
        "image_v",
        "ground_standoff_m",
        "target_lateral_in_body_m",
        "target_height_in_body_m",
        "tool_x_m",
        "tool_y_m",
        "tool_z_m",
    )

    def __init__(
        self,
        model: WholeBodyKinematics,
        config: WholeBodyOptimizerConfig | None = None,
        *,
        solver: WholeBodyQPSolver | None = None,
    ) -> None:
        self.model = model
        self.config = config or WholeBodyOptimizerConfig()
        self.solver = solver or ScipyReferenceBoxQP()

    def _geometry(
        self,
        state: ReducedWholeBodyState,
        task: WholeBodyTask,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        target = _point3(task.target_world_xyz_m, label="world target")
        camera_pose = np.asarray(
            self.model.frame_pose(state, self.model.camera_frame),
            dtype=float,
        )
        tool_pose = np.asarray(
            self.model.frame_pose(state, self.model.tool_frame),
            dtype=float,
        )
        if camera_pose.shape != (4, 4) or tool_pose.shape != (4, 4):
            raise ValueError("whole-body model frame poses must be 4x4")
        camera_target = _homogeneous_inverse(camera_pose) @ np.append(target, 1.0)
        base_delta = target[:2] - np.asarray((state.base_x_m, state.base_y_m))
        planar = float(np.linalg.norm(base_delta))
        return camera_target[:3], tool_pose[:3, 3], planar, float(camera_target[2])

    def _near_weight(self, planar_distance_m: float) -> float:
        span = self.config.transition_start_m - self.config.handoff_planar_m
        return float(
            np.clip(
                (self.config.transition_start_m - planar_distance_m) / span,
                0.0,
                1.0,
            ),
        )

    def _reach_weight(self, planar_distance_m: float) -> float:
        """Blend tool reach only across the final close-range handoff band."""

        span = self.config.tool_transition_start_m - self.config.handoff_planar_m
        linear = float(np.clip(
            (self.config.tool_transition_start_m - planar_distance_m) / span,
            0.0,
            1.0,
        ))
        # Smoothstep avoids a velocity discontinuity when crossing into the
        # final reach band.
        return linear * linear * (3.0 - 2.0 * linear)

    def _residual(
        self,
        state: ReducedWholeBodyState,
        task: WholeBodyTask,
    ) -> tuple[np.ndarray, float, float, float]:
        camera_target, tool_position, planar, depth = self._geometry(state, task)
        if depth <= self.config.camera_min_depth_m:
            raise WholeBodyVisibilityError(
                camera_depth_m=depth,
                minimum_depth_m=self.config.camera_min_depth_m,
            )
        target = _point3(task.target_world_xyz_m, label="world target")
        target_body = np.asarray(
            self.model.target_in_body(state, target),
            dtype=float,
        )
        if target_body.shape != (3,) or not np.isfinite(target_body).all():
            raise ValueError("whole-body model returned an invalid local target")
        offset = _point3(task.tool_target_offset_world_m, label="tool target offset")
        desired_tool = target + offset
        residual = np.asarray(
            (
                camera_target[0] / depth - task.desired_image_u,
                camera_target[1] / depth - task.desired_image_v,
                planar - task.desired_planar_standoff_m,
                target_body[1] - task.desired_target_lateral_in_body_m,
                target_body[2] - task.desired_target_height_in_body_m,
                *(tool_position - desired_tool),
            ),
            dtype=float,
        )
        return residual, planar, depth, self._near_weight(planar)

    def _weights(self, near: float, reach: float) -> np.ndarray:
        # Far away, ground approach dominates.  Near the target, base remains
        # active but body/arm/FOV authority rises continuously—there is no
        # blocking "posture must settle" phase.
        return np.asarray(
            (
                self.config.image_weight * (0.55 + 0.45 * near),
                self.config.image_weight * (0.55 + 0.45 * near),
                self.config.planar_weight,
                self.config.target_lateral_weight * (0.35 + 0.65 * near),
                self.config.target_height_weight * (0.20 + 0.80 * near),
                self.config.tool_position_weight * reach,
                self.config.tool_position_weight * reach,
                self.config.tool_position_weight * reach,
            ),
            dtype=float,
        )

    def _bounds(self, state: ReducedWholeBodyState) -> tuple[np.ndarray, np.ndarray]:
        arm_limit = np.asarray(self.model.arm_velocity_limits, dtype=float)
        if arm_limit.shape != (ARM_DOF,) or not np.isfinite(arm_limit).all():
            raise ValueError("whole-body model must expose six finite arm velocity limits")
        scaled_arm = self.config.arm_velocity_scale * arm_limit
        lower = np.asarray(
            (
                self.config.base_forward_min_mps,
                -self.config.base_yaw_max_rps,
                -self.config.body_roll_rate_max_rps,
                -self.config.body_pitch_rate_max_rps,
                *(-scaled_arm),
            ),
            dtype=float,
        )
        upper = np.asarray(
            (
                self.config.base_forward_max_mps,
                self.config.base_yaw_max_rps,
                self.config.body_roll_rate_max_rps,
                self.config.body_pitch_rate_max_rps,
                *scaled_arm,
            ),
            dtype=float,
        )
        dt = self.config.horizon_dt_s
        lower[2] = max(
            lower[2],
            (-self.config.body_roll_abs_max_rad - state.body_roll_rad) / dt,
        )
        upper[2] = min(
            upper[2],
            (self.config.body_roll_abs_max_rad - state.body_roll_rad) / dt,
        )
        lower[3] = max(
            lower[3],
            (-self.config.body_pitch_abs_max_rad - state.body_pitch_rad) / dt,
        )
        upper[3] = min(
            upper[3],
            (self.config.body_pitch_abs_max_rad - state.body_pitch_rad) / dt,
        )
        # One-step joint-position feasibility is a hard bound.
        joints = np.asarray(state.arm_joints_rad, dtype=float)
        lower[4:] = np.maximum(
            lower[4:],
            (np.asarray(self.model.arm_lower_limits, dtype=float) - joints) / dt,
        )
        upper[4:] = np.minimum(
            upper[4:],
            (np.asarray(self.model.arm_upper_limits, dtype=float) - joints) / dt,
        )
        if np.any(lower > upper):
            raise ValueError("measured arm state lies outside whole-body model limits")
        return lower, upper

    def linearize(
        self,
        state: ReducedWholeBodyState,
        task: WholeBodyTask,
        *,
        previous_velocity: ReducedWholeBodyVelocity | None = None,
    ) -> LinearizedWholeBodyProblem:
        residual, planar, camera_depth, near = self._residual(state, task)
        epsilon = self.config.finite_difference_step
        jacobian = np.zeros((len(residual), CONTROL_DOF), dtype=float)
        for index in range(CONTROL_DOF):
            tangent = np.zeros(CONTROL_DOF)
            tangent[index] = epsilon / self.config.horizon_dt_s
            stepped = self.model.integrate(
                state,
                ReducedWholeBodyVelocity.from_vector(tangent),
                self.config.horizon_dt_s,
            )
            stepped_residual, _planar, _depth, _near = self._residual(stepped, task)
            jacobian[:, index] = (stepped_residual - residual) / epsilon
        # The QP variable is velocity, while the finite-difference derivative
        # above is with respect to a unit state displacement.
        jacobian *= self.config.horizon_dt_s

        reach = self._reach_weight(planar)
        weights = self._weights(near, reach)
        weighted = weights[:, None] * jacobian
        hessian = jacobian.T @ weighted
        gradient = jacobian.T @ (weights * residual)
        hessian += self.config.control_weight * np.eye(CONTROL_DOF)

        previous = (
            np.zeros(CONTROL_DOF)
            if previous_velocity is None
            else previous_velocity.as_vector()
        )
        hessian += self.config.smooth_weight * np.eye(CONTROL_DOF)
        gradient -= self.config.smooth_weight * previous

        manipulability = float(self.model.arm_manipulability(state))
        if self.config.manipulability_weight > 0.0 and near > 0.0:
            gradient_manip = np.zeros(CONTROL_DOF)
            for arm_index in range(ARM_DOF):
                tangent = np.zeros(CONTROL_DOF)
                tangent[4 + arm_index] = epsilon / self.config.horizon_dt_s
                stepped = self.model.integrate(
                    state,
                    ReducedWholeBodyVelocity.from_vector(tangent),
                    self.config.horizon_dt_s,
                )
                gradient_manip[4 + arm_index] = (
                    self.model.arm_manipulability(stepped) - manipulability
                ) / epsilon * self.config.horizon_dt_s
            # Maximizing local manipulability is a linear reward in the QP.
            gradient -= self.config.manipulability_weight * near * gradient_manip

        lower, upper = self._bounds(state)
        hessian = 0.5 * (hessian + hessian.T) + 1e-9 * np.eye(CONTROL_DOF)
        return LinearizedWholeBodyProblem(
            hessian=hessian,
            gradient=gradient,
            lower=lower,
            upper=upper,
            residual=residual,
            jacobian=jacobian,
            residual_names=self._RESIDUAL_NAMES,
            near_weight=near,
            reach_weight=reach,
            planar_distance_m=planar,
            camera_depth_m=camera_depth,
            manipulability=manipulability,
        )

    def solve(
        self,
        state: ReducedWholeBodyState,
        task: WholeBodyTask,
        *,
        previous_velocity: ReducedWholeBodyVelocity | None = None,
        locked_control_indices: Sequence[int] = (),
    ) -> WholeBodyOptimizationResult:
        _camera_target, _tool_position, planar, camera_depth = self._geometry(
            state,
            task,
        )
        if camera_depth <= self.config.camera_min_depth_m:
            return self._visibility_failure(
                state,
                planar_distance_m=planar,
                camera_depth_m=camera_depth,
            )
        problem = self.linearize(
            state,
            task,
            previous_velocity=previous_velocity,
        )
        problem = self._lock_problem(problem, locked_control_indices)
        value = np.asarray(self.solver.solve(problem), dtype=float)
        if value.shape != (CONTROL_DOF,) or not np.isfinite(value).all():
            raise RuntimeError("whole-body QP backend returned an invalid velocity")
        value = np.clip(value, problem.lower, problem.upper)
        return self._evaluate_value(state, task, problem, value, self.solver.name)

    def evaluate_velocity(
        self,
        state: ReducedWholeBodyState,
        task: WholeBodyTask,
        velocity: ReducedWholeBodyVelocity,
        *,
        previous_velocity: ReducedWholeBodyVelocity | None = None,
        locked_control_indices: Sequence[int] = (),
    ) -> WholeBodyOptimizationResult:
        """Replay a proposed bounded velocity against the nonlinear task."""

        problem = self._lock_problem(
            self.linearize(state, task, previous_velocity=previous_velocity),
            locked_control_indices,
        )
        value = np.asarray(velocity.as_vector(), dtype=float)
        if value.shape != (CONTROL_DOF,) or not np.isfinite(value).all():
            raise ValueError("replayed whole-body velocity is invalid")
        if np.any(value < problem.lower - 1e-9) or np.any(value > problem.upper + 1e-9):
            raise ValueError("replayed whole-body velocity exceeds active bounds")
        return self._evaluate_value(
            state,
            task,
            problem,
            np.clip(value, problem.lower, problem.upper),
            f"replay-{self.solver.name}",
        )

    @staticmethod
    def _lock_problem(
        problem: LinearizedWholeBodyProblem,
        locked_control_indices: Sequence[int],
    ) -> LinearizedWholeBodyProblem:
        if not locked_control_indices:
            return problem
        lower = problem.lower.copy()
        upper = problem.upper.copy()
        for raw_index in locked_control_indices:
            index = int(raw_index)
            if index < 0 or index >= CONTROL_DOF:
                raise ValueError(f"locked control index is out of range: {index}")
            lower[index] = 0.0
            upper[index] = 0.0
        return replace(problem, lower=lower, upper=upper)

    def _evaluate_value(
        self,
        state: ReducedWholeBodyState,
        task: WholeBodyTask,
        problem: LinearizedWholeBodyProblem,
        value: np.ndarray,
        backend: str,
    ) -> WholeBodyOptimizationResult:
        velocity = ReducedWholeBodyVelocity.from_vector(value)
        predicted = self.model.integrate(state, velocity, self.config.horizon_dt_s)
        try:
            residual_after, _planar, _depth, _near = self._residual(predicted, task)
        except WholeBodyVisibilityError as error:
            return self._visibility_failure(
                state,
                planar_distance_m=problem.planar_distance_m,
                camera_depth_m=error.camera_depth_m,
            )
        weights = self._weights(problem.near_weight, problem.reach_weight)
        objective_before = float(0.5 * np.sum(weights * problem.residual**2))
        objective_after = float(0.5 * np.sum(weights * residual_after**2))
        return WholeBodyOptimizationResult(
            velocity=velocity,
            predicted_state=predicted,
            backend=backend,
            objective_before=objective_before,
            objective_after=objective_after,
            residual_before=tuple(float(item) for item in problem.residual),
            residual_after=tuple(float(item) for item in residual_after),
            residual_names=problem.residual_names,
            near_weight=problem.near_weight,
            reach_weight=problem.reach_weight,
            planar_distance_m=problem.planar_distance_m,
            camera_depth_m=problem.camera_depth_m,
            manipulability=problem.manipulability,
            success=objective_after <= objective_before + 1e-9,
            reason=(
                "bounded short-horizon intent lowers the coupled task objective"
                if objective_after <= objective_before + 1e-9
                else "linearized intent did not lower the nonlinear replay objective"
            ),
            failure_code=(
                None
                if objective_after <= objective_before + 1e-9
                else "NONLINEAR_OBJECTIVE_NOT_LOWERED"
            ),
            minimum_camera_depth_m=self.config.camera_min_depth_m,
        )

    def _visibility_failure(
        self,
        state: ReducedWholeBodyState,
        *,
        planar_distance_m: float,
        camera_depth_m: float,
    ) -> WholeBodyOptimizationResult:
        """Return a stationary, explicitly non-executable shadow result."""

        zero_velocity = ReducedWholeBodyVelocity.from_vector(
            np.zeros(CONTROL_DOF),
        )
        return WholeBodyOptimizationResult(
            velocity=zero_velocity,
            predicted_state=state,
            backend="visibility-gate",
            objective_before=0.0,
            objective_after=0.0,
            residual_before=(),
            residual_after=(),
            residual_names=(),
            near_weight=self._near_weight(planar_distance_m),
            reach_weight=self._reach_weight(planar_distance_m),
            planar_distance_m=planar_distance_m,
            camera_depth_m=camera_depth_m,
            manipulability=float(self.model.arm_manipulability(state)),
            success=False,
            reason=(
                f"target camera depth {camera_depth_m:.6f} m is not above "
                f"the minimum {self.config.camera_min_depth_m:.6f} m; "
                "stationary intent returned"
            ),
            failure_code=CAMERA_DEPTH_BELOW_MINIMUM,
            minimum_camera_depth_m=self.config.camera_min_depth_m,
        )


__all__ = [
    "WHOLE_BODY_RESULT_SCHEMA",
    "CAMERA_DEPTH_BELOW_MINIMUM",
    "CasadiBoxQP",
    "CasadiWholeBodyUnavailable",
    "LinearizedWholeBodyProblem",
    "ScipyReferenceBoxQP",
    "WholeBodyOptimizationResult",
    "WholeBodyOptimizerConfig",
    "WholeBodyQPSolver",
    "WholeBodyShadowOptimizer",
    "WholeBodyTask",
    "WholeBodyVisibilityError",
]
