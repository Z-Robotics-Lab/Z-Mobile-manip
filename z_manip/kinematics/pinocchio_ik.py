"""Pinocchio-backed bounded multi-start inverse kinematics for PiPER."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from z_manip.planning_control import (
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)

from .chain import KinematicChain
from .robust_ik import IKConfig, IKFailure, IKSolution, RobustIKSolver, _halton


class PinocchioUnavailable(RuntimeError):
    """The selected runtime does not provide the Pinocchio Python bindings."""


class PinocchioIKSolver:
    """Solve SE(3) targets with Pinocchio FK/Jacobians and damped least squares.

    The complete robot URDF is reduced by locking every joint outside the
    requested arm chain.  This preserves the deployed fixed mount transform
    while exposing exactly the six PiPER arm coordinates expected by the rest
    of the planning stack.
    """

    _DAMPING = 1e-4
    _MIN_DAMPING = 1e-7
    _MAX_DAMPING = 1e2
    _INTEGRATION_STEP = 1.0
    _MAX_TANGENT_STEP = 0.70
    _LINE_SEARCH_STEPS = (1.0, 0.5, 0.25, 0.125)
    _WARM_START_CAPACITY = 16
    _WARM_START_RADIUS_M = 0.20
    _WARM_START_COUNT = 1
    _FAILED_WARM_START_POSITION_M = 0.08

    def __init__(
        self,
        urdf_path: str | Path,
        chain: KinematicChain,
        config: IKConfig | None = None,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as error:  # pragma: no cover - host unit env lacks ROS bindings
            raise PinocchioUnavailable(
                "Pinocchio IK was selected but the pinocchio module is unavailable",
            ) from error

        self.pin = pin
        self.chain = chain
        self.config = config or IKConfig()
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        active_names = set(chain.joint_names)
        missing = active_names - set(full_model.names)
        if missing:
            raise ValueError(
                f"Pinocchio URDF is missing arm joints: {sorted(missing)}",
            )
        locked_joint_ids = [
            joint_id
            for joint_id, name in enumerate(full_model.names)
            if joint_id and name not in active_names
        ]
        self.model = pin.buildReducedModel(
            full_model,
            locked_joint_ids,
            pin.neutral(full_model),
        )
        reduced_names = tuple(self.model.names[1:])
        if reduced_names != chain.joint_names or self.model.nq != chain.dof:
            raise ValueError(
                "Pinocchio reduced-model joint order does not match the arm chain: "
                f"{reduced_names!r} != {chain.joint_names!r}",
            )
        self.base_frame_id = self.model.getFrameId(chain.base_link)
        self.tip_frame_id = self.model.getFrameId(chain.tip_link)
        if self.base_frame_id >= self.model.nframes:
            raise ValueError(f"Pinocchio model has no base frame {chain.base_link!r}")
        if self.tip_frame_id >= self.model.nframes:
            raise ValueError(f"Pinocchio model has no tip frame {chain.tip_link!r}")
        self.data = self.model.createData()
        # The dependency-light ranker remains useful because it only orders
        # hypotheses; all acceptance decisions below use Pinocchio.
        self._ranker = RobustIKSolver(chain, self.config)
        self._warm_starts: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self._WARM_START_CAPACITY,
        )
        self._maximum_tip_radius_m = self._maximum_chain_radius(chain)

    @staticmethod
    def _maximum_chain_radius(chain: KinematicChain) -> float:
        """Return a conservative, URDF-derived upper bound on tip radius.

        Rotations preserve vector length, so the sum of every fixed joint
        translation is a proof-safe upper bound for a revolute chain.  A
        prismatic joint adds its largest permitted translation.  This is
        intentionally conservative: it only rejects poses that no numerical
        optimizer can possibly reach.
        """

        radius = 0.0
        for joint in chain.joints:
            radius += float(np.linalg.norm(joint.origin[:3, 3]))
            if joint.joint_type == "prismatic":
                radius += max(abs(float(joint.lower)), abs(float(joint.upper)))
        return radius

    @staticmethod
    def _deduplicated_seeds(
        seeds: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        for seed in seeds:
            bounded = np.minimum(
                np.maximum(np.asarray(seed, dtype=float), lower + 1e-10),
                upper - 1e-10,
            )
            if not any(np.allclose(bounded, other, atol=1e-10) for other in unique):
                unique.append(bounded)
        return unique

    def _prepend_warm_starts(
        self,
        goal: np.ndarray,
        seeds: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> list[np.ndarray]:
        target = goal[:3, 3]
        nearby = sorted(
            (
                (float(np.linalg.norm(position - target)), joints)
                for position, joints in self._warm_starts
            ),
            key=lambda item: item[0],
        )
        warm = [
            joints
            for distance, joints in nearby[: self._WARM_START_COUNT]
            if distance <= self._WARM_START_RADIUS_M
        ]
        return self._deduplicated_seeds(warm + seeds, lower, upper)

    def _remember_warm_start(self, goal: np.ndarray, joints: np.ndarray) -> None:
        self._warm_starts.append((goal[:3, 3].copy(), joints.copy()))

    def make_seed_pose_ranker(
        self,
        current: np.ndarray | None,
        control: PlanningControl | None = None,
    ):
        return self._ranker.make_seed_pose_ranker(current, control)

    def _pose_error(
        self,
        joints: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[float, float, object, object]:
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, joints)
        pin.updateFramePlacements(self.model, self.data)
        base_world = self.data.oMf[self.base_frame_id]
        tip_world = self.data.oMf[self.tip_frame_id]
        desired_world = base_world * pin.SE3(goal[:3, :3], goal[:3, 3])
        position_error = float(
            np.linalg.norm(tip_world.translation - desired_world.translation),
        )
        orientation_error = float(
            np.linalg.norm(
                pin.log3(desired_world.rotation @ tip_world.rotation.T),
            ),
        )
        return position_error, orientation_error, tip_world, desired_world

    def _solution(
        self,
        joints: np.ndarray,
        *,
        position_error: float,
        orientation_error: float,
        iterations: int,
        seed_index: int,
    ) -> IKSolution:
        jacobian = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            joints,
            self.tip_frame_id,
            self.pin.LOCAL,
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        span = np.maximum(self.chain.upper_limits - self.chain.lower_limits, 1e-9)
        normalized_margin = np.minimum(
            (joints - self.chain.lower_limits) / span,
            (self.chain.upper_limits - joints) / span,
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

    def _task_weights(self, position_error: float) -> np.ndarray:
        """Return translation-first task weights for one LM iteration.

        The old implementation minimized metres and radians in the same
        unscaled vector.  For the PiPER work envelope that made a moderate
        wrist-orientation error dominate several centimetres of translation,
        so an otherwise reachable pregrasp was often abandoned far from the
        object.  Translation now receives a tight scale tied to the actual
        acceptance tolerance.  Orientation is introduced gradually after the
        tip enters a small capture region, then becomes a normal soft task.
        """

        position_scale = min(
            self.config.position_scale_m,
            max(2.0 * self.config.position_tolerance_m, 0.01),
        )
        capture_radius = max(3.0 * self.config.position_tolerance_m, 0.025)
        if position_error >= capture_radius:
            orientation_gain = 0.05
        else:
            # Smoothly ramp from a weak viewing-direction preference at the
            # capture boundary to full orientation tracking at the target.
            progress = 1.0 - position_error / capture_radius
            orientation_gain = 0.05 + 0.95 * progress * progress
        return np.asarray(
            [
                1.0 / position_scale,
                1.0 / position_scale,
                1.0 / position_scale,
                orientation_gain / self.config.orientation_scale_rad,
                orientation_gain / self.config.orientation_scale_rad,
                orientation_gain / self.config.orientation_scale_rad,
            ],
            dtype=float,
        )

    @staticmethod
    def _weighted_cost(error_twist: np.ndarray, weights: np.ndarray) -> float:
        return float(np.linalg.norm(weights * np.asarray(error_twist, dtype=float)))

    def _attempt(
        self,
        goal: np.ndarray,
        seed: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        seed_index: int,
        control: PlanningControl,
    ) -> tuple[IKSolution | None, tuple[float, float], np.ndarray]:
        pin = self.pin
        joints = np.minimum(np.maximum(seed, lower + 1e-10), upper - 1e-10)
        best = (float("inf"), float("inf"))
        best_joints = joints.copy()
        damping = self._DAMPING
        for iteration in range(self.config.max_iterations):
            checkpoint(control, "Pinocchio inverse kinematics")
            position_error, orientation_error, tip_world, desired_world = (
                self._pose_error(joints, goal)
            )
            if position_error + orientation_error < sum(best):
                best = (position_error, orientation_error)
                best_joints = joints.copy()
            if (
                position_error < self.config.position_tolerance_m
                and orientation_error < self.config.orientation_tolerance_rad
            ):
                return self._solution(
                    joints,
                    position_error=position_error,
                    orientation_error=orientation_error,
                    iterations=iteration,
                    seed_index=seed_index,
                ), best, best_joints

            tip_from_desired = tip_world.actInv(desired_world)
            error_twist = pin.log6(tip_from_desired).vector
            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                joints,
                self.tip_frame_id,
                pin.LOCAL,
            )
            task_jacobian = -pin.Jlog6(tip_from_desired.inverse()) @ jacobian
            weights = self._task_weights(position_error)
            weighted_jacobian = weights[:, None] * task_jacobian
            weighted_error = weights * error_twist
            normal = weighted_jacobian.T @ weighted_jacobian
            gradient = weighted_jacobian.T @ weighted_error
            tangent = -np.linalg.solve(
                normal + damping * np.eye(self.model.nv),
                gradient,
            )
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > self._MAX_TANGENT_STEP:
                tangent *= self._MAX_TANGENT_STEP / tangent_norm
            current_cost = self._weighted_cost(error_twist, weights)
            accepted = False
            for step_scale in self._LINE_SEARCH_STEPS:
                checkpoint(control, "Pinocchio inverse kinematics line search")
                candidate = pin.integrate(
                    self.model,
                    joints,
                    tangent * (self._INTEGRATION_STEP * step_scale),
                )
                candidate = np.minimum(
                    np.maximum(candidate, lower + 1e-10),
                    upper - 1e-10,
                )
                _, _, candidate_tip, candidate_desired = self._pose_error(
                    candidate,
                    goal,
                )
                candidate_error = pin.log6(
                    candidate_tip.actInv(candidate_desired),
                ).vector
                if self._weighted_cost(candidate_error, weights) < current_cost - 1e-9:
                    joints = candidate
                    damping = max(self._MIN_DAMPING, damping * 0.35)
                    accepted = True
                    break
            if not accepted:
                damping = min(self._MAX_DAMPING, damping * 10.0)
                if damping >= self._MAX_DAMPING:
                    break
        return None, best, best_joints

    def _bounded_solve(
        self,
        goal: np.ndarray,
        seeds: list[np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        timeout_s: float,
        seed_timeout_s: float,
        operation: str,
        control: PlanningControl | None,
    ) -> IKSolution:
        parent = control or PlanningControl()
        target_radius = float(np.linalg.norm(goal[:3, 3]))
        if target_radius > self._maximum_tip_radius_m + self.config.position_tolerance_m:
            raise IKFailure(
                "target lies outside the URDF chain reach bound "
                f"(radius={target_radius:.6f} m, "
                f"maximum={self._maximum_tip_radius_m:.6f} m)",
            )
        seeds = self._prepend_warm_starts(goal, seeds, lower, upper)
        solve_control = parent.limited_to(timeout_s, operation)
        best = (float("inf"), float("inf"))
        best_joints: np.ndarray | None = None
        seed_timeouts = 0
        for seed_index, seed in enumerate(seeds):
            try:
                seed_control = solve_control.limited_to(seed_timeout_s, operation)
                solution, errors, attempt_joints = self._attempt(
                    goal,
                    np.asarray(seed, dtype=float),
                    lower,
                    upper,
                    seed_index=seed_index,
                    control=seed_control,
                )
            except PlanningDeadlineExceeded:
                # A caller-owned deadline/cancellation always propagates. A
                # local seed timeout advances to the next deterministic seed.
                checkpoint(control, operation)
                seed_timeouts += 1
                try:
                    checkpoint(solve_control, operation)
                except PlanningDeadlineExceeded:
                    break
                continue
            if sum(errors) < sum(best):
                best = errors
                best_joints = attempt_joints
            if solution is not None:
                self._remember_warm_start(goal, solution.joints)
                return solution
        if best_joints is not None and best[0] <= self._FAILED_WARM_START_POSITION_M:
            # Adjacent grasp symmetries share almost the same Cartesian
            # position.  Even when this orientation misses tolerance, its
            # closest configuration is a much better next seed than another
            # global Halton restart.
            self._remember_warm_start(goal, best_joints)
        raise IKFailure(
            "no Pinocchio IK solution met tolerances "
            f"(best position={best[0]:.6f} m, orientation={best[1]:.6f} rad; "
            f"seed_timeouts={seed_timeouts})",
        )

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        *,
        control: PlanningControl | None = None,
    ) -> IKSolution:
        goal = RobustIKSolver._validated_goal(target)
        seeds = self._ranker._seeds(current, control)
        return self._bounded_solve(
            goal,
            seeds,
            self.chain.lower_limits,
            self.chain.upper_limits,
            timeout_s=self.config.solve_timeout_s,
            seed_timeout_s=self.config.seed_timeout_s,
            operation="Pinocchio inverse kinematics",
            control=control,
        )

    def solve_continuation(
        self,
        target: np.ndarray,
        current: np.ndarray,
        *,
        max_joint_step_rad: float,
        control: PlanningControl | None = None,
    ) -> IKSolution:
        goal = RobustIKSolver._validated_goal(target)
        reference = np.asarray(current, dtype=float)
        if reference.shape != (self.chain.dof,) or not np.all(np.isfinite(reference)):
            raise ValueError(
                f"current joints must have shape ({self.chain.dof},) and be finite",
            )
        step = float(max_joint_step_rad)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("max_joint_step_rad must be finite and positive")
        lower = np.maximum(self.chain.lower_limits, reference - step)
        upper = np.minimum(self.chain.upper_limits, reference + step)
        if np.any(reference < self.chain.lower_limits) or np.any(
            reference > self.chain.upper_limits
        ):
            raise ValueError("current joints violate the kinematic chain limits")
        seeds = [reference]
        for index in range(1, self.config.continuation_fallback_seeds + 1):
            seeds.append(lower + _halton(index, self.chain.dof) * (upper - lower))
        return self._bounded_solve(
            goal,
            seeds,
            lower,
            upper,
            timeout_s=self.config.continuation_timeout_s,
            seed_timeout_s=self.config.continuation_seed_timeout_s,
            operation="Pinocchio Cartesian continuation IK",
            control=control,
        )


__all__ = ["PinocchioIKSolver", "PinocchioUnavailable"]
