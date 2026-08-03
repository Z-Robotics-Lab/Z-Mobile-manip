"""roboplan ``SimpleIk`` behind the repo's ``IKSolver`` seam.

A drop-in for ``RobustIKSolver``/``PinocchioIKSolver``: same ``IKSolution``
fields, same meanings.  Two of those meanings are not roboplan's:

*   **Acceptance is ours, not SimpleIk's.**  ``SimpleIkOptions`` stops on a
    tip-ORIGIN, isotropic error.  This repo accepts on ``control_point_delta``
    measured at the tool contact point (2026-07-24: a 0.236 rad orientation
    residual inside a loose gate put the PiPER fingers 27 mm off a verified
    grasp) and, when configured, splits the orientation residual about a free
    axis exactly as ``PinocchioIKSolver._orientation_accepted`` does.  So every
    SimpleIk success is re-gated here, and the roboplan tolerances only tune how
    early SimpleIk stops trying.

*   **Restarts are ours, not SimpleIk's.**  ``max_restarts`` is 0 and the
    multi-start ladder is an explicit deterministic Halton sweep.  Measured:
    with ``max_restarts >= 1`` SimpleIk draws its restart from the Scene-global
    RNG, so the same (goal, start, options) returns different joints on repeated
    calls and intermittently fails; at 0 it is a pure descent and two identical
    requests return identical joints.  The ladder costs nothing for that --
    60/60 solved either way, 1.28 ms mean / 4.67 ms p95.
    ``solve_continuation`` reuses the same ladder inside the trust box, which is
    what keeps it on the current IK branch -- SimpleIk has no step bound of its
    own, so the bound is enforced here on the returned solution.

Default OFF: the constructor goes through ``roboplan_runtime.scene_handle``,
which raises ``RoboplanUnavailable('disabled')`` until a deployment opts in.
roboplan is imported lazily inside ``_solve_ik``; this module loads on a host
with neither roboplan nor pinocchio.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from z_manip.kinematics.chain import rotation_log
from z_manip.kinematics.robust_ik import (
    IKConfig,
    IKFailure,
    IKSolution,
    RobustIKSolver,
    _halton,
    control_point_delta,
)
from z_manip.planning_control import (
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)

from .roboplan_runtime import RoboplanConfig, scene_handle

__all__ = ["RoboplanIKSolver"]

_SOLVE = "roboplan inverse kinematics"
_CONTINUATION = "roboplan Cartesian continuation IK"

# Floor for ``SimpleIkOptions.max_time``.  Measured on this model: one SimpleIk
# restart costs ~0.5 ms, and a max_time of 0.0 still runs one full restart, so a
# smaller cap buys nothing and only invites a divide-by-nothing in the library.
# The deadline itself is enforced by ``checkpoint`` between attempts; an attempt
# may therefore overrun the caller's deadline by at most one restart.
_MIN_CALL_BUDGET_S = 1e-3

# Slack when accepting a ``SimpleIk`` solution against the model joint limits.
# Not a tolerance on the robot -- it absorbs the last ulps of a descent that
# converged ON a limit, nothing more.  Anything past it is REJECTED, not
# clipped: clipping would move the configuration away from the one SimpleIk
# collision-checked, and both existing backends structurally cannot return an
# out-of-limit solution (SciPy hard bounds; Pinocchio clamps every iterate), so
# accepting one here is a drop-in violation with a silent consequence --
# grasp_pipeline.py:963 scores ``-joint_limit_penalty / (margin + 1e-3)``, so a
# margin below -1e-3 flips that PENALTY into a bonus and the out-of-limit grasp
# wins the plan.
_JOINT_LIMIT_SLACK_RAD = 1e-9


def _solve_ik(
    scene: object,
    joint_names: Sequence[str],
    frames: tuple[str, str],
    goal_tform: np.ndarray,
    start: np.ndarray,
    options: Mapping[str, object],
) -> tuple[bool, np.ndarray]:
    """One ``SimpleIk`` attempt.

    The only roboplan import site, and the seam the host-side tests replace with
    a fake.  ``solveIk`` writes into the preallocated ``solution`` and returns a
    bool; the buffer holds the last iterate on failure, so it is only read here
    when the call reports success.
    """

    import roboplan.core as rc  # noqa: PLC0415 -- 4090-only dependency
    from roboplan.simple_ik import SimpleIk, SimpleIkOptions  # noqa: PLC0415

    names = list(joint_names)
    solution = rc.JointConfiguration(names, np.zeros(len(names)))
    # A fresh SimpleIk per attempt (measured 39 us, ~4% of a mean solve):
    # SimpleIkOptions is copied by value at construction, so ``max_time`` cannot
    # be retargeted at the remaining deadline any other way.
    solver = SimpleIk(scene, SimpleIkOptions(**dict(options)))
    solved = solver.solveIk(
        rc.CartesianConfiguration(frames[0], frames[1], np.asfortranarray(goal_tform)),
        rc.JointConfiguration(names, np.array(start, dtype=float)),
        solution,
    )
    return bool(solved), np.array(solution.positions, dtype=float)


class RoboplanIKSolver:
    """Solve SE(3) targets with roboplan ``SimpleIk`` on the capsule ``Scene``.

    ``config`` supplies the Scene and roboplan's own solver budget;
    ``ik_config`` supplies acceptance -- the tolerances, the tool control point
    and the free-axis split -- so a pose accepted here is a pose the numpy and
    Pinocchio backends accept, which is what keeps grasp scoring comparable.
    """

    def __init__(
        self,
        config: RoboplanConfig,
        robot_config: object,
        ik_config: IKConfig | None = None,
    ) -> None:
        self.config = config
        self.ik_config = ik_config or IKConfig()
        # Opt-in check FIRST.  A disabled backend must raise exactly
        # RoboplanUnavailable('disabled') and nothing else: a caller that falls
        # back to the numpy backend on that one exception would otherwise crash
        # on a ValueError about a config block it never opted into.
        self.handle = scene_handle(config, robot_config)
        # SimpleIk's stopping norm is measured at the tip origin, isotropically;
        # acceptance below is measured at the tool control point.  A stopping
        # norm LOOSER than acceptance makes every attempt report success into a
        # rejection and burns the whole ladder for nothing, silently, so it fails
        # here instead of at 3am.
        if (
            config.ik_position_tolerance_m > self.ik_config.position_tolerance_m
            or config.ik_orientation_tolerance_rad
            > self.ik_config.orientation_tolerance_rad
        ):
            raise ValueError(
                "roboplan IK stopping tolerances must not be looser than the "
                "acceptance tolerances in IKConfig "
                f"({config.ik_position_tolerance_m} m / "
                f"{config.ik_orientation_tolerance_rad} rad vs "
                f"{self.ik_config.position_tolerance_m} m / "
                f"{self.ik_config.orientation_tolerance_rad} rad)",
            )

        self.base_link = str(robot_config.base_link)
        self.tip_link = str(robot_config.tip_link)
        with self.handle.exclusive() as scene:
            self._joint_names = tuple(
                scene.getJointGroupInfo(self.handle.group).joint_names,
            )
        if len(self._joint_names) != self.handle.dof:
            raise ValueError(
                f"group {self.handle.group} reports {len(self._joint_names)} joint "
                f"names for {self.handle.dof} joints",
            )

        self._offset = np.asarray(
            self.ik_config.position_error_offset_tip_m,
            dtype=float,
        )
        free_axis = np.asarray(self.ik_config.orientation_free_axis, dtype=float)
        self._free_axis = free_axis / float(np.linalg.norm(free_axis))
        lower, upper = self.handle.position_limits
        self._span = np.maximum(upper - lower, 1e-9)
        # The global restart ladder, deterministic and computed once.  Halton
        # rather than random for the same reason ``RobustIKSolver`` uses it: the
        # starts cover the joint box repeatably, so a rerun of a failed plan
        # explores the same hypotheses in the same order.
        self._restarts = tuple(
            lower + _halton(index, self.handle.dof) * (upper - lower)
            for index in range(1, config.ik_restarts + 1)
        )

    # ------------------------------------------------------------------ seams

    def _arm_seed(self, current: object | None) -> np.ndarray:
        lower, upper = self.handle.position_limits
        if current is None:
            # The home pose, not zeros: q_arm = 0 is genuinely self-colliding on
            # this robot (mount vs wrist, 27 mm), and a colliding start makes
            # every collision-checked attempt begin inside an obstacle.
            return self.handle.extract(self.handle.seed)
        values = np.asarray(current, dtype=float)
        if values.shape != (self.handle.dof,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"current joints must have shape ({self.handle.dof},) and be finite",
            )
        return np.clip(values, lower, upper)

    def _errors(
        self,
        scene: object,
        goal: np.ndarray,
        joints: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        actual = np.asarray(
            scene.forwardKinematics(
                self.handle.expand(joints),
                self.tip_link,
                self.base_link,
            ),
            dtype=float,
        )
        position = control_point_delta(
            actual[:3, 3], actual[:3, :3], goal[:3, 3], goal[:3, :3], self._offset,
        )
        residual = rotation_log(goal[:3, :3] @ actual[:3, :3].T)
        return (
            float(np.linalg.norm(position)),
            float(np.linalg.norm(residual)),
            residual,
        )

    def _accepted(
        self,
        goal_rotation: np.ndarray,
        position_error: float,
        orientation_error: float,
        residual: np.ndarray,
    ) -> bool:
        """The repo's acceptance gate, not SimpleIk's.

        Duplicated from ``PinocchioIKSolver._orientation_accepted`` because that
        gate is a bound method on a Pinocchio-owning object and this file must
        import cleanly without it.  The two must stay in step: they decide the
        same question for the same deployment.
        """

        if position_error >= self.ik_config.position_tolerance_m:
            return False
        free_tolerance = self.ik_config.orientation_free_axis_tolerance_rad
        if free_tolerance <= 0.0:
            return orientation_error < self.ik_config.orientation_tolerance_rad
        axis = goal_rotation @ self._free_axis
        axial = abs(float(residual @ axis))
        transverse = float(np.linalg.norm(residual - (residual @ axis) * axis))
        return (
            transverse < self.ik_config.orientation_tolerance_rad
            and axial < free_tolerance
        )

    def _solution(
        self,
        scene: object,
        joints: np.ndarray,
        position_error: float,
        orientation_error: float,
        seed_index: int,
    ) -> IKSolution:
        # Manipulability is a grasp scoring term (grasp_pipeline.py:963), so it
        # is the same quantity the other two backends report: the product of the
        # singular values of the 6x6 arm Jacobian.  ``chain.jacobian`` is
        # base-frame and Pinocchio's is LOCAL; that difference is invisible here
        # because the two differ by an Adjoint, whose determinant is 1.
        jacobian = np.asarray(
            scene.computeFrameJacobian(
                self.handle.expand(joints),
                self.tip_link,
                True,
            ),
            dtype=float,
        )[:, self.handle.v_indices]
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        lower, upper = self.handle.position_limits
        margin = np.minimum(
            (joints - lower) / self._span,
            (upper - joints) / self._span,
        )
        return IKSolution(
            joints=joints.copy(),
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            manipulability=float(np.prod(np.maximum(singular_values, 0.0))),
            # SimpleIk reports no iteration count.  Nothing under ``z_manip``
            # reads this field (only two robust_ik tests do), and 0 is what the
            # existing exact-target fast paths already return.
            iterations=0,
            seed_index=seed_index,
            min_joint_limit_margin=float(np.min(margin)),
        )

    # ----------------------------------------------------------------- ladder

    def _ladder(
        self,
        goal: np.ndarray,
        starts: Iterable[np.ndarray],
        *,
        timeout_s: float,
        operation: str,
        control: PlanningControl | None,
        reference: np.ndarray | None = None,
        step_limit: float | None = None,
        best: tuple[float, float] = (float("inf"), float("inf")),
    ) -> IKSolution:
        parent = control or PlanningControl()
        budget = parent.limited_to(timeout_s, operation)
        lower, upper = self.handle.position_limits
        unsolved = 0
        # SimpleIk converged on its own tip-origin norm and our control-point
        # gate then rejected it.  Split out from ``unsolved`` because the two
        # mean opposite things: "the pose is out of reach" versus "the gate is
        # tighter than the stopping norm", and only the second is tunable.
        gate_rejections = 0
        step_rejections = 0
        limit_rejections = 0
        budget_exhausted = False

        # One acquisition for the whole ladder: roboplan documents Scene
        # collision queries as not thread safe, and every attempt below runs FK,
        # a Jacobian and a collision-checked solve against the same scratch.
        with self.handle.exclusive() as scene:
            # ``SimpleIk`` builds its start through ``Scene::toFullJointPositions``
            # (roboplan scene.cpp:428 -- ``q_out = cur_state_.positions``), so
            # every NON-group coordinate -- the four wheels, twelve leg joints,
            # the chassis -- comes from whatever the previous caller left in the
            # Scene, NOT from the group-sized start passed below.  Pin them to the
            # body ``handle.expand`` assumes, or the collision check and the
            # metrics answer for two different robots.  Invisible today only
            # because the enforced capsule model is PiPER-only (22 ``cap_*``
            # bodies; ``roboplan_scene.write_planning_model`` strips the rest), so
            # no non-arm joint can move a verdict -- add one leg capsule and IK
            # starts clearing the arm against a body that is not there, silently.
            # Deliberately not restored, for the reason roboplan_reachability
            # gives: ``handle.seed`` is the canonical safe pose, and handing the
            # next adapter back what we found is the failure mode, not the fix.
            scene.setJointPositions(self.handle.seed)
            for seed_index, start in enumerate(starts):
                try:
                    budget.checkpoint(operation)
                except PlanningDeadlineExceeded:
                    # A caller-owned deadline or cancellation always propagates;
                    # exhausting only this solver's own budget rejects the pose.
                    checkpoint(control, operation)
                    budget_exhausted = True
                    break
                options = {
                    "group_name": self.handle.group,
                    "max_iters": self.config.ik_max_iterations,
                    # Zero, not one.  Measured: with ``max_restarts >= 1``
                    # SimpleIk draws its restart from the Scene-global RNG, so
                    # the same (goal, start, options) returns different joints
                    # on repeated calls and intermittently fails outright -- a
                    # branch flip in ``solve_continuation`` and unreproducible
                    # plans in both.  At 0 it is a pure deterministic descent
                    # from ``start`` and the multi-start ladder is entirely
                    # ours.  Measured cost of giving up roboplan's restarts:
                    # none, 60/60 solved either way.
                    "max_restarts": 0,
                    "max_linear_error_norm": self.config.ik_position_tolerance_m,
                    "max_angular_error_norm": self.config.ik_orientation_tolerance_rad,
                    # The capsule Scene is the enforced collision model; a
                    # solution that is not collision-free is not usable here.
                    "check_collisions": True,
                    "max_time": max(
                        budget.deadline_s - budget.monotonic_fn(),
                        _MIN_CALL_BUDGET_S,
                    ),
                }
                solved, joints = _solve_ik(
                    scene,
                    self._joint_names,
                    (self.base_link, self.tip_link),
                    goal,
                    start,
                    options,
                )
                if not solved or joints.shape != (self.handle.dof,) or not np.all(
                    np.isfinite(joints)
                ):
                    unsolved += 1
                    continue
                if np.any(joints < lower - _JOINT_LIMIT_SLACK_RAD) or np.any(
                    joints > upper + _JOINT_LIMIT_SLACK_RAD
                ):
                    # Not reachable by either existing backend, so nothing
                    # downstream re-checks it; see _JOINT_LIMIT_SLACK_RAD.
                    limit_rejections += 1
                    continue
                if step_limit is not None and float(
                    np.max(np.abs(joints - reference))
                ) > step_limit:
                    # The whole point of continuation: a solution outside the
                    # trust box is the branch flip the caller rejects anyway
                    # (grasp_pipeline.py:648-654), found one stage earlier.
                    step_rejections += 1
                    continue
                position_error, orientation_error, residual = self._errors(
                    scene, goal, joints,
                )
                if position_error + orientation_error < sum(best):
                    best = (position_error, orientation_error)
                if not self._accepted(
                    goal[:3, :3], position_error, orientation_error, residual,
                ):
                    gate_rejections += 1
                    continue
                checkpoint(control, operation)
                return self._solution(
                    scene, joints, position_error, orientation_error, seed_index,
                )

        counters = (
            f"unsolved_starts={unsolved}; gate_rejections={gate_rejections}; "
            f"limit_rejections={limit_rejections}; "
            f"budget_exhausted={str(budget_exhausted).lower()}"
        )
        if step_limit is None:
            raise IKFailure(
                "no roboplan IK solution met tolerances "
                f"(best position={best[0]:.6f} m, orientation={best[1]:.6f} rad; "
                f"{counters})",
            )
        raise IKFailure(
            "no local roboplan IK continuation met tolerances "
            f"within max_joint_step_rad={step_limit:.6f} "
            f"(best position={best[0]:.6f} m, orientation={best[1]:.6f} rad; "
            f"step_rejections={step_rejections}; {counters})",
        )

    # ----------------------------------------------------------------- public

    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        *,
        control: PlanningControl | None = None,
    ) -> IKSolution:
        checkpoint(control, _SOLVE)
        goal = RobustIKSolver._validated_goal(target)
        return self._ladder(
            goal,
            (self._arm_seed(current), *self._restarts),
            timeout_s=self.config.ik_timeout_s,
            operation=_SOLVE,
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
        """Track a nearby target without leaving the current IK branch."""

        checkpoint(control, _CONTINUATION)
        goal = RobustIKSolver._validated_goal(target)
        reference = np.asarray(current, dtype=float)
        if reference.shape != (self.handle.dof,) or not np.all(np.isfinite(reference)):
            raise ValueError(
                f"current joints must have shape ({self.handle.dof},) and be finite",
            )
        step_limit = float(max_joint_step_rad)
        if not np.isfinite(step_limit) or step_limit <= 0.0:
            raise ValueError("max_joint_step_rad must be finite and positive")
        lower, upper = self.handle.position_limits
        if np.any(reference < lower) or np.any(reference > upper):
            raise ValueError("current joints violate the model joint limits")

        with self.handle.exclusive() as scene:
            position_error, orientation_error, residual = self._errors(
                scene, goal, reference,
            )
            if self._accepted(
                goal[:3, :3], position_error, orientation_error, residual,
            ):
                # Exact-target fast path, and a branch-holding guarantee for the
                # alpha=0 sample of any Cartesian segment.  Needed as well as the
                # ladder because SimpleIk stops on its own tighter tip-origin
                # norm and would walk away from an already-accepted pose.
                return self._solution(
                    scene, reference, position_error, orientation_error, 0,
                )

        trust_lower = np.maximum(lower, reference - step_limit)
        trust_upper = np.minimum(upper, reference + step_limit)
        if np.any(trust_upper - trust_lower <= 2e-10):
            raise IKFailure("roboplan IK continuation has a degenerate trust region")
        starts = (
            reference,
            *(
                trust_lower
                + _halton(index, self.handle.dof) * (trust_upper - trust_lower)
                for index in range(1, self.ik_config.continuation_fallback_seeds + 1)
            ),
        )
        return self._ladder(
            goal,
            starts,
            timeout_s=self.ik_config.continuation_timeout_s,
            operation=_CONTINUATION,
            control=control,
            reference=reference,
            step_limit=step_limit,
            # Seed the reported residual from the pose we are continuing FROM,
            # as RobustIKSolver does.  Without it a ladder whose every attempt
            # is step-rejected reports "best position=inf", which tells the
            # person reading the log at 3am nothing about how far off it was.
            best=(position_error, orientation_error),
        )
