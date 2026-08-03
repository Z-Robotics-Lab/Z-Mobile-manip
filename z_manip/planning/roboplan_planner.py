"""roboplan's C++ RRT-Connect behind this repo's ``JointPlanner`` seam.

Shadows :class:`~z_manip.planning.rrt_connect.JointSpaceRRTConnect` exactly:
same ``plan_joint``/``segment_valid`` signatures, same ``JointTrajectory``
return, same ``PlanningError`` messages for the endpoint rejections, same
``RRTConnectConfig`` knobs, same straight-line early return, same full-path
collision audit before anything is returned.  Swapping the two is a wiring
change, not a caller edit.  Nothing imports this module today; it is reachable
only from a deployment that has already opted into roboplan.

What the C++ backend fixes by construction
==========================================
*   **Nearest neighbour.**  ``rrt_connect.py:219-223`` rescans every node per
    extension.  roboplan keeps a k-d tree.
*   **Shortcutting.**  ``rrt_connect.py:273-289`` samples random index pairs
    only.  ``PathShortcutter`` interleaves a redundant-vertex removal pass
    (``redundant_removal_iters``) that cleans up the micro-segments random
    shortcutting leaves behind, and stops early once shortcuts stop landing
    (``max_convergence_iters``).

Three measured facts this file is built around
==============================================
1.  ``RRT.plan`` reads every *non-group* joint from the Scene's CURRENT joint
    positions, and a freshly built Scene's current positions are invalid (the
    four Go2-W wheels are continuous joints carried as ``(cos, sin)`` and come
    back all-zero).  Without ``setJointPositions`` first, a perfectly good start
    is rejected as ``"Invalid start configuration requested"``.  Measured: every
    plan call fails until the positions are set.
2.  ``max_planning_time`` is checked BETWEEN iterations, not inside one edge
    collision check.  Measured: a 0.30 s budget with a 1e-5 rad check step
    returned successfully after 8.2 s.  So it is an advisory here; the
    authority is ``budget.checkpoint()`` immediately after the call returns
    -- on BOTH exits, because roboplan cannot tell "the geometry defeated me"
    from "you gave me no time" and reports the second as the first.  A
    caller's ``cancel_check`` cannot fire during the call at all.
3.  With this repo's capsule Scene the whole thing is fast: 2.4-3.8 ms to plan,
    ~9 ms to shortcut, 24/24 solved over random in-limit goals.

``rrt_star=True`` + ``fast_return=False`` is deliberately NOT exposed.  It would
give an anytime "keep improving until the deadline" mode the numpy planner
cannot do, but ``RRT.plan`` is one uninterruptible C++ call: with
``fast_return=True`` a cancellation lands within the few milliseconds the call
takes, while ``fast_return=False`` burns the entire budget on every query and a
cancelled grasp candidate would hold the planner for seconds.  Add it when a
caller actually wants path optimality more than it wants to be interruptible.

roboplan is imported lazily inside :func:`_roboplan`: this module must import
cleanly on a host that has neither roboplan nor pinocchio.
"""

from __future__ import annotations

import math
import time

import numpy as np

from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning_control import PlanningControl, checkpoint

from .roboplan_runtime import RoboplanUnavailable, SceneHandle
from .rrt_connect import RRTConnectConfig, _RRTBudget

__all__ = ["RoboplanJointPlanner"]

# roboplan reads ``max_planning_time <= 0`` as UNBOUNDED, so an already-spent
# budget must never round down to zero -- that hands the C++ planner an
# unkillable search on the one code path where we are trying to stop it.  The
# floor is deliberately far below any useful budget: the checkpoint after the
# call is what actually enforces the deadline, this only keeps the search finite.
_MIN_PLANNING_TIME_S = 1e-3

# roboplan's own default, and the right one here: bisection finds the blocking
# sample of a mostly-free edge in far fewer collision queries than a linear
# scan, and at a fixed step size both are complete to the same resolution.
_BISECTION = True


def _roboplan() -> tuple[object, object]:
    """Import roboplan's core + RRT bindings.

    The only import site, and the seam the host-side tests replace with a fake.
    """

    try:
        import roboplan.core as core  # noqa: PLC0415 -- 4090-only dependency
        import roboplan.rrt as rrt  # noqa: PLC0415 -- 4090-only dependency
    except ImportError as error:
        raise RoboplanUnavailable(
            "import-failed",
            f"{type(error).__name__}: {error}",
        ) from error
    return core, rrt


def _advisory_budget(ceiling: float, now: float) -> float:
    """The ``max_planning_time`` roboplan is handed, floored so it stays finite.

    ``PlanningControl`` lets the caller supply ``monotonic_fn``, so ``now`` can
    land on or past ``ceiling`` -- a coarse clock, a jump, a test double.  The
    subtraction then yields ``0.0`` or a negative, both of which roboplan reads
    as "no limit": the one moment we most want the search to stop is the one
    that would make it run forever.
    """

    return max(float(ceiling) - float(now), _MIN_PLANNING_TIME_S)


def _densify(waypoints: np.ndarray, resolution: float, budget: _RRTBudget) -> np.ndarray:
    """Resample a sparse path so no consecutive pair exceeds ``resolution``.

    The shortcutter legitimately collapses a 645-waypoint path to two, and the
    executor's contract (matched to ``JointSpaceRRTConnect``) is that adjacent
    waypoints are never further apart than the collision-checking resolution.
    """

    dense = [waypoints[0]]
    for first, second in zip(waypoints, waypoints[1:]):
        budget.checkpoint("roboplan path densification")
        distance = float(np.linalg.norm(second - first))
        steps = max(1, int(np.ceil(distance / resolution)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            dense.append(first + alpha * (second - first))
    return np.asarray(dense, dtype=float)


class RoboplanJointPlanner:
    """``JointPlanner`` backed by roboplan's RRT-Connect and PathShortcutter.

    Every Scene call happens inside ``SceneHandle.exclusive()``; roboplan
    documents Scene collision queries as not thread safe.
    """

    def __init__(
        self,
        handle: SceneHandle,
        *,
        config: RRTConnectConfig | None = None,
    ) -> None:
        self._handle = handle
        self.config = config or RRTConnectConfig()
        self._core, self._rrt = _roboplan()
        with handle.exclusive() as scene:
            group = scene.getJointGroupInfo(handle.group)
            self.joint_names = tuple(str(name) for name in group.joint_names)
            continuous = bool(getattr(group, "has_continuous_dofs", False))
        if len(self.joint_names) != handle.dof:
            raise ValueError(
                f"group {handle.group} names {len(self.joint_names)} joints but the "
                f"scene handle reports {handle.dof} degrees of freedom",
            )
        # A continuous joint inside the planning group is carried collapsed as
        # (cos, sin) in the Scene but expanded in ``JointPath.positions``, so
        # ``dof`` would stop indexing both.  Nothing in this repo has one; fail
        # closed rather than silently mis-scatter a returned path.
        if continuous:
            raise ValueError(
                f"group {handle.group} has continuous joints; this adapter assumes "
                "one position per degree of freedom",
            )

    # ------------------------------------------------------------- validation

    def _validate_state(
        self,
        scene: object,
        joints: object,
        label: str,
        budget: _RRTBudget,
    ) -> np.ndarray:
        """Fail closed on an unusable endpoint, naming which one and why.

        roboplan collapses both failures into one ``RuntimeError`` per endpoint
        and reports a NaN configuration as "in collision", so the distinction
        callers act on -- retry this grasp from a different seed vs. abandon the
        candidate -- is made here, before roboplan ever sees the request.
        """

        budget.checkpoint(f"roboplan {label} validation")
        values = np.asarray(joints, dtype=float)
        expected = (self._handle.dof,)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise PlanningError(f"{label} joint state must be a finite {expected} vector")
        lower, upper = self._handle.position_limits
        if np.any(values < lower) or np.any(values > upper):
            worst = int(np.argmax(np.maximum(lower - values, values - upper)))
            raise PlanningError(
                f"{label} joint state violates joint limits: "
                f"{self.joint_names[worst]} = {values[worst]:.6f} rad outside "
                f"[{lower[worst]:.6f}, {upper[worst]:.6f}]",
            )
        collides = bool(scene.hasCollisions(self._handle.expand(values)))
        budget.checkpoint(f"roboplan {label} validation")
        if collides:
            raise PlanningError(f"{label} joint state is in collision")
        return values

    def _plan_failure(self, message: str, timeout_s: float) -> PlanningError:
        """Translate roboplan's ``RuntimeError`` text into this repo's reasons."""

        lowered = message.lower()
        for label in ("start", "goal"):
            if f"{label} configuration" in lowered:
                # ``_validate_state`` already proved this endpoint is in-limits
                # and collision-free against the very same Scene, so roboplan is
                # objecting to something our checks cannot see: in practice the
                # Scene's current joint positions, which supply every non-group
                # joint of the configuration it actually evaluates.  Say so --
                # the alternative reads as a spurious infeasible grasp.
                return PlanningError(
                    f"{label} joint state rejected by roboplan after this adapter "
                    f"validated it as in-limits and collision-free: {message}",
                )
        return PlanningError(
            "no collision-free joint path found within "
            f"{self.config.max_iterations} nodes / {timeout_s:.3f} s: {message}",
        )

    # ------------------------------------------------------------- the seam

    def segment_valid(
        self,
        first: object,
        second: object,
        *,
        control: PlanningControl | None = None,
    ) -> bool:
        """Continuously validate one in-limit joint-space segment."""

        checkpoint(control, "roboplan segment validation")
        first_values = np.asarray(first, dtype=float)
        second_values = np.asarray(second, dtype=float)
        expected = (self._handle.dof,)
        if first_values.shape != expected or second_values.shape != expected:
            return False
        if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
            return False
        lower, upper = self._handle.position_limits
        for values in (first_values, second_values):
            if np.any(values < lower) or np.any(values > upper):
                return False
        with self._handle.exclusive() as scene:
            blocked = self._core.hasCollisionsAlongPath(
                scene,
                self._handle.expand(first_values),
                self._handle.expand(second_values),
                self.config.collision_resolution,
                _BISECTION,
                True,  # check_endpoints: matches the numpy planner's semantics
            )
        checkpoint(control, "roboplan segment validation")
        return not bool(blocked)

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
        clock = time.monotonic if control is None else control.monotonic_fn
        # The advisory ceiling roboplan is handed.  Read one tick after the
        # budget's own, so it can only ever be marginally LATER than the local
        # timeout -- harmless, because ``budget.checkpoint()`` right after the
        # call is what decides, and it is the tighter of the two.
        ceiling = float(clock()) + float(timeout_s)
        if control is not None and control.deadline_s is not None:
            ceiling = min(ceiling, float(control.deadline_s))

        with self._handle.exclusive() as scene:
            start = self._validate_state(scene, start_joints, "start", budget)
            goal = self._validate_state(scene, goal_joints, "goal", budget)

            # Straight line first, exactly as rrt_connect.py:321-325 does.  Kept
            # for two reasons: it is part of the contract callers already see
            # (a directly reachable pair yields the direct path, deterministic
            # and minimal), and roboplan does NOT guarantee it -- its trees grow
            # toward random samples and the shortcutter is stochastic, so the
            # trivial query would come back as some longer, seed-dependent path.
            if not bool(
                self._core.hasCollisionsAlongPath(
                    scene,
                    self._handle.expand(start),
                    self._handle.expand(goal),
                    self.config.collision_resolution,
                    _BISECTION,
                    True,
                )
            ):
                budget.checkpoint("roboplan straight-line check")
                return JointTrajectory(
                    self.joint_names,
                    _densify(np.asarray([start, goal]), self.config.collision_resolution, budget),
                )

            budget.checkpoint("roboplan RRT planning")
            previous = np.array(scene.getCurrentJointPositions(), dtype=float, copy=True)
            try:
                # See fact 1 in the module docstring: the RRT reads every
                # non-group joint from here, not from the request.  The
                # shortcutter re-checks every edge it proposes, so it is held
                # inside the same window -- measured, its result is identical
                # either way today, but only because no enforced capsule pair
                # moves with the wheels, and that is a property of the current
                # model rather than a guarantee.
                scene.setJointPositions(self._handle.expand(start))
                advisory = _advisory_budget(ceiling, clock())
                try:
                    path = self._run(scene, start, goal, advisory, timeout_s)
                except PlanningError:
                    # roboplan failed -- but the caller acts on WHY.  The C++
                    # call is uninterruptible, so a deadline that expired or a
                    # cancellation that arrived during it only becomes visible
                    # now, and roboplan reports both as its own "timed out" /
                    # "max nodes" failure.  Translating that straight through
                    # yields a RECOVERABLE PlanningError, and every caller in
                    # grasp_pipeline.py reacts to one by swapping to the next
                    # grasp candidate and planning again -- past a deadline that
                    # is already gone, or for a request nobody wants any more.
                    # Ask the budget first: it raises PlanningAborted when it,
                    # not the geometry, is the reason we stopped.
                    budget.checkpoint("roboplan RRT planning")
                    raise
                budget.checkpoint("roboplan RRT planning")
                if self.config.shortcut_attempts:
                    path = self._core.PathShortcutter(
                        scene,
                        self._core.PathShortcuttingOptions(
                            group_name=self._handle.group,
                            max_step_size=self.config.collision_resolution,
                            max_iters=self.config.shortcut_attempts,
                            seed=self.config.seed,
                        ),
                    ).shortcut(path)
                    budget.checkpoint("roboplan path shortcutting")
            finally:
                # Shared mutable Scene state; leave it as we found it so a peer
                # adapter's next query does not silently answer for our start.
                scene.setJointPositions(previous)

            waypoints = self._as_waypoints(path, start, goal)
            dense = _densify(waypoints, self.config.collision_resolution, budget)
            # Never return a path we have not re-validated end to end
            # (rrt_connect.py:347-354).  This is not redundant with the
            # planner's own checking: it is what catches a bug in the
            # shortcutting, the densification, or the arm -> full-model scatter
            # above, none of which roboplan checked.
            for joints in dense:
                budget.checkpoint("roboplan final collision audit")
                collides = bool(scene.hasCollisions(self._handle.expand(joints)))
                budget.checkpoint("roboplan final collision audit")
                if collides:
                    raise PlanningError(
                        "internal error: path failed final collision audit",
                    )
            return JointTrajectory(self.joint_names, dense)

    # ------------------------------------------------------------- internals

    def _run(
        self,
        scene: object,
        start: np.ndarray,
        goal: np.ndarray,
        max_planning_time: float,
        timeout_s: float,
    ) -> object:
        options = self._rrt.RRTOptions(
            group_name=self._handle.group,
            max_nodes=self.config.max_iterations,
            max_connection_distance=self.config.step_size,
            collision_check_step_size=self.config.collision_resolution,
            collision_check_use_bisection=_BISECTION,
            goal_biasing_probability=self.config.goal_bias,
            max_planning_time=max_planning_time,
            rrt_connect=True,
        )
        planner = self._rrt.RRT(scene, options)
        planner.setRngSeed(self.config.seed)
        try:
            return planner.plan(
                self._core.JointConfiguration(list(self.joint_names), start),
                self._core.JointConfiguration(list(self.joint_names), goal),
            )
        except RuntimeError as error:
            raise self._plan_failure(str(error), timeout_s) from error

    def _as_waypoints(
        self,
        path: object,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Check the returned path before it becomes a robot command."""

        waypoints = np.asarray(path.positions, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1:] != (self._handle.dof,):
            raise PlanningError(
                f"internal error: roboplan returned a {waypoints.shape} path for "
                f"{self._handle.dof} joints",
            )
        if len(waypoints) < 2 or not np.all(np.isfinite(waypoints)):
            raise PlanningError("internal error: roboplan returned an unusable path")
        # Measured exact on the real model.  A drifted endpoint is a silent
        # discontinuity between the plan and the state the executor is in.
        if not np.allclose(waypoints[0], start, atol=1e-9) or not np.allclose(
            waypoints[-1], goal, atol=1e-9
        ):
            raise PlanningError(
                "internal error: roboplan returned a path whose endpoints are not "
                "the requested start and goal",
            )
        return waypoints
