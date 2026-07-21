"""Planner contract — arm motion planning (L1).

Backend (G5, ``docs/plan.md`` §4): MoveIt2 + OMPL RRTConnect as the CEO-mandated
starting point; VAMP (RRTConnect) is a later upgrade requiring per-robot codegen.
Callers depend only on this Protocol, so the swap is config, not a caller edit.

A planner takes a start joint state + a Cartesian or joint goal and returns a
collision-free trajectory (or reports failure so the GRASP retry budget can pick
a different candidate / re-standoff — ``docs/plan.md`` §3a). The planner plans;
IK feasibility of a grasp pose is a separate concern owned by :mod:`z_manip.ik`.

M0 skeleton: contract only. New external dependency (MoveIt2/VAMP) = CEO gate —
declared in the blueprint, not pulled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, eq=False)
class JointTrajectory:
    """A planned joint-space trajectory for the 6 arm joints.

    Attributes:
        joint_names: Ordered active-joint names. Parallel-gripper mimic joints
            are excluded and gripper aperture is commanded separately.
        waypoints: ``(T, N)`` joint positions over ``T`` steps.
        times: ``(T,)`` per-waypoint time-from-start (seconds), or ``None`` if
            the executor re-times.
    """

    joint_names: Sequence[str]
    waypoints: object
    times: Optional[object] = None


class PlanningError(RuntimeError):
    """No collision-free plan was found for the given start/goal.

    Recoverable: the GRASP stage swaps to the next grasp candidate or
    re-standoffs (retry budget in ``docs/plan.md`` §3a).
    """


@runtime_checkable
class Planner(Protocol):
    """Arm motion-planner interface (MoveIt2-RRT baseline / VAMP upgrade)."""

    def plan(
        self,
        start_joints: object,
        goal_pose: object,
        *,
        timeout_s: float = 5.0,
    ) -> JointTrajectory:
        """Plan a collision-free arm trajectory to a Cartesian goal pose.

        Args:
            start_joints: ``(6,)`` current arm joint positions (rad).
            goal_pose: ``(4, 4)`` SE(3) end-effector goal in the arm base frame.
            timeout_s: Planning budget (wall/plan time, not a sim-s gate).

        Raises:
            PlanningError: if no collision-free plan is found.
            NotImplementedError: in M0 — no planner is wired.
        """
        ...
