"""IK stage ③ — the solver (TRAC-IK primary / pick_ik backup).

Contract (``docs/plan.md`` §4a stage ③): solve joint angles for a Cartesian goal
over the 6 arm joints (:data:`z_manip.ik.ARM_IK_JOINTS`). TRAC-IK (SQP, joint-
limit robust) is primary; pick_ik (local for Cartesian, global to rescue a far
initial guess) is the backup. The gripper mimic joints are EXCLUDED — TRAC-IK
does not support mimic joints, and the gripper width is controlled separately
(``docs/plan.md`` §4a, §4 PiPER row).

M0 skeleton: contract only. TRAC-IK / pick_ik = CEO gate (new external
dependency), not pulled here.
"""

from __future__ import annotations

from typing import Optional


class IKError(RuntimeError):
    """No IK solution for the goal within joint limits.

    The caller escalates to stage ④ (DLS) and, failing that, re-standoffs the
    base (``docs/plan.md`` §4a stage ④).
    """


def solve_ik(goal_pose: object, seed_joints: Optional[object] = None) -> object:
    """Solve the 6 arm joints for a Cartesian ``goal_pose``.

    Args:
        goal_pose: ``(4, 4)`` SE(3) end-effector goal in the arm base frame.
        seed_joints: Optional ``(6,)`` seed joint state for the local solver.

    Returns:
        ``(6,)`` joint solution (rad) over :data:`z_manip.ik.ARM_IK_JOINTS`.

    Raises:
        IKError: if no solution is found within joint limits.
        NotImplementedError: in M0 — skeleton (TRAC-IK/pick_ik CEO-gated).
    """
    raise NotImplementedError(
        "IK stage ③ (solver) is an M0 skeleton; see docs/plan.md §4a.",
    )
