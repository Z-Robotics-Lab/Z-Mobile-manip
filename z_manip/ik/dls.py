"""IK stage ④ — near-singular variable-damping DLS fallback.

Contract (``docs/plan.md`` §4a stage ④): when the goal is near a singularity or
the workspace boundary, a plain Jacobian inverse blows up. Damped Least Squares
with variable damping (Chiaverini adaptive / SDLS by SVD, which damps ONLY the
singular direction so away-from-singular precision is not thrown away) gives a
stable, bounded step. If DLS still cannot reach the goal, escalate to a base
re-standoff (the mobile base is the first IK resource — §4a).

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

from typing import Optional


def solve_dls(goal_pose: object, seed_joints: object) -> Optional[object]:
    """Iterate a variable-damping DLS step toward ``goal_pose``.

    Args:
        goal_pose: ``(4, 4)`` SE(3) end-effector goal in the arm base frame.
        seed_joints: ``(6,)`` current joint state to iterate from.

    Returns:
        ``(6,)`` joint solution (rad) if DLS converges, else ``None`` (caller
        then re-standoffs the base).

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError(
        "IK stage ④ (DLS) is an M0 skeleton; see docs/plan.md §4a.",
    )
