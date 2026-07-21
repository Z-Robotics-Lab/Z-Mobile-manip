"""pick(X) skill (L3) — GRASP stage.

Contract (``docs/plan.md`` §3 GRASP, §2 base-pose gate):
    entry:   alignment stable AND base pose gate passes.
    action:  :class:`~z_manip.models.grasp_source.GraspSource` candidates →
             symmetry-expand SE(3) → IK/plan filter (:mod:`z_manip.ik`) →
             MoveIt2-RRT plan to pre-grasp → Cartesian straight-line approach →
             close gripper → lift. Re-check the pose gate throughout (over-
             threshold → abort, arm back to STOW).
    verify:  candidate approach axis vs surface normal <θ_app (30°) AND IK has a
             solution AND plan succeeds; after close, aperture ∈ (0, max); lift Δz
             met (this is a FREE-signal proxy for "picked up" — M3; the real GT
             predicate is the VERIFY stage / M4, actor cannot author it).
    timeout: 40 sim-s.
    degrade: IK no-solution → re-standoff (budget 3); plan fail → next candidate
             (budget 5); close empty → retry close (budget 2); pose unstable →
             STOW + retreat ALIGN (§3a).

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

APPROACH_AXIS_TOLERANCE_DEG = 30.0  # θ_app
# Retry budgets (docs/plan.md §3a).
MAX_IK_RESTANDOFF = 3
MAX_PLAN_CANDIDATES = 5
MAX_CLOSE_RETRIES = 2


def pick(target_pose: object, *, timeout_sim_s: float = 40.0) -> bool:
    """Generate, plan, and execute a grasp on the aligned target.

    Args:
        target_pose: ``(4, 4)`` SE(3) target pose (aligned).
        timeout_sim_s: GRASP budget in **sim seconds** (RTF-scaled).

    Returns:
        ``True`` if a candidate had IK+plan, the gripper closed to a non-zero
        aperture, and the lift executed. NOTE: mechanical-completion signal, NOT
        a grasp-success verdict — the VERIFY stage judges success against GT.

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("pick is an M0 skeleton; see docs/plan.md §3 GRASP.")
