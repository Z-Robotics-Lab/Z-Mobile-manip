"""IK stage ① — candidate reachability / manipulability filter.

Contract (``docs/plan.md`` §4a stage ①): pick a base standoff (Reuleaux inverse-
reachability map) that lands the target inside PiPER's dexterous core, away from
the 626 mm boundary; and/or filter a batch of grasp candidates to the reachable +
collision-free subset via cuRobo GPU batch-IK (~37k solves/s). This is OFF-BOARD
compute — it NEVER runs on the dog's NUC (``docs/plan.md`` §4a note, §9 topology).

M0 skeleton: contract only. cuRobo / Reuleaux = CEO gate, not pulled here.
"""

from __future__ import annotations


def filter_reachable(candidates: object, base_pose: object) -> object:
    """Keep the reachable + collision-free subset of grasp candidates.

    Args:
        candidates: ``(M, 4, 4)`` SE(3) grasp poses in the arm base frame.
        base_pose: Current base pose (to evaluate reach from this standoff).

    Returns:
        A reduced set of reachable candidates (same dtype/shape family).

    Raises:
        NotImplementedError: in M0 — skeleton (off-board cuRobo, CEO-gated).
    """
    raise NotImplementedError(
        "IK stage ① (reachability) is an M0 skeleton; see docs/plan.md §4a.",
    )
