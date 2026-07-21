"""grasp_exec primitive (L2) — pre-grasp → straight-line approach → close → lift.

Contract (``docs/plan.md`` §3 GRASP, §3b): owns the ROS boundary for a grasp
attempt. Build a :class:`~z_manip.models.grasp_source.GraspContext` from the
current observation (cloud/bbox + TF), get candidates from the L1
:class:`~z_manip.models.grasp_source.GraspSource` cascade, expand each by the SE(3)
symmetry family + filter for IK/collision (:mod:`z_manip.ik`), plan a pre-grasp
with the :class:`~z_manip.models.planner.Planner` (MoveIt2-RRT), then execute a
Cartesian straight-line approach (min-jerk, open-loop — D435i min-Z 0.28 m blinds
the last ~30 cm), close the gripper, and lift.

Explicitly NO contact detection during the straight-line sting (gravity/PD false
triggers) — the design deliberately borrows the reference Cartesian approach and
relies on gripper-aperture margin (``docs/plan.md`` §4, §3b, and the "don't copy"
note on close-is-success verification: verify is a separate concern, invariant 1).

M0 skeleton: signature + docstring only. Grasp verification (aperture ∈ (0,max) +
lift Δz + object-follows) is the caller's concern (VERIFY stage), never asserted
from a PASS flag here.
"""

from __future__ import annotations


def grasp_exec(
    grasp_context: object,
    *,
    lift_dz: float = 0.10,
    approach_timeout_sim_s: float = 40.0,
) -> bool:
    """Execute one grasp attempt: pre-grasp → approach → close → lift.

    Args:
        grasp_context: A :class:`~z_manip.models.grasp_source.GraspContext` for
            the target object.
        lift_dz: Lift height after closing (m) — checked against the GRASP gate
            downstream, not certified here.
        approach_timeout_sim_s: GRASP budget in **sim seconds** (RTF-scaled).

    Returns:
        ``True`` if the mechanical sequence completed (candidate had IK/plan,
        gripper closed to a non-zero aperture, lift executed). NOTE: this is a
        mechanical-completion flag, NOT a grasp-success verdict — success is
        judged by the VERIFY stage against ground truth the actor cannot author.

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError(
        "grasp_exec is an M0 skeleton; see docs/plan.md §3 GRASP / §3b.",
    )
