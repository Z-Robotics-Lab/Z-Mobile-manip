"""IK near-limit four-stage pipeline (``docs/plan.md`` §4a).

The PiPER has a 626 mm reach; the dog's back-mounted arm often reaches for far
objects near the workspace boundary where IK is ill-conditioned. The mobile base
is the FIRST IK resource — hard-solving boundary IK is a fallback, not the norm.
The four stages:

    ① :mod:`z_manip.ik.reachability` — candidate reachability / manipulability
       filter (Reuleaux inverse-reachability to pick a base standoff that lands
       the target in PiPER's dexterous core; cuRobo GPU batch-IK to filter the
       reachable+collision-free subset — OFF-BOARD compute, never on the dog NUC).
    ② :mod:`z_manip.ik.symmetry` — parallel-gripper approach-axis relaxation:
       per candidate, enumerate N samples about the approach axis + a 180° yaw
       flip (two-finger swap equivalence), expanding to a family of SE(3) targets.
    ③ :mod:`z_manip.ik.solver` — TRAC-IK (SQP, joint-limit robust; EXCLUDES the
       gripper mimic joint, solves the 6 arm joints only) primary / pick_ik backup.
    ④ :mod:`z_manip.ik.dls` — near-singular variable-damping DLS (Chiaverini /
       SDLS by SVD); still no solution → re-standoff the base.

Skeleton only (M0). No solver is wired; new external dependencies (TRAC-IK /
pick_ik / cuRobo) = CEO gate — declared in the blueprint, not pulled here.
"""

from __future__ import annotations

__all__: list[str] = []

# The 6 arm joints the IK chain solves. The parallel-gripper mimic joints
# (piper_joint7/8) are EXCLUDED — TRAC-IK does not support mimic joints, and the
# gripper width is controlled separately (docs/plan.md §4a, §4 PiPER row).
ARM_IK_JOINTS: tuple[str, ...] = (
    "piper_joint1",
    "piper_joint2",
    "piper_joint3",
    "piper_joint4",
    "piper_joint5",
    "piper_joint6",
)
