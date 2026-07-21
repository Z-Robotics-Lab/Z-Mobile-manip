"""carry skill (L3) — CARRY stage.

Contract (``docs/plan.md`` §3 CARRY, §2 base-pose gate):
    entry:   VERIFY passed.
    action:  arm to CARRY pose (chest-height hold), navigate to the destination;
             the eccentric CARRY payload still holds the base-pose gate.
    verify:  base reaches destination <goal_tol AND aperture stays >0 throughout
             (object not dropped).
    timeout: per path.
    degrade: mid-transit drop (aperture→0 or depth-loss of object) → back to
             SEARCH to re-grasp.

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations


def carry(destination: object, *, goal_tol: float = 0.30) -> bool:
    """Hold the object in CARRY pose and navigate to ``destination``.

    Args:
        destination: Goal pose/point to navigate to (typed at M3+).
        goal_tol: Arrival tolerance at the destination (m).

    Returns:
        ``True`` if the base arrives with the object still held (aperture >0),
        ``False`` on drop.

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("carry is an M0 skeleton; see docs/plan.md §3 CARRY.")
