"""place(X) skill (L3) — PLACE stage.

Contract (``docs/plan.md`` §3 PLACE):
    entry:   at destination.
    action:  plan a place pose (above-destination standoff → Cartesian lower to
             release height) → open gripper → lift arm to hand off → arm to STOW.
    verify:  release height reached; after opening, aperture → open; after lifting,
             the object does NOT follow the gripper (sim GT: object odom vs
             gripper distance >hand-off threshold; free signal: object no longer in
             the depth in front of the gripper) → hand-off success.
    timeout: 20 sim-s.
    degrade: lowering blocked / not handed off → lift + retry (budget 2, §3a);
             exhausted → RECOVER "could not place".

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

MAX_PLACE_RETRIES = 2  # docs/plan.md §3a


def place(destination: object, *, timeout_sim_s: float = 20.0) -> bool:
    """Lower, release, and hand off the held object at ``destination``.

    Args:
        destination: Place target pose/point (typed at M3+).
        timeout_sim_s: PLACE budget in **sim seconds** (RTF-scaled).

    Returns:
        ``True`` on confirmed hand-off (object no longer follows the gripper),
        ``False`` on failure after the retry budget.

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("place is an M0 skeleton; see docs/plan.md §3 PLACE.")
