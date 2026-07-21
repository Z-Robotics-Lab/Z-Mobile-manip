"""approach(X) skill (L3) — APPROACH stage.

Contract (``docs/plan.md`` §3 APPROACH):
    entry:   target 3D pose known.
    action:  :func:`~z_manip.primitives.servo_base.servo_base` two-stage — far
             stage /way_point (localPlanner avoidance); near stage <1.5 m per the
             /cmd_vel takeover timing (retract way-point, manip_servo drives
             /cmd_vel low-speed + terrain gate).
    verify:  base-target plane distance error <standoff_tol (0.10 m) for ≥4 sim-s;
             or STUCK fallback base-target ≤success_radius (0.5 m).
    timeout: 60 sim-s.
    degrade: STUCK → success_radius judge → re-plan once → give up (budget §3a).

M0 skeleton: signature + docstring only. Near-stage /cmd_vel owner (manip_servo)
+ agent_bridge nav_owner extension = CEO gate (G7).
"""

from __future__ import annotations

MAX_APPROACH_REPLANS = 1  # STUCK re-plan budget (docs/plan.md §3a)
STANDOFF_TOL = 0.10       # m
SUCCESS_RADIUS = 0.5      # m


def approach(target_pose: object, *, timeout_sim_s: float = 60.0) -> bool:
    """Servo the base to standoff of ``target_pose`` (far → near stages).

    Args:
        target_pose: ``(4, 4)`` SE(3) target pose from :mod:`z_manip.skills.find`.
        timeout_sim_s: APPROACH budget in **sim seconds** (RTF-scaled).

    Returns:
        ``True`` on reaching standoff, ``False`` on STUCK/timeout failure.

    Raises:
        NotImplementedError: in M0 — skeleton (near-stage owner is CEO-gated G7).
    """
    raise NotImplementedError(
        "approach is an M0 skeleton; see docs/plan.md §3 APPROACH (G7 gated).",
    )
