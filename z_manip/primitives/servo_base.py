"""servo_base primitive (L2) — two-stage mobile-base visual servo.

Contract (``docs/plan.md`` §3b, §1 G7; convergence contract borrowed from
``refs/.../visual_servoing_base/README.md`` §Convergence):

    Far stage (>1.5 m): publish ``/way_point``, borrowing the nav stack's
        localPlanner for obstacle avoidance (half-closed loop).
    Near stage (<1.5 m): per the /cmd_vel takeover timing — retract the way-point
        so pathFollower idles at zero velocity, then manip_servo EXCLUSIVELY
        drives ``/cmd_vel`` (low speed + upright gate), 4-10 Hz on **sim time**.

    Single source of truth: at any instant exactly ONE logical producer of
    ``/cmd_vel`` (nav_owner state = manip_servo enforces this — §1 table). NEVER
    let pathFollower and manip_servo both drive ``/cmd_vel``.

Convergence / termination (reference contract): republish a fresh goal every
``goal_rate`` tick; when within ``stop_update_distance`` for the first time arm a
timer; after ``convergence_duration`` continuously inside the window → SUCCESS;
STUCK fallback judges success by actual proximity to the target within
``success_radius`` (0.5 m). All timers on sim time.

M0 skeleton: signature + docstring only. The near-stage /cmd_vel owner
(manip_servo) + the agent_bridge nav_owner extension is a CEO gate (G7) — NOT
crossed here; declared in the blueprint.
"""

from __future__ import annotations


def servo_base(
    *,
    standoff_tol: float = 0.10,
    stop_update_distance: float = 0.75,
    convergence_duration_sim_s: float = 4.0,
    goal_rate_hz: float = 2.0,
    success_radius: float = 0.5,
    near_stage_threshold: float = 1.5,
    timeout_sim_s: float = 60.0,
) -> bool:
    """Drive the base to standoff via the two-stage servo (far → near).

    Args:
        standoff_tol: Plane distance error target at standoff (m); must hold
            ≥4 sim-s (APPROACH gate, §3).
        stop_update_distance: Convergence window entry threshold (m).
        convergence_duration_sim_s: Seconds continuously inside the window
            before SUCCESS (**sim seconds**).
        goal_rate_hz: Max goal republish rate.
        success_radius: STUCK-fallback success distance to the target (m).
        near_stage_threshold: Range (m) at which the near /cmd_vel stage takes
            over from the far /way_point stage.
        timeout_sim_s: APPROACH budget in **sim seconds** (RTF-scaled).

    Returns:
        ``True`` on convergence to standoff, ``False`` on STUCK/timeout failure.

    Raises:
        NotImplementedError: in M0 — skeleton (near-stage owner is CEO-gated G7).
    """
    raise NotImplementedError(
        "servo_base is an M0 skeleton; near /cmd_vel owner is CEO-gated (G7).",
    )
