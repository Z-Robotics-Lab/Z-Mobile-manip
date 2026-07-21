"""arm_goto primitive (L2) — named pose / joint trajectory to the arm.

Contract: move the PiPER arm to a named pose or follow a joint trajectory. The
named-pose channel mirrors the sim-side contract landing in go2w
(``warehouse_nav.py`` subscribes ``/piper/named_pose`` std_msgs/String, value ∈
{STOW, LOOKOUT, CARRY} — ``docs/plan.md`` §M0). This primitive is the z-manip
producer of that same contract, so sim (Isaac publishes the subscriber) and real
(``real_adapter``) share the topic name and semantics.

The three named poses (design authority: the go2w-side FK-derived pose table,
mirrored here as the *contract* names only — the numeric joint targets live on
the Isaac side, not in z-manip):

    STOW    — arm folded/parked; navigation posture (camera axis up, not for
              perception).
    LOOKOUT — camera level (axis pitch ≈0°, G-b); SEARCH/ALIGN default posture.
    CARRY   — chest-height carry hold; slight down-look to frame the held object.

Owner coordination: while the grasp controller is active (grasp status running)
the arm target is owned by the grasp executor, and named-pose writes are
suppressed (single exclusive owner — ``docs/plan.md`` §M0 pose owner rule). The
z-manip side must respect that ordering; it never contends for the arm target.

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

# The named-pose contract vocabulary. These are the *names* on the wire
# (std_msgs/String on /piper/named_pose); the numeric joint targets are owned by
# the Isaac-side publisher (go2w), NOT duplicated here.
NAMED_POSES: tuple[str, ...] = ("STOW", "LOOKOUT", "CARRY")

# The sim-internal String topic both sim (Isaac subscriber) and real
# (real_adapter) agree on. Not routed through the agent-bridge HTTP face (that
# would touch nav_owner / G7); it is a peer of the existing /piper/grasp_cmd
# String channel.
NAMED_POSE_TOPIC = "/piper/named_pose"


def arm_goto(pose_name: str, *, settle_sim_s: float = 5.0) -> bool:
    """Command the arm to a named pose and wait until it settles.

    Args:
        pose_name: One of :data:`NAMED_POSES`.
        settle_sim_s: Seconds (**sim time**) to wait for the joints to settle
            before declaring arrival (arrival gate: per-joint error <0.05 rad,
            ``docs/plan.md`` §M0 / G-c).

    Returns:
        ``True`` once settled within tolerance.

    Raises:
        ValueError: if ``pose_name`` is not a known named pose.
        NotImplementedError: in M0 — skeleton.
    """
    if pose_name not in NAMED_POSES:
        raise ValueError(
            f"Unknown pose '{pose_name}'; expected one of {NAMED_POSES}",
        )
    raise NotImplementedError(
        "arm_goto is an M0 skeleton; see docs/plan.md §M0 named-pose contract.",
    )
