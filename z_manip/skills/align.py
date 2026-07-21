"""align(X) skill (L3) — ALIGN stage.

Contract (``docs/plan.md`` §3 ALIGN, §2 base-pose gate):
    entry:   at standoff AND base pose gate passes (|pitch|≤12°, |roll|≤10°, read
             /imu or /state_estimation).
    action:  arm LOOKOUT, lock EdgeTAM mask, fine-tune base yaw to face the object
             normal; hold D435i pre-grasp viewpoint ≥0.35 m; re-check the pose
             gate throughout (over-threshold → arm back to STOW + retreat ALIGN).
    verify:  mask IoU frame-to-frame ≥τ_mask (0.5) for ≥K sim-s; base yaw vs
             target normal <φ_yaw (15°); pre-grasp depth valid-ratio >τ_depth;
             pose gate continuously passing.
    timeout: 20 sim-s.
    degrade: mask lost → back to SEARCH; depth holes → retreat; pose unstable →
             STOW + retreat ALIGN (mask-lost budget §3a).

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

MAX_ALIGN_MASK_LOST_RETRIES = 2  # docs/plan.md §3a
MASK_IOU_THRESHOLD = 0.5         # τ_mask
YAW_TOLERANCE_DEG = 15.0         # φ_yaw
# Base pose gate (docs/plan.md §2). Start thresholds; tightened during M0 calib.
BASE_PITCH_GATE_DEG = 12.0
BASE_ROLL_GATE_DEG = 10.0


def align(target_pose: object, *, timeout_sim_s: float = 20.0) -> bool:
    """Lock the mask and yaw-align the base to the object normal at standoff.

    Args:
        target_pose: ``(4, 4)`` SE(3) target pose (refined during ALIGN).
        timeout_sim_s: ALIGN budget in **sim seconds** (RTF-scaled).

    Returns:
        ``True`` when mask + yaw + depth + pose gate all satisfy the ALIGN gate,
        ``False`` on failure (mask lost / depth holes / pose unstable / timeout).

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("align is an M0 skeleton; see docs/plan.md §3 ALIGN.")
