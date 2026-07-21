"""Staged E2E specs (M1–M3) — executable specifications, all skipped until the
milestone lands. Each test body is a placeholder; the CONTRACT lives in the
docstring as Given/When/Then with the exact machine-judgeable gate numbers from
``docs/plan.md`` §3 (state machine) and §5 (milestone gates).

These are the acceptance shapes the bare ``vector-cli`` / zeno face must satisfy,
written now so the target never drifts. When a milestone is implemented, drop
its ``skip`` and wire the body to the same live helpers M0 uses.

All numbers trace to plan.md; sim-time only for every duration/timeout (RTF 0.2
amplifies wall 5×, §3 / pitfall 41). None of these run at M0.
"""

from __future__ import annotations

import pytest

from tests import contract as C  # noqa: F401  (kept for when bodies are wired)
from tests import helpers as H   # noqa: F401


# ============================================================================ M1
@pytest.mark.e2e
@pytest.mark.m1
@pytest.mark.skip(reason="M1 pending")
def test_m1_find_approach():
    """find(X) → SCAN → two-stage approach, landing in the standoff window.

    Given  the chain is green and target X (a can) is in the scene, arm STOW.
    When   the zeno/bridge face receives ``find(X)`` then drives APPROACH:
           far leg publishes /way_point (localPlanner avoidance); near leg
           (<1.5 m) preempts pathFollower and low-speed-directs /cmd_vel
           (§3 APPROACH; G7 /cmd_vel takeover timing).
    Then   base→target planar distance error < standoff_tol = 0.10 m and holds
           for ≥ 4 sim-s (§5 M1 gate); AND the EdgeTAM mask IoU between frames
           is ≥ τ_mask = 0.5 continuously for ≥ 4 sim-s with no lost re-lock
           (§3 SEARCH/ALIGN, §5 M1). STUCK fallback accepts base→target ≤
           success_radius = 0.5 m (§3 APPROACH degrade).
    """
    raise NotImplementedError


@pytest.mark.e2e
@pytest.mark.m1
@pytest.mark.skip(reason="M1 pending")
def test_m1_search_not_found_budget():
    """SEARCH not_found escalates to zeno within the retry budget.

    Given  target X is NOT in the scene, arm STOW.
    When   ``find(X)`` runs SCAN (in-place rotate + LOOKOUT arm sweep) and no
           track reaches τ_det = 0.4 over K = 3 frames within the 30 sim-s
           SEARCH timeout.
    Then   the skill widens scan / re-queries an alias up to N = 2 (§3a budget),
           and on exhaustion terminates reporting "未找到 X" back to zeno —
           it does NOT loop unbounded.
    """
    raise NotImplementedError


# ============================================================================ M2
@pytest.mark.e2e
@pytest.mark.m2
@pytest.mark.skip(reason="M2 pending")
def test_m2_grasp_candidates():
    """Grasp-candidate generation produces frame-aligned, well-oriented grasps.

    Given  M1 reached ALIGN with a stable mask + valid pre-grasp depth on X.
    When   the GraspSource (GT-heuristic/geometric antipodal at pipeline
           bring-up; HGGD later, CUDA 12.8 rebuild gated) runs on the point
           cloud for one frame.
    Then   every frame yields ≥ 1 candidate whose approach axis is within
           θ_app = 30° of the object surface normal (§5 M2 gate); AND each
           candidate pose reprojected into the base frame is numerically
           consistent (frame alignment verified by number, not by eye — §5 M2).
    """
    raise NotImplementedError


# ============================================================================ M3
@pytest.mark.e2e
@pytest.mark.m3
@pytest.mark.skip(reason="M3 pending")
def test_m3_pick():
    """"拿起罐头" — plan + execute a pick, lifting the object with a held grip.

    Given  M2 produced a valid grasp candidate on X in the standoff window,
           base pose gate holding (|pitch|≤12°, |roll|≤10°, §3 GRASP).
    When   the NL command "把罐头拿起来" drives GRASP: MoveIt2-RRT plans the
           pre-grasp, a Cartesian straight-line approach descends, the gripper
           closes, then lifts.
    Then   IK has a solution AND planning succeeds; after closing the gripper
           aperture ∈ (0, max) (not empty, not fully closed on nothing); the
           lift Δz meets target AND the aperture stays > 0 throughout the lift
           (free-signal proxy for "picked up", §5 M3 gate).
    """
    raise NotImplementedError


@pytest.mark.e2e
@pytest.mark.m3
@pytest.mark.skip(reason="M3 pending")
def test_m3_place():
    """PLACE — set the object down at the destination and release cleanly.

    Given  a successful pick (M3), object held, base at the destination.
    When   PLACE plans destination-standoff → Cartesian lower to release height
           → open gripper → lift arm to clear → return STOW (§3 PLACE).
    Then   release height is reached; after opening, aperture → open; after the
           clearing lift the object does NOT follow the gripper — sim GT: object
           odom-to-gripper distance > release threshold; free-signal proxy:
           no object in the depth in front of the gripper (§3 PLACE gate). On
           "not released" it lift-retries ≤ N = 2 then RECOVER (§3a).
    """
    raise NotImplementedError


@pytest.mark.e2e
@pytest.mark.m3
@pytest.mark.skip(reason="M3 pending")
def test_m3_retry_budget_exhaustion():
    """Whole-chain pick retry budget: ≤ 3 attempts, then report to zeno.

    Given  a target X that cannot be picked (e.g. repeatedly ungraspable pose).
    When   the full SEARCH→VERIFY pick loop fails and retries.
    Then   the whole-chain pick attempt count is capped at N = 3 (§3a budget);
           on exhaustion the task terminates and reports "无法完成，把 X 拿来"
           back to zeno — no unbounded whole-chain retry.
    """
    raise NotImplementedError
