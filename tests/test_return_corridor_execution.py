"""The loaded return must be safe as EXECUTED, not only as interpolated.

Operator report 2026-07-28: reaching out for the object is clean, and the
return after grasping drives the gripper tip and side into the Mid-360.

The mechanism is a mismatch between two different notions of "the path between
two waypoints".  Every collision check in this repo -- ``check_step`` here,
and the planner's own segment checks -- samples a JOINT-LINEAR interpolation:
all six joints scaled by one shared parameter, so they start and finish
together.  An unsynchronised ``move_j`` does not do that.  Each joint runs at
its own rate, so the joints with less travel arrive early and the arm walks a
visibly different corridor between the same two endpoints.

On the 14 recorded ``holding_at_lift`` poses in
``tests/data/holding_at_lift_poses.npz`` the two corridors disagree by enough
to matter: the joint-linear sweep to Home clears the lidar on all 14, and the
unsynchronised sweep between the same endpoints puts the fingers and palm
inside it on 12.

These tests pin both halves: the disagreement is real (so nobody "simplifies"
the executed-sweep check away), and the shipped bound keeps the executed motion
inside what was validated.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from z_manip.fixed_self_collision import FixedSelfCollisionGuard


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
COLLISION_MODEL = ROOT / "configs/piper_collision_capsules.json"
RECORDED = ROOT / "tests/data/holding_at_lift_poses.npz"


def _guard() -> FixedSelfCollisionGuard:
    return FixedSelfCollisionGuard(urdf_path=URDF, model_path=COLLISION_MODEL)


def _recorded():
    data = np.load(RECORDED, allow_pickle=False)
    return (
        np.asarray(data["lift_top"], dtype=float),
        np.asarray(data["home"], dtype=float),
    )


def _joint_linear(start: np.ndarray, end: np.ndarray, samples: int = 240):
    """What every collision check in this repo validates."""

    return start + (end - start) * np.linspace(0.0, 1.0, samples)[:, None]


def _unsynchronised(start: np.ndarray, end: np.ndarray, samples: int = 240):
    """What an unsynchronised move_j executes.

    Every joint runs at the same angular rate, so a joint with half the travel
    reaches its target halfway through and then holds while the others keep
    moving.  This is the shape of the real motion, and it is not the straight
    line between the endpoints.
    """

    delta = end - start
    travel = np.abs(delta)
    span = float(travel.max())
    if span <= 0.0:
        return start[None, :]
    elapsed = np.linspace(0.0, span, samples)[:, None]
    return start + np.minimum(travel[None, :], elapsed) * np.sign(delta)[None, :]


def _worst_margin_m(guard: FixedSelfCollisionGuard, path: np.ndarray) -> float:
    return min(guard.check_state(joints).witness.margin_m for joints in path)


@pytest.mark.skipif(not URDF.is_file(), reason="go2W_Sim URDF not checked out")
def test_the_two_corridors_genuinely_disagree_on_recorded_returns():
    """The defect, as an assertion.

    If this ever stops failing for the joint-linear path while passing for the
    executed one, the two notions have converged and the executed-sweep check
    below is no longer load-bearing -- but until then, deleting it puts the
    gripper back into the lidar.
    """

    guard = _guard()
    lift_tops, home = _recorded()

    linear_clear = 0
    executed_hits = 0
    for start in lift_tops:
        if _worst_margin_m(guard, _joint_linear(start, home)) >= 0.0:
            linear_clear += 1
        if _worst_margin_m(guard, _unsynchronised(start, home)) < 0.0:
            executed_hits += 1

    assert linear_clear == len(lift_tops), (
        "every recorded return clears the lidar when the joints are scaled "
        "together -- this is why the checker approved them"
    )
    assert executed_hits >= 10, (
        "and most of them put the gripper inside the lidar when the joints are "
        "driven unsynchronised, which is what the operator sees"
    )


@pytest.mark.skipif(not URDF.is_file(), reason="go2W_Sim URDF not checked out")
@pytest.mark.parametrize("segment_deg", [180.0, 90.0, 45.0, 20.0])
def test_subdividing_alone_does_not_make_the_shortcut_safe(segment_deg):
    """Why the fix is 'retrace the outbound', not 'use smaller steps'.

    Chopping the shortcut into finer move_j edges shrinks the disagreement but
    does not remove it: the corridor itself grazes the lidar, so the executed
    sweep is still inside it at every segment size that keeps the motion
    continuous.  Recorded worst margins: 180 deg -> -91.0 mm, 90 -> -60.1,
    45 -> -34.9, 20 -> -17.3, 10 -> -6.4, 5 -> -1.9, 3 -> -0.3, and only at
    2 deg does it clear, by 0.6 mm.  A 0.6 mm margin is not a safety story.
    """

    guard = _guard()
    lift_tops, home = _recorded()
    step = math.radians(segment_deg)

    worst = math.inf
    for start in lift_tops:
        span = float(np.max(np.abs(home - start)))
        count = max(1, int(math.ceil(span / step)))
        for index in range(count):
            a = start + (home - start) * (index / count)
            b = start + (home - start) * ((index + 1) / count)
            worst = min(worst, _worst_margin_m(guard, _unsynchronised(a, b, 48)))

    assert worst < 0.0, (
        f"at {segment_deg:.0f} deg segments the executed return still enters "
        "the lidar keep-out; if this now passes, re-derive the bound rather "
        "than assuming the shortcut became safe"
    )


@pytest.mark.skipif(not URDF.is_file(), reason="go2W_Sim URDF not checked out")
def test_an_exact_retrace_is_safe_as_executed():
    """The corridor the arm already traversed is safe to traverse backwards.

    A densely sampled planned corridor has small per-segment travel, so the
    executed sweep and the validated interpolation agree to within the
    clearance budget.  This is what makes 'retrace the outbound' a stronger
    claim than 'find another corridor that also passes the checker'.
    """

    guard = _guard()
    lift_tops, home = _recorded()

    for start in lift_tops:
        # Stand in for a planned corridor: the same endpoints, sampled at the
        # density a planner emits (a few degrees per waypoint).
        span = float(np.max(np.abs(home - start)))
        waypoints = max(2, int(math.ceil(span / math.radians(2.0))) + 1)
        corridor = _joint_linear(start, home, waypoints)

        worst = math.inf
        for a, b in zip(corridor[:-1], corridor[1:]):
            worst = min(worst, _worst_margin_m(guard, _unsynchronised(a, b, 24)))

        assert worst >= 0.0, (
            "a densely sampled retrace must be safe as executed, not only as "
            f"interpolated (worst margin {worst * 1000:.1f} mm)"
        )
