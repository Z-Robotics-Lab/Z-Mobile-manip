"""Every leg driven with the object in hand is checked in that configuration.

The outbound plan is validated with an EMPTY gripper -- ``transit`` and
``approach`` treat the object as an obstacle to steer around.  The workflow
then replays those arrays backwards while HOLDING the object.  Reversing a
path is not revalidating it, and the payload adds swept volume out past the
fingers, so the homeward run was being commanded on evidence gathered for a
different robot configuration.  ``plan_holding_return`` is what closes that.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.kinematics.chain import KinematicChain
from z_manip.planning.online_planner import (
    CARRY_MAX_DIP_BELOW_REPLAY_M,
    HoldingReturnPlan,
    OnlinePlanner,
)


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim" / "assets" / "urdf" / "go2w_sensored.urdf"


# The last joint doubles as the tool height under ``_Chain`` below, so each
# pose's height is written down explicitly: Home high, pregrasp above the
# object, grasp on the object, lift top 70 mm clear of it.
def _pose(first, height):
    return np.asarray((first, 0.10, -0.10, 0.0, 0.0, height))


Q_HOME = _pose(0.00, 0.300)
Q_PRE = _pose(0.03, 0.180)
Q_GRASP = _pose(0.04, 0.110)
Q_LIFT = _pose(0.04, 0.180 + 0.070)
# A lateral grasp whose pregrasp sits BELOW the object it picked up.
Q_PRE_BELOW_GRASP = _pose(0.03, 0.100)

TRANSIT = np.vstack((Q_HOME, Q_PRE))
APPROACH = np.vstack((Q_PRE, Q_GRASP))
LIFT = np.vstack((Q_GRASP, Q_LIFT))


class _Chain:
    """A stand-in whose tool height is a chosen linear function of the joints."""

    dof = 6

    def __init__(self, height_weights):
        self.weights = np.asarray(height_weights, dtype=float)

    def forward(self, joints):
        pose = np.eye(4)
        pose[2, 3] = float(np.asarray(joints, dtype=float) @ self.weights)
        return pose


def _planner(height_weights, verdicts=None):
    planner = OnlinePlanner.__new__(OnlinePlanner)
    planner.chain = _Chain(height_weights)
    planner.config = SimpleNamespace(
        grasp_plan=SimpleNamespace(tool_from_tip=np.eye(4)),
    )
    calls: list[dict] = []

    def validate_path(path, **kwargs):
        calls.append({"path": np.asarray(path, dtype=float), **kwargs})
        return bool((verdicts or {}).get(kwargs["segment_name"], True))

    planner.validate_path = validate_path
    return planner, calls


def _plan(planner, **overrides):
    kwargs = {
        "transit_raw": TRANSIT,
        "approach_raw": APPROACH,
        "lift_raw": LIFT,
        "scene_points": np.zeros((3, 3)),
        "target_points": np.zeros((3, 3)),
        "stamp_s": 0.0,
        "required_width_m": 0.03,
    }
    kwargs.update(overrides)
    return planner.plan_holding_return(**kwargs)


# The tool height is read straight off the sixth joint.
DESCENDING = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def test_every_while_holding_leg_is_validated_with_the_payload_attached():
    planner, calls = _planner(DESCENDING)

    plan = _plan(planner)

    named = {call["segment_name"]: call for call in calls}
    assert set(named) == {"lift", "lifted_retreat", "holding_transit"}
    # Both the lift column and the reverse approach are checked as departures
    # from the support with the payload attached.
    lifts = [call for call in calls if call["segment_name"] == "lift"]
    assert len(lifts) == 2
    np.testing.assert_allclose(lifts[0]["path"], LIFT)
    np.testing.assert_allclose(lifts[1]["path"], APPROACH[::-1])
    # The run Home -- the leg on which a carried object reaches the platform.
    np.testing.assert_allclose(named["holding_transit"]["path"], TRANSIT)
    # Every one of them attaches the payload at the grasp pose and passes the
    # planned grasp width so the closed-on-object model is selected.
    for call in calls:
        np.testing.assert_allclose(call["attachment_joints"], Q_GRASP)
        assert call["required_width_m"] == 0.03
    assert plan.segments == {
        "lift": True,
        "reverse_approach": True,
        "carry": True,
        "return_transit": True,
    }


def test_the_carry_is_one_edge_from_the_lift_top_to_the_pregrasp():
    planner, calls = _planner(DESCENDING)

    plan = _plan(planner)

    assert plan.carry_raw.shape == (2, 6)
    np.testing.assert_allclose(plan.carry_raw[0], LIFT[-1])
    np.testing.assert_allclose(plan.carry_raw[-1], TRANSIT[-1])
    retreat = next(c for c in calls if c["segment_name"] == "lifted_retreat")
    np.testing.assert_allclose(retreat["path"], plan.carry_raw)


def test_the_carry_bottoms_out_above_the_pose_the_object_was_picked_from():
    planner, _calls = _planner(DESCENDING)

    plan = _plan(planner)

    heights = plan.carry_tip_height
    assert heights["non_increasing"] is True
    assert heights["lift_top_m"] > heights["pregrasp_m"] > heights["grasp_m"]
    assert heights["minimum_m"] == pytest.approx(heights["pregrasp_m"])
    assert heights["replay_floor_m"] == pytest.approx(heights["grasp_m"])
    assert heights["minimum_above_replay_floor_m"] > CARRY_MAX_DIP_BELOW_REPLAY_M
    assert plan.carry_rejection_reason == ""
    assert plan.carry_valid is True


def test_a_pregrasp_below_the_grasp_does_not_cost_the_operator_the_carry():
    """A lateral pick whose pregrasp sits BELOW the object still carries.

    The fallback for a refused carry is the reverse-lift + reverse-approach
    replay, which sets the object down ON the pick spot and then travels out to
    that same below-grasp pregrasp.  Refusing the carry here would buy nothing
    and cost the operator exactly the put-down this change deletes, so the
    contract is measured against the floor of the replay, not against the grasp
    height on its own.

    Measured on the recorded fleet: 3 of 172 plans have a pregrasp 2.3-27.0 mm
    below their grasp; an absolute "stay above the grasp" rule refused all 3.
    """

    planner, _calls = _planner(DESCENDING)

    plan = _plan(
        planner,
        transit_raw=np.vstack((Q_HOME, Q_PRE_BELOW_GRASP)),
        approach_raw=np.vstack((Q_PRE_BELOW_GRASP, Q_GRASP)),
    )

    heights = plan.carry_tip_height
    assert heights["pregrasp_m"] < heights["grasp_m"]
    # The replay would reach that same pregrasp anyway, so its floor is there.
    assert heights["replay_floor_m"] == pytest.approx(heights["pregrasp_m"])
    assert plan.carry_rejection_reason == ""
    assert plan.carry_valid is True


def test_a_carry_that_dips_below_the_replay_it_replaces_is_refused():
    """A "carry" that dives under both endpoints is a put-down.

    That is precisely the behaviour this replaces, so it is rejected on
    geometry even when the collision checker is happy, and the executor falls
    back to the (separately checked) reverse-lift corridor.
    """

    # The chord from the lift top to the pregrasp is straight in JOINT space,
    # so a tool height that is a quadratic of the joints bows below both of its
    # endpoints along the way.  This one is pinned to zero at both ends (the
    # lift top's first joint is 0.04 and the pregrasp's is 0.03) and dives
    # 200 mm at the midpoint, well under the 110 mm grasp height.
    planner, _calls = _planner(DESCENDING)
    chain = planner.chain
    straight = chain.forward

    def bowed(joints):
        pose = straight(joints)
        first = float(np.asarray(joints, dtype=float)[0])
        pose[2, 3] -= 8000.0 * (first - 0.03) * (0.04 - first)
        return pose

    chain.forward = bowed

    plan = _plan(planner)

    heights = plan.carry_tip_height
    assert heights["minimum_m"] < heights["replay_floor_m"]
    assert plan.carry_valid is False
    assert plan.carry_rejection_reason == "carry_dips_below_the_replay_it_replaces"
    # The corridors that remain are still reported honestly.
    assert plan.legacy_corridor_valid is True
    assert plan.return_transit_valid is True


def test_a_colliding_carry_is_reported_as_a_collision_rejection():
    planner, _calls = _planner(DESCENDING, verdicts={"lifted_retreat": False})

    plan = _plan(planner)

    assert plan.carry_valid is False
    assert plan.carry_rejection_reason == "collision"
    assert plan.legacy_corridor_valid is True


def test_a_rejected_run_home_is_surfaced_rather_than_hidden():
    planner, _calls = _planner(DESCENDING, verdicts={"holding_transit": False})

    plan = _plan(planner)

    assert plan.return_transit_valid is False
    assert plan.document()["segments"]["return_transit"] is False


def test_a_rejected_lift_column_invalidates_the_legacy_replay():
    planner, _calls = _planner(DESCENDING, verdicts={"lift": False})

    plan = _plan(planner)

    assert plan.legacy_corridor_valid is False
    assert plan.segments["lift"] is False
    assert plan.segments["reverse_approach"] is False


def test_the_document_is_the_exact_contract_the_executor_gates_on():
    planner, _calls = _planner(DESCENDING)

    document = _plan(planner).document()

    assert document["schema"] == "z_manip.holding_return_validation.v1"
    assert document["collision_model"] == "closed_on_object_with_platform_fixtures"
    assert document["attached_at"] == "grasp"
    assert document["carry_raw_waypoints"] == 2
    assert set(document["segments"]) == {
        "lift", "reverse_approach", "carry", "return_transit",
    }
    assert isinstance(document["carry_tip_height_m"], dict)


@pytest.mark.parametrize(
    "override",
    [
        {"transit_raw": TRANSIT[:1]},
        {"approach_raw": np.zeros((2, 5))},
        {"lift_raw": np.full((2, 6), np.nan)},
    ],
)
def test_malformed_polylines_are_refused_outright(override):
    planner, _calls = _planner(DESCENDING)

    with pytest.raises(ValueError, match="finite"):
        _plan(planner, **override)


def test_the_plan_is_immutable_evidence():
    planner, _calls = _planner(DESCENDING)

    plan = _plan(planner)

    assert isinstance(plan, HoldingReturnPlan)
    with pytest.raises(Exception):
        plan.segments = {}


# ---------------------------------------------------------------------------
# Replay of recorded plans.
#
# The joint polylines below are the transit_raw/approach_raw/lift_raw arrays
# lifted verbatim out of planned_grasp.npz for six real
# artifacts/go2w_real/interactive_sessions/planning runs, driven through the
# real PiPER chain and the shipped grasp_plan.tool_from_tip.  The three dated
# 20260724-014422 / -063625 / -064414 are the plans on which an absolute
# "the carry must bottom out above the grasp" rule refused the carry and handed
# the operator the put-down back.
# ---------------------------------------------------------------------------

RECORDED_PLANS = ROOT / "tests" / "data" / "holding_return_recorded_plans.npz"
# Measured tip heights (mm) of pregrasp minus grasp on each recorded plan.
RECORDED_PREGRASP_ABOVE_GRASP_MM = {
    "20260723-042644": 50.1,
    "20260724-014422": 2.3,
    "20260724-041614": 13.8,
    "20260724-063625": -27.0,
    "20260724-064414": -22.6,
    "20260727-085157": 31.6,
}


@pytest.fixture(scope="module")
def recorded_planner():
    if not URDF.exists():
        pytest.skip(f"PiPER URDF unavailable: {URDF}")
    if not RECORDED_PLANS.exists():
        pytest.skip("recorded holding-return plans unavailable")
    chain = KinematicChain.from_urdf(URDF, "piper_base_link", "piper_gripper_base")
    tool_from_tip = np.asarray(
        json.loads((ROOT / "configs/go2w_piper.json").read_text(encoding="utf-8"))[
            "grasp_plan"
        ]["tool_from_tip"],
        dtype=float,
    )
    planner = OnlinePlanner.__new__(OnlinePlanner)
    planner.chain = chain
    planner.config = SimpleNamespace(
        grasp_plan=SimpleNamespace(tool_from_tip=tool_from_tip),
    )
    # Collision is checked in its own tests and needs a live scene; this replay
    # is about the behavioural height contract laid over the collision verdict.
    planner.validate_path = lambda *_a, **_k: True
    return planner


@pytest.mark.parametrize("session", sorted(RECORDED_PREGRASP_ABOVE_GRASP_MM))
def test_every_recorded_plan_keeps_its_carry(recorded_planner, session):
    """No recorded plan loses the carry to the height contract.

    Losing it means the arm sets the object back down on its pick spot on the
    way Home, which is exactly what the operator asked to delete.  Three of
    these six (and 3 of all 172 recorded plans) have a pregrasp BELOW the grasp
    pose -- an in-reach lateral pick -- and an absolute contract refused all
    three, sending them to a replay that reaches that identical low pose after
    first putting the object down.
    """

    archive = np.load(RECORDED_PLANS)
    plan = recorded_planner.plan_holding_return(
        transit_raw=archive[f"{session}/transit_raw"],
        approach_raw=archive[f"{session}/approach_raw"],
        lift_raw=archive[f"{session}/lift_raw"],
        scene_points=np.zeros((3, 3)),
        target_points=np.zeros((3, 3)),
        stamp_s=0.0,
        required_width_m=0.04,
    )

    heights = plan.carry_tip_height
    measured_mm = (heights["pregrasp_m"] - heights["grasp_m"]) * 1000.0
    assert measured_mm == pytest.approx(
        RECORDED_PREGRASP_ABOVE_GRASP_MM[session], abs=0.15,
    ), "the recorded geometry moved; re-measure before trusting this contract"
    assert plan.carry_rejection_reason == ""
    assert plan.carry_valid is True
    # And the carry is never lower than the replay it replaces -- which is the
    # whole content of the contract.
    assert heights["minimum_above_replay_floor_m"] >= -CARRY_MAX_DIP_BELOW_REPLAY_M


def test_the_recorded_below_grasp_plans_are_the_ones_an_absolute_rule_refused():
    """Pin the regression these three sessions encode.

    If a future edit reinstates "the carry must bottom out above the grasp",
    these are the plans it silently hands the put-down back to.
    """

    below = {
        session
        for session, value in RECORDED_PREGRASP_ABOVE_GRASP_MM.items()
        if value <= 5.0
    }
    assert below == {"20260724-014422", "20260724-063625", "20260724-064414"}
