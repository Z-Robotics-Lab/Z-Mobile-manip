"""Every leg driven with the object in hand is checked in that configuration.

The outbound plan is validated with an EMPTY gripper -- ``transit`` and
``approach`` treat the object as an obstacle to steer around.  The workflow
then replays those arrays backwards while HOLDING the object.  Reversing a
path is not revalidating it, and the payload adds swept volume out past the
fingers, so the homeward run was being commanded on evidence gathered for a
different robot configuration.  ``plan_holding_return`` is what closes that.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.planning.online_planner import (
    CARRY_MIN_CLEARANCE_ABOVE_GRASP_M,
    HoldingReturnPlan,
    OnlinePlanner,
)


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
    assert heights["minimum_above_grasp_m"] > CARRY_MIN_CLEARANCE_ABOVE_GRASP_M
    assert plan.carry_rejection_reason == ""
    assert plan.carry_valid is True


def test_a_carry_that_would_lower_the_object_back_onto_its_support_is_refused():
    """A "carry" that descends to the grasp height is a put-down.

    That is precisely the behaviour this replaces, so it is rejected on
    geometry even when the collision checker is happy, and the executor falls
    back to the (separately checked) reverse-lift corridor.
    """

    planner, _calls = _planner(DESCENDING)

    plan = _plan(
        planner,
        transit_raw=np.vstack((Q_HOME, Q_PRE_BELOW_GRASP)),
        approach_raw=np.vstack((Q_PRE_BELOW_GRASP, Q_GRASP)),
    )

    assert plan.carry_valid is False
    assert plan.carry_rejection_reason == "carry_descends_to_the_grasp_height"
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
