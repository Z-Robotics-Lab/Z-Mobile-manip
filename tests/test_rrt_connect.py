import numpy as np
import pytest

import z_manip.planning.rrt_connect as rrt_connect
from z_manip.models.planner import PlanningError
from z_manip.planning.rrt_connect import (
    JointSpaceRRTConnect,
    RRTConnectConfig,
    RRTTimeout,
)
from z_manip.planning_control import (
    PlanningAborted,
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
)


def _planner(state_valid, *, max_iterations=2500):
    return JointSpaceRRTConnect(
        joint_names=("x", "y"),
        lower_limits=np.array([-1.0, -1.0]),
        upper_limits=np.array([1.0, 1.0]),
        state_valid=state_valid,
        config=RRTConnectConfig(
            step_size=0.14,
            collision_resolution=0.025,
            max_iterations=max_iterations,
            shortcut_attempts=100,
            seed=17,
        ),
    )


def test_rrt_connect_routes_around_obstacle_and_is_deterministic():
    def valid(joints):
        return np.linalg.norm(joints) > 0.32

    planner = _planner(valid)
    start = np.array([-0.82, 0.0])
    goal = np.array([0.82, 0.0])

    first = planner.plan_joint(start, goal)
    second = planner.plan_joint(start, goal)

    assert np.allclose(first.waypoints, second.waypoints)
    assert np.allclose(first.waypoints[0], start)
    assert np.allclose(first.waypoints[-1], goal)
    assert len(first.waypoints) > 2
    assert all(valid(waypoint) for waypoint in first.waypoints)
    assert np.max(np.linalg.norm(np.diff(first.waypoints, axis=0), axis=1)) <= 0.0251


def test_rrt_connect_rejects_an_impassable_collision_wall():
    planner = _planner(lambda joints: abs(float(joints[0])) > 0.08, max_iterations=400)
    with pytest.raises(PlanningError, match="collision-free joint path"):
        planner.plan_joint(np.array([-0.8, 0.0]), np.array([0.8, 0.0]), timeout_s=1.0)


def test_rrt_connect_fails_closed_for_invalid_endpoint():
    planner = _planner(lambda joints: float(joints[0]) < 0.7)
    with pytest.raises(PlanningError, match="goal joint state is in collision"):
        planner.plan_joint(np.array([-0.8, 0.0]), np.array([0.8, 0.0]))


def test_rrt_cancel_interrupts_inner_segment_sampling():
    planner = _planner(lambda _joints: True)
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 12

    with pytest.raises(PlanningCancelled, match="segment collision checking was cancelled"):
        planner.plan_joint(
            np.array([-0.8, 0.0]),
            np.array([0.8, 0.0]),
            control=PlanningControl(cancel_check=cancelled),
        )

    assert checks == 12


def test_rrt_honors_absolute_monotonic_deadline():
    ticks = 0

    def clock():
        nonlocal ticks
        ticks += 1
        return float(ticks)

    planner = _planner(lambda _joints: True)
    control = PlanningControl(deadline_s=6.0, monotonic_fn=clock)

    with pytest.raises(PlanningDeadlineExceeded, match="monotonic deadline"):
        planner.plan_joint(
            np.array([-0.8, 0.0]),
            np.array([0.8, 0.0]),
            timeout_s=100.0,
            control=control,
        )

    assert ticks == 6


def test_rrt_local_timeout_remains_recoverable_candidate_failure(monkeypatch):
    ticks = 0

    def clock():
        nonlocal ticks
        ticks += 1
        return ticks * 0.1

    monkeypatch.setattr(rrt_connect.time, "monotonic", clock)
    planner = _planner(lambda _joints: True)

    with pytest.raises(RRTTimeout, match="recoverable local RRT timeout") as captured:
        planner.plan_joint(
            np.array([-0.8, 0.0]),
            np.array([0.8, 0.0]),
            timeout_s=0.5,
        )

    assert isinstance(captured.value, PlanningError)
    assert not isinstance(captured.value, PlanningAborted)
