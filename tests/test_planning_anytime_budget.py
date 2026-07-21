import os
from pathlib import Path
import time

import numpy as np
import pytest

from z_manip.kinematics.chain import KinematicChain
from z_manip.kinematics.robust_ik import IKConfig, IKSolution, RobustIKSolver
from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.planner import JointTrajectory
from z_manip.planning.grasp_pipeline import GraspPlanConfig, GraspPlanGenerator
from z_manip.planning.standoff import (
    ReachabilityStandoffConfig,
    ReachabilityStandoffOptimizer,
)
from z_manip.planning_control import (
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)


class SimClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, duration_s):
        self.now += float(duration_s)


class BudgetIK:
    def __init__(self, clock, costs=(), default_cost_s=0.001):
        self.clock = clock
        self.costs = list(costs)
        self.default_cost_s = default_cost_s
        self.calls = 0

    def solve(self, target, current=None, *, control=None):
        checkpoint(control, "test IK")
        cost = self.costs.pop(0) if self.costs else self.default_cost_s
        self.clock.advance(cost)
        checkpoint(control, "test IK")
        self.calls += 1
        joints = np.full(2, float(target[0, 3]))
        return IKSolution(joints, 0.0, 0.0, 0.2, 1, 0, 0.25)


class BudgetPlanner:
    joint_names = ("j1", "j2")

    def __init__(self, clock, cost_s=0.002):
        self.clock = clock
        self.cost_s = cost_s
        self.calls = 0

    def plan_joint(self, start, goal, *, timeout_s, control=None):
        checkpoint(control, "test RRT")
        self.clock.advance(self.cost_s)
        checkpoint(control, "test RRT")
        self.calls += 1
        return JointTrajectory(self.joint_names, np.vstack((start, goal)))

    def segment_valid(self, _first, _second, *, control=None):
        checkpoint(control, "test segment")
        return True


def _pose(position, approach):
    approach = np.asarray(approach, dtype=float)
    approach /= np.linalg.norm(approach)
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference, approach))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    closing = np.cross(reference, approach)
    closing /= np.linalg.norm(closing)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack(
        (closing, np.cross(approach, closing), approach),
    )
    transform[:3, 3] = position
    return transform


def _many_candidates(count=32):
    poses = []
    for index in range(count):
        poses.append(_pose(
            (0.42 + 0.002 * index, -0.08 + 0.005 * (index % 5), 0.24),
            (0.7, 0.2 * ((index % 3) - 1), 0.7),
        ))
    return GraspCandidates(
        grasps=np.stack(poses),
        scores=np.linspace(1.0, 0.4, count),
        centroid=np.mean([pose[:3, 3] for pose in poses], axis=0),
        frame="piper_base_link",
        num_raw=count,
        widths=np.full(count, 0.04),
    )


def test_obstacle_free_32_candidate_plan_returns_first_feasible_inside_budget():
    clock = SimClock()
    ik = BudgetIK(clock)
    planner = BudgetPlanner(clock)
    generator = GraspPlanGenerator(
        ik,
        planner,
        GraspPlanConfig(
            approach_steps=7,
            lift_steps=6,
            symmetry_samples=2,
            max_candidates=32,
            max_hypotheses=64,
            max_feasible_plans=1,
            search_timeout_s=1.0,
            hypothesis_timeout_s=0.5,
        ),
    )
    control = PlanningControl(deadline_s=1.0, monotonic_fn=clock)

    result = generator.plan(
        _many_candidates(),
        current_joints=np.zeros(2),
        control=control,
    )

    assert result.candidate_index == 0
    assert ik.calls == 1 + (7 - 1) + (6 - 1)
    assert planner.calls == 1
    assert clock.now < 0.10


def test_shelf_like_slow_hypothesis_isolated_and_next_oblique_grasp_succeeds():
    clock = SimClock()
    ik = BudgetIK(clock, costs=(0.21,), default_cost_s=0.004)
    planner = BudgetPlanner(clock)
    generator = GraspPlanGenerator(
        ik,
        planner,
        GraspPlanConfig(
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_feasible_plans=1,
            search_timeout_s=1.0,
            hypothesis_timeout_s=0.20,
        ),
    )
    candidates = _many_candidates(2)

    result = generator.plan(
        candidates,
        current_joints=np.zeros(2),
        control=PlanningControl(deadline_s=1.0, monotonic_fn=clock),
    )

    assert result.candidate_index == 1
    assert any(failure.stage == "budget" for failure in result.failures)
    assert clock.now < 0.30


def test_candidate_budget_never_swallows_caller_deadline():
    clock = SimClock()
    generator = GraspPlanGenerator(
        BudgetIK(clock, costs=(0.21,)),
        BudgetPlanner(clock),
        GraspPlanConfig(
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            search_timeout_s=1.0,
            hypothesis_timeout_s=0.5,
        ),
    )

    with pytest.raises(PlanningDeadlineExceeded):
        generator.plan(
            _many_candidates(2),
            current_joints=np.zeros(2),
            control=PlanningControl(deadline_s=0.20, monotonic_fn=clock),
        )


def test_refinement_timeout_returns_existing_feasible_plan():
    clock = SimClock()
    ik = BudgetIK(
        clock,
        costs=(0.005, 0.005, 0.005, 0.005, 0.005, 0.15),
        default_cost_s=0.005,
    )
    generator = GraspPlanGenerator(
        ik,
        BudgetPlanner(clock, cost_s=0.005),
        GraspPlanConfig(
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_feasible_plans=3,
            search_timeout_s=1.0,
            hypothesis_timeout_s=0.5,
            solution_refinement_timeout_s=0.10,
        ),
    )

    result = generator.plan(
        _many_candidates(3),
        current_joints=np.zeros(2),
        control=PlanningControl(deadline_s=1.0, monotonic_fn=clock),
    )

    assert result.candidate_index == 0
    assert clock.now == pytest.approx(0.18)


def test_symmetry_ranking_refinement_timeout_keeps_feasible_plan():
    clock = SimClock()
    rank_calls = 0

    def pose_ranker(_target, *, control=None):
        nonlocal rank_calls
        rank_calls += 1
        if rank_calls == 2:
            clock.advance(0.11)
        checkpoint(control, "test symmetry ranking")
        return 0.0

    generator = GraspPlanGenerator(
        BudgetIK(clock),
        BudgetPlanner(clock),
        GraspPlanConfig(
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_feasible_plans=2,
            search_timeout_s=1.0,
            hypothesis_timeout_s=0.5,
            solution_refinement_timeout_s=0.10,
        ),
    )

    result = generator.plan(
        _many_candidates(2),
        current_joints=np.zeros(2),
        pose_ranker=pose_ranker,
        control=PlanningControl(deadline_s=1.0, monotonic_fn=clock),
    )

    assert result.candidate_index == 0
    assert rank_calls == 2
    assert any(failure.stage == "budget" for failure in result.failures)


def test_expired_search_when_opening_refinement_keeps_feasible_plan():
    clock = SimClock()

    class LateExpiringTrajectory:
        joint_names = ("j1", "j2")

        def __init__(self, start, goal):
            self._waypoints = np.vstack((start, goal))

        @property
        def waypoints(self):
            clock.now = 1.01
            return self._waypoints

    class LateExpiringPlanner:
        joint_names = ("j1", "j2")

        @staticmethod
        def plan_joint(start, goal, *, timeout_s, control=None):
            checkpoint(control, "late-expiring transit")
            return LateExpiringTrajectory(start, goal)

        @staticmethod
        def segment_valid(_first, _second, *, control=None):
            checkpoint(control, "late-expiring segment")
            return True

    generator = GraspPlanGenerator(
        BudgetIK(clock),
        LateExpiringPlanner(),
        GraspPlanConfig(
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_feasible_plans=2,
            search_timeout_s=1.0,
            hypothesis_timeout_s=1.0,
            solution_refinement_timeout_s=0.1,
        ),
    )

    result = generator.plan(
        _many_candidates(1),
        current_joints=np.zeros(2),
        control=PlanningControl(deadline_s=10.0, monotonic_fn=clock),
    )

    assert result.candidate_index == 0
    assert clock.now == pytest.approx(1.01)


def test_shelf_standoff_timeout_probes_near_depth_before_cartesian_product():
    clock = SimClock()
    evaluated = []
    config = ReachabilityStandoffConfig(
        depth_samples=10,
        max_candidates=8,
        max_hypotheses=16,
        search_timeout_s=1.0,
        hypothesis_timeout_s=0.20,
    )

    def evaluate(candidates, _displacement, desired_depth, control):
        evaluated.append((float(candidates.scores[0]), desired_depth))
        clock.advance(0.21 if len(evaluated) == 1 else 0.02)
        checkpoint(control, "shelf downstream planner")
        return {"score": float(candidates.scores[0])}

    choice = ReachabilityStandoffOptimizer(config).select(
        _many_candidates(8),
        current_camera_depth_m=1.2,
        camera_rotation_base=np.array(
            ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ),
        evaluate=evaluate,
        control=PlanningControl(deadline_s=1.0, monotonic_fn=clock),
    )

    assert [sample[1] for sample in evaluated] == pytest.approx([0.75, 0.32])
    assert choice.desired_camera_depth_m == pytest.approx(0.32)
    assert clock.now < 0.30


def test_standoff_refinement_timeout_returns_existing_feasible_choice():
    clock = SimClock()
    calls = 0
    optimizer = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        depth_samples=3,
        max_feasible_choices=2,
        search_timeout_s=1.0,
        hypothesis_timeout_s=0.5,
        solution_refinement_timeout_s=0.10,
    ))

    def evaluate(_candidates, _displacement, desired_depth, control):
        nonlocal calls
        calls += 1
        clock.advance(0.01 if calls == 1 else 0.11)
        checkpoint(control, "standoff refinement benchmark")
        return {"score": desired_depth}

    choice = optimizer.select(
        _many_candidates(2),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ),
        evaluate=evaluate,
        control=PlanningControl(deadline_s=1.0, monotonic_fn=clock),
    )

    assert calls == 2
    assert choice.desired_camera_depth_m == pytest.approx(0.75)


def test_expired_standoff_search_when_opening_refinement_keeps_choice():
    clock = SimClock()

    class LateScore:
        @property
        def score(self):
            clock.now = 1.01
            return 1.0

    optimizer = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        depth_samples=3,
        max_feasible_choices=2,
        search_timeout_s=1.0,
        hypothesis_timeout_s=1.0,
        solution_refinement_timeout_s=0.1,
    ))
    choice = optimizer.select(
        _many_candidates(1),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ),
        evaluate=lambda *_args: LateScore(),
        control=PlanningControl(deadline_s=10.0, monotonic_fn=clock),
    )

    assert choice.desired_camera_depth_m == pytest.approx(0.75)
    assert clock.now == pytest.approx(1.01)


def test_real_piper_side_grasp_fallback_finishes_well_inside_runtime_budget():
    default_urdf = Path(
        "/home/yusenzlabpc/Z-Robotics-Lab/go2W_Sim/assets/urdf/go2w_sensored.urdf",
    )
    urdf = Path(os.environ.get("Z_MANIP_ROBOT_URDF", default_urdf))
    if not urdf.exists():
        pytest.skip(f"PiPER URDF unavailable: {urdf}")
    chain = KinematicChain.from_urdf(
        urdf,
        "piper_base_link",
        "piper_gripper_base",
    )

    class DirectPlanner:
        joint_names = chain.joint_names

        @staticmethod
        def segment_valid(_first, _second, *, control=None):
            checkpoint(control, "benchmark segment")
            return True

        def plan_joint(self, start, goal, *, timeout_s, control=None):
            checkpoint(control, "benchmark transit")
            return JointTrajectory(self.joint_names, np.vstack((start, goal)))

    difficult = np.array([0.20, 0.80, -1.20, 0.30, -0.25, 0.40])
    side_grasp = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
    tool_from_tip = np.eye(4)
    tool_from_tip[2, 3] = 0.17
    grasps = np.stack([
        chain.forward(difficult) @ tool_from_tip,
        chain.forward(side_grasp) @ tool_from_tip,
    ])
    assert abs(float(grasps[1, 2, 2])) < 0.2
    candidates = GraspCandidates(
        grasps=grasps,
        scores=np.array([0.95, 0.90]),
        centroid=np.mean(grasps[:, :3, 3], axis=0),
        frame="piper_base_link",
        num_raw=2,
        widths=np.array([0.04, 0.04]),
    )
    config = GraspPlanConfig(
        pregrasp_distance_m=0.09,
        approach_steps=7,
        lift_steps=6,
        symmetry_samples=2,
        max_feasible_plans=1,
        search_timeout_s=8.0,
        hypothesis_timeout_s=2.5,
        tool_from_tip=tuple(tuple(row) for row in tool_from_tip),
    )
    generator = GraspPlanGenerator(
        RobustIKSolver(
            chain,
            IKConfig(
                random_seeds=14,
                max_feasible_solutions=2,
                solution_refinement_timeout_s=0.05,
            ),
        ),
        DirectPlanner(),
        config,
    )

    started = time.perf_counter()
    result = generator.plan(candidates, current_joints=side_grasp)
    elapsed = time.perf_counter() - started

    assert result.candidate_index == 1
    assert any(failure.stage == "ik" for failure in result.failures)
    assert elapsed < 3.0
