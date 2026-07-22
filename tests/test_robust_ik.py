import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.kinematics.chain import (
    KinematicChain,
    fixed_transform_from_urdf,
    rotation_log,
)
from z_manip.kinematics.robust_ik import IKConfig, IKFailure, RobustIKSolver
from z_manip.planning_control import (
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
)


DEFAULT_URDF = Path(
    "/home/yusenzlabpc/Z-Robotics-Lab/go2W_Sim/assets/urdf/go2w_sensored.urdf"
)


@pytest.fixture(scope="module")
def piper_chain():
    path = Path(os.environ.get("Z_MANIP_ROBOT_URDF", DEFAULT_URDF))
    if not path.exists():
        pytest.skip(f"PiPER URDF unavailable: {path}")
    return KinematicChain.from_urdf(path, "piper_base_link", "piper_gripper_base")


def test_analytic_jacobian_matches_finite_difference(piper_chain):
    q = np.array([0.35, 0.75, -1.15, 0.25, 0.30, -0.45])
    transform = piper_chain.forward(q)
    jacobian = piper_chain.jacobian(q)
    eps = 1e-7
    numeric = np.zeros_like(jacobian)
    for index in range(piper_chain.dof):
        shifted = q.copy()
        shifted[index] += eps
        moved = piper_chain.forward(shifted)
        numeric[:3, index] = (moved[:3, 3] - transform[:3, 3]) / eps
        numeric[3:, index] = rotation_log(
            moved[:3, :3] @ transform[:3, :3].T
        ) / eps
    assert np.allclose(jacobian, numeric, atol=2e-5)


def test_link_frames_end_at_same_transform_as_forward_kinematics(piper_chain):
    q = 0.5 * (piper_chain.lower_limits + piper_chain.upper_limits)
    frames = piper_chain.link_transforms(q)

    assert frames[piper_chain.base_link] == pytest.approx(np.eye(4))
    assert frames[piper_chain.tip_link] == pytest.approx(piper_chain.forward(q))
    assert all(joint.child in frames for joint in piper_chain.joints)


def test_fixed_arm_mount_is_read_from_deployed_urdf():
    if not DEFAULT_URDF.exists():
        pytest.skip(f"PiPER URDF unavailable: {DEFAULT_URDF}")
    transform = fixed_transform_from_urdf(
        DEFAULT_URDF,
        "base",
        "piper_base_link",
    )
    # This is the measured mount written by the base-to-arm calibration tool,
    # not the old nominal CAD placement (0.06, 0, 0.067).
    np.testing.assert_allclose(
        transform[:3, 3],
        (0.128526599, 0.0, 0.081794287),
        atol=1e-9,
    )
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-9)


@pytest.mark.parametrize(
    "q_goal",
    [
        [0.0, 0.43, -0.34, 0.0, 0.0, 0.0],
        [0.45, 0.90, -1.30, 0.55, -0.35, 0.70],
        [-0.55, 1.15, -1.75, -0.45, 0.50, -0.80],
    ],
)
def test_multi_seed_ik_recovers_reachable_piper_poses(piper_chain, q_goal):
    target = piper_chain.forward(np.asarray(q_goal))
    current = np.array([1.7, 2.7, -0.2, 1.2, -0.9, 1.5])
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(position_tolerance_m=0.004, orientation_tolerance_rad=0.035,
                 max_iterations=280, random_seeds=18),
    )

    solution = solver.solve(target, current=current)

    assert solution.position_error_m < 0.004
    assert solution.orientation_error_rad < 0.035
    assert np.all(solution.joints >= piper_chain.lower_limits - 1e-9)
    assert np.all(solution.joints <= piper_chain.upper_limits + 1e-9)
    assert solution.manipulability >= 0.0


def test_ik_fails_cleanly_for_unreachable_pose(piper_chain):
    target = np.eye(4)
    target[:3, 3] = (3.0, 0.0, 2.0)
    solver = RobustIKSolver(piper_chain, IKConfig(max_iterations=80, random_seeds=6))
    with pytest.raises(IKFailure, match="no IK solution"):
        solver.solve(target)


def test_seed_pose_ranker_prefers_an_exact_urdf_seed_pose(piper_chain):
    current = np.array((0.35, 0.75, -1.15, 0.25, 0.30, -0.45))
    solver = RobustIKSolver(piper_chain, IKConfig(random_seeds=3))
    rank = solver.make_seed_pose_ranker(current)
    exact = piper_chain.forward(current)
    far = exact.copy()
    far[:3, 3] += (1.0, -0.5, 0.8)

    assert rank(exact) == pytest.approx(0.0, abs=1e-9)
    assert rank(far) > 5.0


def test_ik_cancel_interrupts_scipy_residual_evaluation(piper_chain):
    target = piper_chain.forward(
        np.array([0.45, 0.90, -1.30, 0.55, -0.35, 0.70]),
    )
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 34

    solver = RobustIKSolver(
        piper_chain,
        IKConfig(max_iterations=280, random_seeds=12),
    )
    control = PlanningControl(cancel_check=cancelled)

    with pytest.raises(PlanningCancelled, match="numerical solve was cancelled"):
        solver.solve(target, current=np.zeros(piper_chain.dof), control=control)

    assert checks == 34


def test_ik_can_score_configured_feasible_seeds_within_solve_budget(
    piper_chain,
    monkeypatch,
):
    q_goal = np.array([0.45, 0.90, -1.30, 0.55, -0.35, 0.70])
    target = piper_chain.forward(q_goal)
    calls = 0

    def solved(_residual, _seed, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(x=q_goal.copy(), nfev=1)

    monkeypatch.setattr('z_manip.kinematics.robust_ik.least_squares', solved)
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(
            random_seeds=14,
            max_feasible_solutions=64,
            solution_refinement_timeout_s=1.0,
        ),
    )

    solution = solver.solve(
        target,
        current=np.zeros(piper_chain.dof),
        control=PlanningControl(deadline_s=10.0, monotonic_fn=lambda: 0.0),
    )

    assert calls == len(solver._seeds(np.zeros(piper_chain.dof)))
    assert calls > 2
    assert solution.position_error_m < 1e-9


def test_ik_default_returns_after_bounded_number_of_feasible_seeds(
    piper_chain,
    monkeypatch,
):
    q_goal = np.array([0.45, 0.90, -1.30, 0.55, -0.35, 0.70])
    target = piper_chain.forward(q_goal)
    calls = 0

    def solved(_residual, _seed, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(x=q_goal.copy(), nfev=1)

    monkeypatch.setattr('z_manip.kinematics.robust_ik.least_squares', solved)
    solver = RobustIKSolver(piper_chain, IKConfig(random_seeds=14))

    solution = solver.solve(target, current=np.zeros(piper_chain.dof))

    assert calls == solver.config.max_feasible_solutions == 2
    assert solution.position_error_m < 1e-9


def test_expired_solve_when_opening_refinement_keeps_feasible_seed(
    piper_chain,
    monkeypatch,
):
    q_goal = np.array([0.45, 0.90, -1.30, 0.55, -0.35, 0.70])
    target = piper_chain.forward(q_goal)
    now = 0.0
    calls = 0
    original_jacobian = piper_chain.jacobian

    def clock():
        return now

    def solved(_residual, _seed, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(x=q_goal.copy(), nfev=1)

    def expiring_jacobian(joints):
        nonlocal now
        result = original_jacobian(joints)
        now = 0.61
        return result

    monkeypatch.setattr('z_manip.kinematics.robust_ik.least_squares', solved)
    monkeypatch.setattr(piper_chain, 'jacobian', expiring_jacobian)
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(
            solve_timeout_s=0.60,
            max_feasible_solutions=2,
            solution_refinement_timeout_s=0.05,
        ),
    )

    solution = solver.solve(
        target,
        current=np.zeros(piper_chain.dof),
        control=PlanningControl(deadline_s=10.0, monotonic_fn=clock),
    )

    assert calls == 1
    assert solution.position_error_m < 1e-9
    assert now == pytest.approx(0.61)


def test_local_ik_seed_and_solve_budgets_reject_only_current_pose(
    piper_chain,
    monkeypatch,
):
    target = np.eye(4)
    target[:3, 3] = (3.0, 0.0, 2.0)
    now = 0.0
    calls = 0

    def clock():
        return now

    def stalls(residual, seed, **_kwargs):
        nonlocal calls, now
        calls += 1
        while True:
            now += 0.06
            residual(seed)

    monkeypatch.setattr('z_manip.kinematics.robust_ik.least_squares', stalls)
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(
            random_seeds=14,
            solve_timeout_s=0.35,
            seed_timeout_s=0.10,
        ),
    )
    control = PlanningControl(deadline_s=5.0, monotonic_fn=clock)

    with pytest.raises(IKFailure, match='solve_budget_exhausted=true') as captured:
        solver.solve(target, current=np.zeros(piper_chain.dof), control=control)

    assert calls <= 4
    assert 'seed_timeouts=' in str(captured.value)


def test_aggregate_ik_deadline_still_propagates(piper_chain, monkeypatch):
    target = np.eye(4)
    target[:3, 3] = (3.0, 0.0, 2.0)
    now = 0.0

    def clock():
        return now

    def stalls(residual, seed, **_kwargs):
        nonlocal now
        while True:
            now += 0.06
            residual(seed)

    monkeypatch.setattr('z_manip.kinematics.robust_ik.least_squares', stalls)
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(solve_timeout_s=0.60, seed_timeout_s=0.10),
    )

    with pytest.raises(PlanningDeadlineExceeded, match='deadline by') as captured:
        solver.solve(
            target,
            current=np.zeros(piper_chain.dof),
            control=PlanningControl(deadline_s=0.15, monotonic_fn=clock),
        )

    assert '(0.150000 s)' not in str(captured.value)


def test_solve_deadline_between_seed_checkpoints_stays_local(
    piper_chain,
    monkeypatch,
):
    target = np.eye(4)
    target[:3, 3] = (3.0, 0.0, 2.0)
    timestamps = iter((0.0, 0.0, 0.0, 0.59, 0.61, 0.61))
    calls = 0

    def clock():
        return next(timestamps, 0.61)

    def unexpected_solve(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError('expired local solve budget reached SciPy')

    monkeypatch.setattr(
        'z_manip.kinematics.robust_ik.least_squares',
        unexpected_solve,
    )
    monkeypatch.setattr(
        RobustIKSolver,
        '_seeds',
        lambda _self, current, _control: [np.asarray(current, dtype=float)],
    )
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(solve_timeout_s=0.60, seed_timeout_s=0.10),
    )

    with pytest.raises(IKFailure, match='solve_budget_exhausted=true'):
        solver.solve(
            target,
            current=np.zeros(piper_chain.dof),
            control=PlanningControl(deadline_s=5.0, monotonic_fn=clock),
        )

    assert calls == 0


def test_continuation_exact_fk_target_holds_current_branch_without_scipy(
    piper_chain,
    monkeypatch,
):
    current = np.array([0.35, 0.75, -1.15, 0.25, 0.30, -0.45])

    def unexpected_solve(*_args, **_kwargs):
        raise AssertionError("an exact continuation target must not invoke SciPy")

    monkeypatch.setattr(
        "z_manip.kinematics.robust_ik.least_squares",
        unexpected_solve,
    )
    solution = RobustIKSolver(piper_chain).solve_continuation(
        piper_chain.forward(current),
        current,
        max_joint_step_rad=0.08,
    )

    np.testing.assert_array_equal(solution.joints, current)
    assert solution.position_error_m < 1e-12
    assert solution.orientation_error_rad < 1e-12
    assert solution.iterations == 0


def test_continuation_tracks_small_fk_steps_without_changing_branch(piper_chain):
    initial = np.array([0.35, 0.75, -1.15, 0.25, 0.30, -0.45])
    total_delta = np.array([0.08, -0.06, 0.07, 0.05, -0.04, 0.06])
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(
            max_iterations=180,
            random_seeds=0,
            continuation_timeout_s=0.30,
            continuation_seed_timeout_s=0.15,
        ),
    )
    current = initial.copy()

    for alpha in np.linspace(1.0 / 6.0, 1.0, 6):
        expected = initial + alpha * total_delta
        solution = solver.solve_continuation(
            piper_chain.forward(expected),
            current,
            max_joint_step_rad=0.08,
        )

        assert np.max(np.abs(solution.joints - current)) <= 0.08
        assert solution.position_error_m < solver.config.position_tolerance_m
        assert solution.orientation_error_rad < solver.config.orientation_tolerance_rad
        np.testing.assert_allclose(solution.joints, expected, atol=2e-5)
        assert solution.seed_index == 0
        current = solution.joints


def test_continuation_rejects_fallback_result_outside_joint_step_gate(
    piper_chain,
    monkeypatch,
):
    current = np.array([0.35, 0.75, -1.15, 0.25, 0.30, -0.45])
    target_joints = current.copy()
    target_joints[0] += 0.04
    escaped = target_joints.copy()
    escaped[0] = current[0] + 0.06
    calls = 0

    def escaped_solution(_residual, _seed, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(x=escaped.copy(), nfev=1)

    monkeypatch.setattr(
        "z_manip.kinematics.robust_ik.least_squares",
        escaped_solution,
    )
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(continuation_fallback_seeds=2),
    )

    with pytest.raises(IKFailure, match="max_joint_step_rad=0.050000"):
        solver.solve_continuation(
            piper_chain.forward(target_joints),
            current,
            max_joint_step_rad=0.05,
        )

    assert calls == 1 + solver.config.continuation_fallback_seeds


def test_continuation_fallback_selects_joint_continuity_lexicographically(
    piper_chain,
    monkeypatch,
):
    current = np.array([0.35, 0.75, -1.15, 0.25, 0.30, -0.45])
    farther = current.copy()
    farther[0] += 0.04
    nearer = current.copy()
    nearer[0] += 0.02
    results = iter((current, farther, nearer))

    def fake_forward(joints):
        pose = np.eye(4)
        if np.array_equal(np.asarray(joints), current):
            pose[0, 3] = 0.10
        return pose

    def solved(_residual, _seed, **_kwargs):
        return SimpleNamespace(x=next(results).copy(), nfev=1)

    monkeypatch.setattr(piper_chain, "forward", fake_forward)
    monkeypatch.setattr(
        "z_manip.kinematics.robust_ik.least_squares",
        solved,
    )
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(continuation_fallback_seeds=2),
    )

    solution = solver.solve_continuation(
        np.eye(4),
        current,
        max_joint_step_rad=0.05,
    )

    np.testing.assert_array_equal(solution.joints, nearer)
    assert solution.seed_index == 2


def test_continuation_propagates_caller_deadline_from_numerical_solve(
    piper_chain,
    monkeypatch,
):
    current = np.array([0.35, 0.75, -1.15, 0.25, 0.30, -0.45])
    target_joints = current.copy()
    target_joints[0] += 0.03
    now = 0.0

    def clock():
        return now

    def stalls(residual, seed, **_kwargs):
        nonlocal now
        while True:
            now += 0.04
            residual(seed)

    monkeypatch.setattr(
        "z_manip.kinematics.robust_ik.least_squares",
        stalls,
    )
    solver = RobustIKSolver(
        piper_chain,
        IKConfig(
            continuation_timeout_s=0.50,
            continuation_seed_timeout_s=0.40,
        ),
    )

    with pytest.raises(PlanningDeadlineExceeded, match="deadline by"):
        solver.solve_continuation(
            piper_chain.forward(target_joints),
            current,
            max_joint_step_rad=0.05,
            control=PlanningControl(deadline_s=0.10, monotonic_fn=clock),
        )
