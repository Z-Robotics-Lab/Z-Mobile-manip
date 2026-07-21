from itertools import islice

import numpy as np
import pytest

from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.grasp_ordering import directionally_diverse_indices
from z_manip.models.planner import PlanningError
from z_manip.planning.standoff import (
    _cost_ranked_hypothesis_order,
    _hypothesis_order,
    ReachabilityStandoffConfig,
    ReachabilityStandoffOptimizer,
)
from z_manip.planning_control import (
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)


def _candidates():
    grasp = np.eye(4)
    grasp[:3, 3] = (1.5, 0.1, 0.2)
    return GraspCandidates(
        np.stack((grasp,)), np.array([0.9]), grasp[:3, 3], "arm_base", 1,
        np.array([0.04]),
    )


def _ranked_candidates():
    low_score = np.eye(4)
    low_score[:3, 3] = (1.5, 0.1, 0.2)
    high_score = np.eye(4)
    high_score[:3, 3] = (1.5, 0.2, 0.2)
    return GraspCandidates(
        np.stack((low_score, high_score)),
        np.array([0.2, 0.9]),
        np.array([1.5, 0.15, 0.2]),
        "arm_base",
        2,
        np.array([0.03, 0.05]),
    )


def _pose_with_approach(approach):
    approach = np.asarray(approach, dtype=float)
    approach /= np.linalg.norm(approach)
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(reference, approach))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    closing = np.cross(reference, approach)
    closing /= np.linalg.norm(closing)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack(
        (closing, np.cross(approach, closing), approach),
    )
    pose[:3, 3] = (1.5, 0.1, 0.2)
    return pose


def test_directional_order_preserves_quality_then_adds_max_min_separation():
    grasps = np.stack((
        _pose_with_approach((0.0, 0.0, 1.0)),
        _pose_with_approach((0.10, 0.0, 0.995)),
        _pose_with_approach((-0.10, 0.0, 0.995)),
        _pose_with_approach((1.0, 0.0, 0.0)),
        _pose_with_approach((0.0, 0.0, -1.0)),
    ))
    scores = np.array((1.0, 0.99, 0.98, 0.5, 0.4))

    order = directionally_diverse_indices(grasps, scores, 3)

    assert order.tolist() == [0, 4, 1]


def test_directional_order_reserves_half_of_cap_for_high_scores():
    approaches = [(0.01 * index, 0.0, 1.0) for index in range(8)]
    approaches.extend((
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ))
    grasps = np.stack([_pose_with_approach(value) for value in approaches])
    scores = np.concatenate((
        np.linspace(1.0, 0.93, 8),
        np.linspace(0.04, 0.01, 4),
    ))

    order = directionally_diverse_indices(grasps, scores, 8)

    assert order[::2].tolist() == [0, 1, 2, 3]
    assert any(index >= 8 for index in order[1::2])


def test_standoff_probes_depth_endpoints_before_bisection():
    evaluated = []

    def evaluate(candidates, displacement, desired_depth):
        evaluated.append((candidates, displacement.copy(), desired_depth))
        if desired_depth > 0.45:
            raise PlanningError("unreachable")
        return {"score": 1.0 - abs(desired_depth - 0.36)}

    optimizer = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        min_camera_depth_m=0.32,
        max_camera_depth_m=0.70,
        depth_samples=9,
        base_motion_penalty=0.02,
    ))
    rotation = np.array(((0, 0, 1), (-1, 0, 0), (0, -1, 0)), dtype=float)
    choice = optimizer.select(
        _candidates(),
        current_camera_depth_m=1.5,
        camera_rotation_base=rotation,
        evaluate=evaluate,
    )

    assert choice.desired_camera_depth_m == pytest.approx(0.32)
    assert choice.base_displacement[0] == pytest.approx(1.5 - 0.32)
    selected_grasp_x = choice.candidates.grasps[0, 0, 3]
    assert selected_grasp_x == pytest.approx(1.5 - choice.base_displacement[0])
    assert [sample[2] for sample in evaluated] == pytest.approx([0.70, 0.32])
    assert all(len(sample[0].grasps) == 1 for sample in evaluated)


def test_standoff_diagonal_order_does_not_starve_candidate_or_depth():
    evaluated = []
    optimizer = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        min_camera_depth_m=0.50,
        max_camera_depth_m=0.70,
        depth_samples=3,
    ))

    def evaluate(candidates, _displacement, desired_depth):
        candidate_y = float(candidates.grasps[0, 1, 3])
        evaluated.append((candidate_y, desired_depth))
        if candidate_y == pytest.approx(0.2):
            raise PlanningError("high-score candidate is unreachable")
        return {"score": 0.5}

    choice = optimizer.select(
        _ranked_candidates(),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)), dtype=float,
        ),
        evaluate=evaluate,
    )

    assert evaluated == pytest.approx([(0.2, 0.70), (0.1, 0.50)])
    assert choice.candidates.scores == pytest.approx([0.2])
    assert choice.candidates.widths == pytest.approx([0.03])
    assert choice.desired_camera_depth_m == pytest.approx(0.50)


def test_standoff_coverage_prefix_visits_every_candidate_and_depth_rank():
    prefix = list(islice(_hypothesis_order(8, 10), 10))

    assert {candidate_rank for candidate_rank, _depth_rank in prefix} == set(range(8))
    assert {_depth_rank for _candidate_rank, _depth_rank in prefix} == set(range(10))


def test_standoff_refinement_prefix_balances_both_marginals():
    prefix = list(islice(_hypothesis_order(8, 10), 16))
    candidate_load = np.bincount([pair[0] for pair in prefix], minlength=8)
    depth_load = np.bincount([pair[1] for pair in prefix], minlength=10)

    assert len(prefix) == len(set(prefix))
    assert candidate_load.tolist() == [2] * 8
    assert int(np.max(depth_load) - np.min(depth_load)) <= 1
    full = list(_hypothesis_order(8, 10))
    assert len(full) == 80
    assert len(set(full)) == 80


def test_standoff_candidate_limit_preserves_approach_diversity():
    approaches = (
        (0.0, 0.0, 1.0),
        (0.05, 0.0, 0.999),
        (-0.05, 0.0, 0.999),
        (1.0, 0.0, 0.0),
    )
    grasps = np.stack([_pose_with_approach(value) for value in approaches])
    candidates = GraspCandidates(
        grasps=grasps,
        scores=np.array((1.0, 0.99, 0.98, 0.5)),
        centroid=np.array((1.5, 0.1, 0.2)),
        frame="arm_base",
        num_raw=4,
        widths=np.full(4, 0.04),
    )
    evaluated = []

    def evaluate(transformed, _displacement, _desired_depth):
        approach = transformed.grasps[0, :3, 2]
        evaluated.append(approach.copy())
        if approach[0] < 0.9:
            raise PlanningError("only the lateral approach is reachable")
        return {"score": 1.0}

    choice = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        depth_samples=2,
        max_candidates=2,
        max_hypotheses=2,
    )).select(
        candidates,
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
    )

    assert len(evaluated) == 2
    assert choice.candidates.grasps[0, 0, 2] > 0.9


def test_pair_cost_nominates_one_reachable_depth_per_direction():
    grasps = []
    for offset in (0.0, 0.01, 0.02):
        grasp = np.eye(4)
        grasp[:3, 3] = (1.5, offset, 0.2)
        grasps.append(grasp)
    candidates = GraspCandidates(
        grasps=np.stack(grasps),
        scores=np.array((0.9, 0.8, 0.7)),
        centroid=np.array((1.5, 0.01, 0.2)),
        frame="arm_base",
        num_raw=3,
        widths=np.array((0.03, 0.04, 0.05)),
    )
    preferred_depth = {0.03: 0.60, 0.04: 0.40, 0.05: 0.50}
    ranked = []
    evaluated = []

    def pair_cost(transformed, _displacement, desired_depth):
        width = float(transformed.widths[0])
        ranked.append((width, desired_depth))
        return abs(desired_depth - preferred_depth[round(width, 2)]) + width * 1e-6

    def evaluate(transformed, _displacement, desired_depth):
        width = float(transformed.widths[0])
        evaluated.append((width, desired_depth))
        if not (width == pytest.approx(0.05) and desired_depth == pytest.approx(0.50)):
            raise PlanningError("pair is unreachable")
        return {"score": 1.0}

    choice = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        min_camera_depth_m=0.40,
        max_camera_depth_m=0.60,
        depth_samples=3,
        max_candidates=3,
        max_hypotheses=3,
    )).select(
        candidates,
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
        pair_cost=pair_cost,
    )

    assert len(ranked) == 9
    assert len(set(ranked)) == 9
    assert evaluated == pytest.approx([
        (0.03, 0.60),
        (0.04, 0.40),
        (0.05, 0.50),
    ])
    assert choice.candidates.widths == pytest.approx([0.05])
    assert choice.desired_camera_depth_m == pytest.approx(0.50)


def test_pair_cost_order_keeps_candidate_and_depth_prefixes_balanced():
    costs = np.tile(np.arange(10, dtype=float), (8, 1))
    prefix = list(islice(_cost_ranked_hypothesis_order(costs), 16))
    candidate_load = np.bincount([pair[0] for pair in prefix], minlength=8)
    depth_load = np.bincount([pair[1] for pair in prefix], minlength=10)

    assert len(prefix) == len(set(prefix))
    assert candidate_load.tolist() == [2] * 8
    assert int(np.max(depth_load) - np.min(depth_load)) <= 1
    assert set(pair[0] for pair in prefix) == set(range(8))
    assert set(pair[1] for pair in prefix) == set(range(10))
    full = list(_cost_ranked_hypothesis_order(costs))
    assert len(full) == 80
    assert len(set(full)) == 80


@pytest.mark.parametrize("shape", ((3, 3), (8, 8), (7, 10), (10, 7)))
def test_pair_cost_order_is_a_fair_full_permutation_for_varied_shapes(shape):
    costs = np.random.default_rng(17).random(shape)
    candidate_count, depth_count = shape
    ordered = list(_cost_ranked_hypothesis_order(costs))

    assert len(ordered) == candidate_count * depth_count
    assert len(set(ordered)) == len(ordered)
    for prefix_size in range(1, len(ordered) + 1):
        prefix = ordered[:prefix_size]
        candidate_load = np.bincount(
            [pair[0] for pair in prefix],
            minlength=candidate_count,
        )
        depth_load = np.bincount(
            [pair[1] for pair in prefix],
            minlength=depth_count,
        )
        assert int(np.max(candidate_load) - np.min(candidate_load)) <= 1
        assert int(np.max(depth_load) - np.min(depth_load)) <= 1


def test_pair_cost_first_matching_avoids_failed_edges_when_finite_exists():
    costs = np.array((
        (0.0, 2.0, 4.0),
        (6.0, 1.0, 3.0),
        (5.0, 7.0, np.inf),
    ))

    first_matching = list(islice(_cost_ranked_hypothesis_order(costs), 3))

    assert all(np.isfinite(costs[pair]) for pair in first_matching)


def test_pair_cost_order_reaches_each_candidates_second_depth_under_default_budget():
    grasps = np.stack([_pose_with_approach((0.0, 0.0, 1.0)) for _ in range(8)])
    candidates = GraspCandidates(
        grasps=grasps,
        scores=np.linspace(0.9, 0.2, 8),
        centroid=np.array((1.5, 0.1, 0.2)),
        frame="arm_base",
        num_raw=8,
        widths=np.linspace(0.03, 0.037, 8),
    )
    side_attempts = 0
    evaluated = []

    def pair_cost(transformed, _displacement, desired_depth):
        width = float(transformed.widths[0])
        return (100.0 if width > 0.0365 else 0.0) + desired_depth

    def evaluate(transformed, _displacement, desired_depth):
        nonlocal side_attempts
        width = float(transformed.widths[0])
        evaluated.append((width, desired_depth))
        if width > 0.0365:
            side_attempts += 1
            if side_attempts == 2:
                return {"score": 1.0}
        raise PlanningError("only the side candidate's second depth is feasible")

    choice = ReachabilityStandoffOptimizer().select(
        candidates,
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
        pair_cost=pair_cost,
    )

    assert side_attempts == 2
    assert len(evaluated) <= 16
    assert choice.candidates.widths == pytest.approx([0.037])


def test_pair_ranking_timeout_falls_back_before_full_exact_search_budget():
    now = 0.0
    evaluated = []

    def pair_cost(_candidates, _displacement, _desired_depth, *, control=None):
        nonlocal now
        now = 0.11
        checkpoint(control, "test pair ranking")
        return 0.0

    def evaluate(_candidates, _displacement, desired_depth, *, control=None):
        evaluated.append((desired_depth, control.deadline_s))
        return {"score": 1.0}

    choice = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        depth_samples=2,
        max_candidates=1,
        max_hypotheses=2,
        pair_ranking_timeout_s=0.10,
        search_timeout_s=0.50,
    )).select(
        _candidates(),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
        pair_cost=pair_cost,
        control=PlanningControl(deadline_s=2.0, monotonic_fn=lambda: now),
    )

    assert choice.desired_camera_depth_m == pytest.approx(0.75)
    assert evaluated == pytest.approx([(0.75, 0.61)])


def test_pair_ranking_error_is_advisory_and_does_not_drop_valid_candidate():
    ranked = []
    evaluated = []

    def pair_cost(transformed, _displacement, _desired_depth):
        width = float(transformed.widths[0])
        ranked.append(width)
        if width > 0.04:
            raise ValueError("malformed heuristic-only pose")
        return 0.0

    def evaluate(transformed, _displacement, _desired_depth):
        width = float(transformed.widths[0])
        evaluated.append(width)
        if width > 0.04:
            raise PlanningError("high-score candidate is invalid")
        return {"score": 1.0}

    choice = ReachabilityStandoffOptimizer(ReachabilityStandoffConfig(
        depth_samples=2,
        max_candidates=2,
        max_hypotheses=2,
    )).select(
        _ranked_candidates(),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
        pair_cost=pair_cost,
    )

    assert sorted(set(ranked)) == pytest.approx([0.03, 0.05])
    assert evaluated == pytest.approx([0.03])
    assert choice.candidates.widths == pytest.approx([0.03])


def test_standoff_default_budget_reaches_rank_seven_candidate():
    grasps = []
    for index in range(8):
        grasp = np.eye(4)
        grasp[:3, 3] = (1.5, 0.01 * index, 0.2)
        grasps.append(grasp)
    candidates = GraspCandidates(
        grasps=np.stack(grasps),
        scores=np.linspace(0.9, 0.2, 8),
        centroid=np.array([1.5, 0.035, 0.2]),
        frame="arm_base",
        num_raw=8,
        widths=np.linspace(0.03, 0.037, 8),
    )
    evaluated = []

    def evaluate(transformed, _displacement, _desired_depth):
        width = float(transformed.widths[0])
        evaluated.append(width)
        if width < 0.0365:
            raise PlanningError("only rank seven is feasible")
        return {"score": 1.0}

    choice = ReachabilityStandoffOptimizer().select(
        candidates,
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
    )

    assert len(evaluated) == 8
    assert choice.candidates.widths == pytest.approx([0.037])


def test_standoff_rejects_budget_that_cannot_cover_both_marginals():
    with pytest.raises(ValueError, match="cover every configured"):
        ReachabilityStandoffConfig(
            depth_samples=10,
            max_candidates=8,
            max_hypotheses=9,
        )


def test_standoff_fails_when_camera_ray_is_vertical_or_no_pose_is_feasible():
    optimizer = ReachabilityStandoffOptimizer()
    vertical_rotation = np.eye(3)
    with pytest.raises(PlanningError, match="horizontal"):
        optimizer.select(
            _candidates(),
            current_camera_depth_m=1.0,
            camera_rotation_base=vertical_rotation,
            evaluate=lambda *_args: None,
        )
    with pytest.raises(PlanningError, match="evaluated_depths=10") as captured:
        optimizer.select(
            _candidates(),
            current_camera_depth_m=1.0,
            camera_rotation_base=np.array(
                ((0, 0, 1), (-1, 0, 0), (0, -1, 0)), dtype=float,
            ),
            evaluate=lambda *_args: (_ for _ in ()).throw(PlanningError("no")),
        )
    assert "depth=0.750m: no" in str(captured.value)
    assert "evaluated_hypotheses=10" in str(captured.value)


def test_standoff_cancel_stops_after_current_downstream_evaluation():
    optimizer = ReachabilityStandoffOptimizer()
    evaluated = []
    cancelled = False

    def evaluate(*_args):
        nonlocal cancelled
        evaluated.append(True)
        cancelled = True
        return {"score": 1.0}

    with pytest.raises(PlanningCancelled, match="downstream evaluation was cancelled"):
        optimizer.select(
            _candidates(),
            current_camera_depth_m=1.0,
            camera_rotation_base=np.array(
                ((0, 0, 1), (-1, 0, 0), (0, -1, 0)), dtype=float,
            ),
            evaluate=evaluate,
            control=PlanningControl(cancel_check=lambda: cancelled),
        )

    assert len(evaluated) == 1


def test_standoff_deadline_stops_after_current_downstream_evaluation():
    optimizer = ReachabilityStandoffOptimizer()
    evaluated = []
    now = 0.0

    def evaluate(candidates, _displacement, _desired_depth):
        nonlocal now
        evaluated.append(float(candidates.grasps[0, 1, 3]))
        now = 2.0
        return {"score": 1.0}

    with pytest.raises(
        PlanningDeadlineExceeded,
        match="monotonic deadline by 1.000000 s",
    ):
        optimizer.select(
            _ranked_candidates(),
            current_camera_depth_m=1.0,
            camera_rotation_base=np.array(
                ((0, 0, 1), (-1, 0, 0), (0, -1, 0)), dtype=float,
            ),
            evaluate=evaluate,
            control=PlanningControl(deadline_s=1.0, monotonic_fn=lambda: now),
        )

    assert evaluated == pytest.approx([0.2])


def test_keyword_only_evaluator_receives_child_control():
    received = []

    def evaluate(_candidates, _displacement, _desired_depth, *, control=None):
        received.append(control)
        return {"score": 1.0}

    ReachabilityStandoffOptimizer().select(
        _candidates(),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
    )

    assert len(received) == 1
    assert isinstance(received[0], PlanningControl)


def test_variadic_evaluator_prefers_keyword_control_without_extra_args():
    received = []

    def evaluate(
        _candidates,
        _displacement,
        _desired_depth,
        *args,
        control=None,
    ):
        received.append((args, control))
        return {"score": 1.0}

    ReachabilityStandoffOptimizer().select(
        _candidates(),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
    )

    assert len(received) == 1
    assert received[0][0] == ()
    assert isinstance(received[0][1], PlanningControl)


def test_positional_only_evaluator_receives_child_control():
    received = []

    def evaluate(_candidates, _displacement, _desired_depth, control, /):
        received.append(control)
        return {"score": 1.0}

    ReachabilityStandoffOptimizer().select(
        _candidates(),
        current_camera_depth_m=1.0,
        camera_rotation_base=np.array(
            ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
            dtype=float,
        ),
        evaluate=evaluate,
    )

    assert len(received) == 1
    assert isinstance(received[0], PlanningControl)


def test_evaluator_body_type_error_is_not_retried_as_legacy():
    calls = 0

    def evaluate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise TypeError("evaluator implementation failed")

    with pytest.raises(TypeError, match="implementation failed"):
        ReachabilityStandoffOptimizer().select(
            _candidates(),
            current_camera_depth_m=1.0,
            camera_rotation_base=np.array(
                ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
                dtype=float,
            ),
            evaluate=evaluate,
        )

    assert calls == 1
