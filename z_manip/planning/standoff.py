"""Choose visual-servo standoff by downstream grasp feasibility."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import inspect
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.grasp_ordering import directionally_diverse_indices
from z_manip.models.planner import PlanningError
from z_manip.planning_control import (
    PlanningAborted,
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)


@dataclass(frozen=True)
class ReachabilityStandoffConfig:
    """Geometric range and hierarchical budgets for standoff search."""

    min_camera_depth_m: float = 0.32
    max_camera_depth_m: float = 0.75
    depth_samples: int = 10
    base_motion_penalty: float = 0.02
    max_candidates: int = 8
    max_hypotheses: int = 16
    max_feasible_choices: int = 1
    search_timeout_s: float = 6.0
    hypothesis_timeout_s: float = 2.5
    solution_refinement_timeout_s: float = 0.35
    pair_ranking_timeout_s: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 < self.min_camera_depth_m < self.max_camera_depth_m:
            raise ValueError("invalid visually safe camera-depth interval")
        if self.depth_samples < 2:
            raise ValueError("standoff optimization needs at least two depth samples")
        if not np.isfinite(self.base_motion_penalty) or self.base_motion_penalty < 0.0:
            raise ValueError("base-motion penalty cannot be negative")
        if (
            self.max_candidates < 1
            or self.max_hypotheses < 1
            or self.max_feasible_choices < 1
        ):
            raise ValueError("standoff candidate and hypothesis counts must be positive")
        if self.max_hypotheses < max(self.max_candidates, self.depth_samples):
            raise ValueError(
                "standoff max_hypotheses must cover every configured "
                "candidate and depth category",
            )
        timeouts = (
            self.pair_ranking_timeout_s,
            self.search_timeout_s,
            self.hypothesis_timeout_s,
            self.solution_refinement_timeout_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in timeouts):
            raise ValueError("standoff anytime-search timeouts must be positive")


@dataclass(frozen=True, eq=False)
class StandoffChoice:
    desired_camera_depth_m: float
    base_displacement: np.ndarray
    candidates: GraspCandidates
    evaluation: object
    score: float


def _score(value: object) -> float:
    if isinstance(value, Mapping):
        result = value.get("score")
    else:
        result = getattr(value, "score", None)
    if result is None or not np.isfinite(float(result)):
        raise PlanningError("standoff evaluator returned no finite score")
    return float(result)


def _depth_priority(depths: np.ndarray) -> list[int]:
    """Probe interval endpoints, then bisect remaining gaps deterministically."""

    if len(depths) <= 1:
        return [0]
    result = [0, len(depths) - 1]
    intervals = [(0, len(depths) - 1)]
    while intervals:
        first, last = intervals.pop(0)
        middle = (first + last) // 2
        if middle in (first, last):
            continue
        result.append(middle)
        intervals.extend(((first, middle), (middle, last)))
    return result


def _hypothesis_order(
    candidate_count: int,
    depth_count: int,
) -> Iterator[tuple[int, int]]:
    """Cover both marginals first, then refine remaining pairs best-first."""

    if candidate_count < 1 or depth_count < 1:
        return
    common = math.gcd(candidate_count, depth_count)
    cycle = math.lcm(candidate_count, depth_count)
    for offset in range(common):
        for rank in range(cycle):
            yield (
                rank % candidate_count,
                (rank + offset) % depth_count,
            )


def _cost_ranked_hypothesis_order(costs: object) -> Iterator[tuple[int, int]]:
    """Seed fair traversal by minimizing failed edges, then ordinal cost."""

    values = np.asarray(costs, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("standoff pair costs must be a non-empty matrix")
    candidate_count, depth_count = values.shape
    ranked_costs = np.empty_like(values)
    ranked_pairs = sorted(
        (
            not np.isfinite(values[candidate_rank, depth_rank]),
            (
                values[candidate_rank, depth_rank]
                if np.isfinite(values[candidate_rank, depth_rank])
                else 0.0
            ),
            candidate_rank,
            depth_rank,
        )
        for candidate_rank in range(candidate_count)
        for depth_rank in range(depth_count)
    )
    for rank, (_failed, _cost, candidate_rank, depth_rank) in enumerate(
        ranked_pairs,
    ):
        ranked_costs[candidate_rank, depth_rank] = float(rank)

    failed = ~np.isfinite(values)
    matching_size = min(candidate_count, depth_count)
    failure_penalty = float(matching_size * values.size + 1)
    assignment_costs = ranked_costs + failure_penalty * failed
    matched_candidates, matched_depths = linear_sum_assignment(assignment_costs)
    first_matching = sorted(
        zip(matched_candidates.tolist(), matched_depths.tolist()),
        key=lambda pair: (assignment_costs[pair], pair[0], pair[1]),
    )
    candidate_order = [pair[0] for pair in first_matching]
    depth_order = [pair[1] for pair in first_matching]
    candidate_order.extend(
        sorted(
            set(range(candidate_count)) - set(candidate_order),
            key=lambda index: (float(np.min(assignment_costs[index])), index),
        ),
    )
    depth_order.extend(
        sorted(
            set(range(depth_count)) - set(depth_order),
            key=lambda index: (float(np.min(assignment_costs[:, index])), index),
        ),
    )
    for candidate_rank, depth_rank in _hypothesis_order(
        candidate_count,
        depth_count,
    ):
        yield candidate_order[candidate_rank], depth_order[depth_rank]


def _evaluator_control_mode(callback: Callable[..., object]) -> str:
    """Classify evaluator control transport without executing its body."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return "legacy"
    sentinel = object()
    try:
        signature.bind(sentinel, sentinel, sentinel, control=sentinel)
    except TypeError:
        try:
            signature.bind(sentinel, sentinel, sentinel, sentinel)
        except TypeError:
            try:
                signature.bind(sentinel, sentinel, sentinel)
            except TypeError as error:
                raise TypeError(
                    "standoff evaluator must accept three observations and "
                    "optional control",
                ) from error
            return "legacy"
        return "positional"
    return "keyword"


def _evaluate(
    callback: Callable[..., object],
    candidates: GraspCandidates,
    displacement: np.ndarray,
    desired_depth: float,
    control: PlanningControl,
    control_mode: str,
) -> object:
    """Invoke the evaluator exactly once using its classified control mode."""

    if control_mode == "positional":
        return callback(candidates, displacement, desired_depth, control)
    if control_mode == "keyword":
        return callback(
            candidates,
            displacement,
            desired_depth,
            control=control,
        )
    return callback(candidates, displacement, desired_depth)


class ReachabilityStandoffOptimizer:
    """Anytime best-first search over observed grasps and safe base depths.

    The target and camera ray come entirely from tracked RGB-D. This layer does
    not assume an object location or a fixed robot reach: the injected evaluator
    is the same complete grasp planner used after visual servo. Each raw grasp
    is evaluated alone so its symmetry family receives full IK, collision, and
    motion-planning validation. Endpoint-first depth probes and diagonal
    candidate/depth traversal prevent either dimension from starving the
    other when downstream IK or motion planning is expensive.
    """

    def __init__(self, config: ReachabilityStandoffConfig | None = None):
        self.config = config or ReachabilityStandoffConfig()

    def select(
        self,
        candidates: GraspCandidates,
        *,
        current_camera_depth_m: float,
        camera_rotation_base: object,
        evaluate: Callable[..., object],
        pair_cost: Callable[..., float] | None = None,
        control: PlanningControl | None = None,
    ) -> StandoffChoice:
        checkpoint(control, "standoff optimization")
        depth = float(current_camera_depth_m)
        rotation = np.asarray(camera_rotation_base, dtype=float)
        if not np.isfinite(depth) or depth <= 0.0:
            raise PlanningError("current tracked camera depth is invalid")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise PlanningError("camera rotation must be a finite 3x3 matrix")
        camera_forward = rotation[:, 2]
        travel_direction = camera_forward.copy()
        travel_direction[2] = 0.0
        horizontal_norm = float(np.linalg.norm(travel_direction))
        if horizontal_norm < 1e-3:
            raise PlanningError("camera ray has no usable horizontal base direction")
        travel_direction /= horizontal_norm
        depth_per_motion = float(np.dot(camera_forward, travel_direction))
        if depth_per_motion <= 0.05:
            raise PlanningError("base forward motion does not reduce tracked camera depth")

        grasps = np.asarray(candidates.grasps, dtype=float)
        scores = np.asarray(candidates.scores, dtype=float)
        centroid = np.asarray(candidates.centroid, dtype=float)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(grasps) == 0:
            raise PlanningError("standoff candidates must contain finite 4x4 grasps")
        if not np.all(np.isfinite(grasps)):
            raise PlanningError("standoff candidates contain non-finite grasps")
        if scores.shape != (len(grasps),) or not np.all(np.isfinite(scores)):
            raise PlanningError("standoff scores do not align with grasp poses")
        widths = (
            None
            if candidates.widths is None
            else np.asarray(candidates.widths, dtype=float)
        )
        if widths is not None and (
            widths.shape != (len(grasps),) or not np.all(np.isfinite(widths))
        ):
            raise PlanningError("standoff widths do not align with grasp poses")
        configured_depths = np.linspace(
            self.config.max_camera_depth_m,
            self.config.min_camera_depth_m,
            self.config.depth_samples,
        )
        # Never command reverse motion here. If already close, only the current
        # observed depth is evaluated and the downstream planner may reject it.
        depths = configured_depths[configured_depths <= depth + 1e-9]
        if len(depths) == 0:
            depths = np.asarray([depth])
        failures: list[tuple[int, float, str]] = []
        feasible: list[StandoffChoice] = []
        parent_control = control or PlanningControl()
        order = directionally_diverse_indices(
            grasps,
            scores,
            self.config.max_candidates,
        )
        depth_order = _depth_priority(depths)
        evaluator_control_mode = _evaluator_control_mode(evaluate)

        def transformed_hypothesis(
            candidate_rank: int,
            depth_rank: int,
        ) -> tuple[int, float, float, np.ndarray, GraspCandidates]:
            candidate_index = int(order[candidate_rank])
            desired_depth = float(depths[depth_order[depth_rank]])
            distance = max(0.0, (depth - desired_depth) / depth_per_motion)
            displacement = travel_direction * distance
            new_grasp = grasps[candidate_index].copy()
            new_grasp[:3, 3] -= displacement
            transformed = GraspCandidates(
                grasps=new_grasp[None, :, :],
                scores=scores[candidate_index:candidate_index + 1].copy(),
                centroid=centroid - displacement,
                frame=candidates.frame,
                num_raw=candidates.num_raw,
                widths=(
                    None
                    if widths is None
                    else widths[candidate_index:candidate_index + 1].copy()
                ),
            )
            return candidate_index, desired_depth, distance, displacement, transformed

        hypothesis_pairs: Iterator[tuple[int, int]]
        if pair_cost is not None:
            pair_cost_control_mode = _evaluator_control_mode(pair_cost)
            pair_costs = np.full((len(order), len(depth_order)), np.inf)
            ranking_complete = True
            ranking_control = parent_control.limited_to(
                self.config.pair_ranking_timeout_s,
                "standoff pair ranking budget",
            )
            for candidate_rank in range(len(order)):
                for depth_rank in range(len(depth_order)):
                    (
                        _candidate_index,
                        desired_depth,
                        _distance,
                        displacement,
                        transformed,
                    ) = transformed_hypothesis(candidate_rank, depth_rank)
                    try:
                        checkpoint(
                            ranking_control,
                            "standoff pair reachability ranking",
                        )
                        cost = float(_evaluate(
                            pair_cost,
                            transformed,
                            displacement,
                            desired_depth,
                            ranking_control,
                            pair_cost_control_mode,
                        ))
                        checkpoint(
                            ranking_control,
                            "standoff pair reachability ranking",
                        )
                        if np.isfinite(cost):
                            pair_costs[candidate_rank, depth_rank] = cost
                    except PlanningCancelled:
                        raise
                    except PlanningDeadlineExceeded:
                        checkpoint(control, "standoff pair reachability ranking")
                        ranking_complete = False
                        break
                    except PlanningAborted:
                        raise
                    except Exception:
                        # Ranking is advisory. Exact downstream planning owns
                        # geometry validation and candidate-level diagnostics.
                        continue
                if not ranking_complete:
                    break
            if ranking_complete:
                hypothesis_pairs = _cost_ranked_hypothesis_order(pair_costs)
            else:
                hypothesis_pairs = _hypothesis_order(len(order), len(depth_order))
        else:
            hypothesis_pairs = _hypothesis_order(len(order), len(depth_order))

        # Advisory ranking has its own short budget. Exact IK/collision/RRT
        # always receives the complete configured standoff search window.
        search_control = parent_control.limited_to(
            self.config.search_timeout_s,
            "standoff search budget",
        )
        refinement_control: PlanningControl | None = None
        for hypothesis_index, (candidate_rank, depth_rank) in enumerate(hypothesis_pairs):
            if hypothesis_index >= self.config.max_hypotheses:
                break
            (
                candidate_index,
                desired_depth,
                distance,
                displacement,
                transformed,
            ) = transformed_hypothesis(candidate_rank, depth_rank)
            active_control = refinement_control or search_control
            try:
                hypothesis_control = active_control.limited_to(
                    self.config.hypothesis_timeout_s,
                    "standoff hypothesis budget",
                )
            except PlanningDeadlineExceeded:
                checkpoint(control, "standoff hypothesis budget")
                break

            try:
                checkpoint(hypothesis_control, "standoff downstream evaluation")
                evaluation = _evaluate(
                    evaluate,
                    transformed,
                    displacement,
                    desired_depth,
                    hypothesis_control,
                    evaluator_control_mode,
                )
                checkpoint(hypothesis_control, "standoff downstream evaluation")
                score = _score(evaluation) - self.config.base_motion_penalty * distance
            except PlanningDeadlineExceeded as error:
                checkpoint(control, "standoff downstream evaluation")
                try:
                    checkpoint(search_control, "standoff search budget")
                except PlanningDeadlineExceeded:
                    failures.append((candidate_index, desired_depth, str(error)))
                    break
                if refinement_control is not None:
                    try:
                        checkpoint(refinement_control, "standoff solution refinement")
                    except PlanningDeadlineExceeded:
                        break
                failures.append((candidate_index, desired_depth, str(error)))
                continue
            except PlanningAborted:
                raise
            except PlanningError as error:
                failures.append((candidate_index, desired_depth, str(error)))
                continue

            feasible.append(StandoffChoice(
                desired_camera_depth_m=desired_depth,
                base_displacement=displacement,
                candidates=transformed,
                evaluation=evaluation,
                score=score,
            ))
            if len(feasible) >= self.config.max_feasible_choices:
                break
            if refinement_control is None:
                try:
                    refinement_control = search_control.limited_to(
                        self.config.solution_refinement_timeout_s,
                        "standoff solution refinement",
                    )
                except PlanningDeadlineExceeded:
                    # Refinement is optional once a complete choice exists.
                    # Preserve caller-owned aborts, but keep the feasible best
                    # when only this optimizer's local search budget elapsed.
                    checkpoint(control, "standoff solution refinement")
                    break

        checkpoint(control, "standoff solution selection")
        if feasible:
            return max(feasible, key=lambda choice: choice.score)
        # First/middle/last hypotheses retain representative reachability and
        # collision evidence while keeping the terminal payload bounded.
        if len(failures) <= 3:
            representative = failures
        else:
            representative = [
                failures[0],
                failures[len(failures) // 2],
                failures[-1],
            ]
        evidence = " | ".join(
            f"candidate={index} depth={sample_depth:.3f}m: {reason}"
            for index, sample_depth, reason in representative
        )
        raise PlanningError(
            "no visually safe standoff produced a feasible grasp plan; "
            f"evaluated_candidates={len({failure[0] for failure in failures})}; "
            f"candidate_limit={len(order)}; "
            f"evaluated_depths={len(depths)}; "
            f"evaluated_hypotheses={len(failures)}; {evidence}"
        )
