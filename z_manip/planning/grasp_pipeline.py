"""Turn scored 6-DoF grasps into fully checked arm motion programs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
import inspect
from typing import Protocol

import numpy as np

from z_manip.ik.symmetry import expand_symmetry
from z_manip.kinematics.robust_ik import IKFailure, IKSolution
from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.grasp_ordering import (
    directionally_diverse_indices,
    lateral_approach_scores,
)
from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning_control import (
    PlanningAborted,
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)


class IKSolver(Protocol):
    def solve(
        self,
        target: np.ndarray,
        current: np.ndarray | None = None,
        *,
        control: PlanningControl | None = None,
    ) -> IKSolution:
        ...


class ContinuationIKSolver(IKSolver, Protocol):
    def solve_continuation(
        self,
        target: np.ndarray,
        current: np.ndarray,
        *,
        max_joint_step_rad: float,
        control: PlanningControl | None = None,
    ) -> IKSolution:
        ...


class JointPlanner(Protocol):
    def plan_joint(
        self,
        start_joints: object,
        goal_joints: object,
        *,
        timeout_s: float = 5.0,
        control: PlanningControl | None = None,
    ) -> JointTrajectory:
        ...

    def segment_valid(
        self,
        first: object,
        second: object,
        *,
        control: PlanningControl | None = None,
    ) -> bool:
        ...


@dataclass(frozen=True)
class GraspPlanConfig:
    """Hierarchical limits for deterministic anytime grasp search."""

    pregrasp_distance_m: float = 0.10
    approach_steps: int = 6
    lift_distance_m: float = 0.10
    lift_steps: int = 5
    lift_direction_base: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fallback_lift_vertical_m: float = 0.045
    fallback_lift_retreat_m: float = 0.025
    symmetry_samples: int = 2
    min_width_m: float = 0.008
    max_width_m: float = 0.075
    planning_timeout_s: float = 4.0
    max_candidates: int = 32
    max_cartesian_joint_step_rad: float = 0.45
    max_feasible_plans: int = 6
    max_hypotheses: int = 16
    search_timeout_s: float = 8.0
    hypothesis_timeout_s: float = 2.5
    solution_refinement_timeout_s: float = 0.35
    joint_limit_penalty: float = 0.02
    lateral_approach_prior_weight: float = 0.0
    overhead_approach_penalty_weight: float = 0.0
    tool_from_tip: tuple[tuple[float, float, float, float], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        if self.pregrasp_distance_m <= 0.0 or self.lift_distance_m <= 0.0:
            raise ValueError("grasp approach and lift distances must be positive")
        if (
            self.fallback_lift_vertical_m <= 0.0
            or self.fallback_lift_retreat_m < 0.0
        ):
            raise ValueError("fallback lift distances must be positive/non-negative")
        if self.approach_steps < 2 or self.lift_steps < 2:
            raise ValueError("Cartesian approach and lift need at least two samples")
        if (
            self.symmetry_samples < 1
            or self.max_candidates < 1
            or self.max_feasible_plans < 1
            or self.max_hypotheses < 1
        ):
            raise ValueError("grasp family and candidate counts must be positive")
        if not 0.0 <= self.min_width_m < self.max_width_m:
            raise ValueError("invalid gripper aperture range")
        if not (
            0.0 <= self.lateral_approach_prior_weight <= 1.0
            and 0.0 <= self.overhead_approach_penalty_weight <= 1.0
        ):
            raise ValueError("grasp approach preference weights must be within [0, 1]")
        timeouts = (
            self.planning_timeout_s,
            self.search_timeout_s,
            self.hypothesis_timeout_s,
            self.solution_refinement_timeout_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in timeouts):
            raise ValueError("planning and anytime-search timeouts must be positive")
        if self.max_cartesian_joint_step_rad <= 0.0:
            raise ValueError("Cartesian IK joint-step limit must be positive")
        if self.joint_limit_penalty < 0.0:
            raise ValueError("joint-limit penalty cannot be negative")
        tool = np.asarray(self.tool_from_tip, dtype=float)
        if tool.shape != (4, 4) or not np.allclose(tool[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError("tool_from_tip must be a homogeneous transform")


@dataclass(frozen=True)
class CandidateFailure:
    candidate_index: int
    symmetry_index: int | None
    stage: str
    reason: str


class _HypothesisRejected(PlanningError):
    """One grasp hypothesis failed without aborting the aggregate search."""

    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage


def _path_validator_control_mode(callback: Callable[..., object]) -> str:
    """Classify a path callback without executing or retrying its body."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError) as error:
        raise TypeError("approach path validator must expose an inspectable signature") from error
    sentinel = object()
    try:
        signature.bind(sentinel, sentinel)
    except TypeError:
        try:
            signature.bind(sentinel, control=sentinel)
        except TypeError:
            try:
                signature.bind(sentinel)
            except TypeError as error:
                raise TypeError(
                    "approach path validator must accept path and optional control",
                ) from error
            return "legacy"
        return "keyword"
    return "positional"


def _accepts_control_keyword(
    callback: Callable[..., object],
    label: str,
    *args: object,
    **kwargs: object,
) -> bool:
    """Classify optional backend control support without calling its body."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        # Opaque extension callables retain the pre-budget legacy invocation.
        return False
    sentinel = object()
    try:
        signature.bind(*args, **kwargs, control=sentinel)
    except TypeError:
        try:
            signature.bind(*args, **kwargs)
        except TypeError as error:
            raise TypeError(
                f"{label} does not implement the required planning interface",
            ) from error
        return False
    return True


def _accepts_optional_keyword(
    callback: Callable[..., object],
    label: str,
    keyword: str,
    *args: object,
    **kwargs: object,
) -> bool:
    """Detect one additive callback keyword while preserving legacy callables."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    sentinel = object()
    extended = dict(kwargs)
    extended[keyword] = sentinel
    try:
        signature.bind(*args, **extended)
    except TypeError:
        try:
            signature.bind(*args, **kwargs)
        except TypeError as error:
            raise TypeError(
                f"{label} does not implement the required planning interface",
            ) from error
        return False
    return True


def _failure_summary(failures: list[CandidateFailure]) -> str:
    """Return bounded, actionable evidence without flooding task diagnostics."""
    if not failures:
        return "no candidate evaluations were recorded"
    counts = Counter(failure.stage for failure in failures)
    stage_counts = ",".join(
        f"{stage}:{count}" for stage, count in sorted(counts.items())
    )
    examples: list[str] = []
    seen: set[tuple[str, str]] = set()
    for failure in reversed(failures):
        key = (failure.stage, failure.reason)
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            f"#{failure.candidate_index}/{failure.symmetry_index} "
            f"{failure.stage}: {failure.reason}"
        )
        if len(examples) >= 3:
            break
    return f"rejections={{{stage_counts}}}; examples=" + " | ".join(examples)


@dataclass(frozen=True, eq=False)
class PlannedGrasp:
    candidate_index: int
    symmetry_index: int
    grasp_pose: np.ndarray
    pregrasp_pose: np.ndarray
    transit: JointTrajectory
    approach_joints: np.ndarray
    lift_joints: np.ndarray
    required_width_m: float | None
    score: float
    failures: tuple[CandidateFailure, ...]
    selected_global_rank: int = 1
    higher_rank_rejection_count: int = 0
    lift_pose: np.ndarray | None = None
    trajectory_refinement: object | None = None
    lateral_approach_bonus: float = 0.0
    overhead_approach_penalty: float = 0.0


class GraspPlanningError(PlanningError):
    """Planning failure that preserves every evaluated hypothesis rejection."""

    def __init__(self, message: str, failures: tuple[CandidateFailure, ...]):
        super().__init__(message)
        self.failures = failures


def _interpolate_pose(first: np.ndarray, second: np.ndarray, steps: int) -> list[np.ndarray]:
    # Orientations are intentionally held equal for grasp approach/lift. Grasp
    # orientation interpolation belongs in a general Cartesian motion module;
    # changing the contact frame while entering the object is unsafe.
    if not np.allclose(first[:3, :3], second[:3, :3], atol=1e-7):
        raise ValueError("Cartesian grasp segments require a fixed orientation")
    poses = []
    for alpha in np.linspace(0.0, 1.0, steps):
        pose = first.copy()
        pose[:3, 3] = first[:3, 3] + alpha * (second[:3, 3] - first[:3, 3])
        poses.append(pose)
    return poses


def grasp_pregrasp_pose(grasp: object, distance_m: float) -> np.ndarray:
    """Return the fixed-orientation pregrasp used by the execution planner."""

    pose = np.asarray(grasp, dtype=float)
    distance = float(distance_m)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("grasp must be a finite 4x4 pose")
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("pregrasp distance must be finite and positive")
    result = pose.copy()
    result[:3, 3] -= distance * result[:3, 2]
    return result


def tool_tip_pose(tool_pose: object, tool_from_tip: object) -> np.ndarray:
    """Map a grasp-tool pose to the IK chain tip using configured geometry."""

    tool = np.asarray(tool_pose, dtype=float)
    offset = np.asarray(tool_from_tip, dtype=float)
    if tool.shape != (4, 4) or offset.shape != (4, 4):
        raise ValueError("tool and tip transforms must be 4x4")
    if not np.all(np.isfinite(tool)) or not np.all(np.isfinite(offset)):
        raise ValueError("tool and tip transforms must be finite")
    return tool @ np.linalg.inv(offset)


class GraspPlanGenerator:
    """Filter aperture, IK and collision planning for every grasp family member."""

    def __init__(
        self,
        ik_solver: IKSolver,
        joint_planner: JointPlanner,
        config: GraspPlanConfig | None = None,
        approach_segment_valid: Callable[[object, object, bool], bool] | None = None,
        lift_segment_valid: Callable[[object, object, object], bool] | None = None,
        approach_path_valid: Callable[..., bool] | None = None,
    ) -> None:
        if approach_segment_valid is not None and approach_path_valid is not None:
            raise ValueError(
                "approach segment and path validators are mutually exclusive",
            )
        self.ik_solver = ik_solver
        self.joint_planner = joint_planner
        sentinel = object()
        self._ik_accepts_control = _accepts_control_keyword(
            self.ik_solver.solve,
            "IK solver",
            sentinel,
            current=sentinel,
        )
        continuation_solve = getattr(self.ik_solver, "solve_continuation", None)
        self._continuation_solve = (
            continuation_solve if callable(continuation_solve) else None
        )
        self._continuation_accepts_control = (
            False
            if self._continuation_solve is None
            else _accepts_control_keyword(
                self._continuation_solve,
                "Cartesian continuation IK solver",
                sentinel,
                current=sentinel,
                max_joint_step_rad=sentinel,
            )
        )
        self._planner_accepts_control = _accepts_control_keyword(
            self.joint_planner.plan_joint,
            "joint planner",
            sentinel,
            sentinel,
            timeout_s=sentinel,
        )
        self._segment_accepts_control = _accepts_control_keyword(
            self.joint_planner.segment_valid,
            "joint segment validator",
            sentinel,
            sentinel,
        )
        self.config = config or GraspPlanConfig()
        self.approach_segment_valid = approach_segment_valid
        self.approach_path_valid = approach_path_valid
        self._approach_path_control_mode = (
            None
            if approach_path_valid is None
            else _path_validator_control_mode(approach_path_valid)
        )
        self.lift_segment_valid = lift_segment_valid
        sentinel = object()
        if approach_path_valid is None:
            self._approach_path_accepts_width = False
        elif self._approach_path_control_mode == "positional":
            self._approach_path_accepts_width = _accepts_optional_keyword(
                approach_path_valid,
                "approach path validator",
                "required_width_m",
                sentinel,
                sentinel,
            )
        elif self._approach_path_control_mode == "keyword":
            self._approach_path_accepts_width = _accepts_optional_keyword(
                approach_path_valid,
                "approach path validator",
                "required_width_m",
                sentinel,
                control=sentinel,
            )
        else:
            self._approach_path_accepts_width = _accepts_optional_keyword(
                approach_path_valid,
                "approach path validator",
                "required_width_m",
                sentinel,
            )
        self._lift_accepts_width = (
            False
            if lift_segment_valid is None
            else _accepts_optional_keyword(
                lift_segment_valid,
                "lift segment validator",
                "required_width_m",
                sentinel,
                sentinel,
                sentinel,
            )
        )

    def _tip_pose(self, tool_pose: np.ndarray) -> np.ndarray:
        return tool_tip_pose(tool_pose, self.config.tool_from_tip)

    def _reachability_order(
        self,
        grasps: np.ndarray,
        order: np.ndarray,
        pose_ranker: Callable[..., float] | None,
        search_control: PlanningControl,
    ) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
        """Use the cheap FK seed ranker to schedule expensive exact IK.

        Candidate zero remains the highest-scored retained grasp.  If it does
        not work, the remaining candidates are ordered by their best symmetry
        cost instead of spending the whole anytime budget walking score order.
        This is deliberately enabled only for first-feasible searches: a
        multi-plan refinement search must retain score order so that it can
        compare the requested number of high-quality complete programs.

        The ranker is advisory.  It receives a small child budget, and any
        timeout falls back to the original deterministic score order.  The
        per-symmetry values are returned so the inner loop does not repeat the
        same FK calculations.
        """

        if (
            pose_ranker is None
            or len(order) < 2
            or self.config.max_feasible_plans != 1
        ):
            return order, {}
        try:
            ranking_control = search_control.limited_to(
                self.config.solution_refinement_timeout_s,
                "grasp reachability ordering budget",
            )
        except PlanningDeadlineExceeded:
            return order, {}

        costs: dict[tuple[int, int], float] = {}
        candidate_costs: dict[int, float] = {}
        try:
            for candidate_index_raw in order:
                candidate_index = int(candidate_index_raw)
                family = expand_symmetry(
                    grasps[candidate_index],
                    n_about_axis=self.config.symmetry_samples,
                )
                best = float("inf")
                for symmetry_index, grasp in enumerate(family):
                    checkpoint(ranking_control, "grasp reachability ordering")
                    target = self._tip_pose(grasp_pregrasp_pose(
                        grasp,
                        self.config.pregrasp_distance_m,
                    ))
                    try:
                        cost = float(pose_ranker(target, control=ranking_control))
                    except PlanningDeadlineExceeded:
                        raise
                    except PlanningAborted:
                        raise
                    except Exception:
                        cost = float("inf")
                    costs[(candidate_index, symmetry_index)] = cost
                    if np.isfinite(cost):
                        best = min(best, cost)
                candidate_costs[candidate_index] = best
        except PlanningDeadlineExceeded:
            return order, {}
        except PlanningAborted:
            raise
        except ValueError:
            # Invalid families are reported by the normal evaluation path.
            return order, {}

        first = int(order[0])
        original_rank = {int(index): rank for rank, index in enumerate(order)}
        remainder = sorted(
            (int(index) for index in order[1:]),
            key=lambda index: (
                not np.isfinite(candidate_costs.get(index, float("inf"))),
                candidate_costs.get(index, float("inf")),
                original_rank[index],
            ),
        )
        return np.asarray((first, *remainder), dtype=int), costs

    def _cartesian_ik(
        self,
        poses: list[np.ndarray],
        seed: np.ndarray,
        control: PlanningControl | None = None,
    ) -> tuple[np.ndarray, IKSolution]:
        if len(poses) < 2:
            raise ValueError("Cartesian IK segments need at least two poses")
        current = np.asarray(seed, dtype=float)
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise IKFailure("Cartesian IK seed must be a finite joint vector")
        # The seed was already solved for poses[0] by the preceding stage. Keep
        # the waypoint for API/execution compatibility without repeating IK.
        joints = [current.copy()]
        last_solution = None
        for pose in poses[1:]:
            checkpoint(control, "grasp Cartesian IK")
            last_solution = self._solve_continuation(
                self._tip_pose(pose),
                current=current,
                control=control,
            )
            next_joints = np.asarray(last_solution.joints, dtype=float)
            if next_joints.shape != current.shape or not np.all(np.isfinite(next_joints)):
                raise IKFailure(
                    "Cartesian IK returned an invalid joint vector",
                )
            if (
                np.max(np.abs(next_joints - current))
                > self.config.max_cartesian_joint_step_rad
            ):
                raise IKFailure(
                    "Cartesian IK changed branch beyond configured joint-step limit",
                )
            current = next_joints
            joints.append(current.copy())
        assert last_solution is not None
        return np.asarray(joints), last_solution

    def _solve_continuation(
        self,
        target: np.ndarray,
        *,
        current: np.ndarray,
        control: PlanningControl | None,
    ) -> IKSolution:
        """Use bounded local IK when supported, with a legacy safe fallback."""

        if self._continuation_solve is None:
            return self._solve(target, current=current, control=control)
        checkpoint(control, "grasp Cartesian continuation IK")
        kwargs = {
            "current": current,
            "max_joint_step_rad": self.config.max_cartesian_joint_step_rad,
        }
        if control is not None and self._continuation_accepts_control:
            kwargs["control"] = control
        result = self._continuation_solve(target, **kwargs)
        checkpoint(control, "grasp Cartesian continuation IK")
        return result

    def _solve(
        self,
        target: np.ndarray,
        *,
        current: np.ndarray,
        control: PlanningControl | None,
    ) -> IKSolution:
        """Pass control only when requested, preserving legacy backends."""

        checkpoint(control, "grasp IK")
        if control is not None and self._ik_accepts_control:
            result = self.ik_solver.solve(target, current=current, control=control)
        else:
            result = self.ik_solver.solve(target, current=current)
        checkpoint(control, "grasp IK")
        return result

    def _segment_valid(
        self,
        first: object,
        second: object,
        *,
        control: PlanningControl | None,
    ) -> bool:
        checkpoint(control, "grasp segment collision checking")
        if control is not None and self._segment_accepts_control:
            result = self.joint_planner.segment_valid(first, second, control=control)
        else:
            result = self.joint_planner.segment_valid(first, second)
        checkpoint(control, "grasp segment collision checking")
        return bool(result)

    def _plan_joint(
        self,
        start: object,
        goal: object,
        *,
        control: PlanningControl | None,
    ) -> JointTrajectory:
        checkpoint(control, "grasp transit planning")
        kwargs = {"timeout_s": self.config.planning_timeout_s}
        if control is not None and self._planner_accepts_control:
            kwargs["control"] = control
        result = self.joint_planner.plan_joint(start, goal, **kwargs)
        checkpoint(control, "grasp transit planning")
        return result

    def _evaluate_hypothesis(
        self,
        *,
        candidate_index: int,
        symmetry_index: int,
        grasp: np.ndarray,
        width: float | None,
        candidate_score: float,
        current: np.ndarray,
        lift_direction: np.ndarray,
        control: PlanningControl,
    ) -> tuple[float, tuple[object, ...]]:
        """Fully validate one 6-DoF grasp under its local child budget."""

        pregrasp = grasp_pregrasp_pose(grasp, self.config.pregrasp_distance_m)
        try:
            pregrasp_solution = self._solve(
                self._tip_pose(pregrasp),
                current=current,
                control=control,
            )
        except PlanningAborted:
            raise
        except IKFailure as error:
            raise _HypothesisRejected(
                "ik",
                f"pregrasp IK failed: {error}",
            ) from error
        try:
            approach_joints, grasp_solution = self._cartesian_ik(
                _interpolate_pose(pregrasp, grasp, self.config.approach_steps),
                pregrasp_solution.joints,
                control,
            )
        except PlanningAborted:
            raise
        except IKFailure as error:
            raise _HypothesisRejected(
                "ik",
                f"approach IK failed: {error}",
            ) from error
        approach_path = np.vstack((pregrasp_solution.joints, approach_joints))
        if self.approach_path_valid is not None:
            checkpoint(control, "grasp approach collision checking")
            width_kwargs = (
                {"required_width_m": width}
                if self._approach_path_accepts_width
                else {}
            )
            if self._approach_path_control_mode == "positional":
                valid = self.approach_path_valid(
                    approach_path,
                    control,
                    **width_kwargs,
                )
            elif self._approach_path_control_mode == "keyword":
                valid = self.approach_path_valid(
                    approach_path,
                    control=control,
                    **width_kwargs,
                )
            else:
                valid = self.approach_path_valid(approach_path, **width_kwargs)
            checkpoint(control, "grasp approach collision checking")
        else:
            valid = True
            segments = tuple(zip(approach_path, approach_path[1:]))
            for index, (first, second) in enumerate(segments):
                checkpoint(control, "grasp approach collision checking")
                if self.approach_segment_valid is None:
                    segment_valid = self._segment_valid(
                        first,
                        second,
                        control=control,
                    )
                else:
                    segment_valid = self.approach_segment_valid(
                        first,
                        second,
                        index == len(segments) - 1,
                    )
                    checkpoint(control, "grasp approach collision checking")
                if not segment_valid:
                    valid = False
                    break
        if not valid:
            raise _HypothesisRejected(
                "approach_collision",
                "Cartesian approach intersects the planning scene",
            )

        # Contact reachability is not enough: the arm must also be able to
        # leave contact while holding the object.  Near the outer workspace a
        # fixed world-Z lift can be infeasible even though contact itself is
        # accurate.  Try that nominal program first, then one shorter
        # up-and-back lift that preserves vertical clearance while moving the
        # wrist toward the arm base and away from the reach boundary.
        nominal_lift = grasp.copy()
        nominal_lift[:3, 3] += self.config.lift_distance_m * lift_direction
        horizontal_to_base = -np.asarray(grasp[:3, 3], dtype=float)
        horizontal_to_base[2] = 0.0
        horizontal_norm = float(np.linalg.norm(horizontal_to_base))
        if horizontal_norm > 1e-9:
            horizontal_to_base /= horizontal_norm
        fallback_lift = grasp.copy()
        fallback_lift[:3, 3] += (
            self.config.fallback_lift_vertical_m * lift_direction
            + self.config.fallback_lift_retreat_m * horizontal_to_base
        )
        lift_attempts = (("nominal", nominal_lift),)
        if not np.allclose(fallback_lift, nominal_lift, atol=1e-9):
            lift_attempts += (("up-and-back", fallback_lift),)

        lift_failures: list[tuple[str, str]] = []
        lift_joints = None
        lift_solution = None
        for lift_name, lift in lift_attempts:
            try:
                candidate_lift_joints, candidate_lift_solution = self._cartesian_ik(
                    _interpolate_pose(grasp, lift, self.config.lift_steps),
                    grasp_solution.joints,
                    control,
                )
            except PlanningAborted:
                raise
            except IKFailure as error:
                lift_failures.append(("ik", f"lift IK failed: {lift_name}: {error}"))
                continue

            lift_path = np.vstack((grasp_solution.joints, candidate_lift_joints))
            lift_valid = True
            for first, second in zip(lift_path, lift_path[1:]):
                checkpoint(control, "grasp lift collision checking")
                if self.lift_segment_valid is None:
                    valid = self._segment_valid(first, second, control=control)
                else:
                    width_kwargs = (
                        {"required_width_m": width}
                        if self._lift_accepts_width
                        else {}
                    )
                    valid = self.lift_segment_valid(
                        first,
                        second,
                        grasp_solution.joints,
                        **width_kwargs,
                    )
                    checkpoint(control, "grasp lift collision checking")
                if not valid:
                    lift_valid = False
                    break
            if not lift_valid:
                lift_failures.append((
                    "lift_collision",
                    f"{lift_name} Cartesian lift intersects the planning scene",
                ))
                continue
            lift_joints = candidate_lift_joints
            lift_solution = candidate_lift_solution
            break

        if lift_joints is None or lift_solution is None:
            stage = (
                "lift_collision"
                if lift_failures and all(item[0] == "lift_collision" for item in lift_failures)
                else "ik"
            )
            raise _HypothesisRejected(
                stage,
                "; ".join(reason for _kind, reason in lift_failures),
            )

        try:
            transit = self._plan_joint(
                current,
                pregrasp_solution.joints,
                control=control,
            )
        except PlanningAborted:
            raise
        except PlanningError as error:
            raise _HypothesisRejected("planning", str(error)) from error

        path = np.asarray(transit.waypoints, dtype=float)
        path_cost = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        solution_score = (
            candidate_score
            + 0.02 * np.log1p(max(0.0, grasp_solution.manipulability))
            + 0.01 * np.log1p(max(0.0, lift_solution.manipulability))
            - 0.005 * path_cost
            - self.config.joint_limit_penalty / (
                min(
                    pregrasp_solution.min_joint_limit_margin,
                    grasp_solution.min_joint_limit_margin,
                    lift_solution.min_joint_limit_margin,
                ) + 1e-3
            )
        )
        return solution_score, (
            candidate_index,
            symmetry_index,
            grasp.copy(),
            pregrasp,
            lift.copy(),
            transit,
            approach_joints,
            lift_joints,
            width,
        )

    def plan(
        self,
        candidates: GraspCandidates,
        *,
        current_joints: object,
        pose_ranker: Callable[..., float] | None = None,
        control: PlanningControl | None = None,
    ) -> PlannedGrasp:
        checkpoint(control, "grasp candidate planning")
        grasps = np.asarray(candidates.grasps, dtype=float)
        scores = np.asarray(candidates.scores, dtype=float)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
            raise PlanningError(f"grasp array must have shape (N, 4, 4), got {grasps.shape}")
        if not np.all(np.isfinite(grasps)):
            raise PlanningError("grasp poses must be finite")
        if scores.shape != (len(grasps),) or not np.all(np.isfinite(scores)):
            raise PlanningError("grasp scores do not align with grasp poses")
        scores, lateral_bonuses, overhead_penalties = lateral_approach_scores(
            grasps,
            scores,
            lateral_weight=self.config.lateral_approach_prior_weight,
            overhead_penalty_weight=(
                self.config.overhead_approach_penalty_weight
            ),
        )
        widths = None if candidates.widths is None else np.asarray(candidates.widths, dtype=float)
        if widths is not None and (
            widths.shape != (len(grasps),) or not np.all(np.isfinite(widths))
        ):
            raise PlanningError("grasp widths do not align with grasp poses")
        current = np.asarray(current_joints, dtype=float)
        if current.ndim != 1 or not np.all(np.isfinite(current)):
            raise PlanningError("current joints must be a finite vector")

        failures: list[CandidateFailure] = []
        feasible: list[tuple[float, tuple[object, ...]]] = []
        parent_control = control or PlanningControl()
        search_control = parent_control.limited_to(
            self.config.search_timeout_s,
            "grasp candidate search budget",
        )
        refinement_control: PlanningControl | None = None
        hypotheses_evaluated = 0
        stop_search = False
        diverse_order = directionally_diverse_indices(
            grasps,
            scores,
            self.config.max_candidates,
        )
        # Candidate diversity decides which bounded subset receives expensive
        # exact checks; it must not decide which member of that subset wins.
        # The former interleaved order could test rank 52 before rank 2 and,
        # with max_feasible_plans=1, return that low-ranked grasp immediately.
        # Evaluate the retained subset in global score order.  Reachability is
        # still used below to order equivalent gripper symmetries.
        order = np.asarray(sorted(
            (int(index) for index in diverse_order),
            key=lambda index: (-float(scores[index]), index),
        ), dtype=int)
        global_score_order = np.argsort(-scores, kind="stable")
        global_ranks = np.empty(len(global_score_order), dtype=int)
        global_ranks[global_score_order] = np.arange(1, len(global_score_order) + 1)
        lift_direction = np.asarray(self.config.lift_direction_base, dtype=float)
        lift_norm = float(np.linalg.norm(lift_direction))
        if lift_norm < 1e-9:
            raise PlanningError("lift direction must be nonzero")
        lift_direction /= lift_norm

        order, reachability_costs = self._reachability_order(
            grasps,
            order,
            pose_ranker,
            search_control,
        )

        for candidate_index in order:
            checkpoint(control, "grasp candidate search")
            width = None if widths is None else float(widths[candidate_index])
            if width is not None and not self.config.min_width_m <= width <= self.config.max_width_m:
                failures.append(CandidateFailure(
                    int(candidate_index), None, "aperture",
                    f"required width {width:.4f} m is outside configured range",
                ))
                continue
            raw_grasp = grasps[candidate_index]
            try:
                family = expand_symmetry(raw_grasp, n_about_axis=self.config.symmetry_samples)
            except ValueError as error:
                failures.append(CandidateFailure(
                    int(candidate_index), None, "geometry", str(error),
                ))
                continue
            indexed_family = list(enumerate(family))
            if hypotheses_evaluated >= self.config.max_hypotheses:
                stop_search = True
                break
            if pose_ranker is not None:
                symmetry_costs: list[tuple[bool, float, int]] = []
                ranking_control = refinement_control or search_control
                for symmetry_index, grasp in indexed_family:
                    cached_cost = reachability_costs.get((
                        int(candidate_index),
                        symmetry_index,
                    ))
                    if cached_cost is None:
                        try:
                            target = self._tip_pose(grasp_pregrasp_pose(
                                grasp,
                                self.config.pregrasp_distance_m,
                            ))
                            cost = float(pose_ranker(target, control=ranking_control))
                        except PlanningDeadlineExceeded:
                            checkpoint(control, "grasp symmetry ranking")
                            failures.append(CandidateFailure(
                                int(candidate_index),
                                symmetry_index,
                                "budget",
                                "grasp symmetry ranking budget expired",
                            ))
                            stop_search = True
                            break
                        except PlanningAborted:
                            raise
                        except Exception:
                            cost = float("inf")
                    else:
                        cost = cached_cost
                    symmetry_costs.append((
                        not np.isfinite(cost),
                        cost if np.isfinite(cost) else 0.0,
                        symmetry_index,
                    ))
                if stop_search:
                    break
                indexed_family.sort(key=lambda item: symmetry_costs[item[0]])
            for symmetry_index, grasp in indexed_family:
                if hypotheses_evaluated >= self.config.max_hypotheses:
                    stop_search = True
                    break
                checkpoint(control, "grasp symmetry search")
                active_control = refinement_control or search_control
                try:
                    hypothesis_control = active_control.limited_to(
                        self.config.hypothesis_timeout_s,
                        "grasp hypothesis budget",
                    )
                except PlanningDeadlineExceeded:
                    checkpoint(control, "grasp hypothesis budget")
                    failures.append(CandidateFailure(
                        int(candidate_index),
                        symmetry_index,
                        "budget",
                        "grasp search/refinement budget expired",
                    ))
                    stop_search = True
                    break
                hypotheses_evaluated += 1
                try:
                    result = self._evaluate_hypothesis(
                        candidate_index=int(candidate_index),
                        symmetry_index=symmetry_index,
                        grasp=grasp,
                        width=width,
                        candidate_score=float(scores[candidate_index]),
                        current=current,
                        lift_direction=lift_direction,
                        control=hypothesis_control,
                    )
                except PlanningDeadlineExceeded as error:
                    checkpoint(control, "grasp hypothesis evaluation")
                    try:
                        checkpoint(search_control, "grasp candidate search budget")
                    except PlanningDeadlineExceeded:
                        failures.append(CandidateFailure(
                            int(candidate_index),
                            symmetry_index,
                            "budget",
                            "grasp candidate search budget expired",
                        ))
                        stop_search = True
                        break
                    if refinement_control is not None:
                        try:
                            checkpoint(
                                refinement_control,
                                "grasp solution refinement",
                            )
                        except PlanningDeadlineExceeded:
                            stop_search = True
                            break
                    failures.append(CandidateFailure(
                        int(candidate_index),
                        symmetry_index,
                        "budget",
                        str(error),
                    ))
                    continue
                except PlanningAborted:
                    raise
                except _HypothesisRejected as error:
                    failures.append(CandidateFailure(
                        int(candidate_index), symmetry_index, error.stage, str(error),
                    ))
                    continue
                feasible.append(result)
                if len(feasible) >= self.config.max_feasible_plans:
                    stop_search = True
                    break
                if refinement_control is None:
                    try:
                        refinement_control = search_control.limited_to(
                            self.config.solution_refinement_timeout_s,
                            "grasp solution refinement",
                        )
                    except PlanningDeadlineExceeded:
                        # The first complete plan remains usable when only the
                        # optional local improvement budget has elapsed.
                        checkpoint(control, "grasp solution refinement")
                        stop_search = True
                        break
            if stop_search:
                break
        checkpoint(control, "grasp solution selection")
        if feasible:
            # Candidate score is the user-visible global rank and remains the
            # primary selection contract.  Motion quality only breaks ties;
            # otherwise a lower-ranked complete plan could silently displace
            # the highest-ranked non-rejected grasp during refinement.
            solution_score, data = max(
                feasible,
                key=lambda item: (
                    float(scores[int(item[1][0])]),
                    item[0],
                ),
            )
            (
                candidate_index, symmetry_index, grasp, pregrasp, lift, transit,
                approach_joints, lift_joints, width,
            ) = data
            return PlannedGrasp(
                candidate_index=candidate_index,
                symmetry_index=symmetry_index,
                grasp_pose=grasp,
                pregrasp_pose=pregrasp,
                transit=transit,
                approach_joints=approach_joints,
                lift_joints=lift_joints,
                required_width_m=width,
                score=solution_score,
                failures=tuple(failures),
                selected_global_rank=int(global_ranks[candidate_index]),
                higher_rank_rejection_count=len({
                    failure.candidate_index
                    for failure in failures
                    if global_ranks[failure.candidate_index]
                    < global_ranks[candidate_index]
                }),
                lift_pose=lift,
                lateral_approach_bonus=float(lateral_bonuses[candidate_index]),
                overhead_approach_penalty=float(
                    overhead_penalties[candidate_index]
                ),
            )
        raise GraspPlanningError(
            "no grasp candidate survived aperture/IK/planning; "
            + _failure_summary(failures),
            tuple(failures),
        )
