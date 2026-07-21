"""Bounded, observation-only SE(2) work-pose optimization.

The optimizer operates entirely in numpy and has no ROS dependency.  Target,
grasp, and scene geometry are captured in the current arm-base frame.  A
candidate mobile-base motion is applied through the configured rigid arm mount,
so every downstream observation is predicted in the future arm-base frame
without consulting simulator or world ground truth.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
import inspect
import math

import numpy as np

from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.planner import PlanningError
from z_manip.planning_control import (
    PlanningAborted,
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)


class WorkPoseFailureCode(str, Enum):
    """Stable machine-readable reasons for rejecting one work-pose sample."""

    MOTION_LIMIT = "motion_limit"
    OUTSIDE_MANIP_CORRIDOR = "outside_manip_corridor"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    HISTORY_DUPLICATE = "history_duplicate"
    EXACT_REJECTED = "exact_rejected"
    EXACT_TIMEOUT = "exact_timeout"
    INVALID_EXACT_RESULT = "invalid_exact_result"


@dataclass(frozen=True)
class WorkPoseFailure:
    code: WorkPoseFailureCode
    stage: str
    reason: str
    relative_base_pose: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class WorkPoseDiagnostics:
    """Bounded evidence from geometric filtering and exact evaluation."""

    sampled_hypotheses: int
    geometric_candidates: int
    ranked_candidates: int
    exact_evaluations: int
    feasible_candidates: int
    rejection_counts: tuple[tuple[WorkPoseFailureCode, int], ...]
    failures: tuple[WorkPoseFailure, ...]
    sample_budget_exhausted: bool
    exact_budget_exhausted: bool

    def rejection_count(self, code: WorkPoseFailureCode) -> int:
        return dict(self.rejection_counts).get(code, 0)


class WorkPoseOptimizationError(PlanningError):
    """No bounded candidate survived, with typed diagnostic evidence."""

    def __init__(self, message: str, diagnostics: WorkPoseDiagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, eq=False)
class WorkPoseObservation:
    """One synchronized observation expressed in the current arm base.

    ``T_platform_piper`` maps arm-base coordinates into the mobile platform
    base.  This explicit convention matters when the arm mount has translation
    or yaw: planar base motion cannot be subtracted directly from arm-frame
    grasps.
    """

    target_pose: np.ndarray
    candidates: GraspCandidates
    scene_points: np.ndarray
    current_joints: np.ndarray
    T_platform_piper: np.ndarray


@dataclass(frozen=True, eq=False)
class WorkPoseCandidate:
    """Materialized future-arm observation for one relative base pose."""

    relative_base_pose: np.ndarray
    predicted_target_pose: np.ndarray
    candidates: GraspCandidates
    scene_points: np.ndarray
    current_joints: np.ndarray
    T_new_piper_current_piper: np.ndarray
    cheap_score: float


@dataclass(frozen=True, eq=False)
class WorkPoseChoice:
    """Selected base work pose and the observations predicted at that pose."""

    relative_base_pose: np.ndarray
    predicted_target_pose: np.ndarray
    candidates: GraspCandidates
    scene_points: np.ndarray
    current_joints: np.ndarray
    T_new_piper_current_piper: np.ndarray
    cheap_score: float
    exact_evaluation: object | None
    score: float
    diagnostics: WorkPoseDiagnostics


@dataclass(frozen=True)
class WorkPoseConfig:
    """Sampling, manipulator-corridor, ranking, and search budgets.

    Radial and lateral samples describe the desired observed target position in
    the *future arm-base frame*.  Candidate platform yaw is the current target
    bearing plus a configured offset.  All values are robot configuration, not
    per-object coordinates.
    """

    radial_distances_m: tuple[float, ...] = (0.52, 0.42, 0.62)
    target_lateral_offsets_m: tuple[float, ...] = (0.0, -0.18, 0.18)
    yaw_offsets_rad: tuple[float, ...] = (
        0.0,
        -math.pi / 12.0,
        math.pi / 12.0,
        -math.pi / 6.0,
        math.pi / 6.0,
    )
    corridor_min_x_m: float = 0.30
    corridor_max_x_m: float = 0.75
    corridor_min_y_m: float = -0.32
    corridor_max_y_m: float = 0.32
    corridor_min_z_m: float = -0.35
    corridor_max_z_m: float = 1.25
    preferred_target_x_m: float = 0.52
    preferred_target_y_m: float = 0.0
    max_base_translation_m: float = 1.45
    max_abs_base_yaw_rad: float = math.pi * 0.75
    target_alignment_weight: float = 1.0
    translation_penalty: float = 0.12
    yaw_penalty: float = 0.08
    exact_score_weight: float = 1.0
    max_sampled_hypotheses: int = 96
    max_ranked_candidates: int = 24
    max_exact_evaluations: int = 16
    max_feasible_choices: int = 2
    search_timeout_s: float = 8.0
    hypothesis_timeout_s: float = 2.5
    dedupe_translation_m: float = 0.04
    dedupe_yaw_rad: float = math.radians(3.0)
    max_diagnostic_failures: int = 12

    def __post_init__(self) -> None:
        samples = (
            self.radial_distances_m,
            self.target_lateral_offsets_m,
            self.yaw_offsets_rad,
        )
        if any(not values for values in samples):
            raise ValueError("work-pose sample sets cannot be empty")
        if any(
            not np.isfinite(value)
            for values in samples
            for value in values
        ):
            raise ValueError("work-pose samples must be finite")
        if any(value <= 0.0 for value in self.radial_distances_m):
            raise ValueError("work-pose radial distances must be positive")
        if not self.corridor_min_x_m < self.corridor_max_x_m:
            raise ValueError("invalid manipulator corridor x interval")
        if not self.corridor_min_y_m < self.corridor_max_y_m:
            raise ValueError("invalid manipulator corridor y interval")
        if not self.corridor_min_z_m < self.corridor_max_z_m:
            raise ValueError("invalid manipulator corridor z interval")
        if not (
            self.corridor_min_x_m
            <= self.preferred_target_x_m
            <= self.corridor_max_x_m
            and self.corridor_min_y_m
            <= self.preferred_target_y_m
            <= self.corridor_max_y_m
        ):
            raise ValueError("preferred target must lie inside manipulator corridor")
        positive_finite = (
            self.max_base_translation_m,
            self.max_abs_base_yaw_rad,
            self.search_timeout_s,
            self.hypothesis_timeout_s,
            self.dedupe_translation_m,
            self.dedupe_yaw_rad,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive_finite):
            raise ValueError("work-pose limits and timeouts must be finite and positive")
        weights = (
            self.target_alignment_weight,
            self.translation_penalty,
            self.yaw_penalty,
            self.exact_score_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("work-pose score weights must be finite and non-negative")
        counts = (
            self.max_sampled_hypotheses,
            self.max_ranked_candidates,
            self.max_exact_evaluations,
            self.max_feasible_choices,
            self.max_diagnostic_failures,
        )
        if any(value < 1 for value in counts):
            raise ValueError("work-pose candidate and diagnostic budgets must be positive")
        if self.max_ranked_candidates > self.max_sampled_hypotheses:
            raise ValueError("ranked work-pose budget cannot exceed sample budget")
        if self.max_exact_evaluations > self.max_ranked_candidates:
            raise ValueError("exact work-pose budget cannot exceed ranked budget")
        if self.max_feasible_choices > self.max_exact_evaluations:
            raise ValueError("feasible-choice budget cannot exceed exact-evaluation budget")


@dataclass(frozen=True, eq=False)
class _WorkPoseProposal:
    relative_base_pose: np.ndarray
    predicted_target_pose: np.ndarray
    T_new_piper_current_piper: np.ndarray
    cheap_score: float


class _DiagnosticsBuilder:
    def __init__(self, max_failures: int):
        self.sampled_hypotheses = 0
        self.geometric_candidates = 0
        self.ranked_candidates = 0
        self.exact_evaluations = 0
        self.feasible_candidates = 0
        self.sample_budget_exhausted = False
        self.exact_budget_exhausted = False
        self._counts: Counter[WorkPoseFailureCode] = Counter()
        self._failures: list[WorkPoseFailure] = []
        self._max_failures = max_failures

    def reject(
        self,
        code: WorkPoseFailureCode,
        stage: str,
        reason: str,
        pose: np.ndarray | None = None,
    ) -> None:
        self._counts[code] += 1
        if len(self._failures) < self._max_failures:
            frozen_pose = None if pose is None else tuple(float(v) for v in pose)
            self._failures.append(WorkPoseFailure(code, stage, reason, frozen_pose))

    def freeze(self) -> WorkPoseDiagnostics:
        return WorkPoseDiagnostics(
            sampled_hypotheses=self.sampled_hypotheses,
            geometric_candidates=self.geometric_candidates,
            ranked_candidates=self.ranked_candidates,
            exact_evaluations=self.exact_evaluations,
            feasible_candidates=self.feasible_candidates,
            rejection_counts=tuple(
                (code, self._counts[code])
                for code in WorkPoseFailureCode
                if self._counts[code]
            ),
            failures=tuple(self._failures),
            sample_budget_exhausted=self.sample_budget_exhausted,
            exact_budget_exhausted=self.exact_budget_exhausted,
        )


def _rigid_transform(value: object, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{label} must have a homogeneous final row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{label} rotation must be right-handed")
    return transform.copy()


def _validate_observation(observation: WorkPoseObservation) -> WorkPoseObservation:
    target = _rigid_transform(observation.target_pose, "target_pose")
    mount = _rigid_transform(observation.T_platform_piper, "T_platform_piper")
    grasps = np.asarray(observation.candidates.grasps, dtype=float)
    scores = np.asarray(observation.candidates.scores, dtype=float)
    centroid = np.asarray(observation.candidates.centroid, dtype=float)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(grasps) < 1:
        raise ValueError("work-pose candidates must contain at least one 4x4 grasp")
    if not np.all(np.isfinite(grasps)):
        raise ValueError("work-pose grasps must be finite")
    if not np.allclose(grasps[:, 3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError("work-pose grasps must have homogeneous final rows")
    if scores.shape != (len(grasps),) or not np.all(np.isfinite(scores)):
        raise ValueError("work-pose scores must align with grasps and be finite")
    if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
        raise ValueError("work-pose centroid must be a finite 3-vector")
    widths = observation.candidates.widths
    if widths is not None:
        width_values = np.asarray(widths, dtype=float)
        if width_values.shape != (len(grasps),) or not np.all(np.isfinite(width_values)):
            raise ValueError("work-pose widths must align with grasps and be finite")
    else:
        width_values = None
    if not observation.candidates.frame:
        raise ValueError("work-pose candidate frame cannot be empty")
    scene = np.asarray(observation.scene_points, dtype=float)
    if scene.ndim != 2 or scene.shape[1] < 3:
        raise ValueError("work-pose scene must be an (N, >=3) array")
    if not np.all(np.isfinite(scene[:, :3])):
        raise ValueError("work-pose scene XYZ must be finite")
    joints = np.asarray(observation.current_joints, dtype=float)
    if joints.ndim != 1 or len(joints) < 1 or not np.all(np.isfinite(joints)):
        raise ValueError("current_joints must be a non-empty finite vector")
    return WorkPoseObservation(
        target_pose=target,
        candidates=GraspCandidates(
            grasps=grasps.copy(),
            scores=scores.copy(),
            centroid=centroid.copy(),
            frame=observation.candidates.frame,
            num_raw=int(observation.candidates.num_raw),
            widths=None if width_values is None else width_values.copy(),
        ),
        scene_points=scene.copy(),
        current_joints=joints.copy(),
        T_platform_piper=mount,
    )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _se2_transform(pose: np.ndarray) -> np.ndarray:
    x, y, yaw = (float(value) for value in pose)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array((
        (cosine, -sine, 0.0, x),
        (sine, cosine, 0.0, y),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))


def _sample_indices(shape: tuple[int, int, int]) -> Iterator[tuple[int, int, int]]:
    """Traverse preference-ordered sample axes without starving one axis."""

    for rank_sum in range(sum(size - 1 for size in shape) + 1):
        for yaw_rank in range(shape[0]):
            for radial_rank in range(shape[1]):
                lateral_rank = rank_sum - yaw_rank - radial_rank
                if 0 <= lateral_rank < shape[2]:
                    yield yaw_rank, radial_rank, lateral_rank


def _pose_is_duplicate(
    pose: np.ndarray,
    others: Iterable[np.ndarray],
    translation_tolerance: float,
    yaw_tolerance: float,
) -> bool:
    for other in others:
        if (
            np.linalg.norm(pose[:2] - other[:2]) <= translation_tolerance
            and abs(_wrap_angle(float(pose[2] - other[2]))) <= yaw_tolerance
        ):
            return True
    return False


def _history_poses(history: Iterable[object]) -> tuple[np.ndarray, ...]:
    result = []
    for index, value in enumerate(history):
        pose = np.asarray(value, dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"work-pose history entry {index} must be a finite SE(2) vector")
        result.append(pose.copy())
    return tuple(result)


def _evaluator_control_mode(callback: Callable[..., object]) -> str:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return "legacy"
    sentinel = object()
    try:
        signature.bind(sentinel, control=sentinel)
    except TypeError:
        try:
            signature.bind(sentinel, sentinel)
        except TypeError:
            try:
                signature.bind(sentinel)
            except TypeError as error:
                raise TypeError(
                    "work-pose evaluator must accept a candidate and optional control",
                ) from error
            return "legacy"
        return "positional"
    return "keyword"


def _evaluate(
    callback: Callable[..., object],
    candidate: WorkPoseCandidate,
    control: PlanningControl,
    mode: str,
) -> object:
    if mode == "keyword":
        return callback(candidate, control=control)
    if mode == "positional":
        return callback(candidate, control)
    return callback(candidate)


def _exact_score(evaluation: object) -> float:
    if isinstance(evaluation, Mapping):
        if evaluation.get("feasible") is False:
            raise PlanningError(str(evaluation.get("reason", "exact evaluation rejected pose")))
        score = evaluation.get("score")
    else:
        if getattr(evaluation, "feasible", True) is False:
            raise PlanningError(str(getattr(evaluation, "reason", "exact evaluation rejected pose")))
        score = getattr(evaluation, "score", None)
    try:
        numeric_score = float(score)
    except (TypeError, ValueError, OverflowError) as error:
        raise PlanningError(
            "work-pose exact evaluator returned no finite score",
        ) from error
    if not np.isfinite(numeric_score):
        raise PlanningError("work-pose exact evaluator returned no finite score")
    return numeric_score


class BoundedSE2WorkPoseOptimizer:
    """Rank bounded base poses, then validate the best with exact planning."""

    def __init__(self, config: WorkPoseConfig | None = None):
        self.config = config or WorkPoseConfig()

    def _proposal(
        self,
        observation: WorkPoseObservation,
        target_platform: np.ndarray,
        bearing: float,
        yaw_offset: float,
        radial: float,
        lateral: float,
    ) -> _WorkPoseProposal:
        mount = observation.T_platform_piper
        desired_arm = np.array((radial, lateral, observation.target_pose[2, 3], 1.0))
        vertical_scale = float(mount[2, 2])
        if abs(vertical_scale) > 1e-6:
            desired_arm[2] = (
                target_platform[2]
                - mount[2, 3]
                - mount[2, 0] * radial
                - mount[2, 1] * lateral
            ) / vertical_scale
        desired_platform = mount @ desired_arm
        yaw = _wrap_angle(bearing + yaw_offset)
        rotation = np.array((
            (math.cos(yaw), -math.sin(yaw)),
            (math.sin(yaw), math.cos(yaw)),
        ))
        translation = target_platform[:2] - rotation @ desired_platform[:2]
        relative_pose = np.array((translation[0], translation[1], yaw))
        base_transform = _se2_transform(relative_pose)
        new_arm_from_current_arm = np.linalg.inv(base_transform @ mount) @ mount
        predicted_target = new_arm_from_current_arm @ observation.target_pose

        x_width = self.config.corridor_max_x_m - self.config.corridor_min_x_m
        y_width = self.config.corridor_max_y_m - self.config.corridor_min_y_m
        x_error = (predicted_target[0, 3] - self.config.preferred_target_x_m) / x_width
        y_error = (predicted_target[1, 3] - self.config.preferred_target_y_m) / y_width
        alignment_cost = self.config.target_alignment_weight * (
            x_error * x_error + y_error * y_error
        )
        motion_cost = (
            self.config.translation_penalty
            * float(np.linalg.norm(relative_pose[:2]))
            / self.config.max_base_translation_m
            + self.config.yaw_penalty
            * abs(yaw)
            / self.config.max_abs_base_yaw_rad
        )
        return _WorkPoseProposal(
            relative_base_pose=relative_pose,
            predicted_target_pose=predicted_target,
            T_new_piper_current_piper=new_arm_from_current_arm,
            cheap_score=-(alignment_cost + motion_cost),
        )

    def _materialize(
        self,
        observation: WorkPoseObservation,
        proposal: _WorkPoseProposal,
    ) -> WorkPoseCandidate:
        transform = proposal.T_new_piper_current_piper
        original = observation.candidates
        grasps = np.asarray(original.grasps, dtype=float)
        transformed_grasps = np.einsum("ij,njk->nik", transform, grasps)
        centroid_h = np.append(np.asarray(original.centroid, dtype=float), 1.0)
        transformed_centroid = (transform @ centroid_h)[:3]
        scene = np.asarray(observation.scene_points, dtype=float).copy()
        if len(scene):
            scene[:, :3] = (
                scene[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            )
        transformed_candidates = GraspCandidates(
            grasps=transformed_grasps,
            scores=np.asarray(original.scores, dtype=float).copy(),
            centroid=transformed_centroid,
            frame=original.frame,
            num_raw=original.num_raw,
            widths=(
                None
                if original.widths is None
                else np.asarray(original.widths, dtype=float).copy()
            ),
        )
        return WorkPoseCandidate(
            relative_base_pose=proposal.relative_base_pose.copy(),
            predicted_target_pose=proposal.predicted_target_pose.copy(),
            candidates=transformed_candidates,
            scene_points=scene,
            current_joints=observation.current_joints.copy(),
            T_new_piper_current_piper=transform.copy(),
            cheap_score=proposal.cheap_score,
        )

    def _choice(
        self,
        candidate: WorkPoseCandidate,
        evaluation: object | None,
        score: float,
        diagnostics: WorkPoseDiagnostics,
    ) -> WorkPoseChoice:
        return WorkPoseChoice(
            relative_base_pose=candidate.relative_base_pose,
            predicted_target_pose=candidate.predicted_target_pose,
            candidates=candidate.candidates,
            scene_points=candidate.scene_points,
            current_joints=candidate.current_joints,
            T_new_piper_current_piper=candidate.T_new_piper_current_piper,
            cheap_score=candidate.cheap_score,
            exact_evaluation=evaluation,
            score=score,
            diagnostics=diagnostics,
        )

    def select(
        self,
        observation: WorkPoseObservation,
        *,
        evaluate: Callable[..., object] | None = None,
        history_relative_base_poses: Iterable[object] = (),
        control: PlanningControl | None = None,
    ) -> WorkPoseChoice:
        """Select a future mobile-base pose using only the current observation."""

        checkpoint(control, "work-pose optimization")
        observed = _validate_observation(observation)
        history = _history_poses(history_relative_base_poses)
        diagnostics = _DiagnosticsBuilder(self.config.max_diagnostic_failures)
        target_platform = observed.T_platform_piper @ observed.target_pose[:, 3]
        horizontal_range = float(np.linalg.norm(target_platform[:2]))
        if horizontal_range < 1e-6:
            raise ValueError("observed target has no stable platform-frame bearing")
        bearing = math.atan2(float(target_platform[1]), float(target_platform[0]))
        proposals: list[_WorkPoseProposal] = []
        accepted_poses: list[np.ndarray] = []
        sample_shape = (
            len(self.config.yaw_offsets_rad),
            len(self.config.radial_distances_m),
            len(self.config.target_lateral_offsets_m),
        )
        total_sample_count = math.prod(sample_shape)
        for yaw_rank, radial_rank, lateral_rank in _sample_indices(sample_shape):
            if diagnostics.sampled_hypotheses >= self.config.max_sampled_hypotheses:
                diagnostics.sample_budget_exhausted = (
                    total_sample_count > diagnostics.sampled_hypotheses
                )
                break
            checkpoint(control, "work-pose geometric sampling")
            diagnostics.sampled_hypotheses += 1
            proposal = self._proposal(
                observed,
                target_platform,
                bearing,
                self.config.yaw_offsets_rad[yaw_rank],
                self.config.radial_distances_m[radial_rank],
                self.config.target_lateral_offsets_m[lateral_rank],
            )
            pose = proposal.relative_base_pose
            translation = float(np.linalg.norm(pose[:2]))
            if (
                translation > self.config.max_base_translation_m
                or abs(float(pose[2])) > self.config.max_abs_base_yaw_rad
            ):
                diagnostics.reject(
                    WorkPoseFailureCode.MOTION_LIMIT,
                    "geometric_filter",
                    f"translation={translation:.3f}m yaw={pose[2]:.3f}rad exceeds bounds",
                    pose,
                )
                continue
            target = proposal.predicted_target_pose[:3, 3]
            if not (
                self.config.corridor_min_x_m <= target[0] <= self.config.corridor_max_x_m
                and self.config.corridor_min_y_m <= target[1] <= self.config.corridor_max_y_m
                and self.config.corridor_min_z_m <= target[2] <= self.config.corridor_max_z_m
            ):
                diagnostics.reject(
                    WorkPoseFailureCode.OUTSIDE_MANIP_CORRIDOR,
                    "geometric_filter",
                    "predicted target remains outside configured arm-base corridor",
                    pose,
                )
                continue
            if _pose_is_duplicate(
                pose,
                accepted_poses,
                self.config.dedupe_translation_m,
                self.config.dedupe_yaw_rad,
            ):
                diagnostics.reject(
                    WorkPoseFailureCode.DUPLICATE_CANDIDATE,
                    "deduplication",
                    "sample duplicates an already accepted bounded pose",
                    pose,
                )
                continue
            if _pose_is_duplicate(
                pose,
                history,
                self.config.dedupe_translation_m,
                self.config.dedupe_yaw_rad,
            ):
                diagnostics.reject(
                    WorkPoseFailureCode.HISTORY_DUPLICATE,
                    "deduplication",
                    "sample repeats a previously attempted work pose",
                    pose,
                )
                continue
            accepted_poses.append(pose)
            proposals.append(proposal)
            diagnostics.geometric_candidates += 1

        proposals.sort(
            key=lambda proposal: (
                -proposal.cheap_score,
                float(np.linalg.norm(proposal.relative_base_pose[:2])),
                abs(float(proposal.relative_base_pose[2])),
                tuple(float(value) for value in proposal.relative_base_pose),
            ),
        )
        proposals = proposals[:self.config.max_ranked_candidates]
        diagnostics.ranked_candidates = len(proposals)
        if not proposals:
            frozen = diagnostics.freeze()
            raise WorkPoseOptimizationError(
                "no bounded SE(2) sample placed the observed target in the manipulator corridor",
                frozen,
            )

        if evaluate is None:
            candidate = self._materialize(observed, proposals[0])
            diagnostics.feasible_candidates = 1
            frozen = diagnostics.freeze()
            return self._choice(
                candidate,
                None,
                candidate.cheap_score,
                frozen,
            )

        parent_control = control or PlanningControl()
        search_control = parent_control.limited_to(
            self.config.search_timeout_s,
            "work-pose search budget",
        )
        evaluator_mode = _evaluator_control_mode(evaluate)
        feasible: list[tuple[float, WorkPoseCandidate, object]] = []
        exact_proposals = proposals[:self.config.max_exact_evaluations]
        diagnostics.exact_budget_exhausted = len(proposals) > len(exact_proposals)
        for proposal in exact_proposals:
            candidate = self._materialize(observed, proposal)
            diagnostics.exact_evaluations += 1
            try:
                hypothesis_control = search_control.limited_to(
                    self.config.hypothesis_timeout_s,
                    "work-pose exact hypothesis budget",
                )
                checkpoint(hypothesis_control, "work-pose exact evaluation")
                evaluation = _evaluate(
                    evaluate,
                    candidate,
                    hypothesis_control,
                    evaluator_mode,
                )
                checkpoint(hypothesis_control, "work-pose exact evaluation")
                exact_score = _exact_score(evaluation)
            except PlanningCancelled:
                raise
            except PlanningDeadlineExceeded as error:
                checkpoint(control, "work-pose exact evaluation")
                diagnostics.reject(
                    WorkPoseFailureCode.EXACT_TIMEOUT,
                    "exact_evaluation",
                    str(error),
                    candidate.relative_base_pose,
                )
                continue
            except PlanningAborted:
                raise
            except PlanningError as error:
                code = (
                    WorkPoseFailureCode.INVALID_EXACT_RESULT
                    if "returned no finite score" in str(error)
                    else WorkPoseFailureCode.EXACT_REJECTED
                )
                diagnostics.reject(
                    code,
                    "exact_evaluation",
                    str(error),
                    candidate.relative_base_pose,
                )
                continue
            total_score = (
                candidate.cheap_score
                + self.config.exact_score_weight * exact_score
            )
            feasible.append((total_score, candidate, evaluation))
            diagnostics.feasible_candidates += 1
            if len(feasible) >= self.config.max_feasible_choices:
                break

        checkpoint(control, "work-pose solution selection")
        if not feasible:
            frozen = diagnostics.freeze()
            raise WorkPoseOptimizationError(
                "no ranked SE(2) work pose passed exact manipulator evaluation",
                frozen,
            )
        score, candidate, evaluation = max(feasible, key=lambda item: item[0])
        frozen = diagnostics.freeze()
        return self._choice(candidate, evaluation, score, frozen)


__all__ = [
    "BoundedSE2WorkPoseOptimizer",
    "WorkPoseCandidate",
    "WorkPoseChoice",
    "WorkPoseConfig",
    "WorkPoseDiagnostics",
    "WorkPoseFailure",
    "WorkPoseFailureCode",
    "WorkPoseObservation",
    "WorkPoseOptimizationError",
]
