"""Bounded recovery state machine for a complete mobile manipulation task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class Stage(str, Enum):
    SEARCH = "search"
    COARSE_NAV = "coarse_nav"
    VISUAL_APPROACH = "visual_approach"
    OBSERVE_GRASP = "observe_grasp"
    PLAN_GRASP = "plan_grasp"
    EXECUTE_GRASP = "execute_grasp"
    VERIFY_GRASP = "verify_grasp"
    CARRY = "carry"
    PLAN_PLACE = "plan_place"
    EXECUTE_PLACE = "execute_place"
    COMPLETE = "complete"
    FAILED = "failed"


class FailureKind(str, Enum):
    NOT_FOUND = "not_found"
    TARGET_LOST = "target_lost"
    NAV_BLOCKED = "nav_blocked"
    VISUAL_APPROACH_FAILED = "visual_approach_failed"
    NO_GRASP = "no_grasp"
    IK_UNREACHABLE = "ik_unreachable"
    PLAN_BLOCKED = "plan_blocked"
    EXECUTION_FAILED = "execution_failed"
    EMPTY_GRASP = "empty_grasp"
    VERIFY_FAILED = "verify_failed"
    PLACE_BLOCKED = "place_blocked"
    RELEASE_FAILED = "release_failed"
    POSTURE_UNSAFE = "posture_unsafe"
    FATAL = "fatal"


@dataclass(frozen=True)
class RetryBudget:
    search_misses: int = 2
    tracker_reacquisitions: int = 2
    nav_replans: int = 1
    ik_restandoffs: int = 3
    plan_candidates: int = 5
    grasp_attempts: int = 3
    place_replans: int = 2
    release_attempts: int = 2

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("retry budgets cannot be negative")


@dataclass(frozen=True)
class StageResult:
    ok: bool
    failure_kind: FailureKind | None = None
    detail: str = ""
    payload: object | None = None

    @classmethod
    def success(cls, payload: object | None = None) -> "StageResult":
        return cls(True, payload=payload)

    @classmethod
    def failure(cls, kind: FailureKind, detail: str = "") -> "StageResult":
        return cls(False, failure_kind=kind, detail=detail)

    def __post_init__(self) -> None:
        if self.ok == (self.failure_kind is not None):
            raise ValueError("success must omit failure_kind and failure must provide it")


@dataclass(frozen=True)
class Transition:
    previous: Stage
    current: Stage
    reason: str
    counters: Mapping[str, int]


_SUCCESSOR = {
    Stage.SEARCH: Stage.COARSE_NAV,
    Stage.COARSE_NAV: Stage.VISUAL_APPROACH,
    Stage.VISUAL_APPROACH: Stage.OBSERVE_GRASP,
    Stage.OBSERVE_GRASP: Stage.PLAN_GRASP,
    Stage.PLAN_GRASP: Stage.EXECUTE_GRASP,
    Stage.EXECUTE_GRASP: Stage.VERIFY_GRASP,
    Stage.VERIFY_GRASP: Stage.CARRY,
    Stage.CARRY: Stage.PLAN_PLACE,
    Stage.PLAN_PLACE: Stage.EXECUTE_PLACE,
    Stage.EXECUTE_PLACE: Stage.COMPLETE,
}


class MobileManipulationStateMachine:
    """Apply stage outcomes with explicit, finite recovery budgets.

    This class contains policy but no platform calls. ROS actions, perception,
    planners and hardware adapters report typed outcomes; the same recovery
    behavior therefore runs in simulation and on the real robot.
    """

    def __init__(self, budget: RetryBudget | None = None):
        self.budget = budget or RetryBudget()
        self.stage = Stage.SEARCH
        self.failure_reason = ""
        self.context: dict[str, object] = {}
        self.counters = {
            "search_misses": 0,
            "tracker_reacquisitions": 0,
            "nav_replans": 0,
            "ik_restandoffs": 0,
            "plan_candidates": 0,
            "grasp_attempts": 0,
            "place_replans": 0,
            "release_attempts": 0,
        }

    @property
    def terminal(self) -> bool:
        return self.stage in (Stage.COMPLETE, Stage.FAILED)

    def _transition(self, previous: Stage, current: Stage, reason: str = "") -> Transition:
        self.stage = current
        if current == Stage.FAILED:
            self.failure_reason = reason or "task failed"
        return Transition(previous, current, reason, dict(self.counters))

    def _increment_or_fail(
        self,
        previous: Stage,
        counter: str,
        limit: int,
        recovery: Stage,
        exhausted: str,
    ) -> Transition:
        self.counters[counter] += 1
        if self.counters[counter] > limit:
            return self._transition(previous, Stage.FAILED, exhausted)
        return self._transition(previous, recovery, f"recovering from {counter}")

    def apply(self, result: StageResult) -> Transition:
        if self.terminal:
            raise RuntimeError(f"cannot apply an event to terminal stage {self.stage.value}")
        previous = self.stage
        if result.ok:
            if result.payload is not None:
                self.context[previous.value] = result.payload
            try:
                next_stage = _SUCCESSOR[previous]
            except KeyError as error:
                raise RuntimeError(f"stage {previous.value} has no success transition") from error
            if previous == Stage.PLAN_GRASP:
                self.counters["plan_candidates"] = 0
            return self._transition(previous, next_stage)

        kind = result.failure_kind
        assert kind is not None
        if kind in (FailureKind.POSTURE_UNSAFE, FailureKind.FATAL):
            return self._transition(
                previous,
                Stage.FAILED,
                result.detail or kind.value,
            )
        if kind == FailureKind.NOT_FOUND:
            return self._increment_or_fail(
                previous,
                "search_misses",
                self.budget.search_misses,
                Stage.SEARCH,
                "target not found within search budget",
            )
        if kind == FailureKind.TARGET_LOST:
            return self._increment_or_fail(
                previous,
                "tracker_reacquisitions",
                self.budget.tracker_reacquisitions,
                Stage.SEARCH,
                "persistent target tracking could not be reacquired",
            )
        if kind in (FailureKind.NAV_BLOCKED, FailureKind.VISUAL_APPROACH_FAILED):
            return self._increment_or_fail(
                previous,
                "nav_replans",
                self.budget.nav_replans,
                Stage.COARSE_NAV,
                "navigation/visual approach budget exhausted",
            )
        if kind in (FailureKind.NO_GRASP, FailureKind.IK_UNREACHABLE):
            self.counters["plan_candidates"] = 0
            return self._increment_or_fail(
                previous,
                "ik_restandoffs",
                self.budget.ik_restandoffs,
                Stage.VISUAL_APPROACH,
                "IK re-standoff budget exhausted",
            )
        if kind == FailureKind.PLAN_BLOCKED:
            self.counters["plan_candidates"] += 1
            if self.counters["plan_candidates"] <= self.budget.plan_candidates:
                return self._transition(previous, Stage.PLAN_GRASP, "trying next grasp candidate")
            self.counters["plan_candidates"] = 0
            return self._increment_or_fail(
                previous,
                "ik_restandoffs",
                self.budget.ik_restandoffs,
                Stage.VISUAL_APPROACH,
                "motion planning re-standoff budget exhausted",
            )
        if kind in (
            FailureKind.EXECUTION_FAILED,
            FailureKind.EMPTY_GRASP,
            FailureKind.VERIFY_FAILED,
        ):
            return self._increment_or_fail(
                previous,
                "grasp_attempts",
                self.budget.grasp_attempts,
                Stage.OBSERVE_GRASP,
                "whole-task grasp attempts exhausted",
            )
        if kind == FailureKind.PLACE_BLOCKED:
            return self._increment_or_fail(
                previous,
                "place_replans",
                self.budget.place_replans,
                Stage.PLAN_PLACE,
                "place planning budget exhausted",
            )
        if kind == FailureKind.RELEASE_FAILED:
            return self._increment_or_fail(
                previous,
                "release_attempts",
                self.budget.release_attempts,
                Stage.EXECUTE_PLACE,
                "object release budget exhausted",
            )
        return self._transition(previous, Stage.FAILED, result.detail or kind.value)
