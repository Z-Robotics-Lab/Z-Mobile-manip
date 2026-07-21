import pytest

from z_manip.orchestration.mobile_manipulation import (
    FailureKind,
    MobileManipulationStateMachine,
    RetryBudget,
    Stage,
    StageResult,
)


def test_happy_path_requires_every_sensor_plan_execute_and_verify_stage():
    machine = MobileManipulationStateMachine()
    expected = (
        Stage.COARSE_NAV,
        Stage.VISUAL_APPROACH,
        Stage.OBSERVE_GRASP,
        Stage.PLAN_GRASP,
        Stage.EXECUTE_GRASP,
        Stage.VERIFY_GRASP,
        Stage.CARRY,
        Stage.PLAN_PLACE,
        Stage.EXECUTE_PLACE,
        Stage.COMPLETE,
    )
    for stage in expected:
        transition = machine.apply(StageResult.success())
        assert transition.current == stage
    assert machine.terminal
    assert machine.failure_reason == ""


def test_plan_failures_try_candidates_then_restand_with_bounded_budget():
    budget = RetryBudget(plan_candidates=2, ik_restandoffs=1)
    machine = MobileManipulationStateMachine(budget)
    for _ in range(4):
        machine.apply(StageResult.success())
    assert machine.stage == Stage.PLAN_GRASP

    assert machine.apply(StageResult.failure(FailureKind.PLAN_BLOCKED)).current == Stage.PLAN_GRASP
    assert machine.apply(StageResult.failure(FailureKind.PLAN_BLOCKED)).current == Stage.PLAN_GRASP
    assert machine.apply(StageResult.failure(FailureKind.PLAN_BLOCKED)).current == Stage.VISUAL_APPROACH

    machine.apply(StageResult.success())
    machine.apply(StageResult.success())
    assert machine.stage == Stage.PLAN_GRASP
    machine.apply(StageResult.failure(FailureKind.PLAN_BLOCKED))
    machine.apply(StageResult.failure(FailureKind.PLAN_BLOCKED))
    exhausted = machine.apply(StageResult.failure(FailureKind.PLAN_BLOCKED))
    assert exhausted.current == Stage.FAILED
    assert "re-standoff" in exhausted.reason


def test_tracking_loss_restarts_search_but_never_loops_forever():
    machine = MobileManipulationStateMachine(RetryBudget(tracker_reacquisitions=1))
    machine.apply(StageResult.success())
    assert machine.stage == Stage.COARSE_NAV
    first = machine.apply(StageResult.failure(FailureKind.TARGET_LOST))
    assert first.current == Stage.SEARCH

    machine.apply(StageResult.success())
    second = machine.apply(StageResult.failure(FailureKind.TARGET_LOST))
    assert second.current == Stage.FAILED
    assert machine.terminal


def test_empty_grasp_reobserves_and_whole_pick_attempts_are_bounded():
    machine = MobileManipulationStateMachine(RetryBudget(grasp_attempts=1))
    for _ in range(5):
        machine.apply(StageResult.success())
    assert machine.stage == Stage.EXECUTE_GRASP
    assert machine.apply(StageResult.failure(FailureKind.EMPTY_GRASP)).current == Stage.OBSERVE_GRASP

    machine.apply(StageResult.success())
    machine.apply(StageResult.success())
    machine.apply(StageResult.success())
    failed = machine.apply(StageResult.failure(FailureKind.EMPTY_GRASP))
    assert failed.current == Stage.FAILED
    assert "grasp attempts" in failed.reason


@pytest.mark.parametrize("kind", [FailureKind.POSTURE_UNSAFE, FailureKind.FATAL])
def test_safety_failures_are_not_retried(kind):
    machine = MobileManipulationStateMachine()
    transition = machine.apply(StageResult.failure(kind, "specific fault"))
    assert transition.current == Stage.FAILED
    assert "specific fault" in transition.reason


def test_events_cannot_advance_a_terminal_task():
    machine = MobileManipulationStateMachine()
    machine.apply(StageResult.failure(FailureKind.FATAL, "stop"))
    with pytest.raises(RuntimeError, match="terminal"):
        machine.apply(StageResult.success())
