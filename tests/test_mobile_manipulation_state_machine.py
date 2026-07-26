import pytest

from z_manip.orchestration.mobile_manipulation import (
    _RECOVERY,
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


def test_recovery_table_counters_name_a_real_counter_and_a_real_budget_field():
    # The table's first element is used both as a ``counters`` key and as a
    # ``RetryBudget`` attribute name.  A copy-paste that names one but not the
    # other would otherwise surface only as a runtime KeyError/AttributeError
    # on a rare failure path.
    machine = MobileManipulationStateMachine()
    budget_fields = set(vars(RetryBudget()))
    for kind, (counter, stage, exhausted) in _RECOVERY.items():
        assert counter in machine.counters, kind
        assert counter in budget_fields, kind
        assert isinstance(stage, Stage), kind
        assert exhausted, kind


def test_recovery_table_covers_every_failure_kind_without_its_own_bookkeeping():
    # Kinds outside the table each carry extra behaviour: terminal safety
    # faults, the plan_candidates reset, and the candidate-retry ladder.
    explicit = {
        FailureKind.POSTURE_UNSAFE,
        FailureKind.FATAL,
        FailureKind.NO_GRASP,
        FailureKind.IK_UNREACHABLE,
        FailureKind.PLAN_BLOCKED,
    }
    assert set(_RECOVERY) | explicit == set(FailureKind)
    assert set(_RECOVERY) & explicit == set()


@pytest.mark.parametrize(
    ("kind", "recovery_stage", "counter"),
    [
        (FailureKind.NOT_FOUND, Stage.SEARCH, "search_misses"),
        (FailureKind.TARGET_LOST, Stage.SEARCH, "tracker_reacquisitions"),
        (FailureKind.NAV_BLOCKED, Stage.COARSE_NAV, "nav_replans"),
        (FailureKind.VISUAL_APPROACH_FAILED, Stage.COARSE_NAV, "nav_replans"),
        (FailureKind.EXECUTION_FAILED, Stage.OBSERVE_GRASP, "grasp_attempts"),
        (FailureKind.EMPTY_GRASP, Stage.OBSERVE_GRASP, "grasp_attempts"),
        (FailureKind.VERIFY_FAILED, Stage.OBSERVE_GRASP, "grasp_attempts"),
        (FailureKind.PLACE_BLOCKED, Stage.PLAN_PLACE, "place_replans"),
        (FailureKind.RELEASE_FAILED, Stage.EXECUTE_PLACE, "release_attempts"),
    ],
)
def test_table_driven_recovery_matches_the_declared_stage_and_counter(
    kind,
    recovery_stage,
    counter,
):
    machine = MobileManipulationStateMachine()
    transition = machine.apply(StageResult.failure(kind))
    assert transition.current == recovery_stage
    assert transition.reason == f"recovering from {counter}"
    assert transition.counters[counter] == 1
