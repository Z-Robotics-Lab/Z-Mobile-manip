"""Structural contracts for node teardown and worker-thread failure guards.

These nodes cannot be constructed here (rclpy is not installed in the test
environment), so the invariants are asserted against the parsed source in the
same style as tests/test_filtered_cloud_pipeline_contract.py.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_NODE = ROOT / "ros2" / "z_manip_task" / "z_manip_task" / "node.py"
PLACE_NODE = ROOT / "ros2" / "z_manip_place" / "z_manip_place" / "node.py"
EDGETAM_NODE = ROOT / "ros2" / "z_manip_edgetam" / "z_manip_edgetam" / "node.py"
PLACE_EVALUATOR = (
    ROOT / "ros2" / "z_manip_place" / "z_manip_place" / "moveit_evaluator.py"
)


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == method_name:
                return child
        raise AssertionError(
            f"{class_name} in {path.relative_to(ROOT)} defines no {method_name}",
        )
    raise AssertionError(f"{path.relative_to(ROOT)} defines no class {class_name}")


def _guarded_ids(function: ast.FunctionDef) -> set[int]:
    """Ids of every node inside the protected body of some ``try``."""
    guarded: set[int] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Try):
            continue
        for statement in node.body:
            guarded.update(id(child) for child in ast.walk(statement))
    return guarded


def test_task_destroy_node_cancels_the_planner_before_shutting_the_executor():
    destroy = ast.unparse(_method(TASK_NODE, "MobileManipulationRuntime", "destroy_node"))

    # cancel_futures only drops queued work, so the running plan has to be
    # told to stop cooperatively or the non-daemon executor thread is joined
    # unbounded at interpreter exit for the rest of the planning budget.
    assert "_future_cancel_event" in destroy
    assert "_worker.shutdown(" in destroy
    assert destroy.index("_future_cancel_event") < destroy.index("_worker.shutdown(")
    assert ".set()" in destroy


def test_placement_node_shutdown_stops_and_joins_its_daemon_planner():
    destroy = ast.unparse(_method(PLACE_NODE, "ObservedPlacementNode", "destroy_node"))

    assert "self._shutting_down = True" in destroy
    # Invalidating the transaction is what makes _planning_cancelled true, so
    # the worker's next checkpoint aborts instead of touching dead handles.
    assert "self._transaction.reset()" in destroy
    assert ".join(timeout=" in destroy
    assert "super().destroy_node()" in destroy


def test_placement_planner_cannot_restart_itself_during_shutdown():
    auto_plan = ast.unparse(_method(PLACE_NODE, "ObservedPlacementNode", "_maybe_auto_plan"))
    start_plan = ast.unparse(_method(PLACE_NODE, "ObservedPlacementNode", "_start_plan"))

    # _plan_worker calls _maybe_auto_plan from its own finally, so both the
    # trigger and the authoritative start have to refuse once teardown began.
    assert "self._shutting_down" in auto_plan
    assert "self._shutting_down" in start_plan


def test_edgetam_worker_loop_handles_every_command_inside_its_failure_guard():
    loop = _method(EDGETAM_NODE, "EdgeTamAdapter", "_worker_loop")
    guarded = _guarded_ids(loop)
    kinds = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Constant) and node.value in {"reset", "init", "frame"}
    ]

    assert kinds, "the worker loop no longer dispatches on command.kind"
    # An escape from the dispatch ends the loop and kills the only consumer of
    # self._commands: the node keeps accepting frames and never produces a
    # result again, recoverable only by restart.
    unguarded = [node.value for node in kinds if id(node) not in guarded]
    assert not unguarded, f"command kinds handled outside the guard: {unguarded}"


def test_placement_evaluator_never_strands_a_created_service_client():
    init = _method(PLACE_EVALUATOR, "MoveItPlacementEvaluator", "__init__")
    tries = [node for node in ast.walk(init) if isinstance(node, ast.Try)]

    assert tries, "a throw on the second create_client would strand the first"
    guarded = ast.unparse(tries[0])
    assert "create_client(GetCartesianPath" in guarded
    assert "destroy_client(self.motion_client)" in guarded


def test_placement_worker_destroys_each_client_independently():
    worker = _method(PLACE_NODE, "ObservedPlacementNode", "_plan_worker")
    guarded = _guarded_ids(worker)
    destroys = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "destroy_client"
    ]

    assert destroys, "the plan worker no longer releases its service clients"
    # The cleanup runs in the worker's finally; an unguarded failure there
    # skips the remaining client and the worker-registry release below it.
    assert all(id(node) in guarded for node in destroys)
