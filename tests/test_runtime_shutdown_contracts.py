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
