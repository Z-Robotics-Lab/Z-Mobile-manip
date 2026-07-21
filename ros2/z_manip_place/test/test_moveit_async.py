"""Cancellation and deadline tests for MoveIt's asynchronous service boundary."""

from types import SimpleNamespace

import pytest


pytest.importorskip('rclpy')

from z_manip.planning_control import (  # noqa: E402
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
)
from z_manip_place.moveit_evaluator import MoveItPlacementEvaluator  # noqa: E402


class _Future:
    def __init__(self, response=None, *, done=False):
        self.response = response
        self.complete = done
        self.cancelled = False

    def done(self):
        return self.complete

    def result(self):
        return self.response

    def cancel(self):
        self.cancelled = True
        self.complete = True


class _Client:
    def __init__(self, future):
        self.future = future
        self.requests = []

    def call_async(self, request):
        self.requests.append(request)
        return self.future


def _evaluator():
    evaluator = MoveItPlacementEvaluator.__new__(MoveItPlacementEvaluator)
    evaluator.config = SimpleNamespace(
        response_timeout_s=0.1,
        service_wait_timeout_s=0.1,
    )
    return evaluator


def test_async_moveit_response_is_returned_without_sync_client_call():
    response = object()
    future = _Future(response, done=True)
    client = _Client(future)

    result = _evaluator()._call_async(
        client,
        object(),
        'test planning',
        PlanningControl(deadline_s=10.0, monotonic_fn=lambda: 0.0),
    )

    assert result is response
    assert len(client.requests) == 1
    assert not future.cancelled


def test_abort_cancels_hung_moveit_future_and_a_retry_uses_a_new_future(
    monkeypatch,
):
    monkeypatch.setattr('z_manip_place.moveit_evaluator.time.sleep', lambda _s: None)
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 4

    old_future = _Future()
    with pytest.raises(PlanningCancelled, match='cancelled'):
        _evaluator()._call_async(
            _Client(old_future),
            object(),
            'hung planning',
            PlanningControl(
                deadline_s=10.0,
                cancel_check=cancelled,
                monotonic_fn=lambda: 0.0,
            ),
        )
    assert old_future.cancelled

    new_response = object()
    new_future = _Future(new_response, done=True)
    assert _evaluator()._call_async(
        _Client(new_future),
        object(),
        'retry planning',
        PlanningControl(deadline_s=10.0, monotonic_fn=lambda: 0.0),
    ) is new_response
    assert not new_future.cancelled


def test_wall_deadline_cancels_a_hung_moveit_future(monkeypatch):
    monkeypatch.setattr('z_manip_place.moveit_evaluator.time.sleep', lambda _s: None)
    now = -0.03

    def monotonic():
        nonlocal now
        now += 0.03
        return now

    future = _Future()
    with pytest.raises(PlanningDeadlineExceeded, match='deadline'):
        _evaluator()._call_async(
            _Client(future),
            object(),
            'deadline planning',
            PlanningControl(deadline_s=0.10, monotonic_fn=monotonic),
        )
    assert future.cancelled
