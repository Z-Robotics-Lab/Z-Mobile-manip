"""Cooperative wall-clock and cancellation control for planning work."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from z_manip.models.planner import PlanningError


class PlanningAborted(PlanningError):
    """Planning stopped by its execution budget rather than infeasibility."""


class PlanningCancelled(PlanningAborted):
    """The caller invalidated planning work that was still in progress."""


class PlanningDeadlineExceeded(PlanningAborted):
    """Planning exhausted an absolute monotonic wall-clock deadline."""


@dataclass(frozen=True)
class PlanningControl:
    """Cooperative control shared by every layer of one planning request.

    ``deadline_s`` is an absolute value from ``monotonic_fn``. The cancellation
    callback should be cheap and thread-safe; a task runtime can back it with a
    ``threading.Event`` or a generation-token comparison. Both controls are
    optional so existing synchronous callers remain unbounded by default.
    """

    deadline_s: float | None = None
    cancel_check: Callable[[], bool] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    monotonic_fn: Callable[[], float] = field(
        default=time.monotonic,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.deadline_s is not None and not math.isfinite(float(self.deadline_s)):
            raise ValueError("planning deadline must be a finite monotonic timestamp")
        if self.cancel_check is not None and not callable(self.cancel_check):
            raise TypeError("planning cancel_check must be callable")
        if not callable(self.monotonic_fn):
            raise TypeError("planning monotonic_fn must be callable")

    def _cancelled(self, operation: str) -> bool:
        if self.cancel_check is None:
            return False
        try:
            return bool(self.cancel_check())
        except Exception as error:
            raise PlanningCancelled(
                f"{operation} cancellation check failed closed: "
                f"{type(error).__name__}: {error}",
            ) from error

    def _now(self, operation: str) -> float:
        try:
            now = float(self.monotonic_fn())
        except Exception as error:
            raise PlanningDeadlineExceeded(
                f"{operation} monotonic deadline clock failed closed: "
                f"{type(error).__name__}: {error}",
            ) from error
        if not math.isfinite(now):
            raise PlanningDeadlineExceeded(
                f"{operation} monotonic deadline clock returned a non-finite value",
            )
        return now

    def checkpoint(self, operation: str = "planning") -> None:
        """Raise a typed exception when this request should stop now."""

        if self._cancelled(operation):
            raise PlanningCancelled(f"{operation} was cancelled")
        if self.deadline_s is not None:
            now = self._now(operation)
            if now >= self.deadline_s:
                raise PlanningDeadlineExceeded(
                    f"{operation} exceeded its monotonic deadline by "
                    f"{now - self.deadline_s:.6f} s",
                )

    def limited_to(self, timeout_s: float, operation: str = "planning") -> PlanningControl:
        """Return a child control capped by a relative wall-clock timeout."""

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("planning timeout must be finite and positive")
        self.checkpoint(operation)
        local_deadline = self._now(operation) + timeout
        deadline = (
            local_deadline
            if self.deadline_s is None
            else min(float(self.deadline_s), local_deadline)
        )
        return PlanningControl(
            deadline_s=deadline,
            cancel_check=self.cancel_check,
            monotonic_fn=self.monotonic_fn,
        )


def checkpoint(control: PlanningControl | None, operation: str) -> None:
    """Fast no-op wrapper used by algorithms with optional control."""

    if control is not None:
        control.checkpoint(operation)


def classify_control_mode(
    callback: Callable[..., object],
    *,
    arity: int,
    requirement: str,
) -> str:
    """Classify how an injected callback transports its optional ``control``.

    Returns ``"keyword"``, ``"positional"`` or ``"legacy"`` without executing
    the callback's body, so a planner can pick one invocation shape per search
    instead of retrying a failed call.  ``arity`` is the number of required
    non-control arguments the callback takes.

    Keyword transport is preferred: an opaque ``*args, **kwargs`` wrapper binds
    every shape, and passing ``control=`` keeps it out of the callback's own
    positional slots.  A callable whose signature cannot be inspected at all
    keeps the pre-budget legacy invocation.  ``requirement`` is the message
    raised when the callback accepts none of the three supported shapes.
    """
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return "legacy"
    sentinel = object()
    required = (sentinel,) * arity
    try:
        signature.bind(*required, control=sentinel)
    except TypeError:
        try:
            signature.bind(*required, sentinel)
        except TypeError:
            try:
                signature.bind(*required)
            except TypeError as error:
                raise TypeError(requirement) from error
            return "legacy"
        return "positional"
    return "keyword"


__all__ = [
    "PlanningAborted",
    "PlanningCancelled",
    "PlanningControl",
    "PlanningDeadlineExceeded",
    "checkpoint",
    "classify_control_mode",
]
