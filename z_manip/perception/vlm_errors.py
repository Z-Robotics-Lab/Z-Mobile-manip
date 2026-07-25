"""VLM exception hierarchy and shared cancellation primitives.

Kept dependency-free (stdlib only) so both the transport layer
(:mod:`z_manip.perception.vlm_transport`) and the orchestrator
(:mod:`z_manip.perception.vlm_openrouter`) can import it without a cycle.
"""

from __future__ import annotations

import threading


class VLMError(RuntimeError):
    """No configured VLM produced a valid structured grounding result."""


class VLMCancellationError(VLMError):
    """The caller invalidated an in-flight VLM request."""


class VLMTransportError(RuntimeError):
    """A typed provider transport failure with an explicit retry contract."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)


class _CombinedCancelEvent:
    """Minimal Event view that is set when any source event is set."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


def _raise_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise VLMCancellationError("VLM request was canceled")


__all__ = [
    "VLMCancellationError",
    "VLMError",
    "VLMTransportError",
]
