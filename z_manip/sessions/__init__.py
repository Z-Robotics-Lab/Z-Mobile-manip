"""Fail-closed interactive session contracts for the workbench UI.

Session identifier/target validation, read-only artifact selection, and the
immutable-artifact freeze/manifest machinery used by the perception and
planning workbench servers.
"""

from __future__ import annotations

from .service import (
    ATTEMPT_SCHEMA,
    MANIFEST_SCHEMA,
    MAX_TARGET_BYTES,
    SESSION_ID_PATTERN,
    STATE_SCHEMA,
    BackendResult,
    ReadOnlyBackend,
    ReadOnlySessionService,
    SessionContractError,
    validate_session_id,
    validate_target_description,
)

__all__ = [
    "ATTEMPT_SCHEMA",
    "MANIFEST_SCHEMA",
    "MAX_TARGET_BYTES",
    "SESSION_ID_PATTERN",
    "STATE_SCHEMA",
    "BackendResult",
    "ReadOnlyBackend",
    "ReadOnlySessionService",
    "SessionContractError",
    "validate_session_id",
    "validate_target_description",
]
