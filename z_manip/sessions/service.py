"""Fail-closed contracts for interactive perception and offline planning.

This module deliberately contains no ROS, SSH, CAN, actuator, or subprocess
integration.  A server constructs :class:`ReadOnlySessionService` with one
fixed backend; action callers can provide only a target description, a safe
session identifier to select, or the parameter-free planning action.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any, Callable, Protocol
import unicodedata


STATE_SCHEMA = "z_manip.interactive_session_state.v1"
ATTEMPT_SCHEMA = "z_manip.interactive_session_attempt.v1"
MANIFEST_SCHEMA = "z_manip.immutable_artifact_manifest.v1"
SESSION_ID_PATTERN = re.compile(
    r"(?:[0-9]{8}-[0-9]{6}|s-[0-9a-f]{32})\Z",
)
MAX_TARGET_BYTES = 160


class SessionContractError(ValueError):
    """An interactive action violated the fixed server-side contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BackendResult:
    """Bounded result returned by the server-configured integration backend."""

    exit_code: int
    error_code: str | None = None
    message: str = ""


class ReadOnlyBackend(Protocol):
    """The only integration seam available to the pure session controller."""

    def run_perception(
        self,
        *,
        target: str,
        output_dir: Path,
        log_path: Path,
    ) -> BackendResult:
        """Produce read-only perception artifacts in ``output_dir``."""

    def run_planning(
        self,
        *,
        perception_dir: Path,
        output_dir: Path,
        log_path: Path,
    ) -> BackendResult:
        """Redo passive/session gates and offline planning."""


def validate_target_description(value: object) -> str:
    """Return one normalized, non-path target description.

    The limit is measured in UTF-8 bytes, which makes the wire contract
    unambiguous for non-ASCII object names.  Unicode control/format/surrogate
    characters, all line separators, and path syntax are rejected.
    """

    if not isinstance(value, str):
        raise SessionContractError("INVALID_TARGET_TYPE", "target must be text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise SessionContractError(
            "INVALID_TARGET_UTF8",
            "target must be valid UTF-8 text",
        ) from error
    if not 1 <= len(encoded) <= MAX_TARGET_BYTES:
        raise SessionContractError(
            "INVALID_TARGET_LENGTH",
            f"target must contain 1..{MAX_TARGET_BYTES} UTF-8 bytes",
        )
    if value != value.strip() or not value.strip():
        raise SessionContractError(
            "INVALID_TARGET_WHITESPACE",
            "target cannot be empty or have leading/trailing whitespace",
        )
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        or character in {"\u2028", "\u2029"}
        for character in value
    ):
        raise SessionContractError(
            "INVALID_TARGET_CONTROL_CHARACTER",
            "target cannot contain control or line-separator characters",
        )
    if "/" in value or "\\" in value or ".." in value:
        raise SessionContractError(
            "INVALID_TARGET_PATH",
            "target cannot contain a filesystem path",
        )
    if re.search(r"(?:^|\s)(?:~|[A-Za-z]:|file:)", value, re.IGNORECASE):
        raise SessionContractError(
            "INVALID_TARGET_PATH",
            "target cannot contain a filesystem path",
        )
    return value


def validate_session_id(value: object) -> str:
    """Accept only the server's timestamp or cryptographically random IDs."""

    if not isinstance(value, str) or SESSION_ID_PATTERN.fullmatch(value) is None:
        raise SessionContractError(
            "INVALID_SESSION_ID",
            "session id must be YYYYMMDD-HHMMSS or s- followed by 32 hex digits",
        )
    if value[0].isdigit():
        try:
            datetime.strptime(value, "%Y%m%d-%H%M%S")
        except ValueError as error:
            raise SessionContractError(
                "INVALID_SESSION_ID",
                "timestamp session id is not a real calendar time",
            ) from error
    return value


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso8601(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SessionContractError(
            "INVALID_SERVER_ARTIFACT",
            f"cannot read {label}: {error}",
        ) from error
    if not isinstance(document, dict):
        raise SessionContractError(
            "INVALID_SERVER_ARTIFACT",
            f"{label} must contain a JSON object",
        )
    return document


def _write_json_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_manifest(directory: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise SessionContractError(
                "INVALID_SERVER_ARTIFACT",
                "artifact trees cannot contain symbolic links",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise SessionContractError(
                "INVALID_SERVER_ARTIFACT",
                "artifact trees can contain only regular files and directories",
            )
        files.append({
            "name": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {
        "schema": MANIFEST_SCHEMA,
        "file_count": len(files),
        "files": files,
    }


def _freeze_tree(path: Path) -> None:
    descendants = sorted(
        path.rglob("*"),
        key=lambda child: len(child.relative_to(path).parts),
        reverse=True,
    )
    for child in descendants:
        if child.is_symlink():
            raise SessionContractError(
                "INVALID_SERVER_ARTIFACT",
                "immutable artifact trees cannot contain symbolic links",
            )
        try:
            child.chmod(0o500 if child.is_dir() else 0o400)
        except PermissionError:
            # Docker versions predating the fixed ``--user`` contract may
            # leave root-owned regular files in an otherwise service-owned
            # attempt directory.  Accept those files only when the service
            # identity already has no write access and neither group nor
            # other can mutate them.  All other ownership/mode failures stay
            # fail-closed, and the artifact manifest is revalidated before a
            # perception session can be selected for planning.
            mode = child.stat(follow_symlinks=False).st_mode
            already_immutable = bool(
                child.is_file()
                and stat.S_ISREG(mode)
                and mode & (stat.S_IWGRP | stat.S_IWOTH) == 0
                and not os.access(child, os.W_OK)
            )
            if not already_immutable:
                raise
    path.chmod(0o500)


class ReadOnlySessionService:
    """Persist immutable perception/planning attempts behind fixed actions.

    Importable controller API for a loopback server::

        start_perception(target: object) -> attempt JSON
        select_perception(session_id: object) -> selection JSON
        start_planning() -> attempt JSON
        status() -> state JSON

    ``start_*`` calls are synchronous and fail with ``ACTION_BUSY`` rather than
    queueing.  Their returned attempt report is the same immutable JSON stored
    in the session.  No caller-provided path, command, or environment exists in
    these signatures.
    """

    def __init__(
        self,
        run_root: Path,
        backend: ReadOnlyBackend,
        *,
        now: Callable[[], datetime] = _utc_now,
        random_token: Callable[[], str] | None = None,
    ) -> None:
        self._root = run_root.expanduser().resolve()
        self._backend = backend
        self._now = now
        self._random_token = random_token or (lambda: secrets.token_hex(16))
        self._lock = threading.Lock()
        for name in ("perception", "planning", "_state"):
            (self._root / name).mkdir(parents=True, exist_ok=True)

    def _new_session(self, action: str) -> tuple[str, Path]:
        timestamp = self._now().astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
        action_root = self._root / action
        for session_id in (timestamp, f"s-{self._random_token()}"):
            validate_session_id(session_id)
            destination = action_root / session_id
            try:
                destination.mkdir(mode=0o700)
            except FileExistsError:
                continue
            return session_id, destination
        raise SessionContractError(
            "SESSION_ID_COLLISION",
            "could not allocate a unique server session id",
        )

    def _resolve_session(self, action: str, session_id: object) -> Path:
        safe_id = validate_session_id(session_id)
        action_root = (self._root / action).resolve(strict=True)
        candidate = action_root / safe_id
        if candidate.is_symlink():
            raise SessionContractError(
                "INVALID_SESSION_PATH",
                "session directories cannot be symbolic links",
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise SessionContractError(
                "SESSION_NOT_FOUND",
                "server session does not exist",
            ) from error
        if resolved.parent != action_root:
            raise SessionContractError(
                "INVALID_SESSION_PATH",
                "resolved session is outside the configured run root",
            )
        return resolved

    def _reference_path(self, action: str, name: str) -> Path:
        return self._root / "_state" / f"{action}_{name}.json"

    def _set_reference(self, action: str, name: str, session_id: str) -> None:
        _write_json_atomic(
            self._reference_path(action, name),
            {
                "schema": STATE_SCHEMA,
                "action": action,
                "session_id": session_id,
                "updated_at": _iso8601(self._now()),
            },
        )

    def _read_reference(self, action: str, name: str) -> str | None:
        path = self._reference_path(action, name)
        if not path.is_file():
            return None
        document = _load_object(path, f"{action} {name} reference")
        if document.get("schema") != STATE_SCHEMA or document.get("action") != action:
            raise SessionContractError(
                "INVALID_SERVER_STATE",
                f"{action} {name} reference has an invalid schema",
            )
        return validate_session_id(document.get("session_id"))

    def _remove_references(self, action: str, names: tuple[str, ...]) -> list[str]:
        removed: list[str] = []
        for name in names:
            path = self._reference_path(action, name)
            if path.is_file():
                path.unlink()
                removed.append(f"{action}_{name}")
        return removed

    def _attempt(self, action: str, session_id: str) -> dict[str, Any]:
        session = self._resolve_session(action, session_id)
        document = _load_object(session / "attempt.json", f"{action} attempt")
        if (
            document.get("schema") != ATTEMPT_SCHEMA
            or document.get("action") != action
            or document.get("session_id") != session_id
        ):
            raise SessionContractError(
                "INVALID_SERVER_STATE",
                "attempt report identity does not match its server session",
            )
        return document

    def _successful_perception(self, session_id: str) -> Path:
        session = self._resolve_session("perception", session_id)
        attempt = self._attempt("perception", session_id)
        if attempt.get("status") != "succeeded":
            raise SessionContractError(
                "PERCEPTION_NOT_SUCCESSFUL",
                "planning can select only a successful perception session",
            )
        perception = session / "perception"
        if perception.is_symlink() or not perception.is_dir():
            raise SessionContractError(
                "INVALID_SERVER_ARTIFACT",
                "selected perception artifacts are unavailable",
            )
        manifest = _load_object(
            session / "perception_manifest.json",
            "perception manifest",
        )
        if manifest != _artifact_manifest(perception):
            raise SessionContractError(
                "PERCEPTION_ARTIFACT_CHANGED",
                "selected perception artifacts no longer match their immutable manifest",
            )
        return perception

    @staticmethod
    def _perception_succeeded(directory: Path, result: BackendResult) -> bool:
        if result.exit_code != 0:
            return False
        try:
            report = _load_object(directory / "report.json", "perception report")
        except SessionContractError:
            return False
        return bool(
            report.get("read_only") is True
            and report.get("grasp_generation_valid") is True
        )

    @staticmethod
    def _planning_succeeded(directory: Path, result: BackendResult) -> bool:
        if result.exit_code != 0:
            return False
        try:
            gate = _load_object(directory / "session_gate.json", "session gate")
            report = _load_object(
                directory / "planning" / "planning_report.json",
                "planning report",
            )
        except SessionContractError:
            return False
        return bool(
            gate.get("planning_ready") is True
            and gate.get("read_only") is True
            and gate.get("planning_only") is True
            and gate.get("motion_commands_published") == 0
            and gate.get("transport_opened") is False
            and report.get("read_only") is True
            and report.get("planning_only") is True
            and report.get("motion_commands_published") == 0
            and report.get("plan_valid") is True
        )

    def run_perception(self, target: object) -> dict[str, Any]:
        """Run one fixed perception action for a validated target description."""

        description = validate_target_description(target)
        if not self._lock.acquire(blocking=False):
            raise SessionContractError("ACTION_BUSY", "another action is in progress")
        try:
            session_id, session = self._new_session("perception")
            started = self._now()
            in_flight = session / ".perception.inflight"
            in_flight.mkdir(mode=0o700)
            log = session / "perception.log"
            try:
                result = self._backend.run_perception(
                    target=description,
                    output_dir=in_flight,
                    log_path=log,
                )
            except Exception as error:  # backend failures become inspectable attempts
                result = BackendResult(
                    exit_code=1,
                    error_code="BACKEND_EXCEPTION",
                    message=f"{type(error).__name__}: {error}",
                )
            perception = session / "perception"
            in_flight.rename(perception)
            succeeded = self._perception_succeeded(perception, result)
            manifest = _artifact_manifest(perception)
            _write_json_exclusive(session / "perception_manifest.json", manifest)
            attempt: dict[str, Any] = {
                "schema": ATTEMPT_SCHEMA,
                "action": "perception",
                "session_id": session_id,
                "status": "succeeded" if succeeded else "failed",
                "target": description,
                "started_at": _iso8601(started),
                "finished_at": _iso8601(self._now()),
                "backend_exit_code": result.exit_code,
                "error": None if succeeded else {
                    "code": result.error_code or "PERCEPTION_OUTPUT_INVALID",
                    "message": result.message or (
                        "perception did not produce a valid read-only result"
                    ),
                },
                "artifacts": {
                    "perception": "perception",
                    "manifest": "perception_manifest.json",
                },
                "safety": {
                    "read_only": True,
                    "motion_commands_published": 0,
                    "can_tx_available": False,
                },
            }
            _write_json_exclusive(session / "attempt.json", attempt)
            _freeze_tree(session)
            self._set_reference("perception", "latest_attempt", session_id)
            if succeeded:
                # A plan is valid only for the exact selected perception.  Do
                # not let a newly perceived object inherit an older green plan.
                self._remove_references("planning", ("latest_attempt", "last_good"))
                self._set_reference("perception", "last_good", session_id)
                self._set_reference("perception", "selected", session_id)
            return attempt
        finally:
            self._lock.release()

    def start_perception(self, target: object) -> dict[str, Any]:
        """Synchronously start perception; alias intended for HTTP handlers."""

        return self.run_perception(target)

    def select_perception(self, session_id: object) -> dict[str, object]:
        """Select one verified successful server session for later planning."""

        safe_id = validate_session_id(session_id)
        self._successful_perception(safe_id)
        self._set_reference("perception", "selected", safe_id)
        return {
            "schema": STATE_SCHEMA,
            "selected_perception_session_id": safe_id,
        }

    def clear_current_context(self) -> dict[str, object]:
        """Invalidate all current task pointers while retaining audit history.

        Immutable perception/planning directories remain available for logs and
        diagnosis, but none can be planned or executed until a new perception
        succeeds.  Home completion uses this to prevent stale plans from a
        previous object or robot cycle being reused.
        """

        with self._lock:
            cleared = self._remove_references(
                "perception", ("selected", "latest_attempt", "last_good"),
            )
            cleared.extend(self._remove_references(
                "planning", ("latest_attempt", "last_good"),
            ))
            return {
                "schema": STATE_SCHEMA,
                "cleared": True,
                "cleared_references": cleared,
                "history_retained": True,
            }

    def run_planning(self) -> dict[str, Any]:
        """Run fixed passive/session gates and offline planning for selection."""

        if not self._lock.acquire(blocking=False):
            raise SessionContractError("ACTION_BUSY", "another action is in progress")
        try:
            session_id, session = self._new_session("planning")
            started = self._now()
            selected_id: str | None = None
            result = BackendResult(
                exit_code=1,
                error_code="NO_SELECTED_PERCEPTION",
                message="select a successful perception session before planning",
            )
            in_flight = session / ".planning.inflight"
            in_flight.mkdir(mode=0o700)
            log = session / "planning.log"
            try:
                selected_id = self._read_reference("perception", "selected")
                if selected_id is None:
                    raise SessionContractError(
                        "NO_SELECTED_PERCEPTION",
                        "select a successful perception session before planning",
                    )
                perception = self._successful_perception(selected_id)
                try:
                    result = self._backend.run_planning(
                        perception_dir=perception,
                        output_dir=in_flight,
                        log_path=log,
                    )
                except Exception as error:  # backend failures become inspectable attempts
                    result = BackendResult(
                        exit_code=1,
                        error_code="BACKEND_EXCEPTION",
                        message=f"{type(error).__name__}: {error}",
                    )
            except SessionContractError as error:
                result = BackendResult(1, error.code, str(error))
            artifacts = session / "artifacts"
            in_flight.rename(artifacts)
            succeeded = self._planning_succeeded(artifacts, result)
            manifest = _artifact_manifest(artifacts)
            _write_json_exclusive(session / "planning_manifest.json", manifest)
            attempt: dict[str, Any] = {
                "schema": ATTEMPT_SCHEMA,
                "action": "planning",
                "session_id": session_id,
                "selected_perception_session_id": selected_id,
                "status": "succeeded" if succeeded else "blocked",
                "started_at": _iso8601(started),
                "finished_at": _iso8601(self._now()),
                "backend_exit_code": result.exit_code,
                "error": None if succeeded else {
                    "code": result.error_code or "PLANNING_OUTPUT_INVALID",
                    "message": result.message or "offline planning was blocked",
                },
                "artifacts": {
                    "planning": "artifacts/planning",
                    "session_gate": "artifacts/session_gate.json",
                    "manifest": "planning_manifest.json",
                },
                "safety": {
                    "read_only": True,
                    "planning_only": True,
                    "motion_commands_published": 0,
                    "can_tx_available": False,
                    "actuator_transport_available": False,
                },
            }
            _write_json_exclusive(session / "attempt.json", attempt)
            _freeze_tree(session)
            self._set_reference("planning", "latest_attempt", session_id)
            if succeeded:
                self._set_reference("planning", "last_good", session_id)
            return attempt
        finally:
            self._lock.release()

    def start_planning(self) -> dict[str, Any]:
        """Synchronously start selected-session planning for an HTTP handler."""

        return self.run_planning()

    def _reference_summary(self, action: str, name: str) -> dict[str, Any] | None:
        session_id = self._read_reference(action, name)
        if session_id is None:
            return None
        attempt = self._attempt(action, session_id)
        return {
            key: attempt.get(key)
            for key in (
                "action",
                "session_id",
                "selected_perception_session_id",
                "status",
                "target",
                "started_at",
                "finished_at",
                "error",
                "safety",
            )
            if key in attempt
        }

    def status(self) -> dict[str, object]:
        """Return bounded JSON state suitable for a future dashboard API."""

        selected = self._read_reference("perception", "selected")
        return {
            "schema": STATE_SCHEMA,
            "read_only": True,
            "busy": self._lock.locked(),
            "selected_perception_session_id": selected,
            "actions": {
                "perception": {
                    "input": "target_description_utf8_1_160",
                    "latest_attempt": self._reference_summary(
                        "perception",
                        "latest_attempt",
                    ),
                    "last_good": self._reference_summary("perception", "last_good"),
                },
                "planning": {
                    "input": "server_selected_perception_session",
                    "latest_attempt": self._reference_summary(
                        "planning",
                        "latest_attempt",
                    ),
                    "last_good": self._reference_summary("planning", "last_good"),
                },
                "grasp_execution": {
                    "available": False,
                    "reason": "read-only session service has no actuator surface",
                },
            },
            "safety": {
                "motion_commands_available": False,
                "actuator_transport_available": False,
                "can_tx_available": False,
                "client_paths_accepted": False,
                "client_commands_accepted": False,
                "client_environment_accepted": False,
            },
        }
