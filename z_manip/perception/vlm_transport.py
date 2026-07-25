"""Wire transport layer for VLM grounding requests.

Two transports live here: a curl-subprocess OpenRouter client with
process-group lifecycle and credential-redacting header/env handling, and a
loopback-only local-grounding HTTP client. Also hosts the shared transport
type aliases and the signature classification used to decide whether a
transport callable accepts a cancellation event.
"""

from __future__ import annotations

import http.client
import inspect
import json
import os
import signal
import subprocess
import threading
import time
from typing import Callable, Mapping
from urllib.parse import urlsplit

import numpy as np

from z_manip.perception.vlm_errors import (
    VLMCancellationError,
    VLMTransportError,
    _raise_if_cancelled,
)
from z_manip.perception.vlm_types import VLMAttemptEvent


LegacyTransport = Callable[
    [str, Mapping[str, object], Mapping[str, str], float],
    Mapping[str, object],
]
CancellableTransport = Callable[
    [str, Mapping[str, object], Mapping[str, str], float, threading.Event],
    Mapping[str, object],
]
Transport = LegacyTransport | CancellableTransport
AttemptCallback = Callable[[VLMAttemptEvent], None]

_MAX_CURL_HEADER_BYTES = 4096
_MAX_LOCAL_GROUNDING_RESPONSE_BYTES = 256 * 1024
_SENSITIVE_ENV_NAME_PARTS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
_SENSITIVE_HEADER_NAME_PARTS = (
    "api-key",
    "authorization",
    "cookie",
    "secret",
    "token",
)


def _validated_loopback_grounding_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("local grounding URL must be an unauthenticated loopback HTTP URL")
    if parsed.path not in {"", "/"}:
        raise ValueError("local grounding URL must not include a path")
    return value.rstrip("/")


def _loopback_grounding_transport(
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
    timeout_s: float,
    cancel_event: threading.Event,
) -> Mapping[str, object]:
    """POST one bounded JSON request to the loopback-only detector service."""

    del headers
    _raise_if_cancelled(cancel_event)
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("local grounding timeout must be finite and positive")
    parsed = urlsplit(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    _validated_loopback_grounding_url(base_url)
    if parsed.path != "/ground" or parsed.query or parsed.fragment:
        raise ValueError("local grounding transport accepts only /ground")
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port,
        timeout=float(timeout_s),
    )
    try:
        connection.request(
            "POST",
            "/ground",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        encoded = response.read(_MAX_LOCAL_GROUNDING_RESPONSE_BYTES + 1)
    except TimeoutError:
        raise
    except OSError as error:
        raise VLMTransportError(
            f"local grounding connection failed: {error}",
            retryable=False,
        ) from error
    finally:
        connection.close()
    _raise_if_cancelled(cancel_event)
    if len(encoded) > _MAX_LOCAL_GROUNDING_RESPONSE_BYTES:
        raise ValueError("local grounding response exceeded its size bound")
    try:
        document = json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("local grounding response was not valid JSON") from error
    if response.status != 200:
        detail = ""
        if isinstance(document, Mapping):
            detail = str(document.get("error", "")).replace("\n", " ")[-256:]
        raise VLMTransportError(
            f"local grounding HTTP {response.status}" + (f": {detail}" if detail else ""),
            retryable=False,
        )
    if not isinstance(document, Mapping):
        raise ValueError("local grounding response is not a JSON object")
    return document


def _transport_accepts_cancel_event(transport: Transport) -> bool:
    """Classify the public transport callback without invoking it."""
    try:
        signature = inspect.signature(transport)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "transport must expose an inspectable four- or five-argument signature",
        ) from error

    sentinel = object()
    try:
        signature.bind(sentinel, sentinel, sentinel, sentinel, sentinel)
    except TypeError:
        try:
            signature.bind(sentinel, sentinel, sentinel, sentinel)
        except TypeError as error:
            raise TypeError(
                "transport must accept (url, payload, headers, timeout_s) with an "
                "optional fifth cancel_event argument",
            ) from error
        return False
    return True


def _signal_process_group(process: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_s: float,
) -> tuple[str, str]:
    """Terminate curl and bound cleanup even if it ignores SIGTERM."""
    # The curl leader may have exited while a descendant still owns an inherited
    # stdout/stderr pipe. Its dedicated process group remains ours to terminate.
    _signal_process_group(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=grace_s)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        try:
            return process.communicate(timeout=grace_s)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("curl process did not exit after SIGKILL") from error


def _curl_header_bytes(headers: Mapping[str, str]) -> bytes:
    """Serialize bounded HTTP headers for curl's inherited anonymous pipe."""
    allowed_name = frozenset(
        "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )
    lines = []
    for key, value in headers.items():
        if not isinstance(key, str) or not key or any(char not in allowed_name for char in key):
            raise ValueError("curl header name is invalid")
        if not isinstance(value, str) or any(
            char in value for char in ("\0", "\r", "\n")
        ):
            raise ValueError(f"curl header value for {key!r} is invalid")
        lines.append(f"{key}: {value}\n")
    encoded = "".join(lines).encode("utf-8")
    if len(encoded) > _MAX_CURL_HEADER_BYTES:
        raise ValueError("curl headers exceed the bounded anonymous pipe capacity")
    return encoded


def _curl_environment(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a child environment without inherited credentials or header secrets."""
    sensitive_values = []
    for key, value in headers.items():
        normalized = key.lower()
        if any(part in normalized for part in _SENSITIVE_HEADER_NAME_PARTS):
            sensitive_values.append(value)
            _, separator, credential = value.partition(" ")
            if separator and len(credential) >= 8:
                sensitive_values.append(credential)
    return {
        key: value
        for key, value in os.environ.items()
        if not any(part in key.upper() for part in _SENSITIVE_ENV_NAME_PARTS)
        and not any(
            len(secret) >= 8 and secret in value
            for secret in sensitive_values
        )
    }


def _open_curl_header_pipe(headers: Mapping[str, str]) -> int | None:
    """Return a readable inherited FD containing headers, or None when empty."""
    encoded = _curl_header_bytes(headers)
    if not encoded:
        return None
    read_fd, write_fd = os.pipe()
    try:
        try:
            if os.write(write_fd, encoded) != len(encoded):
                raise OSError("short write to curl header pipe")
        finally:
            os.close(write_fd)
    except BaseException:
        os.close(read_fd)
        raise
    return read_fd


def _curl_transport(
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
    timeout_s: float,
    cancel_event: threading.Event,
    *,
    poll_interval_s: float = 0.05,
    terminate_grace_s: float = 0.25,
) -> Mapping[str, object]:
    """POST through a cancellable curl process isolated in its own process group."""
    if timeout_s <= 0.0 or poll_interval_s <= 0.0 or terminate_grace_s <= 0.0:
        raise ValueError("curl transport time bounds must be positive")
    _raise_if_cancelled(cancel_event)
    command = [
        "curl", "--disable", "--silent", "--show-error", "--fail-with-body",
        "--retry", "2", "--retry-delay", "1",
        "--max-time", f"{timeout_s:.3f}",
        "--request", "POST", url,
    ]
    # Serialize before allocating an inherited descriptor so invalid payloads
    # cannot leak an unread pipe into this long-lived ROS process.
    body: str | None = json.dumps(payload, separators=(",", ":"))
    header_fd = _open_curl_header_pipe(headers)
    if header_fd is not None:
        command.extend(("--header", f"@/proc/self/fd/{header_fd}"))
    # Passing the image-bearing JSON through stdin avoids argv size limits.
    command.extend(("--data-binary", "@-"))
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=(() if header_fd is None else (header_fd,)),
            env=_curl_environment(headers),
        )
    finally:
        if header_fd is not None:
            os.close(header_fd)
    deadline = time.monotonic() + timeout_s
    group_terminated = False
    try:
        while True:
            if cancel_event.is_set():
                _terminate_process_group(process, grace_s=terminate_grace_s)
                group_terminated = True
                raise VLMCancellationError("VLM request was canceled")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate_process_group(process, grace_s=terminate_grace_s)
                group_terminated = True
                raise TimeoutError(f"VLM request exceeded {timeout_s:.3f}s")
            try:
                stdout, stderr = process.communicate(
                    input=body,
                    timeout=min(poll_interval_s, remaining),
                )
                body = None
                break
            except subprocess.TimeoutExpired:
                # communicate() retains partial I/O state; input must only be
                # supplied on its first call.
                body = None
        _raise_if_cancelled(cancel_event)
        if process.returncode != 0:
            detail = stderr.strip().replace("\n", " ")[-512:]
            suffix = f": {detail}" if detail else ""
            raise VLMTransportError(
                f"OpenRouter curl exited with status {process.returncode}{suffix}",
                retryable=process.returncode in {5, 6, 7, 18, 35, 52, 55, 56, 92},
            )
        decoded = json.loads(stdout)
        if not isinstance(decoded, Mapping):
            raise ValueError("OpenRouter response is not a JSON object")
        return decoded
    except BaseException:
        if not group_terminated:
            _terminate_process_group(process, grace_s=terminate_grace_s)
        raise


__all__: list[str] = []
