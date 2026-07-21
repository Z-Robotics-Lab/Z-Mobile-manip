"""Structured VLM grounding and grasp-affordance reasoning via OpenRouter."""

from __future__ import annotations

import base64
import copy
import http.client
import inspect
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np


DEFAULT_MODELS = (
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3.5-35b-a3b",
)

_BBOX_COORDINATE_SCALES = {
    "normalized_0_1": 1.0,
    "relative_0_1000": 1000.0,
}

GROUNDING_SCOPES = frozenset({
    "grasp_only",
    "grasp_for_place",
    "place_support",
})


class VLMError(RuntimeError):
    """No configured VLM produced a valid structured grounding result."""


class VLMCancellationError(VLMError):
    """The caller invalidated an in-flight VLM request."""


class _CombinedCancelEvent:
    """Minimal Event view that is set when any source event is set."""

    def __init__(self, *events: threading.Event) -> None:
        self._events = events

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


class VLMTransportError(RuntimeError):
    """A typed provider transport failure with an explicit retry contract."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)


@dataclass(frozen=True)
class NormalizedBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(np.isfinite(values)) or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized box coordinates must be finite and in [0, 1]")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("normalized box must have positive area")

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        coordinate_scale: float = 1.0,
    ) -> "NormalizedBox":
        coordinates = np.asarray(value, dtype=float)
        if coordinates.shape != (4,):
            raise ValueError("bbox_xyxy must contain four coordinates")
        scale = float(coordinate_scale)
        if scale not in _BBOX_COORDINATE_SCALES.values():
            raise ValueError("bbox coordinate scale is unsupported")
        if scale == 1000.0 and (
            not np.all(np.isfinite(coordinates))
            or not np.all(coordinates == np.floor(coordinates))
        ):
            raise ValueError(
                "relative_0_1000 bbox coordinates must be finite integers",
            )
        return cls(*(coordinates / scale).tolist())

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return (
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
            int(round(self.x2 * width)),
            int(round(self.y2 * height)),
        )


@dataclass(frozen=True)
class AvoidRegion:
    label: str
    bbox: NormalizedBox


@dataclass(frozen=True)
class PlacementVerification:
    """Semantic object axes that downstream geometry must observe or reject."""

    require_upright: bool
    upright_axis: str
    orientation_symmetry: str
    symmetry_axis: str | None

    def __post_init__(self) -> None:
        axes = {'principal_long', 'principal_middle', 'principal_short'}
        if not isinstance(self.require_upright, bool):
            raise ValueError("require_upright must be boolean")
        if self.upright_axis not in axes:
            raise ValueError("upright_axis must select an observed principal axis")
        if self.orientation_symmetry not in {'none', 'axial'}:
            raise ValueError("orientation_symmetry must be none or axial")
        if self.orientation_symmetry == 'axial':
            if self.symmetry_axis not in axes:
                raise ValueError("axial symmetry requires a principal symmetry_axis")
        elif self.symmetry_axis is not None:
            raise ValueError("symmetry_axis is only valid for axial symmetry")


@dataclass(frozen=True)
class AffordanceResult:
    model: str
    target_label: str
    target_bbox: NormalizedBox
    confidence: float
    grasp_part_label: str | None
    grasp_part_bbox: NormalizedBox | None
    avoid_regions: tuple[AvoidRegion, ...]
    preferred_approach_camera: tuple[float, float, float] | None
    placement_region_label: str | None
    placement_region_bbox: NormalizedBox | None
    placement_avoid_regions: tuple[AvoidRegion, ...]
    placement_verification: PlacementVerification | None
    constraints: tuple[str, ...]
    latency_s: float


@dataclass(frozen=True)
class VLMAttemptEvent:
    """Bounded, credential-free diagnostics for one model attempt."""

    model: str
    attempt: int
    outcome: str
    elapsed_s: float
    detail: str = ""


_OUTPUT_SCHEMA = {
    "name": "mobile_manipulation_affordance",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "target", "grasp_part", "avoid_regions",
            "preferred_approach_camera", "placement_region",
            "placement_avoid_regions", "placement_verification", "constraints",
        ],
        "properties": {
            "target": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "bbox_xyxy", "confidence"],
                "properties": {
                    "label": {"type": "string"},
                    "bbox_xyxy": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4,
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "grasp_part": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object", "additionalProperties": False,
                        "required": ["label", "bbox_xyxy"],
                        "properties": {
                            "label": {"type": "string"},
                            "bbox_xyxy": {
                                "type": "array", "items": {"type": "number"},
                                "minItems": 4, "maxItems": 4,
                            },
                        },
                    },
                ],
            },
            "avoid_regions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["label", "bbox_xyxy"],
                    "properties": {
                        "label": {"type": "string"},
                        "bbox_xyxy": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": 4, "maxItems": 4,
                        },
                    },
                },
            },
            "preferred_approach_camera": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 3, "maxItems": 3,
                    },
                ],
            },
            "placement_region": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object", "additionalProperties": False,
                        "required": ["label", "bbox_xyxy"],
                        "properties": {
                            "label": {"type": "string"},
                            "bbox_xyxy": {
                                "type": "array", "items": {"type": "number"},
                                "minItems": 4, "maxItems": 4,
                            },
                        },
                    },
                ],
            },
            "placement_avoid_regions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["label", "bbox_xyxy"],
                    "properties": {
                        "label": {"type": "string"},
                        "bbox_xyxy": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": 4, "maxItems": 4,
                        },
                    },
                },
            },
            "placement_verification": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "require_upright", "upright_axis",
                            "orientation_symmetry", "symmetry_axis",
                        ],
                        "properties": {
                            "require_upright": {"type": "boolean"},
                            "upright_axis": {
                                "type": "string",
                                "enum": [
                                    "principal_long", "principal_middle",
                                    "principal_short",
                                ],
                            },
                            "orientation_symmetry": {
                                "type": "string",
                                "enum": ["none", "axial"],
                            },
                            "symmetry_axis": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "string",
                                        "enum": [
                                            "principal_long", "principal_middle",
                                            "principal_short",
                                        ],
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
            "constraints": {"type": "array", "items": {"type": "string"}},
        },
    },
}


def _output_schema(
    coordinate_scale: float,
    grounding_scope: str,
) -> dict[str, object]:
    """Bind provider coordinates and semantic fields to one explicit task stage."""
    scale = float(coordinate_scale)
    if scale not in _BBOX_COORDINATE_SCALES.values():
        raise ValueError("bbox coordinate scale is unsupported")
    if grounding_scope not in GROUNDING_SCOPES:
        raise ValueError("grounding scope is unsupported")
    schema = copy.deepcopy(_OUTPUT_SCHEMA)
    properties = schema["schema"]["properties"]
    boxes = (
        properties["target"]["properties"]["bbox_xyxy"],
        properties["grasp_part"]["anyOf"][1]["properties"]["bbox_xyxy"],
        properties["avoid_regions"]["items"]["properties"]["bbox_xyxy"],
        properties["placement_region"]["anyOf"][1]["properties"]["bbox_xyxy"],
        properties["placement_avoid_regions"]["items"]["properties"]["bbox_xyxy"],
    )
    for box in boxes:
        box["items"]["type"] = "integer" if scale == 1000.0 else "number"
        box["items"]["minimum"] = 0.0
        box["items"]["maximum"] = scale
    if grounding_scope in {"grasp_only", "grasp_for_place"}:
        properties["placement_region"] = {"type": "null"}
        properties["placement_avoid_regions"] = {
            "type": "array",
            "maxItems": 0,
        }
        if grounding_scope == "grasp_only":
            properties["placement_verification"] = {"type": "null"}
        else:
            properties["placement_verification"] = copy.deepcopy(
                _OUTPUT_SCHEMA["schema"]["properties"]
                ["placement_verification"]["anyOf"][1],
            )
    else:
        properties["grasp_part"] = {"type": "null"}
        properties["avoid_regions"] = {"type": "array", "maxItems": 0}
        properties["preferred_approach_camera"] = {"type": "null"}
        properties["placement_verification"] = {"type": "null"}
    return schema


def _parse_bbox(
    value: object,
    *,
    coordinate_scale: float,
    field: str,
) -> NormalizedBox:
    """Parse one declared wire-space bbox with a bounded field-path error."""
    try:
        return NormalizedBox.parse(
            value,
            coordinate_scale=coordinate_scale,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}: {error}") from error


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


def _raise_if_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise VLMCancellationError("VLM request was canceled")


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


def _message_text(response: Mapping[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = [
            item.get("text", "") for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "".join(fragments)
    raise ValueError("response message has no text content")


def _parse_json_text(text: str) -> Mapping[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1:] if first_newline >= 0 else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(stripped[start:end + 1])
    if not isinstance(result, Mapping):
        raise ValueError("VLM output is not a JSON object")
    return result


def _box_area(box: NormalizedBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _covered_area(region: NormalizedBox, covers: Sequence[NormalizedBox]) -> float:
    """Return the exact union area of axis-aligned covers clipped to region."""
    clipped = []
    for cover in covers:
        x1 = max(region.x1, cover.x1)
        y1 = max(region.y1, cover.y1)
        x2 = min(region.x2, cover.x2)
        y2 = min(region.y2, cover.y2)
        if x2 > x1 and y2 > y1:
            clipped.append((x1, y1, x2, y2))
    if not clipped:
        return 0.0
    x_edges = sorted({edge for box in clipped for edge in (box[0], box[2])})
    area = 0.0
    for x1, x2 in zip(x_edges, x_edges[1:]):
        if x2 <= x1:
            continue
        intervals = sorted(
            (box[1], box[3])
            for box in clipped
            if box[0] < x2 and box[2] > x1
        )
        if not intervals:
            continue
        covered_y = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_y += current_end - current_start
                current_start, current_end = start, end
        covered_y += current_end - current_start
        area += (x2 - x1) * covered_y
    return area


class OpenRouterVLM:
    """Ground a language goal and reason about grasp regions, never motion."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        models: Sequence[str] | None = None,
        base_url: str | None = None,
        local_grounding_url: str | None = None,
        local_grounding_timeout_s: float = 1.25,
        local_transport: Transport | None = None,
        timeout_s: float = 25.0,
        model_timeouts_s: Mapping[str, float] | Sequence[float] | None = None,
        model_bbox_coordinate_spaces: (
            Mapping[str, str] | Sequence[str] | None
        ) = None,
        transport: Transport | None = None,
        min_confidence: float | None = None,
        max_target_area_ratio: float | None = None,
        min_target_border_margin_ratio: float = 0.002,
        max_semantic_conflict_coverage_ratio: float = 0.95,
        provider_retries: int | None = None,
        timeout_retries: int | None = None,
        hedge_delay_s: float = 0.0,
        attempt_callback: AttemptCallback | None = None,
    ) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        if models is None:
            primary = os.environ.get("Z_MANIP_VLM_MODEL", DEFAULT_MODELS[0])
            fallback = os.environ.get("Z_MANIP_VLM_FALLBACK", DEFAULT_MODELS[1])
            models = tuple(model for model in (primary, fallback) if model)
        self.models = tuple(dict.fromkeys(models))
        if not self.models:
            raise ValueError("at least one VLM model is required")
        self.base_url = (base_url or os.environ.get(
            "Z_MANIP_VLM_BASE_URL", "https://openrouter.ai/api/v1",
        )).rstrip("/")
        local_url = (
            os.environ.get("Z_MANIP_LOCAL_GROUNDING_URL", "")
            if local_grounding_url is None
            else local_grounding_url
        ).strip()
        self.local_grounding_url = (
            _validated_loopback_grounding_url(local_url) if local_url else None
        )
        self.local_grounding_timeout_s = float(local_grounding_timeout_s)
        if (
            not np.isfinite(self.local_grounding_timeout_s)
            or not 0.05 <= self.local_grounding_timeout_s <= 5.0
        ):
            raise ValueError("local grounding timeout must be within [0.05, 5] seconds")
        self.local_transport = local_transport or _loopback_grounding_transport
        self._local_transport_accepts_cancellation = _transport_accepts_cancel_event(
            self.local_transport,
        )
        self.timeout_s = float(timeout_s)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if model_timeouts_s is None:
            configured_timeouts = (self.timeout_s,) * len(self.models)
        elif isinstance(model_timeouts_s, Mapping):
            configured_timeouts = tuple(
                float(model_timeouts_s.get(model, self.timeout_s))
                for model in self.models
            )
        else:
            configured_timeouts = tuple(float(value) for value in model_timeouts_s)
            if len(configured_timeouts) != len(self.models):
                raise ValueError("model_timeouts_s must match the configured model count")
        if not all(
            np.isfinite(value) and value > 0.0 for value in configured_timeouts
        ):
            raise ValueError("model timeouts must be finite and positive")
        self.model_timeouts_s = dict(zip(self.models, configured_timeouts))
        if model_bbox_coordinate_spaces is None:
            configured_coordinate_spaces = ("normalized_0_1",) * len(self.models)
        elif isinstance(model_bbox_coordinate_spaces, Mapping):
            configured_coordinate_spaces = tuple(
                str(model_bbox_coordinate_spaces.get(model, "normalized_0_1"))
                for model in self.models
            )
        else:
            configured_coordinate_spaces = tuple(
                str(value) for value in model_bbox_coordinate_spaces
            )
            if len(configured_coordinate_spaces) != len(self.models):
                raise ValueError(
                    "model_bbox_coordinate_spaces must match the configured model count",
                )
        if not all(
            value in _BBOX_COORDINATE_SCALES
            for value in configured_coordinate_spaces
        ):
            raise ValueError(
                "model_bbox_coordinate_spaces contains an unsupported coordinate space",
            )
        self.model_bbox_coordinate_spaces = dict(zip(
            self.models,
            configured_coordinate_spaces,
        ))
        self.transport = transport or _curl_transport
        self._transport_accepts_cancellation = _transport_accepts_cancel_event(
            self.transport,
        )
        self.min_confidence = float(
            os.environ.get("Z_MANIP_VLM_MIN_CONFIDENCE", "0.20")
            if min_confidence is None else min_confidence
        )
        self.max_target_area_ratio = float(
            os.environ.get("Z_MANIP_VLM_MAX_TARGET_AREA_RATIO", "0.95")
            if max_target_area_ratio is None else max_target_area_ratio
        )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 < self.max_target_area_ratio <= 1.0:
            raise ValueError("max_target_area_ratio must be in (0, 1]")
        self.min_target_border_margin_ratio = float(
            min_target_border_margin_ratio,
        )
        if not 0.0 <= self.min_target_border_margin_ratio < 0.5:
            raise ValueError(
                "min_target_border_margin_ratio must be within [0, 0.5)",
            )
        self.max_semantic_conflict_coverage_ratio = float(
            max_semantic_conflict_coverage_ratio,
        )
        if not (
            np.isfinite(self.max_semantic_conflict_coverage_ratio)
            and 0.0 < self.max_semantic_conflict_coverage_ratio <= 1.0
        ):
            raise ValueError(
                "max_semantic_conflict_coverage_ratio must be in (0, 1]",
            )
        retry_value = (
            os.environ.get("Z_MANIP_VLM_PROVIDER_RETRIES", "1")
            if provider_retries is None else provider_retries
        )
        if isinstance(retry_value, bool):
            raise ValueError("provider_retries must be an integer in [0, 3]")
        try:
            self.provider_retries = int(retry_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "provider_retries must be an integer in [0, 3]",
            ) from error
        if not 0 <= self.provider_retries <= 3:
            raise ValueError("provider_retries must be an integer in [0, 3]")
        timeout_retry_value = (
            os.environ.get("Z_MANIP_VLM_TIMEOUT_RETRIES", "0")
            if timeout_retries is None else timeout_retries
        )
        if isinstance(timeout_retry_value, bool):
            raise ValueError("timeout_retries must be an integer in [0, 3]")
        try:
            self.timeout_retries = int(timeout_retry_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "timeout_retries must be an integer in [0, 3]",
            ) from error
        if not 0 <= self.timeout_retries <= 3:
            raise ValueError("timeout_retries must be an integer in [0, 3]")
        self.hedge_delay_s = float(hedge_delay_s)
        if not np.isfinite(self.hedge_delay_s) or not 0.0 <= self.hedge_delay_s <= 5.0:
            raise ValueError("hedge_delay_s must be finite and within [0, 5]")
        self.attempt_callback = attempt_callback
        self.validation_retries = max(
            0, int(os.environ.get("Z_MANIP_VLM_VALIDATION_RETRIES", "1")),
        )

    def locate_and_reason(
        self,
        jpeg: bytes,
        instruction: str,
        *,
        grounding_scope: str = "grasp_only",
        mime_type: str = "image/jpeg",
        cancel_event: threading.Event | None = None,
    ) -> AffordanceResult:
        """Return target/part grounding and semantic constraints for one frame."""
        call_cancel_event = cancel_event or threading.Event()
        _raise_if_cancelled(call_cancel_event)
        if not jpeg or not instruction.strip():
            raise VLMError("VLM request needs a non-empty image and instruction")
        if grounding_scope not in GROUNDING_SCOPES:
            raise VLMError("VLM grounding scope is unsupported")
        if mime_type not in ("image/jpeg", "image/png", "image/webp"):
            raise VLMError(f"unsupported image MIME type {mime_type!r}")
        failures: list[str] = []
        if self.local_grounding_url and grounding_scope == "grasp_only":
            local_model = "local/grounding-dino-tiny"
            local_started = time.monotonic()
            self._emit_attempt(local_model, 1, "start", 0.0)
            try:
                local_payload = {
                    "schema": "z_manip.local_grounding_request.v1",
                    "instruction": instruction.strip(),
                    "image_base64": base64.b64encode(jpeg).decode("ascii"),
                }
                local_args = (
                    f"{self.local_grounding_url}/ground",
                    local_payload,
                    {},
                    self.local_grounding_timeout_s,
                )
                if self._local_transport_accepts_cancellation:
                    local_response = self.local_transport(
                        *local_args,
                        call_cancel_event,
                    )
                else:
                    local_response = self.local_transport(*local_args)
                _raise_if_cancelled(call_cancel_event)
                if local_response.get("schema") != "z_manip.local_grounding_response.v1":
                    raise ValueError("local grounding response schema is invalid")
                target = local_response.get("target")
                if not isinstance(target, Mapping):
                    raise ValueError("local grounding response has no target")
                model = str(local_response.get("model", local_model)).strip() or local_model
                parsed = {
                    "target": target,
                    "grasp_part": None,
                    "avoid_regions": [],
                    "preferred_approach_camera": None,
                    "placement_region": None,
                    "placement_avoid_regions": [],
                    "placement_verification": None,
                    "constraints": [
                        "local open-vocabulary detector; use full observed object geometry",
                    ],
                }
                local_result = self._result(
                    model,
                    parsed,
                    time.monotonic() - local_started,
                    min_confidence=self.min_confidence,
                    max_target_area_ratio=self.max_target_area_ratio,
                    min_target_border_margin_ratio=(
                        self.min_target_border_margin_ratio
                    ),
                    max_semantic_conflict_coverage_ratio=(
                        self.max_semantic_conflict_coverage_ratio
                    ),
                    bbox_coordinate_scale=1.0,
                    grounding_scope=grounding_scope,
                )
                self._emit_attempt(
                    local_model,
                    1,
                    "success",
                    time.monotonic() - local_started,
                )
                return local_result
            except VLMCancellationError:
                self._emit_attempt(
                    local_model,
                    1,
                    "canceled",
                    time.monotonic() - local_started,
                )
                raise
            except Exception as error:
                _raise_if_cancelled(call_cancel_event)
                detail = self._bounded_error_detail(error)
                self._emit_attempt(
                    local_model,
                    1,
                    "fallback",
                    time.monotonic() - local_started,
                    detail,
                )
                failures.append(f"{local_model}: {detail}")
        if not self.api_key:
            suffix = "; ".join(failures)
            raise VLMError(
                "OPENROUTER_API_KEY is not configured"
                + (f" after local grounding failed: {suffix}" if suffix else "")
            )
        scope_instruction = {
            "grasp_only": (
                "This is a grasp-only pass. Ground the requested physical object and its "
                "grasp affordance. Return placement_region null, placement_avoid_regions "
                "empty, and placement_verification null."
            ),
            "grasp_for_place": (
                "This is a grasp pass for a later observed placement. Ground the requested "
                "physical object and its grasp affordance. Do not reason about the support "
                "yet: return placement_region null and placement_avoid_regions empty. "
                "placement_verification is mandatory and describes the observed object axes "
                "needed to verify the final orientation."
            ),
            "place_support": (
                "This is a fresh placement-support pass after the object was grasped. The "
                "target bbox must continue to identify the visible grasped object so its live "
                "pose can be checked against the frozen object model. placement_region must be "
                "one visible empty supported area on the requested support surface, and "
                "placement_avoid_regions must cover occupied, unsupported, fragile, or edge "
                "areas. Return grasp_part null, avoid_regions empty, "
                "preferred_approach_camera null, and placement_verification null because the "
                "object axes were frozen from the earlier grasp observation."
            ),
        }[grounding_scope]
        image_url = f"data:{mime_type};base64,{base64.b64encode(jpeg).decode('ascii')}"
        for model in self.models:
            _raise_if_cancelled(call_cancel_event)
            started = time.monotonic()
            coordinate_space = self.model_bbox_coordinate_spaces[model]
            coordinate_scale = _BBOX_COORDINATE_SCALES[coordinate_space]
            coordinate_instruction = (
                "Coordinates are normalized xyxy in [0,1]."
                if coordinate_space == "normalized_0_1"
                else (
                    "Coordinates are integer relative xyxy in [0,1000], matching the "
                    "Qwen native grounding space; they are not image pixels."
                )
            )
            payload = {
                "model": model,
                "temperature": 0,
                "max_completion_tokens": 256,
                "reasoning": {
                    "effort": "none",
                    "exclude": True,
                },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": _output_schema(
                        coordinate_scale,
                        grounding_scope,
                    ),
                },
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You ground targets for a mobile manipulation robot. Return only "
                            f"the requested JSON. {coordinate_instruction} All bbox fields "
                            "must use that one coordinate space. "
                            f"{scope_instruction} "
                            "target.bbox_xyxy must tightly cover the entire visible physical "
                            "object from its topmost to bottommost and leftmost to rightmost "
                            "pixels, never only a grasp part, connector, handle, or protrusion. "
                            "grasp_part may be smaller than target. "
                            "Choose one physical instance. grasp_part is the safest visible "
                            "region to grip; avoid_regions include handles, openings, fragile "
                            "parts and occluders when relevant. preferred_approach_camera is "
                            "the gripper travel direction toward contact in optical coordinates "
                            "+x right, +y down, +z forward. Do not invent hidden geometry or "
                            "robot motion. When placement_verification is requested, it must "
                            "explicitly state whether natural upright orientation is required and "
                            "select its observable "
                            "object axis as principal_long, principal_middle, or principal_short. "
                            "Set orientation_symmetry to axial only for a rotationally symmetric "
                            "object and then select its symmetry_axis; for full asymmetric orientation "
                            "use none and symmetry_axis null. Principal axes are undirected geometric "
                            "axes, not camera or robot axes. Do not infer hidden geometry or use an "
                            "object-name default. Use null when visual evidence is insufficient."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction.strip()},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "Z-Manipulation-Stack",
            }
            max_retries = max(
                self.validation_retries,
                self.provider_retries,
                self.timeout_retries,
            )
            for attempt in range(max_retries + 1):
                _raise_if_cancelled(call_cancel_event)
                attempt_number = attempt + 1
                attempt_started = time.monotonic()
                self._emit_attempt(model, attempt_number, "start", 0.0)
                try:
                    transport_args = (
                        f"{self.base_url}/chat/completions",
                        payload,
                        headers,
                        self.model_timeouts_s[model],
                    )
                    response = self._request_transport(
                        transport_args,
                        call_cancel_event,
                    )
                    _raise_if_cancelled(call_cancel_event)
                    parsed = _parse_json_text(_message_text(response))
                    _raise_if_cancelled(call_cancel_event)
                    result = self._result(
                        model,
                        parsed,
                        time.monotonic() - started,
                        min_confidence=self.min_confidence,
                        max_target_area_ratio=self.max_target_area_ratio,
                        min_target_border_margin_ratio=(
                            self.min_target_border_margin_ratio
                        ),
                        max_semantic_conflict_coverage_ratio=(
                            self.max_semantic_conflict_coverage_ratio
                        ),
                        bbox_coordinate_scale=coordinate_scale,
                        grounding_scope=grounding_scope,
                    )
                    _raise_if_cancelled(call_cancel_event)
                    self._emit_attempt(
                        model,
                        attempt_number,
                        "success",
                        time.monotonic() - attempt_started,
                    )
                    return result
                except VLMCancellationError:
                    self._emit_attempt(
                        model,
                        attempt_number,
                        "canceled",
                        time.monotonic() - attempt_started,
                    )
                    raise
                except ValueError as error:
                    _raise_if_cancelled(call_cancel_event)
                    self._emit_attempt(
                        model,
                        attempt_number,
                        "validation_failure",
                        time.monotonic() - attempt_started,
                        self._bounded_error_detail(error),
                    )
                    # A model can occasionally emit a degenerate bbox despite the
                    # schema prompt. Retry validation failures once; transport and
                    # provider failures go straight to the next configured model.
                    if attempt < self.validation_retries:
                        if call_cancel_event.wait(0.2):
                            raise VLMCancellationError("VLM request was canceled")
                        continue
                    failures.append(f"{model}: {type(error).__name__}: {error}")
                except TimeoutError as error:
                    _raise_if_cancelled(call_cancel_event)
                    self._emit_attempt(
                        model,
                        attempt_number,
                        "timeout",
                        time.monotonic() - attempt_started,
                        self._bounded_error_detail(error),
                    )
                    if attempt < self.timeout_retries:
                        continue
                    failures.append(f"{model}: {type(error).__name__}: {error}")
                except VLMTransportError as error:
                    _raise_if_cancelled(call_cancel_event)
                    self._emit_attempt(
                        model,
                        attempt_number,
                        "provider_error",
                        time.monotonic() - attempt_started,
                        self._bounded_error_detail(error),
                    )
                    if error.retryable and attempt < self.provider_retries:
                        if call_cancel_event.wait(0.5):
                            raise VLMCancellationError("VLM request was canceled")
                        continue
                    failures.append(f"{model}: {type(error).__name__}: {error}")
                except Exception as error:  # provider/transport failures degrade
                    _raise_if_cancelled(call_cancel_event)
                    self._emit_attempt(
                        model,
                        attempt_number,
                        "provider_error",
                        time.monotonic() - attempt_started,
                        self._bounded_error_detail(error),
                    )
                    failures.append(f"{model}: {type(error).__name__}: {error}")
                break
        raise VLMError("all VLM models failed: " + "; ".join(failures))

    def _request_transport(
        self,
        transport_args: tuple[object, ...],
        call_cancel_event: threading.Event,
    ) -> Mapping[str, object]:
        """Race two identical cancellable requests after a short hedge delay.

        OpenRouter queue latency is the dominant realtime tail. A delayed
        duplicate leaves the usual single fast request unchanged while making
        an isolated slow provider queue unlikely to block the robot pipeline.
        The losing curl process is canceled and joined before returning.
        """

        if self.hedge_delay_s <= 0.0 or not self._transport_accepts_cancellation:
            if self._transport_accepts_cancellation:
                return self.transport(*transport_args, call_cancel_event)
            return self.transport(*transport_args)

        outcomes: queue.Queue[tuple[int, Mapping[str, object] | None, BaseException | None]] = (
            queue.Queue()
        )
        local_cancel = (threading.Event(), threading.Event())
        threads: list[threading.Thread] = []

        def worker(index: int) -> None:
            try:
                response = self.transport(
                    *transport_args,
                    _CombinedCancelEvent(call_cancel_event, local_cancel[index]),
                )
                outcomes.put((index, response, None))
            except BaseException as error:
                outcomes.put((index, None, error))

        def start(index: int) -> None:
            thread = threading.Thread(
                target=worker,
                args=(index,),
                name=f"openrouter_hedge_{index}",
                daemon=False,
            )
            threads.append(thread)
            thread.start()

        def stop_and_join(winner: int | None = None) -> None:
            for index, event in enumerate(local_cancel):
                if index != winner:
                    event.set()
            for thread in threads:
                thread.join(timeout=max(1.0, self.timeout_s + 1.0))
                if thread.is_alive():
                    raise RuntimeError("hedged VLM transport did not stop")

        start(0)
        first_error: BaseException | None = None
        hedge_deadline = time.monotonic() + self.hedge_delay_s
        while time.monotonic() < hedge_deadline:
            if call_cancel_event.is_set():
                stop_and_join()
                raise VLMCancellationError("VLM request was canceled")
            try:
                index, response, error = outcomes.get(
                    timeout=min(0.05, max(0.0, hedge_deadline - time.monotonic())),
                )
            except queue.Empty:
                continue
            if error is None and response is not None:
                stop_and_join(index)
                return response
            first_error = error
            break
        start(1)
        completed = 1 if first_error is not None else 0
        while completed < 2:
            if call_cancel_event.is_set():
                stop_and_join()
                raise VLMCancellationError("VLM request was canceled")
            try:
                index, response, error = outcomes.get(timeout=0.05)
            except queue.Empty:
                continue
            completed += 1
            if error is None and response is not None:
                stop_and_join(index)
                return response
            if first_error is None:
                first_error = error
        stop_and_join()
        assert first_error is not None
        raise first_error

    def _bounded_error_detail(self, error: BaseException) -> str:
        detail = f"{type(error).__name__}: {error}".replace("\n", " ")
        if self.api_key:
            detail = detail.replace(self.api_key, "[REDACTED]")
        return detail[-384:]

    def _emit_attempt(
        self,
        model: str,
        attempt: int,
        outcome: str,
        elapsed_s: float,
        detail: str = "",
    ) -> None:
        callback = self.attempt_callback
        if callback is None:
            return
        try:
            callback(VLMAttemptEvent(
                model=model,
                attempt=attempt,
                outcome=outcome,
                elapsed_s=max(0.0, float(elapsed_s)),
                detail=detail,
            ))
        except Exception:
            # Diagnostics must never change grounding behavior.
            return

    @staticmethod
    def _result(
        model: str,
        value: Mapping[str, object],
        latency_s: float,
        *,
        min_confidence: float = 0.20,
        max_target_area_ratio: float = 0.95,
        min_target_border_margin_ratio: float = 0.002,
        max_semantic_conflict_coverage_ratio: float = 0.95,
        bbox_coordinate_scale: float = 1.0,
        grounding_scope: str = "grasp_only",
    ) -> AffordanceResult:
        if grounding_scope not in GROUNDING_SCOPES:
            raise ValueError("grounding scope is unsupported")
        target = value.get("target")
        if not isinstance(target, Mapping):
            raise ValueError("VLM result has no target")
        label = str(target.get("label", "")).strip()
        if not label:
            raise ValueError("VLM target label is empty")
        target_box = _parse_bbox(
            target.get("bbox_xyxy"),
            coordinate_scale=bbox_coordinate_scale,
            field="target.bbox_xyxy",
        )
        confidence = float(target.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("VLM confidence must be in [0, 1]")
        area_ratio = (target_box.x2 - target_box.x1) * (target_box.y2 - target_box.y1)
        if confidence < min_confidence:
            raise ValueError(
                f"VLM confidence {confidence:.3f} is below the configured minimum "
                f"{min_confidence:.3f}"
            )
        if area_ratio > max_target_area_ratio:
            raise ValueError(
                f"VLM target box area {area_ratio:.3f} exceeds the configured maximum "
                f"{max_target_area_ratio:.3f}"
            )
        if grounding_scope == "grasp_only" and (
            target_box.x1 <= min_target_border_margin_ratio
            or target_box.y1 <= min_target_border_margin_ratio
            or target_box.x2 >= 1.0 - min_target_border_margin_ratio
            or target_box.y2 >= 1.0 - min_target_border_margin_ratio
        ):
            raise ValueError(
                "VLM grasp target touches the image border; the complete object "
                "must be visible before tracking",
            )

        grasp_part = value.get("grasp_part")
        part_label, part_box = None, None
        if grasp_part is not None:
            if not isinstance(grasp_part, Mapping):
                raise ValueError("grasp_part must be an object or null")
            part_label = str(grasp_part.get("label", "")).strip()
            if not part_label:
                raise ValueError("grasp part label is empty")
            part_box = _parse_bbox(
                grasp_part.get("bbox_xyxy"),
                coordinate_scale=bbox_coordinate_scale,
                field="grasp_part.bbox_xyxy",
            )
        avoid = []
        for index, region in enumerate(value.get("avoid_regions", [])):
            if not isinstance(region, Mapping):
                raise ValueError("avoid region must be an object")
            avoid.append(AvoidRegion(
                str(region.get("label", "")).strip(),
                _parse_bbox(
                    region.get("bbox_xyxy"),
                    coordinate_scale=bbox_coordinate_scale,
                    field=f"avoid_regions[{index}].bbox_xyxy",
                ),
            ))
        if part_box is not None:
            conflict_ratio = _covered_area(
                part_box,
                tuple(region.bbox for region in avoid),
            ) / _box_area(part_box)
            if conflict_ratio >= max_semantic_conflict_coverage_ratio:
                raise ValueError(
                    "grasp_part is covered by avoid_regions "
                    f"({conflict_ratio:.3f})",
                )

        preferred_value = value.get("preferred_approach_camera")
        preferred = None
        if preferred_value is not None:
            direction = np.asarray(preferred_value, dtype=float)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)):
                raise ValueError("preferred approach must be a finite three-vector")
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                raise ValueError("preferred approach must be nonzero")
            preferred = tuple((direction / norm).tolist())
        placement = value.get("placement_region")
        placement_label, placement_box = None, None
        if placement is not None:
            if not isinstance(placement, Mapping):
                raise ValueError("placement_region must be an object or null")
            placement_label = str(placement.get("label", "")).strip()
            if not placement_label:
                raise ValueError("placement region label is empty")
            placement_box = _parse_bbox(
                placement.get("bbox_xyxy"),
                coordinate_scale=bbox_coordinate_scale,
                field="placement_region.bbox_xyxy",
            )
        placement_avoid = []
        for index, region in enumerate(value.get("placement_avoid_regions", [])):
            if not isinstance(region, Mapping):
                raise ValueError("placement avoid region must be an object")
            placement_avoid.append(AvoidRegion(
                str(region.get("label", "")).strip(),
                _parse_bbox(
                    region.get("bbox_xyxy"),
                    coordinate_scale=bbox_coordinate_scale,
                    field=f"placement_avoid_regions[{index}].bbox_xyxy",
                ),
            ))
        if placement_box is not None and grounding_scope == "place_support":
            conflict_ratio = _covered_area(
                placement_box,
                tuple(region.bbox for region in placement_avoid),
            ) / _box_area(placement_box)
            if conflict_ratio >= max_semantic_conflict_coverage_ratio:
                raise ValueError(
                    "placement_region is covered by placement_avoid_regions "
                    f"({conflict_ratio:.3f})",
                )
        verification_value = value.get("placement_verification")
        verification = None
        if verification_value is not None:
            if not isinstance(verification_value, Mapping):
                raise ValueError("placement_verification must be an object or null")
            required_verification = {
                "require_upright", "upright_axis",
                "orientation_symmetry", "symmetry_axis",
            }
            if set(verification_value) != required_verification:
                raise ValueError("placement_verification fields are incomplete or unknown")
            symmetry_axis_value = verification_value.get("symmetry_axis")
            if symmetry_axis_value is not None and not isinstance(
                symmetry_axis_value,
                str,
            ):
                raise ValueError("symmetry_axis must be a string or null")
            verification = PlacementVerification(
                require_upright=verification_value.get("require_upright"),
                upright_axis=str(verification_value.get("upright_axis", "")),
                orientation_symmetry=str(
                    verification_value.get("orientation_symmetry", ""),
                ),
                symmetry_axis=symmetry_axis_value,
            )
        if (
            placement is not None
            and grounding_scope != "place_support"
            and verification is None
        ):
            raise ValueError(
                "visible placement requires explicit placement_verification"
            )
        if grounding_scope == "grasp_only" and (
            placement is not None or placement_avoid or verification is not None
        ):
            raise ValueError(
                "grasp_only must not return placement geometry or verification",
            )
        if grounding_scope == "grasp_for_place" and (
            placement is not None or placement_avoid
        ):
            raise ValueError(
                "grasp_for_place must not return placement geometry",
            )
        if grounding_scope == "grasp_for_place" and verification is None:
            raise ValueError(
                "grasp_for_place requires explicit placement_verification",
            )
        if grounding_scope == "place_support" and placement is None:
            raise ValueError("place_support requires a visible placement_region")
        if grounding_scope == "place_support" and (
            grasp_part is not None
            or avoid
            or preferred is not None
            or verification is not None
        ):
            raise ValueError(
                "place_support must not return grasp geometry or object verification",
            )
        constraints = tuple(str(item).strip() for item in value.get("constraints", []))
        return AffordanceResult(
            model=model,
            target_label=label,
            target_bbox=target_box,
            confidence=confidence,
            grasp_part_label=part_label,
            grasp_part_bbox=part_box,
            avoid_regions=tuple(avoid),
            preferred_approach_camera=preferred,
            placement_region_label=placement_label,
            placement_region_bbox=placement_box,
            placement_avoid_regions=tuple(placement_avoid),
            placement_verification=verification,
            constraints=constraints,
            latency_s=float(latency_s),
        )
