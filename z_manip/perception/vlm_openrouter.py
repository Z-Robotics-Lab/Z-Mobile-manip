"""OpenRouterVLM orchestrator: config, the local-then-remote attempt loop,
prompt/schema assembly, and hedged dual-request transport racing.
"""

from __future__ import annotations

import base64
import os
import queue
import threading
import time
from typing import Mapping, Sequence

import numpy as np

from z_manip.perception.vlm_errors import (
    VLMCancellationError,
    VLMError,
    VLMTransportError,
    _CombinedCancelEvent,
    _raise_if_cancelled,
)
from z_manip.perception.vlm_result_parsing import (
    _message_text,
    _parse_json_text,
    parse_affordance_result,
)
from z_manip.perception.vlm_schema import _output_schema
from z_manip.perception.vlm_transport import (
    AttemptCallback,
    Transport,
    _curl_transport,
    _loopback_grounding_transport,
    _transport_accepts_cancel_event,
    _validated_loopback_grounding_url,
)
from z_manip.perception.vlm_types import (
    _BBOX_COORDINATE_SCALES,
    GROUNDING_SCOPES,
    AffordanceResult,
    VLMAttemptEvent,
)


DEFAULT_MODELS = (
    "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen/qwen3.5-35b-a3b",
)


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
            os.environ.get("Z_MANIP_VLM_MIN_CONFIDENCE", "0.15")
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
            local_model = "local/yoloe-11s-seg"
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
                        "local YOLOE open-vocabulary instance detector; use full observed object geometry",
                    ],
                }
                local_result = parse_affordance_result(
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
                "X-Title": "Z-Mobile-Manip",
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
                    result = parse_affordance_result(
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


__all__ = ["DEFAULT_MODELS", "OpenRouterVLM"]
