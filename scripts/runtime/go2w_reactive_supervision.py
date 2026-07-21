#!/usr/bin/env python3
"""Pure watchdog and JSONL replay helpers for the reactive mobile workflow.

This module has no ROS, WebRTC, CAN, or robot SDK dependency.  It exists so
historical depth-servo traces can exercise the same bounded-wait policy used by
the loopback workbench supervisor without constructing any actuator transport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


POSTURE_WAIT_PHASES = frozenset({
    "posture_adjust",
    "posture_shadow_verified",
    "posture_blocked",
})


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def posture_feedback_snapshot(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the depth-servo wrapper and the direct posture document."""

    wrapper = _mapping(runtime.get("posture_status"))
    document = _mapping(wrapper.get("document")) or wrapper
    feedback = _mapping(document.get("feedback"))
    wrapper_age = _finite(wrapper.get("age_s"))
    body = _mapping(document.get("body_height"))
    feedback_age = _finite(body.get("feedback_age_s"))
    age_s = wrapper_age if wrapper_age is not None else feedback_age
    return {
        "available": document.get("schema") == "z_manip.go2w_posture_status.v1",
        "fresh": feedback.get("fresh") is True,
        "age_s": age_s,
        "phase": str(document.get("phase", "unavailable")),
        "owner": str(document.get("command_owner", "unavailable")),
        "detail": str(document.get("detail", "")),
        "current_height_m": _finite(body.get("current_m")),
        "target_height_m": _finite(body.get("target_m")),
        "current_pitch_rad": _finite(
            _mapping(document.get("attitude")).get("current_pitch_rad")
        ),
        "target_pitch_rad": _finite(
            _mapping(document.get("attitude")).get("target_pitch_rad")
        ),
    }


def ownership_snapshot(runtime: Mapping[str, Any]) -> dict[str, str]:
    """Return one inspectable owner per actuator group.

    Missing feedback is reported as ``unavailable``; an arm-view intent alone
    is deliberately labelled ``intent_only`` rather than pretending an arm
    executor accepted it.
    """

    phase = str(runtime.get("phase", "idle"))
    posture = posture_feedback_snapshot(runtime)
    reactive = _mapping(runtime.get("reactive"))
    arm_view = _mapping(reactive.get("arm_view"))
    arm_feedback = _mapping(runtime.get("arm_view_status"))
    output = _mapping(runtime.get("output"))
    published_linear = _finite(output.get("published_linear_x")) or 0.0
    published_yaw = _finite(output.get("published_angular_z")) or 0.0
    base_owner = "visual_servo" if (
        phase in {"approach", "base_approach"}
        or abs(published_linear) > 1e-9
        or abs(published_yaw) > 1e-9
    ) else "zero_hold"
    arm_owner = str(arm_feedback.get("owner", ""))
    if not arm_owner:
        arm_owner = (
            "intent_only"
            if str(arm_view.get("mode", "hold")) not in {"", "hold"}
            else "none"
        )
    optimizer = _mapping(runtime.get("optimizer"))
    optimizer_owner = str(optimizer.get("owner", "unavailable"))
    return {
        "base": base_owner,
        "body": str(posture["owner"]),
        "arm_view": arm_owner,
        "optimizer": optimizer_owner,
    }


@dataclass(frozen=True)
class ReactiveWatchdogConfig:
    posture_wait_timeout_s: float = 12.0
    feedback_freshness_s: float = 0.75

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.posture_wait_timeout_s)
            or self.posture_wait_timeout_s <= 0
        ):
            raise ValueError("posture wait timeout must be finite and positive")
        if (
            not math.isfinite(self.feedback_freshness_s)
            or self.feedback_freshness_s <= 0
        ):
            raise ValueError("feedback freshness must be finite and positive")


@dataclass(frozen=True)
class ReactiveWatchdogDecision:
    phase: str
    phase_elapsed_s: float
    deadline_s: float | None
    timed_out: bool
    code: str | None
    message: str
    feedback: dict[str, Any]
    owners: dict[str, str]

    def document(self) -> dict[str, Any]:
        return asdict(self)


class ReactivePhaseWatchdog:
    """Bound one blocking reactive phase and expose the exact missing owner."""

    def __init__(self, config: ReactiveWatchdogConfig | None = None) -> None:
        self.config = config or ReactiveWatchdogConfig()
        self.reset()

    def reset(self) -> None:
        self._phase: str | None = None
        self._phase_started_s: float | None = None
        self._last = ReactiveWatchdogDecision(
            phase="idle",
            phase_elapsed_s=0.0,
            deadline_s=None,
            timed_out=False,
            code=None,
            message="idle",
            feedback={},
            owners={},
        )

    @property
    def last(self) -> ReactiveWatchdogDecision:
        return self._last

    def observe(
        self,
        runtime: Mapping[str, Any],
        *,
        now_s: float,
    ) -> ReactiveWatchdogDecision:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("watchdog timestamp must be finite")
        phase = str(
            runtime.get("phase")
            or _mapping(runtime.get("reactive")).get("phase")
            or "idle"
        )
        phase_group = "posture_wait" if phase in POSTURE_WAIT_PHASES else phase
        if phase_group != self._phase:
            self._phase = phase_group
            self._phase_started_s = now
        assert self._phase_started_s is not None
        elapsed = max(0.0, now - self._phase_started_s)
        feedback = posture_feedback_snapshot(runtime)
        owners = ownership_snapshot(runtime)
        timed_out = phase in POSTURE_WAIT_PHASES and elapsed >= self.config.posture_wait_timeout_s
        code: str | None = None
        message = "phase is progressing within its bounded wait"
        if timed_out:
            age_s = feedback.get("age_s")
            feedback_stale = (
                not feedback.get("available")
                or not feedback.get("fresh")
                or not isinstance(age_s, (int, float))
                or float(age_s) > self.config.feedback_freshness_s
            )
            if feedback_stale:
                code = "POSTURE_FEEDBACK_TIMEOUT"
                message = (
                    "posture executor feedback is unavailable or stale; "
                    "reactive control degraded to a stationary terminal state"
                )
            elif owners.get("arm_view") == "intent_only":
                code = "ARM_VIEW_FEEDBACK_TIMEOUT"
                message = (
                    "arm-view intent has no measured executor owner; reactive "
                    "control degraded to a stationary terminal state"
                )
            else:
                code = "POSTURE_SETTLE_TIMEOUT"
                message = (
                    "measured posture did not settle before the bounded deadline; "
                    "reactive control degraded to a stationary terminal state"
                )
        self._last = ReactiveWatchdogDecision(
            phase=phase,
            phase_elapsed_s=elapsed,
            deadline_s=(
                self.config.posture_wait_timeout_s
                if phase in POSTURE_WAIT_PHASES else None
            ),
            timed_out=timed_out,
            code=code,
            message=message,
            feedback=feedback,
            owners=owners,
        )
        return self._last


def replay_trace(
    records: Iterable[Mapping[str, Any]],
    *,
    stall_threshold_s: float = 5.0,
) -> dict[str, Any]:
    """Summarize contiguous phase spans and identify zero-command stalls."""

    rows = [record for record in records if isinstance(record, Mapping)]
    spans: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for record in rows:
        stamp_ns = record.get("updated_unix_ns")
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            continue
        phase = str(record.get("phase", "unknown"))
        output = _mapping(record.get("output"))
        linear = _finite(output.get("published_linear_x")) or 0.0
        yaw = _finite(output.get("published_angular_z")) or 0.0
        reason = str(output.get("reason", ""))
        if active is None or active["phase"] != phase:
            if active is not None:
                spans.append(active)
            active = {
                "phase": phase,
                "start_unix_ns": stamp_ns,
                "end_unix_ns": stamp_ns,
                "samples": 1,
                "all_zero_command": abs(linear) <= 1e-9 and abs(yaw) <= 1e-9,
                "last_reason": reason,
            }
        else:
            active["end_unix_ns"] = stamp_ns
            active["samples"] += 1
            active["all_zero_command"] = bool(active["all_zero_command"]) and (
                abs(linear) <= 1e-9 and abs(yaw) <= 1e-9
            )
            if reason:
                active["last_reason"] = reason
    if active is not None:
        spans.append(active)
    stalls: list[dict[str, Any]] = []
    for span in spans:
        duration_s = max(0.0, (span["end_unix_ns"] - span["start_unix_ns"]) / 1e9)
        span["duration_s"] = duration_s
        if (
            span["phase"] in POSTURE_WAIT_PHASES
            and span["all_zero_command"]
            and duration_s >= stall_threshold_s
        ):
            stalls.append({
                **span,
                "code": "POSTURE_WAIT_STALL",
                "recommended_terminal_phase": "degraded",
            })
    return {
        "schema": "z_manip.reactive_trace_replay.v1",
        "record_count": len(rows),
        "span_count": len(spans),
        "spans": spans,
        "stalls": stalls,
        "passed": not stalls,
    }


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"trace line {line_number} is not an object")
            records.append(value)
    return records
