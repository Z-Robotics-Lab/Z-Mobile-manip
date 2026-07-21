"""Pure Go2W body-posture command contracts and arbitration.

This module intentionally contains no ROS, WebRTC, or Unitree imports.  It is
the testable boundary between reactive target geometry and the NUC transport:

* shadow mode computes the exact SPORT requests but cannot transmit them;
* live mode requires an explicitly injected transport;
* one arbiter serializes Move, posture, and Full Stop command families;
* Full Stop pre-empts and flushes every queued command;
* posture commands are rate/step bounded and require fresh body feedback.

The installed Go2W WebRTC connector exposes SPORT API ids 1007 (Euler), 1008
(Move), 1013 (BodyHeight), and 1024 (GetBodyHeight).  Euler's ``x/y/z``
parameter contract matches Unitree's official SDK.  BodyHeight uses the
connector's scalar ``data`` convention and remains behind this adapter so the
transport representation can be replaced without touching reactive control.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Callable, Mapping, Protocol


POSTURE_STATUS_SCHEMA = "z_manip.go2w_posture_status.v1"


class CommandOwner(str, Enum):
    NONE = "none"
    BASE = "base"
    POSTURE = "posture"
    FULL_STOP = "full_stop"


class PosturePhase(str, Enum):
    IDLE = "idle"
    WAITING_BASE_QUIET = "waiting_base_quiet"
    COMMANDING = "commanding"
    SETTLING = "settling"
    REACHED = "reached"
    BLOCKED = "blocked"
    FAULT = "fault"
    STOPPED = "stopped"


class SportCommand(str, Enum):
    STOP_MOVE = "StopMove"
    EULER = "Euler"
    MOVE = "Move"
    BODY_HEIGHT = "BodyHeight"
    GET_BODY_HEIGHT = "GetBodyHeight"


SPORT_API_ID = {
    SportCommand.STOP_MOVE: 1003,
    SportCommand.EULER: 1007,
    SportCommand.MOVE: 1008,
    SportCommand.BODY_HEIGHT: 1013,
    SportCommand.GET_BODY_HEIGHT: 1024,
}


@dataclass(frozen=True)
class SportRequest:
    command: SportCommand
    parameter: Mapping[str, float]

    @property
    def api_id(self) -> int:
        return SPORT_API_ID[self.command]

    def wire_document(self) -> dict[str, Any]:
        return {"api_id": self.api_id, "parameter": dict(self.parameter)}


class SportTransport(Protocol):
    """Transport supplied only by the single NUC WebRTC command owner."""

    def send(self, request: SportRequest) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PostureLimits:
    # Unitree BodyHeight is an offset from the nominal balanced stance.  Keep
    # the initial manipulation envelope deliberately inside the wider SPORT
    # range until measurements on this Go2W establish a calibrated envelope.
    min_body_height_m: float = -0.12
    max_body_height_m: float = 0.02
    max_abs_roll_rad: float = math.radians(8.0)
    max_abs_pitch_rad: float = math.radians(12.0)
    max_abs_yaw_rad: float = math.radians(8.0)
    max_height_step_m: float = 0.01
    max_angle_step_rad: float = math.radians(2.0)
    min_command_period_s: float = 0.20
    feedback_timeout_s: float = 0.50
    quiet_linear_speed_mps: float = 0.035
    quiet_yaw_rate_rps: float = 0.05
    height_tolerance_m: float = 0.012
    angle_tolerance_rad: float = math.radians(2.0)
    settle_time_s: float = 0.35
    allow_posture_while_moving: bool = False

    def __post_init__(self) -> None:
        values = (
            self.min_body_height_m,
            self.max_body_height_m,
            self.max_abs_roll_rad,
            self.max_abs_pitch_rad,
            self.max_abs_yaw_rad,
            self.max_height_step_m,
            self.max_angle_step_rad,
            self.min_command_period_s,
            self.feedback_timeout_s,
            self.quiet_linear_speed_mps,
            self.quiet_yaw_rate_rps,
            self.height_tolerance_m,
            self.angle_tolerance_rad,
            self.settle_time_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("posture limits must be finite")
        if self.min_body_height_m >= self.max_body_height_m:
            raise ValueError("body-height limits are reversed")
        if any(value <= 0.0 for value in values[2:]):
            raise ValueError("posture rates, bounds, and tolerances must be positive")


@dataclass(frozen=True)
class PostureTarget:
    body_height_m: float
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0


@dataclass(frozen=True)
class PostureFeedback:
    stamp_s: float
    body_height_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    base_linear_x_mps: float = 0.0
    base_linear_y_mps: float = 0.0
    base_yaw_rate_rps: float = 0.0
    source: str = "sport_mode_state"

    def __post_init__(self) -> None:
        numeric = (
            self.stamp_s,
            self.body_height_m,
            self.roll_rad,
            self.pitch_rad,
            self.yaw_rad,
            self.base_linear_x_mps,
            self.base_linear_y_mps,
            self.base_yaw_rate_rps,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("posture feedback must be finite")
        if not self.source:
            raise ValueError("posture feedback source is required")

    @property
    def planar_speed_mps(self) -> float:
        return math.hypot(self.base_linear_x_mps, self.base_linear_y_mps)


@dataclass(frozen=True)
class CommandEvidence:
    sequence: int
    name: str | None
    api_id: int | None
    parameter: Mapping[str, float]
    would_send: bool
    sent: bool
    accepted: bool | None
    reason: str


@dataclass(frozen=True)
class PostureOutput:
    phase: PosturePhase
    owner: CommandOwner
    target: PostureTarget | None
    feedback: PostureFeedback | None
    feedback_age_s: float | None
    command: CommandEvidence

    def status_document(self, *, mode: str) -> dict[str, Any]:
        feedback = self.feedback
        target = self.target
        height_error = (
            None if feedback is None or target is None
            else target.body_height_m - feedback.body_height_m
        )
        pitch_error = (
            None if feedback is None or target is None
            else target.pitch_rad - feedback.pitch_rad
        )
        return {
            "schema": POSTURE_STATUS_SCHEMA,
            "mode": mode,
            "phase": self.phase.value,
            "command_owner": self.owner.value,
            "body_height": {
                "current_m": None if feedback is None else feedback.body_height_m,
                "target_m": None if target is None else target.body_height_m,
                "error_m": height_error,
                "feedback_age_s": self.feedback_age_s,
            },
            "attitude": {
                "current_roll_rad": None if feedback is None else feedback.roll_rad,
                "current_pitch_rad": None if feedback is None else feedback.pitch_rad,
                "current_yaw_rad": None if feedback is None else feedback.yaw_rad,
                "target_roll_rad": None if target is None else target.roll_rad,
                "target_pitch_rad": None if target is None else target.pitch_rad,
                "target_yaw_rad": None if target is None else target.yaw_rad,
                "pitch_error_rad": pitch_error,
            },
            "base": {
                "linear_speed_mps": None if feedback is None else feedback.planar_speed_mps,
                "yaw_rate_rps": None if feedback is None else feedback.base_yaw_rate_rps,
                "quiet": None if feedback is None else (
                    feedback.planar_speed_mps <= 0.035
                    and abs(feedback.base_yaw_rate_rps) <= 0.05
                ),
            },
            "feedback": {
                "fresh": (
                    self.feedback_age_s is not None
                    and self.phase != PosturePhase.BLOCKED
                ),
                "source": None if feedback is None else feedback.source,
            },
            "command": asdict(self.command),
        }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _step(current: float, target: float, maximum: float) -> float:
    return current + _clamp(target - current, -maximum, maximum)


def sport_response_code(response: Mapping[str, Any]) -> int | None:
    """Extract the robot verdict used by the installed WebRTC connector."""
    try:
        return int(response["data"]["header"]["status"]["code"])
    except (KeyError, TypeError, ValueError):
        return None


class SportCommandArbiter:
    """Serialize command families and give Full Stop unconditional priority.

    The adapter is deliberately small: the process owning the one WebRTC
    connection calls :meth:`submit`.  Base Move requests are latest-value
    coalesced, posture requests are FIFO (height then Euler), and Full Stop
    flushes both queues before occupying the next dispatch slot.
    """

    def __init__(self) -> None:
        self._moves: deque[SportRequest] = deque(maxlen=1)
        self._posture: deque[SportRequest] = deque()
        self._stop: SportRequest | None = None
        self.owner = CommandOwner.NONE

    def submit(self, request: SportRequest) -> None:
        if request.command == SportCommand.STOP_MOVE:
            self.full_stop()
            return
        if self._stop is not None:
            return
        if request.command == SportCommand.MOVE:
            self._moves.append(request)
            return
        if request.command in (SportCommand.BODY_HEIGHT, SportCommand.EULER):
            self._posture.append(request)
            return
        raise ValueError(f"unsupported command family: {request.command.value}")

    def full_stop(self) -> None:
        self._moves.clear()
        self._posture.clear()
        self._stop = SportRequest(SportCommand.STOP_MOVE, {})
        self.owner = CommandOwner.FULL_STOP

    def clear_stop(self) -> None:
        self._stop = None
        if self.owner == CommandOwner.FULL_STOP:
            self.owner = CommandOwner.NONE

    def pop_next(self) -> SportRequest | None:
        if self._stop is not None:
            request, self._stop = self._stop, None
            self.owner = CommandOwner.FULL_STOP
            return request
        if self._posture:
            self.owner = CommandOwner.POSTURE
            return self._posture.popleft()
        if self._moves:
            self.owner = CommandOwner.BASE
            return self._moves.pop()
        self.owner = CommandOwner.NONE
        return None

    @property
    def pending(self) -> int:
        return len(self._moves) + len(self._posture) + int(self._stop is not None)


class Go2WPostureAdapter:
    """Feedback-verified, shadow-first BodyHeight/Euler adapter."""

    def __init__(
        self,
        *,
        mode: str = "shadow",
        limits: PostureLimits | None = None,
        transport: SportTransport | None = None,
        arbiter: SportCommandArbiter | None = None,
    ) -> None:
        if mode not in {"shadow", "live"}:
            raise ValueError("mode must be shadow or live")
        if mode == "live" and transport is None:
            raise ValueError("live posture control requires an injected transport")
        self.mode = mode
        self.limits = limits or PostureLimits()
        self.transport = transport
        self.arbiter = arbiter or SportCommandArbiter()
        self.target: PostureTarget | None = None
        self.feedback: PostureFeedback | None = None
        self.phase = PosturePhase.IDLE
        self._last_command_s: float | None = None
        self._settled_since_s: float | None = None
        self._sequence = 0
        self._last_evidence = CommandEvidence(0, None, None, {}, False, False, None, "idle")

    def set_target(self, target: PostureTarget) -> None:
        values = (
            target.body_height_m,
            target.roll_rad,
            target.pitch_rad,
            target.yaw_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("posture target must be finite")
        if not self.limits.min_body_height_m <= target.body_height_m <= self.limits.max_body_height_m:
            raise ValueError("body-height target is outside the configured envelope")
        if abs(target.roll_rad) > self.limits.max_abs_roll_rad:
            raise ValueError("roll target is outside the configured envelope")
        if abs(target.pitch_rad) > self.limits.max_abs_pitch_rad:
            raise ValueError("pitch target is outside the configured envelope")
        if abs(target.yaw_rad) > self.limits.max_abs_yaw_rad:
            raise ValueError("yaw target is outside the configured envelope")
        self.target = target
        self.phase = PosturePhase.COMMANDING
        self._settled_since_s = None

    def observe(self, feedback: PostureFeedback) -> None:
        if self.feedback is not None and feedback.stamp_s < self.feedback.stamp_s:
            raise ValueError("posture feedback time moved backwards")
        self.feedback = feedback

    def cancel(self, *, full_stop: bool = False) -> None:
        self.target = None
        self._settled_since_s = None
        self.phase = PosturePhase.STOPPED if full_stop else PosturePhase.IDLE
        if full_stop:
            self.arbiter.full_stop()
        self._last_evidence = CommandEvidence(
            self._sequence,
            SportCommand.STOP_MOVE.value if full_stop else None,
            SPORT_API_ID[SportCommand.STOP_MOVE] if full_stop else None,
            {},
            full_stop,
            False,
            None,
            "full stop pre-empted posture" if full_stop else "posture cancelled",
        )

    def dispatch_full_stop(self) -> CommandEvidence:
        """Flush all work and dispatch exactly one highest-priority StopMove."""
        self.cancel(full_stop=True)
        request = self.arbiter.pop_next()
        assert request is not None and request.command == SportCommand.STOP_MOVE
        self._last_evidence = self._dispatch(request)
        return self._last_evidence

    def _output(self, now_s: float, *, reason: str = "") -> PostureOutput:
        age = None if self.feedback is None else max(0.0, now_s - self.feedback.stamp_s)
        if reason:
            evidence = CommandEvidence(
                self._sequence, None, None, {}, False, False, None, reason,
            )
        else:
            evidence = self._last_evidence
        return PostureOutput(self.phase, self.arbiter.owner, self.target, self.feedback, age, evidence)

    def _dispatch(self, request: SportRequest) -> CommandEvidence:
        self._sequence += 1
        if self.mode == "shadow":
            return CommandEvidence(
                self._sequence, request.command.value, request.api_id,
                dict(request.parameter), True, False, None, "shadow: command not transmitted",
            )
        assert self.transport is not None
        response = self.transport.send(request)
        code = sport_response_code(response)
        accepted = code in (0, None)
        return CommandEvidence(
            self._sequence, request.command.value, request.api_id,
            dict(request.parameter), True, True, accepted,
            f"robot response code={code}",
        )

    def tick(self, *, now_s: float) -> PostureOutput:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("posture clock must be finite")
        if self.target is None:
            return self._output(now)
        if self.feedback is None:
            self.phase = PosturePhase.BLOCKED
            return self._output(now, reason="measured body feedback is unavailable")
        age = max(0.0, now - self.feedback.stamp_s)
        if age > self.limits.feedback_timeout_s:
            self.phase = PosturePhase.BLOCKED
            return self._output(now, reason="measured body feedback is stale")
        if self.arbiter.owner == CommandOwner.FULL_STOP:
            self.phase = PosturePhase.STOPPED
            return self._output(now, reason="full stop owns the command channel")

        base_quiet = (
            self.feedback.planar_speed_mps <= self.limits.quiet_linear_speed_mps
            and abs(self.feedback.base_yaw_rate_rps) <= self.limits.quiet_yaw_rate_rps
        )
        if not self.limits.allow_posture_while_moving and not base_quiet:
            self.phase = PosturePhase.WAITING_BASE_QUIET
            self._settled_since_s = None
            return self._output(now, reason="waiting for base velocity to settle")

        height_error = self.target.body_height_m - self.feedback.body_height_m
        angle_errors = (
            self.target.roll_rad - self.feedback.roll_rad,
            self.target.pitch_rad - self.feedback.pitch_rad,
            self.target.yaw_rad - self.feedback.yaw_rad,
        )
        inside = (
            abs(height_error) <= self.limits.height_tolerance_m
            and all(abs(error) <= self.limits.angle_tolerance_rad for error in angle_errors)
        )
        if inside:
            if self._settled_since_s is None:
                self._settled_since_s = now
            if now - self._settled_since_s + 1e-9 >= self.limits.settle_time_s:
                self.phase = PosturePhase.REACHED
            else:
                self.phase = PosturePhase.SETTLING
            return self._output(now)
        self._settled_since_s = None

        if self._last_command_s is not None and now - self._last_command_s < self.limits.min_command_period_s:
            self.phase = PosturePhase.COMMANDING
            return self._output(now, reason="posture command rate limited")

        height_request = SportRequest(
            SportCommand.BODY_HEIGHT,
            {"data": _step(
                self.feedback.body_height_m,
                self.target.body_height_m,
                self.limits.max_height_step_m,
            )},
        )
        euler_request = SportRequest(
            SportCommand.EULER,
            {
                "x": _step(self.feedback.roll_rad, self.target.roll_rad, self.limits.max_angle_step_rad),
                "y": _step(self.feedback.pitch_rad, self.target.pitch_rad, self.limits.max_angle_step_rad),
                "z": _step(self.feedback.yaw_rad, self.target.yaw_rad, self.limits.max_angle_step_rad),
            },
        )
        # Finish the previous height+Euler pair before sampling a new pair.
        # Otherwise a slow transport could accumulate stale intermediate
        # postures faster than feedback can confirm them.
        if self.arbiter.pending == 0:
            self.arbiter.submit(height_request)
            self.arbiter.submit(euler_request)
        request = self.arbiter.pop_next()
        assert request is not None
        self._last_evidence = self._dispatch(request)
        self._last_command_s = now
        self.phase = (
            PosturePhase.COMMANDING
            if self._last_evidence.accepted is not False
            else PosturePhase.FAULT
        )
        return self._output(now)


def feedback_from_mapping(message: Mapping[str, Any], *, stamp_s: float) -> PostureFeedback:
    """Normalize a decoded SPORT_MOD_STATE-like mapping.

    The official state schema uses ``body_height``, ``velocity`` and
    ``imu_state.rpy``.  Some WebRTC bridges flatten those keys, so both forms
    are accepted.  Missing measured posture is rejected rather than silently
    replaced by the last command.
    """
    data = message.get("data", message)
    if not isinstance(data, Mapping):
        raise ValueError("sport state payload must be a mapping")
    imu = data.get("imu_state", {})
    if not isinstance(imu, Mapping):
        imu = {}
    rpy = imu.get("rpy", data.get("rpy"))
    velocity = data.get("velocity", (0.0, 0.0, 0.0))
    if not isinstance(rpy, (list, tuple)) or len(rpy) < 3:
        raise ValueError("sport state lacks measured roll/pitch/yaw")
    if not isinstance(velocity, (list, tuple)) or len(velocity) < 3:
        raise ValueError("sport state velocity must contain x/y/yaw")
    if "body_height" not in data:
        raise ValueError("sport state lacks measured body_height")
    return PostureFeedback(
        stamp_s=float(stamp_s),
        body_height_m=float(data["body_height"]),
        roll_rad=float(rpy[0]),
        pitch_rad=float(rpy[1]),
        yaw_rad=float(rpy[2]),
        base_linear_x_mps=float(velocity[0]),
        base_linear_y_mps=float(velocity[1]),
        base_yaw_rate_rps=float(velocity[2]),
        source="sport_mode_state",
    )


__all__ = [
    "POSTURE_STATUS_SCHEMA",
    "CommandEvidence",
    "CommandOwner",
    "Go2WPostureAdapter",
    "PostureFeedback",
    "PostureLimits",
    "PostureOutput",
    "PosturePhase",
    "PostureTarget",
    "SPORT_API_ID",
    "SportCommand",
    "SportCommandArbiter",
    "SportRequest",
    "SportTransport",
    "feedback_from_mapping",
    "sport_response_code",
]
