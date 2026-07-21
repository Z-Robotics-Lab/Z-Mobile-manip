"""Bounded, correlated lifecycle for one observed placement transaction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping


TRANSACTION_CONTROL_SCHEMA = 'z_manip.place_transaction_control.v1'
PLACE_STATUS_SCHEMA = 'z_manip.place_status.v2'


class PlaceTransactionError(ValueError):
    """A transaction message or lifecycle transition is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise PlaceTransactionError(f'duplicate JSON key: {key}')
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise PlaceTransactionError(f'non-finite JSON constant: {value}')


def _strict_object(payload: str, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise PlaceTransactionError(f'{label} is not valid JSON') from error
    if not isinstance(value, Mapping):
        raise PlaceTransactionError(f'{label} must be a JSON object')
    return value


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise PlaceTransactionError(f'{label} must be a bounded non-empty string')
    return value


@dataclass(frozen=True)
class PlaceTransactionIdentity:
    """Exact task/executor ownership of one placement request."""

    goal_id: str
    executor_epoch: str

    def __post_init__(self) -> None:
        _identity(self.goal_id, 'goal_id')
        _identity(self.executor_epoch, 'executor_epoch')


@dataclass(frozen=True)
class PlaceTransactionControl:
    """Strict task-to-place transaction control command."""

    action: str
    identity: PlaceTransactionIdentity

    def __post_init__(self) -> None:
        if self.action != 'abort':
            raise PlaceTransactionError('transaction control action must be abort')

    def to_payload(self) -> dict[str, object]:
        return {
            'schema': TRANSACTION_CONTROL_SCHEMA,
            'action': self.action,
            'goal_id': self.identity.goal_id,
            'executor_epoch': self.identity.executor_epoch,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )


def parse_transaction_control(payload: str) -> PlaceTransactionControl:
    """Parse an exact abort command, rejecting duplicates and extensions."""
    value = _strict_object(payload, 'placement transaction control')
    expected = {'schema', 'action', 'goal_id', 'executor_epoch'}
    if set(value) != expected:
        raise PlaceTransactionError(
            'placement transaction control keys mismatch; '
            f'missing={sorted(expected - set(value))}, '
            f'unknown={sorted(set(value) - expected)}',
        )
    if value['schema'] != TRANSACTION_CONTROL_SCHEMA:
        raise PlaceTransactionError('unsupported placement transaction control schema')
    return PlaceTransactionControl(
        action=value['action'],
        identity=PlaceTransactionIdentity(
            goal_id=value['goal_id'],
            executor_epoch=value['executor_epoch'],
        ),
    )


def parse_region_transaction_identity(payload: str) -> PlaceTransactionIdentity:
    """Recover only an unambiguous v2 request identity for failure correlation."""
    value = _strict_object(payload, 'placement region request')
    schema = value.get('schema_version')
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 2:
        raise PlaceTransactionError('unsupported placement region request schema')
    return PlaceTransactionIdentity(
        goal_id=value.get('goal_id'),
        executor_epoch=value.get('executor_epoch'),
    )


@dataclass(frozen=True)
class PlaceTerminalFailure:
    """Strict correlated terminal status published by the place node."""

    identity: PlaceTransactionIdentity
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reason, str)
            or not self.reason.strip()
            or len(self.reason) > 1024
        ):
            raise PlaceTransactionError('terminal failure reason is invalid')

    def to_payload(self) -> dict[str, object]:
        return {
            'schema': PLACE_STATUS_SCHEMA,
            'state': 'failed',
            'terminal': True,
            'goal_id': self.identity.goal_id,
            'executor_epoch': self.identity.executor_epoch,
            'reason': self.reason,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )


@dataclass(frozen=True)
class PlaceTransactionToken:
    """Generation-bound capability held by exactly one planning worker."""

    identity: PlaceTransactionIdentity
    generation: int


class PlacementTransactionLifecycle:
    """Reject stale workers and bound a request's complete active lifetime."""

    def __init__(self, *, ros_timeout_s: float, wall_timeout_s: float) -> None:
        for label, value in (
            ('ros_timeout_s', ros_timeout_s),
            ('wall_timeout_s', wall_timeout_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise PlaceTransactionError(f'{label} must be finite and positive')
        self.ros_timeout_ns = int(round(ros_timeout_s * 1e9))
        self.wall_timeout_s = float(wall_timeout_s)
        self._generation = 0
        self.reset()

    def reset(self) -> None:
        """Invalidate the current token and unblock the next request."""
        self._generation += 1
        self.identity: PlaceTransactionIdentity | None = None
        self.state = 'idle'
        self.started_ros_ns: int | None = None
        self.last_ros_ns: int | None = None
        self.started_wall_s: float | None = None
        self.last_wall_s: float | None = None

    @property
    def active(self) -> bool:
        return self.identity is not None and self.state in {'pending', 'planning', 'armed'}

    @property
    def armed(self) -> bool:
        return self.identity is not None and self.state == 'armed'

    @staticmethod
    def _validate_clocks(*, now_ros_ns: int, now_wall_s: float) -> None:
        if (
            isinstance(now_ros_ns, bool)
            or not isinstance(now_ros_ns, int)
            or now_ros_ns < 0
        ):
            raise PlaceTransactionError('transaction ROS clock is invalid')
        if not math.isfinite(now_wall_s) or now_wall_s < 0.0:
            raise PlaceTransactionError('transaction wall clock is invalid')

    def begin(
        self,
        identity: PlaceTransactionIdentity,
        *,
        now_ros_ns: int,
        now_wall_s: float,
    ) -> PlaceTransactionToken:
        """Own one request and start immutable pending-to-terminal deadlines."""
        if self.active:
            raise PlaceTransactionError('a placement transaction is already active')
        if not isinstance(identity, PlaceTransactionIdentity):
            raise PlaceTransactionError('placement transaction identity is invalid')
        self._validate_clocks(now_ros_ns=now_ros_ns, now_wall_s=now_wall_s)
        self._generation += 1
        self.identity = identity
        self.state = 'pending'
        self.started_ros_ns = now_ros_ns
        self.last_ros_ns = now_ros_ns
        self.started_wall_s = float(now_wall_s)
        self.last_wall_s = float(now_wall_s)
        return PlaceTransactionToken(identity, self._generation)

    def start_planning(self, token: PlaceTransactionToken) -> None:
        if not self.matches(token) or self.state != 'pending':
            raise PlaceTransactionError('placement planning token is stale')
        self.state = 'planning'

    def matches(self, token: PlaceTransactionToken) -> bool:
        return bool(
            isinstance(token, PlaceTransactionToken)
            and self.identity == token.identity
            and self._generation == token.generation
            and self.active
        )

    def arm(
        self,
        token: PlaceTransactionToken,
        *,
        now_ros_ns: int,
        now_wall_s: float,
    ) -> None:
        """Arm execution without extending the request's original deadlines."""
        if not self.matches(token) or self.state != 'planning':
            raise PlaceTransactionError('aborted or stale planning worker cannot arm')
        self._validate_clocks(now_ros_ns=now_ros_ns, now_wall_s=now_wall_s)
        assert self.last_ros_ns is not None
        assert self.last_wall_s is not None
        if now_ros_ns < self.last_ros_ns:
            raise PlaceTransactionError(
                'placement transaction ROS clock moved backwards before arm',
            )
        if now_wall_s < self.last_wall_s:
            raise PlaceTransactionError(
                'placement transaction wall clock moved backwards before arm',
            )
        assert self.started_ros_ns is not None
        assert self.started_wall_s is not None
        if now_ros_ns - self.started_ros_ns >= self.ros_timeout_ns:
            raise PlaceTransactionError(
                'placement transaction ROS-time deadline exceeded before arm',
            )
        if now_wall_s - self.started_wall_s >= self.wall_timeout_s:
            raise PlaceTransactionError(
                'placement transaction wall-time deadline exceeded before arm',
            )
        self.state = 'armed'
        self.last_ros_ns = now_ros_ns
        self.last_wall_s = float(now_wall_s)

    def abort(self, control: PlaceTransactionControl) -> bool:
        """Reset only an exact active goal/epoch; foreign or stale aborts are ignored."""
        if not isinstance(control, PlaceTransactionControl):
            raise PlaceTransactionError('placement transaction control is invalid')
        if not self.active or control.identity != self.identity:
            return False
        self.reset()
        return True

    def watchdog_reason(self, *, now_ros_ns: int, now_wall_s: float) -> str | None:
        """Return one terminal reason and reset on rollback or either timeout."""
        if not self.active:
            return None
        try:
            self._validate_clocks(
                now_ros_ns=now_ros_ns,
                now_wall_s=now_wall_s,
            )
        except PlaceTransactionError:
            reason = 'placement transaction watchdog clock is invalid'
        else:
            assert self.last_ros_ns is not None
            assert self.started_ros_ns is not None
            assert self.last_wall_s is not None
            assert self.started_wall_s is not None
            if now_ros_ns < self.last_ros_ns:
                reason = 'placement transaction ROS clock moved backwards'
            elif now_wall_s < self.last_wall_s:
                reason = 'placement transaction wall clock moved backwards'
            elif now_ros_ns - self.started_ros_ns >= self.ros_timeout_ns:
                reason = (
                    f'placement transaction {self.state} ROS-time deadline exceeded'
                )
            elif now_wall_s - self.started_wall_s >= self.wall_timeout_s:
                reason = (
                    f'placement transaction {self.state} wall-time deadline exceeded'
                )
            else:
                self.last_ros_ns = now_ros_ns
                self.last_wall_s = float(now_wall_s)
                return None
        self.reset()
        return reason
