"""Task-side strict placement abort and terminal-status protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping


TRANSACTION_CONTROL_SCHEMA = 'z_manip.place_transaction_control.v1'
PLACE_STATUS_SCHEMA = 'z_manip.place_status.v2'


class PlaceTransactionProtocolError(ValueError):
    """A placement transaction protocol payload is malformed."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PlaceTransactionProtocolError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise PlaceTransactionProtocolError(f'non-finite JSON constant: {value}')


def _strict_object(payload: str, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, json.JSONDecodeError) as error:
        raise PlaceTransactionProtocolError(f'{label} is not valid JSON') from error
    if not isinstance(value, Mapping):
        raise PlaceTransactionProtocolError(f'{label} must be a JSON object')
    return value


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise PlaceTransactionProtocolError(
            f'{label} must be a bounded non-empty string',
        )
    return value


def transaction_abort_json(*, goal_id: str, executor_epoch: str) -> str:
    """Serialize the only supported task-to-place control command."""
    payload = {
        'schema': TRANSACTION_CONTROL_SCHEMA,
        'action': 'abort',
        'goal_id': _identity(goal_id, 'goal_id'),
        'executor_epoch': _identity(executor_epoch, 'executor_epoch'),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )


@dataclass(frozen=True)
class PlaceTerminalFailure:
    """One exact place-node terminal failure owned by this task."""

    goal_id: str
    executor_epoch: str
    reason: str


def parse_terminal_place_status(payload: str) -> PlaceTerminalFailure | None:
    """Return a strict v2 terminal failure; ignore legacy/nonterminal status."""
    value = _strict_object(payload, 'placement status')
    if value.get('schema') != PLACE_STATUS_SCHEMA:
        return None
    expected = {
        'schema', 'state', 'terminal', 'goal_id', 'executor_epoch', 'reason',
    }
    if set(value) != expected:
        raise PlaceTransactionProtocolError(
            'placement status keys mismatch; '
            f'missing={sorted(expected - set(value))}, '
            f'unknown={sorted(set(value) - expected)}',
        )
    if value['state'] != 'failed' or value['terminal'] is not True:
        raise PlaceTransactionProtocolError(
            'placement status v2 must be a terminal failure',
        )
    reason = value['reason']
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
        raise PlaceTransactionProtocolError('placement failure reason is invalid')
    return PlaceTerminalFailure(
        goal_id=_identity(value['goal_id'], 'goal_id'),
        executor_epoch=_identity(value['executor_epoch'], 'executor_epoch'),
        reason=reason,
    )
