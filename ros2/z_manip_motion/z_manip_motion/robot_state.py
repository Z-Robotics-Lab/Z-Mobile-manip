"""Pure complete-robot joint-state contracts for MoveIt planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from xml.etree import ElementTree

from .contracts import ContractError


_JOINT_STATE_TYPES = frozenset({"revolute", "continuous", "prismatic"})
_UNSUPPORTED_MOVABLE_TYPES = frozenset({"floating", "planar"})


def movable_joint_names_from_urdf(robot_description: str) -> tuple[str, ...]:
    """Return independent scalar joints that must be measured for MoveIt."""
    try:
        root = ElementTree.fromstring(robot_description)
    except ElementTree.ParseError as error:
        raise ContractError(f"robot description is not valid XML: {error}") from error
    if root.tag != "robot":
        raise ContractError("robot description root must be <robot>")

    result: list[str] = []
    unsupported: list[str] = []
    for joint in root.findall("joint"):
        name = str(joint.attrib.get("name", "")).strip()
        joint_type = str(joint.attrib.get("type", "")).strip()
        if not name or not joint_type:
            raise ContractError("every URDF joint must have a name and type")
        if joint_type in _UNSUPPORTED_MOVABLE_TYPES:
            unsupported.append(name)
        elif joint_type in _JOINT_STATE_TYPES and joint.find("mimic") is None:
            result.append(name)
    if unsupported:
        raise ContractError(
            "scalar JointState assembly does not support planar/floating joints: "
            f"{sorted(unsupported)}",
        )
    if not result:
        raise ContractError("robot description has no independently movable scalar joints")
    if len(set(result)) != len(result):
        raise ContractError("robot description contains duplicate movable joint names")
    return tuple(result)


@dataclass(frozen=True)
class CompleteState:
    """One canonical, complete robot state."""

    names: tuple[str, ...]
    positions: tuple[float, ...]
    stamp_ns: int
    epoch: int


@dataclass(frozen=True)
class StateReadiness:
    """Why a complete state can or cannot be published."""

    missing: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    unstamped: tuple[str, ...] = ()
    inconsistent: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not (
            self.missing or self.stale or self.unstamped or self.inconsistent
        )


class ClockHandoverGuard:
    """Require a stable unique clock epoch before state acceptance resumes."""

    def __init__(self, quiet_s: float) -> None:
        if not math.isfinite(quiet_s) or quiet_s <= 0.0:
            raise ContractError("clock handover quiet time must be finite and positive")
        self._quiet_s = float(quiet_s)
        self._pending = False
        self._candidate_ns = 0
        self._clock_stable_since = 0.0
        self._publisher_gid: tuple[int, ...] | None = None
        self._publisher_stable_since: float | None = None
        self._armed = False

    @property
    def pending(self) -> bool:
        return self._pending

    @staticmethod
    def _validate_sample(stamp_ns: int, now: float) -> None:
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            raise ContractError("ROS clock stamp must be a non-negative integer")
        if not math.isfinite(now):
            raise ContractError("clock handover receipt time must be finite")

    def begin(self, stamp_ns: int, *, now: float) -> None:
        """Start a new epoch candidate at the observed rollback sample."""
        self._validate_sample(stamp_ns, now)
        self._pending = True
        self._candidate_ns = stamp_ns
        self._clock_stable_since = float(now)
        self._publisher_gid = None
        self._publisher_stable_since = None
        self._armed = False

    def observe(
        self,
        stamp_ns: int,
        *,
        now: float,
        publisher_gids: set[tuple[int, ...]] | None = None,
    ) -> bool:
        """Return true only on a post-holdoff advancing sample and fresh graph probe."""
        self._validate_sample(stamp_ns, now)
        if not self._pending:
            raise ContractError("clock handover is not pending")

        progressed = stamp_ns > self._candidate_ns
        if stamp_ns < self._candidate_ns:
            self._clock_stable_since = float(now)
            self._armed = False
        if stamp_ns != self._candidate_ns:
            self._candidate_ns = stamp_ns

        graph_was_probed = publisher_gids is not None
        if graph_was_probed:
            try:
                normalized = {
                    tuple(int(value) for value in gid)
                    for gid in publisher_gids
                }
            except (TypeError, ValueError) as error:
                raise ContractError("clock publisher GIDs must contain bytes") from error
            if any(
                not gid or any(value < 0 or value > 255 for value in gid)
                for gid in normalized
            ):
                normalized = set()
            if len(normalized) != 1:
                self._publisher_gid = None
                self._publisher_stable_since = None
                self._armed = False
            else:
                publisher_gid = next(iter(normalized))
                if publisher_gid != self._publisher_gid:
                    self._publisher_gid = publisher_gid
                    self._publisher_stable_since = float(now)
                    self._clock_stable_since = float(now)
                    self._candidate_ns = stamp_ns
                    self._armed = False
                    progressed = False

        ready = (
            self._publisher_stable_since is not None
            and now - self._publisher_stable_since >= self._quiet_s
            and now - self._clock_stable_since >= self._quiet_s
        )
        if not ready:
            return False
        if not self._armed:
            if progressed:
                self._armed = True
            return False
        return graph_was_probed and progressed

    def finish(self) -> None:
        """Close a handover after the assembler accepted its resume sample."""
        if not self._pending:
            raise ContractError("clock handover is not pending")
        self._pending = False
        self._armed = False


class CompleteJointStateAssembler:
    """Merge named proprioception streams without inventing joint values."""

    def __init__(
        self,
        required_names: Sequence[str],
        *,
        max_age_s: float,
        max_stamp_skew_s: float | None = None,
        expected_sources: Sequence[str] | None = None,
        require_clock: bool = False,
    ) -> None:
        names = tuple(str(name).strip() for name in required_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ContractError("required joint names must be unique and non-empty")
        if not math.isfinite(max_age_s) or max_age_s <= 0.0:
            raise ContractError("complete-state max age must be finite and positive")
        if max_stamp_skew_s is None:
            max_stamp_skew_s = max_age_s
        if not math.isfinite(max_stamp_skew_s) or max_stamp_skew_s <= 0.0:
            raise ContractError("joint-state stamp skew must be finite and positive")
        sources = tuple(str(source).strip() for source in expected_sources or ())
        if any(not source for source in sources) or len(set(sources)) != len(sources):
            raise ContractError("expected joint-state sources must be unique and non-empty")
        if not isinstance(require_clock, bool):
            raise ContractError("require_clock must be boolean")
        self._names = names
        self._required = frozenset(names)
        self._expected_sources = frozenset(sources)
        self._require_clock = require_clock
        self._max_age_s = float(max_age_s)
        self._max_clock_age_ns = round(self._max_age_s * 1_000_000_000)
        self._max_stamp_skew_ns = round(float(max_stamp_skew_s) * 1_000_000_000)
        self._positions: dict[str, float] = {}
        self._received_at: dict[str, float] = {}
        self._stamp_ns: dict[str, int] = {}
        self._source_stamp_ns: dict[str, int] = {}
        self._clock_ns: int | None = None
        self._clock_epoch_floor_ns: int | None = None
        self._clock_accept_after_ns: int | None = None
        self._clock_handover_pending = False
        self._epoch = 0
        self._published_epoch: int | None = None
        self._published_stamp_ns = 0

    @property
    def required_names(self) -> tuple[str, ...]:
        return self._names

    def observe_clock(self, stamp_ns: int) -> bool:
        """Track ROS time and clear state only on an explicit clock rollback."""
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            raise ContractError("ROS clock stamp must be a non-negative integer")
        if self._clock_handover_pending:
            return False
        previous_clock_ns = self._clock_ns
        self._clock_ns = stamp_ns
        if previous_clock_ns is None or stamp_ns >= previous_clock_ns:
            return False

        self._start_clock_epoch(stamp_ns, previous_clock_ns)
        self._clock_handover_pending = True
        return True

    def resume_clock(self, stamp_ns: int) -> None:
        """Accept a clock sample after the stable-reader handover guard passes."""
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            raise ContractError("ROS clock stamp must be a non-negative integer")
        if not self._clock_handover_pending:
            raise ContractError("ROS clock handover is not pending")
        previous_clock_ns = self._clock_ns
        if previous_clock_ns is not None and stamp_ns < previous_clock_ns:
            self._start_clock_epoch(stamp_ns, previous_clock_ns)
        else:
            self._clock_ns = stamp_ns
        self._clock_handover_pending = False

    def _start_clock_epoch(self, stamp_ns: int, previous_clock_ns: int) -> None:
        rollback_ns = previous_clock_ns - stamp_ns
        self._clock_ns = stamp_ns
        self._positions.clear()
        self._received_at.clear()
        self._stamp_ns.clear()
        self._source_stamp_ns.clear()
        self._clock_epoch_floor_ns = stamp_ns
        # For a small overlapping reset, wait until the new clock passes the
        # old clock. Large resets need only one bounded coherence window; old
        # high-stamped frames are then rejected by the current-clock ceiling.
        self._clock_accept_after_ns = stamp_ns + min(
            rollback_ns,
            self._max_stamp_skew_ns,
        )
        self._epoch += 1

    def update(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        *,
        source: str,
        stamp_ns: int,
        received_at: float,
        reference_stamp_ns: int | None = None,
    ) -> bool:
        """Accept one measured source sample, keyed strictly by joint name."""
        source_name = str(source).strip()
        source_names = tuple(str(name).strip() for name in names)
        try:
            source_positions = tuple(float(value) for value in positions)
        except (TypeError, ValueError) as error:
            raise ContractError("joint positions must be numeric") from error
        if len(source_names) != len(source_positions):
            raise ContractError("joint names and positions must have equal length")
        if any(not name for name in source_names):
            raise ContractError("joint state contains an empty name")
        if len(set(source_names)) != len(source_names):
            raise ContractError("joint state contains duplicate names")
        if not source_name:
            raise ContractError("joint-state source must be non-empty")
        if self._expected_sources and source_name not in self._expected_sources:
            raise ContractError(f"unexpected joint-state source: {source_name}")
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns < 0:
            raise ContractError("joint-state stamp must be a non-negative integer")
        if (
            reference_stamp_ns is not None
            and (
                isinstance(reference_stamp_ns, bool)
                or not isinstance(reference_stamp_ns, int)
                or reference_stamp_ns < 0
            )
        ):
            raise ContractError("reference clock stamp must be a non-negative integer")
        if not math.isfinite(received_at):
            raise ContractError("joint-state receipt time must be finite")
        if not all(math.isfinite(value) for value in source_positions):
            raise ContractError("joint state contains a non-finite position")

        measured = tuple(
            (name, position)
            for name, position in zip(source_names, source_positions)
            if name in self._required
        )
        if not measured:
            return False

        if not self._stamp_is_in_clock_epoch(stamp_ns, reference_stamp_ns):
            return False

        previous_stamp_ns = self._source_stamp_ns.get(source_name)
        if previous_stamp_ns is not None and stamp_ns < previous_stamp_ns:
            # Without an explicit /clock rollback, this is either reordering
            # or a broken upstream clock. In both cases fail closed.
            return False
        self._source_stamp_ns[source_name] = stamp_ns

        for name, position in measured:
            self._positions[name] = position
            self._received_at[name] = float(received_at)
            self._stamp_ns[name] = stamp_ns
        return True

    def _stamp_is_in_clock_epoch(
        self,
        stamp_ns: int,
        reference_stamp_ns: int | None,
    ) -> bool:
        current_clock_ns = (
            reference_stamp_ns
            if reference_stamp_ns is not None
            else self._clock_ns
        )
        if self._clock_handover_pending:
            return False
        if current_clock_ns is None:
            return not self._require_clock
        if (
            reference_stamp_ns is None
            and self._clock_accept_after_ns is not None
            and current_clock_ns < self._clock_accept_after_ns
        ):
            return False
        if self._clock_epoch_floor_ns is not None and stamp_ns < self._clock_epoch_floor_ns:
            return False
        if stamp_ns > current_clock_ns + self._max_stamp_skew_ns:
            return False
        if current_clock_ns - stamp_ns > self._max_clock_age_ns:
            return False
        return True

    def readiness(self, *, now: float) -> StateReadiness:
        if not math.isfinite(now):
            raise ContractError("joint-state snapshot time must be finite")
        missing = tuple(name for name in self._names if name not in self._positions)
        stale = tuple(
            name
            for name in self._names
            if name in self._received_at
            and (
                now < self._received_at[name]
                or now - self._received_at[name] > self._max_age_s
            )
        )
        unstamped = tuple(
            name
            for name in self._names
            if name in self._positions and self._stamp_ns.get(name, 0) <= 0
        )
        stamped = tuple(
            self._stamp_ns[name]
            for name in self._names
            if name in self._stamp_ns and self._stamp_ns[name] > 0
        )
        newest_stamp_ns = max(stamped, default=0)
        inconsistent = tuple(
            name
            for name in self._names
            if self._stamp_ns.get(name, 0) > 0
            and newest_stamp_ns - self._stamp_ns[name] > self._max_stamp_skew_ns
        )
        return StateReadiness(
            missing=missing,
            stale=stale,
            unstamped=unstamped,
            inconsistent=inconsistent,
        )

    def snapshot(self, *, now: float) -> CompleteState | None:
        readiness = self.readiness(now=now)
        if not readiness.ready:
            return None
        return CompleteState(
            names=self._names,
            positions=tuple(self._positions[name] for name in self._names),
            # The merged message is only as new as its oldest included sample.
            # Using the newest stamp would make slower joints appear fresher
            # than the measurements that actually supplied their positions.
            stamp_ns=min(self._stamp_ns[name] for name in self._names),
            epoch=self._epoch,
        )

    def next_snapshot(self, *, now: float) -> CompleteState | None:
        """Return a complete state only when its ROS stamp advances in its epoch."""
        complete = self.snapshot(now=now)
        if complete is None:
            return None
        if (
            self._published_epoch == complete.epoch
            and complete.stamp_ns <= self._published_stamp_ns
        ):
            return None
        self._published_epoch = complete.epoch
        self._published_stamp_ns = complete.stamp_ns
        return complete
