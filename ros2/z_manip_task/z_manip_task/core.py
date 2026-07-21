"""ROS-independent safety invariants for the online task runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence

from z_manip.trajectory_digest import canonical_joint_trajectory_sha256


def validate_platform_odometry_frames(
    parent_frame: str,
    child_frame: str,
    *,
    expected_parent_frame: str,
    expected_child_frame: str,
) -> None:
    """Require the configured platform-base odometry frame contract."""
    expected = (str(expected_parent_frame), str(expected_child_frame))
    if not all(expected):
        raise ValueError('platform odometry expected frames must be non-empty')
    observed = (str(parent_frame), str(child_frame))
    if observed != expected:
        raise ValueError(
            'platform odometry frame mismatch: expected '
            f'{expected[0]} -> {expected[1]}, received '
            f'{observed[0] or "<empty>"} -> {observed[1] or "<empty>"}',
        )


def validate_position_hold_frame_contract(
    odometry_child_frame: str,
    velocity_command_frame: str,
) -> None:
    """Require position feedback and commanded velocity to share a base origin."""
    odometry_frame = str(odometry_child_frame)
    command_frame = str(velocity_command_frame)
    if not odometry_frame or not command_frame:
        raise ValueError('position-hold frames must be non-empty')
    if odometry_frame != command_frame:
        raise ValueError(
            'position-hold frame mismatch: odometry tracks '
            f'{odometry_frame}, but velocity commands use {command_frame}',
        )


def base_twist_speed_magnitudes(
    linear_xyz: tuple[float, float, float],
    angular_xyz: tuple[float, float, float],
) -> tuple[float, float]:
    """Return independent SI speed magnitudes for one measured base twist."""
    linear = tuple(float(value) for value in linear_xyz)
    angular = tuple(float(value) for value in angular_xyz)
    if (
        len(linear) != 3
        or len(angular) != 3
        or not all(math.isfinite(value) for value in (*linear, *angular))
    ):
        raise ValueError('base twist must contain finite linear and angular vectors')
    return (
        math.sqrt(sum(value * value for value in linear)),
        math.sqrt(sum(value * value for value in angular)),
    )


class RuntimePhase(str, Enum):
    """Externally observable runtime phase."""

    IDLE = 'idle'
    POSE_SETTLE = 'pose_settle'
    VISUAL_SEARCH = 'visual_search'
    GROUNDING = 'grounding'
    STANDOFF = 'standoff'
    COARSE_NAV = 'coarse_nav'
    NEAR_GROUNDING = 'near_grounding'
    VISUAL_SERVO = 'visual_servo'
    FINAL_GROUNDING = 'final_grounding'
    WAIT_FRESH_OBSERVATION = 'wait_fresh_observation'
    PLANNING = 'planning'
    TRANSIT = 'transit'
    PREGRASP_REOBSERVE = 'pregrasp_reobserve'
    APPROACH_PLANNING = 'approach_planning'
    APPROACH = 'approach'
    CLOSING = 'closing'
    LIFT = 'lift'
    VERIFY = 'verify'
    CARRY = 'carry'
    PICK_COMPLETE = 'pick_complete'
    PLACE_GROUNDING = 'place_grounding'
    PLACE_PLANNING = 'place_planning'
    PLACE_TRANSIT = 'place_transit'
    PLACE_APPROACH = 'place_approach'
    RELEASING = 'releasing'
    PLACE_RETREAT = 'place_retreat'
    POST_RELEASE_VERIFICATION = 'post_release_verification'
    COMPLETE = 'complete'
    CANCELED = 'canceled'
    FAILED = 'failed'


@dataclass(frozen=True)
class ExecutionState:
    """Parsed PiPER executor state."""

    trajectory: str
    owner: str = ''
    gripper: str = ''
    aperture_m: float | None = None
    command_id: int | None = None
    segment: str = ''
    gripper_command_id: int | None = None
    accepted_gripper_aperture_m: float | None = None
    executor_epoch: str = ''
    trajectory_contract_id: str = ''
    trajectory_token: str = ''
    trajectory_received_at: float | None = None
    trajectory_event_token: str = ''
    trajectory_event_received_at: float | None = None
    gripper_received_at: float | None = None

    @property
    def rejected(self) -> bool:
        """Return whether any executor channel reports rejection."""
        values = (self.trajectory, self.owner, self.gripper)
        return any('reject' in value.lower() for value in values)

    @property
    def canceled(self) -> bool:
        """Return whether trajectory ownership was canceled or preempted."""
        return 'cancel' in self.trajectory.lower() or 'preempt' in self.trajectory.lower()


PLACE_CONTRACT_SCHEMA = 'z_manip.place_contract.v2'
_PLACE_CONTRACT_FIELDS = frozenset({
    'schema',
    'schema_version',
    'goal_id',
    'frame_id',
    'joint_names',
    'phase_start_indices',
    'point_count',
    'trajectory_topic',
    'trajectory_digest_sha256',
    'request_id',
    'producer_epoch',
    'generation',
    'observation_stamp_ns',
    'observation_frame_id',
    'executor_epoch',
    'trajectory_contract_id',
    'trajectory_command_highwater',
    'trajectory_source_highwater_ns',
    'gripper_command_highwater',
    'gripper_source_highwater_ns',
})


def _reject_place_contract_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'placement trajectory contract repeats field {key!r}')
        result[key] = value
    return result


def _reject_place_contract_nonfinite(value: str) -> None:
    raise ValueError(
        f'placement trajectory contract contains non-finite {value}',
    )


def parse_place_contract(raw: str) -> dict[str, Any]:
    """Parse the exact placement-plan and executor-snapshot contract."""
    try:
        value = json.loads(
            str(raw),
            object_pairs_hook=_reject_place_contract_duplicate_keys,
            parse_constant=_reject_place_contract_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f'placement trajectory contract is invalid JSON: {error}') \
            from error
    if not isinstance(value, dict):
        raise ValueError('placement trajectory contract must be a JSON object')
    fields = set(value)
    if fields != _PLACE_CONTRACT_FIELDS:
        missing = sorted(_PLACE_CONTRACT_FIELDS - fields)
        unknown = sorted(fields - _PLACE_CONTRACT_FIELDS)
        raise ValueError(
            'placement trajectory contract fields are not exact: '
            f'missing={missing}, unknown={unknown}',
        )
    if (
        value['schema'] != PLACE_CONTRACT_SCHEMA
        or isinstance(value['schema_version'], bool)
        or not isinstance(value['schema_version'], int)
        or value['schema_version'] != 2
    ):
        raise ValueError('unsupported placement trajectory contract schema')

    def token(name: str) -> str:
        candidate = value[name]
        if (
            not isinstance(candidate, str)
            or not candidate.strip()
            or candidate != candidate.strip()
            or len(candidate) > 256
        ):
            raise ValueError(f'placement trajectory contract {name} is invalid')
        return candidate

    def integer(name: str, *, positive: bool) -> int:
        candidate = value[name]
        minimum = 1 if positive else 0
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or not minimum <= candidate < 2**63
        ):
            qualifier = 'positive' if positive else 'nonnegative'
            raise ValueError(
                f'placement trajectory contract {name} must be a {qualifier} integer',
            )
        return candidate

    for name in (
        'goal_id',
        'frame_id',
        'trajectory_topic',
        'trajectory_digest_sha256',
        'request_id',
        'producer_epoch',
        'observation_frame_id',
        'executor_epoch',
        'trajectory_contract_id',
    ):
        token(name)
    for name in (
        'point_count',
        'generation',
        'observation_stamp_ns',
        'trajectory_command_highwater',
        'gripper_command_highwater',
    ):
        integer(name, positive=True)
    for name in (
        'trajectory_source_highwater_ns',
        'gripper_source_highwater_ns',
    ):
        integer(name, positive=False)
    if value['trajectory_contract_id'] != value['goal_id']:
        raise ValueError(
            'placement trajectory contract future trajectory identity mismatches goal',
        )
    digest = value['trajectory_digest_sha256']
    if (
        len(digest) != 64
        or any(character not in '0123456789abcdef' for character in digest)
    ):
        raise ValueError(
            'placement trajectory contract digest must be lowercase SHA-256',
        )

    names = value['joint_names']
    if (
        not isinstance(names, list)
        or not names
        or any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or len(name) > 256
            for name in names
        )
        or len(set(names)) != len(names)
    ):
        raise ValueError('placement trajectory contract joint names are invalid')
    starts = value['phase_start_indices']
    if not isinstance(starts, dict) or set(starts) != {
        'transit', 'approach', 'retreat',
    }:
        raise ValueError('placement trajectory contract phase indices are invalid')
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        for index in starts.values()
    ):
        raise ValueError('placement trajectory contract phase indices are invalid')
    return value


def validate_place_trajectory_content(
    contract: Mapping[str, object],
    trajectory: object,
    *,
    expected_topic: str,
) -> None:
    """Bind one separately delivered JointTrajectory to its exact contract."""
    if not isinstance(contract, Mapping):
        raise ValueError('placement trajectory contract is unavailable')
    if (
        not isinstance(expected_topic, str)
        or not expected_topic
        or expected_topic != expected_topic.strip()
    ):
        raise ValueError('expected placement trajectory topic is invalid')
    if contract.get('trajectory_topic') != expected_topic:
        raise ValueError('placement trajectory topic does not match its contract')
    try:
        header = trajectory.header
        frame_id = header.frame_id
        digest = canonical_joint_trajectory_sha256(
            frame_id=frame_id,
            header_stamp=header.stamp,
            joint_names=trajectory.joint_names,
            points=trajectory.points,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f'placement trajectory content is invalid: {error}') from error
    if frame_id != contract.get('frame_id'):
        raise ValueError('placement trajectory frame does not match its contract')
    if digest != contract.get('trajectory_digest_sha256'):
        raise ValueError('placement trajectory content digest does not match its contract')


MAX_EXECUTION_OCCLUSION_DURATION_S = 3.0


@dataclass(frozen=True)
class ExecutionOcclusionConfig:
    """Bounds for a short, proprioception-checked wrist occlusion window."""

    max_duration_s: float = 3.0
    joint_state_max_age_s: float = 0.25
    execution_status_max_age_s: float = 0.30
    command_ack_timeout_s: float = 0.40
    near_contact_joint_tolerance_rad: float = 0.05
    lift_path_joint_tolerance_rad: float = 0.08
    max_path_regression_samples: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.max_duration_s,
            self.joint_state_max_age_s,
            self.execution_status_max_age_s,
            self.command_ack_timeout_s,
            self.near_contact_joint_tolerance_rad,
            self.lift_path_joint_tolerance_rad,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('execution occlusion limits must be finite and positive')
        if self.max_duration_s > MAX_EXECUTION_OCCLUSION_DURATION_S:
            raise ValueError('execution occlusion duration exceeds the 3.0 s hard limit')
        if self.command_ack_timeout_s >= self.max_duration_s:
            raise ValueError('occlusion command acknowledgement must precede timeout')
        if (
            isinstance(self.max_path_regression_samples, bool)
            or not isinstance(self.max_path_regression_samples, int)
            or self.max_path_regression_samples < 0
        ):
            raise ValueError('occlusion path regression bound must be non-negative')


@dataclass(frozen=True)
class ExecutionOcclusionDecision:
    """One fail-closed prediction decision for the current execution sample."""

    allowed: bool
    reason: str
    mode: str = ''
    path_index: int | None = None


class ExecutionOcclusionGate:
    """Allow only measured near-contact occlusion along an audited grasp path."""

    def __init__(self, config: ExecutionOcclusionConfig | None = None) -> None:
        self.config = config or ExecutionOcclusionConfig()
        self.reset()

    def reset(self) -> None:
        """Discard all near-contact, loss, and execution evidence."""
        self.request_id = ''
        self.producer_epoch = ''
        self.generation: int | None = None
        self.observation_frame_id = ''
        self.observation_serial: int | None = None
        self.observation_stamp_s: float | None = None
        self.observation_stamp_ns: int | None = None
        self.loss_observation_serial: int | None = None
        self.loss_observation_stamp_ns: int | None = None
        self.armed_at_s: float | None = None
        self.loss_at_s: float | None = None
        self.contact_confirmed_at_s: float | None = None
        self.lift_sent_at_s: float | None = None
        self.lift_completed_at_s: float | None = None
        self.near_contact_joints: tuple[float, ...] = ()
        self.maximum_lift_path_index = 0
        self.joint_source_stamp_ns: int | None = None
        self.joint_sequence: int | None = None
        self.loss_joint_source_stamp_ns: int | None = None
        self.loss_joint_sequence: int | None = None
        self.loss_joint_advanced = False
        self.last_lift_path_index: int | None = None
        self.last_path_joint_source_stamp_ns: int | None = None
        self.last_path_joint_sequence: int | None = None
        self._last_time_s: float | None = None

    @property
    def armed(self) -> bool:
        """Return whether exact near-contact evidence has been retained."""
        return self.armed_at_s is not None

    @property
    def loss_active(self) -> bool:
        """Return whether live tracking is currently replaced by prediction."""
        return self.loss_at_s is not None

    @property
    def contact_confirmed(self) -> bool:
        """Return whether measured gripper contact was accepted."""
        return self.contact_confirmed_at_s is not None

    def _time(self, now_s: float) -> float:
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError('execution occlusion time is not finite')
        if self._last_time_s is not None and now < self._last_time_s:
            raise ValueError('execution occlusion clock moved backwards')
        self._last_time_s = now
        return now

    @staticmethod
    def _vector(values: Sequence[float], label: str) -> tuple[float, ...]:
        vector = tuple(float(value) for value in values)
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError(f'{label} must be a finite non-empty vector')
        return vector

    @staticmethod
    def _max_error(first: Sequence[float], second: Sequence[float]) -> float:
        left = ExecutionOcclusionGate._vector(first, 'measured joints')
        right = ExecutionOcclusionGate._vector(second, 'reference joints')
        if len(left) != len(right):
            raise ValueError('execution occlusion joint widths differ')
        return max(abs(a - b) for a, b in zip(left, right))

    @staticmethod
    def _observation_identity(
        serial: int,
        stamp_ns: int,
    ) -> tuple[int, int]:
        if isinstance(serial, bool) or not isinstance(serial, int) or serial <= 0:
            raise ValueError('perception observation serial is invalid')
        if (
            isinstance(stamp_ns, bool)
            or not isinstance(stamp_ns, int)
            or stamp_ns <= 0
        ):
            raise ValueError('perception observation stamp is invalid')
        return serial, stamp_ns

    @staticmethod
    def _joint_source_identity(
        sequence: int,
        stamp_ns: int,
    ) -> tuple[int, int]:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ValueError('joint source sequence is invalid')
        if (
            isinstance(stamp_ns, bool)
            or not isinstance(stamp_ns, int)
            or stamp_ns <= 0
        ):
            raise ValueError('joint source stamp is invalid')
        return sequence, stamp_ns

    @staticmethod
    def _strictly_newer_pair(
        sequence: int,
        stamp_ns: int,
        previous_sequence: int,
        previous_stamp_ns: int,
    ) -> bool:
        return sequence > previous_sequence and stamp_ns > previous_stamp_ns

    @staticmethod
    def _same_pair(
        sequence: int,
        stamp_ns: int,
        previous_sequence: int,
        previous_stamp_ns: int,
    ) -> bool:
        return sequence == previous_sequence and stamp_ns == previous_stamp_ns

    def arm_near_contact(
        self,
        *,
        now_s: float,
        exact_authorized: bool,
        request_id: str,
        producer_epoch: str,
        generation: int,
        observation_serial: int,
        observation_stamp_ns: int,
        observation_frame_id: str,
        measured_joints: Sequence[float],
        approach_endpoint_joints: Sequence[float],
        joint_seen_at_s: float,
        joint_source_stamp_ns: int,
        joint_sequence: int,
    ) -> None:
        """Retain exact geometry only after measured approach completion."""
        now = float(now_s)
        joint_seen = float(joint_seen_at_s)
        if not exact_authorized:
            raise ValueError('near-contact observation is not exactly authorized')
        if (
            not str(request_id)
            or not str(producer_epoch)
            or not str(observation_frame_id)
        ):
            raise ValueError('near-contact perception ownership is unavailable')
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
            or isinstance(observation_serial, bool)
            or not isinstance(observation_serial, int)
            or observation_serial <= 0
        ):
            raise ValueError('near-contact perception generation is invalid')
        observation_serial, observation_stamp_ns = self._observation_identity(
            observation_serial,
            observation_stamp_ns,
        )
        if (
            not math.isfinite(joint_seen)
            or not 0.0 <= now - joint_seen <= self.config.joint_state_max_age_s
        ):
            raise ValueError('near-contact joint feedback is stale')
        source_sequence, source_stamp_ns = self._joint_source_identity(
            joint_sequence,
            joint_source_stamp_ns,
        )
        source_age = now - float(source_stamp_ns) * 1e-9
        if not 0.0 <= source_age <= self.config.joint_state_max_age_s:
            raise ValueError('near-contact joint source stamp is stale')
        endpoint = self._vector(approach_endpoint_joints, 'approach endpoint')
        if self._max_error(measured_joints, endpoint) > (
            self.config.near_contact_joint_tolerance_rad
        ):
            raise ValueError('measured arm did not reach the approach endpoint')
        self.reset()
        self._last_time_s = now
        self.request_id = str(request_id)
        self.producer_epoch = str(producer_epoch)
        self.generation = generation
        self.observation_frame_id = str(observation_frame_id)
        self.observation_serial = observation_serial
        self.observation_stamp_ns = observation_stamp_ns
        self.observation_stamp_s = float(observation_stamp_ns) * 1e-9
        self.armed_at_s = now
        self.near_contact_joints = endpoint
        self.joint_source_stamp_ns = source_stamp_ns
        self.joint_sequence = source_sequence

    def retain_exact_observation(
        self,
        *,
        request_id: str,
        producer_epoch: str,
        generation: int,
        observation_serial: int,
        observation_stamp_ns: int,
        observation_frame_id: str,
    ) -> bool:
        """Advance the exact live-perception watermark without extending time."""
        if self.armed_at_s is None or self.loss_at_s is not None:
            raise ValueError('exact observation retention requires live armed tracking')
        if (
            request_id != self.request_id
            or producer_epoch != self.producer_epoch
            or generation != self.generation
            or observation_frame_id != self.observation_frame_id
        ):
            raise ValueError('exact perception owner changed while execution was armed')
        serial, stamp_ns = self._observation_identity(
            observation_serial,
            observation_stamp_ns,
        )
        assert self.observation_serial is not None
        assert self.observation_stamp_ns is not None
        if self._same_pair(
            serial,
            stamp_ns,
            self.observation_serial,
            self.observation_stamp_ns,
        ):
            return False
        if not self._strictly_newer_pair(
            serial,
            stamp_ns,
            self.observation_serial,
            self.observation_stamp_ns,
        ):
            raise ValueError('exact perception watermark moved backwards or split')
        self.observation_serial = serial
        self.observation_stamp_ns = stamp_ns
        self.observation_stamp_s = float(stamp_ns) * 1e-9
        return True

    def mark_loss(
        self,
        now_s: float,
        *,
        joint_source_stamp_ns: int,
        joint_sequence: int,
    ) -> None:
        """Begin prediction only within the retained near-contact window."""
        now = self._time(now_s)
        if self.armed_at_s is None:
            raise ValueError('execution occlusion was not armed near contact')
        if now - self.armed_at_s > self.config.max_duration_s:
            raise ValueError('near-contact occlusion window already expired')
        if self.loss_at_s is not None:
            return
        source_sequence, source_stamp_ns = self._joint_source_identity(
            joint_sequence,
            joint_source_stamp_ns,
        )
        source_age = now - float(source_stamp_ns) * 1e-9
        if not 0.0 <= source_age <= self.config.joint_state_max_age_s:
            raise ValueError('joint source stamp is stale at perception loss')
        assert self.joint_sequence is not None
        assert self.joint_source_stamp_ns is not None
        if not (
            self._same_pair(
                source_sequence,
                source_stamp_ns,
                self.joint_sequence,
                self.joint_source_stamp_ns,
            )
            or self._strictly_newer_pair(
                source_sequence,
                source_stamp_ns,
                self.joint_sequence,
                self.joint_source_stamp_ns,
            )
        ):
            raise ValueError('joint source moved backwards or split at perception loss')
        self.joint_sequence = source_sequence
        self.joint_source_stamp_ns = source_stamp_ns
        self.loss_joint_sequence = source_sequence
        self.loss_joint_source_stamp_ns = source_stamp_ns
        self.loss_joint_advanced = False
        self.loss_observation_serial = self.observation_serial
        self.loss_observation_stamp_ns = self.observation_stamp_ns
        self.loss_at_s = now

    def tracking_restored(
        self,
        now_s: float,
        *,
        request_id: str,
        producer_epoch: str,
        generation: int,
        observation_serial: int,
        observation_stamp_ns: int,
        observation_frame_id: str,
    ) -> bool:
        """Return to live perception without extending the original window."""
        now = self._time(now_s)
        if self.loss_at_s is None:
            raise ValueError('cannot restore tracking when prediction is inactive')
        if (
            self.armed_at_s is not None
            and now - self.armed_at_s > self.config.max_duration_s
        ):
            raise ValueError('live tracking returned after the occlusion window')
        if (
            request_id != self.request_id
            or producer_epoch != self.producer_epoch
            or generation != self.generation
            or observation_frame_id != self.observation_frame_id
        ):
            raise ValueError('restored perception owner does not match the armed task')
        serial, stamp_ns = self._observation_identity(
            observation_serial,
            observation_stamp_ns,
        )
        assert self.loss_observation_serial is not None
        assert self.loss_observation_stamp_ns is not None
        if not self._strictly_newer_pair(
            serial,
            stamp_ns,
            self.loss_observation_serial,
            self.loss_observation_stamp_ns,
        ):
            return False
        self.observation_serial = serial
        self.observation_stamp_ns = stamp_ns
        self.observation_stamp_s = float(stamp_ns) * 1e-9
        self.loss_at_s = None
        return True

    def confirm_contact(self, now_s: float) -> None:
        """Record stable measured gripper contact before any lift allowance."""
        now = self._time(now_s)
        if self.armed_at_s is None:
            raise ValueError('cannot confirm contact before near-contact arming')
        if (
            self.loss_at_s is not None
            and now - self.armed_at_s > self.config.max_duration_s
        ):
            raise ValueError('contact arrived after the occlusion window')
        self.contact_confirmed_at_s = now

    def note_lift_sent(self, now_s: float) -> None:
        """Record publication of a cached-scene-validated lift trajectory."""
        now = self._time(now_s)
        if self.contact_confirmed_at_s is None:
            raise ValueError('cannot start lift before measured contact')
        if (
            self.loss_at_s is not None
            and self.armed_at_s is not None
            and now - self.armed_at_s > self.config.max_duration_s
        ):
            raise ValueError('lift started after the occlusion window')
        self.lift_sent_at_s = now

    def note_lift_completed(self, now_s: float) -> None:
        """Record identity-checked lift completion for bounded verification."""
        now = self._time(now_s)
        if self.lift_sent_at_s is None:
            raise ValueError('cannot complete a lift that was not sent')
        if (
            self.loss_at_s is not None
            and self.armed_at_s is not None
            and now - self.armed_at_s > self.config.max_duration_s
        ):
            raise ValueError('lift completed after the occlusion window')
        self.lift_completed_at_s = now

    def _joint_fresh(self, now: float, joint_seen_at_s: float) -> bool:
        seen = float(joint_seen_at_s)
        return (
            math.isfinite(seen)
            and 0.0 <= now - seen <= self.config.joint_state_max_age_s
        )

    def _status_fresh(self, now: float, status_seen_at_s: float | None) -> bool:
        if status_seen_at_s is None:
            return False
        seen = float(status_seen_at_s)
        return (
            math.isfinite(seen)
            and 0.0 <= now - seen <= self.config.execution_status_max_age_s
        )

    def _joint_source_decision(
        self,
        *,
        now: float,
        joint_source_stamp_ns: int,
        joint_sequence: int,
        allow_loss_watermark_sample: bool,
    ) -> ExecutionOcclusionDecision | None:
        try:
            sequence, stamp_ns = self._joint_source_identity(
                joint_sequence,
                joint_source_stamp_ns,
            )
        except (TypeError, ValueError) as error:
            return ExecutionOcclusionDecision(False, str(error))
        if not (
            0.0
            <= now - float(stamp_ns) * 1e-9
            <= self.config.joint_state_max_age_s
        ):
            return ExecutionOcclusionDecision(
                False,
                'joint source stamp is stale during occlusion',
            )
        if (
            self.loss_joint_sequence is None
            or self.loss_joint_source_stamp_ns is None
            or self.joint_sequence is None
            or self.joint_source_stamp_ns is None
        ):
            return ExecutionOcclusionDecision(
                False,
                'joint source loss watermark is unavailable',
            )
        same_latest = self._same_pair(
            sequence,
            stamp_ns,
            self.joint_sequence,
            self.joint_source_stamp_ns,
        )
        newer_latest = self._strictly_newer_pair(
            sequence,
            stamp_ns,
            self.joint_sequence,
            self.joint_source_stamp_ns,
        )
        if not (same_latest or newer_latest):
            return ExecutionOcclusionDecision(
                False,
                'joint source moved backwards or split during occlusion',
            )
        if not self.loss_joint_advanced:
            newer_than_loss = self._strictly_newer_pair(
                sequence,
                stamp_ns,
                self.loss_joint_sequence,
                self.loss_joint_source_stamp_ns,
            )
            same_as_loss = self._same_pair(
                sequence,
                stamp_ns,
                self.loss_joint_sequence,
                self.loss_joint_source_stamp_ns,
            )
            if not newer_than_loss:
                if allow_loss_watermark_sample and same_as_loss:
                    return None
                if (
                    same_as_loss
                    and self.loss_at_s is not None
                    and now - self.loss_at_s
                    <= min(
                        self.config.joint_state_max_age_s,
                        self.config.command_ack_timeout_s,
                    )
                ):
                    return ExecutionOcclusionDecision(
                        True,
                        '',
                        mode='awaiting_joint_sample',
                    )
                return ExecutionOcclusionDecision(
                    False,
                    'joint source did not advance after perception loss',
                )
        return None

    def _commit_joint_source(
        self,
        joint_source_stamp_ns: int,
        joint_sequence: int,
    ) -> None:
        """Advance source watermarks only after all phase invariants pass."""
        assert self.loss_joint_sequence is not None
        assert self.loss_joint_source_stamp_ns is not None
        assert self.joint_sequence is not None
        assert self.joint_source_stamp_ns is not None
        if self._strictly_newer_pair(
            joint_sequence,
            joint_source_stamp_ns,
            self.loss_joint_sequence,
            self.loss_joint_source_stamp_ns,
        ):
            self.loss_joint_advanced = True
        if self._strictly_newer_pair(
            joint_sequence,
            joint_source_stamp_ns,
            self.joint_sequence,
            self.joint_source_stamp_ns,
        ):
            self.joint_sequence = joint_sequence
            self.joint_source_stamp_ns = joint_source_stamp_ns

    def _lift_path_decision(
        self,
        measured_joints: Sequence[float],
        lift_path: Sequence[Sequence[float]],
        *,
        joint_source_stamp_ns: int,
        joint_sequence: int,
        commit_progress: bool,
    ) -> ExecutionOcclusionDecision:
        rows = tuple(self._vector(row, 'lift path row') for row in lift_path)
        if len(rows) < 2:
            return ExecutionOcclusionDecision(False, 'lift path is unavailable')
        measured = self._vector(measured_joints, 'measured joints')
        if any(len(row) != len(measured) for row in rows):
            return ExecutionOcclusionDecision(False, 'lift path joint width changed')
        distances = tuple(
            max(abs(value - expected) for value, expected in zip(measured, row))
            for row in rows
        )
        index = min(range(len(rows)), key=distances.__getitem__)
        if distances[index] > self.config.lift_path_joint_tolerance_rad:
            return ExecutionOcclusionDecision(
                False,
                'measured arm left the validated lift path',
                path_index=index,
            )
        if (
            index + self.config.max_path_regression_samples
            < self.maximum_lift_path_index
        ):
            return ExecutionOcclusionDecision(
                False,
                'measured lift progress moved backwards',
                path_index=index,
            )
        if commit_progress:
            assert self.joint_sequence is not None
            assert self.joint_source_stamp_ns is not None
            if (
                self.last_lift_path_index is not None
                and index != self.last_lift_path_index
                and self.last_path_joint_sequence is not None
                and self.last_path_joint_source_stamp_ns is not None
                and not self._strictly_newer_pair(
                    joint_sequence,
                    joint_source_stamp_ns,
                    self.last_path_joint_sequence,
                    self.last_path_joint_source_stamp_ns,
                )
            ):
                return ExecutionOcclusionDecision(
                    False,
                    'lift path changed without a new joint source sample',
                    path_index=index,
                )
            self.maximum_lift_path_index = max(
                self.maximum_lift_path_index,
                index,
            )
            self.last_lift_path_index = index
            self.last_path_joint_sequence = joint_sequence
            self.last_path_joint_source_stamp_ns = joint_source_stamp_ns
        return ExecutionOcclusionDecision(
            True,
            '',
            mode='predicted_lift',
            path_index=index,
        )

    def evaluate(
        self,
        *,
        now_s: float,
        phase: RuntimePhase,
        measured_joints: Sequence[float],
        joint_seen_at_s: float,
        joint_source_stamp_ns: int,
        joint_sequence: int,
        close_command_sent_at_s: float | None = None,
        close_acknowledged: bool = False,
        execution_status_seen_at_s: float | None = None,
        lift_path: Sequence[Sequence[float]] = (),
        lift_execution_active: bool = False,
        lift_execution_completed: bool = False,
        allow_loss_watermark_sample: bool = False,
    ) -> ExecutionOcclusionDecision:
        """Check current proprioception and command progress without an oracle."""
        try:
            now = self._time(now_s)
        except (TypeError, ValueError) as error:
            return ExecutionOcclusionDecision(False, str(error))
        if self.armed_at_s is None or self.loss_at_s is None:
            return ExecutionOcclusionDecision(False, 'predicted occlusion is not active')
        if now - self.armed_at_s > self.config.max_duration_s:
            return ExecutionOcclusionDecision(False, 'predicted occlusion window expired')
        try:
            joint_fresh = self._joint_fresh(now, joint_seen_at_s)
        except (TypeError, ValueError):
            joint_fresh = False
        if not joint_fresh:
            return ExecutionOcclusionDecision(False, 'joint feedback is stale during occlusion')
        joint_source_decision = self._joint_source_decision(
            now=now,
            joint_source_stamp_ns=joint_source_stamp_ns,
            joint_sequence=joint_sequence,
            allow_loss_watermark_sample=allow_loss_watermark_sample,
        )
        if phase is RuntimePhase.CLOSING:
            try:
                endpoint_error = self._max_error(
                    measured_joints,
                    self.near_contact_joints,
                )
            except ValueError as error:
                return ExecutionOcclusionDecision(False, str(error))
            if endpoint_error > self.config.near_contact_joint_tolerance_rad:
                return ExecutionOcclusionDecision(
                    False,
                    'arm moved away from the near-contact endpoint while closing',
                )
            if close_command_sent_at_s is None:
                return ExecutionOcclusionDecision(False, 'close command was not issued')
            try:
                command_age = now - float(close_command_sent_at_s)
            except (TypeError, ValueError):
                return ExecutionOcclusionDecision(False, 'close command time is invalid')
            if not math.isfinite(command_age) or command_age < 0.0:
                return ExecutionOcclusionDecision(False, 'close command time is invalid')
            if close_acknowledged:
                try:
                    status_fresh = self._status_fresh(
                        now,
                        execution_status_seen_at_s,
                    )
                except (TypeError, ValueError):
                    status_fresh = False
                if not status_fresh:
                    return ExecutionOcclusionDecision(
                        False,
                        'gripper close feedback is stale during occlusion',
                    )
            elif command_age > self.config.command_ack_timeout_s:
                return ExecutionOcclusionDecision(
                    False,
                    'gripper close acknowledgement timed out during occlusion',
                )
            if joint_source_decision is not None:
                return joint_source_decision
            self._commit_joint_source(
                joint_source_stamp_ns,
                joint_sequence,
            )
            return ExecutionOcclusionDecision(True, '', mode='predicted_closing')
        if phase is RuntimePhase.LIFT:
            if self.contact_confirmed_at_s is None or self.lift_sent_at_s is None:
                return ExecutionOcclusionDecision(
                    False,
                    'lift occlusion lacks measured contact or trajectory evidence',
                )
            lift_age = now - self.lift_sent_at_s
            if lift_age < 0.0:
                return ExecutionOcclusionDecision(False, 'lift command time is in the future')
            if lift_execution_active or lift_execution_completed:
                try:
                    status_fresh = self._status_fresh(
                        now,
                        execution_status_seen_at_s,
                    )
                except (TypeError, ValueError):
                    status_fresh = False
                if not status_fresh:
                    return ExecutionOcclusionDecision(
                        False,
                        'lift execution feedback is stale during occlusion',
                    )
            elif lift_age > self.config.command_ack_timeout_s:
                return ExecutionOcclusionDecision(
                    False,
                    'lift execution acknowledgement timed out during occlusion',
                )
            try:
                path_decision = self._lift_path_decision(
                    measured_joints,
                    lift_path,
                    joint_source_stamp_ns=joint_source_stamp_ns,
                    joint_sequence=joint_sequence,
                    commit_progress=False,
                )
            except (TypeError, ValueError):
                return ExecutionOcclusionDecision(False, 'lift path is malformed')
            if not path_decision.allowed:
                return path_decision
            if joint_source_decision is not None:
                return joint_source_decision
            self._commit_joint_source(
                joint_source_stamp_ns,
                joint_sequence,
            )
            if allow_loss_watermark_sample:
                return path_decision
            try:
                return self._lift_path_decision(
                    measured_joints,
                    lift_path,
                    joint_source_stamp_ns=joint_source_stamp_ns,
                    joint_sequence=joint_sequence,
                    commit_progress=True,
                )
            except (TypeError, ValueError):
                return ExecutionOcclusionDecision(False, 'lift path is malformed')
        if phase is RuntimePhase.VERIFY:
            if self.contact_confirmed_at_s is None or self.lift_completed_at_s is None:
                return ExecutionOcclusionDecision(
                    False,
                    'verification occlusion lacks completed lift evidence',
                )
            try:
                path_decision = self._lift_path_decision(
                    measured_joints,
                    lift_path,
                    joint_source_stamp_ns=joint_source_stamp_ns,
                    joint_sequence=joint_sequence,
                    commit_progress=False,
                )
            except (TypeError, ValueError):
                return ExecutionOcclusionDecision(False, 'lift path is malformed')
            if not path_decision.allowed:
                return path_decision
            if joint_source_decision is not None:
                return joint_source_decision
            self._commit_joint_source(
                joint_source_stamp_ns,
                joint_sequence,
            )
            if not allow_loss_watermark_sample:
                try:
                    path_decision = self._lift_path_decision(
                        measured_joints,
                        lift_path,
                        joint_source_stamp_ns=joint_source_stamp_ns,
                        joint_sequence=joint_sequence,
                        commit_progress=True,
                    )
                except (TypeError, ValueError):
                    return ExecutionOcclusionDecision(
                        False,
                        'lift path is malformed',
                    )
                if not path_decision.allowed:
                    return path_decision
            return ExecutionOcclusionDecision(
                True,
                '',
                mode='predicted_verify',
                path_index=path_decision.path_index,
            )
        phase_name = phase.value if isinstance(phase, RuntimePhase) else str(phase)
        return ExecutionOcclusionDecision(
            False,
            f'predicted occlusion is forbidden during {phase_name}',
        )


def parse_execution_status(value: str) -> ExecutionState:
    """Parse the version-tolerant semicolon status emitted by the executor."""
    fields = [field.strip() for field in str(value).split(';') if field.strip()]
    if not fields:
        raise ValueError('execution status is empty')
    trajectory = fields[0]
    values: dict[str, str] = {}
    for field in fields[1:]:
        key, separator, item = field.partition('=')
        normalized_key = key.strip()
        if not separator or not normalized_key:
            raise ValueError('execution status field is malformed')
        if normalized_key in values:
            raise ValueError(
                f'execution status repeats field {normalized_key!r}',
            )
        values[normalized_key] = item.strip()
    aperture = None
    if 'aperture' in values:
        aperture = float(values['aperture'])
        if not math.isfinite(aperture) or aperture < 0.0:
            raise ValueError('execution aperture must be finite and non-negative')

    def optional_nonnegative_int(name: str) -> int | None:
        if name not in values:
            return None
        parsed = int(values[name])
        if parsed < 0:
            raise ValueError(f'{name} must be non-negative')
        return parsed

    def optional_identity(name: str) -> str:
        if name not in values:
            return ''
        identity = values[name]
        if (
            not identity
            or len(identity) > 128
            or any(
                ord(character) < 33
                or ord(character) > 126
                or character in ';|='
                for character in identity
            )
        ):
            raise ValueError(f'{name} is invalid')
        return identity

    executor_epoch = optional_identity('executor_epoch')
    trajectory_contract_id = optional_identity('trajectory_contract_id')
    trajectory_token = optional_identity('trajectory_token')
    trajectory_received_at = None
    if 'trajectory_received_at' in values:
        raw_received_at = values['trajectory_received_at']
        if raw_received_at.lower() != 'none':
            trajectory_received_at = float(raw_received_at)
            if (
                not math.isfinite(trajectory_received_at)
                or trajectory_received_at < 0.0
            ):
                raise ValueError(
                    'trajectory_received_at must be finite and non-negative',
                )

    trajectory_event_token = optional_identity('trajectory_event_token')
    if trajectory_event_token == 'none':
        trajectory_event_token = ''
    trajectory_event_received_at = None
    if 'trajectory_event_received_at' in values:
        raw_event_received_at = values['trajectory_event_received_at']
        if raw_event_received_at.lower() != 'none':
            trajectory_event_received_at = float(raw_event_received_at)
            if (
                not math.isfinite(trajectory_event_received_at)
                or trajectory_event_received_at < 0.0
            ):
                raise ValueError(
                    'trajectory_event_received_at must be finite and '
                    'non-negative',
                )
    if bool(trajectory_event_token) != (
        trajectory_event_received_at is not None
    ):
        raise ValueError(
            'trajectory event token and source identity must appear together',
        )

    gripper_received_at = None
    if 'gripper_received_at' in values:
        raw_gripper_received_at = values['gripper_received_at']
        if raw_gripper_received_at.lower() != 'none':
            gripper_received_at = float(raw_gripper_received_at)
            if (
                not math.isfinite(gripper_received_at)
                or gripper_received_at < 0.0
            ):
                raise ValueError(
                    'gripper_received_at must be finite and non-negative',
                )

    segment = values.get('segment', '')
    if segment in {'place_transit', 'place_approach', 'place_retreat'}:
        missing = sorted({
            'executor_epoch',
            'trajectory_contract_id',
            'trajectory_received_at',
        } - values.keys())
        if missing:
            raise ValueError(
                'place execution status omitted identity fields: '
                + ', '.join(missing),
            )
        if trajectory_received_at is None:
            raise ValueError(
                'place trajectory_received_at must be finite and non-negative',
            )
        if segment == 'place_transit':
            if trajectory_contract_id != 'none':
                raise ValueError(
                    'place_transit trajectory contract must be none',
                )
        elif trajectory_contract_id == 'none':
            raise ValueError(
                f'{segment} trajectory contract is missing',
            )

    gripper_command_id = optional_nonnegative_int('gripper_command_id')
    if (
        (gripper_command_id is None and gripper_received_at is not None)
        or (gripper_command_id is not None and gripper_command_id > 0
            and gripper_received_at is None)
        or (gripper_command_id == 0 and gripper_received_at is not None)
    ):
        raise ValueError(
            'gripper command and source identity must appear together',
        )

    accepted_aperture = None
    gripper = values.get('gripper', '')
    if gripper.lower().startswith('accepted:'):
        accepted_aperture = float(gripper.split(':', 1)[1])
        if not math.isfinite(accepted_aperture) or accepted_aperture < 0.0:
            raise ValueError('accepted gripper aperture is invalid')
    return ExecutionState(
        trajectory=trajectory,
        owner=values.get('owner', ''),
        gripper=gripper,
        aperture_m=aperture,
        command_id=optional_nonnegative_int('command_id'),
        segment=segment,
        gripper_command_id=gripper_command_id,
        accepted_gripper_aperture_m=accepted_aperture,
        executor_epoch=executor_epoch,
        trajectory_contract_id=trajectory_contract_id,
        trajectory_token=trajectory_token,
        trajectory_received_at=trajectory_received_at,
        trajectory_event_token=trajectory_event_token,
        trajectory_event_received_at=trajectory_event_received_at,
        gripper_received_at=gripper_received_at,
    )


def grasp_close_aperture(
    required_width_m: float | None,
    *,
    fallback_m: float,
    squeeze_m: float,
    minimum_m: float,
    maximum_m: float,
) -> float:
    """Turn a candidate width into a bounded, object-scaled close command."""
    scalars = (fallback_m, squeeze_m, minimum_m, maximum_m)
    if not all(math.isfinite(float(item)) for item in scalars):
        raise ValueError('gripper aperture settings must be finite')
    if squeeze_m < 0.0 or not 0.0 <= minimum_m < maximum_m:
        raise ValueError('configured gripper aperture bounds are invalid')
    if required_width_m is None:
        value = float(fallback_m)
    else:
        value = float(required_width_m)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError('planned grasp width must be finite and non-negative')
        value -= float(squeeze_m)
    return min(max(value, float(minimum_m)), float(maximum_m))


def validate_grasp_aperture_contract(
    *,
    candidate_min_m: float,
    candidate_max_m: float,
    open_aperture_m: float,
    squeeze_m: float,
    command_min_m: float,
    command_max_m: float,
    contact_margin_m: float,
) -> None:
    """Reject planner, actuator, and verifier aperture ranges with no overlap."""
    values = (
        candidate_min_m,
        candidate_max_m,
        open_aperture_m,
        squeeze_m,
        command_min_m,
        command_max_m,
        contact_margin_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('grasp aperture contract values must be finite')
    if not 0.0 <= candidate_min_m < candidate_max_m:
        raise ValueError('candidate aperture interval is invalid')
    if not 0.0 <= command_min_m < command_max_m:
        raise ValueError('gripper command interval is invalid')
    if squeeze_m < 0.0 or contact_margin_m <= 0.0:
        raise ValueError('squeeze and contact margin are invalid')
    if open_aperture_m > command_max_m or candidate_max_m >= open_aperture_m:
        raise ValueError('candidate or open aperture exceeds the actuator contract')
    widest_close = grasp_close_aperture(
        candidate_max_m,
        fallback_m=command_min_m,
        squeeze_m=squeeze_m,
        minimum_m=command_min_m,
        maximum_m=command_max_m,
    )
    held_min = widest_close + contact_margin_m
    held_max = open_aperture_m - 2.0 * contact_margin_m
    if held_min >= held_max:
        raise ValueError('planner widths leave no grasp-verification contact interval')


@dataclass(frozen=True)
class SynchronizedSerial:
    """One unique, fresh set of perception stream versions."""

    serial: int
    stamp_s: float
    stream_versions: Mapping[str, int]


@dataclass(frozen=True)
class PlacementTrajectorySegment:
    """One canonical placement phase with time rebased to zero."""

    name: str
    positions: tuple[tuple[float, ...], ...]
    times_s: tuple[float, ...]


def trajectory_segment_frame_id(
    segment: str,
    contract_id: str | None,
    *,
    execution_token: str | None = None,
) -> str:
    """Encode bounded transaction identities in trajectory metadata."""
    if not isinstance(segment, str) or not segment:
        raise ValueError('trajectory segment must be non-empty')
    encoded = segment
    if segment in {'place_approach', 'place_retreat'}:
        if not isinstance(contract_id, str) or not 1 <= len(contract_id) <= 128:
            raise ValueError('placement trajectory contract ID has invalid length')
        if any(
            ord(character) < 33
            or ord(character) > 126
            or character in ';|='
            for character in contract_id
        ):
            raise ValueError(
                'placement trajectory contract ID is not bounded ASCII',
            )
        encoded = f'{segment}|contract={contract_id}'
    elif contract_id is not None:
        raise ValueError('non-place trajectory cannot carry a place contract ID')
    if execution_token is None:
        return encoded
    if not isinstance(execution_token, str) or not 1 <= len(execution_token) <= 128:
        raise ValueError('trajectory execution token has invalid length')
    if any(
        ord(character) < 33
        or ord(character) > 126
        or character in ';|='
        for character in execution_token
    ):
        raise ValueError('trajectory execution token is not bounded ASCII')
    return f'{encoded}|token={execution_token}'


def split_placement_trajectory(
    positions: object,
    times_s: object,
    phase_start_indices: Mapping[str, int],
) -> tuple[PlacementTrajectorySegment, ...]:
    """Split an audited joined place path around the release boundary."""
    try:
        rows = tuple(tuple(float(value) for value in row) for row in positions)
        times = tuple(float(value) for value in times_s)
    except (TypeError, ValueError) as error:
        raise ValueError('placement trajectory must be numeric') from error
    if len(rows) < 4 or len(times) != len(rows):
        raise ValueError('placement trajectory is too short or misaligned')
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError('placement trajectory joint width changed')
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError('placement trajectory positions are non-finite')
    if not all(math.isfinite(value) for value in times):
        raise ValueError('placement trajectory times are non-finite')
    if times[0] < 0.0 or any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError('placement trajectory times must increase strictly')
    required = ('transit', 'approach', 'retreat')
    if set(phase_start_indices) != set(required):
        raise ValueError('placement trajectory must declare all phases')
    starts = tuple(int(phase_start_indices[name]) for name in required)
    if not (starts[0] == 0 < starts[1] < starts[2] < len(rows)):
        raise ValueError('placement phase indices are invalid')
    segments = []
    for index, name in enumerate(required):
        begin = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else len(rows)
        if begin > 0:
            begin -= 1
        segment_rows = rows[begin:end]
        segment_times = times[begin:end]
        if len(segment_rows) < 2:
            raise ValueError(f'placement {name} segment contains no motion')
        origin = segment_times[0]
        rebased = tuple(value - origin for value in segment_times)
        segments.append(PlacementTrajectorySegment(
            f'place_{name}', segment_rows, rebased,
        ))
    return tuple(segments)


class ObservationSerialGate:
    """Issue a serial only when all perception streams are fresh and synchronized."""

    def __init__(
        self,
        streams: tuple[str, ...] = ('target', 'target_cloud', 'scene_cloud'),
        *,
        sync_slop_s: float = 0.12,
        max_age_s: float = 0.35,
    ) -> None:
        """Configure stream names, timestamp slop, and maximum age."""
        if not streams or len(set(streams)) != len(streams):
            raise ValueError('streams must be unique and non-empty')
        if sync_slop_s <= 0.0 or max_age_s <= 0.0:
            raise ValueError('synchronization limits must be positive')
        self.streams = streams
        self.sync_slop_s = float(sync_slop_s)
        self.max_age_s = float(max_age_s)
        self._versions = {name: 0 for name in streams}
        self._stamps: dict[str, float] = {}
        self._last_tuple: tuple[int, ...] | None = None
        self._serial = 0

    def update(self, stream: str, stamp_s: float) -> None:
        """Record a new version and timestamp for one observation stream."""
        if stream not in self._versions:
            raise KeyError(f'unknown observation stream {stream!r}')
        stamp = float(stamp_s)
        if not math.isfinite(stamp) or stamp < 0.0:
            raise ValueError('observation stamp must be finite and non-negative')
        previous = self._stamps.get(stream)
        if previous is not None:
            if stamp < previous:
                raise ValueError(f'{stream} time moved backwards')
            if stamp == previous:
                return
        self._versions[stream] += 1
        self._stamps[stream] = stamp

    def snapshot(self, now_s: float) -> SynchronizedSerial | None:
        """Return a serial when all streams form a fresh synchronized set."""
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError('current time must be finite')
        if set(self._stamps) != set(self.streams):
            return None
        stamps = tuple(self._stamps[name] for name in self.streams)
        if max(stamps) - min(stamps) > self.sync_slop_s:
            return None
        newest = max(stamps)
        if now - newest > self.max_age_s or newest - now > self.sync_slop_s:
            return None
        versions = tuple(self._versions[name] for name in self.streams)
        if versions != self._last_tuple:
            self._serial += 1
            self._last_tuple = versions
        return SynchronizedSerial(self._serial, newest, dict(self._versions))


@dataclass(frozen=True)
class SafetyAction:
    """Immediate actuator response requested by the safety core."""

    stop_base: bool = False
    cancel_navigation: bool = False
    cancel_arm: bool = False
    reason: str = ''


class PostureState(str, Enum):
    """Result of evaluating the latest measured platform attitude."""

    WAITING = 'waiting'
    SAFE = 'safe'
    UNSAFE = 'unsafe'


@dataclass(frozen=True)
class PostureAssessment:
    """One fail-closed posture decision with a diagnostic reason."""

    state: PostureState
    reason: str = ''

    @property
    def safe(self) -> bool:
        """Return whether task-owned motion may proceed."""
        return self.state is PostureState.SAFE


class PostureSafetyGate:
    """
    Require fresh, finite, upright state estimation during a task.

    A task may briefly wait for its first state-estimation sample, but motion
    remains inhibited during that interval. Once a sample has been observed,
    stale, future-dated, non-finite, or out-of-bounds attitude is immediately
    unsafe rather than being treated as another startup wait.
    """

    def __init__(
        self,
        *,
        max_roll_rad: float,
        max_pitch_rad: float,
        max_age_s: float,
        acquisition_timeout_s: float,
    ) -> None:
        """Configure attitude limits and state-estimation timing bounds."""
        values = (
            max_roll_rad,
            max_pitch_rad,
            max_age_s,
            acquisition_timeout_s,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError('posture limits and timeouts must be finite and positive')
        self.max_roll_rad = float(max_roll_rad)
        self.max_pitch_rad = float(max_pitch_rad)
        self.max_age_s = float(max_age_s)
        self.acquisition_timeout_s = float(acquisition_timeout_s)
        self._task_started_at_s: float | None = None
        self._roll_rad: float | None = None
        self._pitch_rad: float | None = None
        self._seen_at_s: float | None = None

    def begin(self, now_s: float) -> None:
        """Start acquisition while retaining only a fresh pre-task sample."""
        now = float(now_s)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError('posture task start time must be finite and non-negative')
        self._task_started_at_s = now
        if (
            now == 0.0
            or (
                self._seen_at_s is not None
                and (
                    self._seen_at_s > now
                    or now - self._seen_at_s > self.max_age_s
                )
            )
        ):
            self._roll_rad = None
            self._pitch_rad = None
            self._seen_at_s = None

    def update(self, roll_rad: float, pitch_rad: float, *, seen_at_s: float) -> None:
        """Record the latest attitude and local receipt time."""
        seen_at = float(seen_at_s)
        if not math.isfinite(seen_at) or seen_at < 0.0:
            raise ValueError('posture receipt time must be finite and non-negative')
        self._roll_rad = float(roll_rad)
        self._pitch_rad = float(pitch_rad)
        self._seen_at_s = seen_at

    def assess(self, now_s: float) -> PostureAssessment:
        """Return whether task-owned motion is currently posture-safe."""
        now = float(now_s)
        if not math.isfinite(now) or now < 0.0:
            return PostureAssessment(
                PostureState.UNSAFE,
                'state-estimation freshness clock is invalid',
            )
        if self._task_started_at_s is None:
            raise RuntimeError('posture gate has not started a task')
        if self._task_started_at_s == 0.0 and now > 0.0:
            self._task_started_at_s = now
            if self._seen_at_s is None or self._seen_at_s == 0.0:
                self._roll_rad = None
                self._pitch_rad = None
                self._seen_at_s = None
        if now < self._task_started_at_s:
            return PostureAssessment(
                PostureState.UNSAFE,
                'state-estimation freshness clock moved backwards',
            )
        if self._seen_at_s is None:
            elapsed = now - self._task_started_at_s
            if elapsed <= self.acquisition_timeout_s:
                return PostureAssessment(
                    PostureState.WAITING,
                    'waiting for the first state-estimation posture sample',
                )
            return PostureAssessment(
                PostureState.UNSAFE,
                'state-estimation posture unavailable after acquisition timeout',
            )

        assert self._roll_rad is not None and self._pitch_rad is not None
        if not all(math.isfinite(value) for value in (self._roll_rad, self._pitch_rad)):
            return PostureAssessment(
                PostureState.UNSAFE,
                'state-estimation posture contains a non-finite attitude',
            )
        age = now - self._seen_at_s
        if age < 0.0:
            return PostureAssessment(
                PostureState.UNSAFE,
                'state-estimation posture timestamp is in the future',
            )
        if age > self.max_age_s:
            return PostureAssessment(
                PostureState.UNSAFE,
                f'state-estimation posture is stale ({age:.3f}s > {self.max_age_s:.3f}s)',
            )
        if (
            abs(self._roll_rad) > self.max_roll_rad
            or abs(self._pitch_rad) > self.max_pitch_rad
        ):
            return PostureAssessment(
                PostureState.UNSAFE,
                'base posture limit exceeded: '
                f'roll={self._roll_rad:.4f}rad limit={self.max_roll_rad:.4f}rad, '
                f'pitch={self._pitch_rad:.4f}rad limit={self.max_pitch_rad:.4f}rad',
            )
        return PostureAssessment(PostureState.SAFE)


class TaskGenerationGuard:
    """Invalidate asynchronous results whenever task ownership changes."""

    def __init__(self) -> None:
        self._generation = 0

    @property
    def current(self) -> int:
        """Return the generation assigned to newly submitted work."""
        return self._generation

    def advance(self) -> int:
        """Invalidate every token issued for the previous task."""
        self._generation += 1
        return self._generation

    def accepts(self, generation: int | None) -> bool:
        """Return whether an asynchronous result still belongs to this task."""
        return (
            not isinstance(generation, bool)
            and isinstance(generation, int)
            and generation == self._generation
        )


def terminal_result(phase: RuntimePhase) -> str:
    """Map terminal runtime phases to the stable task-status result field."""
    return {
        RuntimePhase.PICK_COMPLETE: 'pick_complete',
        RuntimePhase.COMPLETE: 'mobile_manip_complete',
        RuntimePhase.CANCELED: 'canceled',
    }.get(phase, '')


def wrap_angle(angle_rad: float) -> float:
    """Return a finite angle in ``[-pi, pi)``."""
    value = float(angle_rad)
    if not math.isfinite(value):
        raise ValueError('angle must be finite')
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _normalized_image_bbox(normalized_xyxy: object) -> tuple[float, float, float, float]:
    """Validate and return one normalized image bounding box."""
    try:
        values = tuple(float(value) for value in normalized_xyxy)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError('target bbox must contain numeric coordinates') from error
    if len(values) != 4:
        raise ValueError('target bbox must contain four coordinates')
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError('target bbox is outside normalized image bounds')
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError('target bbox has no image area')
    return x1, y1, x2, y2


def _image_margin(value: float, label: str) -> float:
    margin = float(value)
    if not math.isfinite(margin) or not 0.0 <= margin < 0.5:
        raise ValueError(f'{label} margin must be in [0, 0.5)')
    return margin


def horizontal_edge_direction(
    normalized_xyxy: object,
    *,
    margin_ratio: float,
) -> int:
    """
    Map a horizontal image-edge violation to a base-yaw direction.

    Positive yaw moves a left-edge target toward image center; negative yaw
    does the same for a right-edge target. A box spanning both margins is not a
    trustworthy recenter observation and is rejected instead of guessed.
    """
    x1, _y1, x2, _y2 = _normalized_image_bbox(normalized_xyxy)
    margin = _image_margin(margin_ratio, 'horizontal')
    touches_left = x1 < margin
    touches_right = x2 > 1.0 - margin
    if touches_left and touches_right:
        raise ValueError('target bbox spans both horizontal safety margins')
    if touches_left:
        return 1
    if touches_right:
        return -1
    return 0


def vertical_edge_direction(
    normalized_xyxy: object,
    *,
    margin_ratio: float,
) -> int:
    """Map vertical clipping to the camera-pitch direction that recenters it."""
    _x1, y1, _x2, y2 = _normalized_image_bbox(normalized_xyxy)
    margin = _image_margin(margin_ratio, 'vertical')
    touches_top = y1 < margin
    touches_bottom = y2 > 1.0 - margin
    if touches_top and touches_bottom:
        raise ValueError('target bbox spans both vertical safety margins')
    if touches_top:
        return 1
    if touches_bottom:
        return -1
    return 0


@dataclass(frozen=True)
class VisualSearchConfig:
    """Robot-independent limits for a bounded odometry-yaw camera search."""

    yaw_step_rad: float = math.radians(20.0)
    max_yaw_offset_rad: float = math.radians(60.0)
    yaw_tolerance_rad: float = math.radians(1.0)
    yaw_gain: float = 1.5
    max_yaw_rate_rps: float = 0.30
    turn_timeout_s: float = 8.0
    max_turn_timeout_s: float = 30.0
    max_planar_drift_m: float = 0.15
    position_hold_deadband_m: float = 0.01
    position_completion_tolerance_m: float = 0.05
    moving_rebound_reacquire_m: float = 0.10
    position_hold_gain_s_inv: float = 1.0
    max_position_hold_speed_mps: float = 0.10
    position_hold_slowdown_radius_m: float = 0.0
    min_position_hold_speed_mps: float = 0.0
    position_hold_timeout_s: float = 4.0
    settle_max_linear_speed_mps: float = 0.035
    settle_max_angular_speed_rps: float = 0.05
    settle_yaw_tolerance_rad: float = math.radians(2.0)
    position_heading_reacquire_tolerance_rad: float = math.radians(4.0)
    min_yaw_rate_rps: float = 0.0
    stationary_wait_timeout_s: float = 0.0
    stationary_quiet_window_s: float = 0.35
    stationary_max_odom_gap_s: float = 0.15
    settle_reacquire_budget_s: float = 2.0
    deadline_grace_s: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.yaw_step_rad,
            self.max_yaw_offset_rad,
            self.yaw_tolerance_rad,
            self.settle_yaw_tolerance_rad,
            self.position_heading_reacquire_tolerance_rad,
            self.yaw_gain,
            self.max_yaw_rate_rps,
            self.turn_timeout_s,
            self.max_turn_timeout_s,
            self.max_planar_drift_m,
            self.position_completion_tolerance_m,
            self.moving_rebound_reacquire_m,
            self.position_hold_gain_s_inv,
            self.max_position_hold_speed_mps,
            self.position_hold_timeout_s,
            self.settle_max_linear_speed_mps,
            self.settle_max_angular_speed_rps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError('visual search limits must be finite and positive')
        if not math.isfinite(self.min_yaw_rate_rps) or self.min_yaw_rate_rps < 0.0:
            raise ValueError('visual search minimum yaw rate must be finite and non-negative')
        if self.min_yaw_rate_rps > self.max_yaw_rate_rps:
            raise ValueError('visual search minimum yaw rate exceeds its maximum')
        if (
            not math.isfinite(self.stationary_wait_timeout_s)
            or self.stationary_wait_timeout_s < 0.0
        ):
            raise ValueError(
                'visual search stationary wait timeout must be finite and non-negative',
            )
        if (
            not math.isfinite(self.stationary_quiet_window_s)
            or self.stationary_quiet_window_s <= 0.0
        ):
            raise ValueError(
                'visual search stationary quiet window must be finite and positive',
            )
        if (
            not math.isfinite(self.stationary_max_odom_gap_s)
            or self.stationary_max_odom_gap_s <= 0.0
            or self.stationary_max_odom_gap_s >= self.stationary_quiet_window_s
        ):
            raise ValueError(
                'visual search stationary odometry gap must be finite, positive, '
                'and shorter than its quiet window',
            )
        if (
            not math.isfinite(self.settle_reacquire_budget_s)
            or self.settle_reacquire_budget_s < 0.0
        ):
            raise ValueError(
                'visual search settle reacquire budget must be finite and non-negative',
            )
        if not math.isfinite(self.deadline_grace_s) or self.deadline_grace_s < 0.0:
            raise ValueError(
                'visual search deadline grace must be finite and non-negative',
            )
        if (
            not math.isfinite(self.position_hold_deadband_m)
            or self.position_hold_deadband_m < 0.0
        ):
            raise ValueError('visual search position deadband must be finite and non-negative')
        if self.position_hold_deadband_m >= self.max_planar_drift_m:
            raise ValueError('visual search position deadband must be below its drift limit')
        if self.position_completion_tolerance_m < self.position_hold_deadband_m:
            raise ValueError(
                'visual search position completion tolerance must be at least its deadband',
            )
        if self.position_completion_tolerance_m >= self.max_planar_drift_m:
            raise ValueError(
                'visual search position completion tolerance must be below its drift limit',
            )
        if not (
            self.position_completion_tolerance_m
            < self.moving_rebound_reacquire_m
            < self.max_planar_drift_m
        ):
            raise ValueError(
                'visual search moving rebound trigger must be above completion '
                'tolerance and below the drift limit',
            )
        slowdown = self.position_hold_slowdown_radius_m
        minimum_speed = self.min_position_hold_speed_mps
        if (
            not math.isfinite(slowdown)
            or slowdown < 0.0
            or not math.isfinite(minimum_speed)
            or minimum_speed < 0.0
        ):
            raise ValueError(
                'visual search position slowdown limits must be finite and '
                'non-negative',
            )
        if (slowdown == 0.0) != (minimum_speed == 0.0):
            raise ValueError(
                'visual search position slowdown radius and minimum speed '
                'must be enabled together',
            )
        if slowdown > 0.0 and not (
            self.position_completion_tolerance_m
            < slowdown
            < self.max_planar_drift_m
        ):
            raise ValueError(
                'visual search position slowdown radius must be above the '
                'completion tolerance and below the drift limit',
            )
        if minimum_speed > self.max_position_hold_speed_mps:
            raise ValueError(
                'visual search minimum position speed exceeds its maximum',
            )
        if self.yaw_step_rad > self.max_yaw_offset_rad:
            raise ValueError('visual search yaw step exceeds maximum coverage')
        if self.max_yaw_offset_rad > math.pi:
            raise ValueError('visual search coverage cannot exceed pi radians')
        if self.yaw_tolerance_rad >= self.yaw_step_rad:
            raise ValueError('visual search tolerance must be smaller than its step')
        if self.settle_yaw_tolerance_rad < self.yaw_tolerance_rad:
            raise ValueError(
                'visual search settle yaw tolerance must be at least its control tolerance',
            )
        if self.settle_yaw_tolerance_rad >= self.yaw_step_rad:
            raise ValueError(
                'visual search settle yaw tolerance must be smaller than its step',
            )
        if not (
            self.settle_yaw_tolerance_rad
            < self.position_heading_reacquire_tolerance_rad
            < self.yaw_step_rad
        ):
            raise ValueError(
                'visual search position heading reacquire tolerance must be '
                'above its settle tolerance and below its step',
            )
        if self.max_turn_timeout_s < self.turn_timeout_s:
            raise ValueError('visual search maximum timeout is below one-step timeout')


class ContinuousMotionQuietWindow:
    """Require consecutive, fresh motion samples below fixed speed limits."""

    def __init__(
        self,
        *,
        quiet_window_s: float,
        max_odom_gap_s: float,
        max_linear_speed_mps: float,
        max_angular_speed_rps: float,
    ) -> None:
        values = (
            float(quiet_window_s),
            float(max_odom_gap_s),
            float(max_linear_speed_mps),
            float(max_angular_speed_rps),
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError('motion quiet-window limits must be finite and positive')
        if values[1] >= values[0]:
            raise ValueError('motion quiet-window odometry gap must be shorter than its window')
        (
            self.quiet_window_s,
            self.max_odom_gap_s,
            self.max_linear_speed_mps,
            self.max_angular_speed_rps,
        ) = values
        self.clear('not_started')

    def clear(self, reason: str = '') -> None:
        """Disarm the window and discard every sample from its previous stop."""
        self.stop_received_at_s: float | None = None
        self.minimum_odom_sequence: int | None = None
        self.minimum_odom_stamp_ns: int | None = None
        self.stable_since_received_at_s: float | None = None
        self.stable_start_odom_stamp_ns: int | None = None
        self.last_received_at_s: float | None = None
        self.last_odom_sequence: int | None = None
        self.last_odom_stamp_ns: int | None = None
        self.last_reset_reason = str(reason)

    def reset(
        self,
        *,
        stop_received_at_s: float,
        minimum_odom_sequence: int,
        minimum_odom_stamp_ns: int,
    ) -> None:
        """Arm a new zero-command epoch without reusing pre-stop odometry."""
        stop = float(stop_received_at_s)
        if not math.isfinite(stop):
            raise ValueError('motion quiet-window stop time must be finite')
        if (
            isinstance(minimum_odom_sequence, bool)
            or int(minimum_odom_sequence) < 0
            or isinstance(minimum_odom_stamp_ns, bool)
            or int(minimum_odom_stamp_ns) <= 0
        ):
            raise ValueError('motion quiet-window odometry boundary is invalid')
        self.clear('post_stop_reset')
        self.stop_received_at_s = stop
        self.minimum_odom_sequence = int(minimum_odom_sequence)
        self.minimum_odom_stamp_ns = int(minimum_odom_stamp_ns)

    def observe(
        self,
        *,
        received_at_s: float,
        odom_sequence: int,
        odom_stamp_ns: int,
        linear_speed_mps: float,
        angular_speed_rps: float,
    ) -> None:
        """Accumulate one post-stop sample, resetting on motion or discontinuity."""
        if (
            self.stop_received_at_s is None
            or self.minimum_odom_sequence is None
            or self.minimum_odom_stamp_ns is None
        ):
            return
        received = float(received_at_s)
        linear = float(linear_speed_mps)
        angular = float(angular_speed_rps)
        if (
            not math.isfinite(received)
            or not math.isfinite(linear)
            or not math.isfinite(angular)
            or linear < 0.0
            or angular < 0.0
            or isinstance(odom_sequence, bool)
            or int(odom_sequence) <= 0
            or isinstance(odom_stamp_ns, bool)
            or int(odom_stamp_ns) <= 0
        ):
            raise ValueError('motion quiet-window sample is invalid')
        sequence = int(odom_sequence)
        stamp_ns = int(odom_stamp_ns)
        if (
            received <= self.stop_received_at_s
            or sequence <= self.minimum_odom_sequence
            or stamp_ns <= self.minimum_odom_stamp_ns
        ):
            return
        previous_received = self.last_received_at_s
        previous_sequence = self.last_odom_sequence
        previous_stamp_ns = self.last_odom_stamp_ns
        if previous_received is not None and received < previous_received:
            raise ValueError('motion quiet-window receipt time moved backwards')
        discontinuous = bool(
            previous_sequence is not None
            and (
                sequence != previous_sequence + 1
                or previous_received is None
                or received - previous_received > self.max_odom_gap_s
                or previous_stamp_ns is None
                or stamp_ns <= previous_stamp_ns
                or (stamp_ns - previous_stamp_ns) * 1e-9 > self.max_odom_gap_s
            )
        )
        self.last_received_at_s = received
        self.last_odom_sequence = sequence
        self.last_odom_stamp_ns = stamp_ns
        stationary = bool(
            linear <= self.max_linear_speed_mps
            and angular <= self.max_angular_speed_rps
        )
        if discontinuous or not stationary:
            self.stable_since_received_at_s = None
            self.stable_start_odom_stamp_ns = None
            self.last_reset_reason = (
                'odom_discontinuity' if discontinuous else 'motion_limit_exceeded'
            )
        if not stationary:
            return
        if self.stable_since_received_at_s is None:
            self.stable_since_received_at_s = received
            self.stable_start_odom_stamp_ns = stamp_ns
            if self.last_reset_reason == 'post_stop_reset':
                self.last_reset_reason = ''

    def ready(
        self,
        *,
        odom_sequence: int,
        odom_stamp_ns: int,
        odom_seen_at_s: float,
    ) -> bool:
        """Return true only for a continuous window ending at the current sample."""
        if (
            self.stable_since_received_at_s is None
            or self.stable_start_odom_stamp_ns is None
            or self.last_received_at_s is None
            or self.last_odom_sequence is None
            or self.last_odom_stamp_ns is None
        ):
            return False
        seen = float(odom_seen_at_s)
        if not math.isfinite(seen) or seen != self.last_received_at_s:
            return False
        if (
            isinstance(odom_sequence, bool)
            or isinstance(odom_stamp_ns, bool)
            or int(odom_sequence) != self.last_odom_sequence
            or int(odom_stamp_ns) != self.last_odom_stamp_ns
        ):
            return False
        receipt_span = self.last_received_at_s - self.stable_since_received_at_s
        source_span = (
            self.last_odom_stamp_ns - self.stable_start_odom_stamp_ns
        ) * 1e-9
        epsilon = 1e-9
        return bool(
            receipt_span + epsilon >= self.quiet_window_s
            and source_span + epsilon >= self.quiet_window_s
        )

    @property
    def stable_duration_s(self) -> float:
        """Report measured receipt-clock progress without aging cached samples."""
        if (
            self.stable_since_received_at_s is None
            or self.last_received_at_s is None
        ):
            return 0.0
        return max(0.0, self.last_received_at_s - self.stable_since_received_at_s)


@dataclass(frozen=True)
class VisualSearchUpdate:
    """One fail-closed visual-search control decision."""

    angular_z: float
    error_rad: float
    complete: bool = False
    timed_out: bool = False
    drift_exceeded: bool = False
    planar_drift_m: float = 0.0
    linear_x: float = 0.0
    linear_y: float = 0.0
    timeout_phase: str = ''


class BoundedYawSearch:
    """
    Generate finite scan viewpoints and close each turn using measured yaw.

    The default sequence alternates around the initial heading.  A horizontal
    image-edge observation can instead request a directional step; every such
    step is still clamped to the same origin-relative coverage envelope. All
    viewpoints in one sequence share its initial map-frame XY anchor.
    """

    def __init__(self, config: VisualSearchConfig | None = None) -> None:
        self.config = config or VisualSearchConfig()
        self._offsets = self._make_offsets()
        self.reset()

    def _make_offsets(self) -> tuple[float, ...]:
        offsets: list[float] = []
        radius = self.config.yaw_step_rad
        epsilon = 1e-9
        while radius <= self.config.max_yaw_offset_rad + epsilon:
            bounded = min(radius, self.config.max_yaw_offset_rad)
            offsets.extend((bounded, -bounded))
            radius += self.config.yaw_step_rad
        if not offsets or abs(offsets[-2]) < self.config.max_yaw_offset_rad - epsilon:
            maximum = self.config.max_yaw_offset_rad
            offsets.extend((maximum, -maximum))
        return tuple(offsets)

    @property
    def position_anchor_xy(self) -> tuple[float, float] | None:
        """Return the fixed odometry-parent-frame anchor for this sequence."""
        return self.start_position_xy

    def reset(self) -> None:
        self.origin_yaw_rad: float | None = None
        self.target_yaw_rad: float | None = None
        self.target_offset_rad: float | None = None
        self.started_at_s: float | None = None
        self.start_position_xy: tuple[float, float] | None = None
        self.allocated_timeout_s: float | None = None
        self.position_hold_started_at_s: float | None = None
        self.position_heading_latched = False
        self.last_update_at_s: float | None = None
        self.previous_error_rad: float | None = None
        self.planar_drift_m = 0.0
        self.position_error_base_xy = (0.0, 0.0)
        self.linear_command_base_xy = (0.0, 0.0)
        self.attempt = 0
        self._scan_index = 0
        self.active = False

    def start(
        self,
        current_yaw_rad: float,
        *,
        now_s: float,
        current_position_xy: tuple[float, float],
        image_edge_direction: int = 0,
    ) -> bool:
        """Start the next view; return false when finite coverage is exhausted."""
        current = wrap_angle(current_yaw_rad)
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError('visual search timestamp must be finite')
        if self.active:
            raise RuntimeError('visual search turn is already active')
        if self.last_update_at_s is not None and now < self.last_update_at_s:
            raise ValueError('visual search time must be finite and monotonic')
        if image_edge_direction not in (-1, 0, 1):
            raise ValueError('image edge direction must be -1, 0, or 1')
        position = tuple(float(value) for value in current_position_xy)
        if len(position) != 2 or not all(math.isfinite(value) for value in position):
            raise ValueError('visual search planar position must contain two finite values')
        if self.origin_yaw_rad is None:
            self.origin_yaw_rad = current
            self.start_position_xy = (position[0], position[1])

        if image_edge_direction:
            current_offset = wrap_angle(current - self.origin_yaw_rad)
            requested = current_offset + image_edge_direction * self.config.yaw_step_rad
            offset = min(max(
                requested,
                -self.config.max_yaw_offset_rad,
            ), self.config.max_yaw_offset_rad)
            if abs(offset - current_offset) <= self.config.yaw_tolerance_rad:
                return False
        else:
            if self._scan_index >= len(self._offsets):
                return False
            offset = self._offsets[self._scan_index]
            self._scan_index += 1

        self.attempt += 1
        self.target_offset_rad = float(offset)
        self.target_yaw_rad = wrap_angle(self.origin_yaw_rad + offset)
        self.previous_error_rad = wrap_angle(self.target_yaw_rad - current)
        self.started_at_s = now
        if self.start_position_xy is None:
            raise RuntimeError('visual search position anchor is unavailable')
        angular_travel = abs(wrap_angle(self.target_yaw_rad - current))
        step_equivalents = max(1.0, angular_travel / self.config.yaw_step_rad)
        self.allocated_timeout_s = min(
            self.config.max_turn_timeout_s,
            self.config.turn_timeout_s * step_equivalents,
        )
        self.position_hold_started_at_s = None
        self.position_heading_latched = False
        self.last_update_at_s = now
        self.planar_drift_m = 0.0
        self.position_error_base_xy = (0.0, 0.0)
        self.linear_command_base_xy = (0.0, 0.0)
        self.active = True
        return True

    def reacquire_current_target(
        self,
        current_yaw_rad: float,
        *,
        now_s: float,
        current_position_xy: tuple[float, float],
        deadline_s: float,
    ) -> None:
        """Re-arm the measured target without consuming another search view."""
        if self.active:
            raise RuntimeError('visual search turn is already active')
        if (
            self.origin_yaw_rad is None
            or self.target_yaw_rad is None
            or self.target_offset_rad is None
            or self.start_position_xy is None
            or self.last_update_at_s is None
        ):
            raise RuntimeError('visual search has no retained target to reacquire')
        current = float(current_yaw_rad)
        now = float(now_s)
        deadline = float(deadline_s)
        if (
            not math.isfinite(current)
            or not math.isfinite(now)
            or not math.isfinite(deadline)
            or now < self.last_update_at_s
            or deadline <= now
        ):
            raise ValueError(
                'visual search reacquisition timing and yaw must be finite and bounded',
            )
        position = tuple(float(value) for value in current_position_xy)
        if len(position) != 2 or not all(math.isfinite(value) for value in position):
            raise ValueError(
                'visual search planar position must contain two finite values',
            )

        # Preserve origin, target, scan index, and attempt. Only the closed-loop
        # controller state is re-armed inside the caller's existing hard deadline.
        self.previous_error_rad = wrap_angle(self.target_yaw_rad - current)
        self.started_at_s = now
        self.allocated_timeout_s = deadline - now
        self.position_hold_started_at_s = None
        self.position_heading_latched = False
        self.last_update_at_s = now
        self.planar_drift_m = math.hypot(
            position[0] - self.start_position_xy[0],
            position[1] - self.start_position_xy[1],
        )
        self.position_error_base_xy = (0.0, 0.0)
        self.linear_command_base_xy = (0.0, 0.0)
        self.active = True

    def _position_hold_command(
        self,
        *,
        current_yaw_rad: float,
        current_position_xy: tuple[float, float],
    ) -> tuple[float, float]:
        """Return a bounded base-frame command toward the map-frame anchor."""
        if self.start_position_xy is None:
            raise RuntimeError('visual search position anchor is unavailable')
        error_map_x = self.start_position_xy[0] - current_position_xy[0]
        error_map_y = self.start_position_xy[1] - current_position_xy[1]
        yaw = wrap_angle(current_yaw_rad)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        error_base_x = cosine * error_map_x + sine * error_map_y
        error_base_y = -sine * error_map_x + cosine * error_map_y
        self.position_error_base_xy = (float(error_base_x), float(error_base_y))

        distance = math.hypot(error_base_x, error_base_y)
        deadband = self.config.position_hold_deadband_m
        if distance <= deadband:
            return (0.0, 0.0)
        speed = min(
            self.config.position_hold_gain_s_inv * distance,
            self.config.max_position_hold_speed_mps,
        )
        slowdown = self.config.position_hold_slowdown_radius_m
        if slowdown > 0.0 and distance < slowdown:
            completion = self.config.position_completion_tolerance_m
            fraction = min(max(
                (distance - completion) / (slowdown - completion),
                0.0,
            ), 1.0)
            slowdown_cap = (
                self.config.min_position_hold_speed_mps
                + fraction
                * (
                    self.config.max_position_hold_speed_mps
                    - self.config.min_position_hold_speed_mps
                )
            )
            speed = min(speed, slowdown_cap)
        scale = speed / distance
        return (float(error_base_x * scale), float(error_base_y * scale))

    def update(
        self,
        current_yaw_rad: float,
        *,
        now_s: float,
        current_position_xy: tuple[float, float],
        measured_angular_speed_rps: float | None = None,
    ) -> VisualSearchUpdate:
        if (
            not self.active
            or self.target_yaw_rad is None
            or self.started_at_s is None
            or self.start_position_xy is None
            or self.allocated_timeout_s is None
        ):
            raise RuntimeError('visual search has no active turn')
        now = float(now_s)
        if (
            not math.isfinite(now)
            or now < self.started_at_s
            or self.last_update_at_s is None
            or now < self.last_update_at_s
        ):
            raise ValueError('visual search time must be finite and monotonic')
        position = tuple(float(value) for value in current_position_xy)
        if len(position) != 2 or not all(math.isfinite(value) for value in position):
            raise ValueError('visual search planar position must contain two finite values')
        measured_angular_speed = (
            None
            if measured_angular_speed_rps is None
            else float(measured_angular_speed_rps)
        )
        if (
            measured_angular_speed is not None
            and (
                not math.isfinite(measured_angular_speed)
                or measured_angular_speed < 0.0
            )
        ):
            raise ValueError(
                'visual search measured angular speed must be finite and non-negative',
            )
        self.last_update_at_s = now
        self.planar_drift_m = math.hypot(
            position[0] - self.start_position_xy[0],
            position[1] - self.start_position_xy[1],
        )
        error = wrap_angle(self.target_yaw_rad - current_yaw_rad)
        previous_error = self.previous_error_rad
        self.previous_error_rad = error
        linear_x, linear_y = self._position_hold_command(
            current_yaw_rad=current_yaw_rad,
            current_position_xy=position,
        )
        self.linear_command_base_xy = (linear_x, linear_y)
        if self.planar_drift_m > self.config.max_planar_drift_m:
            self.active = False
            self.linear_command_base_xy = (0.0, 0.0)
            return VisualSearchUpdate(
                0.0,
                error,
                drift_exceeded=True,
                planar_drift_m=self.planar_drift_m,
            )
        # A command floor can skip the inner gate between odometry samples.
        # Accept that crossing only while it remains inside the settle envelope.
        crossed_target_within_settle_gate = (
            previous_error is not None
            and previous_error * error < 0.0
            and abs(error) <= self.config.settle_yaw_tolerance_rad
        )
        yaw_reached = (
            abs(error) <= self.config.yaw_tolerance_rad
            or crossed_target_within_settle_gate
        )
        # A learned whole-body controller can settle just outside the tighter
        # control gate. Once measured motion is already slow inside the final
        # viewpoint envelope, hand ownership to the independent XY phase. The
        # handoff is sticky while yaw stays inside that envelope; a later
        # excursion still reacquires heading with translation held at zero.
        settled_outer_handoff = (
            measured_angular_speed is not None
            and measured_angular_speed <= self.config.settle_max_angular_speed_rps
            and abs(error) <= self.config.settle_yaw_tolerance_rad
        )
        if (
            self.position_heading_latched
            and abs(error)
            > self.config.position_heading_reacquire_tolerance_rad
        ):
            self.position_heading_latched = False
        position_reached = (
            self.planar_drift_m <= self.config.position_completion_tolerance_m
        )
        heading_accepted = (
            yaw_reached
            or settled_outer_handoff
            or (
                self.position_heading_latched
                and not position_reached
                and abs(error)
                <= self.config.position_heading_reacquire_tolerance_rad
            )
        )
        if (
            position_reached
            and abs(error) <= self.config.settle_yaw_tolerance_rad
        ):
            # Do not excite the whole-body controller by withdrawing a coupled
            # XY correction at the same instant as its final yaw command.
            linear_x = 0.0
            linear_y = 0.0
            self.linear_command_base_xy = (0.0, 0.0)
        if (
            self.position_hold_started_at_s is None
            and now - self.started_at_s
            > self.allocated_timeout_s + self.config.deadline_grace_s
        ):
            self.active = False
            self.linear_command_base_xy = (0.0, 0.0)
            return VisualSearchUpdate(
                0.0,
                error,
                timed_out=True,
                planar_drift_m=self.planar_drift_m,
                timeout_phase='yaw_turn',
            )
        if (
            self.position_hold_started_at_s is not None
            and now - self.position_hold_started_at_s
            > self.config.position_hold_timeout_s + self.config.deadline_grace_s
        ):
            self.active = False
            self.linear_command_base_xy = (0.0, 0.0)
            return VisualSearchUpdate(
                0.0,
                error,
                timed_out=True,
                planar_drift_m=self.planar_drift_m,
                timeout_phase='position_hold',
            )
        if heading_accepted and position_reached:
            self.active = False
            self.linear_command_base_xy = (0.0, 0.0)
            return VisualSearchUpdate(
                0.0,
                error,
                complete=True,
                planar_drift_m=self.planar_drift_m,
            )
        if heading_accepted:
            if self.position_hold_started_at_s is None:
                self.position_hold_started_at_s = now
            self.position_heading_latched = True
            # Once heading is accepted, remove yaw/translation coupling and use
            # the independent bounded position budget to close the XY anchor.
            return VisualSearchUpdate(
                0.0,
                error,
                linear_x=linear_x,
                linear_y=linear_y,
                planar_drift_m=self.planar_drift_m,
            )
        if self.position_hold_started_at_s is not None:
            # Translation disturbed heading after position recovery began.
            # Reacquire yaw without adding another coupled planar command.
            linear_x = 0.0
            linear_y = 0.0
            self.linear_command_base_xy = (0.0, 0.0)
        rate = min(max(
            self.config.yaw_gain * error,
            -self.config.max_yaw_rate_rps,
        ), self.config.max_yaw_rate_rps)
        if abs(rate) < self.config.min_yaw_rate_rps:
            rate = math.copysign(self.config.min_yaw_rate_rps, error)
        return VisualSearchUpdate(
            float(rate),
            error,
            linear_x=linear_x,
            linear_y=linear_y,
            planar_drift_m=self.planar_drift_m,
        )


class RuntimeSafetyCore:
    """Enforce re-observation, execution ordering, and fail-closed behavior."""

    _ACTIVE_PHASES = frozenset(RuntimePhase) - {
        RuntimePhase.IDLE, RuntimePhase.PICK_COMPLETE, RuntimePhase.COMPLETE,
        RuntimePhase.CANCELED, RuntimePhase.FAILED,
    }

    def __init__(self) -> None:
        """Initialize an idle task with no accepted observation or plan."""
        self.phase = RuntimePhase.IDLE
        self.instruction = ''
        self.failure_reason = ''
        self.prospective_serial: int | None = None
        self.required_replan_serial: int | None = None
        self.planned_serial: int | None = None
        self.execution_segment = ''
        self.execution_seen_active = False
        self.highest_command_id = 0
        self.executor_command_highwater: dict[str, int] = {}
        self.execution_command_highwater_snapshot: dict[str, int] = {}
        self.expected_command_id: int | None = None
        self.expected_executor_epoch: str = ''
        self.execution_publish_executor_epoch: str = ''
        self.execution_publish_token: str = ''
        self.minimum_command_id = 1
        self.minimum_trajectory_received_at: float | None = None
        self.expected_trajectory_received_at: float | None = None
        self.place_contract_id = ''
        self.place_executor_epoch = ''
        self.place_request_command_id: int | None = None
        self.place_request_received_at: float | None = None
        self.place_request_gripper_command_id: int | None = None
        self.place_request_gripper_received_at: float | None = None
        self.place_approach_command_id: int | None = None
        self.place_approach_received_at: float | None = None
        self.place_approach_gripper_command_id: int | None = None
        self.place_approach_gripper_received_at: float | None = None
        self.place_release_gripper_command_id: int | None = None
        self.place_release_gripper_received_at: float | None = None
        self._place_highwater_trajectory = ''
        self._place_highwater_owner = ''
        self._place_highwater_segment = ''
        self._place_highwater_contract_id = ''
        self._carry_only = True

    @property
    def active(self) -> bool:
        """Return whether this task still owns or may issue motion."""
        return self.phase in self._ACTIVE_PHASES

    def _clear_execution_binding(self, *, clear_segment: bool = True) -> None:
        """Discard the exact status identity frozen for one trajectory."""
        if clear_segment:
            self.execution_segment = ''
        self.execution_seen_active = False
        self.expected_command_id = None
        self.expected_executor_epoch = ''
        self.execution_publish_executor_epoch = ''
        self.execution_publish_token = ''
        self.expected_trajectory_received_at = None
        self.execution_command_highwater_snapshot = {}

    def _record_executor_command_highwater(
        self,
        status: ExecutionState,
    ) -> None:
        """Retain valid trajectory command high-water independently per epoch."""
        command_id = status.command_id
        received_at = status.trajectory_received_at
        if (
            status.owner != 'trajectory'
            or isinstance(command_id, bool)
            or not isinstance(command_id, int)
            or command_id <= 0
            or received_at is None
            or not math.isfinite(received_at)
            or received_at < 0.0
        ):
            return
        try:
            epoch = self._place_identity_token(
                status.executor_epoch,
                'executor epoch',
            )
        except ValueError:
            return
        self.highest_command_id = max(self.highest_command_id, command_id)
        self.executor_command_highwater[epoch] = max(
            self.executor_command_highwater.get(epoch, 0),
            command_id,
        )

    def begin(self, instruction: str) -> None:
        """Start a new text task and reset all execution state."""
        query = instruction.strip()
        if not query:
            raise ValueError('task instruction must not be empty')
        highest_command_id = self.highest_command_id
        executor_command_highwater = dict(self.executor_command_highwater)
        self.__init__()
        self.highest_command_id = highest_command_id
        self.executor_command_highwater = executor_command_highwater
        self.instruction = query
        self.phase = RuntimePhase.POSE_SETTLE

    def mark_pose_settled(self) -> None:
        """Begin VLM grounding only after the observation pose is stationary."""
        if self.phase is not RuntimePhase.POSE_SETTLE:
            raise RuntimeError(f'cannot settle pose while {self.phase.value}')
        self.phase = RuntimePhase.GROUNDING

    def begin_visual_search(self) -> None:
        """Enter a base-only search turn after the observation arm pose settles."""
        if self.phase is not RuntimePhase.POSE_SETTLE:
            raise RuntimeError(f'cannot search visually while {self.phase.value}')
        self.phase = RuntimePhase.VISUAL_SEARCH

    def mark_visual_search_complete(self) -> None:
        """Require a stationary camera interval before grounding a new view."""
        if self.phase is not RuntimePhase.VISUAL_SEARCH:
            raise RuntimeError(f'cannot finish visual search while {self.phase.value}')
        self.phase = RuntimePhase.POSE_SETTLE

    def restart_grounding(self) -> None:
        """Reset execution state while preserving the active text instruction."""
        if not self.instruction:
            raise RuntimeError('cannot recover a task without an instruction')
        instruction = self.instruction
        self.begin(instruction)

    def request_reobservation(self, after_serial: int) -> None:
        """Retry pre-contact planning only after a strictly newer observation."""
        if self.phase not in (
            RuntimePhase.PLANNING,
            RuntimePhase.PREGRASP_REOBSERVE,
            RuntimePhase.APPROACH_PLANNING,
            RuntimePhase.WAIT_FRESH_OBSERVATION,
            RuntimePhase.FINAL_GROUNDING,
        ):
            raise RuntimeError(f'cannot request reobservation while {self.phase.value}')
        if after_serial < 0:
            raise ValueError('observation serial cannot be negative')
        self.required_replan_serial = int(after_serial) + 1
        self.planned_serial = None
        self._clear_execution_binding()
        self.minimum_trajectory_received_at = None
        self.phase = RuntimePhase.WAIT_FRESH_OBSERVATION

    def mark_standoff(self, observation_serial: int) -> None:
        """Record the prospective-plan frame and enter visual servo."""
        if self.phase not in (RuntimePhase.GROUNDING, RuntimePhase.STANDOFF):
            raise RuntimeError(f'cannot select standoff while {self.phase.value}')
        if observation_serial < 1:
            raise ValueError('observation serial must be positive')
        self.prospective_serial = int(observation_serial)
        self.phase = RuntimePhase.COARSE_NAV

    def mark_coarse_ready(self) -> None:
        """Require near-field semantic re-grounding after coarse navigation."""
        if self.phase is not RuntimePhase.COARSE_NAV:
            raise RuntimeError(f'cannot finish coarse navigation while {self.phase.value}')
        self.phase = RuntimePhase.NEAR_GROUNDING

    def mark_near_grounded(self, observation_serial: int) -> None:
        """Enter visual servo only with a newly VLM-anchored target session."""
        if self.phase is not RuntimePhase.NEAR_GROUNDING:
            raise RuntimeError(f'cannot accept near grounding while {self.phase.value}')
        if observation_serial < 1:
            raise ValueError('near grounding observation serial must be positive')
        self.prospective_serial = int(observation_serial)
        self.phase = RuntimePhase.VISUAL_SERVO

    def mark_servo_complete_for_reground(self) -> None:
        """Require the first exact bundle from a new semantic grounding session."""
        if self.phase is not RuntimePhase.VISUAL_SERVO:
            raise RuntimeError(f'cannot complete servo while {self.phase.value}')
        assert self.prospective_serial is not None
        # The runtime now creates a new request identity and serial gate. Serial
        # values from the pre-servo gate are not comparable across that reset.
        self.required_replan_serial = 1
        self.phase = RuntimePhase.FINAL_GROUNDING

    def begin_replan(self, observation_serial: int) -> None:
        """Accept a post-servo frame only when its serial is new enough."""
        if self.phase not in (
            RuntimePhase.WAIT_FRESH_OBSERVATION,
            RuntimePhase.FINAL_GROUNDING,
        ):
            raise RuntimeError(f'cannot replan while {self.phase.value}')
        assert self.required_replan_serial is not None
        if observation_serial < self.required_replan_serial:
            raise RuntimeError(
                'post-servo plan requires a newly synchronized perception frame',
            )
        self.planned_serial = int(observation_serial)
        self.phase = RuntimePhase.PLANNING

    def plan_ready(self) -> None:
        """Accept a stage-one pregrasp plan and prepare transit execution."""
        if self.phase is not RuntimePhase.PLANNING:
            raise RuntimeError(f'cannot accept a plan while {self.phase.value}')
        self.phase = RuntimePhase.TRANSIT

    def begin_approach_replan(self, observation_serial: int) -> None:
        """Freeze a post-pregrasp observation for the second planning stage."""
        if self.phase is not RuntimePhase.PREGRASP_REOBSERVE:
            raise RuntimeError(
                f'cannot replan an approach while {self.phase.value}',
            )
        if self.planned_serial is None or observation_serial <= self.planned_serial:
            raise RuntimeError(
                'approach planning requires a newer post-pregrasp observation',
            )
        self.planned_serial = int(observation_serial)
        self.phase = RuntimePhase.APPROACH_PLANNING

    def approach_plan_ready(self) -> None:
        """Accept a fresh second-stage plan and permit approach execution."""
        if self.phase is not RuntimePhase.APPROACH_PLANNING:
            raise RuntimeError(
                f'cannot accept an approach plan while {self.phase.value}',
            )
        self.phase = RuntimePhase.APPROACH

    @staticmethod
    def _place_identity_token(value: object, label: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f'{label} must be a string')
        identity = value.strip()
        if (
            not identity
            or identity != value
            or len(identity) > 128
            or any(
                ord(character) < 33
                or ord(character) > 126
                or character in ';|='
                for character in identity
            )
        ):
            raise ValueError(f'{label} is invalid')
        return identity

    def _prepare_place_trajectory(
        self,
        segment: str,
        contract_id: object,
        executor_state: object,
    ) -> None:
        """Freeze the exact executor high-water before one place segment."""
        contract = self._place_identity_token(contract_id, 'place contract ID')
        if not isinstance(executor_state, ExecutionState):
            raise ValueError('place execution high-water is unavailable')
        state = executor_state
        epoch = self._place_identity_token(
            state.executor_epoch,
            'executor epoch',
        )
        if (
            contract != self.place_contract_id
            or epoch != self.place_executor_epoch
        ):
            raise ValueError(
                'place execution contract or executor epoch changed',
            )
        command_id = state.command_id
        received_at = state.trajectory_received_at
        gripper_command_id = state.gripper_command_id
        gripper_received_at = state.gripper_received_at
        if (
            command_id is None
            or command_id <= 0
            or received_at is None
            or not math.isfinite(received_at)
            or received_at < 0.0
            or gripper_command_id is None
            or gripper_command_id <= 0
            or gripper_received_at is None
            or not math.isfinite(gripper_received_at)
            or gripper_received_at < 0.0
        ):
            raise ValueError(
                'place execution command/source high-water is invalid',
            )
        if command_id < self.executor_command_highwater.get(epoch, 0):
            raise ValueError(
                'place execution command snapshot is below the observed high-water',
            )
        previous_segments = {
            'place_transit': 'carry',
            'place_approach': 'place_transit',
            'place_retreat': 'place_approach',
        }
        previous_contract = (
            contract if segment == 'place_retreat' else 'none'
        )
        if (
            state.trajectory != 'succeeded'
            or state.owner != 'trajectory'
            or state.segment != previous_segments[segment]
            or state.trajectory_contract_id != previous_contract
        ):
            raise ValueError(
                f'{segment} high-water does not identify the completed '
                f'{previous_segments[segment]} segment',
            )

        if segment == 'place_retreat' and (
            self.place_approach_command_id is None
            or self.place_approach_received_at is None
            or command_id != self.place_approach_command_id
            or received_at != self.place_approach_received_at
        ):
            raise ValueError(
                'place retreat high-water is not the completed approach',
            )
        if segment == 'place_transit' and (
            self.place_request_command_id is None
            or self.place_request_received_at is None
            or command_id != self.place_request_command_id
            or received_at != self.place_request_received_at
        ):
            raise ValueError(
                'place transit high-water changed during planning',
            )
        if segment in {'place_transit', 'place_approach'} and (
            self.place_request_gripper_command_id is None
            or self.place_request_gripper_received_at is None
            or gripper_command_id != self.place_request_gripper_command_id
            or gripper_received_at != self.place_request_gripper_received_at
        ):
            raise ValueError(
                'place pre-release gripper high-water changed',
            )
        if segment == 'place_retreat' and (
            self.place_release_gripper_command_id is None
            or self.place_release_gripper_received_at is None
            or gripper_command_id != self.place_release_gripper_command_id
            or gripper_received_at != self.place_release_gripper_received_at
        ):
            raise ValueError(
                'place retreat gripper high-water is not the measured release',
            )

        self._record_executor_command_highwater(state)
        self.minimum_command_id = command_id + 1
        self.minimum_trajectory_received_at = received_at
        self.expected_trajectory_received_at = None
        self._place_highwater_trajectory = state.trajectory
        self._place_highwater_owner = state.owner
        self._place_highwater_segment = state.segment
        self._place_highwater_contract_id = state.trajectory_contract_id

    def trajectory_sent(
        self,
        segment: str,
        *,
        place_contract_id: str | None = None,
        executor_state: ExecutionState | None = None,
        executor_epoch: str | None = None,
        published_at_s: float | None = None,
        trajectory_token: str | None = None,
    ) -> None:
        """Record publication of the expected trajectory segment."""
        allowed = {
            'transit': RuntimePhase.TRANSIT,
            'approach': RuntimePhase.APPROACH,
            'lift': RuntimePhase.LIFT,
            'carry': RuntimePhase.CARRY,
            'place_transit': RuntimePhase.PLACE_TRANSIT,
            'place_approach': RuntimePhase.PLACE_APPROACH,
            'place_retreat': RuntimePhase.PLACE_RETREAT,
        }
        if segment not in allowed or self.phase is not allowed[segment]:
            raise RuntimeError(f'cannot send {segment!r} while {self.phase.value}')
        token = self._place_identity_token(
            trajectory_token,
            'trajectory token',
        )
        if segment.startswith('place_'):
            if executor_epoch is not None or published_at_s is not None:
                raise ValueError(
                    'place trajectory cannot carry a non-place publish fence',
                )
            self._prepare_place_trajectory(
                segment,
                place_contract_id,
                executor_state,
            )
        elif place_contract_id is not None or executor_state is not None:
            raise ValueError(
                'non-place trajectory cannot carry a place execution snapshot',
            )
        else:
            epoch = self._place_identity_token(
                executor_epoch,
                'executor epoch',
            )
            if (
                published_at_s is None
                or not math.isfinite(float(published_at_s))
                or float(published_at_s) < 0.0
            ):
                raise ValueError(
                    'non-place trajectory requires a finite publish source fence',
                )
        self.execution_segment = segment
        self._clear_execution_binding(clear_segment=False)
        self.execution_publish_token = token
        if not segment.startswith('place_'):
            self.execution_command_highwater_snapshot = dict(
                self.executor_command_highwater,
            )
            self.execution_publish_executor_epoch = epoch
            self.minimum_command_id = self.highest_command_id + 1
            self.minimum_trajectory_received_at = float(published_at_s)
            self.expected_trajectory_received_at = None

    def _advance_execution_phase(self) -> None:
        if self.phase is RuntimePhase.TRANSIT:
            self.phase = RuntimePhase.PREGRASP_REOBSERVE
        elif self.phase is RuntimePhase.APPROACH:
            self.phase = RuntimePhase.CLOSING
        elif self.phase is RuntimePhase.LIFT:
            self.phase = RuntimePhase.VERIFY
        elif self.phase is RuntimePhase.PLACE_TRANSIT:
            self.phase = RuntimePhase.PLACE_APPROACH
        elif self.phase is RuntimePhase.PLACE_APPROACH:
            assert self.expected_command_id is not None
            assert self.expected_trajectory_received_at is not None
            self.place_approach_command_id = self.expected_command_id
            self.place_approach_received_at = (
                self.expected_trajectory_received_at
            )
            self.phase = RuntimePhase.RELEASING
        elif self.phase is RuntimePhase.PLACE_RETREAT:
            self.phase = RuntimePhase.POST_RELEASE_VERIFICATION
        else:
            self.phase = (
                RuntimePhase.PICK_COMPLETE
                if self._carry_only
                else RuntimePhase.PLACE_GROUNDING
            )
        self._clear_execution_binding()
        self.minimum_trajectory_received_at = None

    def _place_execution_update(self, status: ExecutionState) -> SafetyAction:
        if (
            not self.place_contract_id
            or not self.place_executor_epoch
            or not self.execution_publish_token
            or self.minimum_trajectory_received_at is None
        ):
            return self.fail('place execution transaction is not frozen')
        if status.rejected or status.canceled:
            if (
                status.trajectory_event_token == self.execution_publish_token
                or status.trajectory_token == self.execution_publish_token
            ):
                return self.fail(f'arm execution failed: {status.trajectory}')
            return SafetyAction()
        if status.trajectory_token != self.execution_publish_token:
            if (
                status.trajectory == 'active'
                and status.owner == 'trajectory'
                and status.executor_epoch == self.place_executor_epoch
                and isinstance(status.command_id, int)
                and not isinstance(status.command_id, bool)
                and status.command_id >= self.minimum_command_id
                and status.trajectory_received_at is not None
                and status.trajectory_received_at
                > self.minimum_trajectory_received_at
            ):
                return self.fail(
                    'executor accepted an unexpected place trajectory token',
                )
            return SafetyAction()
        if (
            status.command_id is None
            or status.trajectory_received_at is None
            or not status.executor_epoch
            or not status.trajectory_contract_id
        ):
            return self.fail('place execution status omitted transaction identity')
        if status.executor_epoch != self.place_executor_epoch:
            return self.fail('place execution executor epoch changed')
        highwater_command = self.minimum_command_id - 1
        highwater_source = self.minimum_trajectory_received_at
        if (
            status.command_id == highwater_command
            and status.trajectory_received_at == highwater_source
        ):
            if not (
                status.trajectory == self._place_highwater_trajectory
                and status.owner == self._place_highwater_owner
                and status.segment == self._place_highwater_segment
                and status.trajectory_contract_id
                == self._place_highwater_contract_id
            ):
                return self.fail(
                    'place execution high-water identity changed during handoff',
                )
            if self.expected_command_id is not None or self.execution_seen_active:
                return self.fail(
                    'place execution replayed the frozen high-water after active',
                )
            return SafetyAction()
        if (
            status.command_id <= highwater_command
            or status.trajectory_received_at <= highwater_source
        ):
            return self.fail('place execution status is stale or replayed')

        expected_contract = (
            'none'
            if self.execution_segment == 'place_transit'
            else self.place_contract_id
        )
        if (
            status.owner != 'trajectory'
            or status.segment != self.execution_segment
            or status.trajectory_contract_id != expected_contract
        ):
            return self.fail(
                'place execution identity mismatch: '
                f'owner={status.owner!r} segment={status.segment!r} '
                f'contract={status.trajectory_contract_id!r}',
            )
        if self.expected_command_id is None:
            if status.trajectory != 'active':
                return self.fail(
                    'place execution did not start with active status',
                )
            self.expected_command_id = status.command_id
            self.expected_trajectory_received_at = (
                status.trajectory_received_at
            )
        elif (
            status.command_id != self.expected_command_id
            or status.trajectory_received_at
            != self.expected_trajectory_received_at
        ):
            return self.fail(
                'place trajectory command/source identity changed during execution',
            )

        self._record_executor_command_highwater(status)
        if status.trajectory == 'active':
            self.execution_seen_active = True
        elif status.trajectory == 'succeeded':
            if not self.execution_seen_active:
                return self.fail(
                    'place execution succeeded before active status',
                )
            if self.phase is RuntimePhase.PLACE_APPROACH:
                if (
                    status.gripper_command_id is None
                    or status.gripper_received_at is None
                ):
                    return self.fail(
                        'place approach omitted gripper high-water identity',
                    )
                self.place_approach_gripper_command_id = (
                    status.gripper_command_id
                )
                self.place_approach_gripper_received_at = (
                    status.gripper_received_at
                )
            self._advance_execution_phase()
        else:
            return self.fail(
                f'place execution status is invalid: {status.trajectory}',
            )
        return SafetyAction()

    def _place_release_update(self, status: ExecutionState) -> SafetyAction:
        if (
            self.place_approach_command_id is None
            or self.place_approach_received_at is None
            or self.place_approach_gripper_command_id is None
            or self.place_approach_gripper_received_at is None
        ):
            return self.fail('place release lacks completed approach identity')
        if status.rejected or status.canceled:
            return self.fail(f'arm execution failed: {status.trajectory}')
        if (
            status.trajectory != 'succeeded'
            or status.owner != 'trajectory'
            or status.segment != 'place_approach'
            or status.command_id != self.place_approach_command_id
            or status.executor_epoch != self.place_executor_epoch
            or status.trajectory_contract_id != self.place_contract_id
            or status.trajectory_received_at
            != self.place_approach_received_at
        ):
            return self.fail(
                'place release feedback does not match the completed approach',
            )
        gripper_command_id = status.gripper_command_id
        gripper_received_at = status.gripper_received_at
        if gripper_command_id is None or gripper_received_at is None:
            return self.fail('place release omitted gripper source identity')
        pre_command = self.place_approach_gripper_command_id
        pre_source = self.place_approach_gripper_received_at
        assert pre_command is not None and pre_source is not None
        if (
            gripper_command_id == pre_command
            and gripper_received_at == pre_source
            and self.place_release_gripper_command_id is None
        ):
            return SafetyAction()
        if self.place_release_gripper_command_id is None:
            if (
                gripper_command_id <= pre_command
                or gripper_received_at <= pre_source
                or gripper_received_at <= self.place_approach_received_at
            ):
                return self.fail(
                    'place release gripper command/source did not advance',
                )
            self.place_release_gripper_command_id = gripper_command_id
            self.place_release_gripper_received_at = gripper_received_at
        elif (
            gripper_command_id != self.place_release_gripper_command_id
            or gripper_received_at != self.place_release_gripper_received_at
        ):
            return self.fail(
                'place release gripper command/source identity changed',
            )
        return SafetyAction()

    def _place_planning_execution_update(
        self,
        status: ExecutionState,
    ) -> SafetyAction:
        if (
            self.place_request_command_id is None
            or self.place_request_received_at is None
        ):
            return self.fail('place planning executor snapshot is unavailable')
        if (
            status.trajectory != 'succeeded'
            or status.owner != 'trajectory'
            or status.segment != 'carry'
            or status.command_id != self.place_request_command_id
            or status.executor_epoch != self.place_executor_epoch
            or status.trajectory_contract_id != 'none'
            or status.trajectory_received_at
            != self.place_request_received_at
            or status.gripper_command_id
            != self.place_request_gripper_command_id
            or status.gripper_received_at
            != self.place_request_gripper_received_at
        ):
            return self.fail(
                'place planning executor snapshot changed',
            )
        return SafetyAction()

    def execution_update(self, status: ExecutionState) -> SafetyAction:
        """Advance only after active-to-succeeded or fail on rejection."""
        if self.phase is RuntimePhase.PLACE_PLANNING and self.place_contract_id:
            return self._place_planning_execution_update(status)
        if self.phase is RuntimePhase.RELEASING and self.place_contract_id:
            return self._place_release_update(status)
        if self.phase not in (
            RuntimePhase.TRANSIT,
            RuntimePhase.APPROACH,
            RuntimePhase.LIFT,
            RuntimePhase.CARRY,
            RuntimePhase.PLACE_TRANSIT,
            RuntimePhase.PLACE_APPROACH,
            RuntimePhase.PLACE_RETREAT,
        ):
            return SafetyAction()
        if not self.execution_segment:
            return SafetyAction()
        if self.execution_segment.startswith('place_'):
            return self._place_execution_update(status)
        if status.rejected or status.canceled:
            if (
                status.trajectory_event_token == self.execution_publish_token
                or status.trajectory_token == self.execution_publish_token
            ):
                return self.fail(f'arm execution failed: {status.trajectory}')
            return SafetyAction()
        if status.command_id is None:
            return self.fail('arm execution status omitted command_id')
        received_at = status.trajectory_received_at
        try:
            executor_epoch = self._place_identity_token(
                status.executor_epoch,
                'executor epoch',
            )
        except ValueError:
            executor_epoch = ''
        if (
            not executor_epoch
            or not status.trajectory_token
            or received_at is None
            or not math.isfinite(received_at)
            or received_at < 0.0
        ):
            return self.fail(
                'arm execution status omitted executor epoch/token/source identity',
            )
        if status.trajectory_token != self.execution_publish_token:
            epoch_highwater = self.execution_command_highwater_snapshot.get(
                executor_epoch,
                0,
            )
            if (
                status.trajectory == 'active'
                and status.owner == 'trajectory'
                and executor_epoch == self.execution_publish_executor_epoch
                and self.minimum_trajectory_received_at is not None
                and received_at + 1e-6 >= self.minimum_trajectory_received_at
                and status.command_id > epoch_highwater
            ):
                return self.fail(
                    'executor accepted an unexpected trajectory token',
                )
            return SafetyAction()
        if self.expected_command_id is None:
            if (
                not self.execution_publish_executor_epoch
                or not self.execution_publish_token
                or self.minimum_trajectory_received_at is None
            ):
                return self.fail('arm execution publish fence is unavailable')
            if executor_epoch != self.execution_publish_executor_epoch:
                return self.fail('arm execution executor epoch changed after publish')
            if received_at + 1e-6 < self.minimum_trajectory_received_at:
                return SafetyAction()
            epoch_highwater = self.execution_command_highwater_snapshot.get(
                executor_epoch,
                0,
            )
            if status.command_id <= epoch_highwater:
                return SafetyAction()
        if status.owner != 'trajectory' or status.segment != self.execution_segment:
            return self.fail(
                'arm execution identity mismatch: '
                f'owner={status.owner!r} segment={status.segment!r}',
            )
        if self.expected_command_id is None:
            if status.trajectory != 'active':
                return SafetyAction()
            self.expected_command_id = status.command_id
            self.expected_executor_epoch = executor_epoch
            self.expected_trajectory_received_at = received_at
        elif (
            status.command_id != self.expected_command_id
            or executor_epoch != self.expected_executor_epoch
            or received_at != self.expected_trajectory_received_at
        ):
            return self.fail(
                'arm trajectory executor/source identity changed during execution',
            )
        if status.trajectory == 'active':
            self.execution_seen_active = True
            self._record_executor_command_highwater(status)
        elif status.trajectory == 'succeeded' and self.execution_seen_active:
            self._record_executor_command_highwater(status)
            self._advance_execution_phase()
        return SafetyAction()

    def close_complete(self) -> None:
        """Advance from measured gripper settling to the lift segment."""
        if self.phase is not RuntimePhase.CLOSING:
            raise RuntimeError(f'cannot finish closing while {self.phase.value}')
        self.phase = RuntimePhase.LIFT

    def verification_complete(self, *, carry_only: bool) -> None:
        """Enter planned carry motion before reporting any task result."""
        if self.phase is not RuntimePhase.VERIFY:
            raise RuntimeError(f'cannot finish verification while {self.phase.value}')
        self._carry_only = bool(carry_only)
        self.phase = RuntimePhase.CARRY

    def place_request_sent(
        self,
        *,
        place_contract_id: str,
        executor_state: ExecutionState,
    ) -> None:
        """Wait for an observed placement planner contract."""
        if self.phase is not RuntimePhase.PLACE_GROUNDING:
            raise RuntimeError(f'cannot request placement while {self.phase.value}')
        contract = self._place_identity_token(
            place_contract_id,
            'place contract ID',
        )
        if not isinstance(executor_state, ExecutionState):
            raise ValueError('place request executor high-water is unavailable')
        epoch = self._place_identity_token(
            executor_state.executor_epoch,
            'executor epoch',
        )
        command_id = executor_state.command_id
        received_at = executor_state.trajectory_received_at
        gripper_command_id = executor_state.gripper_command_id
        gripper_received_at = executor_state.gripper_received_at
        if (
            executor_state.trajectory != 'succeeded'
            or executor_state.owner != 'trajectory'
            or executor_state.segment != 'carry'
            or executor_state.trajectory_contract_id != 'none'
            or command_id is None
            or command_id <= 0
            or received_at is None
            or not math.isfinite(received_at)
            or received_at < 0.0
            or gripper_command_id is None
            or gripper_command_id <= 0
            or gripper_received_at is None
            or not math.isfinite(gripper_received_at)
            or gripper_received_at < 0.0
            or command_id < self.executor_command_highwater.get(epoch, 0)
        ):
            raise ValueError(
                'place request lacks the completed carry executor high-water',
            )
        self.place_contract_id = contract
        self.place_executor_epoch = epoch
        self.place_request_command_id = command_id
        self.place_request_received_at = received_at
        self.place_request_gripper_command_id = gripper_command_id
        self.place_request_gripper_received_at = gripper_received_at
        self.place_approach_command_id = None
        self.place_approach_received_at = None
        self.place_approach_gripper_command_id = None
        self.place_approach_gripper_received_at = None
        self.place_release_gripper_command_id = None
        self.place_release_gripper_received_at = None
        self._record_executor_command_highwater(executor_state)
        self.phase = RuntimePhase.PLACE_PLANNING

    @staticmethod
    def _place_source_stamp_ns(value: float | None, label: str) -> int:
        if value is None or not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f'{label} source high-water is invalid')
        stamp_ns = int(round(value * 1e9))
        if not 0 <= stamp_ns < 2**63:
            raise RuntimeError(f'{label} source high-water exceeds supported range')
        return stamp_ns

    def place_plan_ready(self, contract: Mapping[str, object]) -> None:
        """Accept a plan only when it echoes the frozen executor high-water."""
        if self.phase is not RuntimePhase.PLACE_PLANNING:
            raise RuntimeError(f'cannot accept placement plan while {self.phase.value}')
        if (
            not self.place_contract_id
            or not self.place_executor_epoch
            or self.place_request_command_id is None
            or self.place_request_gripper_command_id is None
        ):
            raise RuntimeError('place execution transaction is not frozen')
        if not isinstance(contract, Mapping):
            raise ValueError('placement plan executor snapshot is unavailable')
        expected: dict[str, object] = {
            'goal_id': self.place_contract_id,
            'trajectory_contract_id': self.place_contract_id,
            'executor_epoch': self.place_executor_epoch,
            'trajectory_command_highwater': self.place_request_command_id,
            'trajectory_source_highwater_ns': self._place_source_stamp_ns(
                self.place_request_received_at,
                'trajectory',
            ),
            'gripper_command_highwater': (
                self.place_request_gripper_command_id
            ),
            'gripper_source_highwater_ns': self._place_source_stamp_ns(
                self.place_request_gripper_received_at,
                'gripper',
            ),
        }
        for field, frozen in expected.items():
            observed = contract.get(field)
            if (
                isinstance(frozen, int)
                and (
                    isinstance(observed, bool)
                    or not isinstance(observed, int)
                )
            ) or observed != frozen:
                raise ValueError(
                    f'placement plan executor snapshot mismatches {field}',
                )
        self.phase = RuntimePhase.PLACE_TRANSIT

    def release_complete(self) -> None:
        """Advance to collision-checked retreat after measured gripper opening."""
        if self.phase is not RuntimePhase.RELEASING:
            raise RuntimeError(f'cannot finish release while {self.phase.value}')
        if (
            self.place_approach_command_id is None
            or self.place_approach_received_at is None
            or self.place_release_gripper_command_id is None
            or self.place_release_gripper_received_at is None
        ):
            raise RuntimeError('cannot finish release without completed approach')
        self.phase = RuntimePhase.PLACE_RETREAT

    def post_release_verification_complete(self) -> None:
        """Finish only after correlated observed placement evidence succeeds."""
        if self.phase is not RuntimePhase.POST_RELEASE_VERIFICATION:
            raise RuntimeError(
                'cannot finish post-release verification while '
                f'{self.phase.value}',
            )
        self.phase = RuntimePhase.COMPLETE

    def perception_invalid(self, reason: str) -> SafetyAction:
        """Fail an active task when its perception contract is invalid."""
        if self.phase not in self._ACTIVE_PHASES:
            return SafetyAction()
        return self.fail(reason)

    def posture_invalid(self, reason: str) -> SafetyAction:
        """Fail an active task and cancel every motion owner on posture loss."""
        if not self.active:
            return SafetyAction()
        return self.fail(reason)

    def cancel(self) -> SafetyAction:
        """Enter an idempotent terminal cancellation and stop every actuator owner."""
        self.phase = RuntimePhase.CANCELED
        self.failure_reason = ''
        self.prospective_serial = None
        self.required_replan_serial = None
        self.planned_serial = None
        self._clear_execution_binding()
        self.minimum_trajectory_received_at = None
        self.minimum_command_id = self.highest_command_id + 1
        self._carry_only = True
        return SafetyAction(
            stop_base=True,
            cancel_navigation=True,
            cancel_arm=True,
        )

    def fail(self, reason: str) -> SafetyAction:
        """Enter a terminal failure and request all actuator stops."""
        self.failure_reason = reason.strip() or 'runtime safety failure'
        self.phase = RuntimePhase.FAILED
        self._clear_execution_binding()
        self.minimum_trajectory_received_at = None
        return SafetyAction(True, True, True, self.failure_reason)
