"""ROS-independent health contract for VLM-seeded persistent tracking."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Iterable


class ContractPhase(str, Enum):
    IDLE = 'idle'
    WAITING_FRAME = 'waiting_frame'
    GROUNDING = 'grounding'
    WAITING_TRACKER = 'waiting_tracker'
    TRACKING = 'tracking'
    FAILED = 'failed'


class FailureCode(str, Enum):
    NONE = ''
    CAMERA_FRAME_TIMEOUT = 'camera_frame_timeout'
    GROUNDING_TIMEOUT = 'grounding_timeout'
    GROUNDING_FAILED = 'grounding_failed'
    TRACKER_ACQUISITION_TIMEOUT = 'tracker_acquisition_timeout'
    TRACKER_REPORTED_LOSS = 'tracker_reported_loss'
    EMPTY_DETECTIONS = 'empty_detections'
    TRACK_ID_CHANGED = 'track_id_changed'
    TARGET_MISSING = 'selected_target_missing'
    CLOUD_TOO_SMALL = 'selected_cloud_too_small'
    TRACKER_DATA_STALE = 'tracker_data_stale'
    CLOCK_ROLLBACK = 'clock_rollback'


@dataclass(frozen=True)
class GroundingRequest:
    """One task-owned semantic request carried across the ROS String boundary."""

    request_id: str
    instruction: str
    scope: str


GROUNDING_SCOPES = frozenset({
    'grasp_only',
    'grasp_for_place',
    'place_support',
})


def parse_grounding_request(
    payload: str,
    *,
    legacy_request_id: str | None = None,
) -> GroundingRequest:
    """Parse the versioned request envelope, optionally accepting plain text."""
    if not isinstance(payload, str) or not payload.strip() or len(payload) > 8192:
        raise ValueError('grounding request is not a bounded non-empty string')
    clean = payload.strip()
    if not clean.startswith('{'):
        if legacy_request_id is None:
            raise ValueError('grounding request is not a versioned envelope')
        request_id = str(legacy_request_id).strip()
        instruction = clean
    else:
        try:
            value = json.loads(clean)
        except (TypeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError('grounding request is not valid JSON') from error
        if not isinstance(value, dict):
            raise ValueError('unsupported grounding request schema')
        schema = value.get('schema')
        if schema == 'z_manip.grounding_request.v1':
            if set(value) != {'schema', 'request_id', 'instruction'}:
                raise ValueError('unsupported grounding request schema')
            scope = 'grasp_only'
        elif schema == 'z_manip.grounding_request.v2':
            if set(value) != {'schema', 'request_id', 'instruction', 'scope'}:
                raise ValueError('unsupported grounding request schema')
            scope = value.get('scope')
            if not isinstance(scope, str) or scope not in GROUNDING_SCOPES:
                raise ValueError('grounding request scope is invalid')
        else:
            raise ValueError('unsupported grounding request schema')
        request_id = value.get('request_id')
        instruction = value.get('instruction')
        if not isinstance(request_id, str) or not isinstance(instruction, str):
            raise ValueError('grounding request identity and instruction must be strings')
        request_id = request_id.strip()
        instruction = instruction.strip()
    if (
        not request_id
        or len(request_id) > 128
        or any(ord(character) < 0x21 or ord(character) > 0x7e for character in request_id)
    ):
        raise ValueError('grounding request identity is invalid')
    if not instruction or len(instruction) > 4096:
        raise ValueError('grounding instruction is invalid')
    if not clean.startswith('{'):
        scope = 'grasp_only'
    return GroundingRequest(
        request_id=request_id,
        instruction=instruction,
        scope=scope,
    )


@dataclass(frozen=True)
class PixelBox:
    center_x: float
    center_y: float
    size_x: float
    size_y: float


@dataclass(frozen=True)
class TrackerFailureReport:
    """One versioned adapter failure tied to an exact semantic seed image."""

    seed_id: str
    seed_stamp_ns: int
    reason_code: str
    reason: str
    replay_candidates: int
    replay_selected: int
    replay_span_ns: int
    acquisition_live_updates: int


@dataclass(frozen=True)
class TrackerFrameManifest:
    """Success authority tying one observation frame to an exact seed epoch."""

    seed_id: str
    seed_stamp_ns: int
    adapter_generation: int
    result_stamp_ns: int
    frame_id: str
    session_id: str
    track_id: str


@dataclass(frozen=True)
class ExactObservationBundle:
    """Three controller-facing tracker outputs from one exact source frame."""

    generation: int
    stamp_ns: int
    frame_id: str
    detections: object
    target: object
    cloud: object
    manifest: object


class ExactObservationBundler:
    """Bound and join cross-topic observations without mixing frames or epochs."""

    _KINDS = frozenset({'detections', 'target', 'cloud', 'manifest'})

    def __init__(self, max_pending_bundles: int = 12) -> None:
        if (
            isinstance(max_pending_bundles, bool)
            or not isinstance(max_pending_bundles, int)
            or max_pending_bundles < 1
        ):
            raise ValueError('max_pending_bundles must be a positive integer')
        self._max_pending_bundles = max_pending_bundles
        self.reset(generation=0, seed_stamp_ns=None)

    def reset(self, *, generation: int, seed_stamp_ns: int | None) -> None:
        """Discard all partial bundles and establish a new seed epoch."""
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError('bundle generation must be a non-negative integer')
        if seed_stamp_ns is not None and (
            isinstance(seed_stamp_ns, bool)
            or not isinstance(seed_stamp_ns, int)
            or not 0 <= seed_stamp_ns <= (1 << 63) - 1
        ):
            raise ValueError('bundle seed timestamp is invalid')
        self._generation = generation
        self._seed_stamp_ns = seed_stamp_ns
        self._last_emitted_stamp_ns = seed_stamp_ns
        self._pending: OrderedDict[tuple[int, str], dict[str, object]] = OrderedDict()

    def add(
        self,
        kind: str,
        *,
        generation: int,
        stamp_ns: int,
        frame_id: str,
        payload: object,
    ) -> ExactObservationBundle | None:
        """Add one stream item and return only an exact complete bundle."""
        if kind not in self._KINDS:
            raise ValueError('unknown observation bundle stream')
        if generation != self._generation or self._seed_stamp_ns is None:
            return None
        if (
            isinstance(stamp_ns, bool)
            or not isinstance(stamp_ns, int)
            or not 0 <= stamp_ns <= (1 << 63) - 1
        ):
            raise ValueError('observation timestamp is invalid')
        clean_frame = str(frame_id).strip()
        if not clean_frame or len(clean_frame) > 256:
            raise ValueError('observation frame_id is invalid')
        if (
            stamp_ns <= self._seed_stamp_ns
            or (
                self._last_emitted_stamp_ns is not None
                and stamp_ns <= self._last_emitted_stamp_ns
            )
        ):
            return None

        key = (stamp_ns, clean_frame)
        partial = self._pending.setdefault(key, {})
        partial[kind] = payload
        self._pending.move_to_end(key)
        while len(self._pending) > self._max_pending_bundles:
            self._pending.popitem(last=False)
        if not self._KINDS <= partial.keys():
            return None

        bundle = ExactObservationBundle(
            generation=generation,
            stamp_ns=stamp_ns,
            frame_id=clean_frame,
            detections=partial['detections'],
            target=partial['target'],
            cloud=partial['cloud'],
            manifest=partial['manifest'],
        )
        self._last_emitted_stamp_ns = stamp_ns
        self._pending = OrderedDict(
            (pending_key, value)
            for pending_key, value in self._pending.items()
            if pending_key[0] > stamp_ns
        )
        return bundle


def parse_tracker_frame_manifest(payload: str) -> TrackerFrameManifest:
    """Parse a bounded success manifest without accepting ambiguous epochs."""
    if not isinstance(payload, str) or len(payload) > 4096:
        raise ValueError('tracker frame manifest is not a bounded string')
    try:
        report = json.loads(payload)
    except (TypeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('tracker frame manifest is not valid JSON') from error
    if (
        not isinstance(report, dict)
        or report.get('schema') != 'z_manip.tracker_frame.v1'
    ):
        raise ValueError('unsupported tracker frame manifest schema')
    strings = {
        name: report.get(name)
        for name in ('seed_id', 'frame_id', 'session_id', 'track_id')
    }
    if any(not isinstance(value, str) for value in strings.values()):
        raise ValueError('tracker frame identity fields must be strings')
    strings = {name: value.strip() for name, value in strings.items()}
    if any(not value or len(value) > 256 for value in strings.values()):
        raise ValueError('tracker frame identity fields are invalid')
    integers = {
        name: report.get(name)
        for name in ('seed_stamp_ns', 'adapter_generation', 'result_stamp_ns')
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= (1 << 63) - 1
        for value in integers.values()
    ):
        raise ValueError('tracker frame integer fields are invalid')
    if integers['result_stamp_ns'] <= integers['seed_stamp_ns']:
        raise ValueError('tracker frame does not follow its seed frame')
    return TrackerFrameManifest(
        seed_id=strings['seed_id'],
        seed_stamp_ns=integers['seed_stamp_ns'],
        adapter_generation=integers['adapter_generation'],
        result_stamp_ns=integers['result_stamp_ns'],
        frame_id=strings['frame_id'],
        session_id=strings['session_id'],
        track_id=strings['track_id'],
    )


def parse_tracker_failure_report(payload: str) -> TrackerFailureReport:
    """Parse a bounded tracker failure without accepting ambiguous identities."""
    if not isinstance(payload, str) or len(payload) > 4096:
        raise ValueError('tracker failure payload is not a bounded string')
    try:
        report = json.loads(payload)
    except (TypeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('tracker failure is not valid JSON') from error
    if (
        not isinstance(report, dict)
        or report.get('schema') != 'z_manip.tracker_failure.v1'
    ):
        raise ValueError('unsupported tracker failure schema')
    seed_id = report.get('seed_id')
    seed_stamp_ns = report.get('seed_stamp_ns')
    reason_code = report.get('reason_code')
    reason = report.get('reason')
    replay_candidates = report.get('replay_candidates')
    replay_selected = report.get('replay_selected')
    replay_span_ns = report.get('replay_span_ns')
    acquisition_live_updates = report.get('acquisition_live_updates')
    if not all(isinstance(value, str) for value in (seed_id, reason_code, reason)):
        raise ValueError('tracker failure fields must be strings')
    if (
        isinstance(seed_stamp_ns, bool)
        or not isinstance(seed_stamp_ns, int)
        or not 0 <= seed_stamp_ns <= (1 << 63) - 1
    ):
        raise ValueError('tracker failure seed timestamp is invalid')
    counts = (
        replay_candidates,
        replay_selected,
        replay_span_ns,
        acquisition_live_updates,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise ValueError('tracker failure replay counters are invalid')
    if (
        replay_candidates > 1_000_000
        or replay_selected > 1_000_000
        or replay_span_ns > 86_400_000_000_000
        or acquisition_live_updates > 1_000_000
    ):
        raise ValueError('tracker failure replay counters exceed safety limits')
    if replay_selected > replay_candidates:
        raise ValueError('tracker failure replay selection exceeds candidates')
    seed_id = seed_id.strip()
    if len(seed_id) > 256:
        raise ValueError('tracker failure seed identity exceeds safety limit')
    reason_code = reason_code.replace('\n', ' ').strip()[:128]
    reason = reason.replace('\n', ' ').strip()[:256]
    if not seed_id or not reason_code:
        raise ValueError('tracker failure identity or code is empty')
    return TrackerFailureReport(
        seed_id=seed_id,
        seed_stamp_ns=seed_stamp_ns,
        reason_code=reason_code,
        reason=reason,
        replay_candidates=replay_candidates,
        replay_selected=replay_selected,
        replay_span_ns=replay_span_ns,
        acquisition_live_updates=acquisition_live_updates,
    )


@dataclass(frozen=True)
class ContractSnapshot:
    phase: ContractPhase
    generation: int
    request_id: str
    instruction: str
    target_label: str
    confidence: float
    track_id: str
    failure: FailureCode

    @property
    def valid(self) -> bool:
        return self.phase is ContractPhase.TRACKING


def normalized_xyxy_to_pixel_box(
    xyxy: Iterable[float],
    width: int,
    height: int,
) -> PixelBox:
    """Convert a normalized, positive-area xyxy box to vision_msgs geometry."""
    values = tuple(float(value) for value in xyxy)
    if len(values) != 4:
        raise ValueError('xyxy must contain four values')
    x1, y1, x2, y2 = values
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError('normalized box coordinates must be finite and in [0, 1]')
    if x2 <= x1 or y2 <= y1:
        raise ValueError('normalized box must have positive area')
    return PixelBox(
        center_x=0.5 * (x1 + x2) * width,
        center_y=0.5 * (y1 + y2) * height,
        size_x=(x2 - x1) * width,
        size_y=(y2 - y1) * height,
    )


def expand_pixel_box(
    box: PixelBox,
    width: int,
    height: int,
    fraction: float,
) -> PixelBox:
    """Expand a VLM box on every side and clip it to the source image.

    Vision-language grounding is accurate enough to identify an instance but
    can trim a low-contrast object edge. EdgeTAM needs a prompt that contains
    the complete instance, so the semantic box remains unchanged in the
    affordance record while only the segmentation seed is padded.
    """

    padding = float(fraction)
    if width <= 0 or height <= 0:
        raise ValueError('image dimensions must be positive')
    if (
        not math.isfinite(padding)
        or not 0.0 <= padding <= 1.0
        or not all(math.isfinite(value) for value in (
            box.center_x,
            box.center_y,
            box.size_x,
            box.size_y,
        ))
        or box.size_x <= 0.0
        or box.size_y <= 0.0
    ):
        raise ValueError('pixel box and expansion fraction are invalid')
    half_width = box.size_x * (0.5 + padding)
    half_height = box.size_y * (0.5 + padding)
    x1 = max(0.0, box.center_x - half_width)
    y1 = max(0.0, box.center_y - half_height)
    x2 = min(float(width), box.center_x + half_width)
    y2 = min(float(height), box.center_y + half_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('expanded pixel box is empty')
    return PixelBox(
        center_x=0.5 * (x1 + x2),
        center_y=0.5 * (y1 + y2),
        size_x=x2 - x1,
        size_y=y2 - y1,
    )


class TrackingContract:
    """
    Gate controller inputs on one grounding and a fresh persistent track.

    All timestamps are caller-provided monotonic seconds. ROS header time is kept
    on the relayed messages, while health deadlines continue to work if sim time
    pauses or jumps.
    """

    def __init__(
        self,
        *,
        frame_wait_timeout_s: float = 2.0,
        grounding_timeout_s: float = 35.0,
        acquisition_timeout_s: float = 8.0,
        data_timeout_s: float = 0.5,
        min_cloud_points: int = 24,
    ) -> None:
        limits = (
            frame_wait_timeout_s,
            grounding_timeout_s,
            acquisition_timeout_s,
            data_timeout_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in limits):
            raise ValueError('contract timeouts must be finite and positive')
        if min_cloud_points < 1:
            raise ValueError('min_cloud_points must be positive')
        self.frame_wait_timeout_s = float(frame_wait_timeout_s)
        self.grounding_timeout_s = float(grounding_timeout_s)
        self.acquisition_timeout_s = float(acquisition_timeout_s)
        self.data_timeout_s = float(data_timeout_s)
        self.min_cloud_points = int(min_cloud_points)
        self.generation = 0
        self._last_time_s: float | None = None
        self._reset_state()

    def reset(self) -> int:
        """Invalidate the current epoch, return to idle, and rebase the clock."""
        self.generation += 1
        self._last_time_s = None
        self._reset_state()
        return self.generation

    def _reset_state(self) -> None:
        self.phase = ContractPhase.IDLE
        self.request_id = ''
        self.instruction = ''
        self.target_label = ''
        self.confidence = 0.0
        self.failure = FailureCode.NONE
        self._phase_started_at = 0.0
        self._tracker_seen_true = False
        self._tracker_true_at: float | None = None
        self._detection_ids: frozenset[str] = frozenset()
        self._detections_at: float | None = None
        self._track_id = ''
        self._target_at: float | None = None
        self._cloud_at: float | None = None

    def request(
        self,
        instruction: str,
        *,
        now_s: float,
        request_id: str = '',
    ) -> int:
        query = instruction.strip()
        if not query:
            raise ValueError('grounding instruction must not be empty')
        if not isinstance(request_id, str):
            raise ValueError('grounding request identity must be a string')
        identity = request_id.strip()
        if identity and (
            len(identity) > 128
            or any(ord(character) < 0x21 or ord(character) > 0x7e for character in identity)
        ):
            raise ValueError('grounding request identity is invalid')
        # A new user command is an explicit epoch boundary (for example after a
        # simulator restart), while rollback inside one command remains fatal.
        self._last_time_s = None
        now = self._time(now_s)
        self.generation += 1
        self._reset_state()
        self.phase = ContractPhase.WAITING_FRAME
        self.request_id = identity
        self.instruction = query
        self._phase_started_at = now
        return self.generation

    def grounding_started(self, generation: int, *, now_s: float) -> bool:
        if generation != self.generation:
            return False
        if self.phase is not ContractPhase.WAITING_FRAME:
            raise RuntimeError(f'cannot start grounding while {self.phase.value}')
        self.phase = ContractPhase.GROUNDING
        self._phase_started_at = self._time(now_s)
        return True

    def grounding_succeeded(
        self,
        generation: int,
        *,
        target_label: str,
        confidence: float,
        now_s: float,
    ) -> bool:
        if generation != self.generation:
            return False
        if self.phase is not ContractPhase.GROUNDING:
            raise RuntimeError(f'cannot finish grounding while {self.phase.value}')
        if not target_label.strip() or not math.isfinite(confidence):
            raise ValueError('grounding result needs a label and finite confidence')
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('grounding confidence must be in [0, 1]')
        self.target_label = target_label.strip()
        self.confidence = float(confidence)
        self.phase = ContractPhase.WAITING_TRACKER
        self._phase_started_at = self._time(now_s)
        return True

    def grounding_failed(self, generation: int) -> bool:
        if generation != self.generation:
            return False
        self._fail(FailureCode.GROUNDING_FAILED)
        return True

    def tracker_status(self, is_tracking: bool, *, now_s: float) -> None:
        if self.phase not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING):
            return
        now = self._time(now_s)
        if self.phase is ContractPhase.FAILED:
            return
        if is_tracking:
            self._tracker_seen_true = True
            self._tracker_true_at = now
            self._promote_if_complete(now)
        elif self._tracker_seen_true or self.phase is ContractPhase.TRACKING:
            self._fail(FailureCode.TRACKER_REPORTED_LOSS)

    def tracker_failed(self, *, now_s: float) -> None:
        """Fail immediately on a seed-correlated adapter terminal report."""
        if self.phase not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING):
            return
        self._time(now_s)
        if self.phase is ContractPhase.FAILED:
            return
        self._fail(FailureCode.TRACKER_REPORTED_LOSS)

    def detections(self, track_ids: Iterable[str], *, now_s: float) -> None:
        if self.phase not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING):
            return
        now = self._time(now_s)
        ids = frozenset(
            str(track_id).strip()
            for track_id in track_ids
            if str(track_id).strip()
        )
        self._detection_ids = ids
        self._detections_at = now if ids else None
        if self.phase is ContractPhase.TRACKING:
            if not ids:
                self._fail(FailureCode.EMPTY_DETECTIONS)
                return
            if self._track_id not in ids:
                self._fail(FailureCode.TARGET_MISSING)
                return
        self._promote_if_complete(now)

    def selected_target(self, track_id: str, *, now_s: float) -> None:
        if self.phase not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING):
            return
        candidate = str(track_id).strip()
        if not candidate:
            if self.phase is ContractPhase.TRACKING:
                self._fail(FailureCode.TARGET_MISSING)
            return
        if self._track_id and candidate != self._track_id:
            self._fail(FailureCode.TRACK_ID_CHANGED)
            return
        self._track_id = candidate
        now = self._time(now_s)
        self._target_at = now
        self._promote_if_complete(now)

    def selected_cloud(self, point_count: int, *, now_s: float) -> None:
        if self.phase not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING):
            return
        if int(point_count) < self.min_cloud_points:
            if self.phase is ContractPhase.TRACKING:
                self._fail(FailureCode.CLOUD_TOO_SMALL)
            return
        # A cloud received before a selected ID cannot be associated safely.
        if not self._track_id:
            return
        now = self._time(now_s)
        self._cloud_at = now
        self._promote_if_complete(now)

    def observation_bundle(
        self,
        track_ids: Iterable[str],
        *,
        selected_track_id: str,
        point_count: int,
        now_s: float,
    ) -> None:
        """Accept detections, selected target, and cloud as one exact frame."""
        if self.phase not in (ContractPhase.WAITING_TRACKER, ContractPhase.TRACKING):
            return
        now = self._time(now_s)
        if self.phase is ContractPhase.FAILED:
            return
        ids = frozenset(
            str(track_id).strip()
            for track_id in track_ids
            if str(track_id).strip()
        )
        candidate = str(selected_track_id).strip()
        if not ids:
            self._fail(FailureCode.EMPTY_DETECTIONS)
            return
        if not candidate or candidate not in ids:
            self._fail(FailureCode.TARGET_MISSING)
            return
        if self._track_id and candidate != self._track_id:
            self._fail(FailureCode.TRACK_ID_CHANGED)
            return
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count < self.min_cloud_points
        ):
            self._fail(FailureCode.CLOUD_TOO_SMALL)
            return
        self._detection_ids = ids
        self._track_id = candidate
        self._detections_at = now
        self._target_at = now
        self._cloud_at = now
        self._promote_if_complete(now)

    def tick(self, *, now_s: float) -> ContractSnapshot:
        now = self._time(now_s)
        elapsed = now - self._phase_started_at
        if self.phase is ContractPhase.WAITING_FRAME and elapsed > self.frame_wait_timeout_s:
            self._fail(FailureCode.CAMERA_FRAME_TIMEOUT)
        elif self.phase is ContractPhase.GROUNDING and elapsed > self.grounding_timeout_s:
            self._fail(FailureCode.GROUNDING_TIMEOUT)
        elif self.phase is ContractPhase.WAITING_TRACKER:
            if elapsed > self.acquisition_timeout_s:
                self._fail(FailureCode.TRACKER_ACQUISITION_TIMEOUT)
            else:
                self._promote_if_complete(now)
        elif self.phase is ContractPhase.TRACKING:
            timestamps = (
                self._tracker_true_at,
                self._detections_at,
                self._target_at,
                self._cloud_at,
            )
            if any(
                stamp is None
                or now - stamp < 0.0
                or now - stamp > self.data_timeout_s
                for stamp in timestamps
            ):
                self._fail(FailureCode.TRACKER_DATA_STALE)
        return self.snapshot

    @property
    def snapshot(self) -> ContractSnapshot:
        return ContractSnapshot(
            phase=self.phase,
            generation=self.generation,
            request_id=self.request_id,
            instruction=self.instruction,
            target_label=self.target_label,
            confidence=self.confidence,
            track_id=self._track_id,
            failure=self.failure,
        )

    def _promote_if_complete(self, now_s: float) -> None:
        if self.phase is not ContractPhase.WAITING_TRACKER:
            return
        timestamps = (
            self._tracker_true_at,
            self._detections_at,
            self._target_at,
            self._cloud_at,
        )
        if (
            self._tracker_seen_true
            and self._track_id
            and self._track_id in self._detection_ids
            and all(
                stamp is not None
                and 0.0 <= now_s - stamp <= self.data_timeout_s
                for stamp in timestamps
            )
        ):
            self.phase = ContractPhase.TRACKING
            self._phase_started_at = now_s

    def _fail(self, reason: FailureCode) -> None:
        self.phase = ContractPhase.FAILED
        self.failure = reason

    def _time(self, value: float) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('timestamp must be finite')
        if self._last_time_s is not None and result < self._last_time_s:
            self._fail(FailureCode.CLOCK_ROLLBACK)
            return self._last_time_s
        self._last_time_s = result
        return result
