#!/usr/bin/env python3
"""Run one evidence-backed, no-ground-truth mobile-manipulation acceptance."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TASK_STATUS_TOPIC = '/z_manip/task/status'
TASK_REQUEST_TOPIC = '/z_manip/task/request'
PLACE_STATUS_TOPIC = '/z_manip/place/status'
POST_RELEASE_VERIFICATION_TOPIC = '/z_manip/place/post_release_verification'
EXECUTION_STATUS_TOPIC = '/piper/execution_status'
TASK_STATUS_SCHEMA = 'z_manip.task_status.v1'
POST_RELEASE_VERIFICATION_SCHEMA = 'z_manip.post_release_verification.v2'
POST_RELEASE_VERIFICATION_RESULT = 'post_release_target_stable_in_region'
POST_RELEASE_OBSERVATION_SOURCE = 'synchronized_rgbd_pointcloud'
TASK_POST_RELEASE_FIELDS = frozenset({
    'expected_goal_id', 'expected_release_gripper_command_id',
    'expected_request_id', 'expected_producer_epoch', 'expected_generation',
    'expected_frame_id', 'expected_planning_observation_stamp_ns', 'verified',
    'sample_count', 'stable_duration_s',
})
POST_RELEASE_V2_FIELDS = frozenset({
    'schema', 'state', 'result', 'failure', 'observation_source',
    'goal_id', 'place_goal_id', 'release_gripper_command_id', 'request_id',
    'producer_epoch', 'generation', 'frame_id', 'geometry_frame_id',
    'planning_observation_stamp_ns', 'release_ack_stamp_ns',
    'observation_start_stamp_ns', 'first_observation_stamp_ns',
    'last_observation_stamp_ns', 'first_status_stamp_ns',
    'last_status_stamp_ns', 'first_rgb_stamp_ns', 'first_depth_stamp_ns',
    'first_target_stamp_ns', 'last_rgb_stamp_ns', 'last_depth_stamp_ns',
    'last_target_stamp_ns', 'last_joint_stamp_ns',
    'last_execution_status_received_ns', 'sample_count', 'target_point_count',
    'stable_duration_s', 'max_target_motion_m', 'region_support_fraction',
    'target_gripper_clearance_m', 'target_depth_correspondence_max_error_m',
    'object_position_error_m', 'object_orientation_error_rad',
    'object_upright_error_rad', 'object_registration_inlier_fraction',
    'object_registration_rms_m', 'object_orientation_mode',
    'planned_object_pose', 'observed_object_center_m',
    'rejected_sample_count', 'rejected_sample_reasons',
})
PICK_SUCCESS_PHASES = (
    'planning',
    'transit',
    'pregrasp_reobserve',
    'approach_planning',
    'approach',
    'closing',
    'lift',
    'verify',
    'carry',
)
SUCCESS_PHASES = (
    'place_transit',
    'place_approach',
    'releasing',
    'place_retreat',
    'complete',
)
TERMINAL_PHASES = {'complete', 'failed', 'canceled', 'pick_complete'}
DEFAULT_INSTRUCTION = (
    'Pick up the mustard bottle from the shelf and place it upright on the '
    'empty lower shelf, away from other objects and shelf edges.'
)
DEFAULT_CRITICAL_NODES = (
    'z_manip_urdf_root_alias',
    'vlm_edgetam_bridge',
    'z_manip_edgetam',
    'z_manip_complete_joint_state',
    'z_manip_robot_state_publisher',
    'z_manip_coarse_navigation',
    'z_manip_observed_placement',
    'z_manip_task_runtime',
)
DEFAULT_CRITICAL_TOPICS = (
    '/z_manip/perception/status',
    '/track_3d/frame_manifest',
    '/z_manip/motion/complete_joint_states',
    '/monitored_planning_scene',
    '/z_manip/navigation/status',
    '/z_manip/place/status',
    TASK_STATUS_TOPIC,
)
DEFAULT_UPSTREAM_TOPICS = (
    '/clock',
    '/camera/color/image_raw',
    '/camera/color/camera_info',
    '/camera/aligned_depth_to_color/image_raw',
    '/odom_base_link',
    '/piper/state',
    EXECUTION_STATUS_TOPIC,
)
DEFAULT_BAG_TOPICS = (
    '/clock',
    TASK_REQUEST_TOPIC,
    TASK_STATUS_TOPIC,
    '/z_manip/navigation/status',
    '/z_manip/perception/status',
    PLACE_STATUS_TOPIC,
    '/z_manip/perception/target_3d',
    '/z_manip/perception/tracked_detections_2d',
    '/z_manip/perception/target_pointcloud',
    '/z_manip/perception/scene_pointcloud',
    '/z_manip/perception/affordance',
    '/z_manip/grounding/request',
    '/z_manip/grounding/reset',
    '/z_manip/visual_search/active',
    '/z_manip/coarse_nav/perception_loss_authorization',
    '/track_3d/seed_request',
    '/track_3d/seed_offer_manifest',
    '/track_3d/seed_status',
    '/track_3d/exact_seed_image',
    '/track_3d/init_bbox',
    '/track_3d/reset',
    '/track_3d/is_tracking',
    '/track_3d/failure',
    '/track_3d/frame_manifest',
    '/track_3d/detections_2d',
    '/track_3d/selected_target_3d',
    '/track_3d/selected_target_pointcloud',
    '/z_manip/place/region_request',
    '/z_manip/place/trajectory',
    '/z_manip/place/trajectory_contract',
    POST_RELEASE_VERIFICATION_TOPIC,
    '/piper/joint_trajectory',
    '/piper/gripper_aperture',
    '/piper/trajectory_status',
    EXECUTION_STATUS_TOPIC,
    '/piper/state',
    '/piper/cancel',
    '/piper/named_pose',
    '/goal_reached',
    '/cancel_goal',
    '/way_point',
    '/navigation_cmd_vel',
    '/local_movement_cmd_vel',
    '/odom_base_link',
)


class AcceptanceError(RuntimeError):
    """A fail-closed infrastructure or evidence error."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate JSON key: {key}')
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'non-finite JSON constant: {value}')


def _json_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _ordered_subsequence(
    required: Sequence[str],
    observed: Sequence[str],
) -> bool:
    index = 0
    for value in observed:
        if index < len(required) and value == required[index]:
            index += 1
    return index == len(required)


def _pick_attempt_completed_before_place(observed: Sequence[str]) -> bool:
    """Require one complete pick attempt before the first place transit."""
    progress = 0
    for phase in observed:
        if phase == PICK_SUCCESS_PHASES[0]:
            progress = 1
            continue
        if phase == SUCCESS_PHASES[0]:
            return progress == len(PICK_SUCCESS_PHASES)
        if phase not in PICK_SUCCESS_PHASES:
            continue
        if (
            0 < progress < len(PICK_SUCCESS_PHASES)
            and phase == PICK_SUCCESS_PHASES[progress]
        ):
            progress += 1
        else:
            progress = 0
    return False


def _latch_terminal_receipt(
    current_s: float | None,
    phase: object,
    now_s: float,
) -> float | None:
    """Retain the first terminal receipt despite repeated latched statuses."""
    if current_s is not None or phase not in TERMINAL_PHASES:
        return current_s
    return float(now_s)


def _execution_fields(raw: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    parts = str(raw).split(';')
    if not parts or not parts[0] or '=' in parts[0]:
        return {}
    fields['trajectory'] = parts[0]
    for part in parts[1:]:
        key, separator, value = part.partition('=')
        if not separator or not key or key in fields:
            return {}
        fields[key] = value
    return fields


def _execution_contract(
    fields: dict[str, str],
) -> tuple[str, str, float] | None:
    contract_id = fields.get('trajectory_contract_id', '')
    executor_epoch = fields.get('executor_epoch', '')
    received_at = _finite_float(fields.get('trajectory_received_at'))
    if (
        not contract_id
        or not executor_epoch
        or len(contract_id) > 128
        or len(executor_epoch) > 128
        or any(
            ord(character) < 33
            or ord(character) > 126
            or character in ';|='
            for character in contract_id + executor_epoch
        )
        or received_at is None
        or received_at < 0.0
    ):
        return None
    return contract_id, executor_epoch, received_at


def _source_stamp(
    fields: dict[str, str],
    key: str,
) -> tuple[str, float] | None:
    raw = fields.get(key)
    value = _finite_float(raw)
    if raw is None or raw != raw.strip() or value is None or value < 0.0:
        return None
    return raw, value


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    canonical = result >= 0 and str(value).strip() == str(result)
    return result if canonical else None


def _finite_vector(value: object, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    values = tuple(_finite_float(item) for item in value)
    if any(item is None for item in values):
        return None
    return tuple(float(item) for item in values if item is not None)


def _proper_se3(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    rows = tuple(_finite_vector(row, 4) for row in value)
    if any(row is None for row in rows):
        return False
    matrix = tuple(row for row in rows if row is not None)
    if any(
        abs(matrix[3][index] - expected) > 1e-7
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        return False
    rotation = tuple(row[:3] for row in matrix[:3])
    for first in range(3):
        for second in range(3):
            dot = sum(
                rotation[row][first] * rotation[row][second]
                for row in range(3)
            )
            expected = 1.0 if first == second else 0.0
            if abs(dot - expected) > 1e-5:
                return False
    determinant = (
        rotation[0][0] * (
            rotation[1][1] * rotation[2][2]
            - rotation[1][2] * rotation[2][1]
        )
        - rotation[0][1] * (
            rotation[1][0] * rotation[2][2]
            - rotation[1][2] * rotation[2][0]
        )
        + rotation[0][2] * (
            rotation[1][0] * rotation[2][1]
            - rotation[1][1] * rotation[2][0]
        )
    )
    return abs(determinant - 1.0) <= 1e-5


@dataclass(frozen=True)
class TrajectoryEvent:
    """One immutable executor trajectory identity at a local receipt order."""

    goal_id: str
    executor_epoch: str
    command_id: int
    source_token: str
    source_s: float
    observed_order: int


@dataclass(frozen=True)
class ReleaseObservation:
    """Measured open-gripper evidence while approach is already succeeded."""

    goal_id: str
    executor_epoch: str
    approach_command_id: int
    approach_source_token: str
    approach_source_s: float
    gripper_command_id: int
    gripper_source_token: str
    gripper_source_s: float
    measured_aperture_m: float
    observed_order: int


def _same_trajectory_identity(
    first: TrajectoryEvent,
    second: TrajectoryEvent,
) -> bool:
    return (
        first.goal_id == second.goal_id
        and first.executor_epoch == second.executor_epoch
        and first.command_id == second.command_id
        and first.source_token == second.source_token
        and first.source_s == second.source_s
    )


def _same_release_identity(
    first: ReleaseObservation,
    second: ReleaseObservation,
) -> bool:
    return (
        first.goal_id == second.goal_id
        and first.executor_epoch == second.executor_epoch
        and first.approach_command_id == second.approach_command_id
        and first.approach_source_token == second.approach_source_token
        and first.approach_source_s == second.approach_source_s
        and first.gripper_command_id == second.gripper_command_id
        and first.gripper_source_token == second.gripper_source_token
        and first.gripper_source_s == second.gripper_source_s
    )


def _source_ns(source_s: float) -> int:
    return int(round(float(source_s) * 1e9))


@dataclass
class AcceptanceEvidence:
    """Correlate task, place-planner, and measured executor evidence."""

    instruction: str
    release_min_aperture_m: float = 0.065
    release_stable_samples: int = 3
    release_stable_tolerance_m: float = 0.0008
    post_release_min_stable_duration_s: float = 0.50
    post_release_min_samples: int = 3
    post_release_min_target_points: int = 24
    post_release_max_target_motion_m: float = 0.025
    post_release_min_region_support_fraction: float = 0.80
    post_release_min_gripper_clearance_m: float = 0.04
    post_release_max_rgbd_target_skew_s: float = 0.025
    post_release_max_joint_target_skew_s: float = 0.12
    post_release_max_target_depth_correspondence_m: float = 0.012
    post_release_max_object_position_error_m: float = 0.04
    post_release_max_object_orientation_error_rad: float = 0.35
    post_release_max_object_upright_error_rad: float = 0.26
    post_release_min_object_registration_inlier_fraction: float = 0.55
    post_release_max_object_registration_rms_m: float = 0.025
    task_sent: bool = False
    task_publish_count: int = 0
    phase_history: list[str] = field(default_factory=list)
    task_status_count: int = 0
    terminal_status: dict[str, Any] | None = None
    planned_place_goal_ids: set[str] = field(default_factory=set)
    place_status_count: int = 0
    placement_window_open: bool = False
    execution_observation_count: int = 0
    execution_segments: dict[tuple[str, int], str] = field(
        default_factory=dict,
    )
    execution_conflicts: set[tuple[str, int]] = field(default_factory=set)
    approach_active_events: dict[tuple[str, int], TrajectoryEvent] = field(
        default_factory=dict,
    )
    approach_succeeded_events: dict[tuple[str, int], TrajectoryEvent] = field(
        default_factory=dict,
    )
    retreat_active_events: dict[tuple[str, int], TrajectoryEvent] = field(
        default_factory=dict,
    )
    retreat_succeeded_events: dict[tuple[str, int], TrajectoryEvent] = field(
        default_factory=dict,
    )
    release_observations: dict[
        tuple[str, int], list[ReleaseObservation]
    ] = field(default_factory=dict)
    release_conflicts: set[tuple[str, int]] = field(default_factory=set)
    post_release_status_count: int = 0
    post_release_verifications: dict[
        tuple[object, ...], dict[str, Any]
    ] = field(default_factory=dict)

    def mark_task_published(self) -> None:
        """Open evidence collection for the one allowed request publication."""
        if self.task_sent:
            raise AcceptanceError(
                'task request publication attempted more than once',
            )
        self.task_sent = True
        self.task_publish_count += 1

    def observe_task_status(self, raw: str) -> dict[str, Any] | None:
        """Retain only versioned status owned by this exact instruction."""
        value = _json_object(raw)
        if (
            not self.task_sent
            or value is None
            or value.get('schema') != TASK_STATUS_SCHEMA
            or value.get('instruction') != self.instruction
        ):
            return None
        phase = value.get('phase')
        if not isinstance(phase, str) or not phase:
            return None
        self.task_status_count += 1
        if not self.phase_history or self.phase_history[-1] != phase:
            self.phase_history.append(phase)
        if phase in (
            'place_approach', 'releasing', 'place_retreat', 'complete',
        ):
            self.placement_window_open = True
        if phase in TERMINAL_PHASES and self.terminal_status is None:
            self.terminal_status = value
        return value

    def observe_place_status(self, raw: str) -> dict[str, Any] | None:
        """Retain observed-place planning success and its strict goal ID."""
        if not self.task_sent:
            return None
        value = _json_object(raw)
        if value is None:
            return None
        self.place_status_count += 1
        if value.get('state') != 'planned':
            return value
        detail = _json_object(value.get('detail', ''))
        goal_id = None if detail is None else detail.get('goal_id')
        if isinstance(goal_id, str) and goal_id:
            self.planned_place_goal_ids.add(goal_id)
        return value

    def observe_execution_status(self, raw: str) -> dict[str, str] | None:
        """Observe one fail-closed physical place-execution transition."""
        if not self.task_sent or not self.placement_window_open:
            return None
        fields = _execution_fields(raw)
        if not fields:
            return None
        self.execution_observation_count += 1
        observed_order = self.execution_observation_count
        command_id = _nonnegative_int(fields.get('command_id'))
        contract = _execution_contract(fields)
        source = _source_stamp(fields, 'trajectory_received_at')
        event: TrajectoryEvent | None = None
        if (
            command_id is not None
            and contract is not None
            and source is not None
        ):
            goal_id, executor_epoch, source_s = contract
            source_token, parsed_source_s = source
            if parsed_source_s == source_s:
                event = TrajectoryEvent(
                    goal_id=goal_id,
                    executor_epoch=executor_epoch,
                    command_id=command_id,
                    source_token=source_token,
                    source_s=source_s,
                    observed_order=observed_order,
                )
                self._observe_trajectory_event(fields, event)

        if (
            event is None
            or fields.get('segment') != 'place_approach'
            or fields.get('trajectory') != 'succeeded'
        ):
            return fields
        event_key = (event.executor_epoch, event.command_id)
        succeeded = self.approach_succeeded_events.get(event_key)
        if (
            succeeded is None
            or event_key in self.execution_conflicts
            or not _same_trajectory_identity(succeeded, event)
        ):
            return fields
        gripper_id = _nonnegative_int(fields.get('gripper_command_id'))
        accepted = fields.get('gripper', '')
        if gripper_id is None or not accepted.startswith('accepted:'):
            return fields
        accepted_aperture = _finite_float(accepted.split(':', 1)[1])
        measured_aperture = _finite_float(fields.get('aperture'))
        gripper_source = _source_stamp(fields, 'gripper_received_at')
        if (
            accepted_aperture is not None
            and accepted_aperture >= self.release_min_aperture_m
            and measured_aperture is not None
            and gripper_source is not None
        ):
            gripper_source_token, gripper_source_s = gripper_source
            if gripper_source_s <= event.source_s:
                return fields
            observation = ReleaseObservation(
                goal_id=event.goal_id,
                executor_epoch=event.executor_epoch,
                approach_command_id=event.command_id,
                approach_source_token=event.source_token,
                approach_source_s=event.source_s,
                gripper_command_id=gripper_id,
                gripper_source_token=gripper_source_token,
                gripper_source_s=gripper_source_s,
                measured_aperture_m=measured_aperture,
                observed_order=observed_order,
            )
            release_key = (event.executor_epoch, gripper_id)
            samples = self.release_observations.setdefault(release_key, [])
            if samples and not _same_release_identity(samples[0], observation):
                self.release_conflicts.add(release_key)
                return fields
            samples.append(observation)
        return fields

    def _observe_trajectory_event(
        self,
        fields: dict[str, str],
        event: TrajectoryEvent,
    ) -> None:
        segment = fields.get('segment')
        if segment not in {'place_approach', 'place_retreat'}:
            return
        key = (event.executor_epoch, event.command_id)
        prior_segment = self.execution_segments.setdefault(key, segment)
        if prior_segment != segment:
            self.execution_conflicts.add(key)
            return
        if segment == 'place_approach':
            active_events = self.approach_active_events
            succeeded_events = self.approach_succeeded_events
        else:
            active_events = self.retreat_active_events
            succeeded_events = self.retreat_succeeded_events
        state = fields.get('trajectory')
        if state == 'active':
            existing = active_events.get(key)
            if key in succeeded_events or (
                existing is not None
                and not _same_trajectory_identity(existing, event)
            ):
                self.execution_conflicts.add(key)
                return
            active_events.setdefault(key, event)
        elif state == 'succeeded':
            active = active_events.get(key)
            existing = succeeded_events.get(key)
            if (
                active is None
                or active.observed_order >= event.observed_order
                or not _same_trajectory_identity(active, event)
                or (
                    existing is not None
                    and not _same_trajectory_identity(existing, event)
                )
            ):
                self.execution_conflicts.add(key)
                return
            succeeded_events.setdefault(key, event)

    def observe_post_release_verification(
        self,
        raw: str,
    ) -> dict[str, Any] | None:
        """Validate independent post-release RGB-D geometry evidence."""
        if not self.task_sent:
            return None
        value = _json_object(raw)
        if value is None:
            return None
        self.post_release_status_count += 1
        if not self._valid_post_release_contract(value):
            return value
        identity = (
            value['goal_id'], value['release_gripper_command_id'],
            value['request_id'], value['producer_epoch'], value['generation'],
            value['frame_id'], value['planning_observation_stamp_ns'],
        )
        self.post_release_verifications[identity] = value
        return value

    def _valid_post_release_contract(self, value: dict[str, Any]) -> bool:
        if (
            set(value) != POST_RELEASE_V2_FIELDS
            or
            value.get('schema') != POST_RELEASE_VERIFICATION_SCHEMA
            or value.get('state') != 'verified'
            or value.get('result') != POST_RELEASE_VERIFICATION_RESULT
            or value.get('failure') != ''
            or value.get('observation_source')
            != POST_RELEASE_OBSERVATION_SOURCE
        ):
            return False
        strings = (
            value.get('goal_id'),
            value.get('place_goal_id'),
            value.get('request_id'),
            value.get('producer_epoch'),
            value.get('frame_id'),
            value.get('geometry_frame_id'),
        )
        if (
            any(not isinstance(item, str) or not item for item in strings)
            or value.get('goal_id') != value.get('place_goal_id')
        ):
            return False
        generation = _nonnegative_int(value.get('generation'))
        release_id = _nonnegative_int(value.get('release_gripper_command_id'))
        planning_stamp = _nonnegative_int(
            value.get('planning_observation_stamp_ns'),
        )
        release_stamp = _nonnegative_int(value.get('release_ack_stamp_ns'))
        observation_start = _nonnegative_int(
            value.get('observation_start_stamp_ns'),
        )
        first_stamp = _nonnegative_int(value.get('first_observation_stamp_ns'))
        last_stamp = _nonnegative_int(value.get('last_observation_stamp_ns'))
        first_status = _nonnegative_int(value.get('first_status_stamp_ns'))
        last_status = _nonnegative_int(value.get('last_status_stamp_ns'))
        first_rgb = _nonnegative_int(value.get('first_rgb_stamp_ns'))
        first_depth = _nonnegative_int(value.get('first_depth_stamp_ns'))
        first_target = _nonnegative_int(value.get('first_target_stamp_ns'))
        last_rgb = _nonnegative_int(value.get('last_rgb_stamp_ns'))
        last_depth = _nonnegative_int(value.get('last_depth_stamp_ns'))
        last_target = _nonnegative_int(value.get('last_target_stamp_ns'))
        last_joint = _nonnegative_int(value.get('last_joint_stamp_ns'))
        last_execution = _nonnegative_int(
            value.get('last_execution_status_received_ns'),
        )
        sample_count = _nonnegative_int(value.get('sample_count'))
        target_points = _nonnegative_int(value.get('target_point_count'))
        rejected_count = _nonnegative_int(value.get('rejected_sample_count'))
        rejected_reasons = value.get('rejected_sample_reasons')
        timestamps = (
            planning_stamp, release_stamp, observation_start, first_stamp,
            last_stamp, first_status, last_status, first_rgb, first_depth,
            first_target, last_rgb, last_depth, last_target, last_joint,
            last_execution,
        )
        if (
            generation is None
            or generation == 0
            or release_id is None
            or any(stamp is None or stamp <= 0 for stamp in timestamps)
            or not (
                planning_stamp
                < release_stamp
                <= observation_start
                < first_stamp
                < last_stamp
            )
            or first_status != first_stamp
            or last_status != last_stamp
            or first_target != first_stamp
            or last_target != last_stamp
            or last_execution < release_stamp
            or sample_count is None
            or sample_count < self.post_release_min_samples
            or target_points is None
            or target_points < self.post_release_min_target_points
            or rejected_count is None
            or not isinstance(rejected_reasons, list)
            or len(rejected_reasons) > rejected_count
            or any(
                not isinstance(reason, str) or not reason
                for reason in rejected_reasons
            )
        ):
            return False
        assert first_rgb is not None and first_depth is not None
        assert first_target is not None and last_rgb is not None
        assert last_depth is not None and last_target is not None
        assert last_joint is not None and last_execution is not None
        assert release_stamp is not None and observation_start is not None
        first_sources = (first_rgb, first_depth, first_target)
        last_sources = (last_rgb, last_depth, last_target)
        if (
            min(first_sources) <= observation_start
            or max(first_sources) - min(first_sources)
            > int(self.post_release_max_rgbd_target_skew_s * 1e9)
            or max(last_sources) - min(last_sources)
            > int(self.post_release_max_rgbd_target_skew_s * 1e9)
            or abs(last_joint - last_target)
            > int(self.post_release_max_joint_target_skew_s * 1e9)
        ):
            return False
        stable_duration = _finite_float(value.get('stable_duration_s'))
        max_motion = _finite_float(value.get('max_target_motion_m'))
        support = _finite_float(value.get('region_support_fraction'))
        clearance = _finite_float(value.get('target_gripper_clearance_m'))
        depth_error = _finite_float(
            value.get('target_depth_correspondence_max_error_m'),
        )
        position_error = _finite_float(value.get('object_position_error_m'))
        orientation_error = _finite_float(
            value.get('object_orientation_error_rad'),
        )
        upright_error = _finite_float(value.get('object_upright_error_rad'))
        registration_inlier = _finite_float(
            value.get('object_registration_inlier_fraction'),
        )
        registration_rms = _finite_float(
            value.get('object_registration_rms_m'),
        )
        orientation_mode = value.get('object_orientation_mode')
        pose_valid = _proper_se3(value.get('planned_object_pose'))
        center_valid = _finite_vector(value.get('observed_object_center_m'), 3)
        assert first_stamp is not None and last_stamp is not None
        stamp_duration = (last_stamp - first_stamp) * 1e-9
        return bool(
            stable_duration is not None
            and stable_duration >= self.post_release_min_stable_duration_s
            and stamp_duration >= self.post_release_min_stable_duration_s
            and abs(stable_duration - stamp_duration) <= 1e-6
            and max_motion is not None
            and 0.0 <= max_motion <= self.post_release_max_target_motion_m
            and support is not None
            and self.post_release_min_region_support_fraction <= support <= 1.0
            and clearance is not None
            and clearance >= self.post_release_min_gripper_clearance_m
            and depth_error is not None
            and 0.0 <= depth_error
            <= self.post_release_max_target_depth_correspondence_m
            and position_error is not None
            and 0.0 <= position_error
            <= self.post_release_max_object_position_error_m
            and orientation_error is not None
            and 0.0 <= orientation_error
            <= self.post_release_max_object_orientation_error_rad
            and upright_error is not None
            and 0.0 <= upright_error
            <= self.post_release_max_object_upright_error_rad
            and registration_inlier is not None
            and self.post_release_min_object_registration_inlier_fraction
            <= registration_inlier <= 1.0
            and registration_rms is not None
            and 0.0 <= registration_rms
            <= self.post_release_max_object_registration_rms_m
            and orientation_mode in {'full', 'axial'}
            and pose_valid
            and center_valid is not None
        )

    def _stable_release_windows(
        self,
        goal_id: object,
    ) -> list[tuple[ReleaseObservation, ...]]:
        if not isinstance(goal_id, str) or not goal_id:
            return []
        windows: list[tuple[ReleaseObservation, ...]] = []
        required = self.release_stable_samples
        for key, samples in self.release_observations.items():
            if key in self.release_conflicts or len(samples) < required:
                continue
            for end in range(required, len(samples) + 1):
                recent = tuple(samples[end - required:end])
                apertures = tuple(
                    sample.measured_aperture_m for sample in recent
                )
                orders = tuple(sample.observed_order for sample in recent)
                if (
                    recent[0].goal_id == goal_id
                    and all(
                        _same_release_identity(recent[0], sample)
                        for sample in recent[1:]
                    )
                    and all(
                        first < second
                        for first, second in zip(orders, orders[1:])
                    )
                    and min(apertures) >= self.release_min_aperture_m
                    and max(apertures) - min(apertures)
                    <= self.release_stable_tolerance_m
                ):
                    windows.append(recent)
        return windows

    def _valid_execution_chains(
        self,
        goal_id: object,
    ) -> list[tuple[
        TrajectoryEvent,
        TrajectoryEvent,
        tuple[ReleaseObservation, ...],
        TrajectoryEvent,
        TrajectoryEvent,
    ]]:
        chains: list[tuple[
            TrajectoryEvent,
            TrajectoryEvent,
            tuple[ReleaseObservation, ...],
            TrajectoryEvent,
            TrajectoryEvent,
        ]] = []
        for release_window in self._stable_release_windows(goal_id):
            release = release_window[0]
            approach_key = (
                release.executor_epoch,
                release.approach_command_id,
            )
            approach_active = self.approach_active_events.get(approach_key)
            approach_succeeded = self.approach_succeeded_events.get(
                approach_key,
            )
            if (
                approach_key in self.execution_conflicts
                or approach_active is None
                or approach_succeeded is None
                or not _same_trajectory_identity(
                    approach_active,
                    approach_succeeded,
                )
                or approach_succeeded.goal_id != release.goal_id
                or approach_succeeded.source_token
                != release.approach_source_token
                or not (
                    approach_active.observed_order
                    < approach_succeeded.observed_order
                    <= release_window[0].observed_order
                )
            ):
                continue
            for retreat_key, retreat_succeeded in (
                self.retreat_succeeded_events.items()
            ):
                retreat_active = self.retreat_active_events.get(retreat_key)
                if (
                    retreat_key in self.execution_conflicts
                    or retreat_active is None
                    or not _same_trajectory_identity(
                        retreat_active,
                        retreat_succeeded,
                    )
                    or retreat_active.goal_id != release.goal_id
                    or retreat_active.executor_epoch != release.executor_epoch
                    or not (
                        approach_active.command_id
                        < retreat_active.command_id
                    )
                    or not (
                        approach_active.source_s
                        < release.gripper_source_s
                        < retreat_active.source_s
                    )
                    or not (
                        release_window[-1].observed_order
                        < retreat_active.observed_order
                        < retreat_succeeded.observed_order
                    )
                ):
                    continue
                chains.append((
                    approach_active,
                    approach_succeeded,
                    release_window,
                    retreat_active,
                    retreat_succeeded,
                ))
        return chains

    @staticmethod
    def _terminal_matches_verification(
        terminal: dict[str, Any],
        verification: dict[str, Any],
    ) -> bool:
        expected = terminal.get('post_release_verification')
        if (
            not isinstance(expected, dict)
            or set(expected) != TASK_POST_RELEASE_FIELDS
        ):
            return False
        expected_pairs = (
            ('expected_goal_id', 'goal_id'),
            (
                'expected_release_gripper_command_id',
                'release_gripper_command_id',
            ),
            ('expected_request_id', 'request_id'),
            ('expected_producer_epoch', 'producer_epoch'),
            ('expected_generation', 'generation'),
            ('expected_frame_id', 'frame_id'),
            (
                'expected_planning_observation_stamp_ns',
                'planning_observation_stamp_ns',
            ),
            ('sample_count', 'sample_count'),
        )
        stable_duration = _finite_float(expected.get('stable_duration_s'))
        verified_duration = _finite_float(
            verification.get('stable_duration_s'),
        )
        return bool(
            expected.get('verified') is True
            and all(
                expected.get(expected_key) == verification.get(actual_key)
                for expected_key, actual_key in expected_pairs
            )
            and stable_duration is not None
            and verified_duration is not None
            and abs(stable_duration - verified_duration) <= 1e-9
        )

    def _verified_execution_chain(
        self,
        terminal: dict[str, Any],
        chains: Sequence[tuple[
            TrajectoryEvent,
            TrajectoryEvent,
            tuple[ReleaseObservation, ...],
            TrajectoryEvent,
            TrajectoryEvent,
        ]],
    ) -> bool:
        for verification in self.post_release_verifications.values():
            if not self._terminal_matches_verification(terminal, verification):
                continue
            release_id = _nonnegative_int(
                verification.get('release_gripper_command_id'),
            )
            release_ack_ns = _nonnegative_int(
                verification.get('release_ack_stamp_ns'),
            )
            observation_start_ns = _nonnegative_int(
                verification.get('observation_start_stamp_ns'),
            )
            if (
                release_id is None
                or release_ack_ns is None
                or observation_start_ns is None
            ):
                continue
            for chain in chains:
                release = chain[2][0]
                retreat = chain[3]
                if (
                    release.gripper_command_id == release_id
                    and _source_ns(release.gripper_source_s) <= release_ack_ns
                    and _source_ns(retreat.source_s) <= observation_start_ns
                ):
                    return True
        return False

    def checks(self, *, bag_closed_cleanly: bool) -> dict[str, bool]:
        """Return every independent acceptance predicate."""
        terminal = self.terminal_status or {}
        goal_id = terminal.get('place_goal_id')
        goal_correlated = (
            isinstance(goal_id, str)
            and bool(goal_id)
            and goal_id in self.planned_place_goal_ids
        )
        approach_success = any(
            key not in self.execution_conflicts
            and event.goal_id == goal_id
            and (active := self.approach_active_events.get(key)) is not None
            and _same_trajectory_identity(active, event)
            and active.observed_order < event.observed_order
            for key, event in self.approach_succeeded_events.items()
        )
        stable_release = bool(self._stable_release_windows(goal_id))
        retreat_success = any(
            key not in self.execution_conflicts
            and event.goal_id == goal_id
            and (active := self.retreat_active_events.get(key)) is not None
            and _same_trajectory_identity(active, event)
            and active.observed_order < event.observed_order
            for key, event in self.retreat_succeeded_events.items()
        )
        chains = self._valid_execution_chains(goal_id)
        physical_event_order = bool(chains)
        pick_phase_order = _pick_attempt_completed_before_place(
            self.phase_history,
        )
        phase_order = _ordered_subsequence(SUCCESS_PHASES, self.phase_history)
        place_plan_available = terminal.get('place_plan_available') is True
        place_execution_evidence = all((
            goal_correlated,
            place_plan_available,
            phase_order,
            approach_success,
            stable_release,
            retreat_success,
            physical_event_order,
        ))
        post_release_verified = self._verified_execution_chain(
            terminal,
            chains,
        )
        observed_place_success = bool(
            place_execution_evidence and post_release_verified
        )
        return {
            'task_published_exactly_once': self.task_publish_count == 1,
            'terminal_phase_complete': terminal.get('phase') == 'complete',
            'terminal_result_mobile_manip_complete': (
                terminal.get('result') == 'mobile_manip_complete'
            ),
            'pick_two_stage_phase_order_observed': pick_phase_order,
            'place_planner_success_observed': goal_correlated,
            'place_execution_phase_order_observed': phase_order,
            'place_plan_available_at_completion': place_plan_available,
            'place_approach_active_then_succeeded_observed': approach_success,
            'stable_measured_gripper_release_observed': stable_release,
            'place_retreat_active_then_succeeded_observed': retreat_success,
            'physical_place_event_order_observed': physical_event_order,
            'place_execution_evidence': place_execution_evidence,
            'terminal_post_release_identity_correlated': post_release_verified,
            'post_release_target_stable_in_region_observed': (
                post_release_verified
            ),
            'observed_place_success': observed_place_success,
            'bag_closed_cleanly': bool(bag_closed_cleanly),
        }

    def summary(self) -> dict[str, Any]:
        """Serialize bounded evidence useful for an acceptance audit."""
        return {
            'task_publish_count': self.task_publish_count,
            'task_status_count': self.task_status_count,
            'place_status_count': self.place_status_count,
            'post_release_status_count': self.post_release_status_count,
            'execution_observation_count': self.execution_observation_count,
            'phase_history': list(self.phase_history),
            'pick_two_stage_phase_order_observed': (
                _pick_attempt_completed_before_place(
                    self.phase_history,
                )
            ),
            'terminal_status': self.terminal_status,
            'planned_place_goal_ids': sorted(self.planned_place_goal_ids),
            'release_accepted_command_ids': sorted(
                key[1] for key in self.release_observations
            ),
            'release_sample_counts': {
                f'{key[0]}:{key[1]}': len(value)
                for key, value in sorted(
                    self.release_observations.items(),
                )
            },
            'approach_active_command_ids': sorted(
                key[1] for key in self.approach_active_events
            ),
            'approach_succeeded_command_ids': sorted(
                key[1] for key in self.approach_succeeded_events
            ),
            'retreat_active_command_ids': sorted(
                key[1] for key in self.retreat_active_events
            ),
            'retreat_succeeded_command_ids': sorted(
                key[1] for key in self.retreat_succeeded_events
            ),
            'execution_conflicts': sorted(
                f'{key[0]}:{key[1]}' for key in self.execution_conflicts
            ),
            'release_conflicts': sorted(
                f'{key[0]}:{key[1]}' for key in self.release_conflicts
            ),
            'post_release_verified_goal_ids': sorted(
                {
                    str(value['goal_id'])
                    for value in self.post_release_verifications.values()
                },
            ),
        }


class EventWriter:
    """Write flush-on-event JSON Lines evidence."""

    def __init__(self, path: Path) -> None:
        """Open a new event file without replacing prior evidence."""
        self._stream = path.open('x', encoding='utf-8')

    def write(self, kind: str, payload: object) -> None:
        """Append and flush one timestamped event."""
        value = {
            'wall_time_utc': datetime.now(timezone.utc).isoformat(),
            'monotonic_s': time.monotonic(),
            'kind': kind,
            'payload': payload,
        }
        self._stream.write(json.dumps(value, separators=(',', ':')) + '\n')
        self._stream.flush()

    def close(self) -> None:
        """Close the event stream after process cleanup."""
        self._stream.close()


def _canonical_namespace(value: str) -> str:
    clean = str(value).strip()
    if not clean or clean == '/':
        return '/'
    return '/' + clean.strip('/')


def _qualified_name(namespace: str, node_name: str) -> str:
    prefix = '' if namespace == '/' else namespace
    return f'{prefix}/{node_name}'


def _critical_graph_ready(
    node_counts: dict[str, int],
    topic_counts: dict[str, int],
) -> bool:
    """Require exactly one owner for every complete-runtime graph contract."""
    counts = (*node_counts.values(), *topic_counts.values())
    return bool(counts) and all(value == 1 for value in counts)


def _default_upstream_topics(use_sim_time: str) -> tuple[str, ...]:
    if use_sim_time == 'true':
        return DEFAULT_UPSTREAM_TOPICS
    return tuple(
        topic for topic in DEFAULT_UPSTREAM_TOPICS if topic != '/clock'
    )


def _endpoint_name(endpoint: object) -> str:
    name = str(getattr(endpoint, 'node_name', '')).strip('/')
    namespace = _canonical_namespace(
        str(getattr(endpoint, 'node_namespace', '/')),
    )
    return _qualified_name(namespace, name) if name else ''


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def _group_alive(process: subprocess.Popen[Any]) -> bool:
    pgid = process.pid
    proc = Path('/proc')
    if proc.is_dir():
        try:
            entries = tuple(proc.iterdir())
        except OSError:
            entries = ()
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                line = (entry / 'stat').read_text()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            close = line.rfind(')')
            fields = line[close + 2:].split() if close >= 0 else ()
            if len(fields) < 3:
                continue
            try:
                member_pgid = int(fields[2])
            except ValueError:
                continue
            if member_pgid == pgid and fields[0] != 'Z':
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_process_group(
    process: subprocess.Popen[Any] | None,
    *,
    interrupt_timeout_s: float,
) -> tuple[bool, int | None]:
    if process is None:
        return True, None
    if _group_alive(process):
        os.killpg(process.pid, signal.SIGINT)
        deadline = time.monotonic() + interrupt_timeout_s
        while _group_alive(process) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.02)
        if _group_alive(process):
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + min(5.0, interrupt_timeout_s)
            while _group_alive(process) and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.02)
            if _group_alive(process):
                os.killpg(process.pid, signal.SIGKILL)
                deadline = time.monotonic() + 2.0
                while _group_alive(process) and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.02)
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    return not _group_alive(process), process.returncode


def _path_argument(value: str, description: str) -> str:
    clean = str(value).strip()
    if not clean:
        raise AcceptanceError(f'{description} is required')
    if not Path(clean).is_file():
        raise AcceptanceError(f'{description} is not a file: {clean}')
    return clean


def _optional_path_argument(value: str, description: str) -> str | None:
    """Validate an optional file without assigning a deployment default."""
    clean = str(value).strip()
    if not clean:
        return None
    return _path_argument(clean, description)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run one supervised mobile-manipulation acceptance task.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--instruction', default=DEFAULT_INSTRUCTION)
    parser.add_argument('--ros-domain-id', type=int, default=184)
    parser.add_argument('--namespace', default='/')
    parser.add_argument(
        '--supervisor',
        default=os.environ.get(
            'Z_MANIP_SUPERVISOR',
            '/usr/local/bin/z-manip-mobile-manipulation',
        ),
    )
    parser.add_argument(
        '--robot-description-file',
        default=os.environ.get(
            'Z_MANIP_ROBOT_DESCRIPTION_FILE',
            '/robot/assets/urdf/go2w_sensored.urdf',
        ),
    )
    parser.add_argument(
        '--stack-config-path',
        default=os.environ.get(
            'Z_MANIP_STACK_CONFIG',
            '/opt/z_manip/configs/go2w_piper.json',
        ),
    )
    parser.add_argument(
        '--task-platform-parameters',
        default=os.environ.get('Z_MANIP_TASK_PLATFORM_PARAMETERS', ''),
        help=(
            'Optional z_manip_task platform YAML; omit for generic/real behavior'
        ),
    )
    parser.add_argument(
        '--collision-model-file',
        default=os.environ.get(
            'Z_MANIP_COLLISION_MODEL_FILE',
            '/opt/z_manip/configs/piper_collision_capsules.json',
        ),
    )
    parser.add_argument(
        '--use-sim-time',
        choices=('true', 'false'),
        default='true',
    )
    parser.add_argument('--output-root', type=Path, default=Path('/tmp'))
    parser.add_argument('--task-timeout', type=float, default=1200.0)
    parser.add_argument(
        '--runtime-readiness-timeout', type=float, default=60.0,
    )
    parser.add_argument(
        '--recorder-readiness-timeout', type=float, default=30.0,
    )
    parser.add_argument('--terminal-drain-time', type=float, default=3.0)
    parser.add_argument('--shutdown-timeout', type=float, default=15.0)
    parser.add_argument('--release-min-aperture-m', type=float, default=0.065)
    parser.add_argument('--release-stable-samples', type=int, default=3)
    parser.add_argument(
        '--release-stable-tolerance-m', type=float, default=0.0008,
    )
    parser.add_argument(
        '--post-release-min-stable-duration-s', type=float, default=0.50,
    )
    parser.add_argument('--post-release-min-samples', type=int, default=3)
    parser.add_argument(
        '--post-release-min-target-points', type=int, default=24,
    )
    parser.add_argument(
        '--post-release-max-target-motion-m', type=float, default=0.025,
    )
    parser.add_argument(
        '--post-release-min-region-support-fraction',
        type=float,
        default=0.80,
    )
    parser.add_argument(
        '--post-release-min-gripper-clearance-m', type=float, default=0.04,
    )
    parser.add_argument(
        '--post-release-max-rgbd-target-skew-s', type=float, default=0.025,
    )
    parser.add_argument(
        '--post-release-max-joint-target-skew-s', type=float, default=0.12,
    )
    parser.add_argument(
        '--post-release-max-target-depth-correspondence-m',
        type=float,
        default=0.012,
    )
    parser.add_argument(
        '--post-release-max-object-position-error-m', type=float, default=0.04,
    )
    parser.add_argument(
        '--post-release-max-object-orientation-error-rad',
        type=float,
        default=0.35,
    )
    parser.add_argument(
        '--post-release-max-object-upright-error-rad',
        type=float,
        default=0.26,
    )
    parser.add_argument(
        '--post-release-min-object-registration-inlier-fraction',
        type=float,
        default=0.55,
    )
    parser.add_argument(
        '--post-release-max-object-registration-rms-m',
        type=float,
        default=0.025,
    )
    parser.add_argument(
        '--required-upstream-topic',
        action='append',
        default=None,
        help='Replace the default upstream publisher readiness set; repeat.',
    )
    parser.add_argument(
        '--bag-topic',
        action='append',
        default=None,
        help='Replace the default evidence topic set; repeat.',
    )
    parsed = parser.parse_args(argv)
    if not 0 <= parsed.ros_domain_id <= 232:
        parser.error('--ros-domain-id must be in [0, 232]')
    if not str(parsed.instruction).strip():
        parser.error('--instruction must not be empty')
    for name in (
        'task_timeout',
        'recorder_readiness_timeout',
        'terminal_drain_time',
        'shutdown_timeout',
    ):
        if not 0.0 < float(getattr(parsed, name)) <= 86400.0:
            parser.error(f'--{name.replace("_", "-")} must be in (0, 86400]')
    if not 0.0 < parsed.runtime_readiness_timeout <= 300.0:
        parser.error('--runtime-readiness-timeout must be in (0, 300]')
    if (
        not math.isfinite(parsed.release_min_aperture_m)
        or parsed.release_min_aperture_m <= 0.0
        or parsed.release_stable_samples <= 0
        or not math.isfinite(parsed.release_stable_tolerance_m)
        or parsed.release_stable_tolerance_m < 0.0
        or not math.isfinite(parsed.post_release_min_stable_duration_s)
        or parsed.post_release_min_stable_duration_s <= 0.0
        or parsed.post_release_min_samples <= 0
        or parsed.post_release_min_target_points <= 0
        or not math.isfinite(parsed.post_release_max_target_motion_m)
        or parsed.post_release_max_target_motion_m < 0.0
        or not math.isfinite(
            parsed.post_release_min_region_support_fraction
        )
        or not 0.0 < parsed.post_release_min_region_support_fraction <= 1.0
        or not math.isfinite(parsed.post_release_min_gripper_clearance_m)
        or parsed.post_release_min_gripper_clearance_m <= 0.0
        or not math.isfinite(parsed.post_release_max_rgbd_target_skew_s)
        or parsed.post_release_max_rgbd_target_skew_s <= 0.0
        or not math.isfinite(parsed.post_release_max_joint_target_skew_s)
        or parsed.post_release_max_joint_target_skew_s <= 0.0
        or not math.isfinite(
            parsed.post_release_max_target_depth_correspondence_m
        )
        or parsed.post_release_max_target_depth_correspondence_m <= 0.0
        or not math.isfinite(parsed.post_release_max_object_position_error_m)
        or parsed.post_release_max_object_position_error_m <= 0.0
        or not math.isfinite(
            parsed.post_release_max_object_orientation_error_rad
        )
        or parsed.post_release_max_object_orientation_error_rad <= 0.0
        or not math.isfinite(parsed.post_release_max_object_upright_error_rad)
        or parsed.post_release_max_object_upright_error_rad <= 0.0
        or not math.isfinite(
            parsed.post_release_min_object_registration_inlier_fraction
        )
        or not 0.0
        < parsed.post_release_min_object_registration_inlier_fraction <= 1.0
        or not math.isfinite(parsed.post_release_max_object_registration_rms_m)
        or parsed.post_release_max_object_registration_rms_m <= 0.0
    ):
        parser.error('release evidence thresholds are invalid')
    parsed.namespace = _canonical_namespace(parsed.namespace)
    return parsed


def _run_ros_acceptance(
    arguments: argparse.Namespace,
    run_directory: Path,
    event_writer: EventWriter,
) -> tuple[AcceptanceEvidence, str, bool, dict[str, Any]]:
    evidence = AcceptanceEvidence(
        str(arguments.instruction).strip(),
        release_min_aperture_m=float(arguments.release_min_aperture_m),
        release_stable_samples=int(arguments.release_stable_samples),
        release_stable_tolerance_m=float(arguments.release_stable_tolerance_m),
        post_release_min_stable_duration_s=float(
            arguments.post_release_min_stable_duration_s,
        ),
        post_release_min_samples=int(arguments.post_release_min_samples),
        post_release_min_target_points=int(
            arguments.post_release_min_target_points,
        ),
        post_release_max_target_motion_m=float(
            arguments.post_release_max_target_motion_m,
        ),
        post_release_min_region_support_fraction=float(
            arguments.post_release_min_region_support_fraction,
        ),
        post_release_min_gripper_clearance_m=float(
            arguments.post_release_min_gripper_clearance_m,
        ),
        post_release_max_rgbd_target_skew_s=float(
            arguments.post_release_max_rgbd_target_skew_s,
        ),
        post_release_max_joint_target_skew_s=float(
            arguments.post_release_max_joint_target_skew_s,
        ),
        post_release_max_target_depth_correspondence_m=float(
            arguments.post_release_max_target_depth_correspondence_m,
        ),
        post_release_max_object_position_error_m=float(
            arguments.post_release_max_object_position_error_m,
        ),
        post_release_max_object_orientation_error_rad=float(
            arguments.post_release_max_object_orientation_error_rad,
        ),
        post_release_max_object_upright_error_rad=float(
            arguments.post_release_max_object_upright_error_rad,
        ),
        post_release_min_object_registration_inlier_fraction=float(
            arguments.post_release_min_object_registration_inlier_fraction,
        ),
        post_release_max_object_registration_rms_m=float(
            arguments.post_release_max_object_registration_rms_m,
        ),
    )
    state: dict[str, Any] = {
        'terminal_received_at': None,
        'infrastructure_error': '',
    }
    # ROS imports stay behind argument/environment validation so pure contract
    # tests can import this module on a non-ROS host.
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
    except ImportError as error:
        detail = f'{type(error).__name__}: {error}'
        state['infrastructure_error'] = detail
        event_writer.write('acceptance_error', detail)
        return evidence, 'infrastructure_error', False, state

    class Monitor(Node):
        def __init__(self) -> None:
            super().__init__('z_manip_e2e_acceptance')
            reliable = QoSProfile(
                depth=20,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            latched = QoSProfile(
                depth=20,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.task_publisher = self.create_publisher(
                String,
                TASK_REQUEST_TOPIC,
                reliable,
            )
            self.create_subscription(
                String,
                TASK_STATUS_TOPIC,
                self._task_status,
                latched,
            )
            self.create_subscription(
                String,
                PLACE_STATUS_TOPIC,
                self._place_status,
                reliable,
            )
            self.create_subscription(
                String,
                POST_RELEASE_VERIFICATION_TOPIC,
                self._post_release_verification,
                latched,
            )
            self.create_subscription(
                String,
                EXECUTION_STATUS_TOPIC,
                self._execution_status,
                reliable,
            )

        def _task_status(self, message: String) -> None:
            value = evidence.observe_task_status(message.data)
            if value is None:
                return
            event_writer.write('task_status', value)
            state['terminal_received_at'] = _latch_terminal_receipt(
                state['terminal_received_at'],
                value.get('phase'),
                time.monotonic(),
            )

        def _place_status(self, message: String) -> None:
            value = evidence.observe_place_status(message.data)
            if value is not None:
                event_writer.write('place_status', value)

        def _execution_status(self, message: String) -> None:
            value = evidence.observe_execution_status(message.data)
            if value is not None:
                event_writer.write('placement_execution_status', value)

        def _post_release_verification(self, message: String) -> None:
            value = evidence.observe_post_release_verification(message.data)
            if value is not None:
                event_writer.write('post_release_verification', value)

    supervisor: subprocess.Popen[bytes] | None = None
    recorder: subprocess.Popen[bytes] | None = None
    supervisor_log = None
    recorder_log = None
    monitor: Monitor | None = None
    rclpy_started = False
    bag_closed_cleanly = False
    process_summary: dict[str, Any] = {}
    outcome = 'infrastructure_error'

    def spin_until(predicate: Any, timeout_s: float, description: str) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            if supervisor is not None and supervisor.poll() is not None:
                raise AcceptanceError(
                    f'singleton supervisor exited before {description}: '
                    f'{supervisor.returncode}',
                )
            rclpy.spin_once(monitor, timeout_sec=0.10)
        raise AcceptanceError(f'timed out waiting for {description}')

    try:
        supervisor_path = _path_argument(arguments.supervisor, 'supervisor')
        robot_file = _path_argument(
            arguments.robot_description_file,
            'robot description',
        )
        stack_file = _path_argument(
            arguments.stack_config_path,
            'stack config',
        )
        task_platform_file = _optional_path_argument(
            arguments.task_platform_parameters,
            'task platform parameters',
        )
        collision_model_file = _path_argument(
            arguments.collision_model_file,
            'collision model',
        )
        supervisor_log = (run_directory / 'supervisor.log').open('xb')
        supervisor_command = [
            supervisor_path,
            '--namespace', arguments.namespace,
            '--startup-timeout', str(arguments.runtime_readiness_timeout),
            '--shutdown-timeout', str(arguments.shutdown_timeout),
            '--',
            f'use_sim_time:={arguments.use_sim_time}',
            f'robot_description_file:={robot_file}',
            f'stack_config_path:={stack_file}',
            f'collision_model_file:={collision_model_file}',
        ]
        if task_platform_file is not None:
            supervisor_command.append(
                f'task_platform_parameters:={task_platform_file}',
            )
        event_writer.write('supervisor_command', supervisor_command)
        supervisor = subprocess.Popen(
            supervisor_command,
            stdout=supervisor_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        rclpy.init(args=[])
        rclpy_started = True
        monitor = Monitor()
        required_nodes = tuple(
            _qualified_name(arguments.namespace, name)
            for name in DEFAULT_CRITICAL_NODES
        )
        required_runtime_topics = DEFAULT_CRITICAL_TOPICS
        upstream_topics = tuple(
            arguments.required_upstream_topic
            or _default_upstream_topics(arguments.use_sim_time)
        )

        def runtime_ready() -> bool:
            names = [
                _qualified_name(_canonical_namespace(namespace), name)
                for name, namespace in monitor.get_node_names_and_namespaces()
            ]
            node_counts = {name: names.count(name) for name in required_nodes}
            topic_counts = {
                topic: len(monitor.get_publishers_info_by_topic(topic))
                for topic in (*required_runtime_topics, *upstream_topics)
            }
            return _critical_graph_ready(node_counts, topic_counts)

        spin_until(
            runtime_ready,
            float(arguments.runtime_readiness_timeout),
            'unique runtime and upstream publishers',
        )
        event_writer.write('runtime_ready', {
            'nodes': required_nodes,
            'runtime_topics': required_runtime_topics,
            'upstream_topics': upstream_topics,
        })

        bag_directory = run_directory / 'bag'
        recorder_log = (run_directory / 'recorder.log').open('xb')
        bag_topics = tuple(arguments.bag_topic or DEFAULT_BAG_TOPICS)
        recorder_command = [
            'ros2', 'bag', 'record',
            '--storage', 'mcap',
            '--output', str(bag_directory),
            '--disable-keyboard-controls',
            '--topics', *bag_topics,
        ]
        event_writer.write('recorder_command', recorder_command)
        recorder = subprocess.Popen(
            recorder_command,
            stdout=recorder_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        expected_task_node = _qualified_name(
            arguments.namespace,
            'z_manip_task_runtime',
        )

        def request_subscribers_ready() -> bool:
            if recorder is not None and recorder.poll() is not None:
                raise AcceptanceError(
                    'rosbag recorder exited during readiness: '
                    f'{recorder.returncode}',
                )
            endpoints = monitor.get_subscriptions_info_by_topic(
                TASK_REQUEST_TOPIC,
            )
            names = {_endpoint_name(endpoint) for endpoint in endpoints}
            recorder_seen = any('rosbag2_recorder' in name for name in names)
            return (
                expected_task_node in names
                and recorder_seen
                and monitor.task_publisher.get_subscription_count() >= 2
            )

        spin_until(
            request_subscribers_ready,
            float(arguments.recorder_readiness_timeout),
            'task runtime and rosbag request subscriptions',
        )
        event_writer.write('task_subscribers_ready', {
            'names': sorted(
                _endpoint_name(endpoint)
                for endpoint in monitor.get_subscriptions_info_by_topic(
                    TASK_REQUEST_TOPIC,
                )
            ),
            'matched_count': monitor.task_publisher.get_subscription_count(),
        })

        evidence.mark_task_published()
        monitor.task_publisher.publish(String(data=evidence.instruction))
        event_writer.write('task_published', {
            'instruction': evidence.instruction,
            'publish_count': evidence.task_publish_count,
        })
        acknowledged = monitor.task_publisher.wait_for_all_acked(
            Duration(seconds=5.0),
        )
        if not acknowledged:
            raise AcceptanceError(
                'task request was not acknowledged by all subscribers',
            )

        deadline = time.monotonic() + float(arguments.task_timeout)
        while time.monotonic() < deadline:
            if supervisor.poll() is not None:
                raise AcceptanceError(
                    'singleton supervisor exited during task: '
                    f'{supervisor.returncode}',
                )
            if recorder.poll() is not None:
                raise AcceptanceError(
                    'rosbag recorder exited during task: '
                    f'{recorder.returncode}',
                )
            rclpy.spin_once(monitor, timeout_sec=0.10)
            terminal_at = state['terminal_received_at']
            if (
                terminal_at is not None
                and time.monotonic() - terminal_at
                >= float(arguments.terminal_drain_time)
            ):
                break
        else:
            outcome = 'task_timeout'

        if evidence.terminal_status is not None:
            outcome = (
                'terminal_complete'
                if evidence.terminal_status.get('phase') == 'complete'
                else 'terminal_failure'
            )
    except Exception as error:
        state['infrastructure_error'] = f'{type(error).__name__}: {error}'
        event_writer.write('acceptance_error', state['infrastructure_error'])
    finally:
        recorder_stopped, recorder_returncode = _stop_process_group(
            recorder,
            interrupt_timeout_s=float(arguments.shutdown_timeout),
        )
        bag_directory = run_directory / 'bag'
        bag_metadata = bag_directory / 'metadata.yaml'
        bag_payloads = (
            tuple(bag_directory.glob('*.mcap'))
            if bag_directory.is_dir()
            else ()
        )
        bag_closed_cleanly = bool(
            recorder is not None
            and recorder_stopped
            and recorder_returncode in (
                0, -signal.SIGINT, 128 + signal.SIGINT,
            )
            and bag_metadata.is_file()
            and any(path.stat().st_size > 0 for path in bag_payloads)
        )
        supervisor_stopped, supervisor_returncode = _stop_process_group(
            supervisor,
            interrupt_timeout_s=float(arguments.shutdown_timeout),
        )
        process_summary = {
            'recorder_stopped': recorder_stopped,
            'recorder_returncode': recorder_returncode,
            'bag_metadata': str(bag_metadata),
            'bag_payloads': [str(path) for path in bag_payloads],
            'supervisor_stopped': supervisor_stopped,
            'supervisor_returncode': supervisor_returncode,
        }
        if monitor is not None:
            monitor.destroy_node()
        if rclpy_started:
            rclpy.shutdown()
        if supervisor_log is not None:
            supervisor_log.close()
        if recorder_log is not None:
            recorder_log.close()

    return evidence, outcome, bag_closed_cleanly, {
        **state,
        **process_summary,
    }


def run(argv: Sequence[str] | None = None) -> tuple[int, dict[str, Any]]:
    """Run one task and return a code plus machine-readable verdict."""
    arguments = _arguments(argv)
    os.environ['ROS_DOMAIN_ID'] = str(arguments.ros_domain_id)
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run_directory = arguments.output_root / f'zmanip-e2e-{timestamp}'
    run_directory.mkdir(mode=0o755, parents=True, exist_ok=False)
    event_writer = EventWriter(run_directory / 'events.jsonl')
    started_at = datetime.now(timezone.utc)
    received_signal: int | None = None

    def interrupt(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum
        raise AcceptanceError(f'interrupted by signal {signum}')

    previous_handlers = {
        signum: signal.signal(signum, interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        evidence, outcome, bag_closed, runtime = _run_ros_acceptance(
            arguments,
            run_directory,
            event_writer,
        )
        checks = evidence.checks(bag_closed_cleanly=bag_closed)
        passed = all(checks.values())
        verdict = {
            'schema': 'z_manip.e2e_acceptance.v1',
            'verdict': 'pass' if passed else 'fail',
            'outcome': outcome,
            'started_at_utc': started_at.isoformat(),
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'ros_domain_id': arguments.ros_domain_id,
            'rmw_implementation': os.environ['RMW_IMPLEMENTATION'],
            'fastdds_builtin_transports': os.environ[
                'FASTDDS_BUILTIN_TRANSPORTS'
            ],
            'namespace': arguments.namespace,
            'instruction': evidence.instruction,
            'run_directory': str(run_directory),
            'bag_directory': str(run_directory / 'bag'),
            'pick_two_stage_phase_order_observed': checks[
                'pick_two_stage_phase_order_observed'
            ],
            'checks': checks,
            'evidence': evidence.summary(),
            'runtime': runtime,
        }
        if received_signal is not None:
            verdict['interrupted_by_signal'] = received_signal
        _write_json_atomic(run_directory / 'verdict.json', verdict)
        return (0 if passed else 1), verdict
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        event_writer.close()


def main() -> None:
    """Emit exactly one final JSON verdict on stdout."""
    try:
        returncode, verdict = run()
    except Exception as error:
        verdict = {
            'schema': 'z_manip.e2e_acceptance.v1',
            'verdict': 'error',
            'error': f'{type(error).__name__}: {error}',
        }
        returncode = 2
    print(
        json.dumps(verdict, separators=(',', ':'), sort_keys=True),
        flush=True,
    )
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
