"""ROS-independent coarse-navigation policy and recovery budgets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

import numpy as np


class NavPhase(str, Enum):
    """Coarse navigation phase."""

    IDLE = 'idle'
    WAIT_OBSERVATION = 'wait_observation'
    NAVIGATING = 'navigating'
    REACQUIRE = 'reacquire'
    READY = 'ready'
    FAILED = 'failed'


@dataclass(frozen=True)
class NavigationConfig:
    """All policy thresholds, expressed in ROS/simulation time."""

    near_target_depth_m: float = 1.4
    still_speed_mps: float = 0.035
    still_settle_s: float = 0.35
    target_timeout_s: float = 0.55
    observation_wait_timeout_s: float = 12.0
    navigation_timeout_s: float = 90.0
    stall_timeout_s: float = 8.0
    progress_min_net_decrease_m: float = 0.005
    progress_min_slope_mps: float = 0.001
    odometry_timeout_s: float = 0.50
    min_displacement_m: float = 0.10
    max_displacement_m: float = 3.0
    goal_update_threshold_m: float = 0.20
    explicit_goal_tolerance_m: float = 0.25
    explicit_goal_handoff_hysteresis_m: float = 0.08
    max_reacquisitions: int = 2
    max_replans: int = 3

    def __post_init__(self) -> None:
        """Validate finite positive thresholds and bounded retry counts."""
        positive = (
            self.near_target_depth_m,
            self.still_speed_mps,
            self.still_settle_s,
            self.target_timeout_s,
            self.observation_wait_timeout_s,
            self.navigation_timeout_s,
            self.stall_timeout_s,
            self.progress_min_net_decrease_m,
            self.progress_min_slope_mps,
            self.odometry_timeout_s,
            self.min_displacement_m,
            self.max_displacement_m,
            self.goal_update_threshold_m,
            self.explicit_goal_tolerance_m,
            self.explicit_goal_handoff_hysteresis_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('navigation thresholds must be finite and positive')
        if self.min_displacement_m >= self.max_displacement_m:
            raise ValueError('navigation displacement interval is invalid')
        if self.max_reacquisitions < 0 or self.max_replans < 0:
            raise ValueError('navigation retry budgets cannot be negative')


@dataclass(frozen=True, eq=False)
class NavInput:
    """One tick of observed navigation state in the configured map frame."""

    stamp_s: float
    perception_valid: bool
    target_stamp_s: float | None
    target_depth_m: float | None
    base_xy: np.ndarray | None
    target_xy: np.ndarray | None
    suggested_displacement_m: float | None
    base_speed_mps: float
    odom_stamp_s: float | None
    navigation_healthy: bool
    goal_reached: bool
    explicit_goal_xy: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class NavigationTaskRequest:
    """Validated coarse-navigation contract decoded from task status."""

    instruction: str
    task_key: str
    goal_id: str | None
    map_frame: str | None
    map_goal_xy: np.ndarray | None
    map_goal_yaw_rad: float | None
    suggested_displacement_m: float | None
    source: Mapping[str, Any] | None = None

    @property
    def uses_explicit_goal(self) -> bool:
        """Return whether the request carries an immutable map-frame goal."""
        return self.map_goal_xy is not None


def parse_task_navigation_request(
    status: Mapping[str, Any],
) -> NavigationTaskRequest | None:
    """
    Parse one task-status value without depending on ROS message classes.

    A non-``coarse_nav`` status is a valid request to stop navigation and
    therefore returns ``None``.  A present ``work_pose`` is strict: malformed
    explicit goals are rejected rather than silently falling back to the
    target-ray behavior.
    """
    if not isinstance(status, Mapping):
        raise ValueError('task status must be an object')
    if status.get('schema') != 'z_manip.task_status.v1':
        raise ValueError('unsupported task status schema')
    if str(status.get('phase', '')) != 'coarse_nav':
        return None
    instruction = str(status.get('instruction', '')).strip()
    if not instruction:
        raise ValueError('coarse navigation instruction is empty')

    work_pose = status.get('work_pose')
    if work_pose is not None:
        if not isinstance(work_pose, Mapping):
            raise ValueError('work_pose must be an object')
        goal_id = str(work_pose.get('goal_id', '')).strip()
        if not goal_id:
            raise ValueError('work_pose goal_id is empty')
        map_frame_value = work_pose.get('map_frame')
        if not isinstance(map_frame_value, str) or not map_frame_value.strip():
            raise ValueError('work_pose map_frame must be a non-empty string')
        map_frame = map_frame_value.strip()
        goal_xy = np.asarray(work_pose.get('map_goal_xy'), dtype=float)
        if goal_xy.shape != (2,) or not np.all(np.isfinite(goal_xy)):
            raise ValueError('work_pose map_goal_xy must be a finite 2-vector')
        try:
            goal_yaw = float(work_pose.get('map_goal_yaw_rad'))
        except (TypeError, ValueError) as error:
            raise ValueError('work_pose map_goal_yaw_rad must be finite') from error
        if not math.isfinite(goal_yaw):
            raise ValueError('work_pose map_goal_yaw_rad must be finite')
        source = work_pose.get('source')
        if source is not None and not isinstance(source, Mapping):
            raise ValueError('work_pose source must be an object')
        return NavigationTaskRequest(
            instruction=instruction,
            task_key=goal_id,
            goal_id=goal_id,
            map_frame=map_frame,
            map_goal_xy=goal_xy.copy(),
            map_goal_yaw_rad=goal_yaw,
            suggested_displacement_m=None,
            source=None if source is None else dict(source),
        )

    serial = status.get('prospective_serial')
    if serial is None:
        raise ValueError('legacy coarse navigation serial is unavailable')
    try:
        displacement = float(status.get('prospective_base_displacement_m'))
    except (TypeError, ValueError) as error:
        raise ValueError('legacy standoff displacement is unavailable') from error
    if not math.isfinite(displacement) or displacement <= 0.0:
        raise ValueError('legacy standoff displacement is unavailable')
    return NavigationTaskRequest(
        instruction=instruction,
        task_key=f'{instruction}:{serial}',
        goal_id=None,
        map_frame=None,
        map_goal_xy=None,
        map_goal_yaw_rad=None,
        suggested_displacement_m=displacement,
    )


@dataclass(frozen=True, eq=False)
class NavDecision:
    """Commands and state emitted by one policy tick."""

    phase: NavPhase
    waypoint_xy: np.ndarray | None = None
    cancel_navigation: bool = False
    coarse_ready: bool = False
    request_reacquire: bool = False
    reason: str = ''
    replan_count: int = 0
    reacquisition_count: int = 0


class CoarseNavigationCore:
    """Navigate to a bounded work pose and hand off only when settled."""

    def __init__(self, config: NavigationConfig | None = None) -> None:
        """Initialize an idle policy with explicit retry budgets."""
        self.config = config or NavigationConfig()
        self.phase = NavPhase.IDLE
        self.instruction = ''
        self.task_key = ''
        self.failure_reason = ''
        self.replan_count = 0
        self.reacquisition_count = 0
        self.goal_xy: np.ndarray | None = None
        self.uses_explicit_goal = False
        self._goal_mode: str | None = None
        self._planned_target_xy: np.ndarray | None = None
        self._started_at: float | None = None
        self._phase_started_at: float | None = None
        self._still_since: float | None = None
        self._progress_window_started_at_s: float | None = None
        self._progress_samples: deque[tuple[float, float]] = deque()
        self._last_progress_odom_stamp_s: float | None = None
        self._last_fresh_progress_at_s: float | None = None
        self._health_loss_since: float | None = None
        self._goal_reached_armed = False
        self._explicit_handoff_active = False
        self._explicit_resume_pending = False
        self.progress_window_duration_s = 0.0
        self.progress_net_decrease_m: float | None = None
        self.progress_slope_mps: float | None = None
        self.progress_odom_age_s: float | None = None

    def begin(self, instruction: str, task_key: str, *, stamp_s: float) -> None:
        """Start a unique task and wait for its synchronized observation."""
        query = instruction.strip()
        key = task_key.strip()
        stamp = self._finite_time(stamp_s)
        if not query or not key:
            raise ValueError('instruction and task key must be non-empty')
        self.__init__(self.config)
        self.instruction = query
        self.task_key = key
        self.phase = NavPhase.WAIT_OBSERVATION
        self._started_at = stamp
        self._phase_started_at = stamp

    def reset(self) -> None:
        """Return to idle without consuming a recovery budget."""
        self.__init__(self.config)

    def fail(self, reason: str) -> NavDecision:
        """Fail an active adapter contract and issue a navigation cancel."""
        detail = str(reason).strip()
        if not detail:
            raise ValueError('navigation failure reason must be non-empty')
        return self._fail(detail)

    def arm_current_goal(self) -> None:
        """Accept reached evidence after an adapter's causal reset handshake."""
        if self.phase is not NavPhase.NAVIGATING or self.goal_xy is None:
            raise RuntimeError('cannot arm a goal that has not been planned')
        self._goal_reached_armed = True
        self._still_since = None

    @staticmethod
    def _finite_time(value: float) -> float:
        stamp = float(value)
        if not math.isfinite(stamp):
            raise ValueError('navigation time must be finite')
        return stamp

    @staticmethod
    def _xy(value: np.ndarray | None, label: str) -> np.ndarray:
        point = np.asarray(value, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f'{label} must be a finite 2-vector')
        return point

    def _decision(self, **values: object) -> NavDecision:
        return NavDecision(
            phase=self.phase,
            replan_count=self.replan_count,
            reacquisition_count=self.reacquisition_count,
            **values,
        )

    def _fail(self, reason: str) -> NavDecision:
        self.phase = NavPhase.FAILED
        self.failure_reason = reason
        return self._decision(cancel_navigation=True, reason=reason)

    def _reacquire(self, stamp: float, reason: str) -> NavDecision:
        self.reacquisition_count += 1
        if self.reacquisition_count > self.config.max_reacquisitions:
            return self._fail('persistent target tracking reacquisition budget exhausted')
        self.phase = NavPhase.REACQUIRE
        self._phase_started_at = stamp
        self._still_since = None
        return self._decision(
            cancel_navigation=True,
            request_reacquire=True,
            reason=reason,
        )

    def _revoke_explicit_ready(self, stamp: float) -> None:
        """Clear an unproven READY while preserving its immutable goal."""
        if self.phase is not NavPhase.READY or not self.uses_explicit_goal:
            raise RuntimeError('only an explicit READY goal can be revoked')
        self.phase = NavPhase.NAVIGATING
        self._phase_started_at = stamp
        self._still_since = None
        self._goal_reached_armed = False
        self._explicit_handoff_active = False
        self._explicit_resume_pending = True
        self._progress_window_started_at_s = stamp
        self._progress_samples.clear()
        self._last_progress_odom_stamp_s = None
        self._last_fresh_progress_at_s = stamp
        self.progress_window_duration_s = 0.0
        self.progress_net_decrease_m = None
        self.progress_slope_mps = None
        self.progress_odom_age_s = None

    def _wait_for_complete_observation(
        self,
        stamp: float,
        reason: str,
        *,
        allow_reacquire: bool,
    ) -> NavDecision:
        if stamp - float(self._phase_started_at) > self.config.observation_wait_timeout_s:
            detail = f'{reason} timed out'
            if not allow_reacquire:
                return self._fail(detail)
            return self._reacquire(stamp, detail)
        return self._decision(reason=reason)

    def _plan(self, value: NavInput, stamp: float, *, is_replan: bool) -> NavDecision:
        base = self._xy(value.base_xy, 'base position')
        if value.explicit_goal_xy is not None:
            proposed = self._xy(value.explicit_goal_xy, 'explicit map goal').copy()
            target = None
            explicit = True
        else:
            target = self._xy(value.target_xy, 'target position')
            try:
                displacement = float(value.suggested_displacement_m)
            except (TypeError, ValueError):
                return self._fail('reachability standoff displacement is unavailable')
            if not math.isfinite(displacement) or displacement <= 0.0:
                return self._fail('reachability standoff displacement is unavailable')
            direction = target - base
            distance = float(np.linalg.norm(direction))
            if distance < 1e-4:
                return self._fail('observed target direction is degenerate')
            travel = min(max(displacement, self.config.min_displacement_m),
                         self.config.max_displacement_m, distance)
            proposed = base + travel * direction / distance
            explicit = False
        mode = 'explicit' if explicit else 'legacy'
        if self._goal_mode is not None and mode != self._goal_mode:
            return self._fail('navigation goal mode changed without a new task key')
        if is_replan:
            self.replan_count += 1
            if self.replan_count > self.config.max_replans:
                return self._fail('coarse navigation replan budget exhausted')
        self.goal_xy = proposed
        self.uses_explicit_goal = explicit
        self._goal_mode = mode
        self._planned_target_xy = None if target is None else target.copy()
        self.phase = NavPhase.NAVIGATING
        self._phase_started_at = stamp
        self._reset_progress_window(
            value,
            stamp,
            float(np.linalg.norm(proposed - base)),
        )
        self._still_since = None
        # A latched success from the previous waypoint cannot complete this
        # goal.  Observe at least one not-reached tick after publishing before
        # accepting the next goal-reached edge.
        self._goal_reached_armed = False
        self._explicit_handoff_active = False
        self._explicit_resume_pending = False
        return self._decision(waypoint_xy=proposed.copy())

    def _odom_stamp_if_fresh(self, value: NavInput, stamp: float) -> float | None:
        """Return a synchronized odometry stamp, never a cached timer sample."""
        try:
            odom_stamp = float(value.odom_stamp_s)
        except (TypeError, ValueError):
            self.progress_odom_age_s = None
            return None
        if not math.isfinite(odom_stamp):
            self.progress_odom_age_s = None
            return None
        age = stamp - odom_stamp
        self.progress_odom_age_s = age
        if age < -1e-6 or age > self.config.odometry_timeout_s:
            return None
        return odom_stamp

    def _reset_progress_window(
        self,
        value: NavInput,
        stamp: float,
        goal_distance_m: float,
    ) -> None:
        """Start one immutable progress window for a dispatched waypoint."""
        self._progress_window_started_at_s = stamp
        self._progress_samples.clear()
        self._last_progress_odom_stamp_s = None
        self._last_fresh_progress_at_s = None
        self.progress_window_duration_s = 0.0
        self.progress_net_decrease_m = None
        self.progress_slope_mps = None
        odom_stamp = self._odom_stamp_if_fresh(value, stamp)
        if odom_stamp is not None:
            self._progress_samples.append((odom_stamp, goal_distance_m))
            self._last_progress_odom_stamp_s = odom_stamp
            self._last_fresh_progress_at_s = stamp

    def _progress_window_metrics(
        self,
    ) -> tuple[float, float, float] | None:
        """Measure robust net decrease and least-squares slope over one window."""
        if len(self._progress_samples) < 2:
            return None
        end_stamp = self._progress_samples[-1][0]
        cutoff = end_stamp - self.config.stall_timeout_s
        if self._progress_samples[0][0] > cutoff:
            return None

        samples = list(self._progress_samples)
        if samples[0][0] < cutoff:
            before_stamp, before_distance = samples[0]
            after_stamp, after_distance = samples[1]
            if after_stamp <= before_stamp:
                return None
            fraction = (cutoff - before_stamp) / (after_stamp - before_stamp)
            boundary_distance = before_distance + fraction * (
                after_distance - before_distance
            )
            samples[0] = (cutoff, boundary_distance)

        stamps = np.asarray([item[0] for item in samples], dtype=float)
        distances = np.asarray([item[1] for item in samples], dtype=float)
        centered = stamps - float(np.mean(stamps))
        denominator = float(np.dot(centered, centered))
        if denominator <= 0.0:
            return None
        progress_slope = -float(
            np.dot(centered, distances - float(np.mean(distances))) / denominator,
        )

        # One-eighth-window endpoint bands suppress the measured gait cycle
        # without hiding a sustained circle or retreat.  The one-second cap
        # keeps the same policy responsive for longer deployment windows.
        endpoint_band_s = min(1.0, self.config.stall_timeout_s / 8.0)
        start_distances = distances[stamps <= cutoff + endpoint_band_s]
        end_distances = distances[stamps >= end_stamp - endpoint_band_s]
        if start_distances.size == 0 or end_distances.size == 0:
            return None
        net_decrease = float(
            np.median(start_distances) - np.median(end_distances),
        )
        return end_stamp - cutoff, net_decrease, progress_slope

    def _record_goal_progress(
        self,
        value: NavInput,
        stamp: float,
        goal_distance_m: float,
    ) -> bool | None:
        """Return whether a complete fresh-odometry window shows progress."""
        odom_stamp = self._odom_stamp_if_fresh(value, stamp)
        if odom_stamp is None:
            return None
        previous_stamp = self._last_progress_odom_stamp_s
        if previous_stamp is not None and odom_stamp <= previous_stamp:
            return None
        self._last_progress_odom_stamp_s = odom_stamp
        self._last_fresh_progress_at_s = stamp
        self._progress_samples.append((odom_stamp, goal_distance_m))

        cutoff = odom_stamp - self.config.stall_timeout_s
        while (
            len(self._progress_samples) >= 2
            and self._progress_samples[1][0] <= cutoff
        ):
            self._progress_samples.popleft()

        metrics = self._progress_window_metrics()
        if metrics is None:
            if self._progress_samples:
                self.progress_window_duration_s = max(
                    0.0,
                    odom_stamp - self._progress_samples[0][0],
                )
            return None
        duration_s, net_decrease_m, slope_mps = metrics
        self.progress_window_duration_s = duration_s
        self.progress_net_decrease_m = net_decrease_m
        self.progress_slope_mps = slope_mps
        return (
            net_decrease_m >= self.config.progress_min_net_decrease_m
            and slope_mps >= self.config.progress_min_slope_mps
        )

    def _stale_odometry_timeout(self, stamp: float) -> bool:
        """Bound missing fresh odometry by the unchanged stall window."""
        started_at = (
            self._last_fresh_progress_at_s
            if self._last_fresh_progress_at_s is not None
            else self._progress_window_started_at_s
        )
        return (
            started_at is not None
            and stamp - started_at > self.config.stall_timeout_s
        )

    def _near_handoff(self, value: NavInput, stamp: float) -> NavDecision | None:
        try:
            depth = float(value.target_depth_m)
        except (TypeError, ValueError):
            return self._fail('observed target depth is invalid')
        if not math.isfinite(depth) or depth <= 0.0:
            return self._fail('observed target depth is invalid')
        if depth > self.config.near_target_depth_m:
            self._still_since = None
            return None
        if value.base_speed_mps > self.config.still_speed_mps:
            self._still_since = None
            return self._decision(cancel_navigation=True, reason='near target; waiting for stop')
        if self._still_since is None:
            self._still_since = stamp
        if stamp - self._still_since < self.config.still_settle_s:
            return self._decision(cancel_navigation=True, reason='settling before visual handoff')
        self.phase = NavPhase.READY
        return self._decision(cancel_navigation=True, coarse_ready=True)

    def _explicit_goal_handoff(
        self,
        value: NavInput,
        stamp: float,
        goal_distance_m: float,
    ) -> NavDecision | None:
        """Hand an approximate work pose to visual servo using measured XY."""
        entry_tolerance = self.config.explicit_goal_tolerance_m
        exit_tolerance = (
            entry_tolerance + self.config.explicit_goal_handoff_hysteresis_m
        )
        if not self._explicit_handoff_active and goal_distance_m <= entry_tolerance:
            self._explicit_handoff_active = True
        if self._explicit_handoff_active and goal_distance_m > exit_tolerance:
            self._explicit_handoff_active = False
            self._still_since = None
        if not self._explicit_handoff_active:
            self._still_since = None
            if value.goal_reached:
                return self._decision(
                    reason='goal_reached ignored outside explicit XY tolerance',
                )
            return None
        if value.base_speed_mps > self.config.still_speed_mps:
            self._still_since = None
            return self._decision(
                cancel_navigation=True,
                reason='within coarse work-pose tolerance; waiting for stop',
            )
        if self._still_since is None:
            self._still_since = stamp
        if stamp - self._still_since < self.config.still_settle_s:
            return self._decision(
                cancel_navigation=True,
                reason='settling inside coarse work-pose tolerance',
            )
        self.phase = NavPhase.READY
        return self._decision(cancel_navigation=True, coarse_ready=True)

    def update(self, value: NavInput) -> NavDecision:
        """Advance navigation, recovery, progress, and near-field gates."""
        if self.phase in (NavPhase.IDLE, NavPhase.FAILED):
            return self._decision()
        if self.phase is NavPhase.READY and not self.uses_explicit_goal:
            return self._decision()
        stamp = self._finite_time(value.stamp_s)
        assert self._started_at is not None
        assert self._phase_started_at is not None
        if stamp < self._phase_started_at:
            return self._fail('navigation time moved backwards')
        if stamp - self._started_at > self.config.navigation_timeout_s:
            return self._fail('coarse navigation timeout')
        if not value.navigation_healthy:
            if self.phase is NavPhase.READY:
                self._revoke_explicit_ready(stamp)
            if self._health_loss_since is None:
                self._health_loss_since = stamp
                self.replan_count += 1
            if self.replan_count > self.config.max_replans:
                return self._fail('navigation health recovery budget exhausted')
            if stamp - self._health_loss_since > self.config.stall_timeout_s:
                return self._fail('navigation health did not recover before timeout')
            return self._decision(
                cancel_navigation=True,
                reason='navigation unhealthy; holding for recovery',
            )
        health_recovered = self._health_loss_since is not None
        self._health_loss_since = None

        explicit_requested = (
            self.uses_explicit_goal or value.explicit_goal_xy is not None
        )
        if not explicit_requested:
            target_fresh = (
                value.perception_valid
                and value.target_stamp_s is not None
                and math.isfinite(float(value.target_stamp_s))
                and 0.0 <= stamp - float(value.target_stamp_s) <= self.config.target_timeout_s
            )
            if not target_fresh:
                if self.phase is NavPhase.WAIT_OBSERVATION:
                    if stamp - self._phase_started_at > self.config.observation_wait_timeout_s:
                        return self._reacquire(stamp, 'initial target observation timed out')
                    return self._decision(reason='waiting for fresh target observation')
                if self.phase is NavPhase.REACQUIRE:
                    if stamp - self._phase_started_at > self.config.observation_wait_timeout_s:
                        return self._reacquire(stamp, 'target reacquisition timed out')
                    return self._decision(cancel_navigation=True, reason='reacquiring target')
                return self._reacquire(stamp, 'persistent target tracking lost')

        if not math.isfinite(value.base_speed_mps) or value.base_speed_mps < 0.0:
            if self.phase in (NavPhase.WAIT_OBSERVATION, NavPhase.REACQUIRE):
                return self._wait_for_complete_observation(
                    stamp, 'waiting for finite SLAM velocity',
                    allow_reacquire=not explicit_requested,
                )
            return self._fail('base speed is invalid')
        odom_stamp = self._odom_stamp_if_fresh(value, stamp)
        if odom_stamp is None:
            if self.phase is NavPhase.READY:
                self.replan_count += 1
                if self.replan_count > self.config.max_replans:
                    return self._fail(
                        'explicit READY odometry recovery budget exhausted',
                    )
                self._revoke_explicit_ready(stamp)
                return self._decision(
                    cancel_navigation=True,
                    reason='explicit READY revoked while odometry is stale',
                )
            if self.phase in (NavPhase.WAIT_OBSERVATION, NavPhase.REACQUIRE):
                return self._wait_for_complete_observation(
                    stamp, 'waiting for fresh SLAM odometry',
                    allow_reacquire=not explicit_requested,
                )
            if self._stale_odometry_timeout(stamp):
                return self._fail(
                    'fresh SLAM odometry unavailable for progress window',
                )
            return self._decision(reason='waiting for fresh SLAM odometry')
        try:
            self._xy(value.base_xy, 'base position')
            if value.explicit_goal_xy is not None:
                self._xy(value.explicit_goal_xy, 'explicit map goal')
            else:
                self._xy(value.target_xy, 'target position')
                displacement = float(value.suggested_displacement_m)
                if not math.isfinite(displacement) or displacement <= 0.0:
                    raise ValueError('standoff displacement is invalid')
        except (TypeError, ValueError):
            if self.phase in (NavPhase.WAIT_OBSERVATION, NavPhase.REACQUIRE):
                return self._wait_for_complete_observation(
                    stamp, 'waiting for complete navigation observation',
                    allow_reacquire=not explicit_requested,
                )
            return self._fail('navigation observation became incomplete')

        if self._goal_mode is not None:
            if self.uses_explicit_goal:
                try:
                    explicit_goal = self._xy(
                        value.explicit_goal_xy, 'explicit map goal',
                    )
                except (TypeError, ValueError):
                    return self._fail('active explicit map goal is unavailable')
                if not np.array_equal(explicit_goal, self.goal_xy):
                    return self._fail(
                        'explicit map goal changed without a new task key',
                    )
            elif value.explicit_goal_xy is not None:
                return self._fail('navigation goal mode changed without a new task key')

        if not explicit_requested:
            handoff = self._near_handoff(value, stamp)
            if handoff is not None:
                return handoff
        if self.phase in (NavPhase.WAIT_OBSERVATION, NavPhase.REACQUIRE):
            return self._plan(value, stamp, is_replan=self.phase is NavPhase.REACQUIRE)
        if self._explicit_resume_pending:
            return self._plan(value, stamp, is_replan=False)
        if health_recovered:
            return self._plan(value, stamp, is_replan=False)

        assert self.phase in (NavPhase.NAVIGATING, NavPhase.READY)
        assert self.goal_xy is not None
        base = self._xy(value.base_xy, 'base position')
        if self.uses_explicit_goal:
            target = None
        else:
            target = self._xy(value.target_xy, 'target position')
        goal_distance = float(np.linalg.norm(self.goal_xy - base))
        if self.phase is NavPhase.READY:
            exit_tolerance = (
                self.config.explicit_goal_tolerance_m
                + self.config.explicit_goal_handoff_hysteresis_m
            )
            if goal_distance <= exit_tolerance:
                return self._decision(
                    cancel_navigation=True,
                    coarse_ready=True,
                    reason='coarse work pose remains inside handoff hysteresis',
                )
            return self._plan(value, stamp, is_replan=True)
        if self.uses_explicit_goal:
            handoff = self._explicit_goal_handoff(value, stamp, goal_distance)
            if handoff is not None:
                return handoff
        progress = self._record_goal_progress(value, stamp, goal_distance)
        if progress is False:
            return self._plan(value, stamp, is_replan=True)
        if value.goal_reached and not self.uses_explicit_goal:
            return self._plan(value, stamp, is_replan=True)

        if not self.uses_explicit_goal:
            assert target is not None
            assert self._planned_target_xy is not None
            if (
                np.linalg.norm(target - self._planned_target_xy)
                > self.config.goal_update_threshold_m
            ):
                return self._plan(value, stamp, is_replan=True)
        return self._decision()
