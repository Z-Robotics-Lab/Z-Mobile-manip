"""Focused tests for final visual-search settle validation."""

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from z_manip_task.core import (
    ContinuousMotionQuietWindow,
    RuntimePhase,
    RuntimeSafetyCore,
    VisualSearchUpdate,
)


def _pose_settle_core(*, after_visual_search: bool) -> RuntimeSafetyCore:
    core = RuntimeSafetyCore()
    core.begin('pick the observed target')
    if after_visual_search:
        core.begin_visual_search()
        core.mark_visual_search_complete()
    return core


def _harness(
    *,
    anchor,
    position,
    odom_seen_at,
    target_yaw: float = 0.3,
    measured_yaw: float | None = 0.3,
    linear_speed_mps: float | None = 0.0,
    angular_speed_rps: float | None = 0.0,
    source_stamp_s: float | None = None,
    odom_sequence: int = 6,
    minimum_odom_sequence: int = 1,
    minimum_odom_stamp_s: float = 8.0,
    settle_started_at: float = 9.0,
    stationary_deadline_s: float = 14.0,
    correction_deadline_s: float = 13.2,
    absolute_deadline_s: float = 14.0,
    after_visual_search: bool = True,
):
    class Parameter:
        def __init__(self, value) -> None:
            self.value = value

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    class Search:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                max_planar_drift_m=0.15,
                position_completion_tolerance_m=0.05,
                moving_rebound_reacquire_m=0.10,
                yaw_tolerance_rad=0.0174532925,
                settle_yaw_tolerance_rad=0.0349065850,
                settle_max_linear_speed_mps=0.035,
                settle_max_angular_speed_rps=0.05,
                stationary_wait_timeout_s=4.0,
                stationary_quiet_window_s=0.35,
                stationary_max_odom_gap_s=0.15,
                settle_reacquire_budget_s=4.2,
            )
            self.active = False
            self.attempt = 2
            self.reacquire_calls = []
            self.update_calls = []
            self.next_update = None

        def reacquire_current_target(self, yaw, **kwargs) -> None:
            self.reacquire_calls.append((yaw, kwargs))
            self.active = True

        def update(self, *args, **kwargs) -> VisualSearchUpdate:
            self.update_calls.append((args, kwargs))
            assert self.next_update is not None
            update = self.next_update
            if update.complete or update.timed_out or update.drift_exceeded:
                self.active = False
            return update

    class Harness:
        def __init__(self) -> None:
            self._core = _pose_settle_core(
                after_visual_search=after_visual_search,
            )
            self._pose_settle_until = 10.0
            self._pose_settle_started_at = settle_started_at
            self._pose_settle_last_tick_at = settle_started_at
            self._visual_search_settle_reference = (
                None
                if anchor is None
                else SimpleNamespace(
                    position_anchor_xy=anchor,
                    target_yaw_rad=target_yaw,
                    started_at_s=settle_started_at,
                    stop_started_at_s=settle_started_at,
                    minimum_odom_sequence=minimum_odom_sequence,
                    minimum_odom_stamp_ns=int(round(minimum_odom_stamp_s * 1e9)),
                    correction_deadline_s=correction_deadline_s,
                    absolute_deadline_s=absolute_deadline_s,
                    stationary_deadline_s=stationary_deadline_s,
                    reacquire_count=0,
                )
            )
            self._position_xy = position
            self._yaw = measured_yaw
            self._odom_seen_at = odom_seen_at
            effective_stamp = odom_seen_at if source_stamp_s is None else source_stamp_s
            self._odom_stamp_ns = (
                None if effective_stamp is None else int(round(effective_stamp * 1e9))
            )
            self._odom_sequence = odom_sequence
            self._base_linear_speed_mps = linear_speed_mps
            self._base_angular_speed_rps = angular_speed_rps
            self._base_yaw_rate_rps = 0.0
            self._visual_search = Search()
            self._visual_search_stationarity = ContinuousMotionQuietWindow(
                quiet_window_s=0.35,
                max_odom_gap_s=0.15,
                max_linear_speed_mps=0.035,
                max_angular_speed_rps=0.05,
            )
            if anchor is not None:
                self._visual_search_stationarity.reset(
                    stop_received_at_s=settle_started_at,
                    minimum_odom_sequence=minimum_odom_sequence,
                    minimum_odom_stamp_ns=int(round(minimum_odom_stamp_s * 1e9)),
                )
                if (
                    odom_seen_at is not None
                    and effective_stamp is not None
                    and odom_sequence >= minimum_odom_sequence + 5
                    and effective_stamp - 0.4 > minimum_odom_stamp_s
                    and odom_seen_at - 0.4 > settle_started_at
                    and linear_speed_mps is not None
                    and angular_speed_rps is not None
                    and all(map(math.isfinite, (
                        linear_speed_mps,
                        angular_speed_rps,
                    )))
                ):
                    for index in range(5):
                        offset = -0.4 + 0.1 * index
                        self._visual_search_stationarity.observe(
                            received_at_s=odom_seen_at + offset,
                            odom_sequence=odom_sequence - 4 + index,
                            odom_stamp_ns=int(round(
                                (effective_stamp + offset) * 1e9,
                            )),
                            linear_speed_mps=linear_speed_mps,
                            angular_speed_rps=angular_speed_rps,
                        )
            self._visual_search_error_rad = None
            self._config = SimpleNamespace(
                robot=SimpleNamespace(platform_base_frame='base_link'),
            )
            self._visual_search_active_pub = Publisher()
            self._velocity_pub = Publisher()
            self.zero_commands = 0
            self.actions = []
            self.grounding_requests = 0

        def get_parameter(self, name: str) -> Parameter:
            return Parameter({
                'control_period_s': 0.05,
                'visual_search_odom_timeout_s': 0.5,
                'visual_search_settle_s': 0.75,
            }[name])

        def get_clock(self):
            return SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: object()),
            )

        def _publish_zero(self) -> None:
            self.zero_commands += 1

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

        def _publish_grounding_request(self) -> None:
            self.grounding_requests += 1

    return Harness()


def _feed_stationary(harness, start_s: float, end_s: float) -> None:
    """Feed a continuous 10 Hz post-stop window into one unit harness."""
    sequence = max(
        harness._odom_sequence,
        harness._visual_search_stationarity.last_odom_sequence or 0,
    )
    count = int(round((end_s - start_s) / 0.1))
    for index in range(count + 1):
        stamp_s = start_s + 0.1 * index
        sequence += 1
        harness._odom_sequence = sequence
        harness._odom_stamp_ns = int(round(stamp_s * 1e9))
        harness._odom_seen_at = stamp_s
        harness._base_linear_speed_mps = 0.02
        harness._base_angular_speed_rps = 0.04
        harness._visual_search_stationarity.observe(
            received_at_s=stamp_s,
            odom_sequence=sequence,
            odom_stamp_ns=harness._odom_stamp_ns,
            linear_speed_mps=harness._base_linear_speed_mps,
            angular_speed_rps=harness._base_angular_speed_rps,
        )


def test_ordinary_lookout_settle_does_not_require_visual_search_odometry() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=None,
        position=None,
        odom_seen_at=None,
        measured_yaw=None,
        linear_speed_mps=None,
        angular_speed_rps=None,
        after_visual_search=False,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.GROUNDING
    assert harness.grounding_requests == 1
    assert harness.actions == []
    assert harness._pose_settle_until is None
    assert harness._visual_search_active_pub.messages == []


def test_visual_search_completion_retains_anchor_for_settle_validation() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    class Parameter:
        def __init__(self, value) -> None:
            self.value = value

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    class Search:
        position_anchor_xy = (1.0, 2.0)
        target_yaw_rad = 0.3
        allocated_timeout_s = 8.0
        config = SimpleNamespace(
            max_planar_drift_m=0.15,
            position_completion_tolerance_m=0.05,
            position_hold_timeout_s=4.0,
            stationary_wait_timeout_s=4.0,
            stationary_quiet_window_s=0.35,
            stationary_max_odom_gap_s=0.15,
            settle_reacquire_budget_s=6.0,
        )

        def __init__(self) -> None:
            self.measured_angular_speed_rps = None

        def update(self, *_args, **kwargs) -> VisualSearchUpdate:
            self.measured_angular_speed_rps = kwargs.get(
                'measured_angular_speed_rps',
            )
            return VisualSearchUpdate(0.0, 0.01, complete=True, planar_drift_m=0.04)

    class Harness:
        def __init__(self) -> None:
            self._core = _pose_settle_core(after_visual_search=False)
            self._core.begin_visual_search()
            self._yaw = 0.0
            self._position_xy = (1.04, 2.0)
            self._odom_seen_at = 9.9
            self._visual_search = Search()
            self._visual_search_stationarity = ContinuousMotionQuietWindow(
                quiet_window_s=0.35,
                max_odom_gap_s=0.15,
                max_linear_speed_mps=0.035,
                max_angular_speed_rps=0.05,
            )
            self._visual_search_error_rad = None
            self._visual_search_edge_direction = 1
            self._visual_search_settle_reference = None
            self._pose_settle_until = None
            self._pose_settle_started_at = None
            self._pose_settle_last_tick_at = None
            self._odom_sequence = 5
            self._odom_stamp_ns = 9_900_000_000
            self._base_angular_speed_rps = 0.20
            self._base_yaw_rate_rps = 0.014
            self._config = SimpleNamespace(
                robot=SimpleNamespace(platform_base_frame='base_link'),
            )
            self._visual_search_active_pub = Publisher()
            self._velocity_pub = Publisher()
            self.zero_commands = 0
            self.actions = []

        def get_parameter(self, name: str) -> Parameter:
            return Parameter({
                'control_period_s': 0.05,
                'visual_search_odom_timeout_s': 0.5,
                'visual_search_settle_s': 0.75,
            }[name])

        def get_clock(self):
            return SimpleNamespace(
                now=lambda: SimpleNamespace(to_msg=lambda: object()),
            )

        def _publish_zero(self) -> None:
            self.zero_commands += 1

        def _apply_safety(self, action) -> None:
            self.actions.append(action)

    harness = Harness()

    MobileManipulationRuntime._visual_search_tick(harness, 10.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search_settle_reference.position_anchor_xy == (1.0, 2.0)
    assert harness._visual_search_settle_reference.target_yaw_rad == pytest.approx(0.3)
    assert harness._visual_search_settle_reference.started_at_s == pytest.approx(10.0)
    assert harness._visual_search_settle_reference.minimum_odom_sequence == 5
    assert (
        harness._visual_search_settle_reference.minimum_odom_stamp_ns
        == 9_900_000_000
    )
    assert (
        harness._visual_search_settle_reference.stationary_deadline_s
        == pytest.approx(14.80)
    )
    assert harness._pose_settle_until == pytest.approx(10.75)
    assert harness._pose_settle_started_at == pytest.approx(10.0)
    assert harness._pose_settle_last_tick_at == pytest.approx(10.0)
    assert harness._visual_search.measured_angular_speed_rps == pytest.approx(0.014)
    assert [message.data for message in harness._visual_search_active_pub.messages] == [
        True,
    ]
    assert harness.zero_commands == 1
    assert harness.actions == []


def test_visual_search_settle_reacquires_fresh_drift_beyond_completion_tolerance() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.14, 2.0),
        odom_seen_at=9.8,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search.active
    assert len(harness._visual_search.reacquire_calls) == 1
    yaw, kwargs = harness._visual_search.reacquire_calls[0]
    assert yaw == pytest.approx(0.3)
    assert kwargs['current_position_xy'] == (1.14, 2.0)
    assert kwargs['deadline_s'] == pytest.approx(13.2)
    assert harness.grounding_requests == 0
    assert harness.actions == []
    assert harness._visual_search_settle_reference is not None
    assert harness._visual_search_settle_reference.reacquire_count == 1


def test_visual_search_settle_waits_for_finite_measured_motion_to_stop() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        linear_speed_mps=0.0744,
        angular_speed_rps=0.0929,
        stationary_deadline_s=13.0,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search_settle_reference is not None
    assert harness._pose_settle_started_at == pytest.approx(9.0)
    assert harness._pose_settle_until == pytest.approx(10.0)
    assert harness.grounding_requests == 0
    assert harness.actions == []

    _feed_stationary(harness, 12.0, 12.4)
    MobileManipulationRuntime._finish_pose_settle(harness, 12.5)

    assert harness._core.phase is RuntimePhase.GROUNDING
    assert harness._visual_search_settle_reference is None
    assert harness._pose_settle_started_at is None
    assert harness._pose_settle_last_tick_at is None
    assert harness._pose_settle_until is None
    assert harness.grounding_requests == 1
    assert harness._visual_search_active_pub.messages[-1].data is False
    assert harness.actions == []


def test_visual_search_settle_fails_when_finite_motion_outlives_absolute_deadline() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=12.9,
        source_stamp_s=12.9,
        linear_speed_mps=0.0744,
        angular_speed_rps=0.0929,
        stationary_deadline_s=13.0,
        correction_deadline_s=12.2,
        absolute_deadline_s=13.0,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 13.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'absolute settle deadline expired' in harness._core.failure_reason
    assert 'linear=0.0744m/s' in harness._core.failure_reason
    assert 'angular=0.0929rad/s' in harness._core.failure_reason
    assert harness._visual_search_settle_reference is None
    assert harness._pose_settle_until is None
    assert harness.grounding_requests == 0
    assert len(harness.actions) == 1


def test_visual_search_settle_corrects_rebound_before_waiting_for_stationarity() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        target_yaw=0.3,
        measured_yaw=0.34,
        linear_speed_mps=0.0744,
        angular_speed_rps=0.0929,
        stationary_deadline_s=13.0,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search_settle_reference is not None
    assert harness._visual_search.active
    assert len(harness._visual_search.reacquire_calls) == 1
    assert harness.grounding_requests == 0
    assert harness.actions == []
    _, kwargs = harness._visual_search.reacquire_calls[0]
    assert kwargs['deadline_s'] == pytest.approx(13.2)


def test_visual_search_settle_waits_out_moderate_moving_position_rebound() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.08, 2.0),
        odom_seen_at=9.8,
        linear_speed_mps=0.08,
        angular_speed_rps=0.02,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert not harness._visual_search.active
    assert harness._visual_search.reacquire_calls == []
    assert harness.grounding_requests == 0
    assert harness.actions == []

    _feed_stationary(harness, 10.1, 10.5)
    MobileManipulationRuntime._finish_pose_settle(harness, 10.55)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search.active
    assert len(harness._visual_search.reacquire_calls) == 1
    assert harness.grounding_requests == 0
    assert harness.actions == []


def test_visual_search_settle_reacquires_urgent_moving_position_rebound() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.11, 2.0),
        odom_seen_at=9.8,
        linear_speed_mps=0.08,
        angular_speed_rps=0.02,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search.active
    assert len(harness._visual_search.reacquire_calls) == 1
    assert harness.grounding_requests == 0
    assert harness.actions == []


def test_visual_search_settle_reacquires_rebound_before_fixed_dwell() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.081, 2.0),
        odom_seen_at=9.1,
    )
    harness._visual_search.config.moving_rebound_reacquire_m = 0.08

    assert MobileManipulationRuntime._monitor_visual_search_settle_rebound(
        harness,
        9.2,
    )
    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search.active
    assert len(harness._visual_search.reacquire_calls) == 1
    assert harness._visual_search.planar_drift_m == pytest.approx(0.081)
    assert harness.grounding_requests == 0
    assert harness.actions == []


def test_visual_search_settle_live_rebound_keeps_hard_drift_gate() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.151, 2.0),
        odom_seen_at=9.1,
    )
    harness._visual_search.config.moving_rebound_reacquire_m = 0.08

    assert MobileManipulationRuntime._monitor_visual_search_settle_rebound(
        harness,
        9.2,
    )
    assert harness._core.phase is RuntimePhase.FAILED
    assert not harness._visual_search.active
    assert harness._visual_search.reacquire_calls == []
    assert 'planar drift' in harness._core.failure_reason


def test_visual_search_correction_uses_yaw_rate_not_three_axis_norm() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.081, 2.0),
        odom_seen_at=9.1,
        angular_speed_rps=0.20,
    )
    harness._base_yaw_rate_rps = 0.01
    harness._visual_search.active = True
    harness._visual_search.next_update = VisualSearchUpdate(
        angular_z=0.0,
        error_rad=0.01,
        linear_x=-0.02,
    )

    MobileManipulationRuntime._visual_search_settle_correction_tick(
        harness,
        9.2,
    )

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search.update_calls[-1][1][
        'measured_angular_speed_rps'
    ] == pytest.approx(0.01)
    assert harness._velocity_pub.messages


def test_moderate_moving_rebound_preserves_half_the_correction_budget() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.08, 2.0),
        odom_seen_at=11.11,
        source_stamp_s=11.11,
        linear_speed_mps=0.08,
        angular_speed_rps=0.02,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 11.12)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search.active
    assert len(harness._visual_search.reacquire_calls) == 1
    _, kwargs = harness._visual_search.reacquire_calls[0]
    assert kwargs['deadline_s'] == pytest.approx(13.2)
    assert harness.grounding_requests == 0
    assert harness.actions == []


def test_visual_search_settle_fresh_stationary_sample_wins_at_deadline() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=13.0,
        source_stamp_s=13.0,
        linear_speed_mps=0.02,
        angular_speed_rps=0.04,
        stationary_deadline_s=13.0,
        correction_deadline_s=12.2,
        absolute_deadline_s=13.0,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 13.0)

    assert harness._core.phase is RuntimePhase.GROUNDING
    assert harness.grounding_requests == 1
    assert harness.actions == []


def test_visual_search_settle_rejects_stationary_sample_after_absolute_deadline() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=14.1,
        source_stamp_s=14.1,
        linear_speed_mps=0.02,
        angular_speed_rps=0.04,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 14.15)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'absolute settle deadline expired' in harness._core.failure_reason
    assert harness.grounding_requests == 0
    assert len(harness.actions) == 1


def test_stationary_subdeadline_does_not_preempt_total_recovery_budget() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=13.0,
        source_stamp_s=13.0,
        linear_speed_mps=0.08,
        angular_speed_rps=0.09,
        stationary_deadline_s=13.0,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 13.01)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness.grounding_requests == 0
    assert harness.actions == []

    _feed_stationary(harness, 13.2, 13.6)
    MobileManipulationRuntime._finish_pose_settle(harness, 13.65)

    assert harness._core.phase is RuntimePhase.GROUNDING
    assert harness.grounding_requests == 1
    assert harness.actions == []


def test_visual_search_settle_requires_odometry_strictly_after_stop() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        odom_sequence=4,
        minimum_odom_sequence=4,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'no post-stop odometry sample' in harness._core.failure_reason
    assert harness.grounding_requests == 0


def test_visual_search_settle_rejects_advanced_sequence_with_same_source_stamp() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        source_stamp_s=9.8,
        odom_sequence=5,
        minimum_odom_sequence=4,
        minimum_odom_stamp_s=9.8,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'no post-stop odometry sample' in harness._core.failure_reason
    assert harness.grounding_requests == 0


@pytest.mark.parametrize('source_stamp_s', (9.0, 10.1))
def test_visual_search_settle_requires_fresh_source_timestamp(
    source_stamp_s: float,
) -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        source_stamp_s=source_stamp_s,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'source stamp is stale' in harness._core.failure_reason
    assert harness.grounding_requests == 0


def test_visual_search_settle_rejects_clock_rollback() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.9,
        settle_started_at=10.1,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'settle clock moved backwards' in harness._core.failure_reason
    assert harness.grounding_requests == 0


@pytest.mark.parametrize(
    ('position', 'odom_seen_at', 'reason'),
    (
        (None, 9.8, 'lost odometry pose'),
        ((1.0, 2.0), None, 'lost odometry pose'),
        ((1.0, 2.0), 9.0, 'odometry receipt is stale'),
        ((1.0, 2.0), 10.1, 'odometry receipt is stale'),
    ),
)
def test_visual_search_settle_fails_closed_without_fresh_odometry(
    position,
    odom_seen_at,
    reason: str,
) -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=position,
        odom_seen_at=odom_seen_at,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert reason in harness._core.failure_reason
    assert harness.grounding_requests == 0
    assert len(harness.actions) == 1
    assert harness.actions[0].stop_base
    assert harness.actions[0].cancel_navigation
    assert harness.actions[0].cancel_arm
    assert harness._visual_search_settle_reference is None
    assert harness._pose_settle_until is None


def test_visual_search_settle_fails_closed_beyond_hard_drift_gate() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.151, 2.0),
        odom_seen_at=9.8,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'settle planar drift 0.151m exceeds 0.150m' in harness._core.failure_reason
    assert harness.grounding_requests == 0
    assert harness._visual_search_settle_reference is None


@pytest.mark.parametrize(
    ('linear_speed_mps', 'angular_speed_rps'),
    (
        (0.036, 0.0),
        (0.0, 0.051),
        (0.036, 0.051),
    ),
)
def test_visual_search_settle_fails_closed_while_platform_is_moving(
    linear_speed_mps: float,
    angular_speed_rps: float,
) -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        linear_speed_mps=linear_speed_mps,
        angular_speed_rps=angular_speed_rps,
        stationary_deadline_s=10.0,
        correction_deadline_s=9.8,
        absolute_deadline_s=10.0,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'absolute settle deadline expired' in harness._core.failure_reason
    assert harness.grounding_requests == 0
    assert harness._visual_search_settle_reference is None


@pytest.mark.parametrize(
    ('linear_speed_mps', 'angular_speed_rps'),
    (
        (None, 0.0),
        (0.0, None),
        (float('nan'), 0.0),
        (0.0, float('inf')),
    ),
)
def test_visual_search_settle_requires_finite_measured_speeds(
    linear_speed_mps,
    angular_speed_rps,
) -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        linear_speed_mps=linear_speed_mps,
        angular_speed_rps=angular_speed_rps,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'speed is unavailable or non-finite' in harness._core.failure_reason
    assert harness.grounding_requests == 0
    assert harness._visual_search_settle_reference is None


def test_visual_search_settle_reacquires_retained_target_yaw() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        target_yaw=0.3,
        measured_yaw=0.38,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search_settle_reference is not None
    assert harness._visual_search_settle_reference.stationary_deadline_s == 14.0
    assert harness._visual_search.active
    assert harness._visual_search.attempt == 2
    assert len(harness._visual_search.reacquire_calls) == 1
    yaw, kwargs = harness._visual_search.reacquire_calls[0]
    assert yaw == pytest.approx(0.38)
    assert kwargs['current_position_xy'] == (1.02, 2.0)
    assert kwargs['deadline_s'] == pytest.approx(13.2)
    assert harness.grounding_requests == 0
    assert harness.actions == []


def test_visual_search_settle_correction_restarts_zero_dwell_and_odom_gate() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        target_yaw=0.3,
        measured_yaw=0.38,
        stationary_deadline_s=14.0,
    )
    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)
    harness._visual_search.next_update = VisualSearchUpdate(
        0.0,
        -0.005,
        complete=True,
        planar_drift_m=0.02,
    )
    harness._yaw = 0.305
    harness._odom_sequence += 1
    harness._odom_stamp_ns = 10_390_000_000
    harness._odom_seen_at = 10.39

    MobileManipulationRuntime._visual_search_settle_correction_tick(
        harness,
        10.4,
    )

    reference = harness._visual_search_settle_reference
    assert reference is not None
    assert reference.stationary_deadline_s == pytest.approx(14.0)
    assert reference.minimum_odom_sequence == 7
    assert reference.minimum_odom_stamp_ns == 10_390_000_000
    assert harness._pose_settle_started_at == pytest.approx(10.4)
    assert harness._pose_settle_until == pytest.approx(11.15)
    assert not harness._visual_search.active
    assert harness._visual_search.attempt == 2
    assert harness.zero_commands == 1
    assert harness._visual_search_active_pub.messages[-1].data is True

    harness._yaw = 0.3
    _feed_stationary(harness, 10.7, 11.1)
    MobileManipulationRuntime._finish_pose_settle(harness, 11.15)

    assert harness._core.phase is RuntimePhase.GROUNDING
    assert harness.grounding_requests == 1
    assert harness._visual_search_active_pub.messages[-1].data is False
    assert harness.actions == []


def test_visual_search_settle_rebound_can_reacquire_again_without_new_budget() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        target_yaw=0.3,
        measured_yaw=0.38,
        stationary_deadline_s=14.0,
    )
    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)
    harness._visual_search.active = False
    reference = harness._visual_search_settle_reference
    harness._visual_search_settle_reference = replace(
        reference,
        stop_started_at_s=10.2,
        minimum_odom_sequence=harness._odom_sequence,
        minimum_odom_stamp_ns=harness._odom_stamp_ns,
    )
    harness._visual_search_stationarity.reset(
        stop_received_at_s=10.2,
        minimum_odom_sequence=harness._odom_sequence,
        minimum_odom_stamp_ns=harness._odom_stamp_ns,
    )
    _feed_stationary(harness, 10.5, 10.9)
    harness._pose_settle_started_at = 10.2
    harness._pose_settle_last_tick_at = 10.2
    harness._pose_settle_until = 11.0

    MobileManipulationRuntime._finish_pose_settle(harness, 11.0)

    assert harness._core.phase is RuntimePhase.POSE_SETTLE
    assert harness._visual_search.active
    assert harness._visual_search.attempt == 2
    assert len(harness._visual_search.reacquire_calls) == 2
    assert all(
        call[1]['deadline_s'] == pytest.approx(13.2)
        for call in harness._visual_search.reacquire_calls
    )
    assert harness._visual_search_settle_reference.stationary_deadline_s == 14.0


def test_visual_search_settle_rebound_fails_when_correction_budget_is_spent() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=10.0,
        source_stamp_s=10.0,
        target_yaw=0.3,
        measured_yaw=0.38,
        stationary_deadline_s=10.8,
        correction_deadline_s=10.0,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.FAILED
    assert 'pose correction deadline expired' in harness._core.failure_reason
    assert harness._visual_search.reacquire_calls == []
    assert harness.grounding_requests == 0


def test_visual_search_settle_accepts_parking_rebound_inside_outer_tolerance() -> None:
    pytest.importorskip('rclpy')
    from z_manip_task.node import MobileManipulationRuntime

    harness = _harness(
        anchor=(1.0, 2.0),
        position=(1.02, 2.0),
        odom_seen_at=9.8,
        target_yaw=0.3,
        measured_yaw=0.333,
    )

    MobileManipulationRuntime._finish_pose_settle(harness, 10.0)

    assert harness._core.phase is RuntimePhase.GROUNDING
    assert harness.grounding_requests == 1
    assert harness.actions == []
