"""ROS-independent coarse navigation policy tests."""

import math

import numpy as np
import pytest

from z_manip_navigation.core import (
    CoarseNavigationCore,
    NavigationConfig,
    NavInput,
    NavPhase,
    parse_task_navigation_request,
)


def _input(stamp, **updates):
    values = {
        'stamp_s': stamp,
        'perception_valid': True,
        'target_stamp_s': stamp,
        'target_depth_m': 2.0,
        'base_xy': np.array([0.0, 0.0]),
        'target_xy': np.array([3.0, 0.0]),
        'suggested_displacement_m': 1.2,
        'base_speed_mps': 0.2,
        'odom_stamp_s': stamp,
        'navigation_healthy': True,
        'goal_reached': False,
    }
    values.update(updates)
    return NavInput(**values)


def test_waypoint_uses_observed_map_ray_and_standoff_displacement():
    core = CoarseNavigationCore()
    core.begin('pick item', 'task:1', stamp_s=0.0)
    decision = core.update(_input(0.1, target_xy=np.array([3.0, 4.0])))
    np.testing.assert_allclose(decision.waypoint_xy, [0.72, 0.96])
    assert core.phase is NavPhase.NAVIGATING


def test_explicit_work_pose_is_published_exactly_without_ray_reconstruction():
    core = CoarseNavigationCore()
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    decision = core.update(_input(
        0.1,
        target_xy=np.array([-20.0, 8.0]),
        suggested_displacement_m=None,
        explicit_goal_xy=np.array([3.25, -1.75]),
    ))
    np.testing.assert_array_equal(decision.waypoint_xy, [3.25, -1.75])
    assert core.uses_explicit_goal


def test_explicit_work_pose_near_depth_cannot_bypass_first_waypoint():
    core = CoarseNavigationCore(NavigationConfig(
        near_target_depth_m=1.5, still_settle_s=0.3,
    ))
    core.begin('pick item', 'work-pose-near', stamp_s=0.0)
    goal = np.array([0.8, -0.25])
    first = core.update(_input(
        0.1,
        target_depth_m=0.4,
        base_speed_mps=0.0,
        goal_reached=True,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    np.testing.assert_array_equal(first.waypoint_xy, goal)
    assert not first.cancel_navigation
    assert not first.coarse_ready
    assert core.phase is NavPhase.NAVIGATING


def test_explicit_work_pose_navigation_ignores_stale_perception():
    core = CoarseNavigationCore()
    core.begin('pick item', 'work-pose-frozen', stamp_s=0.0)
    goal = np.array([1.1, -0.35])
    first = core.update(_input(
        0.1,
        perception_valid=False,
        target_stamp_s=None,
        target_depth_m=None,
        target_xy=None,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    np.testing.assert_array_equal(first.waypoint_xy, goal)
    assert not first.request_reacquire
    tracking_lost = core.update(_input(
        1.0,
        perception_valid=False,
        target_stamp_s=None,
        target_depth_m=None,
        target_xy=None,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert not tracking_lost.request_reacquire
    assert core.reacquisition_count == 0
    assert core.phase is NavPhase.NAVIGATING


def test_explicit_work_pose_uses_measured_xy_and_continuous_stillness():
    core = CoarseNavigationCore(NavigationConfig(still_settle_s=0.3))
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    goal = np.array([0.9, -0.4])
    core.update(_input(
        0.1, target_depth_m=0.2, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))

    # The local planner may never assert goal_reached. Measured map-frame XY
    # enters the coarse handoff region and commands a stop instead.
    not_reached = core.update(_input(
        0.5, target_depth_m=0.2, base_xy=np.array([0.82, -0.4]),
        base_speed_mps=0.0,
        goal_reached=False, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert not_reached.cancel_navigation and not not_reached.coarse_ready
    reached = core.update(_input(
        0.6, target_depth_m=0.2, base_xy=np.array([0.82, -0.4]),
        base_speed_mps=0.0,
        goal_reached=True, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert reached.cancel_navigation and not reached.coarse_ready

    # Any movement after goal_reached restarts the continuous settle window.
    moving = core.update(_input(
        0.75, target_depth_m=5.0, base_xy=np.array([0.82, -0.4]),
        base_speed_mps=0.1,
        goal_reached=True, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert moving.cancel_navigation and not moving.coarse_ready
    settling = core.update(_input(
        0.8, target_depth_m=None, base_xy=np.array([0.82, -0.4]),
        base_speed_mps=0.0,
        goal_reached=True, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert settling.cancel_navigation and not settling.coarse_ready
    ready = core.update(_input(
        1.11, target_depth_m=None, base_xy=np.array([0.82, -0.4]),
        base_speed_mps=0.0,
        goal_reached=True, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert ready.cancel_navigation and ready.coarse_ready
    assert core.phase is NavPhase.READY


def test_latched_reached_cannot_complete_new_explicit_waypoint():
    core = CoarseNavigationCore(NavigationConfig(still_settle_s=0.1))
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    goal = np.array([0.9, -0.4])
    core.update(_input(
        0.1, goal_reached=True, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    stale = core.update(_input(
        0.3, base_speed_mps=0.0, goal_reached=True,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert not stale.cancel_navigation
    assert not stale.coarse_ready
    assert core.phase is NavPhase.NAVIGATING


def test_causal_adapter_reset_can_arm_fast_reached_evidence():
    core = CoarseNavigationCore(NavigationConfig(still_settle_s=0.1))
    core.begin('pick item', 'work-pose-fast', stamp_s=0.0)
    goal = np.array([0.9, -0.4])
    core.update(_input(
        0.1,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    core.arm_current_goal()

    reached = core.update(_input(
        0.11,
        base_xy=np.array([0.82, -0.4]),
        base_speed_mps=0.0,
        goal_reached=True,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert not reached.coarse_ready
    ready = core.update(_input(
        0.22,
        base_xy=np.array([0.82, -0.4]),
        base_speed_mps=0.0,
        goal_reached=True,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert ready.coarse_ready
    assert ready.cancel_navigation


def test_explicit_goal_reached_far_from_xy_cannot_become_ready():
    core = CoarseNavigationCore(NavigationConfig(
        explicit_goal_tolerance_m=0.15,
        still_settle_s=0.1,
    ))
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    goal = np.array([1.0, 0.0])
    core.update(_input(
        0.1, explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    core.update(_input(
        0.2, base_xy=np.array([0.0, 0.0]), goal_reached=False,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    false_positive = core.update(_input(
        0.3, base_xy=np.array([0.0, 0.0]), base_speed_mps=0.0,
        goal_reached=True, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert not false_positive.cancel_navigation
    assert not false_positive.coarse_ready
    assert 'XY tolerance' in false_positive.reason
    still_far = core.update(_input(
        0.6, base_xy=np.array([0.0, 0.0]), base_speed_mps=0.0,
        goal_reached=True, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert not still_far.coarse_ready
    assert core.phase is NavPhase.NAVIGATING


def test_explicit_handoff_latches_during_bounded_braking_drift():
    core = CoarseNavigationCore(NavigationConfig(
        explicit_goal_tolerance_m=0.25,
        explicit_goal_handoff_hysteresis_m=0.08,
        still_settle_s=0.1,
    ))
    core.begin('pick item', 'work-pose-braking', stamp_s=0.0)
    goal = np.array([1.0, 0.0])
    core.update(_input(
        0.1, base_xy=np.array([0.0, 0.0]), explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))

    braking = core.update(_input(
        0.2, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.08,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert braking.cancel_navigation and not braking.coarse_ready
    drifted = core.update(_input(
        0.3, base_xy=np.array([0.705, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert drifted.cancel_navigation and not drifted.coarse_ready
    ready = core.update(_input(
        0.41, base_xy=np.array([0.705, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert ready.cancel_navigation and ready.coarse_ready


def test_explicit_ready_replans_same_goal_after_hysteresis_exit():
    core = CoarseNavigationCore(NavigationConfig(
        explicit_goal_tolerance_m=0.25,
        explicit_goal_handoff_hysteresis_m=0.08,
        still_settle_s=0.1,
    ))
    core.begin('pick item', 'work-pose-rebound', stamp_s=0.0)
    goal = np.array([1.0, 0.0])
    core.update(_input(
        0.1, base_xy=np.array([0.0, 0.0]), explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    core.update(_input(
        0.2, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    ready = core.update(_input(
        0.31, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert ready.coarse_ready

    boundary = core.update(_input(
        0.4, base_xy=np.array([0.67, 0.0]), base_speed_mps=0.08,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert boundary.coarse_ready
    assert boundary.waypoint_xy is None
    assert core.phase is NavPhase.READY

    rebound = core.update(_input(
        0.45, base_xy=np.array([0.669, 0.0]), base_speed_mps=0.08,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    np.testing.assert_array_equal(rebound.waypoint_xy, goal)
    assert not rebound.coarse_ready
    assert core.phase is NavPhase.NAVIGATING
    assert core.task_key == 'work-pose-rebound'
    assert core.replan_count == 1


@pytest.mark.parametrize('loss', ('health', 'odometry'))
def test_explicit_ready_revokes_until_navigation_evidence_recovers(loss):
    core = CoarseNavigationCore(NavigationConfig(still_settle_s=0.1))
    core.begin('pick item', f'work-pose-{loss}', stamp_s=0.0)
    goal = np.array([1.0, 0.0])
    core.update(_input(
        0.1, base_xy=np.array([0.0, 0.0]), explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    core.update(_input(
        0.2, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert core.update(_input(
        0.31, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    )).coarse_ready

    changes = (
        {'navigation_healthy': False}
        if loss == 'health'
        else {'odom_stamp_s': 0.0}
    )
    revoked = core.update(_input(
        0.9, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
        **changes,
    ))
    assert revoked.cancel_navigation
    assert not revoked.coarse_ready
    assert core.phase is NavPhase.NAVIGATING
    assert core.replan_count == 1

    resumed = core.update(_input(
        1.0, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    np.testing.assert_array_equal(resumed.waypoint_xy, goal)
    assert core.task_key == f'work-pose-{loss}'
    assert core.replan_count == 1


def test_explicit_ready_stale_odometry_recovery_is_bounded():
    core = CoarseNavigationCore(NavigationConfig(
        still_settle_s=0.1,
        max_replans=0,
    ))
    core.begin('pick item', 'work-pose-stale', stamp_s=0.0)
    goal = np.array([1.0, 0.0])
    core.update(_input(
        0.1, explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    core.update(_input(
        0.2, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    assert core.update(_input(
        0.31, base_xy=np.array([0.76, 0.0]), base_speed_mps=0.0,
        explicit_goal_xy=goal, suggested_displacement_m=None,
    )).coarse_ready

    failed = core.update(_input(
        0.9, odom_stamp_s=0.0, base_xy=np.array([0.76, 0.0]),
        base_speed_mps=0.0, explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert failed.cancel_navigation
    assert core.phase is NavPhase.FAILED
    assert 'budget exhausted' in failed.reason


def test_explicit_work_pose_does_not_follow_target_map_jitter():
    core = CoarseNavigationCore()
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    goal = np.array([0.9, -0.4])
    core.update(_input(0.1, explicit_goal_xy=goal, suggested_displacement_m=None))
    decision = core.update(_input(
        0.2,
        target_xy=np.array([8.0, 5.0]),
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert decision.waypoint_xy is None
    np.testing.assert_array_equal(core.goal_xy, goal)


def test_explicit_work_pose_stall_republishes_same_goal_with_bounded_budget():
    core = CoarseNavigationCore(NavigationConfig(
        stall_timeout_s=0.3, max_replans=1,
    ))
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    goal = np.array([0.9, -0.4])
    core.update(_input(0.1, explicit_goal_xy=goal, suggested_displacement_m=None))
    replanned = core.update(_input(
        0.41, explicit_goal_xy=goal, suggested_displacement_m=None,
    ))
    np.testing.assert_array_equal(replanned.waypoint_xy, goal)
    assert replanned.replan_count == 1


def test_bag_compressed_slow_approach_is_not_a_false_stall():
    """Regression from the 2026-07-14 office MCAP map/odom trajectory."""
    core = CoarseNavigationCore()
    core.begin('pick mustard', 'work-bag-regression', stamp_s=0.0)
    goal = np.array([0.5810558062220369, -0.20119651071002764])
    samples = (
        (0.03, -0.3114, 0.0331),
        (5.03, 0.0369, 0.0284),
        (10.03, 0.1506, 0.0027),
        (15.03, 0.1896, -0.0209),
        (20.03, 0.2196, -0.0392),
        (25.03, 0.2968, -0.0659),
        (30.03, 0.3226, -0.0778),
        (35.03, 0.3320, -0.0781),
        (40.03, 0.3850, -0.0969),
        (45.03, 0.3855, -0.1023),
        (48.07, 0.4027, -0.1072),
    )
    first_time, first_x, first_y = samples[0]
    first = core.update(_input(
        first_time,
        base_xy=np.array([first_x, first_y]),
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    np.testing.assert_array_equal(first.waypoint_xy, goal)

    for stamp, x, y in samples[1:]:
        decision = core.update(_input(
            stamp,
            base_xy=np.array([x, y]),
            explicit_goal_xy=goal,
            suggested_displacement_m=None,
        ))
        assert decision.waypoint_xy is None
        assert core.phase is NavPhase.NAVIGATING

    assert core.replan_count == 0
    assert core.progress_net_decrease_m >= 0.005
    assert core.progress_slope_mps >= 0.001
    assert np.linalg.norm(goal - samples[-1][1:]) > 0.20


@pytest.mark.parametrize('motion', ('stationary_noise', 'orbit', 'away'))
def test_fixed_progress_window_rejects_non_approach_motion(motion):
    core = CoarseNavigationCore(NavigationConfig(
        stall_timeout_s=2.0,
        max_replans=1,
    ))
    core.begin('pick item', f'work-{motion}', stamp_s=0.0)
    goal = np.array([0.0, 0.0])
    times = np.linspace(0.1, 2.1, 21)
    decisions = []
    for index, stamp in enumerate(times):
        if motion == 'stationary_noise':
            x = 1.0 + (0.001 if index % 2 else -0.001)
            y = 0.001 if index % 4 < 2 else -0.001
        elif motion == 'orbit':
            angle = 0.8 * float(stamp)
            x, y = math.cos(angle), math.sin(angle)
        else:
            x, y = 1.0 + 0.02 * float(stamp), 0.0
        decisions.append(core.update(_input(
            float(stamp),
            base_xy=np.array([x, y]),
            explicit_goal_xy=goal,
            suggested_displacement_m=None,
        )))

    assert decisions[-1].waypoint_xy is not None
    assert decisions[-1].replan_count == 1


def test_stale_odometry_fails_without_replanning_from_cached_pose():
    core = CoarseNavigationCore(NavigationConfig(
        stall_timeout_s=0.3,
        odometry_timeout_s=0.05,
        max_replans=3,
    ))
    core.begin('pick item', 'work-stale-odom', stamp_s=0.0)
    goal = np.array([0.9, -0.4])
    core.update(_input(
        0.1,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    waiting = core.update(_input(
        0.2,
        odom_stamp_s=0.1,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert waiting.waypoint_xy is None
    assert 'fresh SLAM odometry' in waiting.reason
    failed = core.update(_input(
        0.41,
        odom_stamp_s=0.1,
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert failed.cancel_navigation
    assert failed.waypoint_xy is None
    assert core.phase is NavPhase.FAILED
    assert core.replan_count == 0


def test_explicit_work_pose_cannot_change_without_new_task_key():
    core = CoarseNavigationCore()
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    core.update(_input(
        0.1, explicit_goal_xy=np.array([0.9, -0.4]),
        suggested_displacement_m=None,
    ))
    rejected = core.update(_input(
        0.2, target_depth_m=1.0, base_speed_mps=0.0,
        explicit_goal_xy=np.array([0.8, -0.4]), suggested_displacement_m=None,
    ))
    assert rejected.cancel_navigation
    assert core.phase is NavPhase.FAILED
    assert 'new task key' in rejected.reason


def test_active_explicit_work_pose_cannot_fall_back_to_legacy_mode():
    core = CoarseNavigationCore()
    core.begin('pick item', 'work-pose-7', stamp_s=0.0)
    core.update(_input(
        0.1, explicit_goal_xy=np.array([0.9, -0.4]),
        suggested_displacement_m=None,
    ))
    rejected = core.update(_input(
        0.2, explicit_goal_xy=None, suggested_displacement_m=0.8,
    ))
    assert rejected.cancel_navigation
    assert core.phase is NavPhase.FAILED
    assert 'explicit map goal is unavailable' in rejected.reason


def test_near_handoff_requires_cancel_and_continuous_stillness():
    core = CoarseNavigationCore(NavigationConfig(
        near_target_depth_m=1.5, still_settle_s=0.3,
    ))
    core.begin('pick', 'task:1', stamp_s=0.0)
    moving = core.update(_input(0.1, target_depth_m=1.4, base_speed_mps=0.1))
    assert moving.cancel_navigation and not moving.coarse_ready
    settling = core.update(_input(0.2, target_depth_m=1.4, base_speed_mps=0.0))
    assert settling.cancel_navigation and not settling.coarse_ready
    ready = core.update(_input(0.51, target_depth_m=1.4, base_speed_mps=0.0))
    assert ready.cancel_navigation and ready.coarse_ready
    assert core.phase is NavPhase.READY


def test_legacy_near_handoff_still_precedes_waypoint_planning():
    core = CoarseNavigationCore(NavigationConfig(
        near_target_depth_m=1.5, still_settle_s=0.2,
    ))
    core.begin('pick', 'legacy:near', stamp_s=0.0)
    first = core.update(_input(
        0.1, target_depth_m=1.0, base_speed_mps=0.0,
    ))
    assert first.waypoint_xy is None
    assert first.cancel_navigation and not first.coarse_ready
    ready = core.update(_input(
        0.31, target_depth_m=1.0, base_speed_mps=0.0,
    ))
    assert ready.waypoint_xy is None
    assert ready.cancel_navigation and ready.coarse_ready
    assert core.phase is NavPhase.READY


def test_tracking_loss_cancels_and_reacquires_with_bounded_budget():
    core = CoarseNavigationCore(NavigationConfig(
        max_reacquisitions=1, observation_wait_timeout_s=0.2,
    ))
    core.begin('pick', 'task:1', stamp_s=0.0)
    core.update(_input(0.1))
    lost = core.update(_input(
        0.2, perception_valid=False, target_stamp_s=None,
    ))
    assert lost.cancel_navigation and lost.request_reacquire
    assert core.phase is NavPhase.REACQUIRE
    failed = core.update(_input(
        0.5, perception_valid=False, target_stamp_s=None,
    ))
    assert failed.cancel_navigation
    assert core.phase is NavPhase.FAILED


def test_stall_causes_bounded_replan_from_latest_observation():
    core = CoarseNavigationCore(NavigationConfig(
        stall_timeout_s=0.3, max_replans=1,
    ))
    core.begin('pick', 'task:1', stamp_s=0.0)
    first = core.update(_input(0.1))
    np.testing.assert_allclose(first.waypoint_xy, [1.2, 0.0])
    replanned = core.update(_input(
        0.41, base_xy=np.array([0.0, 0.0]), target_xy=np.array([3.0, 1.0]),
    ))
    assert replanned.waypoint_xy is not None
    assert replanned.replan_count == 1


def test_goal_reached_outside_near_field_replans_not_ready():
    core = CoarseNavigationCore(NavigationConfig(max_replans=1))
    core.begin('pick', 'task:1', stamp_s=0.0)
    core.update(_input(0.1))
    decision = core.update(_input(0.2, goal_reached=True))
    assert decision.waypoint_xy is not None
    assert not decision.coarse_ready


def test_normal_base_progress_does_not_move_the_waypoint_horizon():
    core = CoarseNavigationCore()
    core.begin('pick', 'task:1', stamp_s=0.0)
    core.update(_input(0.1))
    decision = core.update(_input(
        0.2, base_xy=np.array([0.4, 0.0]), target_xy=np.array([3.0, 0.0]),
    ))
    assert decision.waypoint_xy is None
    np.testing.assert_allclose(core.goal_xy, [1.2, 0.0])


def test_missing_standoff_waits_without_publishing_waypoint():
    core = CoarseNavigationCore()
    core.begin('pick', 'task:1', stamp_s=0.0)
    decision = core.update(_input(0.1, suggested_displacement_m=None))
    assert core.phase is NavPhase.WAIT_OBSERVATION
    assert decision.waypoint_xy is None


def test_initial_missing_odometry_waits_without_throwing():
    core = CoarseNavigationCore()
    core.begin('pick', 'task:1', stamp_s=0.0)
    decision = core.update(_input(
        0.1, base_xy=None, base_speed_mps=float('inf'),
    ))
    assert core.phase is NavPhase.WAIT_OBSERVATION
    assert 'waiting' in decision.reason


def test_explicit_goal_missing_odometry_fails_without_legacy_reacquisition():
    core = CoarseNavigationCore(NavigationConfig(
        observation_wait_timeout_s=0.2,
    ))
    core.begin('pick', 'work-pose-1', stamp_s=0.0)
    goal = np.array([0.8, -0.2])

    waiting = core.update(_input(
        0.1,
        base_xy=None,
        base_speed_mps=float('inf'),
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert core.phase is NavPhase.WAIT_OBSERVATION
    assert not waiting.request_reacquire

    failed = core.update(_input(
        0.21,
        base_xy=None,
        base_speed_mps=float('inf'),
        explicit_goal_xy=goal,
        suggested_displacement_m=None,
    ))
    assert core.phase is NavPhase.FAILED
    assert failed.cancel_navigation
    assert not failed.request_reacquire
    assert core.reacquisition_count == 0
    assert 'timed out' in failed.reason


def test_navigation_health_loss_cancels_once_then_replans_after_recovery():
    core = CoarseNavigationCore(NavigationConfig(max_replans=1))
    core.begin('pick', 'task:1', stamp_s=0.0)
    core.update(_input(0.1))
    unhealthy = core.update(_input(0.2, navigation_healthy=False))
    assert unhealthy.cancel_navigation and unhealthy.replan_count == 1
    still_unhealthy = core.update(_input(0.3, navigation_healthy=False))
    assert still_unhealthy.replan_count == 1
    recovered = core.update(_input(0.4, navigation_healthy=True))
    assert recovered.waypoint_xy is not None
    assert core.phase is NavPhase.NAVIGATING


def test_parse_explicit_work_pose_contract_uses_goal_id_as_task_key():
    request = parse_task_navigation_request({
        'schema': 'z_manip.task_status.v1',
        'phase': 'coarse_nav',
        'instruction': 'pick mustard',
        'prospective_serial': 12,
        'prospective_base_displacement_m': 99.0,
        'work_pose': {
            'goal_id': 'work-epoch4-generation2',
            'map_frame': 'map',
            'map_goal_xy': [1.25, -0.75],
            'map_goal_yaw_rad': -0.42,
            'source': {'epoch': 4, 'generation': 2, 'request_id': 'ground-9'},
        },
    })
    assert request is not None
    assert request.task_key == 'work-epoch4-generation2'
    assert request.goal_id == request.task_key
    assert request.map_frame == 'map'
    np.testing.assert_array_equal(request.map_goal_xy, [1.25, -0.75])
    assert request.map_goal_yaw_rad == -0.42
    assert request.suggested_displacement_m is None
    assert request.source == {
        'epoch': 4, 'generation': 2, 'request_id': 'ground-9',
    }


@pytest.mark.parametrize('work_pose', [
    {
        'goal_id': '', 'map_frame': 'map',
        'map_goal_xy': [1.0, 2.0], 'map_goal_yaw_rad': 0.0,
    },
    {
        'goal_id': 'g1', 'map_frame': 'map',
        'map_goal_xy': [1.0], 'map_goal_yaw_rad': 0.0,
    },
    {
        'goal_id': 'g1', 'map_frame': 'map',
        'map_goal_xy': [1.0, float('nan')], 'map_goal_yaw_rad': 0.0,
    },
    {
        'goal_id': 'g1', 'map_frame': 'map',
        'map_goal_xy': [1.0, 2.0], 'map_goal_yaw_rad': float('inf'),
    },
    {
        'goal_id': 'g1', 'map_frame': 'map', 'map_goal_xy': [1.0, 2.0],
        'map_goal_yaw_rad': 0.0, 'source': 'not-an-object',
    },
    {
        'goal_id': 'g1', 'map_goal_xy': [1.0, 2.0],
        'map_goal_yaw_rad': 0.0,
    },
    {
        'goal_id': 'g1', 'map_frame': '', 'map_goal_xy': [1.0, 2.0],
        'map_goal_yaw_rad': 0.0,
    },
    {
        'goal_id': 'g1', 'map_frame': 42, 'map_goal_xy': [1.0, 2.0],
        'map_goal_yaw_rad': 0.0,
    },
])
def test_parse_explicit_work_pose_rejects_malformed_contract(work_pose):
    with pytest.raises(ValueError):
        parse_task_navigation_request({
            'schema': 'z_manip.task_status.v1',
            'phase': 'coarse_nav',
            'instruction': 'pick mustard',
            'work_pose': work_pose,
        })


def test_parse_legacy_contract_preserves_target_ray_fallback():
    request = parse_task_navigation_request({
        'schema': 'z_manip.task_status.v1',
        'phase': 'coarse_nav',
        'instruction': 'pick mustard',
        'prospective_serial': 3,
        'prospective_base_displacement_m': 0.8,
    })
    assert request is not None
    assert request.task_key == 'pick mustard:3'
    assert request.goal_id is None
    assert request.map_frame is None
    assert request.map_goal_xy is None
    assert request.suggested_displacement_m == 0.8


def test_parse_non_coarse_status_requests_navigation_deactivation():
    assert parse_task_navigation_request({
        'schema': 'z_manip.task_status.v1',
        'phase': 'visual_approach',
    }) is None


def test_parse_rejects_non_object_task_status():
    with pytest.raises(ValueError, match='must be an object'):
        parse_task_navigation_request([])
