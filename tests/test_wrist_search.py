import math

import numpy as np
import pytest

from z_manip.control.wrist_search import (
    BoundedWristSearch,
    WristSearchConfig,
    WristSearchPhase,
)


def test_search_views_are_finite_bounded_and_near_center_first():
    search = BoundedWristSearch()
    config = search.config
    assert search.views[0].pan_offset_rad == 0.0
    assert search.views[0].pitch_offset_rad == 0.0
    # Tie the bound to the configured envelope: hard-coding 36 deg here made
    # this assertion vacuous the moment the envelope shrank, and would have let
    # a later widening past the configured maximum through unnoticed.
    assert all(
        abs(view.pan_offset_rad) <= config.max_pan_offset_rad + 1e-9
        for view in search.views
    )
    assert all(
        abs(view.pitch_offset_rad) <= config.max_pitch_offset_rad + 1e-9
        for view in search.views
    )
    assert len({(view.pan_offset_rad, view.pitch_offset_rad) for view in search.views}) == len(search.views)
    # 5 pan levels x 5 pitch levels; the axis change did not grow the grid.
    assert len(search.views) == 25
    steps = [
        math.hypot(
            second.pan_offset_rad - first.pan_offset_rad,
            second.pitch_offset_rad - first.pitch_offset_rad,
        )
        for first, second in zip(search.views, search.views[1:])
    ]
    # The finite grid never performs the naive +max -> -max cross-sweep.
    assert max(steps) < 2.0 * (config.max_pan_offset_rad + config.max_pitch_offset_rad)
    assert max(steps) < math.radians(72)


def test_default_grid_pans_before_it_pitches():
    """The first new viewpoint must be the left-right head-shake.

    Real searches are short (14 recorded episodes, median one view), so a
    raster that buries the pan behind two pure-pitch views would never widen
    the field in practice.  The walk is ordered in camera-view space, where a
    15 deg pan moves the optical axis 13.3 deg against a 14 deg pitch's 14.0
    deg, so the pan is genuinely the nearer neighbour rather than a tie-break
    artefact of comparing raw joint radians.
    """

    search = BoundedWristSearch()
    assert search.views[1].pitch_offset_rad == 0.0
    assert abs(search.views[1].pan_offset_rad) == pytest.approx(
        search.config.pan_step_rad
    )
    assert search.views[2].pitch_offset_rad == 0.0
    assert abs(search.views[2].pan_offset_rad) == pytest.approx(
        2.0 * search.config.pan_step_rad
    )


def test_pan_axis_is_the_base_yaw_and_pitch_stays_on_the_wrist():
    config = WristSearchConfig()
    # PiPER J1. Index 3 (J4) rotates about the forearm and rolls the horizon;
    # tests/test_wrist_search_camera_axes.py proves that from the URDF.
    assert config.pan_joint_index == 0
    # PiPER J5 elevation, which the operator is explicitly happy with.
    assert config.pitch_joint_index == 4


def test_search_requires_measured_settle_then_two_of_n_confidence():
    config = WristSearchConfig(
        settle_s=0.2,
        detector_hz=5.0,
        observations_per_view=4,
    )
    search = BoundedWristSearch(config)
    anchor = np.zeros(6)
    start = search.start(anchor, now_s=1.0)
    assert start.phase is WristSearchPhase.MOVE
    settling = search.update_motion(anchor, now_s=1.1)
    assert settling.phase is WristSearchPhase.SETTLE
    observing = search.update_motion(anchor, now_s=1.31)
    assert observing.phase is WristSearchPhase.OBSERVE
    first = search.observe(visible=True, confidence=0.8, now_s=1.32)
    assert first.phase is WristSearchPhase.OBSERVE
    assert first.confirmations == 1
    # A 422 / blur frame ABSTAINS: it does not veto the running confirmation.
    abstain = search.observe(visible=False, confidence=None, now_s=1.53)
    assert abstain.confirmations == 1
    found = search.observe(visible=True, confidence=0.7, now_s=1.74)
    assert found.phase is WristSearchPhase.FOUND
    assert found.confirmations == 2


def _observe_at_view(search, anchor, now):
    search.start(anchor, now_s=now)
    search.update_motion(anchor, now_s=now + 0.01)
    search.update_motion(anchor, now_s=now + 0.05)


def test_confident_negative_resets_but_abstain_does_not():
    config = WristSearchConfig(
        settle_s=0.01,
        detector_hz=100.0,
        observations_per_view=6,
        confirmations_required=2,
    )
    search = BoundedWristSearch(config)
    anchor = np.zeros(6)
    _observe_at_view(search, anchor, 0.0)
    now = 0.06

    def step(visible, conf):
        nonlocal now
        now += 0.02
        return search.observe(visible=visible, confidence=conf, now_s=now)

    assert step(True, 0.5).confirmations == 1
    # A real detection scoring below the threshold is a confident negative.
    assert step(True, 0.05).confirmations == 0
    assert step(True, 0.5).confirmations == 1
    # An abstain (no detection) never vetoes the rebuilt confirmation.
    assert step(False, None).confirmations == 1


def test_empty_view_never_confirms_under_abstain_only_frames():
    config = WristSearchConfig(
        settle_s=0.01,
        detector_hz=100.0,
        observations_per_view=3,
        confirmations_required=2,
    )
    search = BoundedWristSearch(config)
    anchor = np.zeros(6)
    _observe_at_view(search, anchor, 0.0)
    now = 0.06
    decision = None
    for _ in range(3):
        now += 0.02
        decision = search.observe(visible=False, confidence=None, now_s=now)
    # Every frame abstained (a genuinely empty view): confirmation stays zero
    # and the search moves on rather than false-confirming.
    assert decision.confirmations == 0
    assert decision.phase in (WristSearchPhase.MOVE, WristSearchPhase.EXHAUSTED)


def test_skip_anchor_view_starts_raster_at_first_new_viewpoint():
    search = BoundedWristSearch(WristSearchConfig(skip_anchor_view=True))
    anchor = np.zeros(6)
    decision = search.start(anchor, now_s=0.0)
    assert decision.view is not None and decision.view.index == 1
    # The anchor stays view 0 so the executor can restore it after a search.
    assert search.views[0].pan_offset_rad == 0.0
    assert search.views[0].pitch_offset_rad == 0.0
    # The default preserves the legacy behaviour (begin at the anchor view).
    legacy = BoundedWristSearch().start(anchor, now_s=0.0)
    assert legacy.view is not None and legacy.view.index == 0


def test_low_confidence_advances_to_next_fixed_view():
    config = WristSearchConfig(
        settle_s=0.1,
        detector_hz=10.0,
        observations_per_view=2,
        confirmations_required=2,
    )
    search = BoundedWristSearch(config)
    anchor = np.asarray((0.1, 0.2, -0.3, 0.0, 0.25, 0.0))
    search.start(anchor, now_s=0.0)
    search.update_motion(anchor, now_s=0.01)
    search.update_motion(anchor, now_s=0.12)
    search.observe(visible=False, confidence=None, now_s=0.13)
    decision = search.observe(visible=True, confidence=0.2, now_s=0.24)
    assert decision.phase is WristSearchPhase.MOVE
    assert decision.view.index == 1
    target = np.asarray(decision.target_joints_rad)
    changed = np.flatnonzero(np.abs(target - anchor) > 1e-9)
    assert set(changed).issubset({config.pan_joint_index, config.pitch_joint_index})


def test_search_exhausts_without_looping_forever():
    config = WristSearchConfig(
        settle_s=0.01,
        detector_hz=100.0,
        observations_per_view=1,
        confirmations_required=1,
        confidence_threshold=0.9,
    )
    search = BoundedWristSearch(config)
    anchor = np.zeros(6)
    now = 0.0
    search.start(anchor, now_s=now)
    for view in search.views:
        target = np.asarray(search._target())
        now += 0.01
        search.update_motion(target, now_s=now)
        now += 0.02
        search.update_motion(target, now_s=now)
        now += 0.02
        decision = search.observe(visible=False, confidence=None, now_s=now)
    assert decision.phase is WristSearchPhase.EXHAUSTED
    assert decision.target_joints_rad is None


def test_stop_is_terminal_and_configuration_is_fail_closed():
    search = BoundedWristSearch()
    search.start(np.zeros(6), now_s=1.0)
    assert search.stop().phase is WristSearchPhase.STOPPED
    with pytest.raises(ValueError):
        WristSearchConfig(confidence_threshold=0.0)
    with pytest.raises(ValueError):
        WristSearchConfig(confirmations_required=4, observations_per_view=3)


def test_default_confirmation_threshold_matches_grounding_service_contract():
    config = WristSearchConfig()
    assert config.confidence_threshold == pytest.approx(0.15)
    assert config.confirmations_required == 2
    assert config.reset_on_confident_negative is True
    assert config.observations_per_view == 3
    assert config.joint_tolerance_rad == pytest.approx(math.radians(1.0))


def test_deadline_still_covers_the_whole_grid_at_the_measured_per_view_cost():
    """The bound must be real: big enough to finish, small enough to bind.

    Per view: one remote fixed-view command (ESTIMATE ~5.9 s, from the 13.4 s
    pre-collapse cost minus the ~9.1 s passive-CAN gate plus its ~1.57 s
    replacement) + settle_s + observations_per_view / detector_hz.
    """

    config = WristSearchConfig()
    views = len(BoundedWristSearch(config).views)
    per_view_s = 5.9 + config.settle_s + config.observations_per_view / config.detector_hz
    assert config.max_search_s >= views * per_view_s
    # ...and it is still a deadline, not an unbounded scan: the stale 360 s
    # budget came from the superseded ~12 s per-view figure.
    assert config.max_search_s <= 1.5 * views * per_view_s
    assert config.max_search_s < 360.0
