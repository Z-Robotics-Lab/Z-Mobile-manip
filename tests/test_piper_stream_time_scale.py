"""The streamed approach must never be stretched past the arm's own rate.

Defect: ``APPROACH_STREAM_REFERENCE_SPEED_PERCENT = 15`` with the operator
running ``speed_percent = 5`` produced ``time_scale = max(1, 15/5) = 3``.  The
streamer then issued 105 ``move_j`` targets 59.5 ms apart, each advancing
0.0677 deg.  PiPER at 5% covers that in 7.7 ms and then stands still for the
remaining 51.8 ms: 105 accel/decel restarts at 17 Hz, duty cycle 0.23, and a
visibly segmented final descent.  The pregrasp transit looked smooth by
contrast only because it coalesces to a single ``move_j`` edge.

These tests fix the invariant, not the constant: a streamed schedule may be
stretched only as far as the arm's MEASURED continuous joint rate requires,
and never further than the legacy ratio (so it can never be slower than what
shipped).  Re-introducing a reference speed above the operator's run speed as
an unconditional multiplier fails here.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_staged_grasp_executor.py"
SPEC = importlib.util.spec_from_file_location("piper_stream_time_scale_executor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXECUTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXECUTOR
SPEC.loader.exec_module(EXECUTOR)


def constant_rate_path(peak_deg_s: float, duration_s: float, samples: int = 106):
    """A dense schedule whose peak per-joint rate is exactly ``peak_deg_s``."""

    times = np.linspace(0.0, duration_s, samples)
    travel = math.radians(peak_deg_s) * duration_s
    positions = np.zeros((samples, 6))
    positions[:, 0] = np.linspace(0.0, travel, samples)
    return positions, times


# Peak per-joint rate measured on the dense approach/lift arrays of the three
# recorded speed_percent=5 sessions (planned_grasp.npz, artifacts/go2w_real).
RECORDED_APPROACH_PEAK_DEG_S = (5.72, 6.06, 8.22)
RECORDED_LIFT_PEAK_DEG_S = (11.80, 13.43, 14.21)
# Measured PiPER continuous joint rate at speed_percent=5.
ARM_RATE_AT_SPEED_5_DEG_S = 8.80


def scale(
    peak_deg_s, *, speed_percent, duration_s=2.0, reference=15, holding_load=False
):
    positions, times = constant_rate_path(peak_deg_s, duration_s)
    return EXECUTOR.stream_time_scale(
        positions,
        times,
        speed_percent=speed_percent,
        reference_speed_percent=reference,
        holding_load=holding_load,
    )


@pytest.mark.parametrize("peak_deg_s", RECORDED_APPROACH_PEAK_DEG_S)
def test_recorded_approach_is_not_stretched_at_the_operator_speed(peak_deg_s):
    """This is the defect, expressed as an assertion.

    Every recorded approach descends slower than the arm's own 8.80 deg/s at
    speed 5, so no stretch is warranted at all.  The shipped rule stretched
    them 3x regardless, which is what the operator sees as segmentation.
    """

    assert peak_deg_s < ARM_RATE_AT_SPEED_5_DEG_S
    value = scale(peak_deg_s, speed_percent=5)
    assert value <= 1.10, (
        "the approach stream is being stretched beyond what the arm needs; "
        "a reference speed above the operator run speed has been reintroduced"
    )


@pytest.mark.parametrize("peak_deg_s", RECORDED_LIFT_PEAK_DEG_S)
def test_an_unloaded_profile_is_stretched_only_as_far_as_the_arm_requires(peak_deg_s):
    # A profile this fast genuinely outruns the arm at speed 5 and must still
    # stretch -- but to the arm's requirement, not to a fixed 3x.
    value = scale(peak_deg_s, speed_percent=5)
    budget = (
        EXECUTOR.PIPER_JOINT_RATE_DEG_S_PER_SPEED_PERCENT
        * 5
        * EXECUTOR.STREAM_RATE_UTILISATION
    )
    assert value == pytest.approx(peak_deg_s / budget, rel=1e-9)
    assert 1.0 < value < 3.0


@pytest.mark.parametrize("peak_deg_s", RECORDED_LIFT_PEAK_DEG_S)
def test_a_holding_arm_keeps_the_legacy_stretch(peak_deg_s):
    """The measured rate does not cover a loaded arm, so it is not applied.

    ``PIPER_JOINT_RATE_DEG_S_PER_SPEED_PERCENT`` was regressed from unloaded
    move_j legs.  Applying it to the recorded lift peaks at speed 5 would raise
    the commanded joint rate from 3.78 to 7.65 deg/s -- roughly double, at 90%
    of a constant with no loaded evidence behind it, on the one leg where a
    tracking failure means dropping the payload.  The judder this whole change
    cures is on the UNLOADED approach descent, so gating on load costs nothing.
    """

    loaded = scale(peak_deg_s, speed_percent=5, holding_load=True)
    unloaded = scale(peak_deg_s, speed_percent=5, holding_load=False)

    assert loaded == pytest.approx(max(1.0, 15 / 5.0))
    assert loaded > unloaded


def test_a_holding_arm_is_never_commanded_faster_than_what_ships_today():
    """Sweep the whole legal range; the loaded path must be exactly legacy."""

    for peak_deg_s in (0.01, 1.0, 5.72, 14.21, 60.0, 400.0):
        for speed_percent in range(1, EXECUTOR.MAX_SPEED_PERCENT + 1):
            legacy = max(1.0, 15 / float(speed_percent))
            value = scale(peak_deg_s, speed_percent=speed_percent, holding_load=True)
            assert value == pytest.approx(legacy), (peak_deg_s, speed_percent)


def test_no_op_at_or_above_the_reference_speed():
    for peak_deg_s in RECORDED_APPROACH_PEAK_DEG_S + RECORDED_LIFT_PEAK_DEG_S:
        for speed_percent in (15, 20, 30, 50):
            for holding in (False, True):
                assert (
                    scale(
                        peak_deg_s,
                        speed_percent=speed_percent,
                        holding_load=holding,
                    )
                    == 1.0
                )


def test_never_slower_than_shipped_and_never_compresses_the_planner_profile():
    """The two-sided clamp, over the whole legal speed range.

    Upper bound: the legacy ratio, so this change can never make any motion
    slower than what ships today.  Lower bound: 1.0, so the planner's bounded
    quintic velocity/acceleration profile is never compressed.
    """

    for peak_deg_s in (0.01, 1.0, 5.72, 14.21, 60.0, 400.0):
        for speed_percent in range(1, EXECUTOR.MAX_SPEED_PERCENT + 1):
            legacy = max(1.0, 15 / float(speed_percent))
            value = scale(peak_deg_s, speed_percent=speed_percent)
            assert 1.0 <= value <= legacy + 1e-12, (peak_deg_s, speed_percent)


def test_the_wrist_search_reference_speed_is_unaffected():
    """The other production caller must not change behaviour at any speed.

    ``piper_wrist_search_executor`` streams a 30 deg/s reference profile with
    ``reference_speed_percent=12`` over a 1..12 speed range.  Its demand always
    exceeds the arm budget by more than the legacy ratio does, so the clamp
    binds and the result is bit-identical to today at every legal speed.
    """

    for speed_percent in range(1, 13):
        legacy = max(1.0, 12 / float(speed_percent))
        assert scale(30.0, speed_percent=speed_percent, reference=12) == legacy


def test_the_effective_command_interval_stops_being_a_visible_ratchet():
    """The recorded speed-5 approach, before and after.

    106 samples over 2.073 s.  The shipped 3x stretch spaced them 59.5 ms
    apart and each hop took the arm 7.7 ms, leaving it stationary for 87% of
    every tick.  Undoing the stretch restores the planner's own 19.8 ms
    cadence.
    """

    positions, times = constant_rate_path(6.06, 2.073, samples=106)
    nominal_interval_s = float(np.median(np.diff(times)))
    shipped = max(1.0, 15 / 5.0)
    fixed = EXECUTOR.stream_time_scale(
        positions,
        times,
        speed_percent=5,
        reference_speed_percent=15,
        holding_load=False,
    )

    assert nominal_interval_s * shipped > 0.055
    assert nominal_interval_s * fixed < 0.025
    # Same commands, same geometry, only the spacing changes.
    assert len(positions) == 106


def test_the_fixed_ratio_is_not_reintroduced_in_the_streamer():
    """Guard the call site, not just the helper."""

    source = SCRIPT.read_text(encoding="utf-8")
    assert "time_scale = max(1.0, reference_speed_percent" not in source, (
        "execute_timed_joint_path is stretching by a fixed reference/run speed "
        "ratio again; derive the stretch from the measured arm rate instead"
    )
    assert "stream_time_scale(" in source


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += max(0.0, float(duration))


class _StreamingRobot:
    def __init__(self, start: np.ndarray) -> None:
        self.joints = np.asarray(start, dtype=float).copy()
        self.stamp = 1.0
        self.commands: list[tuple[str, object]] = []

    def get_joint_angles(self):
        self.stamp += 1.0
        return SimpleNamespace(msg=self.joints.tolist(), timestamp=self.stamp)

    def get_arm_status(self):
        self.stamp += 1.0
        return SimpleNamespace(
            msg=SimpleNamespace(arm_status=0, motion_status=0, err_code=0),
            timestamp=self.stamp,
        )

    def set_speed_percent(self, speed: int) -> None:
        self.commands.append(("speed", speed))

    def enable(self) -> bool:
        return True

    def move_j(self, target: list[float]) -> None:
        self.joints = np.asarray(target, dtype=float)
        self.commands.append(("move_j", tuple(target)))


def test_streaming_sends_the_same_targets_only_sooner():
    positions, times = constant_rate_path(6.06, 2.073, samples=40)
    robot = _StreamingRobot(positions[0])
    clock = _FakeClock()

    EXECUTOR.execute_timed_joint_path(
        robot,
        positions,
        times,
        EXECUTOR.CommandGuard(),
        speed_percent=5,
        segment_timeout_s=1.0,
        start_tolerance_rad=0.01,
        feedback_tolerance_rad=0.01,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    sent = [value for name, value in robot.commands if name == "move_j"]
    assert len(sent) == len(positions) - 1
    np.testing.assert_allclose(np.asarray(sent), positions[1:])
    # 2.073 s of schedule, not 6.22 s.
    assert clock.value <= times[-1] + 0.05
