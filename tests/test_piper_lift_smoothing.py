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
SPEC = importlib.util.spec_from_file_location("piper_lift_smoothing_executor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXECUTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXECUTOR
SPEC.loader.exec_module(EXECUTOR)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += max(0.0, float(duration))


class StreamingRobot:
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
        self.commands.append(("enable", True))
        return True

    def move_j(self, target: list[float]) -> None:
        self.joints = np.asarray(target, dtype=float)
        self.commands.append(("move_j", tuple(target)))


def _q(value: float, axis: int = 0) -> np.ndarray:
    result = np.zeros(6)
    result[axis] = value
    return result


def test_resampled_lift_must_follow_every_raw_corner() -> None:
    raw = np.vstack((_q(0.0), _q(0.20), _q(0.20, axis=1)))
    valid = np.vstack((
        _q(0.0),
        _q(0.10),
        _q(0.20),
        np.asarray((0.15, 0.05, 0.0, 0.0, 0.0, 0.0)),
        np.asarray((0.10, 0.10, 0.0, 0.0, 0.0, 0.0)),
        np.asarray((0.05, 0.15, 0.0, 0.0, 0.0, 0.0)),
        _q(0.20, axis=1),
    ))
    EXECUTOR.validate_resampled_path_on_raw_polyline(valid, raw)

    shortcut = np.vstack((raw[0], (raw[0] + raw[-1]) / 2.0, raw[-1]))
    with pytest.raises(EXECUTOR.SafetyError, match="shortcut|leaves"):
        EXECUTOR.validate_resampled_path_on_raw_polyline(shortcut, raw)


def test_timed_lift_streams_at_recorded_cadence_and_finishes_with_feedback() -> None:
    path = np.vstack((_q(0.0), _q(0.02), _q(0.06), _q(0.10)))
    times_s = np.asarray((0.0, 0.05, 0.10, 0.15))
    clock = FakeClock()
    robot = StreamingRobot(path[0])
    guard = EXECUTOR.CommandGuard()

    final = EXECUTOR.execute_timed_joint_path(
        robot,
        path,
        times_s,
        guard,
        speed_percent=15,
        segment_timeout_s=1.0,
        start_tolerance_rad=0.01,
        feedback_tolerance_rad=0.01,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    move_targets = [value for name, value in robot.commands if name == "move_j"]
    assert len(move_targets) == len(path) - 1
    assert times_s[-1] <= clock.value <= times_s[-1] + 0.05
    np.testing.assert_allclose(final, path[-1])
    assert guard.path_motion_started is True


def test_timed_lift_stretches_schedule_at_lower_requested_speed() -> None:
    path = np.vstack((_q(0.0), _q(0.05), _q(0.10)))
    times_s = np.asarray((0.0, 0.10, 0.20))
    clock = FakeClock()

    EXECUTOR.execute_timed_joint_path(
        StreamingRobot(path[0]),
        path,
        times_s,
        EXECUTOR.CommandGuard(),
        speed_percent=5,
        segment_timeout_s=1.0,
        start_tolerance_rad=0.01,
        feedback_tolerance_rad=0.01,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert 0.60 <= clock.value <= 0.65


def test_timed_lift_resyncs_through_bounded_host_jitter() -> None:
    """A late host re-anchors the schedule instead of abandoning the lift.

    Live NUC evidence 2026-07-23: four consecutive mobile handoffs grasped
    the object and then died at the old fail-closed lag gate ~1.5s into the
    lift stream, stranding a held object.  Bounded jitter must stretch the
    schedule, command every target exactly once, and complete.
    """

    class LateClock(FakeClock):
        def sleep(self, duration: float) -> None:
            self.value += max(0.0, float(duration)) + 0.40

    path = np.vstack((_q(0.0), _q(0.05), _q(0.10)))
    clock = LateClock()
    robot = StreamingRobot(path[0])

    final = EXECUTOR.execute_timed_joint_path(
        robot,
        path,
        np.asarray((0.0, 0.05, 0.10)),
        EXECUTOR.CommandGuard(),
        speed_percent=15,
        segment_timeout_s=1.0,
        start_tolerance_rad=0.01,
        feedback_tolerance_rad=0.01,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    move_targets = [value for name, value in robot.commands if name == "move_j"]
    assert len(move_targets) == len(path) - 1
    np.testing.assert_allclose(final, path[-1])


def test_timed_lift_fails_closed_when_resync_budget_is_exhausted() -> None:
    class StalledClock(FakeClock):
        def sleep(self, duration: float) -> None:
            self.value += max(0.0, float(duration)) + 2.0

    path = np.vstack((_q(0.0), _q(0.05), _q(0.10)))
    clock = StalledClock()
    robot = StreamingRobot(path[0])

    with pytest.raises(EXECUTOR.SafetyError, match="resync budget"):
        EXECUTOR.execute_timed_joint_path(
            robot,
            path,
            np.asarray((0.0, 0.05, 0.10)),
            EXECUTOR.CommandGuard(),
            speed_percent=15,
            segment_timeout_s=1.0,
            start_tolerance_rad=0.01,
            feedback_tolerance_rad=0.01,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    # One target was legitimately commanded before the budget ran out; the
    # abort must not burst the remainder.
    move_targets = [value for name, value in robot.commands if name == "move_j"]
    assert len(move_targets) == 1


class StallingRobot(StreamingRobot):
    """Accepts every target and reports success while the joints do not move.

    This is what a slipped payload, a seized joint, or a controller that has
    silently stopped executing looks like from the stream's point of view: the
    SDK call returns, arm status stays clean, and only the feedback disagrees.
    """

    def move_j(self, target: list[float]) -> None:
        self.commands.append(("move_j", tuple(target)))


class LaggingRobot(StreamingRobot):
    """Tracks correctly but always trails the commanded target."""

    def __init__(self, start, lag_rad: float) -> None:
        super().__init__(start)
        self.lag_rad = float(lag_rad)
        self._previous = np.asarray(start, dtype=float).copy()

    def move_j(self, target: list[float]) -> None:
        commanded = np.asarray(target, dtype=float)
        step = commanded - self._previous
        norm = float(np.max(np.abs(step)))
        if norm > 0.0:
            shortfall = min(self.lag_rad, norm)
            self.joints = commanded - step / norm * shortfall
        else:
            self.joints = commanded
        self._previous = commanded
        self.commands.append(("move_j", tuple(target)))


# Median max-joint excursion over 162 recorded lift plans. An earlier revision
# of these tests used 30 deg -- 2.5x this -- which guaranteed the backstop fired
# and hid the fact that a fixed 6 deg floor never fires on the majority of real
# lifts at the speeds actually used.
RECORDED_LIFT_EXCURSION_DEG = 12.16


def _long_stream(excursion_deg: float = RECORDED_LIFT_EXCURSION_DEG):
    """A path the length of a real recorded lift."""

    samples = 60
    times_s = np.linspace(0.0, 3.0, samples)
    path = np.zeros((samples, 6))
    path[:, 0] = np.linspace(0.0, math.radians(excursion_deg), samples)
    return path, times_s


def _run(robot, path, times_s, **kwargs):
    clock = FakeClock()
    return EXECUTOR.execute_timed_joint_path(
        robot,
        path,
        times_s,
        kwargs.pop("guard", EXECUTOR.CommandGuard()),
        speed_percent=kwargs.pop("speed_percent", 15),
        segment_timeout_s=5.0,
        start_tolerance_rad=0.01,
        feedback_tolerance_rad=0.05,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        **kwargs,
    )


def test_a_stalled_arm_fails_closed_instead_of_streaming_to_completion() -> None:
    """The defect: schedule lag is a TIME check, so a stalled arm looked fine.

    Every remaining target was still commanded and the stage reported success,
    which for a loaded leg means reporting a delivered object that is not there.
    """

    path, times_s = _long_stream()
    with pytest.raises(EXECUTOR.SafetyError, match="lost joint tracking"):
        _run(StallingRobot(path[0]), path, times_s)


def test_ordinary_tracking_lag_does_not_trip_the_backstop() -> None:
    """A real arm always trails the stream; the backstop must ignore that."""

    path, times_s = _long_stream()
    final = _run(LaggingRobot(path[0], math.radians(1.0)), path, times_s)
    assert final is not None


def test_the_backstop_reports_where_and_by_how_much_it_diverged() -> None:
    path, times_s = _long_stream()
    with pytest.raises(EXECUTOR.SafetyError) as excinfo:
        _run(StallingRobot(path[0]), path, times_s)
    message = str(excinfo.value)
    assert "deg from the commanded target" in message
    assert "waypoint" in message


def test_a_holding_arm_gets_more_time_than_an_empty_one_end_to_end() -> None:
    """The load gate, exercised through the streamer rather than the helper."""

    # 6.0 deg/s peak: a recorded approach profile.  Inside the empty arm's
    # 7.65 deg/s budget at speed 5, above the loaded arm's 5.18 deg/s -- so the
    # same geometry is retimed differently by load, which is the whole point.
    step_rad = math.radians(0.6)
    path = np.vstack((_q(0.0), _q(step_rad), _q(2.0 * step_rad)))
    times_s = np.asarray((0.0, 0.10, 0.20))

    loaded_guard = EXECUTOR.CommandGuard()
    loaded_guard.holding_load = True
    loaded_clock = FakeClock()
    EXECUTOR.execute_timed_joint_path(
        StreamingRobot(path[0]), path, times_s, loaded_guard,
        speed_percent=5, segment_timeout_s=1.0,
        start_tolerance_rad=0.01, feedback_tolerance_rad=0.01,
        monotonic=loaded_clock.monotonic, sleep=loaded_clock.sleep,
    )

    unloaded_clock = FakeClock()
    EXECUTOR.execute_timed_joint_path(
        StreamingRobot(path[0]), path, times_s, EXECUTOR.CommandGuard(),
        speed_percent=5, segment_timeout_s=1.0,
        start_tolerance_rad=0.01, feedback_tolerance_rad=0.01,
        monotonic=unloaded_clock.monotonic, sleep=unloaded_clock.sleep,
    )

    legacy_duration_s = float(times_s[-1]) * max(1.0, 15 / 5.0)
    # The loaded leg is given more time than the empty one, and still less than
    # the shipped fixed ratio -- refusing to retime it at all would forfeit the
    # judder fix on a leg that ratchets at 17 Hz just like the approach.
    assert unloaded_clock.value < loaded_clock.value < legacy_duration_s


@pytest.mark.parametrize("speed_percent", [5, 15, 20])
@pytest.mark.parametrize("excursion_deg", [9.14, 12.16, 22.07])
def test_a_stall_is_caught_across_the_recorded_lift_excursion_range(
    excursion_deg, speed_percent
) -> None:
    """p10 / p50 / p90 of 162 recorded lift plans, at the speeds actually used.

    A bound that does not sit under the path's own excursion cannot see a total
    stall at all -- the arm sits at the start and the deviation tops out at the
    distance the profile intended.  A fixed 6 deg floor failed this on the
    majority of these.
    """

    path, times_s = _long_stream(excursion_deg)
    with pytest.raises(EXECUTOR.SafetyError, match="lost joint tracking"):
        _run(StallingRobot(path[0]), path, times_s, speed_percent=speed_percent)


@pytest.mark.parametrize("speed_percent", [5, 15, 20])
@pytest.mark.parametrize("excursion_deg", [9.14, 12.16, 22.07])
def test_ordinary_lag_survives_across_the_same_range(
    excursion_deg, speed_percent
) -> None:
    """The other half: the tightened bound must not abort a healthy lift."""

    path, times_s = _long_stream(excursion_deg)
    final = _run(
        LaggingRobot(path[0], math.radians(1.0)),
        path,
        times_s,
        speed_percent=speed_percent,
    )
    assert final is not None


def test_a_loaded_leg_is_budgeted_at_the_loaded_rate() -> None:
    """A holding arm delivers about three quarters of the unloaded rate.

    Budgeting it at the unloaded constant asks for roughly twice what it can
    track, on the one leg where losing the profile drops the payload.
    """

    assert (
        EXECUTOR.PIPER_LOADED_JOINT_RATE_DEG_S_PER_SPEED_PERCENT
        < EXECUTOR.PIPER_JOINT_RATE_DEG_S_PER_SPEED_PERCENT
    )

    # A profile that fits the unloaded budget but exceeds the loaded one.
    peak_deg_s = 1.30 * 5 * EXECUTOR.STREAM_RATE_UTILISATION
    positions = np.zeros((80, 6))
    times = np.linspace(0.0, 2.0, 80)
    positions[:, 0] = np.linspace(0.0, math.radians(peak_deg_s) * 2.0, 80)

    loaded = EXECUTOR.stream_time_scale(
        positions, times, speed_percent=5, reference_speed_percent=15,
        holding_load=True,
    )
    unloaded = EXECUTOR.stream_time_scale(
        positions, times, speed_percent=5, reference_speed_percent=15,
        holding_load=False,
    )

    assert loaded > unloaded, "a holding arm must be given more time, not less"
    # Still a real improvement on the shipped fixed ratio: the loaded leg
    # ratchets at 17 Hz too, so refusing to retime it at all forfeits the fix.
    assert loaded < max(1.0, 15 / 5.0)
