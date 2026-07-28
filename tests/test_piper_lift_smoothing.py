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


def _long_stream():
    """A path long enough to cross at least one tracking sample."""

    samples = 60
    times_s = np.linspace(0.0, 3.0, samples)
    path = np.zeros((samples, 6))
    path[:, 0] = np.linspace(0.0, math.radians(30.0), samples)
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


def test_a_holding_arm_streams_at_the_legacy_cadence_end_to_end() -> None:
    """The load gate, exercised through the streamer rather than the helper."""

    # 6.0 deg/s peak: a recorded approach profile, comfortably inside the arm's
    # 7.65 deg/s budget at speed 5, so an unloaded leg needs no stretch at all.
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

    # Loaded keeps the shipped 3x stretch; unloaded is allowed to run faster.
    assert loaded_clock.value == pytest.approx(0.60, abs=0.05)
    assert unloaded_clock.value < loaded_clock.value
