from __future__ import annotations

import importlib.util
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


def test_timed_lift_fails_closed_instead_of_bursting_late_targets() -> None:
    class LateClock(FakeClock):
        def sleep(self, duration: float) -> None:
            self.value += max(0.0, float(duration)) + 0.20

    path = np.vstack((_q(0.0), _q(0.05), _q(0.10)))
    clock = LateClock()
    robot = StreamingRobot(path[0])

    with pytest.raises(EXECUTOR.SafetyError, match="schedule lag"):
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

    assert not any(name == "move_j" for name, _value in robot.commands)
