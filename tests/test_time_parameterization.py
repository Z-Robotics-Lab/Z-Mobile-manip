import numpy as np
import pytest

from z_manip.planning.time_parameterization import TimeParameterizationConfig, retime_path


def test_quintic_retiming_respects_joint_velocity_and_acceleration_limits():
    path = np.array([[0.0, 0.0], [0.5, -0.25], [0.8, 0.2]])
    velocity = np.array([0.5, 0.35])
    acceleration = np.array([1.0, 0.7])

    trajectory = retime_path(
        path,
        velocity,
        acceleration,
        TimeParameterizationConfig(sample_period_s=0.005, min_segment_time_s=0.08),
    )

    assert np.allclose(trajectory.positions[0], path[0])
    assert np.allclose(trajectory.positions[-1], path[-1])
    assert np.all(np.diff(trajectory.times_s) > 0.0)
    measured_velocity = (
        np.diff(trajectory.positions, axis=0)
        / np.diff(trajectory.times_s)[:, None]
    )
    assert np.all(np.max(np.abs(measured_velocity), axis=0) <= velocity * 1.01)
    measured_acceleration = (
        np.diff(measured_velocity, axis=0)
        / np.diff(trajectory.times_s[:-1])[:, None]
    )
    assert np.all(np.max(np.abs(measured_acceleration), axis=0) <= acceleration * 1.08)


def test_retiming_rejects_bad_shapes_or_duplicate_only_path():
    with pytest.raises(ValueError, match="at least two"):
        retime_path(np.zeros((1, 2)), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="motion"):
        retime_path(np.zeros((2, 2)), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="align"):
        retime_path(np.zeros((2, 2)), np.ones(3), np.ones(2))
