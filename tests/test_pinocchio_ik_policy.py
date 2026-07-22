from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np

from z_manip.kinematics.pinocchio_ik import PinocchioIKSolver


def _solver_policy() -> PinocchioIKSolver:
    solver = PinocchioIKSolver.__new__(PinocchioIKSolver)
    solver.config = SimpleNamespace(
        position_tolerance_m=0.010,
        position_scale_m=0.025,
        orientation_scale_rad=0.35,
    )
    return solver


def test_task_weights_prioritize_translation_outside_capture_region():
    solver = _solver_policy()

    far = solver._task_weights(0.10)
    near = solver._task_weights(0.0)

    assert np.allclose(far[:3], 50.0)
    assert np.allclose(near[:3], 50.0)
    assert np.all(far[3:] < near[3:])
    assert np.allclose(near[3:], 1.0 / 0.35)


def test_weighted_cost_uses_configured_task_metric():
    error = np.asarray([0.01, 0.0, 0.0, 0.35, 0.0, 0.0])
    weights = np.asarray([50.0, 50.0, 50.0, 2.0, 2.0, 2.0])

    assert PinocchioIKSolver._weighted_cost(error, weights) == np.hypot(0.5, 0.7)


def test_maximum_chain_radius_is_a_conservative_urdf_bound():
    joints = (
        SimpleNamespace(
            joint_type="revolute",
            origin=np.array(
                [[1.0, 0.0, 0.0, 0.3], [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.4], [0.0, 0.0, 0.0, 1.0]],
            ),
            lower=-1.0,
            upper=1.0,
        ),
        SimpleNamespace(
            joint_type="prismatic",
            origin=np.array(
                [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ),
            lower=-0.05,
            upper=0.2,
        ),
    )

    assert np.isclose(
        PinocchioIKSolver._maximum_chain_radius(SimpleNamespace(joints=joints)),
        0.8,
    )


def test_nearby_failed_pose_is_reused_before_global_seeds():
    solver = _solver_policy()
    solver._warm_starts = deque(maxlen=solver._WARM_START_CAPACITY)
    warm = np.full(6, 0.25)
    solver._warm_starts.append((np.array([0.40, 0.0, 0.0]), warm))
    goal = np.eye(4)
    goal[:3, 3] = (0.45, 0.0, 0.0)
    global_seed = np.zeros(6)

    seeds = solver._prepend_warm_starts(
        goal,
        [global_seed],
        np.full(6, -1.0),
        np.full(6, 1.0),
    )

    assert np.array_equal(seeds[0], warm)
    assert np.array_equal(seeds[1], global_seed)


def test_only_nearest_warm_start_precedes_global_seeds():
    solver = _solver_policy()
    solver._warm_starts = deque(maxlen=solver._WARM_START_CAPACITY)
    farther = np.full(6, 0.10)
    nearest = np.full(6, 0.03)
    middle = np.full(6, 0.05)
    solver._warm_starts.extend(
        (
            (np.array([0.10, 0.0, 0.0]), farther),
            (np.array([0.03, 0.0, 0.0]), nearest),
            (np.array([0.05, 0.0, 0.0]), middle),
        ),
    )
    goal = np.eye(4)
    global_seed = np.zeros(6)

    seeds = solver._prepend_warm_starts(
        goal,
        [global_seed],
        np.full(6, -1.0),
        np.full(6, 1.0),
    )

    assert np.array_equal(seeds[0], nearest)
    assert np.array_equal(seeds[1], global_seed)
    assert not any(np.array_equal(seed, farther) for seed in seeds)
    assert not any(np.array_equal(seed, middle) for seed in seeds)
