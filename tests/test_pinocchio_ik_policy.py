from __future__ import annotations

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
