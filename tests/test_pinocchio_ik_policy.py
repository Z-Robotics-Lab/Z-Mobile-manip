from __future__ import annotations

import os
import tempfile
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.kinematics import pinocchio_ik
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


def test_reset_warm_start_clears_seeds_without_touching_radius_floor():
    solver = _solver_policy()
    solver._warm_starts = deque(maxlen=solver._WARM_START_CAPACITY)
    solver._warm_starts.append((np.array([0.1, 0.0, 0.0]), np.full(6, 0.2)))
    radius_before = solver._WARM_START_RADIUS_M

    solver.reset_warm_start()

    assert len(solver._warm_starts) == 0
    # The documented 0.20 m acceptance radius must be untouched by the reset.
    assert solver._WARM_START_RADIUS_M == radius_before == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Resident reduced-model cache (item #10 warm planner model).  Pinocchio is
# absent from the host unit env, so a fake ``pin`` module exercises the pure
# cache/keying logic without the bindings.
# ---------------------------------------------------------------------------
class _FakeModel:
    def __init__(self, names, nq):
        self.names = names
        self.nq = nq


class _FakePin:
    def __init__(self):
        self.reduced_calls = 0

    def buildModelFromUrdf(self, path):
        return _FakeModel(["universe", "j1", "j2", "extra"], 3)

    def neutral(self, model):
        return [0.0] * model.nq

    def buildReducedModel(self, full, locked, neutral):
        self.reduced_calls += 1
        return _FakeModel(["universe", "j1", "j2"], 2)


def _fake_chain(joint_names):
    return SimpleNamespace(joint_names=tuple(joint_names))


def _temp_urdf(text=b"<robot name='r'/>"):
    handle = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


@pytest.fixture(autouse=True)
def _clear_reduced_model_cache():
    pinocchio_ik._REDUCED_MODEL_CACHE.clear()
    yield
    pinocchio_ik._REDUCED_MODEL_CACHE.clear()


def test_reduced_model_is_built_once_then_served_bit_identical():
    pin = _FakePin()
    path = _temp_urdf()
    chain = _fake_chain(("j1", "j2"))
    try:
        first, hit_first = pinocchio_ik._reduced_model_for(pin, path, chain)
        second, hit_second = pinocchio_ik._reduced_model_for(pin, path, chain)
    finally:
        os.unlink(path)

    assert hit_first is False and hit_second is True
    assert pin.reduced_calls == 1
    # A cache hit returns the identical model object, so downstream FK/Jacobian
    # numerics are bit-identical to a private build.
    assert second is first


def test_reduced_model_cache_invalidates_on_urdf_edit():
    pin = _FakePin()
    path = _temp_urdf()
    chain = _fake_chain(("j1", "j2"))
    try:
        pinocchio_ik._reduced_model_for(pin, path, chain)
        # A genuine URDF edit changes mtime (and size) -> fresh build.
        with open(path, "wb") as handle:
            handle.write(b"<robot name='r2'/>   ")
        os.utime(path, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
        _model, hit = pinocchio_ik._reduced_model_for(pin, path, chain)
    finally:
        os.unlink(path)

    assert hit is False
    assert pin.reduced_calls == 2


def test_reduced_model_cache_keys_on_active_joint_set():
    pin = _FakePin()
    path = _temp_urdf()
    try:
        pinocchio_ik._reduced_model_for(pin, path, _fake_chain(("j1", "j2")))
        _model, hit = pinocchio_ik._reduced_model_for(pin, path, _fake_chain(("j1",)))
    finally:
        os.unlink(path)

    assert hit is False
    assert pin.reduced_calls == 2


def test_reduced_model_cache_capacity_is_bounded():
    pin = _FakePin()
    path = _temp_urdf()
    try:
        for index in range(pinocchio_ik._REDUCED_MODEL_CACHE_CAPACITY + 5):
            os.utime(path, ns=(1_000_000 * (index + 1), 1_000_000 * (index + 1)))
            pinocchio_ik._reduced_model_for(pin, path, _fake_chain(("j1", "j2")))
    finally:
        os.unlink(path)

    assert len(pinocchio_ik._REDUCED_MODEL_CACHE) <= (
        pinocchio_ik._REDUCED_MODEL_CACHE_CAPACITY
    )


def test_reduced_model_missing_joint_still_raises():
    pin = _FakePin()
    path = _temp_urdf()
    try:
        with pytest.raises(ValueError, match="missing arm joints"):
            pinocchio_ik._reduced_model_for(pin, path, _fake_chain(("j1", "absent")))
    finally:
        os.unlink(path)
