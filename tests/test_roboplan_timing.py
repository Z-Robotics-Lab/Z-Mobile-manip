"""The TOPP-RA retimer may never run the arm faster than the caller asked.

``retime_path`` is the module that decides the timing of every commanded
trajectory, and swapping it for TOPP-RA moves that decision into a C++ solver
that takes its limits from somewhere else entirely.  Four failure modes here are
silent, so each gets a sharp test:

1.  **The limit hand-off.**  ``TOPPRAOptions`` scales the *Scene's* per-joint
    limits, not the caller's, with a single scalar.  Get the mapping wrong in
    the safe direction and the arm crawls; wrong in the other direction and it
    runs at the URDF's 5 rad/s when the caller asked for 1.  Nothing about the
    returned trajectory says which happened.  Worse, the URDF declares no
    acceleration limits at all, so the Scene answers DBL_MAX and the correct
    scale is subnormal -- a value that looks like a bug and rounds like an
    answer.

2.  **The silent Hermite fallback.**  roboplan's LinearBlend gives up on corner
    blending without a word (measured: byte-identical output for every blend
    deviation from 1e-6 to 1e-2 on a densified path) and the fallback misses
    the acceleration cap by 24-800%.  The adapter audits its own output for
    exactly this; ``test_hermite_on_a_densified_path_is_rejected`` proves the
    audit fires against the real solver rather than being decorative.

3.  **Divergence from the quintic.**  A drop-in that rejects a different set of
    inputs is not a drop-in.  ``test_validation_matches_retime_path`` runs both
    implementations over the same bad inputs and compares the messages.

4.  **The corridor collapse.**  Near-collinear waypoints have to be dropped
    before TOPP-RA sees them (see the module docstring).  Dropping them one at a
    time against their immediate neighbours would flatten a gently curved path
    into a straight line that no one collision-checked.

5.  **Output the audit cannot see.**  A NaN sample compares False against every
    threshold, so it passes each branch of the audit in turn; and a spline is
    under no obligation to stay inside the hull of its knots -- ``adaptive``
    measured 9.5e-2 rad past a hard joint stop on the real model.  Neither
    ``execute_timed_joint_path`` nor ``StagedTrajectorySegment`` would have
    caught either one, so both are pinned here and in a live test.

Everything up to the roboplan marker runs on a host with no roboplan, against
the same ``FakeScene`` ``tests/test_roboplan_runtime`` uses -- real model shape,
nq=28, arm at q[20:26], DBL_MAX accelerations -- so a green suite off the 4090
really does cover the mapping, the collapse and every audit branch.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from z_manip.planning import roboplan_timing as rtm
from z_manip.planning.roboplan_runtime import (
    RoboplanConfig,
    RoboplanUnavailable,
    SceneHandle,
    scene_handle,
)
from z_manip.planning.roboplan_timing import (
    SPLINE_MODES,
    RetimingFailure,
    ToppraRetimer,
)
from z_manip.planning.time_parameterization import (
    TimeParameterizationConfig,
    retime_path,
)

# The real model's shape and the deployment's acceleration limits, shared with
# the runtime's tests rather than re-derived.
from tests.test_roboplan_runtime import (
    ACCELERATION,
    ARM_Q,
    ARM_V,
    DBL_MAX,
    NQ,
    FakeScene,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"

# A caller asking for much less than the URDF's 5.0 / 3.0 rad/s.
CALLER_VELOCITY = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.8])
CALLER_ACCELERATION = np.array(ACCELERATION, dtype=float)


# --------------------------------------------------------------- fake roboplan


class FakeJointPath:
    def __init__(self) -> None:
        self.joint_names: list[str] = []
        self.positions: list[np.ndarray] = []


class FakeOptions:
    def __init__(self) -> None:
        self.dt = 0.01
        self.mode = "Hermite"
        self.velocity_scale = 1.0
        self.acceleration_scale = 1.0
        self.max_adaptive_iterations = 10
        self.max_adaptive_step_size = 0.05
        self.max_blend_deviation = 0.01


class FakeModes:
    Hermite = "Hermite"
    Cubic = "Cubic"
    Adaptive = "Adaptive"
    LinearBlend = "LinearBlend"


class FakeTrajectory:
    def __init__(self, positions, times) -> None:
        self.positions = positions
        self.times = times


class FakeToppra:
    """Records what the adapter asked for, replays what the test wants back.

    The default reply is the knots it was handed, ten seconds apart: endpoints
    preserved, monotonic, and far too slow to trip any cap -- so a test that
    cares about the *request* is not also asserting about the reply.
    """

    SplineFittingMode = FakeModes

    def __init__(self) -> None:
        self.path: FakeJointPath | None = None
        self.options: FakeOptions | None = None
        self.scene: object | None = None
        self.group: str | None = None
        self.reply = None
        self.error: Exception | None = None
        self.TOPPRAOptions = FakeOptions
        self.PathParameterizerTOPPRA = self._parameterizer

    def _parameterizer(self, scene, group):
        self.scene = scene
        self.group = group
        return self

    def generate(self, path, options):
        self.path = path
        self.options = options
        if self.error is not None:
            raise self.error
        if self.reply is not None:
            return FakeTrajectory(*self.reply)
        knots = np.asarray(path.positions, dtype=float)
        return FakeTrajectory(knots, np.arange(len(knots), dtype=float) * 10.0)


class FakeCore:
    JointPath = FakeJointPath


class LockedScene(FakeScene):
    """A ``FakeScene`` that refuses to answer outside ``SceneHandle.exclusive``.

    roboplan documents Scene collision queries as not thread safe, and
    LinearBlend collision-checks the blended path, so every call this adapter
    makes has to be under the lock.  ``handle.scene`` raises when it is not.
    """

    handle: SceneHandle | None = None

    def getJointGroupInfo(self, name):
        assert self.handle is not None and self.handle.scene is self
        return super().getJointGroupInfo(name)

    def getAccelerationLimitVectors(self, group_name=""):
        assert self.handle is not None and self.handle.scene is self
        return super().getAccelerationLimitVectors(group_name)


def make_handle(scene: FakeScene | None = None) -> SceneHandle:
    scene = scene if scene is not None else LockedScene()
    seed = np.zeros(NQ)
    seed[list(ARM_Q)] = 0.1
    handle = SceneHandle(
        scene,
        group="piper_arm",
        q_indices=np.array(ARM_Q),
        v_indices=np.array(ARM_V),
        position_limits=(scene.lower.copy(), scene.upper.copy()),
        velocity_limits=scene.velocity.copy(),
        acceleration_limits=CALLER_ACCELERATION.copy(),
        seed=seed,
    )
    if isinstance(scene, LockedScene):
        scene.handle = handle
    return handle


@pytest.fixture
def fake(monkeypatch):
    """A retimer over the fake Scene, with roboplan's bindings replaced."""

    toppra = FakeToppra()
    monkeypatch.setattr(rtm, "_bindings", lambda: (FakeCore, toppra))
    return ToppraRetimer(make_handle()), toppra


def straight_path(n: int = 4) -> np.ndarray:
    """A path that stays inside ``FakeScene``'s joint box, which is the real one.

    Joint 2 is limited to ``[-2.967, 0]`` and joint 1 to ``[0, 3.14]`` on this
    robot, so a uniform sweep from 0.1 leaves the model on two of six joints --
    which is exactly the trajectory the adapter now refuses to return.
    """

    start = np.array([0.1, 0.1, -0.9, 0.1, 0.1, 0.1])
    return np.array([start + t * np.full(6, 0.3) for t in np.linspace(0.0, 1.0, n)])


def densified(vertices: np.ndarray, step: float = 0.035) -> np.ndarray:
    """The exact densification ``rrt_connect._densify`` performs."""

    dense = [vertices[0]]
    for first, second in zip(vertices, vertices[1:]):
        steps = max(1, int(np.ceil(float(np.linalg.norm(second - first)) / step)))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            dense.append(first + alpha * (second - first))
    return np.asarray(dense)


# ------------------------------------------------------------- drop-in parity

BAD_INPUTS = [
    ("one waypoint", np.zeros((1, 6)), np.ones(6), np.ones(6)),
    ("not a matrix", np.zeros(6), np.ones(6), np.ones(6)),
    ("non-finite path", np.array([[0.0] * 6, [np.nan] * 6]), np.ones(6), np.ones(6)),
    ("no motion", np.zeros((3, 6)), np.ones(6), np.ones(6)),
    ("short velocity", straight_path(), np.ones(5), np.ones(6)),
    ("zero velocity", straight_path(), np.zeros(6), np.ones(6)),
    ("infinite velocity", straight_path(), np.full(6, np.inf), np.ones(6)),
    ("short acceleration", straight_path(), np.ones(6), np.ones(7)),
    ("negative acceleration", straight_path(), np.ones(6), -np.ones(6)),
]


@pytest.mark.parametrize("label,path,velocity,acceleration", BAD_INPUTS, ids=[b[0] for b in BAD_INPUTS])
def test_validation_matches_retime_path(fake, label, path, velocity, acceleration):
    """A drop-in that rejects a different set of inputs is not a drop-in.

    Both messages are compared, not just the type: ``staged_trajectory`` surfaces
    them to an operator, and the two implementations must stay word for word
    interchangeable as one of them is edited.
    """

    retimer, _ = fake
    with pytest.raises(ValueError) as quintic:
        retime_path(path, velocity, acceleration)
    with pytest.raises(ValueError) as toppra:
        retimer.retime_path(path, velocity, acceleration)
    assert str(toppra.value) == str(quintic.value)


def test_a_path_of_the_wrong_width_is_rejected_rather_than_mis_scaled(fake):
    """The one deliberate divergence: the joint group has a fixed arity.

    ``retime_path`` infers the DoF from the path and happily retimes a 5-column
    path against 5 limits.  Handing that to a 6-joint group would have TOPP-RA
    read the sixth column out of whatever came next.
    """

    retimer, _ = fake
    narrow = straight_path()[:, :5]
    assert retime_path(narrow, np.ones(5), np.ones(5)).positions.shape[1] == 5
    with pytest.raises(ValueError, match="5 columns but group 'piper_arm' has 6 joints"):
        retimer.retime_path(narrow, np.ones(5), np.ones(5))


def test_the_returned_type_is_the_existing_schema(fake):
    retimer, _ = fake
    path = straight_path()
    trajectory = retimer.retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    reference = retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert type(trajectory) is type(reference)
    assert trajectory.positions.shape[1] == path.shape[1]
    assert trajectory.times_s.shape == (len(trajectory.positions),)


# ------------------------------------------------------------- limit hand-off


def test_velocity_scale_never_lets_a_joint_exceed_the_callers_cap(fake):
    """The safety direction of the mapping, asserted per joint.

    A single scalar cannot honour six ratios, so it has to round toward the
    tightest.  Rounding the other way is not a performance win, it is the arm
    moving faster than the caller's limit on at least one joint.
    """

    retimer, toppra = fake
    settings = TimeParameterizationConfig()
    retimer.retime_path(straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION, settings)

    scene_velocity = np.asarray(retimer._handle.velocity_limits, dtype=float)
    effective = scene_velocity * toppra.options.velocity_scale
    requested = CALLER_VELOCITY * settings.velocity_scale
    assert np.all(effective <= requested + 1e-12)
    # ...and not needlessly slower than the tightest joint demands.
    assert np.max(effective / requested) == pytest.approx(1.0)


def test_acceleration_scale_round_trips_the_urdfs_dbl_max(fake):
    """The URDF declares no acceleration limits, so the correct scale is subnormal.

    ``handle.acceleration_limits`` is the deployment's finite value and would map
    to a scale of ~1.0 -- i.e. DBL_MAX, i.e. no acceleration limit at all, which
    is a step command to the servo.  The Scene's vector is the one TOPP-RA
    scales, so the Scene's vector is the one that has to be read.
    """

    retimer, toppra = fake
    settings = TimeParameterizationConfig()
    retimer.retime_path(straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION, settings)

    requested = CALLER_ACCELERATION * settings.acceleration_scale
    assert toppra.options.acceleration_scale < 2.2250738585072014e-308  # subnormal
    assert DBL_MAX * toppra.options.acceleration_scale == pytest.approx(
        float(np.min(requested)), rel=1e-12,
    )


def test_a_caller_looser_than_the_urdf_does_not_get_a_scale_above_one(fake):
    """A scale above 1 would drive the arm past its own URDF velocity limit."""

    retimer, toppra = fake
    retimer.retime_path(straight_path(), np.full(6, 50.0), CALLER_ACCELERATION)
    assert toppra.options.velocity_scale == 1.0


def test_config_scales_reach_the_solver(fake):
    """0.65 / 0.55 are the deployment's margin; dropping them doubles the speed."""

    retimer, toppra = fake
    retimer.retime_path(
        straight_path(),
        CALLER_VELOCITY,
        CALLER_ACCELERATION,
        TimeParameterizationConfig(velocity_scale=0.5, acceleration_scale=0.25),
    )
    half = toppra.options.velocity_scale
    retimer.retime_path(
        straight_path(),
        CALLER_VELOCITY,
        CALLER_ACCELERATION,
        TimeParameterizationConfig(velocity_scale=1.0, acceleration_scale=1.0),
    )
    assert half == pytest.approx(0.5 * toppra.options.velocity_scale)


def test_a_scene_limit_that_is_not_positive_fails_closed(fake, monkeypatch):
    retimer, _ = fake
    monkeypatch.setattr(
        type(retimer._handle._scene),
        "getAccelerationLimitVectors",
        lambda self, group_name="": (np.zeros(6), np.zeros(6)),
    )
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "limit-mapping"


def test_sample_period_is_the_solver_dt_and_min_segment_time_is_ignored(fake):
    """``min_segment_time_s`` has no TOPP-RA analogue.

    Pinned rather than left implicit: a deployment that raises it expecting
    gentler starts would get no change at all, and would only find out on the
    robot.
    """

    retimer, toppra = fake
    retimer.retime_path(
        straight_path(),
        CALLER_VELOCITY,
        CALLER_ACCELERATION,
        TimeParameterizationConfig(sample_period_s=0.004, min_segment_time_s=2.5),
    )
    assert toppra.options.dt == pytest.approx(0.004)
    slow = (toppra.options.velocity_scale, toppra.options.acceleration_scale)
    retimer.retime_path(
        straight_path(),
        CALLER_VELOCITY,
        CALLER_ACCELERATION,
        TimeParameterizationConfig(sample_period_s=0.004, min_segment_time_s=0.01),
    )
    assert (toppra.options.velocity_scale, toppra.options.acceleration_scale) == slow


# ------------------------------------------------------------- spline modes


def test_default_mode_is_linear_blend_and_reaches_the_solver(fake):
    retimer, toppra = fake
    assert retimer.mode == "linear-blend"
    retimer.retime_path(straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION)
    assert toppra.options.mode == FakeModes.LinearBlend
    assert toppra.options.max_blend_deviation == pytest.approx(0.01)


@pytest.mark.parametrize("mode", sorted(SPLINE_MODES))
def test_every_advertised_mode_maps_to_a_real_enum_member(mode, monkeypatch):
    toppra = FakeToppra()
    monkeypatch.setattr(rtm, "_bindings", lambda: (FakeCore, toppra))
    ToppraRetimer(make_handle(), mode=mode).retime_path(
        straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION,
    )
    assert toppra.options.mode == getattr(FakeModes, SPLINE_MODES[mode])


def test_an_unknown_mode_is_rejected_at_construction():
    """Not at the first plan, on the robot, three stages into a grasp."""

    with pytest.raises(ValueError, match="unknown spline fitting mode"):
        ToppraRetimer(make_handle(), mode="linear_blend")


@pytest.mark.parametrize(
    "kwargs",
    [{"max_blend_deviation_rad": -0.1}, {"collinear_tolerance_rad": np.nan}],
)
def test_negative_or_non_finite_tuning_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ToppraRetimer(make_handle(), **kwargs)


def test_a_collinear_tolerance_large_enough_to_simplify_the_path_is_rejected():
    """The collapse drops waypoints, and nothing re-checks the chord it leaves.

    roboplan collision-checks the blend it fits, against the knots it was
    handed -- so a tolerance big enough to remove a real corner hands it a
    shortcut through whatever the corner was going around, and the shortcut is
    never checked by anyone.  A de-noiser needs 1e-6 rad; a path simplifier
    belongs upstream, where its output can be collision-checked.
    """

    ToppraRetimer(make_handle(), collinear_tolerance_rad=1e-6)
    with pytest.raises(ValueError, match="simplify the path before retiming"):
        ToppraRetimer(make_handle(), collinear_tolerance_rad=1e-3)


# --------------------------------------------------------- collinear collapse


def test_a_densified_rrt_path_reaches_the_solver_as_its_corners(fake):
    """roboplan abandons corner blending on near-collinear waypoints, silently.

    The collapse is what makes the default mode work at all, so this asserts on
    what TOPP-RA actually received, not on some internal flag.
    """

    retimer, toppra = fake
    corners = np.array(
        [
            [0.0, 0.1, -0.2, 0.0, 0.3, 0.0],
            [0.8, 0.6, -0.9, 0.4, 0.1, 0.5],
            [-0.3, 1.1, -0.4, -0.5, 0.7, -0.6],
            [0.2, 0.4, -1.1, 0.9, -0.2, 0.3],
        ],
    )
    dense = densified(corners)
    assert len(dense) > 150
    retimer.retime_path(dense, CALLER_VELOCITY, CALLER_ACCELERATION)

    knots = np.asarray(toppra.path.positions, dtype=float)
    assert len(knots) == len(corners)
    assert knots == pytest.approx(corners, abs=1e-12)


def test_the_collapse_keeps_every_dropped_point_inside_the_corridor():
    """A per-point test would flatten a curve; this one may not.

    Each waypoint of a finely sampled arc is within a hair of the chord joining
    its immediate neighbours, so dropping them one at a time would leave a
    straight line between the endpoints -- through whatever the arc was bending
    around.
    """

    angle = np.linspace(0.0, 1.2, 200)
    arc = np.stack(
        [np.cos(angle), np.sin(angle), *([np.zeros_like(angle)] * 4)], axis=1,
    )
    tolerance = 1e-3
    kept = rtm._collapse_collinear(arc, tolerance)

    assert 2 < len(kept) < len(arc)
    for point in arc:
        gap = min(
            float(rtm._point_segment_distance(point[None, :], first, second)[0])
            for first, second in zip(kept, kept[1:])
        )
        assert gap <= tolerance + 1e-12
    assert kept[0] == pytest.approx(arc[0])
    assert kept[-1] == pytest.approx(arc[-1])


def test_a_zero_tolerance_collapses_nothing(fake):
    """The honest way to opt out: hand TOPP-RA exactly what the caller passed."""

    retimer = ToppraRetimer(make_handle(), collinear_tolerance_rad=0.0)
    _, toppra = fake
    # Signs per joint, not a uniform sweep: joint 1 is limited to [0, 3.14] and
    # joint 2 to [-2.967, 0].
    dense = densified(
        np.array(
            [
                np.zeros(6),
                np.array([0.5, 0.5, -0.5, 0.5, 0.5, 0.5]),
                np.array([-0.4, 0.4, -0.9, -0.4, -0.4, -0.4]),
            ],
        ),
    )
    retimer.retime_path(dense, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert len(toppra.path.positions) == len(dense)


# ----------------------------------------------------------------- fail closed


def test_a_rejected_path_never_falls_back_to_the_quintic(fake):
    """A caller that asked for TOPP-RA and got quintic timing is executing a
    trajectory it did not compute.  The fallback is the wiring phase's call."""

    retimer, toppra = fake
    toppra.error = RuntimeError("Path must have at least 2 points.")
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "toppra-rejected"
    assert "Path must have at least 2 points" in str(error.value)
    # The quintic is not reachable from this module at all.
    assert not hasattr(rtm, "retime_path")


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("dt must be strictly positive."),
        ValueError("initialGridpoints should be monotonically increasing."),
        IndexError("vector::_M_range_check"),
    ],
)
def test_every_shape_of_solver_error_becomes_one_named_reason(fake, error):
    """Measured: roboplan raises all three for a path it will not accept."""

    retimer, toppra = fake
    toppra.error = error
    with pytest.raises(RetimingFailure) as raised:
        retimer.retime_path(straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION)
    assert raised.value.reason == "toppra-rejected"


def test_a_degenerate_single_sample_reply_is_rejected(fake):
    """Measured: a path TOPP-RA cannot blend comes back as one sample, not an error."""

    retimer, toppra = fake
    toppra.reply = (np.full((1, 6), 0.1), np.zeros(1))
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "toppra-rejected"


def test_a_non_finite_sample_is_rejected(fake):
    """NaN defeats every other branch of the audit by construction.

    ``nan > tolerance`` is False, so a NaN endpoint passes the drift check, a
    NaN sample passes the joint-limit check, and ``np.max`` of a NaN-bearing
    array passes the cap check.  Before this guard existed the adapter returned
    an all-NaN trajectory -- including a NaN grasp pose -- without a word.
    """

    retimer, toppra = fake
    path = straight_path(3)
    for row, column in ((1, 3), (2, 0)):  # interior sample, then the grasp pose
        broken = path.copy()
        broken[row, column] = np.nan
        toppra.reply = (broken, np.array([0.0, 10.0, 20.0]))
        with pytest.raises(RetimingFailure) as error:
            retimer.retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
        assert error.value.reason == "toppra-rejected"
        assert "not finite" in str(error.value)


def test_a_sample_outside_the_joint_limits_is_rejected(fake):
    """A spline is not obliged to stay inside the hull of its knots.

    ``adaptive`` measured 9.5e-2 rad past a hard joint stop on the real model
    (see the live test).  Nothing downstream catches it:
    ``execute_timed_joint_path`` validates only that samples are finite and
    increasing in time, so an out-of-limit target goes to the servo.
    """

    retimer, toppra = fake
    path = straight_path(3)
    lower, upper = retimer._handle.position_limits
    over = path.copy()
    over[1, 2] = upper[2] + 1e-3  # joint 2 is limited to [-2.967, 0]
    toppra.reply = (over, np.array([0.0, 10.0, 20.0]))
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "joint-limit"
    assert "1.000e-03" in str(error.value)

    under = path.copy()
    under[1, 1] = lower[1] - 1e-3  # and joint 1 to [0, 3.14]
    toppra.reply = (under, np.array([0.0, 10.0, 20.0]))
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "joint-limit"


def test_the_acceleration_audit_uses_the_spacing_the_executor_flies(fake):
    """Commanded velocity changes between interval MIDPOINTS, not interval ends.

    ``execute_timed_joint_path`` streams sample ``i`` as a position target due
    at ``times[i]``, so velocity is constant across an interval and steps at its
    midpoint.  TOPP-RA's grid is uniform except for a short final interval
    (measured 0.8 ms against a 20 ms nominal), and dividing by the interval
    width there under-reports the acceleration by up to 1.95x -- the same size
    as the Hermite miss this audit exists to catch.  The reply below is exactly
    1.5x the cap by the spacing the arm flies and 0.78x by the interval width.
    """

    retimer, toppra = fake
    settings = TimeParameterizationConfig()
    times = np.array([0.0, 0.02, 0.04, 0.0408])
    cap = CALLER_ACCELERATION[0] * settings.acceleration_scale
    cruise = 0.3  # well under the velocity cap, so only acceleration can fire
    step = 1.5 * cap * 0.5 * (times[3] - times[1])  # the midpoint spacing
    positions = np.zeros((4, 6))
    positions[1:, 0] = np.cumsum(
        [cruise * 0.02, cruise * 0.02, (cruise + step) * (times[3] - times[2])],
    )
    toppra.reply = (positions, times)
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(positions, CALLER_VELOCITY, CALLER_ACCELERATION, settings)
    assert error.value.reason == "limit-violation"
    assert "acceleration 1.5000x" in str(error.value)


@pytest.mark.parametrize(
    "times,detail",
    [
        (np.array([0.5, 1.5, 2.5]), "non-zero start"),
        (np.array([0.0, 1.0, 1.0]), "repeated sample"),
        (np.array([0.0, 2.0, 1.0]), "time travel"),
    ],
    ids=["non-zero start", "repeated sample", "time travel"],
)
def test_an_unusable_time_base_is_rejected(fake, times, detail):
    """``StagedGraspPlan.flattened`` offsets each stage by the previous duration,
    so a stage that does not start at zero is a silent gap inside a grasp."""

    retimer, toppra = fake
    toppra.reply = (straight_path(3), times)
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(straight_path(3), CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "time-base"


@pytest.mark.parametrize("drift", [5e-4, 5e-7])
def test_a_moved_endpoint_is_rejected(fake, drift):
    """The last waypoint is the grasp pose; near enough is not the same place.

    5e-7 is the interesting one: ``StagedTrajectorySegment.__post_init__``
    compares a trajectory's endpoints against its declared start and goal at
    ``atol=1e-8``, so a drift this adapter waved through would be rejected two
    layers downstream by a message that blames the segment rather than the
    retimer.  The tolerance here tracks that contract, not a round number.
    """

    retimer, toppra = fake
    moved = straight_path(3).copy()
    moved[-1, 0] += drift
    toppra.reply = (moved, np.array([0.0, 10.0, 20.0]))
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(straight_path(3), CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "path-drift"
    assert f"{drift:.3e}" in str(error.value)


def test_output_that_exceeds_the_requested_velocity_is_rejected(fake):
    """The audit that catches roboplan's silent Hermite fallback.

    The reply below is exactly on the path and perfectly monotonic -- everything
    except the timing is right, which is what makes it dangerous.
    """

    retimer, toppra = fake
    path = straight_path(3)
    settings = TimeParameterizationConfig()
    span = float(np.max(np.abs(np.diff(path, axis=0))))
    fast = span / (CALLER_VELOCITY.min() * settings.velocity_scale) * 0.9
    toppra.reply = (path, np.array([0.0, fast, 2.0 * fast]))
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION, settings)
    assert error.value.reason == "limit-violation"


def test_output_that_exceeds_the_requested_acceleration_is_rejected(fake):
    """A full reversal at speed: the corner an unblended polyline flies.

    Velocity stays at half the cap throughout, so only the acceleration term can
    be what fires -- pinned by the multiples in the message.
    """

    retimer, toppra = fake
    # 0.05 rad on every joint, negative on joint 2 because its limit is
    # [-2.967, 0]: the sign does not move any |velocity| or |acceleration|.
    step = np.full(6, 0.05)
    step[2] = -0.05
    path = np.array([np.zeros(6), step, np.zeros(6)])
    toppra.reply = (path, np.array([0.0, 0.2, 0.4]))
    with pytest.raises(RetimingFailure) as error:
        retimer.retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "limit-violation"
    assert "velocity 0.4808x and acceleration 3.0303x" in str(error.value)


def test_a_trajectory_inside_the_caps_is_returned_unchanged(fake):
    """Guard the guard: an audit that rejected everything would also pass above."""

    retimer, toppra = fake
    path = straight_path(3)
    toppra.reply = (path, np.array([0.0, 10.0, 20.0]))
    trajectory = retimer.retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert trajectory.positions == pytest.approx(path)
    assert trajectory.times_s == pytest.approx([0.0, 10.0, 20.0])


# ------------------------------------------------------------ lazy dependency


@pytest.mark.skipif(
    importlib.util.find_spec("roboplan") is not None,
    reason="roboplan is installed here; the missing-library path cannot be exercised",
)
def test_a_missing_library_reports_the_reason_a_caller_falls_back_on():
    """Deliberately unpatched: on any host that is not the 4090 this is the path
    taken, and the wiring phase branches on ``reason``, not on ImportError."""

    with pytest.raises(RoboplanUnavailable) as error:
        ToppraRetimer(make_handle()).retime_path(
            straight_path(), CALLER_VELOCITY, CALLER_ACCELERATION,
        )
    assert error.value.reason == "import-failed"


def test_importing_the_module_does_not_import_roboplan():
    """z_manip/planning never reaches the NUC, but this module must still load
    on any host that has neither roboplan nor pinocchio."""

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import z_manip.planning.roboplan_timing as m;"
            " assert 'roboplan' not in sys.modules;"
            " assert m.ToppraRetimer.__init__ is not None",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# =============================================================== roboplan only


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """The real Scene, plus a collision-free polyline through it."""

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    if not URDF.exists():
        pytest.skip("robot URDF not checked out")

    class Robot:
        urdf_path = URDF
        base_link = "piper_base_link"
        tip_link = "piper_gripper_base"
        acceleration_limits = ACCELERATION

    handle = scene_handle(
        RoboplanConfig(enabled=True, scene_cache_dir=tmp_path_factory.mktemp("scene")),
        Robot(),
    )
    lower, upper = handle.position_limits
    rng = np.random.default_rng(5)
    with handle.exclusive() as scene:
        vertices = [handle.extract(handle.seed)]
        while len(vertices) < 6:
            candidate = rng.uniform(lower, upper)
            # Both the vertex AND the straight segment to it, so the path this
            # retimer is judged on is one rrt_connect could have produced.
            edge = np.linspace(vertices[-1], candidate, 48)
            if not any(scene.hasCollisions(handle.expand(q)) for q in edge):
                vertices.append(candidate)
    return handle, densified(np.asarray(vertices))


def test_retimed_trajectory_is_well_formed(live):
    handle, path = live
    trajectory = ToppraRetimer(handle).retime_path(
        path, CALLER_VELOCITY, CALLER_ACCELERATION,
    )

    assert trajectory.positions.shape[1] == handle.dof
    assert trajectory.positions.shape[0] == len(trajectory.times_s) > 2
    assert np.all(np.isfinite(trajectory.positions))
    assert trajectory.times_s[0] == 0.0
    assert np.all(np.diff(trajectory.times_s) > 0.0)
    assert trajectory.positions[0] == pytest.approx(path[0], abs=1e-9)
    assert trajectory.positions[-1] == pytest.approx(path[-1], abs=1e-9)


def test_peak_velocity_and_acceleration_hold_the_requested_caps(live):
    """The safety assertion, at a tighter tolerance than the adapter's own audit.

    The rates below are the ones the arm is actually commanded, derived from the
    executor rather than from the adapter: ``execute_timed_joint_path`` streams
    sample ``i`` as a position target due at ``times[i]``, so velocity is
    constant across an interval and steps between interval MIDPOINTS.

    The audit passes anything within 2% -- which is 9x the worst measured over 8
    paths x sample periods {20, 10, 5} ms (1.002332, at the finest grid; TOPP-RA
    discretizes its constraint).  This fixture's path at the deployment's 20 ms
    measures 1.000000, so 0.1% here fails if the mapping drifts or the audit is
    loosened, rather than silently absorbing it.
    """

    handle, path = live
    settings = TimeParameterizationConfig()
    trajectory = ToppraRetimer(handle).retime_path(
        path, CALLER_VELOCITY, CALLER_ACCELERATION, settings,
    )
    velocity_cap = CALLER_VELOCITY * settings.velocity_scale

    times = trajectory.times_s
    speed = np.diff(trajectory.positions, axis=0) / np.diff(times)[:, None]
    rate = np.diff(speed, axis=0) / np.diff(0.5 * (times[:-1] + times[1:]))[:, None]
    assert np.max(np.abs(speed) / velocity_cap) <= 1.001
    assert np.max(
        np.abs(rate) / (CALLER_ACCELERATION * settings.acceleration_scale),
    ) <= 1.001

    # Cross-check with an oracle that differentiates nothing, so a shared error
    # in the finite differences above cannot hide in it: the arm cannot cover
    # more joint travel than its cap allows in the time it took.  Loose -- the
    # acceleration cap keeps this trajectory at 0.66 of the bound (measured), so
    # it only resolves an overspeed above ~1.5x -- but it shares no code.
    travel = np.sum(np.abs(np.diff(trajectory.positions, axis=0)), axis=0)
    assert np.all(travel / velocity_cap <= times[-1] * 1.001)


def test_halving_the_requested_limits_halves_the_arm(live):
    """An adapter that ignored its arguments would return the same peaks twice.

    The Scene's own limits are 5 rad/s; both requests below are far under it, so
    only the caller's numbers can be what bind.
    """

    handle, path = live
    retimer = ToppraRetimer(handle)

    def peak(velocity):
        trajectory = retimer.retime_path(path, velocity, CALLER_ACCELERATION)
        speed = np.diff(trajectory.positions, axis=0) / np.diff(trajectory.times_s)[:, None]
        return float(np.max(np.abs(speed)))

    full = peak(CALLER_VELOCITY)
    half = peak(0.5 * CALLER_VELOCITY)
    assert half == pytest.approx(0.5 * full, rel=0.02)


def test_the_retimed_path_is_still_collision_free(live):
    """The corner blend cuts up to ``max_blend_deviation_rad`` off the path the
    caller collision-checked, so every emitted sample is re-checked here."""

    handle, path = live
    trajectory = ToppraRetimer(handle).retime_path(
        path, CALLER_VELOCITY, CALLER_ACCELERATION,
    )
    with handle.exclusive() as scene:
        colliding = [
            index
            for index, q in enumerate(trajectory.positions)
            if scene.hasCollisions(handle.expand(q))
        ]
    assert not colliding, f"{len(colliding)} retimed samples are in self-collision"


def test_the_blend_stays_within_the_configured_deviation(live):
    """The bound the collision argument rests on, measured against the polyline."""

    handle, path = live
    deviation = 0.01
    trajectory = ToppraRetimer(
        handle, max_blend_deviation_rad=deviation,
    ).retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)

    worst = max(
        min(
            float(rtm._point_segment_distance(q[None, :], first, second)[0])
            for first, second in zip(path, path[1:])
        )
        for q in trajectory.positions
    )
    assert worst <= deviation


def test_adaptive_mode_leaves_the_joint_limits_and_is_rejected(live):
    """The joint-limit audit, against the real solver on the suite's own path.

    ``adaptive`` fits a spline through the knots and refines it against the
    Scene's collision model -- which says nothing about joint limits, so the
    spline overshoots one by 9.5e-2 rad (5.5 degrees past a hard stop) and
    roboplan returns it happily.  Neither ``execute_timed_joint_path`` nor
    ``StagedTrajectorySegment`` checks joint limits either, so before this audit
    existed that target reached the servo.
    """

    handle, path = live
    with pytest.raises(RetimingFailure) as error:
        ToppraRetimer(handle, mode="adaptive").retime_path(
            path, CALLER_VELOCITY, CALLER_ACCELERATION,
        )
    assert error.value.reason == "joint-limit"

    lower, upper = handle.position_limits
    inside = ToppraRetimer(handle).retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert np.max(np.maximum(inside.positions - upper, lower - inside.positions)) <= 0.0


def test_hermite_on_a_densified_path_is_rejected(live):
    """Proof the audit fires against the real solver, not just against a fake.

    Hermite is the mode whose contract is exact path adherence, and on a
    densified path it was measured at 3.4-9x the requested acceleration: TOPP-RA
    discretizes its constraint, and a knot between two grid points is not
    constrained.  Silently returning this is the whole failure this adapter
    exists to prevent.
    """

    handle, path = live
    with pytest.raises(RetimingFailure) as error:
        ToppraRetimer(
            handle, mode="hermite", collinear_tolerance_rad=0.0,
        ).retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION)
    assert error.value.reason == "limit-violation"


def test_faster_than_the_quintic_on_a_densified_rrt_path(live, capsys):
    """The reason this module exists, measured on the input rrt_connect emits."""

    handle, path = live
    settings = TimeParameterizationConfig()
    start = time.perf_counter()
    toppra = ToppraRetimer(handle).retime_path(
        path, CALLER_VELOCITY, CALLER_ACCELERATION, settings,
    )
    toppra_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    quintic = retime_path(path, CALLER_VELOCITY, CALLER_ACCELERATION, settings)
    quintic_ms = (time.perf_counter() - start) * 1e3

    speedup = float(quintic.times_s[-1] / toppra.times_s[-1])
    with capsys.disabled():
        print(
            f"\n  {len(path)} waypoints: quintic {quintic.times_s[-1]:.2f} s "
            f"({quintic_ms:.1f} ms) -> toppra {toppra.times_s[-1]:.2f} s "
            f"({toppra_ms:.1f} ms) = {speedup:.2f}x shorter",
        )
    # Measured 9.0-13.1x over eight paths; 3x is the floor below which the
    # collapse or the blend has stopped working and the fallback is in charge.
    assert speedup > 3.0
