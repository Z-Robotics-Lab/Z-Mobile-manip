"""The reach must be ONE continuous motion that still rides the checked path.

Guards the three properties the smooth-reach change depends on:
  1. interior vertices are vias, not stops;
  2. the samples never leave the collision-checked polyline;
  3. the per-joint acceleration cap is still respected at corners.

The second half of the file guards the same properties through the DEPLOYED
planner (``OnlinePlanner.retime_reach`` on ``configs/go2w_piper.json``), across
the transit-to-approach seam.  ``piper_staged_grasp_executor.py:754-762``
checks only that the seam POSITIONS match, which a pair of independently
retimed rest-to-rest legs satisfies perfectly while stopping the arm dead --
that is exactly how the stop-and-go reach survived.  Nothing below is allowed
to pass on positions alone.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from z_manip.configuration import load_stack_config
from z_manip.models.planner import PlanningError
from z_manip.planning.online_planner import OnlinePlanner
from z_manip.planning.time_parameterization import (
    TimeParameterizationConfig,
    retime_path,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
DEPLOYED = ROOT / "configs/go2w_piper.json"

VELOCITY = np.full(6, 3.0)
ACCELERATION = np.full(6, 1.5)
SETTINGS = TimeParameterizationConfig(
    sample_period_s=0.02,
    min_segment_time_s=0.10,
    velocity_scale=0.6,
    acceleration_scale=0.5,
)


def _corner_path():
    """A standoff-shaped path: a straight run, a real corner, a straight descent."""
    run = np.linspace(np.zeros(6), np.full(6, 0.6), 12)
    descent = np.linspace(
        np.full(6, 0.6),
        np.full(6, 0.6) + np.array([0.25, -0.25, 0.0, 0.0, 0.0, 0.0]),
        11,
    )
    return np.vstack((run, descent[1:]))


def _speed(trajectory):
    return (
        np.max(np.abs(np.diff(trajectory.positions, axis=0)), axis=1)
        / np.diff(trajectory.times_s)
    )


def _vertex_speeds(trajectory, path):
    """Commanded speed at each ORIGINAL interior vertex, as a fraction of peak.

    The trajectory legitimately rests at its own two endpoints; what reads as
    a "phase change" to an operator is resting at a vertex in between.
    """
    speed = _speed(trajectory)
    peak = float(np.max(speed))
    fractions = []
    for vertex in path[1:-1]:
        index = int(np.argmin(
            np.max(np.abs(trajectory.positions - vertex), axis=1),
        ))
        index = min(max(index, 1), len(speed) - 1)
        fractions.append(float(min(speed[index - 1], speed[index])) / peak)
    return np.asarray(fractions)


def _distance_to_polyline(point, path):
    best = np.inf
    for start, end in zip(path, path[1:]):
        edge = end - start
        span = float(edge @ edge)
        if span <= 1e-18:
            continue
        alpha = float(np.clip((point - start) @ edge / span, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (start + alpha * edge))))
    return best


def test_interior_vertices_are_vias_not_stops():
    path = _corner_path()
    fractions = _vertex_speeds(
        retime_path(path, VELOCITY, ACCELERATION, SETTINGS), path,
    )
    # The path holds exactly one real corner (run -> descent).  A genuine
    # corner MUST be slowed -- that is the acceleration cap, not a phase
    # change -- but every collinear vertex is pure retiming artifact and has
    # to be crossed at speed.
    slow = fractions < 0.10
    assert int(np.sum(slow)) <= 1, (
        f"{int(np.sum(slow))} interior vertices are still near-stops: "
        f"{np.round(fractions, 4)}"
    )
    assert np.median(fractions) > 0.5, (
        f"collinear vertices are not crossed at speed: {np.round(fractions, 4)}"
    )
    assert np.min(fractions) > 0.0


def test_legacy_budget_still_stops_at_every_vertex():
    path = _corner_path()
    legacy = retime_path(
        path,
        VELOCITY,
        ACCELERATION,
        TimeParameterizationConfig(
            sample_period_s=SETTINGS.sample_period_s,
            min_segment_time_s=SETTINGS.min_segment_time_s,
            velocity_scale=SETTINGS.velocity_scale,
            acceleration_scale=SETTINGS.acceleration_scale,
            corner_acceleration_budget=0.0,
        ),
    )
    # The quintic's velocity is analytically zero at each vertex; a finite
    # difference across the vertex leaves only the sampling residual, so
    # compare against the profile's own peak rather than against zero.
    # Legacy stops at EVERY interior vertex, corner or not.
    fractions = _vertex_speeds(legacy, path)
    assert np.all(fractions < 0.05), np.round(fractions, 4)


def test_samples_stay_on_the_checked_polyline():
    path = _corner_path()
    trajectory = retime_path(path, VELOCITY, ACCELERATION, SETTINGS)
    worst = max(
        _distance_to_polyline(sample, path) for sample in trajectory.positions
    )
    assert worst < 1e-9, f"trajectory left the checked polyline by {worst} rad"
    assert np.allclose(trajectory.positions[0], path[0])
    assert np.allclose(trajectory.positions[-1], path[-1])


def test_corner_respects_the_acceleration_cap():
    path = _corner_path()
    trajectory = retime_path(path, VELOCITY, ACCELERATION, SETTINGS)
    steps = np.diff(trajectory.times_s)
    velocity = np.diff(trajectory.positions, axis=0) / steps[:, None]
    acceleration = np.diff(velocity, axis=0) / steps[:-1, None]
    cap = ACCELERATION * SETTINGS.acceleration_scale
    worst = np.max(np.abs(acceleration), axis=0)
    assert np.all(worst <= cap), f"acceleration {worst} exceeds cap {cap}"


def test_reach_is_faster_than_stopping_at_every_vertex():
    path = _corner_path()
    smooth = retime_path(path, VELOCITY, ACCELERATION, SETTINGS)
    legacy = retime_path(
        path,
        VELOCITY,
        ACCELERATION,
        TimeParameterizationConfig(
            sample_period_s=SETTINGS.sample_period_s,
            min_segment_time_s=SETTINGS.min_segment_time_s,
            velocity_scale=SETTINGS.velocity_scale,
            acceleration_scale=SETTINGS.acceleration_scale,
            corner_acceleration_budget=0.0,
        ),
    )
    assert smooth.times_s[-1] < legacy.times_s[-1]


@pytest.mark.parametrize("count", (2, 3, 4))
def test_short_paths_stay_bounded(count):
    """Both ends of a short path are at rest; it must still finish in seconds.

    A profile that only interpolates between the two node speeds integrates a
    two-waypoint path (rest -> rest) at zero mean speed and runs away to an
    unbounded duration and sample count.
    """
    path = np.linspace(np.zeros(6), np.full(6, 0.3), count)
    trajectory = retime_path(path, VELOCITY, ACCELERATION, SETTINGS)
    assert np.isfinite(trajectory.times_s).all()
    assert 0.0 < trajectory.times_s[-1] < 30.0
    assert len(trajectory.positions) < 2000
    assert np.allclose(trajectory.positions[-1], path[-1])


def test_budget_must_be_a_fraction():
    with pytest.raises(ValueError):
        TimeParameterizationConfig(corner_acceleration_budget=1.0)
    with pytest.raises(ValueError):
        TimeParameterizationConfig(corner_acceleration_budget=-0.1)


# ------------------------------------------- the deployed transit/approach seam


@pytest.fixture(scope="module")
def deployed():
    """The real planner on the deployed config -- real limits, real timing."""

    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    config = load_stack_config(DEPLOYED, environ={"Z_MANIP_ROBOT_URDF": str(URDF)})
    return OnlinePlanner(config)


@pytest.fixture(scope="module")
def stepwise(deployed):
    """The same planner with the legacy stop-at-every-vertex retiming."""

    return OnlinePlanner(replace(
        deployed.config,
        time_parameterization=replace(
            deployed.config.time_parameterization,
            corner_acceleration_budget=0.0,
        ),
    ))


def _reach(corner_deg):
    """A transit into a standoff, then the 11-waypoint 0.10 m descent.

    ``corner_deg`` is the joint-space direction change at the standoff.  The
    deployed reach measures 56.6 deg there; 0 deg is the same reach with the
    transit arriving straight down the descent line, which is the control that
    separates "slowed by the acceleration cap" from "stopped by the retimer".
    """

    start = np.array([0.0, 0.30, -0.55, 0.0, 0.60, 0.0])
    standoff = np.array([0.55, 0.70, -1.05, 0.10, 0.85, -0.20])
    if corner_deg == 0.0:
        heading = standoff - start
        heading /= np.linalg.norm(heading)
        transit = np.vstack((
            start,
            start + 0.4 * np.linalg.norm(standoff - start) * heading,
            standoff,
        ))
        descent = heading
    else:
        transit = np.vstack((
            start,
            start + 0.35 * (standoff - start) + np.array(
                [0.05, -0.03, 0.02, 0.0, 0.01, 0.0]),
            start + 0.70 * (standoff - start) + np.array(
                [0.03, 0.02, -0.01, 0.0, 0.0, 0.0]),
            standoff,
        ))
        incoming = transit[-1] - transit[-2]
        incoming /= np.linalg.norm(incoming)
        offset = np.array([0.0, -0.6, 0.75, 0.0, -0.2, 0.0])
        offset -= (offset @ incoming) * incoming
        offset /= np.linalg.norm(offset)
        angle = np.deg2rad(corner_deg)
        descent = np.cos(angle) * incoming + np.sin(angle) * offset
    # grasp_pipeline._approach_steps_for(0.10) with max_approach_step_m=0.010
    # gives 11 waypoints, so the descent carries 9 interior vias.
    approach = np.vstack([standoff + step * 0.022 * descent for step in range(11)])
    return transit, approach


def _path_speed(trajectory):
    """Per-interval scalar path speed, in rad/s of arclength."""

    return (
        np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1)
        / np.diff(trajectory.times_s)
    )


def _seam_speed(transit, approach):
    """Slowest commanded speed either side of the standoff, and the reach peak."""

    before, after = _path_speed(transit), _path_speed(approach)
    return min(before[-1], after[0]), max(before.max(), after.max())


def _fused(transit, approach):
    """The two slices read back as the single trajectory they were cut from."""

    return (
        np.vstack((transit.positions, approach.positions[1:])),
        np.concatenate((transit.times_s, approach.times_s[1:] + transit.times_s[-1])),
    )


def _corner_speed_ceiling(deployed, transit, approach):
    """Speed the acceleration cap alone permits through the standoff corner.

    Mirrors ``retime_path``'s corner rule: the joint-rate swing across the
    direction change must fit ``corner_acceleration_budget`` of the cap inside
    half a sample period.
    """

    settings = deployed.config.time_parameterization
    incoming = transit[-1] - transit[-2]
    incoming /= np.linalg.norm(incoming)
    outgoing = approach[1] - approach[0]
    outgoing /= np.linalg.norm(outgoing)
    cap = (
        np.asarray(deployed.config.robot.acceleration_limits)
        * settings.acceleration_scale
    )
    swing = np.abs(outgoing - incoming)
    worst = float(np.max(
        swing
        / (settings.corner_acceleration_budget * cap * 0.5 * settings.sample_period_s),
    ))
    return np.inf if worst < 1e-12 else 1.0 / worst


def test_a_straight_standoff_is_crossed_at_speed_not_stopped(deployed, stepwise):
    """THE regression guard: same positions, one moving through, one stopped.

    With the transit arriving straight down the descent line the standoff is
    not a corner at all, so nothing physical may slow the arm there.  Measured
    on the deployed config: fused 0.3616 rad/s = 62.5% of the reach peak,
    against 0.000524 rad/s = 0.08% for two independently retimed legs -- a
    690x separation that no position check can see, because both land on the
    same standoff joints to the bit.
    """

    transit, approach = _reach(corner_deg=0.0)
    smooth, peak = _seam_speed(*deployed.retime_reach(transit, approach))
    stopped, legacy_peak = _seam_speed(*stepwise.retime_reach(transit, approach))

    assert smooth > 0.5 * peak, (
        f"the standoff is still a halt: {smooth:.6f} rad/s = "
        f"{smooth / peak:.2%} of the reach peak {peak:.4f}"
    )
    # The control must actually reproduce the bug, or the test above proves
    # nothing about what changed.
    assert stopped < 0.01 * legacy_peak, (
        f"per-leg retiming no longer stops at the standoff ({stopped:.6f} rad/s)"
    )
    # And the seam is still the exact joint the operator authorized.
    assert np.array_equal(
        deployed.retime_reach(transit, approach)[0].positions[-1],
        transit[-1],
    )


def test_a_real_standoff_corner_is_slowed_only_by_the_acceleration_cap(
    deployed,
    stepwise,
):
    """The 56.6 deg corner DOES slow the arm -- by the cap, not by the retimer.

    An exact polyline corner cannot be flown at speed under a finite
    acceleration limit; the fix is not supposed to pretend otherwise.  What it
    must deliver is that the corner ceiling is the ONLY thing holding the arm
    back.  Measured: ceiling 0.006660 rad/s, fused seam 0.008381 (1.26x the
    ceiling, 2.61% of peak), per-leg seam 0.000833 (0.13x the ceiling, 0.16%).
    """

    transit, approach = _reach(corner_deg=56.6)
    ceiling = _corner_speed_ceiling(deployed, transit, approach)
    smooth, _ = _seam_speed(*deployed.retime_reach(transit, approach))
    stopped, _ = _seam_speed(*stepwise.retime_reach(transit, approach))

    assert smooth >= 0.5 * ceiling, (
        f"seam {smooth:.6f} rad/s is below the {ceiling:.6f} rad/s the "
        "acceleration cap already permits, so something else is stopping it"
    )
    assert stopped < 0.25 * ceiling, (
        f"per-leg retiming no longer stops at the standoff ({stopped:.6f} rad/s)"
    )


@pytest.mark.parametrize("corner_deg", (0.0, 56.6))
def test_the_whole_reach_is_one_time_parameterized_trajectory(deployed, corner_deg):
    """Transit and approach are two slices of one schedule, not two schedules.

    Reads the slices back end to end and requires a strictly increasing clock,
    no duplicated sample at the join, and both dynamic caps respected ACROSS
    the join rather than merely inside each leg.
    """

    transit, approach = _reach(corner_deg)
    first, second = deployed.retime_reach(transit, approach)
    settings = deployed.config.time_parameterization

    assert second.times_s[0] == 0.0
    assert np.array_equal(second.positions[0], first.positions[-1])
    positions, times = _fused(first, second)
    assert np.all(np.diff(times) > 0.0), "the fused schedule is not monotonic"

    velocity = np.diff(positions, axis=0) / np.diff(times)[:, None]
    acceleration = np.diff(velocity, axis=0) / np.diff(times[:-1])[:, None]
    velocity_cap = deployed.chain.velocity_limits * settings.velocity_scale
    acceleration_cap = (
        np.asarray(deployed.config.robot.acceleration_limits)
        * settings.acceleration_scale
    )
    assert np.all(np.max(np.abs(velocity), axis=0) <= velocity_cap * 1.01)
    assert np.all(np.max(np.abs(acceleration), axis=0) <= acceleration_cap * 1.01)

    # The seam is inside the fused profile, not at either end of it.
    seam = len(first.positions) - 1
    assert 0 < seam < len(positions) - 1


def test_the_arm_never_comes_to_rest_between_start_and_grasp(deployed):
    """"完全smooth" stated as one number: no rest once the reach is moving.

    Measured on the straight-standoff reach, as a fraction of the reach peak,
    over every sample from the end of the launch ramp to the start of the
    arrival ramp: the fused profile's deepest dip is 20.21%, two independently
    retimed legs dip to 0.57% -- the arm stopping dead at the standoff and
    starting again, which is the phase change the operator sees.

    The 56.6 deg reach is deliberately NOT the case under test here: its
    corner is genuinely un-flyable on an exact polyline, so it dips to 2.61%
    against the per-leg 0.93% and the two are only 2.8x apart.  That corner is
    covered by ``..._slowed_only_by_the_acceleration_cap`` instead.
    """

    transit, approach = _reach(corner_deg=0.0)

    def deepest_dip(positions, times):
        speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
        # Ignore the launch and arrival ramps: the reach is allowed to start
        # and finish at rest, it is just not allowed to rest in the middle.
        moving = np.flatnonzero(speed > 0.2 * speed.max())
        return speed[moving[0]:moving[-1] + 1].min() / speed.max()

    fused = deepest_dip(*_fused(*deployed.retime_reach(transit, approach)))
    per_leg = deepest_dip(*_fused(
        deployed._retime_joint_path(transit, allow_stationary_hold=True),
        deployed._retime_joint_path(approach),
    ))

    assert fused > 0.01, f"the reach still rests mid-motion, at {fused:.2%} of peak"
    # The control must reproduce the halt, or the bar above is not a bar.
    assert per_leg < 0.01, (
        f"retiming the legs separately no longer halts the arm ({per_leg:.2%})"
    )


def test_the_nine_descent_vias_are_crossed_at_speed(deployed, stepwise):
    """The "分段式前进" the user sees: 10 x 10 mm steps, 9 interior halts.

    Measured on the deployed config, as a fraction of the descent's own peak
    speed: 44.6% .. 99.6% with the fused profile, against a flat 1.24% for
    every one of the nine when each segment is retimed rest-to-rest.  The
    descent's peak itself goes 0.1074 -> 0.2544 rad/s, because a profile that
    need not brake for each via can spend the whole 0.10 m accelerating.
    """

    transit, approach = _reach(corner_deg=56.6)

    def via_fractions(planner):
        _, descent = planner.retime_reach(transit, approach)
        speed = _path_speed(descent)
        peak = speed.max()
        out = []
        for via in approach[1:-1]:
            index = int(np.argmin(np.linalg.norm(descent.positions - via, axis=1)))
            assert np.allclose(descent.positions[index], via, atol=1e-9), (
                "the profile no longer lands a sample on the checked via"
            )
            index = min(max(index, 1), len(speed) - 1)
            out.append(min(speed[index - 1], speed[index]) / peak)
        return np.asarray(out)

    smooth = via_fractions(deployed)
    assert len(smooth) == 9
    assert np.all(smooth > 0.30), f"descent vias are still halts: {np.round(smooth, 3)}"
    assert np.all(via_fractions(stepwise) < 0.05)


def test_the_slice_endpoints_are_the_authorized_joints(deployed):
    """The executor's seam check (:754-762) must still hold, to the bit.

    Slicing one profile in two is only safe if the cut lands exactly on the
    standoff; a slice that merely landed NEAR it would make the executor nudge
    the arm onto the authorized joint at the one sample it is slowest.
    """

    transit, approach = _reach(corner_deg=56.6)
    first, second = deployed.retime_reach(transit, approach)

    assert np.array_equal(first.positions[0], transit[0])
    assert np.array_equal(first.positions[-1], transit[-1])
    assert np.array_equal(second.positions[0], approach[0])
    assert np.array_equal(second.positions[-1], approach[-1])


def test_retime_reach_refuses_a_disjoint_approach(deployed):
    """A fused profile across a gap would invent an unchecked edge."""

    transit, approach = _reach(corner_deg=56.6)
    approach = approach + 0.05
    with pytest.raises(PlanningError, match="transit endpoint"):
        deployed.retime_reach(transit, approach)
