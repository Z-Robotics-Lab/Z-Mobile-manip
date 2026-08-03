"""TOPP-RA retiming: a drop-in for :func:`retime_path`, fail-closed, default OFF.

``retime_path`` samples a rest-to-rest quintic per segment, and its own docstring
concedes the cost: "Stopping at path vertices is conservative".  ``rrt_connect``
densifies every path to ``collision_resolution`` (0.035 rad) before returning it,
so the arm decelerates to a full stop several hundred times on the way to a
grasp.  Measured over eight densified paths through the real model, this adapter
is **11.1x shorter in execution time** for the same caller limits (9.0x worst,
13.1x best); the suite's own fully collision-checked path measures 8.95x, 212.9 s
-> 23.8 s.  Compute cost is a wash: 17.3 ms against the quintic's 19.1 ms.

Three things about roboplan's TOPP-RA are not obvious and every one of them is a
silent failure if you get it wrong.

**1. It reads the limits from the Scene, not from the arguments.**
``TOPPRAOptions`` exposes only a scalar ``velocity_scale`` /
``acceleration_scale`` multiplying the Scene's own per-joint limit vectors, so
the caller's per-joint vectors have to be mapped onto one scalar each:

    scale = min_j( requested_j / scene_j ),  clipped to (0, 1]

The minimum, because a single scalar cannot honour six different ratios and the
only safe rounding is toward the tightest -- every joint then ends up at or
below what the caller asked for.  The clip, because a scale above 1 would drive
the arm past the URDF's own limit.  For velocity this is an ordinary number
(measured 0.2 for a 1.0 rad/s request against the URDF's 5.0).  For acceleration
it is not: ``go2w_sensored.urdf`` declares no acceleration limits, so roboplan
answers DBL_MAX and the ratio lands in the subnormal range (1.5 / 1.798e308 =
8.34e-309).  That is deliberate and it was measured to round-trip exactly --
requested 1.5 rad/s^2, achieved peak 1.500 -- but it is why the Scene's
acceleration vector is read here rather than ``SceneHandle.acceleration_limits``
(which is the deployment's finite value and would map to a scale of ~1.0, i.e.
no acceleration limit at all).  If a build ever ships with flush-to-zero
enabled, that ratio underflows to 0, and TOPP-RA rejects a zero scale outright
rather than running unconstrained.

**2. It abandons corner blending on a densified path, silently.**
``LinearBlend`` rounds each corner within ``max_blend_deviation``; a
near-collinear corner needs a blend arc far longer than the 0.035 rad gap
between densified waypoints, and roboplan then falls back to ``Hermite`` without
saying so.  Measured on a 421-point densified path: every deviation from 1e-6 to
1e-2 produced byte-identical Hermite output.  So near-collinear runs are
collapsed here first (:func:`_collapse_collinear`), which is geometry-preserving
-- no retained point moves, and every dropped point lay within
``collinear_tolerance_rad`` of the chord that replaces it.  Measured: 421 knots
-> 6, worst deviation from the original polyline 1.0e-15 rad.

**3. Its output can miss the acceleration cap by 24-4300%.**
TOPP-RA discretizes its constraint along the path, and a knot between two grid
points is not constrained.  Measured peak acceleration against the requested
cap: LinearBlend on the collapsed path 1.000000 (exact, over 12 paths x 3 sample
periods); Hermite 1.24 on the same collapsed path and 8.99 on the densified one;
LinearBlend with blending disabled 44 (it takes corners at speed -- a step
command to the servo).  Nothing here trusts the solver: the emitted samples are
checked for finiteness, for the caller's caps, and for the model's joint limits,
and a miss raises.  All three have been observed to matter -- ``adaptive``
measured a sample 9.5e-2 rad (5.5 degrees) past a hard joint stop on the same
path, which nothing downstream would have caught either.

**Spline mode.**  Default ``linear-blend``, the only mode measured to hold both
caps while staying a *bounded* distance from the requested path.  ``hermite``
adheres exactly but stops at every waypoint (no better than the quintic) and
still missed the acceleration cap by 24%.  ``cubic`` and ``adaptive`` deviated
0.47 rad -- 27 degrees at the elbow -- from the requested path on a 6-knot path;
``adaptive`` collision-checks that deviation against the Scene, ``cubic`` does
not.  The default is not risk-free either: **the blend cuts every corner by up
to ``max_blend_deviation_rad`` (default 0.01 rad, measured actual 5.1e-3), so
the executed path is not the path the caller collision-checked.**  roboplan
re-checks the blended path against this Scene -- the enforced capsule model,
which is the same model ``rrt_connect`` checks against -- and falls back to
Hermite if it hits, and that fallback is precisely what the output cap check
above catches.  What it cannot catch is an obstacle the Scene does not carry.

**Fail closed.**  Every rejection raises :class:`RetimingFailure` with a named
``reason``; none of them falls back to the quintic.  A caller that asked for
TOPP-RA and silently got quintic timing would be executing a trajectory it did
not compute.  Choosing a fallback is the wiring phase's decision, not this
module's.

``TimeParameterizationConfig.min_segment_time_s`` has no analogue here and is
ignored: TOPP-RA has no per-segment durations, and the acceleration cap already
bounds how abruptly a short move can start.  Pinned by a test so nobody assumes
otherwise.

roboplan is imported lazily inside :func:`_bindings`; this module must import on
a host that has neither roboplan nor pinocchio.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .roboplan_runtime import RoboplanUnavailable, SceneHandle
from .time_parameterization import (
    TimedJointTrajectory,
    TimeParameterizationConfig,
    _validate_limits,
)

__all__ = ["RetimingFailure", "SPLINE_MODES", "ToppraRetimer"]

#: Config-file slug -> roboplan ``SplineFittingMode`` member.  Strings, not the
#: enum, so a deployment can name a mode without importing a 4090-only library.
SPLINE_MODES: Mapping[str, str] = {
    "hermite": "Hermite",
    "cubic": "Cubic",
    "adaptive": "Adaptive",
    "linear-blend": "LinearBlend",
}

# Fraction by which the emitted samples may exceed the caller's caps before this
# adapter refuses the trajectory.  Re-measured over 8 collision-free paths x
# sample periods {20, 10, 5} ms: the default mode sits on the constraint at
# 20 ms (1.000000) and overshoots slightly as the grid refines, worst 1.002332 at
# 5 ms -- TOPP-RA discretizes its constraint and a sample between grid points is
# unconstrained, so the miss GROWS with a finer dt.  The 2% is ~9x that worst
# case and still far under the failure this check exists to catch: the Hermite
# fallback measured 1.24-3.70, blending disabled measured 44.  It is NOT slack
# for the short final sample interval -- that interval never sets the peak
# (measured identical peaks with and without it).
_LIMIT_TOLERANCE = 1.02

# The first and last emitted sample must BE the caller's first and last
# waypoint, not merely near them -- the last one is the grasp pose.  Measured
# drift with every mode: <= 1.1e-15 rad.  1e-8 rad, not something looser,
# because StagedTrajectorySegment.__post_init__ compares the trajectory's
# endpoints against its declared start/goal at atol=1e-8: anything this adapter
# waves through above that is rejected two layers downstream, by a message that
# blames the segment rather than the retimer.
_ENDPOINT_TOLERANCE_RAD = 1e-8

# A sample may sit this far outside the model's joint limits before the
# trajectory is refused.  Not decorative: measured, ``adaptive`` on the suite's
# own 542-waypoint path emits a sample 9.5e-2 rad (5.5 degrees) past a hard
# joint stop, and neither roboplan nor execute_timed_joint_path checks -- the
# latter validates only that samples are finite and increasing in time.  1e-6
# rad absorbs spline noise on a waypoint that legitimately sits ON its limit.
_JOINT_LIMIT_TOLERANCE_RAD = 1e-6

# Above this, the collinear collapse stops being a de-noiser and becomes a path
# simplifier: every dropped point is replaced by a chord up to the tolerance
# away, and NOTHING re-checks that chord -- roboplan collision-checks the blend
# it fits, not the knots it was handed.  Measured float residue from
# rrt_connect's `first + alpha * delta` densification is 1e-15 rad, so 1e-6
# leaves nine orders of headroom while staying 4.5 orders under the 0.035 rad
# resolution at which that path was collision-checked in the first place.
_MAX_COLLINEAR_TOLERANCE_RAD = 1e-6


class RetimingFailure(RuntimeError):
    """TOPP-RA could not produce a trajectory this adapter is willing to return.

    ``reason`` is a stable slug the wiring phase may branch on:
    ``toppra-rejected`` (the solver refused the path, or returned fewer than two
    samples, or returned a non-finite one), ``limit-mapping`` (the caller's
    limits do not map onto the Scene's), ``time-base`` (times do not start at
    zero and strictly increase), ``path-drift`` (an endpoint moved),
    ``joint-limit`` (a sample lies outside the model's joint limits),
    ``limit-violation`` (the emitted samples exceed the caller's own caps).
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(
            f"toppra retiming failed ({reason})" + (f": {detail}" if detail else ""),
        )
        self.reason = reason


def _bindings() -> tuple[object, object]:
    """Import roboplan's ``core`` and ``toppra`` modules.

    The only import site, and the seam the host-side tests replace with a fake.
    """

    try:
        import roboplan.core as core  # noqa: PLC0415 -- 4090-only dependency
        import roboplan.toppra as toppra  # noqa: PLC0415
    except ImportError as error:
        raise RoboplanUnavailable(
            "import-failed",
            f"{type(error).__name__}: {error}",
        ) from error
    return core, toppra


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    axis = end - start
    span = float(axis @ axis)
    if span < 1e-30:
        return np.linalg.norm(points - start, axis=1)
    alpha = np.clip((points - start) @ axis / span, 0.0, 1.0)
    return np.linalg.norm(points - (start + alpha[:, None] * axis), axis=1)


def _collapse_collinear(path: np.ndarray, tolerance: float) -> np.ndarray:
    """Drop waypoints that lie within ``tolerance`` of the chord replacing them.

    A corridor walk, not a per-point test: a point is dropped only if it -- and
    every point already dropped since the last retained one -- stays inside the
    corridor around the chord that will actually be flown.  Dropping points one
    at a time against their immediate neighbours would let the error accumulate
    along a gently curved path until the result bore no relation to the input.

    See the module docstring for why this happens at all.  ``tolerance <= 0``
    keeps every waypoint, which is the honest way to opt out.
    """

    if tolerance <= 0.0 or len(path) < 3:
        return path
    # ponytail: O(n * run) -- each candidate re-checks the whole open run, which
    # is 4-7 ms on the 400-600 point paths rrt_connect emits (TOPP-RA itself is
    # 11 ms) but quadratic if a path ever arrives with thousands of collinear
    # points.  Upgrade path is Douglas-Peucker, same deviation guarantee, n log n.
    kept = [0]
    anchor = 0
    for index in range(1, len(path) - 1):
        if np.any(
            _point_segment_distance(
                path[anchor + 1:index + 2], path[anchor], path[index + 1],
            )
            > tolerance
        ):
            kept.append(index)
            anchor = index
    kept.append(len(path) - 1)
    return path[kept]


def _limit_scale(requested: np.ndarray, available: np.ndarray, label: str) -> float:
    """Map per-joint caps onto TOPP-RA's single scalar scale.  See the docstring."""

    if available.shape != requested.shape:
        raise RetimingFailure(
            "limit-mapping",
            f"scene reports {available.shape} {label} limits for a "
            f"{requested.shape} request",
        )
    if not np.all(np.isfinite(available)) or np.any(available <= 0.0):
        raise RetimingFailure(
            "limit-mapping",
            f"scene {label} limits are not finite and positive: {available}",
        )
    scale = float(np.min(requested / available))
    # Underflow to zero is the flush-to-zero build described in the docstring;
    # TOPP-RA would reject it anyway, but not with a reason anyone can act on.
    if not np.isfinite(scale) or scale <= 0.0:
        raise RetimingFailure(
            "limit-mapping",
            f"requested {label} limits map onto a scale of {scale!r}",
        )
    return min(scale, 1.0)


class ToppraRetimer:
    """A :func:`retime_path`-shaped retimer backed by roboplan's TOPP-RA.

    ``retimer.retime_path`` takes the same arguments in the same order and
    returns the same :class:`TimedJointTrajectory`, so a caller holding one is a
    flag away from the quintic.  It raises the same ``ValueError`` with the same
    message for the same bad input, plus one the quintic cannot have: the joint
    group has a fixed arity, so a path of the wrong width is rejected here
    rather than mis-scaled.

    It also refuses two trajectories the quintic would return, both because it
    knows the model and the quintic does not: samples outside the group's joint
    limits (``joint-limit``), which includes a *caller* path that was already
    outside them, and samples that are not finite.  Both are ``RetimingFailure``
    with a named reason, never a silent repair.
    """

    def __init__(
        self,
        handle: SceneHandle,
        *,
        mode: str = "linear-blend",
        max_blend_deviation_rad: float = 0.01,
        collinear_tolerance_rad: float = 1e-9,
    ) -> None:
        if mode not in SPLINE_MODES:
            raise ValueError(
                f"unknown spline fitting mode {mode!r}; expected one of "
                f"{sorted(SPLINE_MODES)}",
            )
        # 0.01 rad is roboplan's own default and measured out at 5.1e-3 rad of
        # actual corner cutting.  Zero is allowed and means "fly the polyline
        # exactly", but a polyline is flown through its corners at speed:
        # measured 44x the acceleration cap, which fails closed below.
        if not np.isfinite(max_blend_deviation_rad) or max_blend_deviation_rad < 0.0:
            raise ValueError("max blend deviation must be finite and non-negative")
        # 1e-9 rad is pure float noise from rrt_connect's `first + alpha * delta`
        # densification (measured residual 1e-15), seven orders below its 0.035
        # rad step and six below the tightest enforced self-collision clearance
        # (3.8 mm at home).  It is a de-noiser, not a path simplifier, and the
        # ceiling above keeps it one: a chord that replaces real corners is a
        # path nobody collision-checked.
        if (
            not np.isfinite(collinear_tolerance_rad)
            or not 0.0 <= collinear_tolerance_rad <= _MAX_COLLINEAR_TOLERANCE_RAD
        ):
            raise ValueError(
                "collinear tolerance must be finite and within "
                f"[0, {_MAX_COLLINEAR_TOLERANCE_RAD}] rad; simplify the path "
                "before retiming it if you mean to move it further than that",
            )
        self._handle = handle
        self.mode = mode
        self.max_blend_deviation_rad = float(max_blend_deviation_rad)
        self.collinear_tolerance_rad = float(collinear_tolerance_rad)

    def retime_path(
        self,
        waypoints: object,
        velocity_limits: object,
        acceleration_limits: object,
        config: TimeParameterizationConfig | None = None,
    ) -> TimedJointTrajectory:
        """Time-parameterize ``waypoints`` under the caller's per-joint caps."""

        settings = config or TimeParameterizationConfig()
        path = np.asarray(waypoints, dtype=float)
        # Mirrored from retime_path, message for message, in the same order --
        # tests/test_roboplan_timing.py pins the parity by running both.
        if path.ndim != 2 or len(path) < 2:
            raise ValueError("joint path needs at least two waypoints")
        if not np.all(np.isfinite(path)):
            raise ValueError("joint path contains a non-finite value")
        dof = path.shape[1]
        velocity = (
            _validate_limits(velocity_limits, dof, "velocity") * settings.velocity_scale
        )
        acceleration = (
            _validate_limits(acceleration_limits, dof, "acceleration")
            * settings.acceleration_scale
        )
        if dof != self._handle.dof:
            raise ValueError(
                f"joint path has {dof} columns but group "
                f"{self._handle.group!r} has {self._handle.dof} joints",
            )
        if np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) < 1e-12):
            raise ValueError("joint path contains no motion")

        core, toppra = _bindings()
        options = toppra.TOPPRAOptions()
        options.dt = settings.sample_period_s
        options.mode = getattr(toppra.SplineFittingMode, SPLINE_MODES[self.mode])
        options.max_blend_deviation = self.max_blend_deviation_rad
        # handle.velocity_limits IS the Scene's upper vector, so no lock needed;
        # acceleration is not, which is the whole point below.
        options.velocity_scale = _limit_scale(
            velocity, np.asarray(self._handle.velocity_limits, dtype=float), "velocity",
        )
        knots = _collapse_collinear(path, self.collinear_tolerance_rad)

        with self._handle.exclusive() as scene:
            # NOT handle.acceleration_limits: that is the deployment's finite
            # value, and TOPP-RA scales the Scene's DBL_MAX one.  See docstring.
            options.acceleration_scale = _limit_scale(
                acceleration,
                np.asarray(
                    scene.getAccelerationLimitVectors(self._handle.group)[1], dtype=float,
                ),
                "acceleration",
            )
            joint_path = core.JointPath()
            joint_path.joint_names = list(
                scene.getJointGroupInfo(self._handle.group).joint_names,
            )
            joint_path.positions = [np.asarray(row, dtype=float) for row in knots]
            try:
                result = toppra.PathParameterizerTOPPRA(
                    scene, self._handle.group,
                ).generate(joint_path, options)
            except Exception as error:  # noqa: BLE001 -- see below
                # Deliberately broad, and only around the solve: measured,
                # roboplan raises RuntimeError ("Path must have at least 2
                # points"), ValueError ("initialGridpoints should be
                # monotonically increasing") and IndexError (vector range check,
                # adaptive mode) for path rejections.  All of them mean the same
                # thing to a caller.  A binding that changed shape underneath us
                # is NOT one of them and must stay loud, hence the narrow scope.
                raise RetimingFailure(
                    "toppra-rejected", f"{type(error).__name__}: {error}",
                ) from error

        return self._audit(
            np.asarray(result.positions, dtype=float),
            np.asarray(result.times, dtype=float),
            path,
            velocity,
            acceleration,
        )

    def _audit(
        self,
        positions: np.ndarray,
        times: np.ndarray,
        path: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> TimedJointTrajectory:
        """Refuse anything the solver returned that the caller did not ask for."""

        if positions.ndim != 2 or len(positions) < 2 or positions.shape != (
            len(times), path.shape[1],
        ):
            # Measured: a path TOPP-RA cannot blend can come back as a single
            # sample rather than as an error.
            raise RetimingFailure(
                "toppra-rejected",
                f"returned {positions.shape} positions and {times.shape} times",
            )
        if not np.all(np.isfinite(positions)):
            # First, because NaN defeats every check below by construction: it
            # compares False against any threshold.  Measured against this audit
            # before the guard existed -- an all-NaN reply, grasp pose included,
            # passed the drift check, the limit check and the cap check and was
            # returned to the caller.
            raise RetimingFailure(
                "toppra-rejected",
                f"{int(np.count_nonzero(~np.isfinite(positions)))} of "
                f"{positions.size} emitted joint values are not finite",
            )
        if (
            not np.all(np.isfinite(times))
            or times[0] != 0.0
            or not np.all(np.diff(times) > 0.0)
        ):
            # times[0] must be 0: StagedGraspPlan.flattened() concatenates stages
            # by adding the previous stage's duration, so a non-zero start is a
            # silent gap in the middle of a grasp.
            raise RetimingFailure(
                "time-base",
                f"times must start at 0 and strictly increase, got "
                f"[{times[0]!r} .. {times[-1]!r}]",
            )
        drift = max(
            float(np.max(np.abs(positions[0] - path[0]))),
            float(np.max(np.abs(positions[-1] - path[-1]))),
        )
        if drift > _ENDPOINT_TOLERANCE_RAD:
            raise RetimingFailure(
                "path-drift", f"endpoint moved by {drift:.3e} rad",
            )
        lower, upper = self._handle.position_limits
        excursion = float(
            np.max(np.maximum(positions - upper, lower - positions)),
        )
        if excursion > _JOINT_LIMIT_TOLERANCE_RAD:
            # A spline is not obliged to stay inside the hull of its knots.  The
            # blend is (it cuts corners inward), which is why the default mode
            # never trips this; ``adaptive`` measured 9.5e-2 rad outside.
            raise RetimingFailure(
                "joint-limit",
                f"a sample lies {excursion:.3e} rad outside the model limits",
            )

        # The executor streams sample i as a position target due at times[i]
        # (piper_staged_grasp_executor.execute_timed_joint_path), so the
        # commanded velocity is constant across an interval and changes between
        # interval MIDPOINTS -- the spacing below.  On TOPP-RA's uniform grid it
        # equals times[i+1]-times[i], but the final interval is short (measured
        # 0.8 ms against a 20 ms nominal) and using the interval width there
        # under-reports the real acceleration by up to 1.95x (measured), which is
        # the size of the failure this check exists to catch.
        speed = np.diff(positions, axis=0) / np.diff(times)[:, None]
        peak_velocity = float(np.max(np.abs(speed) / velocity))
        peak_acceleration = 0.0
        if len(times) > 2:
            midpoints = 0.5 * (times[:-1] + times[1:])
            rate = np.diff(speed, axis=0) / np.diff(midpoints)[:, None]
            peak_acceleration = float(np.max(np.abs(rate) / acceleration))
        if max(peak_velocity, peak_acceleration) > _LIMIT_TOLERANCE:
            raise RetimingFailure(
                "limit-violation",
                f"emitted peak velocity {peak_velocity:.4f}x and acceleration "
                f"{peak_acceleration:.4f}x the requested cap",
            )
        return TimedJointTrajectory(positions, times)
