"""The whole roboplan path, once, on the real robot -- and nobody's word for it.

Every adapter in this package has its own test file and every one of them
passes.  What none of them covers is the SEAM.  The stack composes
``filter_reachable`` -> ``RoboplanIKSolver`` -> ``RoboplanJointPlanner``
(+ shortcutter) -> ``ToppraRetimer``, and the failures that live *between* those
stages are all quiet ones:

*   the two joint planners name and bound their joints from different files --
    the SRDF group for roboplan, the URDF chain for the numpy checker that
    verifies it -- so a reordering is a plausible-looking trajectory to the
    wrong pose, and nothing downstream compares them;
*   the joints the batch declares feasible are what the RRT is asked to reach,
    and roboplan validates endpoints against the Scene's CURRENT positions
    rather than against the request;
*   the retimer is allowed to cut up to ``max_blend_deviation_rad`` off every
    corner, so what reaches the servo is NOT the polyline the planner
    collision-checked (measured here: 9.6e-3 rad off it at the worst sample).

So this file runs the pipeline once and then judges the RESULT rather than the
stages.  Both oracles are the repo's own and share no code with roboplan:
collisions are re-decided by the enforced capsule model
(``tests.test_roboplan_planner._oracle``, i.e. ``fixed_self_collision``'s own
segment math) and the reached pose by ``tests.test_roboplan_ik._pose_residual``,
which deliberately re-derives the residual instead of calling the solver's error
functions.  A pipeline validated by its own collision checker proves nothing.

Failure is part of the pipeline, so the three ways it happens are driven end to
end too, and each has to arrive as a NAMED reason rather than as a crash or a
short list: a pose outside the workspace, a pose buried inside the robot's own
body, and a batch deadline too tight to finish.  Plus the obstacle the capsule
``Scene`` structurally CANNOT see, which is the whole reason the stack wires
``SceneVerifiedJointPlanner`` in front of the search
(``online_planner._joint_planner``).

Everything here needs the real library, so the whole file skips off the 4090 --
the host-side coverage lives in the six per-adapter test modules.  Measured in
the container: 3.6 s for the file, of which 0.65 s is the Scene build.
Deterministic by construction: fixed rng for the candidate corpus,
``config.rrt.seed`` for the search, and SimpleIk's own restarts are pinned off.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.configuration import load_stack_config
from z_manip.kinematics.chain import KinematicChain
from z_manip.models.planner import PlanningError
from z_manip.planning import roboplan_runtime as rt
from z_manip.planning.roboplan_ik import RoboplanIKSolver
from z_manip.planning.roboplan_planner import RoboplanJointPlanner
from z_manip.planning.roboplan_reachability import (
    COLLISION,
    DEADLINE,
    REACHABILITY_REASONS,
    SOLVED,
    UNREACHABLE,
    filter_reachable,
)
from z_manip.planning.roboplan_runtime import RoboplanConfig
from z_manip.planning.roboplan_timing import ToppraRetimer
from z_manip.planning.roboplan_wiring import SceneVerifiedJointPlanner
from z_manip.planning.rrt_connect import JointSpaceRRTConnect
from z_manip.planning_control import PlanningControl

# The two independent oracles, imported rather than copied: both were written
# for exactly this job in a sibling module, and a fourth transcription of the
# capsule math is a fourth place for it to drift.
from tests.test_roboplan_ik import _pose_residual
from tests.test_roboplan_planner import _oracle

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
STACK_CONFIG = ROOT / "configs/go2w_piper.json"

# Grasp candidates whose straight line from home is BLOCKED.  Without this the
# corpus tests ``roboplan_planner``'s straight-line early return and nothing
# else: measured there, only 9 of 340 random in-limit collision-free goals need
# an actual search.  Costs 299 draws / 0.69 s for four goals, once per module.
_DETOUR_GOALS = 4

# 2.5 m along base +x.  The PiPER's reach is ~0.6 m, so this is unreachable by
# an order of magnitude rather than by a margin the solver could argue with.
_OUT_OF_REACH_M = 2.5

# A batch that measures 82 ms in full, cut at 5 ms.  Not a race: the two
# infeasible plants alone cost 25-38 ms EACH, because the gate walks its whole
# Halton ladder before it will say no, so no machine finishes this corpus in
# 5 ms and ``partial`` is guaranteed.
_TIGHT_DEADLINE_S = 0.005

# Half-width of the box only the deployed checker knows about, centred on the
# transit's own midpoint tip position -- i.e. deliberately ON the path roboplan
# just returned.  0.08 m is wider than the 0.04 rad collision resolution can
# step past.
_OBSTACLE_HALF_WIDTH_M = 0.08

# Executor-frame cap tolerance.  Measured on this trajectory: velocity 0.364x,
# acceleration 1.0000000000014x -- TOPP-RA sits exactly ON the acceleration
# constraint, so anything at or below 1.0 would fail on float noise, and
# anything above ~1.002 stops resolving the failures this exists to catch (the
# Hermite fallback measured 1.24-3.70, blending disabled measured 44).
_CAP_TOLERANCE = 1.001

# The retimed samples must leave the planned polyline by at least this much, or
# the collision audit below is re-checking points the planner already checked
# and proves nothing about the blend.  Measured max deviation 9.6e-3 rad against
# the retimer's 0.01 rad blend budget.
_MIN_BLEND_DEVIATION_RAD = 1e-3

_PLAN_TIMEOUT_S = 5.0


def _point_segment_distance(point, start, end):
    axis = end - start
    span = float(axis @ axis)
    alpha = (
        0.0 if span < 1e-18 else float(np.clip((point - start) @ axis / span, 0.0, 1.0))
    )
    return float(np.linalg.norm(point - (start + alpha * axis)))


def _verifier(chain, stack, state_valid):
    """The deployed numpy planner, exactly as ``online_planner`` builds it."""

    return JointSpaceRRTConnect(
        joint_names=chain.joint_names,
        lower_limits=chain.lower_limits,
        upper_limits=chain.upper_limits,
        state_valid=state_valid,
        config=stack.rrt,
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run the whole pipeline once; every test below judges the same result."""

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    if not URDF.exists():
        pytest.skip("robot URDF not checked out")

    rt.clear_scene_cache()
    stack = load_stack_config(
        STACK_CONFIG, environ={"Z_MANIP_ROBOT_URDF": str(URDF)},
    )
    # The deployed config with ``enabled`` flipped and nothing else: every
    # solver tunable, and every limit, is the one the robot runs with.
    config = RoboplanConfig(
        enabled=True, scene_cache_dir=tmp_path_factory.mktemp("roboplan-e2e"),
    )
    solver = RoboplanIKSolver(config, stack.robot, stack.ik)
    handle = solver.handle
    search = RoboplanJointPlanner(handle, config=stack.rrt)
    retimer = ToppraRetimer(handle)
    chain = KinematicChain.from_urdf(URDF, stack.robot.base_link, stack.robot.tip_link)
    colliding = _oracle()
    # What the stack builds when the flag is on: roboplan searches, the deployed
    # numpy checker has the last word.  Its predicate here is the enforced
    # capsule model, which is a deployment's checker minus the point cloud.
    planner = SceneVerifiedJointPlanner(
        search, _verifier(chain, stack, lambda q: not colliding(q)),
    )

    home = handle.extract(handle.seed)
    lower, upper = handle.position_limits
    rng = np.random.default_rng(20260803)
    goals: list[np.ndarray] = []
    while len(goals) < _DETOUR_GOALS:
        candidate = rng.uniform(lower, upper)
        # Input SELECTION, not an oracle: every collision assertion below runs
        # against ``colliding``, which never touches roboplan.
        if colliding(candidate) or search.segment_valid(home, candidate):
            continue
        goals.append(candidate)

    out_of_reach = chain.forward(home).copy()
    out_of_reach[0, 3] += _OUT_OF_REACH_M
    # q_arm = 0 is genuinely self-colliding (mount vs wrist, 27 mm), so its tip
    # pose is a grasp buried inside the robot's own body.
    targets = tuple(
        [chain.forward(q) for q in goals]
        + [chain.forward(np.zeros(handle.dof)), out_of_reach],
    )

    report = filter_reachable(solver, targets)
    detours = [
        verdict
        for verdict in report.feasible
        if not search.segment_valid(home, verdict.solution.joints)
    ]
    assert detours, (
        "no feasible candidate needs a search; the pipeline below would be "
        "exercising the straight-line early return and nothing else"
    )
    chosen = detours[0]
    goal = chosen.solution.joints
    path = planner.plan_joint(home, goal, timeout_s=_PLAN_TIMEOUT_S)
    trajectory = retimer.retime_path(
        path.waypoints,
        chain.velocity_limits,
        np.asarray(stack.robot.acceleration_limits, dtype=float),
        stack.time_parameterization,
    )

    yield SimpleNamespace(
        stack=stack,
        solver=solver,
        handle=handle,
        search=search,
        planner=planner,
        retimer=retimer,
        chain=chain,
        colliding=colliding,
        home=home,
        targets=targets,
        buried=len(targets) - 2,
        out_of_reach=len(targets) - 1,
        report=report,
        chosen=chosen,
        goal=goal,
        path=np.asarray(path.waypoints, dtype=float),
        trajectory=trajectory,
    )
    rt.clear_scene_cache()


def _caps(pipeline):
    settings = pipeline.stack.time_parameterization
    return (
        pipeline.chain.velocity_limits * settings.velocity_scale,
        np.asarray(pipeline.stack.robot.acceleration_limits, dtype=float)
        * settings.acceleration_scale,
    )


# ------------------------------------------------------------ the happy path


def test_every_candidate_is_resolved_to_a_named_reason_and_none_is_dropped(pipeline):
    """The batch's contract, and the one claim its own collision check cannot
    make: a candidate reported feasible is collision-free by the REPO's math.

    ``solved`` is what selects the grasp that gets executed, so a verdict that
    came from roboplan agreeing with itself would be the whole safety argument.
    """

    report = pipeline.report
    assert len(report.verdicts) == len(pipeline.targets)
    assert {verdict.index for verdict in report.verdicts} == set(
        range(len(pipeline.targets)),
    )
    assert not report.partial
    assert all(verdict.reason in REACHABILITY_REASONS for verdict in report.verdicts)

    reasons = {verdict.index: verdict.reason for verdict in report.verdicts}
    assert reasons[pipeline.out_of_reach] == UNREACHABLE
    assert reasons[pipeline.buried] == COLLISION
    assert all(reasons[index] == SOLVED for index in range(_DETOUR_GOALS))

    for verdict in report.feasible:
        assert not pipeline.colliding(verdict.solution.joints), (
            f"candidate {verdict.index} is reported feasible but its joints "
            "collide by the repo's own capsule model"
        )
    # The plant is only a "buried" grasp if it really is buried; otherwise the
    # verdict above is right for the wrong reason.
    assert pipeline.colliding(np.zeros(pipeline.handle.dof))


def test_every_commanded_sample_is_collision_free_by_the_repos_own_capsule_math(
    pipeline,
):
    """The load-bearing safety claim of the whole pipeline.

    Checked on the RETIMED samples, not on the planned waypoints: the corner
    blend moves the arm off the polyline the planner validated (asserted below,
    or this audit would be re-checking points that were already checked), and
    nothing between the retimer and the servo looks again.

    Not decorative.  Mutation-checked against the hazard
    ``roboplan_timing._MAX_COLLINEAR_TOLERANCE_RAD`` was written for -- raising
    that ceiling turns the collinear collapse into a path simplifier and flies
    the chord instead of the detour -- which this fails on (samples 85+ collide)
    and every other check in the pipeline passes.
    """

    positions = pipeline.trajectory.positions
    lower, upper = pipeline.handle.position_limits
    assert np.all(positions >= lower - 1e-12) and np.all(positions <= upper + 1e-12)
    assert positions[0] == pytest.approx(pipeline.home, abs=1e-9)
    assert positions[-1] == pytest.approx(pipeline.goal, abs=1e-9)
    assert pipeline.trajectory.times_s[0] == 0.0
    assert np.all(np.diff(pipeline.trajectory.times_s) > 0.0)

    path = pipeline.path
    deviation = max(
        min(
            _point_segment_distance(sample, first, second)
            for first, second in zip(path, path[1:])
        )
        for sample in positions
    )
    assert deviation > _MIN_BLEND_DEVIATION_RAD, (
        f"retimed samples stay {deviation:.2e} rad from the planned polyline; "
        "this audit is no longer checking anything the planner did not"
    )

    offenders = [
        index for index, q in enumerate(positions) if pipeline.colliding(q)
    ]
    assert not offenders, (
        f"samples {offenders} of {len(positions)} collide by the repo's capsule model"
    )


def test_the_timed_trajectory_holds_the_configured_velocity_and_acceleration_caps(
    pipeline,
):
    """The caps the deployment configured, in the frame the executor commands.

    ``execute_timed_joint_path`` streams sample ``i`` as a position target due
    at ``times[i]``, so velocity is constant across an interval and changes
    between interval MIDPOINTS.  Differencing on interval widths instead
    under-reports the real acceleration by up to 1.95x, which is the size of the
    failure this is here to catch.

    ACCELERATION is what binds under the deployed limits: measured 1.000x
    against 0.364x for velocity, so the velocity line here cannot fail on its
    own on this trajectory -- forcing ``options.velocity_scale = 1.0`` changes
    nothing about the output.  What the caller's velocity request maps onto is
    pinned where it can bite, in
    ``test_roboplan_timing.test_halving_the_requested_limits_halves_the_arm``.
    """

    velocity_cap, acceleration_cap = _caps(pipeline)
    times = pipeline.trajectory.times_s
    speed = np.diff(pipeline.trajectory.positions, axis=0) / np.diff(times)[:, None]
    rate = np.diff(speed, axis=0) / np.diff(0.5 * (times[:-1] + times[1:]))[:, None]

    assert np.max(np.abs(speed) / velocity_cap) <= _CAP_TOLERANCE
    assert np.max(np.abs(rate) / acceleration_cap) <= _CAP_TOLERANCE
    # Differentiation-free cross-check: the arm cannot cover more joint travel
    # than its cap allows in the time the trajectory takes.  Shares no code with
    # the finite differences above, so a common error cannot hide in both.
    travel = np.sum(np.abs(np.diff(pipeline.trajectory.positions, axis=0)), axis=0)
    assert np.all(travel / velocity_cap <= times[-1] * _CAP_TOLERANCE)


def test_the_trajectory_ends_on_the_requested_grasp_pose_within_the_ik_tolerance(
    pipeline,
):
    """The point of the pipeline: the last commanded sample IS the grasp.

    Measured from the URDF chain against the requested target, at the tool
    CONTACT point and with the deployment's free-axis split -- the same gate
    ``RoboplanIKSolver`` accepts on, recomputed rather than asked for.  This is
    the only assertion that ties the four stages together: an IK solution the
    planner subtly failed to reach, or a retimed endpoint that drifted, both
    land here and nowhere else.

    It is a CHAIN check, not a gate check: SimpleIk's own stopping norm is
    tighter than the acceptance tolerances, so neutering
    ``RoboplanIKSolver._accepted`` leaves this passing (measured).  The gate
    itself is pinned in ``tests/test_roboplan_ik.py``.
    """

    settings = pipeline.stack.ik
    position, _, transverse, axial = _pose_residual(
        pipeline.chain.forward(pipeline.trajectory.positions[-1]),
        pipeline.targets[pipeline.chosen.index],
        settings.position_error_offset_tip_m,
        settings.orientation_free_axis,
    )
    assert position < settings.position_tolerance_m
    assert transverse < settings.orientation_tolerance_rad
    assert axial < settings.orientation_free_axis_tolerance_rad


def test_the_shortcutter_is_actually_in_the_loop(pipeline):
    """``shortcut_attempts`` = 100 in the deployed config, and a wiring that
    dropped it would still return a valid, deterministic, collision-free path --
    just a longer one that the arm flies for no reason.  Measured 3.226 -> 2.848
    rad of joint travel (12%) on this query."""

    raw = RoboplanJointPlanner(
        pipeline.handle, config=replace(pipeline.stack.rrt, shortcut_attempts=0),
    )
    unshortcut = np.asarray(
        raw.plan_joint(pipeline.home, pipeline.goal, timeout_s=_PLAN_TIMEOUT_S).waypoints,
    )

    def travel(path):
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    assert travel(pipeline.path) < travel(unshortcut)


def test_the_pipeline_is_reproducible_end_to_end(pipeline):
    """A replanned grasp must not become a different motion between retries --
    the retry budget replans the same request after a transient failure, and a
    second answer means the one that was reviewed is not the one executed."""

    path = np.asarray(
        pipeline.planner.plan_joint(
            pipeline.home, pipeline.goal, timeout_s=_PLAN_TIMEOUT_S,
        ).waypoints,
    )
    assert path.shape == pipeline.path.shape and np.array_equal(path, pipeline.path)

    trajectory = pipeline.retimer.retime_path(
        path,
        pipeline.chain.velocity_limits,
        np.asarray(pipeline.stack.robot.acceleration_limits, dtype=float),
        pipeline.stack.time_parameterization,
    )
    assert np.array_equal(trajectory.positions, pipeline.trajectory.positions)
    assert np.array_equal(trajectory.times_s, pipeline.trajectory.times_s)


def test_the_two_joint_planners_agree_on_joint_order_and_limits(pipeline):
    """roboplan names and bounds the group from the SRDF, the deployed checker
    from the URDF chain.  ``SceneVerifiedJointPlanner`` compares the names; the
    LIMITS nothing compares, and a disagreement there means the verifier is
    validating a path against a box the search never had -- silently, until a
    grasp near a joint stop is rejected or, worse, is not."""

    lower, upper = pipeline.handle.position_limits
    assert pipeline.search.joint_names == pipeline.chain.joint_names
    assert pipeline.chain.lower_limits == pytest.approx(lower)
    assert pipeline.chain.upper_limits == pytest.approx(upper)
    # The retimer maps the caller's per-joint velocity caps onto the Scene's, so
    # a chain/Scene mismatch there silently rescales the whole trajectory.
    assert pipeline.chain.velocity_limits == pytest.approx(
        np.asarray(pipeline.handle.velocity_limits),
    )


# --------------------------------------------------------- the failure paths


def test_a_deadline_too_tight_names_the_unknowns_instead_of_condemning_them(pipeline):
    """A cut batch must lose CANDIDATES, never a grasp's feasibility.

    The failure this pins shipped once: the gate raises the same exception for
    its own per-candidate slice and for the caller's blown batch budget, and
    swallowing both sent a half-evaluated candidate to the reason probes, which
    named it ``collision``.  Measured then: 12 of 12 partial batches turned a
    genuinely solved grasp into an infeasible verdict.
    """

    control = PlanningControl(deadline_s=time.monotonic() + _TIGHT_DEADLINE_S)
    partial = filter_reachable(pipeline.solver, pipeline.targets, control=control)

    # Not truncated: every candidate still comes back, exactly once.
    assert len(partial.verdicts) == len(pipeline.targets)
    assert {verdict.index for verdict in partial.verdicts} == set(
        range(len(pipeline.targets)),
    )
    assert partial.partial
    # Anytime, in the caller's order: what was cut is a suffix of the input.
    assert {
        verdict.index for verdict in partial.verdicts if verdict.reason == DEADLINE
    } == set(range(partial.evaluated, len(pipeline.targets)))

    full = {verdict.index: verdict.reason for verdict in pipeline.report.verdicts}
    for verdict in partial.verdicts:
        assert verdict.reason in (DEADLINE, full[verdict.index]), (
            f"candidate {verdict.index} is {full[verdict.index]} with a budget but "
            f"{verdict.reason} without one; the clock invented a reason"
        )


def test_a_path_through_an_obstacle_only_the_deployed_checker_sees_is_refused(
    pipeline,
):
    """The capsule ``Scene`` holds the robot and nothing else.

    So the search cannot see a perceived obstacle, and the guard in front of it
    is the entire difference between "roboplan plans transit" and "roboplan
    deletes every obstacle from transit planning".  Both halves are asserted:
    the raw search really does drive through the box, and the composed planner
    really does refuse the result -- with a named ``PlanningError``, not a
    truncated path and not a fallback nobody asked for.
    """

    centre = pipeline.chain.forward(
        pipeline.trajectory.positions[len(pipeline.trajectory.positions) // 2],
    )[:3, 3]

    def inside(joints) -> bool:
        tip = pipeline.chain.forward(np.asarray(joints, dtype=float))[:3, 3]
        return bool(np.all(np.abs(tip - centre) < _OBSTACLE_HALF_WIDTH_M))

    assert any(inside(q) for q in pipeline.path), (
        "the roboplan search returned a path clear of the box; this test would "
        "pass without the guard it exists to check"
    )

    blind = SceneVerifiedJointPlanner(
        pipeline.search,
        _verifier(
            pipeline.chain,
            pipeline.stack,
            lambda q: not (pipeline.colliding(q) or inside(q)),
        ),
    )
    with pytest.raises(PlanningError, match="crosses geometry its Scene cannot see"):
        blind.plan_joint(pipeline.home, pipeline.goal, timeout_s=_PLAN_TIMEOUT_S)
