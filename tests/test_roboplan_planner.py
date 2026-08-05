"""``RoboplanJointPlanner`` must be swappable for ``JointSpaceRRTConnect``.

The failures worth testing here are the quiet ones.  A planner that crashes gets
noticed; these do not:

1.  **The Scene's current joint positions.**  roboplan's ``RRT`` reads every
    *non-group* joint from ``Scene::getCurrentJointPositions``, not from the
    request.  A freshly built Scene's are invalid -- the four Go2-W wheels are
    continuous joints carried as ``(cos, sin)`` and come back all-zero -- so a
    perfectly good start is rejected as "Invalid start configuration requested".
    Read as a grasp candidate being infeasible, that looks like bad luck.

2.  **``max_planning_time`` is not a bound.**  Measured on the real model: a
    0.30 s budget with a 1e-5 rad collision step returned *successfully* after
    8.2 s, because the budget is only checked between iterations.  Handing
    roboplan the caller's deadline is therefore not enough; the adapter has to
    re-check it after the uninterruptible C++ call returns.  And ``0.0`` means
    UNBOUNDED, so a spent budget that rounds to zero is the worst case of all.

3.  **A path nobody re-checked.**  Shortcutting, densification and the
    6-vector -> 28-vector scatter all happen after roboplan stopped looking.

Everything above the ``real roboplan`` divider runs on a host with no roboplan,
against a fake whose Scene has the real model's shape (nq=28, arm at q[20:26],
wheels as (cos, sin)) driven by a caller-supplied collision predicate.  Below the
divider, the collision oracle is the repo's OWN capsule math -- an oracle that
called back into roboplan would prove nothing about roboplan.
"""

from __future__ import annotations

import inspect
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import z_manip.planning.roboplan_planner as rp
from z_manip.fixed_self_collision import _segment_distance
from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning.grasp_pipeline import JointPlanner
from z_manip.planning.roboplan_planner import RoboplanJointPlanner
from z_manip.planning.roboplan_runtime import RoboplanUnavailable, SceneHandle
from z_manip.planning.rrt_connect import JointSpaceRRTConnect, RRTConnectConfig, RRTTimeout
from z_manip.planning_control import (
    PlanningAborted,
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
SRDF = ROOT / "ros2/z_manip_motion/config/piper.srdf"
CAPSULES = ROOT / "configs/piper_collision_capsules.json"
HOME = ROOT / "configs/piper_home.json"

# The real model, measured: nq=28, arm at q[20:26], four continuous wheels.
NQ = 28
ARM_Q = (20, 21, 22, 23, 24, 25)
ARM_V = (16, 17, 18, 19, 20, 21)
WHEEL_Q = tuple(range(12, 20))
LOWER = np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
UPPER = np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.0944])
JOINT_NAMES = tuple(f"piper_joint{i}" for i in range(1, 7))
ACCELERATION = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0])


def _seed_vector(arm: np.ndarray) -> np.ndarray:
    """A valid full configuration: wheels on the unit circle, arm as given."""

    seed = np.zeros(NQ)
    seed[list(WHEEL_Q)] = np.tile((1.0, 0.0), len(WHEEL_Q) // 2)
    seed[list(ARM_Q)] = arm
    return seed


START = np.array([-0.8, 1.0, -1.0, 0.0, 0.0, 0.0])
GOAL = np.array([0.8, 1.0, -1.0, 0.0, 0.0, 0.0])
SEED_FULL = _seed_vector(START)


# --------------------------------------------------------------------- fakes


class _Group:
    def __init__(self, *, continuous: bool = False, names=JOINT_NAMES) -> None:
        self.q_indices = np.array(ARM_Q, dtype=np.int32)
        self.v_indices = np.array(ARM_V, dtype=np.int32)
        self.joint_names = list(names)
        self.has_continuous_dofs = continuous


class FakeScene:
    """Real model shape; collision verdict comes from ``blocked(q_arm)``."""

    def __init__(self, blocked=None, *, continuous: bool = False, names=JOINT_NAMES) -> None:
        self._blocked = blocked or (lambda _arm: False)
        self._continuous = continuous
        self._names = names
        self.current = np.zeros(NQ)  # the degenerate state the docstring warns of
        self.set_calls: list[np.ndarray] = []
        self.collision_checks = 0

    def getJointGroupInfo(self, name):
        assert name == "piper_arm"
        return _Group(continuous=self._continuous, names=self._names)

    def getCurrentJointPositions(self):
        return self.current.copy()

    def setJointPositions(self, q):
        q = np.asarray(q, dtype=float)
        assert q.shape == (NQ,)
        self.current = q.copy()
        self.set_calls.append(q.copy())

    def hasCollisions(self, q, debug=False):
        self.collision_checks += 1
        return bool(self._blocked(np.asarray(q, dtype=float)[list(ARM_Q)]))


class FakePath:
    def __init__(self, positions) -> None:
        self.positions = np.asarray(positions, dtype=float)
        self.joint_names = list(JOINT_NAMES)


class FakeCore:
    """``roboplan.core``: only what the adapter touches."""

    def __init__(self, scene: FakeScene) -> None:
        self._scene = scene
        self.shortcutters: list[dict] = []
        self.shortcut_result = None
        # What the Scene's current positions were when the shortcutter ran: it
        # re-checks every edge it proposes, so it needs the planning base too.
        self.current_at_shortcut: list[np.ndarray] = []

    # -- the adapter's continuous-segment primitive
    def hasCollisionsAlongPath(self, scene, q_start, q_end, step, bisection, endpoints):
        first = np.asarray(q_start, dtype=float)[list(ARM_Q)]
        second = np.asarray(q_end, dtype=float)[list(ARM_Q)]
        steps = max(1, int(np.ceil(float(np.linalg.norm(second - first)) / step)))
        samples = np.linspace(0.0, 1.0, steps + 1)
        return any(
            scene.hasCollisions(_seed_vector(first + a * (second - first)))
            for a in (samples if endpoints else samples[1:-1])
        )

    class JointConfiguration:
        def __init__(self, joint_names, positions) -> None:
            self.joint_names = list(joint_names)
            self.positions = np.asarray(positions, dtype=float)

    def PathShortcuttingOptions(self, **kwargs):
        return kwargs

    def PathShortcutter(self, scene, options):
        self.shortcutters.append(options)
        self.current_at_shortcut.append(scene.getCurrentJointPositions())
        result = self.shortcut_result
        return type(
            "_S", (), {"shortcut": staticmethod(lambda path: result if result is not None else path)}
        )()


class FakeRRTModule:
    """``roboplan.rrt``: records the options, replays a scripted outcome."""

    def __init__(self, scene: FakeScene, outcomes) -> None:
        self._scene = scene
        self.outcomes = list(outcomes)
        self.options: list[dict] = []
        self.seeds: list[int] = []
        # What the Scene's current joint positions were at plan() time.  The
        # whole point of test_scene_positions_are_set_before_planning.
        self.current_at_plan: list[np.ndarray] = []

    def RRTOptions(self, **kwargs):
        return kwargs

    def RRT(self, scene, options):
        module = self

        class _RRT:
            def __init__(self) -> None:
                module.options.append(options)

            def setRngSeed(self, seed):
                module.seeds.append(int(seed))

            def plan(self, start, goal):
                module.current_at_plan.append(scene.getCurrentJointPositions())
                outcome = module.outcomes.pop(0) if module.outcomes else None
                if isinstance(outcome, Exception):
                    raise outcome
                if outcome is None:
                    outcome = np.vstack((start.positions, goal.positions))
                return FakePath(outcome)

        return _RRT()


def _handle(scene: FakeScene) -> SceneHandle:
    return SceneHandle(
        scene,
        group="piper_arm",
        q_indices=np.array(ARM_Q),
        v_indices=np.array(ARM_V),
        position_limits=(LOWER, UPPER),
        velocity_limits=np.full(6, 3.0),
        acceleration_limits=ACCELERATION,
        seed=SEED_FULL,
    )


def _planner(
    monkeypatch,
    *,
    blocked=None,
    outcomes=(),
    config: RRTConnectConfig | None = None,
    continuous: bool = False,
):
    scene = FakeScene(blocked, continuous=continuous)
    core, rrt = FakeCore(scene), FakeRRTModule(scene, outcomes)
    monkeypatch.setattr(rp, "_roboplan", lambda: (core, rrt))
    planner = RoboplanJointPlanner(
        _handle(scene),
        config=config or RRTConnectConfig(seed=17),
    )
    return planner, scene, core, rrt


def _wall(q_arm) -> bool:
    """A slab across joint 1, open only where joint 2 is lifted clear."""

    return abs(float(q_arm[0])) < 0.3 and float(q_arm[1]) < 2.0


#: The one path through :func:`_wall`, used wherever a test needs the fake RRT
#: to succeed.  Every segment stays either outside the slab or above it.
DETOUR = np.array(
    [
        START,
        [-0.8, 2.5, -1.0, 0.0, 0.0, 0.0],
        [0.8, 2.5, -1.0, 0.0, 0.0, 0.0],
        GOAL,
    ]
)


# ------------------------------------------------------------ the seam itself


def test_signatures_match_the_protocol_and_the_planner_it_shadows():
    """Callers pass ``timeout_s``/``control`` by keyword; a renamed or reordered
    parameter type-checks fine and fails at the call site instead."""

    for name in ("plan_joint", "segment_valid"):
        ours = inspect.signature(getattr(RoboplanJointPlanner, name))
        assert ours == inspect.signature(getattr(JointPlanner, name)), name
        assert ours == inspect.signature(getattr(JointSpaceRRTConnect, name)), name


def test_module_imports_without_roboplan_installed():
    """Checked in a subprocess: a sibling test may already have imported it."""

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import z_manip.planning.roboplan_planner as m;"
            " assert 'roboplan' not in sys.modules;"
            " assert m.RoboplanJointPlanner is not None",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("roboplan") is not None,
    reason="roboplan is installed here; the missing-library path cannot be exercised",
)
def test_missing_library_reports_a_named_reason():
    """Fails fast at wiring time, with the slug a caller falls back on."""

    with pytest.raises(RoboplanUnavailable) as error:
        RoboplanJointPlanner(_handle(FakeScene()))
    assert error.value.reason == "import-failed"


def test_continuous_joints_in_the_group_are_refused(monkeypatch):
    """A collapsed (cos, sin) DoF would make ``dof`` stop indexing the path."""

    scene = FakeScene(continuous=True)
    monkeypatch.setattr(rp, "_roboplan", lambda: (FakeCore(scene), FakeRRTModule(scene, ())))
    with pytest.raises(ValueError, match="continuous joints"):
        RoboplanJointPlanner(_handle(scene))


@pytest.mark.parametrize("names", [JOINT_NAMES[:5], JOINT_NAMES + ("piper_joint7",)])
def test_a_group_whose_joint_count_disagrees_with_the_handle_is_refused(monkeypatch, names):
    """The SRDF names the group; the capsule model and the limit vectors size it.

    Editing ``piper_arm`` in piper.srdf without re-deriving the rest is the
    realistic drift, and it is silent: ``joint_names`` labels the returned
    ``JointTrajectory`` while ``dof`` sizes its columns, so a mismatch ships a
    trajectory whose names and numbers refer to different joints -- and
    ``JointConfiguration(list(joint_names), start)`` hands roboplan a
    disagreeing pair on the way in.
    """

    scene = FakeScene(names=names)
    monkeypatch.setattr(rp, "_roboplan", lambda: (FakeCore(scene), FakeRRTModule(scene, ())))
    with pytest.raises(ValueError, match=f"names {len(names)} joints"):
        RoboplanJointPlanner(_handle(scene))


# --------------------------------------------------- the Scene-state gotcha


def test_scene_positions_are_set_before_planning_and_restored_after(monkeypatch):
    """The single non-obvious integration fact -- see docstring point 1."""

    planner, scene, core, rrt = _planner(monkeypatch, blocked=_wall)
    rrt.outcomes.append(DETOUR)
    before = scene.getCurrentJointPositions()

    planner.plan_joint(START, GOAL)

    assert len(rrt.current_at_plan) == 1
    seen = rrt.current_at_plan[0]
    assert seen[list(ARM_Q)] == pytest.approx(START)
    # Not merely "the arm is right": the wheels must be on the unit circle or
    # roboplan rejects the whole configuration as invalid.
    assert np.linalg.norm(seen[list(WHEEL_Q[:2])]) == pytest.approx(1.0)
    # The shortcutter re-checks its own edges, so it is held inside the same
    # window; restoring between the two would silently change what it checks.
    assert core.current_at_shortcut == [pytest.approx(seen)]
    assert scene.getCurrentJointPositions() == pytest.approx(before)


def test_scene_positions_are_restored_even_when_planning_raises(monkeypatch):
    planner, scene, _, rrt = _planner(monkeypatch, blocked=_wall)
    rrt.outcomes.append(RuntimeError("RRT timed out after 5.000000 seconds."))
    before = scene.getCurrentJointPositions()

    with pytest.raises(PlanningError):
        planner.plan_joint(START, GOAL)
    assert scene.getCurrentJointPositions() == pytest.approx(before)


# ------------------------------------------------------ straight-line return


def test_directly_reachable_pair_returns_the_straight_line_without_planning(monkeypatch):
    """Kept from rrt_connect.py:321-325 on purpose: roboplan does NOT guarantee
    it -- its trees grow toward random samples and the shortcutter is
    stochastic, so the trivial query would come back seed-dependent."""

    planner, _, _, rrt = _planner(monkeypatch)
    trajectory = planner.plan_joint(START, GOAL)

    assert not rrt.options, "a directly reachable pair still ran the RRT"
    assert isinstance(trajectory, JointTrajectory)
    assert trajectory.joint_names == JOINT_NAMES
    waypoints = np.asarray(trajectory.waypoints)
    assert waypoints[0] == pytest.approx(START)
    assert waypoints[-1] == pytest.approx(GOAL)
    direction = GOAL - START
    alphas = (waypoints - START) @ direction / (direction @ direction)
    assert np.allclose(waypoints, START + alphas[:, None] * direction, atol=1e-12)
    assert np.all(np.diff(alphas) > 0.0), "the direct path must advance monotonically"


@pytest.mark.parametrize("blocked,name", [(None, "straight line"), (_wall, "searched path")])
def test_dense_output_never_exceeds_the_collision_resolution(monkeypatch, blocked, name):
    """The executor's contract, and what makes the final audit meaningful.

    Both branches, because the straight-line early return is the one taken by
    ~97% of queries (measured, see the ``real`` fixture) and returning its two
    raw endpoints satisfies every other assertion in this file: collinear,
    monotonic, right endpoints, collision-free.  It just hands the executor a
    3 rad jump.
    """

    config = RRTConnectConfig(collision_resolution=0.025, seed=17)
    planner, _, _, rrt = _planner(monkeypatch, blocked=blocked, config=config)
    rrt.outcomes.append(DETOUR)

    waypoints = np.asarray(planner.plan_joint(START, GOAL).waypoints)
    assert bool(rrt.options) is (blocked is not None), name
    assert np.max(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)) <= 0.0251


# --------------------------------------------------------- fail-closed input


@pytest.mark.parametrize("label,bad", [("start", np.zeros(5)), ("goal", np.zeros((1, 6)))])
def test_wrong_shaped_endpoint_names_which_one(monkeypatch, label, bad):
    planner, _, _, _ = _planner(monkeypatch)
    args = (bad, GOAL) if label == "start" else (START, bad)
    with pytest.raises(PlanningError, match=f"{label} joint state must be a finite"):
        planner.plan_joint(*args)


def test_non_finite_endpoint_is_rejected_before_roboplan_sees_it(monkeypatch):
    """roboplan reports a NaN configuration as "in collision"; that is wrong and
    it sends the caller looking for an obstacle that does not exist."""

    planner, _, _, rrt = _planner(monkeypatch)
    with pytest.raises(PlanningError, match="goal joint state must be a finite"):
        planner.plan_joint(START, [0.0, 1.0, -1.0, np.nan, 0.0, 0.0])
    assert not rrt.options


def test_out_of_limit_endpoint_names_the_joint_and_the_bound(monkeypatch):
    """"Out of limits" and "in collision" call for different caller responses."""

    planner, _, _, _ = _planner(monkeypatch)
    bad = GOAL.copy()
    bad[1] = -1.0  # joint2's lower limit is 0.0
    with pytest.raises(PlanningError, match="goal joint state violates joint limits") as error:
        planner.plan_joint(START, bad)
    assert "piper_joint2" in str(error.value)
    assert "0.000000" in str(error.value)


def test_colliding_endpoint_says_in_collision(monkeypatch):
    planner, _, _, rrt = _planner(monkeypatch, blocked=lambda arm: float(arm[0]) > 0.5)
    with pytest.raises(PlanningError, match="goal joint state is in collision"):
        planner.plan_joint(START, GOAL)
    assert not rrt.options, "a colliding goal still ran the RRT"


def test_timeout_must_be_positive(monkeypatch):
    planner, _, _, _ = _planner(monkeypatch)
    with pytest.raises(PlanningError, match="planning timeout must be positive"):
        planner.plan_joint(START, GOAL, timeout_s=0.0)


# ------------------------------------------------------------ error mapping


@pytest.mark.parametrize(
    "message,label",
    [
        ("Invalid start configuration requested, cannot plan!", "start"),
        ("Start configuration is in collision, cannot plan!", "start"),
        ("Invalid goal configuration requested, cannot plan!", "goal"),
        ("Goal configuration is in collision, cannot plan!", "goal"),
    ],
)
def test_endpoint_rejection_from_roboplan_names_the_endpoint(monkeypatch, message, label):
    """These are the real strings roboplan raises (measured).  The adapter has
    already proved both endpoints good, so hitting one means the Scene state is
    wrong -- which must not read as "this grasp is infeasible"."""

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    rrt.outcomes.append(RuntimeError(message))
    with pytest.raises(PlanningError) as error:
        planner.plan_joint(START, GOAL)
    assert not isinstance(error.value, RRTTimeout)
    assert f"{label} joint state rejected by roboplan" in str(error.value)
    assert message in str(error.value)


@pytest.mark.parametrize(
    "message",
    ["RRT timed out after 5.000000 seconds.", "Added maximum number of nodes (4000)."],
)
def test_search_exhaustion_matches_the_numpy_planner_message(monkeypatch, message):
    """``tests/test_rrt_connect.py`` matches "collision-free joint path"; a
    caller that branches on it must keep working after the swap."""

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    rrt.outcomes.append(RuntimeError(message))
    with pytest.raises(PlanningError, match="collision-free joint path") as error:
        planner.plan_joint(START, GOAL)
    assert message in str(error.value)


# ------------------------------------------------- deadline / timeout arithmetic


def _clock(start=100.0):
    now = [start]
    return (lambda: now[0]), now


def _overrun_during_plan(rrt, cell, value):
    """Make ``RRT.plan`` write ``value`` into ``cell[0]`` before it returns.

    Stands in for the one thing the adapter cannot observe from the inside: the
    C++ call is uninterruptible, so a deadline expiring or a cancellation
    arriving mid-call is only visible once it has already returned -- whether it
    returned a path or raised.
    """

    original = rrt.RRT

    def patched(scene, options):
        instance = original(scene, options)
        real_plan = instance.plan

        def plan(start, goal):
            try:
                return real_plan(start, goal)
            finally:
                cell[0] = value

        instance.plan = plan
        return instance

    rrt.RRT = patched


def test_roboplan_gets_the_tighter_of_the_local_timeout_and_the_deadline(monkeypatch):
    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    read, now = _clock()

    rrt.outcomes.append(DETOUR)
    planner.plan_joint(START, GOAL, timeout_s=9.0,
                       control=PlanningControl(deadline_s=now[0] + 2.0, monotonic_fn=read))
    assert rrt.options[-1]["max_planning_time"] == pytest.approx(2.0)

    rrt.outcomes.append(DETOUR)
    planner.plan_joint(START, GOAL, timeout_s=1.5,
                       control=PlanningControl(deadline_s=now[0] + 60.0, monotonic_fn=read))
    assert rrt.options[-1]["max_planning_time"] == pytest.approx(1.5)


@pytest.mark.parametrize("now", [100.0, np.nextafter(100.0, np.inf), 130.0])
def test_a_spent_budget_never_asks_roboplan_for_an_unbounded_search(now):
    """``max_planning_time <= 0`` means UNBOUNDED to roboplan, so the one moment
    we most want the search to stop is the one that would make it infinite.

    ``PlanningControl`` lets the caller own ``monotonic_fn``, so a coarse or
    jumping clock really can land on or past the ceiling.
    """

    assert rp._advisory_budget(100.0, now) == rp._MIN_PLANNING_TIME_S > 0.0
    assert rp._advisory_budget(100.0, 98.5) == pytest.approx(1.5)


def test_a_deadline_crossed_during_the_uninterruptible_call_still_aborts(monkeypatch):
    """``RRT.plan`` is one C++ call: ``cancel_check`` cannot fire inside it and
    ``max_planning_time`` overran by 8.2 s on a measured hostile query.  The
    checkpoint straight after is the only thing standing between a blown
    deadline and a trajectory going to the arm."""

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    now = [100.0]
    control = PlanningControl(deadline_s=105.0, monotonic_fn=lambda: now[0])

    rrt.outcomes.append(DETOUR)
    _overrun_during_plan(rrt, now, 130.0)  # the C++ call ran 25 s past a 5 s deadline
    with pytest.raises(PlanningDeadlineExceeded, match="monotonic deadline"):
        planner.plan_joint(START, GOAL, timeout_s=60.0, control=control)


@pytest.mark.parametrize(
    "message",
    ["RRT timed out after 5.000000 seconds.", "Added maximum number of nodes (4000)."],
)
def test_a_deadline_blown_during_a_FAILED_call_is_not_reported_as_infeasible(
    monkeypatch, message
):
    """The same overrun, but roboplan also failed -- and that inverts the answer.

    roboplan cannot tell "the geometry defeated me" from "you gave me no time",
    so both come back as its own timeout/max-nodes ``RuntimeError``.  Translated
    straight through, that is a *recoverable* ``PlanningError``, and the ten
    ``except PlanningAborted`` sites in grasp_pipeline.py mean recoverable is
    read as "try the next grasp candidate" -- so a request whose deadline has
    already expired keeps planning, one candidate at a time.  The budget, not
    the message, has to decide.
    """

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    now = [100.0]
    control = PlanningControl(deadline_s=105.0, monotonic_fn=lambda: now[0])

    rrt.outcomes.append(RuntimeError(message))
    _overrun_during_plan(rrt, now, 130.0)
    with pytest.raises(PlanningDeadlineExceeded, match="monotonic deadline"):
        planner.plan_joint(START, GOAL, timeout_s=60.0, control=control)


def test_a_cancellation_arriving_during_a_FAILED_call_is_not_reported_as_infeasible(
    monkeypatch,
):
    """Same inversion via ``cancel_check``: nobody wants this request any more,
    so it must abandon rather than come back as one more infeasible candidate."""

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    cancelled = [False]

    rrt.outcomes.append(RuntimeError("Added maximum number of nodes (4000)."))
    _overrun_during_plan(rrt, cancelled, True)
    with pytest.raises(PlanningCancelled):
        planner.plan_joint(
            START, GOAL, control=PlanningControl(cancel_check=lambda: cancelled[0])
        )


def test_a_genuinely_infeasible_query_stays_recoverable_when_the_budget_is_intact(
    monkeypatch,
):
    """The other half of the fix: with time left, roboplan's failure still means
    what it says.  Turning every search failure into ``PlanningAborted`` would
    abandon the whole request on the first hard grasp candidate."""

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    now = [100.0]
    rrt.outcomes.append(RuntimeError("Added maximum number of nodes (4000)."))
    with pytest.raises(PlanningError, match="collision-free joint path") as error:
        planner.plan_joint(
            START,
            GOAL,
            timeout_s=60.0,
            control=PlanningControl(deadline_s=200.0, monotonic_fn=lambda: now[0]),
        )
    assert not isinstance(error.value, PlanningAborted)


def test_local_timeout_stays_a_recoverable_candidate_failure(monkeypatch):
    """Mirrors ``test_rrt_local_timeout_remains_recoverable_candidate_failure``:
    the GRASP retry budget swaps candidates on ``PlanningError`` but abandons
    the request on ``PlanningAborted``.  Getting this backwards silently turns
    one slow plan into a failed task."""

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    now = [100.0]
    monkeypatch.setattr(rp.time, "monotonic", lambda: now[0])
    rrt.outcomes.append(DETOUR)
    _overrun_during_plan(rrt, now, 110.0)
    with pytest.raises(RRTTimeout, match="recoverable local RRT timeout") as error:
        planner.plan_joint(START, GOAL, timeout_s=0.5)
    assert isinstance(error.value, PlanningError)
    assert not isinstance(error.value, PlanningAborted)


# --------------------------------------------------------- output validation


def test_a_path_the_backend_called_valid_is_still_audited(monkeypatch):
    """Shortcutting, densification and the 6 -> 28 scatter all happen after
    roboplan stopped looking; rrt_connect.py:347-354 is why this exists."""

    planner, _, core, rrt = _planner(monkeypatch, blocked=_wall)
    rrt.outcomes.append(DETOUR)
    # A "shortcut" straight through the wall -- exactly what a shortcutter
    # driven by a stale or differently-configured ACM would hand back.
    core.shortcut_result = FakePath(np.vstack((START, GOAL)))

    with pytest.raises(PlanningError, match="final collision audit"):
        planner.plan_joint(START, GOAL)


@pytest.mark.parametrize(
    "positions,match",
    [
        (np.zeros((3, 5)), "path for 6 joints"),
        (np.zeros((1, 6)), "unusable path"),
        (np.array([[np.nan] * 6, [0.0] * 6]), "unusable path"),
    ],
)
def test_a_malformed_returned_path_is_refused(monkeypatch, positions, match):
    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    rrt.outcomes.append(positions)
    with pytest.raises(PlanningError, match=match):
        planner.plan_joint(START, GOAL)


def test_endpoint_drift_in_the_returned_path_is_refused(monkeypatch):
    """A path that does not start where the arm is is a jump command."""

    planner, _, _, rrt = _planner(monkeypatch, blocked=_wall)
    rrt.outcomes.append(np.vstack((START + 0.02, DETOUR[1:])))
    with pytest.raises(PlanningError, match="endpoints are not the requested"):
        planner.plan_joint(START, GOAL)


# ------------------------------------------------------------- config mapping


def test_config_knobs_reach_the_backend_unswapped(monkeypatch):
    """``step_size`` and ``collision_resolution`` are both lengths in radians:
    swapping them plans perfectly well, just at the wrong fidelity."""

    config = RRTConnectConfig(
        step_size=0.11,
        collision_resolution=0.021,
        max_iterations=1234,
        goal_bias=0.13,
        shortcut_attempts=42,
        seed=99,
    )
    planner, _, core, rrt = _planner(monkeypatch, blocked=_wall, config=config)
    rrt.outcomes.append(DETOUR)
    planner.plan_joint(START, GOAL)

    options = rrt.options[-1]
    assert options["max_connection_distance"] == pytest.approx(0.11)
    assert options["collision_check_step_size"] == pytest.approx(0.021)
    assert options["max_nodes"] == 1234
    assert options["goal_biasing_probability"] == pytest.approx(0.13)
    assert options["rrt_connect"] is True
    assert options["group_name"] == "piper_arm"
    assert rrt.seeds == [99]
    assert core.shortcutters[-1] == {
        "group_name": "piper_arm",
        "max_step_size": pytest.approx(0.021),
        "max_iters": 42,
        "seed": 99,
    }


def test_shortcutting_can_be_switched_off(monkeypatch):
    config = RRTConnectConfig(shortcut_attempts=0, seed=17)
    planner, _, core, rrt = _planner(monkeypatch, blocked=_wall, config=config)
    rrt.outcomes.append(DETOUR)
    planner.plan_joint(START, GOAL)
    assert not core.shortcutters


# ------------------------------------------------------------ segment_valid


@pytest.mark.parametrize(
    "first,second",
    [
        (np.zeros(5), GOAL),
        (START, np.zeros(7)),
        (START, [0.0, 1.0, -1.0, 0.0, 0.0, np.inf]),
        (START, np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0])),  # joint2 below its limit
    ],
)
def test_segment_valid_reports_false_rather_than_raising(monkeypatch, first, second):
    """It is a predicate on a candidate edge, called inside search loops; a
    raise here aborts a plan that merely had one bad candidate."""

    planner, _, _, _ = _planner(monkeypatch)
    assert planner.segment_valid(first, second) is False


def test_segment_valid_matches_the_continuous_verdict(monkeypatch):
    planner, _, _, _ = _planner(monkeypatch, blocked=_wall)
    assert planner.segment_valid(START, GOAL) is False
    assert planner.segment_valid(START, np.array([-0.5, 1.0, -1.0, 0.0, 0.0, 0.0])) is True


@pytest.mark.parametrize("endpoint", [0, -1])
def test_segment_valid_checks_both_endpoints_not_just_the_interior(monkeypatch, endpoint):
    """``check_endpoints=True`` is load-bearing, and defaults are not free.

    ``JointSpaceRRTConnect.segment_valid`` tests the first state explicitly and
    sweeps alpha in (0, 1], so both ends are covered.  roboplan's
    ``hasCollisionsAlongPath`` takes ``check_endpoints`` as a keyword with its
    own default; passing it wrong makes an edge that ENDS inside an obstacle
    report valid, and this predicate is what a search loop trusts to reject a
    candidate edge.  The obstacle here is a point at one endpoint only, so
    nothing in the interior can stand in for it.
    """

    other = np.array([-0.5, 1.0, -1.0, 0.0, 0.0, 0.0])
    edge = (START, other)
    blocked_at = edge[endpoint]
    planner, _, _, _ = _planner(
        monkeypatch,
        blocked=lambda arm: bool(np.allclose(arm, blocked_at)),
    )

    assert planner.segment_valid(*edge) is False
    # Guard the guard: the same edge with the obstacle removed is valid, so the
    # False above is the endpoint check and not some unrelated rejection.
    clear, _, _, _ = _planner(monkeypatch)
    assert clear.segment_valid(*edge) is True


def test_segment_valid_does_not_disturb_the_scene(monkeypatch):
    """It is called far more often than plan_joint; leaving the Scene's current
    positions somewhere else would change a peer adapter's answers."""

    planner, scene, _, _ = _planner(monkeypatch, blocked=_wall)
    planner.segment_valid(START, GOAL)
    assert not scene.set_calls


# ============================================================ real roboplan ==

pytestmark_real = pytest.mark.skipif(not URDF.exists(), reason="robot URDF not checked out")


def _oracle():
    """Collision predicate built from the repo's OWN capsule math.

    Deliberately independent of roboplan: an oracle that asked roboplan whether
    roboplan's path was collision-free would assert nothing.  Same enforced
    pairs, same radii, and ``_segment_distance`` is the tested implementation
    ``FixedSelfCollisionGuard`` runs on the robot.
    """

    from z_manip.collision.pointcloud import RobotCollisionModel
    from z_manip.kinematics.chain import KinematicChain

    model = RobotCollisionModel.from_mapping(json.loads(CAPSULES.read_text()))
    chain = KinematicChain.from_urdf(URDF, "piper_base_link", "piper_gripper_base")
    by_name = {capsule.name: capsule for capsule in model.capsules}
    pairs = [tuple(pair) for pair in model.self_collision.pairs]

    def ends(capsule, transforms):
        start_tf, end_tf = transforms[capsule.start_frame], transforms[capsule.end_frame]
        return (
            start_tf[:3, :3] @ np.asarray(capsule.start_offset, float) + start_tf[:3, 3],
            end_tf[:3, :3] @ np.asarray(capsule.end_offset, float) + end_tf[:3, 3],
        )

    def colliding(q_arm) -> bool:
        transforms = chain.link_transforms(np.asarray(q_arm, dtype=float))
        for first, second in pairs:
            a, b = by_name[first], by_name[second]
            if _segment_distance(*ends(a, transforms), *ends(b, transforms)) < a.radius + b.radius:
                return True
        return False

    return colliding


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    """The real Scene plus two goal sets: directly reachable, and not.

    Measured while writing this: of 340 random in-limit, collision-free arm
    configurations, only **9** have a straight line from home that is blocked.
    That number is why the straight-line early return is kept -- it answers
    ~97% of queries in one collision check -- and why every search test below
    must draw from ``detour`` instead of from random goals, or it silently
    tests the early return and nothing else.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    from z_manip.planning import roboplan_runtime as rt

    rt.clear_scene_cache()

    class Robot:
        urdf_path = URDF
        base_link = "piper_base_link"
        tip_link = "piper_gripper_base"
        acceleration_limits = ACCELERATION

    handle = rt.scene_handle(
        rt.RoboplanConfig(
            enabled=True,
            scene_cache_dir=tmp_path_factory.mktemp("roboplan"),
            srdf_path=SRDF,
            collision_model_path=CAPSULES,
            home_path=HOME,
        ),
        Robot(),
    )
    colliding = _oracle()
    home = np.asarray(json.loads(HOME.read_text())["joint_radians"], float)
    lower, upper = handle.position_limits
    # ``segment_valid`` sorts the draws; it is input SELECTION, not the oracle.
    # Every collision assertion below runs against ``colliding``, which never
    # touches roboplan.
    sorter = RoboplanJointPlanner(handle, config=RRTConnectConfig(seed=17))
    rng = np.random.default_rng(5)
    direct, detour = [], []
    while len(detour) < 10 or len(direct) < 2:
        candidate = rng.uniform(lower, upper)
        if colliding(candidate):
            continue
        (direct if sorter.segment_valid(home, candidate) else detour).append(candidate)
    yield SimpleNamespace(
        handle=handle, colliding=colliding, home=home, direct=direct, detour=detour
    )
    rt.clear_scene_cache()


def _plan(real, goal, **config):
    planner = RoboplanJointPlanner(
        real.handle, config=RRTConnectConfig(seed=17, **config)
    )
    return np.asarray(planner.plan_joint(real.home, goal).waypoints)


@pytest.mark.parametrize("index", range(6))
def test_every_returned_sample_is_collision_free_by_the_repos_own_math(real, index):
    """The load-bearing safety claim, checked against the enforced capsules
    rather than against the library that produced the path.

    Drawn from ``detour`` so the RRT, the shortcutter, the densifier and the
    6 -> 28 scatter are all genuinely exercised.
    """

    goal = real.detour[index]
    waypoints = _plan(real, goal)

    lower, upper = real.handle.position_limits
    assert np.all(waypoints >= lower - 1e-12) and np.all(waypoints <= upper + 1e-12)
    assert waypoints[0] == pytest.approx(real.home)
    assert waypoints[-1] == pytest.approx(goal)
    assert np.max(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)) <= 0.0351
    offenders = [i for i, q in enumerate(waypoints) if real.colliding(q)]
    assert not offenders, f"waypoints {offenders} collide by the repo\'s capsule model"


def test_a_directly_reachable_goal_returns_the_straight_line(real):
    """~97% of queries land here, and roboplan's own answer would be a longer,
    seed-dependent path -- see the fixture docstring."""

    goal = real.direct[0]
    waypoints = _plan(real, goal)
    direction = goal - real.home
    alphas = (waypoints - real.home) @ direction / (direction @ direction)
    assert np.allclose(waypoints, real.home + alphas[:, None] * direction, atol=1e-12)
    assert np.all(np.diff(alphas) > 0.0)
    assert not any(real.colliding(q) for q in waypoints)


def test_plans_are_deterministic_under_a_fixed_seed(real):
    """A replanned grasp must not become a different motion between retries."""

    goal = real.detour[6]
    first, second = _plan(real, goal), _plan(real, goal)
    other = np.asarray(
        RoboplanJointPlanner(real.handle, config=RRTConnectConfig(seed=4))
        .plan_joint(real.home, goal).waypoints
    )
    assert first.shape == second.shape and np.allclose(first, second)
    # Guard the guard: if the seed did not reach the sampler, "deterministic"
    # above would hold trivially.
    assert first.shape != other.shape or not np.allclose(first, other)


def test_the_real_scene_is_left_exactly_as_it_was_found(real):
    with real.handle.exclusive() as scene:
        before = np.array(scene.getCurrentJointPositions(), dtype=float, copy=True)
    _plan(real, real.detour[7])
    with real.handle.exclusive() as scene:
        assert scene.getCurrentJointPositions() == pytest.approx(before)


def test_a_colliding_goal_fails_closed_on_the_real_model(real):
    """``q_arm = 0`` self-collides by 27 mm (mount vs wrist)."""

    planner = RoboplanJointPlanner(real.handle, config=RRTConnectConfig(seed=17))
    with pytest.raises(PlanningError, match="goal joint state is in collision"):
        planner.plan_joint(real.home, np.zeros(6))


def test_the_adapter_stops_at_its_deadline_when_roboplan_overruns(real):
    """The hostile query from the module docstring, on the real model.

    A 1e-4 rad collision step with 3 rad edges makes ONE edge check cost most
    of a second, and roboplan only tests ``max_planning_time`` between
    iterations -- so it sails past the budget it was handed.  What must not
    happen is a trajectory being returned after the caller's deadline is gone.
    """

    planner = RoboplanJointPlanner(
        real.handle,
        config=RRTConnectConfig(step_size=3.0, collision_resolution=1e-4, seed=3),
    )
    started = time.monotonic()
    with pytest.raises(RRTTimeout, match="recoverable local RRT timeout"):
        planner.plan_joint(real.home, real.detour[8], timeout_s=0.05)
    overrun = time.monotonic() - started
    # If this ever drops to ~0.05 s, roboplan started honouring the budget and
    # the "advisory only" note in roboplan_planner.py can be retired.
    assert overrun > 0.05, "roboplan respected max_planning_time; re-read the docstring"


def test_node_exhaustion_is_a_recoverable_planning_error(real):
    """Recoverable, not ``PlanningAborted``: the GRASP retry budget swaps
    candidates on one and abandons the request on the other."""

    planner = RoboplanJointPlanner(
        real.handle,
        config=RRTConnectConfig(step_size=0.02, max_iterations=3, seed=17),
    )
    with pytest.raises(PlanningError, match="collision-free joint path") as error:
        planner.plan_joint(real.home, real.detour[9], timeout_s=5.0)
    assert not isinstance(error.value, PlanningAborted)


def test_beats_the_numpy_planner_it_replaces_on_identical_queries(real, capsys):
    """Success rate and latency against ``JointSpaceRRTConnect``: same goals,
    same config, same collision semantics.  Queries that need an actual search,
    because on the trivial ones both planners just return the straight line."""

    config = RRTConnectConfig(seed=17)
    lower, upper = real.handle.position_limits
    goals = real.detour[:6]
    numpy_planner = JointSpaceRRTConnect(
        joint_names=JOINT_NAMES,
        lower_limits=np.asarray(lower),
        upper_limits=np.asarray(upper),
        state_valid=lambda q: not real.colliding(q),
        config=config,
    )
    roboplan_planner = RoboplanJointPlanner(real.handle, config=config)

    def run(planner):
        solved, latencies = 0, []
        for goal in goals:
            started = time.perf_counter()
            try:
                planner.plan_joint(real.home, goal, timeout_s=5.0)
                solved += 1
            except PlanningError:
                pass
            latencies.append(time.perf_counter() - started)
        return solved, float(np.mean(latencies)) * 1e3

    fast_solved, fast_ms = run(roboplan_planner)
    slow_solved, slow_ms = run(numpy_planner)
    with capsys.disabled():
        print(
            f"\n  roboplan {fast_solved}/{len(goals)} @ {fast_ms:.1f} ms mean"
            f" | numpy {slow_solved}/{len(goals)} @ {slow_ms:.1f} ms mean"
            f" ({slow_ms / fast_ms:.0f}x)"
        )
    assert fast_solved >= slow_solved
    # Measured 65-70x across runs.  A bare ``fast_ms < slow_ms`` would still pass
    # if the C++ backend regressed to within a nanosecond of the numpy planner,
    # which is the only reason to carry this adapter at all; 5x leaves an order
    # of magnitude of machine-to-machine headroom and still fails on that.
    assert slow_ms > 5.0 * fast_ms, f"{slow_ms:.1f} ms vs {fast_ms:.1f} ms"
