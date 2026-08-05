"""What ``roboplan_reachability`` gets wrong SILENTLY if it is wrong at all.

The module replaces the seed-FK proxy that ``grasp_pipeline.py:585-615``
documents as having chosen the wrong grasp on every recorded day, so the
failures worth pinning are the ones that would still look like a working filter:

1.  **A truncated batch that reads as a complete one.**  With
    ``max_feasible_plans=1`` the head of this list IS the executed grasp.  A
    blown budget that dropped the tail instead of marking it would have the
    caller "choose" from a silently shortened pool -- exactly the 102-candidates-
    never-evaluated failure being fixed.  ``len(verdicts) == len(targets)``,
    always.

2.  **A ``solved`` verdict that collides.**  Feasibility here is a promise that
    the arm can be commanded to the pose.

3.  **A second acceptance gate.**  ``RoboplanIKSolver`` accepts at the tool
    control point; SimpleIk stops on a tip-origin norm.  If this module gated on
    the latter it would hand the pipeline a feasible list the pipeline rejects,
    and nothing would report the disagreement.

4.  **A mis-named reason.**  Reason strings land in status documents and the
    offline tests replay them; ``unreachable`` (move the base) and
    ``joint-limit`` (re-orient it) prescribe opposite actions.

Everything above the ``real roboplan`` divider runs without the library, against
a fake Scene shaped like the real model (nq=28, arm at q[20:26], v at v[16:22]),
a duck-typed stand-in for the solver, and a fake of the one ``_solve_ik`` seam.
A green suite off the 4090 therefore really does cover the deadline handling,
the reason ladder, the ordering and the collision re-check.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from z_manip.kinematics.robust_ik import IKConfig, IKFailure, IKSolution
from z_manip.planning import roboplan_reachability as rr
from z_manip.planning.roboplan_reachability import (
    COLLISION,
    DEADLINE,
    JOINT_LIMIT,
    REACHABILITY_REASONS,
    SOLVED,
    UNREACHABLE,
    ReachabilityReport,
    ReachabilityVerdict,
    filter_reachable,
)
from z_manip.planning.roboplan_runtime import RoboplanConfig, SceneHandle
from z_manip.planning_control import PlanningCancelled, PlanningControl

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
SRDF = ROOT / "ros2/z_manip_motion/config/piper.srdf"
CAPSULES = ROOT / "configs/piper_collision_capsules.json"
HOME = ROOT / "configs/piper_home.json"

NQ = 28
NV = 24
ARM_Q = (20, 21, 22, 23, 24, 25)
ARM_V = (16, 17, 18, 19, 20, 21)
BASE, TIP = "piper_base_link", "piper_gripper_base"

# Outcome codes, carried in the target's x translation so a fake verdict is a
# pure function of its own target -- which is what the permutation test needs.
CODE_SOLVED = 0
CODE_SOLVED_BUT_COLLIDES = 1
CODE_COLLISION = 2
CODE_JOINT_LIMIT = 3
CODE_UNREACHABLE = 4
# Arm value the fake Scene calls colliding, distinct from the clean solve's 0.1.
_COLLIDING_JOINT = 0.9
# The probes must walk exactly the gate's ladder or a reachable pose gets
# reported as "move the base".  These are the fake gate's OWN restart points,
# and deliberately neither the same count nor the same values as
# ``RoboplanConfig().ik_restarts`` would produce: a test that recomputed the
# ladder from the config would share the implementation's arithmetic and could
# not tell "took the gate's ladder" from "rebuilt something that matches today".
_FAKE_RESTARTS = tuple(np.full(6, 0.05 * i) for i in range(1, 10))
_LADDER = 1 + len(_FAKE_RESTARTS)


def target(code: int, index: int = 0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = (float(code), float(index), 0.25)
    return pose


def _code(pose) -> int:
    return int(round(float(np.asarray(pose)[0, 3])))


class SimClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeScene:
    """Real model shape; collides exactly on the marked arm configuration."""

    def __init__(self, log: list) -> None:
        self.log = log

    def getJointGroupInfo(self, name):
        assert name == "piper_arm"
        return type("Group", (), {"joint_names": [f"joint{i}" for i in range(6)]})()

    def setJointPositions(self, q) -> None:
        self.log.append(("setJointPositions", np.array(q, dtype=float)))

    def hasCollisions(self, q, debug: bool = False) -> bool:
        self.log.append(("hasCollisions", np.array(q, dtype=float)))
        return bool(np.allclose(np.asarray(q)[list(ARM_Q)], _COLLIDING_JOINT))


def _solution(joints: np.ndarray) -> IKSolution:
    return IKSolution(
        joints=joints,
        position_error_m=0.0012,
        orientation_error_rad=0.011,
        manipulability=0.42,
        iterations=0,
        seed_index=0,
        min_joint_limit_margin=0.45,
    )


class FakeSolver:
    """Duck-typed ``RoboplanIKSolver``: the gate, and nothing else."""

    def __init__(self, log: list, clock: SimClock | None = None, cost_s: float = 0.0):
        self.log = log
        self.clock = clock
        self.cost_s = cost_s
        self.config = RoboplanConfig()
        self.base_link, self.tip_link = BASE, TIP
        seed = np.zeros(NQ)
        seed[list(range(12, 20))[::2]] = 1.0  # wheels as valid (cos, sin) pairs
        seed[list(ARM_Q)] = (0.0, 0.8, -0.9, 0.0, 0.3, 0.0)
        self.handle = SceneHandle(
            FakeScene(log),
            group="piper_arm",
            q_indices=np.array(ARM_Q),
            v_indices=np.array(ARM_V),
            position_limits=(np.full(6, -1.0), np.full(6, 1.0)),
            velocity_limits=np.full(6, 3.0),
            acceleration_limits=np.array((1.5, 1.5, 1.5, 2.0, 2.0, 2.0)),
            seed=seed,
        )
        # The gate's own multi-start ladder, which the probes must reuse.
        self._restarts = _FAKE_RESTARTS
        # One buffer reused across calls, like SimpleIk's own solution slot: if
        # the module handed it out uncopied a caller could corrupt the solver.
        self._joints = np.full(6, 0.1)

    def solve(self, target_pose, current=None, *, control=None):
        code = _code(target_pose)
        self.log.append(("gate", code, control))
        if self.clock is not None:
            self.clock.now += self.cost_s
        if code == CODE_SOLVED:
            return _solution(self._joints)
        if code == CODE_SOLVED_BUT_COLLIDES:
            return _solution(np.full(6, _COLLIDING_JOINT))
        raise IKFailure("fake solver: no solution")


def fake_solve_ik(log: list):
    """Stand-in for ``roboplan_ik._solve_ik``, driven by the target's code."""

    def solve_ik(scene, joint_names, frames, goal, start, options):
        code = _code(goal)
        angular = float(options["max_angular_error_norm"])
        log.append(("probe", code, angular, int(options["max_restarts"]),
                    bool(options["check_collisions"]), np.array(start, dtype=float)))
        if code == CODE_COLLISION:
            return True, np.zeros(6)
        if code == CODE_JOINT_LIMIT:
            # Reachable as a POINT, not as a pose: only the relaxed probe finds it.
            return angular > 1.0, np.zeros(6)
        return False, np.zeros(6)

    return solve_ik


@pytest.fixture
def fake(monkeypatch):
    log: list = []
    monkeypatch.setattr(rr, "_solve_ik", fake_solve_ik(log))
    return FakeSolver(log), log


def run(solver, targets, **kwargs) -> ReachabilityReport:
    return filter_reachable(solver, targets, **kwargs)


def _probes(log) -> list:
    return [entry for entry in log if entry[0] == "probe"]


# ------------------------------------------------------------------ never lost


def test_empty_input_returns_an_empty_report_without_touching_the_scene(fake):
    solver, log = fake
    report = run(solver, [])
    assert report.verdicts == ()
    assert report.evaluated == 0
    assert report.partial is False
    assert report.feasible == ()
    assert log == []


def test_a_blown_budget_marks_the_tail_and_never_drops_it(fake):
    """The failure this module exists to remove: a silently shortened pool."""

    _, log = fake
    clock = SimClock()
    solver = FakeSolver(log, clock=clock, cost_s=0.01)
    targets = [target(CODE_SOLVED, i) for i in range(40)]

    report = run(
        solver, targets,
        control=PlanningControl(deadline_s=0.055, monotonic_fn=clock),
    )

    assert len(report.verdicts) == len(targets), "candidates were truncated, not marked"
    assert report.partial is True
    assert report.evaluated == 6
    assert {v.index for v in report.verdicts} == set(range(40))
    unknown = [v for v in report.verdicts if v.reason == DEADLINE]
    assert len(unknown) == 34
    # Unknown is not infeasible: it must sort behind everything that got a real
    # answer, so a caller taking verdicts[0] never gambles on an unevaluated one.
    assert [v.reason for v in report.verdicts[-34:]] == [DEADLINE] * 34
    assert all(v.solution is None for v in unknown)


def test_an_exhausted_budget_still_returns_every_candidate(fake):
    solver, _ = fake
    report = run(
        solver, [target(CODE_SOLVED, i) for i in range(5)],
        control=PlanningControl(deadline_s=-1.0, monotonic_fn=SimClock()),
    )
    assert report.evaluated == 0
    assert report.partial is True
    assert [v.reason for v in report.verdicts] == [DEADLINE] * 5


def _deadline_at(solver, clock, when: float, on_call: int):
    """Make the gate behave like the real one: blow the clock mid-solve.

    ``RoboplanIKSolver.solve`` raises ``PlanningDeadlineExceeded`` for its own
    per-candidate pathology slice AND for the caller's blown batch budget, so a
    fake that only ever raises ``IKFailure`` cannot exercise the branch where the
    two are confused.
    """

    inner = solver.solve
    calls = {"n": 0}

    def solve(target_pose, current=None, *, control=None):
        calls["n"] += 1
        if calls["n"] == on_call:
            clock.now = when
        if control is not None:
            control.checkpoint("gate")
        return inner(target_pose, current, control=control)

    solver.solve = solve
    return solver


def test_a_candidate_the_deadline_cut_through_is_unknown_not_infeasible(fake):
    """The regression that shipped once, and it deleted good grasps.

    The gate raises one exception for two causes.  Swallowing both into "no
    accepted solution" sends a half-evaluated candidate on to the reason probes,
    which name it confidently -- measured on the real model, 12 of 12 partial
    batches turned a genuinely SOLVED grasp into ``collision``, i.e. dropped a
    reachable grasp out of ``report.feasible`` on the list whose head is
    executed.  A cut candidate is UNKNOWN: ``deadline``, no probes, no reason.
    """

    _, log = fake
    clock = SimClock()
    solver = _deadline_at(FakeSolver(log, clock=clock), clock, when=5.0, on_call=3)
    targets = [target(CODE_SOLVED, i) for i in range(4)]

    report = run(solver, targets, control=PlanningControl(deadline_s=1.0, monotonic_fn=clock))

    assert [(v.index, v.reason) for v in report.verdicts] == [
        (0, SOLVED), (1, SOLVED), (2, DEADLINE), (3, DEADLINE),
    ], "a candidate the deadline cut through was given a fabricated reason"
    assert report.evaluated == 2
    assert report.partial is True
    # Not one probe: past the deadline there is nothing left to diagnose with,
    # and 26 probe starts is 52 ms of work spent after the budget was gone.
    assert _probes(log) == []


def test_the_probes_stop_at_the_batch_deadline_rather_than_overrunning_it(fake):
    """The probes are 2 x 13 starts; unbounded, that is 52 ms past the deadline."""

    _, log = fake
    clock = SimClock()
    solver = FakeSolver(log, clock=clock)
    control = PlanningControl(deadline_s=1.0, monotonic_fn=clock)
    # Every probe start costs; the deadline lands part-way through the ladder.
    original = rr._solve_ik

    def ticking(*args, **kwargs):
        clock.now += 0.4
        return original(*args, **kwargs)

    rr._solve_ik = ticking
    try:
        report = run(solver, [target(CODE_UNREACHABLE)], control=control)
    finally:
        rr._solve_ik = original

    assert [v.reason for v in report.verdicts] == [DEADLINE]
    assert len(_probes(log)) <= 3, "the probe ladder ran on past the batch deadline"


def test_cancellation_propagates_rather_than_masquerading_as_a_deadline(fake):
    """A cancelled request is void; returning a partial result would hide that."""

    solver, _ = fake
    with pytest.raises(PlanningCancelled):
        run(
            solver, [target(CODE_SOLVED, i) for i in range(4)],
            control=PlanningControl(cancel_check=lambda: True),
        )


def test_each_candidate_gets_its_own_slice_of_the_batch_budget(fake):
    """One pathological pose may not eat the budget the other 200 need.

    ``RoboplanIKSolver.solve`` spends ``config.ik_timeout_s`` = 0.6 s -- the
    WHOLE deployed budget -- unless it is handed a tighter control.
    """

    solver, log = fake
    control = PlanningControl(deadline_s=100.0, monotonic_fn=SimClock())
    run(solver, [target(CODE_SOLVED, i) for i in range(3)], control=control)

    budgets = [entry[2] for entry in log if entry[0] == "gate"]
    assert len(budgets) == 3
    for budget in budgets:
        assert budget is not None and budget is not control
        assert budget.deadline_s == pytest.approx(rr._SOLVE_TIMEOUT_S)
    assert rr._SOLVE_TIMEOUT_S < solver.config.ik_timeout_s


def test_a_malformed_target_is_rejected_before_any_solve(fake):
    """Discovering it mid-batch would burn the budget and look like a deadline."""

    solver, log = fake
    skewed = np.eye(4)
    skewed[:3, :3] = np.array(((1.0, 0.4, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="reachability target 2"):
        run(solver, [target(CODE_SOLVED), target(CODE_SOLVED), skewed])
    assert log == []


# ------------------------------------------------------------- the reason ladder


@pytest.mark.parametrize(
    ("code", "reason", "gate_calls", "probe_calls"),
    [
        (CODE_SOLVED, SOLVED, 1, 0),
        (CODE_SOLVED_BUT_COLLIDES, COLLISION, 1, 0),
        # The pose probe answers on its first start.
        (CODE_COLLISION, COLLISION, 1, 1),
        # The pose probe exhausts the ladder, then the point probe answers.
        (CODE_JOINT_LIMIT, JOINT_LIMIT, 1, _LADDER + 1),
        (CODE_UNREACHABLE, UNREACHABLE, 1, 2 * _LADDER),
    ],
)
def test_each_failure_gets_its_own_name_at_its_own_cost(
    fake, code, reason, gate_calls, probe_calls,
):
    solver, log = fake
    report = run(solver, [target(code)])
    assert [v.reason for v in report.verdicts] == [reason]
    assert len([e for e in log if e[0] == "gate"]) == gate_calls
    assert len(_probes(log)) == probe_calls


def test_the_reason_probes_never_gate_on_collisions_and_never_use_the_rng(fake):
    """Two properties the probes must hold, both invisible if they break.

    ``check_collisions`` on would make ``collision`` unreachable as a reason --
    the probe would fail for the same cause the gate did and every colliding
    pose would be called ``unreachable``.  ``max_restarts`` above zero would draw
    from the Scene-global RNG (roboplan_ik measured this) and the reason attached
    to a candidate would stop being reproducible.
    """

    solver, log = fake
    run(solver, [target(CODE_UNREACHABLE)])
    probes = _probes(log)
    # Exactly the gate's ladder, twice: a shorter one told an operator to move
    # the base for 18 of 211 poses the arm could already reach.
    assert len(probes) == 2 * _LADDER
    assert all(restarts == 0 for _, _, _, restarts, _, _ in probes)
    assert all(collisions is False for _, _, _, _, collisions, _ in probes)
    # The pose probe uses the configured tolerance; the point probe cannot be
    # gated by orientation at all.
    angles = {angular for _, _, angular, _, _, _ in probes}
    assert angles == {solver.config.ik_orientation_tolerance_rad, rr._ANY_ORIENTATION_RAD}


def test_the_probes_walk_the_gates_own_ladder_not_a_copy_of_the_formula(fake):
    """The starts must come OFF the solver, not be rebuilt from the config.

    ``collision`` means "the same starts that failed with collisions on
    succeeded with them off".  Rebuilding the ladder from
    ``config.ik_restarts`` makes that true only until someone retunes one of the
    two, and the symptom of divergence is a reachable pose reported
    ``unreachable`` -- an operator told to move the base.  So the fake gate's
    ladder is deliberately a different length and different values from what the
    config formula yields; matching it is only possible by reading the gate.
    """

    solver, log = fake
    assert _LADDER != 1 + solver.config.ik_restarts, "the oracle stopped being independent"

    run(solver, [target(CODE_UNREACHABLE)], current=np.full(6, 0.3))
    walked = [entry[5] for entry in _probes(log)][:_LADDER]
    expected = [np.full(6, 0.3), *_FAKE_RESTARTS]
    assert len(walked) == len(expected)
    for got, want in zip(walked, expected):
        assert got == pytest.approx(want)


def test_a_solution_that_collides_is_not_reported_solved(fake):
    """Feasibility is a promise the arm can be commanded there.

    The gate collision-checks internally, so this re-check is redundant while
    every upstream assumption holds -- which is exactly why its removal would
    not show up in any other test.
    """

    solver, log = fake
    report = run(solver, [target(CODE_SOLVED_BUT_COLLIDES)])
    assert [v.reason for v in report.verdicts] == [COLLISION]
    assert report.verdicts[0].solution is None
    checked = [q for name, *rest in log if name == "hasCollisions" for q in rest]
    assert len(checked) == 1
    assert checked[0][list(ARM_Q)] == pytest.approx([_COLLIDING_JOINT] * 6)
    # The rest of the body under that check is the real seed, not a stale pose.
    others = [i for i in range(NQ) if i not in ARM_Q]
    assert checked[0][others] == pytest.approx(solver.handle.seed[others])


def test_every_reason_emitted_is_a_declared_one(fake):
    solver, _ = fake
    codes = (CODE_SOLVED, CODE_SOLVED_BUT_COLLIDES, CODE_COLLISION,
             CODE_JOINT_LIMIT, CODE_UNREACHABLE)
    report = run(solver, [target(code, i) for i, code in enumerate(codes)])
    emitted = {v.reason for v in report.verdicts}
    assert emitted <= set(REACHABILITY_REASONS)
    assert emitted == {SOLVED, COLLISION, JOINT_LIMIT, UNREACHABLE}


# ------------------------------------------------------------------- the seed


def test_the_body_pose_is_pinned_before_the_first_solve(fake):
    """SimpleIk fills non-arm coordinates from the Scene's CURRENT positions.

    ``RoboplanIKSolver`` does not pin them, so whatever the previous adapter left
    there would otherwise decide which body the candidates are checked against.
    """

    solver, log = fake
    run(solver, [target(CODE_SOLVED, i) for i in range(3)])
    names = [entry[0] for entry in log]
    assert names.index("setJointPositions") < names.index("gate")
    assert log[names.index("setJointPositions")][1] == pytest.approx(
        np.asarray(solver.handle.seed),
    )


def test_current_joints_override_the_home_seed(fake):
    solver, log = fake
    seen: list = []
    solver.solve = lambda t, current=None, *, control=None: (
        seen.append(np.array(current, dtype=float)) or _solution(np.full(6, 0.1))
    )
    run(solver, [target(CODE_SOLVED)], current=np.full(6, 0.25))
    assert seen[0] == pytest.approx(np.full(6, 0.25))
    run(solver, [target(CODE_SOLVED)])
    assert seen[1] == pytest.approx(solver.handle.extract(solver.handle.seed))


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(NQ), [0.0] * 6 + [np.nan]])
def test_a_wrong_shaped_current_fails_closed(fake, bad):
    solver, _ = fake
    with pytest.raises(ValueError):
        run(solver, [target(CODE_SOLVED)], current=bad)


# ---------------------------------------------------------------- the ordering


def test_feasible_first_and_otherwise_the_callers_own_order(fake):
    """The caller's order carries the perception score; this module keeps it."""

    solver, _ = fake
    codes = [
        CODE_UNREACHABLE, CODE_SOLVED, CODE_COLLISION, CODE_SOLVED,
        CODE_JOINT_LIMIT, CODE_SOLVED,
    ]
    report = run(solver, [target(code, i) for i, code in enumerate(codes)])

    assert [v.index for v in report.verdicts] == [1, 3, 5, 0, 2, 4]
    assert [v.reason for v in report.verdicts[:3]] == [SOLVED] * 3
    assert [v.index for v in report.feasible] == [1, 3, 5]
    # The three infeasible reasons share a rank: no implied "closer to feasible".
    assert [v.reason for v in report.verdicts[3:]] == [UNREACHABLE, COLLISION, JOINT_LIMIT]


def test_permuting_the_input_permutes_the_output_and_changes_nothing_else(fake):
    solver, _ = fake
    codes = [CODE_SOLVED, CODE_COLLISION, CODE_SOLVED, CODE_UNREACHABLE, CODE_JOINT_LIMIT]
    forward = run(solver, [target(code, i) for i, code in enumerate(codes)])
    reverse = run(solver, [target(code, i) for i, code in enumerate(reversed(codes))])

    assert [codes[v.index] for v in forward.verdicts] == [
        CODE_SOLVED, CODE_SOLVED, CODE_COLLISION, CODE_UNREACHABLE, CODE_JOINT_LIMIT,
    ]
    assert [list(reversed(codes))[v.index] for v in reverse.verdicts] == [
        CODE_SOLVED, CODE_SOLVED, CODE_JOINT_LIMIT, CODE_UNREACHABLE, CODE_COLLISION,
    ]


def test_repeating_a_batch_gives_an_identical_answer(fake):
    solver, _ = fake
    targets = [target(code, i) for i, code in
               enumerate((CODE_SOLVED, CODE_COLLISION, CODE_SOLVED, CODE_UNREACHABLE))]
    first, second = run(solver, targets), run(solver, targets)
    assert [(v.index, v.reason) for v in first.verdicts] == \
           [(v.index, v.reason) for v in second.verdicts]


# ------------------------------------------------------------------ the metrics


def test_a_solved_verdict_carries_the_gates_own_ik_solution(fake):
    """Not recomputed here: the caller is invited to sort by these fields, and a
    second computation of ``manipulability`` is a second definition of it."""

    solver, _ = fake
    solution = run(solver, [target(CODE_SOLVED)]).verdicts[0].solution
    assert isinstance(solution, IKSolution)
    assert solution.joints == pytest.approx(np.full(6, 0.1))
    assert solution.position_error_m == pytest.approx(0.0012)
    assert solution.orientation_error_rad == pytest.approx(0.011)
    assert solution.manipulability == pytest.approx(0.42)
    assert solution.min_joint_limit_margin == pytest.approx(0.45)


# ------------------------------------------------------------ dataclass contracts


def test_a_verdict_cannot_disagree_with_itself():
    with pytest.raises(ValueError, match="disagree"):
        ReachabilityVerdict(0, SOLVED)
    with pytest.raises(ValueError, match="disagree"):
        ReachabilityVerdict(0, UNREACHABLE, object())
    with pytest.raises(ValueError, match="unknown reachability reason"):
        ReachabilityVerdict(0, "infeasible", None)
    with pytest.raises(ValueError, match="non-negative"):
        ReachabilityVerdict(-1, UNREACHABLE)


def test_importing_the_module_does_not_import_roboplan():
    """It has to load on the NUC's host tooling and on any non-4090 machine."""

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import z_manip.planning.roboplan_reachability as m;"
            " assert 'roboplan' not in sys.modules;"
            " assert m.REACHABILITY_REASONS[0] == 'solved'",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------- real roboplan


class Robot:
    """The stack config's ``robot`` block, duck-typed."""

    def __init__(self) -> None:
        self.urdf_path = URDF
        self.base_link = BASE
        self.tip_link = TIP
        self.acceleration_limits = (1.5, 1.5, 1.5, 2.0, 2.0, 2.0)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    from z_manip.planning import roboplan_runtime as rt
    from z_manip.planning.roboplan_ik import RoboplanIKSolver

    rt.clear_scene_cache()
    solver = RoboplanIKSolver(
        RoboplanConfig(
            enabled=True,
            scene_cache_dir=tmp_path_factory.mktemp("reachability"),
            srdf_path=SRDF,
            collision_model_path=CAPSULES,
            home_path=HOME,
        ),
        Robot(),
        # The deployed acceptance gate, configs/go2w_piper.json:ik.
        IKConfig(
            position_tolerance_m=0.006,
            orientation_tolerance_rad=0.06,
            orientation_free_axis_tolerance_rad=0.1745329252,
            orientation_free_axis=(0.0, 0.0, 1.0),
        ),
    )
    yield solver
    rt.clear_scene_cache()


def _fk(handle, arm_configs) -> list[np.ndarray]:
    with handle.exclusive() as scene:
        return [
            np.array(scene.forwardKinematics(handle.expand(q), TIP, BASE), dtype=float)
            for q in arm_configs
        ]


@pytest.fixture(scope="module")
def corpus(real):
    """Targets whose right answer is known before the module runs, INTERLEAVED.

    Four sources:

    ``clear``     FK of collision-free arm configurations: reachable, and the
                  reaching configuration is known not to collide.
    ``colliding`` FK of self-colliding arm configurations: reachable, and at
                  least one reaching configuration is known to collide.
    ``zero``      q_arm = 0, which overlaps mount and wrist by 27 mm.
    ``far``       the home pose translated metres away: not reachable at all.

    They are then shuffled under a fixed seed, and that is not cosmetic.  An
    accepted pose costs 0.95 ms and a rejected one 25-38 ms, so any test with a
    BUDGET measures only the prefix it got through -- and concatenated in
    source order the first 150 entries are all cheap accepts.  Measured on the
    ordered list, the 0.6 s throughput test evaluated 164 targets at 271/s and
    the "deployed pool" test resolved 64 in 0.10 s; on the same corpus
    interleaved, 56 at 93/s and 56-57 of 64 in 0.60 s.  The ordered numbers are the
    all-reachable best case wearing a mixed corpus's name.
    """

    handle = real.handle
    lower, upper = handle.position_limits
    rng = np.random.default_rng(23)
    samples = rng.uniform(lower, upper, size=(500, 6))
    with handle.exclusive() as scene:
        clear = [q for q in samples if not scene.hasCollisions(handle.expand(q))]
        colliding = [q for q in samples if scene.hasCollisions(handle.expand(q))]
    home_pose = _fk(handle, [handle.extract(handle.seed)])[0]
    far = []
    for offset in (2.0, 3.0, 4.0):
        shifted = home_pose.copy()
        shifted[0, 3] += offset
        far.append(shifted)

    # (target, generating arm configuration).  The source is the ground truth a
    # collision verdict is checked against; the far targets have none.
    constructed = list(zip(
        _fk(handle, clear[:150]) + _fk(handle, colliding[:60]) + _fk(handle, [np.zeros(6)]),
        list(clear[:150]) + list(colliding[:60]) + [np.zeros(6)],
    ))
    pairs = constructed + [(pose, None) for pose in far]
    order = np.random.default_rng(7).permutation(len(pairs))
    shuffled = [pairs[i] for i in order]
    return {
        "targets": [pose for pose, _ in shuffled],
        "sources": [source for _, source in shuffled],
        "zero": _fk(handle, [np.zeros(6)]),
        "far": far,
        "constructed": len(constructed),
    }




@pytest.fixture(scope="module")
def truth(real, corpus):
    """What an UNHURRIED batch says about the same targets.

    The oracle for every budgeted run below.  It shares no logic with the
    module's deadline handling -- it is the same module with the deadline taken
    away -- so a reason that differs under a budget is a reason the budget
    invented.
    """

    report = filter_reachable(real, corpus["targets"])
    assert report.partial is False, "the uncapped oracle run was itself truncated"
    return {v.index: v.reason for v in report.verdicts}


def test_every_solved_verdict_really_is_collision_free(real, corpus):
    """The safety direction.  A false ``solved`` commands the arm into itself."""

    handle = real.handle
    targets = corpus["targets"]
    report = filter_reachable(real, targets)

    assert len(report.verdicts) == len(targets)
    assert report.partial is False
    assert report.feasible, "nothing solved at all; the corpus or the seam is broken"
    with handle.exclusive() as scene:
        for verdict in report.feasible:
            q_full = handle.expand(verdict.solution.joints)
            assert not scene.hasCollisions(q_full), (
                f"candidate {verdict.index} reported solved but self-collides"
            )
            assert scene.isValidConfiguration(q_full)
            assert np.isfinite(verdict.solution.manipulability)


def test_a_solved_verdict_is_one_the_pipeline_also_accepts(real, corpus):
    """The feasible list has to be feasible to the gate the pipeline uses.

    Re-solving each solved target through ``RoboplanIKSolver.solve`` directly
    must not raise: if this module gated on SimpleIk's tip-origin stopping norm
    instead of the tool-control-point acceptance, these would diverge and the
    pipeline would reject candidates this filter had promised.
    """

    targets = corpus["targets"]
    report = filter_reachable(real, targets)
    # Without this the whole test is a loop over an empty list -- it passed
    # against a filter that returned nothing feasible at all.
    checked = report.feasible[:40]
    assert len(checked) == 40, f"only {len(checked)} feasible verdicts to re-solve"
    for verdict in checked:
        real.solve(targets[verdict.index])  # raises IKFailure on divergence


def test_every_collision_verdict_really_does_collide(real, corpus):
    """``collision`` claims a reaching configuration exists and self-collides.

    Checked against the configuration each target was GENERATED from rather than
    by re-solving.  Re-solving was the first attempt and it was wrong: any
    verification search weaker than the module's own ladder fails on poses the
    module legitimately found, and the test then reports a module bug that is
    really a test bug.  The generating configuration needs no search -- it
    reaches the target by construction and its collision state is a fact about
    the model.
    """

    handle = real.handle
    targets, sources = corpus["targets"], corpus["sources"]
    report = filter_reachable(real, targets)
    flagged = [v for v in report.verdicts if v.reason == COLLISION]
    assert flagged, "no collision verdicts from a corpus built out of colliding poses"

    with handle.exclusive() as scene:
        for verdict in flagged:
            source = sources[verdict.index]
            assert source is not None, (
                f"candidate {verdict.index} is a deliberately-distant target and "
                "no configuration reaches it; the honest reason is unreachable"
            )
            full = handle.expand(source)
            achieved = np.asarray(
                scene.forwardKinematics(full, TIP, BASE), dtype=float,
            )
            assert np.allclose(achieved, targets[verdict.index], atol=1e-9), (
                f"candidate {verdict.index} was named a collision but no "
                "configuration is known to reach it"
            )
            assert scene.hasCollisions(full), (
                f"candidate {verdict.index} was named a collision but the "
                "configuration reaching it is collision-free"
            )


def test_the_documented_self_colliding_pose_is_named_collision(real, corpus):
    """q_arm = 0 overlaps mount and wrist by 27 mm and is exactly reachable.

    Not a sample: this is the pose ``roboplan_runtime`` refuses to seed from, so
    a verdict of ``solved`` or ``unreachable`` here means the ladder is wired
    wrong, not that the corpus was unlucky.
    """

    assert [v.reason for v in filter_reachable(real, corpus["zero"]).verdicts] == [
        COLLISION,
    ]


def test_a_target_outside_the_workspace_is_named_unreachable(real, corpus):
    """``unreachable`` says move the base; ``joint-limit`` says re-orient it."""

    report = filter_reachable(real, corpus["far"])
    assert [v.reason for v in report.verdicts] == [UNREACHABLE] * len(corpus["far"])


def test_home_pose_is_solved_from_its_own_seed(real):
    """If the commanded home pose is not feasible, nothing downstream can be."""

    handle = real.handle
    report = filter_reachable(real, _fk(handle, [handle.extract(handle.seed)]))
    assert [v.reason for v in report.verdicts] == [SOLVED]
    assert report.verdicts[0].solution.min_joint_limit_margin > 0.0


def test_the_grasp_selection_is_reproducible_run_to_run(real, corpus):
    """Head of this list = the executed grasp, so it may not wander."""

    targets = corpus["targets"]
    first, second = filter_reachable(real, targets), filter_reachable(real, targets)
    assert first.feasible, "nothing feasible; reproducibility of an empty list is free"
    assert [v.index for v in first.feasible] == [v.index for v in second.feasible]
    # Reasons too, not only the feasible split: the whole batch is a
    # deterministic descent from fixed starts once no cap fires.
    assert [(v.index, v.reason) for v in first.verdicts] == [
        (v.index, v.reason) for v in second.verdicts
    ]
    for a, b in zip(first.feasible, second.feasible):
        assert a.solution.joints == pytest.approx(b.solution.joints)


def test_no_reachable_pose_is_ever_told_to_move_the_base(real, corpus):
    """Ground truth, not a sample: all but three corpus targets ARE reachable.

    Every ``constructed`` target is the forward kinematics of a real arm
    configuration, so a configuration reaching it provably exists and
    ``unreachable`` -- "the point is outside the workspace, move the base" -- is
    simply false for any of them.  Only the three deliberately-distant targets
    earn it.  This is the test that forced the probes onto the gate's full start
    ladder: at three starts 18 of these came back ``unreachable``.
    """

    report = filter_reachable(real, corpus["targets"])
    far_indices = {
        index for index, source in enumerate(corpus["sources"]) if source is None
    }
    assert len(far_indices) == 3
    named = {v.index for v in report.verdicts if v.reason == UNREACHABLE}
    wrong = sorted(named - far_indices)
    assert not wrong, (
        f"{len(wrong)} of {corpus['constructed']} reachable poses were reported "
        f"unreachable: {wrong[:10]}"
    )
    assert named == far_indices


def test_a_partial_batch_never_invents_a_reason(real, corpus, truth):
    """The regression that shipped once, on the real model.

    The gate raises ``PlanningDeadlineExceeded`` both for its own per-candidate
    pathology slice and for the caller's blown batch budget.  Swallowing both
    into "no accepted solution" sent the half-evaluated candidate on to the
    reason probes, which named it ``collision`` -- measured, 12 partial batches
    out of 12 turned a genuinely SOLVED grasp into an infeasible one, deleting a
    reachable grasp from ``report.feasible`` on the list whose head is executed.

    Run on the targets the unhurried batch calls ``solved``, and ONLY those,
    because a mixed corpus barely detects this: the deadline lands inside
    whichever solve was in flight, rejects cost 25-38 ms against an accept's
    0.95 ms, so the cut candidate is nearly always an infeasible one -- whose
    fabricated reason coincides with its true one.  Measured against the broken
    version, the mixed corpus caught it in 1 of 10 budgets and this subset in 10
    of 10.  A 10%-sensitive regression test is worse than none.

    On an all-reachable batch the oracle needs no bookkeeping: any verdict that
    is neither ``solved`` nor ``deadline`` is invented.
    """

    reachable = [
        pose for index, pose in enumerate(corpus["targets"]) if truth[index] == SOLVED
    ]
    assert len(reachable) > 100, "not enough reachable targets to cut through"
    for budget_s in (0.02, 0.03, 0.04, 0.06, 0.08):
        report = filter_reachable(
            real, reachable,
            control=PlanningControl(deadline_s=time.monotonic() + budget_s),
        )
        assert report.partial is True, f"{budget_s} s was not actually a tight budget"
        invented = [
            (v.index, v.reason) for v in report.verdicts
            if v.reason not in (SOLVED, DEADLINE)
        ]
        assert not invented, (
            f"{budget_s} s budget: {len(invented)} reachable candidates were "
            f"reported infeasible because the deadline cut through them: "
            f"{invented[:5]}"
        )


def test_the_deployed_hypothesis_pool_inside_the_deployed_budget(real, corpus):
    """The argument, in the deployment's own units -- and its honest margin.

    ``configs/go2w_piper.json`` caps the search at ``max_hypotheses`` = 64 and
    spends ``ik.solve_timeout_s`` = 0.6 s on ONE IK call, with
    ``max_feasible_plans`` = 1 -- so the proxy's ordering of those 64 IS the
    grasp choice, and one of them is then planned.

    Measured here on an interleaved pool (~30% infeasible): 56-57 of 64 resolved
    in 0.60 s (it sits ON the budget, so the last one or two vary run to run).  Not "all 64 with 6x to spare" -- that figure came from a pool of 64
    all-reachable targets, where every solve is the 0.95 ms accept and none is
    the 25-38 ms reject.  56 for real against 1 is still the whole argument, and
    the handful the budget did not reach are marked, not dropped.
    """

    pool = (corpus["targets"] * 4)[:64]
    started = time.perf_counter()
    report = filter_reachable(
        real, pool, control=PlanningControl(deadline_s=time.monotonic() + 0.6),
    )
    elapsed = time.perf_counter() - started
    histogram = Counter(v.reason for v in report.verdicts)

    print(
        f"\ndeployed pool: {report.evaluated} of 64 hypotheses resolved in "
        f"{elapsed:.3f} s of the 0.6 s budget; {dict(histogram)}",
    )
    assert len(report.verdicts) == 64
    # The floor is the claim being defended -- "many more than
    # max_feasible_plans = 1" -- not a regression bound on one loaded container.
    assert report.evaluated >= 32, (
        f"only {report.evaluated} of the deployed 64-hypothesis pool resolved in "
        "0.6 s; solving cannot replace the proxy ranker at this rate"
    )
    assert histogram[SOLVED] > 0
    # Overshoot is one in-flight solver call, not one whole candidate: the gate
    # slice, or a single probe start.
    assert elapsed < 0.6 + 2 * rr._SOLVE_TIMEOUT_S, (
        f"overshot the deadline by {elapsed - 0.6:.3f} s"
    )


def test_throughput_beats_one_plan_per_budget(real, corpus):
    """Past the pool, how far does the budget stretch on a realistic mix?

    Measured on the INTERLEAVED corpus: an accepted pose costs 0.95 ms and a
    rejected one 25-38 ms, so a run that only gets through an all-reachable
    prefix reports ~3x the honest rate (271/s against 93/s, measured).
    """

    targets = (corpus["targets"] * 4)[:600]
    started = time.perf_counter()
    report = filter_reachable(
        real, targets, control=PlanningControl(deadline_s=time.monotonic() + 0.6),
    )
    elapsed = time.perf_counter() - started

    histogram = Counter(v.reason for v in report.verdicts)
    print(
        f"\nreachability throughput: {report.evaluated} candidates fully checked "
        f"in {report.elapsed_s:.3f} s "
        f"({report.evaluated / max(report.elapsed_s, 1e-9):.0f}/s); "
        f"partial={report.partial}; {dict(histogram)}",
    )
    assert len(report.verdicts) == len(targets)
    assert elapsed < 0.6 + 2 * rr._SOLVE_TIMEOUT_S, (
        f"overshot the deadline by {elapsed - 0.6:.3f} s"
    )
    # Measured 93/s on this mix, so ~56 in the budget.
    assert report.evaluated >= 32, (
        f"only {report.evaluated} candidates in 0.6 s; "
        "solving cannot replace the proxy ranker at this rate"
    )
    # The interleaving is load-bearing: without it the evaluated prefix is all
    # accepts and this test measures the best case under a mixed corpus's name.
    assert histogram[SOLVED] > 0 and histogram[COLLISION] > 0


def test_a_real_tight_budget_returns_a_marked_partial(real, corpus):
    targets = (corpus["targets"] * 4)[:600]
    report = filter_reachable(
        real, targets, control=PlanningControl(deadline_s=time.monotonic() + 0.02),
    )
    assert report.partial is True
    assert 0 < report.evaluated < len(targets)
    assert {v.index for v in report.verdicts} == set(range(len(targets)))
    tail = report.verdicts[report.evaluated:]
    assert len(tail) == len(targets) - report.evaluated
    assert all(v.reason == DEADLINE for v in tail)
