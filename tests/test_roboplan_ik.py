"""``RoboplanIKSolver`` must be a drop-in, and the ways it silently would not be.

An IK backend that swaps in cleanly and returns slightly different numbers is
worse than one that crashes, because grasp scoring keeps working and nobody
looks.  These failures get sharp tests here:

1.  **The acceptance gate.**  ``SimpleIkOptions`` stops on a tip-ORIGIN,
    isotropic error.  This repo accepts at the tool contact point: on
    2026-07-24 a 0.236 rad orientation residual inside a loose gate put the
    PiPER fingers 27 mm off a verified grasp.  A backend that reports SimpleIk's
    own verdict reintroduces exactly that bug --
    ``test_position_gate_is_measured_at_the_tool_control_point``.

2.  **The continuation branch.**  ``grasp_pipeline`` falls back to global
    ``solve`` when a backend has no ``solve_continuation``, which is the branch
    flip continuation exists to prevent (grasp_pipeline.py:669-670).  Both are
    implemented here, so the binding is pinned rather than assumed.

3.  **The step bound.**  ``SimpleIk`` has no ``max_joint_step_rad``, so the
    bound is enforced on the returned solution.  Forget it and the caller's own
    guard (grasp_pipeline.py:648-654) turns a branch flip into a late rejection
    with no diagnostic.

4.  **The scoring fields.**  ``manipulability`` and ``min_joint_limit_margin``
    are consumed as grasp scoring terms; stubbing either to 0 changes which
    grasp is chosen and nothing fails.

5.  **The deadline.**  A ``PlanningControl`` deadline must reach
    ``SimpleIkOptions.max_time``, and exhausting the caller's budget must
    propagate while exhausting this solver's own budget must not.

6.  **The joint limits.**  ``SimpleIk`` takes no bounds argument.  Neither
    existing backend can emit an out-of-limit solution, so nothing downstream
    re-checks -- and ``min_joint_limit_margin`` past -1e-3 turns
    grasp_pipeline.py:963's joint-limit penalty into a bonus.

7.  **``check_collisions``.**  The one option that makes the capsule Scene an
    enforced model rather than decoration.  Targets sampled from collision-free
    configurations barely exercise it, so the real-model test uses targets whose
    natural basin is BLOCKED: measured, 0/40 solved with the flag, 38/40 solved
    and all 38 in collision without it.

8.  **Gate drift.**  The free-axis acceptance split is duplicated from
    ``PinocchioIKSolver._orientation_accepted``; a FROZEN-SAFETY flag with two
    implementations is how one backend silently gets a wider gate.

9.  **The body pose under the arm.**  ``SimpleIk`` reads every non-group
    coordinate from the shared Scene's CURRENT positions, so the configuration
    it collision-checks and the one ``handle.expand`` measures can be different
    robots -- ``test_the_body_pose_is_pinned_to_the_one_the_metrics_assume``.

Everything above runs on a host with no roboplan, against a fake ``Scene``
carrying the real model's shape (nq=28, arm at q[20:26]) and a fake ``SimpleIk``
whose answers the test chooses.  The tests behind ``importorskip`` measure the
real thing: solve rate, accuracy, collision-freedom and determinism.

Accuracy there is asserted against ``_pose_residual``, recomputed from the
model.  Asserting ``solution.position_error_m`` instead only asks the solver
whether it thinks it succeeded: patching ``_errors`` to return a constant zero
-- a solver that accepts any pose at any distance -- passed every real-model
test in an earlier revision of this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.kinematics.robust_ik import IKConfig, IKFailure
from z_manip.planning import roboplan_ik as rpik
from z_manip.planning import roboplan_runtime as rt
from z_manip.planning.roboplan_ik import RoboplanIKSolver
from z_manip.planning.roboplan_runtime import RoboplanConfig, RoboplanUnavailable
from z_manip.planning_control import (
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
SRDF = ROOT / "ros2/z_manip_motion/config/piper.srdf"
CAPSULES = ROOT / "configs/piper_collision_capsules.json"
STACK_CONFIG = ROOT / "configs/go2w_piper.json"

# The real model, measured: nq=28, nv=24, arm at q[20:26] / v[16:22].
NQ = 28
NV = 24
ARM_Q = (20, 21, 22, 23, 24, 25)
ARM_V = (16, 17, 18, 19, 20, 21)
WHEEL_Q = tuple(range(12, 20))
DOF = 6
HOME = (0.1, 0.2, 0.3, 0.0, 0.3, 0.0)
LOWER = np.array([-1.0, -1.0, -1.0, -np.pi, -1.2, -np.pi])
UPPER = np.array([1.0, 1.0, 1.0, np.pi, 1.2, np.pi])
# The deployed tool: the contact TCP sits this far along tip +Z.
CONTACT_TCP_Z_M = 0.116675


# --------------------------------------------------------------- fake model


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose_residual(
    actual: np.ndarray,
    goal: np.ndarray,
    offset,
    free_axis,
) -> tuple[float, float, float, float]:
    """``(control-point metres, geodesic rad, transverse rad, axial rad)``.

    Written out here rather than imported.  An oracle that calls the solver's
    own ``control_point_delta``/``rotation_log`` re-derives the answer from the
    implementation under test, so it agrees with a solver that measures nothing:
    patching ``RoboplanIKSolver._errors`` to return a constant zero passed every
    real-model test in this file before this helper existed.  The rotation
    vector is recovered from the antisymmetric part, a different formula path
    from ``chain.rotation_log`` (no near-pi branch is needed: an accepted
    residual is a few hundredths of a radian).
    """

    offset = np.asarray(offset, dtype=float)
    delta = (actual[:3, 3] + actual[:3, :3] @ offset) - (
        goal[:3, 3] + goal[:3, :3] @ offset
    )
    relative = goal[:3, :3] @ actual[:3, :3].T
    angle = float(np.arccos(np.clip(0.5 * (np.trace(relative) - 1.0), -1.0, 1.0)))
    vee = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
    )
    rotvec = (
        np.zeros(3) if angle < 1e-12 else vee * (angle / (2.0 * np.sin(angle)))
    )
    axis = np.asarray(free_axis, dtype=float)
    axis = goal[:3, :3] @ (axis / float(np.linalg.norm(axis)))
    axial = abs(float(rotvec @ axis))
    return (
        float(np.linalg.norm(delta)),
        angle,
        float(np.linalg.norm(rotvec - (rotvec @ axis) * axis)),
        axial,
    )


def _fk(q_arm) -> np.ndarray:
    """A gantry plus a ZYX wrist: six DOF, analytic, non-singular at HOME."""

    q = np.asarray(q_arm, dtype=float)
    tform = np.eye(4)
    tform[:3, 3] = q[:3]
    tform[:3, :3] = _rot_z(q[3]) @ _rot_y(q[4]) @ _rot_x(q[5])
    return tform


def _jacobian(q_arm) -> np.ndarray:
    q = np.asarray(q_arm, dtype=float)
    jac = np.zeros((6, DOF))
    jac[:3, :3] = np.eye(3)
    jac[3:, 3] = (0.0, 0.0, 1.0)
    jac[3:, 4] = _rot_z(q[3]) @ (0.0, 1.0, 0.0)
    jac[3:, 5] = _rot_z(q[3]) @ _rot_y(q[4]) @ (1.0, 0.0, 0.0)
    return jac


class _Group:
    q_indices = np.array(ARM_Q, dtype=np.int32)
    v_indices = np.array(ARM_V, dtype=np.int32)
    joint_names = [f"piper_joint{i}" for i in range(1, 7)]


class FakeScene:
    """The real model's shape with an analytic arm, so errors are computable."""

    def __init__(self) -> None:
        # Shared mutable Scene state, modelled because SimpleIk READS it: see
        # ``FakeIk.__call__``.  A fresh Scene's positions are all-zero, which is
        # exactly what the real one reports (the wheels are continuous joints
        # carried as ``(cos, sin)``, so all-zero is not even a valid pose).
        self.current = np.zeros(NQ)

    def getJointGroupInfo(self, name):
        assert name == "piper_arm"
        return _Group()

    def getCurrentJointPositions(self):
        return self.current.copy()

    def setJointPositions(self, q):
        # No clamping, matching ``Scene::setJointPositions``: it size-checks and
        # assigns.  A fake that quietly repaired a bad pose would hide one.
        values = np.asarray(q, dtype=float)
        assert values.shape == (NQ,), f"setJointPositions expected {NQ}, got {values.shape}"
        self.current = np.array(values, dtype=float, copy=True)

    def clampToValidConfiguration(self, q):
        out = np.array(q, dtype=float, copy=True)
        for i in range(0, len(WHEEL_Q), 2):
            pair = out[list(WHEEL_Q[i:i + 2])]
            norm = float(np.linalg.norm(pair))
            out[list(WHEEL_Q[i:i + 2])] = (1.0, 0.0) if norm < 1e-9 else pair / norm
        out[list(ARM_Q)] = np.clip(out[list(ARM_Q)], LOWER, UPPER)
        return out

    def getPositionLimitVectors(self, group_name=""):
        return LOWER.copy(), UPPER.copy()

    def getVelocityLimitVectors(self, group_name=""):
        return -np.full(DOF, 3.0), np.full(DOF, 3.0)

    def hasCollisions(self, q, debug=False):
        return False

    def forwardKinematics(self, q, frame_name, base_frame=""):
        assert (frame_name, base_frame) == ("piper_gripper_base", "piper_base_link")
        return _fk(np.asarray(q, dtype=float)[list(ARM_Q)])

    def computeFrameJacobian(self, q, frame_name, local=True):
        jac = np.zeros((6, NV))
        jac[:, list(ARM_V)] = _jacobian(np.asarray(q, dtype=float)[list(ARM_Q)])
        return jac


class Robot:
    """The stack config's ``robot`` block, duck-typed."""

    urdf_path = URDF
    base_link = "piper_base_link"
    tip_link = "piper_gripper_base"
    acceleration_limits = (1.5, 1.5, 1.5, 2.0, 2.0, 2.0)


class FakeIk:
    """A recording stand-in for one ``SimpleIk`` attempt.

    ``replies`` are consumed in order; once exhausted every further attempt
    fails, which is what a genuinely unreachable target looks like.
    """

    def __init__(self, *replies, clock=None, cost_s: float = 0.0) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self._clock = clock
        self._cost_s = cost_s

    def __call__(self, scene, joint_names, frames, goal, start, options):
        # ``checked`` is what roboplan would collision-check: exactly
        # ``Scene::toFullJointPositions`` (scene.cpp:428) -- the group
        # coordinates from ``start``, every other one from the Scene's CURRENT
        # positions, not from anything the adapter passes in.
        checked = np.asarray(scene.getCurrentJointPositions(), dtype=float).copy()
        checked[list(ARM_Q)] = np.asarray(start, dtype=float)
        self.calls.append(
            {
                "start": np.array(start, dtype=float),
                "checked": checked,
                "options": dict(options),
                "goal": np.array(goal, dtype=float),
                "names": tuple(joint_names),
                "frames": tuple(frames),
            },
        )
        if self._clock is not None:
            self._clock[0] += self._cost_s
        index = len(self.calls) - 1
        if index >= len(self.replies):
            return False, np.zeros(DOF)
        solved, joints = self.replies[index]
        return solved, np.asarray(joints, dtype=float)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """An enabled config over a fake Scene, plus a default-permissive IKConfig."""

    rt.clear_scene_cache()
    urdf = tmp_path / "robot.urdf"
    urdf.write_bytes(URDF.read_bytes() if URDF.exists() else b"<robot name='x'/>")
    (tmp_path / "planning.srdf").write_bytes(
        SRDF.read_bytes() if SRDF.exists() else b"<robot name='x'/>",
    )
    (tmp_path / "capsules.json").write_bytes(
        CAPSULES.read_bytes() if CAPSULES.exists() else b"{}",
    )
    (tmp_path / "home.json").write_text(json.dumps({"joint_radians": list(HOME)}))
    monkeypatch.setattr(rt, "_new_scene", lambda *a, **k: FakeScene())

    config = RoboplanConfig(
        enabled=True,
        scene_cache_dir=tmp_path / "out",
        srdf_path=tmp_path / "planning.srdf",
        collision_model_path=tmp_path / "capsules.json",
        home_path=tmp_path / "home.json",
    )
    robot = Robot()
    robot.urdf_path = urdf
    yield config, robot
    rt.clear_scene_cache()


def _ik_config(**overrides) -> IKConfig:
    values = {"position_tolerance_m": 0.003, "orientation_tolerance_rad": 0.06}
    values.update(overrides)
    return IKConfig(**values)


def _solver(env, ik_config=None, **config_overrides) -> RoboplanIKSolver:
    config, robot = env
    return RoboplanIKSolver(
        replace(config, **config_overrides),
        robot,
        ik_config or _ik_config(),
    )


# ------------------------------------------------------------- default off


def test_default_config_is_off_and_never_touches_roboplan(env, monkeypatch):
    """This checkout is a live deploy: constructing a solver may not opt in."""

    _config, robot = env
    monkeypatch.setattr(
        rt,
        "_new_scene",
        lambda *a, **k: pytest.fail("a disabled backend built a Scene"),
    )
    monkeypatch.setattr(
        rpik,
        "_solve_ik",
        lambda *a, **k: pytest.fail("a disabled backend called SimpleIk"),
    )
    with pytest.raises(RoboplanUnavailable) as error:
        RoboplanIKSolver(RoboplanConfig(), robot)
    assert error.value.reason == "disabled"


def test_importing_the_module_does_not_import_roboplan():
    """Checked in a subprocess: a sibling test may already have imported it."""

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import z_manip.planning.roboplan_ik as m;"
            " assert 'roboplan' not in sys.modules; assert m.RoboplanIKSolver",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_stopping_norm_looser_than_acceptance_fails_closed(env):
    """Otherwise every attempt 'succeeds' into a rejection and the ladder burns.

    0.0005 m, not 0.001: the roboplan stopping norm IS 0.001 (see
    ``RoboplanConfig.ik_position_tolerance_m`` and the sweep recorded there), so
    an acceptance gate of 0.001 is equal, not looser, and would not trip this
    guard.  This value has to stay strictly tighter than that default.
    """

    with pytest.raises(ValueError, match="must not be looser"):
        _solver(env, _ik_config(position_tolerance_m=0.0005))


def test_a_disabled_backend_reports_disabled_and_not_a_config_error(env):
    """Opt-in is checked before anything else this constructor can object to.

    The wiring layer selects a backend by catching ``RoboplanUnavailable`` and
    falling back to the numpy solver.  If the tolerance guard above runs first,
    a deployment that never opted in still crashes with a ``ValueError`` about a
    ``roboplan`` config block it does not use -- on a live deploy, from a
    default-off feature.
    """

    _config, robot = env
    with pytest.raises(RoboplanUnavailable) as error:
        RoboplanIKSolver(
            RoboplanConfig(),  # enabled=False
            robot,
            _ik_config(position_tolerance_m=0.001),  # and a tolerance conflict
        )
    assert error.value.reason == "disabled"


# --------------------------------------------------------- protocol binding


def test_grasp_pipeline_binds_the_continuation_branch(env, monkeypatch):
    """The silent-degradation bug at grasp_pipeline.py:669-670, pinned.

    A backend without ``solve_continuation`` falls back to global ``solve`` --
    the branch flip continuation exists to prevent.  A backend whose
    ``solve_continuation`` does not accept ``control`` loses the deadline.
    """

    from z_manip.planning.grasp_pipeline import GraspPlanGenerator

    monkeypatch.setattr(rpik, "_solve_ik", FakeIk())

    class _Planner:
        def plan_joint(self, start, goal, *, timeout_s=5.0, control=None):
            raise AssertionError("unused")

        def segment_valid(self, first, second, *, control=None):
            return True

    solver = _solver(env)
    generator = GraspPlanGenerator(solver, _Planner())
    # Bound to OUR continuation, not merely non-None: the fallback that
    # ``_solve_continuation`` takes when this is None is the global solve.
    assert generator._continuation_solve.__func__ is RoboplanIKSolver.solve_continuation
    assert generator._continuation_solve.__self__ is solver
    assert generator._continuation_accepts_control is True
    assert generator._ik_accepts_control is True


# ----------------------------------------------------------- acceptance gate


def test_position_gate_is_measured_at_the_tool_control_point(env, monkeypatch):
    """The 2026-07-24 incident: 0 mm at the tip, 5.8 mm at the fingers.

    The returned configuration puts the TIP exactly on the goal and tilts the
    wrist 0.05 rad -- inside the 0.06 rad orientation tolerance.  Measured at
    the tip origin that is a perfect solution; measured at the contact TCP
    116.675 mm ahead it is 5.8 mm off, past the 3 mm gate.
    """

    goal_joints = np.array([0.2, 0.1, 0.4, 0.0, 0.0, 0.0])
    tilted = goal_joints + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
    goal = _fk(goal_joints)

    at_tip = _solver(env)
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, tilted)))
    accepted = at_tip.solve(goal)
    assert accepted.position_error_m == pytest.approx(0.0, abs=1e-12)

    at_tcp = _solver(
        env,
        _ik_config(position_error_offset_tip_m=(0.0, 0.0, CONTACT_TCP_Z_M)),
    )
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, tilted)))
    with pytest.raises(IKFailure) as error:
        at_tcp.solve(goal)
    # 2 * L * sin(theta / 2) at the contact point, not 0 at the tip.  The
    # counter says the pose was reachable and the gate turned it down, which is
    # the whole distinction the incident turned on.
    assert "best position=0.005833" in str(error.value)
    assert "gate_rejections=1" in str(error.value)


def test_free_axis_tolerance_is_honoured(env, monkeypatch):
    """Set to 0.1745 rad about Z in configs/go2w_piper.json; not honouring it
    would silently reject poses ``PinocchioIKSolver`` accepts."""

    goal_joints = np.array([0.2, 0.1, 0.4, 0.0, 0.0, 0.0])
    # A pure roll about the goal tip Z: it moves the contact TCP not at all.
    rolled = goal_joints + np.array([0.0, 0.0, 0.0, 0.10, 0.0, 0.0])
    tilted = goal_joints + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.10])
    goal = _fk(goal_joints)
    free = _ik_config(
        orientation_free_axis_tolerance_rad=0.1745329252,
        orientation_free_axis=(0.0, 0.0, 1.0),
    )

    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, rolled)))
    solution = _solver(env, free).solve(goal)
    assert solution.orientation_error_rad == pytest.approx(0.10)

    # The same magnitude transverse to the free axis stays outside the gate.
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, tilted)))
    with pytest.raises(IKFailure):
        _solver(env, free).solve(goal)

    # And with the study flag off, the roll is rejected too: 0.10 > 0.06.
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, rolled)))
    with pytest.raises(IKFailure):
        _solver(env).solve(goal)


def test_the_free_axis_gate_matches_the_pinocchio_backend_exactly(env, monkeypatch):
    """One safety gate, two copies -- pinned together so they cannot drift.

    ``RoboplanIKSolver._accepted`` duplicates
    ``PinocchioIKSolver._orientation_accepted`` (pinocchio_ik.py:463) because
    the original is a bound method on a Pinocchio-owning object.  A FROZEN-
    SAFETY study flag with two independent implementations is how a gate gets
    widened on one backend only, so this drives both over a grid that straddles
    every boundary and demands the same verdict.  It runs on the host:
    ``pinocchio_ik`` imports the library lazily, so the gate is reachable here.
    Only the free-axis normalisation (pinocchio_ik.py:218) is mirrored below;
    the branching logic under test is the real one.
    """

    from z_manip.kinematics.pinocchio_ik import PinocchioIKSolver

    class _Desired:
        def __init__(self, rotation):
            self.rotation = rotation

    monkeypatch.setattr(rpik, "_solve_ik", FakeIk())
    rng = np.random.default_rng(3)
    verdicts = []
    for ik_config in (
        _ik_config(),
        _ik_config(
            orientation_free_axis_tolerance_rad=0.1745329252,
            orientation_free_axis=(0.0, 0.0, 1.0),
        ),
        _ik_config(
            orientation_free_axis_tolerance_rad=0.09,
            orientation_free_axis=(0.0, 1.0, 0.0),
        ),
    ):
        solver = _solver(env, ik_config)
        axis = np.asarray(ik_config.orientation_free_axis, dtype=float)
        mirror = SimpleNamespace(
            config=ik_config,
            _orientation_free_axis=axis / float(np.linalg.norm(axis)),
        )
        for _ in range(200):
            goal_rotation = (
                _rot_z(rng.uniform(-np.pi, np.pi))
                @ _rot_y(rng.uniform(-1.0, 1.0))
                @ _rot_x(rng.uniform(-1.0, 1.0))
            )
            # Scaled to straddle both 0.06 and 0.1745 in each component.
            residual = rng.normal(scale=0.09, size=3)
            geodesic = float(np.linalg.norm(residual))
            ours = solver._accepted(goal_rotation, 0.0, geodesic, residual)
            theirs = PinocchioIKSolver._orientation_accepted(
                mirror, _Desired(goal_rotation), residual, geodesic,
            )
            assert ours is theirs, (
                f"gate drift at residual={residual} free_axis="
                f"{ik_config.orientation_free_axis}"
            )
            verdicts.append(ours)
    # Both verdicts must actually occur, or "they agree" means "both said no".
    assert 0 < sum(verdicts) < len(verdicts)


def test_scoring_fields_are_real_not_stubbed(env, monkeypatch):
    """``manipulability`` and ``min_joint_limit_margin`` decide which grasp wins."""

    joints = np.array([0.2, 0.1, 0.4, 0.0, 0.3, 0.0])
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, joints)))
    solution = _solver(env).solve(_fk(joints))

    span = UPPER - LOWER
    expected_margin = float(
        np.min(np.minimum((joints - LOWER) / span, (UPPER - joints) / span)),
    )
    assert solution.min_joint_limit_margin == pytest.approx(expected_margin)
    assert 0.0 < solution.min_joint_limit_margin < 0.5
    # |det J| rather than prod(svd): the same number by a different route, so
    # this does not re-derive the answer from the implementation's own formula.
    # It also pins the v_indices slice -- the full 6x24 Jacobian has no det, and
    # slicing the wrong six columns gives a singular one.
    assert solution.manipulability == pytest.approx(
        abs(float(np.linalg.det(_jacobian(joints)))),
    )
    assert solution.manipulability > 0.0


def test_a_solution_outside_the_joint_limits_is_rejected(env, monkeypatch):
    """``SimpleIk`` takes no bounds argument; neither existing backend can emit
    one of these (SciPy hard bounds, Pinocchio clamps every iterate).

    It matters beyond tidiness: ``grasp_pipeline.py:963`` scores
    ``-joint_limit_penalty / (min_joint_limit_margin + 1e-3)``, so a margin past
    -1e-3 turns that penalty into a BONUS and the out-of-limit grasp wins.
    """

    beyond = np.array([0.2, 0.1, 0.4, 0.0, 1.7, 0.0])  # joint 5 is 0.5 rad past
    assert beyond[4] > UPPER[4]
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, beyond)))
    with pytest.raises(IKFailure) as error:
        _solver(env, ik_restarts=1).solve(_fk(beyond))
    assert "limit_rejections=1" in str(error.value)
    assert "unsolved_starts=1" in str(error.value)

    # A hair outside is a converged descent sitting ON the limit, not a bug.
    on_limit = np.array([0.2, 0.1, 0.4, 0.0, UPPER[4] + 1e-12, 0.0])
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, on_limit)))
    assert _solver(env).solve(_fk(on_limit)).min_joint_limit_margin < 1e-9


def test_cancellation_propagates_and_releases_the_scene_lock(env, monkeypatch):
    """A cancelled request must not come back as ``IKFailure`` -- the caller
    would read that as "this grasp is unreachable" and drop the candidate.

    And the ladder holds the process-global Scene lock across every attempt, so
    an escape that leaked it would deadlock every later plan, not fail one.
    """

    monkeypatch.setattr(rpik, "_solve_ik", FakeIk())
    solver = _solver(env)
    cancelled = [False]
    control = PlanningControl(cancel_check=lambda: cancelled[0])
    cancelled[0] = True
    with pytest.raises(PlanningCancelled):
        solver.solve(_fk(HOME), control=control)

    cancelled[0] = False
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, np.array(HOME))))
    assert solver.solve(_fk(HOME)).joints == pytest.approx(HOME)


# ------------------------------------------------------------------ ladder


def test_never_uses_roboplans_own_random_restarts(env, monkeypatch):
    """Measured on the real model: at ``max_restarts >= 1`` SimpleIk draws its
    restart from the Scene-global RNG, so identical requests return different
    joints and sometimes fail.  Every start this solver uses is its own."""

    recorder = FakeIk()
    monkeypatch.setattr(rpik, "_solve_ik", recorder)
    with pytest.raises(IKFailure):
        _solver(env, ik_restarts=4).solve(_fk(HOME))

    assert len(recorder.calls) == 5  # the seed plus four explicit restarts
    assert all(call["options"]["max_restarts"] == 0 for call in recorder.calls)
    assert all(call["options"]["check_collisions"] is True for call in recorder.calls)
    assert recorder.calls[0]["start"] == pytest.approx(HOME)
    assert recorder.calls[0]["frames"] == ("piper_base_link", "piper_gripper_base")


def test_the_body_pose_is_pinned_to_the_one_the_metrics_assume(env, monkeypatch):
    """The collision check and the metrics must answer for the SAME robot.

    ``SimpleIk`` takes every non-group coordinate from the Scene's current
    positions (``Scene::toFullJointPositions``, scene.cpp:428), so without a pin
    the arm is collision-checked against whatever body pose the previous caller
    left behind -- while ``_solution``/``_errors`` compute the Jacobian and the
    pose error via ``handle.expand``, which uses ``handle.seed`` as the body.

    Today no non-arm joint can change a verdict: the enforced model
    (``configs/piper_collision_capsules.json``) is 22 PiPER capsules and
    ``roboplan_scene.write_planning_model`` strips the rest.  That is what makes
    this worth pinning rather than leaving to be noticed -- the day a leg or
    chassis capsule is added, IK starts clearing the arm against a body that is
    not there and nothing reports it.
    """

    recorder = FakeIk()
    monkeypatch.setattr(rpik, "_solve_ik", recorder)
    solver = _solver(env, ik_restarts=2)

    # What a previous caller left in the shared Scene: the chassis a metre off
    # and all four wheels a quarter turn round.
    stale = np.array(solver.handle.seed, dtype=float, copy=True)
    stale[:3] += 1.0
    stale[list(WHEEL_Q)] = np.tile((0.0, 1.0), 4)
    with solver.handle.exclusive() as scene:
        scene.setJointPositions(stale)
        assert scene.getCurrentJointPositions() == pytest.approx(stale)

    with pytest.raises(IKFailure):
        solver.solve(_fk(HOME))

    body = [index for index in range(NQ) if index not in ARM_Q]
    assert len(recorder.calls) == 3  # the seed start plus two restarts
    for call in recorder.calls:
        # Against ``handle.expand``, not against a copy of the seed: the
        # invariant is that the body SimpleIk checked is the body the metrics
        # were computed on, and expand is where the second one comes from.
        expected = solver.handle.expand(call["start"])
        assert call["checked"][body] == pytest.approx(expected[body])
        assert call["checked"][body] != pytest.approx(stale[body])


def test_the_restart_ladder_is_deterministic(env, monkeypatch):
    """A rerun of a failed plan must explore the same hypotheses in order."""

    starts = []
    for _ in range(2):
        recorder = FakeIk()
        monkeypatch.setattr(rpik, "_solve_ik", recorder)
        with pytest.raises(IKFailure):
            _solver(env, ik_restarts=6).solve(_fk(HOME))
        starts.append(np.array([call["start"] for call in recorder.calls]))
    assert starts[0] == pytest.approx(starts[1])
    # Distinct starts, all inside the joint box: a ladder of duplicates is a
    # ladder of one.
    assert len(np.unique(np.round(starts[0], 9), axis=0)) == len(starts[0])
    assert np.all(starts[0] >= LOWER) and np.all(starts[0] <= UPPER)


def test_failure_message_reports_the_best_residual(env, monkeypatch):
    near = np.array(HOME) + np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, near)))
    with pytest.raises(IKFailure) as error:
        _solver(env, ik_restarts=2).solve(_fk(HOME))
    message = str(error.value)
    assert "no roboplan IK solution met tolerances" in message
    assert "best position=0.020000 m" in message
    assert "unsolved_starts=2" in message
    assert "gate_rejections=1" in message
    assert "budget_exhausted=false" in message


def test_current_joints_are_validated(env, monkeypatch):
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk())
    solver = _solver(env)
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        solver.solve(_fk(HOME), np.zeros(5))
    with pytest.raises(ValueError, match="finite"):
        solver.solve(_fk(HOME), np.full(DOF, np.nan))


# ---------------------------------------------------------------- deadline


def test_caller_deadline_reaches_simple_ik_max_time(env, monkeypatch):
    """A deadline that never reaches ``max_time`` lets one attempt outlive the
    whole request budget."""

    clock = [100.0]
    recorder = FakeIk(clock=clock, cost_s=0.01)
    monkeypatch.setattr(rpik, "_solve_ik", recorder)
    control = PlanningControl(deadline_s=100.05, monotonic_fn=lambda: clock[0])
    with pytest.raises(PlanningDeadlineExceeded):
        # ik_timeout_s is 0.60, far past the caller's 0.05 s: the caller wins.
        _solver(env, ik_restarts=20, ik_timeout_s=0.60).solve(_fk(HOME), control=control)

    budgets = [call["options"]["max_time"] for call in recorder.calls]
    assert budgets[0] == pytest.approx(0.05)
    assert budgets == sorted(budgets, reverse=True)
    assert all(budget <= 0.05 + 1e-9 for budget in budgets)


def test_local_budget_exhaustion_is_a_failure_not_an_abort(env, monkeypatch):
    """Burning this solver's own budget rejects the pose; burning the caller's
    aborts the request.  The two must not be confused."""

    clock = [100.0]
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk(clock=clock, cost_s=0.05))
    control = PlanningControl(monotonic_fn=lambda: clock[0])
    with pytest.raises(IKFailure) as error:
        _solver(env, ik_restarts=20, ik_timeout_s=0.12).solve(_fk(HOME), control=control)
    assert "budget_exhausted=true" in str(error.value)


# ------------------------------------------------------------ continuation


def test_continuation_rejects_a_branch_flip(env, monkeypatch):
    """``SimpleIk`` has no step bound; this is the one that enforces it."""

    goal_joints = np.array([0.2, 0.1, 0.4, 0.0, 0.0, 0.0])
    flipped = goal_joints + np.array([0.0, 0.0, 0.0, 2.9, 0.0, 0.0])
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk((True, flipped), (True, flipped)))
    with pytest.raises(IKFailure) as error:
        _solver(env).solve_continuation(
            _fk(goal_joints),
            np.array(HOME),
            max_joint_step_rad=0.45,
        )
    message = str(error.value)
    assert "no local roboplan IK continuation met tolerances" in message
    assert "max_joint_step_rad=0.450000" in message
    assert "step_rejections=2" in message


def test_continuation_accepts_a_bounded_fallback(env, monkeypatch):
    """A rejected first attempt must fall back INSIDE the trust box, not to a
    global restart -- the fallback starts are what make that possible."""

    current = np.array(HOME)
    goal_joints = current + np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    flipped = goal_joints + np.array([0.0, 0.0, 0.0, 2.9, 0.0, 0.0])
    recorder = FakeIk((True, flipped), (True, goal_joints))
    monkeypatch.setattr(rpik, "_solve_ik", recorder)

    solution = _solver(env).solve_continuation(
        _fk(goal_joints), current, max_joint_step_rad=0.45,
    )
    assert float(np.max(np.abs(solution.joints - current))) <= 0.45
    assert solution.seed_index == 1
    starts = np.array([call["start"] for call in recorder.calls])
    assert np.all(np.abs(starts - current) <= 0.45 + 1e-12)


def test_continuation_holds_an_already_accepted_configuration(env, monkeypatch):
    """The alpha=0 sample of every Cartesian segment.  SimpleIk stops on its own
    tighter tip norm and would walk away from a pose already inside our gate."""

    current = np.array(HOME)
    monkeypatch.setattr(
        rpik,
        "_solve_ik",
        lambda *a, **k: pytest.fail("an accepted pose was re-solved"),
    )
    solution = _solver(env).solve_continuation(
        _fk(current), current, max_joint_step_rad=0.45,
    )
    assert solution.joints == pytest.approx(current)
    assert solution.iterations == 0
    assert solution.manipulability > 0.0


def test_continuation_validates_its_step_bound(env, monkeypatch):
    monkeypatch.setattr(rpik, "_solve_ik", FakeIk())
    solver = _solver(env)
    goal = _fk(np.array(HOME) + 0.2)
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="max_joint_step_rad"):
            solver.solve_continuation(goal, np.array(HOME), max_joint_step_rad=bad)
    with pytest.raises(ValueError, match="joint limits"):
        solver.solve_continuation(
            goal, np.full(DOF, 9.0), max_joint_step_rad=0.4,
        )


# ------------------------------------------------------- the real thing


@pytest.fixture(scope="module")
def real():
    """The deployed stack config against the real capsule Scene."""

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    from z_manip.configuration import load_stack_config

    stack = load_stack_config(
        STACK_CONFIG, environ={"Z_MANIP_ROBOT_URDF": str(URDF)},
    )
    rt.clear_scene_cache()
    config = RoboplanConfig(enabled=True, scene_cache_dir=Path("/tmp/z_manip_roboplan_ik"))
    solver = RoboplanIKSolver(config, stack.robot, stack.ik)
    yield solver, stack
    rt.clear_scene_cache()


def _sample_targets(solver, count: int, seed: int, *, colliding: bool):
    """Targets that are the FK of sampled configurations of a chosen kind."""

    handle = solver.handle
    lower, upper = handle.position_limits
    rng = np.random.default_rng(seed)
    targets = []
    with handle.exclusive() as scene:
        while len(targets) < count:
            q = rng.uniform(lower, upper)
            if scene.hasCollisions(handle.expand(q)) is not colliding:
                continue
            targets.append(
                np.asarray(
                    scene.forwardKinematics(
                        handle.expand(q), solver.tip_link, solver.base_link,
                    ),
                    dtype=float,
                ),
            )
    return targets


def _measured(solver, stack, target, solution):
    """Recompute this solution's pose error from the model, independently."""

    with solver.handle.exclusive() as scene:
        actual = np.asarray(
            scene.forwardKinematics(
                solver.handle.expand(solution.joints),
                solver.tip_link,
                solver.base_link,
            ),
            dtype=float,
        )
    return _pose_residual(
        actual,
        target,
        stack.ik.position_error_offset_tip_m,
        stack.ik.orientation_free_axis,
    )


@pytest.mark.usefixtures("real")
def test_real_model_solves_accurately_and_collision_free(real):
    """Solve rate, accuracy at the CONTACT point, and collision-freedom.

    The targets are the forward kinematics of collision-free sampled
    configurations, so every one of them is known reachable: a failure here is a
    solver failure, not an unreachable pose.

    Every accuracy assertion below is against ``_pose_residual``, recomputed
    from the model, NOT against ``solution.position_error_m``.  Asserting the
    solver's own report is not an accuracy test: patching
    ``RoboplanIKSolver._errors`` to return a constant zero -- a solver that
    accepts any pose at any distance, which is the 2026-07-24 incident with the
    brakes cut -- passed this test in its earlier form.
    """

    solver, stack = real
    handle = solver.handle
    lower, upper = handle.position_limits
    targets = _sample_targets(solver, 120, 17, colliding=False)

    latencies = []
    solved = []
    for target in targets:
        started = time.perf_counter()
        try:
            solution = solver.solve(target)
        except IKFailure:
            latencies.append(time.perf_counter() - started)
            continue
        latencies.append(time.perf_counter() - started)
        solved.append(solution)

        position, geodesic, transverse, axial = _measured(
            solver, stack, target, solution,
        )
        assert position < stack.ik.position_tolerance_m
        # The transverse residual carries the finger closing line; only the roll
        # about the tool axis may take the relaxed free-axis tolerance.
        assert transverse < stack.ik.orientation_tolerance_rad
        assert axial < stack.ik.orientation_free_axis_tolerance_rad
        # ...and the reported fields are the errors that were actually measured,
        # because grasp scoring and every operator log read those, not these.
        assert solution.position_error_m == pytest.approx(position, abs=1e-9)
        assert solution.orientation_error_rad == pytest.approx(geodesic, abs=1e-9)

        assert solution.manipulability > 0.0
        with handle.exclusive() as scene:
            assert not scene.hasCollisions(handle.expand(solution.joints))
        assert np.all(solution.joints >= lower - 1e-9)
        assert np.all(solution.joints <= upper + 1e-9)

    milliseconds = np.asarray(latencies) * 1e3
    print(
        f"\nroboplan IK: {len(solved)}/{len(targets)} solved, "
        f"mean {milliseconds.mean():.2f} ms, "
        f"p95 {np.percentile(milliseconds, 95):.2f} ms",
    )
    # The numpy backends clear this comfortably; a backend that solved half the
    # reachable poses would still pass every unit test above.
    assert len(solved) >= 0.90 * len(targets)


@pytest.mark.usefixtures("real")
def test_real_model_never_returns_a_colliding_solution(real):
    """``check_collisions`` must actually reach ``SimpleIkOptions``.

    Targets sampled from collision-free configurations barely exercise it -- the
    nearest basin is collision-free by construction.  These targets are the FK
    of CO LLIDING configurations, and measured on this model the flag is the whole
    difference: 0/40 solved with it on, 38/40 solved with it off and every one of
    those 38 solutions in collision.  Drop the option, misspell it, or let a
    future refactor build ``SimpleIkOptions`` without it and this test fails 24
    times over; the previous suite would not have noticed.
    """

    solver, _stack = real
    handle = solver.handle
    returned = 0
    for target in _sample_targets(solver, 24, 41, colliding=True):
        try:
            solution = solver.solve(target)
        except IKFailure:
            continue
        returned += 1
        with handle.exclusive() as scene:
            assert not scene.hasCollisions(handle.expand(solution.joints)), (
                "IK returned a self-colliding configuration"
            )
    print(f"\nroboplan IK, blocked targets: {returned}/24 returned an alternative")


@pytest.mark.usefixtures("real")
def test_real_model_continuation_never_exceeds_the_step_bound(real):
    """Walk a Cartesian line the way ``_cartesian_ik`` does and check the bound
    the caller would otherwise catch one stage too late."""

    solver, stack = real
    handle = solver.handle
    step_limit = 0.45
    current = handle.extract(handle.seed)
    with handle.exclusive() as scene:
        start_pose = np.asarray(
            scene.forwardKinematics(
                handle.expand(current), solver.tip_link, solver.base_link,
            ),
            dtype=float,
        )

    latencies = []
    steps = 0
    for index in range(1, 13):
        target = start_pose.copy()
        target[2, 3] += 0.008 * index
        started = time.perf_counter()
        try:
            solution = solver.solve_continuation(
                target, current, max_joint_step_rad=step_limit,
            )
        except IKFailure:
            break
        latencies.append(time.perf_counter() - started)
        assert float(np.max(np.abs(solution.joints - current))) <= step_limit
        # Bounding the step is half the contract.  A ``solve_continuation`` that
        # returns ``current`` unchanged satisfies the bound perfectly and tracks
        # nothing -- and passed this test in its earlier form, all 12 samples.
        position, _geodesic, transverse, axial = _measured(
            solver, stack, target, solution,
        )
        assert position < stack.ik.position_tolerance_m
        assert transverse < stack.ik.orientation_tolerance_rad
        assert axial < stack.ik.orientation_free_axis_tolerance_rad
        current = solution.joints
        steps += 1

    assert steps >= 8, f"continuation gave up after {steps} of 12 samples"
    milliseconds = np.asarray(latencies) * 1e3
    print(
        f"\nroboplan continuation: {steps}/12 steps, "
        f"mean {milliseconds.mean():.2f} ms, "
        f"p95 {np.percentile(milliseconds, 95):.2f} ms",
    )


@pytest.mark.usefixtures("real")
def test_real_model_solutions_are_deterministic(real):
    """Two identical requests must return identical joints.

    ``SimpleIk``'s own restarts draw from the ``Scene`` RNG, which RRT shares;
    this solver uses its own Halton ladder instead, so determinism holds without
    reseeding shared state.
    """

    solver, _stack = real
    compared = 0
    for target in _sample_targets(solver, 8, 23, colliding=False):
        try:
            first = solver.solve(target)
        except IKFailure:
            continue
        second = solver.solve(target)
        assert first.joints == pytest.approx(second.joints, abs=0.0, rel=0.0)
        assert first.seed_index == second.seed_index
        assert first.manipulability == pytest.approx(second.manipulability)
        compared += 1
    # Without this the test is vacuous when every target fails, which is exactly
    # the state a broken solver is in.
    assert compared >= 6, f"only {compared} of 8 targets were solved at all"
