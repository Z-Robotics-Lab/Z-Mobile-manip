"""The roboplan backends are wired in, and OFF, and provably still off.

This file exists for one assertion and its neighbours: with no ``roboplan``
config section and no env var, ``OnlinePlanner`` builds exactly what it built
before this branch -- ``RobustIKSolver``, ``JointSpaceRRTConnect``,
``retime_path`` -- and never touches roboplan at all.  Everything under
``z_manip/planning/roboplan_*`` is opt-in on a live deploy; the switch is the
only thing standing between a running robot and a planner nobody has flown yet,
so the OFF path is tested harder than the ON one.

The second reason it exists is ``SceneVerifiedJointPlanner``.  The roboplan
joint planner searches against the capsule ``Scene``, which contains the robot
and nothing else: it cannot see the perceived point cloud.  Selecting it without
re-checking its output would silently delete every perceived obstacle from
transit planning, and the plan would look perfectly healthy on the way through
the box.  ``test_a_path_the_deployed_checker_rejects_is_not_returned`` is that
guard, and the bad waypoint sits in the MIDDLE of the path on purpose: an
implementation that only checked the endpoints would otherwise pass.

Nothing here needs roboplan or pinocchio.  The switches are exercised with
fakes, because the failure they can produce is a wiring failure, not a numerical
one -- the numerics are pinned in the per-adapter suites.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from z_manip.configuration import load_stack_config
from z_manip.kinematics.robust_ik import RobustIKSolver
from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning import online_planner as op
from z_manip.planning.online_planner import OnlinePlanner
from z_manip.planning.roboplan_runtime import RoboplanConfig, RoboplanUnavailable
from z_manip.planning.roboplan_wiring import (
    env_backend,
    RoboplanBackends,
    SceneVerifiedJointPlanner,
)
from z_manip.planning.rrt_connect import JointSpaceRRTConnect
from z_manip.planning.time_parameterization import retime_path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
DEPLOYED = ROOT / "configs/go2w_piper.json"

SWITCHES = (
    "Z_MANIP_IK_BACKEND",
    "Z_MANIP_JOINT_PLANNER_BACKEND",
    "Z_MANIP_TIMING_BACKEND",
)

try:  # the inverse of the usual gate: one test needs roboplan to be ABSENT
    import roboplan.core  # noqa: F401

    HAS_ROBOPLAN = True
except Exception:  # pragma: no cover - depends on the machine, not the branch
    HAS_ROBOPLAN = False


@pytest.fixture
def clean_env(monkeypatch):
    """No developer's shell decides which planner the robot flies."""

    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture(scope="module")
def deployed_config():
    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    return load_stack_config(DEPLOYED, environ={"Z_MANIP_ROBOT_URDF": str(URDF)})


@pytest.fixture(scope="module")
def stale_config(tmp_path_factory):
    """The bind-mounted config from before this branch: no roboplan section."""

    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    values = json.loads(DEPLOYED.read_text())
    values.pop("roboplan")
    values["robot"]["urdf_path"] = str(URDF)
    values["collision_model"] = str(ROOT / "configs/piper_collision_capsules.json")
    path = tmp_path_factory.mktemp("stale") / "stack.json"
    path.write_text(json.dumps(values))
    return load_stack_config(path, environ={})


# --------------------------------------------------------------- the OFF path


@pytest.mark.parametrize("fixture", ("deployed_config", "stale_config"))
def test_the_default_stack_builds_exactly_what_it_built_before(
    fixture,
    request,
    clean_env,
):
    """The one test that stands between this branch and the robot.

    Run twice: once against the deployed config (which now carries an explicit
    disabled section) and once against a config with no section at all, which
    is what a stale read-only mount looks like while the package is already
    hot-reloaded.  Both must produce the pre-branch stack.
    """

    config = request.getfixturevalue(fixture)

    def never(*args, **kwargs):
        raise AssertionError("the default stack built a roboplan Scene")

    clean_env.setattr(op, "scene_handle", never)

    planner = OnlinePlanner(config)

    assert planner.ik_backend == "robust"
    assert isinstance(planner.ik, RobustIKSolver)
    # 'pinocchio' sets a checker here and 'robust' does not; the default must
    # stay the one that leaves every configured capsule pair with the
    # point-cloud checker.
    assert planner.mesh_self_collision is None
    assert planner.ik_ranker is planner.ik
    assert planner.joint_planner_backend == "rrt"
    assert planner.timing_backend == "quintic"
    assert planner.retime is retime_path
    assert planner._roboplan_planner is None

    predicate = object()
    built = planner._joint_planner(predicate)
    assert type(built) is JointSpaceRRTConnect
    assert built.state_valid is predicate
    assert built.config is config.rrt
    assert built.joint_names == planner.chain.joint_names
    assert np.array_equal(built.lower_limits, planner.chain.lower_limits)
    assert np.array_equal(built.upper_limits, planner.chain.upper_limits)


def test_the_deployed_config_keeps_every_roboplan_backend_off(deployed_config):
    """configs/go2w_piper.json is the live deploy; it says so explicitly."""

    section = deployed_config.roboplan
    assert section is not None, "the deployed config lost its roboplan section"
    assert section.runtime.enabled is False
    assert (section.ik, section.joint_planner, section.timing) == (False, False, False)


def test_a_config_without_the_section_still_loads(stale_config):
    """The stale-mount case: an older config file must not stop the stack."""

    assert stale_config.roboplan is None


def test_every_planner_and_retimer_goes_through_the_switch():
    """Both switches must cover every call site, not just the ones I found.

    A third ``JointSpaceRRTConnect(...)`` or a direct ``retime_path(...)``
    added later would quietly opt itself out of the backend selection, and
    nothing at runtime would say so.
    """

    source = (ROOT / "z_manip/planning/online_planner.py").read_text()
    assert source.count("JointSpaceRRTConnect(\n") == 1
    assert source.count("retime_path(\n") == 0
    assert source.count("self.retime(\n") == 2
    assert source.count("self._joint_planner(") == 2


@pytest.mark.parametrize(
    "variable,value,message",
    (
        ("Z_MANIP_IK_BACKEND", "moveit", "unsupported Z_MANIP_IK_BACKEND: 'moveit'"),
        (
            "Z_MANIP_JOINT_PLANNER_BACKEND",
            "ompl",
            "unsupported Z_MANIP_JOINT_PLANNER_BACKEND: 'ompl'",
        ),
        (
            "Z_MANIP_TIMING_BACKEND",
            "toppra",
            "unsupported Z_MANIP_TIMING_BACKEND: 'toppra'",
        ),
    ),
)
def test_an_unknown_backend_name_is_a_startup_error(
    variable,
    value,
    message,
    deployed_config,
    clean_env,
):
    """Unchanged for IK, and the same shape for the two new switches."""

    clean_env.setenv(variable, value)
    with pytest.raises(ValueError, match=re.escape(message)):
        OnlinePlanner(deployed_config)


# ---------------------------------------------------------------- the ON path


def _enabled(**selectors):
    return RoboplanBackends(runtime=RoboplanConfig(enabled=True), **selectors)


def test_selecting_roboplan_ik_swaps_the_solver_and_keeps_the_numpy_ranker(
    deployed_config,
    clean_env,
):
    """The ranker is the part that would have crashed the first live plan.

    ``online_planner`` asks its IK solver for a seed pose ranker whenever the
    caller supplies none, which is every ordinary plan.  roboplan's SimpleIk has
    no such method, so without the fallback this backend raises AttributeError
    in the middle of ``_plan`` rather than at construction.
    """

    seen = {}

    class FakeSolver:
        def __init__(self, config, robot_config, ik_config=None):
            seen["args"] = (config, robot_config, ik_config)

    clean_env.setattr(op, "RoboplanIKSolver", FakeSolver)
    config = replace(deployed_config, roboplan=_enabled(ik=True))

    planner = OnlinePlanner(config)

    assert isinstance(planner.ik, FakeSolver)
    assert planner.ik_backend == "roboplan"
    assert seen["args"] == (config.roboplan.runtime, config.robot, config.ik)
    # The capsule Scene checks itself inside the solver; the point-cloud checker
    # keeps every configured pair, exactly as it does for 'robust'.
    assert planner.mesh_self_collision is None
    assert isinstance(planner.ik_ranker, RobustIKSolver)
    assert planner.ik_ranker is not planner.ik
    assert callable(planner.ik_ranker.make_seed_pose_ranker)


def test_an_env_var_overrides_the_config_selector(deployed_config, clean_env):
    """A deployment that opted in can still be pinned back from the shell."""

    clean_env.setattr(op, "RoboplanIKSolver", lambda *a, **k: object())
    clean_env.setenv("Z_MANIP_IK_BACKEND", " Robust ")
    config = replace(deployed_config, roboplan=_enabled(ik=True))

    planner = OnlinePlanner(config)

    assert planner.ik_backend == "robust"
    assert isinstance(planner.ik, RobustIKSolver)


def test_selecting_the_roboplan_planner_wraps_it_in_the_deployed_checker(
    deployed_config,
    clean_env,
):
    """The switch must not hand the raw roboplan planner to the pipeline."""

    handle = object()
    built = {}

    class FakePlanner:
        def __init__(self, scene, *, config=None):
            built["scene"] = scene
            built["config"] = config
            self.joint_names = names

    clean_env.setattr(op, "scene_handle", lambda *a, **k: handle)
    clean_env.setattr(op, "RoboplanJointPlanner", FakePlanner)
    config = replace(deployed_config, roboplan=_enabled(joint_planner=True))
    names = tuple(OnlinePlanner(deployed_config).chain.joint_names)

    planner = OnlinePlanner(config)
    composed = planner._joint_planner(lambda joints: True)

    assert planner.joint_planner_backend == "roboplan"
    assert built["scene"] is handle
    # The deployed RRT knobs, not a second copy transcribed from RoboplanConfig.
    assert built["config"] is config.rrt
    assert isinstance(composed, SceneVerifiedJointPlanner)
    assert composed._search is planner._roboplan_planner
    assert type(composed._verifier) is JointSpaceRRTConnect
    # Timing was not selected, so the quintic stays.
    assert planner.retime is retime_path


def test_selecting_roboplan_timing_replaces_retime_path(deployed_config, clean_env):
    """Both retiming call sites follow the switch, including the hold path."""

    class FakeRetimer:
        def __init__(self, scene):
            self.scene = scene

        def retime_path(self, waypoints, velocity, acceleration, config=None):
            return ("retimed", np.asarray(waypoints).shape)

    clean_env.setattr(op, "scene_handle", lambda *a, **k: "handle")
    clean_env.setattr(op, "ToppraRetimer", FakeRetimer)
    config = replace(deployed_config, roboplan=_enabled(timing=True))

    planner = OnlinePlanner(config)
    path = np.linspace(0.0, 0.4, 3 * planner.chain.dof).reshape(3, planner.chain.dof)

    assert planner.timing_backend == "roboplan"
    assert planner.retime is not retime_path
    assert planner._retime_joint_path(path) == ("retimed", (3, planner.chain.dof))
    # The planner was not selected, so the numpy RRT stays.
    assert planner._roboplan_planner is None
    assert type(planner._joint_planner(lambda joints: True)) is JointSpaceRRTConnect


def test_one_scene_serves_both_roboplan_backends(deployed_config, clean_env):
    """Two handles would mean two 618 ms builds and two independent locks."""

    calls = []

    clean_env.setattr(
        op,
        "scene_handle",
        lambda *args, **kwargs: calls.append(args) or "handle",
    )
    clean_env.setattr(op, "RoboplanJointPlanner", lambda scene, config=None: scene)
    clean_env.setattr(
        op,
        "ToppraRetimer",
        lambda scene: type("R", (), {"retime_path": staticmethod(lambda *a: None)})(),
    )
    config = replace(
        deployed_config,
        roboplan=_enabled(joint_planner=True, timing=True),
    )

    OnlinePlanner(config)

    assert len(calls) == 1


# -------------------------------------------------------- fail-closed opt-in


def test_a_disabled_backend_fails_at_construction_with_a_named_reason(
    deployed_config,
    clean_env,
):
    """The forced-on-but-disabled case, which is the same on every machine.

    Construction, not the first plan: a bringup that dies three seconds into a
    grasp is a bringup nobody can diagnose from the robot.
    """

    clean_env.setenv("Z_MANIP_JOINT_PLANNER_BACKEND", "roboplan")

    with pytest.raises(RoboplanUnavailable) as failure:
        OnlinePlanner(deployed_config)

    assert failure.value.reason == "disabled"


@pytest.mark.skipif(HAS_ROBOPLAN, reason="this test needs roboplan to be absent")
def test_enabling_roboplan_without_the_library_names_the_reason(
    deployed_config,
    clean_env,
):
    """Not an ImportError from inside a planning call three layers down."""

    config = replace(deployed_config, roboplan=_enabled(joint_planner=True))

    with pytest.raises(RoboplanUnavailable) as failure:
        OnlinePlanner(config)

    assert failure.value.reason == "import-failed"


def test_the_real_stack_composes_with_every_backend_enabled(deployed_config, clean_env):
    """The one thing fakes cannot answer: does the composition hold on the model.

    Two integration risks live here and nowhere else.  The SRDF group order and
    the URDF chain order are maintained in different files, and
    ``SceneVerifiedJointPlanner`` refuses to run if they disagree.  And the path
    roboplan emits has to be a shape the deployed ``JointSpaceRRTConnect``
    predicate accepts -- a width or dtype mismatch would make ``segment_valid``
    return False for a reason that has nothing to do with collision, and every
    plan would be rejected as if the scene were full.
    """

    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    config = replace(
        deployed_config,
        roboplan=_enabled(ik=True, joint_planner=True, timing=True),
    )

    planner = OnlinePlanner(config)

    assert type(planner.ik).__name__ == "RoboplanIKSolver"
    assert planner.retime is not retime_path
    assert isinstance(planner.ik_ranker, RobustIKSolver)

    home = np.asarray(
        json.loads((ROOT / "configs/piper_home.json").read_text())["joint_radians"],
        dtype=float,
    )
    goal = home.copy()
    goal[1] += 0.2

    permissive = planner._joint_planner(lambda joints: True)
    assert isinstance(permissive, SceneVerifiedJointPlanner)
    path = permissive.plan_joint(home, goal, timeout_s=5.0)
    assert path.joint_names == planner.chain.joint_names
    assert path.waypoints.shape[1] == planner.chain.dof
    assert np.allclose(path.waypoints[0], home)
    assert np.allclose(path.waypoints[-1], goal)

    # Same plan, deployed checker refusing everything: the verification is what
    # decides, not roboplan's own verdict on its own Scene.
    with pytest.raises(PlanningError, match="its Scene cannot see"):
        planner._joint_planner(lambda joints: False).plan_joint(home, goal)


# ------------------------------------------------------------ the config block


def test_a_selector_without_enabled_is_rejected_at_load(tmp_path):
    """A deployment that believes it switched a backend and did not."""

    with pytest.raises(ValueError, match="ik is selected but roboplan enabled is false"):
        RoboplanBackends.from_mapping({"enabled": False, "ik": True})


def test_a_typo_in_the_section_is_rejected(tmp_path):
    """RoboplanConfig owns the knobs; a misspelled one must not default."""

    with pytest.raises(ValueError, match="unknown roboplan fields"):
        RoboplanBackends.from_mapping({"enabled": True, "ik_timeout": 1.0})


@pytest.mark.parametrize("value", ("true", 1, None))
def test_a_non_boolean_selector_is_rejected(value):
    """``bool("false")`` is True, and this flag swaps a planner on a robot."""

    with pytest.raises(ValueError, match="selector must be a boolean"):
        RoboplanBackends.from_mapping({"enabled": True, "timing": value})


def test_the_section_reaches_the_stack_config(tmp_path):
    """End to end through the real loader, including the optional-section path."""

    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    values = json.loads(DEPLOYED.read_text())
    values["robot"]["urdf_path"] = str(URDF)
    values["collision_model"] = str(ROOT / "configs/piper_collision_capsules.json")
    values["roboplan"] = {"enabled": True, "timing": True, "ik_restarts": 3}
    path = tmp_path / "stack.json"
    path.write_text(json.dumps(values))

    config = load_stack_config(path, environ={})

    assert config.roboplan.timing is True
    assert config.roboplan.ik is False
    assert config.roboplan.runtime.ik_restarts == 3
    assert config.schema_version == 2  # an OPTIONAL section is not a migration


# ------------------------------------------------------------- env_backend


def test_env_backend_prefers_the_environment_over_the_selector():
    assert env_backend("V", "rrt", True, {"V": "rrt"}) == "rrt"
    assert env_backend("V", "rrt", False, {"V": "roboplan"}) == "roboplan"
    assert env_backend("V", "rrt", True, {}) == "roboplan"
    assert env_backend("V", "rrt", False, {}) == "rrt"


def test_env_backend_normalizes_like_the_existing_ik_switch():
    assert env_backend("V", "quintic", False, {"V": "  RoboPlan\n"}) == "roboplan"


def test_env_backend_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unsupported V: 'rrt-connect'"):
        env_backend("V", "rrt", False, {"V": "rrt-connect"})


# ------------------------------------------------ SceneVerifiedJointPlanner


NAMES = ("j1", "j2")


class _Search:
    """Stands in for RoboplanJointPlanner: plans, sees no point cloud."""

    def __init__(self, waypoints):
        self.joint_names = NAMES
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.calls = []

    def plan_joint(self, start, goal, *, timeout_s=5.0, control=None):
        self.calls.append((start, goal, timeout_s, control))
        return JointTrajectory(NAMES, self.waypoints)


class _Verifier:
    """Stands in for JointSpaceRRTConnect: the deployed collision predicate."""

    def __init__(self, blocked=()):
        self.joint_names = NAMES
        self.blocked = [np.asarray(state, dtype=float) for state in blocked]
        self.segments = []

    def segment_valid(self, first, second, *, control=None):
        self.segments.append((np.asarray(first), np.asarray(second), control))
        return not any(
            np.allclose(state, first) or np.allclose(state, second)
            for state in self.blocked
        )


def test_a_path_the_deployed_checker_rejects_is_not_returned():
    """The whole reason the wrapper exists.

    roboplan's Scene holds the robot and nothing else, so a path straight
    through a perceived obstacle comes back as a clean success.  Five waypoints
    with the third blocked, because that is the only arrangement no *pair* at
    either end of the path touches: verifying just the first and last segment
    has to fail this test, and it does.
    """

    path = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]
    planner = SceneVerifiedJointPlanner(_Search(path), _Verifier(blocked=([0.2, 0.0],)))

    with pytest.raises(PlanningError, match="crosses geometry its Scene cannot see"):
        planner.plan_joint([0.0, 0.0], [0.4, 0.0])
    with pytest.raises(PlanningError, match="segment 1 of 4"):
        planner.plan_joint([0.0, 0.0], [0.4, 0.0])


def test_an_accepted_path_is_returned_after_every_segment_is_checked():
    """Full coverage, in order, with the caller's control passed through."""

    path = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]
    search = _Search(path)
    verifier = _Verifier()
    planner = SceneVerifiedJointPlanner(search, verifier)
    control = object()

    result = planner.plan_joint([0.0, 0.0], [0.2, 0.0], timeout_s=2.0, control=control)

    assert np.array_equal(result.waypoints, np.asarray(path))
    assert search.calls == [([0.0, 0.0], [0.2, 0.0], 2.0, control)]
    assert [
        (first.tolist(), second.tolist(), seen is control)
        for first, second, seen in verifier.segments
    ] == [
        ([0.0, 0.0], [0.1, 0.0], True),
        ([0.1, 0.0], [0.2, 0.0], True),
    ]


def test_a_degenerate_path_is_still_collision_checked():
    """A start-equals-goal plan is the one a caller executes without looking."""

    planner = SceneVerifiedJointPlanner(
        _Search([[0.4, 0.4]]),
        _Verifier(blocked=([0.4, 0.4],)),
    )

    with pytest.raises(PlanningError, match="segment 0 of 1"):
        planner.plan_joint([0.4, 0.4], [0.4, 0.4])


def test_an_empty_path_is_rejected_rather_than_waved_through():
    planner = SceneVerifiedJointPlanner(_Search(np.zeros((0, 2))), _Verifier())

    with pytest.raises(PlanningError, match="empty joint path"):
        planner.plan_joint([0.0, 0.0], [0.1, 0.0])


def test_segment_valid_asks_the_deployed_checker_not_the_scene():
    """The capsule Scene's pairs are a subset of the checker's; skip the lock."""

    search = _Search([[0.0, 0.0], [0.1, 0.0]])
    verifier = _Verifier()
    planner = SceneVerifiedJointPlanner(search, verifier)

    assert planner.segment_valid([0.0, 0.0], [0.1, 0.0]) is True
    assert len(verifier.segments) == 1
    assert search.calls == []


def test_disagreeing_joint_order_is_refused_at_construction():
    """The SRDF group and the URDF chain are maintained independently."""

    search = _Search([[0.0, 0.0]])
    search.joint_names = ("j2", "j1")

    with pytest.raises(ValueError, match="disagree on joint order"):
        SceneVerifiedJointPlanner(search, _Verifier())
