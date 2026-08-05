"""``roboplan_runtime`` owns the facts every adapter would otherwise re-derive.

Four failure modes here are silent, which is why each gets a sharp test rather
than a shallow one:

1.  **Stale Scene.**  The URDF is generated into a sibling repo by a build step
    that rewrites it in place, so a regeneration can land on the same size and
    even the same mtime.  A stat-keyed cache would keep planning against the old
    geometry and never say so -- ``test_urdf_content_change_rebuilds`` pins the
    content hash.

2.  **DBL_MAX acceleration.**  roboplan reports acceleration limits from the
    URDF, and ``go2w_sensored.urdf`` declares none: measured, every arm entry is
    1.797e308.  TOPPRA turns that into a zero-duration segment, i.e. a step
    command to the servo.  The fake Scene below returns DBL_MAX deliberately.

3.  **The wheels.**  ``nq=28`` but ``nv=24``; the four Go2-W wheels are
    continuous joints stored as ``(cos, sin)``, so a zero-filled full vector is
    not merely wrong, it is *invalid*, and a 6-vector arm config is not a slice.

4.  **The lock.**  roboplan documents Scene collision queries as not thread
    safe.  An unsynchronized query does not crash; it returns a plausible
    verdict computed from another thread's scratch.

Everything except the last two tests runs without roboplan, against a fake Scene
that reproduces the real model's shape (nq=28, arm at q[20:26], DBL_MAX
accelerations) -- so a green suite off the 4090 really does cover the caching,
the key derivation, the locking and the vector algebra.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from z_manip.planning import roboplan_runtime as rt
from z_manip.planning.roboplan_runtime import (
    RoboplanConfig,
    RoboplanUnavailable,
    scene_handle,
)

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
SRDF = ROOT / "ros2/z_manip_motion/config/piper.srdf"
CAPSULES = ROOT / "configs/piper_collision_capsules.json"
HOME = ROOT / "configs/piper_home.json"

# The real model, measured: nq=28, nv=24, arm at q[20:26] / v[16:22].
NQ = 28
ARM_Q = (20, 21, 22, 23, 24, 25)
ARM_V = (16, 17, 18, 19, 20, 21)
# Four continuous wheels carried as (cos, sin) pairs.
WHEEL_Q = tuple(range(12, 20))
DBL_MAX = 1.7976931348623157e308
ACCELERATION = (1.5, 1.5, 1.5, 2.0, 2.0, 2.0)  # configs/go2w_piper.json


class _Group:
    def __init__(self) -> None:
        self.q_indices = np.array(ARM_Q, dtype=np.int32)
        self.v_indices = np.array(ARM_V, dtype=np.int32)
        self.joint_names = [f"piper_joint{i}" for i in range(1, 7)]


class FakeScene:
    """A stand-in with the real model's shape, limits and wheel encoding."""

    def __init__(self, *, colliding: bool = False, velocity: tuple[float, ...] | None = None):
        self.lower = np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
        self.upper = np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.0944])
        self.velocity = np.array(velocity if velocity is not None else (5.0,) * 5 + (3.0,))
        self.colliding = colliding
        self.calls = 0

    def getJointGroupInfo(self, name):
        assert name == "piper_arm"
        return _Group()

    def getCurrentJointPositions(self):
        # Deliberately the degenerate all-zeros the module docstring warns about.
        return np.zeros(NQ)

    def clampToValidConfiguration(self, q):
        out = np.array(q, dtype=float, copy=True)
        for i in range(0, len(WHEEL_Q), 2):
            pair = out[list(WHEEL_Q[i:i + 2])]
            norm = float(np.linalg.norm(pair))
            out[list(WHEEL_Q[i:i + 2])] = (1.0, 0.0) if norm < 1e-9 else pair / norm
        out[list(ARM_Q)] = np.clip(out[list(ARM_Q)], self.lower, self.upper)
        return out

    def getPositionLimitVectors(self, group_name=""):
        return self.lower.copy(), self.upper.copy()

    def getVelocityLimitVectors(self, group_name=""):
        return -self.velocity.copy(), self.velocity.copy()

    def getAccelerationLimitVectors(self, group_name=""):
        # What the real Scene returns for this URDF.  Nothing may use it.
        return np.full(6, -DBL_MAX), np.full(6, DBL_MAX)

    def hasCollisions(self, q, debug=False):
        self.calls += 1
        return self.colliding


class Robot:
    """The stack config's ``robot`` block, duck-typed."""

    def __init__(self, urdf: Path, acceleration=ACCELERATION):
        self.urdf_path = urdf
        self.base_link = "piper_base_link"
        self.tip_link = "piper_gripper_base"
        self.acceleration_limits = acceleration


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """An enabled config over copies of the real inputs, with a fake Scene."""

    rt.clear_scene_cache()
    urdf = tmp_path / "robot.urdf"
    urdf.write_bytes(URDF.read_bytes() if URDF.exists() else b"<robot name='x'/>")
    for source, name in ((SRDF, "planning.srdf"), (CAPSULES, "capsules.json"), (HOME, "home.json")):
        (tmp_path / name).write_bytes(source.read_bytes())

    built: list[FakeScene] = []

    def build(config, urdf_path, base_link, tip_link, acceleration_limits):
        # Recorded, not ignored: the Scene reports DBL_MAX acceleration unless
        # these reach it, and an unbounded Scene silently un-constrains TOPP-RA.
        # test_the_scene_is_told_the_deployment_acceleration_limits pins that.
        built.append(FakeScene())
        built[-1].built_with_acceleration = np.asarray(acceleration_limits, float)
        return built[-1]

    monkeypatch.setattr(rt, "_new_scene", build)
    config = RoboplanConfig(
        enabled=True,
        scene_cache_dir=tmp_path / "out",
        srdf_path=tmp_path / "planning.srdf",
        collision_model_path=tmp_path / "capsules.json",
        home_path=tmp_path / "home.json",
    )
    yield config, Robot(urdf), built
    rt.clear_scene_cache()


# ---------------------------------------------------------------- default off


def test_default_config_is_off_and_never_touches_roboplan(fake, monkeypatch):
    """This checkout is a live deploy: nothing may build a Scene by accident."""

    config, robot, built = fake
    monkeypatch.setattr(
        rt,
        "_new_scene",
        lambda *a, **k: pytest.fail("a disabled backend built a Scene"),
    )
    assert RoboplanConfig().enabled is False
    with pytest.raises(RoboplanUnavailable) as error:
        scene_handle(RoboplanConfig(), robot)
    assert error.value.reason == "disabled"
    assert not built


def test_importing_the_module_does_not_import_roboplan():
    """The whole point of the lazy import: this module must load anywhere.

    Checked in a subprocess because by the time this test runs, a sibling test
    in the same session may already have imported roboplan for its own reasons.
    """

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import z_manip.planning.roboplan_runtime as m;"
            " assert 'roboplan' not in sys.modules;"
            " assert m.RoboplanConfig().enabled is False",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_enabled_must_be_a_real_boolean():
    """``bool("false")`` is True, and this flag swaps the planner on a robot."""

    with pytest.raises(ValueError, match="boolean"):
        RoboplanConfig.from_mapping({"enabled": "false"})


@pytest.mark.skipif(
    importlib.util.find_spec("roboplan") is not None,
    reason="roboplan is installed here; the missing-library path cannot be exercised",
)
def test_missing_library_reports_a_named_reason(tmp_path):
    """On the NUC and on any non-4090 host this is the path that must be taken.

    Deliberately calls the unpatched ``_new_scene``: the point is that the
    ``ImportError`` is mapped to a named reason a caller can fall back on,
    rather than escaping as a bare import failure.
    """

    rt.clear_scene_cache()
    config = RoboplanConfig(enabled=True, scene_cache_dir=tmp_path)
    with pytest.raises(RoboplanUnavailable) as error:
        scene_handle(config, Robot(URDF))
    assert error.value.reason == "import-failed"


# ------------------------------------------------------------------- caching


def test_scene_is_built_once_and_shared(fake):
    config, robot, built = fake
    first = scene_handle(config, robot)
    second = scene_handle(config, robot)
    assert first is second
    assert len(built) == 1


def test_solver_knobs_do_not_invalidate_the_scene(fake):
    """Retuning IK must not pay the 618 ms rebuild -- it changes no geometry."""

    config, robot, built = fake
    first = scene_handle(config, robot)
    second = scene_handle(replace(config, ik_timeout_s=0.2, rrt_max_iterations=99), robot)
    assert first is second
    assert len(built) == 1


def test_urdf_content_change_rebuilds_even_at_an_unchanged_mtime(fake):
    """The sibling repo regenerates this file in place; mtime is not evidence."""

    config, robot, built = fake
    scene_handle(config, robot)
    stat = robot.urdf_path.stat()
    text = robot.urdf_path.read_text()
    robot.urdf_path.write_text(text.replace("robot", "robot ", 1))
    os.utime(robot.urdf_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert robot.urdf_path.stat().st_mtime_ns == stat.st_mtime_ns
    scene_handle(config, robot)
    assert len(built) == 2


def test_capsule_model_change_rebuilds(fake):
    """The capsule JSON is the enforced collision model -- editing it is safety."""

    config, robot, built = fake
    scene_handle(config, robot)
    model = json.loads(config.collision_model_path.read_text())
    model["capsules"][0]["radius"] = float(model["capsules"][0]["radius"]) + 0.01
    config.collision_model_path.write_text(json.dumps(model))
    scene_handle(config, robot)
    assert len(built) == 2


def test_home_pose_change_rebuilds(fake):
    config, robot, built = fake
    scene_handle(config, robot)
    home = json.loads(config.home_path.read_text())
    home["joint_radians"][0] += 0.01
    config.home_path.write_text(json.dumps(home))
    assert scene_handle(config, robot).seed[ARM_Q[0]] == pytest.approx(
        home["joint_radians"][0]
    )
    assert len(built) == 2


def test_missing_input_fails_closed(fake):
    config, robot, built = fake
    config.collision_model_path.unlink()
    with pytest.raises(ValueError, match="could not read roboplan input"):
        scene_handle(config, robot)
    assert not built


# ------------------------------------------------------------- derived facts


def test_acceleration_limits_come_from_the_deployment_not_the_urdf(fake):
    """The URDF declares none; roboplan answers DBL_MAX; TOPPRA would divide by it."""

    config, robot, _ = fake
    handle = scene_handle(config, robot)
    assert np.all(np.isfinite(handle.acceleration_limits))
    assert handle.acceleration_limits == pytest.approx(ACCELERATION)
    assert np.all(handle.acceleration_limits < 1e3)


def test_non_finite_velocity_limit_fails_closed(fake, monkeypatch):
    config, robot, _ = fake
    monkeypatch.setattr(
        rt,
        "_new_scene",
        lambda *a, **k: FakeScene(velocity=(5.0, 5.0, 5.0, 5.0, 5.0, np.inf)),
    )
    with pytest.raises(ValueError, match="finite"):
        scene_handle(config, robot)


def test_acceleration_limit_arity_mismatch_fails_closed(fake):
    config, robot, _ = fake
    with pytest.raises(ValueError, match="acceleration limits"):
        scene_handle(config, Robot(robot.urdf_path, acceleration=(1.5,) * 5))


def test_shared_vectors_are_read_only(fake):
    """One handle serves every adapter; a scribble would poison all of them."""

    config, robot, _ = fake
    handle = scene_handle(config, robot)
    for vector in (
        handle.seed,
        handle.velocity_limits,
        handle.acceleration_limits,
        handle.q_indices,
        *handle.position_limits,
    ):
        with pytest.raises(ValueError):
            vector[0] = 0.0


# ---------------------------------------------------------------- seed & home


def test_seed_keeps_the_wheels_valid(fake):
    """``np.zeros(nq)`` leaves (cos, sin) = (0, 0) on all four wheels."""

    config, robot, _ = fake
    handle = scene_handle(config, robot)
    assert handle.nq == NQ
    assert handle.dof == 6
    for i in range(0, len(WHEEL_Q), 2):
        pair = handle.seed[list(WHEEL_Q[i:i + 2])]
        assert float(np.linalg.norm(pair)) == pytest.approx(1.0)


def test_seed_is_the_commanded_home_pose(fake):
    config, robot, _ = fake
    home = json.loads(config.home_path.read_text())["joint_radians"]
    assert scene_handle(config, robot).seed[list(ARM_Q)] == pytest.approx(home)


def test_home_outside_the_joint_limits_is_not_silently_clamped(fake):
    """A stale home file must fail loudly, not seed every solve from elsewhere."""

    config, robot, _ = fake
    home = json.loads(config.home_path.read_text())
    home["joint_radians"][1] = -1.0  # joint2's lower limit is 0.0
    config.home_path.write_text(json.dumps(home))
    with pytest.raises(ValueError, match="outside the model joint limits"):
        scene_handle(config, robot)


def test_colliding_home_pose_refuses_to_seed(fake, monkeypatch):
    """q_arm = 0 self-collides by 27 mm; a colliding seed poisons every solve."""

    config, robot, _ = fake
    monkeypatch.setattr(rt, "_new_scene", lambda *a, **k: FakeScene(colliding=True))
    with pytest.raises(ValueError, match="self-collision"):
        scene_handle(config, robot)


def test_home_pose_arity_mismatch_fails_closed(fake):
    config, robot, _ = fake
    home = json.loads(config.home_path.read_text())
    home["joint_radians"] = home["joint_radians"][:5]
    config.home_path.write_text(json.dumps(home))
    with pytest.raises(ValueError, match="home pose"):
        scene_handle(config, robot)


# ------------------------------------------------------- arm <-> full vectors


def test_expand_extract_round_trip_preserves_the_non_arm_dofs(fake):
    config, robot, _ = fake
    handle = scene_handle(config, robot)
    q_arm = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    full = handle.expand(q_arm)

    assert full.shape == (NQ,)
    assert handle.extract(full) == pytest.approx(q_arm)
    untouched = [i for i in range(NQ) if i not in ARM_Q]
    assert full[untouched] == pytest.approx(handle.seed[untouched])
    # A fresh array every call: two adapters must not alias one buffer.
    assert full is not handle.expand(q_arm)
    handle.expand(q_arm)[0] = 99.0
    assert handle.seed[0] != 99.0


def test_expand_accepts_an_explicit_base(fake):
    config, robot, _ = fake
    handle = scene_handle(config, robot)
    base = handle.expand(np.zeros(6))
    base[0] = 1.25  # a different chassis pose
    full = handle.expand(np.full(6, 0.1), base=base)
    assert full[0] == pytest.approx(1.25)
    assert handle.extract(full) == pytest.approx(np.full(6, 0.1))


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros(7), np.zeros((1, 6)), np.zeros(NQ)])
def test_expand_rejects_a_wrong_shaped_arm_vector(fake, bad):
    """Handing expand() a full vector is the classic version of this bug."""

    config, robot, _ = fake
    handle = scene_handle(config, robot)
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        handle.expand(bad)


def test_expand_rejects_a_non_finite_arm_vector(fake):
    config, robot, _ = fake
    handle = scene_handle(config, robot)
    with pytest.raises(ValueError, match="finite"):
        handle.expand([0.0, 0.0, np.nan, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros(NQ - 1), np.zeros((NQ, 1))])
def test_extract_rejects_a_wrong_shaped_full_vector(fake, bad):
    config, robot, _ = fake
    handle = scene_handle(config, robot)
    with pytest.raises(ValueError, match=r"shape \(28,\)"):
        handle.extract(bad)


# --------------------------------------------------------------- thread safety


def test_scene_is_unreachable_without_the_lock(fake):
    """Not a style rule: an unlocked query returns a plausible WRONG verdict."""

    config, robot, _ = fake
    handle = scene_handle(config, robot)
    with pytest.raises(RuntimeError, match="not thread safe"):
        handle.scene
    with handle.exclusive() as scene:
        assert handle.scene is scene


def test_exclusive_serializes_peer_threads(fake):
    config, robot, _ = fake
    handle = scene_handle(config, robot)
    entered = threading.Event()
    peer_got_in = threading.Event()

    def peer():
        with handle.exclusive():
            peer_got_in.set()

    with handle.exclusive():
        thread = threading.Thread(target=peer, daemon=True)
        thread.start()
        entered.set()
        assert not peer_got_in.wait(0.25), "a peer thread entered a held Scene lock"
    thread.join(timeout=5.0)
    assert peer_got_in.is_set()
    assert not thread.is_alive()


def test_exclusive_is_reentrant_and_restores_ownership(fake):
    """A helper that takes the lock may be called from inside a holder."""

    config, robot, _ = fake
    handle = scene_handle(config, robot)
    with handle.exclusive():
        with handle.exclusive():
            assert handle.scene is not None
        # The inner exit must not revoke the outer holder's access.
        assert handle.scene is not None
    with pytest.raises(RuntimeError):
        handle.scene


def test_other_threads_cannot_read_the_scene_property(fake):
    config, robot, _ = fake
    handle = scene_handle(config, robot)
    seen: list[object] = []

    def peer():
        try:
            handle.scene
        except RuntimeError as error:
            seen.append(error)

    with handle.exclusive():
        thread = threading.Thread(target=peer)
        thread.start()
        thread.join(timeout=5.0)
    assert len(seen) == 1


# ------------------------------------------------------- config validation


@pytest.mark.parametrize(
    "overrides",
    [
        {"ik_position_tolerance_m": 0.0},
        {"ik_orientation_tolerance_rad": -0.01},
        {"ik_timeout_s": float("inf")},
        {"ik_max_iterations": 0},
        {"ik_restarts": 0},
        {"rrt_timeout_s": 0.0},
        {"rrt_step_rad": -0.1},
        {"rrt_collision_resolution_rad": 0.0},
        {"rrt_max_iterations": 0},
        {"velocity_scale": 0.0},
        {"velocity_scale": 1.5},
        {"acceleration_scale": 0.0},
        {"acceleration_scale": 1.01},
        {"group": ""},
    ],
)
def test_config_rejects_unusable_values(overrides):
    with pytest.raises(ValueError):
        RoboplanConfig(**overrides)


def test_from_mapping_coerces_paths_and_keeps_the_backend_off_by_default():
    config = RoboplanConfig.from_mapping({"srdf_path": "~/piper.srdf"})
    assert config.enabled is False
    assert isinstance(config.srdf_path, Path)
    assert not str(config.srdf_path).startswith("~")


def test_from_mapping_rejects_a_typo_rather_than_defaulting_it():
    with pytest.raises(ValueError, match="unknown roboplan fields"):
        RoboplanConfig.from_mapping({"enabled": True, "ik_restart": 4})


def test_from_mapping_rejects_a_non_mapping():
    with pytest.raises(ValueError, match="must be an object"):
        RoboplanConfig.from_mapping([("enabled", True)])


# ------------------------------------------------------------- real roboplan


@pytest.fixture(scope="module")
def real_handle(tmp_path_factory):
    pytest.importorskip("roboplan.core", reason="roboplan is a 4090-only dependency")
    if not URDF.exists():
        pytest.skip("robot URDF not checked out")
    rt.clear_scene_cache()
    config = RoboplanConfig(
        enabled=True,
        scene_cache_dir=tmp_path_factory.mktemp("roboplan"),
        srdf_path=SRDF,
        collision_model_path=CAPSULES,
        home_path=HOME,
    )
    handle = scene_handle(config, Robot(URDF))
    yield handle
    rt.clear_scene_cache()


def test_real_scene_matches_the_derived_facts(real_handle):
    """The fake above claims nq=28 and arm at q[20:26]; check that on the model."""

    assert real_handle.nq == NQ
    assert real_handle.q_indices.tolist() == list(ARM_Q)
    assert real_handle.v_indices.tolist() == list(ARM_V)
    assert np.all(np.isfinite(real_handle.velocity_limits))
    assert real_handle.acceleration_limits == pytest.approx(ACCELERATION)
    lower, upper = real_handle.position_limits
    home = np.asarray(json.loads(HOME.read_text())["joint_radians"], float)
    assert np.all(lower <= home) and np.all(home <= upper)


def test_real_seed_is_valid_and_collision_free_where_zeros_are_not(real_handle):
    """Pins both documented gotchas at once, on the real model.

    ``np.zeros(nq)`` is INVALID (the wheels' (cos, sin) pairs degenerate) and
    ``q_arm = 0`` is genuinely self-colliding (mount vs wrist, 27 mm).  The seed
    this module hands every adapter must be neither.
    """

    with real_handle.exclusive() as scene:
        assert scene.isValidConfiguration(real_handle.seed)
        assert not scene.hasCollisions(real_handle.seed)
        assert not scene.isValidConfiguration(np.zeros(real_handle.nq))
        assert scene.hasCollisions(real_handle.expand(np.zeros(6)))


def test_real_scene_accepts_the_read_only_seed(real_handle):
    """The handle's vectors are read-only; the bindings must still take them."""

    with real_handle.exclusive() as scene:
        assert scene.hasCollisions(real_handle.seed) is False
        pose = scene.forwardKinematics(real_handle.seed, "piper_gripper_base")
        assert pose.shape == (4, 4) and np.all(np.isfinite(pose))


def test_the_scene_is_told_the_deployment_acceleration_limits(fake):
    """The Scene must learn the caps, or TOPP-RA runs unconstrained.

    ``go2w_sensored.urdf`` declares no acceleration limits and Pinocchio does
    not parse URDF 1.2 extended ones, so a Scene built without the joint-limit
    YAML answers DBL_MAX.  ``roboplan.toppra`` reads its limits FROM the Scene,
    so that is not cosmetic: TOPP-RA then has no acceleration constraint and
    the caller is left auditing output after the fact.  Measured that way, 1
    path in 8 came back 1.72x over the requested cap.
    """

    config, robot, built = fake
    handle = scene_handle(config, robot)
    assert handle is not None
    assert len(built) == 1
    np.testing.assert_allclose(
        built[0].built_with_acceleration,
        np.asarray(robot.acceleration_limits, dtype=float),
    )
