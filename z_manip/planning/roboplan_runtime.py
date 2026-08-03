"""The one place a roboplan adapter gets a ``Scene``.

Every roboplan-backed adapter (IK, RRT, TOPPRA, OInK) needs the same three
things, and gets them wrong in the same three ways, so they are owned here once
and ``build_scene`` is called from nowhere else.

*   **The Scene itself.**  Building one costs 624 ms (measured) and depends only
    on the URDF, the SRDF and the enforced capsule model -- never on a solver
    tunable -- so it is built once per process and shared by content hash; a hit
    is 0.062 ms.

*   **The derived facts.**  ``q_indices``, the limit vectors and a safe seed.
    The model is ``nq=28`` but ``nv=24``: the four Go2-W wheels are CONTINUOUS
    joints carried as ``(cos, sin)`` pairs, so ``np.zeros(nq)`` is not a valid
    configuration and a 6-vector arm configuration is not a slice of a full one.
    ``expand``/``extract`` are the only supported conversion.

*   **Serialization.**  roboplan's own documentation states that ``Scene``
    collision queries are not thread safe -- they mutate shared scratch, and
    ``hasCollisionsAlongPath`` says so explicitly.  This planner runs under a
    deadline-driven control plane that may call from more than one thread, so
    the ``Scene`` is reachable only inside ``SceneHandle.exclusive()``; reading
    ``handle.scene`` without the lock raises instead of corrupting silently.

Default OFF.  ``RoboplanConfig.enabled`` is ``False`` and ``scene_handle`` fails
closed with reason ``disabled`` until a deployment explicitly turns it on, so
adding this module changes no running code path.

roboplan is imported lazily inside functions: this module must import cleanly on
a host that has neither roboplan nor pinocchio.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np

from .staged_trajectory import _readonly_vector

__all__ = [
    "RoboplanConfig",
    "RoboplanUnavailable",
    "SceneHandle",
    "scene_handle",
    "clear_scene_cache",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


class RoboplanUnavailable(RuntimeError):
    """roboplan was selected but this runtime cannot provide it.

    ``reason`` is a stable slug callers may branch on when deciding whether to
    fall back to the numpy planner: ``disabled`` (the deployment never opted
    in) or ``import-failed`` (opted in, but the library is not installed --
    every NUC, and any host that is not the 4090).
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(
            f"roboplan unavailable ({reason})" + (f": {detail}" if detail else ""),
        )
        self.reason = reason


@dataclass(frozen=True)
class RoboplanConfig:
    """Opt-in settings for the roboplan planning backend.

    Only the four paths and ``group`` reach the ``Scene``; every solver knob
    below them is read by a downstream adapter and is deliberately *not* part of
    the scene cache key, so retuning IK does not pay the 618 ms rebuild.
    """

    #: Default OFF.  This checkout is a live deploy; nothing may switch to
    #: roboplan without a deployment explicitly setting this.
    enabled: bool = False
    #: Where the derived capsule-only URDF/SRDF pair is written.  These are
    #: build products of ``roboplan_scene.write_planning_model``, not inputs.
    scene_cache_dir: Path = Path("/tmp/z_manip_roboplan")
    srdf_path: Path = _REPO_ROOT / "ros2/z_manip_motion/config/piper.srdf"
    collision_model_path: Path = _REPO_ROOT / "configs/piper_collision_capsules.json"
    home_path: Path = _REPO_ROOT / "configs/piper_home.json"
    group: str = "piper_arm"

    # IK (roboplan SimpleIk).  Tolerances mirror ``IKConfig`` so a solution
    # accepted by either backend is accepted by both; measured 0.82 ms mean /
    # 1.65 ms p95 per solve, so the budget below is ~360 p95 solves.
    ik_position_tolerance_m: float = 0.003
    ik_orientation_tolerance_rad: float = 0.03
    ik_max_iterations: int = 220
    ik_restarts: int = 12
    ik_timeout_s: float = 0.60

    # RRT-Connect.  ``step_rad``/``collision_resolution_rad`` mirror
    # ``RRTConnectConfig``; measured 270 ms mean over 16/16 solves.
    rrt_timeout_s: float = 5.0
    rrt_step_rad: float = 0.18
    rrt_collision_resolution_rad: float = 0.035
    rrt_max_iterations: int = 4000

    # TOPPRA.  Same scales ``TimeParameterizationConfig`` already ships, so
    # swapping the retimer does not silently speed the arm up.
    velocity_scale: float = 0.65
    acceleration_scale: float = 0.55

    def __post_init__(self) -> None:
        # Not coerced: this is the flag that swaps the planner on a live deploy,
        # and ``bool("false")`` is True.
        if not isinstance(self.enabled, bool):
            raise ValueError("roboplan enabled must be a boolean")
        for name in ("scene_cache_dir", "srdf_path", "collision_model_path", "home_path"):
            value = getattr(self, name)
            if not str(value):
                raise ValueError(f"roboplan {name} must be a path")
            object.__setattr__(self, name, Path(value).expanduser())
        if not self.group:
            raise ValueError("roboplan group name must not be empty")
        positive = (
            self.ik_position_tolerance_m,
            self.ik_orientation_tolerance_rad,
            self.ik_timeout_s,
            self.rrt_timeout_s,
            self.rrt_step_rad,
            self.rrt_collision_resolution_rad,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "roboplan tolerances, step sizes, and timeouts must be positive",
            )
        if self.ik_max_iterations < 1 or self.rrt_max_iterations < 1:
            raise ValueError("roboplan iteration limits must be at least one")
        if self.ik_restarts < 1:
            raise ValueError("roboplan IK needs at least one start")
        if not 0.0 < self.velocity_scale <= 1.0:
            raise ValueError("velocity scale must be in (0, 1]")
        if not 0.0 < self.acceleration_scale <= 1.0:
            raise ValueError("acceleration scale must be in (0, 1]")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "RoboplanConfig":
        """Parse an optional stack-config ``roboplan`` section.

        Unknown fields are rejected -- a typo'd knob that silently kept its
        default is how a deployment ends up believing it retuned the planner.
        Missing fields keep their defaults, including ``enabled=False``: this
        block is opt-in, and requiring a deployment to spell out every solver
        tunable just to switch it on is how blocks get copy-pasted stale.
        """

        if not isinstance(values, Mapping):
            raise ValueError("roboplan section must be an object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown roboplan fields: {sorted(unknown)}")
        return cls(**dict(values))


class SceneHandle:
    """A shared ``Scene`` plus the facts adapters would otherwise re-derive.

    Every vector it exposes is read-only: one handle serves every adapter in the
    process, so an adapter scribbling on ``seed`` or on a limit vector would
    corrupt all of them.
    """

    def __init__(
        self,
        scene: object,
        *,
        group: str,
        q_indices: np.ndarray,
        v_indices: np.ndarray,
        position_limits: tuple[np.ndarray, np.ndarray],
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        seed: np.ndarray,
    ) -> None:
        self._scene = scene
        self.group = str(group)
        self.q_indices = q_indices
        self.v_indices = v_indices
        self.position_limits = position_limits
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits
        self.seed = seed
        self.dof = int(len(q_indices))
        self.nq = int(len(seed))
        # ponytail: one global lock per Scene, serializing every collision query
        # through it.  Overhead is not the issue -- measured 0.48 us per acquire
        # against a 12-27 us capsule check, ~4%.  The ceiling is CONCURRENCY:
        # the whole process does one collision query at a time however many
        # planner threads exist, and a 270 ms RRT call blocks every peer for its
        # duration.  Upgrade path: roboplan exposes no per-thread scratch, so
        # the fix is one Scene per worker thread (624 ms and a full collision
        # model each) keyed by thread ident, not a finer-grained lock.  Today
        # the control plane plans one request at a time.
        self._lock = threading.RLock()
        self._owner: int | None = None

    @property
    def scene(self) -> object:
        """The roboplan ``Scene`` -- only inside :meth:`exclusive`.

        Guarded rather than merely documented because the failure it prevents is
        silent: an unsynchronized collision query returns a *plausible* verdict
        computed from another thread's half-written scratch.
        """

        if self._owner != threading.get_ident():
            raise RuntimeError(
                "roboplan Scene accessed without SceneHandle.exclusive(); "
                "Scene collision queries are not thread safe",
            )
        return self._scene

    @contextmanager
    def exclusive(self) -> Iterator[object]:
        """Hold the scene lock for the duration of a block of Scene calls.

        Reentrant, so a helper that takes the lock may be called from inside a
        caller that already holds it.
        """

        with self._lock:
            previous = self._owner
            self._owner = threading.get_ident()
            try:
                yield self._scene
            finally:
                self._owner = previous

    def expand(self, q_arm: object, *, base: object | None = None) -> np.ndarray:
        """Scatter a ``dof``-vector arm configuration into a full ``nq`` vector.

        ``base`` defaults to the safe seed, which matters: the wheels are
        continuous joints stored as ``(cos, sin)``, so a zero-filled base leaves
        ``(0, 0)`` on all four and every query against it is invalid.
        """

        arm = np.asarray(q_arm, dtype=float)
        if arm.shape != (self.dof,):
            raise ValueError(
                f"arm configuration must have shape ({self.dof},), got {arm.shape}",
            )
        if not np.all(np.isfinite(arm)):
            raise ValueError("arm configuration must be finite")
        full = np.array(self.seed if base is None else base, dtype=float, copy=True)
        if full.shape != (self.nq,):
            raise ValueError(
                f"base configuration must have shape ({self.nq},), got {full.shape}",
            )
        full[self.q_indices] = arm
        return full

    def extract(self, q_full: object) -> np.ndarray:
        """Gather the arm's ``dof`` values out of a full ``nq`` vector."""

        full = np.asarray(q_full, dtype=float)
        if full.shape != (self.nq,):
            raise ValueError(
                f"full configuration must have shape ({self.nq},), got {full.shape}",
            )
        return np.array(full[self.q_indices], dtype=float)


# Process-resident Scene cache.  Keyed on the CONTENT of every input file, not
# their stat identity: the URDF is generated into a sibling repo (go2W_Sim) by a
# build step that rewrites it in place, so a same-size regeneration is routine
# and an mtime-keyed cache would happily keep planning against the old geometry.
# Hashing all four inputs costs 0.024 ms (measured, 41 kB URDF).
_SCENE_CACHE: "OrderedDict[tuple[object, ...], SceneHandle]" = OrderedDict()
# Two: one live robot, plus a slot so a mid-session URDF regeneration does not
# evict-and-rebuild if the old model is still referenced.
_SCENE_CACHE_CAPACITY = 2
_SCENE_CACHE_LOCK = threading.Lock()


def clear_scene_cache() -> None:
    """Drop every cached Scene.  For tests and for a deliberate model reload."""

    with _SCENE_CACHE_LOCK:
        _SCENE_CACHE.clear()


def _content_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        try:
            digest.update(path.read_bytes())
        except OSError as error:
            raise ValueError(f"could not read roboplan input {path}: {error}") from error
        digest.update(b"\0")
    return digest.hexdigest()


def _new_scene(
    config: RoboplanConfig,
    urdf_path: Path,
    base_link: str,
    tip_link: str,
    acceleration_limits: np.ndarray,
) -> object:
    """Import roboplan and build the Scene.  The only import site, and the seam
    the host-side tests replace with a fake."""

    try:
        import roboplan.core  # noqa: F401,PLC0415 -- 4090-only dependency
    except ImportError as error:
        raise RoboplanUnavailable(
            "import-failed",
            f"{type(error).__name__}: {error}",
        ) from error

    from .roboplan_scene import build_scene  # noqa: PLC0415 -- pulls roboplan

    return build_scene(
        urdf_path,
        config.srdf_path,
        config.collision_model_path,
        base_link=base_link,
        tip_link=tip_link,
        out_dir=config.scene_cache_dir,
        # Without this the Scene reports DBL_MAX acceleration and any TOPP-RA
        # built on it is unconstrained -- see roboplan_scene.write_joint_limits.
        # These limits are already in the cache key, so a config that changes
        # them cannot be served a stale Scene.
        acceleration_limits=acceleration_limits,
    )


def _readonly_indices(value: object, label: str) -> np.ndarray:
    indices = np.array(value, dtype=int, copy=True)
    if indices.ndim != 1 or len(indices) < 1 or np.any(indices < 0):
        raise ValueError(f"{label} must be a non-empty vector of non-negative indices")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError(f"{label} contains a repeated index")
    indices.setflags(write=False)
    return indices


def _home_joints(path: Path, dof: int) -> np.ndarray:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load home pose {path}: {error}") from error
    if not isinstance(raw, dict) or "joint_radians" not in raw:
        raise ValueError(f"home pose {path} must carry joint_radians")
    home = _readonly_vector(raw["joint_radians"], f"home pose {path}", dof)
    return np.array(home, dtype=float)


def _build_handle(
    config: RoboplanConfig,
    urdf_path: Path,
    base_link: str,
    tip_link: str,
    acceleration_limits: np.ndarray,
) -> SceneHandle:
    scene = _new_scene(config, urdf_path, base_link, tip_link, acceleration_limits)
    group = scene.getJointGroupInfo(config.group)
    q_indices = _readonly_indices(group.q_indices, "group q indices")
    v_indices = _readonly_indices(group.v_indices, "group v indices")
    dof = len(q_indices)
    if dof != len(acceleration_limits):
        raise ValueError(
            f"group {config.group} has {dof} joints but "
            f"{len(acceleration_limits)} acceleration limits were configured",
        )

    lower, upper = (
        _readonly_vector(bound, "roboplan position limit", dof)
        for bound in scene.getPositionLimitVectors(config.group)
    )
    if np.any(lower >= upper):
        raise ValueError("roboplan position limits are empty on at least one joint")
    # roboplan reports velocity/acceleration limits straight from the URDF, and
    # go2w_sensored.urdf declares no <limit effort=... > acceleration at all:
    # measured, every arm entry comes back as 1.7976931348623157e+308 (DBL_MAX).
    # Handing that to TOPPRA yields a zero-duration segment -- a step command to
    # the servo.  So acceleration comes from the deployment config only (same
    # split online_planner.py:1042 already uses), and a non-finite VELOCITY
    # limit, which would fail the same way, fails closed here instead.
    velocity = _readonly_vector(
        scene.getVelocityLimitVectors(config.group)[1],
        "roboplan velocity limit",
        dof,
    )
    if np.any(velocity <= 0.0):
        raise ValueError("roboplan reports a non-positive joint velocity limit")

    home = _home_joints(config.home_path, dof)
    seed = np.array(
        scene.clampToValidConfiguration(scene.getCurrentJointPositions()),
        dtype=float,
    )
    seed[q_indices] = home
    seed = np.array(scene.clampToValidConfiguration(seed), dtype=float)
    # Clamping legitimately renormalizes the wheels' (cos, sin) pairs, so only
    # the arm is compared: a home file outside the URDF limits must NOT be
    # silently clamped into a different pose that every solve then seeds from.
    if not np.allclose(seed[q_indices], home, atol=1e-9):
        raise ValueError(
            f"home pose {config.home_path} lies outside the model joint limits",
        )
    # q_arm = 0 is genuinely self-colliding on this robot (mount vs wrist, 27 mm
    # overlap); home clears every enforced pair by 3.8 mm.  If the seed collides,
    # every collision-checked solve starts inside an obstacle and the failure
    # shows up as unexplained IK/RRT failure rates, not as an error here.
    if scene.hasCollisions(seed):
        raise ValueError(
            f"home pose {config.home_path} is in self-collision; refusing to seed from it",
        )

    return SceneHandle(
        scene,
        group=config.group,
        q_indices=q_indices,
        v_indices=v_indices,
        position_limits=(lower, upper),
        velocity_limits=velocity,
        acceleration_limits=acceleration_limits,
        seed=_readonly_vector(seed, "seed configuration"),
    )


def scene_handle(config: RoboplanConfig, robot_config: object) -> SceneHandle:
    """Return the process-wide ``SceneHandle`` for this robot, building on miss.

    ``robot_config`` is the stack config's ``robot`` block: ``urdf_path``,
    ``base_link``, ``tip_link`` and ``acceleration_limits``.  Raises
    :class:`RoboplanUnavailable` when the backend is off or the library is
    absent, and ``ValueError`` when an input file is missing or stale.
    """

    if not config.enabled:
        raise RoboplanUnavailable(
            "disabled",
            "RoboplanConfig.enabled is False; the numpy planner stays in charge",
        )

    urdf_path = Path(robot_config.urdf_path).expanduser().resolve()
    base_link = str(robot_config.base_link)
    tip_link = str(robot_config.tip_link)
    acceleration = _readonly_vector(
        robot_config.acceleration_limits,
        "robot acceleration limits",
    )
    if np.any(acceleration <= 0.0):
        raise ValueError("robot acceleration limits must be positive")

    key = (
        _content_digest(
            (urdf_path, config.srdf_path, config.collision_model_path, config.home_path),
        ),
        str(urdf_path),
        base_link,
        tip_link,
        config.group,
        str(config.scene_cache_dir),
        tuple(acceleration.tolist()),
    )
    # Built under the lock, not outside it: unlike the reduced-model cache in
    # pinocchio_ik this fires once per process, so serializing a concurrent
    # second caller behind the 618 ms build is strictly cheaper than letting it
    # build and discard a duplicate Scene.
    with _SCENE_CACHE_LOCK:
        cached = _SCENE_CACHE.get(key)
        if cached is not None:
            _SCENE_CACHE.move_to_end(key)
            return cached
        handle = _build_handle(config, urdf_path, base_link, tip_link, acceleration)
        _SCENE_CACHE[key] = handle
        while len(_SCENE_CACHE) > _SCENE_CACHE_CAPACITY:
            _SCENE_CACHE.popitem(last=False)
        return handle
