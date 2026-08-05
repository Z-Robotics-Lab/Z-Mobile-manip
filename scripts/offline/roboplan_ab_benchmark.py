#!/usr/bin/env python3
"""A/B the deployed numpy planning stack against the opt-in roboplan backends.

Offline and transport-free: it loads the real URDF, the enforced capsule model
and the deployed stack config, runs both paths over identical fixed-seed inputs
and prints a table.  It opens no transport, touches no hardware, issues no
motion command and never enables roboplan in the deployed config -- the
``RoboplanConfig`` built here lives and dies inside this process.

What each row does and does NOT measure:

* **IK.**  ``RobustIKSolver`` does not collision-check during the solve, so its
  solve rate and its *usable* (collision-free) solve rate are two different
  numbers.  Both are printed; the second is the one a grasp pipeline can spend.
  Pose error for both backends is measured by one independent oracle
  (``KinematicChain.forward``), never by the solver's own report.
* **Planning.**  Both planners are handed the same start/goal pairs, the same
  ``RRTConnectConfig`` and the same world.  This row therefore measures SEARCH,
  not model.  The deployment difference -- roboplan's ``Scene`` cannot see the
  perceived point cloud at all -- is a wiring fact, not a benchmark result; see
  ``z_manip/planning/roboplan_wiring.py``.  To keep the row honest the scene
  cloud is placed out of the arm's reach and that inertness is asserted, not
  assumed.
* **Collision checking.**  The two checkers are not interchangeable: the
  point-cloud checker tests the perceived cloud *and* the enforced
  self-collision pairs and adds the deployment clearance, while the ``Scene``
  tests self-collision only against deliberately inflated capsule radii.  The
  agreement counts are what say whether the us/state figures are comparable at
  all.

A backend that will not build degrades to a skipped row carrying the reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from z_manip.collision.pointcloud import (  # noqa: E402
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
)
from z_manip.configuration import load_stack_config  # noqa: E402
from z_manip.kinematics.chain import KinematicChain, rotation_log  # noqa: E402
from z_manip.kinematics.robust_ik import IKFailure, RobustIKSolver  # noqa: E402
from z_manip.models.planner import PlanningError  # noqa: E402
from z_manip.planning.roboplan_ik import RoboplanIKSolver  # noqa: E402
from z_manip.planning.roboplan_planner import RoboplanJointPlanner  # noqa: E402
from z_manip.planning.roboplan_runtime import (  # noqa: E402
    RoboplanConfig,
    RoboplanUnavailable,
    scene_handle,
)
from z_manip.planning.roboplan_timing import RetimingFailure, ToppraRetimer  # noqa: E402
from z_manip.planning.rrt_connect import JointSpaceRRTConnect, RRTConnectConfig  # noqa: E402
from z_manip.planning.time_parameterization import retime_path  # noqa: E402

SCHEMA = "z_mobile_manip.roboplan_ab_benchmark.v1"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/go2w_piper.json"

# The scene cloud exists only to satisfy the point-cloud checker's fail-closed
# freshness gate, so it is parked a metre BELOW the arm base -- further than the
# PiPER's ~0.7 m reach plus the 0.01 m clearance and 0.003 m point radius.  If a
# capsule ever did touch it the planning row would silently stop being an A/B of
# two searches over one world, so ``_inert_scene_cloud`` asserts the inertness
# instead of trusting this comment.
_SCENE_PLANE_Z_M = -1.0
_SCENE_POINTS = 4000
_SCENE_HALF_SPAN_M = 1.0

# Enough restarts of the timed loop to swamp ``perf_counter`` overhead on a
# 27 us call without turning the row into a minute of wall time.
_COLLISION_REPEATS = 3

# The per-call planning timeout.  Not a config field -- ``JointPlanner.plan_joint``
# takes it as an argument -- so it is pinned here and handed to both backends.
_PLAN_TIMEOUT_S = 5.0


class Skipped(RuntimeError):
    """A backend this host cannot exercise.  Degrades one row, not the run."""


# ----------------------------------------------------------------- reporting


@dataclass(frozen=True)
class Metric:
    """One measured quantity on both paths, plus which direction is better."""

    label: str
    old: float | int | None
    new: float | int | None
    fmt: str = "{:.3f}"
    better: str | None = None  # "high" | "low" | None for context-only rows

    def winner(self) -> str:
        if self.better is None or self.old is None or self.new is None:
            return "-"
        old, new = float(self.old), float(self.new)
        # A NaN reaches here whenever a side measured nothing (every path
        # rejected, say).  Comparing it silently declares a winner, and every
        # NaN comparison is False, so the loser would be whichever side the
        # arithmetic happened to favour.
        if not (np.isfinite(old) and np.isfinite(new)):
            return "-"
        if abs(new - old) <= 1e-12:
            return "tie"
        return "new" if (new > old) == (self.better == "high") else "old"

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "old": self.old,
            "new": self.new,
            "better": self.better,
            "winner": self.winner(),
        }


@dataclass(frozen=True)
class Comparison:
    name: str
    old_label: str
    new_label: str
    notes: tuple[str, ...] = ()
    metrics: tuple[Metric, ...] = ()
    skipped: str | None = None

    def to_json(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "name": self.name,
            "old": self.old_label,
            "new": self.new_label,
            "notes": list(self.notes),
        }
        if self.skipped is not None:
            document["skipped"] = self.skipped
        else:
            document["metrics"] = [metric.to_json() for metric in self.metrics]
        return document


def _cell(value: float | int | None, fmt: str) -> str:
    return "n/a" if value is None else fmt.format(value)


def render(comparisons: list[Comparison], header: dict[str, Any]) -> str:
    lines = [
        "=" * 84,
        f"{SCHEMA}",
        f"  model  {header['urdf']}",
        f"  config {header['config']}",
        f"  seed   {header['seed']}    roboplan {header['roboplan_version']}",
        "=" * 84,
    ]
    for comparison in comparisons:
        lines.append("")
        lines.append(f"{comparison.name.upper()}")
        lines.append(f"  old: {comparison.old_label}")
        lines.append(f"  new: {comparison.new_label}")
        for note in comparison.notes:
            lines.append(f"  . {note}")
        if comparison.skipped is not None:
            lines.append(f"  SKIPPED: {comparison.skipped}")
            continue
        lines.append(
            f"  {'metric':<40}{'old':>13}{'new':>13}{'winner':>9}",
        )
        lines.append(f"  {'-' * 75}")
        for metric in comparison.metrics:
            lines.append(
                f"  {metric.label:<40}"
                f"{_cell(metric.old, metric.fmt):>13}"
                f"{_cell(metric.new, metric.fmt):>13}"
                f"{metric.winner():>9}",
            )
    lines.append("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ fixtures


def _timed(call: Callable[[], Any]) -> tuple[Any, float, BaseException | None]:
    """Run ``call`` and return (result, seconds, error).  Never raises."""

    started = time.perf_counter()
    try:
        result = call()
    except (IKFailure, PlanningError, RetimingFailure, ValueError) as error:
        return None, time.perf_counter() - started, error
    return result, time.perf_counter() - started, None


def _stats(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _inert_scene_cloud(
    checker: PointCloudCollisionChecker,
    states: np.ndarray,
) -> None:
    """Fail closed unless the parked cloud blocks nothing the arm can reach.

    The planning row only compares two searches if both search the same world.
    A cloud that clipped a capsule would make the numpy planner solve a strictly
    harder problem and the row would read as a roboplan win.
    """

    blocked = [
        int(index)
        for index, joints in enumerate(states)
        if checker.check_state(joints).kind == "scene"
    ]
    if blocked:
        raise RuntimeError(
            f"the parked benchmark cloud is inside the arm's reach: "
            f"{len(blocked)} of {len(states)} sampled states hit it "
            f"(first index {blocked[0]}); the planning row would not be an A/B",
        )


@dataclass
class Fixtures:
    """Everything both paths share, built once from the deployed config."""

    stack: Any
    chain: KinematicChain
    collision_model: RobotCollisionModel
    checker: PointCloudCollisionChecker
    rng_seed: int
    _handle: Any = field(default=None, init=False, repr=False)
    _handle_error: str | None = field(default=None, init=False, repr=False)

    @property
    def handle(self) -> Any:
        """The shared ``SceneHandle``; a build failure skips a row, not the run."""

        if self._handle is not None:
            return self._handle
        if self._handle_error is not None:
            raise Skipped(self._handle_error)
        try:
            self._handle = scene_handle(self.roboplan_config, self.stack.robot)
        except (RoboplanUnavailable, ValueError, RuntimeError) as error:
            self._handle_error = f"{type(error).__name__}: {error}"
            raise Skipped(self._handle_error) from error
        return self._handle

    @property
    def roboplan_config(self) -> RoboplanConfig:
        # enabled=True here and ONLY here: this benchmark opts itself in, the
        # deployed configs/go2w_piper.json stays false.
        return RoboplanConfig(enabled=True)

    def rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng(self.rng_seed + salt)

    def collision_free_states(self, count: int, salt: int) -> np.ndarray:
        """Sample in-limit states that BOTH checkers call collision-free.

        Both, because the ``Scene``'s radii are inflated and the point-cloud
        checker adds the deployment clearance: a state only one of them accepts
        would make one backend fail on an endpoint for a model reason and the
        row would report that as a search failure.
        """

        rng = self.rng(salt)
        scene_ok = self.scene_validity()
        accepted: list[np.ndarray] = []
        # Bounded: uniform sampling of this arm's joint box lands collision-free
        # well over half the time, so 40x headroom is generous rather than tuned.
        for _ in range(40 * count):
            if len(accepted) >= count:
                break
            joints = rng.uniform(self.chain.lower_limits, self.chain.upper_limits)
            if self.checker.is_state_valid(joints) and scene_ok(joints):
                accepted.append(joints)
        if len(accepted) < count:
            raise Skipped(
                f"only {len(accepted)} of {count} sampled states were "
                "collision-free under both checkers",
            )
        return np.asarray(accepted, dtype=float)

    def scene_validity(self) -> Callable[[np.ndarray], bool]:
        handle = self.handle

        def valid(joints: np.ndarray) -> bool:
            with handle.exclusive() as scene:
                return not bool(scene.hasCollisions(handle.expand(joints)))

        return valid


def build_fixtures(config_path: Path, seed: int, sanity_states: int) -> Fixtures:
    stack = load_stack_config(config_path)
    chain = KinematicChain.from_urdf(
        stack.robot.urdf_path, stack.robot.base_link, stack.robot.tip_link,
    )
    model = RobotCollisionModel.from_mapping(
        json.loads(stack.collision_model_path.read_text(encoding="utf-8")),
    )
    stamp_s = 1_000_000.0
    checker = PointCloudCollisionChecker(
        chain=chain,
        model=model,
        frame_provider=chain.link_transforms,
        config=PointCloudCollisionConfig(
            clearance=model.scene_clearance_m,
            point_radius=model.point_radius_m,
            scene_noise_tolerance=model.scene_noise_tolerance_m,
            scene_noise_min_support_points=model.scene_noise_min_support_points,
            segment_joint_step=stack.rrt.collision_resolution,
        ),
        now_fn=lambda: stamp_s,
    )
    rng = np.random.default_rng(seed)
    cloud = np.column_stack((
        rng.uniform(-_SCENE_HALF_SPAN_M, _SCENE_HALF_SPAN_M, _SCENE_POINTS),
        rng.uniform(-_SCENE_HALF_SPAN_M, _SCENE_HALF_SPAN_M, _SCENE_POINTS),
        np.full(_SCENE_POINTS, _SCENE_PLANE_Z_M),
    ))
    checker.update_scene(cloud, stamp_s=stamp_s)
    _inert_scene_cloud(
        checker,
        rng.uniform(
            chain.lower_limits, chain.upper_limits, size=(sanity_states, chain.dof),
        ),
    )
    return Fixtures(
        stack=stack, chain=chain, collision_model=model, checker=checker, rng_seed=seed,
    )


# ---------------------------------------------------------------------- rows


def _pose_errors(chain: KinematicChain, joints: np.ndarray, goal: np.ndarray) -> tuple[float, float]:
    """Independent oracle: never the solver's own residual report."""

    actual = chain.forward(joints)
    position = float(np.linalg.norm(actual[:3, 3] - goal[:3, 3]))
    orientation = float(np.linalg.norm(rotation_log(goal[:3, :3] @ actual[:3, :3].T)))
    return position, orientation


@dataclass
class _IKRun:
    latency_s: list[float] = field(default_factory=list)
    solved: list[bool] = field(default_factory=list)
    collision_free: list[bool] = field(default_factory=list)
    position_error_m: list[float] = field(default_factory=list)
    orientation_error_rad: list[float] = field(default_factory=list)


def _run_ik(
    solver: Any,
    targets: np.ndarray,
    seed_joints: np.ndarray,
    chain: KinematicChain,
    scene_ok: Callable[[np.ndarray], bool],
) -> _IKRun:
    run = _IKRun()
    for goal in targets:
        solution, elapsed, error = _timed(lambda g=goal: solver.solve(g, seed_joints))
        run.latency_s.append(elapsed)
        run.solved.append(error is None)
        if solution is None:
            run.collision_free.append(False)
            continue
        # Collision is judged by the Scene for BOTH backends, so the column is
        # one predicate's opinion of two solvers rather than two predicates.
        run.collision_free.append(scene_ok(solution.joints))
        position, orientation = _pose_errors(chain, solution.joints, goal)
        run.position_error_m.append(position)
        run.orientation_error_rad.append(orientation)
    return run


def _throughput(
    latency_s: list[float], collision_free: list[bool], budget_s: float,
) -> tuple[int, int, bool]:
    """Candidates fully resolved, usable ones, and whether the budget ran out.

    The third value matters: a backend fast enough to finish the whole candidate
    list inside the budget reports a count that is the SAMPLE SIZE, not a
    throughput, and printing it without saying so invents a measurement.
    """

    elapsed = 0.0
    resolved = usable = 0
    for cost, free in zip(latency_s, collision_free):
        if elapsed + cost > budget_s:
            return resolved, usable, True
        elapsed += cost
        resolved += 1
        usable += int(free)
    return resolved, usable, False


def compare_ik(fixtures: Fixtures, count: int) -> tuple[Comparison, Comparison]:
    stack, chain = fixtures.stack, fixtures.chain
    rng = fixtures.rng(101)
    sources = rng.uniform(chain.lower_limits, chain.upper_limits, size=(count, chain.dof))
    targets = np.stack([chain.forward(joints) for joints in sources])
    seed_joints = np.asarray(
        json.loads(
            (REPOSITORY_ROOT / "configs/piper_home.json").read_text(encoding="utf-8"),
        )["joint_radians"],
        dtype=float,
    )
    budget_s = float(stack.ik.solve_timeout_s)
    notes = [
        f"{count} targets = FK of uniform in-limit joint samples, both solvers "
        "seeded from configs/piper_home.json",
        "pose error measured by KinematicChain.forward for both, not by the "
        "solver's own residual",
        "collision-free judged by the roboplan capsule Scene for both",
        "reported orientation error is the full geodesic; acceptance uses the "
        f"deployed anisotropic gate (free axis {stack.ik.orientation_free_axis_tolerance_rad} rad), "
        "so an accepted solution may legitimately exceed orientation_tolerance_rad",
    ]

    try:
        scene_ok = fixtures.scene_validity()
        roboplan_solver = RoboplanIKSolver(
            fixtures.roboplan_config, stack.robot, stack.ik,
        )
    except (Skipped, RoboplanUnavailable, ValueError) as error:
        reason = str(error) if isinstance(error, Skipped) else f"{type(error).__name__}: {error}"
        return (
            Comparison(
                "inverse kinematics", "RobustIKSolver", "RoboplanIKSolver",
                tuple(notes), skipped=reason,
            ),
            Comparison(
                f"candidate throughput inside ik.solve_timeout_s = {budget_s:.2f} s",
                "RobustIKSolver", "RoboplanIKSolver", (), skipped=reason,
            ),
        )

    old = _run_ik(RobustIKSolver(chain, stack.ik), targets, seed_joints, chain, scene_ok)
    new = _run_ik(roboplan_solver, targets, seed_joints, chain, scene_ok)
    source_free = sum(scene_ok(joints) for joints in sources)
    notes.append(
        f"{source_free}/{count} of the source configurations are themselves "
        "collision-free, so no backend can reach 100% usable",
    )
    notes.append(
        f"restart ladders differ by construction: IKConfig.random_seeds="
        f"{stack.ik.random_seeds} vs RoboplanConfig.ik_restarts="
        f"{fixtures.roboplan_config.ik_restarts}",
    )

    old_latency, new_latency = _stats(old.latency_s), _stats(new.latency_s)
    metrics = (
        Metric("solve rate (met tolerances)",
               sum(old.solved) / count, sum(new.solved) / count, "{:.1%}", "high"),
        Metric("usable rate (solved AND collision-free)",
               sum(old.collision_free) / count, sum(new.collision_free) / count,
               "{:.1%}", "high"),
        Metric("collision-free share of solutions",
               sum(old.collision_free) / max(sum(old.solved), 1),
               sum(new.collision_free) / max(sum(new.solved), 1), "{:.1%}", "high"),
        Metric("latency mean (ms)",
               1e3 * old_latency["mean"], 1e3 * new_latency["mean"], "{:.2f}", "low"),
        Metric("latency p95 (ms)",
               1e3 * old_latency["p95"], 1e3 * new_latency["p95"], "{:.2f}", "low"),
        Metric("position error mean (mm)",
               1e3 * _stats(old.position_error_m)["mean"],
               1e3 * _stats(new.position_error_m)["mean"], "{:.3f}", "low"),
        Metric("orientation error mean (mrad)",
               1e3 * _stats(old.orientation_error_rad)["mean"],
               1e3 * _stats(new.orientation_error_rad)["mean"], "{:.3f}", "low"),
    )
    ik = Comparison(
        "inverse kinematics", "RobustIKSolver (scipy, no collision check)",
        "RoboplanIKSolver (SimpleIk on the capsule Scene)", tuple(notes), metrics,
    )

    old_resolved, old_usable, old_spent = _throughput(
        old.latency_s, old.collision_free, budget_s,
    )
    new_resolved, new_usable, new_spent = _throughput(
        new.latency_s, new.collision_free, budget_s,
    )
    throughput_notes = [
        "derived from the per-candidate latencies measured above, in the same "
        "fixed candidate order",
        "one hard failure costs a backend its whole budget, which is the point "
        "of the row",
    ]
    for label, spent, resolved, latency in (
        ("old", old_spent, old_resolved, old.latency_s),
        ("new", new_spent, new_resolved, new.latency_s),
    ):
        if not spent:
            throughput_notes.append(
                f"{label} ran out of CANDIDATES, not budget: {resolved} is the "
                f"sample size ({sum(latency):.3f} s of the {budget_s:.2f} s "
                "spent).  Its true throughput is higher; raise --ik-targets",
            )
    throughput = Comparison(
        f"candidate throughput inside ik.solve_timeout_s = {budget_s:.2f} s",
        "RobustIKSolver", "RoboplanIKSolver",
        tuple(throughput_notes),
        (
            Metric("candidates fully resolved", old_resolved, new_resolved, "{:.0f}", "high"),
            Metric("of those, usable (collision-free)", old_usable, new_usable, "{:.0f}", "high"),
        ),
    )
    return ik, throughput


def _path_length(waypoints: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))


def _searchable_pairs(
    states: np.ndarray, old: Any, new: Any, queries: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Keep only the pairs BOTH backends actually have to search.

    Filtering on one backend's predicate would be a rigged row.  The deployment
    clearance makes the numpy predicate the stricter of the two on this model,
    so a pair it calls blocked is routinely one roboplan answers with its
    straight-line early return in microseconds -- and the latency column would
    read as a three-order-of-magnitude search win over a query roboplan never
    searched.  The rejected counts come back too: which side alone blocked a
    pair is the divergence between the two collision models, and it is more
    interesting than the latency.
    """

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    counts = {"examined": 0, "both_free": 0, "old_only": 0, "new_only": 0}
    for start, goal in zip(states[0::2], states[1::2]):
        counts["examined"] += 1
        old_blocked = not old.segment_valid(start, goal)
        new_blocked = not new.segment_valid(start, goal)
        if old_blocked and new_blocked:
            pairs.append((start, goal))
            if len(pairs) >= queries:
                break
        elif old_blocked:
            counts["old_only"] += 1
        elif new_blocked:
            counts["new_only"] += 1
        else:
            counts["both_free"] += 1
    return pairs, counts


def compare_planning(fixtures: Fixtures, queries: int, pool: int) -> Comparison:
    stack, chain = fixtures.stack, fixtures.chain
    old_label = "JointSpaceRRTConnect (numpy + PointCloudCollisionChecker)"
    new_label = "RoboplanJointPlanner (roboplan RRT-Connect on the Scene)"
    notes = [
        "identical RRTConnectConfig from configs/go2w_piper.json for both, "
        f"identical start/goal pairs, fixed seeds, timeout {_PLAN_TIMEOUT_S} s",
        "queries whose straight line is already free are excluded: both take "
        "the same early return and would only dilute the search comparison",
        "this row compares SEARCH only -- roboplan's Scene structurally cannot "
        "see the point cloud, which is why the wiring composes the two",
    ]
    try:
        handle = fixtures.handle
        states = fixtures.collision_free_states(2 * pool, salt=202)
        numpy_planner = JointSpaceRRTConnect(
            joint_names=chain.joint_names,
            lower_limits=chain.lower_limits,
            upper_limits=chain.upper_limits,
            state_valid=fixtures.checker.is_state_valid,
            config=stack.rrt,
        )
        roboplan_planner = RoboplanJointPlanner(handle, config=stack.rrt)
    except (Skipped, RoboplanUnavailable, ValueError) as error:
        reason = str(error) if isinstance(error, Skipped) else f"{type(error).__name__}: {error}"
        return Comparison("joint planning", old_label, new_label, tuple(notes), skipped=reason)

    pairs, counts = _searchable_pairs(states, numpy_planner, roboplan_planner, queries)
    if not pairs:
        return Comparison(
            "joint planning", old_label, new_label, tuple(notes),
            skipped=(
                f"no sampled pair of {counts['examined']} needed a search on "
                "both backends"
            ),
        )
    # Worth printing even though it is not an A/B number: it says how thin the
    # world roboplan's Scene can see actually is.  A search benchmark over this
    # world is a benchmark of a few-percent tail, and reading the latency rows
    # without that in front of you overstates what either planner contributes to
    # a real transit.
    notes.append(
        f"{len(pairs)} queries kept out of {counts['examined']} sampled pairs: "
        f"{counts['both_free']} were straight-line free for both, "
        f"{counts['old_only']} blocked only for the deployed checker, "
        f"{counts['new_only']} blocked only for the Scene",
    )

    unshortcut = RRTConnectConfig(
        step_size=stack.rrt.step_size,
        collision_resolution=stack.rrt.collision_resolution,
        max_iterations=stack.rrt.max_iterations,
        goal_bias=stack.rrt.goal_bias,
        shortcut_attempts=0,
        seed=stack.rrt.seed,
    )

    def run(planner: Any) -> tuple[int, list[float], list[float]]:
        solved, latency, lengths = 0, [], []
        for start, goal in pairs:
            path, elapsed, error = _timed(
                lambda s=start, g=goal: planner.plan_joint(
                    s, g, timeout_s=_PLAN_TIMEOUT_S,
                ),
            )
            latency.append(elapsed)
            if error is None and path is not None:
                solved += 1
                lengths.append(_path_length(np.asarray(path.waypoints, dtype=float)))
        return solved, latency, lengths

    old_solved, old_latency, old_after = run(numpy_planner)
    _, _, old_before = run(
        JointSpaceRRTConnect(
            joint_names=chain.joint_names,
            lower_limits=chain.lower_limits,
            upper_limits=chain.upper_limits,
            state_valid=fixtures.checker.is_state_valid,
            config=unshortcut,
        ),
    )
    new_solved, new_latency, new_after = run(roboplan_planner)
    _, _, new_before = run(RoboplanJointPlanner(handle, config=unshortcut))

    count = len(pairs)
    return Comparison(
        "joint planning", old_label, new_label, tuple(notes),
        (
            Metric("success rate", old_solved / count, new_solved / count, "{:.1%}", "high"),
            Metric("latency mean (ms)",
                   1e3 * _stats(old_latency)["mean"], 1e3 * _stats(new_latency)["mean"],
                   "{:.1f}", "low"),
            Metric("latency p95 (ms)",
                   1e3 * _stats(old_latency)["p95"], 1e3 * _stats(new_latency)["p95"],
                   "{:.1f}", "low"),
            Metric("path length before shortcut (rad)",
                   _stats(old_before)["mean"], _stats(new_before)["mean"], "{:.3f}", "low"),
            Metric("path length after shortcut (rad)",
                   _stats(old_after)["mean"], _stats(new_after)["mean"], "{:.3f}", "low"),
        ),
    )


def _peak_fractions(
    trajectory: Any,
    velocity_cap: np.ndarray,
    acceleration_cap: np.ndarray,
) -> tuple[float, float]:
    """Finite-difference peaks as a fraction of the caps the config asks for.

    The same estimator for both retimers, so it under-reports both identically
    rather than favouring either.
    """

    positions = np.asarray(trajectory.positions, dtype=float)
    times = np.asarray(trajectory.times_s, dtype=float)
    dt = np.maximum(np.diff(times), 1e-12)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    if len(velocity) < 2:
        return float(np.max(np.abs(velocity) / velocity_cap)), float("nan")
    midpoint = np.maximum(0.5 * (dt[:-1] + dt[1:]), 1e-12)
    acceleration = np.diff(velocity, axis=0) / midpoint[:, None]
    return (
        float(np.max(np.abs(velocity) / velocity_cap)),
        float(np.max(np.abs(acceleration) / acceleration_cap)),
    )


def compare_timing(fixtures: Fixtures, paths: int, vertices: int) -> Comparison:
    stack, chain = fixtures.stack, fixtures.chain
    old_label = "retime_path (rest-to-rest quintic per segment)"
    new_label = "ToppraRetimer (roboplan TOPP-RA, linear-blend)"
    settings = stack.time_parameterization
    notes = [
        f"{paths} polylines of {vertices} collision-free vertices, densified at "
        f"rrt.collision_resolution = {stack.rrt.collision_resolution} rad",
        "both retimers receive the SAME array and the same per-joint caps "
        "(URDF velocity, robot.acceleration_limits)",
        f"caps scaled by time_parameterization: velocity x{settings.velocity_scale}, "
        f"acceleration x{settings.acceleration_scale}",
        "peaks are finite differences of the emitted samples, the same "
        "estimator for both",
    ]
    try:
        handle = fixtures.handle
        states = fixtures.collision_free_states(paths * vertices, salt=303)
        retimer = ToppraRetimer(handle)
    except Skipped as error:
        return Comparison("time parameterization", old_label, new_label, tuple(notes), skipped=str(error))
    except (RoboplanUnavailable, ValueError) as error:
        return Comparison(
            "time parameterization", old_label, new_label, tuple(notes),
            skipped=f"{type(error).__name__}: {error}",
        )

    resolution = stack.rrt.collision_resolution
    densified: list[np.ndarray] = []
    for index in range(paths):
        knots = states[index * vertices:(index + 1) * vertices]
        dense = [knots[0]]
        for first, second in zip(knots, knots[1:]):
            steps = max(1, int(np.ceil(float(np.linalg.norm(second - first)) / resolution)))
            for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
                dense.append(first + alpha * (second - first))
        densified.append(np.asarray(dense, dtype=float))
    # The quintic is rest-to-rest PER SEGMENT, so on a densified path it comes to
    # a full stop at every one of these waypoints and its duration scales with
    # the COUNT, not with the caps.  Printing the count is what keeps the
    # duration row from reading as a solver-quality result -- and this is the
    # deployed shape: staged_trajectory.py retimes the joint planner's densified
    # output directly.
    notes.append(
        f"mean input length {np.mean([len(path) for path in densified]):.0f} "
        "waypoints; the quintic stops at every one of them, which is why its "
        "peak velocity collapses while its acceleration still saturates",
    )

    velocity = np.asarray(chain.velocity_limits, dtype=float)
    acceleration = np.asarray(stack.robot.acceleration_limits, dtype=float)
    if not (
        np.all(np.isfinite(velocity)) and np.all(velocity > 0.0)
        and np.all(np.isfinite(acceleration)) and np.all(acceleration > 0.0)
    ):
        # Both retimers would reject every path for the same reason and the row
        # would read "0 vs 0" instead of naming the missing URDF limit.
        return Comparison(
            "time parameterization", old_label, new_label, tuple(notes),
            skipped=f"non-positive limits: velocity={velocity}, acceleration={acceleration}",
        )
    velocity_cap = velocity * settings.velocity_scale
    acceleration_cap = acceleration * settings.acceleration_scale

    def run(retime: Callable[[np.ndarray], Any]) -> tuple[list[float], list[float], list[float], list[str]]:
        durations, peak_v, peak_a, failures = [], [], [], []
        for path in densified:
            trajectory, _, error = _timed(lambda p=path: retime(p))
            if error is not None or trajectory is None:
                failures.append(f"{type(error).__name__}: {error}")
                continue
            durations.append(float(trajectory.times_s[-1]))
            fraction_v, fraction_a = _peak_fractions(
                trajectory, velocity_cap, acceleration_cap,
            )
            peak_v.append(fraction_v)
            peak_a.append(fraction_a)
        return durations, peak_v, peak_a, failures

    old_duration, old_v, old_a, old_failures = run(
        lambda path: retime_path(path, velocity, acceleration, settings),
    )
    new_duration, new_v, new_a, new_failures = run(
        lambda path: retimer.retime_path(path, velocity, acceleration, settings),
    )
    for label, failures in (("quintic", old_failures), ("toppra", new_failures)):
        if failures:
            notes.append(f"{label} rejected {len(failures)}/{paths}: {failures[0]}")

    return Comparison(
        "time parameterization", old_label, new_label, tuple(notes),
        (
            Metric("paths retimed", len(old_duration), len(new_duration), "{:.0f}", "high"),
            Metric("trajectory duration mean (s)",
                   _stats(old_duration)["mean"], _stats(new_duration)["mean"], "{:.3f}", "low"),
            Metric("peak velocity / cap",
                   _stats(old_v)["mean"], _stats(new_v)["mean"], "{:.3f}", "high"),
            Metric("peak acceleration / cap",
                   _stats(old_a)["mean"], _stats(new_a)["mean"], "{:.3f}", "high"),
            Metric("worst peak velocity / cap",
                   max(old_v, default=None), max(new_v, default=None), "{:.3f}", None),
            Metric("worst peak acceleration / cap",
                   max(old_a, default=None), max(new_a, default=None), "{:.3f}", None),
        ),
    )


def compare_collision(fixtures: Fixtures, states: int) -> Comparison:
    chain = fixtures.chain
    old_label = f"PointCloudCollisionChecker ({_SCENE_POINTS}-point cloud + capsules)"
    new_label = "roboplan Scene.hasCollisions (capsules only)"
    notes = [
        "NOT interchangeable: the checker also tests the perceived cloud and "
        f"adds clearance {fixtures.collision_model.scene_clearance_m} m; the "
        "Scene tests self-collision only, on inflated radii",
        "the Scene's lock is taken once for the whole loop, as a planner takes "
        "it; handle.expand is inside the timed path because a planner pays it",
        f"{_COLLISION_REPEATS} timed repeats over the same states, best kept, "
        "one untimed warmup",
        "the cloud is parked out of reach, so the checker's cKDTree queries "
        "return empty: a real near-field cloud costs it MORE than shown here, "
        "and nothing more on the Scene side",
    ]
    try:
        handle = fixtures.handle
    except Skipped as error:
        return Comparison("collision checking", old_label, new_label, tuple(notes), skipped=str(error))

    rng = fixtures.rng(404)
    samples = rng.uniform(
        chain.lower_limits, chain.upper_limits, size=(states, chain.dof),
    )
    for joints in samples[:20]:
        fixtures.checker.is_state_valid(joints)
    old_best = float("inf")
    for _ in range(_COLLISION_REPEATS):
        started = time.perf_counter()
        old_verdicts = [fixtures.checker.is_state_valid(joints) for joints in samples]
        old_best = min(old_best, time.perf_counter() - started)

    with handle.exclusive() as scene:
        for joints in samples[:20]:
            scene.hasCollisions(handle.expand(joints))
        new_best = float("inf")
        for _ in range(_COLLISION_REPEATS):
            started = time.perf_counter()
            new_verdicts = [
                not bool(scene.hasCollisions(handle.expand(joints)))
                for joints in samples
            ]
            new_best = min(new_best, time.perf_counter() - started)

    scene_stricter = sum(old and not new for old, new in zip(old_verdicts, new_verdicts))
    checker_stricter = sum(new and not old for old, new in zip(old_verdicts, new_verdicts))
    notes.append(
        "checker-stricter states are expected (clearance); scene-stricter "
        "states are expected (inflated radii).  A zero on BOTH would mean the "
        "two models had converged, which they have not",
    )

    # The one state whose verdict has an operational consequence, so it is
    # asserted rather than left to the aggregate counts: the commanded home pose
    # clears the enforced mount-vs-wrist pair by 3.8 mm, which is INSIDE the
    # deployment's 10 mm self-collision clearance but outside the Scene's
    # inflated radii.  The two backends therefore disagree about whether the
    # robot's own parked pose is a legal planning endpoint.
    home = np.asarray(
        json.loads(
            (REPOSITORY_ROOT / "configs/piper_home.json").read_text(encoding="utf-8"),
        )["joint_radians"],
        dtype=float,
    )
    home_result = fixtures.checker.check_state(home)
    if not home_result.valid:
        gap = "" if home_result.distance is None or home_result.threshold is None else (
            f" (gap {home_result.distance:.4f} m vs threshold "
            f"{home_result.threshold:.4f} m)"
        )
        notes.append(f"home rejected by the checker: {home_result.reason}{gap}")
    with handle.exclusive() as scene:
        home_scene_ok = not bool(scene.hasCollisions(handle.expand(home)))

    return Comparison(
        "collision checking", old_label, new_label, tuple(notes),
        (
            Metric("per state (us)",
                   1e6 * old_best / states, 1e6 * new_best / states, "{:.1f}", "low"),
            Metric("states called collision-free",
                   sum(old_verdicts), sum(new_verdicts), "{:.0f}", None),
            # Read down the column: under "old", the states only the checker
            # rejects; under "new", the states only the Scene rejects.
            Metric("states this side alone rejects",
                   checker_stricter, scene_stricter, "{:.0f}", None),
            Metric("accepts configs/piper_home.json (1 = yes)",
                   int(home_result.valid), int(home_scene_ok), "{:.0f}", None),
        ),
    )


# ---------------------------------------------------------------------- main


def _roboplan_version() -> str:
    import importlib.metadata  # noqa: PLC0415 -- only needed for the header

    try:
        return str(importlib.metadata.version("roboplan"))
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="write the versioned JSON document here")
    # Measured: the roboplan solver clears ~112 candidates inside the deployed
    # 0.6 s budget, so a smaller default makes the throughput row report the
    # sample size instead of a throughput.
    parser.add_argument("--ik-targets", type=int, default=160)
    parser.add_argument("--plan-queries", type=int, default=12)
    # Measured on this model: only ~4% of random collision-free pairs need a
    # search at all, so the pool has to be an order of magnitude bigger than the
    # number of queries or the row skips itself.
    parser.add_argument("--plan-pair-pool", type=int, default=400)
    parser.add_argument("--timing-paths", type=int, default=8)
    parser.add_argument("--timing-vertices", type=int, default=4)
    parser.add_argument("--collision-states", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(
        args.ik_targets, args.plan_queries, args.timing_paths, args.collision_states,
    ) < 1 or args.timing_vertices < 2:
        raise SystemExit("every sample count must be positive and paths need 2+ vertices")
    if args.plan_pair_pool < args.plan_queries:
        raise SystemExit("--plan-pair-pool must be at least --plan-queries")
    fixtures = build_fixtures(args.config, args.seed, sanity_states=256)

    ik, throughput = compare_ik(fixtures, args.ik_targets)
    comparisons = [
        ik,
        throughput,
        compare_planning(fixtures, args.plan_queries, args.plan_pair_pool),
        compare_timing(fixtures, args.timing_paths, args.timing_vertices),
        compare_collision(fixtures, args.collision_states),
    ]
    header = {
        "urdf": str(fixtures.stack.robot.urdf_path),
        "config": str(args.config),
        "seed": args.seed,
        "roboplan_version": _roboplan_version(),
    }
    print(render(comparisons, header), end="")
    if args.output:
        document = {
            "schema": SCHEMA,
            "offline": True,
            "transport_opened": False,
            "motion_commands_sent": 0,
            **header,
            "sample_counts": {
                "ik_targets": args.ik_targets,
                "plan_queries": args.plan_queries,
                "plan_pair_pool": args.plan_pair_pool,
                "timing_paths": args.timing_paths,
                "timing_vertices": args.timing_vertices,
                "collision_states": args.collision_states,
            },
            "comparisons": [comparison.to_json() for comparison in comparisons],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
