"""Batch grasp feasibility decided by solving, not by ranking a proxy.

WHAT THIS REPLACES
``grasp_pipeline.py:585-615`` orders candidates by a seed-pose FK distance
(``robust_ik.make_seed_pose_ranker``).  That ranker carries no grasp quality:
its orientation term is measured against a fixed set of canned seed rotations,
so sorting by its magnitude means "prefer whichever grasp points most like a
stock seed", and with ``max_feasible_plans=1`` that ordering IS the grasp
selection.  Measured over 183 recorded plans: a lateral grasp was rank 1 in
91-100% of sessions but was executed in 9% of 07-28 sessions; 102 better-ranked
candidates were never evaluated at all; the executed pose sat in the 21st-31st
percentile of its own pool's required reach every single day.

The proxy existed because a real IK solve was too expensive to run on every
candidate.  It is not any more, but the honest margin is thinner than the mean
solve time suggests, because the cost is wildly asymmetric: an ACCEPTED pose
costs 0.95 ms and a REJECTED one costs 25-38 ms, since saying no means
exhausting the whole start ladder.  Measured in-container on the real model
against ``configs/go2w_piper.json``, which caps the search at ``max_hypotheses``
= 64 and budgets 0.6 s for ONE ``ik.solve_timeout_s``:

    all 64 reachable        0.10 s, complete            (3/3 runs)
    64 at 30% infeasible    0.60 s, 56-57 of 64 resolved (5/5 runs)

So: a pool of 64 is resolved *at* the deployed budget, not comfortably inside
it, and whether it completes depends on how many of the 64 are infeasible.  An
earlier version of this docstring quoted only the first row and claimed "6x the
budget to spare"; that number was measured on an all-reachable prefix and is not
what a mixed pool does.  The claim that survives measurement is the one the
argument actually needs: the proxy ORDERS 64 candidates by a stand-in and
``max_feasible_plans`` = 1 then evaluates one of them, while this filter answers
~56-64 of them for real in the same budget and marks any remainder ``deadline``
rather than dropping it.  Full-corpus rate on that mix is 119 candidates/s.

THE GATE IS ``RoboplanIKSolver``, NOT A SECOND SOLVER
Feasibility is whatever ``roboplan_ik.RoboplanIKSolver.solve`` says it is.  That
matters more than it looks: SimpleIk stops on a tip-ORIGIN isotropic norm, while
this repo accepts on ``control_point_delta`` at the tool contact point with a
free-axis orientation split, and the two disagree by centimetres of fingertip
(2026-07-24, 27 mm off a verified grasp).  A filter that gated on SimpleIk's own
norm would hand the pipeline candidates the pipeline then rejects -- a feasible
list that is not feasible, which is worse than the proxy it replaced.

Only the two REASON-DIAGNOSING probes call ``_solve_ik`` directly, and they
never produce an ``IKSolution``; they answer "was the pose reachable at all"
with a bool.  There is exactly one IK adapter in this package.

WHAT IT ORDERS BY -- AND WHAT IT REFUSES TO ORDER BY
Feasibility, then the caller's own order.  Nothing else.

Solved candidates come first, resolved failures next, never-evaluated candidates
last; WITHIN each class the caller's input order is preserved bit for bit,
because that order carries the perception score and the lateral-approach prior
and this module knows nothing about either.

Sorting the solved candidates by ``position_error_m`` would be the same mistake
in a new costume: position error is bounded above by the acceptance tolerance
for every solution that passed the gate, so its variation below the gate is
solver noise, not grasp quality.

``manipulability`` and ``min_joint_limit_margin`` are a different case -- they
are real physical quantities, and there is a plausible story that both predict
whether the Cartesian approach and lift that follow a pregrasp will survive
(``grasp_pipeline._cartesian_ik`` aborts on a joint step exceeding
``max_cartesian_joint_step_rad``, which is exactly what a near-singular pregrasp
produces).  Plausible is not measured.  Both are returned on every solved
verdict so a caller holding evidence can sort by them; this module does not,
because swapping one unvalidated scalar for another unvalidated scalar is how
the comment quoted above came to be written.  The experiment that would change
this: replay the recorded corpus, correlate pregrasp manipulability against
approach-stage IK failure, and if it separates, sort here and say so.

NAMED REASONS
Every input index appears in the output exactly once, carrying one of:

``solved``      a collision-free configuration reaching the target.
``collision``   the pose IS reachable -- the same ladder solves it with
                collisions off -- but no collision-free way there was found.
                The grasp is geometrically fine; the arm folds into itself to
                make it.  Note the honest verb: found, not "exists".
``joint-limit`` the target POSITION is reachable but the full 6-DoF pose is not,
                collisions ignored.  The wrist ran out of range there: re-orient
                the base rather than moving it closer.
``unreachable`` not even the position is reachable, collisions ignored.  Outside
                the workspace: move the base.
``deadline``    never evaluated.  UNKNOWN, *not* infeasible.

Only the gate decides; the two probes that separate ``collision`` from
``joint-limit`` from ``unreachable`` never return joints and cannot flip a
verdict.  They run the gate's own start ladder -- literally
``solver._restarts``, not a second copy of the formula that produced it -- with
collisions off.  An under-powered probe cannot make a candidate infeasible, but
it can tell an operator to move the base when the arm could reach the pose all
along, and measured, a shortened ladder did exactly that to 18 of 211 targets.

ANYTIME AND DEADLINE-HONEST
``len(report.verdicts) == len(targets)`` always; whatever the budget did not
reach carries ``deadline`` and sets ``report.partial``.  ``deadline`` sorts last
precisely because it is UNKNOWN -- the correct response to ``partial`` is a
larger budget, not a gamble on an unevaluated candidate, which is the behaviour
being removed.  Cancellation (``PlanningCancelled``) propagates instead: it
means the request is void, not short of time.

A candidate the batch deadline cut THROUGH is ``deadline`` too, and getting that
wrong is subtle enough to have shipped once: the gate raises
``PlanningDeadlineExceeded`` for its own per-candidate pathology slice and for
the caller's blown batch budget alike, and swallowing both into "no solution"
sent the half-evaluated candidate on to the reason probes, which happily named
it ``collision``.  Measured on the real model, 12 partial batches out of 12
turned one genuinely SOLVED grasp into an infeasible verdict that way -- a good
grasp deleted from the feasible list, which is the failure this module exists to
remove, wearing the module's own badge.  So every deadline is re-checked against
the CALLER's control before a reason is invented, and the probes checkpoint
between starts: past the deadline the answer is "unknown", never a reason.

Each candidate gets its own ``limited_to`` slice, so one pathological pose
cannot eat the batch, and overshoot past the deadline is bounded by whichever
single solver call was in flight -- at most ``_SOLVE_TIMEOUT_S``, or one
``_DIAGNOSTIC_TIMEOUT_S`` probe start.

THE SHARED-SCENE HAZARD
``SimpleIk`` builds its start configuration through
``Scene::toFullJointPositions``, which fills every NON-arm coordinate -- all four
wheels, all twelve leg joints, the chassis -- from the Scene's *current* joint
positions, whatever the previous caller left there, not from anything passed in.
``RoboplanIKSolver._ladder`` pins them to ``handle.seed`` on every call, so the
gate carries its own guarantee; this module pins them once more for the batch
because the two reason PROBES call ``_solve_ik`` directly, and a candidate whose
gate call raised before reaching the ladder arrives at them with that pin never
having run.  The mutation deliberately outlives the batch: ``handle.seed`` is the
canonical safe pose, and handing the next adapter back whatever the previous one
left is the failure mode, not the fix.  It all happens inside
``handle.exclusive()``, so no peer observes a half-written state.

roboplan is imported lazily, inside ``roboplan_ik._solve_ik``, so this file
loads on a host that has neither roboplan nor pinocchio and host tests drive
every path through fakes of that one seam and of the solver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from z_manip.kinematics.robust_ik import IKFailure, IKSolution
from z_manip.planning_control import (
    PlanningControl,
    PlanningDeadlineExceeded,
    checkpoint,
)

from .roboplan_ik import RoboplanIKSolver, _solve_ik
from .staged_trajectory import _readonly_pose, _readonly_vector

__all__ = [
    "COLLISION",
    "DEADLINE",
    "JOINT_LIMIT",
    "REACHABILITY_REASONS",
    "SOLVED",
    "UNREACHABLE",
    "ReachabilityReport",
    "ReachabilityVerdict",
    "filter_reachable",
]

# Exported, not left as bare literals at the call sites: ``reason == "solved"``
# against a typo is a comparison that silently never matches, on the branch that
# decides whether a grasp is executed.
SOLVED = "solved"
COLLISION = "collision"
JOINT_LIMIT = "joint-limit"
UNREACHABLE = "unreachable"
DEADLINE = "deadline"

#: Every reason this module can emit.  Exported because the offline tests replay
#: reason strings out of recorded status documents and must be able to reject a
#: string this module never produces.
REACHABILITY_REASONS = (SOLVED, COLLISION, JOINT_LIMIT, UNREACHABLE, DEADLINE)

# Feasible first, then anything resolved as infeasible, then the unknowns.  The
# three infeasible reasons share a rank on purpose: none of them is "closer to
# feasible" than the others, and implying an order between them would be a new
# proxy.  See the module docstring.
_REASON_RANK = {
    SOLVED: 0,
    COLLISION: 1,
    JOINT_LIMIT: 1,
    UNREACHABLE: 1,
    DEADLINE: 2,
}

_OPERATION = "roboplan reachability filtering"

# Per-candidate slice of the batch budget.  A PATHOLOGY BOUND, deliberately set
# so that it never fires in normal operation -- not a knob for trimming the
# gate's ladder.  Measured over a 214-target mixed corpus: a solve costs 0.95 ms,
# a REJECTION costs 25-38 ms (p95 24.6, max 39.7) because RoboplanIKSolver walks
# all 1+ik_restarts Halton starts before it will say no.  60 ms is 1.5x that
# worst case.
#
# Trimming looked tempting and is a trap: at a 6 ms slice the batch runs 90%
# faster (236/s against 124/s) but the cap starts landing INSIDE the ladder, and
# because it is wall clock the number of starts a candidate gets then depends on
# machine load -- measured, three identical batches returned three different
# feasible sets, and this list's head is the executed grasp.  Above the ladder's
# own cost the gate is a pure deterministic descent from fixed starts and the
# answer is reproducible.  ``config.ik_timeout_s`` (0.6 s) is not used here: it
# is a whole-request budget, and spending it on one candidate is what the proxy
# already did.
_SOLVE_TIMEOUT_S = 0.060
# Per-start cap for the reason probes.  Also a pathology bound and also
# non-binding: measured, tripling it to 6 ms changed not one verdict or reason
# on a 214-target corpus, because a collision-free probe either descends onto
# the goal or diverges well inside 2 ms.
_DIAGNOSTIC_TIMEOUT_S = 0.002
# ``max_angular_error_norm`` for the position-only probe.  Larger than pi, so
# the orientation term can never gate convergence and the probe answers exactly
# "can the arm put its tip on this POINT at all".
_ANY_ORIENTATION_RAD = 1.0e3


@dataclass(frozen=True)
class ReachabilityVerdict:
    """One candidate's resolved feasibility.

    ``solution`` carries the joints and the real IK metrics and is present for
    exactly the ``solved`` verdicts -- an absent solution on a ``solved`` verdict
    (or a present one on a failure) is a caller reading a field that does not
    mean what it looks like, so it is rejected here rather than downstream.
    """

    index: int
    reason: str
    solution: IKSolution | None = None

    def __post_init__(self) -> None:
        if int(self.index) != self.index or self.index < 0:
            raise ValueError("reachability verdict index must be a non-negative integer")
        object.__setattr__(self, "index", int(self.index))
        if self.reason not in _REASON_RANK:
            raise ValueError(
                f"unknown reachability reason {self.reason!r}; "
                f"expected one of {REACHABILITY_REASONS}",
            )
        if (self.solution is None) == (self.reason == SOLVED):
            raise ValueError(
                f"reason {self.reason!r} and solution presence disagree",
            )


@dataclass(frozen=True)
class ReachabilityReport:
    """Every input candidate, ordered feasible-first.  Never truncated."""

    verdicts: tuple[ReachabilityVerdict, ...]
    elapsed_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdicts", tuple(self.verdicts))
        if not np.isfinite(self.elapsed_s) or self.elapsed_s < 0.0:
            raise ValueError("reachability elapsed time must be finite and non-negative")

    @property
    def evaluated(self) -> int:
        """How many candidates actually got a solve.  The throughput number."""

        return sum(1 for verdict in self.verdicts if verdict.reason != DEADLINE)

    @property
    def partial(self) -> bool:
        """True when the budget ran out with candidates still unevaluated."""

        return self.evaluated < len(self.verdicts)

    @property
    def feasible(self) -> tuple[ReachabilityVerdict, ...]:
        """The solved candidates, in order.  The point of the module."""

        return tuple(v for v in self.verdicts if v.reason == SOLVED)


def _targets(value: object) -> tuple[np.ndarray, ...]:
    """Validate every target before any solving starts.

    Up front, not lazily: a malformed pose is a caller bug, and discovering it
    250 candidates into a budget would burn the budget AND report a partial
    result that looks like a deadline.
    """

    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError(f"reachability targets must be a sequence: {error}") from error
    return tuple(
        _readonly_pose(item, f"reachability target {index}")
        for index, item in enumerate(items)
    )


def _reachable(
    scene: object,
    solver: RoboplanIKSolver,
    joint_names: Sequence[str],
    target: np.ndarray,
    starts: Sequence[np.ndarray],
    *,
    angular_tolerance_rad: float,
    control: PlanningControl | None,
) -> bool:
    """Ignoring collisions, can the arm reach this goal from any of ``starts``?

    Never returns joints.  This answers a labelling question, so the acceptance
    gate that ``RoboplanIKSolver`` applies to real solutions is deliberately not
    applied: SimpleIk's own stopping norm is the right, cheap instrument for
    "was the pose within reach at all".

    That norm is the ``RoboplanConfig`` tolerance, which is the tighter of the
    two -- the gate ACCEPTS at the looser ``IKConfig`` one -- so in principle a
    point the gate would have taken can be called out of reach here.  Measured
    over 211 targets that are all reachable by construction, it never happened,
    which is why the looser value is not plumbed through for it.

    ``PlanningDeadlineExceeded`` from the checkpoint propagates: past the batch
    deadline this candidate has no reason, it has ``deadline``.
    """

    options = {
        "group_name": solver.handle.group,
        "max_iters": solver.config.ik_max_iterations,
        # Zero for the same reason roboplan_ik pins it to zero: with restarts on,
        # SimpleIk draws them from the Scene-global RNG and the reason attached
        # to a candidate stops being reproducible.
        "max_restarts": 0,
        "max_linear_error_norm": solver.config.ik_position_tolerance_m,
        "max_angular_error_norm": angular_tolerance_rad,
        "check_collisions": False,
        "max_time": _DIAGNOSTIC_TIMEOUT_S,
    }
    for start in starts:
        checkpoint(control, _OPERATION)
        solved, _ = _solve_ik(
            scene,
            joint_names,
            (solver.base_link, solver.tip_link),
            target,
            start,
            options,
        )
        if solved:
            return True
    return False


def _evaluate(
    index: int,
    target: np.ndarray,
    *,
    scene: object,
    solver: RoboplanIKSolver,
    joint_names: Sequence[str],
    seed: np.ndarray,
    starts: Sequence[np.ndarray],
    control: PlanningControl | None,
) -> ReachabilityVerdict:
    """Resolve one candidate, or raise ``PlanningDeadlineExceeded`` unresolved.

    Raising is the point: a candidate the batch deadline cut through has no
    answer, and the caller turns that into ``deadline``.
    """

    handle = solver.handle
    try:
        # Inside the try: ``limited_to`` checkpoints its parent first, so the
        # batch deadline can land here too, between the loop's checkpoint and
        # this line.
        budget = (control or PlanningControl()).limited_to(_SOLVE_TIMEOUT_S, _OPERATION)
        solution = solver.solve(target, seed, control=budget)
    except IKFailure:
        solution = None
    except PlanningDeadlineExceeded:
        # The gate cannot tell these apart -- it raises the same exception for
        # its own per-candidate pathology slice and for the caller's blown batch
        # budget -- so ask the caller's control which one it was.  Only the slice
        # is an answer ABOUT the candidate; a blown batch budget re-raises here
        # and the candidate stays unknown.  Swallowing both is what let a solved
        # grasp be reported ``collision`` in 12 of 12 partial batches.
        checkpoint(control, _OPERATION)
        solution = None

    if solution is not None:
        # Redundant today -- the gate collision-checks, its restarts are pinned
        # off, and this batch pinned the body pose -- and kept anyway, at 12 us
        # against a 0.95 ms accepted solve, because "solved means collision-free
        # on the body the robot is actually in" is this module's own contract
        # and should not be inherited from another module's option dict.
        if not scene.hasCollisions(handle.expand(solution.joints)):
            return ReachabilityVerdict(index, SOLVED, solution)
        return ReachabilityVerdict(index, COLLISION)

    if _reachable(
        scene, solver, joint_names, target, starts,
        angular_tolerance_rad=solver.config.ik_orientation_tolerance_rad,
        control=control,
    ):
        return ReachabilityVerdict(index, COLLISION)
    if _reachable(
        scene, solver, joint_names, target, starts,
        angular_tolerance_rad=_ANY_ORIENTATION_RAD,
        control=control,
    ):
        return ReachabilityVerdict(index, JOINT_LIMIT)
    return ReachabilityVerdict(index, UNREACHABLE)


def filter_reachable(
    solver: RoboplanIKSolver,
    targets: object,
    *,
    current: object | None = None,
    control: PlanningControl | None = None,
) -> ReachabilityReport:
    """Resolve every candidate tip pose to a named feasibility verdict.

    ``targets`` is a sequence of ``(4, 4)`` tip-frame goals expressed in the
    solver's ``base_link``, i.e. what ``grasp_pipeline._tip_pose`` already
    produces.  ``current`` seeds every solve and defaults to the handle's home
    arm pose.
    """

    started = time.perf_counter()
    handle = solver.handle
    poses = _targets(targets)
    seed = (
        handle.extract(handle.seed)
        if current is None
        else np.array(_readonly_vector(current, "current joints", handle.dof), dtype=float)
    )
    if not poses:
        return ReachabilityReport((), time.perf_counter() - started)

    verdicts: list[ReachabilityVerdict] = []
    # One acquisition for the whole batch.  ``solver.solve`` takes it again per
    # candidate; the handle's lock is reentrant and documented so.
    with handle.exclusive() as scene:
        joint_names = list(scene.getJointGroupInfo(handle.group).joint_names)
        # Module docstring, "the shared-Scene hazard this batch owns".
        scene.setJointPositions(handle.seed)
        # The gate's own ladder, taken from the gate rather than rebuilt from the
        # same formula: "the same starts that failed with collisions on succeeded
        # with them off" is what ``collision`` means, and a second copy of the
        # formula makes that true only until someone retunes one of them.
        # Measured, a SHORTER probe ladder is not a cheap approximation but a
        # wrong answer -- at 3 starts, 18 of 211 pose-reachable targets came back
        # ``unreachable``, an operator instruction to move the base; at the
        # gate's own 13, none, for 5% of the batch.
        starts = (seed, *solver._restarts)
        for index, target in enumerate(poses):
            try:
                checkpoint(control, _OPERATION)
                verdicts.append(_evaluate(
                    index,
                    target,
                    scene=scene,
                    solver=solver,
                    joint_names=joint_names,
                    seed=seed,
                    starts=starts,
                    control=control,
                ))
            except PlanningDeadlineExceeded:
                # Before or during this candidate, it makes no difference: it is
                # not appended, so it falls into the ``deadline`` tail below.
                break

    evaluated = len(verdicts)
    verdicts.extend(
        ReachabilityVerdict(index, DEADLINE) for index in range(evaluated, len(poses))
    )
    # Stable sort on the reason rank alone: within a rank the caller's input
    # order survives untouched, which is the whole ordering contract.
    verdicts.sort(key=lambda verdict: _REASON_RANK[verdict.reason])
    return ReachabilityReport(tuple(verdicts), time.perf_counter() - started)
