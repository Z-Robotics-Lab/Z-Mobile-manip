"""Which subsystems roboplan owns, and the check its ``Scene`` cannot make.

Two things every roboplan adapter deliberately left to a wiring layer.

**Selection.**  Each subsystem is chosen by its own env var, and when that is
unset by the optional ``roboplan`` section of the stack config.  Everything is
OFF unless a deployment says so twice: the section has to set ``enabled`` AND
the subsystem has to be selected.  A selector without ``enabled`` is rejected at
config load rather than at the first plan, because the alternative is a bringup
that dies halfway through with ``RoboplanUnavailable('disabled')``.

**The perceived scene.**  This is the load-bearing part.  ``RoboplanIKSolver``
and ``ToppraRetimer`` only ever *add* a check, so selecting them can reject a
motion the numpy path allowed but never allow one it rejected.
``RoboplanJointPlanner`` is not like that: it searches against the capsule
``Scene``, which contains the robot and nothing else.  It has never seen the
point cloud, so swapping it in unguarded would silently delete every perceived
obstacle from transit planning -- the arm would plan a clean straight line
through a box on the table and report success.  :class:`SceneVerifiedJointPlanner`
is what stands between the flag and that: roboplan searches, and the deployed
``JointSpaceRRTConnect`` predicate still has the last word on every waypoint it
returns.

Default OFF.  ``RoboplanBackends()`` selects nothing, ``StackConfig.roboplan``
is ``None`` for a config that omits the section, and both resolve to exactly
today's numpy backends.

roboplan is never imported here, directly or transitively: this module loads on
a host with neither roboplan nor pinocchio.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning_control import PlanningControl

from .roboplan_runtime import RoboplanConfig

__all__ = [
    "RoboplanBackends",
    "SceneVerifiedJointPlanner",
    "env_backend",
]

#: Section keys that pick a subsystem rather than tune a solver.  Kept apart
#: from ``RoboplanConfig``'s own fields so ``from_mapping`` can keep rejecting
#: unknown knobs.
_SELECTORS = ("ik", "joint_planner", "timing")


@dataclass(frozen=True)
class RoboplanBackends:
    """Which subsystems this deployment lets roboplan own.  All default OFF.

    ``runtime`` is the solver configuration every adapter shares -- one per
    deployment, so they share one cached ``Scene`` instead of paying 618 ms
    each.
    """

    runtime: RoboplanConfig = field(default_factory=RoboplanConfig)
    ik: bool = False
    joint_planner: bool = False
    timing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RoboplanConfig):
            raise ValueError("roboplan runtime must be a RoboplanConfig")
        for name in _SELECTORS:
            value = getattr(self, name)
            # Not coerced, for the reason RoboplanConfig.enabled is not:
            # bool("false") is True, and this flag swaps a planner on a robot.
            if not isinstance(value, bool):
                raise ValueError(f"roboplan {name} selector must be a boolean")
            if value and not self.runtime.enabled:
                raise ValueError(
                    f"roboplan {name} is selected but roboplan enabled is false",
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "RoboplanBackends":
        """Parse the optional stack-config ``roboplan`` section."""

        if not isinstance(values, Mapping):
            raise ValueError("roboplan section must be an object")
        selectors = {name: values[name] for name in _SELECTORS if name in values}
        runtime = {
            key: value for key, value in values.items() if key not in selectors
        }
        return cls(runtime=RoboplanConfig.from_mapping(runtime), **selectors)


def env_backend(
    variable: str,
    fallback: str,
    selected: bool,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve one backend switch: the env var wins, else the config selector.

    Same shape as the ``Z_MANIP_IK_BACKEND`` dispatch this repo already ships,
    including the ``strip().lower()``.  An unrecognised name is a startup error,
    never a silent fall back to ``fallback``: a typo'd backend that keeps
    planning is a deployment that believes it is running code it is not.
    """

    values = os.environ if environ is None else environ
    chosen = str(
        values.get(variable, "roboplan" if selected else fallback),
    ).strip().lower()
    if chosen not in (fallback, "roboplan"):
        raise ValueError(f"unsupported {variable}: {chosen!r}")
    return chosen


class SceneVerifiedJointPlanner:
    """roboplan searches; the deployed collision predicate still decides.

    ``search`` plans against the capsule ``Scene`` alone -- self-collision and
    nothing else -- so every waypoint it returns is re-checked with the same
    ``JointSpaceRRTConnect`` instance the numpy backend would have used, which
    carries the perceived point cloud, the attached payload and the fixed Go2-W
    fixtures.  ``segment_valid`` goes straight to that instance: its capsule
    self-collision set is already a superset of the Scene's, so asking roboplan
    first would only cost a lock.

    The verification is not a formality.  ``plan_joint``'s contract is that
    adjacent waypoints are no further apart than ``collision_resolution``, and
    ``segment_valid`` interpolates at exactly that resolution, so checking each
    consecutive pair reproduces the numpy planner's own guarantee rather than
    sampling a weaker one.
    """

    def __init__(self, search: object, verifier: object) -> None:
        self._search = search
        self._verifier = verifier
        # Both planners name their joints from independent sources -- the SRDF
        # group for roboplan, the URDF chain for the numpy planner.  A different
        # order would mean the verified path and the commanded path are indexed
        # differently, which no downstream check would catch.
        search_names = tuple(getattr(search, "joint_names", ()))
        verifier_names = tuple(getattr(verifier, "joint_names", ()))
        if search_names != verifier_names:
            raise ValueError(
                "roboplan and numpy joint planners disagree on joint order: "
                f"{search_names} != {verifier_names}",
            )

    def plan_joint(
        self,
        start_joints: object,
        goal_joints: object,
        *,
        timeout_s: float = 5.0,
        control: PlanningControl | None = None,
    ) -> JointTrajectory:
        """Plan with roboplan, then prove the result against the real scene."""

        path = self._search.plan_joint(
            start_joints,
            goal_joints,
            timeout_s=timeout_s,
            control=control,
        )
        waypoints = np.asarray(path.waypoints, dtype=float)
        if waypoints.ndim != 2 or len(waypoints) < 1:
            raise PlanningError("roboplan returned an empty joint path")
        # A single-waypoint path is still checked, as a degenerate segment: a
        # start-equals-goal plan is exactly the one a caller would execute
        # without looking.
        pairs = (
            tuple(zip(waypoints, waypoints[1:]))
            if len(waypoints) > 1
            else ((waypoints[0], waypoints[0]),)
        )
        for index, (first, second) in enumerate(pairs):
            if not self._verifier.segment_valid(first, second, control=control):
                # ponytail: fail closed, no fallback to the numpy planner.  The
                # ceiling is that an obstacle-rich scene makes the roboplan
                # backend fail every candidate instead of quietly costing 1.2 s
                # per plan; that is the direction an opt-in backend should fail,
                # and it is visible.  Upgrade path when the throughput matters:
                # push the point cloud into the Scene as an OcTree (measured
                # 1.3 ms for 4000 points) so the search sees it in the first
                # place, which needs a new seam in roboplan_planner.
                raise PlanningError(
                    "roboplan joint path crosses geometry its Scene cannot see: "
                    f"segment {index} of {len(pairs)} is invalid for the "
                    "deployed collision checker",
                )
        return path

    def segment_valid(
        self,
        first: object,
        second: object,
        *,
        control: PlanningControl | None = None,
    ) -> bool:
        """Validate one segment with the deployed checker, not with the Scene."""

        return bool(self._verifier.segment_valid(first, second, control=control))
