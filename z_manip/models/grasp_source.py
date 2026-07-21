"""GraspSource contract — pluggable 6-DoF grasp-generation backend (L1).

Adopted verbatim in shape from the reference stack's proven contract
(``refs/vector_manipulation_stack/.../grasp/base.py``: ``GraspContext`` /
``GraspCandidates`` / ``GraspGenerator.generate``). A backend takes ONE object
observation and returns scored 6-DoF candidates that the arm can reach and that
clear the scene. Backends share nothing but this contract:

    HGGD (A budget, sim) · AnyGrasp (dense) · geometric antipodal (always-on,
    zero-dependency CPU fallback + A/B anchor) · GT-heuristic (bring-up only).

Backend selection + graceful degradation is a cascade: try each in order, keep
the first that yields a reachable, collision-free grasp; the geometric antipodal
backend is the deterministic last resort. See :func:`select_grasp_source` (the
skeleton mirror of the reference ``registry.build_generator`` +
``GRASP_CASCADE``).

Everything here is ROS-message-light on purpose: the L2 ``grasp_exec`` primitive
owns the ROS boundary (parsing clouds, resolving TF, publishing feedback) and
hands the backend validated numpy, so backends stay unit-testable without a
running node.

M0 skeleton: contracts only, no inference. Concrete backends land in M2
(``docs/plan.md`` §5; HGGD gated on a CUDA-12.8/sm_120 recompile feasibility
test, §4b).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

# NOTE (M0): the two carriers below are declared with ``object`` field types and
# free-standing docstrings rather than importing ``numpy`` / ``vision_msgs``.
# The skeleton must ``py_compile`` with zero third-party deps (see pyproject).
# When M2 implements a backend, retype these to ``np.ndarray`` /
# ``vision_msgs.msg.Detection3D`` (additive: same field names, tighter types).


class GraspGenerationError(RuntimeError):
    """A backend could not produce a usable grasp for this goal.

    Recoverable at the primitive level: ``grasp_exec`` aborts the current
    attempt with this message and lets the skill layer's retry budget
    (``docs/plan.md`` §3a) decide re-try vs escalate. Raised for empty model
    output, candidates that are all IK-infeasible or all scene-colliding, or an
    inference-server failure.
    """


# eq=False on the carriers below: they will hold numpy arrays once implemented,
# and a generated ``__eq__`` would compare arrays with ``==`` (an array, not a
# bool) and raise. Identity equality is all a per-goal carrier needs.


@dataclass(frozen=True, eq=False)
class GraspContext:
    """Per-goal observation and resolved inputs handed to a backend.

    Built once per grasp goal by the L2 ``grasp_exec`` primitive after it has
    validated and parsed every ROS message. A backend reads only the fields its
    path needs (a point-cloud backend uses ``object_points``; a cuboid/GT-
    heuristic backend uses ``bbox``).

    Attributes:
        object_points: ``(N, 3)`` float32 object cloud in ``source_frame``, or
            ``None`` on a bbox-only (cuboid / GT-heuristic) path.
        bbox: Object oriented bounding box (``vision_msgs`` Detection3D once
            typed); ``None`` on the point-cloud path.
        source_frame: TF frame the backend's raw grasps come out in. The
            primitive resolves observations into the arm base frame up front, so
            in practice this is that frame and ``t_target_src`` is the identity.
        t_target_src: ``(4, 4)`` SE(3) mapping ``source_frame`` into the arm base
            frame, used by the IK filter (identity once everything is in the arm
            base frame).
        scene_points: ``(S, 3)`` raw scene cloud in ``source_frame`` (the object
            is NOT excluded — each backend removes its own object before
            collision checking), or ``None`` when no scene is needed.
        progress_cb: ``(phase, progress)`` sink for action feedback.
    """

    object_points: Optional[object]
    bbox: Optional[object]
    source_frame: str
    t_target_src: object
    scene_points: Optional[object]
    progress_cb: Callable[[str, float], None]
    affordance: Optional[object] = None


@dataclass(frozen=True, eq=False)
class GraspCandidates:
    """A backend's output: filtered, scored grasps in ``frame``.

    Grasps here have already had IK-infeasible and scene-colliding poses
    dropped — the primitive's shared pipeline only transforms, ranks, and
    visualizes what survives. ``num_raw`` carries the pre-filter count through
    for feedback/metrics.

    Attributes:
        grasps: ``(M, 4, 4)`` SE(3) grasp poses in ``frame``, source-gripper
            (Franka TCP) convention.
        scores: ``(M,)`` per-grasp model confidence in ``[0, 1]``.
        centroid: ``(3,)`` object centroid in ``frame`` (cloud mean or OBB
            centre) — the proximity term for downstream ranking.
        frame: TF frame ``grasps`` and ``centroid`` are expressed in.
        num_raw: Grasp count before IK / collision filtering.
        widths: ``(M,)`` per-grasp required gripper opening (meters), aligned
            with ``grasps``, or ``None`` when the backend does not predict a
            width (the aperture filter is then a no-op).
    """

    grasps: object
    scores: object
    centroid: object
    frame: str
    num_raw: int
    widths: Optional[object] = None


@runtime_checkable
class GraspSource(Protocol):
    """Interface every grasp-generation backend implements.

    Concrete backends (HGGD, AnyGrasp, geometric antipodal, GT-heuristic) differ
    entirely in how they produce candidates; they agree only on
    :meth:`generate`. Structural typing (``Protocol``) keeps L2/L3 callers
    zero-coupled to any concrete backend.
    """

    def generate(self, context: GraspContext) -> GraspCandidates:
        """Produce IK- and scene-collision-filtered grasps for one goal.

        Args:
            context: The per-goal observation and resolved inputs.

        Returns:
            Filtered, scored candidates in ``context.source_frame``.

        Raises:
            GraspGenerationError: if no usable grasp survives — empty model
                output, every candidate unreachable or scene-colliding, or an
                inference failure.
        """
        ...


class GraspSourceBase(abc.ABC):
    """Optional ABC base for backends that want shared plumbing (M2+).

    Backends may either satisfy the :class:`GraspSource` Protocol structurally
    or subclass this. Kept minimal here; shared filtering/scoring helpers land
    alongside the first real backend.
    """

    @abc.abstractmethod
    def generate(self, context: GraspContext) -> GraspCandidates:
        """See :meth:`GraspSource.generate`."""
        raise NotImplementedError


# geometry_type keys — part of the grasp-request contract; fixed strings, not
# free to rename. They name a backend. Cascade order mirrors the reference
# ``GRASP_CASCADE``: learned backends first, geometric antipodal last as the
# deterministic fallback.
GEOMETRY_ANTIPODAL = "antipodal"   # zero-dep CPU geometric baseline / fallback
GEOMETRY_HGGD = "hggd"             # A-budget learned (sim, 5080)
GEOMETRY_DENSE = "dense"           # AnyGrasp dense backend
GEOMETRY_GT = "gt_heuristic"       # bring-up-only ground-truth heuristic
ALL_GEOMETRIES: tuple[str, ...] = (
    GEOMETRY_HGGD,
    GEOMETRY_DENSE,
    GEOMETRY_ANTIPODAL,
    GEOMETRY_GT,
)
GRASP_CASCADE: tuple[str, ...] = (GEOMETRY_HGGD, GEOMETRY_DENSE, GEOMETRY_ANTIPODAL)


def select_grasp_source(geometry_type: str) -> GraspSource:
    """Construct the configured grasp backend.

    Mirrors the reference ``registry.build_generator``: adding a backend is a
    new ``models/`` module plus one branch here; nothing else changes. A
    dispatch function, not a plugin framework — the backend set is small and
    fixed.

    Args:
        geometry_type: One of :data:`ALL_GEOMETRIES` (already normalized).

    Raises:
        GraspGenerationError: if ``geometry_type`` is unknown.
        NotImplementedError: when a learned backend is not installed.
    """
    if geometry_type not in ALL_GEOMETRIES:
        raise GraspGenerationError(
            f"Unknown geometry_type '{geometry_type}'; "
            f"expected one of {ALL_GEOMETRIES}",
        )
    if geometry_type == GEOMETRY_ANTIPODAL:
        from .antipodal_grasp import AntipodalGraspSource

        return AntipodalGraspSource()
    if geometry_type in (GEOMETRY_HGGD, GEOMETRY_DENSE):
        from z_manip.inference import GraspInferenceClient, GraspInferenceConfig

        from .learned_grasp import LearnedGraspSource

        prefix = "Z_MANIP_HGGD_" if geometry_type == GEOMETRY_HGGD else "Z_MANIP_DENSE_"
        try:
            config = GraspInferenceConfig.from_env(prefix=prefix)
        except ValueError as error:
            raise GraspGenerationError(
                f"{geometry_type} inference is not configured: {error}",
            ) from error
        return LearnedGraspSource(GraspInferenceClient(config))
    raise GraspGenerationError(f"grasp backend {geometry_type!r} is not installed")


def cascade_generate(
    context: GraspContext,
    *,
    cascade: Sequence[str] = GRASP_CASCADE,
) -> GraspCandidates:
    """Run backends in ``cascade`` order; keep the first that yields grasps.

    The graceful-degradation contract: on :class:`GraspGenerationError` from one
    backend, fall through to the next; the geometric antipodal backend is the
    deterministic last resort. (Reference: ``registry.GRASP_CASCADE`` + node walk.)
    """
    failures = []
    for geometry_type in cascade:
        try:
            return select_grasp_source(geometry_type).generate(context)
        except (GraspGenerationError, NotImplementedError) as exc:
            failures.append(f"{geometry_type}: {exc}")
    raise GraspGenerationError("all grasp backends failed: " + "; ".join(failures))
