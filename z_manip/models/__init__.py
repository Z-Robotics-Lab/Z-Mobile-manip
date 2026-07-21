"""L1 model layer — every perception / grasp / planning model behind a Protocol.

The skills and primitives above never import a concrete model. They depend
only on the Protocols declared in this package, so a backend (YOLO-E vs another
detector, HGGD vs AnyGrasp vs a geometric antipodal fallback, MoveIt2-RRT vs
VAMP) is swapped by config, not by editing callers. This is the ``modularity``
lesson borrowed from the reference stacks (``typing.Protocol`` zero-coupling,
R2 ``modularity.md`` §181-208).

Contracts (see each module):

- :mod:`z_manip.models.grasp_source` — ``GraspSource`` (a.k.a. GraspGenerator):
  ``generate(GraspContext) -> GraspCandidates``. Backends: HGGD (A budget) /
  AnyGrasp / geometric antipodal (always-on fallback) / GT-heuristic (bring-up).
- :mod:`z_manip.models.detector`     — ``Detector``: open-vocab 2D detection.
- :mod:`z_manip.models.tracker`      — ``Tracker``: EdgeTAM mask stream (track_3d).
- :mod:`z_manip.models.planner`      — ``Planner``: MoveIt2-RRT / VAMP.

Skeleton only (M0). No model is implemented; each Protocol raises
``NotImplementedError`` in its placeholder.
"""

from __future__ import annotations

__all__: list[str] = []
