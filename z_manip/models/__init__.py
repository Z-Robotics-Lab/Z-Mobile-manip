"""L1 model layer — every perception / grasp / planning model behind a Protocol.

The skills and primitives above never import a concrete model. They depend
only on the Protocols declared in this package, so a backend (YOLO-E vs another
detector, HGGD vs AnyGrasp vs a geometric antipodal fallback, MoveIt2-RRT vs
VAMP) is swapped by config, not by editing callers. This is the ``modularity``
lesson borrowed from the reference stacks (``typing.Protocol`` zero-coupling,
R2 ``modularity.md`` §181-208).

Contracts (see each module):

- :mod:`z_manip.models.grasp_source` — ``GraspSource`` (a.k.a. GraspGenerator):
  ``generate(GraspContext) -> GraspCandidates``. Implemented; backends include
  the geometric antipodal fallback (:mod:`z_manip.models.antipodal_grasp`) and
  a learned-model source (:mod:`z_manip.models.learned_grasp`).
- :mod:`z_manip.models.detector`     — ``Detector``: open-vocab 2D detection.
  Still a placeholder; raises ``NotImplementedError``.
- :mod:`z_manip.models.tracker`      — ``Tracker``: EdgeTAM mask stream (track_3d).
  Still a placeholder; raises ``NotImplementedError``.
- :mod:`z_manip.models.planner`      — ``Planner``: MoveIt2-RRT / VAMP.
  Still a placeholder; raises ``NotImplementedError``. The real motion
  planner used today lives in :mod:`z_manip.planning`.
"""

from __future__ import annotations

__all__: list[str] = []
