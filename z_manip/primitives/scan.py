"""scan primitive (L2) — in-place rotate + LOOKOUT arm sweep + per-frame detect.

Contract: with the arm in LOOKOUT (camera level, ``docs/plan.md`` pose table),
rotate the base in place (and optionally sweep the arm) while running the L1
:class:`~z_manip.models.detector.Detector` per frame; hand hits to a VLM
disambiguation step; emit a stable 3D target pose once one ``track_id`` persists
≥K frames (K=3, ``docs/plan.md`` §3 SEARCH gate).

I/O contract (ROS2):
    in:  ``/camera/color/image_raw`` (rgb8), ``/camera/color/camera_info``,
         ``/camera/aligned_depth_to_color/image_raw`` (16UC1 mm), TF.
    out: arm goes LOOKOUT via ``/piper/named_pose``; base rotates via the base
         command owner; a stable Detection3D-style target pose to the caller.
    timeout: SEARCH = 30 sim-s (RTF-scaled; NEVER wall-clock — §3).

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

from typing import Optional, Sequence


def scan(
    queries: Sequence[str],
    *,
    timeout_sim_s: float = 30.0,
    stable_frames: int = 3,
) -> Optional[object]:
    """Rotate-scan for any of ``queries`` and return a stable 3D target pose.

    Args:
        queries: Free-text target descriptions handed to the detector/VLM.
        timeout_sim_s: SEARCH budget in **sim seconds** (RTF-scaled).
        stable_frames: Frames one ``track_id`` must persist before the pose is
            declared stable (K in the SEARCH gate).

    Returns:
        A stable 3D target pose (``(4, 4)`` SE(3) once typed), or ``None`` on
        ``not_found`` (caller escalates per the retry budget, §3a).

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("scan is an M0 skeleton; see docs/plan.md §3 SEARCH.")
