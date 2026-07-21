"""track primitive (L2) — drive the EdgeTAM mask stream (track_3d).

Contract: wrap the L1 :class:`~z_manip.models.tracker.Tracker` (EdgeTAM) as an
atomic primitive. Seed a track from a 2D box (from the detector / VLM-chosen
hit), keep a persistent per-frame mask + fused 3D pose, and report a stable lock
(``is_tracking`` + mask IoU stability) for the ALIGN gate (``τ_mask`` 0.5 for
≥K sim-s, ``docs/plan.md`` §3). Reset the track on exit.

Mirrors the reference ``track_target_base`` primitive: publish the seed bbox,
wait for ``is_tracking``, wait for a stable Detection3D, and reset on terminate
(``refs/.../visual_servoing_base/README.md``).

I/O contract (ROS2):
    in:  ``/camera/color/image_raw`` (rgb8),
         ``/camera/aligned_depth_to_color/image_raw`` (16UC1 mm), a seed bbox.
    out: a persistent mask stream + a fused 3D pose; a reset on terminate.

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

from typing import Optional


def track_init(seed_bbox: object, *, lost_timeout_sim_s: float = 5.0) -> object:
    """Seed a new track from ``seed_bbox`` and wait for a stable lock.

    Args:
        seed_bbox: ``(4,)`` seed box in pixels (from detector / VLM hit).
        lost_timeout_sim_s: Seconds (**sim time**) without a fresh pose before
            declaring the track lost.

    Returns:
        A :class:`~z_manip.models.tracker.TrackState` once locked.

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("track_init is an M0 skeleton; see docs/plan.md §3 ALIGN.")


def track_reset() -> None:
    """Drop the current track (called on skill exit — SUCCESS or FAILURE).

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("track_reset is an M0 skeleton.")
