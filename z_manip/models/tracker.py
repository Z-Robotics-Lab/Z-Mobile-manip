"""Tracker contract — EdgeTAM 3D mask stream (L1, a.k.a. track_3d).

Backend (G4, ``docs/plan.md`` §4): EdgeTAM, the reference stacks' ``track_3d``
component (proven, R2 ``MANIPULATION_STACK_SETUP.md``:266). Seeded with a 2D
box from the :class:`~z_manip.models.detector.Detector`, it returns a persistent
per-frame instance mask plus a fused 3D pose (mask + depth), so the target keeps
a stable ``track_id`` across frames.

Contract shape mirrors the reference ``track_3d`` topic protocol referenced in
``visual_servoing_base``: a bbox seed in, ``is_tracking`` + a stable mask/pose
out, and a reset. The ALIGN gate reads mask IoU stability from this stream
(``τ_mask`` 0.5 for ≥K sim-s, ``docs/plan.md`` §3).

M0 skeleton: contract only. New external dependency (EdgeTAM) = CEO gate —
declared in the blueprint, not pulled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True, eq=False)
class TrackState:
    """One tracker update: the current mask and its fused 3D pose.

    Attributes:
        track_id: Stable identifier persisting across frames while locked.
        is_tracking: Whether the tracker currently holds a confident lock.
        mask: ``(H, W)`` bool/uint8 instance mask, or ``None`` when lost.
            Retype to ``np.ndarray`` at M1.
        pose_3d: ``(4, 4)`` SE(3) target pose in the tracker's frame (mask ×
            depth fusion), or ``None`` when unavailable.
        frame: TF frame ``pose_3d`` is expressed in.
    """

    track_id: int
    is_tracking: bool
    mask: Optional[object]
    pose_3d: Optional[object]
    frame: str


@runtime_checkable
class Tracker(Protocol):
    """EdgeTAM-style persistent mask/pose tracker interface."""

    def init(self, image: object, xyxy: object) -> TrackState:
        """Seed a new track from a 2D box in ``image`` and return first state.

        Args:
            image: ``(H, W, 3)`` rgb8 seed frame.
            xyxy: ``(4,)`` seed box ``(x_min, y_min, x_max, y_max)`` in pixels
                (from the detector / VLM-chosen hit).

        Raises:
            NotImplementedError: in M0 — no tracker is loaded.
        """
        ...

    def update(self, image: object, depth: Optional[object] = None) -> TrackState:
        """Advance the track by one frame; fuse depth for ``pose_3d`` if given.

        Args:
            image: ``(H, W, 3)`` rgb8 frame.
            depth: Optional ``(H, W)`` aligned depth (16UC1 mm semantics, per
                ``/camera/aligned_depth_to_color/image_raw``) for 3D fusion.

        Raises:
            NotImplementedError: in M0.
        """
        ...

    def reset(self) -> None:
        """Drop the current track (called on skill exit — SUCCESS or FAILURE).

        Raises:
            NotImplementedError: in M0.
        """
        ...
