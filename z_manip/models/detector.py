"""Detector contract — open-vocabulary 2D detection / segmentation (L1).

Backend (G4, ``docs/plan.md`` §4): YOLO-E (open-vocab detect + segment), the
same component the reference stacks use. Provider-agnostic: the concrete model
is swapped by config; callers depend only on this Protocol.

The detector answers "where in this frame is <text>?" — it produces 2D
detections (box + optional mask + score) for a free-text class query. The L2
``scan`` primitive runs it per frame during SEARCH; a VLM disambiguation step
(``docs/plan.md`` §4 VLM) picks among multiple hits. The 2D box seeds the
:class:`~z_manip.models.tracker.Tracker` (EdgeTAM) for a persistent mask stream.

M0 skeleton: contract only. No model, no weights. New external dependency
(YOLO-E) = CEO gate — declared in the blueprint, not pulled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, eq=False)
class Detection2D:
    """One open-vocab 2D detection in image pixels.

    Attributes:
        xyxy: ``(4,)`` box ``(x_min, y_min, x_max, y_max)`` in pixels.
        score: Detection confidence in ``[0, 1]`` (compared against ``τ_det``,
            start 0.4, in the SEARCH gate — ``docs/plan.md`` §3).
        label: The text class this detection answers (echo of the query, or the
            open-vocab class the model assigned).
        mask: Optional ``(H, W)`` bool/uint8 instance mask (``None`` if the
            backend is detect-only). Retype to ``np.ndarray`` at M1.
    """

    xyxy: object
    score: float
    label: str
    mask: Optional[object] = None


@runtime_checkable
class Detector(Protocol):
    """Open-vocabulary 2D detector interface.

    Structural typing keeps the L2 ``scan`` primitive zero-coupled to the
    concrete model (YOLO-E or a swap-in).
    """

    def detect(
        self,
        image: object,
        queries: Sequence[str],
        *,
        score_threshold: float = 0.4,
    ) -> list[Detection2D]:
        """Detect open-vocab classes in one RGB frame.

        Args:
            image: ``(H, W, 3)`` rgb8 frame (retype to ``np.ndarray`` at M1),
                as delivered on ``/camera/color/image_raw``.
            queries: Free-text class descriptions to look for.
            score_threshold: Drop detections below this confidence.

        Returns:
            Detections at or above ``score_threshold``, one per instance.

        Raises:
            NotImplementedError: in M0 — no model is loaded.
        """
        ...
