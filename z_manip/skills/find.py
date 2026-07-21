"""find(X) skill (L3) — SEARCH stage.

Contract (``docs/plan.md`` §3 SEARCH):
    entry:   a find(X) request.
    action:  arm STOW → :func:`~z_manip.primitives.scan.scan` (in-place rotate +
             LOOKOUT sweep) + per-frame detect → VLM disambiguation.
    verify:  single-frame detection confidence ≥τ_det (start 0.4) AND the same
             ``track_id`` persists ≥K frames (K=3), yielding a stable 3D pose.
    timeout: 30 sim-s.
    degrade: widen scan → still nothing → report not_found; escalate to zeno for
             re-plan; retry budget SEARCH=2 (§3a), then terminate "X not found".

M0 skeleton: signature + docstring only.
"""

from __future__ import annotations

from typing import Optional

# Retry budget for SEARCH not_found before escalating to zeno (docs/plan.md §3a).
MAX_SEARCH_RETRIES = 2
DETECT_CONFIDENCE_THRESHOLD = 0.4  # τ_det
STABLE_TRACK_FRAMES = 3            # K


def find(target: str, *, timeout_sim_s: float = 30.0) -> Optional[object]:
    """Search for ``target`` and return its stable 3D pose, or ``None``.

    Args:
        target: Free-text description of the object to find.
        timeout_sim_s: SEARCH budget in **sim seconds** (RTF-scaled).

    Returns:
        A stable 3D target pose (``(4, 4)`` SE(3) once typed) on success, or
        ``None`` on not_found after the retry budget is exhausted.

    Raises:
        NotImplementedError: in M0 — skeleton.
    """
    raise NotImplementedError("find is an M0 skeleton; see docs/plan.md §3 SEARCH.")
