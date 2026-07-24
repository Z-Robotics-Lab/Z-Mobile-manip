"""Parallel-gripper approach-axis symmetry expansion.

Contract (``docs/plan.md`` §4a stage ②): a parallel gripper is invariant to a
180° flip about its approach axis (the two fingers swap, an equivalent grasp),
and *small* rotations about the approach axis are usually still valid. So each
raw grasp candidate is expanded into a FAMILY of SE(3) targets — the identity,
the 180° yaw flip, and a bounded band of small tilts around BOTH — markedly
raising the solvable rate before IK (arxiv 2504.19502, relaxed-rigidity). Pure
geometry, zero IK; the downstream solver (stage ③) is the authority on which
family members are feasible.

IMPORTANT — only 0° and 180° about the approach axis are true parallel-jaw
symmetries: they leave the *closing* axis (tool-X, the physical finger-opening
direction) on the same line. A 90°/270° rotation swaps the closing and binormal
axes, so the jaw ends up closing across a completely different object dimension
(observed live as a standing bottle grasped with the fingers PARALLEL to its
body). The family therefore never samples the full circle — every member stays
within ``max_tilt_deg`` of 0° or 180°.
"""

from __future__ import annotations

import math

import numpy as np


def _band_tilts(count: int, max_tilt_rad: float) -> list[float]:
    """``count`` tilts starting at 0 and fanning out symmetrically within a band."""

    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    tilts = [0.0]
    steps = max(1, count // 2)
    step = max_tilt_rad / steps
    magnitude, sign = 1, 1.0
    while len(tilts) < count:
        tilts.append(sign * magnitude * step)
        if sign < 0.0:
            magnitude += 1
        sign *= -1.0
    return tilts[:count]


def expand_symmetry(
    grasp: object,
    *,
    n_about_axis: int = 8,
    max_tilt_deg: float = 25.0,
) -> np.ndarray:
    """Expand one grasp into its approach-axis symmetry family.

    Args:
        grasp: ``(4, 4)`` SE(3) source grasp pose, columns
            ``(closing, binormal, approach)``.
        n_about_axis: Number of family members. Half fan out (within the tilt
            band) around 0°, half around the 180° finger swap; for an even count
            index ``n_about_axis // 2`` is the exact 180-degree flip.
        max_tilt_deg: Half-width of the relaxed-rigidity band. Members never
            rotate the closing axis beyond this from 0° or 180°, so a quarter
            turn (which would reorient the physical jaw-opening axis) is never
            produced.

    Returns:
        ``(n_about_axis, 4, 4)`` equivalent/near-equivalent SE(3) targets.

    Raises:
        ValueError: If the input is not a valid transform or the count is zero.
    """
    transform = np.asarray(grasp, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("grasp must be a finite (4, 4) transform")
    if n_about_axis < 1:
        raise ValueError("n_about_axis must be positive")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("grasp rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("grasp rotation must be right handed")

    max_tilt = math.radians(max(0.0, float(max_tilt_deg)))
    near_count = (n_about_axis + 1) // 2
    flip_count = n_about_axis - near_count
    angles = [0.0 + tilt for tilt in _band_tilts(near_count, max_tilt)]
    angles += [math.pi + tilt for tilt in _band_tilts(flip_count, max_tilt)]

    family = np.repeat(transform[None, :, :], n_about_axis, axis=0)
    for index, angle in enumerate(angles):
        cosine, sine = math.cos(angle), math.sin(angle)
        local_about_approach = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        )
        family[index, :3, :3] = rotation @ local_about_approach
    return family
