"""Parallel-gripper approach-axis symmetry expansion.

Contract (``docs/plan.md`` §4a stage ②): a parallel gripper is invariant to a
180° flip about its approach axis (the two fingers swap, an equivalent grasp),
and small rotations about the approach axis are usually still valid. So each raw
grasp candidate is expanded into a FAMILY of SE(3) targets — N samples about the
approach axis + the 180° yaw flip — markedly raising the solvable rate before IK
(arxiv 2504.19502, relaxed-rigidity). Pure geometry, zero IK; the downstream
solver (stage ③) is the authority on which family members are feasible.

"""

from __future__ import annotations

import numpy as np


def expand_symmetry(grasp: object, *, n_about_axis: int = 8) -> np.ndarray:
    """Expand one grasp into its approach-axis symmetry family.

    Args:
        grasp: ``(4, 4)`` SE(3) source grasp pose.
        n_about_axis: Number of evenly spaced samples over a full turn. For an
            even count this includes the exact 180-degree finger-swap pose.

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

    family = np.repeat(transform[None, :, :], n_about_axis, axis=0)
    for index, angle in enumerate(np.arange(n_about_axis) * (2.0 * np.pi / n_about_axis)):
        cosine, sine = np.cos(angle), np.sin(angle)
        local_about_approach = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        )
        family[index, :3, :3] = rotation @ local_about_approach
    return family
