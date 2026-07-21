"""Deterministic quality-preserving ordering for diverse grasp approaches."""

from __future__ import annotations

import numpy as np


def directionally_diverse_indices(
    grasps: object,
    scores: object,
    limit: int,
) -> np.ndarray:
    """Keep a score quota, then fill the remainder by approach separation."""

    poses = np.asarray(grasps, dtype=float)
    values = np.asarray(scores, dtype=float)
    count = int(limit)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("grasps must contain at least one finite 4x4 pose")
    if not np.all(np.isfinite(poses)):
        raise ValueError("grasp poses must be finite")
    if values.shape != (len(poses),) or not np.all(np.isfinite(values)):
        raise ValueError("grasp scores must be finite and aligned with poses")
    if count < 1:
        raise ValueError("grasp ordering limit must be positive")

    axes = poses[:, :3, 2]
    norms = np.linalg.norm(axes, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-9):
        raise ValueError("grasp approach axes must be finite and nonzero")
    axes = axes / norms[:, None]
    target_count = min(count, len(poses))
    ranked = np.argsort(-values, kind="stable").tolist()
    quality_count = max(1, (target_count + 1) // 2)
    quality = ranked[:quality_count]
    ranked = ranked[quality_count:]
    diverse: list[int] = []
    separation_references = quality.copy()
    while len(quality) + len(diverse) < target_count:
        def separation_key(index: int) -> tuple[float, float]:
            closest_similarity = max(
                float(np.dot(axes[index], axes[kept]))
                for kept in separation_references
            )
            return (1.0 - closest_similarity, float(values[index]))

        chosen = max(ranked, key=separation_key)
        diverse.append(chosen)
        separation_references.append(chosen)
        ranked.remove(chosen)
    selected = [
        index
        for pair in zip(quality, diverse)
        for index in pair
    ]
    if len(quality) > len(diverse):
        selected.append(quality[-1])
    return np.asarray(selected, dtype=int)
