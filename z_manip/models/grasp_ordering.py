"""Deterministic quality-preserving ordering for diverse grasp approaches."""

from __future__ import annotations

import numpy as np


def lateral_approach_scores(
    grasps: object,
    scores: object,
    *,
    lateral_direction: object = (0.0, 1.0, 0.0),
    up_direction: object = (0.0, 0.0, 1.0),
    lateral_weight: float = 0.0,
    overhead_penalty_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bias feasible grasp search toward either side of the mobile base.

    The grasp pose's tool-Z axis is the contact approach direction.  A lateral
    approach keeps the pregrasp/contact segment away from fixtures on the
    Go2W centreline (notably Mid360), while a negative up projection denotes
    an overhead, downward approach.  This is deliberately a *soft* ordering:
    IK and continuous collision checks remain authoritative and may select a
    non-lateral candidate when both side approaches are infeasible.
    """

    poses = np.asarray(grasps, dtype=float)
    values = np.asarray(scores, dtype=float)
    lateral = np.asarray(lateral_direction, dtype=float)
    up = np.asarray(up_direction, dtype=float)
    side_weight = float(lateral_weight)
    overhead_weight = float(overhead_penalty_weight)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or not np.all(np.isfinite(poses))
    ):
        raise ValueError("lateral-approach grasps must be finite Nx4x4 poses")
    if values.shape != (len(poses),) or not np.all(np.isfinite(values)):
        raise ValueError("lateral-approach scores must align with grasps")
    if lateral.shape != (3,) or up.shape != (3,):
        raise ValueError("lateral and up directions must be 3-vectors")
    if not np.all(np.isfinite(lateral)) or not np.all(np.isfinite(up)):
        raise ValueError("lateral and up directions must be finite")
    lateral_norm = float(np.linalg.norm(lateral))
    up_norm = float(np.linalg.norm(up))
    if lateral_norm <= 1e-9 or up_norm <= 1e-9:
        raise ValueError("lateral and up directions must be nonzero")
    if abs(float(lateral @ up) / (lateral_norm * up_norm)) > 1e-6:
        raise ValueError("lateral and up directions must be orthogonal")
    if (
        not np.isfinite(side_weight)
        or not np.isfinite(overhead_weight)
        or not 0.0 <= side_weight <= 1.0
        or not 0.0 <= overhead_weight <= 1.0
    ):
        raise ValueError("lateral bonus and overhead penalty weights must be within [0, 1]")

    approaches = poses[:, :3, 2]
    norms = np.linalg.norm(approaches, axis=1)
    if np.any(norms <= 1e-9):
        raise ValueError("lateral-approach axes must be nonzero")
    approaches = approaches / norms[:, None]
    lateral /= lateral_norm
    up /= up_norm

    # Either robot side is equally valid.  A small horizontal component keeps
    # oblique side entries ahead of vertical entries even when their pure-Y
    # component is moderate.
    lateral_alignment = np.abs(approaches @ lateral)
    vertical_projection = approaches @ up
    horizontal_alignment = np.sqrt(np.maximum(0.0, 1.0 - vertical_projection**2))
    bonuses = side_weight * (
        0.75 * lateral_alignment + 0.25 * horizontal_alignment
    )
    # Negative up projection means pregrasp lies above contact: the exact path
    # that tends to sweep the wrist camera plate over the centreline lidar.
    penalties = overhead_weight * np.maximum(0.0, -vertical_projection)
    return values + bonuses - penalties, bonuses, penalties


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
