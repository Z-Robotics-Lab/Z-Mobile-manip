"""Small numeric helpers shared across the control package.

Kept private to :mod:`z_manip.control`: these are call-site utilities, not a
public API surface.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def point3(value: Sequence[float], *, label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must contain exactly three finite values")
    return point
