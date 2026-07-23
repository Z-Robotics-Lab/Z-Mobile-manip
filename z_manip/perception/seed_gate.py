"""Pure, transport-free admission gates for text-conditioned grasp seeds.

A VLM fallback reports its own confidence and a 2-D box for the requested
object. Nothing downstream re-checks whether that box actually sits where the
instruction says it should. Two failure modes follow: (1) the model narrates a
near object on the robot itself as the requested *distant* target at high
self-reported confidence, and (2) the flat point-count bundle gate that assumes
a near object geometrically kills a genuinely far, small one.

This module holds the small, deterministic decisions that fix both. It knows
nothing about ROS, CUDA, or the network; callers supply an aligned metric depth
frame (or a median already measured from a tracked cloud) and the instruction
text. Every threshold is a plain argument so the ROS layer can bind it from a
parameter file and the offline corpus replay can sweep it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


# Distance qualifiers that promise the target is NOT within arm's reach. A seed
# whose measured depth contradicts one of these is almost always the wrong
# object (a near distractor on the robot's own body, a support surface, etc.).
DISTANCE_QUALIFIERS: tuple[str, ...] = (
    "远处", "远端", "远方", "较远", "远的", "远一点", "distant", "far away",
    "far-away", "faraway",
)
# Standalone Latin word boundaries handled separately so "farm" never matches.
_LATIN_DISTANCE_WORDS: tuple[str, ...] = ("far",)

# Qualifiers that promise the target is small (and, at range, produces very few
# depth pixels). Used to decide when the ROI-zoom second grounding pass and the
# depth-aware bundle gate are worth their extra cost.
SMALL_QUALIFIERS: tuple[str, ...] = (
    "小型", "小的", "小白", "小黑", "小", "tiny", "small",
)


def has_distance_qualifier(instruction: str) -> bool:
    """True when the instruction explicitly places the target far from the arm."""

    text = str(instruction)
    lowered = text.lower()
    if any(token in text or token in lowered for token in DISTANCE_QUALIFIERS):
        return True
    for word in _LATIN_DISTANCE_WORDS:
        index = lowered.find(word)
        while index != -1:
            before = lowered[index - 1] if index > 0 else " "
            after = (
                lowered[index + len(word)]
                if index + len(word) < len(lowered)
                else " "
            )
            if not before.isalpha() and not after.isalpha():
                return True
            index = lowered.find(word, index + 1)
    return False


def has_small_qualifier(instruction: str) -> bool:
    """True when the instruction marks the target as small."""

    text = str(instruction)
    lowered = text.lower()
    return any(token in text or token in lowered for token in SMALL_QUALIFIERS)


@dataclass(frozen=True)
class SeedDepthGateConfig:
    """Depth admission band for a text-conditioned seed."""

    enabled: bool = True
    # A seed for an explicitly distant target must be at least this far away.
    distant_min_z_m: float = 1.2
    # General physical plausibility band applied to every seed.
    sanity_min_z_m: float = 0.25
    sanity_max_z_m: float = 4.0
    # Fraction of in-band depth pixels the box must contain before its median is
    # trusted. Below this the gate abstains rather than rejects.
    min_valid_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not math.isfinite(self.distant_min_z_m) or self.distant_min_z_m <= 0.0:
            raise ValueError("distant_min_z_m must be finite and positive")
        if not (0.0 < self.sanity_min_z_m < self.sanity_max_z_m):
            raise ValueError("sanity depth band must satisfy 0 < min < max")
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be within [0, 1]")


@dataclass(frozen=True)
class SeedDepthMeasurement:
    """Median depth of a normalized box plus the support that produced it."""

    median_z_m: float | None
    valid_fraction: float
    sampled_pixels: int


def median_depth_in_bbox(
    depth_m: np.ndarray,
    bbox_xyxy_normalized: Sequence[float],
    *,
    sanity_min_z_m: float = 0.25,
    sanity_max_z_m: float = 4.0,
) -> SeedDepthMeasurement:
    """Median in-band depth (metres) under a normalized xyxy box.

    ``depth_m`` is an aligned ``H x W`` metric depth image; zeros/NaNs are the
    usual "no return" sentinels and are excluded, as are depths outside the
    sanity band (which are almost always alignment spill from a far wall).
    """

    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth frame must be a non-empty 2-D array")
    height, width = depth.shape
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy_normalized)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("bbox coordinates must be finite")
    # Clamp to the frame and to a well-ordered box.
    px1 = int(math.floor(min(max(x1, 0.0), 1.0) * width))
    px2 = int(math.ceil(min(max(x2, 0.0), 1.0) * width))
    py1 = int(math.floor(min(max(y1, 0.0), 1.0) * height))
    py2 = int(math.ceil(min(max(y2, 0.0), 1.0) * height))
    px1, px2 = min(px1, width - 1), min(max(px2, px1 + 1), width)
    py1, py2 = min(py1, height - 1), min(max(py2, py1 + 1), height)
    patch = depth[py1:py2, px1:px2]
    total = int(patch.size)
    if total == 0:
        return SeedDepthMeasurement(None, 0.0, 0)
    finite = np.isfinite(patch) & (patch > 0.0)
    in_band = finite & (patch >= sanity_min_z_m) & (patch <= sanity_max_z_m)
    valid = patch[in_band]
    valid_fraction = float(valid.size) / float(total)
    if valid.size == 0:
        return SeedDepthMeasurement(None, valid_fraction, total)
    return SeedDepthMeasurement(float(np.median(valid)), valid_fraction, total)


@dataclass(frozen=True)
class SeedDepthDecision:
    accepted: bool
    reason: str
    median_z_m: float | None
    distance_qualified: bool


def evaluate_seed_depth(
    measurement: SeedDepthMeasurement,
    instruction: str,
    config: SeedDepthGateConfig,
) -> SeedDepthDecision:
    """Decide whether a seed's measured depth is consistent with the request.

    The gate fails *open*: a disabled gate, an unmeasurable box, or a box with
    too little depth support all ABSTAIN (accept), so a missing depth frame can
    never regress the live pipeline into rejecting valid seeds.
    """

    distance_qualified = has_distance_qualifier(instruction)
    if not config.enabled:
        return SeedDepthDecision(True, "gate disabled", measurement.median_z_m, distance_qualified)
    z = measurement.median_z_m
    if z is None or measurement.valid_fraction < config.min_valid_fraction:
        return SeedDepthDecision(
            True,
            f"abstain: insufficient depth support (valid_fraction={measurement.valid_fraction:.3f})",
            z,
            distance_qualified,
        )
    if z < config.sanity_min_z_m or z > config.sanity_max_z_m:
        return SeedDepthDecision(
            False,
            f"seed depth {z:.3f} m outside sanity band "
            f"[{config.sanity_min_z_m:.2f}, {config.sanity_max_z_m:.2f}]",
            z,
            distance_qualified,
        )
    if distance_qualified and z < config.distant_min_z_m:
        return SeedDepthDecision(
            False,
            f"instruction requests a distant target but seed depth is {z:.3f} m "
            f"(< {config.distant_min_z_m:.2f} m)",
            z,
            distance_qualified,
        )
    return SeedDepthDecision(True, "seed depth consistent with request", z, distance_qualified)


@dataclass(frozen=True)
class SeedConfidenceConfig:
    """Hygiene for a fallback model's self-reported confidence."""

    # Cap so a VLM self-report can never be treated as high certainty downstream.
    ceiling: float = 0.60
    apply_ceiling: bool = True
    # Optional: require a weak local (YOLOE) box overlapping the fallback box.
    corroboration_enabled: bool = False
    corroboration_floor: float = 0.08
    corroboration_min_iou: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.ceiling <= 1.0:
            raise ValueError("confidence ceiling must be within (0, 1]")
        if not 0.0 <= self.corroboration_floor <= 1.0:
            raise ValueError("corroboration_floor must be within [0, 1]")
        if not 0.0 <= self.corroboration_min_iou <= 1.0:
            raise ValueError("corroboration_min_iou must be within [0, 1]")


def hygiene_confidence(confidence: float, config: SeedConfidenceConfig) -> float:
    """Clamp a self-reported confidence to the configured ceiling."""

    value = float(confidence)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be within [0, 1]")
    if config.apply_ceiling:
        return min(value, config.ceiling)
    return value


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def local_corroborates(
    fallback_bbox_xyxy: Sequence[float],
    local_boxes_xyxy: Sequence[Sequence[float]],
    local_scores: Sequence[float],
    config: SeedConfidenceConfig,
) -> bool:
    """True when some local box scoring >= floor overlaps the fallback box.

    Coordinates for both must share one space (either both normalized or both
    pixel); only their relative overlap matters.
    """

    for box, score in zip(local_boxes_xyxy, local_scores):
        try:
            if float(score) < config.corroboration_floor:
                continue
            if _iou(fallback_bbox_xyxy, box) >= config.corroboration_min_iou:
                return True
        except (TypeError, ValueError):
            continue
    return False


@dataclass(frozen=True)
class BundleGateConfig:
    """Distance-aware minimum for the first frozen tracker bundle.

    The historical flat count assumed a near object. A 6x3 cm charger yields
    ~400 usable depth pixels at 1.3 m but only ~120 at 2.4 m, so the flat gate
    geometrically kills genuinely far, small targets. ``min_points_for_depth``
    scales the demand by the inverse-square pixel footprint, clamped between a
    hard floor (below which any cluster is noise) and the caller's near-field
    ceiling (the CLI ``--min-bundle-target-points`` value).
    """

    enabled: bool = False
    reference_points: int = 400
    reference_depth_m: float = 1.3
    floor_points: int = 120

    def __post_init__(self) -> None:
        if self.reference_points <= 0:
            raise ValueError("reference_points must be positive")
        if not math.isfinite(self.reference_depth_m) or self.reference_depth_m <= 0.0:
            raise ValueError("reference_depth_m must be finite and positive")
        if self.floor_points <= 0:
            raise ValueError("floor_points must be positive")


def min_points_for_depth(
    median_z_m: float | None,
    ceiling_points: int,
    config: BundleGateConfig,
) -> int:
    """Minimum target points required at ``median_z_m``.

    Returns ``ceiling_points`` (the strict near-field demand, i.e. current
    behaviour) when the gate is disabled or the depth is unusable, so a missing
    measurement never *weakens* the gate by accident.
    """

    if not config.enabled or median_z_m is None:
        return int(ceiling_points)
    z = float(median_z_m)
    if not math.isfinite(z) or z <= 0.0:
        return int(ceiling_points)
    scaled = config.reference_points * (config.reference_depth_m / z) ** 2
    floor = min(config.floor_points, ceiling_points)
    return int(max(floor, min(float(ceiling_points), round(scaled))))
