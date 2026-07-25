"""Grounding domain value objects, scope enumeration, and geometry helpers.

Leaf module (numpy only) shared by the schema builder
(:mod:`z_manip.perception.vlm_schema`), the response parser
(:mod:`z_manip.perception.vlm_result_parsing`), and the orchestrator
(:mod:`z_manip.perception.vlm_openrouter`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_BBOX_COORDINATE_SCALES = {
    "normalized_0_1": 1.0,
    "relative_0_1000": 1000.0,
}

GROUNDING_SCOPES = frozenset({
    "grasp_only",
    "grasp_for_place",
    "place_support",
})


@dataclass(frozen=True)
class NormalizedBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(np.isfinite(values)) or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("normalized box coordinates must be finite and in [0, 1]")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("normalized box must have positive area")

    @classmethod
    def parse(
        cls,
        value: object,
        *,
        coordinate_scale: float = 1.0,
    ) -> "NormalizedBox":
        coordinates = np.asarray(value, dtype=float)
        if coordinates.shape != (4,):
            raise ValueError("bbox_xyxy must contain four coordinates")
        scale = float(coordinate_scale)
        if scale not in _BBOX_COORDINATE_SCALES.values():
            raise ValueError("bbox coordinate scale is unsupported")
        if scale == 1000.0 and (
            not np.all(np.isfinite(coordinates))
            or not np.all(coordinates == np.floor(coordinates))
        ):
            raise ValueError(
                "relative_0_1000 bbox coordinates must be finite integers",
            )
        return cls(*(coordinates / scale).tolist())

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return (
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
            int(round(self.x2 * width)),
            int(round(self.y2 * height)),
        )


@dataclass(frozen=True)
class AvoidRegion:
    label: str
    bbox: NormalizedBox


@dataclass(frozen=True)
class PlacementVerification:
    """Semantic object axes that downstream geometry must observe or reject."""

    require_upright: bool
    upright_axis: str
    orientation_symmetry: str
    symmetry_axis: str | None

    def __post_init__(self) -> None:
        axes = {'principal_long', 'principal_middle', 'principal_short'}
        if not isinstance(self.require_upright, bool):
            raise ValueError("require_upright must be boolean")
        if self.upright_axis not in axes:
            raise ValueError("upright_axis must select an observed principal axis")
        if self.orientation_symmetry not in {'none', 'axial'}:
            raise ValueError("orientation_symmetry must be none or axial")
        if self.orientation_symmetry == 'axial':
            if self.symmetry_axis not in axes:
                raise ValueError("axial symmetry requires a principal symmetry_axis")
        elif self.symmetry_axis is not None:
            raise ValueError("symmetry_axis is only valid for axial symmetry")


@dataclass(frozen=True)
class AffordanceResult:
    model: str
    target_label: str
    target_bbox: NormalizedBox
    confidence: float
    grasp_part_label: str | None
    grasp_part_bbox: NormalizedBox | None
    avoid_regions: tuple[AvoidRegion, ...]
    preferred_approach_camera: tuple[float, float, float] | None
    placement_region_label: str | None
    placement_region_bbox: NormalizedBox | None
    placement_avoid_regions: tuple[AvoidRegion, ...]
    placement_verification: PlacementVerification | None
    constraints: tuple[str, ...]
    latency_s: float


@dataclass(frozen=True)
class VLMAttemptEvent:
    """Bounded, credential-free diagnostics for one model attempt."""

    model: str
    attempt: int
    outcome: str
    elapsed_s: float
    detail: str = ""


def _parse_bbox(
    value: object,
    *,
    coordinate_scale: float,
    field: str,
) -> NormalizedBox:
    """Parse one declared wire-space bbox with a bounded field-path error."""
    try:
        return NormalizedBox.parse(
            value,
            coordinate_scale=coordinate_scale,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}: {error}") from error


def _box_area(box: NormalizedBox) -> float:
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _covered_area(region: NormalizedBox, covers: Sequence[NormalizedBox]) -> float:
    """Return the exact union area of axis-aligned covers clipped to region."""
    clipped = []
    for cover in covers:
        x1 = max(region.x1, cover.x1)
        y1 = max(region.y1, cover.y1)
        x2 = min(region.x2, cover.x2)
        y2 = min(region.y2, cover.y2)
        if x2 > x1 and y2 > y1:
            clipped.append((x1, y1, x2, y2))
    if not clipped:
        return 0.0
    x_edges = sorted({edge for box in clipped for edge in (box[0], box[2])})
    area = 0.0
    for x1, x2 in zip(x_edges, x_edges[1:]):
        if x2 <= x1:
            continue
        intervals = sorted(
            (box[1], box[3])
            for box in clipped
            if box[0] < x2 and box[2] > x1
        )
        if not intervals:
            continue
        covered_y = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_y += current_end - current_start
                current_start, current_end = start, end
        covered_y += current_end - current_start
        area += (x2 - x1) * covered_y
    return area


__all__ = [
    "GROUNDING_SCOPES",
    "AffordanceResult",
    "AvoidRegion",
    "NormalizedBox",
    "PlacementVerification",
    "VLMAttemptEvent",
]
