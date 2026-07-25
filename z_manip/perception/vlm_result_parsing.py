"""Provider response text/JSON extraction and validation into AffordanceResult.

``parse_affordance_result`` was previously ``OpenRouterVLM._result``, a
``@staticmethod`` that never touched ``self``; it moves here unchanged as a
plain function since it is a pure function of its arguments.
"""

from __future__ import annotations

from typing import Mapping

import json

import numpy as np

from z_manip.perception.vlm_types import (
    GROUNDING_SCOPES,
    AffordanceResult,
    AvoidRegion,
    PlacementVerification,
    _box_area,
    _covered_area,
    _parse_bbox,
)


def _message_text(response: Mapping[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = [
            item.get("text", "") for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        return "".join(fragments)
    raise ValueError("response message has no text content")


def _parse_json_text(text: str) -> Mapping[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1:] if first_newline >= 0 else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(stripped[start:end + 1])
    if not isinstance(result, Mapping):
        raise ValueError("VLM output is not a JSON object")
    return result


def parse_affordance_result(
    model: str,
    value: Mapping[str, object],
    latency_s: float,
    *,
    min_confidence: float = 0.15,
    max_target_area_ratio: float = 0.95,
    min_target_border_margin_ratio: float = 0.002,
    max_semantic_conflict_coverage_ratio: float = 0.95,
    bbox_coordinate_scale: float = 1.0,
    grounding_scope: str = "grasp_only",
) -> AffordanceResult:
    if grounding_scope not in GROUNDING_SCOPES:
        raise ValueError("grounding scope is unsupported")
    target = value.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("VLM result has no target")
    label = str(target.get("label", "")).strip()
    if not label:
        raise ValueError("VLM target label is empty")
    target_box = _parse_bbox(
        target.get("bbox_xyxy"),
        coordinate_scale=bbox_coordinate_scale,
        field="target.bbox_xyxy",
    )
    confidence = float(target.get("confidence", -1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("VLM confidence must be in [0, 1]")
    area_ratio = (target_box.x2 - target_box.x1) * (target_box.y2 - target_box.y1)
    if confidence < min_confidence:
        raise ValueError(
            f"VLM confidence {confidence:.3f} is below the configured minimum "
            f"{min_confidence:.3f}"
        )
    if area_ratio > max_target_area_ratio:
        raise ValueError(
            f"VLM target box area {area_ratio:.3f} exceeds the configured maximum "
            f"{max_target_area_ratio:.3f}"
        )
    if grounding_scope == "grasp_only" and (
        target_box.x1 <= min_target_border_margin_ratio
        or target_box.y1 <= min_target_border_margin_ratio
        or target_box.x2 >= 1.0 - min_target_border_margin_ratio
        or target_box.y2 >= 1.0 - min_target_border_margin_ratio
    ):
        raise ValueError(
            "VLM grasp target touches the image border; the complete object "
            "must be visible before tracking",
        )

    grasp_part = value.get("grasp_part")
    part_label, part_box = None, None
    if grasp_part is not None:
        if not isinstance(grasp_part, Mapping):
            raise ValueError("grasp_part must be an object or null")
        part_label = str(grasp_part.get("label", "")).strip()
        if not part_label:
            raise ValueError("grasp part label is empty")
        part_box = _parse_bbox(
            grasp_part.get("bbox_xyxy"),
            coordinate_scale=bbox_coordinate_scale,
            field="grasp_part.bbox_xyxy",
        )
    avoid = []
    for index, region in enumerate(value.get("avoid_regions", [])):
        if not isinstance(region, Mapping):
            raise ValueError("avoid region must be an object")
        avoid.append(AvoidRegion(
            str(region.get("label", "")).strip(),
            _parse_bbox(
                region.get("bbox_xyxy"),
                coordinate_scale=bbox_coordinate_scale,
                field=f"avoid_regions[{index}].bbox_xyxy",
            ),
        ))
    if part_box is not None:
        conflict_ratio = _covered_area(
            part_box,
            tuple(region.bbox for region in avoid),
        ) / _box_area(part_box)
        if conflict_ratio >= max_semantic_conflict_coverage_ratio:
            raise ValueError(
                "grasp_part is covered by avoid_regions "
                f"({conflict_ratio:.3f})",
            )

    preferred_value = value.get("preferred_approach_camera")
    preferred = None
    if preferred_value is not None:
        direction = np.asarray(preferred_value, dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("preferred approach must be a finite three-vector")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            raise ValueError("preferred approach must be nonzero")
        preferred = tuple((direction / norm).tolist())
    placement = value.get("placement_region")
    placement_label, placement_box = None, None
    if placement is not None:
        if not isinstance(placement, Mapping):
            raise ValueError("placement_region must be an object or null")
        placement_label = str(placement.get("label", "")).strip()
        if not placement_label:
            raise ValueError("placement region label is empty")
        placement_box = _parse_bbox(
            placement.get("bbox_xyxy"),
            coordinate_scale=bbox_coordinate_scale,
            field="placement_region.bbox_xyxy",
        )
    placement_avoid = []
    for index, region in enumerate(value.get("placement_avoid_regions", [])):
        if not isinstance(region, Mapping):
            raise ValueError("placement avoid region must be an object")
        placement_avoid.append(AvoidRegion(
            str(region.get("label", "")).strip(),
            _parse_bbox(
                region.get("bbox_xyxy"),
                coordinate_scale=bbox_coordinate_scale,
                field=f"placement_avoid_regions[{index}].bbox_xyxy",
            ),
        ))
    if placement_box is not None and grounding_scope == "place_support":
        conflict_ratio = _covered_area(
            placement_box,
            tuple(region.bbox for region in placement_avoid),
        ) / _box_area(placement_box)
        if conflict_ratio >= max_semantic_conflict_coverage_ratio:
            raise ValueError(
                "placement_region is covered by placement_avoid_regions "
                f"({conflict_ratio:.3f})",
            )
    verification_value = value.get("placement_verification")
    verification = None
    if verification_value is not None:
        if not isinstance(verification_value, Mapping):
            raise ValueError("placement_verification must be an object or null")
        required_verification = {
            "require_upright", "upright_axis",
            "orientation_symmetry", "symmetry_axis",
        }
        if set(verification_value) != required_verification:
            raise ValueError("placement_verification fields are incomplete or unknown")
        symmetry_axis_value = verification_value.get("symmetry_axis")
        if symmetry_axis_value is not None and not isinstance(
            symmetry_axis_value,
            str,
        ):
            raise ValueError("symmetry_axis must be a string or null")
        verification = PlacementVerification(
            require_upright=verification_value.get("require_upright"),
            upright_axis=str(verification_value.get("upright_axis", "")),
            orientation_symmetry=str(
                verification_value.get("orientation_symmetry", ""),
            ),
            symmetry_axis=symmetry_axis_value,
        )
    if (
        placement is not None
        and grounding_scope != "place_support"
        and verification is None
    ):
        raise ValueError(
            "visible placement requires explicit placement_verification"
        )
    if grounding_scope == "grasp_only" and (
        placement is not None or placement_avoid or verification is not None
    ):
        raise ValueError(
            "grasp_only must not return placement geometry or verification",
        )
    if grounding_scope == "grasp_for_place" and (
        placement is not None or placement_avoid
    ):
        raise ValueError(
            "grasp_for_place must not return placement geometry",
        )
    if grounding_scope == "grasp_for_place" and verification is None:
        raise ValueError(
            "grasp_for_place requires explicit placement_verification",
        )
    if grounding_scope == "place_support" and placement is None:
        raise ValueError("place_support requires a visible placement_region")
    if grounding_scope == "place_support" and (
        grasp_part is not None
        or avoid
        or preferred is not None
        or verification is not None
    ):
        raise ValueError(
            "place_support must not return grasp geometry or object verification",
        )
    constraints = tuple(str(item).strip() for item in value.get("constraints", []))
    return AffordanceResult(
        model=model,
        target_label=label,
        target_bbox=target_box,
        confidence=confidence,
        grasp_part_label=part_label,
        grasp_part_bbox=part_box,
        avoid_regions=tuple(avoid),
        preferred_approach_camera=preferred,
        placement_region_label=placement_label,
        placement_region_bbox=placement_box,
        placement_avoid_regions=tuple(placement_avoid),
        placement_verification=verification,
        constraints=constraints,
        latency_s=float(latency_s),
    )


__all__ = ["parse_affordance_result"]
