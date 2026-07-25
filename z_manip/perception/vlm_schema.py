"""OpenRouter structured-output JSON Schema construction, gated per grounding scope."""

from __future__ import annotations

import copy

from z_manip.perception.vlm_types import _BBOX_COORDINATE_SCALES, GROUNDING_SCOPES


_OUTPUT_SCHEMA = {
    "name": "mobile_manipulation_affordance",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "target", "grasp_part", "avoid_regions",
            "preferred_approach_camera", "placement_region",
            "placement_avoid_regions", "placement_verification", "constraints",
        ],
        "properties": {
            "target": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "bbox_xyxy", "confidence"],
                "properties": {
                    "label": {"type": "string"},
                    "bbox_xyxy": {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 4, "maxItems": 4,
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "grasp_part": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object", "additionalProperties": False,
                        "required": ["label", "bbox_xyxy"],
                        "properties": {
                            "label": {"type": "string"},
                            "bbox_xyxy": {
                                "type": "array", "items": {"type": "number"},
                                "minItems": 4, "maxItems": 4,
                            },
                        },
                    },
                ],
            },
            "avoid_regions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["label", "bbox_xyxy"],
                    "properties": {
                        "label": {"type": "string"},
                        "bbox_xyxy": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": 4, "maxItems": 4,
                        },
                    },
                },
            },
            "preferred_approach_camera": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array", "items": {"type": "number"},
                        "minItems": 3, "maxItems": 3,
                    },
                ],
            },
            "placement_region": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object", "additionalProperties": False,
                        "required": ["label", "bbox_xyxy"],
                        "properties": {
                            "label": {"type": "string"},
                            "bbox_xyxy": {
                                "type": "array", "items": {"type": "number"},
                                "minItems": 4, "maxItems": 4,
                            },
                        },
                    },
                ],
            },
            "placement_avoid_regions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["label", "bbox_xyxy"],
                    "properties": {
                        "label": {"type": "string"},
                        "bbox_xyxy": {
                            "type": "array", "items": {"type": "number"},
                            "minItems": 4, "maxItems": 4,
                        },
                    },
                },
            },
            "placement_verification": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "require_upright", "upright_axis",
                            "orientation_symmetry", "symmetry_axis",
                        ],
                        "properties": {
                            "require_upright": {"type": "boolean"},
                            "upright_axis": {
                                "type": "string",
                                "enum": [
                                    "principal_long", "principal_middle",
                                    "principal_short",
                                ],
                            },
                            "orientation_symmetry": {
                                "type": "string",
                                "enum": ["none", "axial"],
                            },
                            "symmetry_axis": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "string",
                                        "enum": [
                                            "principal_long", "principal_middle",
                                            "principal_short",
                                        ],
                                    },
                                ],
                            },
                        },
                    },
                ],
            },
            "constraints": {"type": "array", "items": {"type": "string"}},
        },
    },
}


def _output_schema(
    coordinate_scale: float,
    grounding_scope: str,
) -> dict[str, object]:
    """Bind provider coordinates and semantic fields to one explicit task stage."""
    scale = float(coordinate_scale)
    if scale not in _BBOX_COORDINATE_SCALES.values():
        raise ValueError("bbox coordinate scale is unsupported")
    if grounding_scope not in GROUNDING_SCOPES:
        raise ValueError("grounding scope is unsupported")
    schema = copy.deepcopy(_OUTPUT_SCHEMA)
    properties = schema["schema"]["properties"]
    boxes = (
        properties["target"]["properties"]["bbox_xyxy"],
        properties["grasp_part"]["anyOf"][1]["properties"]["bbox_xyxy"],
        properties["avoid_regions"]["items"]["properties"]["bbox_xyxy"],
        properties["placement_region"]["anyOf"][1]["properties"]["bbox_xyxy"],
        properties["placement_avoid_regions"]["items"]["properties"]["bbox_xyxy"],
    )
    for box in boxes:
        box["items"]["type"] = "integer" if scale == 1000.0 else "number"
        box["items"]["minimum"] = 0.0
        box["items"]["maximum"] = scale
    if grounding_scope in {"grasp_only", "grasp_for_place"}:
        properties["placement_region"] = {"type": "null"}
        properties["placement_avoid_regions"] = {
            "type": "array",
            "maxItems": 0,
        }
        if grounding_scope == "grasp_only":
            properties["placement_verification"] = {"type": "null"}
        else:
            properties["placement_verification"] = copy.deepcopy(
                _OUTPUT_SCHEMA["schema"]["properties"]
                ["placement_verification"]["anyOf"][1],
            )
    else:
        properties["grasp_part"] = {"type": "null"}
        properties["avoid_regions"] = {"type": "array", "maxItems": 0}
        properties["preferred_approach_camera"] = {"type": "null"}
        properties["placement_verification"] = {"type": "null"}
    return schema


__all__: list[str] = []
