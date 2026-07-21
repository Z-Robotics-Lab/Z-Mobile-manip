from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "local_grounding_service.py"
SPEC = importlib.util.spec_from_file_location("local_grounding_service", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    (
        ("白色充电器", "a white charger."),
        ("抓取黑色盒子", "a black box."),
        ("white power adapter", "a white power adapter."),
        ("charger", "a charger."),
    ),
)
def test_grounding_prompt_maps_grasp_instruction_to_noun_phrase(instruction, expected):
    assert SERVICE.grounding_prompt(instruction) == expected


def test_grounding_prompt_defers_unknown_chinese_to_remote_vlm():
    assert SERVICE.grounding_prompt("拿起那个东西") is None


def test_select_detection_rejects_broad_support_surface():
    selected = SERVICE.select_detection(
        ((10, 10, 630, 470), (356, 238, 415, 338)),
        (0.70, 0.58),
        ("table", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
    )

    assert selected is not None
    assert selected["label"] == "white charger"
    assert selected["bbox_xyxy"] == pytest.approx(
        (356 / 640, 238 / 480, 415 / 640, 338 / 480),
    )


def test_select_detection_uses_confidence_before_area_tiebreak():
    selected = SERVICE.select_detection(
        ((10, 10, 100, 100), (200, 200, 230, 230)),
        (0.61, 0.60),
        ("target", "distractor"),
        width=640,
        height=480,
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
    )

    assert selected is not None
    assert selected["label"] == "target"


def test_select_detection_rejects_partial_object_clipped_by_image_border():
    selected = SERVICE.select_detection(
        ((434, 140, 639, 334), (356, 238, 415, 338)),
        (0.99, 0.58),
        ("partial chair", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
    )

    assert selected is not None
    assert selected["label"] == "white charger"
