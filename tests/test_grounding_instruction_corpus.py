"""Replay every recorded operator instruction through the local grounder.

The corpus is the target string of all 756 recorded interactive perception
sessions, weighted by how often the operator asked for it. The expected prompt
is the operator's intended target as a normalized English YOLOE class, labelled
independently of the parser so this file measures the grounder rather than
restating it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "local_grounding_service.py"
SPEC = importlib.util.spec_from_file_location("local_grounding_service", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)

CORPUS = json.loads(
    (ROOT / "tests" / "data" / "grounding_instruction_corpus.json").read_text(
        encoding="utf-8"
    )
)
INSTRUCTIONS = CORPUS["instructions"]


def _primary(instruction: str) -> str | None:
    prompts = SERVICE.grounding_prompts(instruction)
    return prompts[0] if prompts else None


def test_corpus_covers_every_recorded_session():
    assert CORPUS["schema"] == "z_manip.grounding_instruction_corpus.v1"
    assert CORPUS["session_count"] == 756
    assert sum(record["sessions"] for record in INSTRUCTIONS) == 756
    assert len(INSTRUCTIONS) == 99


@pytest.mark.parametrize(
    "record",
    INSTRUCTIONS,
    ids=[record["instruction"] for record in INSTRUCTIONS],
)
def test_corpus_instruction_grounds_the_intended_target(record):
    assert _primary(record["instruction"]) == record["expected_prompt"]
    assert SERVICE.spatial_preference(record["instruction"]) == record["expected_spatial"]


def test_corpus_target_identity_and_relation_retention():
    sessions = sum(record["sessions"] for record in INSTRUCTIONS)
    correct = sum(
        record["sessions"]
        for record in INSTRUCTIONS
        if _primary(record["instruction"]) == record["expected_prompt"]
    )
    relational = [record for record in INSTRUCTIONS if record["expected_spatial"]]
    retained = sum(
        record["sessions"]
        for record in relational
        if SERVICE.spatial_preference(record["instruction"]) == record["expected_spatial"]
    )
    # 651/756 before the support split; 0/51 relations survived at all.
    assert correct == sessions
    assert sum(record["sessions"] for record in relational) == 51
    assert retained == 51


def test_no_instruction_invents_a_relation():
    for record in INSTRUCTIONS:
        if record["expected_spatial"] is None:
            assert SERVICE.spatial_preference(record["instruction"]) is None


def test_corpus_never_regresses_a_previously_answerable_instruction():
    # The local tier answering at all is what keeps a request off the 2.7-3.4 s
    # remote path, so no phrase may lose its prompt.
    unanswered = [
        record["instruction"]
        for record in INSTRUCTIONS
        if record["expected_prompt"] is None
    ]
    assert sorted(unanswered) == sorted(
        [
            "拿起桌上的那个白色东西",
            "架子上的白色物体",
            "黑色box",
            "架子上的绿色物体",
            "绿色物体",
            "桌子上的灰色机器狗",
            "机器狗",
        ]
    )
    for instruction in unanswered:
        assert SERVICE.grounding_prompts(instruction) == ()


# Prompt tuples recorded from the deployed image before the support split. The
# charger cohort is 358 of the 756 sessions and must stay byte-identical.
_WHITE_CHARGER = (
    "white charger",
    "white wall charger",
    "white usb charger",
    "white power adapter",
    "white electrical plug",
)
_CHARGER = (
    "charger",
    "wall charger",
    "usb charger",
    "power adapter",
    "electrical plug",
)
_GOLDEN_PROMPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("充电器", _CHARGER),
    ("白色充电器", _WHITE_CHARGER),
    ("黑色箱子上的白色充电器", _WHITE_CHARGER),
    ("黑色台子上的白色充电器", _WHITE_CHARGER),
    ("远处箱子上白色充电器", _WHITE_CHARGER),
    ("竖立的白色充电器", _WHITE_CHARGER),
    ("远处黑色箱子上的白色充电器", _WHITE_CHARGER),
    ("黑色箱子上右边的白色充电器", _WHITE_CHARGER),
    ("黑色箱上面右边的小白色充电器", _WHITE_CHARGER),
    ("箱子上的白色充电器", _WHITE_CHARGER),
    ("远处箱子上的白色充电器", _WHITE_CHARGER),
    ("红色瓶子", ("red bottle",)),
    ("黑色盒子", ("black box",)),
    ("黑色箱子上的黑色盒子", ("black box", "small black box")),
    ("红色可乐", ("red soda bottle",)),
    ("彩色瓶子", ("bottle",)),
)


@pytest.mark.parametrize(("instruction", "expected"), _GOLDEN_PROMPTS)
def test_golden_prompt_tuples_are_byte_identical(instruction, expected):
    assert SERVICE.grounding_prompts(instruction) == expected
