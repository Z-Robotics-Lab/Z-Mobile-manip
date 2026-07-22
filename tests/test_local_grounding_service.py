from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "local_grounding_service.py"
SPEC = importlib.util.spec_from_file_location("local_grounding_service", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)


@pytest.mark.parametrize(
    ("instruction", "expected"),
    (
        ("白色充电器", "white charger"),
        ("抓取黑色盒子", "black box"),
        ("white power adapter", "white power adapter"),
        ("charger", "charger"),
        ("pick up the red bottle", "red bottle"),
    ),
)
def test_grounding_prompt_maps_grasp_instruction_to_noun_phrase(instruction, expected):
    assert SERVICE.grounding_prompt(instruction) == expected


def test_grounding_prompt_defers_unknown_chinese_to_remote_vlm():
    assert SERVICE.grounding_prompt("拿起那个东西") is None


def test_grounding_prompts_keep_aliases_semantically_bounded():
    assert SERVICE.grounding_prompts("白色充电器") == (
        "white charger",
        "white wall charger",
        "white usb charger",
        "white power adapter",
        "white electrical plug",
    )
    assert SERVICE.grounding_prompts("黑色箱子上的黑色盒子") == (
        "black box",
        "small black box",
    )
    assert SERVICE.grounding_prompts("红色瓶子") == ("red bottle",)


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


def test_select_detection_rejects_support_surface_for_small_alias():
    selected = SERVICE.select_detection(
        ((150, 80, 510, 280), (356, 238, 415, 338)),
        (0.84, 0.58),
        ("small black box", "black box"),
        width=640,
        height=480,
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
        maximum_area_ratio_by_label={"small black box": 0.12},
    )

    assert selected is not None
    assert selected["label"] == "black box"


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


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class _FakeBoxes:
    xyxy = _FakeTensor([[10, 10, 30, 30]])
    conf = _FakeTensor([0.9])
    cls = _FakeTensor([0])


class _FakeResult:
    boxes = _FakeBoxes()
    names = {0: "target"}


class _FakeModel:
    def __init__(self):
        self.classes = []
        self.embedding_requests = []
        self.model = self

    def get_text_pe(self, classes, *, cache_clip_model=False):
        self.embedding_requests.append((tuple(classes), cache_clip_model))
        return ("embedding", *classes)

    def set_classes(self, classes, embeddings=None):
        self.classes.append((tuple(classes), embeddings))

    def predict(self, **kwargs):
        assert kwargs["half"] is False
        return [_FakeResult()]


def test_runtime_keeps_dynamic_prompt_inference_fp32():
    runtime = SERVICE.GroundingRuntime(
        model_id="unused.pt",
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
    )
    runtime._model = _FakeModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (64, 64), color=(127, 127, 127))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")

    bottle = runtime.ground(encoded.getvalue(), "bottle")
    assert bottle["prompt"] == "bottle"
    assert bottle["target"]["label"] == "bottle"
    assert bottle["embedding_cache_hit"] is False
    assert bottle["timings_s"]["total"] == pytest.approx(bottle["latency_s"])
    repeated_bottle = runtime.ground(encoded.getvalue(), "bottle")
    assert repeated_bottle["embedding_cache_hit"] is True
    red_bottle = runtime.ground(encoded.getvalue(), "red bottle")
    assert red_bottle["prompt"] == "red bottle"
    assert red_bottle["embedding_cache_hit"] is False
    cached_bottle = runtime.ground(encoded.getvalue(), "bottle")
    assert cached_bottle["prompt"] == "bottle"
    assert cached_bottle["embedding_cache_hit"] is True
    assert runtime._model.embedding_requests == [
        (("bottle",), True),
        (("red bottle",), True),
    ]
    assert runtime._model.classes == [
        (("bottle",), ("embedding", "bottle")),
        (("red bottle",), ("embedding", "red bottle")),
        (("bottle",), ("embedding", "bottle")),
    ]


def test_runtime_text_embedding_cache_is_bounded_and_exact():
    runtime = SERVICE.GroundingRuntime(
        model_id="unused.pt",
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
        text_embedding_cache_size=2,
    )
    runtime._model = _FakeModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (64, 64), color=(127, 127, 127))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")

    for prompt in ("red bottle", "black bottle", "charger", "red bottle"):
        runtime.ground(encoded.getvalue(), prompt)

    assert runtime._model.embedding_requests == [
        (("red bottle",), True),
        (("black bottle",), True),
        (("charger",), True),
        (("red bottle",), True),
    ]
    assert tuple(runtime._text_embeddings) == (("charger",), ("red bottle",))
