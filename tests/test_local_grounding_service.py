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
        self.last_predict_kwargs = dict(kwargs)
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


def test_runtime_defaults_to_640_forward_resolution():
    runtime = SERVICE.GroundingRuntime(
        model_id="unused.pt",
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
    )
    assert runtime.image_size == SERVICE.DEFAULT_IMAGE_SIZE == 640
    runtime._model = _FakeModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (64, 64), color=(127, 127, 127))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")

    runtime.ground(encoded.getvalue(), "bottle")
    assert runtime._model.last_predict_kwargs["imgsz"] == 640


def test_runtime_forwards_configured_image_size_to_predict():
    runtime = SERVICE.GroundingRuntime(
        model_id="unused.pt",
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
        image_size=960,
    )
    assert runtime.image_size == 960
    runtime._model = _FakeModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (64, 64), color=(127, 127, 127))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")

    runtime.ground(encoded.getvalue(), "bottle")
    assert runtime._model.last_predict_kwargs["imgsz"] == 960


def test_runtime_rejects_non_stride_image_size():
    for bad in (0, 31, 100, -960):
        with pytest.raises(ValueError):
            SERVICE.GroundingRuntime(
                model_id="unused.pt",
                minimum_confidence=0.35,
                maximum_area_ratio=0.45,
                image_size=bad,
            )


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


@pytest.mark.parametrize(
    ("instruction", "expected_first"),
    (
        # 箱子 (box) is a support here; the charger must still win because it
        # sorts before the box noun and is present in the phrase.
        ("远处箱子上白色充电器", "white charger"),
        ("箱子上的白色充电器", "white charger"),
        # Newly added zero-hit nouns.
        ("红色可乐", "red soda bottle"),
        ("可乐瓶", "soda bottle"),
        ("黑色airpods", "black wireless earbuds"),
        ("黑色耳机", "black headphones"),
        ("小电器", "small appliance"),
        # 电器 must not degrade the charger (which contains the substring 电器).
        ("白色充电器", "white charger"),
        ("箱子", "box"),
    ),
)
def test_added_nouns_are_identity_preserving_and_correctly_ordered(instruction, expected_first):
    prompts = SERVICE.grounding_prompts(instruction)
    assert prompts and prompts[0] == expected_first


def test_box_support_relation_adds_small_variant():
    assert SERVICE.grounding_prompts("远处箱子上的黑色盒子") == ("black box", "small black box")


@pytest.mark.parametrize(
    ("instruction", "expected"),
    (
        ("远处箱子上白色充电器", True),
        ("远处小白色方块", True),
        ("small charger", True),
        ("tiny block", True),
        ("白色充电器", False),
        ("the farm charger", False),
        ("smaller shelf", False),
    ),
)
def test_roi_zoom_qualifier_detection(instruction, expected):
    assert SERVICE.roi_zoom_qualifier(instruction) is expected


def test_center_crop_region_is_central_half():
    assert SERVICE.center_crop_region(640, 480, 0.5) == (160, 120, 480, 360)
    assert SERVICE.center_crop_region(640, 480, 1.0) == (0, 0, 640, 480)


def test_merge_detection_lists_dedupes_by_iou_keeping_stronger():
    boxes, scores, labels = SERVICE.merge_detection_lists(
        [[0, 0, 10, 10]], [0.5], ["a"],
        [[1, 1, 11, 11], [100, 100, 110, 110]], [0.9, 0.3], ["b", "c"],
        iou_threshold=0.6,
    )
    # The overlapping crop box (0.9) replaces the weaker full-frame box; the
    # disjoint one is appended.
    assert boxes == [[1, 1, 11, 11], [100, 100, 110, 110]]
    assert scores == [0.9, 0.3]
    assert labels == ["b", "c"]


class _RoiFakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)


class _RoiFakeResult:
    def __init__(self, xyxy, conf, cls):
        self.boxes = _RoiFakeBoxes(xyxy, conf, cls)
        self.names = {0: "target"}


class _RoiFakeModel:
    """Full-frame pass finds nothing; the centre-crop pass finds the target."""

    def __init__(self):
        self.model = self
        self.predict_sizes = []

    def get_text_pe(self, classes, *, cache_clip_model=False):
        return ("embedding", *classes)

    def set_classes(self, classes, embeddings=None):
        pass

    def predict(self, **kwargs):
        source = kwargs["source"]
        self.predict_sizes.append(source.size)
        if source.size == (640, 480):
            return [_RoiFakeResult([], [], [])]
        return [_RoiFakeResult([[50, 50, 90, 90]], [0.9], [0])]


def test_roi_zoom_second_pass_maps_crop_detection_to_full_frame():
    runtime = SERVICE.GroundingRuntime(
        model_id="fake.pt",
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
        roi_zoom_enabled=True,
        roi_zoom_fraction=0.5,
    )
    runtime._model = _RoiFakeModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (640, 480), color=(90, 90, 90))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")

    response = runtime.ground(encoded.getvalue(), "远处充电器")

    # Two forwards: full frame then the 320x240 centre crop.
    assert runtime._model.predict_sizes == [(640, 480), (320, 240)]
    assert response["roi_zoom_used"] is True
    # Crop box [50,50,90,90] + crop origin (160,120) => full-frame [210,170,250,210].
    assert response["target"]["bbox_xyxy"] == pytest.approx(
        (210 / 640, 170 / 480, 250 / 640, 210 / 480)
    )


def test_roi_zoom_skipped_without_qualifier():
    runtime = SERVICE.GroundingRuntime(
        model_id="fake.pt",
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
        roi_zoom_enabled=True,
    )
    runtime._model = _RoiFakeModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (640, 480), color=(90, 90, 90))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    # A near, unqualified charger: no ROI pass, and the empty full frame yields
    # no qualified box -> local miss -> LookupError (VLM fallback upstream).
    with pytest.raises(LookupError):
        runtime.ground(encoded.getvalue(), "充电器")
    assert runtime._model.predict_sizes == [(640, 480)]
