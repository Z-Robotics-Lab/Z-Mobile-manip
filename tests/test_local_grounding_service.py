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
        # The particle OPENS the phrase, so the support side of the split is
        # empty and _zh_support_split returns None. The relation is still named
        # and the small-variant alias -- with the 0.12 per-label area cap it
        # arms -- has to survive, or the support outscores the target again.
        ("上面的黑色盒子", ("black box", "small black box")),
        ("顶部的盒子", ("box", "small box")),
        ("上边的盒子", ("box", "small box")),
        ("顶上的方块", ("block", "small block")),
        ("里面的盒子", ("box", "small box")),
        # 里 is a recognised particle, so an explicit container arms it too.
        ("黑色箱子里的黑色盒子", ("black box", "small black box")),
        ("盒子里的方块", ("block", "small block")),
        # No support relation named at all: no alias, no area cap.
        ("盒子", ("box",)),
        ("黑色盒子", ("black box",)),
        ("黑色相机盒子", ("black box",)),
        ("黑色箱子", ("black box",)),
    ),
)
def test_support_relation_arms_the_small_alias_even_with_an_elided_support(
    instruction, expected
):
    assert SERVICE.grounding_prompts(instruction) == expected


class _NamedFakeResult:
    def __init__(self, xyxy, conf, cls, names):
        self.boxes = _RoiFakeBoxes(xyxy, conf, cls)
        self.names = names


class _SupportAndTargetModel:
    """A large storage box with the small black box the operator wants on it.

    YOLOE labels both instances with the small-variant class, so only the
    per-label area cap that the alias arms can tell support from target.
    """

    def __init__(self):
        self.model = self
        self.classes = []

    def get_text_pe(self, classes, *, cache_clip_model=False):
        return ("embedding", *classes)

    def set_classes(self, classes, embeddings=None):
        self.classes.append(tuple(classes))

    def predict(self, **kwargs):
        return [
            _NamedFakeResult(
                [[80, 144, 560, 336], [300, 200, 378, 278]],
                [0.80, 0.55],
                [0, 0],
                {0: "small black box"},
            )
        ]


def test_support_initial_phrase_still_rejects_the_support_end_to_end():
    runtime = SERVICE.GroundingRuntime(
        model_id="fake.pt",
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
    )
    runtime._model = _SupportAndTargetModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (640, 480), color=(90, 90, 90))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")

    response = runtime.ground(encoded.getvalue(), "上面的黑色盒子")

    assert runtime._model.classes == [("black box", "small black box")]
    # 0.30 area ratio at 0.80 confidence is the storage box; without the alias
    # there is no per-label cap and it wins the argmax outright.
    assert response["target"]["confidence"] == pytest.approx(0.55)
    assert response["target"]["area_ratio"] == pytest.approx(
        (378 - 300) * (278 - 200) / (640 * 480)
    )
    assert response["target"]["label"] == "black box"


def test_sentence_separators_carry_no_dead_newline():
    # _parse_instruction collapses every run of whitespace before
    # _before_separator runs, so a "\n" entry could never have matched.
    assert "\n" not in SERVICE._SENTENCE_SEPARATORS


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


@pytest.mark.parametrize(
    ("instruction", "expected"),
    (
        # The support noun precedes the target noun in _ZH_NOUNS, so tuple order
        # alone grounded the thing the target rests on.
        ("箱子上的瓶子", ("bottle",)),
        ("盒子里的杯子", ("cup",)),
        ("黑色箱子上的彩色瓶子", ("bottle",)),
        ("黑色箱子上斜着的彩色瓶子", ("bottle",)),
        ("黑色台子上的瓶子", ("bottle",)),
        ("机器狗身上的小充电器", ("charger", "wall charger", "usb charger",
                                  "power adapter", "electrical plug")),
        # English names the target before the preposition, not after it.
        ("the colourful bottle on the black box", ("colourful bottle",)),
        ("small white charger on the black box in the middile", ("small white charger",)),
        ("white charger on the shelf", ("white charger",)),
        ("the  tissue box  on  black box", ("tissue box",)),
    ),
)
def test_support_relation_grounds_the_target_not_the_support(instruction, expected):
    assert SERVICE.grounding_prompts(instruction) == expected


def test_support_colour_does_not_leak_onto_the_target():
    # 黑色 describes the stand; the bottle has no stated colour.
    assert SERVICE.grounding_prompts("黑色台子上的瓶子") == ("bottle",)
    assert SERVICE.grounding_prompts("黑色箱子上的黑色盒子") == ("black box", "small black box")


def test_trailing_clause_after_a_separator_never_supplies_the_noun():
    assert SERVICE.grounding_prompts("黑色箱子上的瓶子，不要黑色盒子") == ("bottle",)
    assert SERVICE.grounding_prompts(
        "the colourful bottle on the black box， you need to search for it"
    ) == ("colourful bottle",)


def test_english_colour_reaches_an_ascii_noun_lexicon_entry():
    assert SERVICE.grounding_prompts("the black airpods on the black box") == (
        "black wireless earbuds",
    )


@pytest.mark.parametrize(
    ("instruction", "expected"),
    (
        ("黑色箱子上右边的白色充电器", "right"),
        ("黑色箱上面右边的小白色充电器", "right"),
        ("黑色箱上面的右边白色充电器", "right"),
        ("黑色箱子上左边的瓶子", "left"),
        ("远处彩色瓶子", "far"),
        ("中间的瓶子", "middle"),
        ("the bottle on the left", "left"),
        ("the charger in the middle", "middle"),
        # A qualifier that describes the support must not move to the target.
        ("远处箱子上白色充电器", None),
        ("远处地上的白色充电器", None),
        ("small white charger on the black box in the middile", None),
        ("白色充电器", None),
        ("the farm charger", None),
    ),
)
def test_spatial_preference_extraction(instruction, expected):
    assert SERVICE.spatial_preference(instruction) == expected


@pytest.mark.parametrize(
    "instruction",
    (
        # Whole-word matching stops "farm", but it cannot stop a position word
        # that is a genuine attributive part of the object's own name. Each of
        # these carries no spatial intent whatsoever.
        "the left handle",
        "center punch",
        "the near field probe",
        "a distant object",
        "grab the left handle",
        "the far side bottle",
        "pick up the centre punch",
        "the right angle bracket",
    ),
)
def test_attributive_position_word_in_a_name_emits_no_preference(instruction):
    assert SERVICE.spatial_preference(instruction) is None


@pytest.mark.parametrize(
    ("instruction", "expected"),
    (
        # ...while a position word that heads a trailing positional phrase, or
        # a CJK locative in the target segment, still qualifies the target.
        ("the bottle on the left", "left"),
        ("the charger in the middle", "middle"),
        ("the charger on the right side", "right"),
        ("黑色箱子上右边的白色充电器", "right"),
        ("黑色箱上面右边的小白色充电器", "right"),
        ("黑色箱上面的右边白色充电器", "right"),
        ("黑色箱子上左边的瓶子", "left"),
        ("远处彩色瓶子", "far"),
        ("中间的瓶子", "middle"),
    ),
)
def test_real_qualifier_phrasings_still_produce_a_preference(instruction, expected):
    assert SERVICE.spatial_preference(instruction) == expected


@pytest.mark.parametrize(
    "instruction",
    (
        "the bottle to the right of the box",
        "the charger on the left of the shelf",
    ),
)
def test_position_word_before_a_second_noun_is_a_deliberate_miss(instruction):
    # An English position word followed by ANOTHER object name is genuinely
    # ambiguous -- "left handle" and "left of the shelf" are the same shape --
    # and nothing in the 756-session recording disambiguates it: all 51
    # qualifier-carrying sessions are Chinese. Withholding the preference is
    # the safe side of that ambiguity, and it is recorded here rather than
    # discovered later. Revisit with recorded English relational sessions.
    assert SERVICE.spatial_preference(instruction) is None


def test_left_and_right_no_longer_produce_identical_requests():
    left = "黑色箱子上左边的白色充电器"
    right = "黑色箱子上右边的白色充电器"
    assert SERVICE.grounding_prompts(left) == SERVICE.grounding_prompts(right)
    assert SERVICE.spatial_preference(left) == "left"
    assert SERVICE.spatial_preference(right) == "right"


_TWO_INSTANCE_BOXES = ((100, 200, 160, 260), (400, 200, 460, 260))


@pytest.fixture
def spatial_reorder_enabled(monkeypatch):
    """Reach the geometric path that ships dark."""

    monkeypatch.setattr(SERVICE, "SPATIAL_REORDER_ENABLED", True)


def test_spatial_reorder_ships_disabled():
    # The qualifier is parsed, carried in the response and available to a later
    # tier; it does NOT move the grasp target until the reorder has been
    # observed against real YOLOE score distributions on live frames.
    assert SERVICE.SPATIAL_REORDER_ENABLED is False


@pytest.mark.parametrize("preference", ("left", "right", "middle", "near", "far"))
def test_qualifier_leaves_the_selection_on_confidence_argmax_by_default(preference):
    selected = SERVICE.select_detection(
        _TWO_INSTANCE_BOXES,
        (0.52, 0.61),
        ("white charger", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.15,
        maximum_area_ratio=0.45,
        spatial_preference=preference,
    )

    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.61)


def test_recorded_right_qualifier_keeps_the_confident_charger():
    # 黑色箱子上右边的白色充电器, 13 recorded sessions. The 0.80 box is the real
    # charger; the 0.49 box further right is a weak false positive. A 0.6 peer
    # ratio is loose enough to admit it, so the reorder would hand the grasp to
    # the wrong physical object.
    selected = SERVICE.select_detection(
        ((280, 200, 340, 270), (500, 210, 540, 250)),
        (0.80, 0.49),
        ("white charger", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
        spatial_preference="right",
    )

    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.80)


@pytest.mark.parametrize("preference", ("far", "near"))
@pytest.mark.parametrize("reorder_enabled", (False, True))
def test_near_and_far_never_rank_on_apparent_size(
    monkeypatch, preference, reorder_enabled
):
    # 远处彩色瓶子, 10 recorded sessions. Ranking "far" onto the smallest area
    # actively selects the tiny fragments YOLOE emits as noise, so the depth
    # proxy is gone outright -- not merely gated behind the flag.
    monkeypatch.setattr(SERVICE, "SPATIAL_REORDER_ENABLED", reorder_enabled)
    selected = SERVICE.select_detection(
        ((100, 200, 130, 230), (400, 180, 480, 260)),
        (0.43, 0.70),
        ("bottle", "bottle"),
        width=640,
        height=480,
        minimum_confidence=0.35,
        maximum_area_ratio=0.45,
        spatial_preference=preference,
    )

    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.70)
    assert selected["area_ratio"] == pytest.approx(80 * 80 / (640 * 480))


@pytest.mark.parametrize(
    ("preference", "expected_x1"),
    (("left", 100 / 640), ("right", 400 / 640)),
)
def test_enabled_reorder_uses_the_qualifier_to_disambiguate(
    spatial_reorder_enabled, preference, expected_x1
):
    selected = SERVICE.select_detection(
        _TWO_INSTANCE_BOXES,
        (0.52, 0.61),
        ("white charger", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.15,
        maximum_area_ratio=0.45,
        spatial_preference=preference,
    )

    assert selected is not None
    assert selected["bbox_xyxy"][0] == pytest.approx(expected_x1)


def test_select_detection_without_qualifier_is_unchanged():
    selected = SERVICE.select_detection(
        _TWO_INSTANCE_BOXES,
        (0.52, 0.61),
        ("white charger", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.15,
        maximum_area_ratio=0.45,
    )

    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.61)


def test_enabled_reorder_does_not_promote_a_much_weaker_detection(
    spatial_reorder_enabled,
):
    selected = SERVICE.select_detection(
        _TWO_INSTANCE_BOXES,
        (0.18, 0.90),
        ("white charger", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.15,
        maximum_area_ratio=0.45,
        spatial_preference="left",
    )

    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.90)


def test_enabled_reorder_is_ignored_when_boxes_are_not_separated(
    spatial_reorder_enabled,
):
    selected = SERVICE.select_detection(
        ((300, 200, 360, 260), (310, 205, 370, 265)),
        (0.52, 0.61),
        ("white charger", "white charger"),
        width=640,
        height=480,
        minimum_confidence=0.15,
        maximum_area_ratio=0.45,
        spatial_preference="left",
    )

    assert selected is not None
    assert selected["confidence"] == pytest.approx(0.61)


def test_enabled_middle_qualifier_picks_the_central_instance(spatial_reorder_enabled):
    selected = SERVICE.select_detection(
        ((60, 200, 120, 260), (300, 200, 360, 260), (520, 200, 580, 260)),
        (0.61, 0.52, 0.58),
        ("bottle", "bottle", "bottle"),
        width=640,
        height=480,
        minimum_confidence=0.15,
        maximum_area_ratio=0.45,
        spatial_preference="middle",
    )

    assert selected is not None
    assert selected["bbox_xyxy"][0] == pytest.approx(300 / 640)


def test_select_detection_rejects_an_unsupported_qualifier():
    with pytest.raises(ValueError):
        SERVICE.select_detection(
            _TWO_INSTANCE_BOXES,
            (0.52, 0.61),
            ("white charger", "white charger"),
            width=640,
            height=480,
            minimum_confidence=0.15,
            maximum_area_ratio=0.45,
            spatial_preference="behind",
        )


class _TwoInstanceModel:
    """Both instances of the requested class are visible in one frame."""

    def __init__(self):
        self.model = self

    def get_text_pe(self, classes, *, cache_clip_model=False):
        return ("embedding", *classes)

    def set_classes(self, classes, embeddings=None):
        pass

    def predict(self, **kwargs):
        return [
            _RoiFakeResult(
                [[100, 200, 160, 260], [400, 200, 460, 260]],
                [0.52, 0.61],
                [0, 0],
            )
        ]


def _ground_two_instances(instruction):
    runtime = SERVICE.GroundingRuntime(
        model_id="fake.pt",
        minimum_confidence=0.15,
        maximum_area_ratio=0.45,
    )
    runtime._model = _TwoInstanceModel()
    runtime._device = "cuda:0"
    image = Image.new("RGB", (640, 480), color=(90, 90, 90))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    return runtime.ground(encoded.getvalue(), instruction)


@pytest.mark.parametrize(
    ("instruction", "expected_preference"),
    (
        ("黑色箱子上左边的白色充电器", "left"),
        ("黑色箱子上右边的白色充电器", "right"),
        ("黑色箱子上的白色充电器", None),
    ),
)
def test_ground_reports_the_qualifier_without_moving_the_target(
    instruction, expected_preference
):
    response = _ground_two_instances(instruction)

    assert response["prompt"] == "white charger"
    assert response["spatial_preference"] == expected_preference
    # Shipped default: the relation is carried in the response for a later
    # tier, and the box handed to the grasp pipeline is still the argmax.
    assert response["target"]["bbox_xyxy"][0] == pytest.approx(400 / 640)
    assert response["target"]["confidence"] == pytest.approx(0.61)


@pytest.mark.parametrize(
    ("instruction", "expected_x1"),
    (
        ("黑色箱子上左边的白色充电器", 100 / 640),
        ("黑色箱子上右边的白色充电器", 400 / 640),
        ("黑色箱子上的白色充电器", 400 / 640),
    ),
)
def test_ground_applies_the_qualifier_end_to_end_once_enabled(
    spatial_reorder_enabled, instruction, expected_x1
):
    response = _ground_two_instances(instruction)

    assert response["prompt"] == "white charger"
    assert response["target"]["bbox_xyxy"][0] == pytest.approx(expected_x1)
