#!/usr/bin/env python3
"""Loopback-only, GPU-resident YOLOE service for fast 2-D/mask seeding.

The process reads pixels and text only.  It has no ROS, CAN, robot, planning, or
actuator imports. Model weights are baked into the service image; network access
is disabled by the service unit.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
import re
import threading
import time
from typing import Any, Iterable, Mapping


MODEL_ID = "yoloe-11s-seg.pt"
REQUEST_SCHEMA = "z_manip.local_grounding_request.v1"
RESPONSE_SCHEMA = "z_manip.local_grounding_response.v1"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
DEFAULT_TEXT_EMBEDDING_CACHE_SIZE = 64
# YOLOE forward resolution. 640 is the historical baked value; deployment passes
# a larger size for distant small-object recall (see --imgsz / Dockerfile CMD).
DEFAULT_IMAGE_SIZE = 640

# NOTE ON ORDER: the first entry whose Chinese key is a substring of the
# scanned segment wins, so more specific / higher-priority targets must precede
# the broader ones. In particular "充电器" (charger) contains the substring
# "电器", so the generic appliance noun MUST sort after every charger/adapter
# entry or "白色充电器" would degrade to "appliance". Tuple order alone cannot
# decide target-vs-support ("箱子" precedes "瓶子", so "箱子上的瓶子" would ground
# the box); the support split below scopes the scan to the target segment first.
_ZH_NOUNS: tuple[tuple[str, str], ...] = (
    ("电源适配器", "power adapter"),
    ("充电适配器", "charger"),
    ("充电器", "charger"),
    ("适配器", "adapter"),
    ("插头", "electrical plug"),
    ("遥控器", "remote control"),
    ("鼠标", "computer mouse"),
    ("手机", "mobile phone"),
    ("airpods", "wireless earbuds"),
    ("耳机", "headphones"),
    ("方块", "block"),
    ("积木", "block"),
    ("箱子", "box"),
    ("盒子", "box"),
    ("可乐瓶", "soda bottle"),
    ("可乐", "soda bottle"),
    ("瓶子", "bottle"),
    ("杯子", "cup"),
    ("罐子", "can"),
    ("碗", "bowl"),
    ("球", "ball"),
    # Generic appliance last: keep it below "充电器" (which contains "电器").
    ("电器", "small appliance"),
)
_ZH_COLORS: tuple[tuple[str, str], ...] = (
    ("白色", "white"),
    ("白", "white"),
    ("黑色", "black"),
    ("黑", "black"),
    ("红色", "red"),
    ("红", "red"),
    ("绿色", "green"),
    ("绿", "green"),
    ("蓝色", "blue"),
    ("蓝", "blue"),
    ("黄色", "yellow"),
    ("黄", "yellow"),
    ("紫色", "purple"),
    ("紫", "purple"),
)
_EN_COLORS: tuple[str, ...] = (
    "white", "black", "red", "green", "blue", "yellow", "purple",
)

# YOLOE's open-vocabulary score is noticeably phrase-sensitive even when two
# phrases name the same physical category.  Keep a deliberately small alias
# set for categories used by the mobile manipulation stack.  All aliases are
# semantic equivalents; this is not a broad "object" fallback, so a miss still
# falls through to the VLM instead of silently changing target identity.
_EQUIVALENT_PROMPTS: Mapping[str, tuple[str, ...]] = {
    "charger": ("wall charger", "usb charger", "power adapter", "electrical plug"),
    "power adapter": ("charger", "wall charger", "usb charger"),
    "adapter": ("power adapter", "charger", "wall charger"),
    "electrical plug": ("wall charger", "usb charger", "power adapter"),
}


# A support relation names the thing the target rests on or sits inside.
# Chinese puts the target AFTER the particle ("箱子上的瓶子" names the bottle)
# while English puts it BEFORE the preposition ("the bottle on the box"), so the
# two languages have to be split in opposite directions.
_ZH_SUPPORT_PARTICLES: tuple[str, ...] = (
    "上面", "上边", "顶部", "顶上", "里面", "里边", "上", "里", "内",
)
_EN_SUPPORT_PREPOSITIONS: tuple[str, ...] = (
    "on top of", "inside", "atop", "above", "on", "in",
)
# Words that carry no object identity. A trailing English phrase built only from
# these plus a position word qualifies the target itself ("the bottle on the
# left") instead of naming a support.
_EN_POSITIONAL_FILLER: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "its", "image", "frame", "picture", "scene", "view", "side"}
)
# The operator often appends a clarifying clause; everything after the first
# separator explains what to avoid or why, and never names the target noun.
_SENTENCE_SEPARATORS: str = "，,。;；\n"

# Position qualifiers. The longest key wins at the same offset, so "左边" is
# matched ahead of the bare "左".
_SPATIAL_RELATIONS: Mapping[str, str] = {
    "左边": "left", "左侧": "left", "左面": "left", "左": "left",
    "右边": "right", "右侧": "right", "右面": "right", "右": "right",
    "中间": "middle", "中央": "middle", "正中": "middle",
    "远处": "far", "远端": "far", "远方": "far", "较远": "far", "最远": "far",
    "近处": "near", "靠近": "near", "最近": "near",
    "left": "left", "right": "right",
    "middle": "middle", "center": "middle", "centre": "middle",
    "far": "far", "distant": "far", "farthest": "far", "furthest": "far",
    "near": "near", "nearest": "near", "closest": "near",
}
SPATIAL_PREFERENCES: frozenset[str] = frozenset(
    {"left", "right", "middle", "near", "far"}
)


def _token_index(text: str, token: str) -> int:
    """Earliest index of ``token`` in ``text``, or -1.

    ASCII tokens must match a whole word so "far" never fires inside "farm" and
    "in" never fires inside "tin". CJK tokens have no word boundaries, so a
    substring match is the only available rule and is safe for this lexicon.
    """

    if not token.isascii():
        return text.find(token)
    lowered = text.lower()
    index = lowered.find(token)
    while index != -1:
        before = lowered[index - 1] if index > 0 else " "
        after = (
            lowered[index + len(token)]
            if index + len(token) < len(lowered)
            else " "
        )
        if not before.isalpha() and not after.isalpha():
            return index
        index = lowered.find(token, index + 1)
    return -1


def _earliest_token(text: str, tokens: Iterable[str]) -> tuple[int, str] | None:
    """(index, token) of the earliest match; the longest wins at equal offset."""

    best: tuple[int, int, str] | None = None
    for token in tokens:
        index = _token_index(text, token)
        if index < 0:
            continue
        candidate = (index, -len(token), token)
        if best is None or candidate < best:
            best = candidate
    return (best[0], best[2]) if best is not None else None


def _before_separator(text: str) -> str:
    offsets = [text.find(character) for character in _SENTENCE_SEPARATORS]
    index = min((offset for offset in offsets if offset >= 0), default=-1)
    return text if index < 0 else text[:index]


def _zh_noun(segment: str) -> str | None:
    return next((english for chinese, english in _ZH_NOUNS if chinese in segment), None)


def _color(segment: str) -> str | None:
    """Colour modifier of a segment, in either language.

    ``_ZH_NOUNS`` carries a few ASCII product names, so an all-English phrase
    can still resolve through the noun lexicon and needs its colour read here.
    """

    chinese = next(
        (english for chinese, english in _ZH_COLORS if chinese in segment), None
    )
    if chinese is not None:
        return chinese
    match = _earliest_token(segment, _EN_COLORS)
    return None if match is None else match[1]


def _zh_support_split(query: str) -> tuple[str, str] | None:
    """(target, support) around the earliest support particle, or None."""

    match = _earliest_token(query, _ZH_SUPPORT_PARTICLES)
    if match is None:
        return None
    index, particle = match
    support = query[:index]
    target = query[index + len(particle):]
    if not support.strip() or not target.strip():
        return None
    return target, support


def _is_positional_phrase(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    return all(
        word in _EN_POSITIONAL_FILLER or word in _SPATIAL_RELATIONS for word in words
    )


def _en_support_split(query: str) -> tuple[str, str] | None:
    """(target, support) around the earliest support preposition, or None."""

    match = _earliest_token(query, _EN_SUPPORT_PREPOSITIONS)
    if match is None:
        return None
    index, preposition = match
    target = query[:index].strip()
    support = query[index + len(preposition):].strip()
    if not target or not support:
        return None
    return target, support


def _spatial_relation(text: str) -> str | None:
    """Return the position qualifier carried by ``text``, or None."""

    match = _earliest_token(text, _SPATIAL_RELATIONS)
    return None if match is None else _SPATIAL_RELATIONS[match[1]]


def _parse_instruction(instruction: str) -> tuple[tuple[str, ...], str | None]:
    """Return (YOLOE class phrases, position qualifier) for one instruction."""

    query = " ".join(str(instruction).strip().lower().split())
    if not query:
        return (), None
    # Most specific first: the segment naming the target under an explicit
    # support relation, then the whole instruction. The first segment holding a
    # known noun decides both the noun and its colour, so a support phrase can
    # no longer supply either.
    segments: list[tuple[str, str | None]] = []
    zh_split = _zh_support_split(query)
    if zh_split is not None:
        zh_target, zh_support = zh_split
        segments.append((_before_separator(zh_target), zh_support))
        segments.append((zh_target, zh_support))
    en_split = _en_support_split(query)
    if en_split is not None and not _is_positional_phrase(en_split[1]):
        segments.append(en_split)
    segments.append((_before_separator(query), None))
    segments.append((query, None))
    target_text, support = next(
        (
            (segment, segment_support)
            for segment, segment_support in segments
            if _zh_noun(segment) is not None
        ),
        (query, None),
    )
    noun = _zh_noun(target_text)
    color = _color(target_text)
    if noun is not None:
        primary = f"{color} {noun}" if color else noun
        prompts = [primary]
        # Repeating the same noun around a support relation means the requested
        # target is the smaller item on the support, not the support itself.
        if support is not None and noun in {"box", "block"}:
            prompts.append(f"small {primary}")
        for alias in _EQUIVALENT_PROMPTS.get(noun, ()):
            candidate = f"{color} {alias}" if color else alias
            if candidate not in prompts:
                prompts.append(candidate)
        return tuple(prompts), _spatial_relation(target_text)
    if any("\u4e00" <= character <= "\u9fff" for character in query):
        return (), None
    cleaned = re.sub(r"[^a-z0-9\s_-]+", " ", query)
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(
        r"^(?:please\s+)?(?:pick(?:\s+up)?|grasp|grab|find|track|locate|approach)\s+",
        "",
        cleaned,
    )
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned)
    if not cleaned:
        return (), None
    qualifier_text = cleaned
    en_split = _en_support_split(cleaned)
    if en_split is not None:
        en_target, en_support = en_split
        if not _is_positional_phrase(en_support):
            qualifier_text = en_target
        cleaned = en_target
    return (cleaned,), _spatial_relation(qualifier_text)


def grounding_prompts(instruction: str) -> tuple[str, ...]:
    """Return a bounded set of identity-preserving YOLOE class phrases.

    The first item is the legacy prompt returned by :func:`grounding_prompt`.
    Extra items only cover strict category synonyms or, for an object described
    on top of another support, the same category with a ``small`` modifier.
    Ultralytics evaluates the whole tuple in one forward pass, avoiding a slow
    provider fallback when a semantically equivalent phrase scores better.
    """

    return _parse_instruction(instruction)[0]


def spatial_preference(instruction: str) -> str | None:
    """Return the position qualifier that disambiguates between instances."""

    return _parse_instruction(instruction)[1]


def grounding_prompt(instruction: str) -> str | None:
    """Return one concise English YOLOE class, or None for VLM fallback."""

    prompts = grounding_prompts(instruction)
    return prompts[0] if prompts else None


# A distant and/or small target occupies few pixels in the full frame, so YOLOE
# at a fixed forward resolution frequently misses it. When the instruction says
# so, a second forward pass on an upscaled centre crop recovers recall without
# changing the reported geometry (crop detections are mapped back to full-frame
# pixels and merged with the full-frame detections).
_ROI_QUALIFIER_TOKENS: tuple[str, ...] = (
    "远处", "远端", "远方", "较远", "远的", "distant", "far",
    "小型", "小的", "小白", "小黑", "小", "tiny", "small",
)


def roi_zoom_qualifier(instruction: str) -> bool:
    """True when the instruction marks the target as distant and/or small."""

    text = str(instruction)
    return any(_token_index(text, token) >= 0 for token in _ROI_QUALIFIER_TOKENS)


def center_crop_region(width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    """Central ``fraction`` sub-window as an integer (x0, y0, x1, y1) box."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("crop fraction must be within (0, 1]")
    crop_w = max(1, int(round(width * fraction)))
    crop_h = max(1, int(round(height * fraction)))
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    return x0, y0, x0 + crop_w, y0 + crop_h


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
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


def merge_detection_lists(
    boxes_a: list[list[float]],
    scores_a: list[float],
    labels_a: list[str],
    boxes_b: list[list[float]],
    scores_b: list[float],
    labels_b: list[str],
    *,
    iou_threshold: float = 0.6,
) -> tuple[list[list[float]], list[float], list[str]]:
    """Union two detection lists, dropping the weaker of any overlapping pair.

    Both lists must already be in the same (full-frame pixel) coordinate space.
    Detections from ``a`` are kept preferentially on a tie so the full-frame
    geometry wins when the crop merely rediscovers the same object.
    """

    merged_boxes = [list(box) for box in boxes_a]
    merged_scores = [float(score) for score in scores_a]
    merged_labels = [str(label) for label in labels_a]
    for box, score, label in zip(boxes_b, scores_b, labels_b):
        candidate = tuple(float(value) for value in box)
        replaced = False
        duplicate = False
        for index, existing in enumerate(merged_boxes):
            if _iou_xyxy(candidate, tuple(existing)) >= iou_threshold:
                duplicate = True
                if float(score) > merged_scores[index]:
                    merged_boxes[index] = list(box)
                    merged_scores[index] = float(score)
                    merged_labels[index] = str(label)
                    replaced = True
                break
        if not duplicate and not replaced:
            merged_boxes.append(list(box))
            merged_scores.append(float(score))
            merged_labels.append(str(label))
    return merged_boxes, merged_scores, merged_labels


# A position qualifier may only reorder detections the detector already rates
# as comparable, so an incidental relation word cannot promote a weak box.
SPATIAL_PEER_CONFIDENCE_RATIO = 0.6
# ...and only when the qualified boxes are actually separated along the axis
# the qualifier names, in normalized frame units / relative area.
SPATIAL_MINIMUM_CENTROID_SPREAD = 0.05
SPATIAL_MINIMUM_AREA_SPREAD = 1.25


def _confidence(candidate: Mapping[str, object]) -> float:
    return float(candidate["confidence"])


def _centre_x(candidate: Mapping[str, object]) -> float:
    box = candidate["bbox_xyxy"]
    return (float(box[0]) + float(box[2])) / 2.0


def _apply_spatial_preference(
    candidates: list[dict[str, object]],
    preference: str,
) -> dict[str, object] | None:
    """Pick the instance the operator's position qualifier names, or None."""

    if len(candidates) < 2:
        return None
    if preference in {"left", "right", "middle"}:
        centres = [(_centre_x(candidate), candidate) for candidate in candidates]
        offsets = sorted(centre for centre, _ in centres)
        if offsets[-1] - offsets[0] < SPATIAL_MINIMUM_CENTROID_SPREAD:
            return None
        if preference == "left":
            return min(centres, key=lambda item: (item[0], -_confidence(item[1])))[1]
        if preference == "right":
            return max(centres, key=lambda item: (item[0], _confidence(item[1])))[1]
        middle = (offsets[(len(offsets) - 1) // 2] + offsets[len(offsets) // 2]) / 2.0
        return min(
            centres,
            key=lambda item: (abs(item[0] - middle), -_confidence(item[1])),
        )[1]
    # Apparent size is the only depth cue a single 2-D frame carries, so a
    # near/far qualifier ranks on box area.
    areas = [(float(candidate["area_ratio"]), candidate) for candidate in candidates]
    smallest = min(area for area, _ in areas)
    largest = max(area for area, _ in areas)
    if smallest <= 0.0 or largest < smallest * SPATIAL_MINIMUM_AREA_SPREAD:
        return None
    if preference == "near":
        return max(areas, key=lambda item: (item[0], _confidence(item[1])))[1]
    return min(areas, key=lambda item: (item[0], -_confidence(item[1])))[1]


def select_detection(
    boxes_xyxy: object,
    scores: object,
    labels: object,
    *,
    width: int,
    height: int,
    minimum_confidence: float,
    maximum_area_ratio: float,
    maximum_area_ratio_by_label: Mapping[str, float] | None = None,
    minimum_border_margin_ratio: float = 0.002,
    spatial_preference: str | None = None,
) -> dict[str, object] | None:
    """Select the strongest complete, finite, object-scale detection.

    A grasp seed must describe a fully observed object. Boxes clipped against
    an image edge are commonly the robot gripper, furniture, or another partial
    foreground object. Passing those boxes to the tracker creates a stable but
    geometrically meaningless mask, so reject them before confidence ranking.

    ``spatial_preference`` names which of several comparable instances the
    operator asked for. Every class phrase in one request denotes the same
    category by construction, so any qualified detection is a plausible
    instance and may be reordered by position.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum confidence must be within [0, 1]")
    if not 0.0 < maximum_area_ratio < 1.0:
        raise ValueError("maximum area ratio must be within (0, 1)")
    if not 0.0 <= minimum_border_margin_ratio < 0.5:
        raise ValueError("minimum border margin ratio must be within [0, 0.5)")
    if spatial_preference is not None and spatial_preference not in SPATIAL_PREFERENCES:
        raise ValueError("spatial preference is unsupported")
    area_limits = dict(maximum_area_ratio_by_label or {})
    if any(not 0.0 < limit < 1.0 for limit in area_limits.values()):
        raise ValueError("per-label maximum area ratios must be within (0, 1)")
    candidates: list[tuple[float, float, dict[str, object]]] = []
    for index, raw_box in enumerate(boxes_xyxy):
        try:
            box = [float(value) for value in raw_box]
            confidence = float(scores[index])
        except (IndexError, TypeError, ValueError):
            continue
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            continue
        if not math.isfinite(confidence) or confidence < minimum_confidence:
            continue
        x1, y1, x2, y2 = box
        x1 = min(float(width), max(0.0, x1))
        y1 = min(float(height), max(0.0, y1))
        x2 = min(float(width), max(0.0, x2))
        y2 = min(float(height), max(0.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        border_x = minimum_border_margin_ratio * width
        border_y = minimum_border_margin_ratio * height
        if (
            x1 <= border_x
            or y1 <= border_y
            or x2 >= width - border_x
            or y2 >= height - border_y
        ):
            continue
        label = "object"
        try:
            candidate_label = str(labels[index]).strip()
            if candidate_label:
                label = candidate_label
        except (IndexError, TypeError):
            pass
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
        label_area_limit = min(maximum_area_ratio, area_limits.get(label, maximum_area_ratio))
        if area_ratio < 0.0002 or area_ratio > label_area_limit:
            continue
        result = {
            "label": label,
            "bbox_xyxy": [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ],
            "confidence": confidence,
            "area_ratio": area_ratio,
        }
        # Confidence is primary.  A tiny area tiebreak prevents a broad support
        # surface from winning when its score is effectively identical.
        candidates.append((confidence, -area_ratio, result))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = candidates[0][2]
    if spatial_preference is None:
        return best
    peers = [
        result
        for confidence, _, result in candidates
        if confidence >= float(best["confidence"]) * SPATIAL_PEER_CONFIDENCE_RATIO
    ]
    return _apply_spatial_preference(peers, spatial_preference) or best


def _result_boxes(
    result: Any,
    prompt: str,
    names: Mapping[int, str],
) -> tuple[list[list[float]], list[float], list[str]]:
    """Extract (xyxy, scores, labels) from one Ultralytics result."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], [], []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    labels = [str(names.get(int(class_id), prompt)) for class_id in class_ids]
    return xyxy, scores, labels


class GroundingRuntime:
    """One persistent CUDA YOLOE model guarded against concurrent forwards."""

    def __init__(
        self,
        *,
        model_id: str,
        minimum_confidence: float,
        maximum_area_ratio: float,
        text_embedding_cache_size: int = DEFAULT_TEXT_EMBEDDING_CACHE_SIZE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        roi_zoom_enabled: bool = True,
        roi_zoom_fraction: float = 0.5,
    ) -> None:
        if text_embedding_cache_size < 1:
            raise ValueError("text_embedding_cache_size must be positive")
        if image_size < 32 or image_size % 32 != 0:
            raise ValueError("image_size must be a positive multiple of 32")
        if not 0.0 < roi_zoom_fraction <= 1.0:
            raise ValueError("roi_zoom_fraction must be within (0, 1]")
        self.model_id = model_id
        self.minimum_confidence = minimum_confidence
        self.maximum_area_ratio = maximum_area_ratio
        self.text_embedding_cache_size = text_embedding_cache_size
        self.image_size = image_size
        self.roi_zoom_enabled = bool(roi_zoom_enabled)
        self.roi_zoom_fraction = float(roi_zoom_fraction)
        self._lock = threading.Lock()
        self._model: Any = None
        self._device = "unloaded"
        self._classes: tuple[str, ...] = ()
        self._text_embeddings: OrderedDict[tuple[str, ...], Any] = OrderedDict()

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("local grounding requires CUDA")
        torch.backends.cuda.matmul.allow_tf32 = True
        self._device = "cuda:0"
        self._model = YOLO(self.model_id, task="segment")
        self._model.to(self._device)

    def _embedding_for(self, classes: tuple[str, ...]) -> tuple[Any, bool]:
        """Return exact text embeddings without rebuilding MobileCLIP.

        Ultralytics' public ``YOLOE.set_classes`` calls ``get_text_pe`` without
        ``cache_clip_model=True``.  That rebuilds MobileCLIP for every new
        prompt even though the detector process is persistent.  Cache both the
        encoder and its tiny exact-phrase outputs.  The key includes the full
        normalized class phrase, so this does not merge semantically distinct
        targets such as ``red bottle`` and ``black bottle``.
        """

        cached = self._text_embeddings.get(classes)
        if cached is not None:
            self._text_embeddings.move_to_end(classes)
            return cached, True
        backend = getattr(self._model, "model", None)
        encoder = getattr(backend, "get_text_pe", None)
        if encoder is not None:
            try:
                embedding = encoder(list(classes), cache_clip_model=True)
            except TypeError:
                # Compatibility with older Ultralytics builds that expose the
                # backend encoder but not the persistent-CLIP argument.
                embedding = encoder(list(classes))
        else:
            embedding = self._model.get_text_pe(list(classes))
        self._text_embeddings[classes] = embedding
        self._text_embeddings.move_to_end(classes)
        while len(self._text_embeddings) > self.text_embedding_cache_size:
            self._text_embeddings.popitem(last=False)
        return embedding, False

    def ground(self, image_bytes: bytes, instruction: str) -> dict[str, object]:
        request_started = time.perf_counter()
        prompts, preference = _parse_instruction(instruction)
        if not prompts:
            raise LookupError("instruction has no supported local noun phrase")
        prompt = prompts[0]
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        decode_finished = time.perf_counter()
        embedding_cache_hit = False
        with self._lock:
            self.load()
            requested_classes = prompts
            embedding_cache_hit = self._classes == requested_classes
            if self._classes != requested_classes:
                embedding, embedding_cache_hit = self._embedding_for(requested_classes)
                self._model.set_classes(
                    list(requested_classes),
                    embeddings=embedding,
                )
                self._classes = requested_classes
            embedding_finished = time.perf_counter()
            predict_kwargs = dict(
                device=self._device,
                imgsz=self.image_size,
                conf=min(0.20, self.minimum_confidence),
                iou=0.55,
                # YOLOE builds new text embeddings whenever the prompt changes.
                # Ultralytics' half=True path permanently converts the detection
                # head to FP16, while a later MobileCLIP embedding is FP32.  The
                # next dynamic prompt then fails inside the head with a
                # Float/Half matmul mismatch.  Keep the persistent model in FP32;
                # TF32 is enabled in load(), so RTX inference remains fast and
                # arbitrary successive prompts remain type-stable.
                half=False,
                max_det=24,
                retina_masks=False,
                verbose=False,
            )
            result = self._model.predict(source=image, **predict_kwargs)[0]
            # ROI zoom: a distant/small target occupies too few pixels for the
            # full-frame forward. Run one extra forward on an upscaled centre
            # crop and merge the crop detections (mapped back to full-frame
            # pixels) so recall improves without changing reported geometry.
            roi_origin: tuple[int, int] | None = None
            roi_result = None
            roi_used = (
                self.roi_zoom_enabled and roi_zoom_qualifier(instruction)
            )
            if roi_used:
                x0, y0, x1, y1 = center_crop_region(
                    width, height, self.roi_zoom_fraction
                )
                roi_origin = (x0, y0)
                roi_result = self._model.predict(
                    source=image.crop((x0, y0, x1, y1)), **predict_kwargs
                )[0]
        inference_finished = time.perf_counter()
        names = getattr(result, "names", {0: prompt})
        xyxy, scores, labels = _result_boxes(result, prompt, names)
        if roi_result is not None and roi_origin is not None:
            roi_xyxy, roi_scores, roi_labels = _result_boxes(
                roi_result, prompt, getattr(roi_result, "names", names)
            )
            offset_x, offset_y = roi_origin
            roi_xyxy = [
                [box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y]
                for box in roi_xyxy
            ]
            xyxy, scores, labels = merge_detection_lists(
                xyxy, scores, labels, roi_xyxy, roi_scores, roi_labels,
            )
        selected = select_detection(
            xyxy,
            scores,
            labels,
            width=width,
            height=height,
            minimum_confidence=self.minimum_confidence,
            maximum_area_ratio=self.maximum_area_ratio,
            maximum_area_ratio_by_label={
                candidate: min(self.maximum_area_ratio, 0.12)
                for candidate in prompts
                if candidate.startswith("small ")
            },
            spatial_preference=preference,
        )
        if selected is None:
            raise LookupError("local detector produced no qualified object box")
        # Aliases only improve detector recall.  Preserve the canonical legacy
        # target label so downstream identity and artifact semantics do not
        # depend on which equivalent phrase happened to score highest.
        selected["label"] = prompt
        finished = time.perf_counter()
        return {
            "schema": RESPONSE_SCHEMA,
            "model": f"local/yoloe/{os.path.basename(self.model_id)}",
            "prompt": prompt,
            "spatial_preference": preference,
            "target": selected,
            "latency_s": finished - request_started,
            "embedding_cache_hit": embedding_cache_hit,
            "roi_zoom_used": roi_used,
            "timings_s": {
                "decode": decode_finished - request_started,
                "prompt_embedding": embedding_finished - decode_finished,
                "inference": inference_finished - embedding_finished,
                "postprocess": finished - inference_finished,
                "total": finished - request_started,
            },
        }

    def warmup(self) -> None:
        """Run one synthetic forward so the first camera request stays fast."""

        from PIL import Image

        image = Image.new("RGB", (640, 480), color=(127, 127, 127))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG")
        # The first class initializes MobileCLIP and the first dynamic class
        # switch initializes the auxiliary prompt head.  Absorb both costs at
        # service startup so the first real, possibly different target does
        # not pay either one.
        for prompt in ("bottle", "red bottle"):
            try:
                self.ground(encoded.getvalue(), prompt)
            except LookupError:
                # A blank image should normally have no qualified detection;
                # CUDA kernels and prompt caches are warm regardless.
                pass

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device


class GroundingServer(ThreadingHTTPServer):
    runtime: GroundingRuntime


class RequestHandler(BaseHTTPRequestHandler):
    server: GroundingServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)

    def _json(self, status: HTTPStatus, document: Mapping[str, object]) -> None:
        payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._json(HTTPStatus.OK, {
            "schema": "z_manip.local_grounding_health.v1",
            "ready": self.server.runtime.loaded,
            "backend": "yoloe",
            "model": self.server.runtime.model_id,
            "device": self.server.runtime.device,
            "read_only": True,
            "motion_commands_published": 0,
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ground":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= content_length <= MAX_REQUEST_BYTES:
                raise ValueError("request size is invalid")
            document = json.loads(self.rfile.read(content_length))
            if not isinstance(document, Mapping) or document.get("schema") != REQUEST_SCHEMA:
                raise ValueError("request schema is invalid")
            instruction = str(document.get("instruction", "")).strip()
            encoded = document.get("image_base64")
            if not instruction or not isinstance(encoded, str):
                raise ValueError("instruction and image_base64 are required")
            image_bytes = base64.b64decode(encoded, validate=True)
            if not image_bytes:
                raise ValueError("decoded image is empty")
            response = self.server.runtime.ground(image_bytes, instruction)
        except LookupError as error:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {
                "schema": RESPONSE_SCHEMA,
                "error": str(error),
                "fallback_recommended": True,
            })
            return
        except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": f"{type(error).__name__}: {error}",
                "fallback_recommended": True,
            })
            return
        self._json(HTTPStatus.OK, response)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--minimum-confidence", type=float, default=0.35)
    parser.add_argument("--maximum-area-ratio", type=float, default=0.45)
    parser.add_argument(
        "--text-embedding-cache-size",
        type=int,
        default=DEFAULT_TEXT_EMBEDDING_CACHE_SIZE,
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--roi-zoom",
        dest="roi_zoom",
        action="store_true",
        default=True,
        help="run a second forward on an upscaled centre crop for distant/small targets",
    )
    parser.add_argument(
        "--no-roi-zoom",
        dest="roi_zoom",
        action="store_false",
        help="disable the ROI-zoom second forward pass",
    )
    parser.add_argument("--roi-zoom-fraction", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.host not in {"127.0.0.1", "::1"}:
        raise ValueError("local grounding service may bind only to loopback")
    runtime = GroundingRuntime(
        model_id=args.model,
        minimum_confidence=args.minimum_confidence,
        maximum_area_ratio=args.maximum_area_ratio,
        text_embedding_cache_size=args.text_embedding_cache_size,
        image_size=args.imgsz,
        roi_zoom_enabled=args.roi_zoom,
        roi_zoom_fraction=args.roi_zoom_fraction,
    )
    runtime.load()
    runtime.warmup()
    server = GroundingServer((args.host, args.port), RequestHandler)
    server.runtime = runtime
    print(
        f"local grounding ready on {args.host}:{args.port} model={args.model} "
        f"imgsz={runtime.image_size} device={runtime.device}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
