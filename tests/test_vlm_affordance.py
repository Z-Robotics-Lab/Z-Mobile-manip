import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import pytest

from z_manip.perception.vlm_affordance import (
    _curl_transport,
    _terminate_process_group,
    OpenRouterVLM,
    VLMCancellationError,
    VLMError,
    VLMTransportError,
)


def _response(content):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


def test_local_grounding_fast_path_returns_without_remote_provider():
    remote_calls = []
    events = []

    def local_transport(url, payload, headers, timeout_s, _cancel_event):
        assert url == "http://127.0.0.1:8771/ground"
        assert payload["instruction"] == "白色充电器"
        assert payload["image_base64"]
        assert headers == {}
        assert timeout_s == pytest.approx(1.25)
        return {
            "schema": "z_manip.local_grounding_response.v1",
            "model": "local/IDEA-Research/grounding-dino-tiny",
            "target": {
                "label": "white charger",
                "bbox_xyxy": [0.55, 0.49, 0.65, 0.71],
                "confidence": 0.56,
            },
        }

    result = OpenRouterVLM(
        api_key="",
        models=("qwen/remote",),
        local_grounding_url="http://127.0.0.1:8771",
        local_transport=local_transport,
        transport=lambda *_args: remote_calls.append(True),
        attempt_callback=events.append,
    ).locate_and_reason(b"jpeg-data", "白色充电器")

    assert result.model == "local/IDEA-Research/grounding-dino-tiny"
    assert result.target_label == "white charger"
    assert result.target_bbox.to_pixels(640, 480) == (352, 235, 416, 341)
    assert remote_calls == []
    assert [(event.model, event.outcome) for event in events] == [
        ("local/grounding-dino-tiny", "start"),
        ("local/grounding-dino-tiny", "success"),
    ]


def test_local_grounding_failure_falls_back_to_remote_vlm():
    remote_calls = []
    events = []

    def local_transport(*_args):
        raise VLMTransportError("local detector has no qualified box", retryable=False)

    def remote_transport(_url, payload, _headers, _timeout_s, _cancel_event):
        remote_calls.append(payload["model"])
        return _response({
            "target": {
                "label": "charger",
                "bbox_xyxy": [0.55, 0.49, 0.65, 0.71],
                "confidence": 0.8,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/remote",),
        local_grounding_url="http://127.0.0.1:8771",
        local_transport=local_transport,
        transport=remote_transport,
        attempt_callback=events.append,
    ).locate_and_reason(b"jpeg-data", "unknown target")

    assert result.model == "qwen/remote"
    assert remote_calls == ["qwen/remote"]
    assert [(event.model, event.outcome) for event in events] == [
        ("local/grounding-dino-tiny", "start"),
        ("local/grounding-dino-tiny", "fallback"),
        ("qwen/remote", "start"),
        ("qwen/remote", "success"),
    ]


def test_grasp_grounding_rejects_partial_target_at_image_border():
    def local_transport(_url, _payload, _headers, _timeout, _cancel_event):
        return {
            "schema": "z_manip.local_grounding_response.v1",
            "model": "local/IDEA-Research/grounding-dino-tiny",
            "target": {
                "label": "a charger",
                "bbox_xyxy": [0.67, 0.29, 0.999, 0.69],
                "confidence": 0.99,
            },
        }

    vlm = OpenRouterVLM(
        api_key="",
        models=("unused",),
        local_grounding_url="http://127.0.0.1:8771",
        local_transport=local_transport,
    )

    with pytest.raises(VLMError, match="image border"):
        vlm.locate_and_reason(b"jpeg", "charger")


def test_local_grounding_url_must_be_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        OpenRouterVLM(
            api_key="key",
            models=("qwen/remote",),
            local_grounding_url="http://192.168.3.8:8771",
        )


def test_openrouter_vlm_uses_schema_and_normalizes_grounding_result():
    requests = []

    def transport(url, payload, headers, timeout_s, _cancel_event):
        requests.append((url, payload, headers, timeout_s))
        return _response({
            "target": {
                "label": "red mug",
                "bbox_xyxy": [0.1, 0.2, 0.4, 0.8],
                "confidence": 0.91,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": {
                "label": "empty shelf area",
                "bbox_xyxy": [0.55, 0.2, 0.85, 0.55],
            },
            "placement_avoid_regions": [
                {"label": "occupied", "bbox_xyxy": [0.7, 0.3, 0.8, 0.5]},
            ],
            "placement_verification": None,
            "constraints": ["keep clear of the occupied area"],
        })

    client = OpenRouterVLM(
        api_key="not-a-real-key",
        models=("qwen/test-vl",),
        transport=transport,
    )
    result = client.locate_and_reason(
        b"jpeg-data",
        "place the red mug on the shelf",
        grounding_scope="place_support",
    )

    assert result.target_label == "red mug"
    assert result.target_bbox.to_pixels(640, 480) == (64, 96, 256, 384)
    assert result.grasp_part_label is None
    assert result.preferred_approach_camera is None
    assert result.placement_region_label == "empty shelf area"
    assert result.placement_region_bbox.to_pixels(640, 480) == (352, 96, 544, 264)
    assert result.placement_verification is None
    assert result.model == "qwen/test-vl"
    url, payload, headers, timeout_s = requests[0]
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer not-a-real-key"
    assert payload["max_completion_tokens"] == 256
    assert payload["reasoning"] == {"effort": "none", "exclude": True}
    assert payload["response_format"]["type"] == "json_schema"
    target_items = payload["response_format"]["json_schema"]["schema"][
        "properties"
    ]["target"]["properties"]["bbox_xyxy"]["items"]
    assert target_items["minimum"] == 0.0
    assert target_items["maximum"] == 1.0
    properties = payload["response_format"]["json_schema"]["schema"]["properties"]
    assert properties["grasp_part"] == {"type": "null"}
    assert properties["avoid_regions"]["maxItems"] == 0
    assert properties["preferred_approach_camera"] == {"type": "null"}
    assert properties["placement_verification"] == {"type": "null"}
    assert "target bbox must continue to identify the visible grasped object" in (
        payload["messages"][0]["content"]
    )
    assert "entire visible physical object" in payload["messages"][0]["content"]
    assert "never only a grasp part" in payload["messages"][0]["content"]
    image_part = payload["messages"][1]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert timeout_s > 0.0


def test_grasp_for_place_schema_defers_support_geometry_but_keeps_axes():
    requests = []

    def transport(_url, payload, _headers, _timeout_s, _cancel_event):
        requests.append(payload)
        return _response({
            "target": {
                "label": "mustard bottle",
                "bbox_xyxy": [0.4, 0.2, 0.5, 0.8],
                "confidence": 0.9,
            },
            "grasp_part": {
                "label": "bottle body",
                "bbox_xyxy": [0.41, 0.3, 0.49, 0.7],
            },
            "avoid_regions": [],
            "preferred_approach_camera": [0.0, 0.0, 1.0],
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": {
                "require_upright": True,
                "upright_axis": "principal_long",
                "orientation_symmetry": "axial",
                "symmetry_axis": "principal_long",
            },
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/test-vl",),
        transport=transport,
    ).locate_and_reason(
        b"image",
        "pick the mustard bottle and place it upright",
        grounding_scope="grasp_for_place",
    )

    assert result.placement_region_bbox is None
    assert result.placement_avoid_regions == ()
    assert result.placement_verification.require_upright
    properties = requests[0]["response_format"]["json_schema"]["schema"][
        "properties"
    ]
    assert properties["placement_region"] == {"type": "null"}
    assert properties["placement_avoid_regions"]["maxItems"] == 0
    assert properties["placement_verification"]["type"] == "object"


def test_grasp_for_place_fails_closed_without_object_axis_verification():
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test-vl",),
        transport=lambda *_args: _response({
            "target": {
                "label": "mustard bottle",
                "bbox_xyxy": [0.4, 0.2, 0.5, 0.8],
                "confidence": 0.9,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        }),
    )

    with pytest.raises(VLMError, match="requires explicit placement_verification"):
        client.locate_and_reason(
            b"image",
            "pick and later place the mustard bottle upright",
            grounding_scope="grasp_for_place",
        )


def test_model_native_bbox_coordinate_space_is_explicitly_normalized():
    requests = []

    def transport(_url, payload, _headers, _timeout_s, _cancel_event):
        requests.append(payload)
        return _response({
            "target": {
                "label": "mustard bottle",
                "bbox_xyxy": [400, 200, 500, 800],
                "confidence": 0.9,
            },
            "grasp_part": {
                "label": "bottle body",
                "bbox_xyxy": [410, 300, 490, 700],
            },
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/test-vl",),
        model_bbox_coordinate_spaces=("relative_0_1000",),
        transport=transport,
    ).locate_and_reason(b"image", "pick the mustard bottle")

    assert result.target_bbox.to_pixels(640, 480) == (256, 96, 320, 384)
    assert result.grasp_part_bbox.to_pixels(640, 480) == (262, 144, 314, 336)
    payload = requests[0]
    assert "integer relative xyxy in [0,1000]" in payload[
        "messages"
    ][0]["content"]
    target_items = payload["response_format"]["json_schema"]["schema"][
        "properties"
    ]["target"]["properties"]["bbox_xyxy"]["items"]
    assert target_items["minimum"] == 0.0
    assert target_items["maximum"] == 1000.0
    assert target_items["type"] == "integer"


def test_native_bbox_contract_rejects_mixed_normalized_coordinates():
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test-vl",),
        model_bbox_coordinate_spaces=("relative_0_1000",),
        transport=lambda *_args: _response({
            "target": {
                "label": "mustard bottle",
                "bbox_xyxy": [400, 200, 500, 800],
                "confidence": 0.9,
            },
            "grasp_part": {
                "label": "mixed-scale body",
                "bbox_xyxy": [0.41, 0.3, 0.49, 0.7],
            },
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        }),
    )

    with pytest.raises(
        VLMError,
        match=r"grasp_part\.bbox_xyxy:.*finite integers",
    ):
        client.locate_and_reason(b"image", "pick the mustard bottle")


@pytest.mark.parametrize(
    "coordinate_spaces",
    (("normalized_0_1",), ("normalized_0_1", "extra"), ("pixels", "pixels")),
)
def test_model_bbox_coordinate_spaces_must_match_models(coordinate_spaces):
    with pytest.raises(ValueError, match="model_bbox_coordinate_spaces"):
        OpenRouterVLM(
            api_key="key",
            models=("qwen/primary", "qwen/fallback"),
            model_bbox_coordinate_spaces=coordinate_spaces,
        )


def test_openrouter_vlm_falls_back_to_next_model():
    attempted = []

    def transport(_url, payload, _headers, _timeout_s, _cancel_event):
        attempted.append(payload["model"])
        if len(attempted) == 1:
            raise TimeoutError("provider timed out")
        return _response({
            "target": {"label": "box", "bbox_xyxy": [0.2, 0.2, 0.8, 0.8], "confidence": 0.8},
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/primary", "qwen/fallback"),
        transport=transport,
    ).locate_and_reason(b"image", "pick the box")

    assert attempted == ["qwen/primary", "qwen/fallback"]
    assert result.model == "qwen/fallback"


def test_openrouter_vlm_uses_model_specific_timeouts_and_reports_attempts():
    requests = []
    events = []

    def transport(_url, payload, _headers, timeout_s, _cancel_event):
        requests.append((payload["model"], timeout_s))
        if payload["model"] == "qwen/primary":
            raise TimeoutError("primary exceeded its provider budget")
        return _response({
            "target": {
                "label": "mustard bottle",
                "bbox_xyxy": [0.4, 0.3, 0.5, 0.7],
                "confidence": 0.9,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/primary", "qwen/fallback"),
        timeout_s=25.0,
        model_timeouts_s=(40.0, 33.0),
        attempt_callback=events.append,
        transport=transport,
    ).locate_and_reason(b"image", "pick the mustard bottle")

    assert result.model == "qwen/fallback"
    assert requests == [("qwen/primary", 40.0), ("qwen/fallback", 33.0)]
    assert [(event.model, event.outcome) for event in events] == [
        ("qwen/primary", "start"),
        ("qwen/primary", "timeout"),
        ("qwen/fallback", "start"),
        ("qwen/fallback", "success"),
    ]
    assert all("key" not in event.detail for event in events)


def test_openrouter_vlm_retries_one_typed_transient_provider_failure():
    attempted = []
    events = []

    def transport(_url, payload, _headers, _timeout_s, _cancel_event):
        attempted.append(payload["model"])
        if len(attempted) == 1:
            raise VLMTransportError("temporary TLS connect failure", retryable=True)
        return _response({
            "target": {
                "label": "mustard bottle",
                "bbox_xyxy": [0.4, 0.3, 0.5, 0.7],
                "confidence": 0.9,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/primary", "qwen/fallback"),
        provider_retries=1,
        attempt_callback=events.append,
        transport=transport,
    ).locate_and_reason(b"image", "pick the mustard bottle")

    assert result.model == "qwen/primary"
    assert attempted == ["qwen/primary", "qwen/primary"]
    assert [(event.attempt, event.outcome) for event in events] == [
        (1, "start"),
        (1, "provider_error"),
        (2, "start"),
        (2, "success"),
    ]


def test_openrouter_vlm_retries_one_timeout_on_the_same_fast_model():
    attempted = []
    events = []

    def transport(_url, payload, _headers, _timeout_s, _cancel_event):
        attempted.append(payload["model"])
        if len(attempted) == 1:
            raise TimeoutError("provider queue exceeded the realtime budget")
        return _response({
            "target": {
                "label": "charger",
                "bbox_xyxy": [0.4, 0.3, 0.5, 0.7],
                "confidence": 0.9,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/fast",),
        timeout_retries=1,
        attempt_callback=events.append,
        transport=transport,
    ).locate_and_reason(b"image", "pick the charger")

    assert result.model == "qwen/fast"
    assert attempted == ["qwen/fast", "qwen/fast"]
    assert [(event.attempt, event.outcome) for event in events] == [
        (1, "start"),
        (1, "timeout"),
        (2, "start"),
        (2, "success"),
    ]


def test_openrouter_vlm_hedge_returns_the_first_success_and_cancels_loser():
    calls = []
    first_canceled = threading.Event()

    def transport(_url, _payload, _headers, _timeout_s, cancel_event):
        index = len(calls)
        calls.append(index)
        if index == 0:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline and not cancel_event.is_set():
                time.sleep(0.005)
            if cancel_event.is_set():
                first_canceled.set()
                raise VLMCancellationError("hedge loser canceled")
            raise TimeoutError("slow primary")
        return _response({
            "target": {
                "label": "charger",
                "bbox_xyxy": [0.4, 0.3, 0.5, 0.7],
                "confidence": 0.9,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    started = time.monotonic()
    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/fast",),
        hedge_delay_s=0.02,
        transport=transport,
    ).locate_and_reason(b"image", "pick the charger")

    assert result.model == "qwen/fast"
    assert calls == [0, 1]
    assert first_canceled.wait(timeout=0.2)
    assert time.monotonic() - started < 0.2


@pytest.mark.parametrize(
    "model_timeouts",
    ((40.0,), (40.0, 0.0), (40.0, float("nan"))),
)
def test_openrouter_vlm_rejects_invalid_model_timeout_contract(model_timeouts):
    with pytest.raises(ValueError, match="model.*timeout"):
        OpenRouterVLM(
            api_key="key",
            models=("qwen/primary", "qwen/fallback"),
            model_timeouts_s=model_timeouts,
        )


@pytest.mark.parametrize("provider_retries", (-1, 4, True, "bad"))
def test_openrouter_vlm_rejects_invalid_provider_retry_contract(provider_retries):
    with pytest.raises(ValueError, match="provider_retries"):
        OpenRouterVLM(
            api_key="key",
            models=("qwen/primary",),
            provider_retries=provider_retries,
        )


@pytest.mark.parametrize("timeout_retries", (-1, 4, True, "bad"))
def test_openrouter_vlm_rejects_invalid_timeout_retry_contract(timeout_retries):
    with pytest.raises(ValueError, match="timeout_retries"):
        OpenRouterVLM(
            api_key="key",
            models=("qwen/primary",),
            timeout_retries=timeout_retries,
        )


@pytest.mark.parametrize("hedge_delay_s", (-1.0, 5.1, float("nan")))
def test_openrouter_vlm_rejects_invalid_hedge_delay(hedge_delay_s):
    with pytest.raises(ValueError, match="hedge_delay_s"):
        OpenRouterVLM(
            api_key="key",
            models=("qwen/primary",),
            hedge_delay_s=hedge_delay_s,
        )


def test_openrouter_vlm_rejects_grasp_region_covered_by_avoid_union():
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test",),
        transport=lambda *_args: _response({
            "target": {
                "label": "bottle",
                "bbox_xyxy": [0.2, 0.2, 0.8, 0.8],
                "confidence": 0.9,
            },
            "grasp_part": {
                "label": "body",
                "bbox_xyxy": [0.3, 0.3, 0.5, 0.7],
            },
            "avoid_regions": [
                {"label": "left unsafe", "bbox_xyxy": [0.3, 0.3, 0.4, 0.7]},
                {"label": "right unsafe", "bbox_xyxy": [0.4, 0.3, 0.5, 0.7]},
            ],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        }),
    )

    with pytest.raises(VLMError, match="grasp_part is covered"):
        client.locate_and_reason(b"image", "pick the bottle")


def test_openrouter_vlm_rejects_place_region_covered_by_avoid_union():
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test",),
        transport=lambda *_args: _response({
            "target": {
                "label": "box",
                "bbox_xyxy": [0.1, 0.2, 0.3, 0.5],
                "confidence": 0.9,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": {
                "label": "empty support",
                "bbox_xyxy": [0.5, 0.3, 0.8, 0.7],
            },
            "placement_avoid_regions": [
                {"label": "occupied upper", "bbox_xyxy": [0.5, 0.3, 0.8, 0.5]},
                {"label": "unsupported lower", "bbox_xyxy": [0.5, 0.5, 0.8, 0.7]},
            ],
            "placement_verification": None,
            "constraints": [],
        }),
    )

    with pytest.raises(VLMError, match="placement_region is covered"):
        client.locate_and_reason(
            b"image",
            "place the box on the shelf",
            grounding_scope="place_support",
        )


def test_openrouter_vlm_supports_legacy_four_argument_transport():
    requests = []

    def transport(url, payload, headers, timeout_s):
        requests.append((url, payload["model"], headers, timeout_s))
        return _response({
            "target": {
                "label": "legacy box",
                "bbox_xyxy": [0.2, 0.2, 0.8, 0.8],
                "confidence": 0.8,
            },
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        })

    result = OpenRouterVLM(
        api_key="key",
        models=("qwen/legacy",),
        transport=transport,
    ).locate_and_reason(b"image", "pick the box")

    assert result.target_label == "legacy box"
    assert len(requests) == 1
    assert requests[0][1] == "qwen/legacy"


def test_transport_body_type_error_is_not_retried_with_legacy_signature():
    argument_counts = []

    def transport(*args):
        argument_counts.append(len(args))
        raise TypeError("transport implementation failed")

    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test",),
        transport=transport,
    )

    with pytest.raises(VLMError, match="transport implementation failed"):
        client.locate_and_reason(b"image", "pick")

    assert argument_counts == [5]


def test_openrouter_vlm_rejects_invalid_or_missing_grounding():
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test",),
        transport=lambda *_args: _response({
            "target": {"label": "bad", "bbox_xyxy": [0.8, 0.2, 0.1, 0.9], "confidence": 0.5},
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        }),
    )
    with pytest.raises(VLMError, match="all VLM models failed"):
        client.locate_and_reason(b"image", "pick")


@pytest.mark.parametrize(
    "target",
    [
        {"label": "not visible", "bbox_xyxy": [0.0, 0.0, 1.0, 1.0], "confidence": 0.0},
        {"label": "uncertain", "bbox_xyxy": [0.1, 0.1, 0.4, 0.4], "confidence": 0.1},
    ],
)
def test_openrouter_vlm_rejects_sentinel_or_low_confidence_grounding(target):
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test",),
        transport=lambda *_args: _response({
            "target": target,
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        }),
    )
    with pytest.raises(VLMError, match="confidence"):
        client.locate_and_reason(b"image", "pick")


def test_openrouter_vlm_rejects_nearly_full_target_box():
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test",),
        transport=lambda *_args: _response({
            "target": {"label": "wall", "bbox_xyxy": [0.0, 0.0, 0.98, 1.0], "confidence": 0.9},
            "grasp_part": None,
            "avoid_regions": [],
            "preferred_approach_camera": None,
            "placement_region": None,
            "placement_avoid_regions": [],
            "placement_verification": None,
            "constraints": [],
        }),
    )
    with pytest.raises(VLMError, match="area"):
        client.locate_and_reason(b"image", "pick")


def test_openrouter_vlm_requires_key_without_leaking_it():
    client = OpenRouterVLM(api_key="", models=("qwen/test",))
    with pytest.raises(VLMError, match="OPENROUTER_API_KEY"):
        client.locate_and_reason(b"image", "pick")


def test_cancellation_prevents_validation_retry_and_model_fallback(monkeypatch):
    attempted = []
    cancel_event = threading.Event()

    def transport(_url, payload, _headers, _timeout_s, call_cancel_event):
        attempted.append(payload["model"])
        call_cancel_event.set()
        return _response({"target": {"bbox_xyxy": [0, 0, 0, 0]}})

    monkeypatch.setenv("Z_MANIP_VLM_VALIDATION_RETRIES", "2")
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/primary", "qwen/fallback"),
        transport=transport,
    )

    with pytest.raises(VLMCancellationError, match="canceled"):
        client.locate_and_reason(b"image", "pick", cancel_event=cancel_event)

    assert attempted == ["qwen/primary"]


def test_curl_cancellation_terminates_then_kills_its_process_group(monkeypatch):
    started = threading.Event()
    killed = threading.Event()
    popen_kwargs = {}
    signals = []
    api_key = 'openrouter-test-secret-7b3d11'

    class HungProcess:
        pid = 4321
        returncode = None

        def poll(self):
            return self.returncode

        def communicate(self, **kwargs):
            input_value = kwargs.get('input')
            timeout = kwargs.get('timeout')
            if input_value is not None:
                assert '"model":"qwen/test"' in input_value
                started.set()
            if killed.is_set():
                self.returncode = -signal.SIGKILL
                return "", ""
            raise subprocess.TimeoutExpired("curl", timeout)

    process = HungProcess()

    def popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        assert command[-2:] == ["--data-binary", "@-"]
        assert not any('"model"' in argument for argument in command)
        assert not any(api_key in argument for argument in command)
        assert not any('Authorization' in argument for argument in command)
        header_argument = command[command.index('--header') + 1]
        header_fd = int(header_argument.rsplit('/', 1)[-1])
        assert kwargs['pass_fds'] == (header_fd,)
        assert os.read(header_fd, 4096) == (
            f'Authorization: Bearer {api_key}\n'.encode()
        )
        assert 'OPENROUTER_API_KEY' not in kwargs['env']
        assert not any(api_key in value for value in kwargs['env'].values())
        return process

    def killpg(pid, sig):
        assert pid == process.pid
        signals.append(sig)
        if sig == signal.SIGKILL:
            killed.set()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr("z_manip.perception.vlm_affordance.os.killpg", killpg)
    monkeypatch.setenv('OPENROUTER_API_KEY', api_key)
    cancel_event = threading.Event()
    outcome = []

    def invoke():
        try:
            _curl_transport(
                "https://example.invalid/chat/completions",
                {"model": "qwen/test"},
                {"Authorization": f"Bearer {api_key}"},
                5.0,
                cancel_event,
                poll_interval_s=0.01,
                terminate_grace_s=0.01,
            )
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert started.wait(timeout=0.5)
    cancel_event.set()
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], VLMCancellationError)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert popen_kwargs["start_new_session"] is True
    assert popen_kwargs["stdin"] is subprocess.PIPE


def test_process_group_cleanup_kills_descendant_after_curl_leader_exits(monkeypatch):
    signals = []

    class ExitedLeaderWithOpenDescendantPipes:
        pid = 5432
        returncode = 0

        def poll(self):
            return self.returncode

        def communicate(self, *, timeout):
            if signals and signals[-1] == signal.SIGKILL:
                return '', ''
            raise subprocess.TimeoutExpired('curl', timeout)

    monkeypatch.setattr(
        'z_manip.perception.vlm_affordance.os.killpg',
        lambda pid, sig: signals.append(sig) if pid == 5432 else None,
    )

    assert _terminate_process_group(
        ExitedLeaderWithOpenDescendantPipes(),
        grace_s=0.01,
    ) == ('', '')
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_invalid_payload_does_not_open_a_header_pipe(monkeypatch):
    opened = False

    def open_pipe(_headers):
        nonlocal opened
        opened = True
        raise AssertionError('header pipe must not open before payload serialization')

    monkeypatch.setattr(
        'z_manip.perception.vlm_affordance._open_curl_header_pipe',
        open_pipe,
    )

    with pytest.raises(TypeError, match='JSON serializable'):
        _curl_transport(
            'https://example.invalid/chat/completions',
            {'invalid': object()},
            {'Authorization': 'Bearer secret-value'},
            1.0,
            threading.Event(),
        )
    assert not opened


def test_invalid_response_cleans_descendants_after_curl_leader_exits(monkeypatch):
    signals = []

    class ExitedLeaderWithDetachedDescendant:
        pid = 6543
        returncode = 0
        calls = 0

        def poll(self):
            return self.returncode

        def communicate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return 'not-json', ''
            if signals and signals[-1] == signal.SIGKILL:
                return '', ''
            raise subprocess.TimeoutExpired('curl', kwargs.get('timeout'))

    process = ExitedLeaderWithDetachedDescendant()
    monkeypatch.setattr(subprocess, 'Popen', lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        'z_manip.perception.vlm_affordance.os.killpg',
        lambda pid, sig: signals.append(sig) if pid == process.pid else None,
    )

    with pytest.raises(json.JSONDecodeError):
        _curl_transport(
            'https://example.invalid/chat/completions',
            {'model': 'qwen/test'},
            {},
            1.0,
            threading.Event(),
            terminate_grace_s=0.01,
        )
    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_precanceled_call_never_starts_transport():
    called = False

    def transport(*_args):
        nonlocal called
        called = True
        raise AssertionError("transport must not start")

    cancel_event = threading.Event()
    cancel_event.set()
    client = OpenRouterVLM(
        api_key="key",
        models=("qwen/test",),
        transport=transport,
    )

    started = time.monotonic()
    with pytest.raises(VLMCancellationError):
        client.locate_and_reason(b"image", "pick", cancel_event=cancel_event)

    assert time.monotonic() - started < 0.1
    assert not called


def test_curl_cancellation_reaps_a_real_transport_process(tmp_path, monkeypatch):
    fake_curl = tmp_path / 'curl'
    pid_file = tmp_path / 'curl.pid'
    fake_curl.write_text(
        '#!/usr/bin/env python3\n'
        'import os\n'
        'from pathlib import Path\n'
        'import sys\n'
        'import time\n'
        'Path(os.environ["CURL_TEST_PID_FILE"]).write_text(str(os.getpid()))\n'
        'sys.stdin.buffer.read()\n'
        'time.sleep(60)\n',
    )
    fake_curl.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    monkeypatch.setenv('CURL_TEST_PID_FILE', str(pid_file))
    cancel_event = threading.Event()
    outcome = []

    def invoke():
        try:
            _curl_transport(
                'https://example.invalid/chat/completions',
                {'model': 'qwen/test'},
                {},
                5.0,
                cancel_event,
                poll_interval_s=0.01,
                terminate_grace_s=0.1,
            )
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 0.5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    cancel_event.set()
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], VLMCancellationError)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
