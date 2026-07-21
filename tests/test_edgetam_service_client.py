import base64
from copy import deepcopy
import io
import json
from urllib import error as urlerror

import numpy as np
import pytest

import z_manip.perception.edgetam_service_client as edgetam_client_module
from z_manip.perception.edgetam_service_client import (
    PROTOCOL_VERSION,
    EdgeTamProtocolError,
    EdgeTamServiceClient,
    EdgeTamServiceError,
    EdgeTamTrackingLost,
    EdgeTamTransportError,
    UrllibJsonTransport,
    decode_coco_rle,
    encode_coco_rle,
)


JPEG = b"\xff\xd8mock-jpeg\xff\xd9"


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, path, payload, timeout_s):
        self.calls.append((method, path, payload, timeout_s))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return deepcopy(response)


def _mask(offset=0):
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 2 + offset : 7 + offset] = True
    return mask


def _track_response(*, frame_seq=0, track_id="track-a", mask=None, score=0.91):
    mask = _mask() if mask is None else np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    bbox = [0, 0, 1, 1]
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {
        "protocol": PROTOCOL_VERSION,
        "status": "tracking",
        "session_id": "pick-17",
        "track_id": track_id,
        "frame_seq": frame_seq,
        "image_size": [10, 8],
        "bbox_xyxy": bbox,
        "score": score,
        "mask_rle": encode_coco_rle(mask),
    }


def test_coco_rle_round_trip_uses_column_major_bool_mask():
    mask = np.array(
        [
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    encoded = encode_coco_rle(mask)
    decoded = decode_coco_rle(encoded)

    assert encoded["size"] == [3, 4]
    assert decoded.dtype == np.bool_
    assert np.array_equal(decoded, mask)


def test_init_and_update_preserve_session_track_and_sequence():
    init_response = _track_response()
    update_response = _track_response(frame_seq=1, mask=_mask(offset=1))
    transport = FakeTransport([init_response, update_response])
    client = EdgeTamServiceClient(transport=transport)

    initial = client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")
    updated = client.update(JPEG)

    assert client.active
    assert initial.track_id == updated.track_id == "track-a"
    assert updated.frame_seq == 1
    assert np.array_equal(updated.mask, _mask(offset=1))
    assert not updated.mask.flags.writeable
    init_payload = transport.calls[0][2]
    assert init_payload["protocol"] == PROTOCOL_VERSION
    assert base64.b64decode(init_payload["image_jpeg_b64"], validate=True) == JPEG
    assert transport.calls[1][2]["frame_seq"] == 1


def test_health_and_idempotent_local_reset_use_explicit_endpoints():
    health = {
        "protocol": PROTOCOL_VERSION,
        "status": "ok",
        "model_loaded": False,
    }
    reset = {
        "protocol": PROTOCOL_VERSION,
        "status": "reset",
        "session_id": "pick-17",
    }
    transport = FakeTransport([health, _track_response(), reset])
    client = EdgeTamServiceClient(transport=transport)

    assert client.health()["model_loaded"] is False
    client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")
    client.reset()
    client.reset()

    assert not client.active
    assert [call[1] for call in transport.calls] == [
        "/health",
        "/v1/sessions/init",
        "/v1/sessions/reset",
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response.update(track_id="track-b"),
        lambda response: response.update(frame_seq=7),
        lambda response: response.update(image_size=[11, 8]),
        lambda response: response.update(bbox_xyxy=[1, 1, 4, 4]),
        lambda response: response.update(score=float("nan")),
        lambda response: response["mask_rle"].update(counts=[79]),
        lambda response: response.update(mask_rle=encode_coco_rle(np.zeros((8, 10), bool))),
    ],
    ids=[
        "track-id-jump",
        "wrong-sequence",
        "image-size-change",
        "bbox-mask-disagreement",
        "non-finite-score",
        "invalid-rle",
        "empty-mask",
    ],
)
def test_update_fails_closed_on_malformed_or_lost_tracking(mutate):
    update_response = _track_response(frame_seq=1)
    mutate(update_response)
    transport = FakeTransport([_track_response(), update_response])
    client = EdgeTamServiceClient(transport=transport)
    client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")

    with pytest.raises(EdgeTamServiceError):
        client.update(JPEG)

    assert not client.active
    with pytest.raises(EdgeTamTrackingLost, match="no active"):
        client.update(JPEG)


def test_out_of_order_caller_frame_is_rejected_before_network_and_clears_lock():
    transport = FakeTransport([_track_response()])
    client = EdgeTamServiceClient(transport=transport)
    client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")

    with pytest.raises(EdgeTamTrackingLost, match="expected 1, got 2"):
        client.update(JPEG, frame_seq=2)

    assert not client.active
    assert len(transport.calls) == 1


def test_transport_timeout_fails_closed_and_is_normalized():
    transport = FakeTransport([_track_response(), TimeoutError("deadline")])
    client = EdgeTamServiceClient(transport=transport)
    client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")

    with pytest.raises(EdgeTamTransportError, match="deadline"):
        client.update(JPEG)

    assert not client.active


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (410, "tracking_lost"),
        (410, "session_frame_limit"),
        (404, "unknown_session"),
        (409, "out_of_order"),
        (422, "image_size_changed"),
    ],
)
def test_terminal_http_session_faults_are_tracking_loss(
    monkeypatch,
    status,
    code,
):
    payload = json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "status": "error",
            "error": {"code": code, "message": "remote session ended"},
        },
    ).encode("utf-8")

    def fail_request(*_args, **_kwargs):
        raise urlerror.HTTPError(
            "http://127.0.0.1:8092/v1/sessions/update",
            status,
            "session ended",
            {},
            io.BytesIO(payload),
        )

    monkeypatch.setattr(edgetam_client_module.urlrequest, "urlopen", fail_request)
    transport = UrllibJsonTransport("http://127.0.0.1:8092")

    with pytest.raises(
        EdgeTamTrackingLost,
        match=rf"EdgeTAM HTTP {status} \({code}\)",
    ) as error:
        transport.request("POST", "/v1/sessions/update", {}, 1.0)
    assert error.value.reason_code == code


def test_nonterminal_http_fault_remains_a_transport_error(monkeypatch):
    payload = json.dumps(
        {
            "protocol": PROTOCOL_VERSION,
            "status": "error",
            "error": {
                "code": "session_capacity",
                "message": "service is at capacity",
            },
        },
    ).encode("utf-8")

    def fail_request(*_args, **_kwargs):
        raise urlerror.HTTPError(
            "http://127.0.0.1:8092/v1/sessions/init",
            503,
            "unavailable",
            {},
            io.BytesIO(payload),
        )

    monkeypatch.setattr(edgetam_client_module.urlrequest, "urlopen", fail_request)
    transport = UrllibJsonTransport("http://127.0.0.1:8092")

    with pytest.raises(
        EdgeTamTransportError,
        match=r"EdgeTAM HTTP 503 \(session_capacity\)",
    ):
        transport.request("POST", "/v1/sessions/init", {}, 1.0)


def test_local_idle_timeout_fails_before_sending_a_stale_frame():
    now = [10.0]
    transport = FakeTransport([_track_response()])
    client = EdgeTamServiceClient(
        transport=transport,
        session_idle_timeout_s=0.5,
        monotonic=lambda: now[0],
    )
    client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")
    now[0] = 10.51

    with pytest.raises(EdgeTamTrackingLost, match="idle timeout"):
        client.update(JPEG)

    assert not client.active
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("jpeg", "bbox", "session", "match"),
    [
        (b"not-jpeg", [1, 1, 9, 7], "pick-17", "JPEG"),
        (JPEG, [1, 1, 1, 7], "pick-17", "non-empty"),
        (JPEG, [1, 1, 9, 7], "has spaces", "session_id"),
    ],
)
def test_invalid_init_input_never_reaches_transport(jpeg, bbox, session, match):
    transport = FakeTransport([])
    client = EdgeTamServiceClient(transport=transport)

    with pytest.raises(EdgeTamProtocolError, match=match):
        client.init(jpeg, bbox, session_id=session)

    assert transport.calls == []


def test_score_threshold_is_configurable_and_fails_closed():
    transport = FakeTransport([_track_response(score=0.49)])
    client = EdgeTamServiceClient(transport=transport, min_score=0.5)

    with pytest.raises(EdgeTamTrackingLost, match="confidence"):
        client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")

    assert not client.active


def test_declared_image_size_is_bounded_before_rle_allocation():
    response = _track_response()
    response["image_size"] = [100_000, 100_000]
    response["bbox_xyxy"] = [1, 1, 2, 2]
    response["mask_rle"] = {
        "encoding": "coco_rle",
        "size": [100_000, 100_000],
        "counts": [9_999_999_999, 1],
    }
    client = EdgeTamServiceClient(transport=FakeTransport([response]))

    with pytest.raises(EdgeTamProtocolError, match="pixel limit"):
        client.init(JPEG, [1, 1, 9, 7], session_id="pick-17")

    assert not client.active
