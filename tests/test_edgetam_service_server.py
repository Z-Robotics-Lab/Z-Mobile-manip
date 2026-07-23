"""Bounded-state and failure-cleanup tests for the standalone EdgeTAM service."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


_SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "docker"
    / "edgetam_service"
    / "server.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "z_manip_edgetam_service_server",
    _SERVER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = server
_SPEC.loader.exec_module(server)


def _state(frame_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        processed_frames={index: object() for index in range(frame_count)},
        output_dict_per_obj={
            0: {
                "cond_frame_outputs": {0: {"conditioning": True}},
                "non_cond_frame_outputs": {
                    index: object() for index in range(1, frame_count)
                },
            },
        },
        frames_tracked_per_obj={
            0: {index: object() for index in range(frame_count)},
        },
    )


def test_stream_pruning_keeps_model_required_window_and_conditioning() -> None:
    backend = server.EdgeTamBackend(
        server.ServiceConfig(stream_history_frames=4),
    )
    backend._model = SimpleNamespace(
        config=SimpleNamespace(
            num_maskmem=7,
            max_object_pointers_in_encoder=16,
        ),
    )
    state = _state(41)

    backend._prune_streaming_state(state, frame_seq=40)

    expected = set(range(25, 41))
    assert set(state.processed_frames) == expected
    assert set(
        state.output_dict_per_obj[0]["non_cond_frame_outputs"],
    ) == expected
    assert set(state.frames_tracked_per_obj[0]) == expected
    assert state.output_dict_per_obj[0]["cond_frame_outputs"] == {
        0: {"conditioning": True},
    }


def test_stream_pruning_honors_larger_operator_limit() -> None:
    backend = server.EdgeTamBackend(
        server.ServiceConfig(stream_history_frames=32),
    )
    backend._model = SimpleNamespace(
        config=SimpleNamespace(
            num_maskmem=7,
            max_object_pointers_in_encoder=16,
        ),
    )
    state = _state(41)

    backend._prune_streaming_state(state, frame_seq=40)

    assert set(state.processed_frames) == set(range(9, 41))


def test_unexpected_backend_failure_drops_and_resets_session(monkeypatch) -> None:
    state = object()

    class BrokenBackend:
        loaded = True

        def update(self, *_args, **_kwargs):
            raise RuntimeError("synthetic CUDA fault")

        def reset(self, value):
            assert value is state
            resets.append(value)

    resets: list[object] = []
    application = server.EdgeTamApplication(
        server.ServiceConfig(),
        backend=BrokenBackend(),
    )
    session = server.TrackingSession(
        session_id="session-1",
        track_id="track-1",
        inference_state=state,
        image_size=(8, 8),
        last_frame_seq=0,
        last_access_s=server.time.monotonic(),
    )
    application._sessions[session.session_id] = session
    monkeypatch.setattr(
        server,
        "decode_jpeg",
        lambda _value, _config: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="synthetic CUDA fault"):
        application.update(
            {
                "protocol": server.PROTOCOL_VERSION,
                "session_id": session.session_id,
                "frame_seq": 1,
                "image_jpeg_b64": "unused",
            },
        )

    assert application._sessions == {}
    assert session.closed
    assert resets == [state]


def _reference_coco_rle(mask: np.ndarray) -> list[int]:
    """Original per-pixel RLE, kept here as the equivalence oracle."""
    flat = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    counts: list[int] = []
    current = False
    run = 0
    for pixel in flat:
        value = bool(pixel)
        if value == current:
            run += 1
        else:
            counts.append(run)
            current = value
            run = 1
    counts.append(run)
    return counts


def test_vectorized_coco_rle_matches_reference_and_round_trips() -> None:
    rng = np.random.default_rng(20260723)
    cases = [
        np.zeros((5, 7), dtype=bool),
        np.ones((5, 7), dtype=bool),
        np.array([[True]], dtype=bool),
        np.array([[False]], dtype=bool),
    ]
    for _ in range(40):
        h = int(rng.integers(1, 9))
        w = int(rng.integers(1, 9))
        cases.append(rng.integers(0, 2, size=(h, w)).astype(bool))
    for mask in cases:
        encoded = server.encode_coco_rle(mask)
        assert encoded["counts"] == _reference_coco_rle(mask)
        assert encoded["size"] == [int(mask.shape[0]), int(mask.shape[1])]
        assert sum(encoded["counts"]) == mask.size


class _ScriptedBackend:
    """Backend that replays a fixed track/coast script per update call."""

    loaded = True

    def __init__(self, script: list[object], *, image_size: tuple[int, int]) -> None:
        self._script = list(script)
        self._image_size = image_size
        self.calls = 0
        self.resets: list[object] = []

    def update(self, _state, _image, _frame_seq):
        step = self._script[self.calls]
        self.calls += 1
        if step == "coast":
            raise server.TrackingFailure("empty or low-score mask")
        width, height = self._image_size
        mask = np.ones((height, width), dtype=bool)
        return mask, 0.9

    def reset(self, state):
        self.resets.append(state)


def _seed_session(application, *, image_size=(8, 8)):
    state = object()
    session = server.TrackingSession(
        session_id="pick-1",
        track_id="track-1",
        inference_state=state,
        image_size=image_size,
        last_frame_seq=0,
        last_access_s=server.time.monotonic(),
    )
    application._sessions[session.session_id] = session
    return session


def _update(application, frame_seq):
    return application.update(
        {
            "protocol": server.PROTOCOL_VERSION,
            "session_id": "pick-1",
            "frame_seq": frame_seq,
            "image_jpeg_b64": "unused",
        },
    )


def test_empty_mask_coasts_and_preserves_session(monkeypatch) -> None:
    monkeypatch.setattr(
        server, "decode_jpeg", lambda _v, _c: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    backend = _ScriptedBackend(["coast", "coast"], image_size=(8, 8))
    application = server.EdgeTamApplication(
        server.ServiceConfig(max_coast_frames=2),
        backend=backend,
    )
    session = _seed_session(application)

    response = _update(application, 1)

    assert response["status"] == "coasting"
    assert response["session_id"] == "pick-1"
    assert response["track_id"] == "track-1"
    assert response["frame_seq"] == 1
    assert response["image_size"] == [8, 8]
    assert "mask_rle" not in response and "score" not in response
    # The session and its inference state survive for an instant relock.
    assert application._sessions == {session.session_id: session}
    assert session.coast_frames == 1
    assert backend.resets == []


def test_coast_exhaustion_drops_and_resets_session(monkeypatch) -> None:
    monkeypatch.setattr(
        server, "decode_jpeg", lambda _v, _c: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    backend = _ScriptedBackend(["coast", "coast"], image_size=(8, 8))
    application = server.EdgeTamApplication(
        server.ServiceConfig(max_coast_frames=1),
        backend=backend,
    )
    session = _seed_session(application)

    assert _update(application, 1)["status"] == "coasting"
    with pytest.raises(server.ServiceFault) as fault:
        _update(application, 2)

    assert fault.value.code == "tracking_lost"
    assert application._sessions == {}
    assert session.closed
    assert backend.resets == [session.inference_state]


def test_tracking_frame_resets_coast_counter(monkeypatch) -> None:
    monkeypatch.setattr(
        server, "decode_jpeg", lambda _v, _c: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    backend = _ScriptedBackend(["coast", "track", "coast"], image_size=(8, 8))
    application = server.EdgeTamApplication(
        server.ServiceConfig(max_coast_frames=1),
        backend=backend,
    )
    session = _seed_session(application)

    assert _update(application, 1)["status"] == "coasting"
    assert session.coast_frames == 1
    tracking = _update(application, 2)
    assert tracking["status"] == "tracking"
    assert session.coast_frames == 0
    # Because the counter reset, another single coast is still permitted.
    assert _update(application, 3)["status"] == "coasting"
    assert application._sessions == {session.session_id: session}


def test_coast_frames_environment_must_be_non_negative() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("EDGETAM_MAX_COAST_FRAMES", "-1")
        with pytest.raises(ValueError, match="EDGETAM_MAX_COAST_FRAMES"):
            server.ServiceConfig.from_env()


def test_frame_ceiling_default_is_high_enough_to_avoid_periodic_reacquire() -> None:
    # A low ceiling forced an identity-destroying re-acquire every few minutes;
    # streaming state is pruned per frame so the ceiling can be large.
    assert server.ServiceConfig().max_frames_per_session >= 100_000


def test_stream_history_environment_must_be_positive() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("EDGETAM_STREAM_HISTORY_FRAMES", "0")
        with pytest.raises(ValueError, match="streaming history"):
            server.ServiceConfig.from_env()


def test_service_confidence_environment_is_bounded() -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("EDGETAM_MIN_SCORE", "1.1")
        with pytest.raises(ValueError, match="EDGETAM_MIN_SCORE"):
            server.ServiceConfig.from_env()
