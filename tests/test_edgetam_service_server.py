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
