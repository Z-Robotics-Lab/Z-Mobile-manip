from __future__ import annotations

import numpy as np
import pytest

from z_manip.inference import (
    GRASP_CONVENTION,
    PROTOCOL_VERSION,
    GraspInferenceClient,
    GraspInferenceConfig,
    GraspInferenceProtocolError,
    GraspInferenceTimeout,
    GraspInferenceUnavailable,
    ZmqMsgpackTransport,
)
from z_manip.models.grasp_source import GraspGenerationError


def _array(array):
    value = np.ascontiguousarray(array)
    return {"dtype": value.dtype.str, "shape": list(value.shape), "data": value.tobytes()}


def _base_response(**overrides):
    response = {
        "protocol": PROTOCOL_VERSION,
        "status": "ok",
        "provider": "test-provider",
        "model": "six-dof-net",
        "model_version": "2026.07",
    }
    response.update(overrides)
    return response


class FakeTransport:
    def __init__(self, responses=None, error=None):
        self.responses = responses or {}
        self.error = error
        self.calls = []

    def request(self, operation, payload, *, timeout_s):
        self.calls.append((operation, payload, timeout_s))
        if self.error is not None:
            raise self.error
        return self.responses[operation]


def _client(transport, *, max_grasps=8, monotonic_fn=None):
    kwargs = {} if monotonic_fn is None else {"monotonic_fn": monotonic_fn}
    return GraspInferenceClient(
        GraspInferenceConfig(
            provider="test-provider",
            endpoint="memory://grasp",
            timeout_s=0.25,
            max_grasps=max_grasps,
        ),
        transport=transport,
        **kwargs,
    )


def _observation():
    points = np.array(
        [[0.42, -0.02, 0.10], [0.44, 0.01, 0.12], [0.43, 0.00, 0.16]],
        dtype=np.float32,
    )
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    bounds = np.array([[0.2, -0.3, 0.0], [0.8, 0.3, 0.7]], dtype=np.float32)
    return points, colors, bounds


def _valid_infer_response():
    grasps = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    grasps[:, :3, 3] = ((0.43, 0.0, 0.12), (0.44, 0.01, 0.13))
    return _base_response(
        frame="base_link",
        convention=GRASP_CONVENTION,
        grasps=_array(grasps),
        scores=_array(np.array([0.91, 0.73], dtype=np.float32)),
        widths=_array(np.array([0.052, 0.061], dtype=np.float32)),
    )


def test_infer_sends_only_observation_geometry_and_validates_six_dof_result():
    transport = FakeTransport({"infer": _valid_infer_response()})
    points, colors, bounds = _observation()

    result = _client(transport).infer(
        object_points=points,
        colors=colors,
        scene_bounds=bounds,
        frame="base_link",
    )

    assert result.grasps.shape == (2, 4, 4)
    assert np.allclose(np.linalg.det(result.grasps[:, :3, :3]), 1.0)
    assert result.scores.tolist() == pytest.approx([0.91, 0.73])
    assert result.widths.tolist() == pytest.approx([0.052, 0.061])
    assert result.frame == "base_link"
    assert result.convention == GRASP_CONVENTION
    assert (result.provider, result.model, result.model_version) == (
        "test-provider",
        "six-dof-net",
        "2026.07",
    )

    operation, payload, timeout = transport.calls[0]
    assert operation == "infer"
    assert timeout == pytest.approx(0.25)
    assert set(payload) == {
        "object_points",
        "colors",
        "scene_bounds",
        "frame",
        "convention",
        "max_grasps",
    }
    assert not {"object_pose", "gt_pose", "world_object_pose"}.intersection(payload)
    assert payload["object_points"]["shape"] == [3, 3]
    assert payload["colors"]["dtype"] == np.dtype(np.uint8).str


def test_health_and_metadata_use_versioned_contract():
    transport = FakeTransport(
        {
            "health": _base_response(ready=True),
            "metadata": _base_response(
                convention=GRASP_CONVENTION,
                operations=["health", "metadata", "infer"],
            ),
        },
    )
    client = _client(transport)

    health = client.health()
    metadata = client.metadata()

    assert health.ready
    assert metadata.operations == ("health", "metadata", "infer")
    assert metadata.convention == GRASP_CONVENTION
    assert [call[0] for call in transport.calls] == ["health", "metadata"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda response: response.update(frame="map"), "frame"),
        (lambda response: response.update(convention="z_closing"), "convention"),
        (lambda response: response.update(scores=[0.5]), "one-to-one"),
        (lambda response: response.update(widths=[-0.01, 0.04]), "positive"),
    ],
)
def test_malformed_responses_fail_closed(mutate, message):
    response = _valid_infer_response()
    mutate(response)
    points, colors, bounds = _observation()
    with pytest.raises(GraspInferenceProtocolError, match=message):
        _client(FakeTransport({"infer": response})).infer(
            object_points=points,
            colors=colors,
            scene_bounds=bounds,
            frame="base_link",
        )


def test_left_handed_grasp_is_rejected():
    response = _valid_infer_response()
    grasps = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    grasps[0, 0, 0] = -1.0
    response["grasps"] = _array(grasps)
    points, colors, bounds = _observation()

    with pytest.raises(GraspInferenceProtocolError, match="left-handed"):
        _client(FakeTransport({"infer": response})).infer(
            object_points=points,
            colors=colors,
            scene_bounds=bounds,
            frame="base_link",
        )


def test_forbidden_pose_field_is_rejected_even_when_nested():
    response = _valid_infer_response()
    response["diagnostics"] = {"object_pose": np.eye(4).tolist()}
    points, colors, bounds = _observation()
    with pytest.raises(GraspInferenceProtocolError, match="forbidden pose fields"):
        _client(FakeTransport({"infer": response})).infer(
            object_points=points,
            colors=colors,
            scene_bounds=bounds,
            frame="base_link",
        )


def test_timeout_and_provider_failure_are_recoverable_by_geometric_cascade():
    timeout_client = _client(FakeTransport(error=TimeoutError("deadline")))
    points, colors, bounds = _observation()
    with pytest.raises(GraspInferenceTimeout) as raised:
        timeout_client.infer(
            object_points=points,
            colors=colors,
            scene_bounds=bounds,
            frame="base_link",
        )
    assert isinstance(raised.value, GraspGenerationError)

    failure = _base_response(status="error", error="GPU unavailable")
    with pytest.raises(GraspInferenceUnavailable, match="GPU unavailable"):
        _client(FakeTransport({"infer": failure})).infer(
            object_points=points,
            scene_bounds=bounds,
            frame="base_link",
        )


def test_environment_config_and_lazy_optional_transport_dependencies():
    config = GraspInferenceConfig.from_env(
        {
            "Z_MANIP_GRASP_PROVIDER": "remote-hggd",
            "Z_MANIP_GRASP_ENDPOINT": "tcp://10.0.0.8:5557",
            "Z_MANIP_GRASP_TIMEOUT_S": "2.25",
            "Z_MANIP_GRASP_MAX_GRASPS": "48",
        },
    )
    assert config.provider == "remote-hggd"
    assert config.endpoint == "tcp://10.0.0.8:5557"
    assert config.timeout_s == pytest.approx(2.25)
    assert config.max_grasps == 48

    def unavailable_import(_name):
        raise ModuleNotFoundError("not installed")

    transport = ZmqMsgpackTransport("tcp://127.0.0.1:5557", importer=unavailable_import)
    with pytest.raises(GraspInferenceUnavailable, match="pyzmq.*msgpack"):
        transport.request("health", {}, timeout_s=0.1)


def test_bad_request_never_reaches_transport_and_object_pose_is_not_an_api_input():
    transport = FakeTransport({"infer": _valid_infer_response()})
    client = _client(transport)
    points, colors, bounds = _observation()
    with pytest.raises(GraspInferenceProtocolError, match="outside scene_bounds"):
        client.infer(
            object_points=points + 2.0,
            colors=colors,
            scene_bounds=bounds,
            frame="base_link",
        )
    assert transport.calls == []

    with pytest.raises(TypeError, match="object_pose"):
        client.infer(
            object_points=points,
            scene_bounds=bounds,
            frame="base_link",
            object_pose=np.eye(4),
        )
