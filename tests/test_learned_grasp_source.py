import numpy as np
import pytest

from z_manip.inference import GRASP_CONVENTION, GraspInferenceResult
from z_manip.models.grasp_source import GraspContext, GraspGenerationError
from z_manip.models.learned_grasp import LearnedGraspSource


class FakeClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _context(points, scene=None):
    return GraspContext(
        object_points=np.asarray(points, dtype=np.float32),
        bbox=None,
        source_frame="piper_base_link",
        t_target_src=np.eye(4),
        scene_points=None if scene is None else np.asarray(scene, dtype=np.float32),
        progress_cb=lambda *_args: None,
    )


def _result(frame="piper_base_link"):
    poses = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    poses[:, :3, 3] = ((0.4, 0.0, 0.2), (0.42, 0.01, 0.22))
    return GraspInferenceResult(
        grasps=poses,
        scores=np.array([0.9, 0.7]),
        widths=np.array([0.045, 0.052]),
        frame=frame,
        convention=GRASP_CONVENTION,
        provider="hggd",
        model="model",
        model_version="1",
    )


def test_learned_source_adapts_observation_contract_without_object_pose():
    points = np.array([[0.39, -0.02, 0.18], [0.44, 0.02, 0.25], [0.4, 0.0, 0.2]])
    scene = np.vstack((points, [[-0.2, -0.3, 0.0], [0.9, 0.4, 0.8]]))
    client = FakeClient(_result())

    candidates = LearnedGraspSource(client).generate(_context(points, scene))

    assert candidates.grasps.shape == (2, 4, 4)
    assert candidates.widths.tolist() == pytest.approx([0.045, 0.052])
    assert candidates.centroid == pytest.approx(np.median(points, axis=0))
    request = client.calls[0]
    assert set(request) == {"object_points", "colors", "scene_bounds", "frame"}
    assert "object_pose" not in request
    assert np.all(request["scene_bounds"][0] < np.min(scene, axis=0))
    assert np.all(request["scene_bounds"][1] > np.max(scene, axis=0))


def test_learned_source_rejects_frame_drift_and_empty_cloud():
    points = np.array([[0.4, 0.0, 0.2], [0.41, 0.01, 0.21]])
    with pytest.raises(GraspGenerationError, match="frame"):
        LearnedGraspSource(FakeClient(_result(frame="map"))).generate(_context(points))
    with pytest.raises(GraspGenerationError, match="point cloud"):
        LearnedGraspSource(FakeClient(_result())).generate(_context(np.empty((0, 3))))
