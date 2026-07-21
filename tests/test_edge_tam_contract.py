import numpy as np
import pytest

from z_manip.perception.edge_tam import (
    EdgeTamObservation,
    EdgeTamTopics,
    EdgeTamTrackBuffer,
    TrackingLost,
)
from z_manip.perception.rgbd import BoundingBox


def _observation(track_id=1, stamp_s=1.0, points=40):
    return EdgeTamObservation(
        track_id=track_id,
        is_tracking=True,
        bbox=BoundingBox(10, 20, 80, 120),
        object_points=np.ones((points, 3), dtype=np.float32),
        frame="camera_color_optical_frame",
        stamp_s=stamp_s,
        score=0.9,
    )


def test_edge_tam_topics_are_explicit_and_namespaced():
    topics = EdgeTamTopics()
    assert topics.init_bbox == "/track_3d/init_bbox"
    assert topics.selected_pointcloud == "/track_3d/selected_target_pointcloud"
    assert all(value.startswith("/") for value in topics.__dict__.values())


def test_edge_tam_buffer_preserves_identity_and_rejects_stale_data():
    buffer = EdgeTamTrackBuffer(max_age_s=0.4, min_points=20)
    buffer.begin("mug body", BoundingBox(10, 20, 80, 120), stamp_s=0.8)
    buffer.ingest(_observation(stamp_s=1.0))

    assert buffer.latest(now_s=1.2).track_id == 1
    with pytest.raises(TrackingLost, match="stale"):
        buffer.latest(now_s=1.41)


def test_edge_tam_buffer_fails_closed_on_track_id_jump_or_thin_cloud():
    buffer = EdgeTamTrackBuffer(max_age_s=1.0, min_points=20)
    buffer.begin("target", BoundingBox(0, 0, 20, 20), stamp_s=0.0)
    buffer.ingest(_observation(track_id=4, points=40))
    with pytest.raises(TrackingLost, match="changed"):
        buffer.ingest(_observation(track_id=8, stamp_s=1.1, points=40))

    buffer.reset()
    buffer.begin("target", BoundingBox(0, 0, 20, 20), stamp_s=2.0)
    with pytest.raises(TrackingLost, match="point cloud"):
        buffer.ingest(_observation(track_id=1, stamp_s=2.1, points=3))


def test_edge_tam_buffer_stops_immediately_when_tracker_reports_loss():
    buffer = EdgeTamTrackBuffer()
    buffer.begin("target", BoundingBox(0, 0, 20, 20), stamp_s=0.0)
    lost = _observation()
    lost = EdgeTamObservation(**{**lost.__dict__, "is_tracking": False})
    with pytest.raises(TrackingLost, match="reported loss"):
        buffer.ingest(lost)
    with pytest.raises(TrackingLost, match="no current"):
        buffer.latest(now_s=1.0)
