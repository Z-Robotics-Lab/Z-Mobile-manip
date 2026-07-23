"""Unit tests for the ROS-independent fail-closed tracking core."""

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from z_manip_edgetam.core import (
    AcquisitionGate,
    CameraIntrinsics,
    center_box_to_half_open,
    FailClosedTracker,
    project_mask_depth,
    project_mask_depth_geometry,
    project_scene_depth,
    register_seed_bbox_to_latest,
    ReseedRegistrationConfig,
    RgbdFrame,
    TrackerFailure,
)


JPEG = b'\xff\xd8unit-test\xff\xd9'


@dataclass(frozen=True, eq=False)
class FakeTrack:
    """Minimal valid service result used by the injected fake client."""

    session_id: str
    track_id: str
    frame_seq: int
    image_size: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    mask: np.ndarray


def make_track(
    *,
    session_id: str,
    frame_seq: int,
    bbox: tuple[int, int, int, int] = (1, 1, 4, 4),
    track_id: str = 'target-7',
    size: tuple[int, int] = (6, 5),
) -> FakeTrack:
    """Create a result whose bbox exactly bounds its half-open mask."""
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = True
    return FakeTrack(
        session_id=session_id,
        track_id=track_id,
        frame_seq=frame_seq,
        image_size=size,
        bbox_xyxy=bbox,
        score=0.91,
        mask=mask,
    )


class FakeClient:
    """Stateful fake exposing the real service client's injectable boundary."""

    def __init__(self) -> None:
        self.active = False
        self.session_id = ''
        self.update_calls: list[int] = []
        self.reset_calls = 0
        self.next_track_id = 'target-7'
        self.next_bbox = (1, 1, 4, 4)
        self.update_error: Exception | None = None

    def init(
        self,
        _image_jpeg: bytes,
        bbox_xyxy: tuple[int, int, int, int],
        *,
        session_id: str,
        frame_seq: int = 0,
    ) -> FakeTrack:
        """Initialize a deterministic fake identity."""
        self.active = True
        self.session_id = session_id
        return make_track(
            session_id=session_id,
            frame_seq=frame_seq,
            bbox=bbox_xyxy,
        )

    def update(self, _image_jpeg: bytes, *, frame_seq: int | None = None) -> FakeTrack:
        """Return one configured update or raise its configured failure."""
        assert frame_seq is not None
        if self.update_error is not None:
            raise self.update_error
        self.update_calls.append(frame_seq)
        return make_track(
            session_id=self.session_id,
            frame_seq=frame_seq,
            bbox=self.next_bbox,
            track_id=self.next_track_id,
        )

    def reset(self) -> None:
        """Record that the fail-closed core discarded remote state."""
        self.active = False
        self.reset_calls += 1


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    return (
        int(cols.min()),
        int(rows.min()),
        int(cols.max()) + 1,
        int(rows.max()) + 1,
    )


class MaskSequenceClient(FakeClient):
    """Fake client that returns exact custom masks in service sequence order."""

    def __init__(self, masks: list[np.ndarray]) -> None:
        super().__init__()
        assert masks
        self._masks = [np.asarray(mask, dtype=bool) for mask in masks]
        self._result_index = 0
        self.frame_seq_offset = 0

    def _result(self, frame_seq: int) -> FakeTrack:
        mask = self._masks[self._result_index]
        height, width = mask.shape
        return FakeTrack(
            session_id=self.session_id,
            track_id=self.next_track_id,
            frame_seq=frame_seq + self.frame_seq_offset,
            image_size=(width, height),
            bbox_xyxy=_mask_bbox(mask),
            score=0.91,
            mask=mask,
        )

    def init(
        self,
        _image_jpeg: bytes,
        bbox_xyxy: tuple[int, int, int, int],
        *,
        session_id: str,
        frame_seq: int = 0,
    ) -> FakeTrack:
        self.active = True
        self.session_id = session_id
        assert bbox_xyxy == _mask_bbox(self._masks[0])
        return self._result(frame_seq)

    def update(self, _image_jpeg: bytes, *, frame_seq: int | None = None) -> FakeTrack:
        assert frame_seq is not None
        self.update_calls.append(frame_seq)
        self._result_index += 1
        assert self._result_index < len(self._masks)
        return self._result(frame_seq)


def frame(stamp_ns: int, *, depth: np.ndarray | None = None) -> RgbdFrame:
    """Create one aligned synthetic RGB-D frame."""
    if depth is None:
        depth = np.ones((5, 6), dtype=np.float32)
    return RgbdFrame(
        stamp_ns=stamp_ns,
        frame_id='camera_color_optical_frame',
        image_jpeg=JPEG,
        width=6,
        height=5,
        depth_m=depth,
        intrinsics=CameraIntrinsics(fx=2.0, fy=2.0, cx=2.0, cy=2.0),
    )


def sized_frame(
    stamp_ns: int,
    *,
    width: int,
    height: int,
    depth: np.ndarray | None = None,
) -> RgbdFrame:
    """Create a configurable aligned frame for mask-continuity tests."""
    if depth is None:
        depth = np.ones((height, width), dtype=np.float32)
    return RgbdFrame(
        stamp_ns=stamp_ns,
        frame_id='camera_color_optical_frame',
        image_jpeg=JPEG,
        width=width,
        height=height,
        depth_m=depth,
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=16.0, cy=16.0),
    )


def mask_tracker(masks: list[np.ndarray]) -> tuple[FailClosedTracker, MaskSequenceClient]:
    """Initialize a configurable tracker with the first custom mask."""
    client = MaskSequenceClient(masks)
    subject = FailClosedTracker(
        client,
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=2,
        max_points=1_000,
        session_id_factory=lambda: 'test-session',
    )
    height, width = masks[0].shape
    subject.initialize(
        stamp_ns=10,
        image_jpeg=JPEG,
        width=width,
        height=height,
        bbox_xyxy=_mask_bbox(masks[0]),
        label='mustard bottle',
    )
    return subject, client


def tracker(client: FakeClient, *, min_points: int = 2) -> FailClosedTracker:
    """Create a deterministic core around a fake service client."""
    return FailClosedTracker(
        client,
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=min_points,
        max_points=100,
        session_id_factory=lambda: 'test-session',
    )


def initialize(subject: FailClosedTracker) -> None:
    """Initialize the common fake target."""
    subject.initialize(
        stamp_ns=10,
        image_jpeg=JPEG,
        width=6,
        height=5,
        bbox_xyxy=(1, 1, 4, 4),
        label='mustard bottle',
    )


def test_center_box_conversion_is_enclosing_clamped_and_half_open() -> None:
    assert center_box_to_half_open(
        2.25,
        2.75,
        3.5,
        2.5,
        width=6,
        height=5,
    ) == (0, 1, 4, 4)
    assert center_box_to_half_open(
        -0.2,
        1.0,
        1.0,
        2.0,
        width=6,
        height=5,
    ) == (0, 0, 1, 2)
    with pytest.raises(ValueError, match='empty'):
        center_box_to_half_open(-3.0, 1.0, 1.0, 1.0, width=6, height=5)


def _registration_image() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Create deterministic, independently textured scene and target regions."""
    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.0)
    bbox = (92, 58, 148, 126)
    x1, y1, x2, y2 = bbox
    roi = rng.integers(0, 256, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    image[y1:y2, x1:x2] = roi
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), 255, 2)
    return image, bbox


def _low_feature_registration_image(
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Create a tiny target whose surrounding scene cannot supply evidence."""
    rng = np.random.default_rng(21)
    background = rng.integers(0, 256, size=(180, 240), dtype=np.uint8)
    seed = background.copy()
    bbox = (100, 65, 115, 95)
    x1, y1, x2, y2 = bbox
    seed[y1:y2, x1:x2] = 190
    cv2.rectangle(seed, (x1, y1), (x2 - 1, y2 - 1), 245, 1)
    return seed, background, bbox


def _jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        '.jpg',
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    assert ok
    return encoded.tobytes()


def _registration_config() -> ReseedRegistrationConfig:
    return ReseedRegistrationConfig(
        min_global_tracks=16,
        min_global_inliers=12,
        min_roi_tracks=6,
        min_roi_inliers=5,
    )


def test_latest_frame_reseed_transfers_box_with_dual_registration() -> None:
    seed, bbox = _registration_image()
    transform = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    latest = cv2.warpAffine(
        seed,
        transform,
        (seed.shape[1], seed.shape[0]),
        borderMode=cv2.BORDER_REFLECT101,
    )

    result = register_seed_bbox_to_latest(
        _jpeg(seed),
        _jpeg(latest),
        bbox,
        width=seed.shape[1],
        height=seed.shape[0],
        config=_registration_config(),
    )

    assert np.allclose(result.bbox_xyxy, (96, 55, 152, 123), atol=1)
    assert result.global_inliers >= 12
    assert result.roi_inliers >= 5
    assert result.center_delta_ratio < 0.01


def test_reseed_rejects_target_motion_relative_to_static_scene() -> None:
    seed, bbox = _registration_image()
    latest = seed.copy()
    x1, y1, x2, y2 = bbox
    target = seed[y1:y2, x1:x2].copy()
    latest[y1:y2, x1:x2] = 127
    latest[y1:y2, x1 + 14:x2 + 14] = target

    with pytest.raises(TrackerFailure, match='inconsistent') as error:
        register_seed_bbox_to_latest(
            _jpeg(seed),
            _jpeg(latest),
            bbox,
            width=seed.shape[1],
            height=seed.shape[0],
            config=_registration_config(),
        )

    assert error.value.reason_code == 'seed_reseed_registration'


def test_tiny_low_feature_reseed_rejects_removed_target() -> None:
    seed, background, bbox = _low_feature_registration_image()

    with pytest.raises(
        TrackerFailure,
        match='ROI reseed has too few seed features',
    ) as error:
        register_seed_bbox_to_latest(
            _jpeg(seed),
            _jpeg(background),
            bbox,
            width=seed.shape[1],
            height=seed.shape[0],
            config=_registration_config(),
        )

    assert error.value.reason_code == 'seed_reseed_registration'


def test_tiny_low_feature_reseed_rejects_relative_target_motion() -> None:
    seed, background, bbox = _low_feature_registration_image()
    x1, y1, x2, y2 = bbox
    latest = background.copy()
    latest[y1:y2, x1 + 8:x2 + 8] = seed[y1:y2, x1:x2]

    with pytest.raises(
        TrackerFailure,
        match='ROI reseed has too few seed features',
    ) as error:
        register_seed_bbox_to_latest(
            _jpeg(seed),
            _jpeg(latest),
            bbox,
            width=seed.shape[1],
            height=seed.shape[0],
            config=_registration_config(),
        )

    assert error.value.reason_code == 'seed_reseed_registration'


def test_reseed_rejects_scene_change_low_texture_and_bad_jpeg() -> None:
    seed, bbox = _registration_image()
    changed = np.random.default_rng(99).integers(
        0,
        256,
        size=seed.shape,
        dtype=np.uint8,
    )
    flat = np.full_like(seed, 127)
    cases = (
        (_jpeg(seed), _jpeg(changed)),
        (_jpeg(flat), _jpeg(flat)),
        (_jpeg(seed), b'\xff\xd8broken\xff\xd9'),
    )

    for seed_jpeg, latest_jpeg in cases:
        with pytest.raises(TrackerFailure) as error:
            register_seed_bbox_to_latest(
                seed_jpeg,
                latest_jpeg,
                bbox,
                width=seed.shape[1],
                height=seed.shape[0],
                config=_registration_config(),
            )
        assert error.value.reason_code == 'seed_reseed_registration'


def test_acquisition_gate_requires_consecutive_validated_updates() -> None:
    gate = AcquisitionGate(3)

    assert not gate.accept()
    assert not gate.accept()
    assert gate.accept()
    assert gate.ready
    assert gate.accepted_updates == 3

    gate.reset()
    assert not gate.ready
    assert gate.accepted_updates == 0


def test_projection_uses_only_current_mask_and_aligned_depth() -> None:
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 1] = True
    mask[1, 2] = True
    depth = np.zeros((3, 4), dtype=np.float32)
    depth[1, 1] = 1.0
    depth[1, 2] = 1.02
    points = project_mask_depth(
        mask,
        depth,
        CameraIntrinsics(fx=2.0, fy=4.0, cx=1.0, cy=1.0),
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=2,
        max_points=10,
    )
    np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [0.51, 0.0, 1.02]])
    assert points.dtype == np.float32
    assert not points.flags.writeable


def test_projection_rejects_sparse_invalid_depth() -> None:
    mask = np.ones((2, 2), dtype=bool)
    depth = np.array([[np.nan, 0.0], [5.0, 1.0]], dtype=np.float32)
    with pytest.raises(TrackerFailure, match='sparse'):
        project_mask_depth(
            mask,
            depth,
            CameraIntrinsics(2.0, 2.0, 1.0, 1.0),
            min_depth_m=0.1,
            max_depth_m=3.0,
            min_points=2,
            max_points=10,
        )


def test_dominant_depth_cluster_rejects_touching_far_background_band() -> None:
    """A connected 2-D mask must not join a shelf/wall depth discontinuity."""
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 1:6] = True
    depth = np.zeros_like(mask, dtype=np.float32)
    depth[1:5, 1:4] = 1.0
    depth[1:5, 4:6] = 2.4
    intrinsics = CameraIntrinsics(fx=10.0, fy=12.0, cx=3.0, cy=2.0)

    projection = project_mask_depth_geometry(
        mask,
        depth,
        intrinsics,
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=4,
        max_points=5,
    )

    assert int(projection.mask.sum()) == 12
    assert projection.points_xyz.shape == (5, 3)
    assert projection.pixels_uv.shape == (5, 2)
    assert projection.rejected_pixels_uv.shape == (8, 2)
    np.testing.assert_allclose(projection.points_xyz[:, 2], 1.0)
    np.testing.assert_array_equal(
        np.unique(projection.rejected_pixels_uv[:, 0]),
        [4, 5],
    )
    expected_x = (
        projection.pixels_uv[:, 0].astype(float) - intrinsics.cx
    ) * projection.points_xyz[:, 2] / intrinsics.fx
    expected_y = (
        projection.pixels_uv[:, 1].astype(float) - intrinsics.cy
    ) * projection.points_xyz[:, 2] / intrinsics.fy
    np.testing.assert_allclose(projection.points_xyz[:, 0], expected_x)
    np.testing.assert_allclose(projection.points_xyz[:, 1], expected_y)

    scene = project_scene_depth(
        projection.mask,
        depth,
        intrinsics,
        target_dilation_px=2,
        stride=3,
        min_depth_m=0.1,
        max_depth_m=3.0,
        max_points=100,
        restore_pixels_uv=projection.rejected_pixels_uv,
    )
    # All eight rejected background pixels are restored even though dilation
    # covers them and most are not sampled by the stride-three scene grid.
    assert np.count_nonzero(np.isclose(scene[:, 2], 2.4)) == 8


def test_depth_cluster_fails_closed_when_no_component_meets_minimum() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 1:3] = True
    mask[2, 4:6] = True
    depth = np.ones((4, 6), dtype=np.float32)

    with pytest.raises(TrackerFailure, match='dominant target depth cluster'):
        project_mask_depth_geometry(
            mask,
            depth,
            CameraIntrinsics(4.0, 4.0, 2.5, 1.5),
            min_depth_m=0.1,
            max_depth_m=3.0,
            min_points=3,
            max_points=10,
        )


def test_equal_depth_components_choose_first_raster_component() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 3] = True
    mask[3, 1] = True
    depth = np.ones((5, 5), dtype=np.float32)

    projection = project_mask_depth_geometry(
        mask,
        depth,
        CameraIntrinsics(4.0, 4.0, 2.0, 2.0),
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=1,
        max_points=10,
    )

    np.testing.assert_array_equal(projection.pixels_uv, [[3, 1]])
    np.testing.assert_array_equal(projection.rejected_pixels_uv, [[1, 3]])


def test_init_then_strict_update_produces_identity_and_cloud() -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)

    observation = subject.update(frame(11))

    assert observation.session_id == 'test-session'
    assert observation.track_id == 'target-7'
    assert observation.label == 'mustard bottle'
    assert observation.bbox_xyxy == (1, 1, 4, 4)
    assert observation.points_xyz.shape == (9, 3)
    assert observation.mask.shape == (5, 6)
    assert observation.mask.flags.writeable is False
    assert observation.pixels_uv.shape == (9, 2)
    assert observation.pixels_uv.flags.writeable is False
    assert observation.rejected_pixels_uv.shape == (0, 2)
    assert observation.rejected_pixels_uv.flags.writeable is False
    assert client.update_calls == [1]
    subject.update(frame(12))
    assert client.update_calls == [1, 2]


def test_tracker_rejects_full_frame_drift_instead_of_reusing_old_seed_box() -> None:
    client = FakeClient()
    client.next_bbox = (0, 0, 6, 5)
    subject = tracker(client)
    initialize(subject)

    with pytest.raises(TrackerFailure, match='mask area is unsafe') as error:
        subject.update(frame(11))

    assert error.value.reason_code == 'mask_area_limit'
    assert not subject.active
    assert not client.active


def test_rgb_replay_advances_sequence_before_first_depth_observation() -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)

    subject.replay_rgb(stamp_ns=11, image_jpeg=JPEG, width=6, height=5)
    observation = subject.update(frame(12))

    assert client.update_calls == [1, 2]
    assert subject.last_stamp_ns == 12
    assert observation.stamp_ns == 12
    assert observation.points_xyz.shape == (9, 3)


def test_rgb_replay_fails_closed_on_mask_discontinuity_with_typed_reason() -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)
    client.next_bbox = (4, 1, 6, 4)

    with pytest.raises(TrackerFailure, match='mask continuity broke') as error:
        subject.replay_rgb(stamp_ns=11, image_jpeg=JPEG, width=6, height=5)

    assert error.value.reason_code == 'mask_continuity'
    assert not subject.active
    assert not client.active


def test_rgb_replay_after_depth_observation_fails_closed() -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)
    subject.update(frame(11))

    with pytest.raises(TrackerFailure, match='before the first depth') as error:
        subject.replay_rgb(stamp_ns=12, image_jpeg=JPEG, width=6, height=5)

    assert error.value.reason_code == 'invalid_replay_phase'
    assert not subject.active
    assert not client.active


def test_tracker_publishes_only_dominant_geometry_for_background_leak() -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)
    depth = np.ones((5, 6), dtype=np.float32)
    depth[2, 2] = 2.4

    observation = subject.update(frame(11, depth=depth))

    assert observation.points_xyz.shape == (8, 3)
    np.testing.assert_allclose(observation.points_xyz[:, 2], 1.0)
    np.testing.assert_array_equal(observation.rejected_pixels_uv, [[2, 2]])
    assert not observation.mask[2, 2]
    assert int(observation.mask.sum()) == 8


def test_project_scene_depth_excludes_dilated_target_pixels() -> None:
    depth = np.ones((5, 5), dtype=np.float32)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    points = project_scene_depth(
        mask,
        depth,
        CameraIntrinsics(10.0, 10.0, 2.0, 2.0),
        target_dilation_px=1,
        stride=1,
        min_depth_m=0.2,
        max_depth_m=2.0,
        max_points=100,
    )
    assert points.shape == (16, 3)


@pytest.mark.parametrize('bad_stamp', [9, 10])
def test_duplicate_or_out_of_order_frame_fails_closed(bad_stamp: int) -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)

    with pytest.raises(TrackerFailure, match='out of order'):
        subject.update(frame(bad_stamp))

    assert not subject.active
    assert not client.active
    assert client.reset_calls == 1
    assert client.update_calls == []


def test_identity_change_fails_closed() -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)
    client.next_track_id = 'different-target'

    with pytest.raises(TrackerFailure, match='identity changed'):
        subject.update(frame(11))

    assert not subject.active
    assert not client.active


def test_stable_service_id_with_disjoint_mask_fails_closed() -> None:
    """A transport UUID is insufficient when the tracked pixels jump identity."""
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)
    client.next_bbox = (4, 1, 6, 4)

    with pytest.raises(TrackerFailure, match='mask continuity broke'):
        subject.update(frame(11))

    assert not subject.active
    assert not client.active


def test_mobile_camera_reanchors_disjoint_mask_with_live_3d_continuity() -> None:
    """A bounded camera turn may move one tracked object beyond its old mask."""
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:16, 4:12] = True
    shifted = np.zeros_like(stable)
    shifted[8:16, 14:22] = True
    shifted_again = np.zeros_like(stable)
    shifted_again[8:16, 20:28] = True
    client = MaskSequenceClient([stable, stable, shifted, shifted_again])
    subject = FailClosedTracker(
        client,
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=2,
        max_points=1_000,
        allow_motion_reanchor=True,
        min_motion_reanchor_area_ratio=0.60,
        max_motion_reanchor_displacement_ratio=1.25,
        session_id_factory=lambda: 'test-session',
    )
    subject.initialize(
        stamp_ns=10,
        image_jpeg=JPEG,
        width=32,
        height=32,
        bbox_xyxy=_mask_bbox(stable),
        label='charger',
    )

    subject.update(sized_frame(1_000_000_010, width=32, height=32))
    moved = subject.update(sized_frame(2_000_000_010, width=32, height=32))
    moved_again = subject.update(sized_frame(3_000_000_010, width=32, height=32))

    assert moved is not None
    assert moved_again is not None
    assert moved.bbox_xyxy == (14, 8, 22, 16)
    assert moved_again.bbox_xyxy == (20, 8, 28, 16)
    assert subject.active
    assert client.reset_calls == 0


def test_contained_mask_scale_collapse_recovers_without_reanchoring() -> None:
    """A one-frame 0.147x collapse needs two stable frames before publication."""
    stable = np.zeros((80, 80), dtype=bool)
    stable[10:49, 10:55] = True
    collapsed = np.zeros_like(stable)
    collapsed[26:32, 11:54] = True
    assert int(stable.sum()) == 1_755
    assert int(collapsed.sum()) == 258
    subject, client = mask_tracker([stable, collapsed, stable, stable])

    assert subject.update(sized_frame(
        1_000_000_010,
        width=80,
        height=80,
    )) is None
    assert subject.pending_mask_anomaly
    assert subject.update(sized_frame(
        2_000_000_010,
        width=80,
        height=80,
    )) is None
    assert subject.pending_mask_anomaly
    recovered = subject.update(sized_frame(
        3_000_000_010,
        width=80,
        height=80,
    ))

    assert recovered is not None
    assert not subject.pending_mask_anomaly
    assert int(recovered.mask.sum()) == 1_755
    assert client.update_calls == [1, 2, 3]
    assert client.reset_calls == 0


def test_persistent_contained_mask_scale_collapse_fails_closed() -> None:
    stable = np.zeros((80, 80), dtype=bool)
    stable[10:49, 10:55] = True
    collapsed = np.zeros_like(stable)
    collapsed[26:32, 11:54] = True
    subject, client = mask_tracker(
        [stable, collapsed, collapsed, collapsed],
    )

    assert subject.update(sized_frame(
        1_000_000_010,
        width=80,
        height=80,
    )) is None
    assert subject.update(sized_frame(
        2_000_000_010,
        width=80,
        height=80,
    )) is None
    with pytest.raises(TrackerFailure, match='collapse persisted') as error:
        subject.update(sized_frame(
            3_000_000_010,
            width=80,
            height=80,
        ))

    assert error.value.reason_code == 'mask_scale_collapse'
    assert not subject.active
    assert not client.active


def test_identity_change_during_pending_collapse_is_immediately_terminal() -> None:
    stable = np.zeros((80, 80), dtype=bool)
    stable[10:49, 10:55] = True
    collapsed = np.zeros_like(stable)
    collapsed[26:32, 11:54] = True
    subject, client = mask_tracker([stable, collapsed, stable])

    assert subject.update(sized_frame(
        1_000_000_010,
        width=80,
        height=80,
    )) is None
    client.next_track_id = 'different-target'
    with pytest.raises(TrackerFailure, match='identity changed'):
        subject.update(sized_frame(
            2_000_000_010,
            width=80,
            height=80,
        ))

    assert not subject.active
    assert client.reset_calls == 1


@pytest.mark.parametrize('fault', ('session', 'sequence', 'image_size'))
def test_pending_collapse_cannot_bypass_service_protocol(fault: str) -> None:
    stable = np.zeros((80, 80), dtype=bool)
    stable[10:49, 10:55] = True
    collapsed = np.zeros_like(stable)
    collapsed[26:32, 11:54] = True
    subject, client = mask_tracker([stable, collapsed, stable])
    assert subject.update(sized_frame(
        1_000_000_010,
        width=80,
        height=80,
    )) is None
    assert subject.last_stamp_ns == 1_000_000_010
    assert client.update_calls == [1]

    if fault == 'session':
        client.session_id = 'different-session'
        next_frame = sized_frame(2_000_000_010, width=80, height=80)
        match = 'session identity changed'
    elif fault == 'sequence':
        client.frame_seq_offset = 1
        next_frame = sized_frame(2_000_000_010, width=80, height=80)
        match = 'sequence changed'
    else:
        next_frame = sized_frame(2_000_000_010, width=81, height=80)
        match = 'dimensions changed'

    with pytest.raises(TrackerFailure, match=match):
        subject.update(next_frame)

    assert not subject.active
    assert client.reset_calls == 1
    assert client.update_calls == ([1] if fault == 'image_size' else [1, 2])


def test_remote_five_pixel_component_does_not_expand_identity_bbox() -> None:
    stable = np.zeros((80, 80), dtype=bool)
    stable[10:20, 10:20] = True
    raw = stable.copy()
    raw[70, 70:75] = True
    subject, client = mask_tracker([stable, raw])

    observation = subject.update(sized_frame(
        1_000_000_010,
        width=80,
        height=80,
    ))

    assert observation is not None
    assert observation.bbox_xyxy == (10, 10, 20, 20)
    assert int(observation.mask.sum()) == 100
    assert observation.mask_diagnostics.raw_pixels == 105
    assert observation.mask_diagnostics.cleaned_pixels == 100
    assert observation.mask_diagnostics.component_count == 2
    assert observation.mask_diagnostics.rejected_pixels == 5
    assert observation.mask_diagnostics.largest_rejected_component_pixels == 5
    assert observation.mask_diagnostics.selected_overlap_pixels == 100
    assert observation.mask_diagnostics.selection_mode == 'reference_overlap'
    assert observation.mask_diagnostics.rejected_ratio == pytest.approx(5 / 105)
    assert (
        observation.mask_diagnostics.largest_rejected_to_selected_ratio
        == pytest.approx(0.05)
    )
    assert client.reset_calls == 0


def test_bounded_nine_percent_remote_speck_is_sensor_noise_not_identity_loss() -> None:
    stable = np.zeros((80, 80), dtype=bool)
    stable[10:20, 10:20] = True
    raw = stable.copy()
    raw[70:72, 70:75] = True
    subject, client = mask_tracker([stable, raw])

    observation = subject.update(sized_frame(
        1_000_000_010,
        width=80,
        height=80,
    ))

    assert observation is not None
    assert observation.mask_diagnostics.rejected_ratio == pytest.approx(
        10 / 110,
    )
    assert (
        observation.mask_diagnostics.largest_rejected_to_selected_ratio
        == pytest.approx(0.1)
    )
    assert observation.bbox_xyxy == (10, 10, 20, 20)
    assert client.reset_calls == 0


def test_larger_competing_component_is_never_silently_discarded() -> None:
    stable = np.zeros((100, 100), dtype=bool)
    stable[10:30, 10:30] = True
    ambiguous = np.zeros_like(stable)
    ambiguous[10:30, 20:40] = True
    ambiguous[60:85, 60:85] = True
    subject, client = mask_tracker([stable, ambiguous])

    with pytest.raises(TrackerFailure, match='ambiguous competing') as error:
        subject.update(sized_frame(
            1_000_000_010,
            width=100,
            height=100,
        ))

    assert error.value.reason_code == 'mask_component_ambiguity'
    assert not subject.active
    assert client.reset_calls == 1


def test_raw_mask_flood_fails_before_component_cleanup() -> None:
    stable = np.zeros((80, 80), dtype=bool)
    stable[10:20, 10:20] = True
    flooded = stable.copy()
    flooded[25:79, 25:79] = True
    subject, client = mask_tracker([stable, flooded])

    with pytest.raises(TrackerFailure, match='raw mask area') as error:
        subject.update(sized_frame(
            1_000_000_010,
            width=80,
            height=80,
        ))

    assert error.value.reason_code == 'mask_area_limit'
    assert not subject.active
    assert client.reset_calls == 1


def test_raw_initialization_mask_flood_resets_remote_session() -> None:
    flooded = np.zeros((80, 80), dtype=bool)
    flooded[5:65, 5:65] = True
    client = MaskSequenceClient([flooded])
    subject = tracker(client)

    with pytest.raises(TrackerFailure, match='initialization failed') as error:
        subject.initialize(
            stamp_ns=10,
            image_jpeg=JPEG,
            width=80,
            height=80,
            bbox_xyxy=_mask_bbox(flooded),
            label='unsafe target',
        )

    assert error.value.reason_code == 'mask_area_limit'
    assert not subject.active
    assert not client.active
    assert client.reset_calls == 1


def test_ambiguous_initialization_components_reset_remote_session() -> None:
    ambiguous = np.zeros((100, 100), dtype=bool)
    ambiguous[10:30, 10:30] = True
    ambiguous[60:85, 60:85] = True
    client = MaskSequenceClient([ambiguous])
    subject = tracker(client)

    with pytest.raises(TrackerFailure, match='initialization failed') as error:
        subject.initialize(
            stamp_ns=10,
            image_jpeg=JPEG,
            width=100,
            height=100,
            bbox_xyxy=_mask_bbox(ambiguous),
            label='ambiguous target',
        )

    assert error.value.reason_code == 'mask_component_ambiguity'
    assert not subject.active
    assert not client.active
    assert client.reset_calls == 1


def test_live_mask_scale_expansion_has_a_distinct_typed_failure() -> None:
    stable = np.zeros((32, 32), dtype=bool)
    stable[10:12, 10:15] = True
    expanded = np.zeros_like(stable)
    expanded[8:14, 8:15] = True
    subject, client = mask_tracker([stable, expanded])

    with pytest.raises(TrackerFailure, match='mask scale expanded') as error:
        subject.update(sized_frame(1_000_000_010, width=32, height=32))

    assert error.value.reason_code == 'mask_scale_expansion'
    assert not subject.active
    assert not client.active


def test_rgb_replay_requires_nominal_iou_without_live_geometry() -> None:
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:13, 8:16] = True
    borderline = np.zeros_like(stable)
    borderline[7:9, 9:16] = True
    subject, client = mask_tracker([stable, borderline])

    with pytest.raises(TrackerFailure, match='depth_unavailable') as error:
        subject.replay_rgb(
            stamp_ns=11,
            image_jpeg=JPEG,
            width=32,
            height=32,
        )

    assert error.value.reason_code == 'mask_continuity'
    assert not subject.active
    assert not client.active


def test_first_live_update_requires_nominal_iou_without_centroid_history() -> None:
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:13, 8:16] = True
    borderline = np.zeros_like(stable)
    borderline[7:9, 9:16] = True
    subject, client = mask_tracker([stable, borderline])

    with pytest.raises(TrackerFailure, match='centroid_history_unavailable') as error:
        subject.update(sized_frame(1_000_000_010, width=32, height=32))

    assert error.value.reason_code == 'mask_continuity'
    assert not subject.active
    assert not client.active


def test_borderline_iou_requires_consistent_area_overlap_motion_and_depth() -> None:
    """A single bounded shape change can pass without lowering nominal IoU."""
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:13, 8:16] = True
    borderline = np.zeros_like(stable)
    # area ratio=14/40=0.35, overlap=7/14=0.50, IoU=7/47<0.15.
    borderline[7:9, 9:16] = True
    subject, client = mask_tracker([stable, stable, borderline, stable])

    subject.update(sized_frame(1_000_000_010, width=32, height=32))
    changed = subject.update(sized_frame(2_000_000_010, width=32, height=32))
    recovered = subject.update(sized_frame(3_000_000_010, width=32, height=32))

    assert int(changed.mask.sum()) == 14
    assert int(recovered.mask.sum()) == 40
    assert client.update_calls == [1, 2, 3]
    assert subject.active


def test_borderline_iou_is_bounded_by_stable_anchor_and_frame_budget() -> None:
    """A persistent marginal mask cannot promote itself into a new anchor."""
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:13, 8:16] = True
    borderline = np.zeros_like(stable)
    borderline[7:9, 9:16] = True
    subject, client = mask_tracker(
        [stable, stable, borderline, borderline, borderline],
    )

    subject.update(sized_frame(1_000_000_010, width=32, height=32))
    subject.update(sized_frame(2_000_000_010, width=32, height=32))
    subject.update(sized_frame(3_000_000_010, width=32, height=32))
    with pytest.raises(TrackerFailure, match='soft_frame_budget') as error:
        subject.update(sized_frame(4_000_000_010, width=32, height=32))

    assert error.value.reason_code == 'mask_continuity'
    assert not subject.active
    assert not client.active


def test_borderline_iou_rejects_sparse_depth_geometry() -> None:
    """2-D agreement cannot excuse a mask with too little connected live depth."""
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:13, 8:16] = True
    borderline = np.zeros_like(stable)
    borderline[7:9, 9:16] = True
    subject, client = mask_tracker([stable, stable, borderline])
    subject.update(sized_frame(1_000_000_010, width=32, height=32))
    sparse_depth = np.zeros((32, 32), dtype=np.float32)
    sparse_depth[7, 9:13] = 1.0

    with pytest.raises(TrackerFailure, match='depth_retention') as error:
        subject.update(sized_frame(
            2_000_000_010,
            width=32,
            height=32,
            depth=sparse_depth,
        ))

    assert error.value.reason_code == 'mask_continuity'
    assert not subject.active
    assert not client.active


def test_low_iou_with_weak_overlap_is_not_treated_as_borderline_jitter() -> None:
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:16, 8:16] = True
    shifted = np.zeros_like(stable)
    shifted[8:16, 14:22] = True
    subject, client = mask_tracker([stable, shifted])

    with pytest.raises(TrackerFailure, match='overlap_ratio') as error:
        subject.update(sized_frame(1_000_000_010, width=32, height=32))

    assert error.value.reason_code == 'mask_continuity'
    assert not subject.active
    assert not client.active


def test_low_iou_with_thirty_five_percent_overlap_fails_closed() -> None:
    stable = np.zeros((32, 32), dtype=bool)
    stable[8:13, 8:16] = True
    weak_overlap = np.zeros_like(stable)
    weak_overlap[7:9, 9:19] = True
    subject, client = mask_tracker([stable, weak_overlap])

    with pytest.raises(TrackerFailure, match='overlap_ratio') as error:
        subject.update(sized_frame(1_000_000_010, width=32, height=32))

    assert error.value.reason_code == 'mask_continuity'
    assert not subject.active
    assert not client.active


def test_implausible_three_dimensional_target_jump_fails_closed() -> None:
    """Reject a stable 2-D mask whose measured 3-D centroid teleports."""
    client = FakeClient()
    subject = FailClosedTracker(
        client,
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=2,
        max_points=100,
        max_centroid_speed_mps=0.5,
        session_id_factory=lambda: 'test-session',
    )
    initialize(subject)
    subject.update(frame(1_000_000_010))
    jumped_depth = np.full((5, 6), 2.0, dtype=np.float32)

    with pytest.raises(TrackerFailure, match='centroid continuity broke'):
        subject.update(frame(1_100_000_010, depth=jumped_depth))

    assert not subject.active
    assert not client.active


def test_initialization_session_change_fails_closed() -> None:
    class BadSessionClient(FakeClient):
        """Return a different session identity from initialization."""

        def init(
            self,
            _image_jpeg: bytes,
            bbox_xyxy: tuple[int, int, int, int],
            *,
            session_id: str,
            frame_seq: int = 0,
        ) -> FakeTrack:
            self.active = True
            self.session_id = 'different-session'
            return make_track(
                session_id=self.session_id,
                frame_seq=frame_seq,
                bbox=bbox_xyxy,
            )

    client = BadSessionClient()
    subject = tracker(client)
    with pytest.raises(TrackerFailure, match='initialization failed'):
        initialize(subject)
    assert not subject.active
    assert not client.active


def test_sparse_target_depth_fails_closed_after_service_update() -> None:
    client = FakeClient()
    subject = tracker(client, min_points=4)
    initialize(subject)
    depth = np.zeros((5, 6), dtype=np.float32)
    depth[1, 1] = 1.0

    with pytest.raises(TrackerFailure, match='sparse'):
        subject.update(frame(11, depth=depth))

    assert client.update_calls == [1]
    assert not subject.active
    assert not client.active


def test_transport_exception_is_redacted_and_fails_closed() -> None:
    client = FakeClient()
    subject = tracker(client)
    initialize(subject)
    client.update_error = TimeoutError('secret transport detail')

    with pytest.raises(TrackerFailure, match=r'update failed \(TimeoutError\)') as error:
        subject.update(frame(11))

    assert 'secret transport detail' not in str(error.value)
    assert not subject.active
    assert not client.active


def test_remote_tracking_loss_reason_code_survives_core_redaction() -> None:
    class RemoteTrackingLost(RuntimeError):
        reason_code = 'tracking_lost'

    client = FakeClient()
    subject = tracker(client)
    initialize(subject)
    client.update_error = RemoteTrackingLost('untrusted remote detail')

    with pytest.raises(TrackerFailure, match='RemoteTrackingLost') as error:
        subject.update(frame(11))

    assert error.value.reason_code == 'tracking_lost'
    assert 'untrusted remote detail' not in str(error.value)


def test_result_bbox_must_exactly_bound_mask() -> None:
    class BadBoxClient(FakeClient):
        """Return a bbox that is valid but does not bound its mask."""

        def update(
            self,
            _image_jpeg: bytes,
            *,
            frame_seq: int | None = None,
        ) -> FakeTrack:
            assert frame_seq is not None
            result = make_track(session_id=self.session_id, frame_seq=frame_seq)
            return FakeTrack(
                session_id=result.session_id,
                track_id=result.track_id,
                frame_seq=result.frame_seq,
                image_size=result.image_size,
                bbox_xyxy=(0, 0, 4, 4),
                score=result.score,
                mask=result.mask,
            )

    client = BadBoxClient()
    subject = tracker(client)
    initialize(subject)
    with pytest.raises(TrackerFailure, match='exact half-open'):
        subject.update(frame(11))
    assert not subject.active


@dataclass(frozen=True)
class FakeCoast:
    """Session-preserving keep-alive result mirroring EdgeTamCoast."""

    session_id: str
    track_id: str
    frame_seq: int
    image_size: tuple[int, int]
    coasting: bool = True


class ScriptedClient(FakeClient):
    """Fake client replaying a fixed 'track'/'coast'/'coast_badid' script."""

    def __init__(self, script: list[str]) -> None:
        super().__init__()
        self._script = list(script)
        self._index = 0

    def update(self, _image_jpeg: bytes, *, frame_seq: int | None = None) -> object:
        assert frame_seq is not None
        self.update_calls.append(frame_seq)
        kind = self._script[self._index]
        self._index += 1
        if kind == 'track':
            return make_track(
                session_id=self.session_id,
                frame_seq=frame_seq,
                bbox=self.next_bbox,
                track_id=self.next_track_id,
            )
        if kind == 'coast':
            return FakeCoast(self.session_id, self.next_track_id, frame_seq, (6, 5))
        if kind == 'coast_badid':
            return FakeCoast(self.session_id, 'intruder', frame_seq, (6, 5))
        raise AssertionError(kind)


def _coast_tracker(
    script: list[str],
    *,
    max_coast: int = 32,
) -> tuple[FailClosedTracker, ScriptedClient]:
    client = ScriptedClient(script)
    subject = FailClosedTracker(
        client,
        min_depth_m=0.1,
        max_depth_m=3.0,
        min_points=2,
        max_points=100,
        max_tracking_coast_frames=max_coast,
        session_id_factory=lambda: 'test-session',
    )
    initialize(subject)
    return subject, client


def test_coast_preserves_identity_and_relocks_without_reset() -> None:
    subject, client = _coast_tracker(['track', 'coast', 'track'])

    first = subject.update(frame(11))
    coasted = subject.update(frame(12))
    relock = subject.update(frame(13))

    assert first is not None and relock is not None
    # The coast produced no observation but did not tear the identity down.
    assert coasted is None
    assert subject.active
    assert client.reset_calls == 0
    assert first.track_id == relock.track_id == 'target-7'
    assert first.session_id == relock.session_id
    # The service timeline advanced in strict lockstep across the coast.
    assert client.update_calls == [1, 2, 3]


def test_coast_leaves_validated_anchor_untouched() -> None:
    subject, _client = _coast_tracker(['track', 'coast'])
    subject.update(frame(11))
    anchor_before = subject._last_mask.copy()
    validated_before = subject._last_validated_stamp_ns

    assert subject.update(frame(12)) is None

    assert subject.active
    assert np.array_equal(subject._last_mask, anchor_before)
    # Only the service timeline advances; the validated anchor stamp is frozen.
    assert subject._last_validated_stamp_ns == validated_before
    assert subject._last_stamp_ns == 12


def test_coast_beyond_window_fails_closed_with_typed_reason() -> None:
    subject, client = _coast_tracker(
        ['track', 'coast', 'coast', 'coast'],
        max_coast=2,
    )
    subject.update(frame(11))
    assert subject.update(frame(12)) is None
    assert subject.update(frame(13)) is None

    with pytest.raises(TrackerFailure, match='coasted beyond') as error:
        subject.update(frame(14))

    assert error.value.reason_code == 'tracking_coast_exhausted'
    assert not subject.active
    assert client.reset_calls >= 1


def test_coast_identity_mismatch_is_immediately_terminal() -> None:
    subject, _client = _coast_tracker(['track', 'coast_badid'])
    subject.update(frame(11))

    with pytest.raises(TrackerFailure, match='identity changed'):
        subject.update(frame(12))

    assert not subject.active


@pytest.mark.parametrize('bad', [-1, True])
def test_invalid_coast_window_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match='max_tracking_coast_frames'):
        FailClosedTracker(
            FakeClient(),
            min_points=2,
            max_tracking_coast_frames=bad,
        )
