import numpy as np
import pytest

from z_manip.perception.rgbd import temporal_median_depth


def test_temporal_median_depth_rejects_flicker_and_fills_single_dropouts():
    frames = np.full((5, 2, 3), 500, dtype=np.uint16)
    frames[:, 0, 0] = (500, 540, 470, 535, 465)
    frames[2, 0, 1] = 0
    frames[:3, 1, 2] = 0

    filtered, report = temporal_median_depth(
        frames,
        min_valid_fraction=0.6,
        max_mad_mm=8.0,
    )

    assert filtered[0, 0] == 0
    assert filtered[0, 1] == 500
    assert filtered[1, 2] == 0
    assert report["frame_count"] == 5
    assert report["minimum_observations"] == 3
    assert report["rejected_unstable_pixels"] == 1
    assert report["rejected_low_support_pixels"] == 1


@pytest.mark.parametrize(
    "frames,fraction,mad",
    [
        (np.zeros((2, 3, 3)), 0.6, 8.0),
        (np.zeros((3, 3, 3)), 0.0, 8.0),
        (np.zeros((3, 3, 3)), 0.6, 0.0),
    ],
)
def test_temporal_median_depth_rejects_invalid_configuration(frames, fraction, mad):
    with pytest.raises(ValueError):
        temporal_median_depth(frames, min_valid_fraction=fraction, max_mad_mm=mad)
