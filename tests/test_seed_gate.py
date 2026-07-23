import numpy as np
import pytest

from z_manip.perception.seed_gate import (
    BundleGateConfig,
    SeedConfidenceConfig,
    SeedDepthGateConfig,
    SeedDepthMeasurement,
    evaluate_seed_depth,
    has_distance_qualifier,
    has_small_qualifier,
    hygiene_confidence,
    local_corroborates,
    median_depth_in_bbox,
    min_points_for_depth,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("远处箱子上白色充电器", True),
        ("远处黑色箱子上的白色充电器", True),
        ("white charger on the distant box", True),
        ("far away charger", True),
        ("白色充电器", False),
        ("地上的彩色瓶子", False),
        ("the farm charger", False),  # 'far' must be a whole word
    ),
)
def test_distance_qualifier_detection(text, expected):
    assert has_distance_qualifier(text) is expected


def test_small_qualifier_detection():
    assert has_small_qualifier("远处小白色方块")
    assert has_small_qualifier("small white charger")
    assert not has_small_qualifier("白色充电器")


def test_median_depth_excludes_zero_and_out_of_band():
    depth = np.full((100, 100), 0.0, dtype=np.float32)
    depth[40:60, 40:60] = 1.5          # the object
    depth[0:5, 0:5] = 9.0              # far wall spill (out of band)
    m = median_depth_in_bbox(depth, (0.4, 0.4, 0.6, 0.6))
    assert m.median_z_m == pytest.approx(1.5, abs=1e-3)
    assert m.valid_fraction > 0.9


def test_median_depth_abstains_when_no_valid_pixels():
    depth = np.zeros((50, 50), dtype=np.float32)
    m = median_depth_in_bbox(depth, (0.1, 0.1, 0.4, 0.4))
    assert m.median_z_m is None
    assert m.valid_fraction == 0.0


def test_depth_gate_rejects_near_seed_for_distant_instruction():
    cfg = SeedDepthGateConfig()
    m = SeedDepthMeasurement(median_z_m=0.964, valid_fraction=0.9, sampled_pixels=400)
    decision = evaluate_seed_depth(m, "远处箱子上白色充电器", cfg)
    assert decision.accepted is False
    assert decision.distance_qualified is True


def test_depth_gate_retains_genuine_far_seed():
    cfg = SeedDepthGateConfig()
    m = SeedDepthMeasurement(median_z_m=1.5, valid_fraction=0.9, sampled_pixels=400)
    assert evaluate_seed_depth(m, "远处彩色瓶子", cfg).accepted is True


def test_depth_gate_ignores_near_seed_without_distance_word():
    cfg = SeedDepthGateConfig()
    m = SeedDepthMeasurement(median_z_m=0.7, valid_fraction=0.9, sampled_pixels=400)
    assert evaluate_seed_depth(m, "白色充电器", cfg).accepted is True


def test_depth_gate_sanity_band_rejects_absurd_depth():
    cfg = SeedDepthGateConfig()
    m = SeedDepthMeasurement(median_z_m=6.0, valid_fraction=0.9, sampled_pixels=400)
    assert evaluate_seed_depth(m, "白色充电器", cfg).accepted is False


def test_depth_gate_abstains_on_disabled_or_unmeasured():
    disabled = SeedDepthGateConfig(enabled=False)
    m = SeedDepthMeasurement(median_z_m=0.5, valid_fraction=0.9, sampled_pixels=400)
    assert evaluate_seed_depth(m, "远处充电器", disabled).accepted is True
    none = SeedDepthMeasurement(median_z_m=None, valid_fraction=0.0, sampled_pixels=0)
    assert evaluate_seed_depth(none, "远处充电器", SeedDepthGateConfig()).accepted is True


def test_confidence_ceiling_clamps_only_above():
    cfg = SeedConfidenceConfig(ceiling=0.60)
    assert hygiene_confidence(0.99, cfg) == pytest.approx(0.60)
    assert hygiene_confidence(0.30, cfg) == pytest.approx(0.30)
    off = SeedConfidenceConfig(apply_ceiling=False)
    assert hygiene_confidence(0.99, off) == pytest.approx(0.99)


def test_local_corroboration_requires_overlap_and_floor():
    cfg = SeedConfidenceConfig(corroboration_enabled=True, corroboration_floor=0.08)
    vlm = (0.4, 0.4, 0.6, 0.6)
    assert local_corroborates(vlm, [(0.42, 0.42, 0.58, 0.58)], [0.12], cfg)
    assert not local_corroborates(vlm, [(0.0, 0.0, 0.1, 0.1)], [0.5], cfg)
    assert not local_corroborates(vlm, [(0.42, 0.42, 0.58, 0.58)], [0.02], cfg)


def test_min_points_scales_inverse_square_and_clamps():
    cfg = BundleGateConfig(enabled=True, reference_points=400, reference_depth_m=1.3, floor_points=120)
    assert min_points_for_depth(1.3, 400, cfg) == 400        # reference depth == ceiling
    assert min_points_for_depth(0.9, 400, cfg) == 400        # nearer clamps to ceiling
    assert min_points_for_depth(2.0, 400, cfg) == 169        # 400*(1.3/2.0)^2
    assert min_points_for_depth(3.0, 400, cfg) == 120        # floored
    # Disabled or unmeasured => strict near-field ceiling (current behaviour).
    assert min_points_for_depth(2.0, 400, BundleGateConfig(enabled=False)) == 400
    assert min_points_for_depth(None, 400, cfg) == 400


def test_min_points_respects_higher_cli_ceiling():
    cfg = BundleGateConfig(enabled=True)
    # A 1200-point near-field ceiling: at range the reference geometry dominates.
    assert min_points_for_depth(1.5, 1200, cfg) == 300       # 400*(1.3/1.5)^2
    assert min_points_for_depth(1.3, 1200, cfg) == 400
