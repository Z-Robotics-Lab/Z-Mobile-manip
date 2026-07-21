"""Tests for VLM semantic constraints on exact-mask point clouds."""

import numpy as np
import pytest

from z_manip_task.planning import select_semantic_target_points


def _grid():
    u, v = np.meshgrid(np.arange(10), np.arange(10))
    uv = np.column_stack((u.ravel(), v.ravel()))
    xyz = np.column_stack((uv * 0.01, np.ones(100)))
    return xyz, uv


def test_grasp_part_is_selected_and_avoid_region_removed():
    xyz, uv = _grid()
    affordance = {
        'grasp_part': {'bbox_xyxy_normalized': [0.0, 0.0, 0.6, 1.0]},
        'avoid_regions': [
            {'bbox_xyxy_normalized': [0.0, 0.0, 0.2, 1.0]},
        ],
    }
    result = select_semantic_target_points(
        xyz, uv, affordance, image_width=10, image_height=10, min_points=20,
    )
    assert result.mode == 'vlm_grasp_part'
    assert np.all(result.points[:, 0] >= 0.02)


def test_small_grasp_part_reports_explicit_fallback():
    xyz, uv = _grid()
    affordance = {
        'grasp_part': {'bbox_xyxy_normalized': [0.0, 0.0, 0.1, 0.1]},
        'avoid_regions': [],
    }
    result = select_semantic_target_points(
        xyz, uv, affordance, image_width=10, image_height=10, min_points=30,
    )
    assert result.mode.startswith('full_target_grasp_part_below_min:')
    assert result.selected_count == 100


def test_avoid_regions_never_fall_back_to_unsafe_pixels():
    xyz, uv = _grid()
    affordance = {
        'grasp_part': None,
        'avoid_regions': [
            {'bbox_xyxy_normalized': [0.0, 0.0, 0.9, 1.0]},
        ],
    }
    with pytest.raises(ValueError, match='avoid regions'):
        select_semantic_target_points(
            xyz, uv, affordance, image_width=10, image_height=10, min_points=30,
        )


def test_vlm_regions_follow_current_tracked_target_scale_and_translation():
    u, v = np.meshgrid(np.arange(40, 60), np.arange(30, 70))
    uv = np.column_stack((u.ravel(), v.ravel()))
    xyz = np.column_stack((uv * 0.001, np.ones(len(uv))))
    affordance = {
        'target': {'bbox_xyxy_normalized': [0.1, 0.2, 0.3, 0.6]},
        'grasp_part': {'bbox_xyxy_normalized': [0.2, 0.2, 0.3, 0.6]},
        'avoid_regions': [],
    }
    result = select_semantic_target_points(
        xyz, uv, affordance, image_width=100, image_height=100, min_points=100,
    )
    assert result.mode == 'vlm_grasp_part'
    assert np.min(result.points[:, 0]) >= 0.050
