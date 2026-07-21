"""Regression contracts for collision geometry in semantic observations."""

from types import SimpleNamespace

import numpy as np

from z_manip_task.node import MobileManipulationRuntime


def test_semantic_observation_filters_shelf_leakage_from_collision_target():
    """Mask-edge shelf samples must not become part of the carried payload."""
    rng = np.random.default_rng(18)
    target = rng.normal(
        loc=(0.52, 0.0, 0.16),
        scale=(0.012, 0.025, 0.055),
        size=(360, 3),
    )
    shelf_leakage = rng.normal(
        loc=(0.78, 0.0, 0.16),
        scale=(0.008, 0.035, 0.07),
        size=(48, 3),
    )
    raw_mask_cloud = np.vstack((target, shelf_leakage))

    runtime = MobileManipulationRuntime.__new__(MobileManipulationRuntime)
    runtime._target_cloud = raw_mask_cloud
    runtime._target_uv = np.tile((320.0, 240.0), (len(raw_mask_cloud), 1))
    runtime._scene_cloud = np.array(((1.2, 0.0, 0.0),), dtype=float)
    runtime._target_camera = np.array((0.0, 0.0, 0.52), dtype=float)
    runtime._camera_origin_piper = np.zeros(3, dtype=float)
    runtime._camera_rotation_piper = np.eye(3, dtype=float)
    runtime._joint_state = SimpleNamespace(position=(0.0,) * 6)
    runtime._affordance = {}
    runtime._image_size = (640, 480)
    runtime.get_parameter = lambda name: SimpleNamespace(
        value=40 if name == 'semantic_min_points' else None,
    )

    observation = MobileManipulationRuntime._semantic_observation(
        runtime,
        serial=7,
        stamp_s=12.5,
    )

    # Semantic selection still sees the complete aligned mask. Collision and
    # attachment geometry uses the dominant, locally supported object layer.
    assert len(observation.target_points) == len(raw_mask_cloud)
    assert 320 <= len(observation.target_collision_points) <= 380
    assert np.max(observation.target_collision_points[:, 0]) < 0.60
    assert np.max(observation.target_points[:, 0]) > 0.74
