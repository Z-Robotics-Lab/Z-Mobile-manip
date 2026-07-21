import numpy as np
import pytest

from z_manip.models.planner import PlanningError
from z_manip.planning import (
    NormalizedPlacementRegion,
    ObservedPlacementConfig,
    ObservedPlacementInput,
    ObservedPlacementPlanner,
    PlacementConstraints,
    PlacementMotionEvaluation,
)
from z_manip.planning_control import PlanningCancelled, PlanningControl


def _basis(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return normal, u, v


def _observed_plane(
    *,
    normal=(0.0, 0.0, 1.0),
    gravity=None,
    half_size=0.28,
    resolution=31,
    extent=(0.08, 0.06, 0.10),
    obstacles=(),
    region=None,
    constraints=None,
    stamp_skew=0.01,
):
    normal, u, v = _basis(normal)
    gravity = -normal if gravity is None else np.asarray(gravity, dtype=float)
    coordinates = np.linspace(-half_size, half_size, resolution)
    organized = np.empty((resolution, resolution, 3), dtype=float)
    for row, y in enumerate(coordinates):
        for column, x in enumerate(coordinates):
            organized[row, column] = x * u + y * v
    scene = organized.reshape(-1, 3).copy()
    obstacle_clouds = []
    for center, size, height in obstacles:
        center = np.asarray(center, dtype=float)
        offsets = np.linspace(-size / 2.0, size / 2.0, 5)
        obstacle_clouds.append(np.asarray([
            center + dx * u + dy * v + z * normal
            for dx in offsets for dy in offsets
            for z in np.linspace(0.02, height, 5)
        ]))
    if obstacle_clouds:
        scene = np.vstack((scene, *obstacle_clouds))
    return ObservedPlacementInput(
        organized_points=organized,
        scene_points=scene,
        region=region or NormalizedPlacementRegion((0.0, 0.0, 1.0, 1.0)),
        constraints=constraints or PlacementConstraints(min_clearance_m=0.025),
        gravity=gravity,
        object_extent_m=np.asarray(extent, dtype=float),
        tool_from_object=np.eye(4),
        organized_frame="arm_base",
        scene_frame="arm_base",
        organized_stamp_s=12.0,
        scene_stamp_s=12.0 + stamp_skew,
    )


def _planner(**overrides):
    values = dict(
        ransac_iterations=100,
        min_plane_points=60,
        sample_spacing_m=0.04,
        support_neighbor_radius_m=0.03,
        footprint_samples_per_axis=4,
        yaw_samples=4,
        max_geometric_candidates=40,
    )
    values.update(overrides)
    return ObservedPlacementPlanner(ObservedPlacementConfig(**values))


def _accept(candidate, _current):
    return PlacementMotionEvaluation(score=1.0, motion={"yaw": candidate.yaw_rad})


def test_rotated_gravity_plane_generates_surface_normal_entry_and_yaw_family():
    normal = np.array((0.0, 1.0, 1.0)) / np.sqrt(2.0)
    observation = _observed_plane(normal=normal)
    seen_rotations = []

    def evaluate(candidate, _current):
        seen_rotations.append(candidate.place_pose[:3, :3].copy())
        return {"score": 2.0 - abs(candidate.yaw_rad - np.pi / 2.0)}

    result = _planner().plan(observation, current_joints=np.zeros(6), evaluate=evaluate)

    np.testing.assert_allclose(result.plane.normal, normal, atol=1e-6)
    delta = result.candidate.preplace_pose[:3, 3] - result.candidate.place_pose[:3, 3]
    np.testing.assert_allclose(delta / np.linalg.norm(delta), normal, atol=1e-6)
    assert len(seen_rotations) > 1
    assert any(not np.allclose(rotation, seen_rotations[0]) for rotation in seen_rotations[1:])
    assert np.dot(result.candidate.place_pose[:3, 2], normal) == pytest.approx(1.0)


def test_scene_kdtree_keeps_placement_footprint_away_from_observed_obstacle():
    obstacle_center = np.array((-0.12, 0.0, 0.0))
    observation = _observed_plane(
        obstacles=((obstacle_center, 0.10, 0.16),),
        extent=(0.10, 0.08, 0.12),
    )

    result = _planner().plan(observation, current_joints=np.zeros(6), evaluate=_accept)

    planar_distance = np.linalg.norm(
        (result.candidate.support_position - obstacle_center)[:2],
    )
    assert planar_distance > 0.10
    assert result.candidate.obstacle_clearance_m >= 0.025


def test_vlm_avoid_region_removes_support_and_motion_evaluator_ranks_survivors():
    observation = _observed_plane(
        region=NormalizedPlacementRegion(
            (0.0, 0.0, 1.0, 1.0),
            avoid_xyxy=((0.40, 0.40, 0.60, 0.60),),
        ),
        constraints=PlacementConstraints(
            min_clearance_m=0.01,
            preferred_yaw_rad=0.7,
            yaw_tolerance_rad=0.0,
        ),
    )
    rejected = 0

    def evaluate(candidate, _current):
        nonlocal rejected
        if candidate.support_position[0] < 0.0:
            rejected += 1
            raise PlanningError("injected continuous-collision check rejected path")
        return PlacementMotionEvaluation(score=float(candidate.support_position[0]))

    result = _planner().plan(observation, current_joints=np.zeros(6), evaluate=evaluate)

    assert rejected > 0
    assert result.rejected_by_motion == rejected
    assert result.candidate.support_position[0] >= 0.0
    assert result.candidate.yaw_rad == pytest.approx(0.7)
    # The normalized avoid region creates a support hole around the image center.
    assert np.linalg.norm(result.candidate.support_position[:2]) > 0.05


def test_object_scale_changes_available_space_without_category_hardcoding():
    small = _observed_plane(half_size=0.16, extent=(0.07, 0.05, 0.08))
    large = _observed_plane(half_size=0.16, extent=(0.42, 0.35, 0.18))
    planner = _planner(sample_spacing_m=0.03)

    result = planner.plan(small, current_joints=np.zeros(6), evaluate=_accept)
    assert result.candidate.support_fraction == pytest.approx(1.0)
    with pytest.raises(PlanningError, match="full boundary support"):
        planner.plan(large, current_joints=np.zeros(6), evaluate=_accept)


def test_no_space_between_obstacles_fails_closed_before_motion_evaluation():
    obstacles = (
        ((-0.13, -0.13, 0.0), 0.20, 0.18),
        ((-0.13, 0.13, 0.0), 0.20, 0.18),
        ((0.13, -0.13, 0.0), 0.20, 0.18),
        ((0.13, 0.13, 0.0), 0.20, 0.18),
    )
    observation = _observed_plane(
        half_size=0.23,
        extent=(0.12, 0.10, 0.12),
        obstacles=obstacles,
        constraints=PlacementConstraints(min_clearance_m=0.035),
    )
    calls = 0

    def evaluate(*_args):
        nonlocal calls
        calls += 1
        return {"score": 1.0}

    with pytest.raises(PlanningError, match="observed obstacle clearance"):
        _planner().plan(observation, current_joints=np.zeros(6), evaluate=evaluate)
    assert calls == 0


def test_unsynchronized_or_non_supporting_observations_fail_closed():
    with pytest.raises(PlanningError, match="not synchronized"):
        _planner().plan(
            _observed_plane(stamp_skew=0.2),
            current_joints=np.zeros(6),
            evaluate=_accept,
        )

    horizontal = _observed_plane(
        normal=(0.0, 0.0, 1.0),
        gravity=(1.0, 0.0, 0.0),
    )
    with pytest.raises(PlanningError, match="gravity-consistent"):
        _planner().plan(horizontal, current_joints=np.zeros(6), evaluate=_accept)


def test_motion_layer_must_return_finite_score_and_can_reject_every_pose():
    observation = _observed_plane()
    with pytest.raises(PlanningError, match="survived IK/collision/motion"):
        _planner().plan(
            observation,
            current_joints=np.zeros(6),
            evaluate=lambda *_args: (_ for _ in ()).throw(PlanningError("no IK")),
        )
    with pytest.raises(PlanningError, match="survived IK/collision/motion"):
        _planner().plan(
            observation,
            current_joints=np.zeros(6),
            evaluate=lambda *_args: {"trajectory": "missing score"},
        )


def test_cancellation_stops_candidate_evaluation_before_the_next_moveit_call():
    observation = _observed_plane()
    calls = 0

    def evaluate(candidate, current):
        nonlocal calls
        calls += 1
        return _accept(candidate, current)

    control = PlanningControl(cancel_check=lambda: calls >= 1)
    with pytest.raises(PlanningCancelled, match="cancelled"):
        _planner().plan(
            observation,
            current_joints=np.zeros(6),
            evaluate=evaluate,
            control=control,
        )
    assert calls == 1
