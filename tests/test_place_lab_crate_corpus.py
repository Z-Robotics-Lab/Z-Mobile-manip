"""Replay one recorded lab frame of the real crate through the placement planner.

``tests/data/place_lab_crate_scene.npz`` is the collision cloud the live
perception pass published for session 20260717-161213, mapped through that
session's hand-eye transform onto its stride-2 image lattice.  The scene is the
white lattice storage crate the charger was picked from: its interior floor is
the support plane and its walls are what the gripper has to clear.  The carried
object is the charger, derived from the same session's tracked-target cloud.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from z_manip.models.planner import PlanningError
from z_manip.planning.placement import (
    NormalizedPlacementRegion,
    ObservedPlacementInput,
    ObservedPlacementPlanner,
    PlacementConstraints,
    PlacementDeploymentConfig,
    PlacementMotionEvaluation,
)
from z_manip.verification.place import derive_carried_object_geometry


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
CRATE = DATA / "place_lab_crate_scene.npz"
PICK = DATA / "place_target_backprojection.npz"

# The crate interior, in normalized image coordinates of the recorded frame.
CRATE_INTERIOR = NormalizedPlacementRegion((0.10, 0.20, 0.90, 0.85))


def _deployment():
    values = json.loads((ROOT / "configs/go2w_piper.json").read_text())
    return PlacementDeploymentConfig.from_mapping(values["placement"])


def _search_config(**overrides):
    # Reduced search: the 24-candidate cap is the deployed value and keeps the
    # geometry stage well inside the grasp stage's measured 1.82 s budget.
    return _deployment().planner_config(
        ransac_iterations=120,
        sample_spacing_m=0.05,
        yaw_samples=4,
        footprint_samples_per_axis=4,
        **overrides,
    )


def _carried():
    pick = np.load(PICK, allow_pickle=False)
    return derive_carried_object_geometry(
        pick["target_points_camera"],
        base_from_camera=pick["base_from_camera"],
        tool_pose_base=pick["grasp_pose_base"],
        tool_pose_source="planning_report.grasp_pose",
        frame="piper_base_link",
    )


def _scene():
    crate = np.load(CRATE, allow_pickle=False)
    organized = crate["organized_points"].astype(float)
    return organized, organized[np.all(np.isfinite(organized), axis=2)]


def _observation(*, tool_from_object=None, region=CRATE_INTERIOR, extent=None):
    organized, cloud = _scene()
    carried = _carried()
    return ObservedPlacementInput(
        organized_points=organized,
        scene_points=cloud,
        region=region,
        constraints=PlacementConstraints(min_clearance_m=0.02),
        gravity=(0.0, 0.0, -1.0),
        object_extent_m=carried.object_extent_m if extent is None else extent,
        tool_from_object=(
            carried.tool_from_object if tool_from_object is None else tool_from_object
        ),
        organized_frame="piper_base_link",
        scene_frame="piper_base_link",
        organized_stamp_s=1.0,
        scene_stamp_s=1.0,
    )


def _plan(config=None, **observation_kwargs):
    planner = ObservedPlacementPlanner(config or _search_config())
    return planner.plan(
        _observation(**observation_kwargs),
        current_joints=np.zeros(6),
        evaluate=lambda candidate, _current: PlacementMotionEvaluation(score=1.0),
    )


def _off_plane_points(cloud, plane, config):
    height = np.abs((cloud - plane.origin) @ plane.normal)
    return cloud[height > config.plane_exclusion_m]


def test_recorded_crate_floor_is_fitted_as_the_support_plane():
    result = _plan()
    plane = result.plane

    # Gravity-up in the arm base frame, tilted only by the platform pose.
    assert float(np.dot(plane.normal, (0.0, 0.0, 1.0))) > np.cos(np.deg2rad(10.0))
    assert plane.inlier_count > 10_000
    assert plane.rms_error_m < 0.006
    assert plane.inlier_polygon_uv is not None
    assert plane.inlier_polygon_uv.shape[1] == 3


def test_recorded_crate_yields_a_candidate_whose_probe_points_clear_the_walls():
    config = _search_config()
    _, cloud = _scene()
    result = _plan(config)
    candidate = result.candidate

    probes = np.asarray(config.tool_probe_points_m, dtype=float)
    assert len(probes) == 6
    world = probes @ candidate.place_pose[:3, :3].T + candidate.place_pose[:3, 3]
    walls = _off_plane_points(cloud, result.plane, config)
    assert len(walls) > 1_000
    gaps, _ = cKDTree(walls).query(world, k=1)

    assert float(gaps.min()) >= config.tool_probe_clearance_m
    # Measured on this frame: the winning pose sits ~18 mm off the crate walls.
    assert float(gaps.min()) > 0.015


def test_probe_points_are_evaluated_against_the_real_wall_geometry():
    # An inflated probe radius must shrink the admissible set, otherwise the
    # probe model is inert config rather than a working clearance test.
    wide = _plan(_search_config(
        tool_probe_clearance_m=0.05, max_geometric_candidates=100_000,
    ))
    deployed = _plan(_search_config(max_geometric_candidates=100_000))

    assert wide.geometric_candidates < deployed.geometric_candidates


def test_release_clearance_lifts_the_tool_above_the_fitted_crate_floor():
    config = _search_config()
    contact = _search_config(release_clearance_m=0.0)
    carried = _carried()

    raised = _plan(config).candidate
    resting = _plan(contact).candidate
    normal = raised.surface_normal

    lift = float((raised.place_pose[:3, 3] - resting.place_pose[:3, 3]) @ normal)
    assert lift == pytest.approx(config.release_clearance_m, abs=1e-9)
    # object_pose keeps meaning "resting on the fitted plane"; the object the
    # tool is still holding at place_pose is that pose raised by the clearance.
    assert float(
        (raised.object_pose[:3, 3] - resting.object_pose[:3, 3]) @ normal
    ) == pytest.approx(0.0, abs=1e-12)
    held = raised.place_pose @ carried.tool_from_object
    assert float(
        (held[:3, 3] - raised.object_pose[:3, 3]) @ normal
    ) == pytest.approx(config.release_clearance_m, abs=1e-9)
    # The object bottom is above the observed floor, not pressed into it.
    floor_gap = float(
        (held[:3, 3] - raised.support_position) @ normal
    ) - 0.5 * carried.object_extent_m[2]
    assert floor_gap == pytest.approx(config.release_clearance_m, abs=1e-9)


def test_geometry_stage_stays_inside_the_grasp_stage_budget():
    config = _search_config()
    observation = _observation()
    planner = ObservedPlacementPlanner(config)

    started = time.perf_counter()
    planner.plan(
        observation,
        current_joints=np.zeros(6),
        evaluate=lambda candidate, _current: PlacementMotionEvaluation(score=1.0),
    )
    elapsed = time.perf_counter() - started

    # Measured ~0.10 s on this frame at the deployed 24-candidate cap; the bar
    # is loose enough for a loaded machine but catches an order-of-magnitude
    # regression against the grasp stage's measured 1.82 s.
    assert elapsed < 1.0


def test_a_place_pose_outside_the_observed_inlier_polygon_is_refused():
    carried = _carried()
    # Same held object, but the tool now stands 0.8 m to one side of it, so
    # every supported release drives the gripper over plane the sensor never
    # measured.
    displaced = carried.tool_from_object.copy()
    displaced[:3, 3] += displaced[:3, :3] @ np.asarray((0.8, 0.0, 0.0))

    with pytest.raises(PlanningError, match="outside the observed support-plane"):
        _plan(tool_from_object=displaced)


def test_an_object_larger_than_the_crate_floor_fails_closed():
    with pytest.raises(PlanningError, match="full boundary support"):
        _plan(extent=np.asarray((0.9, 0.8, 0.1)))
