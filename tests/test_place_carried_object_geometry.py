"""Carried-object geometry must come out of artifacts a real run already wrote.

The fixture holds one recorded pick: the tracked-target cloud, the hand-eye
transform, and the tool pose from that session's planning report.  Nothing here
needs a second sensing pass while the object is held.
"""

from pathlib import Path

import numpy as np
import pytest

from z_manip.models.planner import PlanningError
from z_manip.planning import (
    NormalizedPlacementRegion,
    ObservedPlacementInput,
    ObservedPlacementPlanner,
    PlacementConstraints,
)
from z_manip.verification.place import (
    CARRIED_OBJECT_GEOMETRY_SCHEMA,
    carried_object_geometry_record,
    derive_carried_object_geometry,
    derive_from_session_artifacts,
)


FIXTURE = Path(__file__).resolve().parent / "data/place_target_backprojection.npz"


def _frame():
    return np.load(FIXTURE, allow_pickle=False)


def _report(frame):
    return {
        "base_from_camera": frame["base_from_camera"],
        "grasp_pose": frame["grasp_pose_base"],
        "planning_frame": str(frame["planning_frame"]),
    }


def _geometry(**overrides):
    frame = _frame()
    return derive_from_session_artifacts(
        frame["target_points_camera"], _report(frame), **overrides,
    )


def test_recorded_pick_yields_a_planner_admissible_tool_from_object():
    geometry = _geometry()

    # The placement planner rejects anything that is not right-handed SE(3);
    # a PCA basis is only right-handed after an explicit sign fix.
    rotation = geometry.tool_from_object[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(
        geometry.tool_from_object[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12,
    )
    assert geometry.frame == "piper_base_link"
    assert geometry.observed_point_count == 9473
    assert geometry.tool_pose_source == "planning_report.grasp_pose"

    observation = ObservedPlacementInput(
        organized_points=np.zeros((2, 2, 3)),
        scene_points=np.zeros((0, 3)),
        region=NormalizedPlacementRegion((0.0, 0.0, 1.0, 1.0)),
        constraints=PlacementConstraints(),
        gravity=(0.0, 0.0, -1.0),
        object_extent_m=geometry.object_extent_m,
        tool_from_object=geometry.tool_from_object,
        organized_frame=geometry.frame,
        scene_frame=geometry.frame,
        organized_stamp_s=0.0,
        scene_stamp_s=0.0,
    )
    # The SE(3) and extent gates pass; only the (empty) scene cloud is rejected.
    with pytest.raises(PlanningError, match="scene cloud has too few finite points"):
        ObservedPlacementPlanner().plan(
            observation, current_joints=np.zeros(6), evaluate=lambda *_: None,
        )


def test_object_pose_reconstructs_from_the_tool_pose_it_was_frozen_against():
    frame = _frame()
    geometry = _geometry()

    np.testing.assert_allclose(
        frame["grasp_pose_base"] @ geometry.tool_from_object,
        geometry.object_pose_base,
        atol=1e-9,
    )


def test_single_view_extent_is_inflated_and_both_values_are_carried():
    geometry = _geometry()
    bare = _geometry(extent_margin_m=0.0)

    # One camera view never observes the far face, so the raw oriented box is a
    # lower bound; shipping only the raw value would let a too-small footprint
    # be accepted.
    np.testing.assert_allclose(geometry.observed_extent_m, bare.object_extent_m)
    np.testing.assert_allclose(
        geometry.object_extent_m,
        geometry.observed_extent_m + 2.0 * geometry.extent_margin_m,
    )
    assert np.all(geometry.object_extent_m > geometry.observed_extent_m)
    # The recorded charger measures roughly 0.12 x 0.067 x 0.042 m from this view.
    np.testing.assert_allclose(
        geometry.observed_extent_m, (0.12163, 0.06704, 0.04225), atol=5e-5,
    )


def test_record_names_its_schema_and_does_not_disguise_the_lower_bound():
    geometry = _geometry()
    record = carried_object_geometry_record(
        geometry, sources={"target_points.npy": "sha256:deadbeef"},
    )

    assert record["schema"] == CARRIED_OBJECT_GEOMETRY_SCHEMA
    assert record["extent_is_observed_lower_bound"] is True
    assert record["extent_basis"] == "single_view_oriented_box_plus_margin"
    assert record["extent_margin_m"] == pytest.approx(0.01)
    assert record["observed_extent_m"] != record["object_extent_m"]
    assert record["tool_pose_source"] == "planning_report.grasp_pose"
    assert record["sources"] == {"target_points.npy": "sha256:deadbeef"}
    assert np.shape(record["tool_from_object"]) == (4, 4)
    assert geometry.record["schema"] == CARRIED_OBJECT_GEOMETRY_SCHEMA


def test_derivation_fails_closed_on_unusable_inputs():
    frame = _frame()
    points = frame["target_points_camera"]
    report = _report(frame)

    with pytest.raises(ValueError, match="finite points"):
        derive_from_session_artifacts(points[:10], report)
    with pytest.raises(ValueError, match="shape"):
        derive_from_session_artifacts(points[:, :2], report)
    with pytest.raises(ValueError, match="extent margin"):
        derive_from_session_artifacts(points, report, extent_margin_m=-0.001)
    with pytest.raises(ValueError, match="missing grasp_pose"):
        derive_from_session_artifacts(
            points, {k: v for k, v in report.items() if k != "grasp_pose"},
        )

    skewed = np.array(report["base_from_camera"], dtype=float)
    skewed[:3, :3] *= 1.5
    with pytest.raises(ValueError, match="base_from_camera rotation"):
        derive_carried_object_geometry(
            points,
            base_from_camera=skewed,
            tool_pose_base=report["grasp_pose"],
            tool_pose_source="planning_report.grasp_pose",
            frame="piper_base_link",
        )
    with pytest.raises(ValueError, match="tool-pose source"):
        derive_carried_object_geometry(
            points,
            base_from_camera=report["base_from_camera"],
            tool_pose_base=report["grasp_pose"],
            tool_pose_source="",
            frame="piper_base_link",
        )


def test_degenerate_planar_cloud_is_rejected_before_it_becomes_a_footprint():
    flat = np.zeros((200, 3))
    flat[:, 0] = np.linspace(0.0, 0.1, 200)

    with pytest.raises(ValueError, match="degenerate"):
        derive_carried_object_geometry(
            flat,
            base_from_camera=np.eye(4),
            tool_pose_base=np.eye(4),
            tool_pose_source="unit-test",
            frame="piper_base_link",
            extent_margin_m=0.0,
        )
