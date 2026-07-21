from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_planning_dry_run.py"
SPEC = importlib.util.spec_from_file_location("piper_planning_dry_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DRY_RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRY_RUN)


def test_load_transform_and_transform_geometry(tmp_path):
    transform = np.eye(4)
    transform[:3, 3] = (1.0, -2.0, 0.5)
    path = tmp_path / "transform.json"
    path.write_text(json.dumps({"base_from_camera": transform.tolist()}))

    loaded = DRY_RUN.load_transform(path)
    points = DRY_RUN.transform_points([[0.1, 0.2, 0.3]], loaded)
    poses = DRY_RUN.transform_poses(np.eye(4)[None, :, :], loaded)

    np.testing.assert_allclose(loaded, transform)
    np.testing.assert_allclose(points[0], (1.1, -1.8, 0.8))
    np.testing.assert_allclose(poses[0], transform)


def test_support_approach_prior_prefers_direction_opposite_lift():
    grasps = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    grasps[0, :3, 2] = (0.0, 0.0, -1.0)
    grasps[1, :3, 2] = (1.0, 0.0, 0.0)
    grasps[2, :3, 2] = (0.0, 0.0, 1.0)

    ranked, bonuses = DRY_RUN.support_approach_scores(
        grasps,
        np.ones(3),
        (0.0, 0.0, 1.0),
        weight=0.15,
    )

    np.testing.assert_allclose(bonuses, (0.15, 0.0, 0.0))
    np.testing.assert_allclose(ranked, (1.15, 1.0, 1.0))


def test_support_approach_prior_is_optional_and_bounded():
    grasp = np.eye(4)[None, :, :]
    ranked, bonuses = DRY_RUN.support_approach_scores(
        grasp,
        (0.8,),
        (0.0, 0.0, 1.0),
        weight=0.0,
    )
    np.testing.assert_allclose(ranked, (0.8,))
    np.testing.assert_allclose(bonuses, (0.0,))

    with pytest.raises(ValueError, match=r"within \[0, 0.5\]"):
        DRY_RUN.support_approach_scores(
            grasp,
            (0.8,),
            (0.0, 0.0, 1.0),
            weight=0.6,
        )


def test_support_approach_prior_does_not_promote_edge_grasp_over_central_grasp():
    grasps = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    grasps[:, :3, 2] = (0.0, 0.0, -1.0)

    ranked, bonuses = DRY_RUN.support_approach_scores(
        grasps,
        (1.06, 1.02),
        (0.0, 0.0, 1.0),
        weight=0.5,
        centralities=(0.30, 0.95),
    )

    assert ranked[1] > ranked[0]
    np.testing.assert_allclose(bonuses, (0.15, 0.475))


def test_grasp_centrality_scores_object_center_above_edge():
    x, y, z = np.meshgrid(
        np.linspace(-0.012, 0.012, 5),
        np.linspace(-0.022, 0.022, 7),
        np.linspace(-0.008, 0.008, 3),
    )
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    grasps = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    grasps[1, :3, 3] = (0.010, 0.016, 0.0)

    centralities = DRY_RUN.grasp_centrality_scores(
        grasps,
        (0.0, 0.0, 0.0),
        points,
    )

    assert centralities[0] == pytest.approx(1.0)
    assert centralities[1] < 0.6


def test_closest_segment_witness_reports_both_points():
    cloud = np.asarray(((0.5, 0.2, 0.0), (2.0, 0.0, 0.0)))

    scene, capsule, distance = DRY_RUN._closest_segment_witness(
        cloud,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(scene, (0.5, 0.2, 0.0))
    np.testing.assert_allclose(capsule, (0.5, 0.0, 0.0))
    assert distance == pytest.approx(0.2)


def test_rigid_pose_error_reports_translation_and_geodesic_rotation():
    actual = np.eye(4)
    target = np.eye(4)
    target[:3, 3] = (0.003, -0.004, 0.0)
    angle = np.deg2rad(20.0)
    target[:3, :3] = (
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )

    position, orientation = DRY_RUN.rigid_pose_error(actual, target)

    assert position == pytest.approx(0.005)
    assert orientation == pytest.approx(angle)


def test_planning_only_d435_noise_defaults_are_practical_and_bounded(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "piper_planning_dry_run.py",
            "--artifacts", "/tmp/a",
            "--config", "/tmp/c",
            "--urdf", "/tmp/u",
            "--joints", "0,0,0,0,0,0",
            "--camera-calibration", "/tmp/calibration.json",
            "--output", "/tmp/o",
        ],
    )
    arguments = DRY_RUN._arguments()
    assert arguments.scene_noise_tolerance_m == pytest.approx(0.003)
    assert arguments.scene_noise_min_support_points == 2


def test_scene_uncertainty_uses_sensor_noise_and_hand_eye_residuals():
    report = {"temporal_depth_filter": {"frame_count": 9, "mad_p95_mm": 1.0}}
    calibration = {"translation_rmse_m": 0.006, "rotation_rmse_rad": 0.018}
    target = np.array(((0.0, 0.0, 0.4), (0.01, 0.0, 0.42)))

    result = DRY_RUN.calibrated_scene_uncertainty(report, calibration, target)

    assert result["temporal_frame_count"] == 9
    assert result["sensor_three_sigma_m"] == pytest.approx(0.003)
    assert result["target_depth_m"] == pytest.approx(0.41)
    assert 0.009 < result["applied_clearance_m"] < 0.011


def test_scene_uncertainty_never_exceeds_validated_piper_clearance():
    result = DRY_RUN.calibrated_scene_uncertainty(
        {"temporal_depth_filter": {"frame_count": 5, "mad_p95_mm": 2.0}},
        {"translation_rmse_m": 0.012, "rotation_rmse_rad": 0.03},
        np.array(((0.0, 0.0, 0.4), (0.0, 0.0, 0.42))),
    )

    assert result["raw_clearance_m"] > 0.010
    assert result["maximum_clearance_m"] == pytest.approx(0.010)
    assert result["applied_clearance_m"] == pytest.approx(0.010)


def test_supervised_scene_clearance_is_bounded():
    assert DRY_RUN._supervised_scene_clearance("0.004") == pytest.approx(0.004)
    assert DRY_RUN._supervised_scene_clearance("0.010") == pytest.approx(0.010)
    with pytest.raises(Exception, match="within"):
        DRY_RUN._supervised_scene_clearance("0.0005")


def test_failure_records_attach_matching_collision_witness():
    class Failure:
        candidate_index = 3
        symmetry_index = 2
        stage = "approach_collision"
        reason = "blocked"

    records = DRY_RUN._failure_records(
        (Failure(),),
        {(3, 2): {"schema": "z_manip.collision_witness.v1"}},
    )

    assert records[0]["collision_witness"]["schema"] == "z_manip.collision_witness.v1"


def test_load_transform_fails_closed_on_invalid_rotation(tmp_path):
    transform = np.eye(4)
    transform[0, 0] = 2.0
    path = tmp_path / "invalid.npy"
    np.save(path, transform)

    with pytest.raises(ValueError, match="orthonormal"):
        DRY_RUN.load_transform(path)


def test_real_camera_transform_requires_explicit_calibration_metadata(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"base_from_camera": np.eye(4).tolist()}))

    with pytest.raises(ValueError, match="calibrated=true"):
        DRY_RUN.load_transform(path, require_calibrated=True)

    path.write_text(json.dumps({
        "calibrated": True,
        "base_from_camera": np.eye(4).tolist(),
    }))
    np.testing.assert_allclose(
        DRY_RUN.load_transform(path, require_calibrated=True),
        np.eye(4),
    )


def test_eye_in_hand_calibration_is_composed_with_current_fk(tmp_path):
    class Chain:
        tip_link = "piper_gripper_base"

        def forward(self, joints):
            assert np.asarray(joints).shape == (6,)
            value = np.eye(4)
            value[:3, 3] = (1.0, 2.0, 3.0)
            return value

    tip_from_camera = np.eye(4)
    tip_from_camera[:3, 3] = (0.1, -0.2, 0.3)
    path = tmp_path / "hand_eye.json"
    path.write_text(json.dumps({
        "schema": "z_manip.piper_camera_calibration.v1",
        "calibrated": True,
        "synthetic": False,
        "calibration_id": "test-calibration",
        "mount_type": "eye_in_hand",
        "tip_link": "piper_gripper_base",
        "camera_frame": "camera_color_optical_frame",
        "tip_from_camera": tip_from_camera.tolist(),
        "sample_count": 8,
        "quality": {
            "rotation_axis_rank": 3,
            "max_pair_rotation_rad": 0.8,
            "translation_rmse_m": 0.001,
            "rotation_rmse_rad": 0.01,
        },
        "quality_limits": {
            "min_samples": 8,
            "min_rotation_axis_rank": 2,
            "min_rotation_span_rad": 0.35,
            "max_translation_rmse_m": 0.01,
            "max_rotation_rmse_rad": 0.035,
        },
    }))

    resolved, metadata = DRY_RUN.resolve_base_from_camera(
        path,
        real_camera=True,
        chain=Chain(),
        joints=np.zeros(6),
        source_frame="camera_color_optical_frame",
    )

    expected = np.eye(4)
    expected[:3, 3] = (1.1, 1.8, 3.3)
    np.testing.assert_allclose(resolved, expected)
    assert metadata["mount_type"] == "eye_in_hand"


def test_real_planning_rejects_synthetic_or_failed_quality_calibration(tmp_path):
    class Chain:
        tip_link = "piper_gripper_base"

        def forward(self, _joints):
            return np.eye(4)

    document = {
        "schema": "z_manip.piper_camera_calibration.v1",
        "calibrated": True,
        "synthetic": True,
        "calibration_id": "synthetic-test",
        "mount_type": "eye_in_hand",
        "tip_link": "piper_gripper_base",
        "camera_frame": "camera_color_optical_frame",
        "tip_from_camera": np.eye(4).tolist(),
        "sample_count": 8,
        "quality": {
            "rotation_axis_rank": 3,
            "max_pair_rotation_rad": 0.8,
            "translation_rmse_m": 0.001,
            "rotation_rmse_rad": 0.01,
        },
        "quality_limits": {
            "min_samples": 8,
            "min_rotation_axis_rank": 2,
            "min_rotation_span_rad": 0.35,
            "max_translation_rmse_m": 0.01,
            "max_rotation_rmse_rad": 0.035,
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="synthetic"):
        DRY_RUN.resolve_base_from_camera(
            path,
            real_camera=True,
            chain=Chain(),
            joints=np.zeros(6),
            source_frame="camera_color_optical_frame",
        )

    document["synthetic"] = False
    document["quality"]["translation_rmse_m"] = 0.02
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="quality gates"):
        DRY_RUN.resolve_base_from_camera(
            path,
            real_camera=True,
            chain=Chain(),
            joints=np.zeros(6),
            source_frame="camera_color_optical_frame",
        )


def test_planning_dry_run_has_no_ros_or_transport_calls():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {"rclpy", "can", "piper_sdk", "pyAgxArm"}
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    forbidden_calls = {"publish", "send", "sendall", "sendmsg", "sendto"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_calls
    }

    assert imported.isdisjoint(forbidden_imports)
    assert calls == set()
    assert "PlanningControl(deadline_s=config.grasp_plan.search_timeout_s)" not in source
    assert "PlanningControl().limited_to(" in source
    assert "args.search_timeout_s" in source
    assert "args.symmetry_samples" in source
    assert "args.max_hypotheses" in source
    assert "config.grasp_plan.hypothesis_timeout_s" in source
    assert "planned.failures[-12:]" not in source
    assert '"rejections_truncated": False' in source
    assert "transit_times_s" in source
    assert "transit_raw=raw_transit" in source
    assert "approach_raw=raw_approach" in source
    assert "lift_raw=raw_lift" in source
    assert '"raw_paths_collision_validated": True' in source
    assert '"planned_grasp_sha256": planned_grasp_sha256' in source
    assert '"selected_global_rank": int(planned.selected_global_rank)' in source
    assert 'planned.higher_rank_rejection_count' in source
    assert '"transit_duration_s"' in source
