from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "ros2" / "z_manip_edgetam" / "z_manip_edgetam" / "core.py"
NODE = ROOT / "ros2" / "z_manip_edgetam" / "z_manip_edgetam" / "node.py"
DRY_RUN = ROOT / "scripts" / "runtime" / "go2w_perception_dry_run.py"


def test_target_and_scene_clouds_project_the_same_filtered_rgbd_frame():
    core = CORE.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")

    # The stabilized depth is produced by the dedicated depth worker and
    # joined back by exact source stamp before any projection consumes it.
    assert "depth_m, report = self._depth_filter.update(" in node
    assert "depth_m=depth_m," in node
    assert "depth_filter=report," in node
    assert "_join_filtered_depth(" in node
    scene_projection = (
        "project_scene_depth(\n            observation.mask,\n            frame.depth_m,"
    )
    target_projection = (
        "project_mask_depth_geometry(\n"
        "                current_mask,\n                frame.depth_m,"
    )
    assert scene_projection in node
    assert target_projection in core
    assert "'applied_to': ['target_pointcloud', 'scene_pointcloud']" in core
    assert "'depth_filter': frame.depth_filter" in node


def test_planning_snapshot_consumes_filtered_cloud_pair_not_raw_depth():
    source = DRY_RUN.read_text(encoding="utf-8")

    assert '"/z_manip/perception/target_pointcloud"' in source
    assert '"/z_manip/perception/scene_pointcloud"' in source
    assert '"/track_3d/frame_manifest"' in source
    assert '"scene_source": "edgetam_motion_adaptive_filtered_scene"' in source
    assert "depth_to_scene_cloud" not in source
    assert "temporal_median_depth" not in source
    assert '"/camera/aligned_depth_to_color/image_raw"' not in source


def test_planning_snapshot_filters_mask_edge_outliers_before_grasp_geometry():
    source = DRY_RUN.read_text(encoding="utf-8")

    assert "filter_object_cloud," in source
    assert "filtered_points = filter_object_cloud(" in source
    assert "object_points=filtered_points," in source
    assert "target_exclusion_mask(\n        pixel_excluded_scene_points,\n        filtered_points," in source
    assert 'np.save(args.output / "target_points.npy", filtered_points)' in source
    assert '"filtered_target_points": len(filtered_points)' in source
