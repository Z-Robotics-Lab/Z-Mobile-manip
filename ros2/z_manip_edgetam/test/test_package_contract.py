"""Static deployment contract tests for the standalone ROS package."""

from pathlib import Path
import re
import xml.etree.ElementTree as ElementTree


ROOT = Path(__file__).resolve().parents[1]


def test_config_exposes_bridge_topics_and_failure_limits() -> None:
    text = (ROOT / 'config' / 'edgetam.yaml').read_text(encoding='utf-8')
    required = (
        '/camera/color/image_raw',
        '/camera/aligned_depth_to_color/image_raw',
        '/camera/color/camera_info',
        '/track_3d/init_bbox',
        '/track_3d/reset',
        '/track_3d/seed_request',
        '/track_3d/exact_seed_image',
        '/track_3d/seed_offer_manifest',
        '/track_3d/seed_status',
        '/track_3d/is_tracking',
        '/track_3d/failure',
        '/track_3d/frame_manifest',
        '/track_3d/detections_2d',
        '/track_3d/selected_target_3d',
        '/track_3d/selected_target_pointcloud',
        'service_url:',
        'service_timeout_s:',
        'min_cloud_points:',
        'min_mask_iou:',
        'hard_min_mask_iou:',
        'min_mask_area_ratio:',
        'max_mask_displacement_ratio:',
        'min_mask_overlap_ratio:',
        'min_mask_bbox_iou:',
        'max_soft_continuity_frames:',
        'min_soft_depth_mask_retention:',
        'max_contained_collapse_recovery_frames:',
        'max_rejected_mask_ratio:',
        'max_largest_rejected_to_selected_ratio:',
        'cluster_max_depth_jump_m:',
        'cluster_max_depth_jump_ratio:',
        'max_pending_frames:',
        'reseed_roi_expansion_ratio:',
        'reseed_max_forward_backward_error_px:',
        'reseed_min_global_inliers:',
        'reseed_min_roi_inliers:',
        'reseed_max_global_roi_center_delta_ratio:',
        'min_acquisition_live_updates:',
        'max_acquisition_pending_frames:',
        'max_result_stamp_lag_s:',
        'seed_offer_timeout_s:',
    )
    assert all(value in text for value in required)


def test_node_wires_every_mask_continuity_limit_into_the_core() -> None:
    text = (ROOT / 'z_manip_edgetam' / 'node.py').read_text(encoding='utf-8')
    names = (
        'hard_min_mask_iou',
        'min_mask_area_ratio',
        'max_mask_displacement_ratio',
        'min_mask_overlap_ratio',
        'min_mask_bbox_iou',
        'max_soft_continuity_frames',
        'min_soft_depth_mask_retention',
        'allow_motion_reanchor',
        'min_motion_reanchor_area_ratio',
        'max_motion_reanchor_displacement_ratio',
        'max_contained_collapse_recovery_frames',
        'max_rejected_mask_ratio',
        'max_largest_rejected_to_selected_ratio',
    )
    for name in names:
        assert re.search(
            rf"get_parameter\(\s*['\"]{re.escape(name)}['\"]\s*,?\s*\)",
            text,
        )
        assert f'{name}=' in text


def test_ros_adapter_has_no_sparse_replay_worker_path() -> None:
    text = (ROOT / 'z_manip_edgetam' / 'node.py').read_text(encoding='utf-8')
    assert "command.kind == 'replay'" not in text
    assert 'def _run_replay' not in text


def test_seed_offer_is_reliable_and_not_sourced_from_raw_color_subscription() -> None:
    text = (ROOT / 'z_manip_edgetam' / 'node.py').read_text(encoding='utf-8')
    assert "self._topic('seed_request_topic')" in text
    assert "self._topic('seed_image_topic')" in text
    assert "self._topic('seed_offer_manifest_topic')" in text
    assert 'CompressedImage' in text
    assert 'DurabilityPolicy.TRANSIENT_LOCAL' in text
    assert 'ClockType.STEADY_TIME' in text
    assert '_seed_owner_producer_epoch' in text
    assert 'self._seed_offer = offer_to_publish' in text
    assert 'self._seed_image_pub.publish(image)' in text
    assert 'self._seed_offer_manifest_pub.publish(manifest)' in text
    assert 'self._seed_image_pub.publish(color_msg)' not in text


def test_package_declares_runtime_message_dependencies() -> None:
    root = ElementTree.parse(ROOT / 'package.xml').getroot()
    dependencies = {node.text for node in root.findall('exec_depend')}
    assert {
        'rclpy',
        'cv_bridge',
        'message_filters',
        'sensor_msgs',
        'sensor_msgs_py',
        'std_msgs',
        'vision_msgs',
    } <= dependencies


def test_launch_starts_only_the_adapter_not_isaac_or_rviz() -> None:
    text = (ROOT / 'launch' / 'edgetam.launch.py').read_text(encoding='utf-8')
    assert "package='z_manip_edgetam'" in text
    assert 'isaac' not in text.lower()
    assert 'rviz' not in text.lower()
