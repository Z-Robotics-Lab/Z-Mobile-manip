"""Static deployment tests for the standalone ROS placement package."""

import ast
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_config_externalizes_observation_motion_and_visualization_contracts():
    """Keep all runtime interfaces and safety thresholds configurable."""
    text = (ROOT / 'config' / 'place.yaml').read_text(encoding='utf-8')
    required = (
        '/z_manip/place/region_request',
        '/z_manip/perception/scene_pointcloud',
        '/z_manip/place/candidates',
        '/z_manip/place/selected_poses',
        '/z_manip/place/trajectory',
        '/z_manip/place/trajectory_contract',
        '/z_manip/place/post_release_verification',
        '/z_manip/perception/target_pointcloud',
        '/z_manip/perception/status',
        '/piper/execution_status',
        '/z_manip/place/transaction_control',
        '/compute_cartesian_path',
        'max_sync_skew_s:',
        'max_snapshot_age_s:',
        'planning_rgbd_cache_size: 20',
        'transaction_ros_timeout_s: 30.0',
        'transaction_wall_timeout_s: 60.0',
        'transaction_watchdog_period_s: 0.05',
        'ransac_distance_m:',
        'tool_clearance_radius_m:',
        'min_cartesian_fraction:',
        'moveit_response_timeout_s: 5.0',
        'joint_velocity_limits:',
        'collision_model_file:',
        'attached_collision_clearance_m: 0.02',
        'attached_collision_segment_joint_step_rad: 0.025',
        'attached_collision_extent_samples_per_axis: 5',
        'attached_collision_carried_object_scene_exclusion_m: 0.012',
        'post_release_min_stable_duration_s: 0.50',
        'post_release_max_target_motion_m: 0.025',
        'post_release_min_gripper_clearance_m: 0.04',
        'post_release_target_depth_correspondence_tolerance_m: 0.012',
        'post_release_object_position_tolerance_m: 0.04',
        'post_release_object_orientation_tolerance_rad: 0.35',
        'post_release_upright_tolerance_rad: 0.26',
        'post_release_max_axial_transverse_ratio: 1.90',
        'post_release_max_symmetry_axis_alignment_error_rad: 0.12',
        'post_release_min_signed_upright_profile_asymmetry: 0.10',
        'post_release_min_signed_upright_profile_alignment: 0.60',
        'post_release_max_object_orientation_motion_rad: 0.10',
        'post_release_min_registration_inlier_fraction: 0.55',
        'post_release_fk_position_tolerance_m: 0.015',
        'post_release_fk_orientation_tolerance_rad: 0.10',
        'post_release_max_rejection_diagnostics: 8',
        'gripper_probe_points_tool_flat:',
    )
    assert all(value in text for value in required)


def test_node_publishes_every_audited_trajectory_point():
    """The ROS serializer must append inside the waypoint loop."""
    source = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    loop = source.index('for contract_point in output.trajectory.points:')
    append = source.index('trajectory.points.append(point)', loop)
    loop_line = source[source.rfind('\n', 0, loop) + 1:loop]
    loop_indent = len(loop_line) - len(loop_line.lstrip())
    append_line = source[source.rfind('\n', 0, append) + 1:append]
    assert len(append_line) - len(append_line.lstrip()) > loop_indent


def test_signed_upright_and_orientation_window_parameters_reach_core_config():
    """Keep every new stability threshold wired from ROS overrides to core."""
    source = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    expected = {
        'min_signed_upright_profile_asymmetry': 0.10,
        'min_signed_upright_profile_alignment': 0.60,
        'max_object_orientation_motion_rad': 0.10,
    }
    declared = {}
    configured = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == 'declare_parameter'
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            declared[str(node.args[0].value)] = node.args[1].value
        if isinstance(node.func, ast.Name) and node.func.id == (
            'PostReleaseVerificationConfig'
        ):
            configured.update({keyword.arg: keyword.value for keyword in node.keywords})

    for field, default in expected.items():
        assert declared[f'post_release_{field}'] == default
        assert ast.unparse(configured[field]) == f"float(value('{field}'))"


def test_payload_collision_audit_precedes_candidate_motion_score():
    """A bare-robot MoveIt result cannot enter feasible ranking by itself."""
    evaluator = (
        ROOT / 'z_manip_place' / 'moveit_evaluator.py'
    ).read_text(encoding='utf-8')
    node = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    assert evaluator.index(
        'self.attached_collision_auditor.audit(',
    ) < evaluator.index('joint_distance =')
    required = (
        'object_reference_points_object',
        'object_extent_m=request.object_extent_m',
        'planning_from_kinematic_base',
        'attached_collision.bind_snapshot(',
        'attached_collision.clear_snapshot()',
    )
    assert all(value in node for value in required)


def test_tf_buffer_tracks_sim_clock_epoch_changes():
    """A simulator restart must clear transforms from its previous epoch."""
    source = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    assert 'Buffer(node=self)' in source


def test_planning_rgbd_is_exact_keyed_and_never_uses_latest_or_slop_fallback():
    """Schema-v2 source identity must select one immutable RGB-D frame."""
    source = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    core = (ROOT / 'z_manip_place' / 'core.py').read_text(encoding='utf-8')
    assert 'message_filters.TimeSynchronizer(' in source
    assert 'message_filters.ApproximateTimeSynchronizer(' not in source
    assert 'self._rgbd: _RgbdCache | None' not in source
    assert 'self._planning_rgbd.get(key)' in source
    assert '_exact_rgbd_source_key' in source
    assert '_validate_exact_planning_sources' in source
    assert 'planning_rgbd_cache_size' in source
    assert 'camera_info_stamp_ns' in core
    assert (
        'placement request, RGB, depth, and camera-info stamps must match exactly'
        in core
    )


def test_every_late_planning_input_retries_the_exact_request_assembler():
    """A request cannot depend on one lucky callback arrival ordering."""
    source = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    functions = {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }
    for callback in (
        '_on_execution_status',
        '_on_perception_status',
        '_on_target_cloud',
        '_on_rgbd',
        '_on_scene',
        '_on_joints',
    ):
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_maybe_auto_plan'
            for node in ast.walk(functions[callback])
        ), callback


def test_package_declares_tf_moveit_sensor_and_visualization_dependencies():
    """Declare every ROS runtime boundary used by the adapter."""
    root = ElementTree.parse(ROOT / 'package.xml').getroot()
    dependencies = {node.text for node in root.findall('exec_depend')}
    assert {
        'cv_bridge',
        'diagnostic_msgs',
        'message_filters',
        'moveit_msgs',
        'python3-scipy',
        'rclpy',
        'sensor_msgs',
        'sensor_msgs_py',
        'tf2_ros',
        'trajectory_msgs',
        'visualization_msgs',
        'z_manip',
    } <= dependencies


def test_node_correlates_post_release_evidence_without_ground_truth():
    """Keep exact producer, release, RGB-D, and measured-state gates deployed."""
    source = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    core = (ROOT / 'z_manip_place' / 'core.py').read_text(encoding='utf-8')
    required_node_contracts = (
        'ObservedPerceptionIdentity',
        '_on_perception_status',
        '_on_target_cloud',
        '_on_execution_status',
        'place_approach',
        'gripper_command_id',
        'capture_observed_region_geometry',
    )
    required_core_contracts = (
        'z_manip.post_release_verification.v2',
        'synchronized_rgbd_pointcloud',
        'post-release observation source clock moved backwards',
        'post-release target support inside selected region is insufficient',
        'post-release gripper has not cleared target',
    )
    assert all(value in source for value in required_node_contracts)
    assert all(value in core for value in required_core_contracts)
    forbidden = ('isaac', 'prim_path', 'sim_object_pose', 'ground_truth')
    assert all(value not in source.lower() for value in forbidden)


def test_post_release_qos_requires_durable_late_start_reader():
    """Both endpoints must request durability for late-start replay."""
    rclpy_qos = pytest.importorskip('rclpy.qos')
    publisher_qos = rclpy_qos.QoSProfile(
        depth=1,
        reliability=rclpy_qos.ReliabilityPolicy.RELIABLE,
        durability=rclpy_qos.DurabilityPolicy.TRANSIENT_LOCAL,
    )
    task_subscriber_qos = rclpy_qos.QoSProfile(
        depth=1,
        reliability=rclpy_qos.ReliabilityPolicy.RELIABLE,
        durability=rclpy_qos.DurabilityPolicy.TRANSIENT_LOCAL,
    )
    compatibility, reason = rclpy_qos.qos_check_compatible(
        publisher_qos,
        task_subscriber_qos,
    )
    assert compatibility != rclpy_qos.QoSCompatibility.ERROR, reason

    source = (ROOT / 'z_manip_place' / 'node.py').read_text(encoding='utf-8')
    start = source.index('post_release_qos = QoSProfile(')
    stop = source.index('self._post_release_publisher', start)
    profile = source[start:stop]
    assert 'depth=1' in profile
    assert 'reliability=ReliabilityPolicy.RELIABLE' in profile
    assert 'durability=DurabilityPolicy.TRANSIENT_LOCAL' in profile
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'VOLATILE reader' in readme
    assert 'RELIABLE/TRANSIENT_LOCAL durability' in readme


def test_launch_owns_neither_simulator_nor_rviz():
    """Keep restart and visualization ownership in top-level bringup."""
    text = (ROOT / 'launch' / 'place.launch.py').read_text(encoding='utf-8')
    assert "package='z_manip_place'" in text
    assert "'robot_description_file'" in text
    assert "'collision_model_file'" in text
    assert "'kinematic_base_link'" in text
    assert "'tool_link'" in text
    assert "'transaction_control_topic'" in text
    assert "'transaction_ros_timeout_s'" in text
    assert "'transaction_wall_timeout_s'" in text
    assert 'isaac' not in text.lower()
    assert 'rviz' not in text.lower()
