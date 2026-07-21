"""Static ROS package and no-ground-truth contract checks."""

import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_has_required_topics_and_no_truth_subscription():
    source = (ROOT / 'z_manip_task' / 'node.py').read_text()
    required = (
        '/z_manip/perception/valid',
        '/z_manip/perception/target_3d',
        '/z_manip/perception/target_pointcloud',
        '/z_manip/perception/scene_pointcloud',
        '/z_manip/perception/affordance',
        '/z_manip/perception/status',
        '/z_manip/coarse_nav/perception_loss_authorization',
        '/z_manip/visual_search/active',
        '/z_manip/grounding/reset',
        '/piper/execution_status',
        '/z_manip/place/post_release_verification',
        '/z_manip/place/transaction_control',
        '/z_manip/task/status',
    )
    config = (ROOT / 'config' / 'runtime.yaml').read_text()
    runtime_parameters = yaml.safe_load(config)[
        'z_manip_task_runtime'
    ]['ros__parameters']
    stack = source + config
    assert all(topic in stack for topic in required)
    assert 'self._config.topics.local_velocity' in source
    assert 'self._config.topics.arm_trajectory' in source
    assert 'self._config.topics.gripper_aperture' in source
    assert "Bool, self._topic('cancel_goal_topic')" in source
    assert 'self._cancel_nav_pub.publish(Bool(data=True))' in source
    assert '/objects/' not in source
    assert '/ground_truth' not in source
    assert 'DurabilityPolicy.TRANSIENT_LOCAL' in source
    assert "self._topic('debug_markers_topic'), latched_debug" in source
    assert "self._topic('debug_path_topic'), latched_debug" in source
    assert 'BoundedYawSearch' in source
    assert 'Buffer(node=self)' in source
    assert 'visual_search_horizontal_margin_ratio' in stack
    assert 'visual_search_position_hold_deadband_m' in config
    assert 'visual_search_settle_yaw_tolerance_rad' in config
    assert 'visual_search_position_heading_reacquire_tolerance_rad' in config
    assert 'visual_search_min_yaw_rate_rps' in config
    assert 'visual_search_position_completion_tolerance_m' in config
    assert 'visual_search_moving_rebound_reacquire_m' in config
    assert 'visual_search_position_hold_gain_s_inv' in config
    assert 'visual_search_max_position_hold_speed_mps' in config
    assert 'visual_search_position_hold_timeout_s' in config
    assert 'visual_search_settle_max_linear_speed_mps' in config
    assert 'visual_search_settle_max_angular_speed_rps' in config
    assert 'visual_search_stationary_wait_timeout_s' in config
    assert 'visual_search_stationary_quiet_window_s' in config
    assert 'visual_search_stationary_max_odom_gap_s' in config
    assert 'visual_search_settle_reacquire_budget_s' in config
    assert 'command.twist.linear.x = update.linear_x' in source
    assert 'command.twist.linear.y = update.linear_y' in source
    assert "'position_anchor_frame'" in source
    assert "'position_anchor_xy_m'" in source
    assert "'position_error_base_xy_m'" in source
    assert "'linear_command_base_xy_mps'" in source
    assert "'settle_stationary_deadline_s'" in source
    assert "'stationary_wait_timeout_s'" in source
    assert 'posture_state_max_age_s' in config
    assert 'posture_state_acquisition_timeout_s' in config
    assert runtime_parameters['coarse_nav_posture_violation_dwell_s'] == 0.15
    assert 'standoff_planning_budget_s' in config
    assert 'grasp_planning_budget_s' in config
    assert 'carry_planning_budget_s' in config
    assert 'place_planning_wall_timeout_s' in config
    for parameter in (
        'coarse_nav_arrival_settle_s',
        'coarse_nav_arrival_stop_timeout_s',
        'coarse_nav_arrival_max_linear_speed_mps',
        'coarse_nav_arrival_max_angular_speed_rps',
        'coarse_nav_arrival_max_xy_excursion_m',
        'coarse_nav_arrival_max_yaw_excursion_rad',
        'coarse_nav_arrival_max_odom_gap_s',
        'frozen_coarse_nav_authorization_period_s',
    ):
        assert parameter in config
    assert (
        runtime_parameters['coarse_nav_arrival_max_linear_speed_mps']
        == 0.05
    )
    assert (
        runtime_parameters['visual_search_position_completion_tolerance_m']
        == 0.06
    )
    assert runtime_parameters['visual_search_moving_rebound_reacquire_m'] == 0.10
    assert runtime_parameters['visual_search_max_position_hold_speed_mps'] == 0.05
    assert runtime_parameters['visual_search_position_hold_slowdown_radius_m'] == 0.0
    assert runtime_parameters['visual_search_min_position_hold_speed_mps'] == 0.0
    platform_parameters = yaml.safe_load(
        (ROOT / 'config' / 'go2w_sim.yaml').read_text(),
    )['z_manip_task_runtime']['ros__parameters']
    assert (
        platform_parameters['visual_search_position_completion_tolerance_m']
        == 0.125
    )
    assert platform_parameters['coarse_nav_arrival_max_linear_speed_mps'] == 0.080
    assert runtime_parameters['coarse_nav_arrival_max_xy_excursion_m'] == 0.010
    assert runtime_parameters['coarse_nav_arrival_max_yaw_excursion_rad'] == 0.010
    assert 'linear_speed_mps=nav_speed' in source
    assert 'angular_speed_rps=abs(angular_components[2])' in source
    assert platform_parameters['visual_search_moving_rebound_reacquire_m'] == 0.14
    assert (
        platform_parameters['visual_search_position_hold_slowdown_radius_m']
        == 0.14
    )
    assert platform_parameters['visual_search_min_position_hold_speed_mps'] == 0.020
    assert runtime_parameters['semantic_reground_timeout_s'] == 105.0
    assert runtime_parameters['approach_planning_budget_s'] == 15.0
    assert runtime_parameters['pregrasp_dispatch_feedback_wait_timeout_s'] == 1.0
    assert runtime_parameters['pregrasp_reobserve_timeout_s'] == 8.0
    assert runtime_parameters['pregrasp_joint_tolerance_rad'] == 0.05
    assert (
        runtime_parameters['pregrasp_max_observation_joint_skew_s'] == 0.12
    )
    assert (
        runtime_parameters['approach_planning_target_drift_tolerance_m']
        == 0.025
    )
    assert runtime_parameters['approach_execution_joint_state_max_age_s'] == 0.25
    assert runtime_parameters['approach_execution_joint_wait_timeout_s'] == 1.0
    assert runtime_parameters['execution_status_max_age_s'] == 0.50
    assert "name == 'near_view_joint_positions'" in source
    assert 'ParameterDescriptor(dynamic_typing=True)' in source
    package_xml = (ROOT / 'package.xml').read_text()
    assert '<exec_depend>rcl_interfaces</exec_depend>' in package_xml
    assert runtime_parameters['approach_planning_geometry_trim_mad_scale'] == 4.5
    assert runtime_parameters['approach_planning_geometry_extent_percentile'] == 2.0
    assert runtime_parameters['approach_planning_geometry_max_extent_change_m'] == 0.008
    assert runtime_parameters['approach_planning_geometry_max_extent_ratio'] == 1.25
    assert (
        runtime_parameters['approach_planning_geometry_axis_separation_ratio']
        == 1.25
    )
    assert (
        runtime_parameters['approach_planning_geometry_max_orientation_change_rad']
        == 0.3490658504
    )
    assert 'PlanningControl' in source
    assert 'PostureSafetyGate' in source
    assert 'ExecutionOcclusionGate' in source
    assert '_execution_occlusion_target_cloud' in source
    assert '_execution_occlusion_scene_cloud' in source
    for parameter in (
        'execution_occlusion_max_duration_s',
        'execution_occlusion_joint_state_max_age_s',
        'execution_occlusion_status_max_age_s',
        'execution_occlusion_command_ack_timeout_s',
        'execution_occlusion_near_contact_tolerance_rad',
        'execution_occlusion_lift_path_tolerance_rad',
        'execution_occlusion_max_path_regression_samples',
        'execution_occlusion_verification_reacquire_timeout_s',
    ):
        assert parameter in config
    for parameter in (
        'post_release_verification_timeout_s',
        'post_release_verification_wall_timeout_s',
        'post_release_min_stable_duration_s',
        'post_release_min_samples',
        'post_release_min_target_points',
        'post_release_max_target_motion_m',
        'post_release_min_region_support_fraction',
        'post_release_min_gripper_clearance_m',
        'post_release_max_rgbd_target_skew_s',
        'post_release_max_joint_target_skew_s',
        'post_release_max_target_depth_correspondence_m',
        'post_release_max_object_position_error_m',
        'post_release_max_object_orientation_error_rad',
        'post_release_max_object_upright_error_rad',
        'post_release_min_object_registration_inlier_fraction',
        'post_release_max_object_registration_rms_m',
        'carried_object_min_points',
        'carried_object_trim_mad_scale',
        'carried_object_extent_percentile',
        'carried_object_min_extent_m',
        'carried_object_min_axis_separation_ratio',
        'carried_object_max_axial_transverse_ratio',
        'carried_object_max_reference_points',
        'carried_object_max_joint_target_skew_s',
    ):
        assert parameter in config
    assert 'parse_post_release_verification' in source
    assert 'object_reference_points_object' in source
    assert 'object_reference_identity' in source
    assert "'schema_version': 2" in source
    assert 'RuntimePhase.POST_RELEASE_VERIFICATION' in source
    assert 'required_affordance_generation' in source
    assert 'z_manip.grounding_request.v2' in source
    assert "grounding_scope = 'place_support'" in source
    assert "grounding_scope = 'grasp_for_place'" in source
    assert '_required_perception_request_id' in source
    assert '_bound_perception_producer_epoch' in source
    assert 'require_fresh_target_lock=not predicted_baseline' in source
    assert 'allow_predicted_target_baseline=predicted_baseline' in source
    assert 'establish_baseline_before_lift' in source
    assert "'state_estimation_topic': '/odom_base_link'" in source
    assert 'state_estimation_topic: /odom_base_link' in config
    assert "'platform_odometry_parent_frame': 'map'" in source
    assert "'platform_odometry_child_frame': 'base_link'" in source
    assert 'validate_platform_odometry_frames(' in source


def test_terminal_release_sends_place_abort_before_identity_can_be_cleared():
    source = (ROOT / 'z_manip_task' / 'node.py').read_text()
    start = source.index('def _release_terminal_ownership(self)')
    stop = source.index('def _apply_safety(self, action', start)
    body = source[start:stop]
    assert body.index('_publish_place_abort_once(self)') < body.index(
        'self._terminal_ownership_released = True',
    )


def test_go2w_yaw_deadband_override_is_not_the_node_default():
    source = (ROOT / 'z_manip_task' / 'node.py').read_text()
    runtime = yaml.safe_load((ROOT / 'config' / 'runtime.yaml').read_text())
    parameters = runtime['z_manip_task_runtime']['ros__parameters']

    assert parameters['visual_search_min_yaw_rate_rps'] == 0.06
    assert parameters['visual_search_yaw_tolerance_rad'] == 0.0087266463
    assert parameters['visual_search_turn_timeout_s'] == 10.0
    assert "'visual_search_min_yaw_rate_rps': 0.0" in source
    assert "'visual_search_yaw_tolerance_rad': 0.0174532925" in source
    assert "2.0 * float(self.get_parameter('control_period_s').value)" in source


def test_go2w_stationary_wait_override_is_not_the_node_default():
    source = (ROOT / 'z_manip_task' / 'node.py').read_text()
    runtime = yaml.safe_load((ROOT / 'config' / 'runtime.yaml').read_text())
    parameters = runtime['z_manip_task_runtime']['ros__parameters']

    assert parameters['visual_search_stationary_wait_timeout_s'] == 4.0
    assert parameters['visual_search_stationary_quiet_window_s'] == 0.35
    assert parameters['visual_search_stationary_max_odom_gap_s'] == 0.15
    assert parameters['visual_search_settle_reacquire_budget_s'] == 12.0
    assert "'visual_search_stationary_wait_timeout_s': 0.0" in source
    assert "'visual_search_settle_reacquire_budget_s': 2.0" in source


def test_task_accepts_navigation_bounded_braking_handoff_envelope():
    task = yaml.safe_load((ROOT / 'config' / 'runtime.yaml').read_text())[
        'z_manip_task_runtime'
    ]['ros__parameters']
    navigation = yaml.safe_load(
        (ROOT.parent / 'z_manip_navigation' / 'config' / 'navigation.yaml').read_text(),
    )['z_manip_coarse_navigation']['ros__parameters']

    expected_max_error = (
        navigation['explicit_goal_tolerance_m']
        + navigation['explicit_goal_handoff_hysteresis_m']
    )
    assert math.isclose(
        task['work_pose_goal_tolerance_m'],
        expected_max_error,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_partial_view_axial_gate_matches_recorded_rgbd_observability():
    source = (ROOT / 'z_manip_task' / 'node.py').read_text()
    runtime = yaml.safe_load((ROOT / 'config' / 'runtime.yaml').read_text())
    parameters = runtime['z_manip_task_runtime']['ros__parameters']

    assert parameters['carried_object_max_axial_transverse_ratio'] == 1.90
    assert "'carried_object_max_axial_transverse_ratio': 1.90" in source


def test_manifest_and_entrypoint_are_installed():
    setup = (ROOT / 'setup.py').read_text()
    manifest = (ROOT / 'package.xml').read_text()
    assert 'mobile_manipulation_runtime = z_manip_task.node:main' in setup
    for dependency in ('diagnostic_msgs', 'vision_msgs', 'visualization_msgs', 'tf2_ros'):
        assert f'<exec_depend>{dependency}</exec_depend>' in manifest


def test_post_release_observation_source_matches_place_and_acceptance():
    assignment = (
        "POST_RELEASE_OBSERVATION_SOURCE = 'synchronized_rgbd_pointcloud'"
    )
    sources = (
        ROOT / 'z_manip_task' / 'post_release_verification.py',
        ROOT.parent / 'z_manip_place' / 'z_manip_place' / 'core.py',
    )

    assert all(assignment in path.read_text() for path in sources)
    acceptance = ROOT.parents[1] / 'scripts' / 'runtime' / (
        'mobile_manipulation_acceptance.py'
    )
    if acceptance.is_file():
        assert assignment in acceptance.read_text()


def test_post_release_v2_schema_matches_place_task_and_acceptance():
    assignment = (
        'POST_RELEASE_VERIFICATION_SCHEMA = '
        "'z_manip.post_release_verification.v2'"
    )
    sources = (
        ROOT / 'z_manip_task' / 'post_release_verification.py',
        ROOT.parent / 'z_manip_place' / 'z_manip_place' / 'core.py',
        ROOT.parents[1] / 'scripts' / 'runtime' / (
            'mobile_manipulation_acceptance.py'
        ),
    )
    assert all(assignment in path.read_text() for path in sources)


def test_complete_launch_composes_every_runtime_owner():
    source = (ROOT / 'launch' / 'mobile_manipulation.launch.py').read_text()
    for package in (
        'z_manip_ros', 'z_manip_edgetam', 'z_manip_motion',
        'z_manip_navigation', 'z_manip_place', 'z_manip_task',
    ):
        assert f"'{package}'" in source
    assert "'start_edge_tam': 'false'" in source
    assert "'platform_base_frame'" in source
    assert "'urdf_root_frame'" in source
    assert "'kinematic_base_link'" in source
    assert "'tool_link'" in source
    assert "'robot_description_file': robot" in source
    assert "'collision_model_file': collision_model" in source
    assert "'Z_MANIP_COLLISION_MODEL_FILE'" in source
    assert "'state_max_age_s'" in source
    assert "'state_max_stamp_skew_s'" in source
    assert "'clock_handover_quiet_s'" in source
    assert "DeclareLaunchArgument('namespace', default_value='/')" in source
    assert 'PushRosNamespace(namespace)' in source
    assert "'start_motion_plan_bridge': 'false'" in source
    assert "executable='static_transform_publisher'" in source
    assert "'use_sim_time': use_sim_time" in source


def test_task_launch_exposes_sim_or_real_clock_selection():
    source = (ROOT / 'launch' / 'task_runtime.launch.py').read_text()
    assert "DeclareLaunchArgument('use_sim_time', default_value='true')" in source
    assert "'use_sim_time': LaunchConfiguration('use_sim_time')" in source


def test_near_view_is_enabled_only_by_an_explicit_platform_profile():
    generic = yaml.safe_load((ROOT / 'config' / 'runtime.yaml').read_text())[
        'z_manip_task_runtime'
    ]['ros__parameters']
    go2w_sim = yaml.safe_load((ROOT / 'config' / 'go2w_sim.yaml').read_text())[
        'z_manip_task_runtime'
    ]['ros__parameters']

    assert generic['near_view_pose'] == ''
    assert generic['near_view_joint_positions'] == []
    assert generic['near_view_timeout_s'] == 12.0
    assert generic['near_view_joint_tolerance_rad'] == 0.05
    assert go2w_sim == {
        'near_view_pose': 'MANIP_LOOKOUT',
        'near_view_settle_s': 3.0,
        'near_view_timeout_s': 12.0,
        'near_view_joint_positions': [0.0, 1.0, -0.71, 0.0, 0.30, 0.0],
        'near_view_joint_tolerance_rad': 0.05,
        'coarse_nav_arrival_max_linear_speed_mps': 0.080,
        'coarse_nav_arrival_max_angular_speed_rps': 0.05,
        'visual_search_position_completion_tolerance_m': 0.125,
        'visual_search_position_hold_slowdown_radius_m': 0.14,
        'visual_search_min_position_hold_speed_mps': 0.020,
        'visual_search_moving_rebound_reacquire_m': 0.14,
    }

    task_launch = (ROOT / 'launch' / 'task_runtime.launch.py').read_text()
    mobile_launch = (ROOT / 'launch' / 'mobile_manipulation.launch.py').read_text()
    assert "DeclareLaunchArgument(\n            'platform_parameters'" in task_launch
    assert "parameters = [LaunchConfiguration('runtime_parameters')]" in task_launch
    assert task_launch.index('parameters.append(platform_parameters)') < (
        task_launch.index("'stack_config_path': LaunchConfiguration")
    )
    assert "'task_platform_parameters'" in mobile_launch
    assert "'platform_parameters': task_platform_parameters" in mobile_launch


def test_mobile_bringup_has_one_arm_trajectory_command_owner():
    """The task runtime, not the standalone bridge, owns mobile arm commands."""
    mobile_launch = (ROOT / 'launch' / 'mobile_manipulation.launch.py').read_text()
    motion_launch = (
        ROOT.parent / 'z_manip_motion' / 'launch' / 'moveit_planning.launch.py'
    ).read_text()
    task_source = (ROOT / 'z_manip_task' / 'node.py').read_text()

    assert "'start_motion_plan_bridge': 'false'" in mobile_launch
    assert (
        'condition=IfCondition(LaunchConfiguration("start_motion_plan_bridge"))'
        in motion_launch
    )
    assert 'DeclareLaunchArgument("start_motion_plan_bridge", default_value="true")' in (
        motion_launch
    )
    assert 'self._config.topics.arm_trajectory' in task_source


def test_perception_safety_and_task_local_velocity_channels_are_separate():
    """Fail-closed perception cannot contend for task-local velocity ownership."""
    perception = yaml.safe_load(
        (
            ROOT.parent / 'z_manip_ros' / 'config' / 'perception.yaml'
        ).read_text(),
    )['vlm_edgetam_bridge']['ros__parameters']
    stack = json.loads((ROOT.parents[1] / 'configs' / 'go2w_piper.json').read_text())

    assert perception['stop_cmd_topic'] == '/safety_cmd_vel'
    assert stack['topics']['local_velocity'] == '/local_movement_cmd_vel'
    assert perception['stop_cmd_topic'] != stack['topics']['local_velocity']


def test_frozen_navigation_authorization_topic_and_heartbeat_match_bridge():
    task = yaml.safe_load((ROOT / 'config' / 'runtime.yaml').read_text())[
        'z_manip_task_runtime'
    ]['ros__parameters']
    perception = yaml.safe_load(
        (ROOT.parent / 'z_manip_ros' / 'config' / 'perception.yaml').read_text(),
    )['vlm_edgetam_bridge']['ros__parameters']

    assert (
        task['frozen_coarse_nav_authorization_topic']
        == perception['frozen_coarse_nav_authorization_topic']
    )
    assert (
        3.0 * task['frozen_coarse_nav_authorization_period_s']
        <= perception['frozen_coarse_nav_authorization_timeout_s'] + 1e-12
    )
    assert perception['frozen_coarse_nav_authorization_timeout_s'] <= 0.35
