"""Launch the perception bridge and its contract-compatible EdgeTAM adapter."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


COLOR = '/camera/color/image_raw'
DEPTH = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO = '/camera/color/camera_info'


def generate_launch_description() -> LaunchDescription:
    config = LaunchConfiguration('config')
    tracker_config = LaunchConfiguration('tracker_config')
    tracker_service_url = LaunchConfiguration('tracker_service_url')
    tracker_data_timeout_s = LaunchConfiguration('tracker_data_timeout_s')
    tracker_frame_wait_timeout_s = LaunchConfiguration(
        'tracker_frame_wait_timeout_s',
    )
    tracker_max_result_stamp_lag_s = LaunchConfiguration(
        'tracker_max_result_stamp_lag_s',
    )
    tracker_sync_timeout_s = LaunchConfiguration('tracker_sync_timeout_s')
    tracker_sync_queue_size = LaunchConfiguration('tracker_sync_queue_size')
    stop_cmd_topic = LaunchConfiguration('stop_cmd_topic')
    frozen_coarse_nav_authorization_topic = LaunchConfiguration(
        'frozen_coarse_nav_authorization_topic',
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    bridge = Node(
        package='z_manip_ros',
        executable='vlm_edgetam_bridge',
        name='vlm_edgetam_bridge',
        output='screen',
        parameters=[config, {
            'use_sim_time': use_sim_time,
            'tracker_data_timeout_s': ParameterValue(
                tracker_data_timeout_s,
                value_type=float,
            ),
            'frame_wait_timeout_s': ParameterValue(
                tracker_frame_wait_timeout_s,
                value_type=float,
            ),
            'stop_cmd_topic': stop_cmd_topic,
            'frozen_coarse_nav_authorization_topic': (
                frozen_coarse_nav_authorization_topic
            ),
        }],
        remappings=[
            ('/track_3d/seed_request', '/track_3d/seed_request'),
            ('/track_3d/exact_seed_image', '/track_3d/exact_seed_image'),
            ('/track_3d/seed_offer_manifest', '/track_3d/seed_offer_manifest'),
            ('/track_3d/init_bbox', '/track_3d/init_bbox'),
            ('/track_3d/is_tracking', '/track_3d/is_tracking'),
            ('/track_3d/detections_2d', '/track_3d/detections_2d'),
            ('/track_3d/selected_target_3d', '/track_3d/selected_target_3d'),
            (
                '/track_3d/selected_target_pointcloud',
                '/track_3d/selected_target_pointcloud',
            ),
        ],
    )

    edge_tam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('z_manip_edgetam'),
                'launch',
                'edgetam.launch.py',
            ]),
        ),
        condition=IfCondition(LaunchConfiguration('start_edge_tam')),
        launch_arguments={
            'config': tracker_config,
            'service_url': tracker_service_url,
            'max_result_stamp_lag_s': tracker_max_result_stamp_lag_s,
            'sync_queue_size': tracker_sync_queue_size,
            'sync_timeout_s': tracker_sync_timeout_s,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'config',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('z_manip_ros'), 'config', 'perception.yaml'],
                ),
            ),
            DeclareLaunchArgument('use_sim_time', default_value='false'),
            DeclareLaunchArgument('start_edge_tam', default_value='true'),
            DeclareLaunchArgument('tracker_data_timeout_s', default_value='1.0'),
            DeclareLaunchArgument(
                'tracker_frame_wait_timeout_s',
                default_value='2.0',
            ),
            DeclareLaunchArgument(
                'tracker_max_result_stamp_lag_s',
                default_value='0.5',
            ),
            DeclareLaunchArgument('tracker_sync_timeout_s', default_value='0.5'),
            DeclareLaunchArgument('tracker_sync_queue_size', default_value='20'),
            DeclareLaunchArgument('stop_cmd_topic', default_value='/safety_cmd_vel'),
            DeclareLaunchArgument(
                'frozen_coarse_nav_authorization_topic',
                default_value=(
                    '/z_manip/coarse_nav/perception_loss_authorization'
                ),
            ),
            DeclareLaunchArgument(
                'tracker_config',
                default_value=PathJoinSubstitution([
                    FindPackageShare('z_manip_edgetam'),
                    'config',
                    'edgetam.yaml',
                ]),
            ),
            DeclareLaunchArgument(
                'tracker_service_url',
                default_value='http://127.0.0.1:8092',
            ),
            edge_tam,
            bridge,
        ],
    )
