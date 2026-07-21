"""Launch the standalone EdgeTAM ROS adapter."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Return a configurable adapter launch description."""
    default_config = os.path.join(
        get_package_share_directory('z_manip_edgetam'),
        'config',
        'edgetam.yaml',
    )
    config = LaunchConfiguration('config')
    service_url = LaunchConfiguration('service_url')
    max_result_stamp_lag_s = LaunchConfiguration('max_result_stamp_lag_s')
    sync_queue_size = LaunchConfiguration('sync_queue_size')
    sync_timeout_s = LaunchConfiguration('sync_timeout_s')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription(
        [
            DeclareLaunchArgument('config', default_value=default_config),
            DeclareLaunchArgument(
                'service_url',
                default_value='http://127.0.0.1:8092',
            ),
            DeclareLaunchArgument('use_sim_time', default_value='true'),
            DeclareLaunchArgument('sync_timeout_s', default_value='0.5'),
            DeclareLaunchArgument('sync_queue_size', default_value='20'),
            DeclareLaunchArgument('max_result_stamp_lag_s', default_value='0.5'),
            Node(
                package='z_manip_edgetam',
                executable='edgetam_adapter',
                name='z_manip_edgetam',
                output='screen',
                parameters=[config, {
                    'service_url': service_url,
                    'sync_timeout_s': ParameterValue(
                        sync_timeout_s,
                        value_type=float,
                    ),
                    'sync_queue_size': ParameterValue(
                        sync_queue_size,
                        value_type=int,
                    ),
                    'max_result_stamp_lag_s': ParameterValue(
                        max_result_stamp_lag_s,
                        value_type=float,
                    ),
                    'use_sim_time': use_sim_time,
                }],
            ),
        ],
    )
