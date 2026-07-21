"""Launch the standalone observed placement adapter."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return one configurable placement node without simulator ownership."""
    default_config = os.path.join(
        get_package_share_directory('z_manip_place'),
        'config',
        'place.yaml',
    )
    config = LaunchConfiguration('config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_description_file = LaunchConfiguration('robot_description_file')
    collision_model_file = LaunchConfiguration('collision_model_file')
    kinematic_base_link = LaunchConfiguration('kinematic_base_link')
    tool_link = LaunchConfiguration('tool_link')
    transaction_control_topic = LaunchConfiguration('transaction_control_topic')
    transaction_ros_timeout = LaunchConfiguration('transaction_ros_timeout_s')
    transaction_wall_timeout = LaunchConfiguration('transaction_wall_timeout_s')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'robot_description_file',
            default_value=os.environ.get('Z_MANIP_ROBOT_DESCRIPTION_FILE', ''),
        ),
        DeclareLaunchArgument(
            'collision_model_file',
            default_value=os.environ.get('Z_MANIP_COLLISION_MODEL_FILE', ''),
        ),
        DeclareLaunchArgument(
            'kinematic_base_link',
            default_value='piper_base_link',
        ),
        DeclareLaunchArgument('tool_link', default_value='piper_gripper_base'),
        DeclareLaunchArgument(
            'transaction_control_topic',
            default_value='/z_manip/place/transaction_control',
        ),
        DeclareLaunchArgument('transaction_ros_timeout_s', default_value='30.0'),
        DeclareLaunchArgument('transaction_wall_timeout_s', default_value='60.0'),
        Node(
            package='z_manip_place',
            executable='observed_placement',
            name='z_manip_observed_placement',
            output='screen',
            parameters=[config, {
                'use_sim_time': use_sim_time,
                'robot_description_file': robot_description_file,
                'collision_model_file': collision_model_file,
                'kinematic_base_link': kinematic_base_link,
                'tool_link': tool_link,
                'transaction_control_topic': transaction_control_topic,
                'transaction_ros_timeout_s': transaction_ros_timeout,
                'transaction_wall_timeout_s': transaction_wall_timeout,
            }],
        ),
    ])
