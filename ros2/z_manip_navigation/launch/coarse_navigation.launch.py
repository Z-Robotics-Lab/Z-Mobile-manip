"""Launch observed-target coarse navigation."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the parameterized coarse navigation launch description."""
    parameters = LaunchConfiguration('navigation_parameters')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'navigation_parameters',
            default_value=PathJoinSubstitution([
                FindPackageShare('z_manip_navigation'), 'config', 'navigation.yaml',
            ]),
        ),
        Node(
            package='z_manip_navigation',
            executable='coarse_navigation',
            name='z_manip_coarse_navigation',
            output='screen',
            parameters=[parameters, {'use_sim_time': use_sim_time}],
        ),
    ])
