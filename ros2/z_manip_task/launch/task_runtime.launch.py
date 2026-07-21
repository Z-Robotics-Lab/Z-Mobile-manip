"""Launch the online mobile manipulation task runtime."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _runtime_node(context):
    """Apply an optional deployment profile after the generic runtime YAML."""
    parameters = [LaunchConfiguration('runtime_parameters')]
    platform_parameters = LaunchConfiguration(
        'platform_parameters',
    ).perform(context).strip()
    if platform_parameters:
        parameters.append(platform_parameters)
    parameters.append({
        'stack_config_path': LaunchConfiguration('stack_config_path'),
        'use_sim_time': LaunchConfiguration('use_sim_time'),
        'place_transaction_control_topic': LaunchConfiguration(
            'transaction_control_topic',
        ),
    })
    return [Node(
        package='z_manip_task',
        executable='mobile_manipulation_runtime',
        name='z_manip_task_runtime',
        output='screen',
        parameters=parameters,
    )]


def generate_launch_description() -> LaunchDescription:
    """Build a runtime launch description with an external stack config."""
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'transaction_control_topic',
            default_value='/z_manip/place/transaction_control',
        ),
        DeclareLaunchArgument(
            'stack_config_path',
            default_value=EnvironmentVariable('Z_MANIP_STACK_CONFIG', default_value=''),
            description='Absolute path to the root stack schema-v2 JSON configuration',
        ),
        DeclareLaunchArgument(
            'runtime_parameters',
            default_value=PathJoinSubstitution([
                FindPackageShare('z_manip_task'), 'config', 'runtime.yaml',
            ]),
            description='Runtime YAML; normally supplied by the top-level bringup',
        ),
        DeclareLaunchArgument(
            'platform_parameters',
            default_value='',
            description=(
                'Optional platform YAML applied after the generic runtime YAML'
            ),
        ),
        OpaqueFunction(function=_runtime_node),
    ])
