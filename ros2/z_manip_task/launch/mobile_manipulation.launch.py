"""Bring up the complete no-ground-truth mobile manipulation runtime."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def _launch(package: str, filename: str, arguments: dict[str, object]):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare(package), 'launch', filename,
        ])),
        launch_arguments=arguments.items(),
    )


def generate_launch_description() -> LaunchDescription:
    """Compose perception, navigation, MoveIt, task, and placement nodes."""
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot = LaunchConfiguration('robot_description_file')
    collision_model = LaunchConfiguration('collision_model_file')
    stack = LaunchConfiguration('stack_config_path')
    platform_frame = LaunchConfiguration('platform_base_frame')
    urdf_root_frame = LaunchConfiguration('urdf_root_frame')
    kinematic_base_link = LaunchConfiguration('kinematic_base_link')
    tool_link = LaunchConfiguration('tool_link')
    state_max_age = LaunchConfiguration('state_max_age_s')
    state_max_stamp_skew = LaunchConfiguration('state_max_stamp_skew_s')
    clock_handover_quiet = LaunchConfiguration('clock_handover_quiet_s')
    namespace = LaunchConfiguration('namespace')
    task_platform_parameters = LaunchConfiguration('task_platform_parameters')
    transaction_control_topic = LaunchConfiguration('transaction_control_topic')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('namespace', default_value='/'),
        DeclareLaunchArgument(
            'transaction_control_topic',
            default_value='/z_manip/place/transaction_control',
        ),
        DeclareLaunchArgument(
            'robot_description_file',
            default_value=EnvironmentVariable(
                'Z_MANIP_ROBOT_DESCRIPTION_FILE', default_value='',
            ),
        ),
        DeclareLaunchArgument(
            'collision_model_file',
            default_value=EnvironmentVariable(
                'Z_MANIP_COLLISION_MODEL_FILE', default_value='',
            ),
        ),
        DeclareLaunchArgument(
            'stack_config_path',
            default_value=EnvironmentVariable('Z_MANIP_STACK_CONFIG', default_value=''),
        ),
        DeclareLaunchArgument(
            'task_platform_parameters',
            default_value=EnvironmentVariable(
                'Z_MANIP_TASK_PLATFORM_PARAMETERS',
                default_value='',
            ),
            description=(
                'Optional z_manip_task platform YAML; empty keeps generic behavior'
            ),
        ),
        DeclareLaunchArgument('platform_base_frame', default_value='base_link'),
        DeclareLaunchArgument('urdf_root_frame', default_value='base'),
        DeclareLaunchArgument(
            'kinematic_base_link',
            default_value='piper_base_link',
        ),
        DeclareLaunchArgument(
            'tool_link',
            default_value='piper_gripper_base',
        ),
        DeclareLaunchArgument('state_max_age_s', default_value='0.25'),
        DeclareLaunchArgument('state_max_stamp_skew_s', default_value='0.25'),
        DeclareLaunchArgument('clock_handover_quiet_s', default_value='0.5'),
        GroupAction([
            PushRosNamespace(namespace),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='z_manip_urdf_root_alias',
                arguments=[
                    '--frame-id', platform_frame,
                    '--child-frame-id', urdf_root_frame,
                ],
            ),
            _launch('z_manip_ros', 'perception.launch.py', {
                'use_sim_time': use_sim_time,
                'start_edge_tam': 'false',
            }),
            _launch('z_manip_edgetam', 'edgetam.launch.py', {
                'use_sim_time': use_sim_time,
            }),
            _launch('z_manip_motion', 'moveit_planning.launch.py', {
                'use_sim_time': use_sim_time,
                'robot_description_file': robot,
                'point_cloud_topic': '/z_manip/perception/scene_pointcloud',
                'start_motion_plan_bridge': 'false',
                'state_max_age_s': state_max_age,
                'state_max_stamp_skew_s': state_max_stamp_skew,
                'clock_handover_quiet_s': clock_handover_quiet,
            }),
            _launch('z_manip_navigation', 'coarse_navigation.launch.py', {
                'use_sim_time': use_sim_time,
            }),
            _launch('z_manip_place', 'place.launch.py', {
                'use_sim_time': use_sim_time,
                'robot_description_file': robot,
                'collision_model_file': collision_model,
                'kinematic_base_link': kinematic_base_link,
                'tool_link': tool_link,
                'transaction_control_topic': transaction_control_topic,
            }),
            _launch('z_manip_task', 'task_runtime.launch.py', {
                'stack_config_path': stack,
                'platform_parameters': task_platform_parameters,
                'use_sim_time': use_sim_time,
                'transaction_control_topic': transaction_control_topic,
            }),
        ]),
    ])
