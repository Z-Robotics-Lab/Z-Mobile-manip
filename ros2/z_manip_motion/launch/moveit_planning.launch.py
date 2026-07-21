"""Start a plan-only MoveIt move_group and the audited trajectory bridge."""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from z_manip_motion.moveit_config import (
    moveit_sensor_parameters,
    normalized_robot_description,
)


def _read_text(path: str, label: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise RuntimeError(f"{label} file does not exist: {candidate}")
    return candidate.read_text()


def _read_yaml(path: str, label: str):
    try:
        value = yaml.safe_load(_read_text(path, label))
    except yaml.YAMLError as error:
        raise RuntimeError(f"invalid {label} YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} YAML must contain a mapping: {path}")
    return value


def _launch_setup(context):
    robot_description_file = LaunchConfiguration("robot_description_file").perform(context)
    srdf_file = LaunchConfiguration("srdf_file").perform(context)
    kinematics_file = LaunchConfiguration("kinematics_file").perform(context)
    ompl_file = LaunchConfiguration("ompl_file").perform(context)
    joint_limits_file = LaunchConfiguration("joint_limits_file").perform(context)
    sensors_file = LaunchConfiguration("sensors_file").perform(context)

    robot_description = normalized_robot_description(robot_description_file)
    semantic = _read_text(srdf_file, "SRDF")
    kinematics = _read_yaml(kinematics_file, "kinematics")
    ompl = _read_yaml(ompl_file, "OMPL")
    joint_limits = _read_yaml(joint_limits_file, "joint limits")
    sensors = _read_yaml(sensors_file, "3-D sensors")

    point_cloud_topic = LaunchConfiguration("point_cloud_topic").perform(context)
    depth_image_topic = LaunchConfiguration("depth_image_topic").perform(context)
    filtered_cloud_topic = LaunchConfiguration("filtered_cloud_topic").perform(context)
    sensor_max_range = float(LaunchConfiguration("sensor_max_range").perform(context))
    arm_state_topic = LaunchConfiguration("arm_state_topic").perform(context)
    platform_state_topic = LaunchConfiguration("platform_state_topic").perform(context)
    complete_state_topic = LaunchConfiguration("state_topic").perform(context)
    sensor_parameters = moveit_sensor_parameters(
        sensors,
        point_cloud_topic=point_cloud_topic,
        depth_image_topic=depth_image_topic,
        filtered_cloud_topic=filtered_cloud_topic,
        max_range=sensor_max_range,
    )

    move_group_parameters = {
        "robot_description": robot_description,
        "robot_description_semantic": semantic,
        "robot_description_kinematics": kinematics,
        "robot_description_planning": joint_limits,
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl,
        "allow_trajectory_execution": False,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "octomap_frame": LaunchConfiguration("octomap_frame").perform(context),
        "octomap_resolution": float(
            LaunchConfiguration("octomap_resolution").perform(context),
        ),
        "use_sim_time": LaunchConfiguration("use_sim_time").perform(context).lower()
        == "true",
    }
    move_group_parameters.update(sensor_parameters)
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[move_group_parameters],
        remappings=[
            ("joint_states", LaunchConfiguration("state_topic")),
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        condition=IfCondition(LaunchConfiguration("start_move_group")),
    )
    complete_state = Node(
        package="z_manip_motion",
        executable="complete_joint_state",
        name="z_manip_complete_joint_state",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time").perform(context).lower()
            == "true",
            "robot_description_file": robot_description_file,
            "input_topics": list(dict.fromkeys([arm_state_topic, platform_state_topic])),
            "output_topic": complete_state_topic,
            "state_max_age_s": float(
                LaunchConfiguration("state_max_age_s").perform(context),
            ),
            "state_max_stamp_skew_s": float(
                LaunchConfiguration("state_max_stamp_skew_s").perform(context),
            ),
            "clock_handover_quiet_s": float(
                LaunchConfiguration("clock_handover_quiet_s").perform(context),
            ),
        }],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="z_manip_robot_state_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_robot_state_publisher")),
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": LaunchConfiguration("use_sim_time").perform(context).lower()
            == "true",
        }],
        remappings=[
            ("joint_states", LaunchConfiguration("state_topic")),
        ],
    )
    bridge = Node(
        package="z_manip_motion",
        executable="motion_plan_bridge",
        name="z_manip_motion_plan_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_motion_plan_bridge")),
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time").perform(context).lower()
                == "true",
                "state_topic": LaunchConfiguration("state_topic").perform(context),
                "joint_goal_topic": LaunchConfiguration("joint_goal_topic").perform(context),
                "pose_goal_topic": LaunchConfiguration("pose_goal_topic").perform(context),
                "trajectory_topic": LaunchConfiguration("trajectory_topic").perform(context),
                "planning_service": LaunchConfiguration("planning_service").perform(context),
                "planning_group": LaunchConfiguration("planning_group").perform(context),
                "planning_frame": LaunchConfiguration("planning_frame").perform(context),
                "end_effector_link": LaunchConfiguration("end_effector_link").perform(context),
                "state_max_age_s": float(
                    LaunchConfiguration("state_max_age_s").perform(context),
                ),
            },
        ],
    )
    return [complete_state, robot_state_publisher, move_group, bridge]


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("z_manip_motion"))
    config = share / "config"
    arguments = [
        DeclareLaunchArgument(
            "robot_description_file",
            default_value=EnvironmentVariable(
                "Z_MANIP_ROBOT_DESCRIPTION_FILE",
                default_value="",
            ),
            description="Absolute external URDF path; required.",
        ),
        DeclareLaunchArgument("srdf_file", default_value=str(config / "piper.srdf")),
        DeclareLaunchArgument(
            "kinematics_file",
            default_value=str(config / "kinematics_trac_ik.yaml"),
        ),
        DeclareLaunchArgument("ompl_file", default_value=str(config / "ompl_planning.yaml")),
        DeclareLaunchArgument("joint_limits_file", default_value=str(config / "joint_limits.yaml")),
        DeclareLaunchArgument(
            "sensors_file",
            default_value=str(config / "sensors_3d_pointcloud.yaml"),
        ),
        DeclareLaunchArgument("arm_state_topic", default_value="/piper/state"),
        DeclareLaunchArgument("platform_state_topic", default_value="/go2w/joint_states"),
        DeclareLaunchArgument(
            "state_topic",
            default_value="/z_manip/motion/complete_joint_states",
        ),
        DeclareLaunchArgument("state_max_age_s", default_value="0.25"),
        DeclareLaunchArgument("state_max_stamp_skew_s", default_value="0.25"),
        DeclareLaunchArgument("clock_handover_quiet_s", default_value="0.5"),
        DeclareLaunchArgument("joint_goal_topic", default_value="/z_manip/motion/joint_goal"),
        DeclareLaunchArgument("pose_goal_topic", default_value="/z_manip/motion/pose_goal"),
        DeclareLaunchArgument("trajectory_topic", default_value="/piper/joint_trajectory"),
        DeclareLaunchArgument("planning_service", default_value="/plan_kinematic_path"),
        DeclareLaunchArgument("point_cloud_topic", default_value="/camera/depth/color/points"),
        DeclareLaunchArgument(
            "depth_image_topic",
            default_value="/camera/aligned_depth_to_color/image_raw",
        ),
        DeclareLaunchArgument(
            "filtered_cloud_topic",
            default_value="/z_manip/motion/filtered_cloud",
        ),
        DeclareLaunchArgument("planning_group", default_value="piper_arm"),
        DeclareLaunchArgument("planning_frame", default_value="base"),
        DeclareLaunchArgument("end_effector_link", default_value="piper_gripper_base"),
        DeclareLaunchArgument("octomap_frame", default_value="base"),
        DeclareLaunchArgument("octomap_resolution", default_value="0.025"),
        DeclareLaunchArgument("sensor_max_range", default_value="2.5"),
        DeclareLaunchArgument("start_move_group", default_value="true"),
        DeclareLaunchArgument("start_robot_state_publisher", default_value="true"),
        DeclareLaunchArgument("start_motion_plan_bridge", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]
    return LaunchDescription([*arguments, OpaqueFunction(function=_launch_setup)])
