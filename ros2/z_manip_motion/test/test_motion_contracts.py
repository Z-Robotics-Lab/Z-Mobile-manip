from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

from z_manip_motion.contracts import (
    PIPER_ARM_JOINTS,
    ContractError,
    TrajectoryPointData,
    normalize_quaternion,
    ordered_joint_positions,
    validate_planned_trajectory,
    validate_trajectory_start,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def test_srdf_group_is_six_axis_arm_and_stops_before_gripper_joints():
    root = ElementTree.parse(CONFIG / "piper.srdf").getroot()
    group = root.find("./group[@name='piper_arm']")
    assert group is not None
    chain = group.find("chain")
    assert chain is not None
    assert chain.attrib == {
        "base_link": "piper_base_link",
        "tip_link": "piper_gripper_base",
    }
    state_names = tuple(
        joint.attrib["name"]
        for joint in root.findall("./group_state[@group='piper_arm']/joint")
    )
    assert state_names == PIPER_ARM_JOINTS
    text = (CONFIG / "piper.srdf").read_text()
    assert "piper_joint7" not in text
    assert "piper_joint8" not in text


def test_kinematics_ompl_and_octomap_configs_are_deployable_and_switchable():
    kdl = yaml.safe_load((CONFIG / "kinematics.yaml").read_text())
    trac = yaml.safe_load((CONFIG / "kinematics_trac_ik.yaml").read_text())
    ompl = yaml.safe_load((CONFIG / "ompl_planning.yaml").read_text())
    cloud = yaml.safe_load((CONFIG / "sensors_3d_pointcloud.yaml").read_text())
    depth = yaml.safe_load((CONFIG / "sensors_3d_depth.yaml").read_text())

    assert kdl["piper_arm"]["kinematics_solver"] == (
        "kdl_kinematics_plugin/KDLKinematicsPlugin"
    )
    assert "TRAC_IKKinematicsPlugin" in trac["piper_arm"]["kinematics_solver"]
    assert ompl["piper_arm"]["default_planner_config"] == "RRTConnect"
    assert ompl["planner_configs"]["RRTConnect"]["type"] == "geometric::RRTConnect"
    assert cloud["sensors"]["wrist_point_cloud"]["sensor_plugin"].endswith(
        "PointCloudOctomapUpdater"
    )
    assert depth["sensors"]["wrist_depth"]["sensor_plugin"].endswith(
        "DepthImageOctomapUpdater"
    )


def test_current_state_is_name_ordered_while_goals_reject_extra_joints():
    names = ("leg_joint", *reversed(PIPER_ARM_JOINTS))
    positions = (99.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)
    assert ordered_joint_positions(
        names,
        positions,
        allow_extra=True,
    ) == pytest.approx((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    with pytest.raises(ContractError, match="non-group"):
        ordered_joint_positions(names, positions, allow_extra=False)
    with pytest.raises(ContractError, match="missing"):
        ordered_joint_positions(PIPER_ARM_JOINTS[:-1], [0.0] * 5, allow_extra=False)


def test_trajectory_audit_reorders_six_joints_and_fails_closed():
    incoming = tuple(reversed(PIPER_ARM_JOINTS))
    points = [
        TrajectoryPointData(
            positions=(6, 5, 4, 3, 2, 1),
            velocities=(0, 0, 0, 0, 0, 0),
            time_from_start_s=0.1,
        ),
        TrajectoryPointData(
            positions=(6.1, 5.1, 4.1, 3.1, 2.1, 1.1),
            time_from_start_s=0.2,
        ),
    ]
    assert validate_planned_trajectory(incoming, points) == (5, 4, 3, 2, 1, 0)

    with pytest.raises(ContractError, match="empty"):
        validate_planned_trajectory(PIPER_ARM_JOINTS, [])
    with pytest.raises(ContractError, match="wrong length"):
        validate_planned_trajectory(
            PIPER_ARM_JOINTS,
            [TrajectoryPointData(positions=(0.0,) * 5, time_from_start_s=0.1)],
        )
    with pytest.raises(ContractError, match="increase strictly"):
        validate_planned_trajectory(
            PIPER_ARM_JOINTS,
            [
                TrajectoryPointData((0.0,) * 6, time_from_start_s=0.1),
                TrajectoryPointData((0.0,) * 6, time_from_start_s=0.1),
            ],
        )


def test_pose_quaternion_is_normalized_and_degenerate_input_rejected():
    assert normalize_quaternion((0.0, 0.0, 0.0, 2.0)) == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ContractError, match="near-zero"):
        normalize_quaternion((0.0, 0.0, 0.0, 0.0))


def test_trajectory_start_must_still_match_latest_measured_state():
    validate_trajectory_start(
        (0.0, 0.8, -1.2, 0.0, 0.4, 0.0),
        (0.01, 0.79, -1.2, 0.0, 0.4, 0.0),
        tolerance=0.02,
    )
    with pytest.raises(ContractError, match="differs from measured"):
        validate_trajectory_start(
            (0.0, 0.8, -1.2, 0.0, 0.4, 0.0),
            (0.2, 0.8, -1.2, 0.0, 0.4, 0.0),
            tolerance=0.02,
        )


def test_package_is_plan_only_parameterized_and_contains_no_sim_truth_contract():
    package = ElementTree.parse(ROOT / "package.xml").getroot()
    dependencies = {element.text for element in package.findall("exec_depend")}
    assert {
        "moveit_msgs",
        "moveit_ros_move_group",
        "moveit_ros_perception",
        "moveit_planners_ompl",
        "robot_state_publisher",
        "rosgraph_msgs",
    } <= dependencies

    node_source = (ROOT / "z_manip_motion" / "moveit_plan_bridge.py").read_text()
    assembler_source = (
        ROOT / "z_manip_motion" / "joint_state_assembler.py"
    ).read_text()
    launch_source = (ROOT / "launch" / "moveit_planning.launch.py").read_text()
    assert "GetMotionPlan" in node_source
    assert "FollowJointTrajectory" not in node_source
    assert '"allow_trajectory_execution": False' in launch_source
    for parameter in (
        "robot_description_file",
        "arm_state_topic",
        "platform_state_topic",
        "state_topic",
        "state_max_age_s",
        "state_max_stamp_skew_s",
        "clock_handover_quiet_s",
        "trajectory_topic",
        "planning_service",
        "octomap_frame",
    ):
        assert parameter in launch_source
    assert 'executable="complete_joint_state"' in launch_source
    assert 'executable="robot_state_publisher"' in launch_source
    assert 'name="z_manip_robot_state_publisher"' in launch_source
    assert '"robot_description": robot_description' in launch_source
    assert 'default_value="/z_manip/motion/complete_joint_states"' in launch_source
    assert '("joint_states", LaunchConfiguration("state_topic"))' in launch_source
    assert 'default_value=str(config / "kinematics_trac_ik.yaml")' in launch_source
    assert "move_group_parameters.update(sensor_parameters)" in launch_source
    assert 'DeclareLaunchArgument("start_motion_plan_bridge", default_value="true")' in (
        launch_source
    )
    assert 'DeclareLaunchArgument("start_robot_state_publisher", default_value="true")' in (
        launch_source
    )
    assert (
        'condition=IfCondition(LaunchConfiguration("start_robot_state_publisher"))'
        in launch_source
    )
    assert (
        'condition=IfCondition(LaunchConfiguration("start_motion_plan_bridge"))'
        in launch_source
    )
    assert "message.header.stamp.sec" in assembler_source
    assert "complete.stamp_ns" in assembler_source
    assert "output.header.stamp.sec = complete.stamp_ns" in assembler_source
    text_suffixes = {".py", ".xml", ".yaml", ".srdf", ".md", ".cfg"}
    runtime_roots = (ROOT / "z_manip_motion", ROOT / "launch", ROOT / "config")
    combined = "\n".join(
        path.read_text()
        for runtime_root in runtime_roots
        for path in runtime_root.rglob("*")
        if path.is_file() and path.suffix in text_suffixes
    )
    assert "/objects/" not in combined
    assert "get_object_gt_pose" not in combined


def test_joint_state_node_keeps_rclpy_readers_stable_across_clock_resets():
    source = (ROOT / "z_manip_motion" / "joint_state_assembler.py").read_text()

    assert "self._state_subscriptions = self._create_state_subscriptions()" in source
    assert "self._subscriptions =" not in source
    assert "destroy_subscription" not in source
    assert source.count("self._create_state_subscriptions()") == 1
    assert source.count("get_publishers_info_by_topic") == 1
    assert "_CLOCK_GRAPH_PROBE_PERIOD_S" in source
    assert "state acceptance quarantined" in source
