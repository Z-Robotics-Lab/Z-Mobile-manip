"""ROS 2 bridge that plans with MoveIt and publishes only audited trajectories."""

from __future__ import annotations

import time
import math
from copy import deepcopy

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetMotionPlan
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .contracts import (
    PIPER_ARM_JOINTS,
    ContractError,
    TrajectoryPointData,
    normalize_quaternion,
    ordered_joint_positions,
    validate_arm_joint_names,
    validate_position,
    validate_planned_trajectory,
    validate_trajectory_start,
    validate_workspace_bounds,
)


class MotionPlanBridge(Node):
    """Call ``GetMotionPlan`` and publish successful arm-only trajectories."""

    def __init__(self) -> None:
        super().__init__("z_manip_motion_plan_bridge")
        self.declare_parameter("planning_group", "piper_arm")
        self.declare_parameter("arm_joint_names", list(PIPER_ARM_JOINTS))
        self.declare_parameter("planning_frame", "base")
        self.declare_parameter("end_effector_link", "piper_gripper_base")
        self.declare_parameter("state_topic", "/piper/state")
        self.declare_parameter("joint_goal_topic", "/z_manip/motion/joint_goal")
        self.declare_parameter("pose_goal_topic", "/z_manip/motion/pose_goal")
        self.declare_parameter("trajectory_topic", "/piper/joint_trajectory")
        self.declare_parameter("planning_service", "/plan_kinematic_path")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planner_id", "RRTConnect")
        self.declare_parameter("planning_attempts", 3)
        self.declare_parameter("allowed_planning_time_s", 3.0)
        self.declare_parameter("service_timeout_s", 4.0)
        self.declare_parameter("state_max_age_s", 0.25)
        self.declare_parameter("velocity_scaling", 0.35)
        self.declare_parameter("acceleration_scaling", 0.25)
        self.declare_parameter("joint_tolerance_rad", 0.015)
        self.declare_parameter("position_tolerance_m", 0.008)
        self.declare_parameter("orientation_tolerance_rad", 0.06)
        self.declare_parameter("trajectory_start_tolerance_rad", 0.03)
        self.declare_parameter("workspace_min", [-1.0, -1.0, -0.3])
        self.declare_parameter("workspace_max", [1.0, 1.0, 1.5])

        self._group = self.get_parameter("planning_group").value
        self._joints = validate_arm_joint_names(
            self.get_parameter("arm_joint_names").value,
        )
        self._planning_frame = self.get_parameter("planning_frame").value
        self._tip_link = self.get_parameter("end_effector_link").value
        self._workspace_min, self._workspace_max = validate_workspace_bounds(
            self.get_parameter("workspace_min").value,
            self.get_parameter("workspace_max").value,
        )
        for label, value in (
            ("planning_group", self._group),
            ("planning_frame", self._planning_frame),
            ("end_effector_link", self._tip_link),
        ):
            if not isinstance(value, str) or not value:
                raise ContractError(f"{label} must be non-empty")

        state_topic = self.get_parameter("state_topic").value
        joint_goal_topic = self.get_parameter("joint_goal_topic").value
        pose_goal_topic = self.get_parameter("pose_goal_topic").value
        trajectory_topic = self.get_parameter("trajectory_topic").value
        planning_service = self.get_parameter("planning_service").value
        self._trajectory_publisher = self.create_publisher(
            JointTrajectory,
            trajectory_topic,
            5,
        )
        self.create_subscription(
            JointState,
            state_topic,
            self._on_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(JointState, joint_goal_topic, self._on_joint_goal, 5)
        self.create_subscription(PoseStamped, pose_goal_topic, self._on_pose_goal, 5)
        self._plan_client = self.create_client(GetMotionPlan, planning_service)
        self._deadline_timer = self.create_timer(0.05, self._check_deadline)

        self._current_positions: tuple[float, ...] | None = None
        self._state_received_at: float | None = None
        self._pending = None
        self._pending_started_at: float | None = None
        self._pending_token = 0
        self.get_logger().info(
            f"planning group={self._group} joints={list(self._joints)}; "
            f"successful plans publish to {trajectory_topic}",
        )

    def _on_state(self, message: JointState) -> None:
        try:
            positions = ordered_joint_positions(
                message.name,
                message.position,
                expected=self._joints,
                allow_extra=True,
            )
        except ContractError as error:
            self.get_logger().error(f"rejecting malformed arm state: {error}")
            self._current_positions = None
            self._state_received_at = None
            return
        self._current_positions = positions
        self._state_received_at = time.monotonic()

    def _fresh_start_state(self) -> tuple[float, ...]:
        if self._current_positions is None or self._state_received_at is None:
            raise ContractError("no valid arm state has been received")
        age = time.monotonic() - self._state_received_at
        max_age = float(self.get_parameter("state_max_age_s").value)
        if not math.isfinite(max_age) or max_age <= 0.0 or not 0.0 <= age <= max_age:
            raise ContractError(f"arm state is stale ({age:.3f} s > {max_age:.3f} s)")
        return self._current_positions

    def _on_joint_goal(self, message: JointState) -> None:
        try:
            goal = ordered_joint_positions(
                message.name,
                message.position,
                expected=self._joints,
                allow_extra=False,
            )
            tolerance = float(self.get_parameter("joint_tolerance_rad").value)
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ContractError("joint_tolerance_rad must be positive")
            constraints = Constraints()
            constraints.name = "joint_goal"
            for name, position in zip(self._joints, goal):
                constraint = JointConstraint()
                constraint.joint_name = name
                constraint.position = position
                constraint.tolerance_above = tolerance
                constraint.tolerance_below = tolerance
                constraint.weight = 1.0
                constraints.joint_constraints.append(constraint)
            self._begin_plan(constraints, "joint goal")
        except ContractError as error:
            self.get_logger().error(f"joint goal rejected without execution: {error}")

    def _on_pose_goal(self, message: PoseStamped) -> None:
        try:
            frame = message.header.frame_id or self._planning_frame
            pose_position = validate_position(
                (message.pose.position.x, message.pose.position.y, message.pose.position.z),
            )
            quaternion = normalize_quaternion(
                (
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                    message.pose.orientation.w,
                ),
            )
            position_tolerance = float(self.get_parameter("position_tolerance_m").value)
            orientation_tolerance = float(
                self.get_parameter("orientation_tolerance_rad").value,
            )
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (position_tolerance, orientation_tolerance)
            ):
                raise ContractError("pose tolerances must be positive")

            constraints = Constraints()
            constraints.name = "pose_goal"
            position = PositionConstraint()
            position.header = deepcopy(message.header)
            position.header.frame_id = frame
            position.link_name = self._tip_link
            sphere = SolidPrimitive()
            sphere.type = SolidPrimitive.SPHERE
            sphere.dimensions = [position_tolerance]
            sphere_pose = Pose()
            sphere_pose.position.x, sphere_pose.position.y, sphere_pose.position.z = pose_position
            sphere_pose.orientation.w = 1.0
            position.constraint_region.primitives = [sphere]
            position.constraint_region.primitive_poses = [sphere_pose]
            position.weight = 1.0

            orientation = OrientationConstraint()
            orientation.header = deepcopy(position.header)
            orientation.link_name = self._tip_link
            (
                orientation.orientation.x,
                orientation.orientation.y,
                orientation.orientation.z,
                orientation.orientation.w,
            ) = quaternion
            orientation.absolute_x_axis_tolerance = orientation_tolerance
            orientation.absolute_y_axis_tolerance = orientation_tolerance
            orientation.absolute_z_axis_tolerance = orientation_tolerance
            orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
            orientation.weight = 1.0
            constraints.position_constraints = [position]
            constraints.orientation_constraints = [orientation]
            self._begin_plan(constraints, "pose goal")
        except ContractError as error:
            self.get_logger().error(f"pose goal rejected without execution: {error}")

    def _begin_plan(self, goal: Constraints, label: str) -> None:
        if self._pending is not None:
            raise ContractError("a MoveIt request is already pending")
        if not self._plan_client.service_is_ready():
            raise ContractError("MoveIt planning service is unavailable")
        start_positions = self._fresh_start_state()
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = self._group
        motion.pipeline_id = self.get_parameter("pipeline_id").value
        motion.planner_id = self.get_parameter("planner_id").value
        motion.num_planning_attempts = int(self.get_parameter("planning_attempts").value)
        motion.allowed_planning_time = float(
            self.get_parameter("allowed_planning_time_s").value,
        )
        motion.max_velocity_scaling_factor = float(
            self.get_parameter("velocity_scaling").value,
        )
        motion.max_acceleration_scaling_factor = float(
            self.get_parameter("acceleration_scaling").value,
        )
        if (
            motion.num_planning_attempts < 1
            or not math.isfinite(motion.allowed_planning_time)
            or motion.allowed_planning_time <= 0.0
        ):
            raise ContractError("planning attempts and allowed time must be positive")
        for value, parameter in (
            (motion.max_velocity_scaling_factor, "velocity_scaling"),
            (motion.max_acceleration_scaling_factor, "acceleration_scaling"),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ContractError(f"{parameter} must be in (0, 1]")

        state = RobotState()
        state.joint_state.header.stamp = self.get_clock().now().to_msg()
        state.joint_state.name = list(self._joints)
        state.joint_state.position = list(start_positions)
        state.is_diff = True
        motion.start_state = state
        motion.goal_constraints = [goal]
        workspace = motion.workspace_parameters
        workspace.header.stamp = state.joint_state.header.stamp
        workspace.header.frame_id = self._planning_frame
        workspace.min_corner.x, workspace.min_corner.y, workspace.min_corner.z = self._workspace_min
        workspace.max_corner.x, workspace.max_corner.y, workspace.max_corner.z = self._workspace_max

        self._pending_token += 1
        token = self._pending_token
        self._pending_started_at = time.monotonic()
        self._pending = self._plan_client.call_async(request)
        self._pending.add_done_callback(lambda future: self._on_plan_done(token, future))
        self.get_logger().info(f"submitted collision-aware MoveIt {label}")

    def _check_deadline(self) -> None:
        if self._pending is None or self._pending_started_at is None:
            return
        timeout = float(self.get_parameter("service_timeout_s").value)
        if math.isfinite(timeout) and timeout > 0.0 and time.monotonic() - self._pending_started_at <= timeout:
            return
        future = self._pending
        self._pending = None
        self._pending_started_at = None
        self._pending_token += 1
        future.cancel()
        self.get_logger().error("MoveIt service timed out; no trajectory published")

    def _on_plan_done(self, token: int, future) -> None:
        if token != self._pending_token or future is not self._pending:
            return
        self._pending = None
        self._pending_started_at = None
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f"MoveIt service failed; no execution: {error}")
            return
        try:
            plan = response.motion_plan_response
        except AttributeError:
            self.get_logger().error("MoveIt returned a malformed response; no execution")
            return
        if plan.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt planning failed with code {plan.error_code.val}; no execution",
            )
            return
        if plan.group_name != self._group:
            self.get_logger().error(
                f"MoveIt returned group {plan.group_name!r}, expected {self._group!r}; "
                "no execution",
            )
            return
        try:
            output = self._audited_trajectory(plan.trajectory.joint_trajectory)
        except ContractError as error:
            self.get_logger().error(f"unsafe MoveIt trajectory rejected: {error}")
            return
        self._trajectory_publisher.publish(output)
        self.get_logger().info(
            f"published audited trajectory with {len(output.points)} points",
        )

    def _audited_trajectory(self, trajectory: JointTrajectory) -> JointTrajectory:
        pure_points = [
            TrajectoryPointData(
                positions=point.positions,
                velocities=point.velocities,
                accelerations=point.accelerations,
                effort=point.effort,
                time_from_start_s=(
                    float(point.time_from_start.sec)
                    + float(point.time_from_start.nanosec) * 1e-9
                ),
            )
            for point in trajectory.points
        ]
        order = validate_planned_trajectory(
            trajectory.joint_names,
            pure_points,
            expected=self._joints,
        )
        measured = self._fresh_start_state()
        first_positions = [trajectory.points[0].positions[index] for index in order]
        validate_trajectory_start(
            first_positions,
            measured,
            tolerance=float(
                self.get_parameter("trajectory_start_tolerance_rad").value,
            ),
        )
        result = JointTrajectory()
        result.header = deepcopy(trajectory.header)
        result.joint_names = list(self._joints)
        for source in trajectory.points:
            point = JointTrajectoryPoint()
            point.positions = [source.positions[index] for index in order]
            for field_name in ("velocities", "accelerations", "effort"):
                values = getattr(source, field_name)
                if values:
                    setattr(point, field_name, [values[index] for index in order])
            point.time_from_start = deepcopy(source.time_from_start)
            result.points.append(point)
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotionPlanBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
