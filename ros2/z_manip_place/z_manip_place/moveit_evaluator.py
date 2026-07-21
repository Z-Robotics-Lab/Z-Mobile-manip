"""MoveIt-backed IK, collision, and path evaluator for placement candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan
import numpy as np
from shape_msgs.msg import SolidPrimitive

from z_manip.models.planner import PlanningError
from z_manip.planning import PlacementCandidate
from z_manip.planning_control import checkpoint, PlanningControl

from .attached_collision import AttachedObjectPathAuditor
from .core import (
    combine_trajectory_segments,
    EvaluatedPlacementMotion,
    RawTrajectorySegment,
)


@dataclass(frozen=True)
class MoveItPlacementConfig:
    """MoveIt request and trajectory audit parameters."""

    planning_frame: str
    planning_group: str
    tool_link: str
    joint_names: tuple[str, ...]
    joint_velocity_limits: tuple[float, ...]
    planning_pipeline: str
    planner_id: str
    planning_attempts: int
    allowed_planning_time_s: float
    service_wait_timeout_s: float
    response_timeout_s: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    cartesian_step_m: float
    cartesian_jump_threshold: float
    min_cartesian_fraction: float
    min_waypoint_duration_s: float
    continuity_tolerance_rad: float

    def __post_init__(self) -> None:
        if not self.planning_frame or not self.planning_group or not self.tool_link:
            raise ValueError('MoveIt placement frames and group must be non-empty')
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError('MoveIt placement joint names must be unique')
        limits = np.asarray(self.joint_velocity_limits, dtype=float)
        if limits.shape != (len(self.joint_names),) or np.any(limits <= 0.0):
            raise ValueError('joint velocity limits must be positive and match joints')
        positive = (
            self.allowed_planning_time_s,
            self.service_wait_timeout_s,
            self.response_timeout_s,
            self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.cartesian_step_m,
            self.min_waypoint_duration_s,
            self.continuity_tolerance_rad,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError('MoveIt placement tolerances and timeouts must be positive')
        if self.planning_attempts < 1:
            raise ValueError('MoveIt placement attempts must be positive')
        if self.cartesian_jump_threshold < 0.0:
            raise ValueError('Cartesian jump threshold cannot be negative')
        if not 0.0 < self.min_cartesian_fraction <= 1.0:
            raise ValueError('minimum Cartesian fraction must be in (0, 1]')


def _pose(matrix: np.ndarray) -> Pose:
    result = Pose()
    result.position.x, result.position.y, result.position.z = (
        float(value) for value in matrix[:3, 3]
    )
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    result.orientation.x = qx / norm
    result.orientation.y = qy / norm
    result.orientation.z = qz / norm
    result.orientation.w = qw / norm
    return result


def _duration_seconds(duration: object) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


class MoveItPlacementEvaluator:
    """Evaluate transit with OMPL and surface-normal phases as Cartesian paths."""

    def __init__(
        self,
        node: object,
        config: MoveItPlacementConfig,
        *,
        motion_service: str,
        cartesian_service: str,
        attached_collision_auditor: AttachedObjectPathAuditor,
    ) -> None:
        self.node = node
        self.config = config
        self.motion_client = node.create_client(GetMotionPlan, motion_service)
        self.cartesian_client = node.create_client(GetCartesianPath, cartesian_service)
        self.attached_collision_auditor = attached_collision_auditor
        self.goal_id = ''

    def set_goal_id(self, goal_id: str) -> None:
        """Bind the immutable output contract identifier for one planning run."""
        self.goal_id = goal_id

    def _state(self, positions: np.ndarray) -> RobotState:
        state = RobotState()
        state.joint_state.header.stamp = self.node.get_clock().now().to_msg()
        state.joint_state.name = list(self.config.joint_names)
        state.joint_state.position = [float(value) for value in positions]
        state.is_diff = True
        return state

    def _constraints(self, pose: np.ndarray) -> Constraints:
        target = _pose(pose)
        constraints = Constraints()
        constraints.name = 'observed_placement_pose'
        position = PositionConstraint()
        position.header.frame_id = self.config.planning_frame
        position.link_name = self.config.tool_link
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [self.config.position_tolerance_m]
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [target]
        position.weight = 1.0
        orientation = OrientationConstraint()
        orientation.header.frame_id = self.config.planning_frame
        orientation.link_name = self.config.tool_link
        orientation.orientation = target.orientation
        orientation.absolute_x_axis_tolerance = self.config.orientation_tolerance_rad
        orientation.absolute_y_axis_tolerance = self.config.orientation_tolerance_rad
        orientation.absolute_z_axis_tolerance = self.config.orientation_tolerance_rad
        orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
        orientation.weight = 1.0
        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        return constraints

    def _wait(
        self,
        client: object,
        label: str,
        control: PlanningControl | None,
    ) -> None:
        operation = f'{label} service wait'
        wait_control = (
            PlanningControl(
                deadline_s=time.monotonic() + self.config.service_wait_timeout_s,
            )
            if control is None
            else control.limited_to(
                self.config.service_wait_timeout_s,
                operation,
            )
        )
        while True:
            checkpoint(wait_control, operation)
            if client.wait_for_service(timeout_sec=0.02):
                checkpoint(wait_control, operation)
                return

    def _call_async(
        self,
        client: object,
        request: object,
        label: str,
        control: PlanningControl | None,
    ) -> object:
        """Poll one cancellable future inside both local and transaction bounds."""
        operation = f'{label} response'
        call_control = (
            PlanningControl(
                deadline_s=time.monotonic() + self.config.response_timeout_s,
            )
            if control is None
            else control.limited_to(self.config.response_timeout_s, operation)
        )
        checkpoint(call_control, operation)
        future = client.call_async(request)
        try:
            while not future.done():
                checkpoint(call_control, operation)
                time.sleep(0.01)
            checkpoint(call_control, operation)
            response = future.result()
            if response is None:
                raise PlanningError(f'{label} MoveIt service returned no response')
            return response
        except BaseException:
            if not future.done():
                future.cancel()
            raise

    def _retime(self, positions: np.ndarray) -> np.ndarray:
        limits = np.asarray(self.config.joint_velocity_limits, dtype=float)
        times = [0.0]
        for first, second in zip(positions, positions[1:]):
            duration = max(
                self.config.min_waypoint_duration_s,
                float(np.max(np.abs(second - first) / limits)),
            )
            times.append(times[-1] + duration)
        return np.asarray(times)

    def _segment(
        self,
        phase: str,
        trajectory: object,
        expected_start: np.ndarray,
    ) -> RawTrajectorySegment:
        names = tuple(trajectory.joint_names)
        if set(names) != set(self.config.joint_names) or len(names) != len(set(names)):
            raise PlanningError(f'{phase} MoveIt trajectory has the wrong joints')
        positions = np.asarray([point.positions for point in trajectory.points], dtype=float)
        if positions.ndim != 2 or positions.shape[1] != len(names) or len(positions) < 1:
            raise PlanningError(f'{phase} MoveIt trajectory is empty or malformed')
        start_in_order = np.asarray([
            expected_start[self.config.joint_names.index(name)] for name in names
        ])
        if np.max(np.abs(positions[0] - start_in_order)) > 1e-7:
            positions = np.vstack((start_in_order, positions))
            times = np.asarray([0.0] + [
                _duration_seconds(point.time_from_start)
                for point in trajectory.points
            ])
        else:
            times = np.asarray([
                _duration_seconds(point.time_from_start)
                for point in trajectory.points
            ])
        if len(positions) < 2:
            raise PlanningError(f'{phase} MoveIt trajectory contains no motion')
        if times.shape != (len(positions),) or times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
            canonical_indices = [
                names.index(name) for name in self.config.joint_names
            ]
            times = self._retime(positions[:, canonical_indices])
        return RawTrajectorySegment(phase, names, positions, times)

    def _transit(
        self,
        current: np.ndarray,
        goal: np.ndarray,
        control: PlanningControl | None,
    ) -> RawTrajectorySegment:
        self._wait(self.motion_client, 'motion planning', control)
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = self.config.planning_group
        motion.pipeline_id = self.config.planning_pipeline
        motion.planner_id = self.config.planner_id
        motion.num_planning_attempts = self.config.planning_attempts
        motion.allowed_planning_time = self.config.allowed_planning_time_s
        motion.start_state = self._state(current)
        motion.goal_constraints = [self._constraints(goal)]
        response = self._call_async(
            self.motion_client,
            request,
            'motion planning',
            control,
        )
        result = response.motion_plan_response
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise PlanningError(f'MoveIt transit failed with code {result.error_code.val}')
        return self._segment('transit', result.trajectory.joint_trajectory, current)

    def _cartesian(
        self,
        phase: str,
        current: np.ndarray,
        goal: np.ndarray,
        control: PlanningControl | None,
    ) -> RawTrajectorySegment:
        self._wait(self.cartesian_client, 'Cartesian planning', control)
        request = GetCartesianPath.Request()
        request.header.frame_id = self.config.planning_frame
        request.start_state = self._state(current)
        request.group_name = self.config.planning_group
        request.link_name = self.config.tool_link
        request.waypoints = [_pose(goal)]
        request.max_step = self.config.cartesian_step_m
        request.jump_threshold = self.config.cartesian_jump_threshold
        request.avoid_collisions = True
        response = self._call_async(
            self.cartesian_client,
            request,
            f'{phase} Cartesian planning',
            control,
        )
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise PlanningError(
                f'MoveIt {phase} Cartesian path failed with code '
                f'{response.error_code.val}',
            )
        fraction_is_safe = (
            math.isfinite(response.fraction)
            and response.fraction >= self.config.min_cartesian_fraction
        )
        if not fraction_is_safe:
            raise PlanningError(
                f'MoveIt {phase} Cartesian path covered only {response.fraction:.3f}',
            )
        return self._segment(phase, response.solution.joint_trajectory, current)

    def evaluate(
        self,
        candidate: PlacementCandidate,
        current: np.ndarray,
        *,
        control: PlanningControl | None = None,
    ) -> EvaluatedPlacementMotion:
        """Return only a complete collision-checked three-phase motion."""
        if not self.goal_id:
            raise PlanningError('placement evaluator has no active goal_id')
        checkpoint(control, 'placement MoveIt evaluation')
        transit = self._transit(current, candidate.preplace_pose, control)
        transit_end = np.asarray(transit.positions[-1])[
            [transit.joint_names.index(name) for name in self.config.joint_names]
        ]
        approach = self._cartesian(
            'approach', transit_end, candidate.place_pose, control,
        )
        approach_end = np.asarray(approach.positions[-1])[
            [approach.joint_names.index(name) for name in self.config.joint_names]
        ]
        retreat = self._cartesian(
            'retreat', approach_end, candidate.retreat_pose, control,
        )
        segments = (transit, approach, retreat)
        self.attached_collision_auditor.audit(
            segments=segments,
            control=control,
        )
        checkpoint(control, 'placement trajectory combination')
        trajectory = combine_trajectory_segments(
            goal_id=self.goal_id,
            frame_id=self.config.planning_frame,
            expected_joint_names=self.config.joint_names,
            start_positions=current,
            segments=segments,
            continuity_tolerance_rad=self.config.continuity_tolerance_rad,
        )
        positions = np.asarray([point.positions for point in trajectory.points])
        joint_distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        return EvaluatedPlacementMotion(
            score=-0.01 * joint_distance,
            trajectory=trajectory,
        )


__all__ = ['MoveItPlacementConfig', 'MoveItPlacementEvaluator']
