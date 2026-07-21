"""Perception-driven grasp and motion planning composition."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable

import numpy as np

from z_manip.collision.contact_phase import check_target_contact_approach
from z_manip.collision.gripper_aperture import (
    collision_aperture_for_grasp,
    with_parallel_gripper_aperture,
)
from z_manip.collision.pointcloud import (
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
)
from z_manip.collision.pinocchio_self import PinocchioSelfCollisionChecker
from z_manip.configuration import StackConfig
from z_manip.ik.symmetry import expand_symmetry
from z_manip.kinematics import (
    fixed_transform_from_urdf,
    KinematicChain,
    PinocchioIKSolver,
)
from z_manip.kinematics.robust_ik import IKFailure, RobustIKSolver
from z_manip.models.grasp_source import (
    cascade_generate,
    GraspCandidates,
    GraspContext,
)
from z_manip.models.planner import PlanningError
from z_manip.perception.rgbd import filter_object_cloud
from z_manip.planning.grasp_pipeline import (
    CandidateFailure,
    grasp_pregrasp_pose,
    GraspPlanGenerator,
    PlannedGrasp,
    tool_tip_pose,
)
from z_manip.planning.rrt_connect import JointSpaceRRTConnect
from z_manip.planning.standoff import ReachabilityStandoffOptimizer
from z_manip.planning.time_parameterization import retime_path, TimedJointTrajectory
from z_manip.planning.work_pose import (
    BoundedSE2WorkPoseOptimizer,
    WorkPoseCandidate,
    WorkPoseDiagnostics,
    WorkPoseObservation,
    WorkPoseOptimizationError,
)
from z_manip.planning_control import checkpoint, PlanningControl


@dataclass(frozen=True, eq=False)
class PerceptionObservation:
    """One synchronized observation, expressed in the PiPER base frame."""

    serial: int
    stamp_s: float
    target_points: np.ndarray
    target_collision_points: np.ndarray
    scene_points: np.ndarray
    target_position_camera: np.ndarray
    camera_origin_piper: np.ndarray
    camera_rotation_piper: np.ndarray
    affordance: object | None


@dataclass(frozen=True, eq=False)
class MotionProgram:
    """Fully checked and retimed grasp execution program."""

    planned: PlannedGrasp
    transit: TimedJointTrajectory
    approach: TimedJointTrajectory
    lift: TimedJointTrajectory


@dataclass(frozen=True, eq=False)
class PregraspTransitProgram:
    """Stage-one output that deliberately cannot expose grasp-phase motion."""

    observation_serial: int
    candidate_index: int
    symmetry_index: int
    score: float
    failures: tuple[CandidateFailure, ...]
    transit: TimedJointTrajectory


@dataclass(frozen=True, eq=False)
class GraspCompletionProgram:
    """Freshly replanned approach, gripper aperture, and lift motion."""

    observation_serial: int
    candidate_index: int
    symmetry_index: int
    grasp_pose: np.ndarray
    pregrasp_pose: np.ndarray
    required_width_m: float | None
    score: float
    failures: tuple[CandidateFailure, ...]
    approach: TimedJointTrajectory
    lift: TimedJointTrajectory


@dataclass(frozen=True, eq=False)
class ProspectiveWorkPose:
    """Observation-derived base motion that puts the target in arm workspace."""

    relative_base_pose: np.ndarray
    desired_camera_depth_m: float
    predicted_target_position_piper: np.ndarray
    selection_mode: str
    kinematic_precheck_feasible: bool
    diagnostics: WorkPoseDiagnostics
    rejected_precheck_diagnostics: WorkPoseDiagnostics | None = None


@dataclass(frozen=True, eq=False)
class SemanticPointSelection:
    """VLM-constrained point selection with an explicit fallback reason."""

    points: np.ndarray
    mode: str
    selected_count: int
    source_count: int


def _immutable_timed_trajectory(
    trajectory: TimedJointTrajectory,
) -> TimedJointTrajectory:
    """Detach a retimed trajectory from writable planner-owned arrays."""
    positions = np.array(trajectory.positions, dtype=float, copy=True)
    times_s = np.array(trajectory.times_s, dtype=float, copy=True)
    positions.setflags(write=False)
    times_s.setflags(write=False)
    return TimedJointTrajectory(positions=positions, times_s=times_s)


def _immutable_pose(pose: object, label: str) -> np.ndarray:
    value = np.array(pose, dtype=float, copy=True)
    if (
        value.shape != (4, 4)
        or not np.all(np.isfinite(value))
        or not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise PlanningError(f'{label} must be a finite homogeneous transform')
    value.setflags(write=False)
    return value


def _join_contiguous_joint_paths(
    first: object,
    second: object,
    *,
    label: str,
) -> np.ndarray:
    """Join planner phases while rejecting a discontinuous hidden joint jump."""
    first_path = np.asarray(first, dtype=float)
    second_path = np.asarray(second, dtype=float)
    if (
        first_path.ndim != 2
        or second_path.ndim != 2
        or first_path.shape[1:] != second_path.shape[1:]
        or len(first_path) < 1
        or len(second_path) < 1
        or not np.all(np.isfinite(first_path))
        or not np.all(np.isfinite(second_path))
    ):
        raise PlanningError(f'{label} contains an invalid joint path')
    if not np.allclose(first_path[-1], second_path[0], atol=1e-7, rtol=0.0):
        raise PlanningError(f'{label} has a discontinuous phase boundary')
    joined = np.vstack((first_path, second_path[1:]))
    if len(joined) > 1:
        moving = np.linalg.norm(np.diff(joined, axis=0), axis=1) >= 1e-12
        joined = joined[np.concatenate(([True], moving))]
    if len(joined) < 2:
        raise PlanningError(f'{label} contains no executable motion')
    return joined


def _attached_target_at_joints(
    chain: KinematicChain,
    target_points: np.ndarray,
    attachment_joints: np.ndarray,
    joints: np.ndarray,
) -> np.ndarray:
    """Transform an attached payload from its observed state to new joints."""
    base_t_tip_at_attachment = chain.forward(attachment_joints)
    tip_t_base_at_attachment = np.linalg.inv(base_t_tip_at_attachment)
    target_tip = (
        target_points @ tip_t_base_at_attachment[:3, :3].T
        + tip_t_base_at_attachment[:3, 3]
    )
    base_t_tip = chain.forward(joints)
    return (
        target_tip @ base_t_tip[:3, :3].T
        + base_t_tip[:3, 3]
    )


def _attached_target_departure_direction(
    chain: KinematicChain,
    target_points: np.ndarray,
    attachment_joints: np.ndarray,
    end_joints: np.ndarray,
) -> np.ndarray:
    """Measure payload-centroid departure in the chain base frame."""
    target_at_end = _attached_target_at_joints(
        chain,
        target_points,
        attachment_joints,
        end_joints,
    )
    displacement = np.mean(target_at_end, axis=0) - np.mean(target_points, axis=0)
    distance = float(np.linalg.norm(displacement))
    if not np.isfinite(distance) or distance <= 1e-9:
        raise ValueError('lift path has no measurable payload departure direction')
    return displacement / distance


def select_semantic_target_points(
    xyz: object,
    uv: object | None,
    affordance: object | None,
    *,
    image_width: int,
    image_height: int,
    min_points: int = 40,
) -> SemanticPointSelection:
    """
    Apply semantic contact and no-grasp regions to an aligned mask cloud.

    Avoid regions are always removed. A grasp-part region is preferred when it
    contains enough supported depth points; otherwise the function explicitly
    reports fallback to the remaining whole-target mask.
    """
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('target xyz must be a finite (N, 3) array')
    if min_points < 1 or image_width < 1 or image_height < 1:
        raise ValueError('image dimensions and min_points must be positive')
    if uv is None or not isinstance(affordance, dict):
        if len(points) < min_points:
            raise ValueError('target cloud has too few points')
        return SemanticPointSelection(
            points, 'full_target_no_pixel_semantics', len(points), len(points),
        )
    pixels = np.asarray(uv, dtype=float)
    if pixels.shape != (len(points), 2) or not np.all(np.isfinite(pixels)):
        raise ValueError('target uv must align with xyz and be finite')

    current_target = np.array([
        np.min(pixels[:, 0]) / image_width,
        np.min(pixels[:, 1]) / image_height,
        (np.max(pixels[:, 0]) + 1.0) / image_width,
        (np.max(pixels[:, 1]) + 1.0) / image_height,
    ])
    initial_target = None
    target_data = affordance.get('target')
    if isinstance(target_data, dict):
        candidate = np.asarray(
            target_data.get('bbox_xyxy_normalized'),
            dtype=float,
        )
        if (
            candidate.shape == (4,)
            and np.all(np.isfinite(candidate))
            and np.all(candidate >= 0.0)
            and np.all(candidate <= 1.0)
            and candidate[2] > candidate[0]
            and candidate[3] > candidate[1]
        ):
            initial_target = candidate

    def region_mask(box: object) -> np.ndarray:
        values = np.asarray(box, dtype=float)
        if (
            values.shape != (4,)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or np.any(values > 1.0)
            or values[2] <= values[0]
            or values[3] <= values[1]
        ):
            raise ValueError('affordance region must be normalized positive-area xyxy')
        if initial_target is not None:
            initial_size = initial_target[2:] - initial_target[:2]
            local = np.concatenate((
                (values[:2] - initial_target[:2]) / initial_size,
                (values[2:] - initial_target[:2]) / initial_size,
            ))
            local = np.clip(local, 0.0, 1.0)
            if local[2] <= local[0] or local[3] <= local[1]:
                return np.zeros(len(points), dtype=bool)
            current_size = current_target[2:] - current_target[:2]
            values = np.concatenate((
                current_target[:2] + local[:2] * current_size,
                current_target[:2] + local[2:] * current_size,
            ))
        x1, y1, x2, y2 = values * np.array(
            [image_width, image_height, image_width, image_height], dtype=float,
        )
        return (
            (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2)
            & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
        )

    allowed = np.ones(len(points), dtype=bool)
    for region in affordance.get('avoid_regions', []):
        if not isinstance(region, dict):
            raise ValueError('avoid region must be an object')
        allowed &= ~region_mask(region.get('bbox_xyxy_normalized'))
    if int(np.count_nonzero(allowed)) < min_points:
        raise ValueError('semantic avoid regions leave too few safe target points')

    grasp_part = affordance.get('grasp_part')
    if isinstance(grasp_part, dict) and grasp_part.get('bbox_xyxy_normalized') is not None:
        preferred = allowed & region_mask(grasp_part['bbox_xyxy_normalized'])
        count = int(np.count_nonzero(preferred))
        if count >= min_points:
            return SemanticPointSelection(points[preferred], 'vlm_grasp_part', count, len(points))
        return SemanticPointSelection(
            points[allowed],
            f'full_target_grasp_part_below_min:{count}<{min_points}',
            int(np.count_nonzero(allowed)),
            len(points),
        )
    return SemanticPointSelection(
        points[allowed],
        'full_target_no_grasp_part',
        int(np.count_nonzero(allowed)),
        len(points),
    )


class OnlinePlanner:
    """Compose root stack algorithms without importing ROS messages."""

    def __init__(
        self,
        config: StackConfig,
        *,
        grasp_generate: Callable[[GraspContext], GraspCandidates] = cascade_generate,
    ) -> None:
        """Load robot/collision models and configure grasp generation."""
        self.config = config
        self.chain = KinematicChain.from_urdf(
            config.robot.urdf_path,
            config.robot.base_link,
            config.robot.tip_link,
        )
        if len(config.robot.acceleration_limits) != self.chain.dof:
            raise ValueError(
                'robot acceleration limit count does not match arm chain DOF: '
                f'{len(config.robot.acceleration_limits)} != {self.chain.dof}',
            )
        model_data = json.loads(config.collision_model_path.read_text())
        self.collision_model = RobotCollisionModel.from_mapping(model_data)
        self.ik_backend = os.environ.get(
            'Z_MANIP_IK_BACKEND',
            'robust',
        ).strip().lower()
        if self.ik_backend == 'pinocchio':
            self.ik = PinocchioIKSolver(
                config.robot.urdf_path,
                self.chain,
                config.ik,
            )
            self.mesh_self_collision = PinocchioSelfCollisionChecker(
                config.robot.urdf_path,
                self.chain,
            )
        elif self.ik_backend == 'robust':
            self.ik = RobustIKSolver(self.chain, config.ik)
            self.mesh_self_collision = None
        else:
            raise ValueError(
                f'unsupported Z_MANIP_IK_BACKEND: {self.ik_backend!r}',
            )
        self.standoff = ReachabilityStandoffOptimizer(config.standoff)
        self.work_pose = BoundedSE2WorkPoseOptimizer(config.work_pose)
        self.T_platform_piper = fixed_transform_from_urdf(
            config.robot.urdf_path,
            config.robot.mount_parent_link,
            config.robot.base_link,
        )
        self.grasp_generate = grasp_generate

    def grasp_collision_aperture(self, required_width_m: object) -> float:
        """Return the plan-specific PiPER collision aperture."""

        return collision_aperture_for_grasp(
            required_width_m,
            open_aperture_m=(
                self.config.tool_geometry.collision_open_aperture_m
            ),
            grasp_margin_m=(
                self.config.tool_geometry.collision_grasp_margin_m
            ),
        )

    def _collision_model_for_grasp_width(
        self,
        required_width_m: object,
    ) -> RobotCollisionModel:
        aperture = self.grasp_collision_aperture(required_width_m)
        return with_parallel_gripper_aperture(
            self.collision_model,
            open_aperture_m=(
                self.config.tool_geometry.collision_open_aperture_m
            ),
            aperture_m=aperture,
            closing_axis=self.config.tool_geometry.tip_closing_axis,
        )

    def candidates(
        self,
        observation: PerceptionObservation,
        control: PlanningControl | None = None,
    ) -> GraspCandidates:
        """Generate learned-then-geometric 6-DoF grasps from observed points."""
        checkpoint(control, 'grasp candidate generation')
        filtered = filter_object_cloud(
            observation.target_points,
            viewpoint=observation.camera_origin_piper,
        )
        context = GraspContext(
            object_points=filtered,
            bbox=None,
            source_frame=self.config.robot.base_link,
            t_target_src=np.eye(4),
            scene_points=observation.scene_points,
            progress_cb=lambda _phase, _progress: None,
            affordance=observation.affordance,
        )
        generated = self.grasp_generate(context)
        checkpoint(control, 'grasp candidate generation')
        return generated

    def _plan(
        self,
        candidates: GraspCandidates,
        *,
        scene_points: np.ndarray,
        target_points: np.ndarray,
        current_joints: np.ndarray,
        stamp_s: float,
        pose_ranker: Callable[..., float] | None = None,
        control: PlanningControl | None = None,
    ) -> PlannedGrasp:
        def checker_with_target(
            *,
            allowed_contact_capsules: tuple[str, ...] = (),
            attachment_joints: np.ndarray | None = None,
            collision_model: RobotCollisionModel | None = None,
        ) -> PointCloudCollisionChecker:
            checkpoint(control, 'planning-scene collision checker setup')
            model = self.collision_model if collision_model is None else collision_model
            checker = PointCloudCollisionChecker(
                chain=self.chain,
                model=model,
                frame_provider=self.chain.link_transforms,
                config=PointCloudCollisionConfig(
                    clearance=model.scene_clearance_m,
                    point_radius=model.point_radius_m,
                    scene_noise_tolerance=model.scene_noise_tolerance_m,
                    scene_noise_min_support_points=(
                        model.scene_noise_min_support_points
                    ),
                    segment_joint_step=self.config.rrt.collision_resolution,
                ),
                now_fn=lambda: float(stamp_s),
                self_collision_checker=(
                    None
                    if self.mesh_self_collision is None
                    else self.mesh_self_collision.check_state
                ),
            )
            checker.update_scene(scene_points, stamp_s=stamp_s)
            if attachment_joints is None:
                checker.update_target(
                    target_points,
                    allowed_contact_capsules=allowed_contact_capsules,
                )
            else:
                checker.update_attached_target(
                    target_points,
                    attachment_joints=attachment_joints,
                    allowed_contact_capsules=allowed_contact_capsules,
                    allow_initial_scene_contact=True,
                    departure_direction_base=(
                        self.config.grasp_plan.lift_direction_base
                    ),
                )
            checkpoint(control, 'planning-scene collision checker setup')
            return checker

        transit_checker = checker_with_target()
        open_contact_checker = checker_with_target(
            allowed_contact_capsules=(
                self.collision_model.target_contact_capsules
            ),
        )
        contact_checkers: dict[float, PointCloudCollisionChecker] = {}
        attached_checker: PointCloudCollisionChecker | None = None
        attached_at: np.ndarray | None = None
        attached_aperture: float | None = None

        def approach_path_valid(
            path: object,
            path_control: PlanningControl | None,
            *,
            required_width_m: float | None = None,
        ) -> bool:
            aperture = self.grasp_collision_aperture(required_width_m)
            contact_checker = contact_checkers.get(aperture)
            if contact_checker is None:
                contact_checker = checker_with_target(
                    allowed_contact_capsules=(
                        self.collision_model.target_contact_capsules
                    ),
                    collision_model=self._collision_model_for_grasp_width(
                        required_width_m,
                    ),
                )
                contact_checkers[aperture] = contact_checker
            approach = check_target_contact_approach(
                path,
                no_contact=transit_checker,
                finger_contact=open_contact_checker,
                allowed_contact_capsules=self.collision_model.target_contact_capsules,
                control=path_control,
            )
            if not approach.valid:
                return False
            positions = np.asarray(path, dtype=float)
            checkpoint(path_control, 'closed-gripper final contact audit')
            valid = contact_checker.check_state(positions[-1]).valid
            checkpoint(path_control, 'closed-gripper final contact audit')
            return valid

        def lift_segment_valid(
            first: object,
            second: object,
            attachment_joints: object,
            *,
            required_width_m: float | None = None,
        ) -> bool:
            nonlocal attached_aperture, attached_at, attached_checker
            attachment = np.asarray(attachment_joints, dtype=float)
            aperture = self.grasp_collision_aperture(required_width_m)
            if (
                attached_at is None
                or not np.array_equal(attached_at, attachment)
                or attached_aperture != aperture
            ):
                attached_checker = checker_with_target(
                    allowed_contact_capsules=self.collision_model.target_contact_capsules,
                    attachment_joints=attachment,
                    collision_model=self._collision_model_for_grasp_width(
                        required_width_m,
                    ),
                )
                attached_at = attachment.copy()
                attached_aperture = aperture
            assert attached_checker is not None
            return attached_checker.is_segment_valid(first, second)

        joint_planner = JointSpaceRRTConnect(
            joint_names=self.chain.joint_names,
            lower_limits=self.chain.lower_limits,
            upper_limits=self.chain.upper_limits,
            state_valid=transit_checker.is_state_valid,
            config=self.config.rrt,
        )
        if pose_ranker is None:
            pose_ranker = self.ik.make_seed_pose_ranker(current_joints, control)
        return GraspPlanGenerator(
            self.ik,
            joint_planner,
            self.config.grasp_plan,
            approach_path_valid=approach_path_valid,
            lift_segment_valid=lift_segment_valid,
        ).plan(
            candidates,
            current_joints=current_joints,
            pose_ranker=pose_ranker,
            control=control,
        )

    def _new_checker(
        self,
        *,
        scene_points: np.ndarray,
        stamp_s: float,
        collision_model: RobotCollisionModel | None = None,
    ) -> PointCloudCollisionChecker:
        """Create one fail-closed checker over a synchronized scene snapshot."""
        model = self.collision_model if collision_model is None else collision_model
        checker = PointCloudCollisionChecker(
            chain=self.chain,
            model=model,
            frame_provider=self.chain.link_transforms,
            config=PointCloudCollisionConfig(
                clearance=model.scene_clearance_m,
                point_radius=model.point_radius_m,
                scene_noise_tolerance=model.scene_noise_tolerance_m,
                scene_noise_min_support_points=(
                    model.scene_noise_min_support_points
                ),
                segment_joint_step=self.config.rrt.collision_resolution,
            ),
            now_fn=lambda: float(stamp_s),
            self_collision_checker=(
                None
                if self.mesh_self_collision is None
                else self.mesh_self_collision.check_state
            ),
        )
        checker.update_scene(scene_points, stamp_s=stamp_s)
        return checker

    def prospective_standoff(
        self,
        observation: PerceptionObservation,
        current_joints: np.ndarray,
        control: PlanningControl | None = None,
        *,
        history_relative_base_poses: tuple[np.ndarray, ...] = (),
    ) -> ProspectiveWorkPose:
        """
        Choose a bounded SE(2) work pose before near-field visual servo.

        The current RGB-D observation is transformed into candidate future arm
        frames. A bounded pregrasp IK probe ranks those candidates, but failure
        of that prospective probe does not strand a distant target: the best
        geometry-only pose is retained and the full collision-checked plan is
        rebuilt from a fresh post-servo observation.
        """
        candidates = self.candidates(observation, control)
        target_pose = np.eye(4)
        target_pose[:3, 3] = np.asarray(candidates.centroid, dtype=float)
        work_observation = WorkPoseObservation(
            target_pose=target_pose,
            candidates=candidates,
            scene_points=observation.scene_points,
            current_joints=np.asarray(current_joints, dtype=float),
            T_platform_piper=self.T_platform_piper,
        )

        def evaluate(
            candidate: WorkPoseCandidate,
            *,
            control: PlanningControl | None = None,
        ) -> dict[str, object]:
            checkpoint(control, 'work-pose pregrasp IK evaluation')
            scores = np.asarray(candidate.candidates.scores, dtype=float)
            order = np.argsort(-scores, kind='stable')[:2]
            joint_span = np.maximum(
                self.chain.upper_limits - self.chain.lower_limits,
                1e-9,
            )
            feasible: list[tuple[float, int, int, object]] = []
            failures: list[str] = []
            for candidate_index in order:
                raw_grasp = np.asarray(
                    candidate.candidates.grasps[int(candidate_index)],
                    dtype=float,
                )
                family = expand_symmetry(
                    raw_grasp,
                    n_about_axis=self.config.grasp_plan.symmetry_samples,
                )
                for symmetry_index, grasp in enumerate(family):
                    checkpoint(control, 'work-pose pregrasp IK evaluation')
                    target_tip = tool_tip_pose(
                        grasp_pregrasp_pose(
                            grasp,
                            self.config.grasp_plan.pregrasp_distance_m,
                        ),
                        self.config.grasp_plan.tool_from_tip,
                    )
                    try:
                        solution = self.ik.solve(
                            target_tip,
                            current=candidate.current_joints,
                            control=control,
                        )
                    except (IKFailure, PlanningError) as error:
                        failures.append(str(error))
                        continue
                    continuity = float(np.linalg.norm(
                        (solution.joints - candidate.current_joints) / joint_span,
                    ))
                    score = (
                        float(scores[int(candidate_index)])
                        + 0.5 * float(solution.min_joint_limit_margin)
                        - 0.05 * continuity
                    )
                    feasible.append((
                        score,
                        int(candidate_index),
                        symmetry_index,
                        solution,
                    ))
            if not feasible:
                detail = failures[-1] if failures else 'no grasp hypotheses were evaluated'
                raise PlanningError(f'prospective pregrasp IK rejected work pose: {detail}')
            score, candidate_index, symmetry_index, solution = max(
                feasible,
                key=lambda item: item[0],
            )
            return {
                'score': score,
                'candidate_index': candidate_index,
                'symmetry_index': symmetry_index,
                'solution': solution,
            }

        rejected_diagnostics = None
        try:
            choice = self.work_pose.select(
                work_observation,
                evaluate=evaluate,
                history_relative_base_poses=history_relative_base_poses,
                control=control,
            )
            selection_mode = 'kinematic_precheck'
            kinematic_feasible = True
        except WorkPoseOptimizationError as error:
            rejected_diagnostics = error.diagnostics
            choice = self.work_pose.select(
                work_observation,
                history_relative_base_poses=history_relative_base_poses,
                control=control,
            )
            selection_mode = 'bounded_geometry_fallback'
            kinematic_feasible = False

        predicted_target = np.asarray(
            choice.predicted_target_pose[:3, 3],
            dtype=float,
        )
        predicted_camera = (
            observation.camera_rotation_piper.T
            @ (predicted_target - observation.camera_origin_piper)
        )
        raw_depth = float(predicted_camera[2])
        if not np.isfinite(raw_depth) or raw_depth <= 0.0:
            raise PlanningError('selected work pose puts the target behind the camera')
        desired_depth = float(np.clip(
            raw_depth,
            self.config.standoff.min_camera_depth_m,
            self.config.standoff.max_camera_depth_m,
        ))
        return ProspectiveWorkPose(
            relative_base_pose=choice.relative_base_pose.copy(),
            desired_camera_depth_m=desired_depth,
            predicted_target_position_piper=predicted_target.copy(),
            selection_mode=selection_mode,
            kinematic_precheck_feasible=kinematic_feasible,
            diagnostics=choice.diagnostics,
            rejected_precheck_diagnostics=rejected_diagnostics,
        )

    def _plan_from_observation(
        self,
        observation: PerceptionObservation,
        current_joints: np.ndarray,
        control: PlanningControl | None,
    ) -> PlannedGrasp:
        """Build one fully feasible grasp hypothesis from one observation."""
        candidates = self.candidates(observation, control)
        return self._plan(
            candidates,
            scene_points=observation.scene_points,
            target_points=observation.target_collision_points,
            current_joints=current_joints,
            stamp_s=observation.stamp_s,
            control=control,
        )

    def _retime_joint_path(
        self,
        path: object,
        *,
        allow_stationary_hold: bool = False,
    ) -> TimedJointTrajectory:
        waypoints = np.asarray(path, dtype=float)
        if (
            allow_stationary_hold
            and waypoints.ndim == 2
            and len(waypoints) >= 1
            and waypoints.shape[1:] == (self.chain.dof,)
            and np.all(np.isfinite(waypoints))
            and (
                len(waypoints) == 1
                or np.all(np.linalg.norm(
                    np.diff(waypoints, axis=0),
                    axis=1,
                ) < 1e-12)
            )
        ):
            endpoint = waypoints[-1].copy()
            return TimedJointTrajectory(
                positions=np.vstack((endpoint, endpoint)),
                times_s=np.array((
                    0.0,
                    self.config.time_parameterization.min_segment_time_s,
                )),
            )
        return retime_path(
            waypoints,
            self.chain.velocity_limits,
            np.asarray(self.config.robot.acceleration_limits, dtype=float),
            self.config.time_parameterization,
        )

    def pregrasp_program(
        self,
        observation: PerceptionObservation,
        current_joints: np.ndarray,
        control: PlanningControl | None = None,
    ) -> PregraspTransitProgram:
        """
        Plan stage one while withholding provisional approach and lift paths.

        The complete grasp hypothesis is still checked, including Cartesian
        approach, target contact, lift IK, and collision feasibility. Only its
        collision-free transit to pregrasp crosses this API boundary.
        """
        planned = self._plan_from_observation(observation, current_joints, control)
        checkpoint(control, 'pregrasp trajectory retiming')
        transit = self._retime_joint_path(
            np.asarray(planned.transit.waypoints, dtype=float),
            allow_stationary_hold=True,
        )
        return PregraspTransitProgram(
            observation_serial=int(observation.serial),
            candidate_index=int(planned.candidate_index),
            symmetry_index=int(planned.symmetry_index),
            score=float(planned.score),
            failures=tuple(planned.failures),
            transit=_immutable_timed_trajectory(transit),
        )

    def grasp_completion_program(
        self,
        pregrasp: PregraspTransitProgram,
        observation: PerceptionObservation,
        current_joints: np.ndarray,
        control: PlanningControl | None = None,
    ) -> GraspCompletionProgram:
        """
        Replan approach, aperture, and lift after stage-one execution.

        ``observation`` must be newer than the observation used for stage one.
        The returned approach begins at the measured ``current_joints`` and
        includes any collision-free correction needed to reach the freshly
        selected pregrasp before its Cartesian contact approach.
        """
        if not isinstance(pregrasp, PregraspTransitProgram):
            raise TypeError('pregrasp must be a PregraspTransitProgram')
        if int(observation.serial) <= pregrasp.observation_serial:
            raise PlanningError(
                'grasp completion requires an observation newer than pregrasp planning',
            )
        measured_joints = np.asarray(current_joints, dtype=float)
        if (
            measured_joints.shape != (self.chain.dof,)
            or not np.all(np.isfinite(measured_joints))
        ):
            raise PlanningError('measured pregrasp joints are invalid')

        planned = self._plan_from_observation(observation, measured_joints, control)
        correction = np.asarray(planned.transit.waypoints, dtype=float)
        if (
            correction.ndim != 2
            or len(correction) < 1
            or correction.shape[1:] != measured_joints.shape
            or not np.allclose(correction[0], measured_joints, atol=1e-7, rtol=0.0)
        ):
            raise PlanningError(
                'fresh pregrasp correction does not start at measured joints',
            )
        approach_path = _join_contiguous_joint_paths(
            correction,
            planned.approach_joints,
            label='fresh grasp approach',
        )
        lift_path = _join_contiguous_joint_paths(
            approach_path[-1:],
            planned.lift_joints,
            label='fresh grasp lift',
        )
        checkpoint(control, 'grasp completion trajectory retiming')
        approach = self._retime_joint_path(approach_path)
        lift = self._retime_joint_path(lift_path)
        return GraspCompletionProgram(
            observation_serial=int(observation.serial),
            candidate_index=int(planned.candidate_index),
            symmetry_index=int(planned.symmetry_index),
            grasp_pose=_immutable_pose(planned.grasp_pose, 'fresh grasp pose'),
            pregrasp_pose=_immutable_pose(
                planned.pregrasp_pose,
                'fresh pregrasp pose',
            ),
            required_width_m=(
                None
                if planned.required_width_m is None
                else float(planned.required_width_m)
            ),
            score=float(planned.score),
            failures=tuple(planned.failures),
            approach=_immutable_timed_trajectory(approach),
            lift=_immutable_timed_trajectory(lift),
        )

    def final_program(
        self,
        observation: PerceptionObservation,
        current_joints: np.ndarray,
        control: PlanningControl | None = None,
    ) -> MotionProgram:
        """Rebuild candidates and a plan from a post-servo observation."""
        planned = self._plan_from_observation(observation, current_joints, control)
        checkpoint(control, 'grasp trajectory retiming')
        transit_path = np.asarray(planned.transit.waypoints, dtype=float)
        approach_path = np.vstack((transit_path[-1], planned.approach_joints))
        lift_path = np.vstack((approach_path[-1], planned.lift_joints))
        return MotionProgram(
            planned=planned,
            transit=self._retime_joint_path(transit_path),
            approach=self._retime_joint_path(approach_path),
            lift=self._retime_joint_path(lift_path),
        )

    def validate_path(
        self,
        path: object,
        *,
        scene_points: object,
        target_points: object,
        stamp_s: float,
        segment_name: str,
        attachment_joints: object | None = None,
        required_width_m: object | None = None,
        control: PlanningControl | None = None,
    ) -> bool:
        """Revalidate a trajectory phase against the newest perception snapshot."""
        checkpoint(control, 'trajectory path revalidation')
        positions = np.asarray(path, dtype=float)
        scene = np.asarray(scene_points, dtype=float)
        target = np.asarray(target_points, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != self.chain.dof or len(positions) < 2:
            return False
        if scene.ndim != 2 or scene.shape[1] != 3 or target.ndim != 2 or target.shape[1] != 3:
            return False
        try:
            if segment_name in ('lift', 'carry', 'place_transit'):
                if attachment_joints is None:
                    return False
                collision_model = (
                    self._collision_model_for_grasp_width(required_width_m)
                    if segment_name == 'lift'
                    else self.collision_model
                )
                checker = self._new_checker(
                    scene_points=scene,
                    stamp_s=stamp_s,
                    collision_model=collision_model,
                )
                departure_direction = None
                if segment_name == 'lift':
                    departure_direction = _attached_target_departure_direction(
                        self.chain,
                        target,
                        np.asarray(attachment_joints, dtype=float),
                        positions[-1],
                    )
                checker.update_attached_target(
                    target,
                    attachment_joints=attachment_joints,
                    allowed_contact_capsules=self.collision_model.target_contact_capsules,
                    allow_initial_scene_contact=segment_name == 'lift',
                    departure_direction_base=departure_direction,
                )
                return all(
                    checker.is_segment_valid(first, second)
                    for first, second in zip(positions, positions[1:])
                )
            if segment_name == 'place_approach':
                if attachment_joints is None:
                    return False
                attachment = np.asarray(attachment_joints, dtype=float)
                blocked = self._new_checker(scene_points=scene, stamp_s=stamp_s)
                blocked.update_attached_target(
                    target,
                    attachment_joints=attachment,
                    allowed_contact_capsules=self.collision_model.target_contact_capsules,
                )
                segments = tuple(zip(positions, positions[1:]))
                if not all(
                    blocked.is_segment_valid(first, second, control=control)
                    for first, second in segments[:-1]
                ):
                    return False
                final = positions[-1]
                before_final = positions[-2]
                final_target = _attached_target_at_joints(
                    self.chain,
                    target,
                    attachment,
                    final,
                )
                prior_target = _attached_target_at_joints(
                    self.chain,
                    target,
                    attachment,
                    before_final,
                )
                departure = np.mean(prior_target, axis=0) - np.mean(
                    final_target,
                    axis=0,
                )
                departure_norm = float(np.linalg.norm(departure))
                if not np.isfinite(departure_norm) or departure_norm <= 1e-9:
                    return False
                support_contact = self._new_checker(
                    scene_points=scene,
                    stamp_s=stamp_s,
                )
                support_contact.update_attached_target(
                    final_target,
                    attachment_joints=final,
                    allowed_contact_capsules=self.collision_model.target_contact_capsules,
                    allow_initial_scene_contact=True,
                    departure_direction_base=departure / departure_norm,
                )
                return support_contact.is_segment_valid(
                    final,
                    before_final,
                    control=control,
                )
            if segment_name == 'place_retreat':
                checker = self._new_checker(scene_points=scene, stamp_s=stamp_s)
                checker.update_target(target)
                return all(
                    checker.is_segment_valid(first, second, control=control)
                    for first, second in zip(positions, positions[1:])
                )
            if segment_name == 'transit':
                checker = self._new_checker(scene_points=scene, stamp_s=stamp_s)
                checker.update_target(target)
                return all(
                    checker.is_segment_valid(first, second)
                    for first, second in zip(positions, positions[1:])
                )
            if segment_name == 'approach':
                no_contact = self._new_checker(scene_points=scene, stamp_s=stamp_s)
                no_contact.update_target(target)
                open_contact = self._new_checker(
                    scene_points=scene,
                    stamp_s=stamp_s,
                )
                open_contact.update_target(
                    target,
                    allowed_contact_capsules=(
                        self.collision_model.target_contact_capsules
                    ),
                )
                closed_contact = self._new_checker(
                    scene_points=scene,
                    stamp_s=stamp_s,
                    collision_model=self._collision_model_for_grasp_width(
                        required_width_m,
                    ),
                )
                closed_contact.update_target(
                    target,
                    allowed_contact_capsules=self.collision_model.target_contact_capsules,
                )
                approach = check_target_contact_approach(
                    positions,
                    no_contact=no_contact,
                    finger_contact=open_contact,
                    allowed_contact_capsules=(
                        self.collision_model.target_contact_capsules
                    ),
                    control=control,
                )
                return (
                    approach.valid
                    and closed_contact.check_state(positions[-1]).valid
                )
            return False
        except (TypeError, ValueError):
            return False

    def joint_motion(
        self,
        *,
        current_joints: object,
        goal_joints: object,
        scene_points: object,
        target_points: object,
        stamp_s: float,
        control: PlanningControl | None = None,
    ) -> TimedJointTrajectory:
        """Plan and retime a collision-checked posture while holding the target."""
        checkpoint(control, 'attached-object joint planning')
        current = np.asarray(current_joints, dtype=float)
        goal = np.asarray(goal_joints, dtype=float)
        scene = np.asarray(scene_points, dtype=float)
        target = np.asarray(target_points, dtype=float)
        checker = self._new_checker(scene_points=scene, stamp_s=stamp_s)
        checker.update_attached_target(
            target,
            attachment_joints=current,
            allowed_contact_capsules=self.collision_model.target_contact_capsules,
        )
        planner = JointSpaceRRTConnect(
            joint_names=self.chain.joint_names,
            lower_limits=self.chain.lower_limits,
            upper_limits=self.chain.upper_limits,
            state_valid=checker.is_state_valid,
            config=self.config.rrt,
        )
        path = planner.plan_joint(
            current,
            goal,
            timeout_s=self.config.grasp_plan.planning_timeout_s,
            control=control,
        )
        checkpoint(control, 'attached-object trajectory retiming')
        return retime_path(
            path.waypoints,
            self.chain.velocity_limits,
            self.config.robot.acceleration_limits,
            self.config.time_parameterization,
        )
