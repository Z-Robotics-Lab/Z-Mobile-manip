"""Payload-aware placement collision tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.collision import (
    CapsuleSpec,
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
)
from z_manip.kinematics import KinematicChain
from z_manip.models.planner import PlanningError
from z_manip.planning import PlacementCandidate
from z_manip_place.attached_collision import (
    AttachedCollisionAuditConfig,
    AttachedObjectPathAuditor,
)
from z_manip_place.core import RawTrajectorySegment


_SLIDER_URDF = """
<robot name="payload_slider">
  <link name="base"/>
  <link name="tool"/>
  <joint name="slide" type="prismatic">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _geometry(tmp_path):
    urdf = tmp_path / 'payload_slider.urdf'
    urdf.write_text(_SLIDER_URDF, encoding='utf-8')
    chain = KinematicChain.from_urdf(urdf, 'base', 'tool')
    model = RobotCollisionModel(
        capsules=(CapsuleSpec(
            'wrist',
            'tool',
            'tool',
            0.015,
            start_offset=(0.0, 0.0, -0.02),
            end_offset=(0.0, 0.0, 0.02),
        ),),
        target_contact_capsules=('wrist',),
    )
    return chain, model


def _reference():
    return np.asarray([
        (x, y, z)
        for x in (-0.03, 0.03)
        for y in (-0.03, 0.03)
        for z in (-0.03, 0.03)
    ])


def _scene():
    values = np.linspace(-0.035, 0.035, 7)
    yy, zz = np.meshgrid(0.20 + values, values)
    shelf_side = np.column_stack((
        np.full(yy.size, 0.25),
        yy.ravel(),
        zz.ravel(),
    ))
    far = np.column_stack((
        np.linspace(-0.8, 0.8, 40),
        np.full(40, 0.8),
        np.full(40, 0.8),
    ))
    return np.vstack((shelf_side, far))


def _auditor(tmp_path):
    chain, model = _geometry(tmp_path)
    auditor = AttachedObjectPathAuditor(
        chain=chain,
        collision_model=model,
        config=AttachedCollisionAuditConfig(
            clearance_m=0.008,
            point_radius_m=0.002,
            segment_joint_step_rad=0.02,
            min_scene_points=16,
            max_attached_points=256,
            extent_samples_per_axis=4,
            carried_object_scene_exclusion_m=0.005,
        ),
    )
    tool_from_object = np.eye(4)
    tool_from_object[1, 3] = 0.20
    auditor.bind_snapshot(
        scene_points=_scene(),
        scene_stamp_s=10.0,
        planning_from_kinematic_base=np.eye(4),
        attachment_joints=np.array((0.0,)),
        object_reference_points_object=_reference(),
        object_extent_m=np.array((0.06, 0.06, 0.06)),
        tool_from_object=tool_from_object,
    )
    return chain, model, auditor


def _segments(direction):
    sign = float(direction)
    return (
        RawTrajectorySegment(
            'transit',
            ('slide',),
            np.array(((0.0,), (sign * 0.50,))),
            np.array((0.0, 0.5)),
        ),
        RawTrajectorySegment(
            'approach',
            ('slide',),
            np.array(((sign * 0.50,), (sign * 0.60,))),
            np.array((0.0, 0.2)),
        ),
        RawTrajectorySegment(
            'retreat',
            ('slide',),
            np.array(((sign * 0.60,), (sign * 0.40,))),
            np.array((0.0, 0.3)),
        ),
    )


def test_payload_rejects_shelf_collision_that_bare_arm_misses(tmp_path):
    chain, model, auditor = _auditor(tmp_path)
    bare = PointCloudCollisionChecker(
        chain=chain,
        model=model,
        frame_provider=chain.link_transforms,
        config=PointCloudCollisionConfig(
            clearance=0.008,
            point_radius=0.002,
            min_scene_points=16,
            segment_joint_step=0.02,
        ),
        now_fn=lambda: 10.0,
    )
    bare.update_scene(_scene(), stamp_s=10.0)
    assert bare.is_segment_valid(np.array((0.0,)), np.array((0.50,)))

    with pytest.raises(PlanningError, match='attached_target'):
        auditor.audit(segments=_segments(1.0))

    auditor.audit(segments=_segments(-1.0))


def _candidate(direction):
    sign = float(direction)

    def pose(x):
        result = np.eye(4)
        result[0, 3] = sign * x
        return result

    return PlacementCandidate(
        support_position=np.array((sign * 0.6, 0.0, 0.0)),
        surface_normal=np.array((1.0, 0.0, 0.0)),
        yaw_rad=0.0,
        object_pose=pose(0.6),
        preplace_pose=pose(0.5),
        place_pose=pose(0.6),
        retreat_pose=pose(0.4),
        support_fraction=1.0,
        obstacle_clearance_m=0.1,
        geometric_score=1.0,
    )


def test_only_payload_safe_candidate_reaches_moveit_ranking(tmp_path):
    pytest.importorskip('geometry_msgs.msg')
    pytest.importorskip('moveit_msgs.msg')
    from z_manip_place.moveit_evaluator import MoveItPlacementEvaluator

    _chain, _model, auditor = _auditor(tmp_path)
    evaluator = MoveItPlacementEvaluator.__new__(MoveItPlacementEvaluator)
    evaluator.goal_id = 'place-payload-audit'
    evaluator.attached_collision_auditor = auditor
    evaluator.config = SimpleNamespace(
        joint_names=('slide',),
        planning_frame='base',
        continuity_tolerance_rad=0.01,
    )

    def transit(_current, goal, _control):
        direction = 1.0 if goal[0, 3] > 0.0 else -1.0
        return _segments(direction)[0]

    def cartesian(phase, _current, goal, _control):
        direction = 1.0 if goal[0, 3] > 0.0 else -1.0
        return _segments(direction)[1 if phase == 'approach' else 2]

    evaluator._transit = transit
    evaluator._cartesian = cartesian

    with pytest.raises(PlanningError, match='attached_target'):
        evaluator.evaluate(_candidate(1.0), np.array((0.0,)))
    evaluation = evaluator.evaluate(_candidate(-1.0), np.array((0.0,)))

    assert np.isfinite(evaluation.score)
    assert evaluation.trajectory.goal_id == 'place-payload-audit'
    assert evaluation.trajectory.points[-1].phase == 'retreat'


@pytest.mark.parametrize(
    'changes,match',
    (
        ({'segment_joint_step_rad': 0.0}, 'joint step'),
        ({'extent_samples_per_axis': 1}, 'at least two'),
        ({'carried_object_scene_exclusion_m': -0.1}, 'non-negative'),
    ),
)
def test_attached_collision_configuration_fails_closed(changes, match):
    with pytest.raises(ValueError, match=match):
        AttachedCollisionAuditConfig(**changes)
