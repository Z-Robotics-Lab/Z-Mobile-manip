"""Planning-layer integration for geometry-triggered grasp contact."""

from types import SimpleNamespace

import numpy as np
import pytest

from z_manip.collision import CapsuleSpec, RobotCollisionModel
from z_manip.kinematics import KinematicChain
from z_manip_task.planning import OnlinePlanner


_URDF = """
<robot name='contact_slider'>
  <link name='base'/>
  <link name='tool'/>
  <joint name='approach' type='prismatic'>
    <parent link='base'/>
    <child link='tool'/>
    <axis xyz='1 0 0'/>
    <limit lower='-1' upper='1' effort='1' velocity='1'/>
  </joint>
</robot>
"""


def _planner(
    tmp_path,
    *,
    palm_offset=-0.08,
    finger_radius=0.025,
    scene_clearance=0.02,
    point_radius=0.005,
):
    urdf = tmp_path / 'contact_slider.urdf'
    urdf.write_text(_URDF)
    planner = OnlinePlanner.__new__(OnlinePlanner)
    planner.chain = KinematicChain.from_urdf(urdf, 'base', 'tool')
    planner.mesh_self_collision = None
    # This one-DOF contact fixture exercises point-cloud phase semantics only;
    # fixed Go2W attachments are covered by test_fixed_fixture_planning.py.
    planner._fixed_fixture_state_valid = lambda _joints: True
    planner._fixed_fixture_path_valid = lambda _path: True
    planner.collision_model = RobotCollisionModel(
        capsules=(
            CapsuleSpec(
                'palm',
                'tool',
                'tool',
                0.018,
                start_offset=(palm_offset, 0.0, 0.0),
                end_offset=(palm_offset, 0.0, 0.0),
            ),
            CapsuleSpec(
                'finger_left',
                'tool',
                'tool',
                finger_radius,
                start_offset=(0.0, 0.03, 0.0),
                end_offset=(0.0, 0.03, 0.0),
            ),
            CapsuleSpec(
                'finger_right',
                'tool',
                'tool',
                finger_radius,
                start_offset=(0.0, -0.03, 0.0),
                end_offset=(0.0, -0.03, 0.0),
            ),
        ),
        target_contact_capsules=('finger_left', 'finger_right'),
        scene_clearance_m=scene_clearance,
        point_radius_m=point_radius,
    )
    planner.config = SimpleNamespace(
        rrt=SimpleNamespace(collision_resolution=0.005),
        tool_geometry=SimpleNamespace(
            collision_open_aperture_m=0.06,
            collision_grasp_margin_m=0.004,
            tip_closing_axis=(0.0, 1.0, 0.0),
        ),
    )
    return planner


def _scene():
    return np.column_stack((
        np.linspace(1.5, 1.8, 32),
        np.full(32, 1.5),
        np.full(32, 1.5),
    ))


def _scene_with(*points):
    groups = [_scene()]
    groups.extend(
        np.asarray(points_group, dtype=float).reshape((-1, 3))
        for points_group in points
    )
    return np.vstack(groups)


def test_online_planner_rejects_acceleration_limits_with_wrong_dof(monkeypatch):
    chain = SimpleNamespace(dof=6)
    monkeypatch.setattr(
        KinematicChain,
        'from_urdf',
        lambda *_args, **_kwargs: chain,
    )
    config = SimpleNamespace(robot=SimpleNamespace(
        urdf_path='robot.urdf',
        base_link='arm_base',
        tip_link='tool',
        acceleration_limits=(1.0,) * 5,
    ))

    with pytest.raises(ValueError, match='acceleration limit count.*DOF'):
        OnlinePlanner(config)


def _x_plane(x, *, span=0.025, samples=9):
    values = np.linspace(-span, span, samples)
    yy, zz = np.meshgrid(values, values)
    return np.column_stack((
        np.full(yy.size, x),
        yy.ravel(),
        zz.ravel(),
    ))


def _y_plane(y, *, x_center=0.20, span=0.025, samples=15):
    values = np.linspace(-span, span, samples)
    xx, zz = np.meshgrid(x_center + values, values)
    return np.column_stack((
        xx.ravel(),
        np.full(xx.size, y),
        zz.ravel(),
    ))


def test_execution_revalidation_accepts_multi_waypoint_finger_contact(tmp_path):
    planner = _planner(tmp_path)

    assert planner.validate_path(
        np.array([[-0.14], [-0.10], [-0.06], [-0.02], [0.0]]),
        scene_points=_scene(),
        target_points=np.array([
            [0.0, 0.03, 0.0],
            [0.0, -0.03, 0.0],
        ]),
        stamp_s=10.0,
        segment_name='approach',
        required_width_m=0.04,
    )


def test_execution_revalidation_rejects_early_palm_target_contact(tmp_path):
    planner = _planner(tmp_path, palm_offset=0.08)

    assert not planner.validate_path(
        np.array([[-0.16], [-0.12], [-0.08], [-0.04], [0.0]]),
        scene_points=_scene(),
        target_points=np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.03, 0.0],
            [0.0, -0.03, 0.0],
        ]),
        stamp_s=10.0,
        segment_name='approach',
        required_width_m=0.04,
    )


def test_approach_checks_open_sweep_and_required_width_at_final_contact(tmp_path):
    planner = _planner(
        tmp_path,
        finger_radius=0.005,
        scene_clearance=0.0,
        point_radius=0.0,
    )
    required_width = 0.02
    closed_model = planner._collision_model_for_grasp_width(required_width)
    target = np.array([[0.0, 0.5, 0.0]])

    open_only_scene = _scene_with([[0.0, 0.03, 0.0]])
    opened = planner._new_checker(scene_points=open_only_scene, stamp_s=10.0)
    opened.update_target(target)
    closed = planner._new_checker(
        scene_points=open_only_scene,
        stamp_s=10.0,
        collision_model=closed_model,
    )
    closed.update_target(target)
    assert not opened.check_state(np.array([0.0])).valid
    assert closed.check_state(np.array([0.0])).valid
    assert not planner.validate_path(
        np.array([[0.0], [0.01]]),
        scene_points=open_only_scene,
        target_points=target,
        stamp_s=10.0,
        segment_name='approach',
        required_width_m=required_width,
    )

    closed_only_scene = _scene_with([[0.01, 0.012, 0.0]])
    opened = planner._new_checker(scene_points=closed_only_scene, stamp_s=10.0)
    opened.update_target(target)
    closed = planner._new_checker(
        scene_points=closed_only_scene,
        stamp_s=10.0,
        collision_model=closed_model,
    )
    closed.update_target(target)
    assert opened.check_state(np.array([0.01])).valid
    assert not closed.check_state(np.array([0.01])).valid
    assert not planner.validate_path(
        np.array([[0.0], [0.01]]),
        scene_points=closed_only_scene,
        target_points=target,
        stamp_s=10.0,
        segment_name='approach',
        required_width_m=required_width,
    )
    assert planner.validate_path(
        np.array([[0.0], [0.01]]),
        scene_points=open_only_scene,
        target_points=target,
        stamp_s=10.0,
        segment_name='lift',
        attachment_joints=np.array([0.0]),
        required_width_m=required_width,
    )
    assert not planner.validate_path(
        np.array([[0.0], [0.01]]),
        scene_points=closed_only_scene,
        target_points=target,
        stamp_s=10.0,
        segment_name='lift',
        attachment_joints=np.array([0.0]),
        required_width_m=required_width,
    )


def test_lift_revalidation_allows_only_directional_support_departure(tmp_path):
    planner = _planner(tmp_path)
    target = np.array([[0.20, 0.0, 0.0]])
    path = np.array([[0.0], [0.05], [0.10]])

    assert planner.validate_path(
        path,
        scene_points=_scene_with(_x_plane(0.18)),
        target_points=target,
        stamp_s=10.0,
        segment_name='lift',
        attachment_joints=np.array([0.0]),
        required_width_m=0.04,
    )
    assert not planner.validate_path(
        path,
        scene_points=_scene_with(
            _x_plane(0.18),
            _y_plane(0.02),
        ),
        target_points=target,
        stamp_s=10.0,
        segment_name='lift',
        attachment_joints=np.array([0.0]),
        required_width_m=0.04,
    )


def test_carry_and_place_noncontact_phases_remain_strict(tmp_path):
    planner = _planner(tmp_path)
    target = np.array([[0.20, 0.0, 0.0]])
    scene = _scene_with(_x_plane(0.18))
    path = np.array([[0.0], [0.05], [0.10]])

    for segment_name in ('carry', 'place_transit', 'place_approach'):
        assert not planner.validate_path(
            path,
            scene_points=scene,
            target_points=target,
            stamp_s=10.0,
            segment_name=segment_name,
            attachment_joints=np.array([0.0]),
        )


def test_place_approach_allows_only_final_support_facing_contact(tmp_path):
    planner = _planner(tmp_path)
    target = _x_plane(0.20)
    path = np.array([[0.0], [0.04], [0.10]])

    assert planner.validate_path(
        path,
        scene_points=_scene_with(_x_plane(0.305)),
        target_points=target,
        stamp_s=10.0,
        segment_name='place_approach',
        attachment_joints=np.array([0.0]),
    )


def test_place_approach_rejects_final_segment_side_collision(tmp_path):
    planner = _planner(tmp_path)
    target = _x_plane(0.20)
    path = np.array([[0.0], [0.04], [0.10]])

    assert not planner.validate_path(
        path,
        scene_points=_scene_with(
            _x_plane(0.305),
            _y_plane(0.005, x_center=0.30, span=0.01),
        ),
        target_points=target,
        stamp_s=10.0,
        segment_name='place_approach',
        attachment_joints=np.array([0.0]),
    )
