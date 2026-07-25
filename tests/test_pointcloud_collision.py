from __future__ import annotations

import numpy as np
import pytest

from z_manip.collision import (
    CapsuleSpec,
    CollisionResult,
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
    SelfCollisionConfig,
)
from z_manip.kinematics import KinematicChain
from z_manip.planning_control import (
    PlanningCancelled,
    PlanningControl,
    PlanningDeadlineExceeded,
)


_URDF = """
<robot name="slider">
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


def _chain_and_frames(tmp_path):
    urdf = tmp_path / "slider.urdf"
    urdf.write_text(_URDF)
    chain = KinematicChain.from_urdf(urdf, "base", "tool")

    def frames(joints):
        return {"base": np.eye(4), "tool": chain.forward(joints)}

    return chain, frames


def _checker(
    tmp_path,
    *,
    now=None,
    model=None,
    min_points=1,
    max_age=0.5,
    scene_noise_tolerance=0.0,
    scene_noise_min_support_points=1,
    self_collision_checker=None,
):
    chain, frames = _chain_and_frames(tmp_path)
    if model is None:
        model = RobotCollisionModel(
            capsules=(
                CapsuleSpec(
                    "wrist",
                    "tool",
                    "tool",
                    0.04,
                    start_offset=(0.0, 0.0, -0.06),
                    end_offset=(0.0, 0.0, 0.06),
                ),
            ),
        )
    clock = (lambda: 10.0) if now is None else now
    return PointCloudCollisionChecker(
        chain=chain,
        model=model,
        frame_provider=frames,
        config=PointCloudCollisionConfig(
            clearance=0.01,
            point_radius=0.0,
            scene_noise_tolerance=scene_noise_tolerance,
            scene_noise_min_support_points=scene_noise_min_support_points,
            min_scene_points=min_points,
            max_scene_age_s=max_age,
            segment_joint_step=0.02,
        ),
        now_fn=clock,
        self_collision_checker=self_collision_checker,
    )


def _x_plane(x, *, span=0.015, samples=7):
    lateral = np.linspace(-span, span, samples)
    yy, zz = np.meshgrid(lateral, lateral)
    return np.column_stack((
        np.full(yy.size, x),
        yy.ravel(),
        zz.ravel(),
    ))


def _y_plane(y, *, x_center=0.20, span=0.015, samples=13):
    values = np.linspace(-span, span, samples)
    xx, zz = np.meshgrid(x_center + values, values)
    return np.column_stack((
        xx.ravel(),
        np.full(xx.size, y),
        zz.ravel(),
    ))


def test_scene_capsule_collision_and_target_mask_exclusion(tmp_path):
    checker = _checker(tmp_path)
    cloud = np.array([[0.0, 0.045, 0.0], [0.8, 0.8, 0.8]])

    checker.update_scene(cloud, stamp_s=10.0)
    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.kind == "scene"
    assert collision.capsules == ("wrist",)
    assert collision.distance == pytest.approx(0.045)

    kept = checker.update_scene(
        cloud,
        stamp_s=10.0,
        target_mask=np.array([True, False]),
    )
    assert kept == 1
    assert checker.is_state_valid(np.array([0.0]))
    assert checker.is_state_valid(np.array([-0.5]))


def test_scene_capsule_ignores_d435_boundary_band_but_blocks_supported_penetration(tmp_path):
    checker = _checker(
        tmp_path,
        scene_noise_tolerance=0.003,
        scene_noise_min_support_points=2,
    )
    # Capsule radius 40 mm + clearance 10 mm = 50 mm.  A dense-looking
    # boundary mini-cluster 0.4 mm inside that envelope is still sensor noise.
    boundary = np.array([
        [0.0, 0.0496, -0.01],
        [0.0, 0.0497, 0.00],
        [0.0, 0.0498, 0.01],
    ])
    checker.update_scene(boundary, stamp_s=10.0)
    assert checker.check_state(np.array([0.0])).valid

    penetrating = np.array([
        [0.0, 0.0450, -0.01],
        [0.0, 0.0460, 0.01],
        [0.0, 0.0498, 0.00],
    ])
    checker.update_scene(penetrating, stamp_s=10.0)
    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.distance == pytest.approx(0.045)
    assert collision.threshold == pytest.approx(0.05)


def test_joint_segment_detects_thin_obstacle_between_valid_endpoints(tmp_path):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.0, 0.0, 0.0]]), stamp_s=10.0)
    start, end = np.array([-0.5]), np.array([0.5])

    assert checker.is_state_valid(start)
    assert checker.is_state_valid(end)
    result = checker.check_segment(start, end)

    assert not result.valid
    assert result.state_result is not None
    assert result.state_result.kind == "scene"
    assert 0.40 <= result.alpha <= 0.50


def test_joint_segment_cancellation_stops_after_first_interpolation_sample(tmp_path):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=10.0)
    original = checker.check_state
    work_units = []

    def counted_check_state(joints):
        work_units.append(np.asarray(joints).copy())
        return original(joints)

    checker.check_state = counted_check_state
    control = PlanningControl(cancel_check=lambda: len(work_units) >= 1)

    with pytest.raises(PlanningCancelled, match="collision interpolation"):
        checker.check_segment(
            np.array([-0.5]),
            np.array([0.5]),
            max_joint_step=0.04,
            control=control,
        )

    assert len(work_units) == 1


def test_joint_segment_deadline_stops_between_interpolation_samples(tmp_path):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=10.0)
    original = checker.check_state
    work_units = []
    clock_values = iter((0.0, 0.0, 1.0))

    def counted_check_state(joints):
        work_units.append(np.asarray(joints).copy())
        return original(joints)

    checker.check_state = counted_check_state
    control = PlanningControl(
        deadline_s=0.5,
        monotonic_fn=lambda: next(clock_values),
    )

    with pytest.raises(PlanningDeadlineExceeded, match="collision interpolation"):
        checker.check_segment(
            np.array([-0.5]),
            np.array([0.5]),
            max_joint_step=0.04,
            control=control,
        )

    assert len(work_units) == 1


def test_stale_or_insufficient_scene_fails_closed(tmp_path):
    now = [20.0]
    checker = _checker(tmp_path, now=lambda: now[0], min_points=2, max_age=0.25)

    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=20.0)
    insufficient = checker.check_state(np.array([0.0]))
    assert not insufficient.valid
    assert insufficient.kind == "perception"
    assert "requires 2" in insufficient.reason

    checker.update_scene(
        np.array([[0.8, 0.8, 0.8], [0.9, 0.8, 0.8]]),
        stamp_s=20.0,
    )
    assert checker.is_state_valid(np.array([0.0]))
    now[0] = 20.3
    stale = checker.check_state(np.array([0.0]))
    assert not stale.valid
    assert stale.kind == "perception"
    assert "stale" in stale.reason


def test_self_collision_pairs_and_ignore_list_are_configurable(tmp_path):
    crossing = (
        CapsuleSpec(
            "vertical",
            "tool",
            "tool",
            0.02,
            start_offset=(0.0, 0.0, -0.1),
            end_offset=(0.0, 0.0, 0.1),
        ),
        CapsuleSpec(
            "lateral",
            "tool",
            "tool",
            0.02,
            start_offset=(0.0, -0.1, 0.0),
            end_offset=(0.0, 0.1, 0.0),
        ),
    )
    enabled = RobotCollisionModel(
        crossing,
        SelfCollisionConfig(pairs=(("vertical", "lateral"),)),
    )
    checker = _checker(tmp_path, model=enabled)
    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=10.0)
    result = checker.check_state(np.array([0.0]))
    assert not result.valid
    assert result.kind == "self"

    ignored = RobotCollisionModel(
        crossing,
        SelfCollisionConfig(
            pairs=None,
            ignore_pairs=(("lateral", "vertical"),),
        ),
    )
    checker = _checker(tmp_path, model=ignored)
    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=10.0)
    assert checker.is_state_valid(np.array([0.0]))


def test_fixed_platform_capsule_skips_scene_and_target_but_keeps_self_collision(tmp_path):
    model = RobotCollisionModel(
        capsules=(
            CapsuleSpec(
                "platform_sensor",
                "base",
                "base",
                0.05,
                check_scene=False,
                check_target=False,
                supplemental_self_collision=True,
            ),
            CapsuleSpec("moving_arm", "tool", "tool", 0.03),
        ),
        self_collision=SelfCollisionConfig(
            pairs=(("platform_sensor", "moving_arm"),),
        ),
    )
    checker = _checker(tmp_path, model=model)
    checker.update_scene(np.array([[0.0, 0.0, 0.0]]), stamp_s=10.0)
    checker.update_target(np.array([[0.0, 0.0, 0.0]]))

    assert checker.is_state_valid(np.array([0.5]))
    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.kind == "self"
    assert set(collision.capsules) == {"platform_sensor", "moving_arm"}


def test_capsule_self_collision_supplements_valid_mesh_backend(tmp_path):
    crossing = RobotCollisionModel(
        capsules=(
            CapsuleSpec(
                "vertical",
                "tool",
                "tool",
                0.02,
                start_offset=(0.0, 0.0, -0.1),
                end_offset=(0.0, 0.0, 0.1),
                supplemental_self_collision=True,
            ),
            CapsuleSpec(
                "lateral",
                "tool",
                "tool",
                0.02,
                start_offset=(0.0, -0.1, 0.0),
                end_offset=(0.0, 0.1, 0.0),
            ),
        ),
        self_collision=SelfCollisionConfig(
            pairs=(("vertical", "lateral"),),
        ),
    )
    mesh_calls = []

    def valid_mesh_backend(joints):
        mesh_calls.append(np.asarray(joints).copy())
        return CollisionResult(True, "mesh collision-free")

    checker = _checker(
        tmp_path,
        model=crossing,
        self_collision_checker=valid_mesh_backend,
    )
    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=10.0)

    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.kind == "self"
    assert mesh_calls == []


def test_bad_geometry_or_perception_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="radius"):
        CapsuleSpec("bad", "base", "tool", 0.0)

    duplicate = CapsuleSpec("same", "base", "tool", 0.02)
    with pytest.raises(ValueError, match="unique"):
        RobotCollisionModel((duplicate, duplicate))

    unknown_frame = RobotCollisionModel(
        (CapsuleSpec("unknown", "base", "not_in_urdf", 0.02),),
    )
    with pytest.raises(ValueError, match="outside the kinematic chain"):
        _checker(tmp_path, model=unknown_frame)

    checker = _checker(tmp_path)
    with pytest.raises(ValueError, match="target_mask"):
        checker.update_scene(
            np.zeros((2, 3)),
            stamp_s=10.0,
            target_mask=np.array([True]),
        )
    assert not checker.is_state_valid(np.array([0.0]))


def test_target_contact_is_allowed_only_for_named_gripper_capsule(tmp_path):
    model = RobotCollisionModel(capsules=(
        CapsuleSpec(
            "gripper", "tool", "tool", 0.04,
            start_offset=(0.0, 0.0, -0.04),
            end_offset=(0.0, 0.0, 0.04),
        ),
        CapsuleSpec(
            "wrist", "tool", "tool", 0.04,
            start_offset=(0.0, 0.20, -0.04),
            end_offset=(0.0, 0.20, 0.04),
        ),
    ))
    checker = _checker(tmp_path, model=model)
    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=10.0)

    checker.update_target(np.array([[0.0, 0.0, 0.0]]))
    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.kind == "target"
    assert collision.capsules == ("gripper",)

    checker.update_target(
        np.array([[0.0, 0.0, 0.0]]),
        allowed_contact_capsules=("gripper",),
    )
    assert checker.is_state_valid(np.array([0.0]))

    checker.update_target(
        np.array([[0.0, 0.20, 0.0]]),
        allowed_contact_capsules=("gripper",),
    )
    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.capsules == ("wrist",)


def test_attached_target_moves_with_tip_and_blocks_payload_scene_collision(tmp_path):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.70, 0.0, 0.0]]), stamp_s=10.0)
    checker.update_attached_target(
        np.array([[0.20, 0.0, 0.0]]),
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
    )

    assert checker.is_state_valid(np.array([0.0]))
    collision = checker.check_state(np.array([0.5]))
    assert not collision.valid
    assert collision.kind == "attached_target"
    assert collision.distance == pytest.approx(0.0)


def test_attached_target_rejects_unknown_contact_capsule(tmp_path):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.8, 0.8, 0.8]]), stamp_s=10.0)
    with pytest.raises(ValueError, match="unknown capsules"):
        checker.update_attached_target(
            np.array([[0.2, 0.0, 0.0]]),
            attachment_joints=np.array([0.0]),
            allowed_contact_capsules=("not-a-link",),
        )


def test_attached_scene_contact_requires_explicit_contact_band(tmp_path):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.20, 0.0, 0.0]]), stamp_s=10.0)
    checker.update_attached_target(
        np.array([[0.20, 0.0, 0.0]]),
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
        allow_scene_contact=True,
    )
    assert checker.is_state_valid(np.array([0.0]))


def test_attached_target_departure_is_pure_and_reverse_segment_is_rejected(tmp_path):
    scene = np.vstack((_x_plane(0.195), [[0.70, 0.0, 0.0]]))
    target = _x_plane(0.20)
    strict = _checker(tmp_path)
    strict.update_scene(scene, stamp_s=10.0)
    strict.update_attached_target(
        target,
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
    )
    assert strict.check_state(np.array([0.0])).kind == "attached_target"

    departure = _checker(tmp_path)
    departure.update_scene(scene, stamp_s=10.0)
    departure.update_attached_target(
        target,
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
        allow_initial_scene_contact=True,
        departure_direction_base=np.array([1.0, 0.0, 0.0]),
    )

    assert departure.is_state_valid(np.array([0.0]))
    assert departure.is_state_valid(np.array([0.05]))
    assert departure.is_state_valid(np.array([0.0]))
    assert departure.check_segment(np.array([0.0]), np.array([0.05])).valid
    returned = departure.check_segment(np.array([0.05]), np.array([0.0]))
    assert not returned.valid
    assert returned.state_result is not None
    assert returned.state_result.kind == "attached_target"
    assert "reverses" in returned.reason
    collision = departure.check_state(np.array([0.50]))
    assert not collision.valid
    assert collision.kind == "attached_target"


def test_support_manifold_tolerates_sparse_curved_bottom_sampling(tmp_path):
    checker = _checker(tmp_path)
    support = _x_plane(0.2005, span=0.015, samples=6)
    checker.update_scene(
        np.vstack((support, [[0.70, 0.0, 0.0]])),
        stamp_s=10.0,
    )
    checker.update_attached_target(
        np.array([[0.20, 0.0, 0.0]]),
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
        allow_initial_scene_contact=True,
        departure_direction_base=np.array([1.0, 0.0, 0.0]),
    )

    assert checker.is_state_valid(np.array([0.0]))


def test_support_manifold_accepts_directional_shelf_edge_contact(tmp_path):
    checker = _checker(tmp_path)
    xx, zz = np.meshgrid(
        np.linspace(0.194, 0.195, 9),
        np.linspace(-0.015, 0.015, 13),
    )
    edge = np.column_stack((
        xx.ravel(),
        np.full(xx.size, 0.001),
        zz.ravel(),
    ))
    target = np.column_stack((
        np.full(13, 0.20),
        np.zeros(13),
        np.linspace(-0.015, 0.015, 13),
    ))
    checker.update_scene(
        np.vstack((_x_plane(0.195, samples=13), edge, [[0.70, 0.0, 0.0]])),
        stamp_s=10.0,
    )
    checker.update_attached_target(
        target,
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
        allow_initial_scene_contact=True,
        departure_direction_base=np.array([1.0, 0.0, 0.0]),
    )

    assert checker.is_state_valid(np.array([0.0]))


def test_support_manifold_keeps_locally_valid_seeds_across_depth_bands(tmp_path):
    checker = _checker(tmp_path)
    target = _x_plane(0.20, span=0.015, samples=9)
    support = np.vstack((
        _x_plane(0.195, span=0.015, samples=9),
        _x_plane(0.180, span=0.015, samples=9),
    ))
    checker.update_scene(
        np.vstack((support, [[0.70, 0.0, 0.0]])),
        stamp_s=10.0,
    )
    checker.update_attached_target(
        target,
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
        allow_initial_scene_contact=True,
        departure_direction_base=np.array([1.0, 0.0, 0.0]),
    )

    assert checker.is_state_valid(np.array([0.0]))


@pytest.mark.parametrize(
    "obstacle_kind",
    ("lateral", "departure-facing"),
    ids=("lateral", "departure-facing"),
)
def test_attached_target_departure_exempts_only_support_side(tmp_path, obstacle_kind):
    checker = _checker(tmp_path)
    obstacle = (
        _y_plane(0.005)
        if obstacle_kind == "lateral"
        else _x_plane(0.205)
    )
    checker.update_scene(
        np.vstack((_x_plane(0.195), obstacle, [[0.70, 0.0, 0.0]])),
        stamp_s=10.0,
    )
    checker.update_attached_target(
        _x_plane(0.20),
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
        allow_initial_scene_contact=True,
        departure_direction_base=np.array([1.0, 0.0, 0.0]),
    )

    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.kind == "attached_target"


def test_departure_exception_keeps_robot_links_against_full_scene(tmp_path):
    model = RobotCollisionModel(capsules=(
        CapsuleSpec(
            "wrist",
            "tool",
            "tool",
            0.04,
            start_offset=(0.20, 0.0, -0.02),
            end_offset=(0.20, 0.0, 0.02),
        ),
    ))
    checker = _checker(tmp_path, model=model)
    checker.update_scene(
        np.vstack((_x_plane(0.195), [[0.70, 0.0, 0.0]])),
        stamp_s=10.0,
    )
    checker.update_attached_target(
        _x_plane(0.20),
        attachment_joints=np.array([0.0]),
        allowed_contact_capsules=("wrist",),
        allow_initial_scene_contact=True,
        departure_direction_base=np.array([1.0, 0.0, 0.0]),
    )

    collision = checker.check_state(np.array([0.0]))
    assert not collision.valid
    assert collision.kind == "scene"
    assert collision.capsules == ("wrist",)


@pytest.mark.parametrize(
    "direction",
    (None, (0.0, 0.0, 0.0), (1.0, 2.0), (np.nan, 0.0, 1.0)),
)
def test_initial_contact_exception_requires_departure_direction(tmp_path, direction):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.195, 0.0, 0.0]]), stamp_s=10.0)

    with pytest.raises(ValueError, match="departure_direction_base"):
        checker.update_attached_target(
            np.array([[0.20, 0.0, 0.0]]),
            attachment_joints=np.array([0.0]),
            allowed_contact_capsules=("wrist",),
            allow_initial_scene_contact=True,
            departure_direction_base=direction,
        )


def test_attached_target_contact_modes_are_mutually_exclusive(tmp_path):
    checker = _checker(tmp_path)
    checker.update_scene(np.array([[0.20, 0.0, 0.0]]), stamp_s=10.0)

    with pytest.raises(ValueError, match="exclusive"):
        checker.update_attached_target(
            np.array([[0.20, 0.0, 0.0]]),
            attachment_joints=np.array([0.0]),
            allowed_contact_capsules=("wrist",),
            allow_scene_contact=True,
            allow_initial_scene_contact=True,
        )


def _finger_checker(tmp_path, *, exclusion, band=0.006, radius=0.18):
    chain, frames = _chain_and_frames(tmp_path)
    model = RobotCollisionModel(
        capsules=(
            CapsuleSpec(
                "finger_test",
                "tool",
                "tool",
                0.02,
                start_offset=(0.0, 0.0, 0.0),
                end_offset=(0.0, 0.0, 0.001),
            ),
        ),
        finger_support_plane_exclusion=exclusion,
        finger_support_plane_band_m=band,
        finger_support_plane_radius_m=radius,
    )
    return PointCloudCollisionChecker(
        chain=chain,
        model=model,
        frame_provider=frames,
        config=PointCloudCollisionConfig(
            clearance=0.01,
            point_radius=0.0,
            min_scene_points=1,
            max_scene_age_s=0.5,
            segment_joint_step=0.02,
        ),
        now_fn=lambda: 10.0,
    )


def test_finger_support_plane_exclusion_clears_in_plane_graze(tmp_path):
    # Object standing on a support plane at z=0.0, offset laterally so it does
    # not itself contact the finger.  A single floor sample sits right under the
    # finger at the fitted support height.
    target = np.array([[0.20, 0.10, z] for z in np.linspace(0.0, 0.05, 6)])
    floor_graze = np.array([[0.20, 0.0, 0.0]])
    joints = np.array([0.20])

    off = _finger_checker(tmp_path, exclusion=False)
    off.update_scene(floor_graze, stamp_s=10.0)
    off.update_target(target)
    assert off.check_state(joints).kind == "scene"  # graze vetoes the finger

    on = _finger_checker(tmp_path, exclusion=True)
    on.update_scene(floor_graze, stamp_s=10.0)
    on.update_target(target)
    assert on.check_state(joints).valid  # in-plane sample excluded for the finger


def test_finger_support_plane_exclusion_keeps_off_plane_obstacle(tmp_path):
    # Same geometry but the object (hence the fitted support plane) is high up,
    # so the sample under the finger is far below the plane and must stay active.
    target = np.array([[0.20, 0.10, z] for z in np.linspace(0.30, 0.35, 6)])
    obstacle = np.array([[0.20, 0.0, 0.0]])
    joints = np.array([0.20])

    on = _finger_checker(tmp_path, exclusion=True)
    on.update_scene(obstacle, stamp_s=10.0)
    on.update_target(target)
    assert on.check_state(joints).kind == "scene"  # off-plane obstacle preserved


def test_attaching_the_target_invalidates_a_cached_finger_scene_exclusion(tmp_path):
    # update_scene() and update_target() both invalidate the finger-exclusion
    # cache (_clear_finger_scene_exclusion); update_attached_target() did not,
    # leaving a plane fit computed for the pre-pickup resting target pose
    # cached after the target detaches from the scene and attaches to the
    # chain. check_state() re-derives its own "use the exclusion" gate from
    # `self._target_points is not None` on every call, and update_attached_
    # target() already sets that to None, so this is not observable through
    # check_state() today (the audit's own "currently unreachable" framing) —
    # assert the actual invariant directly: every "install new target
    # geometry" entry point must leave no stale finger-exclusion fit cached.
    target = np.array([[0.20, 0.10, z] for z in np.linspace(0.0, 0.05, 6)])
    floor_graze = np.array([[0.20, 0.0, 0.0]])
    joints = np.array([0.20])

    checker = _finger_checker(tmp_path, exclusion=True)
    checker.update_scene(floor_graze, stamp_s=10.0)
    checker.update_target(target)
    checker.check_state(joints)
    assert checker._finger_scene_ready  # caches a plane fit against `target`

    checker.update_attached_target(target, attachment_joints=joints)
    assert not checker._finger_scene_ready
    assert checker._finger_scene_tree is None
    assert checker._finger_scene_points is None


# -- link-frame validation ---------------------------------------------------


def _rigid(rotation=None, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def test_state_checks_validate_link_frames_without_numpy_isclose(
    tmp_path,
    monkeypatch,
):
    """Regression test: `_valid_transform` runs once per link frame per state
    check, i.e. it is on the per-RRT-sample and per-waypoint hot path.  It must
    evaluate the closeness predicate directly instead of calling
    `np.allclose`/`np.isclose`, whose errstate context managers and dtype
    dispatch dominated the cost of every collision query.

    Counting the calls is the only way to see this: the accepted set of
    transforms is unchanged, which the boundary test below pins separately.
    """

    checker = _checker(tmp_path)
    checker.update_scene(np.array([[5.0, 0.0, 0.0]]), stamp_s=10.0)
    real_allclose = np.allclose
    real_isclose = np.isclose
    calls: list[str] = []

    def counting_allclose(*args, **kwargs):
        calls.append("allclose")
        return real_allclose(*args, **kwargs)

    def counting_isclose(*args, **kwargs):
        calls.append("isclose")
        return real_isclose(*args, **kwargs)

    monkeypatch.setattr(np, "allclose", counting_allclose)
    monkeypatch.setattr(np, "isclose", counting_isclose)
    for value in np.linspace(-0.5, 0.5, 12):
        assert checker.check_state(np.array([value])).valid

    assert calls == []


def test_link_frame_validation_keeps_its_exact_rigid_transform_tolerances():
    """Pin the accept/reject boundary of every rigid-transform criterion.

    Nothing exercised the reject path before, so an over-loosened predicate
    would have gone unnoticed while silently admitting a skewed or mirrored
    frame from a misbehaving kinematics service.  The tolerances below are
    numpy's `allclose`/`isclose` defaults for the original calls: 1e-7 on the
    bottom row (1.01e-5 on its unit entry), 1e-5 off-diagonal and 2e-5 on the
    diagonal of R^T R, and 2e-5 on det(R).
    """

    valid = PointCloudCollisionChecker._valid_transform
    assert valid(_rigid(translation=(0.3, -0.2, 0.1)))

    # Bottom row: 1e-7 on the zeros, 1e-7 + 1e-5 on the trailing one.
    inside = _rigid()
    inside[3, 0] = 9e-8
    assert valid(inside)
    outside = _rigid()
    outside[3, 0] = 1.2e-7
    assert not valid(outside)
    inside = _rigid()
    inside[3, 3] = 1.0 + 9e-6
    assert valid(inside)
    outside = _rigid()
    outside[3, 3] = 1.0 + 1.2e-5
    assert not valid(outside)

    # Off-diagonal orthonormality: a pure shear leaves det(R) at exactly 1, so
    # only the 1e-5 off-diagonal bound on R^T R decides.
    shear = np.eye(3)
    shear[0, 1] = 8e-6
    assert valid(_rigid(shear))
    shear = np.eye(3)
    shear[0, 1] = 1.3e-5
    assert not valid(_rigid(shear))

    # Uniform scale by (1 + e) moves R^T R by 2e and det(R) by 3e, so e between
    # 2e-5/3 and 1e-5 is rejected by the determinant bound alone.
    assert valid(_rigid(np.eye(3) * (1.0 + 6e-6)))
    assert not valid(_rigid(np.eye(3) * (1.0 + 8e-6)))

    # Mirrored frames are orthonormal but left-handed.
    reflection = np.diag((-1.0, 1.0, 1.0))
    assert not valid(_rigid(reflection))

    # Malformed inputs stay rejected rather than raising.
    assert not valid(_rigid(np.full((3, 3), np.nan)))
    assert not valid(np.eye(3))
    assert not valid("not a transform")
    assert not valid(None)


def test_a_skewed_frame_provider_transform_fails_closed_with_a_named_frame(
    tmp_path,
):
    """End-to-end reject path: a kinematics service that returns a
    non-orthonormal link transform must invalidate the state, not be trusted.
    """

    chain, frames = _chain_and_frames(tmp_path)

    def skewed(joints):
        transforms = dict(frames(joints))
        matrix = np.array(transforms["tool"], dtype=float)
        matrix[:3, :3] = matrix[:3, :3] * 1.05
        transforms["tool"] = matrix
        return transforms

    checker = _checker(tmp_path)
    checker.update_scene(np.array([[5.0, 0.0, 0.0]]), stamp_s=10.0)
    assert checker.check_state(np.array([0.0])).valid

    checker.frame_provider = skewed
    result = checker.check_state(np.array([0.0]))
    assert not result.valid
    assert result.kind == "kinematics"
    assert "invalid transform for 'tool'" in result.reason
    assert chain.dof == 1


# -- forward kinematics on the state-check hot path ---------------------------


def _fk_attributable_checker(tmp_path):
    """A checker whose frame provider uses `link_transforms`, so every
    `chain.forward` call can be attributed to `check_state` itself."""

    chain, _frames = _chain_and_frames(tmp_path)
    model = RobotCollisionModel(
        capsules=(
            CapsuleSpec(
                "wrist", "tool", "tool", 0.02,
                start_offset=(0.0, 0.0, -0.06), end_offset=(0.0, 0.0, 0.06),
            ),
            CapsuleSpec(
                "palm", "tool", "tool", 0.02,
                start_offset=(0.0, 0.0, 0.06), end_offset=(0.0, 0.0, 0.10),
            ),
            CapsuleSpec(
                "forearm", "base", "tool", 0.02,
                start_offset=(0.0, 0.0, -0.20), end_offset=(0.0, 0.0, -0.10),
            ),
        ),
    )
    return PointCloudCollisionChecker(
        chain=chain,
        model=model,
        frame_provider=chain.link_transforms,
        config=PointCloudCollisionConfig(
            clearance=0.01,
            point_radius=0.0,
            min_scene_points=1,
            max_scene_age_s=0.5,
            segment_joint_step=0.02,
        ),
        now_fn=lambda: 10.0,
    )


def test_check_state_runs_no_forward_kinematics_without_an_attached_payload(
    tmp_path,
    monkeypatch,
):
    """Regression test: the tip transform is consulted only by the attached
    payload checks, so transit and approach states -- the common case, and the
    ones an RRT samples by the thousand -- must not pay a full forward
    kinematics pass they then throw away.
    """

    checker = _fk_attributable_checker(tmp_path)
    checker.update_scene(np.array([[5.0, 0.0, 0.0]]), stamp_s=10.0)
    real_forward = checker.chain.forward
    calls = 0

    def counting_forward(joints):
        nonlocal calls
        calls += 1
        return real_forward(joints)

    monkeypatch.setattr(checker.chain, "forward", counting_forward)

    assert checker.check_state(np.array([0.1])).valid
    assert calls == 0  # no payload attached

    checker.update_target(np.array([[5.0, 0.0, 0.5]]))
    assert checker.check_state(np.array([0.1])).valid
    assert calls == 0  # a static target already lives in the base frame


def test_check_state_inverts_the_tip_transform_once_per_state_when_attached(
    tmp_path,
    monkeypatch,
):
    """With a payload attached the tip transform is needed, but exactly once:
    one forward kinematics pass and one inverse per state, not one inverse per
    checked capsule.
    """

    checker = _fk_attributable_checker(tmp_path)
    checker.update_scene(np.array([[5.0, 0.0, 0.0]]), stamp_s=10.0)
    checker.update_attached_target(
        np.array([[0.0, 0.0, 0.30], [0.0, 0.01, 0.31]]),
        attachment_joints=np.array([0.0]),
    )
    real_forward = checker.chain.forward
    real_inverse = np.linalg.inv
    forward_calls = 0
    inverse_calls = 0

    def counting_forward(joints):
        nonlocal forward_calls
        forward_calls += 1
        return real_forward(joints)

    def counting_inverse(matrix):
        nonlocal inverse_calls
        inverse_calls += 1
        return real_inverse(matrix)

    monkeypatch.setattr(checker.chain, "forward", counting_forward)
    monkeypatch.setattr(np.linalg, "inv", counting_inverse)

    result = checker.check_state(np.array([0.1]))

    assert result.valid
    assert forward_calls == 1
    # Three capsules are checked against the attached payload; the tip
    # transform is inverted once for the state, not once per capsule.
    assert inverse_calls == 1
