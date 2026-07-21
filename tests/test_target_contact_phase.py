from __future__ import annotations

import numpy as np
import pytest

from z_manip.collision import (
    CapsuleSpec,
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
    SegmentCollisionResult,
    check_target_contact_approach,
)
from z_manip.kinematics import KinematicChain
from z_manip.planning_control import PlanningCancelled, PlanningControl


_URDF = """
<robot name="contact_slider">
  <link name="base"/>
  <link name="tool"/>
  <joint name="approach" type="prismatic">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _checker_pair(tmp_path, model, target, *, scene=None):
    urdf = tmp_path / "contact_slider.urdf"
    urdf.write_text(_URDF)
    chain = KinematicChain.from_urdf(urdf, "base", "tool")

    def frames(joints):
        return {"base": np.eye(4), "tool": chain.forward(joints)}

    def build(allowed=()):
        checker = PointCloudCollisionChecker(
            chain=chain,
            model=model,
            frame_provider=frames,
            config=PointCloudCollisionConfig(
                clearance=0.0,
                point_radius=0.0,
                min_scene_points=1,
                segment_joint_step=0.005,
            ),
            now_fn=lambda: 10.0,
        )
        checker.update_scene(
            np.array([[2.0, 2.0, 2.0]]) if scene is None else np.asarray(scene),
            stamp_s=10.0,
        )
        checker.update_target(target, allowed_contact_capsules=allowed)
        return checker

    return build(), build(model.target_contact_capsules)


def _two_finger_model(*, palm_offset=-0.08):
    return RobotCollisionModel(
        capsules=(
            CapsuleSpec(
                "palm",
                "tool",
                "tool",
                0.018,
                start_offset=(palm_offset, 0.0, 0.0),
                end_offset=(palm_offset, 0.0, 0.0),
            ),
            CapsuleSpec(
                "finger_left",
                "tool",
                "tool",
                0.025,
                start_offset=(0.0, 0.03, 0.0),
                end_offset=(0.0, 0.03, 0.0),
            ),
            CapsuleSpec(
                "finger_right",
                "tool",
                "tool",
                0.025,
                start_offset=(0.0, -0.03, 0.0),
                end_offset=(0.0, -0.03, 0.0),
            ),
        ),
        target_contact_capsules=("finger_left", "finger_right"),
    )


def test_geometry_entry_allows_finger_contact_across_multiple_final_segments(tmp_path):
    model = _two_finger_model()
    target = np.array([[0.0, 0.03, 0.0], [0.0, -0.03, 0.0]])
    blocked, finger_contact = _checker_pair(tmp_path, model, target)
    path = np.array([[-0.14], [-0.10], [-0.06], [-0.02], [0.0]])

    result = check_target_contact_approach(
        path,
        no_contact=blocked,
        finger_contact=finger_contact,
        allowed_contact_capsules=model.target_contact_capsules,
    )

    assert result.valid
    assert result.contact_entry_segment == 2
    assert len(path) - 1 - result.contact_entry_segment == 2
    assert "finger-only target contact" in result.reason


def test_palm_contact_before_fingers_cannot_open_the_contact_phase(tmp_path):
    model = _two_finger_model(palm_offset=0.08)
    target = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.03, 0.0],
        [0.0, -0.03, 0.0],
    ])
    blocked, finger_contact = _checker_pair(tmp_path, model, target)

    result = check_target_contact_approach(
        np.array([[-0.16], [-0.12], [-0.08], [-0.04], [0.0]]),
        no_contact=blocked,
        finger_contact=finger_contact,
        allowed_contact_capsules=model.target_contact_capsules,
    )

    assert not result.valid
    assert result.contact_entry_segment is None
    assert result.collision is not None
    assert result.collision.state_result is not None
    assert result.collision.state_result.kind == "target"
    assert result.collision.state_result.capsules == ("palm",)


def test_scene_collision_remains_forbidden_for_allowed_finger_capsule(tmp_path):
    model = _two_finger_model()
    target = np.array([[0.0, 0.03, 0.0], [0.0, -0.03, 0.0]])
    scene = np.array([[0.02, 0.03, 0.0]])
    blocked, finger_contact = _checker_pair(
        tmp_path,
        model,
        target,
        scene=scene,
    )

    result = check_target_contact_approach(
        np.array([[-0.14], [-0.10], [-0.06], [-0.02], [0.0]]),
        no_contact=blocked,
        finger_contact=finger_contact,
        allowed_contact_capsules=model.target_contact_capsules,
    )

    assert not result.valid
    assert result.contact_entry_segment == 2
    assert result.collision is not None
    assert result.collision.state_result is not None
    assert result.collision.state_result.kind == "scene"


def test_empty_contact_allowlist_remains_strict_and_duplicates_are_rejected(tmp_path):
    model = _two_finger_model()
    target = np.array([[0.0, 0.03, 0.0]])
    blocked, finger_contact = _checker_pair(tmp_path, model, target)
    path = np.array([[-0.1], [0.0]])

    strict = check_target_contact_approach(
        path,
        no_contact=blocked,
        finger_contact=finger_contact,
        allowed_contact_capsules=(),
    )
    assert not strict.valid
    assert strict.contact_entry_segment is None

    with pytest.raises(ValueError, match="unique non-empty"):
        check_target_contact_approach(
            path,
            no_contact=blocked,
            finger_contact=finger_contact,
            allowed_contact_capsules=("finger_left", "finger_left"),
        )


def test_contact_path_cancellation_stops_before_second_segment_work_unit():
    work_units = []

    class RecordingChecker:
        def check_segment(self, first, second, *, max_joint_step=None, control=None):
            work_units.append((np.asarray(first), np.asarray(second)))
            return SegmentCollisionResult(True, "collision-free segment")

    checker = RecordingChecker()
    control = PlanningControl(cancel_check=lambda: len(work_units) >= 1)

    with pytest.raises(PlanningCancelled, match="target-contact approach segment 1"):
        check_target_contact_approach(
            np.arange(26, dtype=float)[:, None],
            no_contact=checker,
            finger_contact=checker,
            allowed_contact_capsules=("finger",),
            control=control,
        )

    assert len(work_units) == 1
