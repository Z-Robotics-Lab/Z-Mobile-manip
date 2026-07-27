"""Equivalence tests for the collision-checker latency fast paths.

Both fast paths below are latency-only: the batched frame validation must
accept exactly the transform set the per-frame predicate accepts, and the
self-collision pair prefilter must only skip pairs whose exact segment distance
provably exceeds the acceptance threshold.  These tests pin that equivalence so
a future edit cannot quietly turn either one into an approximation.
"""

from __future__ import annotations

import numpy as np
import pytest

from z_manip.collision import (
    CapsuleSpec,
    PointCloudCollisionChecker,
    PointCloudCollisionConfig,
    RobotCollisionModel,
    SelfCollisionConfig,
)
from z_manip.collision.pointcloud import _segment_distance
from z_manip.kinematics import KinematicChain


_URDF = """
<robot name="arm">
  <link name="base"/>
  <link name="upper"/>
  <link name="fore"/>
  <link name="tool"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/>
    <child link="fore"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="wrist" type="revolute">
    <parent link="fore"/>
    <child link="tool"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _model() -> RobotCollisionModel:
    return RobotCollisionModel(
        capsules=(
            CapsuleSpec(
                "upper", "upper", "fore", 0.03,
                start_offset=(0.0, 0.0, 0.0),
                end_offset=(0.0, 0.0, 0.0),
            ),
            CapsuleSpec(
                "fore", "fore", "tool", 0.03,
                start_offset=(0.0, 0.0, 0.0),
                end_offset=(0.0, 0.0, 0.0),
            ),
            CapsuleSpec(
                "tool", "tool", "tool", 0.02,
                start_offset=(0.0, 0.0, -0.05),
                end_offset=(0.0, 0.0, 0.05),
            ),
            CapsuleSpec(
                "column", "base", "base", 0.05,
                start_offset=(0.0, 0.0, 0.0),
                end_offset=(0.0, 0.0, 0.4),
            ),
        ),
        self_collision=SelfCollisionConfig(
            pairs=(("tool", "column"), ("fore", "column"), ("upper", "tool")),
        ),
    )


def _checker(tmp_path) -> PointCloudCollisionChecker:
    urdf = tmp_path / "arm.urdf"
    urdf.write_text(_URDF)
    chain = KinematicChain.from_urdf(urdf, "base", "tool")
    frames = chain.link_transforms
    return PointCloudCollisionChecker(
        chain=chain,
        model=_model(),
        frame_provider=frames,
        config=PointCloudCollisionConfig(
            clearance=0.01,
            point_radius=0.0,
            min_scene_points=1,
            max_scene_age_s=1.0,
            segment_joint_step=0.02,
        ),
        now_fn=lambda: 10.0,
    )


def _states(count: int, dof: int) -> np.ndarray:
    generator = np.random.default_rng(20260727)
    return generator.uniform(-2.5, 2.5, size=(count, dof))


def test_pair_prefilter_never_skips_a_pair_that_could_collide(tmp_path) -> None:
    checker = _checker(tmp_path)

    for joints in _states(400, checker.chain.dof):
        capsules, problem = checker._world_capsules(joints)
        assert problem is None and capsules is not None
        separable = checker._separable_pairs(capsules)
        for index, (first_index, second_index) in enumerate(
            checker._self_pair_indices
        ):
            if not separable[index]:
                continue
            first, second = capsules[first_index], capsules[second_index]
            distance = _segment_distance(
                first.start, first.end, second.start, second.end,
            )
            threshold = (
                first.spec.radius + second.spec.radius + checker.config.clearance
            )
            assert distance > threshold


def test_self_collision_result_matches_the_unfiltered_scan(tmp_path) -> None:
    checker = _checker(tmp_path)

    def exhaustive(capsules):
        for first_index, second_index in checker._self_pair_indices:
            first, second = capsules[first_index], capsules[second_index]
            distance = _segment_distance(
                first.start, first.end, second.start, second.end,
            )
            threshold = (
                first.spec.radius + second.spec.radius + checker.config.clearance
            )
            if distance <= threshold:
                return first.spec.name, second.spec.name, distance
        return None

    collisions = 0
    for joints in _states(400, checker.chain.dof):
        capsules, problem = checker._world_capsules(joints)
        assert problem is None and capsules is not None
        expected = exhaustive(capsules)
        result = checker._check_self_collision(capsules)
        if expected is None:
            assert result is None
            continue
        collisions += 1
        assert result is not None
        assert result.capsules == expected[:2]
        assert result.distance == expected[2]
    assert collisions, "the sampled states must exercise the colliding branch"


def test_batched_frame_validation_accepts_exactly_the_per_frame_predicate(
    tmp_path,
) -> None:
    checker = _checker(tmp_path)
    chain = checker.chain
    valid = chain.link_transforms(np.zeros(chain.dof))

    accepted = checker._validated_transforms(valid)
    assert accepted is not None
    for name, transform in accepted.items():
        assert np.array_equal(transform, np.asarray(valid[name], dtype=float))

    corruptions = []
    skewed = np.asarray(valid["tool"], dtype=float).copy()
    skewed[:3, :3] *= 1.01
    corruptions.append(skewed)
    reflected = np.asarray(valid["tool"], dtype=float).copy()
    reflected[:3, 0] *= -1.0
    corruptions.append(reflected)
    shifted_row = np.asarray(valid["tool"], dtype=float).copy()
    shifted_row[3, 0] = 1e-3
    corruptions.append(shifted_row)
    infinite = np.asarray(valid["tool"], dtype=float).copy()
    infinite[0, 3] = np.inf
    corruptions.append(infinite)

    for corrupted in corruptions:
        frames = dict(valid)
        frames["tool"] = corrupted
        assert not PointCloudCollisionChecker._valid_transform(corrupted)
        assert checker._validated_transforms(frames) is None
        capsules, problem = checker._world_capsules(np.zeros(chain.dof))
        assert capsules is not None and problem is None

    # A transform that only grazes the tolerance must land on the same side for
    # both predicates, because the batched form reuses the same comparison.
    grazing = np.asarray(valid["tool"], dtype=float).copy()
    grazing[:3, :3] *= 1.0 + 4e-6
    frames = dict(valid)
    frames["tool"] = grazing
    assert PointCloudCollisionChecker._valid_transform(grazing) is (
        checker._validated_transforms(frames) is not None
    )


def test_invalid_frame_set_still_names_the_offending_link(tmp_path) -> None:
    checker = _checker(tmp_path)
    chain = checker.chain
    broken = dict(chain.link_transforms(np.zeros(chain.dof)))
    broken["fore"] = np.zeros((4, 4))

    checker.frame_provider = lambda joints: broken
    capsules, problem = checker._world_capsules(np.zeros(chain.dof))
    assert capsules is None
    assert problem is not None
    assert "invalid transform for 'fore'" in problem.reason


def test_ragged_frame_shapes_fail_closed(tmp_path) -> None:
    checker = _checker(tmp_path)
    chain = checker.chain
    ragged = dict(chain.link_transforms(np.zeros(chain.dof)))
    ragged["tool"] = np.zeros((3, 3))

    checker.frame_provider = lambda joints: ragged
    capsules, problem = checker._world_capsules(np.zeros(chain.dof))
    assert capsules is None
    assert problem is not None
    assert "invalid transform for 'tool'" in problem.reason


@pytest.mark.parametrize("scale", [1.0, 3.0])
def test_prefilter_bound_is_a_true_lower_bound(tmp_path, scale: float) -> None:
    checker = _checker(tmp_path)
    generator = np.random.default_rng(7)
    for _ in range(2000):
        points = generator.normal(0.0, scale, size=(4, 3))
        first_start, first_end, second_start, second_end = points
        exact = _segment_distance(first_start, first_end, second_start, second_end)
        bound = (
            float(np.linalg.norm(
                0.5 * (first_start + first_end) - 0.5 * (second_start + second_end)
            ))
            - 0.5 * float(np.linalg.norm(first_end - first_start))
            - 0.5 * float(np.linalg.norm(second_end - second_start))
        )
        assert bound <= exact + 1e-12
