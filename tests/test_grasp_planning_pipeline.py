import numpy as np
import pytest

from z_manip.kinematics.robust_ik import IKFailure, IKSolution
from z_manip.models.grasp_source import GraspCandidates
from z_manip.models.planner import JointTrajectory, PlanningError
from z_manip.planning.grasp_pipeline import (
    GraspPlanConfig,
    GraspPlanGenerator,
    grasp_pregrasp_pose,
)
from z_manip.planning.rrt_connect import JointSpaceRRTConnect, RRTConnectConfig
from z_manip.planning_control import PlanningCancelled, PlanningControl


class FakeIK:
    def __init__(self, reject_x=()):
        self.reject_x = tuple(reject_x)
        self.calls = []

    def solve(self, target, current=None, *, control=None):
        self.calls.append(np.asarray(target).copy())
        x = float(target[0, 3])
        if any(abs(x - rejected) < 0.035 for rejected in self.reject_x):
            raise IKFailure("no IK solution")
        joints = np.full(2, x)
        return IKSolution(joints, 0.0, 0.0, 0.2, 3, 0)


class FakePlanner:
    joint_names = ("j1", "j2")

    def __init__(self, reject_goal_x=(), reject_segment_x=()):
        self.reject_goal_x = tuple(reject_goal_x)
        self.reject_segment_x = tuple(reject_segment_x)
        self.calls = []

    def plan_joint(self, start, goal, *, timeout_s, control=None):
        self.calls.append((np.asarray(start).copy(), np.asarray(goal).copy(), timeout_s))
        if any(abs(float(goal[0]) - rejected) < 0.035 for rejected in self.reject_goal_x):
            raise PlanningError("blocked")
        return JointTrajectory(self.joint_names, np.vstack((start, goal)))

    def segment_valid(self, first, second, *, control=None):
        lo, hi = sorted((float(first[0]), float(second[0])))
        return not any(lo <= rejected <= hi for rejected in self.reject_segment_x)


def _pose(position, approach=(0.0, 0.0, 1.0)):
    approach = np.asarray(approach, dtype=float)
    approach /= np.linalg.norm(approach)
    reference = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(reference, approach)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    closing = np.cross(reference, approach)
    closing /= np.linalg.norm(closing)
    binormal = np.cross(approach, closing)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((closing, binormal, approach))
    transform[:3, 3] = position
    return transform


def _candidates(poses, widths=None):
    return GraspCandidates(
        grasps=np.stack(poses),
        scores=np.linspace(0.95, 0.65, len(poses)),
        centroid=np.mean([pose[:3, 3] for pose in poses], axis=0),
        frame="arm_base",
        num_raw=len(poses),
        widths=None if widths is None else np.asarray(widths),
    )


def test_pipeline_rejects_width_and_ik_then_plans_oblique_6dof_grasp():
    too_wide = _pose((0.2, 0.0, 0.2))
    unreachable = _pose((0.4, 0.0, 0.2))
    oblique = _pose((0.6, 0.0, 0.25), approach=(1.0, 0.0, 1.0))
    ik = FakeIK(reject_x=(0.4,))
    planner = FakePlanner()
    generator = GraspPlanGenerator(
        ik,
        planner,
        GraspPlanConfig(
            min_width_m=0.01,
            max_width_m=0.08,
            pregrasp_distance_m=0.10,
            approach_steps=4,
            lift_distance_m=0.06,
            lift_steps=3,
            symmetry_samples=2,
        ),
    )

    result = generator.plan(
        _candidates((too_wide, unreachable, oblique), widths=(0.12, 0.04, 0.05)),
        current_joints=np.zeros(2),
    )

    expected_pregrasp = oblique[:3, 3] - 0.10 * oblique[:3, 2]
    assert result.candidate_index == 2
    assert np.allclose(result.grasp_pose, oblique)
    assert np.allclose(result.pregrasp_pose[:3, 3], expected_pregrasp)
    assert result.approach_joints.shape == (4, 2)
    assert result.lift_joints.shape == (3, 2)
    assert np.allclose(result.transit.waypoints[0], np.zeros(2))
    assert result.required_width_m == pytest.approx(0.05)
    assert result.failures[0].stage == "aperture"
    assert any(failure.stage == "ik" for failure in result.failures)


def test_pipeline_uses_global_ik_once_then_bounded_cartesian_continuation():
    class ContinuationIK:
        def __init__(self):
            self.global_targets = []
            self.continuation_calls = []

        @staticmethod
        def _solution(target, seed_index):
            joints = np.asarray((target[0, 3], target[2, 3]), dtype=float)
            return IKSolution(joints, 0.0, 0.0, 0.2, 1, seed_index, 0.25)

        def solve(self, target, current=None, *, control=None):
            del current, control
            self.global_targets.append(np.asarray(target).copy())
            return self._solution(target, 7)

        def solve_continuation(
            self,
            target,
            current,
            *,
            max_joint_step_rad,
            control=None,
        ):
            del control
            self.continuation_calls.append((
                np.asarray(target).copy(),
                np.asarray(current).copy(),
                float(max_joint_step_rad),
            ))
            return self._solution(target, 0)

    ik = ContinuationIK()
    generator = GraspPlanGenerator(
        ik,
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.09,
            approach_steps=4,
            lift_distance_m=0.06,
            lift_steps=3,
            symmetry_samples=1,
            max_feasible_plans=1,
            tool_from_tip=np.eye(4),
        ),
    )
    grasp = _pose((0.5, 0.0, 0.2), approach=(1.0, 0.0, 0.0))

    result = generator.plan(
        _candidates((grasp,), widths=(0.04,)),
        current_joints=np.zeros(2),
    )

    assert len(ik.global_targets) == 1
    assert len(ik.continuation_calls) == (4 - 1) + (3 - 1)
    pregrasp_x = float(ik.global_targets[0][0, 3])
    assert float(ik.continuation_calls[0][0][0, 3]) > pregrasp_x
    assert all(call[2] == pytest.approx(0.45) for call in ik.continuation_calls)
    for (_target, current, _step), next_call in zip(
        ik.continuation_calls,
        ik.continuation_calls[1:],
    ):
        expected = ContinuationIK._solution(_target, 0).joints
        np.testing.assert_allclose(next_call[1], expected)
    assert result.approach_joints.shape == (4, 2)
    assert result.lift_joints.shape == (3, 2)


def test_pipeline_uses_up_and_back_lift_at_outer_workspace_boundary():
    class BoundaryIK:
        @staticmethod
        def _solution(target):
            joints = np.asarray((target[0, 3], target[2, 3]), dtype=float)
            return IKSolution(joints, 0.0, 0.0, 0.2, 1, 0, 0.25)

        def solve(self, target, current=None, *, control=None):
            del current, control
            return self._solution(target)

        def solve_continuation(
            self,
            target,
            current,
            *,
            max_joint_step_rad,
            control=None,
        ):
            del current, max_joint_step_rad, control
            if float(target[2, 3]) > 0.25:
                raise IKFailure("vertical lift leaves the reachable workspace")
            return self._solution(target)

    generator = GraspPlanGenerator(
        BoundaryIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.08,
            approach_steps=3,
            lift_distance_m=0.07,
            lift_steps=3,
            fallback_lift_vertical_m=0.045,
            fallback_lift_retreat_m=0.025,
            symmetry_samples=1,
            tool_from_tip=np.eye(4),
        ),
    )
    grasp = _pose((0.5, 0.0, 0.2), approach=(1.0, 0.0, 0.0))

    result = generator.plan(
        _candidates((grasp,), widths=(0.04,)),
        current_joints=np.zeros(2),
    )

    # The nominal endpoint (0.50, 0.27) is unreachable.  The selected
    # fallback keeps 45 mm of vertical clearance and retreats 25 mm toward
    # the arm base instead of rejecting an otherwise valid contact pose.
    np.testing.assert_allclose(result.lift_joints[-1], (0.475, 0.245), atol=1e-9)


def test_pipeline_tries_next_candidate_after_motion_planning_failure():
    blocked = _pose((0.35, 0.0, 0.2))
    clear = _pose((0.55, 0.0, 0.2))
    generator = GraspPlanGenerator(
        FakeIK(),
        FakePlanner(reject_goal_x=(0.35,)),
        GraspPlanConfig(
            pregrasp_distance_m=0.02,
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
        ),
    )

    result = generator.plan(_candidates((blocked, clear)), current_joints=np.zeros(2))

    assert result.candidate_index == 1
    assert any(failure.stage == "planning" for failure in result.failures)


def test_score_ranking_does_not_starve_lateral_grasp():
    class LateralOnlyIK(FakeIK):
        def solve(self, target, current=None, *, control=None):
            self.calls.append(np.asarray(target).copy())
            if float(target[0, 2]) < 0.9:
                raise IKFailure("only the lateral approach is reachable")
            return IKSolution(np.zeros(2), 0.0, 0.0, 0.2, 1, 0)

    poses = (
        _pose((0.5, 0.0, 0.2), approach=(0.0, 0.0, 1.0)),
        _pose((0.5, 0.0, 0.2), approach=(0.05, 0.0, 0.999)),
        _pose((0.5, 0.0, 0.2), approach=(-0.05, 0.0, 0.999)),
        _pose((0.5, 0.0, 0.2), approach=(1.0, 0.0, 0.0)),
    )
    generator = GraspPlanGenerator(
        LateralOnlyIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.02,
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_candidates=2,
            max_hypotheses=2,
            tool_from_tip=np.eye(4),
        ),
    )

    result = generator.plan(_candidates(poses), current_joints=np.zeros(2))

    assert result.candidate_index == 3


def test_pose_ranker_prioritizes_reachable_symmetry_without_losing_index():
    grasp = _pose((0.5, 0.0, 0.2), approach=(0.0, 0.0, 1.0))
    ik = FakeIK()

    def pose_ranker(target, *, control=None):
        del control
        return 0.0 if float(target[0, 0]) < 0.0 else 10.0

    result = GraspPlanGenerator(
        ik,
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.02,
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=2,
            max_hypotheses=2,
            tool_from_tip=np.eye(4),
        ),
    ).plan(
        _candidates((grasp,)),
        current_joints=np.zeros(2),
        pose_ranker=pose_ranker,
    )

    assert result.symmetry_index == 1
    assert float(ik.calls[0][0, 0]) < 0.0


def test_candidate_diversity_never_runs_lower_global_rank_first():
    class FirstCandidateFails(FakeIK):
        def solve(self, target, current=None, *, control=None):
            self.calls.append(np.asarray(target).copy())
            if len(self.calls) == 1:
                raise IKFailure("highest-ranked candidate is unreachable")
            return IKSolution(np.zeros(2), 0.0, 0.0, 0.2, 1, 0, 0.25)

    poses = (
        _pose((0.40, 0.0, 0.2), approach=(0.0, 0.0, 1.0)),
        _pose((0.41, 0.0, 0.2), approach=(0.05, 0.0, 0.999)),
        _pose((0.42, 0.0, 0.2), approach=(-0.05, 0.0, 0.999)),
        _pose((0.43, 0.0, 0.2), approach=(1.0, 0.0, 0.0)),
    )
    ik = FirstCandidateFails()
    result = GraspPlanGenerator(
        ik,
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.02,
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_candidates=4,
            max_hypotheses=4,
            max_feasible_plans=1,
            tool_from_tip=np.eye(4),
        ),
    ).plan(_candidates(poses), current_joints=np.zeros(2))

    assert result.candidate_index == 1
    assert result.selected_global_rank == 2
    assert result.higher_rank_rejection_count == 1
    expected_second = grasp_pregrasp_pose(poses[1], 0.02)
    np.testing.assert_allclose(ik.calls[1], expected_second)


def test_first_feasible_search_uses_reachability_after_top_score_fails():
    class TopCandidateFails(FakeIK):
        def solve(self, target, current=None, *, control=None):
            self.calls.append(np.asarray(target).copy())
            if len(self.calls) == 1:
                raise IKFailure("highest-ranked candidate is unreachable")
            x = float(target[0, 3])
            return IKSolution(np.full(2, x), 0.0, 0.0, 0.2, 1, 0, 0.25)

    poses = tuple(
        _pose((x, 0.0, 0.2), approach=(0.0, 0.0, 1.0))
        for x in (0.40, 0.50, 0.60, 0.70)
    )
    reachability_cost = {0.40: 0.0, 0.50: 8.0, 0.60: 4.0, 0.70: 1.0}
    rank_calls = []

    def pose_ranker(target, *, control=None):
        del control
        x = float(target[0, 3])
        rank_calls.append(x)
        return reachability_cost[round(x, 2)]

    ik = TopCandidateFails()
    result = GraspPlanGenerator(
        ik,
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.02,
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_candidates=4,
            max_hypotheses=4,
            max_feasible_plans=1,
            tool_from_tip=np.eye(4),
        ),
    ).plan(
        _candidates(poses),
        current_joints=np.zeros(2),
        pose_ranker=pose_ranker,
    )

    assert result.candidate_index == 3
    assert [round(float(call[0, 3]), 2) for call in ik.calls[:2]] == [0.40, 0.70]
    # The advisory batch ranking is cached for the per-candidate symmetry
    # ordering instead of repeating the same FK work during exact search.
    assert len(rank_calls) == 4


@pytest.mark.parametrize(
    ("failed_call", "expected_phase"),
    ((0, "pregrasp"), (1, "approach"), (2, "lift")),
)
def test_ik_rejections_report_the_exact_cartesian_phase(failed_call, expected_phase):
    class PhaseRejectingIK:
        def __init__(self):
            self.calls = 0

        def _solve(self):
            call = self.calls
            self.calls += 1
            if call == failed_call:
                raise IKFailure("phase-specific failure")
            return IKSolution(np.zeros(2), 0.0, 0.0, 0.2, 1, 0, 0.25)

        def solve(self, target, current=None, *, control=None):
            del target, current, control
            return self._solve()

        def solve_continuation(
            self,
            target,
            current,
            *,
            max_joint_step_rad,
            control=None,
        ):
            del target, current, max_joint_step_rad, control
            return self._solve()

    generator = GraspPlanGenerator(
        PhaseRejectingIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.02,
            approach_steps=2,
            fallback_lift_vertical_m=0.10,
            fallback_lift_retreat_m=0.0,
            lift_steps=2,
            symmetry_samples=1,
            max_hypotheses=1,
            tool_from_tip=np.eye(4),
        ),
    )

    with pytest.raises(PlanningError) as captured:
        generator.plan(
            _candidates((_pose((0.5, 0.0, 0.2)),)),
            current_joints=np.zeros(2),
        )

    assert captured.value.failures[0].stage == "ik"
    assert captured.value.failures[0].reason.startswith(
        f"{expected_phase} IK failed:"
    )


def test_phase_collision_callbacks_limit_contact_and_attach_lift_payload():
    approach_calls = []
    lift_calls = []
    generator = GraspPlanGenerator(
        FakeIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.04,
            approach_steps=4,
            lift_steps=3,
            symmetry_samples=1,
        ),
        approach_path_valid=lambda path: (
            approach_calls.append(np.asarray(path).copy()) or True
        ),
        lift_segment_valid=lambda first, second, attachment: (
            lift_calls.append((np.asarray(first), np.asarray(second), np.asarray(attachment)))
            or True
        ),
    )

    result = generator.plan(
        _candidates((_pose((0.5, 0.0, 0.2)),), widths=(0.04,)),
        current_joints=np.zeros(2),
    )

    assert len(approach_calls) == 1
    assert approach_calls[0].shape == (5, 2)
    np.testing.assert_allclose(approach_calls[0][-1], result.approach_joints[-1])
    assert len(lift_calls) == 3
    for _first, _second, attachment in lift_calls:
        np.testing.assert_allclose(attachment, result.approach_joints[-1])


def test_phase_collision_callbacks_receive_the_candidate_required_width():
    approach_widths = []
    lift_widths = []

    def approach(path, *, required_width_m=None):
        approach_widths.append(required_width_m)
        return len(path) >= 2

    def lift(first, second, attachment, *, required_width_m=None):
        lift_widths.append(required_width_m)
        return np.asarray(first).shape == np.asarray(second).shape == np.asarray(attachment).shape

    generator = GraspPlanGenerator(
        FakeIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.04,
            approach_steps=3,
            lift_steps=3,
            symmetry_samples=1,
        ),
        approach_path_valid=approach,
        lift_segment_valid=lift,
    )

    generator.plan(
        _candidates((_pose((0.5, 0.0, 0.2)),), widths=(0.04,)),
        current_joints=np.zeros(2),
    )

    assert approach_widths == pytest.approx([0.04])
    assert lift_widths == pytest.approx([0.04, 0.04, 0.04])


def test_legacy_segment_callback_keeps_final_segment_contact_marker():
    contact_markers = []
    generator = GraspPlanGenerator(
        FakeIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.04,
            approach_steps=4,
            lift_steps=2,
            symmetry_samples=1,
        ),
        approach_segment_valid=lambda _first, _second, contact: (
            contact_markers.append(contact) or True
        ),
    )

    generator.plan(
        _candidates((_pose((0.5, 0.0, 0.2)),), widths=(0.04,)),
        current_joints=np.zeros(2),
        control=PlanningControl(),
    )

    assert contact_markers == [False, False, False, True]


def test_segment_and_path_approach_validators_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        GraspPlanGenerator(
            FakeIK(),
            FakePlanner(),
            approach_segment_valid=lambda _first, _second, _contact: True,
            approach_path_valid=lambda _path: True,
        )


def test_control_aware_path_validator_stops_after_one_work_unit():
    work_units = []

    def validate_path(path, control):
        for joints in path:
            control.checkpoint("test approach work")
            work_units.append(np.asarray(joints).copy())
        return True

    generator = GraspPlanGenerator(
        FakeIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.04,
            approach_steps=24,
            lift_steps=2,
            symmetry_samples=1,
        ),
        approach_path_valid=validate_path,
    )
    control = PlanningControl(cancel_check=lambda: len(work_units) >= 1)

    with pytest.raises(PlanningCancelled, match="test approach work"):
        generator.plan(
            _candidates((_pose((0.5, 0.0, 0.2)),), widths=(0.04,)),
            current_joints=np.zeros(2),
            control=control,
        )

    assert len(work_units) == 1


def test_path_validator_body_type_error_is_not_retried_as_legacy_callback():
    argument_counts = []

    def validate_path(*args):
        argument_counts.append(len(args))
        raise TypeError("path validator implementation failed")

    generator = GraspPlanGenerator(
        FakeIK(),
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.04,
            approach_steps=3,
            lift_steps=2,
            symmetry_samples=1,
        ),
        approach_path_valid=validate_path,
    )

    with pytest.raises(TypeError, match="implementation failed"):
        generator.plan(
            _candidates((_pose((0.5, 0.0, 0.2)),), widths=(0.04,)),
            current_joints=np.zeros(2),
        )

    assert argument_counts == [2]


def test_pipeline_reports_all_candidate_failures_without_fallback_motion():
    generator = GraspPlanGenerator(
        FakeIK(reject_x=(0.2, 0.3)),
        FakePlanner(),
        GraspPlanConfig(pregrasp_distance_m=0.01, symmetry_samples=1),
    )
    with pytest.raises(PlanningError, match="rejections=\\{ik:2\\}") as captured:
        generator.plan(
            _candidates((_pose((0.2, 0.0, 0.2)), _pose((0.3, 0.0, 0.2)))),
            current_joints=np.zeros(2),
        )
    assert "#1/0 ik: pregrasp IK failed: no IK solution" in str(captured.value)
    assert len(captured.value.failures) == 2
    assert [failure.candidate_index for failure in captured.value.failures] == [0, 1]


def test_pipeline_checks_cartesian_approach_segments_for_collision():
    first = _pose((0.4, 0.0, 0.2), approach=(1.0, 0.0, 1.0))
    second = _pose((0.7, 0.0, 0.2))
    generator = GraspPlanGenerator(
        FakeIK(),
        FakePlanner(reject_segment_x=(0.36,)),
        GraspPlanConfig(
            pregrasp_distance_m=0.10,
            approach_steps=5,
            lift_steps=2,
            symmetry_samples=1,
        ),
    )

    result = generator.plan(_candidates((first, second)), current_joints=np.zeros(2))

    assert result.candidate_index == 1
    assert any(failure.stage == "approach_collision" for failure in result.failures)


def test_rejected_approach_does_not_solve_redundant_lift_ik():
    class CountingContinuationIK(FakeIK):
        def __init__(self):
            super().__init__()
            self.continuation_calls = 0

        def solve_continuation(
            self,
            target,
            current,
            *,
            max_joint_step_rad,
            control=None,
        ):
            del max_joint_step_rad, control
            self.continuation_calls += 1
            return self.solve(target, current=current)

    ik = CountingContinuationIK()
    generator = GraspPlanGenerator(
        ik,
        FakePlanner(),
        GraspPlanConfig(
            pregrasp_distance_m=0.06,
            approach_steps=4,
            lift_distance_m=0.07,
            lift_steps=4,
            symmetry_samples=1,
            max_hypotheses=1,
        ),
        approach_path_valid=lambda _path: False,
    )

    with pytest.raises(PlanningError) as captured:
        generator.plan(
            _candidates((_pose((0.5, 0.0, 0.2)),)),
            current_joints=np.zeros(2),
        )

    assert captured.value.failures[0].stage == "approach_collision"
    assert ik.continuation_calls == 3


def test_pipeline_does_not_swallow_typed_planning_cancellation():
    class CancelingPlanner(FakePlanner):
        def plan_joint(self, start, goal, *, timeout_s, control=None):
            self.calls.append((np.asarray(start), np.asarray(goal), timeout_s))
            raise PlanningCancelled("task generation is stale")

    planner = CancelingPlanner()
    generator = GraspPlanGenerator(
        FakeIK(),
        planner,
        GraspPlanConfig(
            pregrasp_distance_m=0.02,
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
        ),
    )

    with pytest.raises(PlanningCancelled, match="task generation is stale"):
        generator.plan(
            _candidates((_pose((0.35, 0.0, 0.2)), _pose((0.55, 0.0, 0.2)))),
            current_joints=np.zeros(2),
            control=PlanningControl(),
        )

    assert len(planner.calls) == 1


def test_control_none_preserves_legacy_backend_signatures():
    class LegacyIK:
        def solve(self, target, current=None):
            joints = np.full(2, float(target[0, 3]))
            return IKSolution(joints, 0.0, 0.0, 0.2, 1, 0, 0.25)

    class LegacyPlanner:
        joint_names = ("j1", "j2")

        def __init__(self):
            self.plan_calls = 0
            self.segment_calls = 0

        def plan_joint(self, start, goal, *, timeout_s):
            self.plan_calls += 1
            return JointTrajectory(self.joint_names, np.vstack((start, goal)))

        def segment_valid(self, _first, _second):
            self.segment_calls += 1
            return True

    planner = LegacyPlanner()
    generator = GraspPlanGenerator(
        LegacyIK(),
        planner,
        GraspPlanConfig(
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_feasible_plans=1,
        ),
    )

    result = generator.plan(
        _candidates((_pose((0.5, 0.0, 0.2)),)),
        current_joints=np.zeros(2),
    )

    assert result.candidate_index == 0
    assert planner.plan_calls == 1
    assert planner.segment_calls == 4


def test_bounded_refinement_selects_better_complete_plan():
    class QualityIK:
        def solve(self, target, current=None, *, control=None):
            x = float(target[0, 3])
            is_better = x > 0.45
            return IKSolution(
                joints=np.full(2, x),
                position_error_m=0.0,
                orientation_error_rad=0.0,
                manipulability=5.0 if is_better else 0.0,
                iterations=1,
                seed_index=0,
                min_joint_limit_margin=0.5 if is_better else 0.005,
            )

    rrt = JointSpaceRRTConnect(
        joint_names=("j1", "j2"),
        lower_limits=np.full(2, -2.0),
        upper_limits=np.full(2, 2.0),
        state_valid=lambda joints: bool(np.all(np.isfinite(joints))),
        config=RRTConnectConfig(
            collision_resolution=0.05,
            max_iterations=100,
            shortcut_attempts=0,
            seed=7,
        ),
    )

    class QualityPlanner:
        joint_names = rrt.joint_names

        def __init__(self):
            self.path_costs = []
            self.segment_calls = 0

        def plan_joint(self, start, goal, *, timeout_s, control=None):
            trajectory = rrt.plan_joint(
                start,
                goal,
                timeout_s=timeout_s,
                control=control,
            )
            path = np.asarray(trajectory.waypoints)
            self.path_costs.append(
                float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            )
            return trajectory

        def segment_valid(self, first, second, *, control=None):
            self.segment_calls += 1
            return rrt.segment_valid(first, second, control=control)

    first = _pose((0.4, 0.0, 0.2))
    better = _pose((0.5, 0.0, 0.2))
    candidates = GraspCandidates(
        grasps=np.stack((first, better)),
        scores=np.array([0.8, 0.8]),
        centroid=np.array([0.45, 0.0, 0.2]),
        frame="arm_base",
        num_raw=2,
        widths=np.array([0.04, 0.04]),
    )
    planner = QualityPlanner()
    generator = GraspPlanGenerator(
        QualityIK(),
        planner,
        GraspPlanConfig(
            approach_steps=2,
            lift_steps=2,
            symmetry_samples=1,
            max_feasible_plans=2,
            solution_refinement_timeout_s=0.35,
        ),
    )

    result = generator.plan(candidates, current_joints=np.ones(2))

    assert len(planner.path_costs) == 2
    assert planner.path_costs[1] < planner.path_costs[0]
    assert planner.segment_calls == 8
    assert result.candidate_index == 1
