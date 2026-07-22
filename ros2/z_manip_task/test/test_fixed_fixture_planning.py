"""Offline integration contracts for immutable Go2W fixture clearance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from z_manip.configuration import load_stack_config
from z_manip_task import planning as planning_module
from z_manip_task.planning import OnlinePlanner


ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT.parent / "go2W_Sim/assets/urdf/go2w_sensored.urdf"

# Both endpoints are clear.  Linear interpolation passes through a measured
# Mid360/palm collision near alpha=0.4.  This regression is deliberately not a
# two-endpoint test: the planner must sample the complete joint-space edge.
EDGE_START = np.asarray((
    1.2827352660515232,
    1.6711694229138814,
    -0.34186831424772457,
    0.34931490848846236,
    0.11322961348658067,
    -1.800730239834958,
))
EDGE_END = np.asarray((
    -1.5823539356772047,
    2.0415123046225023,
    -1.0005359705788517,
    -1.408861622965804,
    0.5141252702428707,
    0.8294463891258359,
))
EDGE_COLLISION = EDGE_START + 0.4 * (EDGE_END - EDGE_START)


def _planner() -> OnlinePlanner:
    config = load_stack_config(
        ROOT / "configs/go2w_piper.json",
        environ={"Z_MANIP_ROBOT_URDF": str(URDF)},
    )
    return OnlinePlanner(config)


class _AlwaysClearSceneChecker:
    """Point-cloud fake that isolates the immutable-fixture planning path."""

    def __init__(self, **_kwargs):
        pass

    def update_scene(self, *_args, **_kwargs):
        return 64

    def update_target(self, *_args, **_kwargs):
        return 1

    def update_attached_target(self, *_args, **_kwargs):
        return 1

    def is_state_valid(self, _joints):
        return True

    def check_state(self, _joints):
        return SimpleNamespace(valid=True)

    def is_segment_valid(self, _first, _second):
        return True


class _FixturePhaseProbe:
    """Exercise every callback installed by :meth:`OnlinePlanner._plan`."""

    def __init__(
        self,
        _ik,
        joint_planner,
        _config,
        *,
        approach_path_valid,
        lift_segment_valid,
    ):
        self.joint_planner = joint_planner
        self.approach_path_valid = approach_path_valid
        self.lift_segment_valid = lift_segment_valid

    def plan(self, _candidates, *, current_joints, pose_ranker, control):
        del current_joints, pose_ranker, control

        # Candidate/pregrasp states enter RRT through this same state-validity
        # callback.  Scene collision is faked clear, so only the real fixed
        # fixture guard can reject this state.
        assert not self.joint_planner.state_valid(EDGE_COLLISION)

        # Safe endpoints are insufficient: RRT must continuously sample the
        # edge and reject its Mid360/palm interior collision.
        assert self.joint_planner.state_valid(EDGE_START)
        assert self.joint_planner.state_valid(EDGE_END)
        assert not self.joint_planner.segment_valid(EDGE_START, EDGE_END)

        # Cartesian approach and attached-object lift use different scene
        # policies, but neither is allowed to bypass robot-mounted fixtures.
        path = np.vstack((EDGE_START, EDGE_END))
        assert not self.approach_path_valid(
            path,
            None,
            required_width_m=0.04,
        )
        assert not self.lift_segment_valid(
            EDGE_START,
            EDGE_END,
            EDGE_START,
            required_width_m=0.04,
        )
        return "fixture-phase-contract-passed"


def test_fixed_fixture_model_keeps_body_lidar_camera_and_plate_geometry():
    planner = _planner()
    supplemental = {
        capsule.name
        for capsule in planner.fixed_fixture_guard.capsules
        if capsule.supplemental
    }

    assert {
        "platform_head",
        "mid360",
        "d435_body",
        "gripper_camera_plate",
    }.issubset(supplemental)
    assert any("mid360" in pair for pair in planner.fixed_fixture_guard.pairs)
    assert any("platform_head" in pair for pair in planner.fixed_fixture_guard.pairs)


def test_fixed_fixture_path_checks_interiors_not_only_safe_endpoints():
    planner = _planner()

    assert planner._fixed_fixture_state_valid(EDGE_START)
    assert planner._fixed_fixture_state_valid(EDGE_END)
    assert not planner._fixed_fixture_state_valid(EDGE_COLLISION)
    assert not planner._fixed_fixture_path_valid(
        np.vstack((EDGE_START, EDGE_END)),
    )

    witness = planner.fixed_fixture_guard.check_state(EDGE_COLLISION).witness
    assert "mid360" in witness.pair
    assert witness.margin_m < 0.0


def test_online_plan_applies_fixed_fixture_guard_to_every_motion_phase(
    monkeypatch,
):
    planner = _planner()
    monkeypatch.setattr(
        planning_module,
        "PointCloudCollisionChecker",
        _AlwaysClearSceneChecker,
    )
    monkeypatch.setattr(
        planning_module,
        "check_target_contact_approach",
        lambda *_args, **_kwargs: SimpleNamespace(valid=True),
    )
    monkeypatch.setattr(
        planning_module,
        "GraspPlanGenerator",
        _FixturePhaseProbe,
    )

    far_scene = np.column_stack((
        np.linspace(10.0, 11.0, 64),
        np.full(64, 10.0),
        np.full(64, 10.0),
    ))
    result = planner._plan(
        object(),
        scene_points=far_scene,
        target_points=np.asarray(((12.0, 12.0, 12.0),)),
        current_joints=EDGE_START,
        stamp_s=10.0,
        pose_ranker=lambda *_args, **_kwargs: 0.0,
    )

    assert result == "fixture-phase-contract-passed"
