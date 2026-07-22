from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/offline/planning_replay_benchmark.py"
SPEC = importlib.util.spec_from_file_location("planning_replay_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def _options(tmp_path, **overrides):
    values = {
        "calibration": tmp_path / "calibration.json",
        "urdf": tmp_path / "robot.urdf",
        "config": tmp_path / "config.json",
        "robot_assets": tmp_path / "assets",
        "runtime_image": "z-manip-runtime:pinocchio",
        "trial_timeout_s": 10.0,
        "search_timeout_s": 6.0,
        "symmetry_samples": 4,
        "max_hypotheses": 64,
        "max_feasible_plans": 1,
        "support_approach_prior_weight": 0.05,
        "scene_clearance_m": 0.001,
        "scene_point_radius_m": 0.001,
        "gripper_scene_radius_scale": 0.6,
        "handoff_reach_m": 0.7,
        "max_trials": None,
        "baseline": None,
        "min_success_rate": 0.5,
        "max_p95_s": 3.0,
        "max_ik_rejection_fraction": 0.5,
        "max_success_rate_drop": 0.0,
        "max_p95_regression_ratio": 1.1,
    }
    values.update(overrides)
    for key in ("calibration", "urdf", "config"):
        values[key].write_text("{}", encoding="utf-8")
    values["robot_assets"].mkdir()
    return argparse.Namespace(**values)


def _bag(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  duration:\n    nanoseconds: 20000000000\n"
        "  starting_time:\n    nanoseconds_since_epoch: 100000000000\n"
        "  message_count: 42\n",
        encoding="utf-8",
    )
    return bag


def _perception(sessions, session_id="p1"):
    root = sessions / "perception" / session_id
    artifacts = root / "perception"
    artifacts.mkdir(parents=True)
    (root / "attempt.json").write_text(json.dumps({
        "session_id": session_id,
        "started_at": "1970-01-01T00:01:41Z",
        "status": "succeeded",
    }), encoding="utf-8")
    for name in BENCH.REQUIRED_PERCEPTION_FILES:
        (artifacts / name).write_bytes(b"x")
    return artifacts


def test_planner_command_is_network_disabled_and_has_no_device_mount(tmp_path):
    options = _options(tmp_path)
    artifacts = tmp_path / "perception"
    output = tmp_path / "output"
    artifacts.mkdir()
    output.mkdir()
    command = BENCH.planner_command(
        artifacts, output,
        {"measured_joints_rad": [0.0] * 6, "planning_start_joints_rad": [0.0] * 6},
        image=options.runtime_image,
        config=options.config,
        calibration=options.calibration,
        robot_assets=options.robot_assets,
        search_timeout_s=6.0,
        symmetry_samples=4,
        max_hypotheses=64,
        max_feasible_plans=1,
        support_approach_prior_weight=0.05,
        scene_clearance_m=0.001,
        scene_point_radius_m=0.001,
        gripper_scene_radius_scale=0.6,
    )

    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "--device" not in command
    assert all("/dev/" not in item for item in command)
    assert f"{artifacts}:/session/perception:ro" in command


def test_batch_replays_only_sessions_in_bag_window_and_reports_thresholds(tmp_path):
    bag = _bag(tmp_path)
    sessions = tmp_path / "sessions"
    _perception(sessions)
    output = tmp_path / "output"
    output.mkdir()
    options = _options(tmp_path)

    def fake_runner(argv, _timeout_s, _log_path):
        destination = Path(argv[argv.index("--output") + 1])
        if str(BENCH.SESSION_GATE) in argv:
            destination.write_text(json.dumps({
                "planning_ready": True,
                "candidate_count": 2,
                "measured_joints_rad": [0.0] * 6,
                "planning_start_joints_rad": [0.0] * 6,
            }), encoding="utf-8")
            return 0, 0.1, False
        host_output = output / "p1" / "planning"
        (host_output / "planning_report.json").write_text(json.dumps({
            "plan_valid": True,
            "base_from_camera": np.eye(4).tolist(),
            "rejection_count": 1,
            "rejections": [{"stage": "approach_collision"}],
            "timings_s": {"search": 1.2, "total": 1.4},
        }), encoding="utf-8")
        np.save(sessions / "perception" / "p1" / "perception" / "target_points.npy", [[0.4, 0.0, 0.0]])
        return 0, 1.5, False

    report = BENCH.build_report(
        bag=bag, sessions_root=sessions, output_root=output,
        options=options, runner=fake_runner,
    )

    assert report["offline"] is True
    assert report["transport_opened"] is False
    assert report["selected_sessions"] == ["p1"]
    assert report["summary"]["success_rate"] == 1.0
    assert report["summary"]["planner_wall_s"]["p95"] == 1.5
    assert report["summary"]["rejection_stages"] == {"approach_collision": 1}
    assert report["summary"]["handoff"]["success_rate"] == 1.0
    assert report["trials"][0]["target_base_range_m"] == 0.4
    assert report["thresholds"]["passed"] is True


def test_handoff_summary_separates_far_perception_from_close_planning():
    summary = BENCH.summarize_trials([
        {
            "status": "failed", "handoff_eligible": False,
            "planner_wall_s": 6.0, "total_wall_s": 6.1,
            "planner_timings_s": {"search": 6.0},
            "rejection_stages": {"ik": 64},
        },
        {
            "status": "succeeded", "handoff_eligible": True,
            "planner_wall_s": 2.2, "total_wall_s": 2.3,
            "planner_timings_s": {"search": 1.1},
            "rejection_stages": {"ik": 4},
        },
    ])

    assert summary["success_rate"] == 0.5
    assert summary["handoff"]["eligible_trials"] == 1
    assert summary["handoff"]["needs_base_approach_trials"] == 1
    assert summary["handoff"]["success_rate"] == 1.0
    assert summary["handoff"]["planner_wall_s"]["p95"] == 2.2


def test_gate_blocked_far_target_is_reported_as_needing_base_approach(tmp_path):
    artifacts = tmp_path / "perception"
    artifacts.mkdir()
    np.save(artifacts / "target_points.npy", [[1.1, 0.0, 0.0]])
    output = tmp_path / "output"
    output.mkdir()
    options = _options(tmp_path)

    def fake_runner(argv, _timeout_s, _log_path):
        destination = Path(argv[argv.index("--output") + 1])
        destination.write_text(json.dumps({
            "planning_ready": False,
            "base_from_camera": np.eye(4).tolist(),
            "handoff_workspace": {
                "target_range_m": 1.1,
                "planning_allowed": False,
                "state": "NEED_BASE_APPROACH",
            },
            "errors": [{"code": "NEED_BASE_APPROACH"}],
        }), encoding="utf-8")
        return 1, 0.05, False

    trial = BENCH.run_trial(
        {"session_id": "far", "artifacts": str(artifacts)},
        output,
        options,
        runner=fake_runner,
    )

    assert trial["status"] == "needs_base_approach"
    assert trial["target_base_range_m"] == 1.1
    assert trial["handoff_eligible"] is False


def test_thresholds_compare_against_baseline():
    current = {
        "success_rate": 0.7,
        "planner_wall_s": {"p95": 3.1},
        "ik_rejection_fraction": 0.4,
    }
    baseline = {"summary": {"success_rate": 0.8, "planner_wall_s": {"p95": 2.0}}}

    result = BENCH.evaluate_thresholds(
        current,
        min_success_rate=None,
        max_p95_s=None,
        max_ik_rejection_fraction=None,
        baseline=baseline,
        max_success_rate_drop=0.05,
        max_p95_regression_ratio=1.2,
    )

    assert result["passed"] is False
    assert [item["passed"] for item in result["checks"]] == [False, False]
