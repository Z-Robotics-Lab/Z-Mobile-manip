from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_planning_session_gate.py"
LAUNCHER = ROOT / "scripts" / "runtime" / "go2w_planning_session.sh"
URDF = ROOT.parent / "go2W_Sim" / "assets" / "urdf" / "go2w_sensored.urdf"
SPEC = importlib.util.spec_from_file_location("piper_planning_session_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _fixtures(tmp_path: Path):
    perception = tmp_path / "perception"
    perception.mkdir()
    stamp = 1_700_000_001_000_000_000
    (perception / "report.json").write_text(json.dumps({
        "read_only": True,
        "stamp_ns": stamp,
        "frame": "camera_color_optical_frame",
        "grasp_generation_valid": True,
        "grasp_generation_error": "",
    }))
    np.savez_compressed(
        perception / "grasp_candidates.npz",
        stamp_ns=np.asarray(stamp, dtype=np.int64),
        frame=np.asarray("camera_color_optical_frame"),
        grasps=np.eye(4)[None, :, :],
    )
    np.save(perception / "target_points.npy", np.asarray([[0.0, 0.0, 0.5]]))
    joints = tmp_path / "joints.json"
    joints.write_text(json.dumps({
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": True,
        "interface_tx_packet_delta": 0,
        "observation_start_unix_ns": stamp - 1_000_000_000,
        "observation_end_unix_ns": stamp + 1_000_000_000,
        "joint_snapshot_span_s": 0.005,
        "max_joint_range_rad": 0.0,
        "joint_positions_rad": [0.0, 0.5, -1.0, 0.0, 0.0, 0.0],
    }))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema": "z_manip.piper_camera_calibration.v1",
        "calibrated": True,
        "synthetic": False,
        "calibration_id": "real-test",
        "camera_frame": "camera_color_optical_frame",
        "tip_from_camera": np.eye(4).tolist(),
        "sample_count": 8,
        "quality": {
            "rotation_axis_rank": 3,
            "max_pair_rotation_rad": 1.0,
            "translation_rmse_m": 0.003,
            "rotation_rmse_rad": 0.01,
        },
        "quality_limits": {
            "min_samples": 8,
            "min_rotation_axis_rank": 2,
            "min_rotation_span_rad": 0.35,
            "max_translation_rmse_m": 0.01,
            "max_rotation_rmse_rad": 0.035,
        },
    }))
    return perception, joints, calibration


def test_synchronized_read_only_session_passes(tmp_path):
    perception, joints, calibration = _fixtures(tmp_path)

    report = GATE.evaluate_session(perception, joints, calibration, URDF)

    assert report["planning_ready"] is True
    assert report["motion_commands_published"] == 0
    assert report["joint_observation_overlaps"] is True
    assert report["candidate_count"] == 1
    assert report["target_centroid_base"] is not None


def test_stale_or_out_of_limit_joint_state_fails_closed(tmp_path):
    perception, joints, calibration = _fixtures(tmp_path)
    document = json.loads(joints.read_text())
    document["observation_start_unix_ns"] += 10_000_000_000
    document["observation_end_unix_ns"] += 10_000_000_000
    document["joint_positions_rad"][4] = 1.5
    joints.write_text(json.dumps(document))

    report = GATE.evaluate_session(perception, joints, calibration, URDF)

    assert report["planning_ready"] is False
    assert {error["code"] for error in report["errors"]} == {
        "STALE_JOINT_REPORT",
        "PLANNING_START_OUTSIDE_URDF",
    }


def test_small_passive_feedback_overshoot_is_projected_for_planning_only(tmp_path):
    perception, joints, calibration = _fixtures(tmp_path)
    document = json.loads(joints.read_text())
    document["joint_positions_rad"][2] = 0.0054
    joints.write_text(json.dumps(document))

    report = GATE.evaluate_session(perception, joints, calibration, URDF)

    assert report["planning_ready"] is True
    assert report["measured_joints_rad"][2] == 0.0054
    assert report["planning_start_joints_rad"][2] == 0.0
    assert report["execution_start_requires_limit_reconciliation"] is True
    assert report["warnings"][0]["code"] == (
        "PLANNING_START_PROJECTED_TO_URDF_LIMIT"
    )


def test_gate_and_launcher_have_no_actuator_transport():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden_imports = {"rclpy", "can", "socket", "subprocess", "piper_sdk", "pyAgxArm"}
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    source = LAUNCHER.read_text(encoding="utf-8")

    assert imports.isdisjoint(forbidden_imports)
    assert "--network none" in source
    assert "piper_planning_dry_run.py:/usr/local/bin/" in source
    assert (
        "configs/piper_collision_capsules.json:"
        "/opt/z_manip/configs/piper_collision_capsules.json:ro"
    ) in source
    assert "ros2/z_manip_task/z_manip_task:$TASK_PACKAGE_CONTAINER:ro" in source
    assert '--joints="$joints_csv"' in source
    assert '--planning-joints="$planning_joints_csv"' in source
    assert '--search-timeout-s "$PLANNING_ONLY_SEARCH_TIMEOUT_S"' in source
    assert '--symmetry-samples "$PLANNING_ONLY_SYMMETRY_SAMPLES"' in source
    assert '--max-hypotheses "$PLANNING_ONLY_MAX_HYPOTHESES"' in source
    assert 'REMOTE_PASSIVE_PROBE="/usr/local/libexec/z-manip/piper_passive_probe.py"' in source
    assert '--interface can0' in source
    assert 'selected_passive_joint_report.json' in source
    assert "cansend" not in source
    assert "/piper/joint_trajectory" not in source
    assert "/local_movement_cmd_vel" not in source
    assert "ros2 action" not in source
