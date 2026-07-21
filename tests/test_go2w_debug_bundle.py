from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_debug_bundle.py"
SPEC = importlib.util.spec_from_file_location("go2w_debug_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


def _perception(directory: Path, *, candidate_count: int = 2) -> None:
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps({
        "read_only": True,
        "request_id": "test-request",
        "instruction": "pick object",
        "frame": "camera_color_optical_frame",
        "stamp_ns": 123,
        "grasp_backend": "antipodal",
        "grasp_generation_valid": True,
        "grasp_candidates": candidate_count,
        "raw_grasp_hypotheses": 16,
    }))
    for name in ("edgetam_mask.png", "edgetam_overlay.png", "grasp_candidates_overlay.png"):
        (directory / name).write_bytes(b"fake-png")
    target = np.column_stack((np.linspace(0.0, 0.1, 2_000), np.zeros(2_000), np.ones(2_000)))
    scene = np.column_stack((np.linspace(-0.2, 0.2, 1_800), np.ones(1_800), np.ones(1_800)))
    np.save(directory / "target_points.npy", target.astype(np.float32))
    np.save(directory / "scene_collision_points.npy", scene.astype(np.float32))
    grasps = np.repeat(np.eye(4)[None, :, :], candidate_count, axis=0)
    grasps[:, 0, 3] = np.arange(candidate_count) * 0.1
    np.savez_compressed(
        directory / "grasp_candidates.npz",
        grasps=grasps,
        scores=np.linspace(0.9, 0.8, candidate_count),
        widths=np.full(candidate_count, 0.03),
        centroid=np.zeros(3),
        frame=np.asarray("camera_color_optical_frame"),
        num_raw=np.asarray(16),
        stamp_ns=np.asarray(123),
    )


def _joint(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": True,
        "interface_tx_packet_delta": 0,
        "joint_positions_rad": [0.0] * 6,
        "joint_ranges_rad": [0.0] * 6,
        "observation_start_unix_ns": 100,
        "observation_end_unix_ns": 200,
    }))


def _calibration(path: Path, *, synthetic: bool = False) -> None:
    path.write_text(json.dumps({
        "schema": "z_manip.piper_camera_calibration.v1",
        "calibrated": True,
        "synthetic": synthetic,
        "calibration_id": "measured-test",
        "mount_type": "eye_in_hand",
        "camera_frame": "camera_color_optical_frame",
        "sample_count": 10,
        "quality": {
            "rotation_axis_rank": 3,
            "max_pair_rotation_rad": 0.8,
            "translation_rmse_m": 0.002,
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


def _planning(directory: Path, *, motion_commands: int = 0) -> None:
    directory.mkdir()
    rejections = [
        {"candidate_index": 0, "symmetry_index": 1, "stage": "ik", "reason": "no solution"},
        {"candidate_index": 1, "symmetry_index": 0, "stage": "approach_collision", "reason": "collision"},
    ]
    (directory / "planning_report.json").write_text(json.dumps({
        "read_only": True,
        "planning_only": True,
        "motion_commands_published": motion_commands,
        "source_frame": "camera_color_optical_frame",
        "planning_frame": "piper_base_link",
        "base_from_camera": np.eye(4).tolist(),
        "plan_valid": True,
        "candidate_index": 0,
        "symmetry_index": 0,
        "selected_global_rank": 1,
        "higher_rank_rejection_count": 0,
        "score": 0.7,
        "required_width_m": 0.03,
        "grasp_pose": np.eye(4).tolist(),
        "pregrasp_pose": np.eye(4).tolist(),
        "transit_waypoints": 2,
        "approach_waypoints": 3,
        "lift_waypoints": 2,
        "rejection_count": len(rejections),
        "rejections": rejections,
    }))
    np.savez_compressed(
        directory / "planned_grasp.npz",
        grasp_pose=np.eye(4),
        pregrasp_pose=np.eye(4),
        transit=np.zeros((2, 6)),
        transit_times_s=np.array([0.0, 0.2]),
        approach=np.ones((3, 6)) * 0.1,
        approach_times_s=np.array([0.0, 0.1, 0.2]),
        lift=np.ones((2, 6)) * 0.2,
        lift_times_s=np.array([0.0, 0.3]),
        current_joints=np.zeros(6),
    )


def _session_gate(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "z_manip.piper_planning_session_gate.v1",
        "planning_ready": True,
        "read_only": True,
        "planning_only": True,
        "source_frame": "camera_color_optical_frame",
        "base_from_camera": np.eye(4).tolist(),
    }))


def _urdf(path: Path) -> None:
    links = ['<link name="piper_base_link"/>']
    joints = []
    parent = "piper_base_link"
    for index in range(1, 7):
        child = f"piper_link{index}"
        links.append(f'<link name="{child}"/>')
        joints.append(
            f'<joint name="piper_joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            '<origin xyz="0.08 0 0.02" rpy="0 0 0"/>'
            '<axis xyz="0 0 1"/><limit lower="-3" upper="3" velocity="2" effort="1"/>'
            '</joint>'
        )
        parent = child
    links.append('<link name="piper_gripper_base"/>')
    joints.append(
        '<joint name="tip_fixed" type="fixed">'
        f'<parent link="{parent}"/><child link="piper_gripper_base"/>'
        '<origin xyz="0.05 0 0" rpy="0 0 0"/>'
        '</joint>'
    )
    path.write_text('<robot name="test">' + ''.join(links + joints) + '</robot>')


def test_missing_joint_calibration_and_plan_are_visible_and_fail_closed(tmp_path):
    perception = tmp_path / "perception"
    _perception(perception)
    output = perception / "debug_bundle.json"

    bundle = BUNDLE.build_bundle(perception, output)
    stages = {stage["name"]: stage for stage in bundle["stages"]}

    assert bundle["schema"] == "z_manip.debug_bundle.v1"
    assert bundle["status"]["ok"] is False
    assert stages["joint_state_gate"]["status"] == "blocked"
    assert stages["calibration_gate"]["error"]["code"] == "MISSING_CALIBRATION"
    assert stages["motion_plan"]["error"]["code"] == "MISSING_PLANNING_REPORT"
    assert bundle["safety"]["motion_commands_published"] == 0
    assert len(bundle["visualization"]["target_cloud"]["points_xyz_m"]) == 1_500
    assert bundle["visualization"]["images"]["candidate_overlay"] == "grasp_candidates_overlay.png"
    assert [item["candidate_id"] for item in bundle["candidates"]] == [0, 1]
    assert bundle["visualization"]["robot_overlay_allowed"] is False


def test_verified_session_gate_keeps_transform_valid_when_planner_crashes(tmp_path):
    perception = tmp_path / "perception"
    joint = tmp_path / "joint.json"
    calibration = tmp_path / "calibration.json"
    gate = tmp_path / "session_gate.json"
    urdf = tmp_path / "robot.urdf"
    _perception(perception)
    _joint(joint)
    _calibration(calibration)
    _session_gate(gate)
    _urdf(urdf)

    bundle = BUNDLE.build_bundle(
        perception,
        tmp_path / "debug_bundle.json",
        session_gate=gate,
        joint_report=joint,
        calibration=calibration,
        urdf=urdf,
    )

    stages = {stage["name"]: stage for stage in bundle["stages"]}
    assert stages["frame_transform"]["status"] == "ok"
    assert stages["frame_transform"]["metrics"]["planning_frame"] == "piper_base_link"
    assert stages["ik"]["error"]["code"] == "MISSING_PLANNING_REPORT"
    assert bundle["visualization"]["frame"] == "piper_base_link"


def test_complete_bundle_preserves_rejections_and_trajectory(tmp_path):
    perception = tmp_path / "perception"
    planning = tmp_path / "planning"
    joint = tmp_path / "joint.json"
    calibration = tmp_path / "calibration.json"
    urdf = tmp_path / "robot.urdf"
    _perception(perception)
    _planning(planning)
    _joint(joint)
    _calibration(calibration)
    _urdf(urdf)

    bundle = BUNDLE.build_bundle(
        perception,
        tmp_path / "debug_bundle.json",
        planning_dir=planning,
        joint_report=joint,
        calibration=calibration,
        urdf=urdf,
    )

    assert bundle["status"]["ok"] is True
    assert bundle["planning"]["included_rejection_count"] == 2
    assert bundle["planning"]["selected_global_rank"] == 1
    assert bundle["planning"]["higher_rank_rejection_count"] == 0
    assert bundle["planning"]["rejections"][1]["stage"] == "approach_collision"
    assert bundle["candidates"][0]["status"] == "selected"
    assert bundle["candidates"][1]["status"] == "rejected"
    assert bundle["selected_plan"]["segments"]["approach"]["source_waypoint_count"] == 3
    assert bundle["selected_plan"]["segments"]["approach"]["duration_s"] == 0.2
    assert bundle["selected_plan"]["segments"]["lift"]["positions_rad"][-1] == [0.2] * 6
    assert bundle["selected_plan"]["selected_global_rank"] == 1
    assert bundle["visualization"]["robot_overlay_allowed"] is True
    assert len(bundle["visualization"]["robot_overlay"]["links_xyz_m"]) == 7
    assert len(bundle["visualization"]["trajectory_xyz_m"]["approach"]) == 3
    assert len(bundle["visualization"]["kinematic_model"]["urdf_sha256"]) == 64
    assert bundle["visualization"]["frame"] == "piper_base_link"
    assert bundle["visualization"]["target_cloud"]["frame"] == "piper_base_link"
    assert bundle["visualization"]["scene_cloud"]["frame"] == "piper_base_link"
    assert {item["frame"] for item in bundle["visualization"]["candidate_axes"]} == {"piper_base_link"}
    assert {item["name"] for item in bundle["visualization"]["reference_axes"]} == {"base", "camera"}


def test_failed_plan_still_has_base_frame_clouds_and_current_robot(tmp_path):
    perception = tmp_path / "perception"
    planning = tmp_path / "planning"
    joint = tmp_path / "joint.json"
    calibration = tmp_path / "calibration.json"
    urdf = tmp_path / "robot.urdf"
    _perception(perception)
    _planning(planning)
    report_path = planning / "planning_report.json"
    report = json.loads(report_path.read_text())
    report["plan_valid"] = False
    report["error"] = "no candidate survived"
    report["measured_joints_rad"] = [0.0] * 6
    report_path.write_text(json.dumps(report))
    (planning / "planned_grasp.npz").unlink()
    _joint(joint)
    _calibration(calibration)
    _urdf(urdf)

    bundle = BUNDLE.build_bundle(
        perception,
        tmp_path / "debug_bundle.json",
        planning_dir=planning,
        joint_report=joint,
        calibration=calibration,
        urdf=urdf,
    )

    assert bundle["selected_plan"] is None
    assert bundle["planning"]["selection_status"] == "no_feasible_candidate"
    assert all(candidate["status"] != "selected" for candidate in bundle["candidates"])
    stages = {stage["name"]: stage for stage in bundle["stages"]}
    assert stages["ik"]["status"] == "ok"
    assert stages["ik"]["metrics"] == {
        "passed_hypotheses": 1,
        "rejected_hypotheses": 1,
    }
    assert stages["collision_check"]["status"] == "failed"
    assert stages["collision_check"]["metrics"] == {
        "passed_hypotheses": 0,
        "rejected_hypotheses": 1,
    }
    assert stages["motion_plan"]["status"] == "blocked"
    assert stages["motion_plan"]["metrics"]["attempted_hypotheses"] == 0
    assert bundle["visualization"]["frame"] == "piper_base_link"
    assert bundle["visualization"]["robot_overlay"]["pose_source"] == "measured_passive_feedback"
    assert len(bundle["visualization"]["robot_overlay"]["links_xyz_m"]) == 7


def test_failed_rrt_is_distinguished_from_upstream_candidate_rejections(tmp_path):
    perception = tmp_path / "perception"
    planning = tmp_path / "planning"
    joint = tmp_path / "joint.json"
    calibration = tmp_path / "calibration.json"
    _perception(perception)
    _planning(planning)
    report_path = planning / "planning_report.json"
    report = json.loads(report_path.read_text())
    report["plan_valid"] = False
    report["error"] = "RRT budget exhausted"
    report["rejections"] = [
        {
            "candidate_index": 0,
            "symmetry_index": 0,
            "stage": "planning",
            "reason": "RRT budget exhausted",
        },
    ]
    report["rejection_count"] = 1
    report_path.write_text(json.dumps(report))
    (planning / "planned_grasp.npz").unlink()
    _joint(joint)
    _calibration(calibration)

    bundle = BUNDLE.build_bundle(
        perception,
        tmp_path / "debug_bundle.json",
        planning_dir=planning,
        joint_report=joint,
        calibration=calibration,
    )
    stages = {stage["name"]: stage for stage in bundle["stages"]}

    assert stages["ik"]["status"] == "ok"
    assert stages["collision_check"]["status"] == "ok"
    assert stages["motion_plan"]["status"] == "failed"
    assert stages["motion_plan"]["metrics"] == {
        "attempted_hypotheses": 1,
        "failed_hypotheses": 1,
    }


def test_stale_joint_report_is_visible_and_blocks_base_overlay(tmp_path):
    perception = tmp_path / "perception"
    joint = tmp_path / "joint.json"
    _perception(perception)
    _joint(joint)
    document = json.loads(joint.read_text())
    document["observation_start_unix_ns"] = 1_000_000_000
    document["observation_end_unix_ns"] = 2_000_000_000
    joint.write_text(json.dumps(document))

    bundle = BUNDLE.build_bundle(
        perception,
        tmp_path / "debug_bundle.json",
        joint_report=joint,
    )
    stages = {stage["name"]: stage for stage in bundle["stages"]}

    assert stages["joint_state_gate"]["status"] == "failed"
    assert stages["joint_state_gate"]["error"]["code"] == "STALE_JOINT_REPORT"
    assert bundle["visualization"]["robot_overlay_allowed"] is False


def test_synthetic_calibration_and_reported_motion_fail_closed(tmp_path):
    perception = tmp_path / "perception"
    planning = tmp_path / "planning"
    joint = tmp_path / "joint.json"
    calibration = tmp_path / "calibration.json"
    _perception(perception)
    _planning(planning, motion_commands=1)
    _joint(joint)
    _calibration(calibration, synthetic=True)

    bundle = BUNDLE.build_bundle(
        perception,
        tmp_path / "debug_bundle.json",
        planning_dir=planning,
        joint_report=joint,
        calibration=calibration,
    )
    stages = {stage["name"]: stage for stage in bundle["stages"]}

    assert bundle["status"]["ok"] is False
    assert stages["calibration_gate"]["status"] == "failed"
    assert stages["ik"]["error"]["code"] == "UPSTREAM_GATE_NOT_VERIFIED"
    assert stages["motion_plan"]["status"] == "blocked"
    assert stages["safety_gate"]["error"]["code"] == "UPSTREAM_MOTION_REPORTED"
    assert bundle["safety"]["motion_commands_published"] == 0


def test_source_has_no_ros_can_or_transport_calls():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden_imports = {"can", "socket", "rclpy", "piper_sdk", "pyAgxArm"}
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
    transport_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"create_publisher", "publish", "send", "sendall", "sendmsg", "sendto"}
    }

    assert imports.isdisjoint(forbidden_imports)
    assert transport_calls == set()
