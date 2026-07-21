from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from z_manip.kinematics.chain import KinematicChain


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_hand_eye_sample.py"
URDF = ROOT.parent / "go2W_Sim" / "assets" / "urdf" / "go2w_sensored.urdf"
SPEC = importlib.util.spec_from_file_location("piper_hand_eye_sample", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAMPLE)


@pytest.fixture(scope="module")
def chain():
    if not URDF.exists():
        pytest.skip(f"PiPER URDF unavailable: {URDF}")
    return KinematicChain.from_urdf(URDF, "piper_base_link", "piper_gripper_base")


def reports(chain, *, moving=False):
    q = 0.5 * (chain.lower_limits + chain.upper_limits)
    start = 1_700_000_000_000_000_000
    ranges = np.full(chain.dof, 0.003 if moving else 0.0002)
    camera = {
        "schema": "z_manip.charuco_camera_sample.v1",
        "read_only": True,
        "valid": True,
        "source_stamp_ns": start + 1_000_000_000,
        "camera_frame": "camera_color_optical_frame",
        "target_frame": "charuco_board",
        "camera_from_target": np.eye(4).tolist(),
    }
    joint = {
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": True,
        "interface_tx_packet_delta": 0,
        "observation_start_unix_ns": start,
        "observation_end_unix_ns": start + 2_000_000_000,
        "joint_snapshot_span_s": 0.005,
        "joint_positions_rad": q.tolist(),
        "joint_ranges_rad": ranges.tolist(),
        "max_joint_range_rad": float(ranges.max()),
    }
    return camera, joint, q


def test_assemble_and_append_read_only_sample(chain, tmp_path):
    camera, joint, q = reports(chain)
    sample = SAMPLE.assemble_sample(camera, joint, chain)
    dataset_path = tmp_path / "samples.json"
    dataset = SAMPLE.append_sample(
        dataset_path,
        sample,
        base_frame=chain.base_link,
        tip_link=chain.tip_link,
        camera_frame="camera_color_optical_frame",
        target_frame="charuco_board",
    )

    assert dataset["synthetic"] is False
    assert len(dataset["samples"]) == 1
    np.testing.assert_allclose(sample["base_from_tip"], chain.forward(q))
    assert sample["safety_evidence"]["can_tx_packet_delta"] == 0
    with pytest.raises(ValueError, match="already present"):
        SAMPLE.append_sample(
            dataset_path,
            sample,
            base_frame=chain.base_link,
            tip_link=chain.tip_link,
            camera_frame="camera_color_optical_frame",
            target_frame="charuco_board",
        )


def test_rejects_motion_and_nonoverlapping_camera_stamp(chain):
    camera, joint, _q = reports(chain, moving=True)
    with pytest.raises(ValueError, match="arm moved"):
        SAMPLE.assemble_sample(camera, joint, chain)

    camera, joint, _q = reports(chain)
    camera["source_stamp_ns"] = joint["observation_end_unix_ns"] + 1_000_000_000
    with pytest.raises(ValueError, match="does not overlap"):
        SAMPLE.assemble_sample(camera, joint, chain)


def test_manual_drag_feedback_outside_planning_limits_is_warning_only_for_calibration(chain):
    camera, joint, _q = reports(chain)
    joint["joint_positions_rad"][5] = float(chain.lower_limits[5] - np.deg2rad(42.0))

    sample = SAMPLE.assemble_sample(camera, joint, chain)
    safety = sample["safety_evidence"]
    assert safety["max_abs_joint_feedback_rad"] == pytest.approx(np.deg2rad(190.0))
    assert safety["joint_limit_policy"] == "warning_only_for_passive_calibration"
    assert safety["planning_limits_enforced_for_automatic_motion"] is True
    assert len(safety["planning_limit_violations"]) == 1
    violation = safety["planning_limit_violations"][0]
    assert violation["joint_index"] == 6
    assert violation["excess_deg"] == pytest.approx(42.0)

    joint["joint_positions_rad"][5] = np.deg2rad(-195.0)
    with pytest.raises(ValueError, match=r"plausibility envelope.*J6=-195\.000deg"):
        SAMPLE.assemble_sample(camera, joint, chain)


def test_source_has_no_ros_can_or_transport_send():
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
        and node.func.attr in {"create_publisher", "publish", "send", "sendto"}
    }

    assert imports.isdisjoint(forbidden_imports)
    assert transport_calls == set()
