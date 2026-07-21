from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts" / "runtime" / "piper_calibration_ui.py"
LAUNCHER = ROOT / "scripts" / "runtime" / "piper_calibration_ui.sh"
HTML = ROOT / "web" / "calibration_dashboard" / "index.html"
SPEC = importlib.util.spec_from_file_location("piper_calibration_ui", SERVER)
assert SPEC is not None and SPEC.loader is not None
UI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UI
SPEC.loader.exec_module(UI)


def test_calibration_workbench_has_no_robot_publish_or_direct_can_transport():
    source = SERVER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {"can", "socketcan", "piper_sdk", "pyAgxArm"}
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
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert imports.isdisjoint(forbidden_imports)
    assert "create_publisher" not in attributes
    assert "publish" not in attributes
    assert "/piper/joint_trajectory" not in source
    assert "/local_movement_cmd_vel" not in source
    assert "z-manip-piper-passive-can-gate\", \"can0\", \"8" in source


def test_calibration_workbench_is_loopback_and_has_only_fixed_post_actions():
    source = SERVER.read_text(encoding="utf-8")

    assert 'LOOPBACK = "127.0.0.1"' in source
    assert 'path.path == "/api/capture"' in source
    assert 'path.path == "/api/solve"' in source
    assert 'path.path == "/api/reset"' in source
    assert "arbitrary command API" in source
    assert "shell=True" not in source
    assert "sys.executable" in source


def test_calibration_dashboard_contains_live_stream_quality_and_actions():
    html = HTML.read_text(encoding="utf-8")

    assert 'src="/api/stream.mjpg"' in html
    assert 'id="capture"' in html
    assert 'id="solve"' in html
    assert 'id="reset"' in html
    assert 'id="reset-dialog"' in html
    assert 'id="operation-message"' in html
    assert "planning_limit_violations" in html
    assert "规划限位提示" in html
    assert "确认清空并重新开始" in html
    assert 'id="coverage"' in html
    assert "重投影误差" in html
    assert "零运动命令" in html
    assert "status.capture_only" in html
    assert "安装外参模式 · 仅采样" in html


def test_capture_only_mode_disables_hand_eye_solver_endpoint():
    source = SERVER.read_text(encoding="utf-8")
    assert 'if self.args.capture_only:' in source
    assert 'capture-only mode does not run the hand-eye solver' in source
    assert '"capture_only": bool(self.args.capture_only)' in source


def test_launcher_does_not_mount_devices_or_start_motion_runtime():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "--device" not in source
    assert "/dev/can" not in source
    assert "move_group" not in source
    assert "controller_manager" not in source
    assert "ROS_DOMAIN_ID=20" in source


def test_reset_archives_session_without_deleting_artifacts(tmp_path):
    dataset = tmp_path / "hand_eye_samples.json"
    calibration = tmp_path / "piper_wrist_camera_calibration.json"
    sample_dir = tmp_path / "sample-01-123"
    dataset.write_text('{"samples":[]}\n', encoding="utf-8")
    calibration.write_text('{"calibrated":false}\n', encoding="utf-8")
    sample_dir.mkdir()
    (sample_dir / "camera_sample.json").write_text("{}\n", encoding="utf-8")

    archive = UI.archive_calibration_session(dataset, calibration)

    assert archive is not None
    assert not dataset.exists()
    assert not calibration.exists()
    assert not sample_dir.exists()
    assert (archive / dataset.name).is_file()
    assert (archive / calibration.name).is_file()
    assert (archive / sample_dir.name / "camera_sample.json").is_file()
    manifest = json.loads((archive / "reset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["recoverable"] is True
    assert len(manifest["moved"]) == 3
