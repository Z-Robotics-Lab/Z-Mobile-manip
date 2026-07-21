from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts/runtime/piper_joint_zero_ui.py"
LAUNCHER = ROOT / "scripts/runtime/piper_joint_zero_ui.sh"
PAGE = ROOT / "web/joint_zero_dashboard/index.html"
SPEC = importlib.util.spec_from_file_location("piper_joint_zero_ui", SERVER)
assert SPEC is not None and SPEC.loader is not None
UI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UI)


def _report(*, ready: bool = True) -> dict[str, object]:
    return {
        "schema": UI.SCHEMA,
        "ready_for_manual_review": ready,
        "read_only": True,
        "motion_commands_published": 0,
        "urdf_modified": False,
        "calibration_id": "piper-joint-zero-test",
        "joint_names": [f"joint{index}" for index in range(1, 7)],
        "joint_zero_offsets_rad": [0.01, -0.02, 0.005, 0.0, -0.003, 0.007],
        "optimizer": {
            "success": ready,
            "message": "converged" if ready else "did not converge",
            "max_abs_offset_rad": 0.052,
            "evaluations": 8,
        },
        "observability": {
            "rank": 6 if ready else 5,
            "required_rank": 6,
            "condition": 42.0,
            "max_condition": 10000.0,
            "singular_values": [120.0, 95.0, 71.0, 55.0, 22.0, 9.0],
            "passes": ready,
            "joint_excitation_rank": 6,
            "joint_excitation_required_rank": 6,
            "linearized_standard_deviation_rad": [0.001] * 6,
        },
        "provenance": {
            "independent_dataset_verified": True,
            "read_only_capture_evidence_verified": True,
            "dataset_sha256": "kinematic-dataset",
            "hand_eye_dataset_sha256": "hand-eye-dataset",
            "mount_dataset_sha256": "mount-dataset",
        },
        "quality": {
            "nominal_training": {
                "translation_rmse_m": 0.02,
                "rotation_rmse_rad": 0.04,
            },
            "calibrated_training": {
                "translation_rmse_m": 0.003,
                "rotation_rmse_rad": 0.01,
            },
            "nominal_validation": {
                "translation_rmse_m": 0.021,
                "rotation_rmse_rad": 0.045,
            },
            "calibrated_validation": {
                "translation_rmse_m": 0.004,
                "rotation_rmse_rad": 0.012,
            },
            "validation_passes": ready,
            "offsets_inside_review_bounds": True,
        },
        "quality_limits": {
            "max_validation_translation_rmse_m": 0.01,
            "max_validation_rotation_rmse_rad": 0.035,
        },
        "sample_count": 24,
        "training_indices": list(range(18)),
        "validation_indices": list(range(18, 24)),
        "caveat": "provisional joint-zero fit",
    }


def _write(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "joint_zero_report.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_load_report_accepts_success_and_minimal_solver_failure(tmp_path):
    success = _write(tmp_path, _report())
    assert UI.load_report(success)["ready_for_manual_review"] is True

    failure = _report(ready=False)
    failure = {
        key: failure[key]
        for key in (
            "schema",
            "ready_for_manual_review",
            "read_only",
            "motion_commands_published",
            "urdf_modified",
        )
    }
    failure["error"] = "ValueError: input frame contract does not match"
    failed_path = _write(tmp_path, failure)
    assert "frame contract" in UI.load_report(failed_path)["error"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("read_only", False),
        ("motion_commands_published", 1),
        ("urdf_modified", True),
    ),
)
def test_load_report_rejects_unsafe_provenance(tmp_path, field, value):
    document = _report()
    document[field] = value
    path = _write(tmp_path, document)
    with pytest.raises(UI.ReportError, match="provenance"):
        UI.load_report(path)


def test_server_is_loopback_only_and_read_only(tmp_path):
    path = _write(tmp_path, _report(ready=False))
    server = UI.create_server(path, port=0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/api/report", timeout=2) as response:
            report = json.load(response)
            assert report["motion_commands_published"] == 0
            assert response.headers["Cache-Control"] == "no-store"
            assert "connect-src 'self'" in response.headers["Content-Security-Policy"]
            assert response.headers.get("Access-Control-Allow-Origin") is None
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(base + "/api/report", method="POST"), timeout=2)
        assert caught.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_server_has_no_robot_transport_or_subprocess_integration():
    source = SERVER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports = {
        "can",
        "rclpy",
        "socket",
        "subprocess",
        "piper_sdk",
        "pyAgxArm",
    }
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
    assert attributes.isdisjoint({
        "create_publisher",
        "publish",
        "send",
        "sendto",
        "Popen",
        "run",
        "system",
    })
    assert 'LOOPBACK = "127.0.0.1"' in source
    assert "0.0.0.0" not in source


def test_page_contains_offsets_quality_observability_safety_and_failures():
    source = PAGE.read_text(encoding="utf-8")
    lowered = source.lower()
    for required in (
        "joint_zero_offsets_rad",
        "nominal_training",
        "calibrated_training",
        "nominal_validation",
        "calibrated_validation",
        "singular_values",
        "joint_excitation_rank",
        "independent_dataset_verified",
        "read_only_capture_evidence_verified",
        "motion_commands_published",
        "urdf_modified",
        "ready_for_manual_review",
        "失败原因",
    ):
        assert required in source
    assert "fetch('/api/report'" in source
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "innerhtml" not in lowered


def test_launcher_only_starts_the_local_report_viewer():
    source = LAUNCHER.read_text(encoding="utf-8").lower()
    assert "8770" in source
    assert "piper_joint_zero_ui.py" in source
    for forbidden in (
        "sudo ",
        "ssh ",
        "ros2 ",
        "docker ",
        "cansend",
        "candump",
        "/cmd_vel",
        "/joint_trajectory",
        "piper_joint_zero_calibrate.py",
    ):
        assert forbidden not in source
