from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_debug_safety_gate.py"
SPEC = importlib.util.spec_from_file_location("go2w_debug_safety_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _calibration(path: Path, *, valid: bool = True) -> None:
    _write_json(path, {
        "schema": "z_manip.piper_camera_calibration.v1",
        "calibrated": valid,
        "synthetic": False,
        "calibration_id": "measured-test",
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
    })


def _joint(path: Path, *, tx_delta: int = 0) -> None:
    _write_json(path, {
        "schema": "z_manip.piper_passive_joint_report.v1",
        "read_only": True,
        "complete_joint_feedback": True,
        "zero_transmit_verified": tx_delta == 0,
        "interface_tx_packet_delta": tx_delta,
        "joint_positions_rad": [0.0] * 6,
        "joint_ranges_rad": [0.0] * 6,
    })


def _reference(path: Path, bundle_parent: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": str(path.relative_to(bundle_parent)),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _bundle(
    root: Path,
    *,
    calibration_valid: bool = True,
    overlay_allowed: bool = True,
    motion_commands: int = 0,
    include_joint: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image = root / "images" / "overlay.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    calibration = root / "calibration.json"
    _calibration(calibration, valid=calibration_valid)
    artifacts = {
        "candidate_overlay": _reference(image, root),
        "camera_calibration": _reference(calibration, root),
    }
    if include_joint:
        joint = root / "joint.json"
        _joint(joint)
        artifacts["joint_report"] = _reference(joint, root)
    path = root / "bundle.json"
    _write_json(path, {
        "schema": "z_manip.debug_bundle.v1",
        "mode": {
            "read_only": True,
            "planning_only": True,
            "offline_artifact_reader": True,
        },
        "safety": {
            "motion_commands_published": motion_commands,
            "transport_opened": False,
            "ros_imported": False,
            "can_opened": False,
            "upstream_reported_motion_commands_published": 0,
        },
        "artifacts": artifacts,
        "visualization": {
            "robot_overlay_allowed": overlay_allowed,
            "images": {"candidate_overlay": "images/overlay.png"},
        },
    })
    return path


def _codes(audit: dict[str, object]) -> set[str]:
    return {str(value["code"]) for value in audit["errors"]}


def test_complete_offline_bundle_with_zero_tx_evidence_passes(tmp_path):
    bundle = _bundle(tmp_path)

    audit = GATE.audit_bundle(bundle, tmp_path)

    assert audit["passed"] is True
    assert audit["calibration_valid"] is True
    assert audit["can_zero_transmit_verified"] is True
    assert audit["robot_overlay_allowed"] is True
    assert audit["motion_commands_published"] == 0


def test_motion_report_and_nonzero_can_tx_fail_closed(tmp_path):
    bundle = _bundle(tmp_path, motion_commands=1)
    unsafe_joint = tmp_path / "unsafe-joint.json"
    _joint(unsafe_joint, tx_delta=1)

    audit = GATE.audit_bundle(bundle, tmp_path, joint_report=unsafe_joint)

    assert audit["passed"] is False
    assert "MOTION_OR_TRANSPORT_REPORTED" in _codes(audit)
    assert "INVALID_OR_NONZERO_TX_JOINT_REPORT" in _codes(audit)
    assert audit["can_zero_transmit_verified"] is False


def test_invalid_calibration_requires_explicitly_disabled_robot_overlay(tmp_path):
    unsafe_bundle = _bundle(
        tmp_path / "unsafe",
        calibration_valid=False,
        overlay_allowed=True,
    )
    safe_bundle = _bundle(
        tmp_path / "safe",
        calibration_valid=False,
        overlay_allowed=False,
    )

    unsafe = GATE.audit_bundle(unsafe_bundle, tmp_path)
    safe = GATE.audit_bundle(safe_bundle, tmp_path)

    assert "ROBOT_OVERLAY_WITHOUT_VALID_CALIBRATION" in _codes(unsafe)
    assert safe["passed"] is True
    assert safe["calibration_valid"] is False
    assert safe["robot_overlay_allowed"] is False


def test_missing_joint_evidence_allows_only_robotless_overlay(tmp_path):
    unsafe_bundle = _bundle(tmp_path / "unsafe", include_joint=False, overlay_allowed=True)
    safe_bundle = _bundle(tmp_path / "safe", include_joint=False, overlay_allowed=False)

    unsafe = GATE.audit_bundle(unsafe_bundle, tmp_path)
    safe = GATE.audit_bundle(safe_bundle, tmp_path)

    assert "ROBOT_OVERLAY_WITHOUT_PASSIVE_JOINT_EVIDENCE" in _codes(unsafe)
    assert safe["passed"] is True
    assert safe["can_zero_transmit_verified"] is False


def test_path_escape_and_sensitive_artifacts_are_rejected(tmp_path):
    root = tmp_path / "artifacts"
    bundle = _bundle(root)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    document = json.loads(bundle.read_text())
    document["artifacts"]["escaped"] = {"path": "../outside.png"}
    document["artifacts"]["secret"] = {"path": ".env"}
    (root / ".env").write_text("SECRET=value")
    _write_json(bundle, document)

    audit = GATE.audit_bundle(bundle, root)

    assert audit["passed"] is False
    assert "UNSAFE_OR_TAMPERED_ARTIFACT" in _codes(audit)


def test_symlink_escape_and_ssh_key_name_are_rejected(tmp_path):
    root = tmp_path / "artifacts"
    bundle = _bundle(root)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (root / "escape.bin").symlink_to(outside)
    ssh_key = root / "id_ed25519"
    ssh_key.write_text("not-a-key")
    document = json.loads(bundle.read_text())
    document["artifacts"]["symlink"] = {"path": "escape.bin"}
    document["artifacts"]["ssh_private_key"] = {"path": "id_ed25519"}
    _write_json(bundle, document)

    audit = GATE.audit_bundle(bundle, root)

    assert audit["passed"] is False
    assert sum(value["code"] == "UNSAFE_OR_TAMPERED_ARTIFACT" for value in audit["errors"]) >= 2


def test_source_has_no_ros_can_subprocess_publish_service_or_goal():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
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
    forbidden_attributes = {
        "call_async",
        "create_client",
        "create_publisher",
        "create_service",
        "publish",
        "send",
        "send_goal",
        "send_goal_async",
        "sendall",
        "sendmsg",
        "sendto",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_attributes
    }

    assert imports.isdisjoint(forbidden_imports)
    assert calls == set()
