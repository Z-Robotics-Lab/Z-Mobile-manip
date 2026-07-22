from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_interactive_sessions.py"


def _module():
    name = "go2w_interactive_sessions_planning_ready_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_ready_tree(root: Path) -> tuple[Path, Path, Path]:
    perception = root / "perception"
    output = root / "planning-action"
    planning = output / "planning"
    perception.mkdir(parents=True)
    planning.mkdir(parents=True)
    joint = perception / "selected_passive_joint_report.json"
    (perception / "report.json").write_text(
        json.dumps({"instruction": "red bottle", "source_stamp_unix_ns": 7}),
        encoding="utf-8",
    )
    joint.write_text(
        json.dumps({
            "read_only": True,
            "joint_positions_rad": [0.0] * 6,
            "source_timestamp_ns": 8,
        }),
        encoding="utf-8",
    )
    (output / "session_gate.json").write_text(json.dumps({
        "planning_ready": True,
        "read_only": True,
        "planning_only": True,
        "motion_commands_published": 0,
        "transport_opened": False,
        "measured_joints_rad": [0.0] * 6,
    }), encoding="utf-8")
    (planning / "planning_report.json").write_text(json.dumps({
        "read_only": True,
        "planning_only": True,
        "motion_commands_published": 0,
        "plan_valid": True,
    }), encoding="utf-8")
    (planning / "planned_grasp.npz").write_bytes(b"immutable-plan")
    return perception, output, joint


def test_plan_ready_marker_requires_and_hashes_exact_evidence(tmp_path, monkeypatch):
    module = _module()
    calibration = tmp_path / "calibration.json"
    urdf = tmp_path / "robot.urdf"
    calibration.write_text("{}", encoding="utf-8")
    urdf.write_text("<robot/>", encoding="utf-8")
    monkeypatch.setattr(module, "CALIBRATION", calibration)
    monkeypatch.setattr(module, "URDF", urdf)
    perception, output, joint = _write_ready_tree(tmp_path)

    evidence = module._planning_ready_evidence(
        perception_dir=perception,
        output_dir=output,
        joint_report=joint,
    )

    assert evidence is not None
    assert evidence["executor_receipt"] is False
    assert set(evidence["evidence_sha256"]) == {
        "calibration",
        "passive_joint_report",
        "perception_report",
        "planned_grasp",
        "planning_report",
        "session_gate",
        "urdf",
    }


def test_plan_ready_marker_fails_closed_without_valid_plan(tmp_path, monkeypatch):
    module = _module()
    calibration = tmp_path / "calibration.json"
    urdf = tmp_path / "robot.urdf"
    calibration.write_text("{}", encoding="utf-8")
    urdf.write_text("<robot/>", encoding="utf-8")
    monkeypatch.setattr(module, "CALIBRATION", calibration)
    monkeypatch.setattr(module, "URDF", urdf)
    perception, output, joint = _write_ready_tree(tmp_path)
    report = output / "planning" / "planning_report.json"
    document = json.loads(report.read_text(encoding="utf-8"))
    document["plan_valid"] = False
    report.write_text(json.dumps(document), encoding="utf-8")

    assert module._planning_ready_evidence(
        perception_dir=perception,
        output_dir=output,
        joint_report=joint,
    ) is None
