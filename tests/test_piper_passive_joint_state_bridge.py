from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "runtime" / "piper_passive_joint_state_bridge.py"


def load_module():
    spec = importlib.util.spec_from_file_location("piper_passive_bridge", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decodes_all_feedback_pairs() -> None:
    module = load_module()
    payload = struct.pack(">ii", 90_000, -45_000)
    for frame_id, expected_indices in module.PAIR_BY_ID.items():
        values = module.decode_joint_pair(frame_id, payload)
        assert tuple(index for index, _ in values) == expected_indices
        assert round(values[0][1], 6) == round(1.5707963267948966, 6)


def test_can_surface_is_receive_only() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden = {"send", "sendall", "sendto", "write"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (calls & forbidden)
    assert "recv" in calls


def test_ros_surface_has_one_telemetry_publisher_and_no_control_surface() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("create_publisher(") == 1
    assert 'topic != "/piper/state"' in source
    for forbidden in (
        "create_subscription(",
        "create_service(",
        "create_client(",
        "ActionClient(",
        "move_j(",
        "enable(",
    ):
        assert forbidden not in source


def test_unit_is_domain_20_and_control_free() -> None:
    unit = (ROOT / "configs" / "z-manip-piper-passive-feedback.service").read_text()
    assert "ROS_DOMAIN_ID=20" in unit
    assert "--topic /piper/state" in unit
    assert "control" not in unit.lower()
