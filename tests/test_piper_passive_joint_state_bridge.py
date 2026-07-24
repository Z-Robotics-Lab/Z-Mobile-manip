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


def test_publish_is_guarded_against_shutdown_context_teardown() -> None:
    # Before the fix, publisher.publish(message) sat bare in the receive loop,
    # so a systemd stop/restart could tear the rcl context down between the
    # top-of-loop rclpy.ok() and the publish -- raising
    # RCLError "publisher's context is invalid" (publisher.c:423, line 138) and
    # crash-looping.  Require the publish to sit inside a try/except whose
    # handler checks rclpy.ok() and breaks the loop on a torn-down context.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

    def _body_has_publish(node: ast.Try) -> bool:
        return any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "publish"
            for stmt in node.body
            for call in ast.walk(stmt)
        )

    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.handlers:
            continue
        if not _body_has_publish(node):
            continue
        handlers_dump = "\n".join(ast.dump(h) for h in node.handlers)
        checks_context = "rclpy" in handlers_dump and "ok" in handlers_dump
        breaks_loop = any(
            isinstance(inner, ast.Break)
            for handler in node.handlers
            for inner in ast.walk(handler)
        )
        if checks_context and breaks_loop:
            guarded = True
    assert guarded, (
        "publisher.publish must be wrapped in a shutdown-aware try/except that "
        "checks rclpy.ok() and breaks the receive loop on a torn-down context"
    )


def test_publish_guard_preserves_can_tolerance_and_tx_safety() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    # Shutdown race handled: the loop exits cleanly when the context is gone.
    assert "if not rclpy.ok():" in source
    assert "publisher.publish(message)" in source
    # CAN-tolerance is NOT regressed: a can0 blip (OSError) still closes and
    # reopens the socket, and a recv timeout still just continues.
    assert "except OSError:" in source
    assert "except TimeoutError:" in source
    assert "channel = None" in source
    # TX-safety fail-loud is NOT regressed: an unknown transmitter still raises
    # rather than being swallowed by the new publish guard.
    assert "can0 TX counter changed while passive bridge was active" in source
    assert "raise RuntimeError(" in source
