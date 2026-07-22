import importlib.util
import ast
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/runtime/go2w_reactive_control_nuc.py"
LAUNCHER = ROOT / "scripts/runtime/go2w_reactive_control_nuc.sh"
PC_BRIDGE = ROOT / "scripts/runtime/go2w_posture_intent_bridge.py"
LIVE_UNIT = ROOT / "configs/z-mobile-manip-go2w-reactive-live.service"
SHADOW_UNIT = ROOT / "configs/z-mobile-manip-go2w-reactive-shadow.service"
PC_LIVE_UNIT = ROOT / "configs/z-mobile-manip-go2w-posture-intent-live.service"


def _load_pc_bridge():
    spec = importlib.util.spec_from_file_location("go2w_posture_intent_bridge", PC_BRIDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_euler_classifier():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "RPC_ERR_SERVER_API_NOT_IMPL"
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_euler_response_outcome":
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_euler_response_outcome"]


def _load_motion_mode_parser():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "MOTION_SWITCHER_CHECK_MODE_API_ID",
        "RPC_ERR_SERVER_API_NOT_IMPL",
    }
    functions = {
        "_status_code",
        "_raw_response_evidence",
        "_motion_mode_evidence",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {"Any": object, "json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_motion_mode_evidence"]


def test_shadow_path_cannot_construct_unitree_transport():
    source = SOURCE.read_text(encoding="utf-8")

    assert "class ShadowReactiveControlNode(_StatusNode)" in source
    assert "shadow: transport was not constructed" in source
    assert "ReactiveUnitreeControlNode()" in source
    assert "if parsed.mode == \"live\"" in source
    assert "UnitreeWebRTCConnection(" not in source


def test_live_requires_exact_gate_but_not_a_hand_written_nominal_height():
    source = SOURCE.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'LIVE_ACK = "I_UNDERSTAND_GO2W_WILL_MOVE"' in source
    assert "Z_MANIP_GO2W_LIVE_ACK" in source
    assert "Z_MANIP_GO2W_NOMINAL_BODY_HEIGHT_M" not in source
    assert 'MODE="${1:-shadow}"' in launcher
    assert "Z_MANIP_GO2W_LIVE_ACK" in launcher
    assert "Z_MANIP_GO2W_NOMINAL_BODY_HEIGHT_M" not in launcher


def test_live_bridge_is_one_owner_with_serialized_stop_move_and_posture():
    source = SOURCE.read_text(encoding="utf-8")

    assert "class ReactiveUnitreeControlNode(UnitreeControlNode, _StatusNode)" in source
    assert "self._sport_request_lock = asyncio.Lock()" in source
    assert "async with self._sport_request_lock" in source
    assert "def _execute_sport_command(" in source
    assert source.count("publish_request_new(") == 1
    assert '"Move", {"x": x, "y": y, "z": yaw}' in source
    assert '("Euler", {"x": roll, "y": pitch, "z": yaw})' in source
    assert '"GetBodyHeight"' not in source
    assert '("BodyHeight",' not in source
    assert 'self._last_code = await self._request_sport("StopMove", {})' in source
    assert "self._stop_latched = True" in source
    assert "self._pending_move = None" in source
    assert "self._pending_posture = None" in source


def test_web_rtc_status_keeps_mode_and_per_command_robot_evidence():
    parse = _load_motion_mode_parser()
    source = SOURCE.read_text(encoding="utf-8")
    response = {
        "type": "res",
        "topic": "rt/api/motion_switcher/response",
        "data": {
            "header": {"status": {"code": 0}},
            "data": '{"form":"1","name":"ai-w"}',
        },
    }

    assert parse(response) == {
        "check_api_id": 1001,
        "robot_code": 0,
        "name": "ai-w",
        "form": "1",
        "api_family": "wheeled_sport",
        "parse_error": None,
        "raw_response": response,
    }
    assert '"motion_switcher_topic": RTC_TOPIC["MOTION_SWITCHER"]' in source
    assert '"Move": None' in source
    assert '"Euler": None' in source
    assert '"StopMove": None' in source
    assert 'self._command_codes["Move"] = self._last_code' in source
    assert 'self._command_codes["StopMove"] = self._last_code' in source


def test_body_height_is_explicitly_unsupported_and_never_queried():
    source = SOURCE.read_text(encoding="utf-8")

    assert '"body_height": False' in source
    assert '"get_body_height": False' in source
    assert '"source": "sport_mode_state"' in source
    assert '"BodyHeight is unsupported; linear.z must be zero"' in source
    assert '"api_id": None' in source


def test_feedback_freshness_gates_posture_and_stop_reset():
    source = SOURCE.read_text(encoding="utf-8")

    assert "_STATE_TIMEOUT_S = 0.50" in source
    assert "measured posture feedback is stale" in source
    assert "cannot verify reached posture" in source
    assert "fresh, detail = self._fresh_feedback()" in source
    assert "cannot release Full Stop" in source
    assert "Full Stop is always" not in source  # NUC reset is still feedback gated.


def test_posture_commands_require_an_explicit_zero_ack():
    source = SOURCE.read_text(encoding="utf-8")

    assert "result.success = code == 0" in source
    assert 'if code == 0:' in source
    assert 'return "accepted"' in source
    assert "code in (0, None)" not in source


def test_euler_api_not_implemented_degrades_instead_of_faulting_forever():
    classify = _load_euler_classifier()
    source = SOURCE.read_text(encoding="utf-8")

    assert classify(0) == "accepted"
    assert classify(3203) == "unsupported"
    assert classify(-1) == "fault"
    assert classify(None) == "fault"
    assert 'self._phase = "unsupported"' in source
    assert '"euler": self._euler_supported' in source
    assert '"euler_state": self._euler_capability_state' in source
    assert 'self._euler_capability_state = "unsupported_for_session"' in source
    assert "degraded to base + arm control" in source


def test_launcher_routes_move_through_guard_only_in_live_mode():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "cmd_vel_guard.launch.py" in launcher
    assert "if [[ \"$MODE\" == live ]]" in launcher
    assert "-r cmd_vel:=/cmd_vel_safe" in launcher
    assert "max_linear_mps:=0.20" in launcher
    assert "pkill" not in launcher
    assert "killall" not in launcher


def test_services_prevent_two_live_webrtc_owners():
    live = LIVE_UNIT.read_text(encoding="utf-8")
    shadow = SHADOW_UNIT.read_text(encoding="utf-8")

    assert "Conflicts=z-manip-go2w-base-control.service" in live
    assert "z-mobile-manip-go2w-reactive-shadow.service" in live
    assert "go2w_reactive_control_nuc.sh live" in live
    assert "EnvironmentFile=" in live
    assert "go2w_reactive_control_nuc.sh shadow" in shadow
    pc_live = PC_LIVE_UNIT.read_text(encoding="utf-8")
    assert "Conflicts=z-mobile-manip-go2w-posture-intent-shadow.service" in pc_live
    assert "go2w_posture_intent_bridge.sh live" in pc_live
    assert "EnvironmentFile=" in pc_live


def test_pc_intent_conversion_is_neutral_relative_bounded_and_finite():
    bridge = _load_pc_bridge()
    target = bridge.bounded_wire_target(
        {
            "schema": bridge.INTENT_SCHEMA,
            "pitch_delta_rad": math.radians(30.0),
        }
    )

    assert target[0] == pytest.approx(0.0)
    assert target[2] == pytest.approx(math.radians(12.0))
    with pytest.raises(ValueError, match="finite"):
        bridge.bounded_wire_target(
            {
                "schema": bridge.INTENT_SCHEMA,
                "body_height_delta_m": math.nan,
                "pitch_delta_rad": 0.0,
            }
        )
    with pytest.raises(ValueError, match="BodyHeight is unsupported"):
        bridge.bounded_wire_target(
            {
                "schema": bridge.INTENT_SCHEMA,
                "body_height_delta_m": -0.01,
                "pitch_delta_rad": 0.0,
            }
        )


def test_pc_live_relay_requires_fresh_unlatched_nuc_feedback():
    bridge = _load_pc_bridge()
    status = {
        "schema": "z_manip.go2w_posture_status.v1",
        "mode": "live",
        "stop_latched": False,
        "feedback": {"fresh": True, "sport_state_age_s": 0.1},
    }

    assert bridge.feedback_is_fresh(status)
    status["stop_latched"] = True
    assert not bridge.feedback_is_fresh(status)
    status["stop_latched"] = False
    status["feedback"]["sport_state_age_s"] = 0.9
    assert not bridge.feedback_is_fresh(status)


def test_pc_relay_suppresses_euler_after_explicit_capability_rejection():
    bridge = _load_pc_bridge()

    assert bridge.euler_is_available({})
    assert bridge.euler_is_available({"capabilities": {"euler": True}})
    assert not bridge.euler_is_available({"capabilities": {"euler": False}})
