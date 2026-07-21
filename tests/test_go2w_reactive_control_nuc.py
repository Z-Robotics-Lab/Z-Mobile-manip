import importlib.util
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
    assert '"GetBodyHeight", {}, timeout_s=self._HEIGHT_QUERY_TIMEOUT_S' in source
    assert "get_body_height_from_response(response)" in source
    assert '"Move", {"x": x, "y": y, "z": yaw}' in source
    assert '("BodyHeight", {"data": height})' in source
    assert '("Euler", {"x": roll, "y": pitch, "z": yaw})' in source
    assert 'self._last_code = await self._request_sport("StopMove", {})' in source
    assert "self._stop_latched = True" in source
    assert "self._pending_move = None" in source
    assert "self._pending_posture = None" in source


def test_get_body_height_feedback_is_raw_observable_and_freshness_gated():
    source = SOURCE.read_text(encoding="utf-8")

    assert 'SPORT_CMD["GetBodyHeight"]' in source
    assert '"raw_response": self._height_query_response' in source
    assert '"parse_path": self._height_query_parse_path' in source
    assert '"parse_error": self._height_query_error' in source
    assert "self._height_query_received_s = time.monotonic()" in source
    assert "Never continue against a previous query" in source
    assert '"sport_mode_state+GetBodyHeight"' in source
    assert 'self._phase not in {"fault", "stopped", "stopping"}' in source


def test_feedback_freshness_gates_posture_and_stop_reset():
    source = SOURCE.read_text(encoding="utf-8")

    assert "_STATE_TIMEOUT_S = 0.50" in source
    assert "measured posture feedback is stale" in source
    assert "measured posture feedback is unsynchronized" in source
    assert "cannot verify reached posture" in source
    assert "fresh, detail = self._fresh_feedback()" in source
    assert "cannot release Full Stop" in source
    assert "Full Stop is always" not in source  # NUC reset is still feedback gated.


def test_posture_commands_require_an_explicit_zero_ack():
    source = SOURCE.read_text(encoding="utf-8")

    assert "result.success = code == 0" in source
    assert "if self._last_code != 0:" in source
    assert "code in (0, None)" not in source


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
            "body_height_delta_m": -0.5,
            "pitch_delta_rad": math.radians(30.0),
        }
    )

    assert target[0] == pytest.approx(-0.12)
    assert target[2] == pytest.approx(math.radians(12.0))
    with pytest.raises(ValueError, match="finite"):
        bridge.bounded_wire_target(
            {
                "schema": bridge.INTENT_SCHEMA,
                "body_height_delta_m": math.nan,
                "pitch_delta_rad": 0.0,
            }
        )


def test_pc_live_relay_requires_fresh_unlatched_nuc_feedback():
    bridge = _load_pc_bridge()
    status = {
        "schema": "z_manip.go2w_posture_status.v1",
        "mode": "live",
        "stop_latched": False,
        "feedback": {"fresh": True},
        "body_height": {"feedback_age_s": 0.1},
    }

    assert bridge.feedback_is_fresh(status)
    status["stop_latched"] = True
    assert not bridge.feedback_is_fresh(status)
    status["stop_latched"] = False
    status["body_height"]["feedback_age_s"] = 0.9
    assert not bridge.feedback_is_fresh(status)
