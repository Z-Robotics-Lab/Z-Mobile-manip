from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "piper_passive_probe.py"
GATE = ROOT / "scripts" / "runtime" / "piper_passive_can_gate.sh"
DOWN = ROOT / "scripts" / "runtime" / "piper_can_down.sh"
INSTALLER = ROOT / "scripts" / "runtime" / "install_nuc_passive_access.sh"
SPEC = importlib.util.spec_from_file_location("piper_passive_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.mark.parametrize(
    ("frame_id", "raw", "indices"),
    [
        (0x2A5, (90_000, -45_000), (0, 1)),
        (0x2A6, (-180_000, 12_500), (2, 3)),
        (0x2A7, (0, 1_000), (4, 5)),
    ],
)
def test_decode_joint_feedback(frame_id, raw, indices):
    decoded = PROBE.decode_joint_pair(frame_id, struct.pack(">ii", *raw))

    assert tuple(index for index, _value in decoded) == indices
    assert decoded[0][1] == pytest.approx(math.radians(raw[0] * 1e-3))
    assert decoded[1][1] == pytest.approx(math.radians(raw[1] * 1e-3))


def test_decode_rejects_wrong_id_and_size():
    with pytest.raises(ValueError, match="unsupported"):
        PROBE.decode_joint_pair(0x155, bytes(8))
    with pytest.raises(ValueError, match="8 bytes"):
        PROBE.decode_joint_pair(0x2A5, bytes(7))


def test_probe_source_has_no_socket_transmit_call():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden = {"send", "sendall", "sendmsg", "sendto", "sendfile"}
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden
    ]

    assert calls == []


class _FakeChannel:
    """Minimal PF_CAN stand-in that fails closed like a bus-down interface."""

    def __init__(self, *, fail_on):
        self._fail_on = fail_on
        self.recv_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def setsockopt(self, *_args):
        return None

    def settimeout(self, *_args):
        return None

    def bind(self, *_args):
        if self._fail_on == "bind":
            raise OSError(100, "Network is down")

    def recv(self, *_args):
        self.recv_calls += 1
        if self._fail_on == "recv":
            raise OSError(100, "Network is down")
        raise AssertionError("recv must not be reached in this scenario")


@pytest.mark.parametrize("fail_on", ["bind", "recv"])
def test_probe_reports_bus_down_fail_closed_without_traceback(
    tmp_path,
    monkeypatch,
    fail_on,
):
    output = tmp_path / "passive.json"
    channel = _FakeChannel(fail_on=fail_on)
    monkeypatch.setattr(PROBE, "_counter", lambda _interface, _name: 0)
    monkeypatch.setattr(PROBE.socket, "socket", lambda *_a, **_k: channel)
    monkeypatch.setattr(
        PROBE,
        "_arguments",
        lambda: SimpleNamespace(interface="can0", duration=0.25, output=output),
    )

    # A raw OSError would escape as SystemExit; the gate must instead return a
    # legible fail-closed exit code and still emit its report.
    return_code = PROBE.main()

    assert return_code == 1
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "z_manip.piper_passive_joint_report.v1"
    assert document["read_only"] is True
    assert document["complete_joint_feedback"] is False
    # Zero-TX evidence is preserved: a down bus never transmits.
    assert document["zero_transmit_verified"] is True
    assert document["interface_tx_packet_delta"] == 0
    assert "can0 down during passive window" in document["passive_window_error"]
    if fail_on == "recv":
        assert channel.recv_calls == 1


def test_root_can_gate_is_fail_closed_and_has_no_can_sender():
    source = GATE.read_text(encoding="utf-8")

    assert "trap cleanup EXIT" in source
    assert 'ip link set "$interface" down' in source
    assert "tx_after_up" in source
    assert "tx_after_probe" in source
    assert "piper_passive_probe.py" in source
    assert "--output" in source
    assert "piper_passive_probe_report.json" in source
    assert "cansend" not in source
    assert "piper_sdk" in source  # forbidden-process guard, never an invocation
    assert "ros2 run" not in source
    assert 'probe="/usr/local/libexec/z-manip/piper_passive_probe.py"' in source


def test_persistent_access_is_scoped_to_passive_gate_and_can_down():
    installer = INSTALLER.read_text(encoding="utf-8")
    down = DOWN.read_text(encoding="utf-8")

    assert "NOPASSWD: ALL" not in installer
    assert "z-manip-piper-passive-can-gate can0 8" in installer
    assert "z-manip-piper-can-down" in installer
    assert "visudo -cf" in installer
    assert "systemctl enable --now ssh.service" in installer
    assert 'interface="can0"' in down
    assert "/usr/sbin/ip link set \"$interface\" down" in down
    assert " up" not in down
