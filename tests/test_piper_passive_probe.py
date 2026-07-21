from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import struct

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
