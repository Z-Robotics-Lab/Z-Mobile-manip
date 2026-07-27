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


class _FakeClock:
    """Virtual monotonic clock so the 8 s fail path costs no wall time."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time_ns(self):
        return int(self.now * 1_000_000_000)


def _millidegrees(radians):
    """Encode radians the way PiPER feedback carries them (0.001 deg units)."""

    return round(radians * 180_000.0 / math.pi)


class _ScriptedChannel:
    """PF_CAN stand-in replaying joint feedback against a virtual clock.

    ``drift_rad_per_s`` ramps J1 (the first joint of the 0x2A5 pair) linearly in
    time so a scripted stream can stand in for an arm that is slowly sagging or
    being nudged rather than one that is genuinely parked.
    """

    def __init__(
        self,
        clock,
        *,
        frame_ids,
        first_frame_s,
        period_s,
        bus_down_s=None,
        drift_rad_per_s=0.0,
    ):
        self._clock = clock
        self._frame_ids = tuple(frame_ids)
        self._next_frame_s = first_frame_s
        self._period_s = period_s
        self._bus_down_s = bus_down_s
        self._drift_rad_per_s = drift_rad_per_s
        self._timeout = None
        self._index = 0
        self.recv_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def setsockopt(self, *_args):
        return None

    def settimeout(self, timeout):
        self._timeout = timeout

    def bind(self, *_args):
        return None

    def recv(self, *_args):
        self.recv_calls += 1
        if self._bus_down_s is not None and self._clock.now >= self._bus_down_s:
            raise OSError(100, "Network is down")
        if self._next_frame_s > self._clock.now + self._timeout:
            self._clock.now += self._timeout
            raise TimeoutError
        self._clock.now = max(self._clock.now, self._next_frame_s)
        self._next_frame_s += self._period_s
        frame_id = self._frame_ids[self._index % len(self._frame_ids)]
        self._index += 1
        first = 1_000
        if frame_id == 0x2A5 and self._drift_rad_per_s:
            first += _millidegrees(self._drift_rad_per_s * self._clock.now)
        return PROBE.CAN_FRAME.pack(frame_id, 8, struct.pack(">ii", first, -2_000))


def _run_scripted_probe(
    monkeypatch,
    output,
    *,
    frame_ids,
    first_frame_s,
    period_s,
    duration=8.0,
    bus_down_s=None,
    drift_rad_per_s=0.0,
):
    clock = _FakeClock()
    channel = _ScriptedChannel(
        clock,
        frame_ids=frame_ids,
        first_frame_s=first_frame_s,
        period_s=period_s,
        bus_down_s=bus_down_s,
        drift_rad_per_s=drift_rad_per_s,
    )
    monkeypatch.setattr(PROBE, "time", clock)
    monkeypatch.setattr(PROBE, "_counter", lambda _interface, _name: 0)
    monkeypatch.setattr(PROBE.socket, "socket", lambda *_a, **_k: channel)
    monkeypatch.setattr(
        PROBE,
        "_arguments",
        lambda: SimpleNamespace(interface="can0", duration=duration, output=output),
    )

    return_code = PROBE.main()
    return return_code, clock, json.loads(output.read_text(encoding="utf-8"))


def test_complete_joint_set_closes_the_window_far_inside_the_duration_bound(
    tmp_path,
    monkeypatch,
):
    return_code, clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=PROBE.JOINT_FEEDBACK_IDS,
        first_frame_s=0.01,
        period_s=1.0 / 600.0,
    )

    assert return_code == 0
    assert document["complete_joint_feedback"] is True
    assert document["zero_transmit_verified"] is True
    # A parked arm leaves at the floor, an order of magnitude inside the bound.
    assert clock.now < PROBE.MIN_OBSERVATION_S + 0.05
    assert clock.now < document["duration_s"] / 4.0


def test_incomplete_joint_set_burns_the_full_duration_and_fails_closed(
    tmp_path,
    monkeypatch,
):
    return_code, clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=(0x2A5, 0x2A6),
        first_frame_s=0.01,
        period_s=0.05,
    )

    assert return_code == 1
    assert document["complete_joint_feedback"] is False
    assert document["joint_positions_rad"][4] is None
    assert document["joint_positions_rad"][5] is None
    assert clock.now >= 8.0


def test_immediate_joint_set_still_holds_the_minimum_observation_floor(
    tmp_path,
    monkeypatch,
):
    return_code, clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=PROBE.JOINT_FEEDBACK_IDS,
        first_frame_s=0.0,
        period_s=0.001,
    )

    assert return_code == 0
    assert document["complete_joint_feedback"] is True
    assert clock.now >= PROBE.MIN_OBSERVATION_S


def test_bus_dropping_before_the_floor_fails_closed_despite_a_complete_set(
    tmp_path,
    monkeypatch,
):
    return_code, clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=PROBE.JOINT_FEEDBACK_IDS,
        first_frame_s=0.0,
        period_s=0.001,
        bus_down_s=0.02,
    )

    assert return_code == 1
    assert document["complete_joint_feedback"] is True
    assert document["zero_transmit_verified"] is True
    assert "can0 down during passive window" in document["passive_window_error"]
    assert clock.now < PROBE.MIN_OBSERVATION_S


def test_duration_shorter_than_the_floor_still_observes_its_whole_window(
    tmp_path,
    monkeypatch,
):
    # The read-only planning sessions poll the probe at --duration 0.25.
    return_code, clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=PROBE.JOINT_FEEDBACK_IDS,
        first_frame_s=0.0,
        period_s=0.001,
        duration=0.25,
    )

    assert return_code == 0
    assert document["complete_joint_feedback"] is True
    assert clock.now >= 0.25


# The absolute still-arm budget piper_hand_eye_sample.py and go2w_debug_bundle.py
# apply to max_joint_range_rad.  Restated here on purpose: the probe must not
# import from its consumers, so these tests are what pins the coupling.
DOWNSTREAM_MAX_JOINT_RANGE_RAD = 0.002


def test_early_exit_budget_scales_with_the_window_instead_of_being_fixed():
    still = [0.0] * 6

    def spread(value):
        return [value] + [0.0] * 5

    # A full-duration window may spend the whole downstream budget...
    assert PROBE.stillness_supports_early_exit(
        spread(PROBE.STILLNESS_REFERENCE_RANGE_RAD), 8.0, 8.0
    )
    # ...and an eighth of that window may spend exactly an eighth of it.
    assert PROBE.stillness_supports_early_exit(spread(0.00025), 1.0, 8.0)
    assert not PROBE.stillness_supports_early_exit(spread(0.00026), 1.0, 8.0)
    # The budget is a rate, so it tracks --duration rather than a hardcoded 8 s.
    assert PROBE.stillness_supports_early_exit(spread(0.0005), 1.0, 4.0)
    assert not PROBE.stillness_supports_early_exit(spread(0.0005), 1.0, 16.0)
    # A joint that has never reported cannot contribute stillness evidence.
    assert not PROBE.stillness_supports_early_exit([None] + [0.0] * 5, 1.0, 8.0)
    assert not PROBE.stillness_supports_early_exit(still, 0.0, 8.0)


def test_a_drifting_joint_holds_the_window_open_for_the_downstream_gate(
    tmp_path,
    monkeypatch,
):
    # J1 creeping at 1 mrad/s is exactly the case an absolute 0.002 rad gate is
    # there to catch: over a full 8 s window it spreads 0.008 rad and is
    # rejected, but a 0.35 s window would have reported 0.00035 rad and sailed
    # through.  The drift-rate guard must refuse the early exit outright.
    return_code, clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=PROBE.JOINT_FEEDBACK_IDS,
        first_frame_s=0.0,
        period_s=1.0 / 600.0,
        drift_rad_per_s=0.001,
    )

    # The probe itself still succeeds: the set completed and nothing transmitted.
    assert return_code == 0
    assert document["complete_joint_feedback"] is True
    assert document["zero_transmit_verified"] is True
    # It observed the whole bound rather than closing at the floor...
    assert clock.now >= 8.0
    assert document["max_joint_range_rad"] == pytest.approx(0.008, rel=0.05)
    # ...so the downstream still-arm gate rejects it exactly as it does today.
    assert document["max_joint_range_rad"] > DOWNSTREAM_MAX_JOINT_RANGE_RAD


def test_a_still_arm_leaves_at_the_floor_and_stays_inside_the_downstream_gate(
    tmp_path,
    monkeypatch,
):
    return_code, clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=PROBE.JOINT_FEEDBACK_IDS,
        first_frame_s=0.0,
        period_s=1.0 / 600.0,
        drift_rad_per_s=0.0,
    )

    assert return_code == 0
    assert document["complete_joint_feedback"] is True
    assert document["max_joint_range_rad"] == 0.0
    assert document["max_joint_range_rad"] <= DOWNSTREAM_MAX_JOINT_RANGE_RAD
    assert PROBE.MIN_OBSERVATION_S <= clock.now < PROBE.MIN_OBSERVATION_S + 0.05


def test_passive_report_schema_keys_match_the_captured_eight_second_format(
    tmp_path,
    monkeypatch,
):
    # Field-for-field key set of the captured 8 s report; the early-exit window
    # changes how long the probe listens, never what it attests to.
    _return_code, _clock, document = _run_scripted_probe(
        monkeypatch,
        tmp_path / "passive.json",
        frame_ids=PROBE.JOINT_FEEDBACK_IDS,
        first_frame_s=0.01,
        period_s=1.0 / 600.0,
    )

    assert sorted(document) == [
        "complete_joint_feedback",
        "duration_s",
        "feedback_span_s",
        "filtered_frame_counts",
        "filtered_frames",
        "first_feedback_unix_ns",
        "interface",
        "interface_rx_packet_delta",
        "interface_tx_packet_delta",
        "joint_positions_deg",
        "joint_positions_rad",
        "joint_ranges_rad",
        "joint_snapshot_span_s",
        "last_feedback_unix_ns",
        "max_joint_range_rad",
        "observation_end_unix_ns",
        "observation_start_unix_ns",
        "passive_window_error",
        "read_only",
        "schema",
        "zero_transmit_verified",
    ]
    assert document["schema"] == "z_manip.piper_passive_joint_report.v1"
    assert document["duration_s"] == 8.0
    assert sorted(document["filtered_frame_counts"]) == ["0x2A5", "0x2A6", "0x2A7"]


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


def test_root_can_gate_tx_counter_comparisons_are_unchanged():
    source = GATE.read_text(encoding="utf-8")
    counter = '"$(<"/sys/class/net/$interface/statistics/tx_packets")"'

    assert f"tx_before={counter}" in source
    assert f"tx_after_up={counter}" in source
    assert f"tx_after_probe={counter}" in source
    assert '[[ "$tx_after_up" -ne "$tx_before" ]]' in source
    assert '[[ "$tx_after_probe" -ne "$tx_before" ]]' in source
    assert source.index("tx_before=") < source.index('ip link set "$interface" up')
    assert source.index('ip link set "$interface" up') < source.index("tx_after_up=")
    assert source.index("tx_after_up=") < source.index("tx_after_probe=")
    assert 'duration="${2:-8}"' in source
    # The bring-up settle shortens but never disappears: a host whose sleep is
    # integer-only falls back to a whole second instead of skipping the wait.
    assert "sleep 0.5 2>/dev/null || sleep 1" in source
    assert source.index("sleep 0.5") < source.index("tx_after_up=")


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
