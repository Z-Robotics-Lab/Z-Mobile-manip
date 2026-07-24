"""Sequencing contract for the checked reverse-home recovery.

The operator complaint these tests pin down: after a grasp, a single Home
request used to stop at the session's recorded planning-start pose (the reverse
corridor endpoint ``transit[0]``) and only a *second* Home command reached the
calibrated software Home.  The nominal success path must now converge directly
to calibrated Home in one request, while genuine failure/abort cases (Home
config missing, or the calibrated-Home delta beyond the bounded convergence
envelope) still fall back to a safe stop at the checked corridor endpoint.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts" / "runtime"
sys.path.insert(0, str(RUNTIME))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUNTIME / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registering the reverse-recovery module lets its own ``import
    # piper_staged_grasp_executor`` land in sys.modules so the stage module's
    # dataclasses resolve their owning module during exec.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REVERSE = _load("piper_reverse_home_recovery_contract", "piper_reverse_home_recovery.py")
STAGE = REVERSE.stage


ARTIFACT_ID = "a" * 64
RECOVER_TOKEN = f"PIPER-RECOVER-{ARTIFACT_ID[:16]}"

Q_HOME_START = np.zeros(6)              # transit[0] == planning-start pose
Q_PRE = np.full(6, 0.10)
Q_GRASP = np.full(6, 0.20)
Q_LIFT = np.full(6, 0.30)              # arm rests here after a lift


def _software_home_document(joints_rad: list[float]) -> dict[str, object]:
    return {
        "schema": "z_manip.piper_software_home.v1",
        "name": "test_home",
        "joint_degrees": [math.degrees(value) for value in joints_rad],
        "joint_radians": list(joints_rad),
        "capture_zero_can_tx_verified": True,
        "interface": "can0",
        "firmware": "S-V1.8-8",
    }


def _write_home(path: Path, joints_rad: list[float]) -> Path:
    path.write_text(json.dumps(_software_home_document(joints_rad)), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# load_software_home: the calibrated target must be evidence-bound and legal
# --------------------------------------------------------------------------

def test_load_software_home_accepts_captured_pose(tmp_path):
    home = _write_home(tmp_path / "home.json", [0.0, 0.08, -0.14, -0.03, 0.26, 0.0])
    loaded = STAGE.load_software_home(home)
    np.testing.assert_allclose(loaded, [0.0, 0.08, -0.14, -0.03, 0.26, 0.0])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc.update(schema="z_manip.something_else.v1"), "software Home schema"),
        (lambda doc: doc.update(capture_zero_can_tx_verified=False), "zero CAN TX"),
        (lambda doc: doc.update(joint_degrees=[999.0, 0, 0, 0, 0, 0]), "degree/radian"),
        (lambda doc: doc.update(
            joint_radians=[math.radians(200.0), 0, 0, 0, 0, 0],
            joint_degrees=[200.0, 0, 0, 0, 0, 0],
        ), "joint limits"),
    ],
)
def test_load_software_home_rejects_untrusted_or_illegal(tmp_path, mutate, message):
    document = _software_home_document([0.0, 0.08, -0.14, -0.03, 0.26, 0.0])
    mutate(document)
    path = tmp_path / "home.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(STAGE.SafetyError, match=message):
        STAGE.load_software_home(path)


# --------------------------------------------------------------------------
# converge_to_measured_home: bounded low-speed hop, fail-closed beyond envelope
# --------------------------------------------------------------------------

def test_converge_steps_within_deadband_and_reaches_home(monkeypatch):
    captured: dict[str, object] = {}

    def fake_execute(_robot, path, _guard, **_kwargs):
        captured["path"] = np.asarray(path, dtype=float)
        return np.asarray(path, dtype=float)[-1]

    monkeypatch.setattr(STAGE, "execute_joint_path", fake_execute)
    home = np.full(6, math.radians(12.0))  # 12deg <= 20deg envelope, > one 5deg step
    final = STAGE.converge_to_measured_home(
        object(), np.zeros(6), home, STAGE.CommandGuard(), speed_percent=5,
    )
    path = captured["path"]
    # ceil(12 / 5) == 3 steps -> 4 waypoints, endpoints exact, every leg <= 5deg.
    assert path.shape == (4, 6)
    np.testing.assert_allclose(path[0], np.zeros(6))
    np.testing.assert_allclose(path[-1], home)
    legs = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    assert float(np.max(legs)) <= math.radians(5.0) + 1e-9
    np.testing.assert_allclose(final, home)


def test_converge_rejects_delta_beyond_envelope(monkeypatch):
    monkeypatch.setattr(
        STAGE,
        "execute_joint_path",
        lambda *_a, **_k: pytest.fail("out-of-envelope convergence must not move the arm"),
    )
    with pytest.raises(STAGE.SafetyError, match="convergence envelope"):
        STAGE.converge_to_measured_home(
            object(), np.zeros(6), np.full(6, math.radians(25.0)),
            STAGE.CommandGuard(), speed_percent=5,
        )


# --------------------------------------------------------------------------
# _load_optional_measured_home: absence/invalidity is a safe skip, not a fault
# --------------------------------------------------------------------------

def test_optional_home_missing_is_safe_skip(tmp_path):
    target, note = REVERSE._load_optional_measured_home(tmp_path / "absent.json")
    assert target is None
    assert "no calibrated software Home" in note


def test_optional_home_invalid_is_safe_skip(tmp_path):
    bad = tmp_path / "home.json"
    bad.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    target, note = REVERSE._load_optional_measured_home(bad)
    assert target is None
    assert "unusable" in note


def test_optional_home_valid_loads(tmp_path):
    good = _write_home(tmp_path / "home.json", [0.0, 0.08, -0.14, -0.03, 0.26, 0.0])
    target, note = REVERSE._load_optional_measured_home(good)
    assert note is None
    np.testing.assert_allclose(target, [0.0, 0.08, -0.14, -0.03, 0.26, 0.0])


# --------------------------------------------------------------------------
# main() sequencing decision (the operator-facing behavior)
# --------------------------------------------------------------------------

def _drive_main(monkeypatch, tmp_path, *, home_arg: Path):
    """Run reverse-home recovery from a post-lift pose with mocked transport.

    Returns ``(exit_code, receipt_dict, joint_path_calls, converge_calls)``.
    """
    report = tmp_path / "planning_report.json"
    report.write_text(json.dumps({"source_stamp_ns": 1}), encoding="utf-8")
    archive = tmp_path / "planned_grasp.npz"
    archive.write_bytes(b"stub")

    fake_artifact = SimpleNamespace(
        artifact_id=ARTIFACT_ID,
        arrays={
            "transit_raw": np.asarray([Q_HOME_START, Q_PRE]),
            "approach_raw": np.asarray([Q_PRE, Q_GRASP]),
            "lift_raw": np.asarray([Q_GRASP, Q_LIFT]),
        },
    )
    monkeypatch.setattr(STAGE, "load_planning_artifact", lambda *a, **k: fake_artifact)
    monkeypatch.setattr(STAGE, "connect_real_arm", lambda *a, **k: (object(), _effector()))
    monkeypatch.setattr(STAGE, "wait_for_initial_arm_feedback", lambda *a, **k: (Q_LIFT.copy(), 1.0))
    monkeypatch.setattr(STAGE, "wait_for_fresh_joint_feedback", lambda *a, **k: (Q_LIFT.copy(), 1.0))
    monkeypatch.setattr(STAGE, "enter_can_joint_control", lambda *a, **k: None)
    monkeypatch.setattr(STAGE, "normalize_gripper_feedback", lambda *_a, **_k: SimpleNamespace(timestamp=0.0))
    monkeypatch.setattr(STAGE, "wait_for_gripper", lambda *a, **k: SimpleNamespace(aperture_m=0.07))
    monkeypatch.setattr(
        STAGE, "restore_gripper_enable_at_current_aperture",
        lambda *a, **k: SimpleNamespace(aperture_m=0.07),
    )
    monkeypatch.setattr(STAGE, "disconnect_quietly", lambda *_a, **_k: None)

    joint_calls: list[np.ndarray] = []

    def fake_execute(_robot, path, _guard, **_kwargs):
        arr = np.asarray(path, dtype=float).copy()
        joint_calls.append(arr)
        return arr[-1]

    converge_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_converge(_robot, current, home, _guard, **_kwargs):
        converge_calls.append((np.asarray(current, dtype=float).copy(), np.asarray(home, dtype=float).copy()))
        return np.asarray(home, dtype=float).copy()

    monkeypatch.setattr(STAGE, "execute_joint_path", fake_execute)
    monkeypatch.setattr(STAGE, "converge_to_measured_home", fake_converge)

    argv = [
        "piper_reverse_home_recovery.py",
        "--planning-report", str(report),
        "--planned-grasp", str(archive),
        "--speed-percent", "5",
        "--execute",
        "--confirm", RECOVER_TOKEN,
        "--home", str(home_arg),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    captured: dict[str, str] = {}
    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    code = REVERSE.main()
    # The final receipt is the only stdout JSON line with the recovery schema.
    receipt = None
    for line in printed:
        try:
            candidate = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(candidate, dict) and candidate.get("schema") == "z_manip.piper_home_recovery.v1":
            receipt = candidate
    return code, receipt, joint_calls, converge_calls


def _effector():
    return SimpleNamespace(
        get_gripper_status=lambda: None,
        move_gripper_m=lambda **_k: None,
    )


def test_nominal_success_converges_directly_to_true_home(monkeypatch, tmp_path):
    # Calibrated Home 12deg from the corridor endpoint -> inside the envelope.
    # J3 is negative-only, so keep every joint legal while 12deg from zero.
    home_joints = [math.radians(d) for d in (12.0, 12.0, -12.0, 12.0, 12.0, 12.0)]
    home_arg = _write_home(tmp_path / "home.json", home_joints)

    code, receipt, joint_calls, converge_calls = _drive_main(
        monkeypatch, tmp_path, home_arg=home_arg,
    )

    assert code == 0
    # The full checked reverse corridor still runs: reverse lift, reverse
    # approach, reverse transit -> lands on the recorded planning start.
    assert len(joint_calls) == 3
    np.testing.assert_allclose(joint_calls[0], [Q_LIFT, Q_LIFT, Q_GRASP])
    np.testing.assert_allclose(joint_calls[1], [Q_GRASP, Q_PRE])
    np.testing.assert_allclose(joint_calls[2], [Q_PRE, Q_HOME_START])
    # ...but it does NOT stop there: one convergence hop from that endpoint to
    # calibrated Home, in the same command.
    assert len(converge_calls) == 1
    start, target = converge_calls[0]
    np.testing.assert_allclose(start, Q_HOME_START)
    np.testing.assert_allclose(target, home_joints)
    assert receipt["converged_to_measured_home"] is True
    np.testing.assert_allclose(receipt["final_joints_rad"], home_joints)
    assert "measured_home_note" not in receipt
    assert receipt["phase"] == "complete"
    assert receipt["returned_home"] is True


def test_abort_beyond_envelope_preserves_reverse_corridor_stop(monkeypatch, tmp_path):
    # Calibrated Home 25deg away on J1 -> beyond the 20deg envelope.
    home_joints = [math.radians(25.0), 0.0, 0.0, 0.0, 0.0, 0.0]
    home_arg = _write_home(tmp_path / "home.json", home_joints)

    code, receipt, joint_calls, converge_calls = _drive_main(
        monkeypatch, tmp_path, home_arg=home_arg,
    )

    assert code == 0
    # Reverse corridor still runs to the checked endpoint...
    assert len(joint_calls) == 3
    # ...and there is NO unplanned direct-home hop: the checked forward path is
    # the only known-safe corridor, so it stops at the recorded planning start.
    assert converge_calls == []
    assert receipt["converged_to_measured_home"] is False
    np.testing.assert_allclose(receipt["final_joints_rad"], Q_HOME_START)
    assert "beyond" in receipt["measured_home_note"]


def test_missing_home_config_preserves_reverse_corridor_stop(monkeypatch, tmp_path):
    code, receipt, joint_calls, converge_calls = _drive_main(
        monkeypatch, tmp_path, home_arg=tmp_path / "absent.json",
    )

    assert code == 0
    assert len(joint_calls) == 3
    assert converge_calls == []
    assert receipt["converged_to_measured_home"] is False
    np.testing.assert_allclose(receipt["final_joints_rad"], Q_HOME_START)
    assert "no calibrated software Home" in receipt["measured_home_note"]
