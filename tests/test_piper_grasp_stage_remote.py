"""Transport-hygiene contract for the staged grasp remote.

Each stage makes five short ssh/scp calls (mkdir, upload, executor, receipt
fetch, cleanup) and a grasp drives three stages as separate processes.  Over
the WiFi link to the NUC a cold SSH handshake costs ~0.40s while a call
multiplexed onto a persisted master costs ~0.01-0.03s, so every ssh/scp call
must carry the connection-multiplexing options against one shared, dedicated
grasp control path.  Live motion is never exercised: the terminal executor call
and every ssh/scp are stubbed; the fetch stub writes a valid stage receipt so
the wrapper's schema gate still runs end to end.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "piper_grasp_stage_remote",
    ROOT / "scripts" / "runtime" / "piper_grasp_stage_remote.py",
)
remote = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(remote)


class FakeRun:
    """Dispatch stage-remote subprocess calls by shape; record every call."""

    def __init__(self, receipt_output: Path, stage: str) -> None:
        self.receipt_output = receipt_output
        self.stage = stage
        self.calls: list[list[str]] = []

    def __call__(self, arguments, *, timeout, capture=False):
        self.calls.append([str(part) for part in arguments])

        def done(rc, out=""):
            return subprocess.CompletedProcess(arguments, rc, stdout=out, stderr=None)

        arg0 = str(arguments[0])
        if arg0 == sys.executable:
            return done(0, json.dumps({"confirmation_token": "PIPER-STAGE-test"}))
        if arg0 == "scp":
            if str(arguments[-1]) == str(self.receipt_output):
                self.receipt_output.parent.mkdir(parents=True, exist_ok=True)
                self.receipt_output.write_text(
                    json.dumps({
                        "schema": "z_manip.piper_stage_receipt.v1",
                        "stage": self.stage,
                        "success": True,
                        "artifact_id": "artifact-xyz",
                    }),
                    encoding="utf-8",
                )
            return done(0)
        tail = str(arguments[-1])
        if "rm -rf" in tail or "mkdir -p" in tail:
            return done(0)
        if tail.startswith("set -e;"):
            return done(0, "stage output\n")
        raise AssertionError(f"unexpected stage-remote subprocess: {arguments}")


def _invoke(tmp_path, monkeypatch, fake: FakeRun) -> dict:
    key = tmp_path / "nuc-key"
    key.write_text("key", encoding="utf-8")
    report = tmp_path / "planning_report.json"
    report.write_text("{}", encoding="utf-8")
    archive = tmp_path / "planned_grasp.npz"
    archive.write_bytes(b"npz")
    monkeypatch.setattr(remote, "NUC_KEY", key)
    monkeypatch.setattr(remote, "_run", fake)
    return remote.execute_remote_stage(
        planning_report=report,
        planned_grasp=archive,
        stage=fake.stage,
        receipt_output=fake.receipt_output,
        prior_receipt=None,
    )


def test_stage_succeeds_and_returns_receipt(tmp_path, monkeypatch):
    fake = FakeRun(tmp_path / "out" / "pregrasp-receipt.json", "pregrasp")
    receipt = _invoke(tmp_path, monkeypatch, fake)
    assert receipt["success"] is True
    assert receipt["stage"] == "pregrasp"


def test_every_ssh_and_scp_call_reuses_one_control_master(tmp_path, monkeypatch):
    fake = FakeRun(tmp_path / "out" / "pregrasp-receipt.json", "pregrasp")
    _invoke(tmp_path, monkeypatch, fake)

    transport_calls = [
        call for call in fake.calls if call and call[0] in ("ssh", "scp")
    ]
    assert transport_calls, "stage remote made no ssh/scp calls"
    control_paths = set()
    for call in transport_calls:
        assert "ControlMaster=auto" in call
        assert "ControlPersist=60" in call
        control_path = next(
            part.split("=", 1)[1]
            for part in call
            if part.startswith("ControlPath=")
        )
        assert control_path.endswith("z-manip-grasp-%C")
        control_paths.add(control_path)
    assert len(control_paths) == 1
