"""Receipt-durability and payload-staging contract of the full-grasp remote wrapper.

Live evidence 2026-07-23: a transient ssh blip during the start-receipt probe
made the wrapper skip the local fetch while its cleanup still deleted the
remote action directory, destroying the only evidence for a physically held
object and surfacing the misleading "handoff evidence is not a regular file"
error.  The invariant under test: the remote directory is deleted ONLY when
receipts landed locally or the probe PROVED no start receipt exists.

Measured 2026-07-27: a place-back cycle re-uploads one byte-identical 320KB
payload once per leg at 0.30-0.53s each.  The second contract here is that the
staging cache may only ever skip that upload when the NUC has itself proved the
staged copy is complete and unmodified -- a leg must never execute against a
stale or partial payload, whatever it costs in re-uploads.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "piper_full_grasp_remote",
    ROOT / "scripts" / "runtime" / "piper_full_grasp_remote.py",
)
wrapper = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(wrapper)


class FakeRuns:
    """Dispatch wrapper subprocess calls by shape; record every invocation."""

    def __init__(
        self,
        *,
        executor_rc: int,
        probe_rcs: tuple[int, ...],
        fetch_rc: int,
        receipt_dir: Path,
        execute_raises_timeout: bool = False,
        payload_staged: bool = False,
    ) -> None:
        self.executor_rc = executor_rc
        self.probe_rcs = list(probe_rcs)
        self.fetch_rc = fetch_rc
        self.receipt_dir = receipt_dir
        self.execute_raises_timeout = execute_raises_timeout
        self.payload_staged = payload_staged
        self.fetch_attempts = 0
        self.payload_uploads = 0
        self.remote_deleted = False
        self.cleanup_command = ""
        self.calls: list[list[str]] = []

    def __call__(self, arguments, *, timeout):
        self.calls.append([str(part) for part in arguments])

        def done(rc, out=""):
            return subprocess.CompletedProcess(arguments, rc, stdout=out, stderr=None)

        tail = str(arguments[-1])
        if str(arguments[0]) == sys.executable:
            return done(0, json.dumps({"confirmation_token": "PIPER-FULL-test"}))
        if str(arguments[0]) == "scp":
            if "*.json" in " ".join(str(part) for part in arguments):
                self.fetch_attempts += 1
                if self.fetch_rc == 0:
                    (self.receipt_dir / "executor-start-receipt.json").write_text(
                        "{}", encoding="utf-8",
                    )
                return done(self.fetch_rc, "" if self.fetch_rc == 0 else "scp: lost connection")
            self.payload_uploads += 1
            return done(0)
        if "echo STAGED" in tail:
            return done(0, "STAGED\n" if self.payload_staged else "STALE\n")
        if "rm -rf" in tail:
            self.remote_deleted = True
            self.cleanup_command = tail
            return done(0)
        if "test -f" in tail:
            rc = self.probe_rcs.pop(0) if self.probe_rcs else 255
            return done(rc)
        if "mkdir -p" in tail:
            return done(0)
        if tail.startswith("set -e;"):
            if self.execute_raises_timeout:
                raise subprocess.TimeoutExpired(cmd=arguments, timeout=timeout)
            return done(self.executor_rc, "executor output\n")
        raise AssertionError(f"unexpected wrapper subprocess: {arguments}")


def _invoke(tmp_path, monkeypatch, fake: FakeRuns) -> int:
    key = tmp_path / "nuc-key"
    key.write_text("key", encoding="utf-8")
    report = tmp_path / "planning_report.json"
    report.write_text("{}", encoding="utf-8")
    archive = tmp_path / "planned_grasp.npz"
    archive.write_bytes(b"npz")
    monkeypatch.setattr(wrapper, "NUC_KEY", key)
    monkeypatch.setattr(wrapper, "run", fake)
    monkeypatch.setattr(wrapper.time, "sleep", lambda _s: None)
    monkeypatch.setattr(sys, "argv", [
        "piper_full_grasp_remote.py",
        "--planning-report", str(report),
        "--planned-grasp", str(archive),
        "--receipt-dir", str(fake.receipt_dir),
        "--speed-percent", "20",
    ])
    return wrapper.main()


def test_unknown_probe_preserves_remote_and_still_attempts_fetch(tmp_path, monkeypatch):
    fake = FakeRuns(
        executor_rc=1,
        probe_rcs=(255, 255, 255),
        fetch_rc=1,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 2
    assert fake.fetch_attempts >= 1
    assert fake.remote_deleted is False


def test_proved_absent_receipt_allows_remote_cleanup(tmp_path, monkeypatch):
    fake = FakeRuns(
        executor_rc=1,
        probe_rcs=(1,),
        fetch_rc=1,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 2
    # The fetch is the probe now, so proving absence costs the one failed fetch
    # that raised the question -- and never a second.
    assert fake.fetch_attempts == 1
    assert fake.remote_deleted is True
    # Nothing landed, so no local directory may survive to imply the leg ran.
    assert not fake.receipt_dir.exists()


def test_unknown_probe_with_successful_fetch_secures_evidence(tmp_path, monkeypatch):
    fake = FakeRuns(
        executor_rc=1,
        probe_rcs=(255, 255, 255),
        fetch_rc=0,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 2
    assert (fake.receipt_dir / "executor-start-receipt.json").is_file()
    assert fake.remote_deleted is True


def test_success_path_fetches_and_cleans_remote(tmp_path, monkeypatch, capsys):
    fake = FakeRuns(
        executor_rc=0,
        probe_rcs=(0,),
        fetch_rc=0,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 0
    assert fake.remote_deleted is True
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["success"] is True
    # A receipt that fetched first time never needs the separate `test -f`.
    assert not [call for call in fake.calls if "test -f" in str(call[-1])]


def test_execute_timeout_before_probe_preserves_remote(tmp_path, monkeypatch):
    fake = FakeRuns(
        executor_rc=0,
        probe_rcs=(),
        fetch_rc=0,
        receipt_dir=tmp_path / "receipts",
        execute_raises_timeout=True,
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 2
    assert fake.remote_deleted is False


def test_every_ssh_and_scp_call_reuses_one_control_master(tmp_path, monkeypatch):
    # Over the WiFi link to the NUC a cold SSH handshake costs ~0.40s, so the
    # ~6-10 ssh/scp calls this wrapper makes per leg (and the three back-to-back
    # leg processes of a place-back cycle) must multiplex over one persisted
    # master.  Every ssh/scp invocation has to carry the multiplexing options
    # against a stable dedicated grasp control path; a bare call would re-pay the
    # handshake and defeat the persisted socket.
    fake = FakeRuns(
        executor_rc=0,
        probe_rcs=(0,),
        fetch_rc=0,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 0

    transport_calls = [
        call for call in fake.calls if call and call[0] in ("ssh", "scp")
    ]
    assert transport_calls, "wrapper made no ssh/scp calls"
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
    # One shared socket path for the whole grasp -- not a per-call path.
    assert len(control_paths) == 1


def test_staged_payload_skips_both_the_mkdir_and_the_scp(tmp_path, monkeypatch, capsys):
    fake = FakeRuns(
        executor_rc=0,
        probe_rcs=(0,),
        fetch_rc=0,
        receipt_dir=tmp_path / "receipts",
        payload_staged=True,
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 0
    assert fake.payload_uploads == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["payload_upload_skipped"] is True
    # The single staging round trip both answered the question and left the
    # directory ready for an upload; nothing else may create it.
    assert len([call for call in fake.calls if "echo STAGED" in str(call[-1])]) == 1


def test_unsecured_evidence_blocks_every_remote_deletion(tmp_path, monkeypatch):
    # The evidence_secured guard is what stands between a stopped leg and an
    # unrecoverable held object; making cleanup detached must not soften it.
    fake = FakeRuns(
        executor_rc=1,
        probe_rcs=(255, 255, 255),
        fetch_rc=1,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 2
    assert not [call for call in fake.calls if "rm -rf" in str(call[-1])]


def test_secured_cleanup_is_detached_and_spares_the_payload(tmp_path, monkeypatch):
    fake = FakeRuns(
        executor_rc=0,
        probe_rcs=(0,),
        fetch_rc=0,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 0
    cleanup = fake.cleanup_command
    # Detached: the leg boundary pays a spawn, not a remote delete.
    assert cleanup.startswith("nohup sh -c ")
    assert cleanup.endswith("&")
    targets = re.search(r"rm -rf -- (.*?);", cleanup).group(1).split()
    assert all(target.startswith(f"{wrapper.REMOTE_ROOT}/") for target in targets)
    assert all("/receipts/" in target for target in targets)


# --- fake NUC ---------------------------------------------------------------

# Stands in for piper_full_grasp_executor.py on the fake NUC.  It reproduces
# exactly the identity binding the real executor enforces -- artifact_id is
# re-derived from the STAGED copies and a confirmation token that does not
# match them is refused -- so a leg that ever ran against a stale or partial
# payload fails here the same way it would on the arm, without any motion.
_STUB_EXECUTOR = '''#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path

args = sys.argv[1:]
flags = {args[i]: args[i + 1] for i in range(0, len(args) - 1) if args[i].startswith("--")}
report = Path(flags["--planning-report"]).read_bytes()
archive = Path(flags["--planned-grasp"]).read_bytes()
artifact_id = hashlib.sha256(report + b"\\0" + archive).hexdigest()
receipt_dir = Path(flags["--receipt-dir"])
with (receipt_dir.parents[2] / "executor-calls.log").open("a", encoding="utf-8") as log:
    log.write(f"{flags.get('--workflow-phase')} {receipt_dir}\\n")
if flags.get("--confirm") != f"PIPER-FULL-{artifact_id[:16]}":
    print("ERROR: real full grasp requires exact confirmation token", file=sys.stderr)
    raise SystemExit(2)
prior = flags.get("--prior-receipt-dir")
prior_sha = None
if prior is not None:
    prior_sha = hashlib.sha256((Path(prior) / "workflow-state.json").read_bytes()).hexdigest()
receipt_dir.mkdir(parents=True, exist_ok=False)
common = {
    "artifact_id": artifact_id,
    "planning_report_sha256": hashlib.sha256(report).hexdigest(),
    "planned_grasp_sha256": hashlib.sha256(archive).hexdigest(),
    "planning_session_id": flags.get("--planning-session-id", ""),
}
(receipt_dir / "executor-start-receipt.json").write_text(
    json.dumps(dict(common, schema="z_manip.piper_executor_start_receipt.v1"), sort_keys=True),
    encoding="utf-8",
)
(receipt_dir / "workflow-state.json").write_text(
    json.dumps(dict(
        common,
        schema="z_manip.piper_grasp_workflow_state.v1",
        phase=flags["--workflow-phase"],
        prior_workflow_sha256=prior_sha,
    ), sort_keys=True),
    encoding="utf-8",
)
'''

_STUB_NOOP = "#!/bin/sh\nexit 0\n"


class FakeNuc:
    """A local directory plus a stub ssh/scp that runs the real remote shell.

    Everything the wrapper sends is executed for real against ``root``: the
    staging probe's sha256sum, the completion marker, the detached reaper.  The
    only substitution is the executor itself, which is swapped for the stand-in
    above because it must never drive an arm.  ``systemctl``/``sudo`` are
    stubbed on PATH and the passive-restore log is redirected into the sandbox
    so the remote shell touches nothing on the host.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir(parents=True)
        for name in ("systemctl", "sudo"):
            (self.bin / name).write_text(_STUB_NOOP, encoding="utf-8")
            (self.bin / name).chmod(0o755)
        self.stub_executor = root / "stub_executor.py"
        self.stub_executor.write_text(_STUB_EXECUTOR, encoding="utf-8")
        self.home = root / "home"
        (self.home / "pyAgxArm").mkdir(parents=True)
        self.actions = root / "full-grasp-actions"
        self.uploads = 0
        self.downloads = 0
        self.truncate_upload_after = None

    def executor_calls(self) -> list[str]:
        log = self.actions / "executor-calls.log"
        return log.read_text(encoding="utf-8").splitlines() if log.is_file() else []

    @staticmethod
    def _operands(arguments: list[str]) -> list[str]:
        rest = list(arguments[1:])
        operands = []
        while rest:
            part = rest.pop(0)
            if part in ("-q", "-p"):
                continue
            if part in ("-i", "-o"):
                rest.pop(0)
                continue
            operands.append(part)
        return operands

    def __call__(self, arguments, *, timeout):
        arguments = [str(part) for part in arguments]

        def done(rc, out=""):
            return subprocess.CompletedProcess(arguments, rc, stdout=out, stderr=None)

        if arguments[0] == sys.executable:
            flags = {
                arguments[i]: arguments[i + 1]
                for i in range(len(arguments) - 1)
                if arguments[i].startswith("--")
            }
            artifact_id = hashlib.sha256(
                Path(flags["--planning-report"]).read_bytes()
                + b"\0"
                + Path(flags["--planned-grasp"]).read_bytes()
            ).hexdigest()
            return done(0, json.dumps({
                "confirmation_token": f"PIPER-FULL-{artifact_id[:16]}",
            }))
        if arguments[0] == "scp":
            return done(*self._copy(self._operands(arguments)))
        if arguments[0] == "ssh":
            return done(*self._shell(self._operands(arguments)[-1]))
        raise AssertionError(f"unexpected wrapper subprocess: {arguments}")

    def _copy(self, operands: list[str]) -> tuple[int, str]:
        *sources, destination = operands
        if destination.startswith(f"{wrapper.NUC_HOST}:"):
            target = Path(destination.split(":", 1)[1])
            into_directory = destination.endswith("/") or target.is_dir()
            # Only the payload lands as a directory drop; the per-leg prior
            # receipt is a single named file and is never cached.
            self.uploads += int(into_directory)
            if not (target if into_directory else target.parent).is_dir():
                return 1, "scp: no such directory"
            for index, source in enumerate(sources):
                data = Path(source).read_bytes()
                landing = target / Path(source).name if into_directory else target
                if self.truncate_upload_after is not None and index >= self.truncate_upload_after:
                    # A connection dropped mid-transfer leaves a short file and
                    # a nonzero scp, exactly as the wrapper must survive.
                    landing.write_bytes(data[: len(data) // 2])
                    return 1, "scp: lost connection"
                landing.write_bytes(data)
            return 0, ""
        self.downloads += 1
        remote = Path(sources[0].split(":", 1)[1])
        matches = sorted(remote.parent.glob(remote.name))
        if not matches:
            return 1, f"scp: {remote}: No such file or directory"
        for match in matches:
            shutil.copy2(match, Path(destination))
        return 0, ""

    def _shell(self, command: str) -> tuple[int, str]:
        command = re.sub(
            r"/usr/bin/python3 \S+piper_full_grasp_executor\.py",
            f"{sys.executable} {self.stub_executor}",
            command,
        )
        command = command.replace(
            "/tmp/z-manip-passive-restore.log", str(self.root / "passive-restore.log"),
        )
        environment = dict(os.environ)
        environment.update({
            "PATH": f"{self.bin}:{environment['PATH']}",
            "HOME": str(self.home),
        })
        # The reaper is deliberately detached; joining it here keeps the
        # assertions below deterministic without weakening the command itself.
        completed = subprocess.run(
            ["sh", "-c", f"{command}\nwait"],
            env=environment,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return completed.returncode, completed.stdout


def _leg(
    tmp_path: Path,
    monkeypatch,
    nuc: FakeNuc,
    *,
    phase: str,
    receipt_dir: Path,
    prior_receipt_dir: Path | None = None,
) -> int:
    key = tmp_path / "nuc-key"
    if not key.exists():
        key.write_text("key", encoding="utf-8")
    monkeypatch.setattr(wrapper, "NUC_KEY", key)
    monkeypatch.setattr(wrapper, "REMOTE_ROOT", str(nuc.actions))
    monkeypatch.setattr(wrapper, "run", nuc)
    monkeypatch.setattr(wrapper.time, "sleep", lambda _s: None)
    argv = [
        "piper_full_grasp_remote.py",
        "--planning-report", str(tmp_path / "planning_report.json"),
        "--planned-grasp", str(tmp_path / "planned_grasp.npz"),
        "--receipt-dir", str(receipt_dir),
        "--speed-percent", "20",
        "--workflow-phase", phase,
        "--planning-session-id", "session-2026-07-27",
    ]
    if prior_receipt_dir is not None:
        argv.extend(("--prior-receipt-dir", str(prior_receipt_dir)))
    monkeypatch.setattr(sys, "argv", argv)
    return wrapper.main()


def _fake_nuc(tmp_path: Path) -> FakeNuc:
    (tmp_path / "planning_report.json").write_text(
        json.dumps({"schema": "z_manip.planning_report.v1", "grasp": [0.1] * 6}),
        encoding="utf-8",
    )
    (tmp_path / "planned_grasp.npz").write_bytes(b"PK\x03\x04" + b"grasp" * 64)
    return FakeNuc(tmp_path / "nuc")


def _staged(nuc: FakeNuc) -> Path:
    payloads = [path for path in nuc.actions.iterdir() if path.is_dir()]
    assert len(payloads) == 1, payloads
    return payloads[0]


def _receipt(receipt_dir: Path) -> dict:
    return json.loads((receipt_dir / "executor-start-receipt.json").read_text(encoding="utf-8"))


def test_three_legs_share_one_upload_and_one_identity(tmp_path, monkeypatch):
    """Pick -> carry -> place-back against a fake NUC: one upload, one identity."""
    nuc = _fake_nuc(tmp_path)
    legs = [
        ("pick-hold", tmp_path / "leg1", None),
        ("return-home-holding", tmp_path / "leg2", tmp_path / "leg1"),
        ("place-back", tmp_path / "leg3", tmp_path / "leg2"),
    ]
    for phase, receipt_dir, prior in legs:
        assert _leg(
            tmp_path, monkeypatch, nuc,
            phase=phase, receipt_dir=receipt_dir, prior_receipt_dir=prior,
        ) == 0

    # The payload crossed the link once for the whole cycle, not once per leg.
    assert nuc.uploads == 1
    assert len(nuc.executor_calls()) == 3

    receipts = [_receipt(receipt_dir) for _phase, receipt_dir, _prior in legs]
    identities = {
        (
            receipt["artifact_id"],
            receipt["planning_report_sha256"],
            receipt["planned_grasp_sha256"],
        )
        for receipt in receipts
    }
    assert len(identities) == 1
    report = (tmp_path / "planning_report.json").read_bytes()
    archive = (tmp_path / "planned_grasp.npz").read_bytes()
    assert identities == {(
        hashlib.sha256(report + b"\0" + archive).hexdigest(),
        hashlib.sha256(report).hexdigest(),
        hashlib.sha256(archive).hexdigest(),
    )}

    # Each continuation names its predecessor's receipt, so the held-object
    # chain the executor re-validates is unbroken across the shared payload.
    chain = [
        json.loads((tmp_path / f"leg{index}" / "workflow-state.json").read_text(encoding="utf-8"))
        for index in (1, 2, 3)
    ]
    assert chain[0]["prior_workflow_sha256"] is None
    for index in (2, 3):
        expected = hashlib.sha256(
            (tmp_path / f"leg{index - 1}" / "workflow-state.json").read_bytes(),
        ).hexdigest()
        assert chain[index - 1]["prior_workflow_sha256"] == expected


def test_mutated_remote_file_forces_a_full_re_upload(tmp_path, monkeypatch):
    nuc = _fake_nuc(tmp_path)
    assert _leg(tmp_path, monkeypatch, nuc, phase="pick-hold", receipt_dir=tmp_path / "leg1") == 0
    assert nuc.uploads == 1
    first = _receipt(tmp_path / "leg1")

    # Anything that edits a staged file -- a truncated write, a stray hand --
    # must invalidate the cache even though the marker still reads correct.
    staged = _staged(nuc)
    marker = (staged / ".manifest-sha").read_text(encoding="utf-8")
    (staged / "planned_grasp.npz").write_bytes(b"tampered")

    assert _leg(
        tmp_path, monkeypatch, nuc,
        phase="return-home-holding", receipt_dir=tmp_path / "leg2",
        prior_receipt_dir=tmp_path / "leg1",
    ) == 0
    assert nuc.uploads == 2
    assert (staged / "planned_grasp.npz").read_bytes() == (tmp_path / "planned_grasp.npz").read_bytes()
    assert (staged / ".manifest-sha").read_text(encoding="utf-8") == marker
    # The re-upload restored the exact planned artifact, so the leg still
    # executed against the identity the host validates.
    assert _receipt(tmp_path / "leg2")["artifact_id"] == first["artifact_id"]


def test_interrupted_upload_makes_the_next_leg_re_upload_before_executing(tmp_path, monkeypatch):
    nuc = _fake_nuc(tmp_path)
    nuc.truncate_upload_after = 2

    assert _leg(tmp_path, monkeypatch, nuc, phase="pick-hold", receipt_dir=tmp_path / "leg1") == 2
    staged = _staged(nuc)
    assert not (staged / ".manifest-sha").exists()
    # A half-uploaded payload must never reach the executor.
    assert nuc.executor_calls() == []
    assert not (tmp_path / "leg1").exists()

    nuc.truncate_upload_after = None
    assert _leg(tmp_path, monkeypatch, nuc, phase="pick-hold", receipt_dir=tmp_path / "leg2") == 0
    assert nuc.uploads == 2
    assert len(nuc.executor_calls()) == 1
    assert _receipt(tmp_path / "leg2")["planned_grasp_sha256"] == hashlib.sha256(
        (tmp_path / "planned_grasp.npz").read_bytes(),
    ).hexdigest()


def test_payload_key_separates_planning_sessions(tmp_path):
    payload = []
    for name, data in (("report.json", b"{}"), ("grasp.npz", b"npz")):
        path = tmp_path / name
        path.write_bytes(data)
        payload.append(path)
    # Identical bytes replanned in a new session are a different plan; they may
    # not land on -- or be served from -- the earlier session's directory.
    assert wrapper._payload_manifest_sha(payload, "session-a") != wrapper._payload_manifest_sha(
        payload, "session-b",
    )


def test_secured_leg_reaps_its_receipts_and_keeps_the_payload(tmp_path, monkeypatch):
    nuc = _fake_nuc(tmp_path)
    assert _leg(tmp_path, monkeypatch, nuc, phase="pick-hold", receipt_dir=tmp_path / "leg1") == 0
    staged = _staged(nuc)
    assert (staged / "planned_grasp.npz").is_file()
    assert list((staged / "receipts").iterdir()) == []


def test_stopped_leg_leaves_its_remote_receipts_in_place(tmp_path, monkeypatch):
    nuc = _fake_nuc(tmp_path)
    assert _leg(tmp_path, monkeypatch, nuc, phase="pick-hold", receipt_dir=tmp_path / "leg1") == 0
    staged = _staged(nuc)

    # A continuation whose receipts cannot be fetched keeps its remote evidence:
    # it is the only record of an object that may still be held.
    def refuse_download(operands, _original=nuc._copy):
        if not operands[-1].startswith(f"{wrapper.NUC_HOST}:"):
            return 1, "scp: lost connection"
        return _original(operands)

    monkeypatch.setattr(nuc, "_copy", refuse_download)
    assert _leg(
        tmp_path, monkeypatch, nuc,
        phase="return-home-holding", receipt_dir=tmp_path / "leg2",
        prior_receipt_dir=tmp_path / "leg1",
    ) == 2
    remaining = sorted(path.name for path in (staged / "receipts").iterdir())
    assert len(remaining) == 1
    assert (staged / "receipts" / remaining[0] / "executor-start-receipt.json").is_file()
