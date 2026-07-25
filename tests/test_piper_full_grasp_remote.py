"""Receipt-durability contract of the full-grasp remote wrapper.

Live evidence 2026-07-23: a transient ssh blip during the start-receipt probe
made the wrapper skip the local fetch while its cleanup still deleted the
remote action directory, destroying the only evidence for a physically held
object and surfacing the misleading "handoff evidence is not a regular file"
error.  The invariant under test: the remote directory is deleted ONLY when
receipts landed locally or the probe PROVED no start receipt exists.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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

# Login home of a NUC that is *not* the lab account baked into the old
# hardcoded remote root, so a regression back to it is visible.
REMOTE_HOME = "/home/robot"


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
    ) -> None:
        self.executor_rc = executor_rc
        self.probe_rcs = list(probe_rcs)
        self.fetch_rc = fetch_rc
        self.receipt_dir = receipt_dir
        self.execute_raises_timeout = execute_raises_timeout
        self.fetch_attempts = 0
        self.remote_deleted = False
        self.calls: list[str] = []

    def __call__(self, arguments, *, timeout):
        def done(rc, out=""):
            return subprocess.CompletedProcess(arguments, rc, stdout=out, stderr=None)

        tail = str(arguments[-1])
        self.calls.append(" ".join(str(part) for part in arguments))
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
            return done(0)
        if "rm -rf" in tail:
            self.remote_deleted = True
            return done(0)
        if "test -f" in tail:
            rc = self.probe_rcs.pop(0) if self.probe_rcs else 255
            return done(rc)
        if "mkdir -p" in tail:
            # The action-directory call also asks the NUC for its own login
            # home, so the fake must answer it the way a real shell would.
            if wrapper.REMOTE_HOME_MARKER in tail:
                return done(0, f"{wrapper.REMOTE_HOME_MARKER}{REMOTE_HOME}\n")
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
    assert fake.fetch_attempts == 0
    assert fake.remote_deleted is True


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


def test_remote_workspace_follows_the_nuc_login_home(tmp_path, monkeypatch):
    """GO2W_NUC_HOST is a documented override and .env.example ships a
    different account, so nothing may pin the staging root to one lab user.

    The wrapper asks the NUC for its own ``$HOME`` while it creates the action
    directory, and every later absolute path (the executor runs after
    ``cd ~/pyAgxArm``) must be built from that answer.
    """
    fake = FakeRuns(
        executor_rc=0,
        probe_rcs=(0,),
        fetch_rc=0,
        receipt_dir=tmp_path / "receipts",
    )
    assert _invoke(tmp_path, monkeypatch, fake) == 0

    assert wrapper.REMOTE_ROOT == "z-manip-runtime/full-grasp-actions"
    # The default *host* may still name the lab account (it is a documented
    # override); no remote *path* may.
    joined = "\n".join(fake.calls)
    assert "/home/yusenzlabnuc" not in joined

    # The creation call is the only one allowed to defer to the remote shell;
    # it must both create the directory under $HOME and report that home back.
    creations = [call for call in fake.calls if "mkdir -p" in call and wrapper.REMOTE_HOME_MARKER in call]
    assert len(creations) == 1
    assert '"$HOME"/z-manip-runtime/full-grasp-actions/' in creations[0]

    # Everything afterwards is absolute and rooted at the reported home: the
    # upload target, the executor invocation, the receipt probe and fetch, and
    # the cleanup.
    prefix = f"{REMOTE_HOME}/z-manip-runtime/full-grasp-actions/"
    for needle in ("scp", "cd ~/pyAgxArm", "test -f", "rm -rf"):
        matching = [call for call in fake.calls if needle in call]
        assert matching, needle
        for call in matching:
            assert prefix in call, (needle, call)


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
