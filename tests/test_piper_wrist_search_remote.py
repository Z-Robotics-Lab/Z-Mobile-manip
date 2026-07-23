"""Transport-hygiene contract for the wrist-search remote launcher.

The launcher is invoked once per bounded wrist view (see
``FixedWristMotion.__call__`` in ``go2w_wrist_search.py``), so a fresh SSH
handshake and a re-upload of the four unchanged payload files on every view is
pure overhead. These tests pin the two fixes: a persisted ControlMaster/mux
transport on both ``ssh`` and ``scp``, and a per-search upload guard that skips
``rm``/``scp`` once the NUC already carries the matching payload.

Live motion is never exercised here: the launcher's terminal executor call is
stubbed out, and the stub ``ssh`` runs only the benign remote bookkeeping
(marker read/write, ``rm``/``mkdir``) against a local fake remote directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_wrist_search_remote.sh"

# The stub ssh must faithfully persist the manifest marker so the guard's
# skip/re-upload decision is exercised end-to-end, but it must never run the
# real per-view executor (which drives the arm). It distinguishes the two by the
# executor's `--view-index` flag, exiting 0 for that command without running it.
_STUB_SSH = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> {ssh_log}
cmd="${{!#}}"
if [[ "$cmd" == *"--view-index"* ]]; then
  exit 0
fi
bash -c "$cmd"
"""

_STUB_SCP = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> {scp_log}
exit 0
"""


def _install_stubs(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"
    scp_log = tmp_path / "scp.log"
    (bin_dir / "ssh").write_text(_STUB_SSH.format(ssh_log=ssh_log), encoding="utf-8")
    (bin_dir / "scp").write_text(_STUB_SCP.format(scp_log=scp_log), encoding="utf-8")
    (bin_dir / "ssh").chmod(0o755)
    (bin_dir / "scp").chmod(0o755)
    return bin_dir, ssh_log, scp_log


def _run(tmp_path: Path, bin_dir: Path, remote_dir: Path) -> subprocess.CompletedProcess[str]:
    key = tmp_path / "key"
    key.write_text("test", encoding="utf-8")
    environment = dict(os.environ)
    environment.update({
        "PATH": f"{bin_dir}:{environment['PATH']}",
        "GO2W_NUC_SSH_KEY": str(key),
        "GO2W_WRIST_SEARCH_REMOTE_DIR": str(remote_dir),
        "Z_MANIP_ENABLE_WRIST_SEARCH": "1",
    })
    return subprocess.run(
        (str(SCRIPT), "1", "5"),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _scp_calls(scp_log: Path) -> int:
    if not scp_log.exists():
        return 0
    return len([line for line in scp_log.read_text(encoding="utf-8").splitlines() if line.strip()])


def test_transport_persists_a_control_master_across_ssh_and_scp(tmp_path: Path) -> None:
    bin_dir, ssh_log, _scp_log = _install_stubs(tmp_path)
    remote_dir = tmp_path / "remote"

    result = _run(tmp_path, bin_dir, remote_dir)

    assert result.returncode == 0, result.stderr
    ssh_calls = [line for line in ssh_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Every ssh invocation (marker probe, rm/mkdir, marker stamp, executor) must
    # carry the multiplexing options so the master socket is shared and outlives
    # the process for the next per-view call.
    for call in ssh_calls:
        assert "ControlMaster=auto" in call
        assert "ControlPersist=60" in call
        assert "ControlPath=" in call


def test_first_view_uploads_then_subsequent_views_skip_the_scp(tmp_path: Path) -> None:
    bin_dir, _ssh_log, scp_log = _install_stubs(tmp_path)
    remote_dir = tmp_path / "remote"

    first = _run(tmp_path, bin_dir, remote_dir)
    assert first.returncode == 0, first.stderr
    assert _scp_calls(scp_log) == 1
    assert (remote_dir / ".manifest-sha").exists()

    # A second per-view invocation with the identical payload must reuse the
    # already-staged files: no rm, no scp.
    second = _run(tmp_path, bin_dir, remote_dir)
    assert second.returncode == 0, second.stderr
    assert _scp_calls(scp_log) == 1


def test_manifest_mismatch_forces_a_fresh_upload(tmp_path: Path) -> None:
    bin_dir, _ssh_log, scp_log = _install_stubs(tmp_path)
    remote_dir = tmp_path / "remote"

    first = _run(tmp_path, bin_dir, remote_dir)
    assert first.returncode == 0, first.stderr
    assert _scp_calls(scp_log) == 1

    # Simulate a stale/partial remote payload: the marker no longer matches the
    # local inputs, so the guard must re-upload rather than trust the NUC copy.
    (remote_dir / ".manifest-sha").write_text("stale", encoding="utf-8")

    second = _run(tmp_path, bin_dir, remote_dir)
    assert second.returncode == 0, second.stderr
    assert _scp_calls(scp_log) == 2
