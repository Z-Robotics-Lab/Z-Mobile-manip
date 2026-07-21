from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/runtime/piper_home_remote.sh"


def _write_plan(directory: Path, *, valid: bool, archive: bool = True) -> None:
    directory.mkdir(parents=True)
    payload = b"checked-trajectory" if valid else b"untrusted-trajectory"
    if archive:
        (directory / "planned_grasp.npz").write_bytes(payload)
    report = {
        "read_only": True,
        "planning_only": True,
        "motion_commands_published": 0,
        "plan_valid": valid,
        "raw_paths_collision_validated": valid,
        "planned_grasp_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (directory / "planning_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )


def _run(tmp_path: Path, interactive_root: Path) -> tuple[subprocess.CompletedProcess[str], str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"
    scp_log = tmp_path / "scp.log"
    for name, log in (("ssh", ssh_log), ("scp", scp_log)):
        executable = bin_dir / name
        executable.write_text(
            f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> {log!s}\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    key = tmp_path / "key"
    key.write_text("test", encoding="utf-8")
    home = tmp_path / "piper_home.json"
    home_payload = json.loads((ROOT / "configs/piper_home.example.json").read_text(encoding="utf-8"))
    home_payload["capture_zero_can_tx_verified"] = True
    home_payload["captured_at"] = "test-fixture"
    home.write_text(json.dumps(home_payload), encoding="utf-8")
    environment = dict(os.environ)
    environment.update({
        "PATH": f"{bin_dir}:{environment['PATH']}",
        "GO2W_NUC_SSH_KEY": str(key),
        "PIPER_HOME_CONFIG": str(home),
        "Z_MANIP_INTERACTIVE_RUN_ROOT": str(interactive_root),
    })
    result = subprocess.run(
        (str(SCRIPT), "5"),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return (
        result,
        ssh_log.read_text(encoding="utf-8") if ssh_log.exists() else "",
        scp_log.read_text(encoding="utf-8") if scp_log.exists() else "",
    )


def test_failed_latest_plan_cannot_block_direct_home(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    failed = root / "planning/20260720-020000/artifacts/planning"
    _write_plan(failed, valid=False, archive=False)

    result, ssh_log, scp_log = _run(tmp_path, root)

    assert result.returncode == 0
    assert "no complete checked planning artifact" in result.stdout
    assert "planned_grasp.npz" not in scp_log
    assert "--planning-report" not in ssh_log
    assert "piper_home_recovery.py" in ssh_log


def test_home_skips_failed_latest_and_uses_previous_checked_path(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    checked = root / "planning/20260720-010000/artifacts/planning"
    failed = root / "planning/20260720-020000/artifacts/planning"
    _write_plan(checked, valid=True)
    _write_plan(failed, valid=False, archive=False)

    result, ssh_log, scp_log = _run(tmp_path, root)

    assert result.returncode == 0
    assert str(checked) in result.stdout
    assert str(checked / "planned_grasp.npz") in scp_log
    assert str(failed) not in scp_log
    assert "--planning-report" in ssh_log
    assert "piper_reverse_home_recovery.py" in ssh_log


def test_home_rejects_archive_whose_digest_does_not_match(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    corrupt = root / "planning/20260720-030000/artifacts/planning"
    _write_plan(corrupt, valid=True)
    (corrupt / "planned_grasp.npz").write_bytes(b"changed-after-report")

    result, ssh_log, scp_log = _run(tmp_path, root)

    assert result.returncode == 0
    assert "no complete checked planning artifact" in result.stdout
    assert "planned_grasp.npz" not in scp_log
    assert "--planning-report" not in ssh_log
