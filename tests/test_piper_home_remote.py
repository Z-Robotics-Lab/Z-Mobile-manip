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


def _environment(tmp_path: Path, bin_dir: Path, interactive_root: Path) -> dict[str, str]:
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
    return environment


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
    environment = _environment(tmp_path, bin_dir, interactive_root)
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


def test_transport_multiplexes_ssh_and_scp_on_a_dedicated_control_path(tmp_path: Path) -> None:
    result, ssh_log, scp_log = _run(tmp_path, tmp_path / "sessions")

    assert result.returncode == 0
    for log in (ssh_log, scp_log):
        calls = [line for line in log.splitlines() if line.strip()]
        assert calls
        for call in calls:
            assert "ControlMaster=auto" in call
            assert "ControlPersist=60" in call
            assert "z-manip-home-%C" in call
            # A shared master would let a Home teardown drop a live grasp leg.
            assert "z-manip-grasp-" not in call


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


# The staging stubs below run the benign remote bookkeeping (marker read/write,
# rm/mkdir, file copy) against a local fake remote directory so the upload guard
# is exercised end to end. The recovery call itself drives the arm, so the stub
# ssh recognises it by its `--execute` flag and exits 0 without running it.
_STAGING_SSH = """#!/usr/bin/env bash
printf 'ssh %s\\n' "$*" >> {log}
cmd="${{!#}}"
if [[ "$cmd" == *"--execute"* ]]; then
  exit 0
fi
bash -c "$cmd"
"""

_STAGING_SCP = """#!/usr/bin/env bash
printf 'scp %s\\n' "$*" >> {log}
count=$(( $(cat {counter} 2>/dev/null || printf 0) + 1 ))
printf '%s' "$count" > {counter}
if [[ "$count" == "${{Z_MANIP_TEST_SCP_FAIL_ON:-none}}" ]]; then
  exit 1
fi
sources=()
while (($#)); do
  case "$1" in
    -q) shift;;
    -i|-o) shift 2;;
    *) sources+=("$1"); shift;;
  esac
done
target="${{sources[-1]#*:}}"
unset 'sources[-1]'
cp "${{sources[@]}}" "$target"
"""


def _run_staged(
    tmp_path: Path,
    interactive_root: Path,
    remote_dir: Path,
    *,
    fail_scp_call: int | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "transport.log"
    counter = tmp_path / "scp.count"
    (bin_dir / "ssh").write_text(_STAGING_SSH.format(log=log), encoding="utf-8")
    (bin_dir / "scp").write_text(_STAGING_SCP.format(log=log, counter=counter), encoding="utf-8")
    (bin_dir / "ssh").chmod(0o755)
    (bin_dir / "scp").chmod(0o755)
    environment = _environment(tmp_path, bin_dir, interactive_root)
    environment["GO2W_HOME_REMOTE_DIR"] = str(remote_dir)
    if fail_scp_call is not None:
        environment["Z_MANIP_TEST_SCP_FAIL_ON"] = str(fail_scp_call)
    result = subprocess.run(
        (str(SCRIPT), "5"),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, [line for line in lines if line.strip()]


def test_manifest_marker_is_stamped_only_after_every_file_lands(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _write_plan(root / "planning/20260720-010000/artifacts/planning", valid=True)
    remote = tmp_path / "remote"

    result, calls = _run_staged(tmp_path, root, remote)

    assert result.returncode == 0, result.stderr
    uploads = [index for index, call in enumerate(calls) if call.startswith("scp ")]
    stamps = [
        index
        for index, call in enumerate(calls)
        if call.startswith("ssh ") and "printf" in call and ".manifest-sha" in call
    ]
    assert len(stamps) == 1
    assert uploads
    assert stamps[0] > max(uploads)
    for name in (
        "piper_home.json",
        "piper_home_recovery.py",
        "piper_reverse_home_recovery.py",
        "piper_staged_grasp_executor.py",
        "planning_report.json",
        "planned_grasp.npz",
    ):
        assert (remote / name).is_file()
    assert (remote / ".manifest-sha").read_text(encoding="utf-8").strip()


def test_unchanged_payload_skips_the_re_upload(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    remote = tmp_path / "remote"

    first, calls = _run_staged(tmp_path, root, remote)
    assert first.returncode == 0, first.stderr
    uploaded = len([call for call in calls if call.startswith("scp ")])
    assert uploaded == 2

    second, calls = _run_staged(tmp_path, root, remote)
    assert second.returncode == 0, second.stderr
    assert len([call for call in calls if call.startswith("scp ")]) == uploaded
    assert "piper_home_recovery.py" in calls[-1]


def test_interrupted_upload_forces_a_full_re_stage(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    remote = tmp_path / "remote"

    interrupted, calls = _run_staged(tmp_path, root, remote, fail_scp_call=2)
    assert interrupted.returncode != 0
    assert (remote / "piper_home.json").is_file()
    assert not (remote / "piper_home_recovery.py").exists()
    assert not (remote / ".manifest-sha").exists()
    partial_uploads = len([call for call in calls if call.startswith("scp ")])

    resumed, calls = _run_staged(tmp_path, root, remote)
    assert resumed.returncode == 0, resumed.stderr
    assert len([call for call in calls if call.startswith("scp ")]) == partial_uploads + 2
    assert (remote / "piper_home_recovery.py").is_file()
    assert (remote / ".manifest-sha").read_text(encoding="utf-8").strip()
