from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "manip"
FSM_MANAGER = ROOT / "scripts" / "runtime" / "go2w_task_fsm.sh"


def _recording_fake(path: Path, log: Path, *, stdout: str = "", exit_code: int = 0) -> None:
    """Write a fake executable that appends '<name> <args>' to a shared log."""
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s %s\\n" "$(basename "$0")" "$*" >> "{log}"\n'
        + (f'printf "%s" {stdout!r}\n' if stdout else "")
        + f"exit {exit_code}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_manip_cli_is_syntax_checked_and_has_fixed_operator_surface() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    source = SCRIPT.read_text(encoding="utf-8")

    for command in (
        "manip start",
        "manip stop",
        "manip bringup",
        "manip status",
        "manip bag start",
        "manip bag stop",
        "manip bag status",
    ):
        assert command in help_result.stdout
    assert "go2w_component_manager.sh" in source
    assert "go2w_rosbag.sh" in source
    assert '"$COMPONENT_MANAGER" install' in source
    assert 'SYSTEMCTL\" --user start' in source
    assert 'SYSTEMCTL\" --user stop' in source
    assert "/api/home/status" in source
    assert "/api/grasp/status" in source
    assert "/api/sessions/status" in source
    assert "argv[1]" in source
    assert "piper_full_grasp_remote\\.py" in source
    assert "cansend" not in source.lower()
    assert "move_j" not in source.lower()
    assert "motionenable" not in source.lower()


def test_manip_url_requires_no_service_or_hardware() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "url"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "http://127.0.0.1:8766/"


def test_help_documents_the_graded_motion_tiers() -> None:
    help_text = subprocess.run(
        ["bash", str(SCRIPT), "--help"], check=True, capture_output=True, text=True
    ).stdout
    # The zero-motion default and the separate motion-enabling step must be
    # spelled out so an operator (and the persona) never conflate them.
    assert "ZERO-MOTION" in help_text
    assert "manip start-base" in help_text
    assert "manip down" in help_text
    assert "manip fsm start|status|down" in help_text
    # The FSM must never grow an arm/base actuator verb by accident.
    source = SCRIPT.read_text(encoding="utf-8")
    for banned in ("cansend", "move_j", "motionenable"):
        assert banned not in source.lower()


def _manip_env(tmp_path: Path, log: Path) -> dict[str, str]:
    """A manip invocation whose every side-effecting helper is a recording fake."""
    for name, binary in (
        ("Z_MANIP_FSM_MANAGER", "fsm"),
        ("Z_MANIP_COMPONENT_MANAGER", "component"),
    ):
        _recording_fake(tmp_path / binary, log)
    _recording_fake(tmp_path / "systemctl", log)
    _recording_fake(tmp_path / "curl", log)  # wait_for_ui health probe succeeds
    env = dict(os.environ)
    env.update(
        Z_MANIP_FSM_MANAGER=str(tmp_path / "fsm"),
        Z_MANIP_COMPONENT_MANAGER=str(tmp_path / "component"),
        Z_MANIP_SYSTEMCTL_BIN=str(tmp_path / "systemctl"),
        Z_MANIP_CURL_BIN=str(tmp_path / "curl"),
    )
    return env


def test_bare_start_is_zero_motion_fsm_plus_ui(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    env = _manip_env(tmp_path, log)
    subprocess.run(["bash", str(SCRIPT), "start"], check=True, env=env,
                   capture_output=True, text=True)
    calls = log.read_text(encoding="utf-8")
    # UI comes up (systemd) AND the task FSM is started, but nothing touches the
    # NUC base chain.
    assert "fsm start" in calls
    assert "component install" in calls
    assert "systemctl --user start" in calls
    assert "reactive-control" not in calls


def test_start_ui_does_not_start_the_fsm(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    env = _manip_env(tmp_path, log)
    subprocess.run(["bash", str(SCRIPT), "start", "ui"], check=True, env=env,
                   capture_output=True, text=True)
    calls = log.read_text(encoding="utf-8")
    assert "systemctl --user start" in calls
    assert "fsm start" not in calls


def test_start_base_is_the_only_motion_enabling_alias(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    env = _manip_env(tmp_path, log)
    subprocess.run(["bash", str(SCRIPT), "start-base"], check=True, env=env,
                   capture_output=True, text=True)
    calls = log.read_text(encoding="utf-8")
    assert "component restart reactive-control" in calls
    assert "fsm" not in calls  # base enablement never touches the FSM manager


def test_down_stops_only_the_fsm(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    env = _manip_env(tmp_path, log)
    subprocess.run(["bash", str(SCRIPT), "down"], check=True, env=env,
                   capture_output=True, text=True)
    calls = log.read_text(encoding="utf-8")
    assert "fsm down" in calls
    assert "component" not in calls


def test_status_task_routes_to_the_fsm_manager(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    env = _manip_env(tmp_path, log)
    subprocess.run(["bash", str(SCRIPT), "status", "task"], check=True, env=env,
                   capture_output=True, text=True)
    calls = log.read_text(encoding="utf-8")
    assert "fsm status" in calls
    assert "component" not in calls
