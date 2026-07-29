"""Hermetic contract tests for scripts/runtime/go2w_task_fsm.sh.

The docker binary is an injected recording fake, so these verify the manager's
fixed surface — which docker verbs it may emit and what it must never touch —
without a daemon, an image, or ROS.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runtime" / "go2w_task_fsm.sh"


def _fake_docker(tmp_path: Path, log: Path, *, running: str = "false") -> Path:
    """A docker fake that records argv and answers inspect with a fixed state."""
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        "case \"$1\" in\n"
        "  inspect)\n"
        "    # State.Running template query and existence checks both land here.\n"
        f'    printf "{running}\\n"\n'
        "    exit 0 ;;\n"
        "  logs)\n"
        "    # Never claim supervisor readiness in the hermetic fake.\n"
        "    exit 0 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def _env(tmp_path: Path, log: Path, *, running: str = "false") -> dict[str, str]:
    env = dict(os.environ)
    env["Z_MANIP_DOCKER_BIN"] = str(_fake_docker(tmp_path, log, running=running))
    # Keep the readiness poll bounded so a start that (correctly) never sees the
    # ready marker fails fast instead of spinning for the live default.
    env["Z_MANIP_TASK_STARTUP_TIMEOUT"] = "1"
    return env


def test_syntax_and_fixed_verbs() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    # Documentation comments may name the things this manager must never DO;
    # judge only the executable lines.
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    # No actuator or base-chain surface, ever.
    for banned in ("cansend", "move_j", "motionenable", "reactive-live",
                   "reactive-control", "ssh "):
        assert banned not in code.lower()
    # The supervisor is the only launch owner; the manager must not bypass it
    # with a bare `ros2 launch` of its own.
    assert "z-manip-mobile-manipulation" in code
    assert "ros2 launch" not in code


def test_status_offline_without_container(tmp_path: Path) -> None:
    log = tmp_path / "docker.log"
    result = subprocess.run(
        ["bash", str(SCRIPT), "status"],
        check=True, capture_output=True, text=True,
        env=_env(tmp_path, log, running="false"),
    )
    component, state, _summary = result.stdout.strip().split("\t")
    assert component == "task"
    assert state == "offline"


def test_down_is_a_clean_stop_never_a_kill(tmp_path: Path) -> None:
    log = tmp_path / "docker.log"
    subprocess.run(
        ["bash", str(SCRIPT), "down"],
        check=True, capture_output=True, text=True,
        env=_env(tmp_path, log, running="true"),
    )
    calls = log.read_text(encoding="utf-8")
    # A clean SIGTERM stop must precede removal; `docker kill` is forbidden.
    assert "stop --time" in calls
    assert "kill" not in calls
    # The manager owns exactly one container name.
    assert "z-manip-task" in calls
    for other in ("z-manip-hw", "z-manip-rgbd", "z-manip-edgetam",
                  "z-manip-perception-runner", "z-manip-planning-runner",
                  "z-manip-ffs"):
        assert other not in calls


def test_start_is_idempotent_when_already_running(tmp_path: Path) -> None:
    log = tmp_path / "docker.log"
    result = subprocess.run(
        ["bash", str(SCRIPT), "start"],
        check=True, capture_output=True, text=True,
        env=_env(tmp_path, log, running="true"),
    )
    assert "already running" in result.stdout
    calls = log.read_text(encoding="utf-8")
    assert "run" not in [line.split()[0] for line in calls.splitlines() if line]


def test_start_invokes_the_singleton_supervisor(tmp_path: Path) -> None:
    log = tmp_path / "docker.log"
    result = subprocess.run(
        ["bash", str(SCRIPT), "start"],
        capture_output=True, text=True,
        env=_env(tmp_path, log, running="false"),
    )
    # The fake never reports the ready marker, so start must fail closed …
    assert result.returncode != 0
    calls = log.read_text(encoding="utf-8")
    run_lines = [line for line in calls.splitlines() if line.startswith("run ")]
    assert len(run_lines) == 1
    run = run_lines[0]
    # … but the run itself must carry the supervisor and its launch arguments.
    assert "z-manip-mobile-manipulation" in run
    assert "--namespace" in run
    assert "use_sim_time:=false" in run
    assert "stack_config_path:=" in run
    assert "robot_description_file:=" in run
    assert "collision_model_file:=" in run
    # Env contract the node fails without (see z_manip.configuration):
    assert "Z_MANIP_ROBOT_URDF=" in run
    # Restart policy stays off: a crash-looping FSM must be visible, not hidden.
    assert "--restart no" in run


@pytest.mark.parametrize("argv", [[], ["bogus"], ["start", "extra"]])
def test_usage_rejects_everything_else(tmp_path: Path, argv: list[str]) -> None:
    log = tmp_path / "docker.log"
    result = subprocess.run(
        ["bash", str(SCRIPT), *argv],
        capture_output=True, text=True,
        env=_env(tmp_path, log),
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
