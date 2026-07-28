"""The Go2W base-transport health check, exercised against stubbed systemd.

This check decides whether to RESTART the LIVE single-owner base bridge. Its
only caller is ``go2w_depth_servo.sh``, which runs it at the start of every
task, so a false "stale" verdict tears down base authority on a working robot
and then blocks for up to 12 s waiting for it to come back.

The regression these tests pin: the three markers the check looks for are
emitted ONCE at bridge startup, while the bridge logs a heartbeat every ~3 s.
Judging health from a fixed tail of the log therefore expires on a timer -- at
240 lines, roughly 12 minutes -- and every task after that restarted a healthy
bridge. Measured live on the NUC 2026-07-28: service active, NRestarts=0, the
markers 441 and 432 lines back, zero of the three inside the 240-line window.

The remote block is extracted from the script rather than restated here, so
these cases cannot drift away from the code that actually ships.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts" / "runtime" / "go2w_base_transport_preflight.sh"

_OK = "\U0001f552 Data Channel Verification: ✅ OK            (12:16:43)"
_OWNER = (
    "[INFO] [1785212208.154758670] [unitree_control]: LIVE single-owner bridge "
    "enabled: Move + capability-gated Euler + StopMove; BodyHeight/GetBodyHeight "
    "are not control dependencies"
)
_FAIL = "[WARN] [unitree_control]: Data channel is not open"
_HEARTBEAT = (
    "[INFO] [1785213567.169527] [unitree_control]: WebRTC motion service "
    "evidence: name='ai-w' form='0' code=0 epoch=1785213567"
)


def _remote_block() -> str:
    """The body of the heredoc that runs on the NUC."""

    source = PREFLIGHT.read_text(encoding="utf-8")
    start = source.index("<<'REMOTE'\n") + len("<<'REMOTE'\n")
    end = source.index("\nREMOTE\n", start)
    return source[start:end]


def _run(tmp_path: Path, *, active: str, invocation: str, journal: list[str]) -> str:
    """Run the remote block with stub systemctl/journalctl on PATH."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (tmp_path / "journal.txt").write_text("\n".join(journal) + "\n", encoding="utf-8")

    (bin_dir / "systemctl").write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            case "$*" in
              *is-active*)  printf '{active}\\n' ;;
              *InvocationID*) printf '{invocation}\\n' ;;
              *) exit 1 ;;
            esac
            """
        ),
        encoding="utf-8",
    )

    # Emulates the real journalctl for BOTH query styles -- the invocation
    # filter AND the `-u SERVICE -n N` tail -- so these cases discriminate on
    # the verdict rather than on which flags happen to be passed. Without the
    # tail emulation a test would "catch" the old implementation merely by
    # failing to understand it.
    (bin_dir / "journalctl").write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            want=""
            pattern=""
            tail_n=""
            prev=""
            for arg in "$@"; do
              case "$arg" in
                _SYSTEMD_INVOCATION_ID=*) want="${{arg#*=}}" ;;
                --grep=*) pattern="${{arg#*=}}" ;;
                -n) prev="-n" ; continue ;;
                *) if [[ "$prev" == "-n" ]]; then tail_n="$arg"; fi ;;
              esac
              prev="$arg"
            done
            if [[ -n "$want" && "$want" != "{invocation}" ]]; then
              exit 0
            fi
            out="$(cat "{tmp_path}/journal.txt")"
            if [[ -n "$tail_n" ]]; then
              out="$(tail -n "$tail_n" <<<"$out")"
            fi
            if [[ -n "$pattern" ]]; then
              grep -E "$pattern" <<<"$out" || true
            else
              printf '%s\\n' "$out"
            fi
            """
        ),
        encoding="utf-8",
    )
    for name in ("systemctl", "journalctl"):
        (bin_dir / name).chmod(0o755)

    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", SERVICE="bridge.service")
    result = subprocess.run(
        ["bash", "-s"],
        input=_remote_block(),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_a_long_running_healthy_bridge_is_not_declared_stale(tmp_path):
    """THE REGRESSION.

    Startup markers followed by far more than 240 heartbeat lines: exactly the
    live NUC state that was restarting a working bridge on every task.
    """

    journal = [_OK, _OWNER] + [_HEARTBEAT] * 500
    assert _run(tmp_path, active="active", invocation="abc123", journal=journal) == "ready"


def test_a_freshly_started_bridge_is_ready(tmp_path):
    journal = [_OK, _OWNER, _HEARTBEAT]
    assert _run(tmp_path, active="active", invocation="abc123", journal=journal) == "ready"


def test_an_inactive_service_is_stale(tmp_path):
    journal = [_OK, _OWNER]
    assert _run(tmp_path, active="inactive", invocation="abc123", journal=journal) == "stale"


def test_a_service_with_no_invocation_is_stale(tmp_path):
    """Fail closed: nothing is running, so nothing has proven anything."""

    assert _run(tmp_path, active="active", invocation="", journal=[_OK, _OWNER]) == "stale"


def test_a_bridge_that_never_verified_this_run_is_stale(tmp_path):
    """The check must still catch a genuinely broken channel.

    Markers from a PREVIOUS invocation must not count -- that is the whole point
    of scoping to the current one.
    """

    assert _run(tmp_path, active="active", invocation="abc123",
                journal=[_HEARTBEAT] * 50) == "stale"
    assert _run(tmp_path, active="active", invocation="abc123",
                journal=[_OK, _HEARTBEAT]) == "stale", "no owner line"
    assert _run(tmp_path, active="active", invocation="abc123",
                journal=[_OWNER, _HEARTBEAT]) == "stale", "no verification line"


def test_a_channel_that_dropped_after_verifying_is_stale(tmp_path):
    journal = [_OK, _OWNER, _HEARTBEAT, _FAIL, _HEARTBEAT]
    assert _run(tmp_path, active="active", invocation="abc123", journal=journal) == "stale"


def test_a_channel_that_recovered_after_dropping_is_ready(tmp_path):
    """Ordering, not mere presence, decides -- and it must survive log volume."""

    journal = [_OWNER, _FAIL, _HEARTBEAT, _OK] + [_HEARTBEAT] * 400
    assert _run(tmp_path, active="active", invocation="abc123", journal=journal) == "ready"


def test_the_health_window_is_not_a_fixed_line_count(tmp_path):
    """Guard the property, not the current implementation.

    Any future rewrite that judges health from a bounded tail of the log
    re-introduces a health check that expires on a timer.
    """

    source = PREFLIGHT.read_text(encoding="utf-8")
    assert "_SYSTEMD_INVOCATION_ID" in source
    remote = _remote_block()
    assert "-n 240" not in remote
    assert "InvocationID" in remote


@pytest.mark.parametrize("heartbeats", [0, 239, 240, 241, 5000])
def test_the_verdict_does_not_depend_on_how_long_the_bridge_has_been_up(
    tmp_path, heartbeats
):
    journal = [_OK, _OWNER] + [_HEARTBEAT] * heartbeats
    assert _run(tmp_path, active="active", invocation="abc123", journal=journal) == "ready"
