"""Supervision contract for the Fast-FoundationStereo depth stack.

FFS is the sole source of obstacle geometry for the planner:

    /camera/ffs_depth_aligned/image_raw
      -> EdgeTAM exact-time colour+depth+camera_info sync
      -> /z_manip/perception/scene_pointcloud
      -> scene_collision_points.npy -> the collision checker

Until this suite existed the stack had no manager entry, no bringup step and no
health gate: it ran only because somebody started it by hand.  When it stopped,
EdgeTAM's synchroniser simply never fired and perception looked like it was
"waiting" rather than broken.

These tests pin the three properties that make that failure loud, and -- more
importantly -- they pin the *consumers* of those properties.  This project's
characteristic regression is a fail-closed gate quietly becoming fail-open
because the consumer that mattered lived in a file the change did not touch, so
the dashboard-side contract is asserted here too.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "scripts" / "runtime" / "go2w_component_manager.sh"
STACK = ROOT / "scripts" / "runtime" / "ffs_depth_stack.sh"
CONTROL_SCRIPT = ROOT / "scripts" / "runtime" / "go2w_planning_control.py"
OBSERVER_SCRIPT = ROOT / "scripts" / "runtime" / "go2w_runtime_observer.py"
DEPTH_TOPIC = "/camera/ffs_depth_aligned/image_raw"


_SPEC = importlib.util.spec_from_file_location("go2w_runtime_observer", OBSERVER_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
OBSERVER = importlib.util.module_from_spec(_SPEC)
# Register before exec: the module defines @dataclass types, which resolve their
# own module out of sys.modules during class creation.
sys.modules.setdefault(_SPEC.name, OBSERVER)
_SPEC.loader.exec_module(OBSERVER)


# --------------------------------------------------------------------------
# Shell-level harness: source the manager (CLI dispatch is skipped when
# sourced) and drive it against fake docker/curl binaries.  Nothing here talks
# to the live stack.
# --------------------------------------------------------------------------
def _enclosing_block(source: str, needle: str, *, radius: int = 6) -> str:
    """Return the lines surrounding every occurrence of ``needle``."""

    lines = source.splitlines()
    window: list[str] = []
    for index, line in enumerate(lines):
        if needle in line:
            window.extend(lines[max(0, index - radius):index + radius])
    return "\n".join(window)


def _fake_bin(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _drive(tmp_path: Path, *, docker: str, curl: str, body: str) -> subprocess.CompletedProcess:
    docker_bin = _fake_bin(tmp_path / "docker.sh", docker)
    curl_bin = _fake_bin(tmp_path / "curl.sh", curl)
    script = (
        "set -uo pipefail\n"
        f'source "{MANAGER}"\n'
        f'DOCKER="{docker_bin}"\n'
        f'CURL="{curl_bin}"\n'
        f"{body}\n"
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


# A relay reporting a real published rate, and the inference service ready.
_DOCKER_PUBLISHING = """
case "$1" in
  inspect)
    case "$*" in
      *State.Running*) printf 'true\\n' ;;
      *RestartCount*) printf '0\\n' ;;
      *) printf 'running; restarts=0\\n' ;;
    esac
    ;;
  logs) printf '[INFO] [123.4] [ffs_depth_relay]: pairs_in=99 published=44 rate~4.4fps rt_p50=59ms\\n' ;;
esac
exit 0
"""

# Containers alive, but the relay has published nothing: the calibration guard
# tripped, or the IR pairs stopped arriving.  This is the silent failure.
_DOCKER_SILENT = """
case "$1" in
  inspect)
    case "$*" in
      *State.Running*) printf 'true\\n' ;;
      *RestartCount*) printf '7\\n' ;;
      *) printf 'running; restarts=7\\n' ;;
    esac
    ;;
  logs) printf '[INFO] [123.4] [ffs_depth_relay]: relay up: waiting\\n' ;;
esac
exit 0
"""

_DOCKER_ABSENT = """
case "$1" in
  inspect) exit 1 ;;
  logs) exit 1 ;;
esac
exit 1
"""

_CURL_READY = """printf '{"ready": true, "model": "x", "iters": 8}\\n'\n"""
_CURL_NOT_READY = """printf '{"ready": false}\\n'\n"""


def test_scripts_are_syntax_checked():
    subprocess.run(["bash", "-n", str(MANAGER)], check=True)
    subprocess.run(["bash", "-n", str(STACK)], check=True)


def test_ffs_is_a_first_class_component_with_every_verb():
    source = MANAGER.read_text(encoding="utf-8")
    lowered = source.lower()

    # Accepted by the fixed component surface, and advertised in all three
    # usage lines (status / restart / logs) like every other component.
    assert "|ffs|" in lowered
    assert lowered.count("|ffs|") >= 4  # valid_component + 3 usage lines
    # Status, restart and logs each have a real ffs arm.
    assert "ffs_ready" in source
    assert "ffs_publish_rate_fps" in source
    assert "ffs_service_ready" in source
    assert '"$FFS_SCRIPT" up' in source
    assert 'wait_until "FFS depth stack" ffs_ready' in source
    # Both restart chains start it, fail-closed, before perception: cold
    # bringup and the perception-all "Restart all" button.
    for chain in ("perception-all)", "cold_bringup_steps()"):
        body = source[source.index(chain):]
        ffs_at = body.index("restart_one ffs || return 1")
        perception_at = body.index("restart_one perception || return 1")
        assert ffs_at < perception_at, f"{chain} must start FFS before perception"
    # Listed in the status/bringup enumerations so `manip status` shows it.
    assert lowered.count("observer rgbd grounding ffs edgetam perception") == 2


def test_status_reports_measured_rate_when_depth_is_really_flowing(tmp_path):
    result = _drive(
        tmp_path,
        docker=_DOCKER_PUBLISHING,
        curl=_CURL_READY,
        body="status_one ffs",
    )
    assert result.returncode == 0, result.stderr
    name, state, summary = result.stdout.strip().split("\t", 2)
    assert (name, state) == ("ffs", "healthy")
    # Honest and specific: the topic and the measured rate, not just "running".
    assert DEPTH_TOPIC in summary
    assert "4.4 fps" in summary
    assert "restarts=0" in summary


def test_running_containers_that_publish_nothing_are_not_healthy(tmp_path):
    """The exact health/reality mismatch this component exists to prevent."""

    result = _drive(
        tmp_path,
        docker=_DOCKER_SILENT,
        curl=_CURL_READY,
        body="status_one ffs",
    )
    assert result.returncode == 0, result.stderr
    name, state, summary = result.stdout.strip().split("\t", 2)
    assert (name, state) == ("ffs", "degraded")
    assert "NOT publishing" in summary
    assert DEPTH_TOPIC in summary
    # Docker's own restart policy churning the container must be visible.
    assert "restarts=7" in summary

    # ffs_ready is the gate restart_one waits on; it must agree with status.
    gate = _drive(
        tmp_path,
        docker=_DOCKER_SILENT,
        curl=_CURL_READY,
        body="ffs_ready && echo GATE=open || echo GATE=closed",
    )
    assert "GATE=closed" in gate.stdout, gate.stderr


def test_inference_service_not_ready_is_degraded(tmp_path):
    result = _drive(
        tmp_path,
        docker=_DOCKER_PUBLISHING,
        curl=_CURL_NOT_READY,
        body="status_one ffs",
    )
    _name, state, summary = result.stdout.strip().split("\t", 2)
    assert state == "degraded"
    assert DEPTH_TOPIC in summary


def test_absent_containers_are_offline_and_name_the_topic(tmp_path):
    result = _drive(
        tmp_path,
        docker=_DOCKER_ABSENT,
        curl=_CURL_NOT_READY,
        body="status_one ffs",
    )
    name, state, summary = result.stdout.strip().split("\t", 2)
    assert (name, state) == ("ffs", "offline")
    assert f"{DEPTH_TOPIC} has no publisher" in summary


def test_healthy_gate_opens_only_when_depth_is_measurably_flowing(tmp_path):
    open_gate = _drive(
        tmp_path,
        docker=_DOCKER_PUBLISHING,
        curl=_CURL_READY,
        body="ffs_ready && echo GATE=open || echo GATE=closed",
    )
    assert "GATE=open" in open_gate.stdout, open_gate.stderr

    absent = _drive(
        tmp_path,
        docker=_DOCKER_ABSENT,
        curl=_CURL_READY,
        body="ffs_ready && echo GATE=open || echo GATE=closed",
    )
    assert "GATE=closed" in absent.stdout, absent.stderr


def test_publishing_floor_does_not_false_alarm_on_the_measured_live_range(tmp_path):
    """Live healthy relay measured 1.3-7.7 fps; none of that may read degraded.

    The relay caps at 10 Hz but is bounded by Wi-Fi IR delivery from the NUC, so
    a threshold picked near the cap would flag a working stack.  The floor only
    answers "is depth arriving at all".
    """

    for rate in ("1.3", "4.0", "7.7"):
        result = _drive(
            tmp_path,
            docker=_DOCKER_PUBLISHING.replace("rate~4.4fps", f"rate~{rate}fps"),
            curl=_CURL_READY,
            body="status_one ffs",
        )
        _name, state, _summary = result.stdout.strip().split("\t", 2)
        assert state == "healthy", f"{rate} fps must not read as degraded"

    zero = _drive(
        tmp_path,
        docker=_DOCKER_PUBLISHING.replace("rate~4.4fps", "rate~0.0fps"),
        curl=_CURL_READY,
        body="status_one ffs",
    )
    _name, state, summary = zero.stdout.strip().split("\t", 2)
    assert state == "degraded"
    assert "NOT publishing" in summary


# --------------------------------------------------------------------------
# The loud surface, and the consumer that renders it.
# --------------------------------------------------------------------------
def test_perception_all_leads_with_ffs_and_names_the_topic(tmp_path):
    """A dead depth stack must not hide behind a generic perception summary."""

    for ffs_state in ("offline", "degraded"):
        result = _drive(
            tmp_path,
            docker=_DOCKER_PUBLISHING,
            curl=_CURL_READY,
            body=(
                "status_state() {\n"
                f'  if [[ "$1" == ffs ]]; then printf "{ffs_state}\\n"; '
                'else printf "healthy\\n"; fi\n'
                "}\n"
                "status_one perception-all"
            ),
        )
        name, state, summary = result.stdout.strip().split("\t", 2)
        assert name == "perception-all"
        # The aggregate mirrors the FFS state instead of averaging it away.
        assert state == ffs_state
        assert DEPTH_TOPIC in summary
        assert "scene_pointcloud" in summary
        assert "manip component restart ffs" in summary

    healthy = _drive(
        tmp_path,
        docker=_DOCKER_PUBLISHING,
        curl=_CURL_READY,
        body='status_state() { printf "healthy\\n"; }\nstatus_one perception-all',
    )
    _name, state, summary = healthy.stdout.strip().split("\t", 2)
    assert state == "healthy"
    assert "FFS depth" in summary


def test_dashboard_still_renders_the_component_carrying_the_ffs_alarm():
    """Guard the consumer, not just the producer.

    ``perception-all`` is the channel the FFS alarm reaches the operator
    through: the dashboard drops any status line whose component name is not in
    VISUAL_COMPONENTS, and renders ``summary`` verbatim on the card.  ``ffs``
    itself is deliberately NOT in that set (adding a card needs edits to
    go2w_planning_control.py and web/debug_dashboard/index.html), so if
    ``perception-all`` were ever dropped the alarm would go silent while every
    test above still passed.
    """

    control = CONTROL_SCRIPT.read_text(encoding="utf-8")
    assert '"perception-all",' in control

    html = (ROOT / "web" / "debug_dashboard" / "index.html").read_text(encoding="utf-8")
    assert 'data-component-state="perception-all"' in html
    assert "detail.textContent = component.summary" in html


def test_no_unbounded_ffs_restart_loop_was_introduced():
    """Restarting a genuinely broken stack in a loop is worse than not trying.

    Recovery is operator-triggered (``manip component restart ffs``) plus the
    one fail-closed start during cold bringup.  Docker's own
    ``--restart unless-stopped`` policy is the only automatic retry, and its
    RestartCount is surfaced in the status summary so those restarts are never
    invisible.
    """

    source = MANAGER.read_text(encoding="utf-8")
    assert "while true" not in source
    # Exactly two call sites, both plain sequential steps in a restart chain
    # (cold bringup and perception-all) -- never inside a loop or retry helper.
    assert source.count("restart_one ffs") == 2
    for line in source.splitlines():
        if "restart_one ffs" in line:
            assert line.strip() == "restart_one ffs || return 1"
    assert "for " not in _enclosing_block(source, "restart_one ffs")
    assert "while " not in _enclosing_block(source, "restart_one ffs")
    # Docker's own restarts are surfaced rather than hidden.
    assert "RestartCount" in source


def test_stack_script_status_reports_the_same_publishing_signal():
    stack = STACK.read_text(encoding="utf-8")
    assert "rate~[0-9.]+fps" in stack
    assert "NOT publishing /camera/ffs_depth_aligned/image_raw" in stack
    # The low-level script points at the managed surface.
    assert "manip component restart ffs" in stack


# --------------------------------------------------------------------------
# Observer: the FFS -> raw-D435 demotion must be recorded, not silently taken.
# --------------------------------------------------------------------------
_CLOUD_K = [605.87, 0.0, 331.78, 0.0, 605.21, 240.24, 0.0, 0.0, 1.0]


def _color(stamp: int, width: int, height: int):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9),
            frame_id="camera_color_optical_frame",
        ),
        stamp=stamp,
        width=width,
        height=height,
        step=width * 3,
        encoding="rgb8",
        data=np.ascontiguousarray(rgb).tobytes(),
    )


def _depth(stamp: int, width: int, height: int):
    depth = np.full((height, width), 1000, dtype="<u2")
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9),
            frame_id="camera_color_optical_frame",
        ),
        width=width,
        height=height,
        step=width * 2,
        encoding="16UC1",
        data=np.ascontiguousarray(depth, dtype="<u2").tobytes(),
    )


@pytest.mark.parametrize(
    ("source", "degraded"),
    [("ffs", False), ("d435_raw", True)],
)
def test_cloud_manifest_records_the_ffs_fallback_explicitly(source, degraded):
    stamp = 1_700_000_000_000_000_001
    _payload, manifest = OBSERVER.encode_point_cloud(
        _color(stamp, 64, 48),
        _depth(stamp, 64, 48),
        _CLOUD_K,
        source=source,
        source_topic="/topic/whatever",
    )

    assert manifest["expected_source"] == "ffs"
    assert manifest["expected_source_topic"] == DEPTH_TOPIC
    assert manifest["degraded"] is degraded
    if degraded:
        reason = manifest["degraded_reason"]
        # Names the missing topic and the real downstream consequence.
        assert DEPTH_TOPIC in reason
        assert "scene_pointcloud" in reason
        assert "manip component restart ffs" in reason
    else:
        assert manifest["degraded_reason"] is None


def test_cloud_manifest_stays_json_serialisable_for_the_dashboard_route():
    """The manifest is served byte-for-byte at /api/cloud/latest.json."""

    stamp = 1_700_000_000_000_000_001
    _payload, manifest = OBSERVER.encode_point_cloud(
        _color(stamp, 64, 48),
        _depth(stamp, 64, 48),
        _CLOUD_K,
        source="d435_raw",
        source_topic="/camera/aligned_depth_to_color/image_raw",
    )
    decoded = json.loads(json.dumps(manifest))
    assert decoded["degraded"] is True
    assert decoded["schema"] == "z_manip.point_cloud_frame.v1"
