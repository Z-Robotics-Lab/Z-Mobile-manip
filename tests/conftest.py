"""pytest fixtures + marker registration for the z-manip contract suite.

The ``chain`` fixture is the single gate on every live test: it runs the go2w
read-only health probe (``~/Desktop/go2w/scripts/nav/status.sh``) and skips the
test unless the whole chain reports green. It NEVER starts, restarts, or tears
anything down — attach-only. If the probe itself can't run (no docker, not on
this host), that is also a skip, not an error.

The suite is import-safe with zero third-party deps: collection (and every
skeleton spec) works on a bare Python; only live tests need the chain + rclpy
(which lives INSIDE the container, reached through the exec seam).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

def _default_status_script() -> str:
    """Find the sibling go2W checkout, retaining the historical fallback."""

    candidates = (
        Path(__file__).resolve().parents[2] / "go2W_Sim/scripts/nav/status.sh",
        Path.home() / "Desktop/go2w/scripts/nav/status.sh",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


# go2w health probe (read-only). Overridable for real-robot / relocated checkouts.
_STATUS_SH = os.environ.get("Z_MANIP_STATUS_SH", _default_status_script())


def _chain_status() -> dict:
    """Run status.sh and parse its one-line JSON. Empty dict ⇒ unavailable."""
    if not Path(_STATUS_SH).exists():
        return {}
    try:
        proc = subprocess.run(
            ["bash", _STATUS_SH],
            capture_output=True, text=True, timeout=40,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    out = (proc.stdout or "").strip().splitlines()
    for line in reversed(out):  # last JSON line is the verdict
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def pytest_configure(config: pytest.Config) -> None:
    """Register the milestone / layer markers (avoids PytestUnknownMarkWarning)."""
    for name, desc in (
        ("m0", "M0: camera stream + three poses + GT (live, attach-only)"),
        ("m05", "M0.5: extra props (parametrized, forward-looking)"),
        ("m1", "M1: find + two-stage approach + tracking (spec/skeleton)"),
        ("m2", "M2: grasp-candidate generation (spec/skeleton)"),
        ("m3", "M3: plan + execute pick/place (spec/skeleton)"),
        ("e2e", "end-to-end smoke / staged specs (bare-cli acceptance face)"),
        ("slow", "runs a multi-second live probe against the chain"),
    ):
        config.addinivalue_line("markers", f"{name}: {desc}")


@pytest.fixture(scope="session")
def chain() -> dict:
    """Gate: yield the chain status dict iff green, else skip the test.

    Green = status.sh reports ``green: true``. A missing/failed probe, or any
    non-green phase, skips — the suite is a passive observer of a chain another
    session owns and may pair-restart at any moment.
    """
    status = _chain_status()
    if not status:
        pytest.skip("chain probe unavailable (no status.sh / not this host)")
    if not (status.get("green") is True):
        pytest.skip(f"chain not green: {status}")
    return status
