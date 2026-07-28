#!/usr/bin/env python3
"""Deterministic fingerprint for code/config loaded by resident workers.

The warm perception and planning workers import Python modules once at
startup.  A Git commit alone is therefore not enough to prove that the
running worker matches the checkout (and it misses dirty changes).  This
module hashes the exact source/config bytes that those workers load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
STACK_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = STACK_ROOT.parent


def runtime_inputs() -> tuple[Path, ...]:
    fixed = (
        SCRIPT_DIR / "z_manip_runtime_fingerprint.py",
        SCRIPT_DIR / "go2w_interactive_sessions.py",
        SCRIPT_DIR / "go2w_perception_dry_run.py",
        SCRIPT_DIR / "go2w_perception_worker.py",
        SCRIPT_DIR / "piper_planning_dry_run.py",
        SCRIPT_DIR / "piper_planning_worker.py",
    )
    discovered = (
        *sorted((STACK_ROOT / "z_manip").rglob("*.py")),
        *sorted((STACK_ROOT / "configs").glob("*.json")),
        *sorted((STACK_ROOT / "configs").glob("*.yaml")),
        *sorted((STACK_ROOT / "configs").glob("*.yml")),
    )
    external_urdf = (
        WORKSPACE_ROOT / "go2W_Sim" / "assets" / "urdf" / "go2w_sensored.urdf"
    )
    return tuple(path for path in (*fixed, *discovered, external_urdf) if path.is_file())


def runtime_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in runtime_inputs():
        try:
            name = path.relative_to(WORKSPACE_ROOT).as_posix()
        except ValueError:
            name = path.name
        data = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


#: How many leading hex characters identify a fingerprint in operator-facing
#: text.  ``go2w_component_manager.sh`` already prints ``${expected:0:12}``; this
#: constant exists so the servo, the supervisor and that script agree instead of
#: each choosing a width.
SHORT_FINGERPRINT_CHARS = 12

RUNTIME_FINGERPRINT_ENV = "Z_MANIP_RUNTIME_FINGERPRINT"


def short_fingerprint(value: str | None) -> str | None:
    """Operator-width form of a fingerprint, or ``None`` for an unknown one."""

    return None if value is None else value[:SHORT_FINGERPRINT_CHARS]


def try_runtime_fingerprint() -> tuple[str | None, str | None]:
    """``(fingerprint, error)`` -- never raises, for use on a live control path.

    ``runtime_fingerprint`` reads ~1.3 MB across 99 files.  THAT IS EXACTLY THE
    SET AN OPERATOR IS EDITING while the servo runs, so a file can vanish or be
    replaced between ``runtime_inputs``'s ``is_file`` check and ``read_bytes``
    -- every editor that saves through a temp file plus ``os.replace`` creates
    that window.  A servo tick may not die because somebody hit save.
    """

    try:
        return runtime_fingerprint(), None
    except (OSError, UnicodeError, ValueError, MemoryError) as error:
        return None, f"{type(error).__name__}: {error}"


def runtime_fingerprint_state(
    *,
    launch: str | None,
    live: str | None,
    error: str | None = None,
) -> dict[str, object]:
    """Pure verdict on whether the deployed tree changed under a running process.

    THE DEPLOYMENT IS THE WORKING TREE.  ``go2w_depth_servo.sh`` bind-mounts
    ``$STACK_ROOT`` read-only and runs the servo straight out of it, so an edit
    on the host is a live deploy into a process that is already running.  One
    recorded session holds ``_tick`` at five different line numbers
    (go2w-depth-servo.log: 1835, 1954, 2043, 2045, 2183) because the file moved
    underneath it, and commits a0f9e10 and 0124a20 each produced a
    resident-worker fingerprint mismatch minutes later.

    ``mutated`` is asserted ONLY when both fingerprints are known and differ.
    Absence is reported as ``unverified`` rather than as a mismatch on purpose:
    every offline replay, every test harness and every non-containerised run
    lacks a launch fingerprint, and manufacturing an alarm for all of them would
    train an operator to ignore the one that matters.  ``unverified`` is carried
    in the document so the absence is still visible.
    """

    mutated = bool(launch is not None and live is not None and launch != live)
    return {
        "launch": launch,
        "live": live,
        "launch_short": short_fingerprint(launch),
        "live_short": short_fingerprint(live),
        "mutated": mutated,
        "unverified": launch is None or live is None,
        "error": error,
    }


#: How often a long-lived process re-hashes the tree it is running out of.
#:
#: A full hash costs a measured 3.8 ms warm over 99 files / 1.3 MB.  That is
#: 7.6% of a 20 Hz servo tick and must not run every tick; at 0.5 Hz it is 0.19%
#: and still bounds the detection latency of a mid-run mutation to one 2 s
#: window.  ONE number, shared by the servo and the supervisor, so the two
#: cannot hold different opinions about how fresh "live" is.
RUNTIME_FINGERPRINT_RECHECK_S = 2.0


class RuntimeFingerprintWatch:
    """Rate-limited watcher over the working tree a process is running from.

    Deliberately a plain class with an injectable clock and probe: the servo's
    node is nested inside ``_run_ros`` and needs rclpy, and the supervisor's
    runner needs a live HTTP server, so a decision that only lived on either
    would be untestable on a host where neither imports.
    """

    def __init__(
        self,
        *,
        launch: str | None,
        recheck_s: float = RUNTIME_FINGERPRINT_RECHECK_S,
        clock: Any = None,
        probe: Any = None,
    ) -> None:
        self._launch = launch
        self._recheck_s = max(0.0, float(recheck_s))
        self._clock = time.monotonic if clock is None else clock
        self._probe = try_runtime_fingerprint if probe is None else probe
        self._live: str | None = None
        self._error: str | None = None
        self._checked_s: float | None = None

    def state(self) -> dict[str, object]:
        """Current verdict, re-hashing the tree at most once per recheck period.

        The FIRST call always probes.  A process that never re-read the tree
        would report ``unverified`` for its whole life, and an operator cannot
        tell that apart from "checked and clean".
        """

        now = self._clock()
        if self._checked_s is None or now - self._checked_s >= self._recheck_s:
            self._live, self._error = self._probe()
            self._checked_s = now
        state = runtime_fingerprint_state(
            launch=self._launch,
            live=self._live,
            error=self._error,
        )
        state["checked_age_s"] = (
            None if self._checked_s is None else max(0.0, now - self._checked_s)
        )
        return state


def main() -> int:
    print(runtime_fingerprint())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
