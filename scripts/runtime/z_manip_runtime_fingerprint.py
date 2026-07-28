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


def external_runtime_inputs() -> tuple[Path, ...]:
    """Hashed inputs that do NOT live under ``STACK_ROOT``.

    THE DEFECT THIS EXISTS FOR, AND IT SHIPPED.  ``runtime_inputs`` ends with
    ``$WORKSPACE_ROOT/go2W_Sim/assets/urdf/go2w_sensored.urdf`` -- the ONE input
    of 99 that is outside the checkout.  ``go2w_depth_servo.sh`` measures the
    fingerprint on the HOST (where that file exists) and exports it as the
    servo's launch baseline, but its ``docker run`` mounted ``$STACK_ROOT`` at
    its own path and the URDF only at the UNRELATED path
    ``/robot/go2w_sensored.urdf``.  ``runtime_inputs``'s ``if path.is_file()``
    filter then silently dropped it inside the container, so the servo's live
    re-measure hashed 98 files where the launcher hashed 99.  Measured on this
    machine against the live checkout:

        launcher (host, 99 files) : 834e35138ae0...
        servo    (container, 98)  : 87e820e6b6c2...

    ``runtime_fingerprint_state`` asserts ``mutated`` whenever both values are
    known and differ, so EVERY live and shadow servo run reported
    ``runtime_fingerprint_mutated: true`` on every trace row and raised
    ``RUNTIME_TREE_MUTATED_DURING_RUN`` on the operator's dashboard, on a
    completely frozen tree.  A deploy alarm that is always on is the exact
    thing ``runtime_fingerprint_state``'s own docstring argues against.

    The fix is a contract, not a special case: the launcher self-mounts every
    path this returns (``--launch-manifest`` below), so the container resolves
    the identical set at the identical absolute paths.  ``unmounted_runtime_inputs``
    is the pure check, and tests/test_runtime_fingerprint_mounts.py runs it
    against the mount list parsed out of the launcher.  Adding a new input
    outside ``STACK_ROOT`` therefore cannot silently re-open this.
    """

    return tuple(
        path for path in runtime_inputs() if not path.is_relative_to(STACK_ROOT)
    )


def unmounted_runtime_inputs(
    mount_sources: Any,
    *,
    inputs: Any = None,
) -> tuple[Path, ...]:
    """Pure: which hashed inputs a container with these bind mounts cannot see.

    ``mount_sources`` is the set of host paths that ``docker run`` mounts AT
    THEIR OWN ABSOLUTE PATH (``-v "$X:$X:ro"``).  A mount that remaps the
    destination (``-v "$URDF:/robot/...:ro"``) does not count and must not: the
    in-container ``runtime_inputs`` looks the file up by the host path, and the
    digest's per-file name key is derived from that path too, so a remapped
    copy changes both the presence test and the name.
    """

    sources = [Path(source) for source in mount_sources]
    candidates = runtime_inputs() if inputs is None else tuple(Path(p) for p in inputs)
    return tuple(
        path
        for path in candidates
        if not any(path == source or path.is_relative_to(source) for source in sources)
    )


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


#: Argument that makes ``main`` print the fingerprint AND, on the following
#: lines, every hashed input a container must be given at its own absolute path.
#:
#: ONE invocation, not two, on purpose: the digest and the mount list have to
#: describe the same measurement.  Two calls would let the URDF appear or vanish
#: between them and hand the servo a baseline measured over a set the mounts do
#: not reproduce -- which is the very failure this argument exists to close.
LAUNCH_MANIFEST_FLAG = "--launch-manifest"


def main(argv: Any = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args == [LAUNCH_MANIFEST_FLAG]:
        # Order matters and the launcher depends on it: line 1 is the digest,
        # every later line is a mount source.  A failure here prints nothing,
        # which the launcher reads as "no fingerprint" -> the servo reports
        # ``unverified`` rather than a fabricated mismatch.
        external = external_runtime_inputs()
        unmountable = [
            path for path in external if ":" in str(path) or "\n" in str(path)
        ]
        if unmountable:
            # ``docker run -v SRC:DST:MODE`` is colon-delimited and the launcher
            # reads this output line by line, so such a path cannot be handed
            # across the boundary at all.  Emitting NOTHING is the fail-closed
            # answer: the launcher exports no fingerprint and the servo reports
            # ``unverified``.  Printing the digest and silently dropping the
            # mount would put back exactly the permanent false ``mutated`` this
            # flag exists to remove.
            print(
                "cannot express these hashed inputs as bind mounts: "
                + ", ".join(str(path) for path in unmountable),
                file=sys.stderr,
            )
            return 3
        print(runtime_fingerprint())
        for path in external:
            print(path)
        return 0
    if args:
        print(f"usage: {__file__} [{LAUNCH_MANIFEST_FLAG}]", file=sys.stderr)
        return 2
    print(runtime_fingerprint())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
