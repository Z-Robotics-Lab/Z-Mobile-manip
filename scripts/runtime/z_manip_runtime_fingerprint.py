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


def main() -> int:
    print(runtime_fingerprint())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
