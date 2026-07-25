"""Shared atomic file-write primitive for scripts/runtime entry points.

Every writer here follows the same crash-safety contract: write to a
process-unique temp file next to the destination, then ``replace`` it into
place, cleaning the temp file up on any failure so a killed write never
leaves a stray ``.tmp`` file behind. Callers own their own serialization
(JSON formatting, encoding, etc.); this module only owns the temp-file
mechanics.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding=encoding)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
