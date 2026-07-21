#!/usr/bin/env python3
"""Replay a recorded depth-servo JSONL trace without importing robot drivers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2w_reactive_supervision import load_jsonl, replay_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect unbounded zero-command posture waits in an offline trace.",
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--stall-threshold-s", type=float, default=5.0)
    parser.add_argument(
        "--expect-stall",
        action="store_true",
        help="return success only when at least one historical stall is reproduced",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = replay_trace(
        load_jsonl(args.trace),
        stall_threshold_s=args.stall_threshold_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    has_stall = bool(report["stalls"])
    return 0 if has_stall == bool(args.expect_stall) else 1


if __name__ == "__main__":
    raise SystemExit(main())
