#!/usr/bin/env python3
"""Run the fixed single-connection full-grasp transaction on the PiPER NUC."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
FULL_EXECUTOR = SCRIPT_DIR / "piper_full_grasp_executor.py"
STAGE_EXECUTOR = SCRIPT_DIR / "piper_staged_grasp_executor.py"
NUC_HOST = os.environ.get("GO2W_NUC_HOST", "yusenzlabnuc@192.168.3.8")
NUC_KEY = Path(os.environ.get(
    "GO2W_NUC_SSH_KEY",
    str(Path.home() / ".ssh" / "id_ed25519_codex_nuc"),
)).expanduser().resolve()
REMOTE_ROOT = "/home/yusenzlabnuc/z-manip-runtime/full-grasp-actions"


def run(arguments: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        shell=False,
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-report", type=Path, required=True)
    parser.add_argument("--planned-grasp", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--speed-percent", type=int, required=True)
    parser.add_argument("--workflow-phase", choices=("full", "pick-hold", "return-home-holding", "place-back"), default="full")
    parser.add_argument("--prior-receipt-dir", type=Path)
    parser.add_argument("--planning-session-id")
    args = parser.parse_args()
    try:
        if not 1 <= args.speed_percent <= 50:
            raise ValueError("speed percent must be within 1-50")
        inputs = [
            NUC_KEY,
            FULL_EXECUTOR,
            STAGE_EXECUTOR,
            args.planning_report.resolve(),
            args.planned_grasp.resolve(),
        ]
        for path in inputs:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"required regular input is unavailable: {path}")
        if args.receipt_dir.exists():
            raise ValueError(f"refusing to overwrite receipt directory: {args.receipt_dir}")
        dry_command = [
            sys.executable,
            str(FULL_EXECUTOR),
            "--planning-report", str(args.planning_report.resolve()),
            "--planned-grasp", str(args.planned_grasp.resolve()),
            "--receipt-dir", str(args.receipt_dir.resolve()),
            "--speed-percent", str(args.speed_percent),
            "--workflow-phase", args.workflow_phase,
        ]
        if args.planning_session_id is not None:
            dry_command.extend(("--planning-session-id", args.planning_session_id))
        if args.prior_receipt_dir is not None:
            dry_command.extend(("--prior-receipt-dir", str(args.prior_receipt_dir.resolve())))
        dry = run(dry_command, timeout=15.0)
        if dry.returncode != 0:
            raise RuntimeError(f"full-grasp validation failed: {dry.stdout.strip()}")
        token = json.loads(dry.stdout).get("confirmation_token")
        if not isinstance(token, str) or not token.startswith("PIPER-FULL-"):
            raise RuntimeError("full-grasp validation omitted its bound token")

        action_id = f"{time.time_ns()}-{secrets.token_hex(5)}"
        remote_dir = f"{REMOTE_ROOT}/{action_id}"
        remote_receipts = f"{remote_dir}/receipts"
        ssh = [
            "ssh", "-i", str(NUC_KEY), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5", NUC_HOST,
        ]
        scp = [
            "scp", "-q", "-i", str(NUC_KEY), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5",
        ]
        created = run([*ssh, f"mkdir -p {shlex.quote(remote_dir)}"], timeout=10.0)
        if created.returncode != 0:
            raise RuntimeError(f"cannot create NUC action directory: {created.stdout.strip()}")
        try:
            copied = run([
                *scp,
                *(str(path) for path in inputs[1:]),
                f"{NUC_HOST}:{remote_dir}/",
            ], timeout=20.0)
            if copied.returncode != 0:
                raise RuntimeError(f"cannot copy full grasp to NUC: {copied.stdout.strip()}")
            remote_prior = None
            if args.prior_receipt_dir is not None:
                prior = args.prior_receipt_dir.resolve()
                state_file = prior / "workflow-state.json"
                if not state_file.is_file() or state_file.is_symlink():
                    raise ValueError("prior receipt directory omits workflow-state.json")
                remote_prior = f"{remote_dir}/prior"
                made_prior = run([*ssh, f"mkdir -p {shlex.quote(remote_prior)}"], timeout=10.0)
                if made_prior.returncode != 0:
                    raise RuntimeError("cannot create NUC prior receipt directory")
                prior_copy = run([
                    *scp, str(state_file), f"{NUC_HOST}:{remote_prior}/workflow-state.json",
                ], timeout=10.0)
                if prior_copy.returncode != 0:
                    raise RuntimeError(f"cannot copy workflow receipt to NUC: {prior_copy.stdout.strip()}")
            command = [
                "/usr/bin/python3", f"{remote_dir}/{FULL_EXECUTOR.name}",
                "--planning-report", f"{remote_dir}/{args.planning_report.name}",
                "--planned-grasp", f"{remote_dir}/{args.planned_grasp.name}",
                "--receipt-dir", remote_receipts,
                "--speed-percent", str(args.speed_percent),
                "--segment-timeout-s", "12",
                "--lift-hold-s", "2",
                "--firmware", "v188",
                "--channel", "can0",
                "--execute",
                "--confirm", token,
                "--workflow-phase", args.workflow_phase,
            ]
            if args.planning_session_id is not None:
                command.extend(("--planning-session-id", args.planning_session_id))
            if remote_prior is not None:
                command.extend(("--prior-receipt-dir", remote_prior))
            remote_shell = (
                "set -e; "
                "systemctl --user stop z-manip-piper-passive-feedback.service; "
                "trap 'sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8 >/tmp/z-manip-passive-restore.log 2>&1 || true; systemctl --user start z-manip-piper-passive-feedback.service' EXIT; "
                "cd ~/pyAgxArm; "
                + " ".join(shlex.quote(part) for part in command)
            )
            executed = run([*ssh, remote_shell], timeout=420.0)
            if executed.stdout:
                print(executed.stdout, end="" if executed.stdout.endswith("\n") else "\n")
            if executed.returncode != 0:
                raise RuntimeError(f"NUC full grasp stopped safely (exit {executed.returncode})")
            args.receipt_dir.parent.mkdir(parents=True, exist_ok=True)
            args.receipt_dir.mkdir(mode=0o700)
            fetched = run([
                *scp,
                f"{NUC_HOST}:{remote_receipts}/*.json",
                f"{args.receipt_dir}/",
            ], timeout=20.0)
            if fetched.returncode != 0:
                raise RuntimeError(f"cannot fetch full-grasp receipts: {fetched.stdout.strip()}")
            print(json.dumps({
                "schema": "z_manip.piper_full_grasp_remote.v1",
                "success": True,
                "speed_percent": args.speed_percent,
                "receipt_dir": str(args.receipt_dir),
            }, sort_keys=True))
            return 0
        finally:
            run([*ssh, f"rm -rf {shlex.quote(remote_dir)}"], timeout=10.0)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
