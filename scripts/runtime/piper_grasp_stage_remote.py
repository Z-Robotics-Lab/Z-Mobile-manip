#!/usr/bin/env python3
"""Execute one artifact-bound PiPER grasp stage on the fixed NUC.

This helper is server-owned.  The dashboard never supplies its paths, token,
CAN channel, firmware, speed, or remote command.  It validates the immutable
planning artifact locally, derives the exact dry-run token, copies only the
required files to a fresh NUC directory, and restores passive feedback on exit.
"""

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
EXECUTOR = SCRIPT_DIR / "piper_staged_grasp_executor.py"
NUC_HOST = os.environ.get("GO2W_NUC_HOST", "yusenzlabnuc@192.168.3.8")
NUC_KEY = Path(os.environ.get(
    "GO2W_NUC_SSH_KEY",
    str(Path.home() / ".ssh" / "id_ed25519_codex_nuc"),
)).expanduser().resolve()
REMOTE_ROOT = "/home/yusenzlabnuc/z-manip-runtime/grasp-actions"
STAGE_MAX_AGE_S = {"pregrasp": 30.0, "approach_close": 180.0, "lift": 180.0}


class RemoteStageError(RuntimeError):
    """The fixed remote stage failed before producing a verified receipt."""


def _run(arguments: list[str], *, timeout: float, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
        shell=False,
        timeout=timeout,
    )


def _dry_run_token(arguments: list[str]) -> str:
    completed = _run([sys.executable, str(EXECUTOR), *arguments], timeout=15.0, capture=True)
    if completed.returncode != 0:
        raise RemoteStageError(f"local stage validation failed: {completed.stdout.strip()}")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RemoteStageError("local stage validation returned invalid JSON") from error
    token = document.get("confirmation_token")
    if not isinstance(token, str) or not token.startswith("PIPER-"):
        raise RemoteStageError("local stage validation omitted its bound token")
    return token


def execute_remote_stage(
    *,
    planning_report: Path,
    planned_grasp: Path,
    stage: str,
    receipt_output: Path,
    prior_receipt: Path | None,
) -> dict[str, object]:
    """Validate and execute one stage, returning its immutable receipt."""

    if stage not in STAGE_MAX_AGE_S:
        raise RemoteStageError(f"unsupported stage: {stage}")
    inputs = [EXECUTOR, NUC_KEY, planning_report, planned_grasp]
    if prior_receipt is not None:
        inputs.append(prior_receipt)
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            raise RemoteStageError(f"required regular input is unavailable: {path}")
    if receipt_output.exists():
        raise RemoteStageError(f"refusing to overwrite receipt: {receipt_output}")

    age = STAGE_MAX_AGE_S[stage]
    local_arguments = [
        "--planning-report", str(planning_report),
        "--planned-grasp", str(planned_grasp),
        "--stage", stage,
        "--max-source-age-s", str(age),
    ]
    prior_flag: str | None = None
    if stage == "approach_close":
        prior_flag = "--pregrasp-receipt"
    elif stage == "lift":
        prior_flag = "--approach-receipt"
    if prior_flag is not None:
        if prior_receipt is None:
            raise RemoteStageError(f"{stage} requires its prior receipt")
        local_arguments.extend((prior_flag, str(prior_receipt)))
    elif prior_receipt is not None:
        raise RemoteStageError("pregrasp cannot consume a prior receipt")
    token = _dry_run_token(local_arguments)

    action_id = f"{time.time_ns()}-{secrets.token_hex(5)}"
    remote_dir = f"{REMOTE_ROOT}/{action_id}"
    ssh_base = [
        "ssh", "-i", str(NUC_KEY), "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5", NUC_HOST,
    ]
    scp_base = [
        "scp", "-q", "-i", str(NUC_KEY), "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5",
    ]
    mkdir = _run([*ssh_base, f"mkdir -p {shlex.quote(remote_dir)}"], timeout=10.0, capture=True)
    if mkdir.returncode != 0:
        raise RemoteStageError(f"cannot create fixed NUC action directory: {mkdir.stdout.strip()}")
    try:
        local_files = [EXECUTOR, planning_report, planned_grasp]
        if prior_receipt is not None:
            local_files.append(prior_receipt)
        copied = _run(
            [*scp_base, *(str(path) for path in local_files), f"{NUC_HOST}:{remote_dir}/"],
            timeout=20.0,
            capture=True,
        )
        if copied.returncode != 0:
            raise RemoteStageError(f"cannot copy grasp stage to NUC: {copied.stdout.strip()}")

        remote_report = f"{remote_dir}/{planning_report.name}"
        remote_npz = f"{remote_dir}/{planned_grasp.name}"
        remote_receipt = f"{remote_dir}/{stage}-receipt.json"
        command = [
            "/usr/bin/python3", f"{remote_dir}/{EXECUTOR.name}",
            "--planning-report", remote_report,
            "--planned-grasp", remote_npz,
            "--stage", stage,
            "--max-source-age-s", str(age),
            "--speed-percent", "5",
            "--segment-timeout-s", "12",
            "--gripper-force-n", "1.0",
            "--firmware", "v188",
            "--channel", "can0",
            "--receipt-output", remote_receipt,
            "--execute",
            "--confirm", token,
        ]
        if prior_flag is not None and prior_receipt is not None:
            command.extend((prior_flag, f"{remote_dir}/{prior_receipt.name}"))
        quoted = " ".join(shlex.quote(part) for part in command)
        remote_shell = (
            "set -e; "
            "systemctl --user stop z-manip-piper-passive-feedback.service; "
            "trap 'sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8 >/tmp/z-manip-passive-restore.log 2>&1 || true; systemctl --user start z-manip-piper-passive-feedback.service' EXIT; "
            f"cd ~/pyAgxArm; {quoted}"
        )
        executed = _run([*ssh_base, remote_shell], timeout=150.0, capture=True)
        if executed.stdout:
            print(executed.stdout, end="" if executed.stdout.endswith("\n") else "\n", flush=True)
        if executed.returncode != 0:
            raise RemoteStageError(f"NUC {stage} execution stopped safely (exit {executed.returncode})")

        receipt_output.parent.mkdir(parents=True, exist_ok=True)
        fetched = _run(
            [*scp_base, f"{NUC_HOST}:{remote_receipt}", str(receipt_output)],
            timeout=15.0,
            capture=True,
        )
        if fetched.returncode != 0:
            raise RemoteStageError(f"cannot fetch verified {stage} receipt: {fetched.stdout.strip()}")
        try:
            receipt = json.loads(receipt_output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RemoteStageError(f"cannot read verified {stage} receipt") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != "z_manip.piper_stage_receipt.v1"
            or receipt.get("stage") != stage
            or receipt.get("success") is not True
        ):
            raise RemoteStageError(f"NUC returned an invalid {stage} receipt")
        return receipt
    finally:
        _run([*ssh_base, f"rm -rf {shlex.quote(remote_dir)}"], timeout=10.0, capture=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-report", type=Path, required=True)
    parser.add_argument("--planned-grasp", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_MAX_AGE_S), required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--prior-receipt", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = execute_remote_stage(
            planning_report=args.planning_report.expanduser().resolve(),
            planned_grasp=args.planned_grasp.expanduser().resolve(),
            stage=args.stage,
            receipt_output=args.receipt_output.expanduser().resolve(),
            prior_receipt=(
                None if args.prior_receipt is None
                else args.prior_receipt.expanduser().resolve()
            ),
        )
        print(json.dumps({
            "schema": "z_manip.piper_remote_stage.v1",
            "stage": args.stage,
            "success": True,
            "receipt": str(args.receipt_output.expanduser().resolve()),
            "artifact_id": receipt.get("artifact_id"),
        }, sort_keys=True))
        return 0
    except (OSError, RemoteStageError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
