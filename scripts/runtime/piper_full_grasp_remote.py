#!/usr/bin/env python3
"""Run the fixed single-connection full-grasp transaction on the PiPER NUC."""

from __future__ import annotations

import argparse
import hashlib
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
# A payload directory now outlives the leg that uploaded it, so the reaper caps
# the cache at the newest few (~320 KB each) rather than at one per leg.
REMOTE_PAYLOAD_KEEP = 8


def _mux_options() -> list[str]:
    """SSH connection-multiplexing options shared by every ssh/scp call.

    This wrapper makes ~6-10 short ssh/scp calls per leg (mkdir, upload,
    optional prior upload, the executor call, the receipt probe/fetch retries,
    and cleanup), and a place-back cycle drives three back-to-back leg
    processes.  Over the WiFi link to the NUC a cold SSH handshake costs
    ~0.40s while a call multiplexed onto a persisted master costs ~0.01-0.03s.
    Reusing one authenticated master across every call -- and persisting it
    (``ControlPersist``) across the successive leg processes -- collapses that
    per-handshake tax to a single connect per grasp.  This mirrors the
    perception path (``FixedReadOnlyBackend._ssh_prefix``) and the wrist-search
    launcher; a dedicated ``grasp`` control path keeps a wedged motion
    connection from ever poisoning the perception transport.  Transport only:
    the remote command, every receipt, and every fail-closed gate are
    unchanged.
    """

    return [
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=60",
        "-o", f"ControlPath={NUC_KEY.parent / 'z-manip-grasp-%C'}",
    ]


def _payload_manifest_sha(payload: list[Path], planning_session_id: str) -> str:
    """Digest that names the staged payload directory on the NUC.

    A place-back cycle drives three legs as three processes over one
    byte-identical 320KB payload, and re-uploading it per leg costs ~0.3-0.5s
    each.  The manifest lines are laid out exactly as ``sha256sum`` prints
    them, so the NUC can recompute this digest over the files it already
    carries and prove they are complete and unmodified.  The session id is
    folded in because a payload staged for one planning session must never be
    reused by another.
    """

    manifest = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in payload
    )
    return hashlib.sha256(
        f"{manifest}session {planning_session_id}\n".encode("utf-8"),
    ).hexdigest()


def _staged_payload_probe(
    *,
    remote_dir: str,
    payload: list[Path],
    planning_session_id: str,
    manifest_sha: str,
) -> str:
    """One round trip that either proves the payload is staged or prepares a re-upload.

    Prints ``STAGED`` only when the NUC's own re-hash of every payload file AND
    the ``.manifest-sha`` completion marker both equal the expected digest, so
    a mutated, truncated or half-uploaded remote copy can never be executed
    against.  Any other outcome clears the marker and (re)creates the
    directory, which is the state the upload below expects.
    """

    names = " ".join(shlex.quote(path.name) for path in payload)
    return (
        f"d={shlex.quote(remote_dir)}; e={shlex.quote(manifest_sha)}; "
        f'a=$({{ cd "$d" && sha256sum -- {names} && '
        f"printf 'session %s\\n' {shlex.quote(planning_session_id)}; }} "
        "2>/dev/null | sha256sum | cut -d' ' -f1); "
        'if [ "$a" = "$e" ] && [ "$(cat "$d/.manifest-sha" 2>/dev/null)" = "$e" ]; '
        "then echo STAGED; "
        'else rm -f "$d/.manifest-sha"; mkdir -p "$d" || exit 1; echo STALE; fi'
    )


def _detached_cleanup(paths: list[str]) -> str:
    """Remote cleanup that a leg boundary must not wait for.

    Deletion latency lands between two legs of a held-object cycle, so the
    reaper is launched detached and the ssh returns as soon as it is spawned.
    The payload cache is bounded by keeping the newest ``REMOTE_PAYLOAD_KEEP``
    directories; a payload whose receipts were never fetched is skipped no
    matter how old, because that tree is the only evidence for an object that
    may still be in the gripper.
    """

    body = (
        "rm -rf -- " + " ".join(shlex.quote(path) for path in paths) + "; "
        f"cd {shlex.quote(REMOTE_ROOT)} 2>/dev/null || exit 0; "
        f"ls -1dt -- */ 2>/dev/null | tail -n +{REMOTE_PAYLOAD_KEEP + 1} | "
        "while read -r stale; do "
        '[ -n "$(ls -A -- "$stale/receipts" 2>/dev/null)" ] || rm -rf -- "$stale"; '
        "done"
    )
    return f"nohup sh -c {shlex.quote(body)} >/dev/null 2>&1 </dev/null &"


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
        payload = inputs[1:]
        session = args.planning_session_id or ""
        manifest_sha = _payload_manifest_sha(payload, session)
        remote_dir = f"{REMOTE_ROOT}/{manifest_sha}"
        # Receipts and the prior-workflow handoff stay keyed on the action, not
        # on the payload: the directory is shared by every leg of a cycle, and
        # the executor refuses a receipt directory that already exists.
        remote_receipts = f"{remote_dir}/receipts/{action_id}"
        mux = _mux_options()
        ssh = [
            "ssh", "-i", str(NUC_KEY), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5",
            *mux, NUC_HOST,
        ]
        scp = [
            "scp", "-q", "-p", "-i", str(NUC_KEY), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5",
            *mux,
        ]
        created = run([*ssh, _staged_payload_probe(
            remote_dir=remote_dir,
            payload=payload,
            planning_session_id=session,
            manifest_sha=manifest_sha,
        )], timeout=10.0)
        if created.returncode != 0:
            raise RuntimeError(f"cannot create NUC action directory: {created.stdout.strip()}")
        payload_staged = created.stdout.strip() == "STAGED"
        started = None
        remote_prior = None
        receipts_fetched = False
        receipt_proved_absent = False
        try:
            if not payload_staged:
                copied = run([
                    *scp,
                    *(str(path) for path in payload),
                    f"{NUC_HOST}:{remote_dir}/",
                ], timeout=20.0)
                if copied.returncode != 0:
                    raise RuntimeError(f"cannot copy full grasp to NUC: {copied.stdout.strip()}")
            if args.prior_receipt_dir is not None:
                prior = args.prior_receipt_dir.resolve()
                state_file = prior / "workflow-state.json"
                if not state_file.is_file() or state_file.is_symlink():
                    raise ValueError("prior receipt directory omits workflow-state.json")
                remote_prior = f"{remote_dir}/prior/{action_id}"
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
                # Stamping the completion marker here, before anything can touch
                # CAN, is what makes a later leg allowed to skip the upload: the
                # marker exists only once a full payload has landed.
                f"printf '%s' {shlex.quote(manifest_sha)} > "
                f"{shlex.quote(remote_dir + '/.manifest-sha')}; "
                # The reactive-view executor is a live CAN owner in its own
                # right; systemd Conflicts= only pairs it with the passive
                # bridge, so the grasp executor must evict it explicitly or
                # two owners can transmit on can0 mid-motion.
                "systemctl --user stop z-mobile-manip-piper-reactive-view.service; "
                "systemctl --user stop z-manip-piper-passive-feedback.service; "
                "trap 'sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8 >/tmp/z-manip-passive-restore.log 2>&1 || true; systemctl --user start z-manip-piper-passive-feedback.service' EXIT; "
                "cd ~/pyAgxArm; "
                + " ".join(shlex.quote(part) for part in command)
            )
            executed = run([*ssh, remote_shell], timeout=420.0)
            if executed.stdout:
                print(executed.stdout, end="" if executed.stdout.endswith("\n") else "\n")
            # Preserve executor-start evidence even when a later motion stage
            # blocks.  Do not manufacture a local receipt directory when the
            # NUC never opened the real transport.
            start_receipt = f"{remote_receipts}/executor-start-receipt.json"
            # Receipts (including a pick-hold workflow-state.json that is the
            # ONLY continuation evidence for a physically held object) must
            # survive one transient ssh/scp blip, so the fetch is retried and
            # the remote copy is never deleted until it has landed.  The fetch
            # itself is the probe: a successful one answers both questions in a
            # single round trip, and only a fetch that failed needs the
            # ``test -f`` to tell "the transport never opened" (which ends the
            # retries) apart from "the link blipped" (which does not).
            args.receipt_dir.parent.mkdir(parents=True, exist_ok=True)
            if not args.receipt_dir.exists():
                args.receipt_dir.mkdir(mode=0o700)
            fetch_error = ""
            for _attempt in range(3):
                fetched = run([
                    *scp,
                    f"{NUC_HOST}:{remote_receipts}/*.json",
                    f"{args.receipt_dir}/",
                ], timeout=20.0)
                if fetched.returncode == 0:
                    receipts_fetched = True
                    break
                fetch_error = fetched.stdout.strip()
                started = run(
                    [*ssh, f"test -f {shlex.quote(start_receipt)}"],
                    timeout=10.0,
                )
                if started.returncode == 1:
                    receipt_proved_absent = True
                    break
                time.sleep(1.0)
            if not receipts_fetched and not any(args.receipt_dir.iterdir()):
                # An empty receipt directory would read downstream as evidence
                # that this leg ran; leave nothing behind that the NUC did not
                # actually hand over.
                args.receipt_dir.rmdir()
            if not receipts_fetched and started is not None and started.returncode == 0:
                # The receipt definitely exists but could not be fetched;
                # surface that as the primary failure.  With an unknown
                # probe, fall through so a nonzero executor exit reports
                # its own (more informative) error instead.
                raise RuntimeError(
                    "cannot fetch full-grasp receipts after retries "
                    f"(remote evidence preserved at {remote_receipts}): {fetch_error}",
                )
            if executed.returncode != 0:
                raise RuntimeError(f"NUC full grasp stopped safely (exit {executed.returncode})")
            if not (args.receipt_dir / "executor-start-receipt.json").is_file():
                raise RuntimeError("NUC executor succeeded without transport-start evidence")
            print(json.dumps({
                "schema": "z_manip.piper_full_grasp_remote.v1",
                "success": True,
                "speed_percent": args.speed_percent,
                "receipt_dir": str(args.receipt_dir),
                "payload_upload_skipped": payload_staged,
            }, sort_keys=True))
            return 0
        finally:
            # Delete this leg's remote receipts ONLY when the evidence is
            # secured: receipts landed locally, or the probe PROVED no start
            # receipt exists (transport never opened, nothing can be held).
            # Any other state -- unknown probe, early exception before the
            # probe ran, failed fetch -- preserves the remote copy, because
            # deleting it could strand a held object with no continuation
            # state.  The cost of a false keep is one stale directory; the
            # cost of a false delete is unrecoverable.
            evidence_secured = receipts_fetched or receipt_proved_absent
            if evidence_secured:
                leftovers = [remote_receipts]
                if remote_prior is not None:
                    leftovers.append(remote_prior)
                run([*ssh, _detached_cleanup(leftovers)], timeout=10.0)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
