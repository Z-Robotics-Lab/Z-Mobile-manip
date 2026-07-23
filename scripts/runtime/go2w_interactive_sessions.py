#!/usr/bin/env python3
"""Fixed read-only perception and offline-planning session actions.

This is an integration adapter for a future loopback UI.  The action surface
contains no path, command, environment, actuator, or arbitrary transport
parameter.  It does not expose grasp execution.  Perception invokes the
existing lab script while repeating the exact passive joint receive gate;
planning revalidates that synchronized report before the network-disabled
offline planner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pwd
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
STACK_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = STACK_ROOT.parent
sys.path.insert(0, str(STACK_ROOT))

from z_manip.read_only_sessions import (  # noqa: E402
    BackendResult,
    ReadOnlySessionService,
    SessionContractError,
)
from z_manip_runtime_fingerprint import runtime_fingerprint  # noqa: E402


RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "go2w_real" / "interactive_sessions"
ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts"
PERCEPTION = SCRIPT_DIR / "go2w_perception_dry_run.py"
SESSION_GATE = SCRIPT_DIR / "piper_planning_session_gate.py"
PLANNER = SCRIPT_DIR / "piper_planning_dry_run.py"
PLANNING_WORKER = SCRIPT_DIR / "piper_planning_worker.py"
STACK_CONFIG = STACK_ROOT / "configs" / "go2w_piper.json"
DEBUG_BUNDLE = SCRIPT_DIR / "go2w_debug_bundle.py"
SAFETY_GATE = SCRIPT_DIR / "go2w_debug_safety_gate.py"
DDS_CONFIG = STACK_ROOT / "docker" / "runtime" / "cyclonedds-go2w-pc.xml"
CALIBRATION = (
    WORKSPACE_ROOT
    / "artifacts"
    / "go2w_real"
    / "calibration"
    / "piper_wrist_camera_calibration.json"
)
URDF = WORKSPACE_ROOT / "go2W_Sim" / "assets" / "urdf" / "go2w_sensored.urdf"
ROBOT_ASSETS = URDF.parent.parent
CONTAINER_URDF = f"/robot_assets/urdf/{URDF.name}"
DEFAULT_RUNTIME_IMAGE = "z-manip-runtime:pinocchio"
DEFAULT_IK_BACKEND = "pinocchio"
PERCEPTION_RUNNER_CONTAINER = "z-manip-perception-runner"
PERCEPTION_RUNNER_ARTIFACT_ROOT = Path("/workspace-artifacts")
PLANNING_RUNNER_CONTAINER = "z-manip-planning-runner"
PLANNING_RUNNER_ARTIFACT_ROOT = Path("/workspace-artifacts")
PLANNING_RUNNER_SCRATCH_ROOT = (
    ARTIFACT_ROOT / "go2w_real" / ".planning_runner_scratch"
)
PLANNING_RUNNER_CONTAINER_SCRATCH_ROOT = Path("/workspace-planning-output")
SAFE_RUNTIME_IMAGE = re.compile(
    r"z-manip-runtime:[a-z0-9][a-z0-9._-]{0,63}\Z",
)
NUC_HOST = "yusenzlabnuc@192.168.3.8"
NUC_KEY = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".ssh" / "id_ed25519_codex_nuc"
REMOTE_PASSIVE_REPORT = "/tmp/z-manip-passive-live.json"
REMOTE_PASSIVE_PROBE = "/usr/local/libexec/z-manip/piper_passive_probe.py"
PASSIVE_CAPTURE_SECONDS = "0.25"
PERCEPTION_ATTEMPTS = 2
MAX_PASSIVE_REPORT_BYTES = 1024 * 1024
MAX_SESSION_GATE_REPORT_BYTES = 256 * 1024
MAX_PLANNING_REPORT_BYTES = 4 * 1024 * 1024
MAX_PLANNED_GRASP_BYTES = 8 * 1024 * 1024
MAX_MODEL_EVIDENCE_BYTES = 8 * 1024 * 1024
PLANNING_RUNNER_SCRATCH_TTL_S = 24 * 60 * 60
MAX_PLANNER_ERROR_CHARS = 600
MAX_PERCEPTION_REPORT_BYTES = 256 * 1024
MAX_PERCEPTION_ERROR_CHARS = 600
MAX_REJECTIONS_TO_SUMMARIZE = 4096
MAX_WORKER_REQUEST_BYTES = 64 * 1024
MAX_WORKER_RESPONSE_BYTES = 8 * 1024 * 1024
SEARCH_TIMEOUT_S = "6"
SYMMETRY_SAMPLES = "4"
MAX_HYPOTHESES = "64"
MAX_CANDIDATES = "64"
MAX_FEASIBLE_PLANS = "1"
SUPPORT_APPROACH_PRIOR_WEIGHT = "0.05"
SUPERVISED_SCENE_CLEARANCE_M = "0.001"
SUPERVISED_SCENE_POINT_RADIUS_M = "0.001"
SUPERVISED_GRIPPER_SCENE_RADIUS_SCALE = "0.60"


@dataclass(frozen=True)
class _WorkerResult:
    """Bounded result returned by a fixed local resident worker."""

    returncode: int
    worker_elapsed_s: float | None = None
    worker_fingerprint: str | None = None


def _fixed_worker_socket_available(socket_path: Path) -> bool:
    """Accept only the server-owned, private Unix socket at a fixed path."""

    try:
        metadata = socket_path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISSOCK(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_gid == os.getegid()
        and metadata.st_mode & 0o077 == 0
    )


def _run_fixed_worker_request(
    socket_path: Path,
    request: Mapping[str, object],
    log_path: Path,
    *,
    expected_fingerprint: str | None = None,
) -> _WorkerResult:
    """Call one private local worker without spawning a Python client.

    This is deliberately Unix-socket-only.  The worker still owns argument
    validation, path confinement, read-only/planning-only policy, and output
    generation; this helper merely removes ``docker exec`` and client import
    overhead from the request transport.
    """

    if not _fixed_worker_socket_available(socket_path):
        raise OSError("fixed resident worker socket is unavailable or unsafe")
    encoded_request = json.dumps(request, separators=(",", ":")).encode("utf-8")
    if len(encoded_request) > MAX_WORKER_REQUEST_BYTES:
        raise ValueError("resident worker request exceeds bounded size")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(encoded_request)
        client.shutdown(socket.SHUT_WR)
        response = bytearray()
        while len(response) <= MAX_WORKER_RESPONSE_BYTES:
            block = client.recv(64 * 1024)
            if not block:
                break
            response.extend(block)
    if len(response) > MAX_WORKER_RESPONSE_BYTES:
        raise RuntimeError("resident worker response exceeds bounded size")
    try:
        document: Any = json.loads(bytes(response))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("resident worker returned malformed JSON") from error
    if not isinstance(document, dict):
        raise RuntimeError("resident worker response is not an object")
    return_code = document.get("return_code")
    output = document.get("output", "")
    worker_elapsed = document.get("elapsed_s")
    worker_fingerprint = document.get("worker_fingerprint")
    if not isinstance(return_code, int) or not isinstance(output, str):
        raise RuntimeError("resident worker response violates its schema")
    if expected_fingerprint is not None and worker_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "resident worker fingerprint mismatch; restart the perception component "
            f"(expected {expected_fingerprint[:12]}, got "
            f"{str(worker_fingerprint)[:12]})",
        )
    with log_path.open("ab") as log:
        log.write(output.encode("utf-8", errors="replace"))
    return _WorkerResult(
        returncode=return_code,
        worker_elapsed_s=(
            float(worker_elapsed)
            if isinstance(worker_elapsed, (int, float))
            and math.isfinite(float(worker_elapsed))
            and float(worker_elapsed) >= 0.0
            else None
        ),
        worker_fingerprint=(
            worker_fingerprint if isinstance(worker_fingerprint, str) else None
        ),
    )


def _append_timing(log_path: Path, stage: str, elapsed_s: float, **fields: object) -> None:
    """Append one machine-readable performance marker to an action log."""

    payload = {
        "schema": "z_manip.interactive_timing.v1",
        "stage": stage,
        "elapsed_s": round(float(elapsed_s), 6),
        **fields,
    }
    with log_path.open("ab") as log:
        log.write((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))


def _cleanup_stale_planning_runner_scratch(
    scratch_root: Path,
    *,
    now_s: float | None = None,
    max_age_s: float = PLANNING_RUNNER_SCRATCH_TTL_S,
) -> None:
    """Remove only old, server-owned warm-planner scratch directories.

    Every request uses ``mkdtemp`` and therefore never reuses these paths.
    Cleanup is deliberately conservative so a concurrent planner cannot be
    removed; symlinks and unrelated entries are never followed or deleted.
    """

    try:
        root_metadata = scratch_root.lstat()
    except OSError:
        return
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        return
    cutoff = (time.time() if now_s is None else float(now_s)) - float(max_age_s)
    try:
        entries = tuple(scratch_root.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith("planning-"):
            continue
        try:
            metadata = entry.lstat()
        except OSError:
            continue
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mtime >= cutoff
        ):
            continue
        try:
            shutil.rmtree(entry)
        except OSError:
            # Cleanup is maintenance only.  A new unique directory remains
            # safe even when an old directory cannot be removed.
            continue


def _planning_runner_report_valid(report_path: Path) -> bool:
    """Validate the minimum bounded output contract of the warm runner."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(report_path, flags)
    except OSError:
        return False
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > MAX_PLANNING_REPORT_BYTES
        ):
            return False
        chunks: list[bytes] = []
        remaining = MAX_PLANNING_REPORT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > MAX_PLANNING_REPORT_BYTES:
            return False
        document: Any = json.loads(encoded.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    finally:
        os.close(descriptor)
    return isinstance(document, dict)


def _bounded_evidence(path: Path, maximum: int) -> bytes | None:
    """Read one bounded regular evidence file without following symlinks."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 1 <= metadata.st_size <= maximum
        ):
            return None
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        return payload if 1 <= len(payload) <= maximum else None
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _planning_ready_evidence(
    *,
    perception_dir: Path,
    output_dir: Path,
    joint_report: Path,
) -> dict[str, object] | None:
    """Bind a pre-visualization plan-ready marker to immutable inputs.

    This marker is timing evidence only, never an executor receipt.  It is
    emitted only after the same fail-closed planning fields required by the
    immutable session controller are present and the execution archive exists.
    """

    paths = {
        "perception_report": (
            perception_dir / "report.json",
            MAX_PERCEPTION_REPORT_BYTES,
        ),
        "passive_joint_report": (joint_report, MAX_PASSIVE_REPORT_BYTES),
        "session_gate": (
            output_dir / "session_gate.json",
            MAX_SESSION_GATE_REPORT_BYTES,
        ),
        "planning_report": (
            output_dir / "planning" / "planning_report.json",
            MAX_PLANNING_REPORT_BYTES,
        ),
        "planned_grasp": (
            output_dir / "planning" / "planned_grasp.npz",
            MAX_PLANNED_GRASP_BYTES,
        ),
        "calibration": (CALIBRATION, MAX_MODEL_EVIDENCE_BYTES),
        "urdf": (URDF, MAX_MODEL_EVIDENCE_BYTES),
    }
    payloads: dict[str, bytes] = {}
    for name, (path, maximum) in paths.items():
        payload = _bounded_evidence(path, maximum)
        if payload is None:
            return None
        payloads[name] = payload
    try:
        gate: Any = json.loads(payloads["session_gate"].decode("utf-8"))
        report: Any = json.loads(payloads["planning_report"].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(gate, dict) or not isinstance(report, dict):
        return None
    if not (
        gate.get("planning_ready") is True
        and gate.get("read_only") is True
        and gate.get("planning_only") is True
        and gate.get("motion_commands_published") == 0
        and gate.get("transport_opened") is False
        and report.get("read_only") is True
        and report.get("planning_only") is True
        and report.get("motion_commands_published") == 0
        and report.get("plan_valid") is True
    ):
        return None
    return {
        "evidence_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
        "executor_receipt": False,
    }


@dataclass(frozen=True)
class ServerRuntimeConfig:
    """Allowlisted runtime settings resolved once from the server process."""

    runtime_image: str = DEFAULT_RUNTIME_IMAGE
    ik_backend: str = DEFAULT_IK_BACKEND

    @classmethod
    def from_server_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "ServerRuntimeConfig":
        """Read only two server-owned keys; request data is never consulted."""

        source = os.environ if environment is None else environment
        runtime_image = source.get(
            "Z_MANIP_RUNTIME_IMAGE",
            DEFAULT_RUNTIME_IMAGE,
        )
        ik_backend = source.get("Z_MANIP_IK_BACKEND", DEFAULT_IK_BACKEND)
        if SAFE_RUNTIME_IMAGE.fullmatch(runtime_image) is None:
            raise ValueError(
                "server runtime image must be a local z-manip-runtime tag",
            )
        if ik_backend != "pinocchio":
            raise ValueError(
                "interactive offline planning supports only pinocchio IK",
            )
        return cls(runtime_image=runtime_image, ik_backend=ik_backend)


def _server_environment(*, python_path: bool = False) -> dict[str, str]:
    """Return a fixed allowlist; no action-supplied environment is inherited."""

    account = pwd.getpwuid(os.getuid())
    environment = {
        "HOME": account.pw_dir,
        "USER": account.pw_name,
        "LOGNAME": account.pw_name,
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if python_path:
        environment["PYTHONPATH"] = str(STACK_ROOT)
    return environment


def _run_logged(
    argv: Sequence[str],
    log_path: Path,
    *,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        return subprocess.run(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            shell=False,
            check=False,
        )


def _six_joint_csv(value: object, label: str) -> str:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{label} must contain six joint positions")
    joints: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} contains a non-numeric joint position")
        joint = float(item)
        if not math.isfinite(joint):
            raise ValueError(f"{label} contains a non-finite joint position")
        joints.append(joint)
    return ",".join(f"{joint:.12g}" for joint in joints)


def _planning_failure_message(output_dir: Path) -> str:
    """Return a bounded summary from the fixed server-owned planner report.

    The report location is derived exclusively from the attempt output
    directory.  Refuse symlinks, non-regular files, oversized JSON, and
    malformed fields so a failed diagnostic read cannot broaden the action
    surface or hide the planner failure behind another exception.
    """

    fallback = (
        "offline planner produced no valid grasp plan; "
        "inspect the latest candidate rejection diagnostics"
    )
    report_path = output_dir / "planning" / "planning_report.json"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(report_path, flags)
    except OSError:
        return fallback
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > MAX_PLANNING_REPORT_BYTES
        ):
            return fallback
        chunks: list[bytes] = []
        remaining = MAX_PLANNING_REPORT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > MAX_PLANNING_REPORT_BYTES:
            return fallback
        document: Any = json.loads(encoded.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return fallback
    finally:
        os.close(descriptor)
    if not isinstance(document, dict):
        return fallback

    raw_error = document.get("error")
    detail = ""
    if isinstance(raw_error, str):
        detail = " ".join(
            "".join(
                character if character.isprintable() else " "
                for character in raw_error
            ).split()
        )
        if len(detail) > MAX_PLANNER_ERROR_CHARS:
            detail = detail[: MAX_PLANNER_ERROR_CHARS - 1].rstrip() + "…"

    raw_rejections = document.get("rejections")
    stage_counts: dict[str, int] = {}
    if isinstance(raw_rejections, list):
        for rejection in raw_rejections[:MAX_REJECTIONS_TO_SUMMARIZE]:
            if not isinstance(rejection, dict):
                continue
            stage = rejection.get("stage")
            if (
                isinstance(stage, str)
                and re.fullmatch(r"[a-z][a-z0-9_/-]{0,31}", stage)
            ):
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

    raw_total = document.get("rejection_count")
    total = (
        raw_total
        if isinstance(raw_total, int)
        and not isinstance(raw_total, bool)
        and 0 <= raw_total <= 1_000_000
        else sum(stage_counts.values())
    )
    summary = ""
    if total or stage_counts:
        counts = ", ".join(
            f"{stage}={count}"
            for stage, count in sorted(
                stage_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        summary = f"rejection summary: {total} total"
        if counts:
            summary += f" ({counts})"

    if detail and summary:
        return f"{detail}; {summary}"
    if detail:
        return detail
    if summary:
        return f"offline planner produced no valid grasp plan; {summary}"
    return fallback


def _planning_failure_disposition(output_dir: Path) -> str | None:
    """Read a typed planner disposition from a bounded regular report."""

    from z_manip.planning.handoff_disposition import classify_planning_report

    report_path = output_dir / "planning" / "planning_report.json"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(report_path, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > MAX_PLANNING_REPORT_BYTES
        ):
            return None
        encoded = os.read(descriptor, MAX_PLANNING_REPORT_BYTES + 1)
        if len(encoded) > MAX_PLANNING_REPORT_BYTES:
            return None
        document: Any = json.loads(encoded.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    finally:
        os.close(descriptor)
    disposition = classify_planning_report(document)
    return None if disposition is None else disposition.state


class FixedReadOnlyBackend:
    """Production adapter containing only repository-owned fixed commands."""

    def __init__(
        self,
        runtime: ServerRuntimeConfig | None = None,
    ) -> None:
        self.runtime = runtime or ServerRuntimeConfig.from_server_environment()

    @staticmethod
    def _ssh_prefix() -> tuple[str, ...]:
        # Reuse the authenticated fixed-host transport across the short passive
        # probe and report fetch.  This removes repeated SSH handshakes while
        # preserving the exact receive-only remote command surface.
        return (
            "/usr/bin/ssh",
            "-i",
            str(NUC_KEY),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            f"ControlPath={NUC_KEY.parent / 'z-manip-%C'}",
            NUC_HOST,
        )

    @staticmethod
    def _passive_report_valid(path: Path) -> bool:
        try:
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(document, dict)
            and document.get("schema")
            == "z_manip.piper_passive_joint_report.v1"
            and document.get("read_only") is True
            and document.get("complete_joint_feedback") is True
            and document.get("zero_transmit_verified") is True
            and document.get("interface_tx_packet_delta") == 0
        )

    @staticmethod
    def _typed_session_gate_block(path: Path) -> BackendResult | None:
        """Return a recoverable gate disposition only from complete evidence.

        A non-zero gate process is normally fail-closed as
        ``SESSION_GATE_BLOCKED``.  ``NEED_BASE_APPROACH`` is the sole typed
        exception because it is not an IK failure: the immutable target cloud
        is simply outside the handoff workspace.  Validate the entire safety
        envelope before trusting that disposition so a truncated or forged
        report cannot downgrade another gate failure into a recoverable one.
        """

        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or not 1 <= path.stat().st_size <= MAX_SESSION_GATE_REPORT_BYTES
            ):
                return None
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(document, dict):
            return None
        workspace = document.get("handoff_workspace")
        errors = document.get("errors")
        safety_valid = bool(
            document.get("schema") == "z_manip.piper_planning_session_gate.v1"
            and document.get("planning_ready") is False
            and document.get("read_only") is True
            and document.get("planning_only") is True
            and document.get("motion_commands_published") == 0
            and document.get("transport_opened") is False
            and document.get("planning_disposition") == "NEED_BASE_APPROACH"
        )
        workspace_valid = bool(
            isinstance(workspace, dict)
            and workspace.get("state") == "NEED_BASE_APPROACH"
            and workspace.get("planning_allowed") is False
            and workspace.get("frame") == "piper_base_link"
        )
        error_valid = bool(
            isinstance(errors, list)
            and any(
                isinstance(error, dict)
                and error.get("code") == "NEED_BASE_APPROACH"
                for error in errors
            )
        )
        if not (safety_valid and workspace_valid and error_valid):
            return None
        try:
            target_range_m = float(workspace["target_range_m"])
            maximum_range_m = float(workspace["maximum_handoff_range_m"])
            if not (
                math.isfinite(target_range_m)
                and math.isfinite(maximum_range_m)
                and target_range_m > maximum_range_m > 0.0
            ):
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        return BackendResult(
            1,
            "NEED_BASE_APPROACH",
            "target remains outside the manipulation handoff workspace "
            f"({target_range_m:.3f} m > {maximum_range_m:.3f} m); "
            "continue base approach before retrying close-range planning",
        )

    def _capture_passive_window(
        self,
        output_dir: Path,
        log_path: Path,
        environment: dict[str, str],
    ) -> BackendResult:
        # The probe atomically writes the remote report and prints the exact
        # same JSON document to stdout. Capture that stdout directly into the
        # local inflight file: a second SSH ``cat`` round-trip used to dominate
        # the warm-track UI path even though it added no safety evidence.
        # stderr remains in the action log for actionable SSH/probe failures.
        passive_command = self._ssh_prefix() + (
            "/usr/bin/python3",
            REMOTE_PASSIVE_PROBE,
            "--interface",
            "can0",
            "--duration",
            PASSIVE_CAPTURE_SECONDS,
            "--output",
            REMOTE_PASSIVE_REPORT,
        )
        live_report = output_dir / "live_passive_joint_report.json"
        temporary_report = output_dir / ".passive_joint_report.inflight"
        temporary_report.unlink(missing_ok=True)
        with temporary_report.open("xb") as report_output, log_path.open("ab") as log:
            passive = subprocess.run(
                passive_command,
                stdin=subprocess.DEVNULL,
                stdout=report_output,
                stderr=log,
                env=environment,
                shell=False,
                check=False,
            )
        if passive.returncode != 0:
            temporary_report.unlink(missing_ok=True)
            return BackendResult(
                passive.returncode,
                "PASSIVE_JOINT_GATE_FAILED",
                "fixed receive-only passive joint gate failed",
            )
        if (
            not 1 <= temporary_report.stat().st_size <= MAX_PASSIVE_REPORT_BYTES
            or not self._passive_report_valid(temporary_report)
        ):
            temporary_report.unlink(missing_ok=True)
            return BackendResult(
                1,
                "PASSIVE_JOINT_REPORT_INVALID",
                "passive joint report lacks zero-TX evidence",
            )
        temporary_report.replace(live_report)
        return BackendResult(0)

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    @staticmethod
    def _perception_outputs_valid(output_dir: Path, target: str) -> bool:
        required = (
            output_dir / "report.json",
            output_dir / "edgetam_mask.png",
            output_dir / "edgetam_overlay.png",
            output_dir / "grasp_candidates.npz",
            output_dir / "grasp_candidates_overlay.png",
            output_dir / "scene_collision_points.npy",
            output_dir / "selected_passive_joint_report.json",
            output_dir / "target_points.npy",
        )
        selected_passive = output_dir / "selected_passive_joint_report.json"
        report = FixedReadOnlyBackend._perception_report(output_dir)
        return bool(
            all(path.is_file() and not path.is_symlink() for path in required)
            and selected_passive.stat().st_size <= MAX_PASSIVE_REPORT_BYTES
            and FixedReadOnlyBackend._passive_report_valid(selected_passive)
            and report is not None
            and report.get("read_only") is True
            and report.get("instruction") == target
        )

    @staticmethod
    def _perception_report(output_dir: Path) -> dict[str, object] | None:
        report_path = output_dir / "report.json"
        try:
            if (
                not report_path.is_file()
                or report_path.is_symlink()
                or not 1 <= report_path.stat().st_size <= MAX_PERCEPTION_REPORT_BYTES
            ):
                return None
            value: Any = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _bounded_perception_detail(value: object) -> str:
        if not isinstance(value, str):
            return ""
        detail = " ".join(
            "".join(
                character if character.isprintable() else " "
                for character in value
            ).split()
        )
        if len(detail) > MAX_PERCEPTION_ERROR_CHARS:
            detail = detail[: MAX_PERCEPTION_ERROR_CHARS - 1].rstrip() + "…"
        return detail

    @classmethod
    def _perception_failure_result(
        cls,
        output_dir: Path,
        return_code: int,
    ) -> BackendResult:
        report = cls._perception_report(output_dir)
        if report is None:
            return BackendResult(
                return_code,
                "PERCEPTION_PROCESS_FAILED",
                "read-only perception process failed without a valid report",
            )
        failure = cls._bounded_perception_detail(report.get("perception_failure"))
        grasp_error = cls._bounded_perception_detail(
            report.get("grasp_generation_error"),
        )
        if failure.startswith("camera_frame_timeout"):
            return BackendResult(
                return_code,
                "PERCEPTION_CAMERA_FRAME_TIMEOUT",
                "RGB-D metadata arrived but the grounding bridge did not receive "
                "a synchronized camera frame after restart",
            )
        if failure.startswith("grounding_failed"):
            return BackendResult(
                return_code,
                "PERCEPTION_TARGET_NOT_FOUND",
                failure,
            )
        if failure.startswith("tracker_reported_loss"):
            return BackendResult(
                return_code,
                "PERCEPTION_TRACKER_LOST",
                failure,
            )
        if grasp_error:
            return BackendResult(
                return_code,
                "GRASP_GEOMETRY_FAILED",
                grasp_error,
            )
        detail = failure or cls._bounded_perception_detail(report.get("error"))
        return BackendResult(
            return_code,
            "PERCEPTION_PROCESS_FAILED",
            detail or "read-only perception process failed",
        )

    @classmethod
    def _perception_retryable(cls, output_dir: Path, return_code: int) -> bool:
        """Retry only failures that a fresh segmentation seed can recover.

        Exit 4 is the explicit post-capture grasp-geometry failure.  An object
        OBB larger than the physical aperture is deterministic for the frozen
        frame, while an ambiguous contact mask may recover on a new seed.
        A camera timeout is retried only when CameraInfo proves that the RGB-D
        source is alive and DDS discovery, rather than hardware, raced startup.
        """

        report = cls._perception_report(output_dir)
        if report is None:
            return return_code == 4
        if return_code == 4:
            filtered_points = report.get("filtered_target_points")
            if (
                not isinstance(filtered_points, bool)
                and isinstance(filtered_points, int)
                and filtered_points > 0
            ):
                # A real identity-valid bundle exists; grasp geometry at this
                # range is deterministic for the frozen scene and the mobile
                # flow advances on a seeded rc4, so a fresh seed would only
                # add seconds and replace the live track that downstream
                # target streaming depends on.
                return False
            grasp_error = report.get("grasp_generation_error")
            return not (
                isinstance(grasp_error, str)
                and "no OBB dimension within gripper aperture" in grasp_error
            )
        if return_code != 5:
            return False
        failure = report.get("perception_failure")
        if isinstance(failure, str) and failure.startswith("tracker_reported_loss"):
            return True
        if not (
            isinstance(failure, str)
            and failure.startswith("camera_frame_timeout")
        ):
            return False
        counts = report.get("message_counts")
        return bool(
            isinstance(counts, dict)
            and isinstance(counts.get("info"), int)
            and counts["info"] >= 5
        )

    @staticmethod
    def _perception_runner_running() -> bool:
        """Return whether the fixed read-only warm runner is available."""

        completed = subprocess.run(
            (
                "/usr/bin/docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                PERCEPTION_RUNNER_CONTAINER,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_server_environment(),
            shell=False,
            check=False,
        )
        return completed.returncode == 0 and completed.stdout.strip() == b"true"

    @staticmethod
    def _planning_runner_running() -> bool:
        """Return whether the fixed network-disabled planner runner is warm."""

        completed = subprocess.run(
            (
                "/usr/bin/docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                PLANNING_RUNNER_CONTAINER,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_server_environment(),
            shell=False,
            check=False,
        )
        return completed.returncode == 0 and completed.stdout.strip() == b"true"

    def run_perception(
        self,
        *,
        target: str,
        output_dir: Path,
        log_path: Path,
    ) -> BackendResult:
        """Run perception while repeatedly capturing synchronized passive joints."""

        total_started = time.monotonic()
        for path in (NUC_KEY, DDS_CONFIG, PERCEPTION):
            if not path.is_file():
                return BackendResult(
                    1,
                    "SERVER_PREFLIGHT_FAILED",
                    f"required server-owned input is unavailable: {path.name}",
                )

        environment = _server_environment()
        environment.update({
            "ROS_DOMAIN_ID": "20",
            "Z_MANIP_RUNTIME_IMAGE": self.runtime.runtime_image,
            "Z_MANIP_ARTIFACT_DIR": str(output_dir),
            "Z_MANIP_REQUIRE_PASSIVE_WINDOW": "1",
        })
        log_path.parent.mkdir(parents=True, exist_ok=True)
        runner_output: Path | None = None
        runner_socket: Path | None = None
        runner_probe_started = time.monotonic()
        try:
            relative_output = output_dir.resolve().relative_to(
                ARTIFACT_ROOT.resolve(),
            )
            candidate = PERCEPTION_RUNNER_ARTIFACT_ROOT / relative_output
            fixed_socket = ARTIFACT_ROOT / "go2w_real" / ".perception_runner.sock"
            if _fixed_worker_socket_available(fixed_socket):
                runner_output = candidate
                runner_socket = fixed_socket
            elif self._perception_runner_running():
                runner_output = candidate
        except ValueError:
            # Tests and explicitly isolated callers may use a temporary output
            # outside the shared immutable artifact tree. Keep the former
            # one-shot container as a safe compatibility fallback.
            pass
        runner_probe_s = time.monotonic() - runner_probe_started
        if runner_output is not None and runner_socket is None:
            command_prefix = (
                "/usr/bin/docker",
                "exec",
                PERCEPTION_RUNNER_CONTAINER,
                "z-manip-go2w-perception-worker",
                "client",
                "--",
            )
            artifact_output = str(runner_output)
        elif runner_output is None:
            command_prefix = (
                "/usr/bin/docker",
                "run",
                "--rm",
                "--user",
                f"{os.geteuid()}:{os.getegid()}",
                "--network",
                "host",
                "-e",
                "HOME=/tmp/z-manip",
                "-e",
                "ROS_LOG_DIR=/tmp/z-manip-ros-logs",
                "-e",
                "ROS_DOMAIN_ID=20",
                "-e",
                "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
                "-e",
                "CYCLONEDDS_URI=file:///config/cyclonedds.xml",
                "-e",
                "PYTHONPATH=/opt/z_manip/python",
                "-v",
                f"{DDS_CONFIG}:/config/cyclonedds.xml:ro",
                "-v",
                (
                    f"{PERCEPTION}:"
                    "/usr/local/bin/z-manip-go2w-perception-dry-run:ro"
                ),
                "-v",
                f"{STACK_ROOT / 'z_manip'}:/opt/z_manip/python/z_manip:ro",
                "-v",
                f"{output_dir}:/artifacts",
                self.runtime.runtime_image,
            )
            artifact_output = "/artifacts"
        else:
            command_prefix = ()
            artifact_output = str(runner_output)
        dry_run_program = () if runner_output is not None else (
            "z-manip-go2w-perception-dry-run",
        )
        base_perception_args = (
            "--instruction",
            target,
            "--output",
            artifact_output,
            "--passive-window",
            f"{artifact_output}/live_passive_joint_report.json",
            "--selected-passive-window",
            f"{artifact_output}/selected_passive_joint_report.json",
            "--timeout",
            "15",
            "--min-bundle-target-points",
            "400",
            # Keep 400 as the near-field ceiling but scale the demand down with
            # the seed's median depth so a genuinely far, small target is not
            # geometrically killed by a flat count sized for a near object.
            "--distance-aware-bundle-gate",
        )
        perception_args = base_perception_args + (
            # A close-range handoff commonly asks for the exact same target
            # that EdgeTAM is already tracking.  The dry-run accepts reuse only
            # when the bridge reports a valid track with the exact instruction
            # SHA-256; otherwise it publishes a fresh grounding transaction.
            # This removes a redundant YOLOE forward and tracker re-seed without
            # allowing stale or semantically different geometry through.
            "--reuse-valid-tracking",
            # Keep the identity/age quality gate explicit at the UI boundary;
            # the resident worker must not turn into a long-lived result cache.
            "--tracking-reuse-max-age",
            "0.5",
        )
        command = command_prefix + dry_run_program + perception_args
        return_code = 1
        passive_capture_s_total = 0.0
        passive_capture_count_total = 0
        for attempt in range(PERCEPTION_ATTEMPTS):
            attempt_args = perception_args
            attempt_command = command
            if attempt:
                # _perception_retryable only admits failures a fresh
                # segmentation seed can recover.  Tracking reuse would replay
                # the exact mask that just failed — a drifted persistent
                # EdgeTAM track stays within the reuse age forever at
                # streaming rate — so the retry must force a new grounding
                # transaction instead of reusing the live track.
                attempt_args = base_perception_args
                attempt_command = (
                    command_prefix + dry_run_program + attempt_args
                )
                for name in (
                    "report.json",
                    "edgetam_mask.png",
                    "edgetam_overlay.png",
                    "grasp_candidates.npz",
                    "grasp_candidates_overlay.png",
                    "scene_collision_points.npy",
                    "selected_passive_joint_report.json",
                    "target_points.npy",
                ):
                    (output_dir / name).unlink(missing_ok=True)
                with log_path.open("ab") as log:
                    log.write(
                        b"Retrying perception with a fresh grounding seed"
                        b" after an invalid geometric mask.\n",
                    )
            attempt_started = time.monotonic()
            process_launch_started = time.monotonic()
            process: subprocess.Popen[bytes] | None = None
            worker_result: list[_WorkerResult] = []
            worker_error: list[Exception] = []
            worker_thread: threading.Thread | None = None
            worker_elapsed_s: float | None = None
            if runner_socket is not None:
                def request_worker() -> None:
                    try:
                        worker_result.append(_run_fixed_worker_request(
                            runner_socket,
                            {"argv": list(attempt_args)},
                            log_path,
                            # Recompute from the current checkout for every
                            # action.  A long-lived UI must reject a worker
                            # that was started before a source/config update,
                            # even when the UI itself has not restarted yet.
                            expected_fingerprint=runtime_fingerprint(),
                        ))
                    except Exception as error:  # surfaced on this request
                        worker_error.append(error)

                worker_thread = threading.Thread(
                    target=request_worker,
                    name="z-manip-perception-request",
                    daemon=True,
                )
                worker_thread.start()
            else:
                with log_path.open("ab") as log:
                    process = subprocess.Popen(
                        attempt_command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        shell=False,
                    )
            process_launch_s = time.monotonic() - process_launch_started
            passive_capture_s = 0.0
            passive_capture_count = 0
            try:
                while (
                    worker_thread.is_alive()
                    if worker_thread is not None
                    else process is not None and process.poll() is None
                ):
                    selected_passive = (
                        output_dir / "selected_passive_joint_report.json"
                    )
                    if (
                        selected_passive.is_file()
                        and not selected_passive.is_symlink()
                        and selected_passive.stat().st_size
                        <= MAX_PASSIVE_REPORT_BYTES
                        and self._passive_report_valid(selected_passive)
                    ):
                        # The dry-run atomically selected this exact zero-TX
                        # evidence. Repeating SSH capture after selection adds
                        # latency but cannot strengthen this immutable request.
                        if worker_thread is not None:
                            worker_thread.join(timeout=0.01)
                        else:
                            time.sleep(0.01)
                        continue
                    passive_capture_started = time.monotonic()
                    passive = self._capture_passive_window(
                        output_dir,
                        log_path,
                        environment,
                    )
                    capture_elapsed = time.monotonic() - passive_capture_started
                    passive_capture_s += capture_elapsed
                    passive_capture_s_total += capture_elapsed
                    passive_capture_count += 1
                    passive_capture_count_total += 1
                    if passive.exit_code != 0:
                        return passive
                if worker_thread is not None:
                    worker_thread.join()
                    if worker_error or not worker_result:
                        detail = (
                            str(worker_error[0])
                            if worker_error
                            else "resident worker returned no result"
                        )
                        with log_path.open("ab") as log:
                            log.write(
                                f"resident perception transport failed: {detail}\n".encode(
                                    "utf-8", errors="replace",
                                ),
                            )
                        return_code = 70
                    else:
                        return_code = worker_result[0].returncode
                        worker_elapsed_s = worker_result[0].worker_elapsed_s
                elif process is not None:
                    return_code = process.wait()
            finally:
                if process is not None:
                    self._stop_process(process)
            _append_timing(
                log_path,
                "perception_attempt",
                time.monotonic() - attempt_started,
                attempt=attempt + 1,
                return_code=return_code,
                runner_warm=runner_output is not None,
                runner_transport=(
                    "unix_socket" if runner_socket is not None else "subprocess"
                ),
                worker_elapsed_s=worker_elapsed_s,
                process_launch_s=round(process_launch_s, 6),
                passive_capture_s=round(passive_capture_s, 6),
                passive_capture_count=passive_capture_count,
            )
            if not self._perception_retryable(output_dir, return_code):
                break
        output_validation_started = time.monotonic()
        outputs_valid = self._perception_outputs_valid(output_dir, target)
        output_validation_s = time.monotonic() - output_validation_started
        report = self._perception_report(output_dir)
        internal_elapsed = (
            float(report["elapsed_s"])
            if report is not None
            and isinstance(report.get("elapsed_s"), (int, float))
            else None
        )
        total_elapsed = time.monotonic() - total_started
        timing_fields: dict[str, object] = {
            "attempts": attempt + 1,
            "return_code": return_code,
            "runner_warm": runner_output is not None,
            "runner_probe_s": round(runner_probe_s, 6),
            "passive_capture_s": round(passive_capture_s_total, 6),
            "passive_capture_count": passive_capture_count_total,
            "output_validation_s": round(output_validation_s, 6),
            "target_identity_valid": bool(
                report is not None and report.get("instruction") == target
            ),
        }
        if internal_elapsed is not None:
            timing_fields["internal_elapsed_s"] = round(internal_elapsed, 6)
            timing_fields["wrapper_overhead_s"] = round(
                max(0.0, total_elapsed - internal_elapsed),
                6,
            )
        if report is not None:
            timing_fields["grounding_mode"] = (
                "reused_tracking"
                if report.get("grounding_reused") is True
                else "fresh_grounding"
            )
        _append_timing(
            log_path,
            "perception_total",
            total_elapsed,
            **timing_fields,
        )
        if return_code == 0 and not outputs_valid:
            return BackendResult(
                1,
                "PERCEPTION_OUTPUT_INVALID",
                "perception omitted synchronized joints or fixed UI overlays",
            )
        if return_code == 0:
            return BackendResult(0)
        return self._perception_failure_result(output_dir, return_code)

    @staticmethod
    def _required_planning_files() -> tuple[Path, ...]:
        return (
            SESSION_GATE,
            PLANNER,
            PLANNING_WORKER,
            STACK_CONFIG,
            DEBUG_BUNDLE,
            SAFETY_GATE,
            CALIBRATION,
            URDF,
        )

    def _build_visualization_bundle(
        self,
        *,
        perception_dir: Path,
        output_dir: Path,
        joint_report: Path,
        log_path: Path,
        environment: dict[str, str],
    ) -> BackendResult:
        bundle = output_dir / "debug_bundle.json"
        planning_dir = output_dir / "planning"
        arguments = [
            sys.executable,
            str(DEBUG_BUNDLE),
            "--perception-dir",
            str(perception_dir),
            "--joint-report",
            str(joint_report),
            "--calibration",
            str(CALIBRATION),
            "--urdf",
            str(URDF),
            "--output",
            str(bundle),
        ]
        session_gate = output_dir / "session_gate.json"
        if session_gate.is_file():
            arguments.extend(("--session-gate", str(session_gate)))
        if (planning_dir / "planning_report.json").is_file():
            arguments.extend(("--planning-dir", str(planning_dir)))
        built = _run_logged(arguments, log_path, environment=environment)
        if built.returncode != 0:
            return BackendResult(
                built.returncode,
                "DEBUG_BUNDLE_FAILED",
                "fixed offline visualization bundle could not be built",
            )
        try:
            document: Any = json.loads(bundle.read_text(encoding="utf-8"))
            images = document["visualization"]["images"]
            safety = document["safety"]
            valid = bool(
                document.get("schema") == "z_manip.debug_bundle.v1"
                and set(images) == {
                    "segmentation_mask",
                    "segmentation_overlay",
                    "candidate_overlay",
                }
                and safety.get("motion_commands_published") == 0
                and safety.get("transport_opened") is False
                and safety.get("can_opened") is False
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            valid = False
        if not valid:
            return BackendResult(
                1,
                "DEBUG_BUNDLE_INVALID",
                "visualization bundle is missing fixed read-only evidence",
            )

        audit = output_dir / "debug_bundle.safety-audit.json"
        audited = _run_logged(
            (
                sys.executable,
                str(SAFETY_GATE),
                "--bundle",
                str(bundle),
                "--artifact-root",
                str(WORKSPACE_ROOT / "artifacts"),
                "--joint-report",
                str(joint_report),
                "--output",
                str(audit),
            ),
            log_path,
            environment=environment,
        )
        if audited.returncode != 0:
            return BackendResult(
                audited.returncode,
                "DEBUG_BUNDLE_SAFETY_GATE_FAILED",
                "visualization bundle did not pass its read-only safety audit",
            )
        try:
            audit_document: Any = json.loads(audit.read_text(encoding="utf-8"))
            audit_valid = bool(
                audit_document.get("schema")
                == "z_manip.debug_safety_audit.v1"
                and audit_document.get("passed") is True
                and audit_document.get("motion_commands_published") == 0
            )
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            audit_valid = False
        if not audit_valid:
            return BackendResult(
                1,
                "DEBUG_BUNDLE_SAFETY_AUDIT_INVALID",
                "visualization safety audit output is invalid",
            )
        return BackendResult(0)

    def run_planning(
        self,
        *,
        perception_dir: Path,
        output_dir: Path,
        log_path: Path,
    ) -> BackendResult:
        """Revalidate capture-time joints, then run fixed offline planning."""

        for path in self._required_planning_files():
            if not path.is_file():
                return BackendResult(
                    1,
                    "SERVER_PREFLIGHT_FAILED",
                    f"required server-owned input is unavailable: {path.name}",
                )

        total_started = time.monotonic()
        environment = _server_environment(python_path=True)
        joint_report = perception_dir / "selected_passive_joint_report.json"
        if not self._passive_report_valid(joint_report):
            return BackendResult(
                1,
                "PASSIVE_JOINT_REPORT_INVALID",
                "immutable perception session lacks synchronized zero-TX joints",
            )

        session_gate_report = output_dir / "session_gate.json"
        gate_started = time.monotonic()
        gate = _run_logged(
            (
                sys.executable,
                str(SESSION_GATE),
                "--perception-dir",
                str(perception_dir),
                "--joint-report",
                str(joint_report),
                "--calibration",
                str(CALIBRATION),
                "--urdf",
                str(URDF),
                "--output",
                str(session_gate_report),
            ),
            log_path,
            environment=environment,
        )
        _append_timing(
            log_path,
            "planning_session_gate",
            time.monotonic() - gate_started,
            return_code=gate.returncode,
        )
        if gate.returncode != 0:
            visualization = self._build_visualization_bundle(
                perception_dir=perception_dir,
                output_dir=output_dir,
                joint_report=joint_report,
                log_path=log_path,
                environment=environment,
            )
            if visualization.exit_code != 0:
                return visualization
            typed_block = self._typed_session_gate_block(session_gate_report)
            if typed_block is not None:
                return BackendResult(
                    gate.returncode,
                    typed_block.error_code,
                    typed_block.message,
                )
            return BackendResult(
                gate.returncode,
                "SESSION_GATE_BLOCKED",
                "passive joint/perception/calibration session gate blocked planning",
            )
        try:
            gate_document: Any = json.loads(
                session_gate_report.read_text(encoding="utf-8"),
            )
            if not isinstance(gate_document, dict):
                raise ValueError("session gate is not an object")
            measured_csv = _six_joint_csv(
                gate_document.get("measured_joints_rad"),
                "measured_joints_rad",
            )
            planning_csv = _six_joint_csv(
                gate_document.get("planning_start_joints_rad"),
                "planning_start_joints_rad",
            )
            if gate_document.get("planning_ready") is not True:
                raise ValueError("session gate is not planning-ready")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            visualization = self._build_visualization_bundle(
                perception_dir=perception_dir,
                output_dir=output_dir,
                joint_report=joint_report,
                log_path=log_path,
                environment=environment,
            )
            if visualization.exit_code != 0:
                return visualization
            return BackendResult(
                1,
                "SESSION_GATE_OUTPUT_INVALID",
                f"session gate output is invalid: {error}",
            )

        planning_dir = output_dir / "planning"
        planning_dir.mkdir(mode=0o700)
        runner_perception: Path | None = None
        runner_planning: Path | None = None
        runner_scratch: Path | None = None
        runner_socket: Path | None = None
        try:
            relative_perception = perception_dir.resolve().relative_to(
                ARTIFACT_ROOT.resolve(),
            )
            fixed_socket = PLANNING_RUNNER_SCRATCH_ROOT / ".planner.sock"
            runner_available = _fixed_worker_socket_available(fixed_socket)
            if runner_available or self._planning_runner_running():
                runner_perception = (
                    PLANNING_RUNNER_ARTIFACT_ROOT / relative_perception
                )
                if runner_available:
                    runner_socket = fixed_socket
                # The warm runner sees all immutable perception/calibration
                # evidence read-only.  It can write only a fresh, server-owned
                # scratch directory; the host atomically promotes that output
                # into this action after the planner process exits.
                PLANNING_RUNNER_SCRATCH_ROOT.mkdir(
                    mode=0o700,
                    parents=True,
                    exist_ok=True,
                )
                _cleanup_stale_planning_runner_scratch(
                    PLANNING_RUNNER_SCRATCH_ROOT,
                )
                runner_scratch = Path(tempfile.mkdtemp(
                    prefix="planning-",
                    dir=PLANNING_RUNNER_SCRATCH_ROOT,
                ))
                runner_planning = (
                    PLANNING_RUNNER_CONTAINER_SCRATCH_ROOT
                    / runner_scratch.name
                )
        except ValueError:
            # Tests and isolated callers outside the fixed artifact root retain
            # the former one-shot, network-disabled compatibility path.
            pass

        planner_args = (
            "z-manip-piper-planning-dry-run",
            "--artifacts",
            str(runner_perception or Path("/session/perception")),
            "--config",
            "/opt/z_manip/configs/go2w_piper.json",
            "--urdf",
            CONTAINER_URDF,
            f"--joints={measured_csv}",
            f"--planning-joints={planning_csv}",
            "--search-timeout-s",
            SEARCH_TIMEOUT_S,
            "--symmetry-samples",
            SYMMETRY_SAMPLES,
            "--max-hypotheses",
            MAX_HYPOTHESES,
            "--max-candidates",
            MAX_CANDIDATES,
            "--max-feasible-plans",
            MAX_FEASIBLE_PLANS,
            "--support-approach-prior-weight",
            SUPPORT_APPROACH_PRIOR_WEIGHT,
            "--scene-clearance-m",
            SUPERVISED_SCENE_CLEARANCE_M,
            "--scene-point-radius-m",
            SUPERVISED_SCENE_POINT_RADIUS_M,
            "--gripper-scene-radius-scale",
            SUPERVISED_GRIPPER_SCENE_RADIUS_SCALE,
            "--camera-calibration",
            (
                str(
                    PLANNING_RUNNER_ARTIFACT_ROOT
                    / CALIBRATION.resolve().relative_to(ARTIFACT_ROOT.resolve())
                )
                if runner_perception is not None
                else "/session/calibration.json"
            ),
            "--output",
            str(runner_planning or Path("/session/planning")),
        )
        if runner_perception is not None and runner_socket is None:
            planner_command = (
                "/usr/bin/docker",
                "exec",
                "-e",
                f"Z_MANIP_IK_BACKEND={self.runtime.ik_backend}",
                PLANNING_RUNNER_CONTAINER,
                "z-manip-piper-planning-worker",
                "client",
                "--",
                *planner_args[1:],
            )
        elif runner_perception is None:
            planner_command = (
                "/usr/bin/docker",
                "run",
                "--rm",
                "--user",
                f"{os.geteuid()}:{os.getegid()}",
                "--network",
                "none",
                "-e",
                "HOME=/tmp/z-manip",
                "-e",
                f"Z_MANIP_IK_BACKEND={self.runtime.ik_backend}",
                "-v",
                f"{perception_dir}:/session/perception:ro",
                "-v",
                f"{planning_dir}:/session/planning",
                "-v",
                f"{CALIBRATION}:/session/calibration.json:ro",
                "-v",
                f"{ROBOT_ASSETS}:/robot_assets:ro",
                "-v",
                f"{PLANNER}:/usr/local/bin/z-manip-piper-planning-dry-run:ro",
                "-v",
                f"{PLANNING_WORKER}:/usr/local/bin/z-manip-piper-planning-worker:ro",
                "-v",
                f"{STACK_CONFIG}:/opt/z_manip/configs/go2w_piper.json:ro",
                "-v",
                f"{STACK_ROOT / 'z_manip'}:/opt/z_manip/python/z_manip:ro",
                self.runtime.runtime_image,
                *planner_args,
            )
        else:
            planner_command = ()
        planner_started = time.monotonic()
        try:
            if runner_socket is not None:
                planner = _run_fixed_worker_request(
                    runner_socket,
                    {
                        "argv": list(planner_args[1:]),
                        "ik_backend": self.runtime.ik_backend,
                    },
                    log_path,
                    expected_fingerprint=runtime_fingerprint(),
                )
            else:
                planner = _run_logged(
                    planner_command,
                    log_path,
                    environment=_server_environment(),
                )
        except (OSError, RuntimeError, ValueError) as error:
            with log_path.open("ab") as log:
                log.write(
                    f"resident planning transport failed: {error}\n".encode(
                        "utf-8", errors="replace",
                    ),
                )
            _append_timing(
                log_path,
                "planning_search",
                time.monotonic() - planner_started,
                return_code=None,
            )
            if runner_scratch is not None:
                shutil.rmtree(runner_scratch, ignore_errors=True)
                return BackendResult(
                    1,
                    "PLANNING_RUNNER_UNAVAILABLE",
                    "warm planner process could not be started",
                )
            raise
        _append_timing(
            log_path,
            "planning_search",
            time.monotonic() - planner_started,
            return_code=planner.returncode,
            runner_transport=(
                "unix_socket" if runner_socket is not None else "subprocess"
            ),
            worker_elapsed_s=(
                planner.worker_elapsed_s
                if isinstance(planner, _WorkerResult)
                else None
            ),
        )
        if runner_scratch is not None:
            runner_report = runner_scratch / "planning_report.json"
            if not _planning_runner_report_valid(runner_report):
                shutil.rmtree(runner_scratch, ignore_errors=True)
                return BackendResult(
                    planner.returncode or 1,
                    "PLANNING_RUNNER_OUTPUT_MISSING",
                    "warm planner exited without a valid bounded planning report",
                )
            try:
                # ``planning_dir`` is still empty: no consumer can observe a
                # partially copied report, and inputs were never writable by
                # the container.  Both paths share the artifact filesystem.
                planning_dir.rmdir()
                os.replace(runner_scratch, planning_dir)
            except OSError as error:
                shutil.rmtree(runner_scratch, ignore_errors=True)
                if not planning_dir.exists():
                    planning_dir.mkdir(mode=0o700)
                return BackendResult(
                    1,
                    "PLANNING_RUNNER_OUTPUT_INVALID",
                    f"warm planner output could not be promoted: {error}",
                )
        ready_evidence = (
            _planning_ready_evidence(
                perception_dir=perception_dir,
                output_dir=output_dir,
                joint_report=joint_report,
            )
            if planner.returncode == 0
            else None
        )
        if ready_evidence is not None:
            _append_timing(
                log_path,
                "planning_ready_pre_visualization",
                time.monotonic() - total_started,
                **ready_evidence,
            )
        visualization_started = time.monotonic()
        visualization = self._build_visualization_bundle(
            perception_dir=perception_dir,
            output_dir=output_dir,
            joint_report=joint_report,
            log_path=log_path,
            environment=environment,
        )
        _append_timing(
            log_path,
            "planning_visualization_and_audit",
            time.monotonic() - visualization_started,
            return_code=visualization.exit_code,
        )
        if visualization.exit_code != 0:
            return visualization
        _append_timing(
            log_path,
            "planning_total",
            time.monotonic() - total_started,
            return_code=planner.returncode,
        )
        return BackendResult(
            planner.returncode,
            (
                None
                if planner.returncode == 0
                else _planning_failure_disposition(output_dir)
                or "OFFLINE_PLANNER_BLOCKED"
            ),
            "" if planner.returncode == 0 else _planning_failure_message(output_dir),
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    perception = actions.add_parser("perception")
    perception.add_argument("target")
    selection = actions.add_parser("select")
    selection.add_argument("session_id")
    actions.add_parser("planning")
    actions.add_parser("status")
    return parser.parse_args()


def main() -> int:
    """Execute one bounded CLI action and print its JSON response."""

    args = _arguments()
    service = ReadOnlySessionService(RUN_ROOT, FixedReadOnlyBackend())
    try:
        if args.action == "perception":
            response = service.run_perception(args.target)
        elif args.action == "select":
            response = service.select_perception(args.session_id)
        elif args.action == "planning":
            response = service.run_planning()
        else:
            response = service.status()
    except SessionContractError as error:
        response = {
            "schema": "z_manip.interactive_session_error.v1",
            "ok": False,
            "error": {"code": error.code, "message": str(error)},
        }
        print(json.dumps(response, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    return 0 if response.get("status") not in {"failed", "blocked"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
