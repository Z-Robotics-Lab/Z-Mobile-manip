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
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
STACK_ROOT = SCRIPT_DIR.parent.parent
WORKSPACE_ROOT = STACK_ROOT.parent
sys.path.insert(0, str(STACK_ROOT))
# ``z_manip_runtime_fingerprint`` below is a SIBLING FILE imported by top-level
# name, which used to work only because ``sys.path[0]`` happened to be this
# directory -- true when the server runs the script, false under a test that
# loads this file through an importlib spec.  The result was 39 failures in
# tests/test_read_only_sessions.py when that file is run ALONE and none when it
# is run after any test that had already inserted this directory: a suite whose
# verdict depended on collection order.  Name the directory explicitly.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from z_manip.read_only_sessions import (  # noqa: E402
    BackendResult,
    ReadOnlySessionService,
    SessionContractError,
)
from z_manip_runtime_fingerprint import (  # noqa: E402
    runtime_fingerprint,
    try_runtime_fingerprint,
)


RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "go2w_real" / "interactive_sessions"
ARTIFACT_ROOT = WORKSPACE_ROOT / "artifacts"
PERCEPTION = SCRIPT_DIR / "go2w_perception_dry_run.py"
SESSION_GATE = SCRIPT_DIR / "piper_planning_session_gate.py"
PLANNER = SCRIPT_DIR / "piper_planning_dry_run.py"
PLANNING_WORKER = SCRIPT_DIR / "piper_planning_worker.py"
STACK_CONFIG = STACK_ROOT / "configs" / "go2w_piper.json"
DEBUG_BUNDLE = SCRIPT_DIR / "go2w_debug_bundle.py"
SAFETY_GATE = SCRIPT_DIR / "go2w_debug_safety_gate.py"
COMPONENT_MANAGER = SCRIPT_DIR / "go2w_component_manager.sh"
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
# The remote probe rejects anything below 0.25 s and must observe all three
# joint-feedback frames inside the window, so this is already its floor and is
# not a latency knob.
PASSIVE_CAPTURE_SECONDS = "0.25"
# Supervision granularity for the perception worker and its in-flight passive
# window.  Small enough that the request returns on the worker's own clock
# instead of the fixed observation period's.
PASSIVE_CAPTURE_POLL_SECONDS = 0.01
PASSIVE_REPORT_SCHEMA = "z_manip.piper_passive_joint_report.v1"
# Every field the zero-TX verdict and the stamp-overlap gate are computed from.
# Used for a *presence* check only: it answers "did the probe finish writing
# this document", never "does the document attest zero transmit".
PASSIVE_REPORT_REQUIRED_FIELDS = (
    "read_only",
    "complete_joint_feedback",
    "zero_transmit_verified",
    "interface_tx_packet_delta",
    "observation_start_unix_ns",
    "observation_end_unix_ns",
    "joint_positions_rad",
    "joint_ranges_rad",
    "max_joint_range_rad",
    "joint_snapshot_span_s",
)
PERCEPTION_ATTEMPTS = 2
# The exact prefix ``go2w_perception_dry_run.py`` writes into
# ``report.perception_failure`` when it refuses a passive window that closed
# before its own bundle wait began.  Written once here and matched once below,
# so the producer and the consumer cannot drift into two spellings.
PASSIVE_WINDOW_STALE_FAILURE_PREFIX = "passive_window_stale"
MAX_PASSIVE_REPORT_BYTES = 1024 * 1024
MAX_SESSION_GATE_REPORT_BYTES = 256 * 1024
MAX_PLANNING_REPORT_BYTES = 4 * 1024 * 1024
MAX_PLANNED_GRASP_BYTES = 8 * 1024 * 1024
MAX_MODEL_EVIDENCE_BYTES = 8 * 1024 * 1024
PLANNING_RUNNER_SCRATCH_TTL_S = 24 * 60 * 60
MAX_PLANNER_ERROR_CHARS = 600
MAX_PERCEPTION_REPORT_BYTES = 256 * 1024
MAX_PERCEPTION_ERROR_CHARS = 600
# Bounded suffix of the worker log persisted into a failed attempt when the
# process died without a structured report.  Small and fixed, so the failure
# path never reads an unbounded log.
MAX_WORKER_STDERR_TAIL_BYTES = 16 * 1024
MAX_WORKER_STDERR_TAIL_LINES = 8
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


class _ResidentWorkerFingerprintMismatch(RuntimeError):
    """A resident worker reported a stale runtime fingerprint.

    Subclasses ``RuntimeError`` so existing callers that catch RuntimeError
    (e.g. the planning transport) keep their behaviour, while the perception
    path can recognise this specific, self-healable condition and restart the
    read-only perception component exactly once before retrying.
    """


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
        raise _ResidentWorkerFingerprintMismatch(
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
        # Cross-request observability counters on the resident backend.  These
        # are incremented only on cold paths (a completed perception attempt or
        # a rc=70 self-heal) and read back into the per-request timing record,
        # so they add no cost to the hot success path.
        #   * grounding reuse "hit" ratio: reused-tracking requests skip a fresh
        #     YOLOE+VLM grounding, so the ratio measures how often the VLM was
        #     avoided.
        #   * rc=70 fingerprint self-heal count and cumulative wall-clock cost.
        self._grounding_reuse_hits = 0
        self._grounding_requests_scored = 0
        self._fingerprint_selfheal_count = 0
        self._fingerprint_selfheal_cost_s = 0.0

    @staticmethod
    def runtime_fingerprint() -> str | None:
        """Fingerprint of the exact bytes the resident workers load right now.

        Recomputed per call, never cached: the whole point is that this checkout
        is the deployment (``go2w_depth_servo.sh`` bind-mounts it read-only and
        runs from it), so a cached value would report the tree as it was when
        this process started rather than as it is.  ``try_runtime_fingerprint``
        never raises -- a file can vanish between the ``is_file`` check and the
        read while somebody is saving.
        """

        return try_runtime_fingerprint()[0]

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
        """THE ZERO-TX GATE: does this document attest the arm never transmitted?

        This is a *verdict*, not a completeness check.  It is False both for a
        document that was killed mid-write and for a complete, well-formed
        document that affirmatively recorded the arm transmitting on CAN.  Those
        two cases must never be conflated on a discard path -- see
        ``_passive_report_structurally_complete``.
        """

        try:
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(document, dict)
            and document.get("schema") == PASSIVE_REPORT_SCHEMA
            and document.get("read_only") is True
            and document.get("complete_joint_feedback") is True
            and document.get("zero_transmit_verified") is True
            and document.get("interface_tx_packet_delta") == 0
        )

    @staticmethod
    def _passive_report_structurally_complete(path: Path) -> bool:
        """Did the probe FINISH WRITING this document?  Not: is the arm quiet.

        The abandon path needs to separate "this document never finished being
        written" from "this document is complete and says the arm transmitted".
        ``_passive_report_valid`` cannot make that distinction -- it returns
        False for both -- so using it to decide what may be deleted silently
        discards an affirmative zero-TX violation and reports the request as a
        success.  That is fail-open, and it is exactly what this predicate
        exists to prevent.

        Structural completeness is deliberately value-blind: parseable JSON, a
        bounded non-empty file, the right schema string, and every field the
        verdict is computed from present.  A document carrying
        ``zero_transmit_verified: false`` / ``interface_tx_packet_delta: 7`` is
        structurally complete, is therefore never deleted, and must reach the
        gate above so the request fails closed.
        """

        try:
            if (
                path.is_symlink()
                or not path.is_file()
                or not 1 <= path.stat().st_size <= MAX_PASSIVE_REPORT_BYTES
            ):
                return False
            document: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return bool(
            isinstance(document, dict)
            and document.get("schema") == PASSIVE_REPORT_SCHEMA
            and all(
                field in document for field in PASSIVE_REPORT_REQUIRED_FIELDS
            )
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
        *,
        stop: Callable[[], bool] | None = None,
    ) -> BackendResult:
        # The probe atomically writes the remote report and prints the exact
        # same JSON document to stdout. Capture that stdout directly into the
        # local inflight file: a second SSH ``cat`` round-trip used to dominate
        # the warm-track UI path even though it added no safety evidence.
        # stderr remains in the action log for actionable SSH/probe failures.
        #
        # ``stop`` is polled while the probe runs and cuts short a window the
        # caller has proven redundant.  The capture must never outlive this
        # call: the session layer hashes every file under ``output_dir`` into
        # the immutable artifact manifest as soon as the action returns, so a
        # background write would invalidate that manifest.  The remote scratch
        # report is rewritten by every probe run and its only other reader
        # (go2w_planning_session.sh) writes it immediately before reading it.
        #
        # Cutting a window short may only ever throw away a document that never
        # finished being written.  A probe that ran to completion and recorded a
        # transmission has produced the single most important result this whole
        # mechanism exists to obtain, and it fails the request whether or not
        # anybody was still waiting for it.
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
        abandoned = False
        try:
            with temporary_report.open("xb") as report_output, log_path.open("ab") as log:
                passive = subprocess.Popen(
                    passive_command,
                    stdin=subprocess.DEVNULL,
                    stdout=report_output,
                    stderr=log,
                    env=environment,
                    shell=False,
                )
                try:
                    while passive.poll() is None:
                        if stop is not None and stop():
                            abandoned = True
                            break
                        time.sleep(PASSIVE_CAPTURE_POLL_SECONDS)
                finally:
                    if abandoned:
                        self._stop_process(passive)
                    returncode = self._reap_process(passive)
        except BaseException:
            # Nothing may survive this call under any exit path: a leftover
            # ``.passive_joint_report.inflight`` in ``output_dir`` invalidates
            # the immutable artifact manifest hashed the instant we return.
            temporary_report.unlink(missing_ok=True)
            raise
        structurally_complete = self._passive_report_structurally_complete(
            temporary_report,
        )
        if abandoned and not structurally_complete:
            # ONLY a document that never finished being written is discarded.
            # A window cut short before the probe emitted anything is not a
            # failed gate and not missing evidence: the request that would have
            # consumed it already holds the zero-TX report it is judged on.
            temporary_report.unlink(missing_ok=True)
            return BackendResult(0)
        if returncode != 0 and not abandoned:
            # A probe that failed on its own is a failed gate, exactly as before,
            # whatever it managed to write.  Only a non-zero status caused by the
            # signal WE sent is discounted -- and then the document, not the exit
            # code, carries the verdict and is evaluated below.
            temporary_report.unlink(missing_ok=True)
            return BackendResult(
                returncode,
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
        """Best-effort teardown that never raises.

        This runs inside the ``finally:`` of ``_capture_passive_window``, still
        inside the open ``.passive_joint_report.inflight`` handle and on the
        perception request path.  An unguarded ``wait`` after ``kill()`` would
        propagate ``TimeoutExpired`` out of that path and strand the inflight
        file in ``output_dir``, invalidating the immutable artifact manifest.
        An unreapable child is left to the OS instead.
        """

        if process.poll() is not None:
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return

    @staticmethod
    def _reap_process(process: subprocess.Popen[bytes]) -> int:
        """Collect an exit status without being able to block forever.

        ``_stop_process`` above is best-effort, so a pathological child can
        still be running here.  Report a non-zero status rather than hanging the
        request: the caller treats that as a failed gate unless the probe left a
        structurally complete document behind.
        """

        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return 1

    @classmethod
    def _clear_inherited_attempt_outputs(cls, output_dir: Path) -> None:
        """Delete EVERY artifact a previous sub-attempt left in ``output_dir``.

        THE DEFECT.  A retry -- the inner fresh-seed retry below, or the rc=70
        resident-worker self-heal in ``run_perception`` -- reuses the SAME
        ``output_dir``.  The self-heal path cleared nothing at all, and the
        inner path cleared everything EXCEPT ``live_passive_joint_report.json``.
        Two things then went wrong at once:

        1. ``_selected_passive_report_valid`` saw the previous sub-attempt's
           selected report, so the supervision loop below never opened another
           passive window.  Recorded: ``passive_capture_count: 0`` on 6 of 6
           self-healed retries (20260727-080308, 20260728-030239, -041144,
           -062834, -070233, -081128), against 2-21 captures on their own first
           sub-attempts.
        2. The dry run's ``--passive-window`` therefore still pointed at a
           window that had closed ~18 s earlier, so its stamp-overlap gate
           rejected every fresh bundle: 911 and 1127 rejections with
           ``widest_rejected_overlap_margin_s`` of -17.8 s and -18.4 s, the full
           15 s budget burnt, and ``PERCEPTION_PROCESS_FAILED`` on a run whose
           own report says ``seed_accepted: true`` with 781 target points.

        Deleting is the fail-CLOSED direction: it can only force a fresh
        capture and a fresh grounding.  It can never make an invalid run look
        valid, because ``_perception_outputs_valid`` still requires every one of
        these files to exist and the selected report to pass the unchanged
        zero-TX gate before the attempt is called a success.

        ``live_passive_joint_report.json`` is listed here and NOT in
        ``_perception_outputs_valid``'s required set on purpose: it is the
        wrapper's inflight evidence, not a session artifact, and it is precisely
        the file whose survival poisoned the retry.
        """

        for name in (
            "report.json",
            "edgetam_mask.png",
            "edgetam_overlay.png",
            "grasp_candidates.npz",
            "grasp_candidates_overlay.png",
            "scene_collision_points.npy",
            "selected_passive_joint_report.json",
            "target_points.npy",
            "live_passive_joint_report.json",
        ):
            (output_dir / name).unlink(missing_ok=True)

    @staticmethod
    def _selected_passive_report_valid(output_dir: Path) -> bool:
        """Return whether the dry run already selected its zero-TX evidence."""

        selected = output_dir / "selected_passive_joint_report.json"
        return bool(
            selected.is_file()
            and not selected.is_symlink()
            and selected.stat().st_size <= MAX_PASSIVE_REPORT_BYTES
            and FixedReadOnlyBackend._passive_report_valid(selected)
        )

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
    def _worker_stderr_tail(cls, log_path: Path | None) -> str:
        """Return a bounded printable tail of the worker's stdout/stderr log.

        The perception worker writes its combined stdout/stderr to the action
        log.  When the process dies without a structured report, that tail is
        the only diagnostic; persist it into the failed attempt so the failure
        is inspectable without shelling into the host.  Reads at most a small
        fixed suffix, so there is no unbounded I/O even for large logs.
        """

        if log_path is None:
            return ""
        try:
            if log_path.is_symlink() or not log_path.is_file():
                return ""
            size = log_path.stat().st_size
            with log_path.open("rb") as handle:
                if size > MAX_WORKER_STDERR_TAIL_BYTES:
                    handle.seek(size - MAX_WORKER_STDERR_TAIL_BYTES)
                raw = handle.read(MAX_WORKER_STDERR_TAIL_BYTES)
        except OSError:
            return ""
        text = raw.decode("utf-8", errors="replace")
        # Drop the structured timing markers the worker emits; keep only the
        # human-readable diagnostic lines for the tail.
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("{")
        ]
        return cls._bounded_perception_detail(" | ".join(lines[-MAX_WORKER_STDERR_TAIL_LINES:]))

    @classmethod
    def _perception_failure_result(
        cls,
        output_dir: Path,
        return_code: int,
        *,
        log_path: Path | None = None,
    ) -> BackendResult:
        report = cls._perception_report(output_dir)
        if report is None:
            message = "read-only perception process failed without a valid report"
            tail = cls._worker_stderr_tail(log_path)
            if tail:
                message = f"{message}; worker stderr tail: {tail}"
            return BackendResult(
                return_code,
                "PERCEPTION_PROCESS_FAILED",
                message,
            )
        failure = cls._bounded_perception_detail(report.get("perception_failure"))
        grasp_error = cls._bounded_perception_detail(
            report.get("grasp_generation_error"),
        )
        if failure.startswith(PASSIVE_WINDOW_STALE_FAILURE_PREFIX):
            # NAMED, and deliberately its own code.  Before this existed the
            # symptom was indistinguishable from a dead worker: the run burnt
            # its whole budget rejecting bundles against a window nothing was
            # refreshing and surfaced as PERCEPTION_PROCESS_FAILED, which routed
            # a plainly detected object into a wrist search.  The remedy is a
            # fresh capture and nothing else -- the zero-TX gate itself is
            # untouched.
            return BackendResult(
                return_code,
                "PERCEPTION_PASSIVE_WINDOW_STALE",
                failure,
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
        A stale passive window is retried unconditionally: it is not a property
        of the scene at all, it is this wrapper having handed the dry run an
        already-closed observation, and the retry's own cleanup is what fixes
        it.
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
        if isinstance(failure, str) and failure.startswith(
            PASSIVE_WINDOW_STALE_FAILURE_PREFIX
        ):
            # FIX BY RE-CAPTURE, NEVER BY RELAXATION.  The next attempt runs
            # through ``_clear_inherited_attempt_outputs``, which deletes the
            # stale window, which is exactly what makes the supervision loop
            # open a fresh one.  Nothing about the zero-TX gate or the
            # stamp-overlap gate changes; the retry simply gets evidence that
            # can satisfy them.  Bounded by PERCEPTION_ATTEMPTS like every
            # other retryable class.
            return True
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

    def _heal_stale_perception_worker(self, log_path: Path) -> bool:
        """Restart the read-only perception component once to clear a stale worker.

        This is the sole recovery for an rc=70 resident-worker fingerprint
        mismatch.  It restarts perception (read-only; motion stays 0) through
        the existing fixed component-manager target and logs the trigger.  It
        does not loosen any evidence gate: the restarted worker recomputes
        ``runtime_fingerprint()`` from the live checkout, so a genuine mid-edit
        checkout still mismatches on the subsequent retry and fails closed with
        the same legible error.
        """

        if not COMPONENT_MANAGER.is_file():
            with log_path.open("ab") as log:
                log.write(
                    b"resident worker fingerprint mismatch: component manager "
                    b"is unavailable; cannot self-heal perception\n",
                )
            return False
        started = time.monotonic()
        restart = _run_logged(
            (str(COMPONENT_MANAGER), "restart", "perception"),
            log_path,
            environment=_server_environment(),
        )
        _append_timing(
            log_path,
            "perception_fingerprint_selfheal",
            time.monotonic() - started,
            trigger="resident_worker_fingerprint_mismatch",
            action="restart_perception",
            return_code=restart.returncode,
        )
        return restart.returncode == 0

    def run_perception(
        self,
        *,
        target: str,
        output_dir: Path,
        log_path: Path,
    ) -> BackendResult:
        """Run perception, self-healing one stale resident-worker episode.

        A resident-worker fingerprint mismatch surfaces as rc=70.  Instead of
        forcing the operator to hammer retry against a stale worker, restart the
        read-only perception component exactly once and retry the request a
        single time.  The cap guarantees no restart loop, and a persistent
        mismatch still fails closed with its legible rc=70 error.
        """

        result, fingerprint_mismatch = self._run_perception_once(
            target=target,
            output_dir=output_dir,
            log_path=log_path,
        )
        if not fingerprint_mismatch:
            return result
        # rc=70 self-heal cost: measure the restart+retry wall clock and persist
        # a running total on this cold path so the fingerprint-mismatch tax is
        # observable without touching the hot success path.
        selfheal_started = time.monotonic()
        if not self._heal_stale_perception_worker(log_path):
            self._persist_fingerprint_selfheal_cost(
                log_path,
                cost_s=time.monotonic() - selfheal_started,
                healed=False,
                healed_return_code=None,
            )
            return result
        # THE SELF-HEAL IS A RETRY AND MUST START FROM AN EMPTY DIRECTORY.
        # This second ``_run_perception_once`` re-enters the attempt loop at
        # attempt==0, so the loop's own ``if attempt:`` cleanup never fires and
        # every artifact of the rc=70 sub-attempt -- including its already
        # closed passive window -- would be inherited.  That is the recorded
        # ``passive_capture_count: 0`` deadlock; see
        # ``_clear_inherited_attempt_outputs``.
        self._clear_inherited_attempt_outputs(output_dir)
        healed, _mismatch_again = self._run_perception_once(
            target=target,
            output_dir=output_dir,
            log_path=log_path,
        )
        self._persist_fingerprint_selfheal_cost(
            log_path,
            cost_s=time.monotonic() - selfheal_started,
            healed=True,
            healed_return_code=healed.exit_code,
        )
        return healed

    def _persist_fingerprint_selfheal_cost(
        self,
        log_path: Path,
        *,
        cost_s: float,
        healed: bool,
        healed_return_code: int | None,
    ) -> None:
        """Record the running rc=70 self-heal count and cumulative cost."""

        self._fingerprint_selfheal_count = (
            getattr(self, "_fingerprint_selfheal_count", 0) + 1
        )
        self._fingerprint_selfheal_cost_s = round(
            getattr(self, "_fingerprint_selfheal_cost_s", 0.0) + max(0.0, cost_s),
            6,
        )
        _append_timing(
            log_path,
            "perception_fingerprint_selfheal_cost",
            cost_s,
            trigger="resident_worker_fingerprint_mismatch",
            selfheal_count=self._fingerprint_selfheal_count,
            selfheal_cost_total_s=self._fingerprint_selfheal_cost_s,
            healed=healed,
            healed_return_code=healed_return_code,
        )

    def _grounding_observability(self, reused: bool) -> dict[str, object]:
        """Score one completed grounding and return the VLM hit-ratio fields.

        A reused-tracking request is a "hit" that skipped a fresh YOLOE+VLM
        grounding, so the ratio measures how often the VLM was avoided.  Scored
        once per attempt that produced a report; a rc=70 transport failure has
        no report and is never scored here, so a self-heal cannot double-count.
        """

        self._grounding_requests_scored = (
            getattr(self, "_grounding_requests_scored", 0) + 1
        )
        if reused:
            self._grounding_reuse_hits = (
                getattr(self, "_grounding_reuse_hits", 0) + 1
            )
        scored = self._grounding_requests_scored
        hits = getattr(self, "_grounding_reuse_hits", 0)
        return {
            "vlm_avoided_reuse_hits": hits,
            "grounding_requests_scored": scored,
            "vlm_reuse_hit_ratio": round(hits / scored, 6) if scored else 0.0,
        }

    def _run_perception_once(
        self,
        *,
        target: str,
        output_dir: Path,
        log_path: Path,
    ) -> tuple[BackendResult, bool]:
        """Run perception once, capturing synchronized passive joints.

        Returns the backend result together with a flag indicating whether the
        attempt failed on a resident-worker fingerprint mismatch (rc=70), which
        the public ``run_perception`` wrapper uses to self-heal exactly once.
        """

        fingerprint_mismatch = False
        total_started = time.monotonic()
        for path in (NUC_KEY, DDS_CONFIG, PERCEPTION):
            if not path.is_file():
                return BackendResult(
                    1,
                    "SERVER_PREFLIGHT_FAILED",
                    f"required server-owned input is unavailable: {path.name}",
                ), fingerprint_mismatch

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
                self._clear_inherited_attempt_outputs(output_dir)
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

            def perception_worker_running() -> bool:
                if worker_thread is not None:
                    return worker_thread.is_alive()
                return process is not None and process.poll() is None

            # Set immediately before each capture is opened.  A window opened
            # while the request already holds selected zero-TX evidence cannot
            # be the attestation for that evidence, and is the only kind of
            # window that may be cut short.
            window_redundant_at_open = False

            def passive_window_unneeded() -> bool:
                # Fail-closed rule for cutting an observation short.
                #
                # "The worker is gone" must NEVER on its own abandon a window.
                # The supervision loop's only exit is the worker going away, so
                # a probe in flight at that moment always satisfies such a
                # predicate -- which abandoned the final window of every single
                # attempt (measured: 373/373 recorded attempts overshot the
                # worker's own completion, so a window was always in flight).
                #
                # Nor may selection alone abandon it.  The stamp-overlap gate
                # admits a sensor stamp up to 250 ms AFTER observation_end, and
                # on the recorded corpus 92 of 610 sessions (15.1%) select
                # exactly such a stamp.  The only probe that ever observes that
                # trailing region is the one already in flight when the dry run
                # selects, so that window is the request's trailing attestation
                # and must produce a verdict.
                #
                # What is left is genuinely redundant: a window that was already
                # superfluous when it opened.  The guard below normally stops
                # one being opened at all, but it reads a report the dry run
                # writes non-atomically, so an extra window can still slip out
                # just after selection.  That one -- and only that one -- may be
                # cut short.
                return window_redundant_at_open and not perception_worker_running()

            try:
                while perception_worker_running():
                    if self._selected_passive_report_valid(output_dir):
                        time.sleep(PASSIVE_CAPTURE_POLL_SECONDS)
                        continue
                    window_redundant_at_open = (
                        self._selected_passive_report_valid(output_dir)
                    )
                    passive_capture_started = time.monotonic()
                    passive = self._capture_passive_window(
                        output_dir,
                        log_path,
                        environment,
                        stop=passive_window_unneeded,
                    )
                    capture_elapsed = time.monotonic() - passive_capture_started
                    passive_capture_s += capture_elapsed
                    passive_capture_s_total += capture_elapsed
                    passive_capture_count += 1
                    passive_capture_count_total += 1
                    if passive.exit_code != 0:
                        return passive, fingerprint_mismatch
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
                        if worker_error and isinstance(
                            worker_error[0], _ResidentWorkerFingerprintMismatch
                        ):
                            # Recoverable: the wrapper restarts the read-only
                            # perception component once and retries.
                            fingerprint_mismatch = True
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
            reused = report.get("grounding_reused") is True
            timing_fields["grounding_mode"] = (
                "reused_tracking" if reused else "fresh_grounding"
            )
            timing_fields.update(self._grounding_observability(reused))
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
            ), fingerprint_mismatch
        if return_code == 0:
            return BackendResult(0), fingerprint_mismatch
        return (
            self._perception_failure_result(
                output_dir, return_code, log_path=log_path,
            ),
            fingerprint_mismatch,
        )

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
