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
import json
import math
import os
import pwd
from pathlib import Path
import re
import stat
import subprocess
import sys
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


RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "go2w_real" / "interactive_sessions"
PERCEPTION = SCRIPT_DIR / "go2w_perception_dry_run.py"
SESSION_GATE = SCRIPT_DIR / "piper_planning_session_gate.py"
PLANNER = SCRIPT_DIR / "piper_planning_dry_run.py"
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
MAX_PLANNING_REPORT_BYTES = 4 * 1024 * 1024
MAX_PLANNER_ERROR_CHARS = 600
MAX_PERCEPTION_REPORT_BYTES = 256 * 1024
MAX_PERCEPTION_ERROR_CHARS = 600
MAX_REJECTIONS_TO_SUMMARIZE = 4096
SEARCH_TIMEOUT_S = "6"
SYMMETRY_SAMPLES = "4"
MAX_HYPOTHESES = "64"
MAX_FEASIBLE_PLANS = "1"
SUPPORT_APPROACH_PRIOR_WEIGHT = "0.5"
SUPERVISED_SCENE_CLEARANCE_M = "0.001"
SUPERVISED_SCENE_POINT_RADIUS_M = "0.001"
SUPERVISED_GRIPPER_SCENE_RADIUS_SCALE = "0.60"


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

    def _capture_passive_window(
        self,
        output_dir: Path,
        log_path: Path,
        environment: dict[str, str],
    ) -> BackendResult:
        # The probe prints its full JSON report on every 250 ms sample. The
        # immutable report is fetched below, so duplicating it into the action
        # log only creates megabytes of noise and makes UI inspection appear
        # stuck. Preserve stderr for actionable SSH/probe failures.
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
        with log_path.open("ab") as log:
            passive = subprocess.run(
                passive_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                env=environment,
                shell=False,
                check=False,
            )
        if passive.returncode != 0:
            return BackendResult(
                passive.returncode,
                "PASSIVE_JOINT_GATE_FAILED",
                "fixed receive-only passive joint gate failed",
            )

        live_report = output_dir / "live_passive_joint_report.json"
        temporary_report = output_dir / ".passive_joint_report.inflight"
        temporary_report.unlink(missing_ok=True)
        with temporary_report.open("xb") as report_output, log_path.open("ab") as log:
            fetched = subprocess.run(
                self._ssh_prefix() + ("cat", REMOTE_PASSIVE_REPORT),
                stdin=subprocess.DEVNULL,
                stdout=report_output,
                stderr=log,
                env=environment,
                shell=False,
                check=False,
            )
        if fetched.returncode != 0:
            temporary_report.unlink(missing_ok=True)
            return BackendResult(
                fetched.returncode,
                "PASSIVE_JOINT_REPORT_UNAVAILABLE",
                "passive joint report could not be retrieved",
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
    def _perception_outputs_valid(output_dir: Path) -> bool:
        required = (
            output_dir / "report.json",
            output_dir / "edgetam_mask.png",
            output_dir / "edgetam_overlay.png",
            output_dir / "grasp_candidates_overlay.png",
            output_dir / "selected_passive_joint_report.json",
        )
        return bool(
            all(path.is_file() and not path.is_symlink() for path in required)
            and required[-1].stat().st_size <= MAX_PASSIVE_REPORT_BYTES
            and FixedReadOnlyBackend._passive_report_valid(required[-1])
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

    def run_perception(
        self,
        *,
        target: str,
        output_dir: Path,
        log_path: Path,
    ) -> BackendResult:
        """Run perception while repeatedly capturing synchronized passive joints."""

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
        command = (
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
            "z-manip-go2w-perception-dry-run",
            "--instruction",
            target,
            "--output",
            "/artifacts",
            "--passive-window",
            "/artifacts/live_passive_joint_report.json",
            "--selected-passive-window",
            "/artifacts/selected_passive_joint_report.json",
            "--timeout",
            "15",
            "--min-bundle-target-points",
            "400",
        )
        total_started = time.monotonic()
        return_code = 1
        for attempt in range(PERCEPTION_ATTEMPTS):
            if attempt:
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
                        b"Retrying perception after an invalid geometric mask.\n",
                    )
            with log_path.open("ab") as log:
                attempt_started = time.monotonic()
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    shell=False,
                )
            try:
                while process.poll() is None:
                    passive = self._capture_passive_window(
                        output_dir,
                        log_path,
                        environment,
                    )
                    if passive.exit_code != 0:
                        return passive
                return_code = process.wait()
            finally:
                self._stop_process(process)
            _append_timing(
                log_path,
                "perception_attempt",
                time.monotonic() - attempt_started,
                attempt=attempt + 1,
                return_code=return_code,
            )
            if not self._perception_retryable(output_dir, return_code):
                break
        outputs_valid = self._perception_outputs_valid(output_dir)
        if return_code == 0 and not outputs_valid:
            return BackendResult(
                1,
                "PERCEPTION_OUTPUT_INVALID",
                "perception omitted synchronized joints or fixed UI overlays",
            )
        _append_timing(
            log_path,
            "perception_total",
            time.monotonic() - total_started,
            attempts=attempt + 1,
            return_code=return_code,
        )
        if return_code == 0:
            return BackendResult(0)
        return self._perception_failure_result(output_dir, return_code)

    @staticmethod
    def _required_planning_files() -> tuple[Path, ...]:
        return (
            SESSION_GATE,
            PLANNER,
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
        planner_started = time.monotonic()
        planner = _run_logged(
            (
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
                f"{STACK_CONFIG}:/opt/z_manip/configs/go2w_piper.json:ro",
                "-v",
                f"{STACK_ROOT / 'z_manip'}:/opt/z_manip/python/z_manip:ro",
                self.runtime.runtime_image,
                "z-manip-piper-planning-dry-run",
                "--artifacts",
                "/session/perception",
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
                "/session/calibration.json",
                "--output",
                "/session/planning",
            ),
            log_path,
            environment=_server_environment(),
        )
        _append_timing(
            log_path,
            "planning_search",
            time.monotonic() - planner_started,
            return_code=planner.returncode,
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
            None if planner.returncode == 0 else "OFFLINE_PLANNER_BLOCKED",
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
