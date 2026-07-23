#!/usr/bin/env python3
"""Replay recorded perception sessions through the offline grasp planner.

The rosbag is used only as an immutable time-window manifest.  Every selected
perception session is revalidated by the filesystem-only session gate, then
planned sequentially in a Docker container with ``--network none`` and no
device mounts.  This program never imports ROS or robot drivers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np


SCHEMA = "z_mobile_manip.offline_planning_replay.v1"
ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
SESSION_GATE = ROOT / "scripts/runtime/piper_planning_session_gate.py"
PLANNER = ROOT / "scripts/runtime/piper_planning_dry_run.py"
CONFIG = ROOT / "configs/go2w_piper.json"
CALIBRATION = (
    WORKSPACE / "artifacts/go2w_real/calibration/piper_wrist_camera_calibration.json"
)
URDF = WORKSPACE / "go2W_Sim/assets/urdf/go2w_sensored.urdf"
ROBOT_ASSETS = URDF.parent.parent
CONTAINER_URDF = f"/robot_assets/urdf/{URDF.name}"
SAFE_IMAGE = re.compile(r"z-manip-runtime:[a-z0-9][a-z0-9._-]{0,63}\Z")
REQUIRED_PERCEPTION_FILES = (
    "grasp_candidates.npz",
    "report.json",
    "scene_collision_points.npy",
    "selected_passive_joint_report.json",
    "target_points.npy",
)
Runner = Callable[[Sequence[str], float, Path], tuple[int, float, bool]]
DEFAULT_HANDOFF_REACH_M = 0.70


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _stamp(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1e9)


def bag_window(bag: Path) -> dict[str, Any]:
    metadata = bag / "metadata.yaml" if bag.is_dir() else bag.parent / "metadata.yaml"
    text = metadata.read_text(encoding="utf-8")
    start = re.search(r"(?m)^\s*nanoseconds_since_epoch:\s*(\d+)\s*$", text)
    duration = re.search(
        r"(?ms)^\s*duration:\s*\n\s*nanoseconds:\s*(\d+)\s*$", text,
    )
    count = re.search(r"(?m)^\s*message_count:\s*(\d+)\s*$", text)
    if start is None or duration is None:
        raise ValueError(f"rosbag metadata has no bounded time window: {metadata}")
    start_ns = int(start.group(1))
    duration_ns = int(duration.group(1))
    return {
        "path": str(bag.resolve()),
        "start_unix_ns": start_ns,
        "end_unix_ns": start_ns + duration_ns,
        "duration_s": duration_ns / 1e9,
        "message_count": int(count.group(1)) if count else None,
    }


def discover_sessions(
    sessions_root: Path,
    *,
    start_ns: int,
    end_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for attempt_path in sorted((sessions_root / "perception").glob("*/attempt.json")):
        attempt = _load_json(attempt_path)
        if attempt is None:
            continue
        stamp_ns = _stamp(attempt.get("started_at"))
        if stamp_ns is None or not start_ns <= stamp_ns <= end_ns:
            continue
        session_id = str(attempt.get("session_id") or attempt_path.parent.name)
        artifacts = attempt_path.parent / "perception"
        missing = [name for name in REQUIRED_PERCEPTION_FILES if not (artifacts / name).is_file()]
        reason = None
        if attempt.get("status") != "succeeded":
            reason = f"perception status is {attempt.get('status', 'unknown')}"
        elif missing:
            reason = "missing artifacts: " + ", ".join(missing)
        item = {
            "session_id": session_id,
            "started_unix_ns": stamp_ns,
            "attempt_path": str(attempt_path.resolve()),
            "artifacts": str(artifacts.resolve()),
        }
        if reason is None:
            selected.append(item)
        else:
            skipped.append({**item, "reason": reason})
    return selected, skipped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(values: object, label: str) -> str:
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError(f"{label} is not a six-joint vector")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} has a non-numeric value")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} has a non-finite value")
        result.append(f"{number:.12g}")
    return ",".join(result)


def gate_command(artifacts: Path, output: Path, *, calibration: Path, urdf: Path) -> list[str]:
    return [
        sys.executable,
        str(SESSION_GATE),
        "--perception-dir", str(artifacts),
        "--joint-report", str(artifacts / "selected_passive_joint_report.json"),
        "--calibration", str(calibration),
        "--urdf", str(urdf),
        "--output", str(output),
    ]


def planner_command(
    artifacts: Path,
    output: Path,
    gate: dict[str, Any],
    *,
    image: str,
    config: Path,
    calibration: Path,
    robot_assets: Path,
    search_timeout_s: float,
    symmetry_samples: int,
    max_hypotheses: int,
    max_feasible_plans: int,
    max_candidates: int,
    support_approach_prior_weight: float,
    scene_clearance_m: float,
    scene_point_radius_m: float,
    gripper_scene_radius_scale: float,
) -> list[str]:
    if SAFE_IMAGE.fullmatch(image) is None:
        raise ValueError("runtime image must be a local z-manip-runtime tag")
    measured = _csv(gate.get("measured_joints_rad"), "measured_joints_rad")
    planning = _csv(gate.get("planning_start_joints_rad"), "planning_start_joints_rad")
    return [
        "/usr/bin/docker", "run", "--rm",
        "--network", "none",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", f"{os.geteuid()}:{os.getegid()}",
        "-e", "HOME=/tmp/z-manip",
        "-e", "Z_MANIP_IK_BACKEND=pinocchio",
        "-v", f"{artifacts}:/session/perception:ro",
        "-v", f"{output}:/session/planning",
        "-v", f"{calibration}:/session/calibration.json:ro",
        "-v", f"{robot_assets}:/robot_assets:ro",
        "-v", f"{PLANNER}:/usr/local/bin/z-manip-piper-planning-dry-run:ro",
        "-v", f"{config}:/opt/z_manip/configs/go2w_piper.json:ro",
        "-v", f"{ROOT / 'z_manip'}:/opt/z_manip/python/z_manip:ro",
        image,
        "z-manip-piper-planning-dry-run",
        "--artifacts", "/session/perception",
        "--config", "/opt/z_manip/configs/go2w_piper.json",
        "--urdf", CONTAINER_URDF,
        f"--joints={measured}",
        f"--planning-joints={planning}",
        "--search-timeout-s", str(search_timeout_s),
        "--symmetry-samples", str(symmetry_samples),
        "--max-hypotheses", str(max_hypotheses),
        "--max-candidates", str(max_candidates),
        "--max-feasible-plans", str(max_feasible_plans),
        "--support-approach-prior-weight", str(support_approach_prior_weight),
        "--scene-clearance-m", str(scene_clearance_m),
        "--scene-point-radius-m", str(scene_point_radius_m),
        "--gripper-scene-radius-scale", str(gripper_scene_radius_scale),
        "--camera-calibration", "/session/calibration.json",
        "--output", "/session/planning",
    ]


def subprocess_runner(argv: Sequence[str], timeout_s: float, log_path: Path) -> tuple[int, float, bool]:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("wb") as log:
            completed = subprocess.run(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    "HOME": str(Path.home()),
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "PYTHONPATH": str(ROOT),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                },
                shell=False,
                check=False,
                timeout=timeout_s,
            )
        return completed.returncode, time.perf_counter() - started, False
    except subprocess.TimeoutExpired:
        return 124, time.perf_counter() - started, True


def _stage_counts(report: dict[str, Any] | None) -> dict[str, int]:
    counts = Counter(
        str(item.get("stage", "unknown"))
        for item in ((report or {}).get("rejections") or [])
        if isinstance(item, dict)
    )
    return dict(sorted(counts.items()))


def target_base_range_m(
    artifacts: Path,
    report: dict[str, Any] | None,
) -> float | None:
    """Return robust target range in the measured arm-base frame."""
    matrix = (report or {}).get("base_from_camera")
    try:
        transform = np.asarray(matrix, dtype=np.float64)
        points = np.asarray(
            np.load(artifacts / "target_points.npy", allow_pickle=False),
            dtype=np.float64,
        )
    except (OSError, ValueError, TypeError):
        return None
    if transform.shape != (4, 4) or points.ndim != 2 or points.shape[1] != 3:
        return None
    if points.shape[0] == 0 or not np.isfinite(transform).all() or not np.isfinite(points).all():
        return None
    base_points = points @ transform[:3, :3].T + transform[:3, 3]
    value = float(np.linalg.norm(np.median(base_points, axis=0)))
    return round(value, 6) if math.isfinite(value) else None


def run_trial(
    session: dict[str, Any],
    output_root: Path,
    options: argparse.Namespace,
    *,
    runner: Runner,
) -> dict[str, Any]:
    session_id = str(session["session_id"])
    trial_root = output_root / session_id
    planning_dir = trial_root / "planning"
    trial_root.mkdir(parents=True, exist_ok=True)
    planning_dir.mkdir()
    artifacts = Path(session["artifacts"])
    gate_path = trial_root / "session_gate.json"
    total_started = time.perf_counter()
    gate_rc, gate_wall_s, gate_timeout = runner(
        gate_command(artifacts, gate_path, calibration=options.calibration, urdf=options.urdf),
        options.trial_timeout_s,
        trial_root / "gate.log",
    )
    gate = _load_json(gate_path)
    if gate_timeout or gate_rc != 0 or not gate or gate.get("planning_ready") is not True:
        target_range_m = None
        if isinstance(gate, dict):
            handoff_workspace = gate.get("handoff_workspace")
            if isinstance(handoff_workspace, dict):
                value = handoff_workspace.get("target_range_m")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    target_range_m = round(float(value), 6)
            if target_range_m is None:
                target_range_m = target_base_range_m(artifacts, gate)
        handoff_eligible = (
            target_range_m <= options.handoff_reach_m
            if target_range_m is not None
            else None
        )
        gate_errors = (gate or {}).get("errors", [])
        error_codes = {
            str(item.get("code"))
            for item in gate_errors
            if isinstance(item, dict)
        }
        status = (
            "needs_base_approach"
            if "NEED_BASE_APPROACH" in error_codes
            else "gate_blocked"
        )
        return {
            **session,
            "status": status,
            "gate_return_code": gate_rc,
            "gate_timeout": gate_timeout,
            "gate_wall_s": round(gate_wall_s, 6),
            "total_wall_s": round(time.perf_counter() - total_started, 6),
            "target_base_range_m": target_range_m,
            "handoff_eligible": handoff_eligible,
            "gate_errors": gate_errors,
            "rejection_stages": {},
        }
    command = planner_command(
        artifacts,
        planning_dir,
        gate,
        image=options.runtime_image,
        config=options.config,
        calibration=options.calibration,
        robot_assets=options.robot_assets,
        search_timeout_s=options.search_timeout_s,
        symmetry_samples=options.symmetry_samples,
        max_hypotheses=options.max_hypotheses,
        max_feasible_plans=options.max_feasible_plans,
        max_candidates=options.max_candidates,
        support_approach_prior_weight=options.support_approach_prior_weight,
        scene_clearance_m=options.scene_clearance_m,
        scene_point_radius_m=options.scene_point_radius_m,
        gripper_scene_radius_scale=options.gripper_scene_radius_scale,
    )
    planner_rc, planner_wall_s, planner_timeout = runner(
        command,
        options.trial_timeout_s,
        trial_root / "planner.log",
    )
    report = _load_json(planning_dir / "planning_report.json")
    plan_valid = bool(report and report.get("plan_valid") is True and planner_rc == 0)
    target_range_m = target_base_range_m(artifacts, report)
    handoff_eligible = (
        target_range_m <= options.handoff_reach_m
        if target_range_m is not None
        else None
    )
    return {
        **session,
        "status": "succeeded" if plan_valid else ("timeout" if planner_timeout else "failed"),
        "gate_return_code": gate_rc,
        "gate_wall_s": round(gate_wall_s, 6),
        "planner_return_code": planner_rc,
        "planner_timeout": planner_timeout,
        "planner_wall_s": round(planner_wall_s, 6),
        "total_wall_s": round(time.perf_counter() - total_started, 6),
        "plan_valid": plan_valid,
        "candidate_count": gate.get("candidate_count"),
        "target_base_range_m": target_range_m,
        "handoff_eligible": handoff_eligible,
        "rejection_count": (report or {}).get("rejection_count", 0),
        "rejection_stages": _stage_counts(report),
        "planner_timings_s": (report or {}).get("timings_s", {}),
        "error": (report or {}).get("error", "planner report unavailable"),
        "report_path": str((planning_dir / "planning_report.json").resolve()),
    }


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 6)


def _latency(values: Iterable[float]) -> dict[str, float | None]:
    cached = list(values)
    return {
        "min": _percentile(cached, 0.0),
        "p50": _percentile(cached, 0.50),
        "p95": _percentile(cached, 0.95),
        "max": _percentile(cached, 1.0),
    }


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    success = sum(item.get("status") == "succeeded" for item in trials)
    rejection_stages: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for item in trials:
        rejection_stages.update(item.get("rejection_stages") or {})
        statuses[str(item.get("status", "unknown"))] += 1
    rejection_total = sum(rejection_stages.values())
    ik_total = rejection_stages.get("ik", 0)
    result = {
        "trials": len(trials),
        "succeeded": success,
        "success_rate": round(success / len(trials), 6) if trials else None,
        "status_counts": dict(sorted(statuses.items())),
        "planner_wall_s": _latency(
            float(item["planner_wall_s"])
            for item in trials
            if isinstance(item.get("planner_wall_s"), (int, float))
        ),
        "total_wall_s": _latency(float(item["total_wall_s"]) for item in trials),
        "planner_search_s": _latency(
            float(item["planner_timings_s"]["search"])
            for item in trials
            if isinstance(item.get("planner_timings_s"), dict)
            and isinstance(item["planner_timings_s"].get("search"), (int, float))
        ),
        "rejection_stages": dict(sorted(rejection_stages.items())),
        "ik_rejection_fraction": (
            round(ik_total / rejection_total, 6) if rejection_total else 0.0
        ),
    }
    eligible = [item for item in trials if item.get("handoff_eligible") is True]
    far = [item for item in trials if item.get("handoff_eligible") is False]
    close_success = sum(item.get("status") == "succeeded" for item in eligible)
    result["handoff"] = {
        "eligible_trials": len(eligible),
        "needs_base_approach_trials": len(far),
        "unknown_range_trials": len(trials) - len(eligible) - len(far),
        "succeeded": close_success,
        "success_rate": (
            round(close_success / len(eligible), 6) if eligible else None
        ),
        "planner_wall_s": _latency(
            float(item["planner_wall_s"])
            for item in eligible
            if isinstance(item.get("planner_wall_s"), (int, float))
        ),
        "planner_search_s": _latency(
            float(item["planner_timings_s"]["search"])
            for item in eligible
            if isinstance(item.get("planner_timings_s"), dict)
            and isinstance(item["planner_timings_s"].get("search"), (int, float))
        ),
    }
    return result


def evaluate_thresholds(
    summary: dict[str, Any],
    *,
    min_success_rate: float | None,
    max_p95_s: float | None,
    max_ik_rejection_fraction: float | None,
    baseline: dict[str, Any] | None,
    max_success_rate_drop: float,
    max_p95_regression_ratio: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: float | None, limit: float, relation: str) -> None:
        passed = bool(
            observed is not None
            and (observed >= limit if relation == ">=" else observed <= limit)
        )
        checks.append({
            "name": name, "observed": observed, "relation": relation,
            "limit": round(limit, 6), "passed": passed,
        })

    if min_success_rate is not None:
        check("minimum_success_rate", summary.get("success_rate"), min_success_rate, ">=")
    if max_p95_s is not None:
        check("maximum_planner_p95_s", summary["planner_wall_s"].get("p95"), max_p95_s, "<=")
    if max_ik_rejection_fraction is not None:
        check(
            "maximum_ik_rejection_fraction",
            summary.get("ik_rejection_fraction"),
            max_ik_rejection_fraction,
            "<=",
        )
    baseline_summary = (baseline or {}).get("summary")
    if isinstance(baseline_summary, dict):
        old_success = baseline_summary.get("success_rate")
        if isinstance(old_success, (int, float)):
            check(
                "baseline_success_rate_drop",
                summary.get("success_rate"),
                float(old_success) - max_success_rate_drop,
                ">=",
            )
        old_p95 = (baseline_summary.get("planner_wall_s") or {}).get("p95")
        if isinstance(old_p95, (int, float)):
            check(
                "baseline_p95_regression_ratio",
                summary["planner_wall_s"].get("p95"),
                float(old_p95) * max_p95_regression_ratio,
                "<=",
            )
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def build_report(
    *,
    bag: Path,
    sessions_root: Path,
    output_root: Path,
    options: argparse.Namespace,
    runner: Runner = subprocess_runner,
) -> dict[str, Any]:
    window = bag_window(bag)
    sessions, skipped = discover_sessions(
        sessions_root,
        start_ns=window["start_unix_ns"],
        end_ns=window["end_unix_ns"],
    )
    if options.max_trials is not None:
        sessions = sessions[: options.max_trials]
    trials = [run_trial(item, output_root, options, runner=runner) for item in sessions]
    summary = summarize_trials(trials)
    baseline = _load_json(options.baseline) if options.baseline else None
    thresholds = evaluate_thresholds(
        summary,
        min_success_rate=options.min_success_rate,
        max_p95_s=options.max_p95_s,
        max_ik_rejection_fraction=options.max_ik_rejection_fraction,
        baseline=baseline,
        max_success_rate_drop=options.max_success_rate_drop,
        max_p95_regression_ratio=options.max_p95_regression_ratio,
    )
    return {
        "schema": SCHEMA,
        "offline": True,
        "transport_opened": False,
        "motion_commands_sent": 0,
        "runner": "docker_network_none",
        "bag": window,
        "profile": {
            "runtime_image": options.runtime_image,
            "ik_backend": "pinocchio",
            "search_timeout_s": options.search_timeout_s,
            "symmetry_samples": options.symmetry_samples,
            "max_hypotheses": options.max_hypotheses,
            "max_feasible_plans": options.max_feasible_plans,
            "support_approach_prior_weight": options.support_approach_prior_weight,
            "scene_clearance_m": options.scene_clearance_m,
            "scene_point_radius_m": options.scene_point_radius_m,
            "gripper_scene_radius_scale": options.gripper_scene_radius_scale,
            "handoff_reach_m": options.handoff_reach_m,
            "config_sha256": _sha256(options.config),
            "urdf_sha256": _sha256(options.urdf),
            "planner_sha256": _sha256(PLANNER),
        },
        "summary": summary,
        "thresholds": thresholds,
        "selected_sessions": [item["session_id"] for item in sessions],
        "skipped_sessions": skipped,
        "trials": trials,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    handoff = summary["handoff"]
    lines = [
        "# Offline planning replay", "",
        f"- Sessions: {summary['succeeded']}/{summary['trials']} succeeded ({summary['success_rate']})",
        f"- Close-range handoff: {handoff['succeeded']}/{handoff['eligible_trials']} succeeded ({handoff['success_rate']})",
        f"- Needs more base approach: {handoff['needs_base_approach_trials']}",
        f"- Close-range planner wall time: p50 {handoff['planner_wall_s']['p50']} s, p95 {handoff['planner_wall_s']['p95']} s",
        f"- Planner wall time: p50 {summary['planner_wall_s']['p50']} s, p95 {summary['planner_wall_s']['p95']} s",
        f"- Planner search: p50 {summary['planner_search_s']['p50']} s, p95 {summary['planner_search_s']['p95']} s",
        f"- Rejections: {summary['rejection_stages']}",
        f"- IK rejection fraction: {summary['ik_rejection_fraction']}",
        f"- Regression thresholds: {'PASS' if report['thresholds']['passed'] else 'FAIL'}",
        "", "## Safety", "",
        "- Docker network: none",
        "- ROS/WebRTC/CAN transports opened: no",
        "- Motion commands sent: 0", "",
    ]
    return "\n".join(lines)


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be within [0, 1]")
    return number


def _gripper_scene_radius_scale(value: str) -> float:
    number = _fraction(value)
    if number < 0.5:
        raise argparse.ArgumentTypeError("value must be within [0.5, 1]")
    return number


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--runtime-image", default="z-manip-runtime:pinocchio")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--urdf", type=Path, default=URDF)
    parser.add_argument("--robot-assets", type=Path, default=ROBOT_ASSETS)
    parser.add_argument("--trial-timeout-s", type=_positive, default=30.0)
    parser.add_argument("--search-timeout-s", type=_positive, default=6.0)
    parser.add_argument("--symmetry-samples", type=int, default=4)
    parser.add_argument("--max-hypotheses", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-feasible-plans", type=int, default=1)
    parser.add_argument("--support-approach-prior-weight", type=float, default=0.05)
    parser.add_argument("--scene-clearance-m", type=_positive, default=0.001)
    parser.add_argument("--scene-point-radius-m", type=_positive, default=0.001)
    parser.add_argument(
        "--gripper-scene-radius-scale",
        type=_gripper_scene_radius_scale,
        default=0.60,
    )
    parser.add_argument(
        "--handoff-reach-m",
        type=_positive,
        default=DEFAULT_HANDOFF_REACH_M,
        help="maximum target range in piper_base_link counted as a handoff trial",
    )
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--min-success-rate", type=_fraction)
    parser.add_argument("--max-p95-s", type=_positive)
    parser.add_argument("--max-ik-rejection-fraction", type=_fraction)
    parser.add_argument("--max-success-rate-drop", type=_fraction, default=0.0)
    parser.add_argument("--max-p95-regression-ratio", type=_positive, default=1.10)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if SAFE_IMAGE.fullmatch(args.runtime_image) is None:
        parser.error("--runtime-image must be a local z-manip-runtime tag")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be positive")
    for path in (args.config, args.calibration, args.urdf, args.robot_assets):
        if not path.exists():
            parser.error(f"required input is missing: {path}")
    if args.output_root.exists():
        if not args.overwrite:
            parser.error("--output-root exists; use --overwrite to replace it")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    return args


def main() -> int:
    args = parse_args()
    report = build_report(
        bag=args.bag,
        sessions_root=args.sessions_root,
        output_root=args.output_root,
        options=args,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0 if not args.strict or report["thresholds"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
