#!/usr/bin/env python3
"""Stress recorded near-field planning sessions without any robot transport.

The tool clones immutable perception artifacts into a temporary workspace,
then applies deterministic sensor-scale perturbations to target geometry,
planning joints, and grasp candidate order.  Each trial invokes the existing
offline planner in an isolated Docker container with networking disabled.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PLANNER = ROOT / "scripts/runtime/piper_planning_dry_run.py"
CONFIG = ROOT / "configs/go2w_piper.json"
CALIBRATION = (
    WORKSPACE
    / "artifacts/go2w_real/calibration/piper_wrist_camera_calibration.json"
)
ROBOT_ASSETS = WORKSPACE / "go2W_Sim/assets"
CONTAINER_URDF = "/robot_assets/urdf/go2w_sensored.urdf"
HOST_URDF = ROBOT_ASSETS / "urdf/go2w_sensored.urdf"


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        if item.is_file():
            item.chmod(0o600)
        elif item.is_dir():
            item.chmod(0o700)
    shutil.rmtree(path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv(values: np.ndarray) -> str:
    return ",".join(f"{float(value):.12g}" for value in values)


def _perturb_artifacts(
    source: Path,
    destination: Path,
    *,
    rng: np.random.Generator,
    target_sigma_m: float,
) -> tuple[np.ndarray, list[int]]:
    shutil.copytree(source, destination)
    destination.chmod(0o700)
    for item in destination.iterdir():
        if item.is_file():
            item.chmod(0o600)
    archive_path = destination / "grasp_candidates.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    candidate_count = len(data["grasps"])
    permutation = rng.permutation(candidate_count)
    offset = rng.normal(0.0, target_sigma_m, 3)
    grasps = np.asarray(data["grasps"])[permutation].copy()
    grasps[:, :3, 3] += offset
    data["grasps"] = grasps
    data["scores"] = np.asarray(data["scores"])[permutation]
    data["widths"] = np.asarray(data["widths"])[permutation]
    data["centroid"] = np.asarray(data["centroid"]) + offset
    perturbed_archive = archive_path.with_suffix(".perturbed.npz")
    np.savez(perturbed_archive, **data)
    perturbed_archive.replace(archive_path)
    target_points = np.load(destination / "target_points.npy", allow_pickle=False)
    perturbed_points = destination / "target_points.perturbed.npy"
    np.save(perturbed_points, target_points + offset)
    perturbed_points.replace(destination / "target_points.npy")
    return offset, permutation.tolist()


def _planner_command(
    artifacts: Path,
    output: Path,
    measured_joints: np.ndarray,
    planning_joints: np.ndarray,
    *,
    image: str,
) -> list[str]:
    return [
        "docker", "run", "--rm", "--network", "none",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", f"{os.geteuid()}:{os.getegid()}",
        "-e", "HOME=/tmp/z-manip", "-e", "Z_MANIP_IK_BACKEND=pinocchio",
        "-v", f"{artifacts}:/session/perception:ro",
        "-v", f"{output}:/session/planning",
        "-v", f"{CALIBRATION}:/session/calibration.json:ro",
        "-v", f"{ROBOT_ASSETS}:/robot_assets:ro",
        "-v", f"{PLANNER}:/usr/local/bin/z-manip-piper-planning-dry-run:ro",
        "-v", f"{CONFIG}:/opt/z_manip/configs/go2w_piper.json:ro",
        "-v", f"{ROOT / 'z_manip'}:/opt/z_manip/python/z_manip:ro",
        image, "z-manip-piper-planning-dry-run",
        "--artifacts", "/session/perception",
        "--config", "/opt/z_manip/configs/go2w_piper.json",
        "--urdf", CONTAINER_URDF,
        f"--joints={_csv(measured_joints)}",
        f"--planning-joints={_csv(planning_joints)}",
        "--search-timeout-s", "6.0", "--symmetry-samples", "4",
        "--max-hypotheses", "64", "--max-candidates", "64",
        "--max-feasible-plans", "1",
        "--support-approach-prior-weight", "0.05",
        "--scene-clearance-m", "0.001", "--scene-point-radius-m", "0.001",
        "--gripper-scene-radius-scale", "0.6",
        "--camera-calibration", "/session/calibration.json",
        "--output", "/session/planning",
    ]


def _joint_limits(urdf: Path) -> tuple[np.ndarray, np.ndarray]:
    # Import lazily so --help and artifact-only unit tests stay lightweight.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from z_manip.kinematics.chain import KinematicChain

    chain = KinematicChain.from_urdf(
        urdf,
        base_link="piper_base_link",
        tip_link="piper_gripper_base",
    )
    if chain.dof != 6:
        raise ValueError(f"expected six PiPER joints, found {chain.dof}")
    return chain.lower_limits.copy(), chain.upper_limits.copy()


def _bounded_planning_seed(
    planning_start: np.ndarray,
    *,
    rng: np.random.Generator,
    sigma_rad: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(planning_start, dtype=float)
    if values.shape != lower.shape or values.shape != upper.shape:
        raise ValueError("planning start and joint limits must have matching shapes")
    if not np.all(np.isfinite(values)):
        raise ValueError("planning start contains a non-finite value")
    if np.any(values < lower - 1e-9) or np.any(values > upper + 1e-9):
        raise ValueError("recorded planning start violates URDF limits")
    raw_offset = rng.normal(0.0, sigma_rad, values.shape)
    perturbed = np.clip(values + raw_offset, lower, upper)
    return perturbed, perturbed - values


def _distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("latency distribution requires finite samples")
    return {
        "min": float(np.min(data)),
        "p50": float(np.quantile(data, 0.50)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--session", action="append", required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-sigma-mm", type=float, default=3.0)
    parser.add_argument("--joint-sigma-deg", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--runtime-image", default="z-manip-runtime:pinocchio")
    parser.add_argument("--urdf", type=Path, default=HOST_URDF)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.target_sigma_mm < 0.0 or args.joint_sigma_deg < 0.0:
        raise SystemExit("perturbation sigmas must be non-negative")
    required_inputs = (
        args.sessions_root,
        args.gate_root,
        CALIBRATION,
        ROBOT_ASSETS,
        args.urdf,
    )
    for required in required_inputs:
        if not required.exists():
            raise SystemExit(f"required offline input does not exist: {required}")
    for session_id in args.session:
        source = args.sessions_root / session_id / "perception"
        gate = args.gate_root / session_id / "session_gate.json"
        if not source.is_dir() or not gate.is_file():
            raise SystemExit(f"incomplete recorded session: {session_id}")
    lower, upper = _joint_limits(args.urdf)
    output = args.output.expanduser().resolve()
    _remove_tree(output)
    output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    trials: list[dict[str, Any]] = []
    for session_id in args.session:
        source = args.sessions_root / session_id / "perception"
        gate = _json(args.gate_root / session_id / "session_gate.json")
        measured = np.asarray(gate["measured_joints_rad"], dtype=float)
        planning_start = np.asarray(gate["planning_start_joints_rad"], dtype=float)
        for repeat in range(args.repeats):
            trial = output / f"{session_id}-{repeat:02d}"
            artifacts = trial / "perception"
            planning = trial / "planning"
            target_offset, permutation = _perturb_artifacts(
                source,
                artifacts,
                rng=rng,
                target_sigma_m=args.target_sigma_mm / 1000.0,
            )
            planning.mkdir()
            planning_joints, joint_offset = _bounded_planning_seed(
                planning_start,
                rng=rng,
                sigma_rad=float(np.deg2rad(args.joint_sigma_deg)),
                lower=lower,
                upper=upper,
            )
            started = time.perf_counter()
            completed = subprocess.run(
                _planner_command(
                    artifacts,
                    planning,
                    measured,
                    planning_joints,
                    image=args.runtime_image,
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=30.0,
            )
            elapsed = time.perf_counter() - started
            report_path = planning / "planning_report.json"
            report = _json(report_path) if report_path.is_file() else {}
            stages = Counter(
                str(item.get("stage", "unknown"))
                for item in report.get("rejections", [])
                if isinstance(item, dict)
            )
            record = {
                "session_id": session_id,
                "repeat": repeat,
                "return_code": completed.returncode,
                "wall_s": elapsed,
                "plan_valid": report.get("plan_valid") is True,
                "target_offset_m": target_offset.tolist(),
                "joint_offset_rad": joint_offset.tolist(),
                "planning_joints_rad": planning_joints.tolist(),
                "candidate_permutation": permutation,
                "rejection_stages": dict(sorted(stages.items())),
                "errors": report.get("errors", []),
                "timings_s": report.get("timings_s", {}),
            }
            trials.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    rejection_totals: Counter[str] = Counter()
    for trial in trials:
        rejection_totals.update(trial["rejection_stages"])
    summary = {
        "schema": "z_mobile_manip.ik_monte_carlo_replay.v1",
        "seed": args.seed,
        "target_sigma_mm": args.target_sigma_mm,
        "joint_sigma_deg": args.joint_sigma_deg,
        "trial_count": len(trials),
        "success_count": sum(trial["plan_valid"] for trial in trials),
        "latency_s": {
            "wall": _distribution([trial["wall_s"] for trial in trials]),
            "planner_search": _distribution(
                [trial["timings_s"]["search"] for trial in trials],
            ),
        },
        "rejection_totals": dict(sorted(rejection_totals.items())),
        "trials": trials,
    }
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
