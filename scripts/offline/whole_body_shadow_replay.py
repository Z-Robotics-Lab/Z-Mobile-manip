#!/usr/bin/env python3
"""Replay recorded Go2W target/joint evidence through the whole-body QP.

This executable has no transport code.  It never imports ROS, WebRTC, CAN, or
the robot SDK, and its report always records zero transmitted commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from z_manip.control.whole_body_model import (
    PinocchioReducedWholeBodyModel,
    PinocchioWholeBodyFrames,
    ReducedWholeBodyState,
)
from z_manip.control.whole_body_optimizer import (
    CasadiBoxQP,
    ScipyReferenceBoxQP,
    WholeBodyShadowOptimizer,
    WholeBodyTask,
)


def _document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _last_tracking_target(path: Path) -> tuple[float, float, float]:
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        target = item.get("target")
        if (
            item.get("tracking")
            and isinstance(target, dict)
            and all(key in target for key in ("x_m", "y_m", "z_m"))
        ):
            last = tuple(float(target[key]) for key in ("x_m", "y_m", "z_m"))
    if last is None:
        raise ValueError(f"trace contains no tracked 3-D target: {path}")
    return last


def replay(args: argparse.Namespace) -> dict[str, Any]:
    calibration = _document(args.calibration)
    if calibration.get("synthetic") is not False or calibration.get("calibrated") is not True:
        raise ValueError("shadow replay requires measured, accepted hand-eye calibration")
    tip_from_camera = np.asarray(calibration.get("tip_from_camera"), dtype=float)
    if tip_from_camera.shape != (4, 4):
        raise ValueError("calibration lacks a 4x4 tip_from_camera transform")
    passive = _document(args.passive_joints)
    joints = passive.get("joint_positions_rad")
    if not isinstance(joints, list) or len(joints) != 6:
        raise ValueError("passive joint report lacks six joint_positions_rad")

    frames = PinocchioWholeBodyFrames(
        camera_frame=str(calibration.get("camera_frame", "camera_color_optical_frame")),
        tool_frame=str(calibration.get("tip_link", "piper_gripper_base")),
    )
    model = PinocchioReducedWholeBodyModel(
        args.urdf,
        frames,
        tool_from_camera_optical=tip_from_camera,
    )
    state = ReducedWholeBodyState(
        base_x_m=0.0,
        base_y_m=0.0,
        base_yaw_rad=0.0,
        body_height_m=args.body_height_m,
        body_roll_rad=0.0,
        body_pitch_rad=0.0,
        arm_joints_rad=tuple(float(value) for value in joints),
    )
    camera_target = np.append(_last_tracking_target(args.trace), 1.0)
    target_world = model.frame_pose(state, model.camera_frame) @ camera_target
    solver = CasadiBoxQP() if args.backend == "casadi" else ScipyReferenceBoxQP()
    optimizer = WholeBodyShadowOptimizer(model, solver=solver)
    task = WholeBodyTask(
        target_world_xyz_m=tuple(float(value) for value in target_world[:3]),
        desired_planar_standoff_m=args.standoff_m,
    )

    ticks = []
    previous = None
    for index in range(args.ticks):
        result = optimizer.solve(state, task, previous_velocity=previous)
        ticks.append(
            {
                "tick": index,
                "planar_distance_m": result.planar_distance_m,
                "image_error": float(np.linalg.norm(result.residual_before[:2])),
                "objective_before": result.objective_before,
                "objective_after": result.objective_after,
                "intent": result.status_document()["intent"],
            },
        )
        state = result.predicted_state
        previous = result.velocity

    return {
        "schema": "z_manip.whole_body_shadow_replay.v1",
        "mode": "shadow",
        "transport_opened": False,
        "motion_commands_sent": 0,
        "backend": solver.name,
        "sources": {
            "urdf": str(args.urdf.resolve()),
            "calibration": str(args.calibration.resolve()),
            "passive_joints": str(args.passive_joints.resolve()),
            "trace": str(args.trace.resolve()),
        },
        "ticks": ticks,
        "final_state": state.as_vector().tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--passive-joints", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--standoff-m", type=float, default=0.52)
    parser.add_argument("--body-height-m", type=float, default=0.0)
    parser.add_argument("--backend", choices=("casadi", "scipy"), default="casadi")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = replay(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
