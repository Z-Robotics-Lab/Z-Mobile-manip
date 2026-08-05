# Z-Mobile-Manip

Supervised mobile manipulation for a Unitree Go2-W EDU quadruped carrying an AgileX PiPER
6-DoF arm and a wrist-mounted Intel RealSense D435.

![Operator workbench](docs/images/workbench.png)

## What this is

The **near-field half** of a three-repository stack: navigation drives the robot to a coarse
standoff, and this system takes over from about a metre out and runs through to a completed
grasp and return. An RTX 4090 workstation does perception, grasp synthesis, IK, planning and
the operator UI; the robot's onboard NUC owns the D435, PiPER `can0`, joint feedback and the
Go2-W WebRTC base transport, joined over `ROS_DOMAIN_ID=20`.

Open-vocabulary text is grounded with **YOLOE**, tracked with **EdgeTAM**, and back-projected
against **Fast-FoundationStereo** depth rather than the D435's raw stream. Grasps come from
**AnyGrasp** with a local antipodal generator as fallback, are solved by a damped Gauss-Newton
**Pinocchio** IK, and are checked against a capsule model of the platform's own fixtures
(`configs/piper_collision_capsules.json` — the only enforced representation of the chassis, NUC
and Mid-360 LiDAR) plus an observed point cloud. Paths come from **RRT-Connect** with quintic
time parameterisation; **CasADi** refines them and solves the 10-DoF whole-body QP during the
servo approach. Motion runs in short-lived, content-addressed executors copied to the NUC per
run, so the arm is never driven by a long-lived process.

```text
target text → grounding → tracking → target point cloud
→ depth-closed-loop approach → stop at ~0.50 m → near-field re-perception
→ grasp candidates → IK → collision check and planning
→ pregrasp → approach → slow close → lift → Home
```

Two chains run on real hardware: **fixed base** (`perception → planning → grasp → Home`) and
**mobile** (`find/track → depth approach → stop → close-range grasp`).

![Pipeline](docs/images/pipeline.png)

## Before you change anything

- **Editing this checkout is a live deploy.** `z_manip/**` and `scripts/runtime/**` are
  bind-mounted read-only into the running robot and execute from here. `ros2/**` is baked into
  `z-manip-runtime:pinocchio` and rebuilt with `scripts/runtime/go2w_perception_lab.sh build` —
  not `docker compose`, which targets a stale `:jazzy` tag.
- **Fail-closed throughout.** A missing calibration, a stale transform or an unproven collision
  corridor stops the robot with a named reason in the status document.
- **Threshold comments are normative.** Each records the incident that motivated it; several are
  not derivable from the code, and more than one has been silently reverted by someone who
  assumed otherwise. Their measurements are in [`docs/evidence/`](docs/evidence/).
- **A green test suite does not mean IK or the QP ran.** `pinocchio` and `casadi` are absent
  from some hosts and those tests skip rather than fail.
- **Posture control is unavailable.** The Go2-W `wheeled_sport` interface exposes no `Euler` or
  `BodyHeight`, so the QP's body roll/pitch DOFs are locked to zero on this platform.
- **Every process boundary is a versioned JSON document** — 79 `z_manip.*.vN` schemas. Open
  `depth_servo_trace.v1` first when something misbehaves. Execution leaves stage receipts that
  the offline tests replay, so regressions are caught against recorded runs, not simulation.

This is a supervised laboratory system, not an unattended product. Keep clearance around the
robot and the physical e-stop in reach during any motion test.

![Go2-W with PiPER arm](docs/images/robot.webp)

## Requirements

- Ubuntu 24.04, ROS 2 Jazzy, CycloneDDS, Docker with the NVIDIA Container Toolkit
- RTX 4090 (24 GB) workstation; Go2-W EDU with SSH-reachable onboard NUC
- AgileX PiPER on 1 Mbps SocketCAN with a matching URDF; wrist RealSense D435/D435i
- Time-synced PC and NUC on the same segment, identical ROS Domain ID

Base transport uses the third-party `unitree_webrtc_connect`; the PiPER backend uses
`pyAgxArm`. AnyGrasp is an optional backend whose SDK, license and weights are not included.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Getting started

```bash
scripts/runtime/manip up          # bring up the full stack
scripts/runtime/manip status      # per-component health, publish rates, transports
```

The workbench serves on `127.0.0.1:8766`. Calibration is not optional — the frame gate refuses
to plan without measured, non-synthetic evidence; start from
[mount and kinematic calibration](docs/piper_mount_and_kinematic_calibration.md).

```bash
python3 -m pytest -q              # offline contract and regression suite
```

## Layout

```text
z_manip/         perception, grasp, IK, planning and runtime modules
scripts/runtime/ manip CLI, bringup, diagnostics, visual servo, executors
web/             local operator and debug workbench
ros2/            ROS 2 bridges, observers and interface packages
configs/         schemas, collision model, service units, sample configs
docker/          GPU inference and planning runtimes
tests/           unit, contract and regression tests
docs/            operations, calibration, configuration, acceptance
```

## Documentation

- [Real-robot operations and recovery](docs/go2w_piper_operations.md)
- [Component manager and one-shot bringup](docs/component_manager.md)
- [Configuration schema and migration](docs/configuration.md)
- [Mount extrinsics and kinematic calibration](docs/piper_mount_and_kinematic_calibration.md)
- [NUC thin-deploy contract](docs/nuc_thin_deploy.md)
- [Staged pick-and-hold contract](docs/staged_pick_hold_contract.md)
- [Mobile manipulation acceptance](docs/mobile-manipulation-acceptance.md)
- [Architecture blueprint and roadmap](docs/plan.md)

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — use, modification and redistribution are
permitted for noncommercial purposes only. Commercial use requires a separate license.

Third-party software, models, robot SDKs and trademarks remain subject to their own licenses
and rights. This project is not affiliated with, sponsored by, or endorsed by Unitree Robotics,
AgileX Robotics, Intel, NVIDIA or their affiliates.
