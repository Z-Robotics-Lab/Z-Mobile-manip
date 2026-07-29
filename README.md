# Z-Mobile-Manip

Supervised mobile manipulation for a Unitree Go2-W EDU quadruped carrying an AgileX PiPER
6-DoF arm and a wrist-mounted Intel RealSense D435.

![Operator workbench](docs/images/workbench.png)

## What this repository is

Z-Mobile-Manip is the **near-field half** of a three-repository stack: navigation drives the
robot to a coarse standoff, and this system takes over from about a metre out and runs through
to a completed grasp and return. An RTX 4090 workstation does perception, grasp synthesis, IK,
planning and the operator UI; the robot's onboard NUC owns the D435, PiPER `can0`, joint
feedback and the Go2-W WebRTC base transport, joined over `ROS_DOMAIN_ID=20`.

The pipeline grounds an open-vocabulary text target with **YOLOE**, tracks it with **EdgeTAM**,
and back-projects it against **Fast-FoundationStereo** depth rather than the D435's raw stream.
Grasps come from **AnyGrasp** with a local antipodal generator as fallback, are solved by a
damped Gauss-Newton **Pinocchio** IK, and are checked against two collision models: a capsule
model of the platform's own fixtures (`configs/piper_collision_capsules.json`, the only enforced
representation of the chassis, NUC and Mid-360 LiDAR) and an observed point cloud of the scene.
Trajectories come from **RRT-Connect** with quintic time parameterisation; **CasADi** refines
them and solves the 10-DoF whole-body QP coupling base forward/yaw and body roll/pitch to the
arm during the servo approach. Motion runs in short-lived, content-addressed executors copied to
the NUC per run, so the arm is never driven by a long-lived process. Seven ROS 2 packages carry
the bridges, and GPU workloads are isolated per image — `z-manip-runtime:pinocchio` bakes ROS 2,
Pinocchio and `ros2/**`; EdgeTAM, YOLOE, AnyGrasp and FFS each run in their own.

Two chains run on real hardware today:

- **Fixed base** — `perception → planning → grasp → return Home`
- **Mobile** — `find/track → depth approach → stop → close-range grasp`

### Before you change anything

- **Editing this checkout is a live deploy.** `z_manip/**` and `scripts/runtime/**` are
  bind-mounted read-only into the running robot and execute from here. `ros2/**` is baked into
  `z-manip-runtime:pinocchio` and rebuilt with `scripts/runtime/go2w_perception_lab.sh build` —
  not with `docker compose`, which targets a stale `:jazzy` tag.
- **Integration seam.** `z_manip_navigation` owns coarse approach as its own state machine
  (`NavPhase`) and hands over at `near_target_depth_m = 1.4` once the base settles below
  `still_speed_mps = 0.035`. From there the near-field servo owns the base.
- **Every process boundary is a versioned JSON document** — 79 `z_manip.*.vN` schemas. Open
  `depth_servo_trace.v1` first when something misbehaves: one row per tick carrying the phase,
  the reason string, the target age, the collision witness, and both the proposed and the
  published command. Execution leaves stage receipts that the offline tests replay, which is why
  regressions here are caught against recorded runs rather than in simulation.
- **Servo state is one enum and one policy table** (`z_manip/control/servo_phase.py`). Every
  phase carries a deadline, an expiry action, an expected base owner and a terminal flag; a
  phase nobody declared inherits a fail-closed policy instead of running unbounded.
- **Fail-closed throughout.** A missing calibration, a stale transform or an unproven collision
  corridor stops the robot, with a named reason in the status document.
- **Posture control is unavailable.** The Go2-W `wheeled_sport` interface exposes no `Euler` or
  `BodyHeight`, so the QP's body roll/pitch DOFs are locked to zero on this platform.
- **A green test suite does not mean IK or the QP ran.** `pinocchio` and `casadi` are absent
  from some hosts and those tests skip rather than fail.
- **Threshold comments are normative.** Each records the incident that motivated it; several are
  not derivable from the code, and more than one has been silently reverted by someone who
  assumed otherwise.

![Go2-W with PiPER arm](docs/images/robot.webp)

This is a supervised laboratory system, not an unattended product. Keep clearance around the
robot and the physical e-stop in reach during any motion test.

## Architecture

| Location | Responsibility |
|---|---|
| 4090 workstation | RGB-D decode, grounding and tracking, point clouds, grasp synthesis, IK, planning, UI |
| Go2-W NUC | D435 ROS service, PiPER `can0`, passive joint feedback, per-run executors, base WebRTC |
| Browser | Perception, planning, execution, Home/reset, Full Stop, diagnostics |

```text
target text → RGB-D grounding → EdgeTAM tracking → target point cloud
→ coarse align, depth-closed-loop approach → stop at ~0.50 m → near-field re-perception
→ grasp candidates → Pinocchio IK → collision check and path planning
→ pregrasp → approach → slow close → smooth lift → Home
```

![Pipeline](docs/images/pipeline.png)

## NUC thin-deploy contract

The two hosts do not run the same Python package. The 4090 imports the full `z_manip`
(Pinocchio IK, CasADi QP, planning); the onboard NUC runs a **thin package** that carries only
what a short-lived executor needs and **deliberately omits Pinocchio and casadi**. The decision
loop lives on the 4090 — the NUC only executes.

`scripts/runtime/install_go2w_reactive_runtime.sh` provisions the NUC by SCP-ing the reactive
executors plus a reduced `z_manip/` tree into `~/.local/lib/z-mobile-manip/`, and — this is the
contract — it copies `configs/nuc-kinematics-init.py` and `configs/nuc-control-init.py` and
**renames them to `__init__.py`** inside `z_manip/kinematics/` and `z_manip/control/`. Those stub
`__init__` files export only the light surface (`KinematicChain` from `chain.py`, the posture
transport marker) instead of the full package `__init__`, so importing `z_manip.kinematics` /
`z_manip.control` on the NUC never pulls Pinocchio/casadi. This keeps the robot's onboard host
free of the heavy IK/QP dependency chain while the executors still resolve the modules they need.

- **Do not** deploy the full package `__init__.py` to the NUC — it would drag in Pinocchio/casadi
  and break the thin-host assumption.
- The stub sources live in `configs/nuc-{kinematics,control}-init.py`; edit them there, not on the
  NUC (the deploy renames them in place).
- Rationale and the cross-machine picture are distilled in go2w-integration
  [`docs/reproduction.md` §A.5](https://github.com/Z-Robotics-Lab/go2w-integration/blob/main/docs/reproduction.md)
  and [`docs/modules.md` module C](https://github.com/Z-Robotics-Lab/go2w-integration/blob/main/docs/modules.md).

Passive CAN telemetry (read-only `/piper/state` at 20 Hz) is a separate, non-owning deploy:
`scripts/runtime/install_nuc_passive_access.sh`.

## Requirements

- Ubuntu 24.04, ROS 2 Jazzy, CycloneDDS, Docker with the NVIDIA Container Toolkit
- RTX 4090 (24 GB) workstation; Go2-W EDU with SSH-reachable onboard NUC
- AgileX PiPER on 1 Mbps SocketCAN with a matching URDF; wrist RealSense D435/D435i
- Time-synced PC and NUC on the same segment, identical ROS Domain ID

Base transport uses the third-party `unitree_webrtc_connect`; the PiPER backend uses
`pyAgxArm`. AnyGrasp is an optional backend whose SDK, license and weights are not included
here. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Getting started

```bash
scripts/runtime/manip up          # bring up the full stack
scripts/runtime/manip status      # per-component health, publish rates, transports
```

The workbench serves on `127.0.0.1:8766`. First-time setup — hand-eye calibration, mount
extrinsics, NUC provisioning — is in `docs/piper_mount_and_kinematic_calibration.md` and
`docs/component_manager.md`. Calibration is not optional: the frame gate refuses to plan
without measured, non-synthetic calibration evidence.

## Safety boundaries

- No motion command is issued without a validated collision corridor and fresh joint feedback.
- On fault the arm is left torqued while holding an object; the electronic e-stop fires only
  when the gripper is provably empty.
- Every phase the servo can emit carries a deadline and a terminal action; an unlisted phase
  fails closed (`z_manip/control/servo_phase.py`).
- Full Stop in the UI and the physical e-stop are independent paths.

## Verification

```bash
python3 -m pytest -q          # offline contract and regression suite
```

`pinocchio` and `casadi` are not importable on every host; tests covering those paths skip
rather than fail, so a green suite does not imply the IK or QP numerical paths were exercised.
Live-hardware checks sit behind a read-only health probe and skip when the chain is down.

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
- [Mobile manipulation acceptance](docs/mobile-manipulation-acceptance.md)
- [Staged pick-and-hold contract](docs/staged_pick_hold_contract.md)
- [Architecture blueprint and roadmap](docs/plan.md)

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — use, modification and redistribution are
permitted for noncommercial purposes only. Commercial use requires a separate license.

Third-party software, models, robot SDKs and trademarks remain subject to their own licenses
and rights. This project is not affiliated with, sponsored by, or endorsed by Unitree Robotics,
AgileX Robotics, Intel, NVIDIA or their affiliates.
