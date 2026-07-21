# Go2W + PiPER + D435 local-manipulation operations

This document is the operator-facing source of truth for the current real
robot stack. It describes the arm-only manipulation baseline that runs across
the Go2W NUC and the 4090 PC, how to recover individual processes, and where
future local/mobile manipulation should attach.

## 1. Deployment boundary

| Machine | Owns | Must not own during normal perception/planning |
| --- | --- | --- |
| Go2W NUC | D435 ROS service, `can0`, receive-only PiPER joint feedback, and the short-lived PiPER execution process started by an explicitly requested Home or grasp action | GPU perception, persistent planning, the web UI, or a second arm command owner |
| 4090 PC | RGB-D decoding, EdgeTAM, target grounding/grasp generation, synchronized session artifacts, Pinocchio IK, collision/path planning, runtime observer, and the loopback UI | Direct SocketCAN access or a continuously running actuator publisher |
| Browser | Fixed Home, perception, planning, full grasp, component restart/status/log actions | Arbitrary shell commands, file paths, ROS topics, joint targets, or trajectories |

Both computers use `ROS_DOMAIN_ID=20`. Camera and passive joint telemetry cross
Wi-Fi through ROS 2/DDS. IK and planning run on the PC; arm execution is copied
to and run on the NUC only for one explicit action. During execution the
receive-only feedback service yields the CAN connection, and the fixed executor
keeps one arm connection for the complete motion before passive feedback is
restored.

The dashboard listens only on `127.0.0.1:8766`. Component-management requests
use a fixed allowlist and cannot be turned into an arbitrary process or shell
launcher.

## 2. Daily cold start

After either computer has rebooted, run this on the 4090 PC:

```bash
manip bringup
```

Then open <http://127.0.0.1:8766/>. The same operation is available under
**Vision services → Bring up all vision services** if the UI itself is already
running.

`bringup` is idempotent and serialized by one `flock`. It performs bounded
health waits for, in order:

1. NUC `d435i.service`;
2. the receive-only `can0` gate and NUC passive joint-feedback service;
3. PC EdgeTAM, RGB-D bridge, perception ROS, subscribe-only runtime observer;
4. the loopback workbench UI.

This cold-start command sends no robot motion command. It assumes the deployed
NUC user services, scoped passwordless passive-CAN rule, SSH key, PC user
services, runtime images, URDF, and camera calibration are already installed.
Those are provisioning inputs, not recreated on every boot.

Verify the result without changing state:

```bash
manip status
```

Every row should be `healthy`. `nuc-camera` requires both an active service and
the physical D435 USB device, while `rgbd` requires a newly received camera
artifact rather than only a running container. `passive-feedback` also requires
`can0` to be `ERROR-ACTIVE`. A live process without its real input is reported
as `degraded`, not healthy.

For UI-only lifecycle control, use:

```bash
manip start
manip stop
manip restart
manip logs ui
```

`manip stop` refuses to stop the workbench while Home, perception, planning,
or grasp execution is active. It stops only the loopback UI/API and leaves the
camera, perception, and passive telemetry components unchanged.

## 3. Normal UI workflow

There are two intended workflows.

### One-click fresh grasp

1. Confirm the workspace is clear and remain beside the emergency stop.
2. Enter the visible target description in **Perception**.
3. Choose a bounded speed from 1–50%.
4. Click **Perception + perform** once.

It is not necessary to click **Run perception** first. The single action:

```text
verify measured Home (or run fixed Home)
→ capture one fresh RGB-D/perception session
→ plan once from Home
→ open gripper
→ transit to pregrasp
→ approach and close
→ smooth timed lift and hold
→ lower on the checked reverse lift
→ release at the original grasp pose
→ reverse approach and transit to Home
```

The plan is completed at Home before movement. There is deliberately no
close-range replan at pregrasp because the D435 depth stream is not dependable
inside its near blind zone.

### Inspect first, then execute

Use **Run perception** and **Run planning** separately to inspect segmentation,
candidates, IK, collision evidence, and the planned trajectory. **Direct
perform** executes only the successful plan bound to the currently selected
perception session. It skips Home, perception, and replanning, so the physical
arm must still match that plan's start state.

**Go Home** is a fixed server-owned recovery pose; the browser cannot supply a
different pose. **Clear demo** removes the active task selection and overlays
but retains historical artifacts and logs.

## 4. Component recovery and logs

Open **Vision services · status, logs and restart** in the dashboard. Each
component has independent **Restart** and **Logs** controls:

- `ui`: loopback workbench/API;
- `nuc-camera`: NUC D435 service;
- `passive-feedback`: receive-only PiPER telemetry;
- `observer`: PC subscribe-only runtime observer;
- `rgbd`: RGB-D decode bridge;
- `edgetam`: EdgeTAM service;
- `perception`: perception ROS container;
- `perception-all`: observer + RGB-D + EdgeTAM + perception ROS.

Restarts are refused while Home, perception, planning, or grasp execution is
active. A vision restart clears the selected perception and plan context so an
old overlay or trajectory cannot become executable after the restart.

Equivalent CLI commands are:

```bash
scripts/runtime/go2w_component_manager.sh restart ui
scripts/runtime/go2w_component_manager.sh restart nuc-camera
scripts/runtime/go2w_component_manager.sh restart passive-feedback
scripts/runtime/go2w_component_manager.sh restart observer
scripts/runtime/go2w_component_manager.sh restart rgbd
scripts/runtime/go2w_component_manager.sh restart edgetam
scripts/runtime/go2w_component_manager.sh restart perception
scripts/runtime/go2w_component_manager.sh restart perception-all

scripts/runtime/go2w_component_manager.sh logs manager 100
scripts/runtime/go2w_component_manager.sh logs ui 100
scripts/runtime/go2w_component_manager.sh logs nuc-camera 100
scripts/runtime/go2w_component_manager.sh logs passive-feedback 100
scripts/runtime/go2w_component_manager.sh logs perception-all 100
```

Log requests accept 1–300 lines and are additionally capped at 24 KB. The
manager's append-only log is stored under
`../artifacts/go2w_real/component_logs/component-manager.log`.

For read-only perception/planning diagnosis without the browser:

```bash
python3 scripts/runtime/go2w_interactive_sessions.py perception "白色充电器"
python3 scripts/runtime/go2w_interactive_sessions.py planning
python3 scripts/runtime/go2w_interactive_sessions.py status
```

The low-level `piper_*executor*` and `piper_*_remote*` programs are execution
internals, not general recovery commands. Routine real-arm execution should use
the fixed UI action so the selected immutable plan, speed, receipts, and return
path stay bound to one transaction.

## 5. Data and session lifecycle

Interactive perception and planning attempts are immutable directories under:

```text
../artifacts/go2w_real/interactive_sessions/
```

Workbench action logs and execution receipts are under:

```text
../artifacts/go2w_real/planning_sessions/
├── piper-home.log
├── piper-grasp.log
└── execution-receipts/
```

The runtime observer writes the latest read-only state and camera snapshot to
`../artifacts/go2w_real/latest/`. UI selection is separate from history:

- new successful perception invalidates the previous plan selection;
- a successful Home, component restart, Clear demo, or completed full grasp
  clears the active task context;
- historical artifacts and logs remain available for diagnosis;
- stale/invalid tracked images are display-only and cannot be used for a new
  plan or execution;
- a large **runtime sequence** value is a monotonic observer counter, not a
  process backlog or number of queued frames.

Only one interactive action and one component-management operation may own
their respective lock at a time. Do not start duplicate dashboard servers or
parallel perception scripts against the same artifact root.

## 6. What can move the robot

The following operations do **not** drive the arm or gripper:

- component `bringup`, `status`, `logs`, and vision-component `restart`;
- live RGB-D/EdgeTAM tracking and runtime observation;
- **Run perception**;
- **Run planning** (network-disabled offline planner);
- **Clear demo** and UI restart.

The following operations can drive the PiPER and require an operator present:

- **Go Home**;
- **Perception + perform** after Home verification/planning succeeds;
- **Direct perform** of the current selected plan.
- **Find → approach → grasp** in live mode, including its optional fixed-view
  J4/J5 wrist-camera search.

Full execution uses only the immutable collision-checked outbound paths and
their exact reverse for place-back/Home. Failure after motion begins can invoke
the executor's stop behavior; component restarts are not a substitute for the
physical emergency stop. Never restart or power-cycle the NUC while the arm is
moving.

### Mobile find → approach → grasp

The mobile action is one server-owned transaction; it does not depend on the
browser remaining open:

```text
fresh open-vocabulary detection in the current D435 frame
→ if absent: finite J4/J5 wrist-camera viewpoints
→ two consecutive detections at confidence >= 0.55
→ fresh EdgeTAM seed and three depth-validated tracking updates
→ filtered target x/z at 5–10 Hz
→ coarse legged-base alignment, body/view adjustment, and curved vx/wz approach
→ zero-speed `handoff_probe` / `handoff_ready` at the 3-D arm corridor
→ latched zero base command
→ fresh close-range perception → IK → plan → grasp
```

The base controller deliberately does not chase exact yaw. Angular correction has a
6 deg deadband, and rotate-in-place is reserved for errors above 25 deg. This prevents
normal Go2W stepping sway from repeatedly reversing the turn command. Once the target
enters the handoff cone, base motion latches stopped and final alignment belongs to
the close-range perception and arm IK stages.

The forward profile is 0.10--0.18 m/s outside the handoff cone. The 0.10 m/s
floor avoids the observed Go2W low-speed gait dead zone, while the 0.18 m/s cap
keeps the far-field approach smooth. The NUC guard permits up to 0.20 m/s so it
does not silently clip this server-owned profile.

The current detector adapter calls the resident local Grounding DINO service;
the search policy is detector-neutral, so YOLOE can replace that adapter later
without changing wrist motion, EdgeTAM, servo, or grasp handoff. Target x/z is
filtered with a five-sample median plus EMA. A single jump above 0.20 m is
rejected. Tracking loss commands zero immediately, allows a stationary 0.75 s
reacquisition window, then runs at most three fresh perception attempts. It
never blind-drives through a missing target.

`view_recovery` and `search_required` are stationary recovery phases. The
supervisor first terminates the base-servo owner (which sends its zero-command
cleanup), then runs the bounded wrist search and a fresh EdgeTAM seed. Only
after a stable target is recovered does it restart base approach. Wrist search
and base velocity therefore never run concurrently. `handoff_probe`,
`handoff_ready`, and the legacy `reached` phase all use the same zero-speed
fresh-grasp handoff; no far-field perception or trajectory is reused.

**Full stop** cancels the server-side workflow, terminates the depth-servo
process, clears its task context, and interrupts a pending local wrist-search
subprocess. The fixed NUC wrist command is still finite and must never replace
the physical emergency stop.

Live wrist search is deliberately disabled after installation/reboot. The
confirmation shown by **Find → approach → grasp** grants one-shot authorization
for only that finite scan; completion, failure, Full Stop, or service restart
removes it. This keeps the normal UI workflow one-click after confirmation.

For a supervised tuning session with an operator physically present, the
workspace clear, and the emergency stop in reach, it may instead be enabled for
that user-service session:

```bash
systemctl --user set-environment Z_MANIP_ENABLE_WRIST_SEARCH=1
systemctl --user restart z-manip-planning-workbench.service
```

After the supervised experiment, lock it again:

```bash
systemctl --user unset-environment Z_MANIP_ENABLE_WRIST_SEARCH
systemctl --user restart z-manip-planning-workbench.service
```

Even with one-shot authorization, the browser can request only a fixed view
index and a bounded 1–12% search
speed; it cannot submit joint angles. The finite grid is anchored to the
measured software Home, changes only J4/J5, requires the other joints to remain
at Home, and has a hard 75 s deadline. While no operator is present, keep the
environment variable unset and use only **Shadow check**.

## 7. Troubleshooting

| Symptom | Action |
| --- | --- |
| UI unavailable after reboot | Run `go2w_component_manager.sh bringup`; if it fails, inspect `logs manager 100`, then `restart ui`. |
| `NUC D435` says degraded/USB absent, or the camera endpoint returns HTTP 503 | Reseat the D435 USB/power connection on the NUC and confirm it appears in `lsusb`. Then run `manip component restart nuc-camera` followed by `manip component restart perception-all`, or run `manip bringup`. The manager now fails immediately while the USB device is absent instead of restarting a process that cannot publish frames. |
| RGB is live but mask/candidates are stale or invalid | Use `restart edgetam`; if unresolved, use `restart perception-all`, then run a fresh perception. |
| Button cannot be clicked | Check whether Home/grasp/session/component status says `running`. Controls remain locked until the active fixed action finishes. Do not start a duplicate server. |
| Old object or trajectory remains on screen | Click **Clear demo**. This clears active context without deleting evidence. |
| Passive feedback is degraded after NUC reboot | Run `bringup`, then inspect `status passive-feedback` and `logs passive-feedback 100`. The manager only reports healthy after the service is active and `can0` is `ERROR-ACTIVE`. |
| Perception succeeds but planning is blocked | Read candidate rejections and the planning artifact. IK/collision rejection is a geometric result, not evidence that restarting perception will help. Capture a fresh view after moving the target/camera, or fix the relevant calibration/model/planner issue. |
| Direct perform is unavailable | A successful plan for the currently selected perception is required, and the physical arm must match its start. Use **Perception + perform** for a new Home-planned transaction. |
| UI appears responsive but state no longer advances | Use **Restart UI** or `restart ui`; if camera/tracking also stopped, restart the affected vision component instead of repeatedly polling the dead endpoint. |

## 8. Controlled shutdown

Do not stop services while an actuator action is running. After the arm is
stationary at Home, the supported limited shutdown commands are:

```bash
scripts/runtime/go2w_planning_workbench.sh stop-ui
scripts/runtime/go2w_perception_lab.sh stop
scripts/runtime/go2w_perception_lab.sh anygrasp-stop   # only when used
```

`go2w_perception_lab.sh stop` stops the PC `z-manip-hw` and `z-manip-rgbd`
containers only and sends no actuator command. The NUC camera and passive
feedback services are intentionally left independent. Use the component
manager's `bringup` or fixed `restart` commands for the next session.

## 9. Extension points for local/mobile manipulation

Keep the current arm-only transaction as a tested leaf capability. Add local
manipulation through existing boundaries rather than expanding the browser into
a generic command console:

1. put base/arm coordination and retry budgets in
   `z_manip/orchestration/mobile_manipulation.py`;
2. use `z_manip/planning/work_pose.py` and `standoff.py` to choose a bounded
   base work pose before the existing Home-planned arm grasp;
3. keep far navigation and near visual alignment behind
   `z_manip/control/approach.py` and the ROS packages under
   `ros2/z_manip_navigation`/`ros2/z_manip_motion`;
4. preserve one motion owner: base and arm motion remain mutually exclusive
   until a separately verified loco-manip milestone;
5. carry the calibrated `base_link → piper_base_link` transform, synchronized
   observations, immutable IDs, and receipts across every new stage;
6. register new long-running services in the component manager's fixed
   allowlist with bounded health checks and logs; never accept a process name,
   topic, pose, or shell command from the browser;
7. extend the UI with explicit task actions and visible stage/owner state, while
   leaving model, IK, collision, planning, and hardware adapters independently
   replaceable.

The next milestone should therefore call the existing local grasp transaction
only after the base has reached and verified a work pose, not merge base control
into the current PiPER executor.
