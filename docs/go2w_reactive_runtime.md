# Go2W reactive posture runtime

This runtime keeps geometry policy, ROS intent routing, and Unitree transport
ownership separate:

```text
PC reactive geometry
  /z_manip/reactive/posture_intent (JSON, neutral-relative deltas)
        |
        v
PC posture intent bridge (shadow by default)
  /go2w/posture_cmd (bounded BodyHeight offset + absolute Euler)
        |
        v
NUC reactive bridge (shadow by default; the only live WebRTC owner)
  Move + BodyHeight + Euler + GetBodyHeight + StopMove -> one SPORT request lock
  /go2w/posture_state -> measured, freshness-qualified status
```

The visual-servo policy publishes this input schema:

```json
{
  "schema": "z_manip.go2w_posture_intent.v1",
  "body_height_delta_m": -0.05,
  "pitch_delta_rad": -0.10
}
```

These values are API-1013 offsets. They are not accumulated on every control
tick. The PC relay clamps them to the calibrated envelope and turns them into
`/go2w/posture_cmd`. The NUC requires fresh `SPORT_MOD_STATE`
attitude/velocity. On firmware that implements API 1024 it also uses measured
`GetBodyHeight` feedback. The tested Go2W EDU firmware returns status code
3203 for API 1024, so the bridge falls back to a bounded API-1013 command-ack
estimate for the height offset while keeping roll/pitch/yaw measured from the
IMU. The fallback is explicitly reported as
`api_1013_command_ack_estimate`; it is never presented as measured height. Its
status uses
`z_manip.go2w_posture_status.v1` on `/go2w/posture_state`.

## Full Stop

Publish `std_msgs/Empty` on `/z_manip/reactive/full_stop`. In live mode the PC
relay forwards it to `/go2w/full_stop`. The NUC immediately latches stop,
drops pending Move and posture work, then sends `StopMove` in the next shared
SPORT request slot. The latch blocks all later motion.

Releasing the latch requires `std_msgs/Empty` on
`/z_manip/reactive/control_reset`. The NUC releases it only when measured SPORT
feedback is both fresh and quiet. Full Stop itself never waits for feedback.

## Shadow workflow

Both launchers default to shadow:

```bash
scripts/runtime/go2w_posture_intent_bridge.sh
scripts/runtime/go2w_reactive_control_nuc.sh
```

The NUC shadow process never constructs `UnitreeControlNode`, so it cannot open
WebRTC. Use it to verify ROS topics and UI status without a motion-capable
transport.

## Live deployment gates

Do not run the legacy `z-manip-go2w-base-control.service` together with the new
live bridge. `z-mobile-manip-go2w-reactive-live.service` declares a systemd
conflict with that service and the shadow service, making it the single WebRTC
owner.

Before supervised live testing, create this NUC-only acknowledgement file:

```bash
mkdir -p ~/.config/z-mobile-manip
cat > ~/.config/z-mobile-manip/go2w-reactive-live.env <<'EOF'
Z_MANIP_GO2W_LIVE_ACK=I_UNDERSTAND_GO2W_WILL_MOVE
# Optional only after moving-posture validation:
# Z_MANIP_GO2W_ALLOW_POSTURE_WHILE_MOVING=1
EOF
chmod 600 ~/.config/z-mobile-manip/go2w-reactive-live.env
```

The PC relay has an independent defense-in-depth gate:

```bash
export Z_MANIP_POSTURE_INTENT_LIVE_ACK=I_UNDERSTAND_POSTURE_INTENTS_REACH_NUC
scripts/runtime/go2w_posture_intent_bridge.sh live
```

The equivalent PC user units are
`z-mobile-manip-go2w-posture-intent-shadow.service` and
`z-mobile-manip-go2w-posture-intent-live.service`; the live unit reads its
acknowledgement from
`~/.config/z-mobile-manip/go2w-posture-intent-live.env`.

Merely installing/enabling the shadow units cannot move the robot. This change
does not deploy, restart, enable, or connect either service automatically.

## Installation targets

The versioned NUC service files are:

- `configs/z-mobile-manip-go2w-reactive-shadow.service`
- `configs/z-mobile-manip-go2w-reactive-live.service`
- `configs/z-mobile-manip-go2w-posture-intent-shadow.service`
- `configs/z-mobile-manip-go2w-posture-intent-live.service`

Install the script under `~/.local/lib/z-mobile-manip/` and the chosen unit
under `~/.config/systemd/user/`, then use `systemctl --user daemon-reload`.
Live status retains the raw GetBodyHeight response, parse path/error, robot
code, sample age, fallback source and query count so an unsupported firmware
response cannot masquerade as measured feedback.

## Integrated whole-body approach

`manip bringup` starts the complete chain: perception, the single NUC WebRTC
owner, the PC posture relay, and the Pinocchio/CasADi runtime image. The depth
servo loads the measured wrist-camera calibration and real Go2W URDF, solves a
bounded CasADi QP at runtime, and sends base velocity plus BodyHeight/Euler
targets through the single owner. The optimizer also computes PiPER joint
velocity as a reachability and conditioning diagnostic. Arm motion is not
streamed while the base walks; after the handoff distance is reached the
existing fresh-perception Pinocchio IK and checked arm trajectory executor own
the arm. This prevents two simultaneous PiPER command owners.

Use `START SHADOW` first after a reboot. A healthy solve reports the
`casadi-qrqp` backend and a decreasing objective without opening a motion
transport. `FIND → APPROACH → GRASP` is the operator-authorized live path, and
`FULL STOP` interrupts the base/posture workflow.

## State heartbeat supervision

The loopback supervisor does not treat a live depth-servo PID as proof that
the control loop is healthy. Every active status document must advance
`updated_unix_ns` within 1.5 seconds. This applies across target waiting,
tracking, base approach, posture adjustment, tracking loss, reacquisition, and
view-search requests; changing phase does not reset the heartbeat deadline.

A missing, stale, frozen, future, or backwards heartbeat terminates the servo
process and latches a stationary `degraded` workflow with
`REACTIVE_STATE_HEARTBEAT_TIMEOUT`. A `reached`, `handoff_probe`, or
`handoff_ready` document is accepted only with a currently valid heartbeat,
so stale state can never launch the grasp transaction. The status API exposes
the heartbeat source stamp, age, elapsed time without progress, and deadline
under `supervision` for replay and UI diagnosis.
