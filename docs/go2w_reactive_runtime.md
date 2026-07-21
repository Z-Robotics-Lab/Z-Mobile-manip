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
`/go2w/posture_cmd`. The NUC rejects posture commands without both fresh
`SPORT_MOD_STATE` attitude/velocity and fresh API-1024 `GetBodyHeight`
feedback. Its status uses
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
Keep the live unit disabled until API-1024 parsing and the physical command
envelope have been inspected in shadow/log replay. Live status retains the raw
GetBodyHeight response, parse path/error, robot code, sample age, and query
count so an unsupported firmware response cannot masquerade as feedback.
