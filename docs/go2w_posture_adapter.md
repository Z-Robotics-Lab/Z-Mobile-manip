# Go2W posture adapter

`z_manip.control.go2w_posture` is the transport-independent boundary for
reactive Go2W body height and Euler posture control. It does not open WebRTC,
publish ROS commands, or move hardware by itself.

## Command ownership

One NUC process must own the WebRTC connection and inject a `SportTransport`
into the adapter. `SportCommandArbiter` then orders all high-level commands:

1. `StopMove` flushes every pending Move/posture command and dispatches first.
2. `BodyHeight` and `Euler` form a serialized posture pair.
3. `Move` is latest-value coalesced, so delayed velocity requests cannot build
   a stale queue.

The default posture policy waits for measured base velocity to become quiet.
Reactive camera retention may explicitly set
`PostureLimits(allow_posture_while_moving=True)`; transport dispatch remains
serialized even then. Full Stop always pre-empts both policies.

## Shadow before live

The default `mode="shadow"` returns the exact SPORT API id and parameters in
`PostureOutput.command` but never calls a transport. `mode="live"` cannot be
constructed without an explicitly injected transport. A live response whose
robot `status.code` is non-zero enters `fault` rather than treating WebRTC ACK
arrival as command acceptance.

The initial manipulation envelope is intentionally bounded to body-height
command `[-0.12, 0.02] m`, pitch `±12°`, roll/yaw `±8°`, height steps of
`0.01 m`, angle steps of `2°`, and a minimum `0.20 s` command period. These
values are configuration, not hidden controller constants.

> The installed WebRTC API calls command 1013 `BodyHeight`; this legacy
> interface represents the height setpoint using `parameter.data`. Before the
> first live deployment, verify whether the Go2W firmware reports that value
> as a nominal-height offset or an absolute measured height, and normalize the
> NUC state bridge accordingly. Euler `x/y/z` is roll/pitch/yaw and matches the
> official Unitree SDK contract.

## Feedback and UI contract

Posture execution requires fresh measured body height, roll/pitch/yaw, and base
velocity. Missing or stale feedback produces `blocked` with no command. Target
completion requires all measured errors to remain within tolerance for the
settling window.

`PostureOutput.status_document()` emits
`z_manip.go2w_posture_status.v1`. Runtime integration should publish it at
`runtime.posture_status`:

```json
{
  "schema": "z_manip.go2w_posture_status.v1",
  "mode": "shadow",
  "phase": "commanding",
  "command_owner": "posture",
  "body_height": {
    "current_m": 0.0,
    "target_m": -0.05,
    "error_m": -0.05,
    "feedback_age_s": 0.02
  },
  "attitude": {
    "current_pitch_rad": 0.0,
    "target_pitch_rad": -0.1,
    "pitch_error_rad": -0.1
  },
  "base": {"linear_speed_mps": 0.0, "yaw_rate_rps": 0.0, "quiet": true},
  "feedback": {"fresh": true, "source": "sport_mode_state"},
  "command": {
    "sequence": 1,
    "name": "BodyHeight",
    "api_id": 1013,
    "parameter": {"data": -0.01},
    "would_send": true,
    "sent": false,
    "accepted": null,
    "reason": "shadow: command not transmitted"
  }
}
```

FOV prediction and IK probing stay in the reactive geometry/handoff layer;
they must not be inferred from transport ACKs.

The concrete ROS/NUC wiring, Full Stop latch, systemd single-owner deployment,
and two independent live gates are documented in
[`go2w_reactive_runtime.md`](go2w_reactive_runtime.md).
