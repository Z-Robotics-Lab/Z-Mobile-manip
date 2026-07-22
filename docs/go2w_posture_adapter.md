# Go2W posture adapter

> Legacy transport-independent adapter note: this document describes the
> older `BodyHeight + Euler` abstraction and is retained for offline contract
> tests. It is **not** the capability declaration for the current Go2W
> `ai-w` runtime. Live capability is determined from MotionSwitcher
> `CheckMode` plus each robot RPC response. In the present `ai-w` session,
> `Move(1008)` works while Go2 `Euler(1007)` returns 3203; wheeled
> `FreeEuler(1051)` is only a boolean mode toggle.

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
4. `GetBodyHeight` (API 1024) is latest-value coalesced and uses the same
   serialized SPORT request owner. It is feedback, never a second WebRTC
   connection.

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

The installed Go2W WebRTC firmware calls command 1013 `BodyHeight`; this
interface represents the height setpoint using `parameter.data`. Completion is
checked against API 1024 `GetBodyHeight`, in the same command domain. The
absolute `SPORT_MOD_STATE.body_height` remains useful telemetry, but the
controller never combines it with a hand-written nominal height. An unknown,
rejected, ambiguous, malformed, or stale API-1024 response blocks posture
commands and exposes its raw response and parse evidence. Euler `x/y/z` is
roll/pitch/yaw.

## Feedback and UI contract

Posture execution requires both fresh API-1024 height-offset feedback and fresh
SPORT state roll/pitch/yaw/base velocity. Missing or stale feedback produces
`blocked` with no command. Target completion requires all measured errors to
remain within tolerance for the settling window.

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
