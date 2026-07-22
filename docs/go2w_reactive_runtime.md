# Go2W reactive whole-body runtime

The live controller uses only motion channels supported by current Go2W
firmware:

```text
10-D control vector
  [base_forward, base_yaw_rate,
   body_roll_rate, body_pitch_rate,
   arm_q1_dot ... arm_q6_dot]

Move(vx, 0, vyaw)       -> rolling base
Euler(roll, pitch, yaw) -> body lean
PiPER qdot(6)           -> wrist-camera view and reach
```

`BodyHeight` (SPORT API 1013) and `GetBodyHeight` (1024) are not control
dependencies. Their availability is motion-service and firmware specific, so
the 10-DOF controller deliberately excludes height from its decision vector
and refuses non-zero `linear.z` instead of pretending the body moved.

The active service matters as much as the numeric API ID. Go2W `ai-w` is the
wheeled sport family: this robot accepts `Move(1008)` but returns 3203 for the
Go2-family `Euler(1007)`. The bridge records read-only MotionSwitcher
`CheckMode(1001)` evidence and the raw command response. It never calls
`SelectMode` or `ReleaseMode` automatically. B2/wheeled `FreeEuler(1051)` is a
boolean mode toggle, not an Euler target command.

The model may retain a fixed zero body-height state for transform compatibility,
but height is not a QP decision variable and cannot block the controller.
Vertical tracking uses target Z in `body`/`piper_base_link`; body pitch and the
six arm joints reduce that error. A broadcast body-height estimate or a D435
ground-plane estimate may be recorded as optional telemetry only.

## Data and command path

```text
D435 target point in camera frame
        |
        v
hand-eye calibration + Pinocchio FK
        |
        v
target (x, y, z) in body / piper_base_link
        |
        v
CasADi 10-D bounded QP at 20 Hz
        |
        +--> /cmd_vel ------------------------> NUC Move owner
        +--> /z_manip/reactive/posture_intent -> NUC Euler owner
        +--> /z_manip/reactive/arm_view_intent
                                                   |
                                                   v
                                      NUC PiPER CAN owner
                                      /z_manip/reactive/arm_view_status
```

The Go2W posture status is measured from SPORT state. Roll and pitch are used
for convergence. Absolute IMU yaw is not compared with a zero-relative Euler
command. Unsupported body-height fields are explicitly marked unsupported,
not replaced with command-ack estimates.

## PiPER CAN ownership

Passive feedback and reactive arm control must never own `can0` together.
`z-mobile-manip-piper-reactive-view.service` therefore conflicts with
`z-manip-piper-passive-feedback.service`.

An operator-started live depth-servo run performs this transaction:

1. Verify the base transport and NUC SSH access.
2. Start `z-mobile-manip-piper-reactive-view.service`; systemd stops the
   conflicting passive listener in the same transaction.
3. Verify the reactive service is active and the passive service is inactive.
4. Start the PC CasADi runtime.
5. On every normal exit, error, `SIGINT`, or `SIGTERM`, stop the reactive arm
   service and restart/verify passive feedback.

If acquisition fails after passive feedback is stopped, the launcher restores
passive feedback before it returns an error. The installer deploys the arm
service but explicitly disables and stops it; it is never enabled at boot.

The arm executor accepts only fresh, monotonically increasing JSON intents on
`/z_manip/reactive/arm_view_intent`. It clips joint velocity, per-cycle motion,
and URDF joint limits; stale input holds the measured pose. Measured joints,
target joints, ownership, accepted sequence, error, and faults are published on
`/z_manip/reactive/arm_view_status`.

## Full Stop

Publish `std_msgs/Empty` on `/z_manip/reactive/full_stop`. The PC stops base,
posture, and arm intent production. The NUC base owner sends `StopMove`; the
PiPER owner clears pending motion and holds its measured pose.

The latch is released only by `/z_manip/reactive/control_reset` after measured
feedback is fresh and quiet. Full Stop itself never waits for feedback.

## Shadow and live workflows

Shadow mode opens no motion transport and does not switch the PiPER CAN owner:

```bash
scripts/runtime/go2w_depth_servo.sh shadow /absolute/path/depth-servo.json
```

The UI's `START SHADOW` path is the normal first check after reboot. A healthy
solve reports the CasADi backend and advancing heartbeat while all transports
remain closed.

Live mode is operator-authorized:

```bash
scripts/runtime/go2w_depth_servo.sh live /absolute/path/depth-servo.json
```

It requires the fixed NUC WebRTC bridge, ROS domain 20, the measured hand-eye
calibration, the whole-body URDF, passwordless NUC SSH, and the on-demand PiPER
executor installed. `FULL STOP` remains available throughout the run.

## Installation

From the repository root:

```bash
scripts/runtime/install_go2w_reactive_runtime.sh
```

The script deploys:

- the NUC Go2W reactive bridge and its live user service;
- the PC posture relay;
- the NUC PiPER reactive-view executor and its user service.

Installation does not start or enable the PiPER reactive-view service. It does
restart the always-on Go2W bridge and PC relay; those services do not move the
arm. A live depth-servo run alone acquires the PiPER owner.

## Runtime supervision

An active status document must advance `updated_unix_ns` within 1.5 seconds.
Missing, stale, future, backwards, or frozen heartbeats terminate the servo and
latch a stationary degraded workflow. Handoff is permitted only when:

- target tracking and transforms are fresh;
- SPORT roll/pitch feedback has converged;
- the arm executor owns CAN and publishes fresh measured joint feedback;
- the arm view error and commanded joint rates are settled;
- the close-range IK probe succeeds.

No unsupported body-height measurement is part of these gates.
