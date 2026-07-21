# Reduced whole-body shadow controller

The first controller is intentionally reduced to commands the real platform
already exposes. It does **not** optimize Go2W leg torques:

```text
state   [base x, base y, yaw, body height, roll, pitch, PiPER q1..q6]
intent  [forward speed, yaw rate, height rate, roll rate, pitch rate, qdot1..qdot6]
```

The missing lateral base velocity is deliberate. It makes the rolling base's
non-holonomic constraint structural. Go2W's internal controller remains the
only leg-level whole-body controller.

## Model and objective

`PinocchioReducedWholeBodyModel` locks every URDF joint except PiPER joints
1–6. A virtual `x/y/yaw/height/roll/pitch` transform wraps Pinocchio FK and
frame Jacobians. The measured `tip_from_camera` hand-eye transform creates the
optical frame outside the URDF; CAD camera placement is not substituted for
calibration.

At each 0.2 s shadow tick, `WholeBodyShadowOptimizer` linearizes:

- target image-plane error, to retain the object in D435 FOV;
- Euclidean target distance on the ground plane;
- target height relative to the body;
- 3-D tool-to-handoff error;
- arm manipulability;
- command magnitude and change from the prior command.

CasADi/QRQP solves the bounded velocity QP. Bounds include base/body rates,
URDF arm velocity limits, and one-step arm position feasibility. Far from the
target, ground approach dominates. Body, camera and arm weights rise
continuously near the handoff zone. There is no discrete "posture must settle"
phase that can block approach.

Camera projection is fail-closed. A synchronized target whose optical-frame
depth is behind the camera or at/below `camera_min_depth_m` is never clamped to
a synthetic positive depth. `linearize()` rejects that geometry, while
`solve()` returns a zero-velocity result with `success=false`,
`executable_intent=false`, and `failure_code=CAMERA_DEPTH_BELOW_MINIMUM`.
This prevents an invalid image projection from becoming a nominal QP command.

This code currently emits intents only. It contains no ROS, WebRTC, CAN or SDK
transport. A report always includes `transport_opened=false` and
`motion_commands_sent=0`.

## Offline verification

Ordinary dependency-light tests use the SciPy reference QP:

```bash
python3 -m pytest -q tests/test_whole_body_optimizer.py
```

The robotics environment verifies real Pinocchio FK/Jacobians and the CasADi
QP:

```bash
/home/yusenzlabpc/Z-Robotics-Lab/z-agent/.venv/bin/python \
  -m pytest -q tests/test_whole_body_optimizer.py
```

Replay a recorded target and passive joint snapshot without opening hardware:

```bash
/home/yusenzlabpc/Z-Robotics-Lab/z-agent/.venv/bin/python \
  scripts/offline/whole_body_shadow_replay.py \
  --urdf ../go2W_Sim/assets/urdf/go2w_sensored.urdf \
  --calibration ../artifacts/go2w_real/calibration/piper_wrist_camera_calibration.json \
  --passive-joints ../artifacts/go2w_real/planning_sessions/20260717-164946/passive_joint_report.json \
  --trace ../artifacts/go2w_real/latest/depth-servo.trace.jsonl \
  --ticks 40 --output /tmp/whole-body-shadow.json
```

Before any live integration, the remaining gates are measured body-height and
Euler feedback, a single serialized SPORT owner, a smooth PiPER view-servo
adapter, collision constraints, and atomic ownership transfer from reactive
view control to the grasp executor.
