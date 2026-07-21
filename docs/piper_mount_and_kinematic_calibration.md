# PiPER mounting and kinematic calibration

## What this calibration can and cannot fix

The fixed transform `base -> piper_base_link` places the arm inside the Go2W
body model.  It matters for whole-body visualization, body/arm collision checks,
base repositioning, and targets expressed by sensors outside the arm chain.

For the wrist D435, an observed target can already be expressed in the PiPER
base frame without the body mounting transform:

```text
arm_from_target = FK(q) * tip_from_camera * camera_from_target
```

Consequently, changing `base -> piper_base_link` does not repair errors caused
by joint encoder zero offsets, link dimensions, compliance, or backlash.  Those
are arm kinematic parameters and need a separate validation/calibration stage.

## Observability requirement

A board fixed somewhere in the room is insufficient.  In that setup both
`body_from_arm` and `body_from_board` are unknown and can be changed together
without changing any camera observation.

Use one of these independent anchors:

1. a ChArUco board on a rigid, dimensioned bracket attached to the Go2W body;
2. a metrology jig that locates the board relative to the body frame; or
3. an external tracker that observes body and board/tool frames.

The bracket/CAD route is the simplest.  Define the structural Go2W `base` frame,
not a drifting `odom` frame.  Measure the full six-degree transform from `base`
to the ChArUco board and record its uncertainty.  A ruler-only estimate should
not claim millimetre uncertainty.

## Safe collection and solve sequence

1. Keep the Go2W base stationary and rigidly attach the board to the body.
2. Reuse the completed `tip_from_camera` hand-eye calibration.
3. With arm execution disabled, manually place the arm in 12–20 diverse poses.
4. At every pose capture the same passive-CAN and ChArUco observation used by
   the hand-eye workbench.  The camera and board must not move within a sample.
5. Create a real anchor file from
   `configs/piper_mount_anchor.example.json`; do not use the identity example.
6. Run the offline solver:

```bash
./scripts/runtime/piper_mount_workbench.sh capture

# After placing the measured anchor JSON at the path printed by the workbench:
./scripts/runtime/piper_mount_workbench.sh solve
```

The result reports a calibrated transform, every sample residual, and the
difference from the current URDF.  It never edits the URDF; applying the result
requires visual review and a separate approval.

## Kinematic accuracy stage after mounting

Use a different validation board or independent measured points and split poses
into fitting and held-out validation sets.  First estimate six joint zero
offsets with tight priors and report their observability/covariance.  Only fit
link lengths when external metric measurements and sufficiently rich poses are
available.  Joint zero, hand-eye, and mounting parameters should eventually be
optimized together with priors because one transform can otherwise absorb
another parameter's error.

Before any grasp execution, require held-out end-to-end error, no-execute IK and
collision visualization, and then staged low-speed approach waypoints.  An
eye-in-hand visual-servo correction near the object can remove residual absolute
model error, but it does not replace collision-safe geometry.

The provisional offline joint-zero fit is available after the mounting result
has passed review:

```bash
PYTHONPATH=. python3 scripts/runtime/piper_joint_zero_calibrate.py \
  --samples /path/to/independent_kinematic_samples.json \
  --hand-eye /path/to/piper_wrist_camera_calibration.json \
  --mount /path/to/piper_mount_calibration.json \
  --anchor /path/to/platform_target_anchor.json \
  --urdf /path/to/go2w_sensored.urdf \
  --output /path/to/piper_joint_zero_calibration.json
```

It uses every fourth pose as held-out validation, reports the six-parameter
observation Jacobian rank/condition and linearized parameter uncertainty.  The
dataset must be different from the hand-eye and mount fitting inputs, contain
passive camera/CAN zero-transmit evidence, and independently excite all six
joints by at least the configured range.  Boundary-hitting, unobservable, or
held-out-degrading fits are rejected.  The output is never applied automatically.

Review a generated report at `127.0.0.1:8770`:

```bash
./scripts/runtime/piper_joint_zero_ui.sh \
  /path/to/piper_joint_zero_calibration.json
```
