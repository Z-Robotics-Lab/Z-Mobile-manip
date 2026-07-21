# z_manip_place

Standalone ROS 2 adapter for the observed placement planner. It fits support
geometry only from synchronized aligned RGB-D and a perception scene cloud,
then ranks candidates through MoveIt IK, collision-aware transit planning, and
collision-checked Cartesian approach/retreat. It never reads Isaac prims,
scene configuration, semantic object poses, or task ground truth.

## Inputs

- `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, and
  `/camera/color/camera_info`: exact-source-stamp RGB-D geometry.
- `/z_manip/perception/scene_pointcloud`: target-excluded current collision cloud.
- `/piper/state`: current arm joints, extracted by configured joint names.
- `/z_manip/perception/target_pointcloud` and
  `/z_manip/perception/status`: exact `x,y,z,u,v` target geometry plus its
  `request_id`, producer epoch, generation, frame, and source stamp.
- `/piper/execution_status`: measured approach completion, release command ID,
  accepted open command, and current gripper aperture.
- TF: stamped transforms from the camera and scene-cloud frames into `base`.
- `/z_manip/place/region_request`: schema-v2 JSON on `std_msgs/String`.

The region request is intentionally class-independent. `tool_from_object`,
`object_extent_m`, the frozen object-frame reference cloud, and orientation
axes must come from one grasp-time transaction, not a simulator transform.
The top-level owner identifies the newer place-time observation;
`object_reference_identity` independently identifies the older grasp-time
model. Both identities and `verification` are mandatory.

```json
{
  "schema_version": 2,
  "goal_id": "place-0007",
  "stamp_ns": 123456789000,
  "image_frame": "wrist_camera_optical_frame",
  "request_id": "place-observation-request",
  "producer_epoch": "perception-epoch-9",
  "generation": 12,
  "region_xyxy": [0.12, 0.18, 0.82, 0.76],
  "avoid_xyxy": [[0.42, 0.31, 0.56, 0.62]],
  "constraints": {
    "min_clearance_m": 0.03,
    "max_surface_tilt_rad": 0.35,
    "preferred_yaw_rad": 1.57,
    "yaw_tolerance_rad": 0.7,
    "min_support_fraction": 1.0
  },
  "object_extent_m": [0.08, 0.05, 0.19],
  "tool_from_object": [
    [1, 0, 0, 0], [0, 1, 0, 0],
    [0, 0, 1, 0.09], [0, 0, 0, 1]
  ],
  "object_reference_identity": {
    "request_id": "grasp-observation-request",
    "producer_epoch": "perception-epoch-8",
    "generation": 7,
    "observation_stamp_ns": 123450000000,
    "frame_id": "wrist_camera_optical_frame"
  },
  "object_reference_points_object": [
    [-0.03, -0.02, -0.09], [0.03, -0.02, -0.09],
    [-0.03, 0.02, -0.09], [0.03, 0.02, -0.09]
  ],
  "verification": {
    "require_upright": true,
    "upright_axis_object": [0, 0, 1],
    "orientation_symmetry": "axial",
    "symmetry_axis_object": [0, 0, 1]
  }
}
```

The reference array is abbreviated above. A real request contains 40-512
unique finite object-frame points whose measured extent agrees with
`object_extent_m`. Its grasp-time stamp must strictly predate place-time
`stamp_ns`; swapping the owner roles is rejected.

`orientation_symmetry` is exactly `none`, `axial`, or `auto`. `none` requires
three observable principal axes. `axial` requires `symmetry_axis_object` and
verifies its tilt while treating rotation about it as equivalent. `auto`
infers only full versus axial observability and rejects fully degenerate
geometry. `symmetry_axis_object` is forbidden for the other two modes.
`upright_axis_object` is an object-frame undirected axis; set
`require_upright=false` only when the requested placement intentionally permits
a non-upright pose. A side grasp therefore needs the observed non-identity
`tool_from_object` rotation rather than an identity placeholder.

The schema-v2 request stamp and image frame must exactly equal the color,
depth, camera-info, target-cloud, and perception-status source key. Planning
selects that immutable key from a bounded RGB-D cache and never falls back to
the latest frame. The independently stamped scene cloud must satisfy
`max_sync_skew_s`; all data, including the joint state and required TF, must
remain inside `max_snapshot_age_s`. Missing, old, mixed-frame, or colliding
source keys fail closed.
Each required source callback re-runs the same request-keyed assembler. Thus a
request that arrives before its identity, RGB-D, target cloud, scene, joints,
or completed-carry status starts automatically when the last exact input
arrives. The request is consumed atomically before one generation worker is
started, so concurrent callbacks cannot start it twice.

## Outputs

- `/z_manip/place/candidates`: green/rejected red approach arrows for RViz.
- `/z_manip/place/selected_poses`: pre-place, place, and retreat tool poses.
- `/z_manip/place/trajectory`: audited `transit/approach/retreat` joint path.
- `/z_manip/place/trajectory_contract`: strict `z_manip.place_contract.v2`
  trajectory metadata, canonical SHA-256 content digest, and immutable executor
  high-water snapshot.
- `/z_manip/place/status`: machine-readable state and failure diagnostics.
- `/z_manip/place/post_release_verification`: schema-v2 correlated observed
  placement evidence, published RELIABLE/TRANSIENT_LOCAL with depth 1.
- `/z_manip/place/plan`: `std_srvs/Trigger` for the latest complete snapshot.

The trajectory is published only if all three phases succeed. Transit uses the
configured MoveIt pipeline. Approach and retreat use `GetCartesianPath` with
`avoid_collisions=true` and a minimum path-fraction gate. The output contract
reorders every phase by configured joint name, verifies segment continuity,
and enforces strictly increasing timestamps.
The contract binds the resolved trajectory topic, while the digest covers the
header frame/stamp, ordered joint names, every point array, and every
`time_from_start`; a separately delivered stale trajectory therefore cannot
borrow a newer contract.

Planning may arm its execution transaction only from an exact completed carry
status: `succeeded`, trajectory owner, `carry` segment, and contract `none`.
The v2 output freezes that status's executor epoch and trajectory/gripper
command and source-time high-water marks, while declaring the place goal as the
future trajectory contract ID. Missing, stale, cross-epoch, or partially
updated snapshots reject the plan before any trajectory is published.

The request also freezes `executor_epoch` before planning. Task termination
publishes exactly one strict `z_manip.place_transaction_control.v1` abort on
`/z_manip/place/transaction_control`; only the exact goal/epoch can clear the
pending, planning, or armed transaction. A planning abort invalidates a
generation token, so a late worker cannot arm or publish. A retry can start
immediately even if an old MoveIt call is still returning: planner, evaluator,
payload-auditor, and worker-registry state are isolated by generation, and old
worker cleanup cannot clear the new transaction. Finite
`transaction_ros_timeout_s` and `transaction_wall_timeout_s` deadlines run in
parallel from request acceptance across pending, planning, and armed states;
no phase transition extends them. A steady-clock timer remains active while
simulation time is paused, and either clock moving backwards is terminal.
MoveIt calls use cancellable asynchronous futures with bounded service and
response polling, so abort or timeout releases the generation's worker and
clients without consuming a late result. Every planning, execution-chain,
post-release, or watchdog failure publishes a strict correlated
`z_manip.place_status.v2` before the transaction is reset for retry.

MoveIt's bare-robot result is not sufficient for candidate ranking. The node
also rebuilds the carried volume from the frozen grasp-time object points and
their measured extent, attaches it through `tool_from_object`, and continuously
checks every joint segment against the same observed scene. Transit is strict;
the last approach segment permits only support-facing contact when audited in
reverse; retreat treats the released object as a fixed obstacle. A failed
payload audit rejects that candidate before its motion score is considered.

After the measured `place_approach` succeeds, the node waits for the matching
open-gripper command and freezes its command ID. Once measured retreat
succeeds, it verifies multiple fresh frames from the same perception request,
producer epoch, and generation.
Both `place_approach` and `place_retreat` require an ordered `active` then
`succeeded` pair with the same positive trajectory `command_id`, source time,
`executor_epoch`, and `trajectory_contract_id`. Approach must exceed the
pre-publication high-water marks; retreat must use a distinct later command.
The release `gripper_command_id` and `gripper_received_at` must fall strictly
between approach and retreat source times. Bare, retained, reordered,
cross-epoch, or cross-contract acknowledgements cannot open observation.
The target cloud must remain supported within the 3-D region observed during
planning, stay within 2.5 cm for at least 0.5 seconds, and clear the configured
measured gripper probes by at least 4 cm. Each unique target pixel must match
the current organized depth XYZ. Geometry registration compares observations
only with the frozen grasp-time model. Axial registration is continuous and
requires equal transverse variances plus a separated symmetry axis. Before
motion planning, the current target must register against that model under the
retained attachment transform. Current RGB-D proves support visibility;
strict named joints, URDF forward kinematics versus stamped tool TF, and
gripper feedback prove separation.

The verifier fails closed on perception loss, stale or mixed identities,
source or ROS clock rollback, insufficient target/support points, loss of
plane contact, a changed release command, or stale robot feedback. Its output
correlates the placement goal, release command, perception ownership, and all
source stamps. Probe geometry and every threshold are externalized in
`config/place.yaml`; no simulator object transform or semantic class shortcut
is accepted.

Depth/TF loss, occlusion, and isolated geometric outliers are recoverable
sample rejections: they clear the continuous dwell window, increment bounded
diagnostics, and do not refresh the valid-observation timeout. A later clean
window may still verify; persistent invalid sensing terminates at
`post_release_observation_timeout_s`. Identity, ownership, command ordering,
source-clock, and release-command violations remain immediate terminal
failures.

Late-start consumers of `/z_manip/place/post_release_verification` must request
RELIABLE/TRANSIENT_LOCAL durability. A VOLATILE reader may be DDS compatible,
but it cannot recover the retained terminal sample after startup.

The exact terminal JSON contract consumed by the task is:

```json
{
  "schema": "z_manip.post_release_verification.v2",
  "state": "verified",
  "result": "post_release_target_stable_in_region",
  "failure": "",
  "observation_source": "synchronized_rgbd_pointcloud",
  "goal_id": "place-0007",
  "place_goal_id": "place-0007",
  "release_gripper_command_id": 12,
  "request_id": "task-request-7",
  "producer_epoch": "bridge-epoch-4",
  "generation": 3,
  "frame_id": "wrist_camera_optical_frame",
  "geometry_frame_id": "base",
  "planning_observation_stamp_ns": 123456789000,
  "release_ack_stamp_ns": 123456990000,
  "observation_start_stamp_ns": 123457010000,
  "first_observation_stamp_ns": 123457020000,
  "last_observation_stamp_ns": 123457570000,
  "first_status_stamp_ns": 123457020000,
  "last_status_stamp_ns": 123457570000,
  "first_rgb_stamp_ns": 123457020000,
  "first_depth_stamp_ns": 123457020000,
  "first_target_stamp_ns": 123457020000,
  "last_rgb_stamp_ns": 123457570000,
  "last_depth_stamp_ns": 123457570000,
  "last_target_stamp_ns": 123457570000,
  "last_joint_stamp_ns": 123457570000,
  "last_execution_status_received_ns": 123457575000,
  "sample_count": 3,
  "target_point_count": 240,
  "stable_duration_s": 0.55,
  "max_target_motion_m": 0.004,
  "region_support_fraction": 0.93,
  "target_gripper_clearance_m": 0.061,
  "target_depth_correspondence_max_error_m": 0.006,
  "object_position_error_m": 0.012,
  "object_orientation_error_rad": 0.08,
  "object_upright_error_rad": 0.06,
  "object_registration_inlier_fraction": 0.88,
  "object_registration_rms_m": 0.011,
  "object_orientation_mode": "axial",
  "planned_object_pose": [[1, 0, 0, 0.42], [0, 1, 0, -0.18], [0, 0, 1, 0.74], [0, 0, 0, 1]],
  "observed_object_center_m": [0.421, -0.176, 0.746],
  "rejected_sample_count": 1,
  "rejected_sample_reasons": ["post-release support is occluded"]
}
```

Every field shown above is mandatory in schema v2, including the planned pose,
observed center, registration metrics, orientation mode, and rejection
diagnostics. Acceptance thresholds are exactly:

- position error <= `post_release_object_position_tolerance_m` (`0.04` m)
- orientation error <= `post_release_object_orientation_tolerance_rad` (`0.35` rad)
- required upright error <= `post_release_upright_tolerance_rad` (`0.26` rad)
- registration inliers >= `post_release_min_registration_inlier_fraction` (`0.55`)
- registration RMS <= `post_release_max_registration_rms_m` (`0.025` m)
- target/depth XYZ error <= `post_release_target_depth_correspondence_tolerance_m` (`0.012` m)
- region support >= `post_release_min_region_support_fraction` (`0.80`)
- gripper clearance >= `post_release_min_gripper_clearance_m` (`0.04` m)
- stable duration >= `post_release_min_stable_duration_s` (`0.50` s) across at least `post_release_min_samples` (`3`) samples

All RGB, depth, target, and perception-status source stamps must be strictly
newer than `observation_start_stamp_ns`. Failed terminal messages use
`state="failed"`, `result="post_release_verification_failed"`, and a non-empty
`failure`; they retain the same fields, with zero sample metrics when no valid
post-release frame was accepted.

## Launch

```bash
colcon build --packages-select z_manip_place
source install/setup.bash
ros2 launch z_manip_place place.launch.py use_sim_time:=true \
  robot_description_file:=/absolute/path/to/go2w_with_arm.urdf \
  collision_model_file:=/absolute/path/to/piper_collision_capsules.json \
  transaction_ros_timeout_s:=30.0 transaction_wall_timeout_s:=60.0
```

All topics, frames, geometry thresholds, synchronization limits, MoveIt
services, tolerances, and fallback joint velocity limits are externalized in
`config/place.yaml`. The launch file owns neither Isaac Sim nor RViz.
