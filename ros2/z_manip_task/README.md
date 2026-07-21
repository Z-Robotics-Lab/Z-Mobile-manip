# z_manip_task

`z_manip_task` is the online, fail-closed ROS 2 task executor. It accepts an
arbitrary text request on `/z_manip/task/request` and composes the root Python
stack's VLM-seeded persistent perception, reachability standoff, visual servo,
6-DoF grasp generation, robust IK, perceived-scene collision checking,
RRT-Connect, retiming, execution, and sensor-only grasp verification.

## Safety and data contract

- Target and scene geometry come only from the synchronized
  `/z_manip/perception/target_3d`, `target_pointcloud`, and `scene_pointcloud`
  topics. The node has no object-truth subscriptions.
- The camera-to-platform transform is resolved from TF at the observation
  stamp. The platform mount-parent to PiPER transform is parsed from the fixed
  path in the configured URDF. The three distinct frame names are external
  `RobotModelConfig` fields.
- The initial full IK/collision plan is used only to choose a reachable camera
  standoff. After servo convergence, a strictly newer synchronized RGB-D
  observation must regenerate the object cloud, candidates, and final plan.
- Every semantic pass publishes a unique `z_manip.grounding_request.v2`
  `request_id` plus an explicit `grasp_only`, `grasp_for_place`, or
  `place_support` scope. Perception status and VLM affordance must echo that exact ID,
  one bridge `producer_epoch`, its local generation, and the same observation
  stamp/frame as all three geometry topics. Unrelated status cannot advance a
  generation floor, and a restarted bridge may safely begin again at
  generation 1 for the task's new request.
- `grasp_for_place` freezes visible object-axis semantics while deferring support
  geometry. A later `place_support` pass continues to track the grasped object as
  `target` for live pose validation and separately observes the support through
  `placement_region` and `placement_avoid_regions`.
- Coarse navigation retains velocity ownership until either the observed target
  depth is inside the configured near threshold or an external navigation
  supervisor publishes `true` on `/z_manip/navigation/coarse_ready`. Required
  base displacement from the standoff optimizer is diagnostic only and is
  never misused as a navigation range.
- LOOKOUT is given a configurable simulation-time settling interval before the
  grounding request is emitted, so VLM initialization never intentionally uses
  a moving-camera frame.
- Each trajectory phase is revalidated immediately before publication against
  the newest fresh scene cloud. The sole exception is a lift already inside the
  bounded near-contact occlusion gate, which is revalidated against its copied
  exact-authorized scene. Planning itself uses an atomic snapshot clock so a
  long RRT call cannot partially expire its own input.
- Standoff, final-grasp, and carry planning each have a configurable aggregate
  monotonic wall-clock budget. Task cancellation signals the active IK/RRT
  worker cooperatively, so a stale plan cannot monopolize the single planner.
- Wrist-camera occlusion may replace live tracking only after the exact
  task-owned request, producer epoch, generation, frame, serial, and integer-ns
  observation stamp retained at measured approach completion. The hard window
  is at most 3.0 seconds and is checked on every tick against both JointState
  receipt age and monotonic ROS source stamps, the exact close-command
  acknowledgement, measured contact, lift executor identity, and monotonic
  progress along the cached-scene-validated lift path. After loss it allows one
  bounded wait for the next source-stamped joint sample; replay, rollback, or a
  split serial/stamp pair is rejected. A copied target may seed only the pre-lift
  verifier baseline; live exact-authorized target tracking must return for grasp
  verification and carry planning. No object truth or unbounded blind timeout
  is used.
- Track loss before near contact, expired or unexpected occlusion, stale data,
  posture violation, executor rejection, path deviation, or invalid trajectory
  start immediately publishes zero base velocity, cancels navigation, and
  cancels the arm.
- Fresh measured roll and pitch are required before LOOKOUT and continuously
  through navigation, planning, grasp, carry, and placement. The freshness and
  first-sample acquisition limits are runtime parameters; missing or invalid
  state estimation never authorizes motion.
- Platform pose, posture, speed, and visual-search drift use base-origin
  odometry from the legacy-named `state_estimation_topic`, which defaults to
  `/odom_base_link`. Every message must match the configurable parent/child
  frame contract (`map -> base_link` by default). Raw `map -> sensor`
  `/state_estimation` is rejected because its sensor lever arm appears to
  translate during an in-place base rotation.
- During visual search, the same measured odometry closes XY position around
  the turn anchor. Map-frame position error is rotated into the current base
  frame, passed through configurable deadband, gain, and vector speed limits,
  then published with the bounded yaw command during the initial turn. Once yaw
  and the separate position-completion tolerance are satisfied, motion stops and
  fresh odometry drives re-planning from the measured pose. Larger XY errors
  close under an independent, finite position-hold timeout with yaw commanded to
  zero. If translation disturbs heading, the controller pauses XY while it
  reacquires yaw. The task-local zero command remains live on every pose-settle
  control tick. At the end of the minimum settle period, fresh odometry is
  checked against the retained search anchor and target yaw. Separately measured
  base linear and angular speed must be below their configurable SI thresholds,
  final yaw must remain within the search tolerance, and the independent 0.15 m
  drift gate must pass before re-grounding. A platform profile may configure a
  finite additional stationary-wait timeout; only finite measured speed above a
  threshold may wait. Freshness and hard drift remain immediately fail-closed,
  while the final yaw gate is evaluated on the first stationary sample. The
  generic timeout is zero, with one configured control period of bounded
  scheduler grace. Both the odometry callback sequence and its source
  `header.stamp` must advance strictly past the stop sample; receipt age and
  source age are checked independently on every retry. Pose-settle and semantic
  re-ground timers fail closed if ROS time moves backwards. The posture gate
  remains fail-closed.
- Visual-search yaw and position-hold deadlines likewise include exactly one
  configured control period of scheduler grace. A completion sample after that
  hard bound is rejected even if its measured pose is otherwise acceptable.
- Publishing `true` on the configurable `/z_manip/task/cancel`
  (`std_msgs/Bool`) performs the same three stops atomically with task-state
  cancellation. It invalidates in-flight planning, resets the task-owned VLM
  and EdgeTAM session through `/z_manip/grounding/reset`, clears grasp and
  placement programs, disables visual search, and publishes terminal
  `phase=canceled,result=canceled` status. `false` is ignored, repeated cancel
  requests are idempotent, and a later text request starts a new task.
- VLM avoid regions are always removed through the EdgeTAM point cloud's
  aligned `u/v` fields. A grasp-part crop is used when it has enough depth
  support; the explicit fallback reason is published in task status.
- Verification uses measured gripper aperture, arm FK/lift, and persistent
  target co-motion. It does not use an object pose oracle.
- Placement completion also requires the versioned
  `/z_manip/place/post_release_verification` result. The task freezes the exact
  planning request, producer epoch, generation, frame, source stamp, place goal,
  and measured release command ID; mismatched, malformed, failed, or timed-out
  geometry evidence fails the task instead of reporting completion. Schema v2
  additionally requires bounded depth correspondence, object position,
  orientation/upright error, registration inlier/RMS, a proper planned SE(3)
  pose, and a finite observed center. Missing, non-finite, legacy, or weak fields
  cannot complete the task.

## Task result and placement

`place_mode:=carry_only` ends in `PICK_COMPLETE`, published as
`result=pick_complete`; it is deliberately distinct from
`MOBILE_MANIP_COMPLETE`. The default `observed_place` mode performs another
VLM grounding pass, derives carried-object extent from its measured target
cloud, and requests `/z_manip/place/region_request`. At verified grasp time it
uses robust PCA on the exact authorized target cloud plus synchronized measured
FK to freeze a non-placeholder `tool_from_object`, observable object axes,
extent, and a bounded object-frame reference cloud. Structured VLM output must
explicitly select natural upright and symmetry principal axes; ambiguous or
contradictory geometry fails closed without an object-class default. The later
placement request carries both place-time perception ownership and the separate
grasp-time reference identity so the verifier cannot construct its reference
from the observation it is testing. It validates the returned
`z_manip.place_contract.v2` trajectory contract and executes `place_transit`,
`place_approach`,
measured gripper release, and `place_retreat` as separately acknowledged
commands. After measured retreat success it enters
`post_release_verification` and publishes `mobile_manip_complete` only after
bounded, synchronized RGB-D evidence shows the target stable in the planned
support region and clear of the measured gripper.

The task freezes the completed-carry executor epoch plus both trajectory and
gripper command/source high-water pairs when it issues the placement request.
It accepts only an exact v2 contract whose goal, future trajectory contract ID,
epoch, and all four high-water values match that request. Duplicate JSON keys,
legacy or unknown fields, and any missing or changed snapshot field fail closed
before `place_transit` can be published.
Because ROS delivers the trajectory and JSON contract on separate topics, the
task also requires the contract's resolved topic and frame plus a canonical
SHA-256 over the complete received `JointTrajectory`; matching point counts and
joint names alone never authorize execution.

That same request carries the frozen executor epoch. After a place request has
been published, terminal ownership release emits exactly one strict abort for
the unchanged goal/epoch before local identity is cleared. The task accepts
only strict correlated `z_manip.place_status.v2` failures and listens throughout
planning, transit, approach, release, retreat, and post-release verification;
foreign or stale goal/epoch statuses are ignored, while a matching late failure
immediately cancels arm and base ownership.

## Debug outputs

- `/z_manip/task/status` (`std_msgs/String`, JSON schema
  `z_manip.task_status.v1`), including the visual-search map anchor,
  base-frame position error, commanded XY velocity, yaw error, and drift
- `/z_manip/debug/markers` (`visualization_msgs/MarkerArray`) for selected
  pregrasp and grasp axes
- `/z_manip/debug/arm_path` (`nav_msgs/Path`) for the planned end-effector path

Debug geometry is transformed into the configured live platform TF frame; the
internal URDF PiPER base frame does not need to exist in the ROS TF graph.

These topics are intended for RViz alongside the perception overlay, target
mask, aligned depth image, selected target cloud, full collision scene, TF,
robot model, navigation path, and PiPER execution diagnostics.

## Launch

Set `Z_MANIP_STACK_CONFIG` and the URDF environment referenced by that JSON,
then launch with the installed runtime YAML:

```bash
ros2 launch z_manip_task task_runtime.launch.py \
  runtime_parameters:=/path/to/runtime.yaml \
  platform_parameters:=/path/to/validated-platform-overrides.yaml
```

`platform_parameters` is optional and is applied after the generic runtime
file. The installed `config/go2w_sim.yaml` enables the simulation adapter's
`MANIP_LOOKOUT` pose together with its measured joint target, tolerance, and
bounded arrival timeout. Leave the argument empty on unknown or real
platforms; the generic runtime does not assume that named pose exists.

Grasp execution is deliberately split at pregrasp. The first planning request
may validate a complete candidate, but its runtime result exposes only the
collision-free transit. After the executor reports that exact command active
then succeeded, the task requires a newer measured joint sample at the
pregrasp endpoint, a stationary arm window, and a newer exact-authorized RGB-D
bundle synchronized to joint feedback. It then reruns candidate generation,
IK, collision checking, and time parameterization from those measured joints
before the approach trajectory can be published. A completed second-stage plan
must then match a strictly newer exact RGB-D bundle, including robust target
extent and observable principal-axis checks, and a newer dual-clock-fresh joint
sample. This prevents a cached plan from authorizing contact after target or
arm motion. The result callback freezes a second JointState sequence/source
watermark, so feedback that arrived while stage-two planning was running cannot
authorize execution; publication waits boundedly for a sample newer than the
completed result.

Every non-placement trajectory also carries a fresh bounded
`trajectory_token` in `JointTrajectory.header.frame_id`, and freezes the latest
executor epoch plus ROS publish time. The adapter must echo that exact token and
report `trajectory_received_at` in the same ROS clock domain. An `ACTIVE` with
another token or epoch, including a delayed old DDS command received after the
new publication, cannot own the command. A different-token `ACTIVE` that was
newly accepted beyond the frozen command/source watermarks immediately fails
the task and requests an arm cancel; only retained pre-publication status is
ignored. This contract is shared by simulation and hardware adapters and does
not depend on simulator state.
Rejected attempts additionally report independent `trajectory_event_token` and
`trajectory_event_received_at` fields. This lets the task correlate and cancel
a rejected current publication without replacing the last accepted command's
ID, segment, token, or receive time.

The top-level bringup owns simulator/RViz lifecycle and supplies the external
config path. This package never starts or restarts either process.
