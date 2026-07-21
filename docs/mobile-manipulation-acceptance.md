# Mobile-manipulation acceptance

`z-manip-mobile-manipulation-acceptance` owns one complete acceptance run. It
starts the stack through the singleton supervisor, waits for one runtime and
one upstream publisher for every critical interface, starts an MCAP recorder,
and waits until both the task runtime and recorder have matched the task
publisher. It then publishes the natural-language request exactly once.

The default simulation invocation is:

```bash
source /opt/ros/jazzy/setup.bash
source /opt/z_manip_ws/install/setup.bash
z-manip-mobile-manipulation-acceptance \
  --ros-domain-id 184 \
  --robot-description-file /robot/assets/urdf/go2w_sensored.urdf \
  --stack-config-path /opt/z_manip/configs/go2w_piper.json \
  --collision-model-file /opt/z_manip/configs/piper_collision_capsules.json \
  --task-platform-parameters \
    /opt/z_manip_ws/install/share/z_manip_task/config/go2w_sim.yaml \
  --output-root /tmp
```

The runner sets `rmw_fastrtps_cpp` and UDPv4 for the entire owned process tree.
It does not start or stop Isaac Sim, RViz, or the navigation stack. Those
producers must already be healthy on domain 184. The same runner can target a
real deployment with `--use-sim-time false` and deployment-specific robot,
stack, and collision-model paths; its observation contract is unchanged. The
real invocation must omit `--task-platform-parameters` until its arm adapter
defines and verifies an equivalent profile. The generic task runtime never
assumes the simulation-only `MANIP_LOOKOUT` named pose. The
same external collision model is passed to grasp, carry, and observed-place
planning so simulation and hardware do not silently use different payload
clearance geometry.

The singleton supervisor uses one persistent `rclpy` graph observer for both
preflight and readiness. Its DDS discovery cache therefore survives every
snapshot. Before launch, the graph must remain free of every critical node and
publisher for the bounded preflight observation window. After launch, every
critical count must remain exactly one for the readiness stability window;
partial or empty snapshots reset that window, and any duplicate fails
immediately. This avoids the Fast DDS false negatives caused by starting a new
short-lived `ros2 --no-daemon` participant for every query.

Each run creates `zmanip-e2e-<UTC>/` containing:

- `bag/`: the closed MCAP and `metadata.yaml`;
- `events.jsonl`: flushed task, placement, and execution evidence;
- `supervisor.log` and `recorder.log`;
- `verdict.json`: the machine-readable final result.

A pass requires all of the following, not merely a terminal success string:

1. one task publication and task status `phase=complete` with
   `result=mobile_manip_complete`;
2. ordered `planning`, `transit`, `pregrasp_reobserve`,
   `approach_planning`, `approach`, `closing`, `lift`, `verify`, and `carry`
   task phases, proving that the two-stage pregrasp/reobservation/approach
   chain ran before placement;
3. an observed-place planner `planned` event correlated to the terminal
   `place_goal_id`;
4. ordered `place_transit`, `place_approach`, `releasing`, `place_retreat`, and
   `complete` task phases;
5. the same placement-approach command observed first `active` and then
   `succeeded` under one goal and executor epoch;
6. executor-measured stable open-gripper feedback from one command and source
   stamp, followed by a distinct later retreat command observed first `active`
   and then `succeeded`;
7. an independent `/z_manip/place/post_release_verification` message with
   schema `z_manip.post_release_verification.v2`, correlated place goal,
   release-command, perception request, producer epoch, generation, frame, and
   planning-observation identities, synchronized RGB-D point-cloud provenance,
   and multi-frame evidence that the target is stable in the requested region,
   upright as requested, registered to its grasp-time geometry, and spatially
   separated from the gripper;
8. a cleanly closed, non-empty MCAP.

The machine-readable `pick_two_stage_phase_order_observed` predicate is
included in `checks`, the evidence summary, and the top-level verdict. It is a
mandatory acceptance predicate, not informational metadata. Its matcher is
attempt-local: every new `planning` phase discards partial pick progress, and a
complete `planning` through `carry` sequence is accepted only when it precedes
the first `place_transit` observed in the global phase history.

Task status schema v1 identifies the natural-language instruction but does not
carry a run ID. Cross-run isolation is therefore an external acceptance fence:
the singleton-supervisor preflight requires a bounded empty critical graph
before it launches a fresh task runtime, then readiness requires exactly one
task runtime and status publisher. This rejects a pre-existing runtime that
could replay status for the same instruction; the attempt-local matcher above
separately prevents bounded retries within the owned run from being stitched
into one successful pick.

Planner success, gripper opening, and retreat success constitute
`place_execution_evidence`; they do **not** constitute observed placement. The
overall verdict remains failed until the independent post-release geometry
contract is present and valid. This prevents a successful motion program from
masking an object that remained in the gripper, fell outside the requested
region, or continued moving.

The executor evidence is a strict receipt-ordered transaction, not a set of
status flags. Its invariant is:

```text
approach active < approach succeeded <= stable release samples
                < retreat active < retreat succeeded <= observation start
```

Approach and retreat must retain immutable command IDs and source stamps from
their `active` through `succeeded` states. The open-gripper evidence uses the
independent `gripper_received_at` field; a later trajectory status cannot
replace that source identity. Commands, source stamps, goal IDs, or executor
epochs cannot be mixed across samples. Duplicate JSON keys or execution fields
are rejected rather than resolved by first- or last-value precedence. The task
terminal status must repeat the exact v2 verification expectation and report it
verified before the independent evidence can satisfy acceptance.

`trajectory_received_at` and `gripper_received_at` are protocol source times in
the ROS clock domain used by post-release verification. Isaac executors derive
them from simulation ROS time. A real executor must derive them from ROS time,
not a device boot, monotonic, or unsynchronized wall clock; otherwise the
cross-component event ordering fails closed.

No object odometry, simulator prim, ground-truth pose, fixed image box, or
hard-coded task pose is consumed by the runner.
