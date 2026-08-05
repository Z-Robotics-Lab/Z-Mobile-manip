# Task-FSM seam E2E — 2026-07-29 (static, zero-motion)

First live bring-up of the `z_manip_task` runtime on the 4090 workstation, and
the recorded request → phase-stream → cancel seam evidence for the z-agent
bridge contract. Static: no camera, no odometry, no arm, no base — the run
proves the seam and the fail-closed guards, not a pick.

## Zero-motion premise (verified before launch)

The full launch includes `z_manip_coarse_navigation`, and the task runtime
itself publishes velocity to the platform `local_velocity` topic
(`/local_movement_cmd_vel` in `configs/go2w_piper.json`), so "motion-inert"
must be proven, not assumed. Evidence, in probe order:

1. **Domain 20 was NOT safe at probe time (20:5x):** the NUC base chain was
   live — `z-mobile-manip-go2w-reactive-live.service` active and the `zdog`
   nav unit running `system_real_robot.launch.py … control_mode:=sport_cmd`,
   with `unitree_control` subscribed to `/cmd_vel_safe` on domain 20. A
   domain-20 FSM launch was therefore ruled out for this session. (Later the
   NUC dropped off Wi-Fi entirely — `no route to host`, its DDS participants
   gone — but "unreachable" cannot certify "chain stopped", so the ruling
   stood.) Domain 20 is additionally blocked by design: the lab perception
   containers already own critical producers (`vlm_edgetam_bridge`,
   `z_manip_edgetam`), so the singleton supervisor's empty-graph preflight
   refuses to start there — fail closed, by intent.
2. **The E2E ran on isolated `ROS_DOMAIN_ID=91`:** `ros2 node list` on domain
   91 was empty before launch (zero participants), and during the run
   `/local_movement_cmd_vel` had `Subscription count: 0` while `/cmd_vel` /
   `/cmd_vel_safe` did not exist at all on that domain. No chassis consumer
   can exist there: motion was impossible by construction.

## Bring-up path (the shipped one, not a bespoke harness)

```bash
ROS_DOMAIN_ID=91 manip fsm start     # scripts/runtime/go2w_task_fsm.sh start
```

The manager started the `z-manip-task` container running the singleton
supervisor, which passed its empty-graph preflight, launched
`mobile_manipulation.launch.py`, and reported:

```text
[z-manip-supervisor] uniquely ready: ROS_DOMAIN_ID=91 namespace=/ pgid=35
```

Node set at readiness: `move_group`, `vlm_edgetam_bridge`, `z_manip_edgetam`,
`z_manip_coarse_navigation`, `z_manip_complete_joint_state`,
`z_manip_observed_placement`, `z_manip_robot_state_publisher`,
`z_manip_task_runtime`, `z_manip_urdf_root_alias`.
`manip status task` → `healthy  singleton task runtime ready;
/z_manip/task/status has a publisher`.

## QoS evidence (z-agent bridge contract)

`ros2 topic info -v /z_manip/task/status`:

```text
Publisher count: 1     Reliability: RELIABLE  History: KEEP_LAST (1)  Durability: TRANSIENT_LOCAL
Subscription count: 1  Reliability: RELIABLE  History: KEEP_LAST (1)  Durability: TRANSIENT_LOCAL
```

Matches `node.py` `_setup_io` (`latched_debug`) and the z-agent bridge's
subscriber QoS. A late-joining `ros2 topic echo` received the latched `idle`
status document immediately (one message before any request) — the latch works.

## Recorded phase stream

Driven with `ros2 topic pub --once` on `/z_manip/task/request`
(`std_msgs/String`) and `/z_manip/task/cancel` (`std_msgs/Bool`), recorded by a
continuous `ros2 topic echo` of `/z_manip/task/status`:

```text
phase=idle         instruction=""                          (latched pre-task)
phase=pose_settle  instruction="approach the black case"   (request #1 accepted)
phase=failed       instruction="approach the black case"   (fail-closed, see below)
phase=canceled     instruction="approach the black case"   (cancel acknowledged)
phase=pose_settle  instruction="approach the mug"          (request #2 accepted — clean recovery)
phase=failed       instruction="approach the mug"
phase=canceled     instruction="approach the mug"
```

The `failed` transitions are the posture guard doing its job with no state
estimation on the isolated domain:

```text
[ERROR] [z_manip_task_runtime]: posture safety violation:
        state-estimation posture unavailable after acquisition timeout
```

That is the honest static-seam ceiling: with odometry, joint states, and
perception present (a domain-20 window), the same request proceeds
`pose_settle → grounding/visual_search → standoff → coarse_nav → …` instead.
What this run proves: request parsing and acceptance, the latched status
stream, per-instruction status attribution, cancel from any state, and that a
fresh task is accepted cleanly after a cancel (no stuck state, no restart
needed).

## What was left running

* `z-manip-task` (FSM, domain 91) — left up and healthy; zero risk on the
  isolated domain, and `manip status task` / `manip logs` work against it.
* Planning-workbench UI — `http://127.0.0.1:8766/` healthy
  (`manip start ui`).
* Nothing else was started; no existing container or NUC service was touched.

For a domain-20 live window (operator + E-stop): stop the lab perception
containers OR accept the supervisor's refusal, confirm the base chain state
explicitly, then `manip down && ROS_DOMAIN_ID=20 manip fsm start`. The
supervisor's preflight makes a double-ownership start impossible either way.
