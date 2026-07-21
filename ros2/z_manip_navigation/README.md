# z_manip_navigation

This package owns the coarse-navigation stage before `z_manip_task` visual
servo. It consumes only SLAM odometry, TF, the validated persistent target, and
the reachability-derived standoff displacement in `z_manip.task_status.v1`.
It does not consume simulator or robot pose oracles.

## Flow

1. A task status enters `coarse_nav` with a prospective observation serial and
   a `prospective_base_displacement_m` selected by downstream IK/collision
   feasibility.
2. The target center is transformed from its camera frame into `map` using TF
   at the observation timestamp. The node combines the observed map ray, SLAM
   base position, and bounded suggested displacement to publish a
   `geometry_msgs/PointStamped` on `/way_point`.
3. `/goal_reached`, SLAM health, target freshness, and fresh-stamped odometry
   are monitored. Goal progress uses the robust endpoint decrease and fitted
   distance slope over one fixed `stall_timeout_s` window, so slow continuous
   approach is distinct from circling, retreat, or a cached pose. A stale/lost
   target immediately publishes
   `/cancel_goal=true`; bounded reacquisition republishes the task grounding
   request, and bounded replanning uses the newest observation.
4. When the observed optical target depth is within `near_target_depth_m`, the
   node cancels navigation. Only after measured base speed remains below the
   configured threshold for the full settle interval does it publish
   `/z_manip/navigation/coarse_ready=true`.

`near_target_depth_m` must match the deployment's camera field of view and
visual-servo capture envelope. The live office deployment may use `1.7`; the
portable default remains `1.4`.

## Topics

Inputs:

- `/z_manip/task/status` (`std_msgs/String`, JSON)
- `/z_manip/perception/valid` (`std_msgs/Bool`)
- `/z_manip/perception/target_3d` (`vision_msgs/Detection3D`)
- `/state_estimation` (`nav_msgs/Odometry`)
- `/state_estimation_health` and `/goal_reached` (`std_msgs/Bool`)

Outputs:

- `/way_point` (`geometry_msgs/PointStamped`, `map` frame)
- `/cancel_goal` and `/z_manip/navigation/coarse_ready` (`std_msgs/Bool`)
- `/z_manip/grounding/request` (`std_msgs/String`) for bounded reacquisition
- `/z_manip/navigation/status` (`std_msgs/String`, JSON diagnostics)

All topic names and policy thresholds are ROS parameters.

## Launch

```bash
ros2 launch z_manip_navigation coarse_navigation.launch.py
```

The top-level bringup owns Isaac Sim and RViz lifecycle. This package never
starts or restarts either process.
