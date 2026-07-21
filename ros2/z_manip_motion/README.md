# z_manip_motion

ROS 2 Jazzy plan-only MoveIt 2 wiring for the six revolute joints of PiPER.
The bridge never calls a trajectory controller action. It sends the latest
measured `/piper/state` and a joint or pose goal to `GetMotionPlan`, audits the
successful arm-only result, then publishes it to `/piper/joint_trajectory`.
Failures, timeouts, stale state, empty paths, extra joints, or malformed paths
publish nothing.

The host must provide MoveIt 2 and an external combined robot URDF. Every mesh
reference is resolved and checked before `move_group` starts. Relative mesh
paths are resolved against the URDF's directory and converted to canonical
`file://` URIs; missing or unsupported resources stop launch. A container must
therefore mount the whole referenced asset tree, not only the URDF file. No
object pose or simulation-specific interface is part of this package.

The point-cloud and depth occupancy updaters are provided by
`moveit_ros_perception`; `moveit_ros_occupancy_map_monitor` alone only provides
the monitor interface and cannot load either sensor plugin.

MoveIt receives `/z_manip/motion/complete_joint_states`, assembled from the
arm and mobile-platform proprioception topics. The assembler derives every
independent revolute, continuous, and prismatic joint from the supplied URDF
and publishes only while all of them are finite and fresh. It never fills a
missing leg or wheel joint with a nominal value. Set `platform_state_topic` to
the robot driver's complete platform `sensor_msgs/JointState` stream; a driver
that already publishes the whole robot can use the same topic for both inputs.
The output copies the oldest upstream ROS measurement stamp included in each
coherent snapshot; it is never fabricated from the assembler node's `now()`.
This prevents slower joints from appearing newer than their source data.
`state_max_stamp_skew_s` bounds the allowed measurement-time spread. In
simulation, only an explicit `/clock` rollback starts a new epoch: cached state
is cleared while the long-lived readers reject state. Acceptance resumes only
after `/clock` has one stable publisher, advances without rollback for the
configured `clock_handover_quiet_s`, and supplies one additional advancing
sample. Samples outside the current clock window are rejected. Without
`/clock`, a source stamp regression is rejected rather than guessed to be a
global reset.

The same complete state drives a stack-owned `robot_state_publisher`, so the
planning-scene self-filter has transforms for every collision link. Disable it
with `start_robot_state_publisher:=false` only when an external publisher owns
the identical combined URDF tree and joint-state contract; two publishers for
the same child frames are invalid.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select z_manip_motion
source install/setup.bash
ros2 launch z_manip_motion moveit_planning.launch.py \
  robot_description_file:=/absolute/path/to/go2w_sensored.urdf
```

TRAC-IK is the default solver. Use `kinematics_file:=.../kinematics.yaml` only
as an explicit KDL fallback. Select `sensors_file:=.../sensors_3d_depth.yaml`
to use a registered depth image instead of a point cloud. Named sensor profiles
are converted to the flat parameter contract required by MoveIt 2 on Jazzy;
launch topic and range arguments override the selected profile. Every external
path, frame, sensor topic, goal topic, state topic, service, and output topic is
a launch argument.
