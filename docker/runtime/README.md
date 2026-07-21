# ROS 2 Jazzy manipulation runtime

This image contains the lightweight `z_manip` Python core, the
`z_manip_ros` perception bridge, the synchronized `z_manip_edgetam` RGB-D
gateway, the fail-closed `z_manip_task` executor, and the `z_manip_motion`
MoveIt bridge, plus observed-target coarse navigation and RGB-D placement
planning. It
installs the ROS Jazzy binary packages for `vision_msgs`, MoveIt KDL and
TRAC-IK kinematics, OMPL, `move_group`, and 3-D occupancy monitoring.

It deliberately does **not** contain API keys, `.env` files, robot assets,
EdgeTAM/model weights, `track_anything`, reference repositories, logs, or
simulation evidence. Supply secrets and the external read-only robot asset
tree only when running a container. EdgeTAM can run as the separate service
image; launch perception with `start_edge_tam:=false` in that topology.

## Build and verify

From the repository root:

```bash
docker build \
  --file docker/runtime/Dockerfile \
  --tag z-manip-runtime:jazzy \
  .

docker run --rm --network host \
  --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  z-manip-runtime:jazzy \
  z-manip-runtime-smoke --node-probe
```

The default base is the current local `navstack:ready` pinned to
`sha256:7c1f317d0bfe33fbdb5067bf0266fc70c273da23bcbd767ebf3c714405118407`.
An official Jazzy ros-base image can be selected explicitly:

```bash
docker build \
  --build-arg BASE_IMAGE=ros:jazzy-ros-base-noble \
  --file docker/runtime/Dockerfile \
  --tag z-manip-runtime:jazzy-ros-base \
  .
```

The Dockerfile-specific ignore file allow-lists only `z_manip/`, the six ROS
packages, and `docker/runtime/`. Docker BuildKit therefore does not send local
secrets, `refs/`, logs, or evidence into the build context.

## Run

Start a reusable host-DDS container:

```bash
export Z_MANIP_ENV_FILE=/absolute/private/z-agent.env
export Z_MANIP_ROBOT_ASSETS=/absolute/path/to/go2W_Sim/assets
docker compose --file docker/runtime/compose.yaml up --detach --build
docker compose --file docker/runtime/compose.yaml exec manipulation-runtime \
  z-manip-runtime-smoke --node-probe
```

The compose profile defaults to the Office simulation's clean DDS domain
(`Z_MANIP_ROS_DOMAIN_ID=184`). Simulation and real deployments use the same
image and ROS contracts; every host participating in one deployment must use
the same domain. Fast DDS is explicitly restricted to UDPv4 transport, not to
localhost, so separately isolated host-network containers share the same graph.
`Z_MANIP_ENV_FILE` is optional at Compose parsing time, but a complete VLM run
must point it at a private file containing `OPENROUTER_API_KEY`; the file is
injected only into the container environment and is never copied into the
image. `Z_MANIP_ROBOT_ASSETS` must name the complete asset directory because
MoveIt validates every mesh referenced relative to the URDF.

Run the complete stack through its singleton supervisor. The lock is scoped by
`ROS_DOMAIN_ID` and ROS namespace, is inherited by the launch root, and is held
until every process in the supervisor-owned launch group exits. A second
bringup fails closed; it never terminates nodes owned by another process.

```bash
docker compose --file docker/runtime/compose.yaml exec manipulation-runtime \
  bash -lc '
    source /opt/ros/jazzy/setup.bash
    source /opt/z_manip_ws/install/setup.bash
    exec z-manip-mobile-manipulation \
      --namespace / \
      -- \
      use_sim_time:=true \
      robot_description_file:=/robot/assets/urdf/go2w_sensored.urdf \
      stack_config_path:=/opt/z_manip/configs/go2w_piper.json \
      collision_model_file:=/opt/z_manip/configs/piper_collision_capsules.json \
      task_platform_parameters:=/opt/z_manip_ws/install/share/z_manip_task/config/go2w_sim.yaml
  '
```

Use `use_sim_time:=false` with the same entry point on the robot. `SIGINT`,
`SIGTERM`, and `SIGHUP` are forwarded as a bounded teardown of only the launch
process group created by this supervisor. Before launch, critical producer
topics and nodes must be absent; after launch, each must have exactly one
publisher/owner. Omit `task_platform_parameters` on real hardware until its arm
adapter supplies a separately validated named-pose profile.

Run the perception contract without trying to start an in-container tracker.
The private environment file is read at container start, never copied into the
image:

```bash
docker run --rm --network host \
  --env-file /absolute/private/z-agent.env \
  --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  z-manip-runtime:jazzy \
  ros2 launch z_manip_ros perception.launch.py \
    use_sim_time:=true start_edge_tam:=false
```

Connect synchronized RGB-D to the independent EdgeTAM HTTP service:

```bash
docker run --rm --network host \
  --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  z-manip-runtime:jazzy \
  ros2 launch z_manip_edgetam edgetam.launch.py \
    service_url:=http://127.0.0.1:8092 use_sim_time:=true
```

Run collision-aware MoveIt planning with the external robot asset tree mounted
read-only. Mounting only the URDF is insufficient when it contains relative
mesh references:

```bash
docker run --rm --network host \
  --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  --volume /absolute/path/to/go2W_Sim/assets:/robot/assets:ro \
  --env Z_MANIP_ROBOT_DESCRIPTION_FILE=/robot/assets/urdf/go2w_sensored.urdf \
  z-manip-runtime:jazzy \
  ros2 launch z_manip_motion moveit_planning.launch.py \
    use_sim_time:=true \
    point_cloud_topic:=/camera/depth/color/points \
    octomap_frame:=base
```

Before `move_group` starts, every URDF mesh is resolved relative to the URDF,
checked, and converted to a canonical local `file://` URI. Missing assets fail
the launch instead of silently removing collision geometry.

The installed Debian package versions are recorded inside the image at
`/opt/z_manip/runtime-package-manifest.txt`. The external `track_anything`
package/model and the robot URDF are required only for their corresponding
full launches; the image build and node smoke do not assume either resource.
