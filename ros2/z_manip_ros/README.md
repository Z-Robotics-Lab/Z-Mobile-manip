# z_manip_ros

ROS 2 Jazzy integration for the production perception path:

1. A versioned request on `/z_manip/grounding/request` publishes an ordered
   reliable/transient-local `arm` on `/track_3d/seed_request`. It carries the
   task request ID, bridge epoch, grounding generation, random nonce, and a
   source-time floor.
2. After exact RGB-D-camera-info admission, the adapter pins one JPEG and emits
   `/track_3d/exact_seed_image` plus `/track_3d/seed_offer_manifest`. The bridge
   accepts them in either DDS order only when token, nonce, adapter generation,
   timestamp, optical frame, and dimensions all agree. The contract deadline is
   advanced before offer admission and again before a ready VLM future may
   commit, so a late image or result never wins a timer scheduling race. The
   bridge then calls `z_manip.perception.vlm_affordance.OpenRouterVLM` exactly
   once.
3. The VLM target box echoes the complete offer token in `Detection2D.id` and
   initializes the in-tree adapter on `/track_3d/init_bbox`. A reset or failed
   future cannot publish a box for a newer request nonce.
4. The adapter calls the independent EdgeTAM HTTP service and owns persistent
   identity, exact-frame mask tracking, and RGB-D point generation.
5. The bridge only republishes controller-facing data while tracking status,
   detections, selected target ID, and selected point cloud are all fresh.

If request B replaces a still-running request A, cancellation remains queue
bounded. Once A drains, a B offer that has exceeded `max_camera_age_s` is not
used: the bridge republishes `arm` for the unchanged B task request ID,
instruction, scope, and grounding generation with a fresh nonce and source-time
floor. This obtains a new exact frame without accepting A's future or silently
starting a new task generation.

There is no per-frame VLM call and no color/depth heuristic tracker in this
path. A lost status, empty detection set, identity change, thin cloud, or stale
stream invalidates the contract and publishes zero `TwistStamped` commands on
the dedicated `/safety_cmd_vel` channel until a new grounding request succeeds.
The task runtime retains sole ownership of `/local_movement_cmd_vel`; a downstream
command arbiter must give the safety channel priority over task-local commands.

## Runtime dependencies

- ROS 2 Jazzy packages listed in `package.xml`
- `z_manip_edgetam` in the same colcon workspace (also declared as a package
  dependency)
- the independent EdgeTAM HTTP service and its model dependencies
- the repository root Python package, installed into the ROS environment so
  `z_manip.perception.vlm_affordance` is importable
- `OPENROUTER_API_KEY` exported into the launch process environment

The API key is not accepted as a ROS parameter and is never logged. Model IDs,
per-model timeouts, topics, depth scale, and cloud gates are ordinary YAML
parameters. Every provider attempt logs its model, attempt number, bounded
outcome, and elapsed time. This makes a primary timeout and fallback visible in
acceptance artifacts without exposing credentials. Structurally valid VLM
output is still rejected when avoid regions cover the proposed grasp part or
placement region.

```bash
python3 -m pip install -e /path/to/Z-Mobile-manip
colcon build --symlink-install --packages-select z_manip_edgetam z_manip_ros
source install/setup.bash
ros2 launch z_manip_ros perception.launch.py \
  use_sim_time:=true tracker_service_url:=http://127.0.0.1:8092
ros2 topic pub --once /z_manip/grounding/request std_msgs/msg/String \
  "{data: '{\"schema\":\"z_manip.grounding_request.v2\",\"request_id\":\"operator-demo-1\",\"instruction\":\"pick the red mug by its body, avoiding the handle\",\"scope\":\"grasp_only\"}'}"
```

Plain-text requests remain accepted only as a legacy bridge input and receive
a bridge-generated identity. The task runtime always uses the versioned
envelope and only authorizes geometry whose status and affordance carry its
exact `request_id`, grounding scope, `producer_epoch`, generation, source
stamp, and frame.

`tracker_config` selects the adapter YAML and `tracker_service_url` selects its
HTTP endpoint. The standalone launch intentionally starts
`z_manip_edgetam/edgetam.launch.py`; the former `track_anything/track_3d`
default cannot provide the versioned exact-frame `/track_3d/frame_manifest`
required by the bridge.

The adapter process is fail-closed single-owner: its first valid `arm` locks the
bridge `producer_epoch` until that adapter exits. Foreign-epoch arm and cancel
messages are ignored, including a delayed cancel from an old transient-local
writer. Supervisors must lifecycle the bridge and adapter together; restarting
only the bridge deliberately cannot take ownership of an existing adapter.

Set `start_edge_tam:=false` when the mobile task launch or another supervisor
already manages the `z_manip_edgetam` adapter. That externally managed adapter
must use the same `/track_3d/*` topics, including `exact_seed_image`,
`seed_request`, `seed_offer_manifest`, `frame_manifest`, and `failure`; the flag
does not relax the bridge contract.

Consumers must gate motion on `/z_manip/perception/valid` and subscribe to the
validated outputs rather than raw tracker topics:

- `/z_manip/perception/tracked_detections_2d`
- `/z_manip/perception/target_3d`
- `/z_manip/perception/target_pointcloud`
- `/z_manip/perception/affordance`
- `/z_manip/perception/status`

Send an `std_msgs/msg/Empty` to `/z_manip/grounding/reset` when a skill exits.
