# z_manip_edgetam

`z_manip_edgetam` adapts the standalone EdgeTAM HTTP service to the ROS 2
tracking contract consumed by `z_manip_ros`. It never reads simulator object
poses, semantic IDs, scene files, or other ground truth. The only 3-D source is
the current EdgeTAM mask combined with timestamp-aligned measured depth and the
camera calibration matrix.

## Data contract

Inputs:

- `/camera/color/image_raw` (`sensor_msgs/Image`)
- `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`)
- `/camera/color/camera_info` (`sensor_msgs/CameraInfo`)
- `/track_3d/seed_request` (`std_msgs/String`, reliable/transient-local
  `z_manip.seed_request.v1` arm/cancel transaction)
- `/track_3d/init_bbox` (`vision_msgs/Detection2DArray`)
- `/track_3d/reset` (`std_msgs/Empty`, legacy unconditional cancel only)

Outputs:

- `/track_3d/exact_seed_image` (`sensor_msgs/CompressedImage`, reliable and
  transient-local; its format field carries the full offer token)
- `/track_3d/seed_offer_manifest` (`std_msgs/String`, reliable and
  transient-local `z_manip.seed_offer.v1` causal identity)
- `/track_3d/seed_status` (`std_msgs/String`, non-terminal handshake diagnostics)
- `/track_3d/is_tracking` (`std_msgs/Bool`)
- `/track_3d/detections_2d` (`vision_msgs/Detection2DArray`)
- `/track_3d/selected_target_3d` (`vision_msgs/Detection3D`)
- `/track_3d/selected_target_pointcloud` (`sensor_msgs/PointCloud2`, fields
  `x,y,z,u,v` so VLM grasp/avoid regions remain aligned)
- `/z_manip/perception/target_mask` (`sensor_msgs/Image`, `mono8`, dominant
  depth-connected target cluster)
- `/z_manip/perception/overlay` (`sensor_msgs/Image`, RGB target/ID overlay)
- `/z_manip/perception/scene_pointcloud` (`sensor_msgs/PointCloud2`, measured
  scene with a configurable dilation of the tracked target removed; valid mask
  pixels rejected from target geometry are restored as collision obstacles)

All names are ROS parameters. RGB, aligned depth, and camera info must have the
same timestamp, image dimensions, and, by default, optical frame ID. Boxes use
the half-open convention `[x1, y1, x2, y2)` throughout. The published point
cloud and 3-D box carry the synchronized optical-frame header.

The target point cloud and 3-D box use the largest 8-connected component whose
neighbouring measured depths satisfy configurable absolute-or-relative jump
limits. This observation-only filter removes far shelf/wall leakage while
retaining exact `u,v` correspondence for every published target point.

Before any area, bbox, or IoU continuity gate, the service mask is split into
8-connected image components. A component overlapping the last validated mask
is selected first; deterministic largest-component fallback is used only when
there is no overlap, after which the unchanged continuity gates still apply. A
raw-area gate runs before cleanup, and cleanup is permitted only when both the
total rejected/raw fraction and largest-rejected/selected fraction satisfy
explicit limits. A comparable competing component is terminal rather than
silently changing identity. Published frame manifests retain these counts,
ratios, limits, and the deterministic selection mode.

The upstream VLM bridge returns a seed after inference latency but preserves the
header of the image that it grounded. This adapter keeps a bounded JPEG cache,
looks up that exact image, and initializes EdgeTAM on it. It then sends later
synchronized frames through one serialized worker. Service frame sequence and
ROS image time must both increase strictly.

An `arm` request carries the task request ID, bridge process epoch, grounding
generation, random request nonce, and a source-time floor. It atomically resets
the previous tracker generation and arms exactly one offer. Only the first
admitted exact RGB-D-camera-info tuple strictly newer than that floor can become
the offer. Its JPEG remains pinned independently of normal cache eviction, and
the image plus manifest bind request nonce, adapter generation, random offer
token, timestamp, optical frame, and dimensions. Duplicate arm delivery is
idempotent and republishes an outstanding offer without resetting it. The first
valid `arm` also locks this adapter process to that bridge `producer_epoch`.
Requests, including `cancel`, from every other epoch are diagnosed and ignored;
the owner changes only when the adapter process restarts. Production supervisors
must therefore start, stop, and restart the bridge and adapter as one pair. A
bridge-only rolling restart is intentionally unsupported rather than allowing an
old producer to cancel a newer task.

The adapter accepts an initialization box only when `Detection2D.id` echoes the
complete current offer token and both headers match the pinned tuple. Stale,
unmatched, duplicate, or malformed boxes are ignored and diagnosed; they never
stop a newer tracking session. Acceptance, ordered `cancel`, failure, legacy
reset, or the finite `seed_offer_timeout_s` watchdog releases the pin. The pin
deadline and watchdog timer use a process-local steady clock, so pausing or
freezing `/clock` cannot retain it indefinitely. The same deadline is checked
immediately before and after latest-frame registration; a bbox cannot commit by
beating a delayed watchdog callback or by crossing the deadline during
registration.

Delayed VLM grounding never sparsely replays a long RGB backlog. Instead, the
seed and latest exact-sync JPEGs are registered before EdgeTAM initialization.
Bidirectional LK tracks independently fit a global background transform and a
seed-ROI transform with RANSAC. Feature counts, inlier ratios, residuals,
rotation, scale, translation, and global/ROI motion agreement are all bounded.
Only then is the seed box transformed and clamped onto the latest cached frame;
the original seed identity and timestamp remain the cross-node authority. A
low-texture image, scene change, target motion relative to the background, bad
JPEG, or unsafe transform fails with `seed_reseed_registration`.

## Failure behavior

The adapter publishes `is_tracking=false` before clearing its session when any
of these conditions occurs:

- the current matching seed fails bounded latest-frame registration or tracker
  initialization;
- RGB, depth, and camera info stop synchronizing exactly;
- image stamps duplicate or move backwards;
- the HTTP service times out, changes session/track identity, returns an empty
  mask, changes image dimensions, or violates the protocol;
- valid target depth falls below `min_cloud_points`;
- serialized inference stops producing fresh results before the result watchdog
  expires. Both acquisition and tracking queues replace older pending work with
  the freshest RGB-D frame; the default tracking window is latest-only.

`is_tracking` uses reliable transient-local QoS. A downstream controller that
starts after a failure therefore observes the latest false state instead of
mistaking a previously cached target or transform for an active track.

A single mask whose cleaned component is a strict subset collapse is handled as
a bounded pending anomaly, not a new anchor. The adapter advances the strict
service sequence but publishes nothing and retains the previous validated mask
and 3-D centroid. It accepts again only after two consecutive source frames pass
the original continuity thresholds. Failure to recover within those two new
frames is terminal. Session, track, sequence, timestamp, and image-size errors
remain immediately terminal even while an anomaly is pending.

For a camera mounted on a moving base, an already depth-validated track may
advance its mask anchor across one non-overlapping image-plane jump. This mode
is opt-in and requires bounded mask area and displacement, sufficient live
depth retention, and a continuous 3-D centroid. It never applies to the first
live update or RGB-only replay. Other discontinuities still require a fresh VLM
seed. The standalone client also clears its local service session on protocol
and transport failures.

## Build and run

Install the root Python package and ROS dependencies, then build this package:

```bash
python3 -m pip install -e /path/to/Z-Mobile-manip
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2/z_manip_edgetam --symlink-install
source install/setup.bash
ros2 launch z_manip_edgetam edgetam.launch.py \
  service_url:=http://127.0.0.1:8092 use_sim_time:=true
```

The default configuration is in `config/edgetam.yaml`. Start the independent
`docker/edgetam_service` deployment before requesting VLM grounding. No API key
is handled by this package. The service loads its model lazily; production
deployments should warm the service before enabling robot motion or increase
`service_timeout_s` for the first request.

## Test without ROS or GPU

The sequencing, identity, bbox, depth projection, and fail-closed behavior live
in `z_manip_edgetam.core.FailClosedTracker`. Its service client is injected, so
the package tests use a deterministic fake and do not import ROS, start a model,
or access a GPU:

```bash
python3 -m pytest -q ros2/z_manip_edgetam/test/test_core.py
```
