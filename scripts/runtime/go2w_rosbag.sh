#!/usr/bin/env bash
set -euo pipefail

# Read-only evidence recorder for mobile-manipulation tuning.  This script
# starts only rosbag2 subscriptions; it never publishes a motion command.
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$STACK_ROOT/.." && pwd)"

DOCKER="${Z_MANIP_DOCKER_BIN:-docker}"
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:jazzy}"
CONTAINER="${Z_MANIP_ROSBAG_CONTAINER:-z-mobile-manip-rosbag}"
BAG_ROOT="${Z_MANIP_ROSBAG_ROOT:-$WORKSPACE_ROOT/artifacts/go2w_real/rosbags}"
QOS_CONFIG="${Z_MANIP_ROSBAG_QOS:-$STACK_ROOT/configs/rosbag_sensor_qos.yaml}"
CYCLONEDDS_CONFIG="${Z_MANIP_CYCLONEDDS_CONFIG:-$STACK_ROOT/docker/runtime/cyclonedds-go2w-pc.xml}"
DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
SPLIT_SECONDS="${Z_MANIP_ROSBAG_SPLIT_SECONDS:-300}"
STATE_FILE="$BAG_ROOT/.active-z-mobile-manip-rosbag"

TOPICS=(
  /camera/color/image_raw
  /camera/color/camera_info
  /camera/aligned_depth_to_color/image_raw
  /nuc/camera/color/image_raw/compressed
  /nuc/camera/color/camera_info
  /nuc/camera/aligned_depth_to_color/image_raw/compressedDepth
  /tf
  /tf_static
  /piper/state
  /piper/joint_trajectory
  /piper/gripper_aperture
  /piper/trajectory_status
  /piper/execution_status
  /piper/cancel
  /piper/named_pose
  /track_3d/detections_2d
  /track_3d/exact_seed_image
  /track_3d/failure
  /track_3d/frame_manifest
  /track_3d/init_bbox
  /track_3d/is_tracking
  /track_3d/reset
  /track_3d/seed_offer_manifest
  /track_3d/seed_request
  /track_3d/seed_status
  /track_3d/selected_target_3d
  /track_3d/selected_target_pointcloud
  /z_manip/grounding/request
  /z_manip/grounding/reset
  /z_manip/perception/affordance
  /z_manip/perception/overlay
  /z_manip/perception/scene_pointcloud
  /z_manip/perception/status
  /z_manip/perception/target_3d
  /z_manip/perception/target_mask
  /z_manip/perception/target_pointcloud
  /z_manip/perception/tracked_detections_2d
  /z_manip/perception/valid
  /z_manip/visual_search/active
  /z_manip/coarse_nav/perception_loss_authorization
  /z_manip/reactive/posture_intent
  /z_manip/reactive/posture_status
  /z_manip/reactive/arm_view_intent
  /z_manip/reactive/arm_view_status
  /z_manip/reactive/control_reset
  /z_manip/reactive/full_stop
  /go2w/posture_cmd
  /go2w/posture_state
  /go2w/control_reset
  /go2w/full_stop
  /navigation_cmd_vel
  /local_movement_cmd_vel
  /odom_base_link
  /goal_reached
  /cancel_goal
  /way_point
)

usage() {
  cat <<'EOF'
Usage:
  go2w_rosbag.sh start [LABEL]  Start a compressed, five-minute-split tuning bag
  go2w_rosbag.sh stop           Stop cleanly and finalize bag metadata
  go2w_rosbag.sh status         Show recorder and disk status
EOF
}

container_running() {
  [[ "$($DOCKER inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" == true ]]
}

active_bag_name() {
  [[ -f "$STATE_FILE" ]] && sed -n '1p' "$STATE_FILE"
}

start_recording() {
  local label="${1:-tuning}" timestamp bag_name commit container_id
  [[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$ ]] || {
    printf 'Invalid label; use 1-48 letters, digits, dot, underscore, or dash.\n' >&2
    return 2
  }
  [[ -r "$QOS_CONFIG" ]] || { printf 'Missing QoS config: %s\n' "$QOS_CONFIG" >&2; return 1; }
  [[ -r "$CYCLONEDDS_CONFIG" ]] || { printf 'Missing CycloneDDS config: %s\n' "$CYCLONEDDS_CONFIG" >&2; return 1; }
  if container_running; then
    printf 'Rosbag is already recording: %s\n' "$(active_bag_name || printf unknown)"
    return 0
  fi
  $DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true
  mkdir -p "$BAG_ROOT"
  timestamp="$(date +%Y%m%d-%H%M%S)"
  bag_name="mobile-tuning-${timestamp}-${label}"
  commit="$(git -C "$STACK_ROOT" rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"

  container_id="$($DOCKER run -d --rm \
    --name "$CONTAINER" --init --network host --stop-signal SIGINT \
    --label z-manip.role=rosbag-recorder \
    -e ROS_DOMAIN_ID="$DOMAIN_ID" \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///config/cyclonedds.xml \
    -e ROS_LOG_DIR=/tmp/ros-log \
    -v "$CYCLONEDDS_CONFIG:/config/cyclonedds.xml:ro" \
    -v "$QOS_CONFIG:/config/rosbag-qos.yaml:ro" \
    -v "$BAG_ROOT:/bags" \
    "$IMAGE" /bin/bash -lc \
    'source /opt/ros/jazzy/setup.bash && exec ros2 bag record "$@"' _ \
      --storage mcap \
      --storage-preset-profile zstd_fast \
      --max-bag-duration "$SPLIT_SECONDS" \
      --qos-profile-overrides-path /config/rosbag-qos.yaml \
      --disable-keyboard-controls \
      --custom-data "project=z-mobile-manip" \
      --custom-data "git_commit=$commit" \
      --custom-data "label=$label" \
      --output "/bags/$bag_name" \
      --topics "${TOPICS[@]}")"

  printf '%s\n%s\n%s\n' "$bag_name" "$container_id" "$(date --iso-8601=seconds)" >"$STATE_FILE"
  local deadline=$((SECONDS + 15))
  while ((SECONDS < deadline)); do
    if ! container_running; then
      printf 'Rosbag recorder exited during startup:\n' >&2
      $DOCKER logs "$CONTAINER" >&2 || true
      return 1
    fi
    if [[ -d "$BAG_ROOT/$bag_name" ]]; then
      printf 'Recording read-only tuning bag: %s\n' "$BAG_ROOT/$bag_name"
      printf 'MCAP zstd_fast; split every %ss; ROS_DOMAIN_ID=%s; %s topics requested.\n' \
        "$SPLIT_SECONDS" "$DOMAIN_ID" "${#TOPICS[@]}"
      return 0
    fi
    sleep 0.25
  done
  printf 'Recorder is running but its output directory was not created in time.\n' >&2
  return 1
}

stop_recording() {
  local bag_name deadline
  bag_name="$(active_bag_name || true)"
  if ! container_running; then
    printf 'Rosbag recorder is not running.%s\n' "${bag_name:+ Last bag: $BAG_ROOT/$bag_name}"
    return 0
  fi
  # SIGINT lets rosbag2 close the MCAP and write metadata.yaml cleanly.
  $DOCKER kill --signal=SIGINT "$CONTAINER" >/dev/null
  deadline=$((SECONDS + 30))
  while container_running && ((SECONDS < deadline)); do sleep 0.25; done
  if container_running; then
    printf 'Recorder did not finalize within 30s; requesting a second graceful stop.\n' >&2
    $DOCKER stop --time 15 "$CONTAINER" >/dev/null
  fi
  rm -f "$STATE_FILE"
  printf 'Rosbag finalized: %s\n' "$BAG_ROOT/${bag_name:-unknown}"
}

show_status() {
  local bag_name=""
  bag_name="$(active_bag_name || true)"
  if container_running; then
    printf 'rosbag  recording  %s\n' "$BAG_ROOT/${bag_name:-unknown}"
    $DOCKER ps --filter "name=^/${CONTAINER}$" --format 'container {{.Status}}'
  else
    printf 'rosbag  stopped%s\n' "${bag_name:+  last=$BAG_ROOT/$bag_name}"
  fi
  df -h "$BAG_ROOT" | awk 'NR==2 {printf "disk    %s available of %s (%s used)\n", $4, $2, $5}'
}

command="${1:-status}"
case "$command" in
  start)
    [[ $# -le 2 ]] || { usage >&2; exit 2; }
    start_recording "${2:-tuning}"
    ;;
  stop)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    stop_recording
    ;;
  status)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    show_status
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
