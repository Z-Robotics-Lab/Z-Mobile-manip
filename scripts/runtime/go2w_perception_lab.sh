#!/usr/bin/env bash
set -euo pipefail

# Perception-only real-hardware bring-up.  This script never starts the task,
# MoveIt, arm, gripper, navigation, or base-control runtimes.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep bring-up on the same verified image used by interactive perception and
# offline Pinocchio planning.  Falling back to the older :jazzy image after a
# reboot silently removes the resident local grounding path and adds a remote
# VLM round trip to every request.
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:pinocchio}"
DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
if [[ -n "${Z_MANIP_ENV_FILE:-}" ]]; then
  ENV_FILE="$Z_MANIP_ENV_FILE"
elif [[ -f "$ROOT_DIR/.env" ]]; then
  ENV_FILE="$ROOT_DIR/.env"
else
  # Backward-compatible fallback for the existing lab deployment.
  ENV_FILE="$ROOT_DIR/../z-agent/.env"
fi
DDS_CONFIG="${Z_MANIP_DDS_CONFIG:-$ROOT_DIR/docker/runtime/cyclonedds-go2w-pc.xml}"
EDGETAM_CACHE="${Z_MANIP_EDGETAM_CACHE:-$HOME/.cache/z-manip/huggingface}"
EDGETAM_MAX_FRAMES_PER_SESSION="${Z_MANIP_EDGETAM_MAX_FRAMES_PER_SESSION:-18000}"
EDGETAM_STREAM_HISTORY_FRAMES="${Z_MANIP_EDGETAM_STREAM_HISTORY_FRAMES:-32}"
EDGETAM_MIN_SCORE="${Z_MANIP_EDGETAM_MIN_SCORE:-0.20}"
ANYGRASP_DIR="${ANYGRASP_DIR:-$HOME/anygrasp}"
ANYGRASP_SERVER_IMAGE="${ANYGRASP_SERVER_IMAGE:-z-manip-anygrasp:cu128}"
ANYGRASP_LICENSE_DIR="${ANYGRASP_LICENSE_DIR:-$ANYGRASP_DIR/license_YusenXie}"
ANYGRASP_CHECKPOINT="${ANYGRASP_CHECKPOINT:-$ANYGRASP_DIR/checkpoint_detection.tar}"
ARTIFACT_DIR="${Z_MANIP_ARTIFACT_DIR:-$ROOT_DIR/../artifacts/go2w_real/latest}"
PERCEPTION_RUNNER_ARTIFACT_ROOT="$ROOT_DIR/../artifacts"
PERCEPTION_RUNNER="z-manip-perception-runner"
PLANNING_RUNNER="z-manip-planning-runner"
PLANNING_RUNNER_SCRATCH_ROOT="$PERCEPTION_RUNNER_ARTIFACT_ROOT/go2w_real/.planning_runner_scratch"
SOAK_MAX_RECOVERIES="${Z_MANIP_SOAK_MAX_RECOVERIES:-2}"
SOAK_RECOVERY_TIMEOUT="${Z_MANIP_SOAK_RECOVERY_TIMEOUT:-25}"

docker_env=(
  --network host
  -e "ROS_DOMAIN_ID=$DOMAIN_ID"
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  -e CYCLONEDDS_URI=file:///config/cyclonedds.xml
  -v "$DDS_CONFIG:/config/cyclonedds.xml:ro"
)

require_file() {
  [[ -f "$1" ]] || { echo "required file is missing: $1" >&2; exit 1; }
}

preflight() {
  local failures=0
  local camera_state=""
  local dds_interface=""
  local local_wifi_ip=""
  local nuc_address="${NUC_HOST##*@}"

  check_file() {
    local label="$1"
    local path="$2"
    if [[ -f "$path" ]]; then
      echo "[ready] $label"
    else
      echo "[missing] $label: $path" >&2
      failures=$((failures + 1))
    fi
  }

  echo "Read-only Go2W perception preflight (ROS_DOMAIN_ID=$DOMAIN_ID)"
  if [[ "$DOMAIN_ID" == "20" ]]; then
    echo "[ready] ROS domain 20"
  else
    echo "[invalid] ROS_DOMAIN_ID must be 20 for this PC/NUC lab" >&2
    failures=$((failures + 1))
  fi
  check_file "NUC SSH key" "$NUC_KEY"
  check_file "CycloneDDS PC profile" "$DDS_CONFIG"
  check_file "VLM environment file" "$ENV_FILE"

  if [[ -f "$DDS_CONFIG" ]]; then
    dds_interface="$(sed -n \
      's/.*<NetworkInterface name="\([^"]*\)".*/\1/p' \
      "$DDS_CONFIG" | head -n 1)"
    if [[ -n "$dds_interface" ]] && ip link show "$dds_interface" >/dev/null 2>&1; then
      local_wifi_ip="$(ip -o -4 addr show dev "$dds_interface" \
        | awk '{split($4, address, "/"); print address[1]; exit}')"
      if [[ -n "$local_wifi_ip" ]]; then
        echo "[ready] DDS Wi-Fi interface $dds_interface=$local_wifi_ip"
      else
        echo "[offline] DDS interface $dds_interface has no IPv4 address" >&2
        failures=$((failures + 1))
      fi
    else
      echo "[missing] DDS Wi-Fi interface from profile: ${dds_interface:-unset}" >&2
      failures=$((failures + 1))
    fi
    if grep -Fq '<AllowMulticast>spdp</AllowMulticast>' "$DDS_CONFIG"; then
      echo "[ready] CycloneDDS SPDP discovery"
    else
      echo "[invalid] CycloneDDS profile does not enable SPDP discovery" >&2
      failures=$((failures + 1))
    fi
    if [[ -n "$local_wifi_ip" ]] \
        && grep -Fq "<Peer address=\"$local_wifi_ip\"/>" "$DDS_CONFIG"; then
      echo "[ready] DDS local peer $local_wifi_ip"
    else
      echo "[invalid] current PC Wi-Fi IP is not a DDS peer" >&2
      failures=$((failures + 1))
    fi
    if grep -Fq "<Peer address=\"$nuc_address\"/>" "$DDS_CONFIG"; then
      echo "[ready] DDS NUC peer $nuc_address"
    else
      echo "[invalid] NUC address $nuc_address is not a DDS peer" >&2
      failures=$((failures + 1))
    fi
  fi

  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[ready] runtime image $IMAGE"
  else
    echo "[missing] runtime image $IMAGE; run '$0 build'" >&2
    failures=$((failures + 1))
  fi
  if docker image inspect z-manip-edgetam:latest >/dev/null 2>&1; then
    echo "[ready] EdgeTAM service image"
  else
    echo "[missing] z-manip-edgetam:latest" >&2
    failures=$((failures + 1))
  fi

  if [[ -f "$NUC_KEY" ]]; then
    if camera_state="$(ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 \
        "$NUC_HOST" \
        'if systemctl --user is-active --quiet d435i.service; then echo active; else echo inactive; fi' \
        2>/dev/null)"; then
      echo "[ready] NUC reachable; d435i.service=$camera_state"
    else
      echo "[offline] NUC is not reachable at $NUC_HOST" >&2
      failures=$((failures + 1))
    fi
  fi

  if [[ -f "$ANYGRASP_CHECKPOINT" && -d "$ANYGRASP_LICENSE_DIR" ]]; then
    echo "[optional] AnyGrasp assets present; daily will verify the license"
  else
    echo "[optional] AnyGrasp incomplete; antipodal fallback remains available"
  fi
  echo "[safety] preflight starts no container and publishes no ROS message"
  if ((failures > 0)); then
    echo "preflight failed: $failures required item(s) unavailable" >&2
    return 1
  fi
  echo "preflight passed"
}

start_nuc_camera() {
  require_file "$NUC_KEY"
  ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST" \
    'systemctl --user start d435i.service && systemctl --user is-active d435i.service'
}

start_edgetam() {
  [[ "$EDGETAM_MAX_FRAMES_PER_SESSION" =~ ^[1-9][0-9]*$ ]] || {
    echo "Z_MANIP_EDGETAM_MAX_FRAMES_PER_SESSION must be a positive integer" >&2
    exit 2
  }
  [[ "$EDGETAM_STREAM_HISTORY_FRAMES" =~ ^[1-9][0-9]*$ ]] || {
    echo "Z_MANIP_EDGETAM_STREAM_HISTORY_FRAMES must be a positive integer" >&2
    exit 2
  }
  [[ "$EDGETAM_MIN_SCORE" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || {
    echo "Z_MANIP_EDGETAM_MIN_SCORE must be a number in [0, 1]" >&2
    exit 2
  }
  mkdir -p "$EDGETAM_CACHE"
  if docker container inspect z-manip-edgetam >/dev/null 2>&1; then
    container_env="$(docker inspect z-manip-edgetam \
      --format '{{range .Config.Env}}{{println .}}{{end}}')"
    if grep -Fxq \
        "EDGETAM_MAX_FRAMES_PER_SESSION=$EDGETAM_MAX_FRAMES_PER_SESSION" \
        <<<"$container_env" \
      && grep -Fxq \
        "EDGETAM_STREAM_HISTORY_FRAMES=$EDGETAM_STREAM_HISTORY_FRAMES" \
        <<<"$container_env" \
      && grep -Fxq \
        "EDGETAM_MIN_SCORE=$EDGETAM_MIN_SCORE" \
        <<<"$container_env"; then
      docker start z-manip-edgetam >/dev/null
    else
      docker rm -f z-manip-edgetam >/dev/null
    fi
  fi
  if ! docker container inspect z-manip-edgetam >/dev/null 2>&1; then
    docker run -d --name z-manip-edgetam --gpus all \
      --restart unless-stopped \
      -p 127.0.0.1:8092:8092 \
      -v "$EDGETAM_CACHE:/models/huggingface" \
      -e EDGETAM_DEVICE=cuda -e HF_HUB_OFFLINE=1 \
      -e "EDGETAM_MAX_FRAMES_PER_SESSION=$EDGETAM_MAX_FRAMES_PER_SESSION" \
      -e "EDGETAM_STREAM_HISTORY_FRAMES=$EDGETAM_STREAM_HISTORY_FRAMES" \
      -e "EDGETAM_MIN_SCORE=$EDGETAM_MIN_SCORE" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      z-manip-edgetam:latest >/dev/null
  fi
  for _ in $(seq 1 20); do
    if curl --silent --fail http://127.0.0.1:8092/health >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  echo "EdgeTAM did not become healthy" >&2
  exit 1
}

start_rgbd_bridge() {
  docker rm -f z-manip-rgbd >/dev/null 2>&1 || true
  docker run -d --name z-manip-rgbd --restart unless-stopped \
    "${docker_env[@]}" "$IMAGE" z-manip-real-rgbd-bridge >/dev/null
}

start_perception() {
  require_file "$ENV_FILE"
  docker rm -f z-manip-hw >/dev/null 2>&1 || true
  docker run -d --name z-manip-hw --restart unless-stopped \
    "${docker_env[@]}" --env-file "$ENV_FILE" \
    "$IMAGE" ros2 launch z_manip_ros perception.launch.py \
      use_sim_time:=false \
      tracker_service_url:=http://127.0.0.1:8092 \
      tracker_sync_timeout_s:=3.0 \
      tracker_sync_queue_size:=60 \
      tracker_data_timeout_s:=4.0 \
      tracker_frame_wait_timeout_s:=8.0 \
      tracker_max_result_stamp_lag_s:=1.5 \
      stop_cmd_topic:=/z_manip/disabled/read_only_stop_cmd >/dev/null
}

start_perception_runner() {
  # Keep the runtime image and container filesystem warm between UI clicks.
  # Each click still starts the fixed Python entrypoint via docker exec.  This
  # container has no CAN/device mount and no actuator environment.
  mkdir -p "$PERCEPTION_RUNNER_ARTIFACT_ROOT"
  docker rm -f "$PERCEPTION_RUNNER" >/dev/null 2>&1 || true
  docker run -d --name "$PERCEPTION_RUNNER" --restart unless-stopped \
    --user "$(id -u):$(id -g)" "${docker_env[@]}" \
    -e HOME=/tmp/z-manip \
    -e ROS_LOG_DIR=/tmp/z-manip-ros-logs \
    -e PYTHONPATH=/opt/z_manip_ws/install/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages:/opt/z_manip/python \
    -e LD_LIBRARY_PATH=/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib \
    -v "$ROOT_DIR/scripts/runtime/go2w_perception_dry_run.py:/usr/local/bin/z-manip-go2w-perception-dry-run:ro" \
    -v "$ROOT_DIR/z_manip:/opt/z_manip/python/z_manip:ro" \
    -v "$PERCEPTION_RUNNER_ARTIFACT_ROOT:/workspace-artifacts" \
    "$IMAGE" sleep infinity >/dev/null
}

start_planning_runner() {
  # Warm only the fixed offline planning filesystem.  The runner has no
  # network, ROS environment, hardware device, CAN socket, or actuator mount.
  # Interactive requests execute the same fixed planner and full safety checks
  # through docker exec, avoiding per-click image/container cold-start jitter.
  mkdir -p "$PERCEPTION_RUNNER_ARTIFACT_ROOT" "$PLANNING_RUNNER_SCRATCH_ROOT"
  docker rm -f "$PLANNING_RUNNER" >/dev/null 2>&1 || true
  docker run -d --name "$PLANNING_RUNNER" --restart unless-stopped \
    --user "$(id -u):$(id -g)" \
    --network none \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    -e HOME=/tmp/z-manip \
    -e PYTHONPATH=/opt/z_manip_ws/install/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages:/opt/z_manip/python \
    -e LD_LIBRARY_PATH=/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib \
    -v "$ROOT_DIR/scripts/runtime/piper_planning_dry_run.py:/usr/local/bin/z-manip-piper-planning-dry-run:ro" \
    -v "$ROOT_DIR/z_manip:/opt/z_manip/python/z_manip:ro" \
    -v "$ROOT_DIR/configs/go2w_piper.json:/opt/z_manip/configs/go2w_piper.json:ro" \
    -v "$ROOT_DIR/../go2W_Sim/assets:/robot_assets:ro" \
    -v "$PERCEPTION_RUNNER_ARTIFACT_ROOT:/workspace-artifacts:ro" \
    -v "$PLANNING_RUNNER_SCRATCH_ROOT:/workspace-planning-output" \
    "$IMAGE" sleep infinity >/dev/null
}

wait_anygrasp_healthy() {
  for _ in $(seq 1 120); do
    if [[ "$(docker inspect z-manip-anygrasp --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
      echo "AnyGrasp server is healthy."
      return 0
    fi
    sleep 2
  done
  echo "AnyGrasp server did not become healthy within 240 s." >&2
  return 1
}

start_anygrasp_server() {
  require_file "$ANYGRASP_CHECKPOINT"
  [[ -d "$ANYGRASP_LICENSE_DIR" ]] || { echo "AnyGrasp license directory is missing: $ANYGRASP_LICENSE_DIR" >&2; return 1; }
  docker rm -f z-manip-anygrasp >/dev/null 2>&1 || true
  docker run -d --name z-manip-anygrasp --restart unless-stopped \
    --gpus all --ipc host --network host \
    -v "$ANYGRASP_LICENSE_DIR:/opt/anygrasp/grasp_detection/license:ro" \
    -v "$ANYGRASP_CHECKPOINT:/opt/anygrasp/grasp_detection/log/checkpoint_detection.tar:ro" \
    "$ANYGRASP_SERVER_IMAGE" >/dev/null
  echo "AnyGrasp server is loading."
  wait_anygrasp_healthy
}

auto_start_anygrasp() {
  if [[ "$(docker inspect z-manip-anygrasp --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
    echo "AnyGrasp server is already healthy."
    return 0
  fi
  if [[ ! -d "$ANYGRASP_DIR" || ! -d "$ANYGRASP_LICENSE_DIR" || ! -f "$ANYGRASP_CHECKPOINT" ]]; then
    echo "AnyGrasp assets are incomplete; using read-only antipodal fallback."
    return 0
  fi
  if "$0" anygrasp-check >/dev/null 2>&1; then
    start_anygrasp_server
  else
    echo "AnyGrasp license is not valid for this PC; using read-only antipodal fallback."
  fi
}

probe() {
  mkdir -p "$ARTIFACT_DIR"
  probe_args=(
    --duration "${PROBE_DURATION_S:-8}"
    --output /artifacts/probe-report.json
  )
  if [[ "${PROBE_REQUIRE_PERCEPTION:-0}" == 1 ]]; then
    probe_args+=(--require-perception)
  fi
  docker run --rm "${docker_env[@]}" \
    -v "$ARTIFACT_DIR:/artifacts" \
    "$IMAGE" \
    z-manip-go2w-perception-probe "${probe_args[@]}"
}

status() {
  echo "NUC camera: $(ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST" 'systemctl --user is-active d435i.service' 2>/dev/null || echo unreachable)"
  docker ps --filter name=z-manip-edgetam --filter name=z-manip-rgbd \
    --filter name=z-manip-hw --filter name=z-manip-perception-runner \
    --filter name=z-manip-planning-runner \
    --filter name=z-manip-anygrasp \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
  if [[ "$(docker inspect z-manip-anygrasp --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
    echo "Grasp proposal backend: AnyGrasp"
  else
    echo "Grasp proposal backend: read-only antipodal fallback (AnyGrasp not healthy)"
  fi
  curl --silent --show-error --fail http://127.0.0.1:8092/health
  echo
  probe
}

case "${1:-status}" in
  build)
    docker build -t "$IMAGE" -f "$ROOT_DIR/docker/runtime/Dockerfile" "$ROOT_DIR"
    ;;
  verify-latest-only)
    mkdir -p "$ARTIFACT_DIR"
    docker run --rm --network none \
      -e ROS_DOMAIN_ID=120 \
      -v "$ARTIFACT_DIR:/artifacts" \
      "$IMAGE" z-manip-go2w-latest-only-verify \
      --output /artifacts/latest-only-report.json
    ;;
  start)
    require_file "$DDS_CONFIG"
    start_nuc_camera
    start_edgetam
    start_rgbd_bridge
    start_perception
    start_perception_runner
    start_planning_runner
    sleep 5
    status
    ;;
  restart-edgetam)
    require_file "$DDS_CONFIG"
    docker rm -f z-manip-edgetam >/dev/null 2>&1 || true
    start_edgetam
    ;;
  restart-rgbd)
    require_file "$DDS_CONFIG"
    start_rgbd_bridge
    ;;
  restart-perception)
    require_file "$DDS_CONFIG"
    start_perception
    start_perception_runner
    start_planning_runner
    ;;
  preflight)
    preflight
    ;;
  status|probe)
    status
    ;;
  stop)
    docker stop z-manip-hw z-manip-rgbd "$PERCEPTION_RUNNER" "$PLANNING_RUNNER" >/dev/null 2>&1 || true
    echo "Stopped perception containers only; no actuator command was sent."
    ;;
  anygrasp-check)
    [[ -d "$ANYGRASP_DIR" ]] || { echo "AnyGrasp directory is missing: $ANYGRASP_DIR" >&2; exit 1; }
    docker compose --project-directory "$ANYGRASP_DIR" \
      -f "$ANYGRASP_DIR/compose.yaml" run --rm anygrasp \
      python /opt/anygrasp_tools/verify_install.py
    ;;
  anygrasp-build)
    docker build -t "$ANYGRASP_SERVER_IMAGE" \
      -f "$ROOT_DIR/docker/anygrasp_runtime/Dockerfile" "$ROOT_DIR"
    ;;
  anygrasp-start)
    "$0" anygrasp-check
    start_anygrasp_server
    ;;
  anygrasp-auto)
    auto_start_anygrasp
    ;;
  anygrasp-stop)
    docker stop z-manip-anygrasp >/dev/null 2>&1 || true
    ;;
  dry-run)
    instruction="${2:-pick the small white USB power adapter with the red port by its white body}"
    mkdir -p "$ARTIFACT_DIR"
    learned_args=()
    passive_args=()
    if [[ "${Z_MANIP_REQUIRE_PASSIVE_WINDOW:-0}" == 1 ]]; then
      passive_args=(
        --passive-window /artifacts/live_passive_joint_report.json
        --selected-passive-window /artifacts/selected_passive_joint_report.json
      )
    fi
    if [[ "$(docker inspect z-manip-anygrasp --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
      learned_args=(--learned-endpoint tcp://127.0.0.1:5557)
    fi
    docker run --rm "${docker_env[@]}" \
      -v "$ARTIFACT_DIR:/artifacts" \
      "$IMAGE" z-manip-go2w-perception-dry-run \
      --instruction "$instruction" --output /artifacts \
      "${passive_args[@]}" "${learned_args[@]}"
    ;;
  soak)
    duration="${2:-120}"
    instruction="${3:-pick the small white USB power adapter with the red port by its white body}"
    [[ "$SOAK_MAX_RECOVERIES" =~ ^[0-9]+$ ]] || { echo "Z_MANIP_SOAK_MAX_RECOVERIES must be a non-negative integer" >&2; exit 2; }
    [[ "$SOAK_RECOVERY_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Z_MANIP_SOAK_RECOVERY_TIMEOUT must be numeric" >&2; exit 2; }
    soak_dir="$ARTIFACT_DIR/soaks/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$soak_dir"
    learned_args=()
    if [[ "$(docker inspect z-manip-anygrasp --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
      learned_args=(--learned-endpoint tcp://127.0.0.1:5557)
    fi
    docker run --rm "${docker_env[@]}" \
      -v "$soak_dir:/artifacts" \
      "$IMAGE" z-manip-go2w-perception-dry-run \
      --instruction "$instruction" --output /artifacts --soak-duration "$duration" \
      --allow-no-grasp-candidates --max-recoveries "$SOAK_MAX_RECOVERIES" \
      --recovery-timeout "$SOAK_RECOVERY_TIMEOUT" \
      "${learned_args[@]}"
    echo "soak report: $soak_dir/report.json"
    ;;
  cycles)
    count="${2:-3}"
    duration="${3:-60}"
    instruction="${4:-pick the small white USB power adapter with the red port by its white body}"
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "cycle count must be a positive integer" >&2; exit 2; }
    [[ "$SOAK_MAX_RECOVERIES" =~ ^[0-9]+$ ]] || { echo "Z_MANIP_SOAK_MAX_RECOVERIES must be a non-negative integer" >&2; exit 2; }
    [[ "$SOAK_RECOVERY_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Z_MANIP_SOAK_RECOVERY_TIMEOUT must be numeric" >&2; exit 2; }
    run_dir="$ARTIFACT_DIR/cycles/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$run_dir"
    learned_args=()
    if [[ "$(docker inspect z-manip-anygrasp --format '{{.State.Health.Status}}' 2>/dev/null || true)" == healthy ]]; then
      learned_args=(--learned-endpoint tcp://127.0.0.1:5557)
    fi
    failures=0
    for cycle in $(seq 1 "$count"); do
      cycle_dir="$run_dir/cycle-$cycle"
      mkdir -p "$cycle_dir"
      echo "read-only cycle $cycle/$count"
      if ! docker run --rm "${docker_env[@]}" \
          -v "$cycle_dir:/artifacts" \
          "$IMAGE" z-manip-go2w-perception-dry-run \
          --instruction "$instruction" --output /artifacts --soak-duration "$duration" \
          --max-recoveries "$SOAK_MAX_RECOVERIES" \
          --recovery-timeout "$SOAK_RECOVERY_TIMEOUT" \
          "${learned_args[@]}"; then
        failures=$((failures + 1))
      fi
    done
    echo "cycle reports: $run_dir"
    if ((failures > 0)); then
      echo "$failures/$count read-only cycles failed" >&2
      exit 1
    fi
    current_target="${run_dir#"$ARTIFACT_DIR"/}/cycle-$count"
    ln -sfn "$current_target" "$ARTIFACT_DIR/current-cycle"
    echo "current cycle artifacts: $ARTIFACT_DIR/current-cycle"
    ;;
  daily)
    count="${2:-3}"
    duration="${3:-20}"
    instruction="${4:-pick the small white USB power adapter with the red port by its white body}"
    "$0" preflight
    "$0" verify-latest-only
    "$0" start
    "$0" anygrasp-auto
    "$0" cycles "$count" "$duration" "$instruction"
    PROBE_REQUIRE_PERCEPTION=1 \
      PROBE_DURATION_S="${DAILY_RESULT_PROBE_DURATION_S:-30}" \
      "$0" probe
    ;;
  *)
    echo "usage: $0 {build|preflight|verify-latest-only|daily [count] [seconds] [instruction]|start|restart-edgetam|restart-rgbd|restart-perception|status|probe|dry-run [instruction]|soak [seconds] [instruction]|cycles [count] [seconds] [instruction]|stop|anygrasp-check|anygrasp-build|anygrasp-auto|anygrasp-start|anygrasp-stop}" >&2
    exit 2
    ;;
esac
