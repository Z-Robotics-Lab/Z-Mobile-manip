#!/usr/bin/env bash
set -euo pipefail

# Fast-FoundationStereo depth stack: learned stereo depth from the wrist D435's
# IR pair, registered into the COLOR frame, published as
# /camera/ffs_depth_aligned/image_raw (16UC1 mm, exact frameset stamps).
#
#   ffs_depth_stack.sh up       start service + relay containers
#   ffs_depth_stack.sh down     stop + remove both
#   ffs_depth_stack.sh status   health/rate summary
#   ffs_depth_stack.sh logs [service|relay] [lines]
#   ffs_depth_stack.sh build    (re)build the ffs:infer image
#
# Consumers switch via ros2/z_manip_edgetam/config/edgetam.yaml depth_topic
# (requires go2w_perception_lab.sh build + component manager restart perception
# because ros2/** and the yaml are baked into z-manip-runtime:pinocchio).
# FALLBACK: set depth_topic back to /camera/aligned_depth_to_color/image_raw,
# rebuild + restart perception, then `ffs_depth_stack.sh down` -- the raw D435
# aligned-depth path is untouched by this stack.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${Z_MANIP_FFS_IMAGE:-ffs:infer}"
RUNTIME_IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:pinocchio}"
FFS_REPO="${Z_MANIP_FFS_REPO:-$ROOT_DIR/../third_party/Fast-FoundationStereo}"
FFS_MODEL_DIR="${Z_MANIP_FFS_MODEL_DIR:-/ffs/weights/23-36-37/model_best_bp2_serialize.pth}"
FFS_ITERS="${Z_MANIP_FFS_ITERS:-8}"
CALIB="$ROOT_DIR/configs/ffs_d435_calibration.json"
DDS_CONFIG="${Z_MANIP_DDS_CONFIG:-$ROOT_DIR/docker/runtime/cyclonedds-go2w-pc.xml}"
DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
# Persistent Triton/inductor kernel cache: without it every service restart
# pays a multi-second JIT recompile before the first depth frame.
JIT_CACHE="${Z_MANIP_FFS_JIT_CACHE:-$HOME/.cache/z-manip/ffs-jit}"

SERVICE=z-manip-ffs-depth
RELAY=z-manip-ffs-relay

require_file() {
  [[ -e "$1" ]] || { echo "required path is missing: $1" >&2; exit 1; }
}

build() {
  (cd "$ROOT_DIR" && docker build -f docker/ffs_service/Dockerfile -t "$IMAGE" .)
}

up() {
  require_file "$FFS_REPO/core/foundation_stereo.py"
  require_file "$CALIB"
  require_file "$DDS_CONFIG"
  mkdir -p "$JIT_CACHE"
  docker rm -f "$SERVICE" "$RELAY" >/dev/null 2>&1 || true
  docker run -d --name "$SERVICE" --gpus all \
    --restart unless-stopped \
    -p 127.0.0.1:8773:8773 \
    -v "$FFS_REPO:/ffs:ro" \
    -v "$CALIB:/config/ffs_calibration.json:ro" \
    -v "$JIT_CACHE:/jit-cache" \
    -e TRITON_CACHE_DIR=/jit-cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/jit-cache/inductor \
    -e "FFS_MODEL_DIR=$FFS_MODEL_DIR" \
    -e "FFS_ITERS=$FFS_ITERS" \
    "$IMAGE" >/dev/null
  echo "waiting for FFS service (model load + first-run KERNEL JIT can take ~2 min)..."
  for _ in $(seq 1 150); do
    if curl --silent --fail http://127.0.0.1:8773/health 2>/dev/null | grep -q '"ready": true'; then
      echo "FFS depth service healthy."
      start_relay
      return
    fi
    sleep 2
  done
  echo "FFS depth service did not become healthy" >&2
  docker logs --tail 30 "$SERVICE" >&2 || true
  exit 1
}

start_relay() {
  docker rm -f "$RELAY" >/dev/null 2>&1 || true
  # The relay subscribes the NUC's *throttled* IR pair topics (produced by
  # ffs-ir-throttle.service on the NUC, <=10 Hz).  Subscribing the full-rate
  # 30 fps compressed IR across Wi-Fi collapses the color stream to ~6 Hz --
  # never point these at /nuc/camera/infra*/image_rect_raw/compressed.
  docker run -d --name "$RELAY" --restart unless-stopped \
    --network host \
    -e "ROS_DOMAIN_ID=$DOMAIN_ID" \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///config/cyclonedds.xml \
    -e FFS_IR1_TOPIC=/nuc/camera/ffs_ir_pair/infra1/compressed \
    -e FFS_IR2_TOPIC=/nuc/camera/ffs_ir_pair/infra2/compressed \
    -e "FFS_MAX_FPS=${Z_MANIP_FFS_MAX_FPS:-10}" \
    -v "$DDS_CONFIG:/config/cyclonedds.xml:ro" \
    -v "$CALIB:/config/ffs_calibration.json:ro" \
    -v "$ROOT_DIR/scripts/runtime/ffs_depth_relay.py:/usr/local/bin/z-manip-ffs-depth-relay:ro" \
    "$RUNTIME_IMAGE" python3 /usr/local/bin/z-manip-ffs-depth-relay >/dev/null
  echo "FFS relay started."
}

down() {
  docker rm -f "$SERVICE" "$RELAY" >/dev/null 2>&1 || true
  echo "FFS depth stack stopped."
}

status() {
  for c in "$SERVICE" "$RELAY"; do
    if docker container inspect "$c" >/dev/null 2>&1; then
      echo "$c: $(docker inspect "$c" --format '{{.State.Status}} (health={{if .State.Health}}{{.State.Health.Status}}{{else}}n/a{{end}})')"
    else
      echo "$c: absent"
    fi
  done
  curl --silent http://127.0.0.1:8773/health 2>/dev/null && echo || echo "service /health unreachable"
  docker logs --tail 3 "$RELAY" 2>/dev/null | grep -E "pairs_in|calib" || true
}

logs() {
  local which="${1:-relay}" lines="${2:-40}"
  case "$which" in
    service) docker logs --tail "$lines" "$SERVICE" ;;
    relay) docker logs --tail "$lines" "$RELAY" ;;
    *) echo "usage: logs [service|relay] [lines]" >&2; exit 2 ;;
  esac
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  logs) shift; logs "$@" ;;
  build) build ;;
  *) echo "usage: $0 {up|down|status|logs [service|relay] [n]|build}" >&2; exit 2 ;;
esac
