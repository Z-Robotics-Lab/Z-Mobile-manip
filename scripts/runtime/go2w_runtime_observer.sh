#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$ROOT_DIR/.." && pwd)"
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:jazzy}"
CONTAINER="${Z_MANIP_RUNTIME_OBSERVER_CONTAINER:-z-manip-runtime-observer}"
DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
DDS_CONFIG="${Z_MANIP_DDS_CONFIG:-$ROOT_DIR/docker/runtime/cyclonedds-go2w-pc.xml}"
OUTPUT="${Z_MANIP_RUNTIME_OBSERVER_OUTPUT:-$ROOT_DIR/../artifacts/go2w_real/latest/runtime-observer.json}"
CAMERA_OUTPUT_NAME="camera-latest.jpg"
OBSERVER="$SCRIPT_DIR/go2w_runtime_observer.py"
URDF="${Z_MANIP_ROBOT_URDF:-$WORKSPACE_ROOT/go2W_Sim/assets/urdf/go2w_sensored.urdf}"
CALIBRATION="${Z_MANIP_CAMERA_CALIBRATION:-$WORKSPACE_ROOT/artifacts/go2w_real/calibration/piper_wrist_camera_calibration.json}"

require_file() {
  [[ -f "$1" ]] || { printf 'required file is missing: %s\n' "$1" >&2; exit 1; }
}

run_observer() {
  require_file "$OBSERVER"
  require_file "$DDS_CONFIG"
  require_file "$URDF"
  require_file "$CALIBRATION"
  [[ "$DOMAIN_ID" == "20" ]] || {
    printf 'ROS_DOMAIN_ID must be 20 for the PC/NUC runtime observer\n' >&2
    exit 2
  }
  output_dir="$(dirname -- "$OUTPUT")"
  output_name="$(basename -- "$OUTPUT")"
  mkdir -p "$output_dir"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  exec docker run --rm --name "$CONTAINER" --init --network host \
    --label z-manip.role=runtime-observer \
    --read-only --cap-drop ALL --security-opt no-new-privileges \
    --tmpfs /tmp:rw,noexec,nosuid,size=16m \
    --user "$(id -u):$(id -g)" \
    -e "ROS_DOMAIN_ID=$DOMAIN_ID" \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///config/cyclonedds.xml \
    -e ROS_LOG_DIR=/tmp/ros-log \
    -e PYTHONPATH=/workspace \
    -v "$DDS_CONFIG:/config/cyclonedds.xml:ro" \
    -v "$OBSERVER:/observer/go2w_runtime_observer.py:ro" \
    -v "$ROOT_DIR/z_manip:/workspace/z_manip:ro" \
    -v "$URDF:/model/robot.urdf:ro" \
    -v "$CALIBRATION:/model/camera-calibration.json:ro" \
    -v "$output_dir:/artifacts" \
    "$IMAGE" \
      python3 /observer/go2w_runtime_observer.py \
        --output "/artifacts/$output_name" \
        --camera-output "/artifacts/$CAMERA_OUTPUT_NAME" \
        --ros-domain-id "$DOMAIN_ID" \
        --urdf /model/robot.urdf \
        --calibration /model/camera-calibration.json \
        --base-link piper_base_link \
        --tip-link piper_gripper_base \
        --platform-urdf-link base \
        --platform-frame base_link \
        --write-period-s 0.10
}

case "${1:-run}" in
  run) run_observer ;;
  stop) docker rm -f "$CONTAINER" >/dev/null 2>&1 || true ;;
  status)
    docker ps -a --filter "name=^/${CONTAINER}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
    ;;
  *) printf 'usage: %s {run|stop|status}\n' "$0" >&2; exit 2 ;;
esac
