#!/usr/bin/env bash
# Start/stop the loopback-only PiPER hand-eye calibration workbench.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$STACK_ROOT/.." && pwd)"
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:jazzy}"
CONTAINER="${Z_MANIP_CALIBRATION_UI_CONTAINER:-z-manip-calibration-ui}"
PORT="${Z_MANIP_CALIBRATION_UI_PORT:-8767}"
ARTIFACT_ROOT="${Z_MANIP_REAL_ARTIFACT_ROOT:-$WORKSPACE_ROOT/artifacts/go2w_real}"
DDS_CONFIG="${Z_MANIP_DDS_CONFIG:-$STACK_ROOT/docker/runtime/cyclonedds-go2w-pc.xml}"
ROBOT_ASSETS="${Z_MANIP_ROBOT_ASSETS:-$WORKSPACE_ROOT/go2W_Sim/assets}"
SSH_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
KNOWN_HOSTS="${GO2W_SSH_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
BOARD_METADATA="${Z_MANIP_CHARUCO_METADATA:-$ARTIFACT_ROOT/charuco/board_A4_landscape_clean.json}"
DATASET="${Z_MANIP_HAND_EYE_DATASET:-$ARTIFACT_ROOT/calibration/hand_eye_samples.json}"
CALIBRATION="${Z_MANIP_CAMERA_CALIBRATION_OUTPUT:-$ARTIFACT_ROOT/calibration/piper_wrist_camera_calibration.json}"
URDF="${Z_MANIP_PIPER_URDF:-$ROBOT_ASSETS/urdf/go2w_sensored.urdf}"
CAPTURE_ONLY="${Z_MANIP_CALIBRATION_CAPTURE_ONLY:-0}"

require_file() {
  [[ -f "$1" ]] || { printf 'required file is missing: %s\n' "$1" >&2; exit 1; }
}

start() {
  require_file "$DDS_CONFIG"
  require_file "$SSH_KEY"
  require_file "$KNOWN_HOSTS"
  require_file "$BOARD_METADATA"
  require_file "$URDF"
  require_file "$SCRIPT_DIR/piper_calibration_ui.py"
  require_file "$STACK_ROOT/web/calibration_dashboard/index.html"
  docker image inspect "$IMAGE" >/dev/null
  mkdir -p "$(dirname -- "$DATASET")"
  for artifact in "$BOARD_METADATA" "$DATASET" "$CALIBRATION"; do
    artifact_relative="$(realpath -m --relative-to="$ARTIFACT_ROOT" "$artifact")"
    [[ "$artifact_relative" != .. && "$artifact_relative" != ../* && "$artifact_relative" != /* ]] || {
      printf 'artifact path must remain under %s: %s\n' "$ARTIFACT_ROOT" "$artifact" >&2
      return 2
    }
  done
  board_relative="$(realpath -m --relative-to="$ARTIFACT_ROOT" "$BOARD_METADATA")"
  dataset_relative="$(realpath -m --relative-to="$ARTIFACT_ROOT" "$DATASET")"
  calibration_relative="$(realpath -m --relative-to="$ARTIFACT_ROOT" "$CALIBRATION")"
  capture_args=()
  [[ "$CAPTURE_ONLY" != 1 ]] || capture_args=(--capture-only)
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker run -d --name "$CONTAINER" --restart unless-stopped --network host \
    -e ROS_DOMAIN_ID=20 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///config/cyclonedds.xml \
    -v "$DDS_CONFIG:/config/cyclonedds.xml:ro" \
    -v "$ARTIFACT_ROOT:/artifacts" \
    -v "$ROBOT_ASSETS:/robot/assets:ro" \
    -v "$SSH_KEY:/ssh/id_ed25519:ro" \
    -v "$KNOWN_HOSTS:/ssh/known_hosts:ro" \
    -v "$SCRIPT_DIR/piper_calibration_ui.py:/usr/local/bin/z-manip-piper-calibration-ui:ro" \
    -v "$SCRIPT_DIR/piper_hand_eye_sample.py:/usr/local/bin/z-manip-piper-hand-eye-sample:ro" \
    -v "$SCRIPT_DIR/piper_hand_eye_calibrate.py:/usr/local/bin/z-manip-piper-hand-eye-calibrate:ro" \
    -v "$SCRIPT_DIR/piper_charuco_tool.py:/usr/local/bin/z-manip-piper-charuco:ro" \
    -v "$STACK_ROOT/web/calibration_dashboard/index.html:/opt/z_manip/calibration-ui/index.html:ro" \
    "$IMAGE" python3 /usr/local/bin/z-manip-piper-calibration-ui \
      --index /opt/z_manip/calibration-ui/index.html \
      --board-metadata /artifacts/"$board_relative" \
      --dataset /artifacts/"$dataset_relative" \
      --calibration /artifacts/"$calibration_relative" \
      --urdf /robot/assets/urdf/"$(basename -- "$URDF")" \
      --charuco-tool /usr/local/bin/z-manip-piper-charuco \
      --sample-tool /usr/local/bin/z-manip-piper-hand-eye-sample \
      --calibrate-tool /usr/local/bin/z-manip-piper-hand-eye-calibrate \
      --ssh-key /ssh/id_ed25519 \
      --known-hosts /ssh/known_hosts \
      --nuc-host "$NUC_HOST" \
      --port "$PORT" "${capture_args[@]}" >/dev/null
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
      printf 'PiPER calibration workbench: http://127.0.0.1:%s/\n' "$PORT"
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://127.0.0.1:$PORT/" >/dev/null 2>&1 &
      fi
      return
    fi
    sleep 1
  done
  docker logs "$CONTAINER" >&2 || true
  return 1
}

case "${1:-start}" in
  start) start ;;
  stop) docker rm -f "$CONTAINER" >/dev/null 2>&1 || true ;;
  status) docker ps --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' ;;
  logs) docker logs --tail 120 "$CONTAINER" ;;
  *) printf 'usage: %s {start|stop|status|logs}\n' "$0" >&2; exit 2 ;;
esac
