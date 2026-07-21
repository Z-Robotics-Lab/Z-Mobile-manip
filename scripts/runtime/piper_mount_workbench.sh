#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$STACK_ROOT/.." && pwd)"
ARTIFACT_ROOT="${Z_MANIP_REAL_ARTIFACT_ROOT:-$WORKSPACE_ROOT/artifacts/go2w_real}"
MOUNT_DIR="${PIPER_MOUNT_DIR:-$ARTIFACT_ROOT/mount_calibration}"
SAMPLES="${PIPER_MOUNT_SAMPLES:-$MOUNT_DIR/hand_eye_samples.json}"
HAND_EYE="${Z_MANIP_CAMERA_CALIBRATION:-$ARTIFACT_ROOT/calibration/piper_wrist_camera_calibration.json}"
ANCHOR="${PIPER_MOUNT_ANCHOR:-$MOUNT_DIR/platform_target_anchor.json}"
URDF="${Z_MANIP_ROBOT_URDF:-$WORKSPACE_ROOT/go2W_Sim/assets/urdf/go2w_sensored.urdf}"
REPORT="${PIPER_MOUNT_REPORT:-$MOUNT_DIR/piper_mount_calibration.json}"
PORT="${PIPER_MOUNT_UI_PORT:-8768}"
UI_CONTAINER="${PIPER_MOUNT_UI_CONTAINER:-piper-mount-report-ui}"
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:jazzy}"

stop_report_ui() {
  docker rm -f "$UI_CONTAINER" >/dev/null 2>&1 || true
}

show_report() {
  [[ -f "$REPORT" ]] || { printf 'mount report is missing: %s\n' "$REPORT" >&2; exit 1; }
  stop_report_ui
  python3 "$SCRIPT_DIR/piper_mount_ui.py" --report "$REPORT" --check >/dev/null
  docker run -d --name "$UI_CONTAINER" --restart unless-stopped \
    --network host \
    -v "$MOUNT_DIR:$MOUNT_DIR:ro" \
    -v "$SCRIPT_DIR/piper_mount_ui.py:/app/server.py:ro" \
    -v "$STACK_ROOT/web/mount_dashboard/index.html:/app/index.html:ro" \
    "$IMAGE" python3 /app/server.py \
      --report "$REPORT" --index /app/index.html --port "$PORT" >/dev/null
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
      printf 'PiPER mount calibration dashboard: http://127.0.0.1:%s/\n' "$PORT"
      return
    fi
    sleep 0.2
  done
  printf 'mount dashboard failed to start\n' >&2
  exit 1
}

case "${1:-capture}" in
  capture)
    mkdir -p "$MOUNT_DIR"
    if [[ ! -f "$ANCHOR" ]]; then
      cp "$STACK_ROOT/configs/piper_mount_anchor.example.json" "$ANCHOR"
    fi
    Z_MANIP_CALIBRATION_UI_CONTAINER=piper-mount-capture-ui \
    Z_MANIP_CALIBRATION_UI_PORT=8769 \
    Z_MANIP_CALIBRATION_CAPTURE_ONLY=1 \
    Z_MANIP_HAND_EYE_DATASET="$SAMPLES" \
    Z_MANIP_CAMERA_CALIBRATION_OUTPUT="$MOUNT_DIR/disabled_hand_eye_output.json" \
      "$SCRIPT_DIR/piper_calibration_ui.sh" start
    printf 'Rigidly body-mount the measured board, then collect 12–20 poses at http://127.0.0.1:8769/\n'
    printf 'Replace the fail-closed anchor template with measured values: %s\n' "$ANCHOR"
    ;;
  solve)
    mkdir -p "$MOUNT_DIR"
    for file in "$SAMPLES" "$HAND_EYE" "$ANCHOR" "$URDF"; do
      [[ -f "$file" ]] || { printf 'required file is missing: %s\n' "$file" >&2; exit 1; }
    done
    solve_rc=0
    PYTHONPATH="$STACK_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      python3 "$SCRIPT_DIR/piper_mount_calibrate.py" \
        --samples "$SAMPLES" --hand-eye "$HAND_EYE" --anchor "$ANCHOR" \
        --urdf "$URDF" --output "$REPORT" || solve_rc=$?
    show_report
    exit "$solve_rc"
    ;;
  show)
    show_report
    ;;
  stop)
    stop_report_ui
    Z_MANIP_CALIBRATION_UI_CONTAINER=piper-mount-capture-ui \
      "$SCRIPT_DIR/piper_calibration_ui.sh" stop
    ;;
  *)
    printf 'usage: %s {capture|solve|show|stop}\n' "$0" >&2
    exit 2
    ;;
esac
