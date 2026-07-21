#!/usr/bin/env bash
set -euo pipefail

# Fixed launcher used by the loopback UI.  The browser can choose only the
# shadow/live mode; topic names, gains, bounds, ROS domain, and executable are
# owned here rather than accepted as arbitrary HTTP input.
if (($# != 2)); then
  printf 'usage: %s {shadow|live} ABSOLUTE_STATUS_PATH\n' "$0" >&2
  exit 2
fi

MODE="$1"
STATUS_PATH="$2"
TRACE_PATH="${STATUS_PATH%.json}.trace.jsonl"
RUNTIME_STATE_PATH="$(dirname -- "$STATUS_PATH")/runtime-observer.json"
case "$MODE" in
  shadow|live) ;;
  *) printf 'mode must be shadow or live\n' >&2; exit 2 ;;
esac
if [[ "$STATUS_PATH" != /* ]]; then
  printf 'status path must be absolute\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DDS_CONFIG="$STACK_ROOT/docker/runtime/cyclonedds-go2w-pc.xml"
RUNTIME_IMAGE="${Z_MANIP_WHOLE_BODY_IMAGE:-z-mobile-manip-whole-body:latest}"
WHOLE_BODY_URDF="${Z_MANIP_WHOLE_BODY_URDF:-$STACK_ROOT/../go2W_Sim/assets/urdf/go2w_sensored.urdf}"
WHOLE_BODY_CALIBRATION="${Z_MANIP_WHOLE_BODY_CALIBRATION:-$STACK_ROOT/../artifacts/go2w_real/calibration/piper_wrist_camera_calibration.json}"
CONTAINER_NAME="z-manip-go2w-depth-servo"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
ARM_OWNER_STARTED=0

release_arm_owner() {
  if [[ "$ARM_OWNER_STARTED" == 1 ]]; then
    if ! ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST" \
      'set -eu; systemctl --user stop z-mobile-manip-piper-reactive-view.service; systemctl --user restart z-manip-piper-passive-feedback.service; systemctl --user is-active --quiet z-manip-piper-passive-feedback.service' \
      >/dev/null 2>&1; then
      printf 'warning: failed to restore the passive PiPER feedback owner on the NUC\n' >&2
    fi
    ARM_OWNER_STARTED=0
  fi
}

acquire_arm_owner() {
  # The reactive service conflicts with the passive listener, so systemd stops
  # the old CAN owner and starts the new one in one transaction. If start-up or
  # the postcondition check fails, restore the passive owner before returning.
  ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST" '
    set -eu
    restore_passive() {
      systemctl --user stop z-mobile-manip-piper-reactive-view.service >/dev/null 2>&1 || true
      systemctl --user restart z-manip-piper-passive-feedback.service
    }
    if ! systemctl --user start z-mobile-manip-piper-reactive-view.service; then
      restore_passive
      exit 1
    fi
    if ! systemctl --user is-active --quiet z-mobile-manip-piper-reactive-view.service ||
       systemctl --user is-active --quiet z-manip-piper-passive-feedback.service; then
      restore_passive
      exit 1
    fi
  '
  ARM_OWNER_STARTED=1
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  release_arm_owner
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -r "$DDS_CONFIG" ]]; then
  printf 'missing PC CycloneDDS profile: %s\n' "$DDS_CONFIG" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  printf 'docker is required for the fixed CycloneDDS depth-servo runtime\n' >&2
  exit 1
fi
[[ -r "$WHOLE_BODY_URDF" ]] || {
  printf 'missing whole-body URDF: %s\n' "$WHOLE_BODY_URDF" >&2
  exit 1
}
[[ -r "$WHOLE_BODY_CALIBRATION" ]] || {
  printf 'missing measured hand-eye calibration: %s\n' "$WHOLE_BODY_CALIBRATION" >&2
  exit 1
}
if ! docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
  printf 'building one-time CasADi whole-body runtime image: %s\n' "$RUNTIME_IMAGE" >&2
  docker build -t "$RUNTIME_IMAGE" \
    -f "$STACK_ROOT/docker/whole_body_runtime/Dockerfile" "$STACK_ROOT"
fi

# A user service can remain active after Unitree's WebRTC data channel has
# closed. In that state ROS accepts /cmd_vel but the robot never receives it.
# Refuse to start a live servo until the fixed NUC transport has either been
# verified or restarted and reconnected. Shadow mode remains transport-free.
if [[ "$MODE" == live ]]; then
  "$SCRIPT_DIR/go2w_base_transport_preflight.sh"
  [[ -f "$NUC_KEY" ]] || { printf 'missing NUC SSH key: %s\n' "$NUC_KEY" >&2; exit 1; }
  acquire_arm_owner
fi

# The workstation host currently has FastDDS only, while every long-running
# Z-Manip ROS process and the NUC use CycloneDDS with explicit Wi-Fi peers.
# Running this small publisher in the runtime image makes discovery
# deterministic across the PC/NUC boundary.  A host FastDDS publisher can be
# visible to local containers yet disappear from the NUC, causing the base
# watchdog to stop a valid approach after the first short motion.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run --rm \
  --name "$CONTAINER_NAME" \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e ROS_DOMAIN_ID=20 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=file:///config/cyclonedds.xml \
  -e PYTHONPATH="$STACK_ROOT:/opt/z_manip/python" \
  -v "$DDS_CONFIG:/config/cyclonedds.xml:ro" \
  -v "$STACK_ROOT:$STACK_ROOT:ro" \
  -v "$WHOLE_BODY_URDF:/robot/go2w_sensored.urdf:ro" \
  -v "$WHOLE_BODY_CALIBRATION:/robot/piper_wrist_camera_calibration.json:ro" \
  -v "$(dirname -- "$STATUS_PATH"):$(dirname -- "$STATUS_PATH"):rw" \
  "$RUNTIME_IMAGE" \
  python3 "$SCRIPT_DIR/go2w_depth_servo.py" \
  --mode "$MODE" \
  --status-file "$STATUS_PATH" \
  --trace-file "$TRACE_PATH" \
  --target-topic /track_3d/selected_target_pointcloud \
  --tracking-topic /track_3d/is_tracking \
  --velocity-topic /cmd_vel \
  --runtime-state "$RUNTIME_STATE_PATH" \
  --runtime-transform-timeout-s 0.50 \
  --whole-body casadi \
  --whole-body-urdf /robot/go2w_sensored.urdf \
  --whole-body-calibration /robot/piper_wrist_camera_calibration.json \
  --desired-depth-m 0.50 \
  --handoff-depth-m 0.52 \
  --handoff-bearing-deg 20 \
  --min-forward-mps 0.10 \
  --max-forward-mps 0.18 \
  --max-yaw-rps 0.12 \
  --target-timeout-s 0.25 \
  --tracking-loss-grace-s 0.75 \
  --rate-hz 20
