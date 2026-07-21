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
RUNTIME_IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:pinocchio}"
CONTAINER_NAME="z-manip-go2w-depth-servo"

if [[ ! -r "$DDS_CONFIG" ]]; then
  printf 'missing PC CycloneDDS profile: %s\n' "$DDS_CONFIG" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  printf 'docker is required for the fixed CycloneDDS depth-servo runtime\n' >&2
  exit 1
fi

# A user service can remain active after Unitree's WebRTC data channel has
# closed. In that state ROS accepts /cmd_vel but the robot never receives it.
# Refuse to start a live servo until the fixed NUC transport has either been
# verified or restarted and reconnected. Shadow mode remains transport-free.
if [[ "$MODE" == live ]]; then
  "$SCRIPT_DIR/go2w_base_transport_preflight.sh"
fi

# The workstation host currently has FastDDS only, while every long-running
# Z-Manip ROS process and the NUC use CycloneDDS with explicit Wi-Fi peers.
# Running this small publisher in the runtime image makes discovery
# deterministic across the PC/NUC boundary.  A host FastDDS publisher can be
# visible to local containers yet disappear from the NUC, causing the base
# watchdog to stop a valid approach after the first short motion.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
exec docker run --rm \
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
  --desired-depth-m 0.50 \
  --handoff-depth-m 0.52 \
  --handoff-bearing-deg 20 \
  --min-forward-mps 0.10 \
  --max-forward-mps 0.18 \
  --max-yaw-rps 0.12 \
  --target-timeout-s 0.25 \
  --tracking-loss-grace-s 0.75 \
  --rate-hz 20
