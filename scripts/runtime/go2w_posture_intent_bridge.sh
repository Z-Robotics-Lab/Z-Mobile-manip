#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-shadow}"
case "$MODE" in
  shadow|live) ;;
  *) printf 'usage: %s [shadow|live]\n' "$0" >&2; exit 2 ;;
esac

if [[ "$MODE" == live ]]; then
  expected="I_UNDERSTAND_POSTURE_INTENTS_REACH_NUC"
  [[ "${Z_MANIP_POSTURE_INTENT_LIVE_ACK:-}" == "$expected" ]] || {
    printf 'live blocked: set Z_MANIP_POSTURE_INTENT_LIVE_ACK=%s\n' "$expected" >&2
    exit 3
  }
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DDS_CONFIG="$STACK_ROOT/docker/runtime/cyclonedds-go2w-pc.xml"
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:pinocchio}"
CONTAINER="z-mobile-manip-posture-intent"

[[ -r "$DDS_CONFIG" ]] || { printf 'missing DDS profile: %s\n' "$DDS_CONFIG" >&2; exit 1; }
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
exec docker run --rm \
  --name "$CONTAINER" \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e ROS_DOMAIN_ID=20 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=file:///config/cyclonedds.xml \
  -e Z_MANIP_POSTURE_INTENT_LIVE_ACK="${Z_MANIP_POSTURE_INTENT_LIVE_ACK:-}" \
  -e PYTHONPATH="$STACK_ROOT:/opt/z_manip/python" \
  -v "$DDS_CONFIG:/config/cyclonedds.xml:ro" \
  -v "$STACK_ROOT:$STACK_ROOT:ro" \
  "$IMAGE" \
  python3 "$SCRIPT_DIR/go2w_posture_intent_bridge.py" --mode "$MODE"
