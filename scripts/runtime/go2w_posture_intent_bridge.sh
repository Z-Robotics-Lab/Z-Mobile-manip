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
set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
exec python3 "$SCRIPT_DIR/go2w_posture_intent_bridge.py" --mode "$MODE"
