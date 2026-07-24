#!/usr/bin/env bash
set -euo pipefail

# NUC-local wrapper for the base-lock publisher.  Sources the same Domain-20
# CycloneDDS environment the reactive live service uses so the published
# /go2w/base_lock command is discovered immediately by the co-located WebRTC
# owner.  Invoked over SSH by the PC orchestrator's base-lock transport.
# It publishes an intent message only; it never opens a robot transport.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

set +u
source "$HOME/unitree_venv/bin/activate"
source /opt/ros/jazzy/setup.bash
source "$HOME/Z-Navigation-Stack/install/setup.bash"
source "$HOME/go2w-nuc/ros2_ws/install/setup.bash"
set -u

export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$HOME/go2w-nuc/bringup/cyclonedds.xml"
export PYTHONUNBUFFERED=1

[[ -r "$HOME/go2w-nuc/bringup/cyclonedds.xml" ]] || {
  printf 'missing NUC CycloneDDS profile\n' >&2
  exit 1
}

exec python3 "$SCRIPT_DIR/go2w_base_lock_publish.py" "$@"
