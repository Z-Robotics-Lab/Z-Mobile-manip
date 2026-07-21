#!/usr/bin/env bash
set -euo pipefail

source "$HOME/go2w-nuc/bringup/ros_env.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export Z_MANIP_PIPER_REACTIVE_ACK=I_UNDERSTAND_PIPER_REACTIVE_VIEW_WILL_MOVE
cd "$HOME/pyAgxArm"
exec /usr/bin/python3 \
  "$HOME/.local/lib/z-mobile-manip/piper_reactive_view_executor.py" \
  --execute --channel can0 --firmware v188 --rate-hz 20 --speed-percent 5
