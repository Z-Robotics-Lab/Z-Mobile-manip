#!/usr/bin/env bash
set -euo pipefail

# ROS setup scripts legitimately probe optional shell variables.  Keep
# nounset disabled only while sourcing them; otherwise Jazzy aborts here with
# `AMENT_TRACE_SETUP_FILES: unbound variable` before the executor can publish
# feedback or accept a reactive-view intent.
set +u
source "$HOME/go2w-nuc/bringup/ros_env.sh"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
export Z_MANIP_PIPER_REACTIVE_ACK=I_UNDERSTAND_PIPER_REACTIVE_VIEW_WILL_MOVE
cd "$HOME/pyAgxArm"
exec /usr/bin/python3 \
  "$HOME/.local/lib/z-mobile-manip/piper_reactive_view_executor.py" \
  --execute --channel can0 --firmware v188 --rate-hz 20 --speed-percent 5 \
  --urdf "$HOME/.local/share/z-mobile-manip/go2w_sensored.urdf" \
  --collision-model "$HOME/.local/share/z-mobile-manip/piper_collision_capsules.json"
