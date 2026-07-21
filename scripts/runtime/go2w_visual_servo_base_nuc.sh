#!/usr/bin/env bash
set -euo pipefail

# NUC-side base transport for the Domain-20 manipulation graph.  The visual
# servo publishes /cmd_vel; cmd_vel_guard is the only producer of
# /cmd_vel_safe; unitree_control consumes only that guarded topic.
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

if [[ ! -r "$HOME/go2w-nuc/bringup/cyclonedds.xml" ]]; then
  printf 'missing NUC CycloneDDS profile: %s\n' \
    "$HOME/go2w-nuc/bringup/cyclonedds.xml" >&2
  exit 1
fi

guard_pid=""
control_pid=""
cleanup() {
  if [[ -n "$guard_pid" && -n "$control_pid" ]]; then
    kill "$guard_pid" "$control_pid" 2>/dev/null || true
    wait "$guard_pid" "$control_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ros2 launch cmd_vel_guard cmd_vel_guard.launch.py \
  command_timeout_s:=0.4 max_linear_mps:=0.20 max_angular_rps:=0.35 &
guard_pid=$!

ros2 launch unitree_webrtc_ros unitree_control.launch.py \
  robot_ip:=192.168.123.161 \
  connection_method:=LocalSTA \
  control_mode:=sport_cmd \
  device_type:=Go2 &
control_pid=$!

wait -n "$guard_pid" "$control_pid"
