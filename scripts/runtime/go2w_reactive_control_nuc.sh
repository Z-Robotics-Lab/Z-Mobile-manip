#!/usr/bin/env bash
set -euo pipefail

# NUC-side single WebRTC owner.  Default invocation is transport-free shadow.
# Live requires BOTH the positional mode and an exact environment gate.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-shadow}"

case "$MODE" in
  shadow|live) ;;
  *) printf 'usage: %s [shadow|live]\n' "$0" >&2; exit 2 ;;
esac

if [[ "$MODE" == live ]]; then
  expected="I_UNDERSTAND_GO2W_WILL_MOVE"
  [[ "${Z_MANIP_GO2W_LIVE_ACK:-}" == "$expected" ]] || {
    printf 'live blocked: set Z_MANIP_GO2W_LIVE_ACK=%s on the NUC\n' "$expected" >&2
    exit 3
  }
fi

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

guard_pid=""
control_pid=""
cleanup() {
  [[ -z "$guard_pid" ]] || kill "$guard_pid" 2>/dev/null || true
  [[ -z "$control_pid" ]] || kill "$control_pid" 2>/dev/null || true
  wait "$guard_pid" "$control_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$MODE" == live ]]; then
  ros2 launch cmd_vel_guard cmd_vel_guard.launch.py \
    command_timeout_s:=0.4 max_linear_mps:=0.20 max_angular_rps:=0.35 &
  guard_pid=$!
fi

python3 "$SCRIPT_DIR/go2w_reactive_control_nuc.py" \
  --mode "$MODE" \
  --ros-args \
  -r cmd_vel:=/cmd_vel_safe \
  -p robot_ip:=192.168.123.161 \
  -p connection_method:=LocalSTA \
  -p control_mode:=sport_cmd \
  -p device_type:=Go2 &
control_pid=$!

if [[ "$MODE" == live ]]; then
  # Neither cmd_vel_guard nor the control node is expected to exit while this
  # unit is live, so "wait -n" finishing at all is already the failure signal
  # -- but its own reported code cannot be trusted: `ros2 launch` swallows a
  # died child and still exits 0 (observed on the NUC as "wlo1: does not
  # match an available interface" killing cmd_vel_guard_node during the wlo1
  # DHCP startup race, while go2w_reactive_control_nuc.sh reported
  # status=0/SUCCESS and Restart=on-failure never fired). Force a nonzero
  # status whenever the race resolves as "success" so systemd always retries
  # a co-process death, regardless of which side raced first.
  set +e
  wait -n "$guard_pid" "$control_pid"
  rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    printf 'go2w_reactive_control_nuc: a live co-process exited reporting success while the bridge was still starting/running; treating as a failure so systemd retries\n' >&2
    rc=1
  fi
  exit "$rc"
else
  wait "$control_pid"
fi
