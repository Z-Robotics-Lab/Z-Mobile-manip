#!/usr/bin/env bash
set -euo pipefail

# Root-only, fail-closed SocketCAN bring-up for the passive PiPER feedback
# probe. This script never starts a ROS node, PiPER SDK, controller, or CAN
# sender. The interface is left UP only after both pre-probe and in-probe TX
# counters prove that the host transmitted zero frames.

interface="${1:-can0}"
duration="${2:-8}"
probe="/usr/local/libexec/z-manip/piper_passive_probe.py"
report="${3:-/tmp/piper_passive_probe_report.json}"
success=0

if [[ "$EUID" -ne 0 ]]; then
  echo "run this gate with sudo; it only configures SocketCAN and receives frames" >&2
  exit 2
fi
if [[ ! "$interface" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
  echo "invalid CAN interface: $interface" >&2
  exit 2
fi
if [[ ! -r "$probe" ]]; then
  echo "passive probe is missing: $probe" >&2
  exit 2
fi
if ! [[ "$duration" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "duration must be numeric" >&2
  exit 2
fi

unsafe_processes="$(pgrep -af \
  'piper_sdk|piper_ctrl|piper_driver|piper_ros|agx_arm|piper_reactive_view|piper_full_grasp|piper_staged_grasp|piper_wrist_search' || true)"
if [[ -n "$unsafe_processes" ]]; then
  echo "refusing passive CAN bring-up while an arm process is present:" >&2
  echo "$unsafe_processes" >&2
  exit 3
fi

cleanup() {
  if [[ "$success" -ne 1 ]]; then
    ip link set "$interface" down >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

ip link set "$interface" down
ip link set "$interface" type can bitrate 1000000
tx_before="$(<"/sys/class/net/$interface/statistics/tx_packets")"
ip link set "$interface" up
sleep 1
tx_after_up="$(<"/sys/class/net/$interface/statistics/tx_packets")"
if [[ "$tx_after_up" -ne "$tx_before" ]]; then
  echo "host TX counter changed during passive interface bring-up" >&2
  exit 4
fi

python3 "$probe" \
  --interface "$interface" \
  --duration "$duration" \
  --output "$report"
tx_after_probe="$(<"/sys/class/net/$interface/statistics/tx_packets")"
if [[ "$tx_after_probe" -ne "$tx_before" ]]; then
  echo "host TX counter changed during passive feedback probe" >&2
  exit 5
fi

success=1
echo "PASS: complete PiPER feedback received with zero host CAN transmission"
echo "read-only report: $report"
ip -details -statistics link show "$interface"
