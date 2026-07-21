#!/usr/bin/env bash
set -euo pipefail

# Deliberately hard-coded safe action for the scoped NUC sudo policy.
interface="can0"
if [[ "$EUID" -ne 0 ]]; then
  echo "run through the installed scoped sudo rule" >&2
  exit 2
fi
/usr/sbin/ip link set "$interface" down
printf 'PASS: %s is down; tx_packets=%s\n' \
  "$interface" \
  "$(<"/sys/class/net/$interface/statistics/tx_packets")"
