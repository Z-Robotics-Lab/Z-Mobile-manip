#!/usr/bin/env bash
# Launch the offline joint-zero report viewer.  No report generation occurs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPORT="${PIPER_JOINT_ZERO_REPORT:-${1:-}}"
PORT="${PIPER_JOINT_ZERO_UI_PORT:-8770}"

if [[ -z "$REPORT" || ! -f "$REPORT" ]]; then
  printf 'usage: %s /absolute/path/to/piper_joint_zero_calibration.json\n' "$0" >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  printf 'PIPER_JOINT_ZERO_UI_PORT must be an integer from 1 to 65535\n' >&2
  exit 2
fi

exec python3 "$SCRIPT_DIR/piper_joint_zero_ui.py" \
  --report "$(readlink -f -- "$REPORT")" \
  --port "$PORT"
