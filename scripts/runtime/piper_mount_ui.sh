#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPORT="${PIPER_MOUNT_REPORT:-${1:-}}"
PORT="${PIPER_MOUNT_UI_PORT:-8768}"
if [[ -z "$REPORT" || ! -f "$REPORT" ]]; then
  printf 'usage: %s /absolute/path/to/piper_mount_calibration.json\n' "$0" >&2
  exit 2
fi
exec python3 "$SCRIPT_DIR/piper_mount_ui.py" --report "$REPORT" --port "$PORT"
