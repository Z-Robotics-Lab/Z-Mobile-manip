#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
REMOTE_DIR="/home/yusenzlabnuc/z-manip-runtime"
REMOTE_ACTION="${GO2W_HOME_REMOTE_DIR:-$REMOTE_DIR/smart-home}"
INTERACTIVE_ROOT="${Z_MANIP_INTERACTIVE_RUN_ROOT:-$STACK_ROOT/../artifacts/go2w_real/interactive_sessions}"
SPEED_PERCENT="${1:-2}"
PIPER_HOME_CONFIG="${PIPER_HOME_CONFIG:-$STACK_ROOT/configs/piper_home.json}"
[[ "$SPEED_PERCENT" =~ ^([1-9]|[1-4][0-9]|50)$ ]] || { printf 'Home speed must be an integer from 1 to 50 percent\n' >&2; exit 2; }

HOME_INPUTS=(
  "$PIPER_HOME_CONFIG"
  "$SCRIPT_DIR/piper_home_recovery.py"
  "$SCRIPT_DIR/piper_reverse_home_recovery.py"
  "$SCRIPT_DIR/piper_staged_grasp_executor.py"
)

for path in "$NUC_KEY" "${HOME_INPUTS[@]}"; do
  [[ -f "$path" ]] || { printf 'required Home input is missing: %s\n' "$path" >&2; exit 1; }
done

# A dedicated control path: sharing the grasp master would let this wrapper's
# teardown drop a connection the grasp legs still depend on.
control_path="$(dirname -- "$NUC_KEY")/z-manip-home-%C"
mux_args=(-o ControlMaster=auto -o ControlPersist=60 -o "ControlPath=$control_path")
ssh_args=(-i "$NUC_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=5 "${mux_args[@]}" "$NUC_HOST")
scp_args=(-q -i "$NUC_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=5 "${mux_args[@]}")

shopt -s nullglob
planning_dirs=("$INTERACTIVE_ROOT"/planning/*/artifacts/planning)
latest_planning=""
for ((index=${#planning_dirs[@]}-1; index>=0; index--)); do
  candidate="${planning_dirs[index]}"
  report="$candidate/planning_report.json"
  archive="$candidate/planned_grasp.npz"
  [[ -f "$report" && ! -L "$report" && -f "$archive" && ! -L "$archive" ]] || continue
  if /usr/bin/python3 - "$report" "$archive" <<'PY'
import hashlib
import hmac
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = report.get("planned_grasp_sha256")
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    valid = (
        report.get("read_only") is True
        and report.get("planning_only") is True
        and report.get("motion_commands_published") == 0
        and report.get("plan_valid") is True
        and report.get("raw_paths_collision_validated") is True
        and isinstance(expected, str)
        and len(expected) == 64
        and hmac.compare_digest(expected, actual)
    )
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
  then
    latest_planning="$candidate"
    break
  fi
done

if [[ -n "$latest_planning" ]]; then
  printf '[home] checked reverse-recovery artifact: %s\n' "$latest_planning"
else
  printf '[home] no complete checked planning artifact; using direct Home recovery\n'
fi

staged_inputs=("${HOME_INPUTS[@]}")
if [[ -n "$latest_planning" ]]; then
  staged_inputs+=("$latest_planning/planning_report.json" "$latest_planning/planned_grasp.npz")
fi

manifest_sha="$(sha256sum "${staged_inputs[@]}" | awk '{print $1}' | sha256sum | awk '{print $1}')"
remote_sha="$(ssh "${ssh_args[@]}" "cat '$REMOTE_ACTION/.manifest-sha' 2>/dev/null || true")"
if [[ "$remote_sha" != "$manifest_sha" ]]; then
  ssh "${ssh_args[@]}" "rm -rf '$REMOTE_ACTION'; mkdir -p '$REMOTE_ACTION'"
  scp "${scp_args[@]}" \
    "$PIPER_HOME_CONFIG" \
    "$NUC_HOST:$REMOTE_ACTION/piper_home.json"
  scp "${scp_args[@]}" \
    "$SCRIPT_DIR/piper_home_recovery.py" \
    "$SCRIPT_DIR/piper_reverse_home_recovery.py" \
    "$SCRIPT_DIR/piper_staged_grasp_executor.py" \
    "$NUC_HOST:$REMOTE_ACTION/"

  if [[ -n "$latest_planning" ]]; then
    scp "${scp_args[@]}" \
      "$latest_planning/planning_report.json" \
      "$latest_planning/planned_grasp.npz" \
      "$NUC_HOST:$REMOTE_ACTION/"
  fi

  # Stamp the marker only after every file has landed, so an interrupted upload
  # forces a full re-stage instead of letting the next call skip onto a partial
  # payload.
  ssh "${ssh_args[@]}" "printf '%s' '$manifest_sha' > '$REMOTE_ACTION/.manifest-sha'"
fi

if [[ -n "$latest_planning" ]]; then
  ssh "${ssh_args[@]}" \
    "set -e; systemctl --user stop z-manip-piper-passive-feedback.service; trap 'sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8 >/tmp/z-manip-passive-restore.log 2>&1 || true; systemctl --user start z-manip-piper-passive-feedback.service' EXIT; cd ~/pyAgxArm; if /usr/bin/python3 '$REMOTE_ACTION/piper_home_recovery.py' --home '$REMOTE_ACTION/piper_home.json' --speed-percent $SPEED_PERCENT --max-recovery-deg 20 --max-step-deg 5 --clear-electronic-estop --execute; then exit 0; fi; token=\$(/usr/bin/python3 '$REMOTE_ACTION/piper_reverse_home_recovery.py' --planning-report '$REMOTE_ACTION/planning_report.json' --planned-grasp '$REMOTE_ACTION/planned_grasp.npz' | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)[\"confirmation_token\"])'); /usr/bin/python3 '$REMOTE_ACTION/piper_reverse_home_recovery.py' --planning-report '$REMOTE_ACTION/planning_report.json' --planned-grasp '$REMOTE_ACTION/planned_grasp.npz' --speed-percent $SPEED_PERCENT --execute --confirm \"\$token\""
else
  ssh "${ssh_args[@]}" \
    "set -e; systemctl --user stop z-manip-piper-passive-feedback.service; trap 'sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8 >/tmp/z-manip-passive-restore.log 2>&1 || true; systemctl --user start z-manip-piper-passive-feedback.service' EXIT; cd ~/pyAgxArm; /usr/bin/python3 '$REMOTE_ACTION/piper_home_recovery.py' --home '$REMOTE_ACTION/piper_home.json' --speed-percent $SPEED_PERCENT --max-recovery-deg 20 --max-step-deg 5 --clear-electronic-estop --execute"
fi
