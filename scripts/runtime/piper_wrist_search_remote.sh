#!/usr/bin/env bash
set -euo pipefail

if (($# != 2)); then
  printf 'usage: %s FIXED_VIEW_INDEX SPEED_PERCENT\n' "$0" >&2
  exit 2
fi
[[ "${Z_MANIP_ENABLE_WRIST_SEARCH:-0}" == 1 ]] || {
  printf 'live wrist search is locked; enable it only while the operator is present\n' >&2
  exit 3
}
VIEW_INDEX="$1"
SPEED_PERCENT="$2"
[[ "$VIEW_INDEX" =~ ^[0-9]+$ ]] || { printf 'view index must be decimal\n' >&2; exit 2; }
[[ "$SPEED_PERCENT" =~ ^([1-9]|1[0-2])$ ]] || { printf 'search speed must be 1..12 percent\n' >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
REMOTE_DIR="${GO2W_WRIST_SEARCH_REMOTE_DIR:-/home/yusenzlabnuc/z-manip-runtime/wrist-search}"

WRIST_SEARCH_INPUTS=(
  "$STACK_ROOT/configs/piper_home.json"
  "$STACK_ROOT/z_manip/control/wrist_search.py"
  "$SCRIPT_DIR/piper_wrist_search_executor.py"
  "$SCRIPT_DIR/piper_staged_grasp_executor.py"
)

for path in "$NUC_KEY" "${WRIST_SEARCH_INPUTS[@]}"; do
  [[ -f "$path" ]] || { printf 'required wrist-search input is missing: %s\n' "$path" >&2; exit 1; }
done

# Reuse one authenticated master connection across the marker probe, the upload,
# and the per-view executor call, and persist it across the successive per-view
# invocations that FixedWristMotion.__call__ makes (one process per view). This
# mirrors the perception path (FixedReadOnlyBackend._ssh_prefix) and collapses
# the ~0.5s handshake each per-view command otherwise re-pays down to ~0.05s.
control_path="$(dirname -- "$NUC_KEY")/z-manip-wrist-%C"
mux_args=(-o ControlMaster=auto -o ControlPersist=60 -o "ControlPath=$control_path")
ssh_args=(-i "$NUC_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=5 "${mux_args[@]}" "$NUC_HOST")
scp_args=(-q -i "$NUC_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=5 "${mux_args[@]}")

# The 4 uploaded files never change within a single wrist search, yet the search
# calls this script once per view. Hash the inputs and stamp the marker on the
# NUC after a complete upload; views 2..N whose marker already matches skip the
# ~1.0s rm+scp entirely and only pay the (multiplexed) executor call.
manifest_sha="$(sha256sum "${WRIST_SEARCH_INPUTS[@]}" | awk '{print $1}' | sha256sum | awk '{print $1}')"
remote_sha="$(ssh "${ssh_args[@]}" "cat '$REMOTE_DIR/.manifest-sha' 2>/dev/null || true")"
if [[ "$remote_sha" != "$manifest_sha" ]]; then
  ssh "${ssh_args[@]}" "rm -rf '$REMOTE_DIR'; mkdir -p '$REMOTE_DIR'"
  scp "${scp_args[@]}" "${WRIST_SEARCH_INPUTS[@]}" "$NUC_HOST:$REMOTE_DIR/"
  # Stamp the marker only after every file has landed, so an interrupted upload
  # never lets a later view skip a re-upload against a partial payload.
  ssh "${ssh_args[@]}" "printf '%s' '$manifest_sha' > '$REMOTE_DIR/.manifest-sha'"
fi

ssh "${ssh_args[@]}" \
  "set -e; systemctl --user stop z-manip-piper-passive-feedback.service; trap 'sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8 >/tmp/z-manip-passive-restore.log 2>&1 || true; systemctl --user start z-manip-piper-passive-feedback.service' EXIT; cd ~/pyAgxArm; /usr/bin/python3 '$REMOTE_DIR/piper_wrist_search_executor.py' --home '$REMOTE_DIR/piper_home.json' --view-index '$VIEW_INDEX' --speed-percent '$SPEED_PERCENT' --execute"
