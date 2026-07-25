#!/usr/bin/env bash
# Shared helpers for scripts/runtime/*.sh.  Source with:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
# Every function below reads its inputs from the sourcing script's own
# variables (NUC_HOST, NUC_KEY, ...) rather than accepting fresh state, so
# behavior stays identical to the inline/duplicated code it replaces.

# require_file PATH [MESSAGE_PREFIX] [TEST_FLAG]
#   Exits 1 with "<MESSAGE_PREFIX>: <PATH>" on stderr unless
#   `test TEST_FLAG PATH` succeeds.  MESSAGE_PREFIX defaults to
#   "required file is missing"; TEST_FLAG defaults to -f (regular file).
#   Pass -e for "any path type" (ffs_depth_stack.sh checks a path that may be
#   reached through a symlinked third-party checkout).
require_file() {
  local path="$1" message="${2:-required file is missing}" flag="${3:--f}"
  test "$flag" "$path" || { printf '%s: %s\n' "$message" "$path" >&2; exit 1; }
}

# nuc_ssh COMMAND...
#   Runs COMMAND on the NUC over a short-lived batch SSH connection, using
#   the caller's NUC_HOST / NUC_KEY globals.  Callers that need additional
#   ssh options (connection multiplexing, IdentitiesOnly, ...) set
#   NUC_SSH_EXTRA_OPTS (an array) before calling.
nuc_ssh() {
  ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 \
    "${NUC_SSH_EXTRA_OPTS[@]}" "$NUC_HOST" "$@"
}

# nuc_scp ARG...
#   Runs scp with the same fixed connection defaults as nuc_ssh.  Callers
#   pass the full source/destination argument list, e.g.
#   `nuc_scp "$file" "$NUC_HOST:dest/"`.  Callers that need additional scp
#   options set NUC_SCP_EXTRA_OPTS (an array) before calling.
nuc_scp() {
  scp -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 \
    "${NUC_SCP_EXTRA_OPTS[@]}" "$@"
}

NUC_SSH_EXTRA_OPTS=()
NUC_SCP_EXTRA_OPTS=()

# nuc_ros_env_bootstrap
#   Sources the NUC-side unitree venv / ROS / nav-stack / go2w-nuc workspace
#   setup scripts.  This exact sequence was byte-identical across every
#   NUC-resident launcher.  nounset is toggled off only for the duration of
#   the sourcing (these upstream setup scripts probe unset variables),
#   matching the inline block this replaces.
nuc_ros_env_bootstrap() {
  set +u
  source "$HOME/unitree_venv/bin/activate"
  source /opt/ros/jazzy/setup.bash
  source "$HOME/Z-Navigation-Stack/install/setup.bash"
  source "$HOME/go2w-nuc/ros2_ws/install/setup.bash"
  set -u
}
