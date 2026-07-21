#!/usr/bin/env bash
# z-manip test entry point (host side).
#
# ATTACH-ONLY: this runner NEVER starts, restarts, or tears down any sim or
# container. The tests only observe a chain that is already up; they SKIP when
# the chain is not green (checked via ~/Desktop/go2w/scripts/nav/status.sh) or
# when a live probe cannot reach it. ROS2 runs INSIDE the navstack container
# through the exec seam (tests/helpers.py); nothing here needs rclpy on the host.
#
# sim vs real: the exec seam is chosen by $Z_MANIP_ROS_EXEC.
#   sim  (default): unset -> `docker exec navstack bash -lc '<ros env>'`
#   real robot    : export Z_MANIP_ROS_EXEC="" -> commands run natively.
#
# NO third-party install: pytest must already be available. We detect it and
# print guidance if missing; we DO NOT pip-install anything into the system.
#
# Usage:
#   scripts/run-tests.sh                 # collection check + M0/e2e live layer
#   scripts/run-tests.sh -m m0           # only the M0 gates
#   scripts/run-tests.sh -m "m0 or e2e"  # M0 + e2e
#   scripts/run-tests.sh --collect-only  # collection only (no chain needed)
#   scripts/run-tests.sh <any pytest args...>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- locate pytest without ever installing it -------------------------------
PYTEST=()
if command -v pytest >/dev/null 2>&1; then
    PYTEST=(pytest)
elif python3 -c "import pytest" >/dev/null 2>&1; then
    PYTEST=(python3 -m pytest)
else
    cat >&2 <<'MSG'
[run-tests] pytest not found on this host.
  This runner will NOT pip-install into the system Python.
  Install pytest in a venv/user site yourself, e.g.:
      python3 -m venv .venv && . .venv/bin/activate && pip install pytest
  then re-run. (Collection-only still needs pytest; the tests themselves need
  no third-party deps — rclpy runs inside the navstack container.)
MSG
    exit 2
fi

echo "[run-tests] using: ${PYTEST[*]}"
echo "[run-tests] exec seam: Z_MANIP_ROS_EXEC=${Z_MANIP_ROS_EXEC-<default: docker exec navstack>}"

# --- always verify collection is clean first --------------------------------
echo "[run-tests] step 1/2: collection check (no chain required)"
"${PYTEST[@]}" --collect-only -q

# If the caller only wants collection, stop here.
for a in "$@"; do
    if [[ "$a" == "--collect-only" ]]; then
        echo "[run-tests] collection-only requested; done."
        exit 0
    fi
done

# --- run the requested layer (default: M0 + e2e) ----------------------------
echo "[run-tests] step 2/2: live run (attach-only; skips if chain not green)"
if [[ $# -eq 0 ]]; then
    set -- -m "m0 or e2e"
fi
exec "${PYTEST[@]}" "$@"
