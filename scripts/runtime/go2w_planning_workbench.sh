#!/usr/bin/env bash
set -euo pipefail

# One command for the read-only perception/planning session and its dashboard.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$STACK_ROOT/.." && pwd)"
RUN_ROOT="${Z_MANIP_PLANNING_RUN_ROOT:-$WORKSPACE_ROOT/artifacts/go2w_real/planning_sessions}"
PORT="${Z_MANIP_DEBUG_UI_PORT:-8766}"
AUDIT_LOG="$RUN_ROOT/debug-dashboard-safety.log"
UI_CONTAINER="${Z_MANIP_DEBUG_UI_CONTAINER:-z-manip-debug-dashboard}"
UI_SERVICE="z-manip-planning-workbench.service"
UI_UNIT_SOURCE="$STACK_ROOT/configs/$UI_SERVICE"
UI_UNIT_TARGET="$HOME/.config/systemd/user/$UI_SERVICE"

stop_legacy_ui() {
  docker rm -f "$UI_CONTAINER" >/dev/null 2>&1 || true
}

install_ui_service() {
  [[ -f "$UI_UNIT_SOURCE" ]] || { printf 'missing UI service unit: %s\n' "$UI_UNIT_SOURCE" >&2; exit 1; }
  mkdir -p "$(dirname -- "$UI_UNIT_TARGET")"
  if ! cmp -s "$UI_UNIT_SOURCE" "$UI_UNIT_TARGET"; then
    install -m 0644 "$UI_UNIT_SOURCE" "$UI_UNIT_TARGET"
    systemctl --user daemon-reload
  fi
  systemctl --user enable "$UI_SERVICE" >/dev/null
}

show_latest() {
  latest="$(readlink -f -- "$RUN_ROOT/latest")"
  bundle="$latest/debug_bundle.json"
  [[ -f "$bundle" ]] || { printf 'latest debug bundle is missing: %s\n' "$bundle" >&2; exit 1; }
  stop_legacy_ui
  python3 "$SCRIPT_DIR/go2w_debug_ui.py" --bundle "$bundle" --check >/dev/null
  python3 "$SCRIPT_DIR/go2w_debug_safety_gate.py" \
    --bundle "$bundle" --artifact-root "$WORKSPACE_ROOT/artifacts" \
    --output "${bundle%.json}.safety-audit.json" >"$AUDIT_LOG"
  install_ui_service
  systemctl --user restart "$UI_SERVICE"
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
      printf 'Go2W planning workbench: http://127.0.0.1:%s/\n' "$PORT"
      printf 'bundle: %s\n' "$bundle"
      printf 'control: fixed planning-only pipeline (%s)\n' "$UI_SERVICE"
      if [[ "${Z_MANIP_OPEN_BROWSER:-1}" == 1 ]] && command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://127.0.0.1:$PORT/" >/dev/null 2>&1 &
      fi
      return
    fi
    sleep 0.2
  done
  systemctl --user status "$UI_SERVICE" --no-pager >&2 || true
  printf 'dashboard failed to start; inspect: journalctl --user -u %s\n' "$UI_SERVICE" >&2
  exit 1
}

case "${1:-run}" in
  run)
    shift || true
    run_rc=0
    "$SCRIPT_DIR/go2w_planning_session.sh" "$@" || run_rc=$?
    show_latest
    exit "$run_rc"
    ;;
  show)
    show_latest
    ;;
  stop-ui)
    stop_legacy_ui
    systemctl --user disable --now "$UI_SERVICE" >/dev/null 2>&1 || true
    ;;
  *)
    printf 'usage: %s {run [instruction]|show|stop-ui}\n' "$0" >&2
    exit 2
    ;;
esac
