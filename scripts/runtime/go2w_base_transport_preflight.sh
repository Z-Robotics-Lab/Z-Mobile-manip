#!/usr/bin/env bash
set -euo pipefail

# Fixed, non-interactive preflight for the Go2W WebRTC command transport.
# The UI cannot supply a host, service, command, or environment override.
NUC_HOST="yusenzlabnuc@192.168.3.8"
NUC_KEY="$HOME/.ssh/id_ed25519_codex_nuc"
SERVICE="z-mobile-manip-go2w-reactive-live.service"
SSH=(ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST")

[[ -f "$NUC_KEY" ]] || {
  printf 'Go2W transport preflight failed: fixed NUC SSH key is missing\n' >&2
  exit 1
}

# The three markers are emitted ONCE, during bridge startup.  Scoping the query
# to the service's current systemd invocation is therefore the only window that
# means what this check means: "did THIS instance verify its data channel and
# take single ownership".
#
# The previous window was `-u SERVICE -n 240`, the last 240 log lines regardless
# of restart boundaries.  The bridge logs a "WebRTC motion service evidence"
# heartbeat every ~3 s, so 240 lines is ~12 minutes: past that, the startup
# markers scroll out and a perfectly healthy transport reads STALE.  Measured on
# the live NUC 2026-07-28 with the service active and NRestarts=0 -- markers 441
# and 432 lines back, zero of the three inside the window, verdict STALE.  Since
# the only caller is go2w_depth_servo.sh, every task started more than ~12 min
# after the bridge came up restarted a working single-owner base bridge and then
# waited up to 12 s for it to return, with no base authority in between.
#
# --grep filters journal-side, so this stays cheap on a long-lived invocation
# instead of shipping the whole heartbeat stream over ssh.
transport_state() {
  "${SSH[@]}" "SERVICE='$SERVICE' bash -s" <<'REMOTE'
set -euo pipefail
active="$(systemctl --user is-active "$SERVICE" 2>/dev/null || true)"
invocation="$(systemctl --user show "$SERVICE" -p InvocationID --value 2>/dev/null || true)"
if [[ "$active" != active || -z "$invocation" ]]; then
  # No running instance to have proven anything: fail closed to a restart.
  printf 'stale\n'
  exit 0
fi
logs="$(journalctl --user _SYSTEMD_INVOCATION_ID="$invocation" --no-pager -o cat \
  --grep='Data Channel Verification:|Data channel is not open|LIVE single-owner bridge enabled' \
  2>/dev/null || true)"
ok_line="$(grep -nF 'Data Channel Verification:' <<<"$logs" | grep -F 'OK' | tail -n1 | cut -d: -f1 || true)"
fail_line="$(grep -nF 'Data channel is not open' <<<"$logs" | tail -n1 | cut -d: -f1 || true)"
owner_line="$(grep -nF 'LIVE single-owner bridge enabled' <<<"$logs" | tail -n1 | cut -d: -f1 || true)"
if [[ -n "$ok_line" && -n "$owner_line" && ( -z "$fail_line" || "$ok_line" -gt "$fail_line" ) ]]; then
  printf 'ready\n'
else
  printf 'stale\n'
fi
REMOTE
}

if [[ "$(transport_state)" != ready ]]; then
  printf 'Go2W WebRTC transport is stale; restarting the fixed NUC service\n' >&2
  # H5.  THIS LINE USED TO BE THE SILENT ABORT.  Under `set -euo pipefail` a
  # `systemctl restart` that fails because the unit hit its start limiter exits
  # this script immediately, and go2w_depth_servo.sh calls it under `set -e`, so
  # the servo launch dies with the operator holding a message about a stale
  # transport -- not about systemd refusing to start anything.  Nothing in this
  # repository has ever called reset-failed.
  #
  # Diagnose first, clear only start-limit-hit, at most once per preflight.  The
  # readiness loop below is untouched, so a bridge that is genuinely broken
  # still fails this preflight and still stops the servo; it just says why.
  limiter="$(
    "${SSH[@]}" "systemctl --user show '$SERVICE' -p Result --value" 2>/dev/null || true
  )"
  if [[ "$limiter" == start-limit-hit ]]; then
    restarts="$(
      "${SSH[@]}" "systemctl --user show '$SERVICE' -p NRestarts --value" 2>/dev/null || true
    )"
    printf '%s tripped its systemd start limiter (Result=start-limit-hit, NRestarts=%s); clearing it once so this preflight reports the real transport fault\n' \
      "$SERVICE" "${restarts:-unknown}" >&2
    "${SSH[@]}" "systemctl --user reset-failed '$SERVICE'" >/dev/null 2>&1 || true
  fi
  if ! "${SSH[@]}" "systemctl --user restart '$SERVICE'"; then
    printf 'Go2W transport preflight failed: could not restart %s on the NUC (check `systemctl --user status %s` there)\n' \
      "$SERVICE" "$SERVICE" >&2
    exit 1
  fi
fi

for _ in $(seq 1 24); do
  if [[ "$(transport_state)" == ready ]]; then
    printf 'Go2W WebRTC transport ready\n'
    exit 0
  fi
  sleep 0.5
done

printf 'Go2W transport preflight failed: WebRTC data channel did not become ready\n' >&2
exit 1
