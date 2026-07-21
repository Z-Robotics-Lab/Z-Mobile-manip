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

transport_state() {
  "${SSH[@]}" "SERVICE='$SERVICE' bash -s" <<'REMOTE'
set -euo pipefail
active="$(systemctl --user is-active "$SERVICE" 2>/dev/null || true)"
logs="$(journalctl --user -u "$SERVICE" -n 240 --no-pager -o cat 2>/dev/null || true)"
ok_line="$(grep -nF 'Data Channel Verification:' <<<"$logs" | grep -F 'OK' | tail -n1 | cut -d: -f1 || true)"
fail_line="$(grep -nF 'Data channel is not open' <<<"$logs" | tail -n1 | cut -d: -f1 || true)"
owner_line="$(grep -nF 'LIVE single-owner bridge enabled' <<<"$logs" | tail -n1 | cut -d: -f1 || true)"
if [[ "$active" == active && -n "$ok_line" && -n "$owner_line" && ( -z "$fail_line" || "$ok_line" -gt "$fail_line" ) ]]; then
  printf 'ready\n'
else
  printf 'stale\n'
fi
REMOTE
}

if [[ "$(transport_state)" != ready ]]; then
  printf 'Go2W WebRTC transport is stale; restarting the fixed NUC service\n' >&2
  "${SSH[@]}" "systemctl --user restart '$SERVICE'"
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
