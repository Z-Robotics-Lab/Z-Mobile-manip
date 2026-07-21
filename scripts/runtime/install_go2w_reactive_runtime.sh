#!/usr/bin/env bash
set -euo pipefail

# Install the single-owner Go2W SPORT bridge on the NUC and the fixed posture
# relay on this workstation. Starting these services does not itself issue a
# movement command; commands arrive only from an operator-started Live run.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
SSH=(ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST")
SCP=(scp -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5)

[[ -f "$NUC_KEY" ]] || { printf 'missing NUC SSH key: %s\n' "$NUC_KEY" >&2; exit 1; }
"${SSH[@]}" 'mkdir -p "$HOME/.local/lib/z-mobile-manip/z_manip/control" "$HOME/.config/systemd/user" "$HOME/.config/z-mobile-manip"'
"${SCP[@]}" \
  "$SCRIPT_DIR/go2w_reactive_control_nuc.py" \
  "$SCRIPT_DIR/go2w_reactive_control_nuc.sh" \
  "$NUC_HOST:.local/lib/z-mobile-manip/"
"${SCP[@]}" \
  "$STACK_ROOT/z_manip/__init__.py" \
  "$NUC_HOST:.local/lib/z-mobile-manip/z_manip/"
"${SCP[@]}" \
  "$STACK_ROOT/configs/nuc-control-init.py" \
  "$STACK_ROOT/z_manip/control/go2w_posture.py" \
  "$NUC_HOST:.local/lib/z-mobile-manip/z_manip/control/"
"${SSH[@]}" 'mv "$HOME/.local/lib/z-mobile-manip/z_manip/control/nuc-control-init.py" "$HOME/.local/lib/z-mobile-manip/z_manip/control/__init__.py"'
"${SCP[@]}" \
  "$STACK_ROOT/configs/z-mobile-manip-go2w-reactive-live.service" \
  "$NUC_HOST:.config/systemd/user/"
"${SCP[@]}" \
  "$STACK_ROOT/configs/go2w-reactive-live.env" \
  "$NUC_HOST:.config/z-mobile-manip/go2w-reactive-live.env"
"${SSH[@]}" 'chmod 0755 "$HOME/.local/lib/z-mobile-manip/go2w_reactive_control_nuc.sh" "$HOME/.local/lib/z-mobile-manip/go2w_reactive_control_nuc.py"; systemctl --user daemon-reload; systemctl --user disable --now z-manip-go2w-base-control.service >/dev/null 2>&1 || true; systemctl --user enable z-mobile-manip-go2w-reactive-live.service; systemctl --user restart z-mobile-manip-go2w-reactive-live.service'

mkdir -p "$HOME/.config/systemd/user" "$HOME/.config/z-mobile-manip"
ln -sfnT \
  "$STACK_ROOT/configs/z-mobile-manip-go2w-posture-intent-live.service" \
  "$HOME/.config/systemd/user/z-mobile-manip-go2w-posture-intent-live.service"
cp -f \
  "$STACK_ROOT/configs/go2w-posture-intent-live.env" \
  "$HOME/.config/z-mobile-manip/go2w-posture-intent-live.env"
systemctl --user daemon-reload
systemctl --user enable z-mobile-manip-go2w-posture-intent-live.service
systemctl --user restart z-mobile-manip-go2w-posture-intent-live.service

printf 'Go2W reactive runtime installed. No motion command was issued.\n'
