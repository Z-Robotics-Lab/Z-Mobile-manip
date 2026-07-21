#!/usr/bin/env bash
set -euo pipefail

# One-time root bootstrap for persistent, least-privilege NUC access.  It does
# not grant a shell, generic sudo, CAN transmission, or actuator execution.
if [[ "$EUID" -ne 0 ]]; then
  echo "run this installer with sudo" >&2
  exit 2
fi

target_user="${1:-${SUDO_USER:-}}"
if [[ -z "$target_user" || "$target_user" == "root" ]]; then
  echo "a non-root NUC username is required" >&2
  exit 2
fi
if ! id "$target_user" >/dev/null 2>&1; then
  echo "unknown NUC user: $target_user" >&2
  exit 2
fi

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
probe_source="$source_dir/piper_passive_probe.py"
gate_source="$source_dir/piper_passive_can_gate.sh"
down_source="$source_dir/piper_can_down.sh"
for source in "$probe_source" "$gate_source" "$down_source"; do
  if [[ ! -f "$source" ]]; then
    echo "missing installer input: $source" >&2
    exit 2
  fi
done

install -d -o root -g root -m 0755 /usr/local/libexec/z-manip
install -o root -g root -m 0755 \
  "$probe_source" \
  /usr/local/libexec/z-manip/piper_passive_probe.py
install -o root -g root -m 0755 \
  "$gate_source" \
  /usr/local/sbin/z-manip-piper-passive-can-gate
install -o root -g root -m 0755 \
  "$down_source" \
  /usr/local/sbin/z-manip-piper-can-down

sudoers_file=/etc/sudoers.d/z-manip-passive-can
temporary="$(mktemp)"
trap 'rm -f "$temporary"' EXIT
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/z-manip-piper-passive-can-gate can0 8\n' \
  "$target_user" >"$temporary"
printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/z-manip-piper-can-down\n' \
  "$target_user" >>"$temporary"
chmod 0440 "$temporary"
visudo -cf "$temporary"
install -o root -g root -m 0440 "$temporary" "$sudoers_file"
visudo -cf "$sudoers_file"

systemctl enable --now ssh.service
loginctl enable-linger "$target_user"

printf 'PASS: persistent scoped access installed for %s\n' "$target_user"
printf 'allowed: sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8\n'
printf 'allowed: sudo -n /usr/local/sbin/z-manip-piper-can-down\n'
