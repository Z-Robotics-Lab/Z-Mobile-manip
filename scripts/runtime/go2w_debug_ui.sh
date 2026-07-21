#!/usr/bin/env bash
# Start the loopback-only offline artifact dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$STACK_ROOT/.." && pwd)"
SERVER="$SCRIPT_DIR/go2w_debug_ui.py"
GENERATOR="$SCRIPT_DIR/go2w_debug_bundle.py"
SAFETY_GATE="$SCRIPT_DIR/go2w_debug_safety_gate.py"

BUNDLE=""
PORT=8766
OPEN_BROWSER=1

usage() {
  printf '%s\n' \
    "Usage: $(basename "$0") [--bundle FILE] [--port PORT] [--no-open]" \
    "" \
    "Without --bundle, regenerate debug_bundle.json from the latest recorded" \
    "perception cycle. Optional planning/calibration inputs are accepted only" \
    "through Z_MANIP_PLANNING_DIR and Z_MANIP_CAMERA_CALIBRATION." \
    "The dashboard always listens on 127.0.0.1."
}

while (($#)); do
  case "$1" in
    --bundle)
      (($# >= 2)) || { printf 'missing value for --bundle\n' >&2; exit 2; }
      BUNDLE="$2"
      shift 2
      ;;
    --port)
      (($# >= 2)) || { printf 'missing value for --port\n' >&2; exit 2; }
      PORT="$2"
      shift 2
      ;;
    --no-open)
      OPEN_BROWSER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) || {
  printf 'port must be an integer from 1 to 65535\n' >&2
  exit 2
}

if [[ -z "$BUNDLE" ]]; then
  if [[ -n "${Z_MANIP_ARTIFACT_ROOT:-}" ]]; then
    ARTIFACT_ROOT="$Z_MANIP_ARTIFACT_ROOT"
  elif [[ -d "$WORKSPACE_ROOT/artifacts/go2w_real/latest" ]]; then
    ARTIFACT_ROOT="$WORKSPACE_ROOT/artifacts/go2w_real/latest"
  else
    ARTIFACT_ROOT="$STACK_ROOT/artifacts/go2w_real/latest"
  fi

  if [[ -f "$ARTIFACT_ROOT/current-cycle/report.json" ]]; then
    PERCEPTION_DIR="$ARTIFACT_ROOT/current-cycle"
  elif [[ -f "$ARTIFACT_ROOT/report.json" ]]; then
    PERCEPTION_DIR="$ARTIFACT_ROOT"
  else
    printf 'no latest perception report under %s\n' "$ARTIFACT_ROOT" >&2
    exit 1
  fi

  BUNDLE="$ARTIFACT_ROOT/debug_bundle.json"
  GENERATE_ARGS=(
    --perception-dir "$PERCEPTION_DIR"
    --output "$BUNDLE"
  )

  if [[ -n "${Z_MANIP_JOINT_REPORT:-}" ]]; then
    GENERATE_ARGS+=(--joint-report "$Z_MANIP_JOINT_REPORT")
  elif [[ -d "$ARTIFACT_ROOT/piper" ]]; then
    LATEST_JOINT="$(find "$ARTIFACT_ROOT/piper" -maxdepth 1 -type f -name 'passive-*.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    [[ -z "$LATEST_JOINT" ]] || GENERATE_ARGS+=(--joint-report "$LATEST_JOINT")
  fi
  [[ -z "${Z_MANIP_PLANNING_DIR:-}" ]] || GENERATE_ARGS+=(--planning-dir "$Z_MANIP_PLANNING_DIR")
  [[ -z "${Z_MANIP_CAMERA_CALIBRATION:-}" ]] || GENERATE_ARGS+=(--calibration "$Z_MANIP_CAMERA_CALIBRATION")
  [[ -z "${Z_MANIP_ROBOT_URDF:-}" ]] || GENERATE_ARGS+=(--urdf "$Z_MANIP_ROBOT_URDF")

  python3 "$GENERATOR" "${GENERATE_ARGS[@]}"
fi

BUNDLE="$(readlink -f -- "$BUNDLE")"
python3 "$SERVER" --bundle "$BUNDLE" --check >/dev/null

if [[ -n "${Z_MANIP_SAFETY_ARTIFACT_ROOT:-}" ]]; then
  SAFETY_ROOT="$Z_MANIP_SAFETY_ARTIFACT_ROOT"
elif [[ -d "$WORKSPACE_ROOT/artifacts" ]]; then
  SAFETY_ROOT="$WORKSPACE_ROOT/artifacts"
else
  SAFETY_ROOT="$(dirname -- "$BUNDLE")"
fi
AUDIT="${BUNDLE%.json}.safety-audit.json"
python3 "$SAFETY_GATE" \
  --bundle "$BUNDLE" \
  --artifact-root "$SAFETY_ROOT" \
  --output "$AUDIT"

URL="http://127.0.0.1:${PORT}/"
printf 'Starting Z-Manip read-only dashboard\n  %s\n  bundle: %s\n  safety audit: %s\n' \
  "$URL" "$BUNDLE" "$AUDIT"
if ((OPEN_BROWSER)) && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 &
fi

exec python3 "$SERVER" --bundle "$BUNDLE" --port "$PORT"
