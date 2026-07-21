#!/usr/bin/env bash
set -euo pipefail

# Fixed-surface process manager for the loopback UI and vision-only runtime.
# This script owns no actuator transport and accepts only the component names
# listed below.  A single flock prevents overlapping restart sequences.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$STACK_ROOT/.." && pwd)"
LAB_SCRIPT="$SCRIPT_DIR/go2w_perception_lab.sh"
UI_UNIT="z-manip-planning-workbench.service"
OBSERVER_UNIT="z-manip-runtime-observer.service"
GROUNDING_UNIT="z-manip-local-grounding.service"
POSTURE_UNIT="z-mobile-manip-go2w-posture-intent-live.service"
REACTIVE_NUC_UNIT="z-mobile-manip-go2w-reactive-live.service"
REACTIVE_INSTALLER="$SCRIPT_DIR/install_go2w_reactive_runtime.sh"
WHOLE_BODY_IMAGE="z-mobile-manip-whole-body:latest"
USER_SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UI_PORT="${Z_MANIP_DEBUG_UI_PORT:-8766}"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
LOCK_FILE="$RUNTIME_DIR/z-manip-component-manager.lock"
LOG_ROOT="${Z_MANIP_COMPONENT_LOG_ROOT:-$WORKSPACE_ROOT/artifacts/go2w_real/component_logs}"
MANAGER_LOG="$LOG_ROOT/component-manager.log"
CAMERA_ARTIFACT="${Z_MANIP_CAMERA_ARTIFACT:-$WORKSPACE_ROOT/artifacts/go2w_real/latest/camera-latest.jpg}"
WAIT_SECONDS="${Z_MANIP_COMPONENT_WAIT_SECONDS:-30}"
NUC_SNAPSHOT_LOADED=0
NUC_CAMERA_STATE="unreachable"
NUC_CAMERA_DEVICE="unreachable"
NUC_PASSIVE_STATE="unreachable"
NUC_BUS_STATE="unreachable"

DOCKER="${Z_MANIP_DOCKER_BIN:-docker}"
CURL="${Z_MANIP_CURL_BIN:-curl}"
SYSTEMCTL="${Z_MANIP_SYSTEMCTL_BIN:-systemctl}"
SYSTEMD_RUN="${Z_MANIP_SYSTEMD_RUN_BIN:-systemd-run}"
JOURNALCTL="${Z_MANIP_JOURNALCTL_BIN:-journalctl}"
SSH="${Z_MANIP_SSH_BIN:-ssh}"

usage() {
  cat >&2 <<'EOF'
usage: go2w_component_manager.sh bringup
       go2w_component_manager.sh install
       go2w_component_manager.sh status [all|ui|nuc-camera|passive-feedback|observer|rgbd|edgetam|perception|perception-all|reactive-control|posture-bridge|whole-body|mobile-control]
       go2w_component_manager.sh restart {ui|nuc-camera|passive-feedback|observer|rgbd|edgetam|perception|perception-all|reactive-control|posture-bridge|whole-body|mobile-control}
       go2w_component_manager.sh logs {manager|ui|nuc-camera|passive-feedback|observer|rgbd|edgetam|perception|perception-all|reactive-control|posture-bridge|mobile-control} [lines]
EOF
  exit 2
}

install_pc_units() {
  local unit source target wants
  wants="$USER_SYSTEMD_DIR/default.target.wants"
  mkdir -p "$USER_SYSTEMD_DIR" "$wants"
  for unit in "$UI_UNIT" "$OBSERVER_UNIT" "$GROUNDING_UNIT" "$POSTURE_UNIT"; do
    source="$STACK_ROOT/configs/$unit"
    target="$USER_SYSTEMD_DIR/$unit"
    [[ -f "$source" ]] || {
      printf 'required user unit is missing: %s\n' "$source" >&2
      return 1
    }
    ln -sfnT "$source" "$target"
    ln -sfnT "../$unit" "$wants/$unit"
  done
  $SYSTEMCTL --user daemon-reload
}

valid_component() {
  case "$1" in
    ui|nuc-camera|passive-feedback|observer|rgbd|edgetam|perception|perception-all|reactive-control|posture-bridge|whole-body|mobile-control) return 0 ;;
    *) return 1 ;;
  esac
}

container_running() {
  [[ "$($DOCKER inspect "$1" --format '{{.State.Running}}' 2>/dev/null || true)" == "true" ]]
}

container_summary() {
  local name="$1"
  $DOCKER inspect "$name" --format '{{.State.Status}}; restarts={{.RestartCount}}' 2>/dev/null \
    || printf 'container missing'
}

status_one() {
  local component="$1" state="offline" summary="not running"
  case "$component" in
    ui)
      if $SYSTEMCTL --user is-active --quiet "$UI_UNIT" 2>/dev/null; then
        state="healthy"
        summary="user service active on 127.0.0.1:$UI_PORT"
      fi
      ;;
    nuc-camera)
      load_nuc_snapshot
      if [[ "$NUC_CAMERA_STATE" == active && "$NUC_CAMERA_DEVICE" == ready ]]; then
        state="healthy"
        summary="NUC D435 USB present; d435i.service active"
      elif [[ "$NUC_CAMERA_STATE" == active ]]; then
        state="degraded"
        summary="d435i.service active but D435 USB device is absent"
      else
        summary="NUC d435i.service $NUC_CAMERA_STATE; D435 USB $NUC_CAMERA_DEVICE"
      fi
      ;;
    passive-feedback)
      load_nuc_snapshot
      if [[ "$NUC_PASSIVE_STATE" == active ]]; then
        if [[ "$NUC_BUS_STATE" == ready ]]; then
          state="healthy"
          summary="NUC passive feedback active; bus ERROR-ACTIVE"
        else
          state="degraded"
          summary="passive feedback active; bus $NUC_BUS_STATE"
        fi
      else
        summary="NUC passive feedback $NUC_PASSIVE_STATE"
      fi
      ;;
    observer)
      if $SYSTEMCTL --user is-active --quiet "$OBSERVER_UNIT" 2>/dev/null \
          && container_running z-manip-runtime-observer; then
        state="healthy"
        summary="subscribe-only runtime observer active"
      else
        summary="runtime observer service/container inactive"
      fi
      ;;
    rgbd)
      if container_running z-manip-rgbd; then
        if camera_artifact_fresh; then
          state="healthy"
          summary="fresh RGB-D frames; $(container_summary z-manip-rgbd)"
        else
          state="degraded"
          summary="bridge running but no fresh RGB-D camera artifact"
        fi
      else
        summary="$(container_summary z-manip-rgbd)"
      fi
      ;;
    edgetam)
      if container_running z-manip-edgetam; then
        if $CURL -fsS --max-time 2 http://127.0.0.1:8092/health >/dev/null 2>&1; then
          state="healthy"
          summary="HTTP health OK; $(container_summary z-manip-edgetam)"
        else
          state="degraded"
          summary="container running; HTTP health unavailable"
        fi
      else
        summary="$(container_summary z-manip-edgetam)"
      fi
      ;;
    perception)
      if container_running z-manip-hw; then
        state="healthy"
        summary="$(container_summary z-manip-hw)"
      else
        summary="$(container_summary z-manip-hw)"
      fi
      ;;
    perception-all)
      local observer_state rgbd_state edgetam_state perception_state
      observer_state="$(status_state observer)"
      rgbd_state="$(status_state rgbd)"
      edgetam_state="$(status_state edgetam)"
      perception_state="$(status_state perception)"
      if [[ "$observer_state" == healthy && "$rgbd_state" == healthy && "$edgetam_state" == healthy && "$perception_state" == healthy ]]; then
        state="healthy"
        summary="observer + RGB-D + EdgeTAM + perception ROS healthy"
      elif [[ "$observer_state" == offline || "$rgbd_state" == offline || "$edgetam_state" == offline || "$perception_state" == offline ]]; then
        state="offline"
        summary="observer=$observer_state, rgbd=$rgbd_state, edgetam=$edgetam_state, perception=$perception_state"
      else
        state="degraded"
        summary="observer=$observer_state, rgbd=$rgbd_state, edgetam=$edgetam_state, perception=$perception_state"
      fi
      ;;
    reactive-control)
      if remote_service_active "$REACTIVE_NUC_UNIT"; then
        state="healthy"
        summary="NUC single WebRTC owner active; Move + BodyHeight + Euler"
      else
        summary="NUC reactive WebRTC owner inactive"
      fi
      ;;
    posture-bridge)
      if $SYSTEMCTL --user is-active --quiet "$POSTURE_UNIT" 2>/dev/null \
          && container_running z-mobile-manip-posture-intent; then
        state="healthy"
        summary="PC posture intent/status relay active"
      else
        summary="posture intent relay service/container inactive"
      fi
      ;;
    whole-body)
      if $DOCKER image inspect "$WHOLE_BODY_IMAGE" >/dev/null 2>&1; then
        state="healthy"
        summary="Pinocchio + CasADi runtime image installed"
      else
        summary="whole-body runtime image missing"
      fi
      ;;
    mobile-control)
      local reactive_state posture_state whole_body_state
      reactive_state="$(status_state reactive-control)"
      posture_state="$(status_state posture-bridge)"
      whole_body_state="$(status_state whole-body)"
      if [[ "$reactive_state" == healthy && "$posture_state" == healthy && "$whole_body_state" == healthy ]]; then
        state="healthy"
        summary="NUC owner + posture relay + Pinocchio/CasADi ready"
      else
        state="degraded"
        summary="reactive=$reactive_state, posture=$posture_state, whole-body=$whole_body_state"
      fi
      ;;
  esac
  printf '%s\t%s\t%s\n' "$component" "$state" "$summary"
}

remote_command() {
  [[ -f "$NUC_KEY" ]] || return 1
  $SSH -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST" "$1"
}

load_nuc_snapshot() {
  if ((NUC_SNAPSHOT_LOADED == 1)); then
    return
  fi
  NUC_SNAPSHOT_LOADED=1
  local snapshot=()
  mapfile -t snapshot < <(remote_command \
    'systemctl --user is-active d435i.service 2>/dev/null || true; if compgen -G "/dev/v4l/by-id/*RealSense*" >/dev/null || lsusb -d 8086: 2>/dev/null | grep -qi RealSense; then echo ready; else echo absent; fi; systemctl --user is-active z-manip-piper-passive-feedback.service 2>/dev/null || true; if ip -details link show can0 2>/dev/null | grep -Fq "state ERROR-ACTIVE"; then echo ready; else echo down; fi' \
    2>/dev/null || true)
  if ((${#snapshot[@]} == 4)); then
    NUC_CAMERA_STATE="${snapshot[0]}"
    NUC_CAMERA_DEVICE="${snapshot[1]}"
    NUC_PASSIVE_STATE="${snapshot[2]}"
    NUC_BUS_STATE="${snapshot[3]}"
  fi
}

remote_service_active() {
  remote_command "systemctl --user is-active --quiet $1" >/dev/null 2>&1
}

remote_bus_ready() {
  remote_command "ip -details link show can0" 2>/dev/null \
    | grep -Fq 'state ERROR-ACTIVE'
}

remote_camera_device_ready() {
  remote_command 'compgen -G "/dev/v4l/by-id/*RealSense*" >/dev/null || lsusb -d 8086: 2>/dev/null | grep -qi RealSense' \
    >/dev/null 2>&1
}

camera_artifact_fresh() {
  [[ -f "$CAMERA_ARTIFACT" ]] || return 1
  local modified now
  modified="$(stat -c %Y "$CAMERA_ARTIFACT" 2>/dev/null)" || return 1
  now="$(date +%s)"
  ((now - modified <= 3))
}

ensure_remote_bus() {
  if remote_bus_ready; then
    return 0
  fi
  if ! remote_command 'test -x /usr/local/sbin/z-manip-piper-passive-can-gate' >/dev/null 2>&1; then
    printf 'NUC passive bus gate is missing; install it with install_nuc_passive_access.sh\n' >&2
    return 1
  fi
  if ! remote_command 'sudo -n /usr/local/sbin/z-manip-piper-passive-can-gate can0 8 >/tmp/z-manip-cold-bringup-passive.log 2>&1'; then
    printf 'NUC passive bus setup failed; inspect /tmp/z-manip-cold-bringup-passive.log and the scoped sudo rule\n' >&2
    return 1
  fi
  if ! remote_bus_ready; then
    printf 'NUC passive bus gate completed but can0 is not ERROR-ACTIVE\n' >&2
    return 1
  fi
}

status_state() {
  status_one "$1" | cut -f2
}

wait_until() {
  local label="$1"
  shift
  local deadline=$((SECONDS + WAIT_SECONDS))
  until "$@"; do
    if ((SECONDS >= deadline)); then
      printf '%s did not become healthy within %ss\n' "$label" "$WAIT_SECONDS" >&2
      return 1
    fi
    sleep 1
  done
}

rgbd_ready() {
  container_running z-manip-rgbd && camera_artifact_fresh
}

edgetam_ready() {
  container_running z-manip-edgetam \
    && $CURL -fsS --max-time 2 http://127.0.0.1:8092/health >/dev/null 2>&1
}

perception_ready() {
  if ! container_running z-manip-hw; then
    return 1
  fi
  $DOCKER exec z-manip-hw bash -lc \
    'source /opt/ros/jazzy/setup.bash && timeout 4 ros2 node list 2>/dev/null | grep -Fxq /vlm_edgetam_bridge && timeout 4 ros2 node list 2>/dev/null | grep -Fxq /z_manip_edgetam' \
    >/dev/null 2>&1
}

observer_ready() {
  $SYSTEMCTL --user is-active --quiet "$OBSERVER_UNIT" 2>/dev/null \
    && container_running z-manip-runtime-observer
}

posture_bridge_ready() {
  $SYSTEMCTL --user is-active --quiet "$POSTURE_UNIT" 2>/dev/null \
    && container_running z-mobile-manip-posture-intent
}

nuc_camera_ready() {
  remote_service_active d435i.service && remote_camera_device_ready
}

passive_feedback_ready() {
  remote_service_active z-manip-piper-passive-feedback.service && remote_bus_ready
}

ui_ready() {
  $SYSTEMCTL --user is-active --quiet "$UI_UNIT" 2>/dev/null \
    && $CURL -fsS --max-time 2 "http://127.0.0.1:$UI_PORT/api/health" >/dev/null 2>&1
}

restart_one() {
  local component="$1"
  case "$component" in
    ui)
      install_pc_units || return 1
      $SYSTEMCTL --user restart "$UI_UNIT" || return 1
      wait_until "UI workbench" ui_ready
      ;;
    nuc-camera)
      if ! remote_camera_device_ready; then
        printf 'D435 USB device is absent on the NUC; reconnect its USB cable before restarting the camera service\n' >&2
        return 1
      fi
      remote_command 'systemctl --user restart d435i.service' || return 1
      wait_until "NUC D435 service" nuc_camera_ready
      ;;
    passive-feedback)
      ensure_remote_bus || return 1
      remote_command 'systemctl --user restart z-manip-piper-passive-feedback.service' || return 1
      wait_until "NUC passive feedback" passive_feedback_ready
      ;;
    observer)
      install_pc_units || return 1
      $SYSTEMCTL --user restart "$OBSERVER_UNIT" || return 1
      wait_until "runtime observer" observer_ready
      ;;
    rgbd)
      if ! remote_camera_device_ready; then
        printf 'D435 USB device is absent on the NUC; reconnect it before restarting the RGB-D bridge\n' >&2
        return 1
      fi
      "$LAB_SCRIPT" restart-rgbd || return 1
      wait_until "RGB-D bridge" rgbd_ready
      ;;
    edgetam)
      "$LAB_SCRIPT" restart-edgetam || return 1
      wait_until "EdgeTAM" edgetam_ready
      ;;
    perception)
      "$LAB_SCRIPT" restart-perception || return 1
      wait_until "perception ROS" perception_ready
      ;;
    perception-all)
      if ! remote_camera_device_ready; then
        printf 'D435 USB device is absent on the NUC; reconnect it before restarting perception-all\n' >&2
        return 1
      fi
      restart_one edgetam || return 1
      restart_one rgbd || return 1
      restart_one perception || return 1
      restart_one observer
      ;;
    reactive-control|posture-bridge|mobile-control)
      "$REACTIVE_INSTALLER" || return 1
      wait_until "NUC reactive control" remote_service_active "$REACTIVE_NUC_UNIT" || return 1
      wait_until "posture relay" posture_bridge_ready
      ;;
    whole-body)
      $DOCKER build -t "$WHOLE_BODY_IMAGE" "$STACK_ROOT/docker/whole_body_runtime" || return 1
      ;;
  esac
}

cold_bringup_steps() {
  printf '[%s] cold bringup begin\n' "$(date --iso-8601=seconds)"
  install_pc_units || return 1
  if ! remote_camera_device_ready; then
    printf 'D435 USB device is absent on the NUC; reconnect its USB cable before bringup\n' >&2
    return 1
  fi
  # A service may remain active after a USB disconnect while publishing zero
  # frames. Always restart it during cold bringup.
  remote_command 'systemctl --user restart d435i.service' || return 1
  wait_until "NUC D435 service" nuc_camera_ready || return 1
  ensure_remote_bus || return 1
  remote_command 'systemctl --user restart z-manip-piper-passive-feedback.service' || return 1
  wait_until "NUC passive feedback" passive_feedback_ready || return 1
  restart_one edgetam || return 1
  restart_one rgbd || return 1
  restart_one perception || return 1
  $SYSTEMCTL --user enable --now "$OBSERVER_UNIT" || return 1
  wait_until "runtime observer" observer_ready || return 1
  $SYSTEMCTL --user enable --now "$UI_UNIT" || return 1
  wait_until "UI workbench" ui_ready || return 1
  "$REACTIVE_INSTALLER" || return 1
  wait_until "NUC reactive control" remote_service_active "$REACTIVE_NUC_UNIT" || return 1
  wait_until "posture relay" posture_bridge_ready || return 1
  if ! $DOCKER image inspect "$WHOLE_BODY_IMAGE" >/dev/null 2>&1; then
    $DOCKER build -t "$WHOLE_BODY_IMAGE" "$STACK_ROOT/docker/whole_body_runtime" || return 1
  fi
  printf '[%s] cold bringup healthy\n' "$(date --iso-8601=seconds)"
}

show_logs() {
  local component="$1" lines="$2"
  case "$component" in
    manager) tail -n "$lines" "$MANAGER_LOG" 2>/dev/null || true ;;
    ui) $JOURNALCTL --user -u "$UI_UNIT" -n "$lines" --no-pager -o cat 2>&1 ;;
    nuc-camera) remote_command "journalctl --user -u d435i.service -n $lines --no-pager -o cat" 2>&1 ;;
    passive-feedback) remote_command "journalctl --user -u z-manip-piper-passive-feedback.service -n $lines --no-pager -o cat" 2>&1 ;;
    observer) $JOURNALCTL --user -u "$OBSERVER_UNIT" -n "$lines" --no-pager -o cat 2>&1 ;;
    rgbd) $DOCKER logs --tail "$lines" z-manip-rgbd 2>&1 ;;
    edgetam) $DOCKER logs --tail "$lines" z-manip-edgetam 2>&1 ;;
    perception) $DOCKER logs --tail "$lines" z-manip-hw 2>&1 ;;
    reactive-control) remote_command "journalctl --user -u $REACTIVE_NUC_UNIT -n $lines --no-pager -o cat" 2>&1 ;;
    posture-bridge) $JOURNALCTL --user -u "$POSTURE_UNIT" -n "$lines" --no-pager -o cat 2>&1 ;;
    perception-all)
      for item in observer edgetam rgbd perception; do
        printf '===== %s =====\n' "$item"
        show_logs "$item" "$lines"
      done
      ;;
    mobile-control)
      for item in reactive-control posture-bridge; do
        printf '===== %s =====\n' "$item"
        show_logs "$item" "$lines"
      done
      ;;
  esac
}

mkdir -p "$LOG_ROOT" "$RUNTIME_DIR"
action="${1:-status}"
component="${2:-all}"

case "$action" in
  install)
    [[ "$component" == all ]] || usage
    install_pc_units
    ;;
  status)
    if [[ "$component" == all ]]; then
      for item in ui nuc-camera passive-feedback observer rgbd edgetam perception perception-all reactive-control posture-bridge whole-body mobile-control; do
        status_one "$item"
      done
    else
      valid_component "$component" || usage
      status_one "$component"
    fi
    ;;
  restart)
    valid_component "$component" || usage
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
      printf 'another component restart is already running\n' >&2
      exit 75
    fi
    printf '[%s] restart begin: %s\n' "$(date --iso-8601=seconds)" "$component" >>"$MANAGER_LOG"
    if restart_one "$component" >>"$MANAGER_LOG" 2>&1; then
      printf '[%s] restart healthy: %s\n' "$(date --iso-8601=seconds)" "$component" >>"$MANAGER_LOG"
    else
      rc=$?
      printf '[%s] restart failed: %s rc=%s\n' "$(date --iso-8601=seconds)" "$component" "$rc" >>"$MANAGER_LOG"
      show_logs "$component" 30 >>"$MANAGER_LOG" 2>&1 || true
      tail -n 35 "$MANAGER_LOG" >&2
      exit "$rc"
    fi
    status_one "$component"
    ;;
  bringup)
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
      printf 'another component restart is already running\n' >&2
      exit 75
    fi
    if cold_bringup_steps >>"$MANAGER_LOG" 2>&1; then
      :
    else
      rc=$?
      printf '[%s] cold bringup failed rc=%s\n' "$(date --iso-8601=seconds)" "$rc" >>"$MANAGER_LOG"
      tail -n 50 "$MANAGER_LOG" >&2
      exit "$rc"
    fi
    for item in ui nuc-camera passive-feedback observer rgbd edgetam perception perception-all reactive-control posture-bridge whole-body mobile-control; do
      status_one "$item"
    done
    ;;
  logs)
    lines="${3:-80}"
    [[ "$lines" =~ ^[0-9]+$ ]] && ((lines >= 1 && lines <= 300)) || usage
    if [[ "$component" != manager ]]; then
      valid_component "$component" || usage
    fi
    show_logs "$component" "$lines" | tail -c 24000
    ;;
  *) usage ;;
esac
